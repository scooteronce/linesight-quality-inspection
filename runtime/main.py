"""LineSight inspection handler running inside Amazon Bedrock AgentCore Runtime."""

import base64
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# The CDK CodeZip bundle vendors Linux ARM64 dependencies under ./vendor.
vendor = Path(__file__).resolve().parent / "vendor"
if vendor.is_dir():
    sys.path.insert(0, str(vendor))

import boto3
from anthropic import AnthropicBedrock, AnthropicBedrockMantle
from bedrock_agentcore.runtime import BedrockAgentCoreApp


REGION = os.getenv("AWS_REGION", "us-west-2")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "anthropic.claude-opus-5-5")
# "mantle" serves current-generation Claude models; "legacy" (InvokeModel) serves
# older generations such as us.anthropic.claude-sonnet-4-6 via inference profiles.
BEDROCK_CLIENT = os.getenv("BEDROCK_CLIENT", "mantle").lower()
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
REFERENCES_PATH = Path(__file__).resolve().parent / "reference_embeddings.json"
MAX_PAYLOAD_BYTES = 5 * 1024 * 1024
MAX_FRAMES = 20
TOP_K = 3

session = boto3.Session(region_name=REGION)
bedrock = session.client("bedrock-runtime")
claude = (AnthropicBedrock if BEDROCK_CLIENT == "legacy" else AnthropicBedrockMantle)(aws_region=REGION)
app = BedrockAgentCoreApp()

SUPPORTED_MEDIA_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
    "image/gif": "image/gif",
}

INSPECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "inspection_status": {"type": "string", "enum": ["PASS", "REVIEW", "FAIL"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "location": {"type": "string"},
                    "evidence": {"type": "string"},
                    # No "minimum": the legacy Bedrock integration rejects it for integers.
                    "frame": {"type": "integer"},
                },
                "required": ["type", "severity", "location", "evidence", "frame"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["inspection_status", "confidence", "defects", "summary"],
    "additionalProperties": False,
}

INSPECTION_PROMPT = """Inspect this manufactured part for visible surface or assembly defects.
Report inspection_status, confidence, the visible defects, and a one-sentence summary.
Use FAIL only for a clearly visible, material defect. Use REVIEW when the image is
blurry, poorly lit, obstructed, unfamiliar, or the evidence is uncertain. Use PASS
when no visible defect is apparent. Do not invent measurements or claim certainty
that the image does not support. Describe each defect location in plain language;
do not invent pixel coordinates or bounding boxes."""

SINGLE_IMAGE_NOTE = 'There is one image, so set "frame" to 1 for every defect.'

MULTI_FRAME_NOTE = """The {count} images are sequential frames of the same part sampled from a video.
Report each distinct defect once, and set "frame" to the frame number (1-{count}) where it
is most clearly visible. Use agreement across frames to confirm or rule out a defect:
a mark visible from several angles is stronger evidence; something visible in only one
frame may be glare or a reflection and deserves REVIEW rather than FAIL unless it is
unambiguous."""


def inspect_frames(frames_base64: list[str], content_type: str) -> dict[str, Any]:
    media_type = SUPPORTED_MEDIA_TYPES.get(content_type)
    if media_type is None:
        raise ValueError("Upload JPEG, PNG, WebP, or GIF frames.")

    multi = len(frames_base64) > 1
    content: list[dict[str, Any]] = []
    for index, frame in enumerate(frames_base64, start=1):
        if multi:
            content.append({"type": "text", "text": f"Frame {index}:"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": frame},
            }
        )
    note = MULTI_FRAME_NOTE.format(count=len(frames_base64)) if multi else SINGLE_IMAGE_NOTE
    content.append({"type": "text", "text": f"{INSPECTION_PROMPT}\n{note}"})

    response = claude.messages.create(
        model=VISION_MODEL_ID,
        max_tokens=2048,
        system="You are a cautious visual quality inspection assistant.",
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": INSPECTION_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        return {
            "inspection_status": "REVIEW",
            "confidence": "low",
            "defects": [],
            "summary": "The model declined to analyze this image. Route the part to manual inspection.",
        }
    if response.stop_reason == "max_tokens":
        raise ValueError("The inspection response was truncated. Try again with fewer or smaller frames.")
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def get_embedding(text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def _load_references() -> list[dict[str, Any]]:
    if not REFERENCES_PATH.is_file():
        return []
    return json.loads(REFERENCES_PATH.read_text())


REFERENCES = _load_references()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def find_similar_references(query_text: str) -> list[dict[str, Any]]:
    """Rank the bundled reference records by cosine similarity to the query.

    The reference library is small, so it ships with the runtime as precomputed
    Titan embeddings instead of a managed vector store with an always-on cost.
    """
    if not REFERENCES:
        return []
    query = get_embedding(query_text)
    scored = sorted(
        (
            {
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
                "similarity": round(_cosine_similarity(query, record.get("embedding", [])), 3),
            }
            for record in REFERENCES
        ),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return scored[:TOP_K]


def _validate_frames(payload: dict[str, Any]) -> list[str]:
    """Return the base64 frames from a payload with image_base64 or frames_base64."""
    frames = payload.get("frames_base64")
    if frames is None:
        single = payload.get("image_base64")
        frames = [single] if single is not None else []
    if not isinstance(frames, list) or not frames:
        raise ValueError("Request must include image_base64 or a non-empty frames_base64 list.")
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"Send at most {MAX_FRAMES} frames.")

    total_bytes = 0
    for frame in frames:
        if not isinstance(frame, str) or not frame:
            raise ValueError("Each frame must be a non-empty base64 string.")
        try:
            decoded = base64.b64decode(frame, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("Frames must contain valid base64 data.") from exc
        if not decoded:
            raise ValueError("Each frame must decode to at least 1 byte.")
        total_bytes += len(decoded)
    if total_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError("Frames must total at most 5 MB after resizing.")
    return frames


@app.entrypoint
def handle_inspection(payload: dict[str, Any]) -> dict[str, Any]:
    content_type = str(payload.get("content_type", "image/jpeg")).lower()
    frames = _validate_frames(payload)

    inspection = inspect_frames(frames, content_type)
    defect_text = "; ".join(
        f"{item.get('type', 'possible defect')}: {item.get('evidence', '')}"
        for item in inspection["defects"]
        if isinstance(item, dict)
    )
    query_text = defect_text or inspection["summary"] or "No visible defect observed."
    try:
        references = find_similar_references(query_text)
    except Exception as exc:  # Preserve the inspection result if retrieval is unavailable.
        references = []
        inspection["retrieval_note"] = f"Reference search unavailable: {type(exc).__name__}"

    return {
        "inspection": inspection,
        "similar_references": references,
        "model_id": VISION_MODEL_ID,
        "frame_count": len(frames),
        "review_note": "AI recommendation only. An inspector makes the final disposition.",
    }


if __name__ == "__main__":
    app.run()
