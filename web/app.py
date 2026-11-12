"""Local Streamlit review screen for the LineSight AgentCore runtime."""

import base64
import json
import os
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import boto3
import cv2
import streamlit as st
from PIL import Image


REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
RUNTIME_ARN = os.getenv("AGENT_RUNTIME_ARN", "")
IMAGE_MAX_EDGE = 1800
FRAME_MAX_EDGE = 1024
MAX_TOTAL_BYTES = 5 * 1024 * 1024
VIDEO_TYPES = {"mp4", "mov", "m4v", "avi", "webm"}

STATUS_BANNERS = {"PASS": st.success, "REVIEW": st.warning, "FAIL": st.error}


def encode_jpeg(image: Image.Image, max_edge: int, quality: int) -> bytes:
    scaled = image.copy()
    scaled.thumbnail((max_edge, max_edge))
    output = BytesIO()
    scaled.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def prepare_image(upload) -> list[bytes]:
    image = Image.open(upload).convert("RGB")
    for quality in (88, 72):
        data = encode_jpeg(image, IMAGE_MAX_EDGE, quality)
        if len(data) <= MAX_TOTAL_BYTES:
            return [data]
    raise ValueError("Image is still over 5 MB after resizing.")


def extract_frames(upload, frame_count: int) -> list[Image.Image]:
    """Sample frames evenly across the uploaded video clip."""
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getvalue())
        temp_path = handle.name
    capture = cv2.VideoCapture(temp_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Could not read any frames from this video.")
        count = min(frame_count, total)
        # Sample midpoints of equal segments so the first and last moments both appear.
        indices = [int((segment + 0.5) * total / count) for segment in range(count)]
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, total - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError("Could not decode frames from this video.")
        return frames
    finally:
        capture.release()
        os.unlink(temp_path)


def prepare_video_frames(upload, frame_count: int) -> list[bytes]:
    frames = extract_frames(upload, frame_count)
    for quality in (80, 65, 50):
        encoded = [encode_jpeg(frame, FRAME_MAX_EDGE, quality) for frame in frames]
        if sum(len(data) for data in encoded) <= MAX_TOTAL_BYTES:
            return encoded
    raise ValueError("Sampled frames exceed 5 MB even at reduced quality. Use fewer frames.")


def invoke_runtime(frames: list[bytes], filename: str) -> dict:
    agent = boto3.client("bedrock-agentcore", region_name=REGION)
    response = agent.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        runtimeSessionId=uuid.uuid4().hex + uuid.uuid4().hex[:1],
        qualifier="DEFAULT",
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(
            {
                "frames_base64": [base64.b64encode(data).decode("ascii") for data in frames],
                "content_type": "image/jpeg",
                "filename": filename,
            }
        ).encode("utf-8"),
    )
    return json.loads(response["response"].read().decode("utf-8"))


def render_result(result: dict, elapsed: float, frames: list[bytes]) -> None:
    inspection = result["inspection"]
    status = str(inspection.get("inspection_status", "REVIEW")).upper()
    banner = STATUS_BANNERS.get(status, st.warning)
    multi_frame = len(frames) > 1

    banner(f"Recommendation: **{status}**")
    st.write(inspection.get("summary", ""))
    st.caption(
        f"Model confidence: {inspection.get('confidence', 'low')} · "
        f"{result.get('model_id', 'unknown model')} · "
        f"{result.get('frame_count', len(frames))} frame(s) · {elapsed:.1f}s"
    )

    defects = inspection.get("defects") or []
    if defects:
        st.markdown("**Visible findings**")
        for defect in defects:
            frame_note = f" · frame {defect.get('frame', 'n/a')}" if multi_frame else ""
            st.markdown(
                f"- **{defect.get('type', 'defect')}** "
                f"(severity: {defect.get('severity', 'n/a')}{frame_note}) — {defect.get('location', '')}"
            )
            if defect.get("evidence"):
                st.caption(f"Evidence: {defect['evidence']}")

    if multi_frame:
        with st.expander("Sampled frames sent to the model"):
            columns = st.columns(min(len(frames), 4))
            for index, data in enumerate(frames, start=1):
                with columns[(index - 1) % len(columns)]:
                    st.image(data, caption=f"Frame {index}", use_container_width=True)

    st.markdown("**Similar references**")
    refs = result.get("similar_references", [])
    if refs:
        for ref in refs:
            metadata = ref.get("metadata", {})
            with st.container(border=True):
                st.markdown(
                    f"**{metadata.get('type', 'reference')}** · "
                    f"severity: {metadata.get('severity', 'n/a')} · "
                    f"similarity: {ref.get('similarity', 'n/a')}"
                )
                st.write(ref.get("text", ""))
    else:
        st.write("No similar references were returned.")

    if inspection.get("retrieval_note"):
        st.warning(inspection["retrieval_note"])
    st.warning(result.get("review_note", "Human review is required for final disposition."))


st.set_page_config(page_title="LineSight Quality Inspection", layout="wide")
st.title("LineSight Quality Inspection")
st.caption(
    "AI-assisted visual inspection with similar reference examples. "
    "A human inspector makes the final disposition."
)

if not RUNTIME_ARN:
    st.info("Set AGENT_RUNTIME_ARN to the AgentCore Runtime ARN from the CDK stack outputs.")

uploaded = st.file_uploader(
    "Upload a product image or a short video clip",
    type=["jpg", "jpeg", "png", "webp", *sorted(VIDEO_TYPES)],
)
if uploaded:
    is_video = Path(uploaded.name).suffix.lstrip(".").lower() in VIDEO_TYPES
    left, right = st.columns([1, 1])
    with left:
        if is_video:
            st.video(uploaded)
            frame_count = st.slider(
                "Frames to sample",
                min_value=2,
                max_value=12,
                value=8,
                help="Frames are sampled evenly across the clip. Each frame adds model cost.",
            )
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
    label = "Inspect video" if is_video else "Inspect image"
    if st.button(label, type="primary", disabled=not RUNTIME_ARN):
        with right:
            try:
                with st.spinner("Inspecting..."):
                    if is_video:
                        frames = prepare_video_frames(uploaded, frame_count)
                    else:
                        frames = prepare_image(uploaded)
                    started = time.monotonic()
                    result = invoke_runtime(frames, uploaded.name)
                    elapsed = time.monotonic() - started
                render_result(result, elapsed, frames)
            except Exception as exc:
                st.error(f"Inspection failed: {type(exc).__name__}: {exc}")
