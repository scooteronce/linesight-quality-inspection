"""Precompute Titan embeddings for the reference records.

Writes runtime/reference_embeddings.json, which ships inside the AgentCore
CodeZip bundle so the runtime can rank references in-process with no
always-on vector store.
"""

import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "defect-references.json"
OUTPUT_PATH = ROOT / "runtime" / "reference_embeddings.json"


def embed(text: str, client) -> list[float]:
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def main() -> None:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    records = json.loads(DATA_PATH.read_text())
    output = [
        {
            "text": record["text"],
            "metadata": {key: value for key, value in record.items() if key != "text"},
            "embedding": embed(record["text"], bedrock),
        }
        for record in records
    ]
    OUTPUT_PATH.write_text(json.dumps(output))
    print(f"Embedded {len(output)} reference records into {OUTPUT_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
