import base64
import json
import math
from unittest.mock import MagicMock, patch

import pytest

import main

# A representative structured-output response with one defect
SCRATCH_FINDING = {
    "inspection_status": "FAIL",
    "confidence": "high",
    "defects": [
        {"type": "scratch", "severity": "high", "location": "center", "evidence": "dark groove", "frame": 1}
    ],
    "summary": "Deep scratch across the part.",
}

CLEAN_FINDING = {
    "inspection_status": "PASS",
    "confidence": "high",
    "defects": [],
    "summary": "Clean surface.",
}


def b64(data):
    return base64.b64encode(data).decode("ascii")


def mock_claude(payload, stop_reason="end_turn"):
    """Patch the module's Claude client so messages.create returns the given inspection JSON."""
    block = MagicMock(type="text", text=json.dumps(payload))
    response = MagicMock(stop_reason=stop_reason, content=[block])
    client = MagicMock()
    client.messages.create.return_value = response
    return patch.object(main, "claude", client), client


def reference(name, vector):
    return {"text": name, "metadata": {"id": name}, "embedding": vector}


class TestValidateFrames:
    def test_accepts_frames_list(self):
        assert len(main._validate_frames({"frames_base64": [b64(b"abc"), b64(b"defg")]})) == 2

    def test_accepts_single_image_payload(self):
        assert len(main._validate_frames({"image_base64": b64(b"abc")})) == 1

    def test_rejects_missing_payload(self):
        with pytest.raises(ValueError, match="image_base64 or a non-empty frames_base64"):
            main._validate_frames({})

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="image_base64 or a non-empty frames_base64"):
            main._validate_frames({"frames_base64": []})

    def test_rejects_too_many_frames(self):
        with pytest.raises(ValueError, match="at most"):
            main._validate_frames({"frames_base64": [b64(b"x")] * (main.MAX_FRAMES + 1)})

    def test_rejects_invalid_base64(self):
        with pytest.raises(ValueError, match="valid base64"):
            main._validate_frames({"frames_base64": ["not-base64!!!"]})

    def test_rejects_non_string_frame(self):
        with pytest.raises(ValueError, match="non-empty base64 string"):
            main._validate_frames({"frames_base64": [12345]})

    def test_rejects_oversized_payload(self):
        big = b64(b"x" * (main.MAX_PAYLOAD_BYTES + 1))
        with pytest.raises(ValueError, match="5 MB"):
            main._validate_frames({"frames_base64": [big]})

    def test_size_limit_applies_to_the_total_not_each_frame(self):
        half = b64(b"x" * (main.MAX_PAYLOAD_BYTES // 2 + 1))
        with pytest.raises(ValueError, match="5 MB"):
            main._validate_frames({"frames_base64": [half, half]})


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert main._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert main._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        assert main._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_scores_zero_instead_of_dividing_by_zero(self):
        assert main._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestFindSimilarReferences:
    def test_returns_top_k_sorted_by_similarity(self):
        library = [
            reference("far", [0.0, 1.0]),
            reference("near", [1.0, 0.0]),
            reference("mid", [math.sqrt(0.5), math.sqrt(0.5)]),
            reference("opposite", [-1.0, 0.0]),
        ]
        with patch.object(main, "REFERENCES", library), \
             patch.object(main, "get_embedding", return_value=[1.0, 0.0]):
            results = main.find_similar_references("a scratch")

        assert [r["metadata"]["id"] for r in results] == ["near", "mid", "far"]
        assert results[0]["similarity"] == pytest.approx(1.0)

    def test_empty_library_returns_empty_without_calling_titan(self):
        with patch.object(main, "REFERENCES", []), \
             patch.object(main, "get_embedding") as get_embedding:
            assert main.find_similar_references("anything") == []
        get_embedding.assert_not_called()

    def test_missing_embeddings_file_yields_an_empty_library(self, tmp_path):
        with patch.object(main, "REFERENCES_PATH", tmp_path / "missing.json"):
            assert main._load_references() == []


class TestInspectFrames:
    def test_single_image_request_shape(self):
        patcher, client = mock_claude(CLEAN_FINDING)
        with patcher:
            result = main.inspect_frames([b64(b"img")], "image/jpeg")

        assert result["inspection_status"] == "PASS"
        call = client.messages.create.call_args.kwargs
        content = call["messages"][0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert 'set "frame" to 1' in content[-1]["text"]
        assert call["output_config"]["format"]["schema"] == main.INSPECTION_SCHEMA

    def test_multi_frame_request_interleaves_labels(self):
        patcher, client = mock_claude(CLEAN_FINDING)
        with patcher:
            main.inspect_frames([b64(b"a"), b64(b"b"), b64(b"c")], "image/png")

        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert [block["type"] for block in content] == [
            "text", "image", "text", "image", "text", "image", "text",
        ]
        assert content[0]["text"] == "Frame 1:"
        assert content[4]["text"] == "Frame 3:"
        assert "sequential frames" in content[-1]["text"]

    def test_jpg_alias_maps_to_jpeg_media_type(self):
        patcher, client = mock_claude(CLEAN_FINDING)
        with patcher:
            main.inspect_frames([b64(b"img")], "image/jpg")
        content = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert content[0]["source"]["media_type"] == "image/jpeg"

    def test_rejects_unsupported_media_type(self):
        with pytest.raises(ValueError, match="JPEG, PNG, WebP, or GIF"):
            main.inspect_frames([b64(b"img")], "image/tiff")

    def test_refusal_routes_to_review_instead_of_raising(self):
        patcher, _ = mock_claude(CLEAN_FINDING, stop_reason="refusal")
        with patcher:
            result = main.inspect_frames([b64(b"img")], "image/jpeg")

        assert result["inspection_status"] == "REVIEW"
        assert result["confidence"] == "low"
        assert "manual inspection" in result["summary"]

    def test_truncated_response_raises(self):
        patcher, _ = mock_claude(CLEAN_FINDING, stop_reason="max_tokens")
        with patcher, pytest.raises(ValueError, match="truncated"):
            main.inspect_frames([b64(b"img")], "image/jpeg")


class TestHandleInspection:
    def test_full_flow_returns_inspection_references_and_metadata(self):
        patcher, _ = mock_claude(SCRATCH_FINDING)
        refs = [{"text": "ref", "metadata": {"id": "surface-scratch-03"}, "similarity": 0.9}]
        with patcher, patch.object(main, "find_similar_references", return_value=refs) as find:
            result = main.handle_inspection({"image_base64": b64(b"img"), "content_type": "image/jpeg"})

        assert result["inspection"]["inspection_status"] == "FAIL"
        assert result["similar_references"] == refs
        assert result["frame_count"] == 1
        assert result["model_id"] == main.VISION_MODEL_ID
        # The retrieval query is built from the defect type and evidence.
        assert "scratch" in find.call_args.args[0]
        assert "dark groove" in find.call_args.args[0]

    def test_retrieval_failure_keeps_the_inspection_result(self):
        patcher, _ = mock_claude(SCRATCH_FINDING)
        with patcher, patch.object(main, "find_similar_references", side_effect=ConnectionError("down")):
            result = main.handle_inspection({"image_base64": b64(b"img")})

        assert result["inspection"]["inspection_status"] == "FAIL"
        assert result["similar_references"] == []
        assert "ConnectionError" in result["inspection"]["retrieval_note"]

    def test_pass_with_no_defects_queries_by_summary(self):
        patcher, _ = mock_claude(CLEAN_FINDING)
        with patcher, patch.object(main, "find_similar_references", return_value=[]) as find:
            main.handle_inspection({"image_base64": b64(b"img")})

        assert find.call_args.args[0] == "Clean surface."
