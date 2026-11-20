import base64
import io
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

import app


class FakeUpload(io.BytesIO):
    """Stands in for a Streamlit UploadedFile: a BytesIO with a filename."""

    name = "clip.mp4"


def make_video(path, frame_count=90, size=(640, 480)):
    """Write a small MP4 with a moving rectangle so sampled frames differ."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(frame_count):
        frame = np.full((size[1], size[0], 3), 200, np.uint8)
        cv2.rectangle(frame, (i * 5, 100), (i * 5 + 80, 220), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def video_upload(tmp_path, **kwargs):
    data = make_video(tmp_path / "clip.mp4", **kwargs).read_bytes()
    return FakeUpload(data)


def test_prepare_image_returns_one_jpeg_within_limits():
    source = BytesIO()
    Image.new("RGB", (2400, 1600), (120, 120, 120)).save(source, format="PNG")
    source.seek(0)

    (data,) = app.prepare_image(source)

    image = Image.open(BytesIO(data))
    assert image.format == "JPEG"
    assert max(image.size) <= app.IMAGE_MAX_EDGE
    assert len(data) <= app.MAX_TOTAL_BYTES


def test_prepare_image_keeps_aspect_ratio():
    source = BytesIO()
    Image.new("RGB", (3600, 1800), (120, 120, 120)).save(source, format="PNG")
    source.seek(0)

    (data,) = app.prepare_image(source)

    image = Image.open(BytesIO(data))
    assert image.size == (1800, 900)


def test_extract_frames_samples_the_requested_count(tmp_path):
    frames = app.extract_frames(video_upload(tmp_path), 8)
    assert len(frames) == 8
    assert all(isinstance(frame, Image.Image) for frame in frames)


def test_extract_frames_clamps_to_the_clip_length(tmp_path):
    frames = app.extract_frames(video_upload(tmp_path, frame_count=3), 8)
    assert len(frames) == 3


def test_extract_frames_covers_the_whole_clip(tmp_path):
    # The rectangle moves right over time, so the first and last sampled
    # frames should differ if sampling spans the clip.
    frames = app.extract_frames(video_upload(tmp_path), 4)
    first = np.asarray(frames[0])
    last = np.asarray(frames[-1])
    assert not np.array_equal(first, last)


def test_extract_frames_rejects_an_unreadable_file(tmp_path):
    upload = FakeUpload(b"this is not a video")
    with pytest.raises(ValueError, match="video"):
        app.extract_frames(upload, 4)


def test_prepare_video_frames_returns_jpegs_within_budget(tmp_path):
    encoded = app.prepare_video_frames(video_upload(tmp_path), 8)

    assert len(encoded) == 8
    assert sum(len(data) for data in encoded) <= app.MAX_TOTAL_BYTES
    for data in encoded:
        image = Image.open(BytesIO(data))
        assert image.format == "JPEG"
        assert max(image.size) <= app.FRAME_MAX_EDGE


def test_prepare_image_rejects_an_image_that_cannot_fit_the_budget():
    source = BytesIO()
    Image.new("RGB", (2400, 1600), (120, 120, 120)).save(source, format="PNG")
    source.seek(0)
    with patch.object(app, "MAX_TOTAL_BYTES", 10), pytest.raises(ValueError, match="still over 5 MB"):
        app.prepare_image(source)


def test_prepare_video_frames_rejects_a_batch_that_cannot_fit(tmp_path):
    with patch.object(app, "MAX_TOTAL_BYTES", 10), pytest.raises(ValueError, match="reduced quality"):
        app.prepare_video_frames(video_upload(tmp_path), 4)


def mock_capture(frame_count, reads):
    """A cv2.VideoCapture stand-in that reports frame_count and yields the given reads."""
    capture = MagicMock()
    capture.get.return_value = float(frame_count)
    capture.read.side_effect = reads
    return capture


def test_extract_frames_skips_frames_that_fail_to_read():
    good = (True, np.zeros((4, 4, 3), np.uint8))
    capture = mock_capture(90, [(False, None), good, good, good])
    with patch.object(app.cv2, "VideoCapture", return_value=capture):
        frames = app.extract_frames(FakeUpload(b"ignored"), 4)
    assert len(frames) == 3


def test_extract_frames_raises_when_no_frame_decodes():
    capture = mock_capture(90, [(False, None)] * 4)
    with patch.object(app.cv2, "VideoCapture", return_value=capture):
        with pytest.raises(ValueError, match="decode frames"):
            app.extract_frames(FakeUpload(b"ignored"), 4)


def test_invoke_runtime_sends_base64_frames_to_the_runtime():
    client = MagicMock()
    client.invoke_agent_runtime.return_value = {"response": BytesIO(b'{"ok": true}')}

    with patch.object(app.boto3, "client", return_value=client), \
         patch.object(app, "RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-west-2:0:runtime/test"):
        result = app.invoke_runtime([b"one", b"two"], "part.jpg")

    assert result == {"ok": True}
    call = client.invoke_agent_runtime.call_args.kwargs
    assert call["agentRuntimeArn"] == "arn:aws:bedrock-agentcore:us-west-2:0:runtime/test"
    assert call["qualifier"] == "DEFAULT"
    payload = json.loads(call["payload"])
    assert payload["frames_base64"] == [
        base64.b64encode(b"one").decode("ascii"),
        base64.b64encode(b"two").decode("ascii"),
    ]
    assert payload["content_type"] == "image/jpeg"
    assert payload["filename"] == "part.jpg"
