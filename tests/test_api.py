from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_detector
from app.schemas import BoundingBox, DetectResponse, Detection


class StubDetector:
    def predict(self, image_bytes: bytes, filename: str) -> DetectResponse:
        return DetectResponse(
            filename=filename,
            width=32,
            height=32,
            model="stub",
            detections=[
                Detection(
                    label="person",
                    confidence=0.91,
                    box=BoundingBox(x1=1, y1=2, x2=20, y2=30),
                )
            ],
        )


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_image() -> None:
    app.dependency_overrides[get_detector] = lambda: StubDetector()
    client = TestClient(app)

    response = client.post(
        "/api/v1/detect",
        files={"image": ("sample.png", make_png(), "image/png")},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.png"
    assert body["detections"][0]["label"] == "person"


def test_rejects_non_image_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/detect",
        files={"image": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400

