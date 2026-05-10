from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_detector, get_task_queue
from app.schemas import BoundingBox, DetectResponse, Detection, TaskStatusResponse


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


class StubTaskQueue:
    def __init__(self) -> None:
        self.task_id = ""
        self.image_paths: list[str] = []

    def enqueue_batch_detect(self, task_id: str, image_paths: list[str]) -> str:
        self.task_id = task_id
        self.image_paths = image_paths
        return task_id

    def get_status(self, task_id: str) -> TaskStatusResponse:
        return TaskStatusResponse(
            task_id=task_id,
            status="pending",
            total=len(self.image_paths) if self.image_paths else 1,
            completed=0,
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


def test_create_batch_detect_task() -> None:
    queue = StubTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: queue
    client = TestClient(app)

    response = client.post(
        "/api/v1/batch-detect",
        files=[
            ("images", ("sample-1.png", make_png(), "image/png")),
            ("images", ("sample-2.png", make_png(), "image/png")),
        ],
    )

    app.dependency_overrides.clear()
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["total"] == 2
    assert queue.task_id == body["task_id"]
    assert len(queue.image_paths) == 2


def test_get_task_status() -> None:
    queue = StubTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: queue
    client = TestClient(app)

    response = client.get("/api/v1/tasks/task-1")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
