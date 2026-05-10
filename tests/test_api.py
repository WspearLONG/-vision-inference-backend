from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_detector, get_task_queue
from app.schemas import BoundingBox, DetectResponse, Detection, TaskStatusResponse


class StubDetector:
    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str | None = None,
        confidence: float | None = None,
        image_size: int | None = None,
    ) -> DetectResponse:
        return DetectResponse(
            filename=filename,
            width=32,
            height=32,
            model=model_id or "stub",
            confidence_threshold=confidence,
            image_size=image_size,
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
        self.video_path = ""
        self.inference_options: dict = {}

    def enqueue_batch_detect(self, task_id: str, image_paths: list[str], inference_options: dict) -> str:
        self.task_id = task_id
        self.image_paths = image_paths
        self.inference_options = inference_options
        return task_id

    def enqueue_video_detect(self, task_id: str, video_path: str, inference_options: dict) -> str:
        self.task_id = task_id
        self.video_path = video_path
        self.inference_options = inference_options
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


def test_root() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert body["endpoints"]["video_tasks"] == "/api/v1/video-tasks"
    assert body["endpoints"]["models"] == "/api/v1/models"


def test_list_models() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json()[0]["id"] == "yolov8n"


def test_detect_image() -> None:
    app.dependency_overrides[get_detector] = lambda: StubDetector()
    client = TestClient(app)

    response = client.post(
        "/api/v1/detect?model_id=yolov8n&confidence=0.5&image_size=320",
        files={"image": ("sample.png", make_png(), "image/png")},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.png"
    assert body["model"] == "yolov8n"
    assert body["confidence_threshold"] == 0.5
    assert body["image_size"] == 320
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
        "/api/v1/batch-detect?model_id=yolov8n&confidence=0.4&image_size=320",
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
    assert body["model"] == "yolov8n"
    assert queue.task_id == body["task_id"]
    assert len(queue.image_paths) == 2
    assert queue.inference_options["confidence"] == 0.4
    assert queue.inference_options["image_size"] == 320


def test_get_task_status() -> None:
    queue = StubTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: queue
    client = TestClient(app)

    response = client.get("/api/v1/tasks/task-1")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_get_empty_task_artifacts() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/tasks/missing-task/artifacts")

    assert response.status_code == 200
    assert response.json() == {"task_id": "missing-task", "artifacts": []}


def test_create_video_task() -> None:
    queue = StubTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: queue
    client = TestClient(app)

    response = client.post(
        "/api/v1/video-tasks?model_id=yolov8n&confidence=0.45&frame_stride=5&max_frames=10",
        files={"video": ("sample.mp4", b"fake-video-bytes", "video/mp4")},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["filename"] == "sample.mp4"
    assert body["model"] == "yolov8n"
    assert body["frame_stride"] == 5
    assert body["max_frames"] == 10
    assert queue.task_id == body["task_id"]
    assert queue.video_path.endswith("sample.mp4")
    assert queue.inference_options["confidence"] == 0.45


def test_rejects_non_video_upload() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/video-tasks",
        files={"video": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
