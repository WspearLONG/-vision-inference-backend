from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.schemas import BatchDetectCreateResponse, DetectResponse, HealthResponse, TaskArtifactsResponse, TaskStatusResponse
from app.services.detector import Detector, YOLODetector
from app.services.queue import RedisTaskQueue, TaskQueue
from app.services.storage import create_task_id, list_task_artifacts, read_task_result, save_uploads


@lru_cache
def get_yolo_detector() -> YOLODetector:
    return YOLODetector(get_settings())


def get_detector() -> Detector:
    return get_yolo_detector()


def get_task_queue(settings: Settings = Depends(get_settings)) -> TaskQueue:
    return RedisTaskQueue(settings)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vision Inference Backend",
        version="0.1.0",
        description="A production-oriented backend template for computer vision model serving.",
    )
    Path(get_settings().output_dir).mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=get_settings().output_dir), name="artifacts")

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            environment=settings.app_env,
        )

    @app.post("/api/v1/detect", response_model=DetectResponse)
    async def detect(
        image: UploadFile = File(...),
        settings: Settings = Depends(get_settings),
        detector: Detector = Depends(get_detector),
    ) -> DetectResponse:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="file must be an image",
            )

        image_bytes = await image.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"image exceeds {settings.max_upload_mb} MB upload limit",
            )

        return detector.predict(image_bytes=image_bytes, filename=image.filename or "upload")

    @app.post("/api/v1/batch-detect", response_model=BatchDetectCreateResponse, status_code=status.HTTP_202_ACCEPTED)
    async def batch_detect(
        images: list[UploadFile] = File(...),
        settings: Settings = Depends(get_settings),
        task_queue: TaskQueue = Depends(get_task_queue),
    ) -> BatchDetectCreateResponse:
        task_id = create_task_id()
        image_paths = await save_uploads(images, settings, task_id)
        task_queue.enqueue_batch_detect(task_id=task_id, image_paths=image_paths)
        return BatchDetectCreateResponse(task_id=task_id, status="pending", total=len(image_paths))

    @app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(
        task_id: str,
        task_queue: TaskQueue = Depends(get_task_queue),
    ) -> TaskStatusResponse:
        return task_queue.get_status(task_id)

    @app.get("/api/v1/tasks/{task_id}/result")
    def get_task_result(task_id: str, settings: Settings = Depends(get_settings)) -> dict:
        result = read_task_result(settings, task_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="task result is not available",
            )
        return result

    @app.get("/api/v1/tasks/{task_id}/artifacts", response_model=TaskArtifactsResponse)
    def get_task_artifacts(task_id: str, settings: Settings = Depends(get_settings)) -> TaskArtifactsResponse:
        artifacts = list_task_artifacts(settings, task_id)
        return TaskArtifactsResponse(task_id=task_id, artifacts=artifacts)

    return app


app = create_app()
