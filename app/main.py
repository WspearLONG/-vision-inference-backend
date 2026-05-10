from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db, init_db
from app.schemas import (
    BatchDetectCreateResponse,
    DetectResponse,
    HealthResponse,
    RootResponse,
    TaskArtifactsResponse,
    TaskStatusResponse,
    VideoTaskCreateResponse,
)
from app.services.detector import Detector, YOLODetector
from app.services.queue import RedisTaskQueue, TaskQueue
from app.services.storage import create_task_id, list_task_artifacts, read_task_result, save_uploads, save_video_upload
from app.services.tasks import create_task_record, create_video_task_record, get_task_status_from_db, list_task_artifacts_from_db


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

    @app.on_event("startup")
    def startup() -> None:
        init_db()

    @app.get("/", response_model=RootResponse)
    def root(settings: Settings = Depends(get_settings)) -> RootResponse:
        return RootResponse(
            service=settings.app_name,
            docs="/docs",
            health="/health",
            endpoints={
                "detect_image": "/api/v1/detect",
                "batch_detect": "/api/v1/batch-detect",
                "video_tasks": "/api/v1/video-tasks",
                "task_status": "/api/v1/tasks/{task_id}",
                "task_result": "/api/v1/tasks/{task_id}/result",
                "task_artifacts": "/api/v1/tasks/{task_id}/artifacts",
            },
        )

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
        db: Session = Depends(get_db),
    ) -> BatchDetectCreateResponse:
        task_id = create_task_id()
        image_paths = await save_uploads(images, settings, task_id)
        create_task_record(db, task_id, image_paths)
        task_queue.enqueue_batch_detect(task_id=task_id, image_paths=image_paths)
        return BatchDetectCreateResponse(task_id=task_id, status="pending", total=len(image_paths))

    @app.post("/api/v1/video-tasks", response_model=VideoTaskCreateResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_video_task(
        video: UploadFile = File(...),
        settings: Settings = Depends(get_settings),
        task_queue: TaskQueue = Depends(get_task_queue),
        db: Session = Depends(get_db),
    ) -> VideoTaskCreateResponse:
        task_id = create_task_id()
        video_path = await save_video_upload(video, settings, task_id)
        create_video_task_record(db, task_id, video_path)
        task_queue.enqueue_video_detect(task_id=task_id, video_path=video_path)
        return VideoTaskCreateResponse(
            task_id=task_id,
            status="pending",
            filename=Path(video_path).name,
            frame_stride=settings.video_frame_stride,
            max_frames=settings.max_video_frames,
        )

    @app.get("/api/v1/tasks/{task_id}/artifacts", response_model=TaskArtifactsResponse)
    def get_task_artifacts(
        task_id: str,
        settings: Settings = Depends(get_settings),
        db: Session = Depends(get_db),
    ) -> TaskArtifactsResponse:
        artifacts = list_task_artifacts_from_db(db, task_id) or list_task_artifacts(settings, task_id)
        return TaskArtifactsResponse(task_id=task_id, artifacts=artifacts)

    @app.get("/api/v1/video-tasks/{task_id}/artifacts", response_model=TaskArtifactsResponse)
    def get_video_task_artifacts(
        task_id: str,
        settings: Settings = Depends(get_settings),
        db: Session = Depends(get_db),
    ) -> TaskArtifactsResponse:
        artifacts = list_task_artifacts_from_db(db, task_id) or list_task_artifacts(settings, task_id)
        return TaskArtifactsResponse(task_id=task_id, artifacts=artifacts)

    @app.get("/api/v1/task-artifacts/{task_id}", response_model=TaskArtifactsResponse)
    def get_task_artifacts_alias(
        task_id: str,
        settings: Settings = Depends(get_settings),
        db: Session = Depends(get_db),
    ) -> TaskArtifactsResponse:
        artifacts = list_task_artifacts_from_db(db, task_id) or list_task_artifacts(settings, task_id)
        return TaskArtifactsResponse(task_id=task_id, artifacts=artifacts)

    @app.get("/api/v1/tasks/{task_id}/result")
    def get_task_result(task_id: str, settings: Settings = Depends(get_settings)) -> dict:
        result = read_task_result(settings, task_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="task result is not available",
            )
        return result

    @app.get("/api/v1/video-tasks/{task_id}/result")
    def get_video_task_result(task_id: str, settings: Settings = Depends(get_settings)) -> dict:
        return get_task_result(task_id, settings)

    @app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(
        task_id: str,
        task_queue: TaskQueue = Depends(get_task_queue),
        db: Session = Depends(get_db),
    ) -> TaskStatusResponse:
        queue_status = task_queue.get_status(task_id)
        db_status = get_task_status_from_db(db, task_id)
        if queue_status.status != "not_found":
            return queue_status
        if db_status is not None:
            return db_status
        return queue_status

    @app.get("/api/v1/video-tasks/{task_id}", response_model=TaskStatusResponse)
    def get_video_task_status(
        task_id: str,
        task_queue: TaskQueue = Depends(get_task_queue),
        db: Session = Depends(get_db),
    ) -> TaskStatusResponse:
        return get_task_status(task_id, task_queue, db)

    return app


app = create_app()
