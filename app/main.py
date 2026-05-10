from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import DetectResponse, HealthResponse
from app.services.detector import Detector, YOLODetector


@lru_cache
def get_yolo_detector() -> YOLODetector:
    return YOLODetector(get_settings())


def get_detector() -> Detector:
    return get_yolo_detector()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vision Inference Backend",
        version="0.1.0",
        description="A production-oriented backend template for computer vision model serving.",
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

    return app


app = create_app()
