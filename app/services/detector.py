from io import BytesIO
from functools import lru_cache
from typing import Protocol

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.schemas import BoundingBox, DetectResponse, Detection
from app.services.model_registry import get_model_config


class Detector(Protocol):
    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str | None = None,
        confidence: float | None = None,
        image_size: int | None = None,
    ) -> DetectResponse:
        ...


class YOLODetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str | None = None,
        confidence: float | None = None,
        image_size: int | None = None,
    ) -> DetectResponse:
        model_config = get_model_config(model_id)
        resolved_confidence = model_config.default_confidence if confidence is None else confidence
        resolved_image_size = model_config.default_image_size if image_size is None else image_size
        image = _load_image(image_bytes)
        results = _load_yolo_model(model_config.path).predict(
            source=image,
            conf=resolved_confidence,
            imgsz=resolved_image_size,
            verbose=False,
        )

        detections: list[Detection] = []
        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=str(names.get(cls_id, cls_id)),
                    confidence=confidence,
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return DetectResponse(
            filename=filename,
            width=image.width,
            height=image.height,
            detections=detections,
            model=model_config.id,
            confidence_threshold=resolved_confidence,
            image_size=resolved_image_size,
        )


@lru_cache
def _load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ultralytics is not installed; install requirements.txt first",
        ) from exc
    return YOLO(model_path)


def _load_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is not a valid image",
        ) from exc
