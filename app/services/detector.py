from io import BytesIO
from typing import Protocol

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.schemas import BoundingBox, DetectResponse, Detection


class Detector(Protocol):
    def predict(self, image_bytes: bytes, filename: str) -> DetectResponse:
        ...


class YOLODetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="ultralytics is not installed; install requirements.txt first",
                ) from exc
            self._model = YOLO(self.settings.model_name)
        return self._model

    def predict(self, image_bytes: bytes, filename: str) -> DetectResponse:
        image = _load_image(image_bytes)
        results = self.model.predict(
            source=image,
            conf=self.settings.confidence_threshold,
            imgsz=self.settings.image_size,
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
            model=self.settings.model_name,
        )


def _load_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is not a valid image",
        ) from exc

