from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings


class ModelConfig(BaseModel):
    id: str
    name: str
    path: str
    task_type: str = "detection"
    default_confidence: float = Field(ge=0, le=1)
    default_image_size: int = Field(gt=0)
    description: str | None = None


@lru_cache
def load_model_registry() -> dict[str, ModelConfig]:
    settings = get_settings()
    registry_dir = Path(settings.model_config_dir)
    registry: dict[str, ModelConfig] = {}

    if registry_dir.exists():
        for path in sorted(registry_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            model = ModelConfig(**data)
            registry[model.id] = model

    if not registry:
        model = ModelConfig(
            id="default",
            name="Default YOLO model",
            path=settings.model_name,
            task_type="detection",
            default_confidence=settings.confidence_threshold,
            default_image_size=settings.image_size,
            description="Fallback model from environment settings.",
        )
        registry[model.id] = model

    return registry


def list_models() -> list[ModelConfig]:
    return list(load_model_registry().values())


def get_model_config(model_id: str | None) -> ModelConfig:
    registry = load_model_registry()
    if model_id is None:
        return registry.get("yolov8n") or next(iter(registry.values()))
    if model_id not in registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model '{model_id}' is not registered",
        )
    return registry[model_id]


def resolve_inference_options(
    model_id: str | None,
    confidence: float | None,
    image_size: int | None,
) -> tuple[ModelConfig, float, int]:
    model = get_model_config(model_id)
    resolved_confidence = model.default_confidence if confidence is None else confidence
    resolved_image_size = model.default_image_size if image_size is None else image_size

    if not 0 <= resolved_confidence <= 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="confidence must be between 0 and 1",
        )
    if resolved_image_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_size must be greater than 0",
        )

    return model, resolved_confidence, resolved_image_size

