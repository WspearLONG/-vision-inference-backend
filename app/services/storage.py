import json
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import Settings


def create_task_id() -> str:
    return uuid.uuid4().hex


def task_upload_dir(settings: Settings, task_id: str) -> Path:
    return Path(settings.upload_dir) / task_id


def task_output_path(settings: Settings, task_id: str) -> Path:
    return Path(settings.output_dir) / f"{task_id}.json"


async def save_uploads(files: list[UploadFile], settings: Settings, task_id: str) -> list[str]:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="at least one image is required",
        )

    upload_dir = task_upload_dir(settings, task_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    saved_paths: list[str] = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{file.filename or 'upload'} must be an image",
            )

        data = await file.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{file.filename or 'upload'} exceeds {settings.max_upload_mb} MB upload limit",
            )

        filename = Path(file.filename or f"{uuid.uuid4().hex}.jpg").name
        path = upload_dir / filename
        path.write_bytes(data)
        saved_paths.append(str(path))

    return saved_paths


def write_task_result(settings: Settings, task_id: str, payload: dict) -> str:
    output_path = task_output_path(settings, task_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def read_task_result(settings: Settings, task_id: str) -> dict | None:
    output_path = task_output_path(settings, task_id)
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))

