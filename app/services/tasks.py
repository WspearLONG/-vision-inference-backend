from pathlib import Path

from sqlalchemy.orm import Session

from app.models import TaskArtifact, TaskImage, VisionTask
from app.schemas import TaskStatusResponse


def create_task_record(db: Session, task_id: str, image_paths: list[str]) -> VisionTask:
    task = VisionTask(id=task_id, status="pending", total=len(image_paths), completed=0)
    task.images = [
        TaskImage(task_id=task_id, filename=Path(image_path).name, path=image_path)
        for image_path in image_paths
    ]
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def update_task_status(
    db: Session,
    task_id: str,
    status: str,
    completed: int | None = None,
    result_path: str | None = None,
    error: str | None = None,
) -> None:
    task = db.get(VisionTask, task_id)
    if task is None:
        return
    task.status = status
    if completed is not None:
        task.completed = completed
    if result_path is not None:
        task.result_path = result_path
    if error is not None:
        task.error = error
    db.commit()


def replace_task_artifacts(db: Session, task_id: str, artifacts: list[dict]) -> None:
    task = db.get(VisionTask, task_id)
    if task is None:
        return
    task.artifacts.clear()
    for artifact in artifacts:
        task.artifacts.append(
            TaskArtifact(
                task_id=task_id,
                filename=artifact["filename"],
                path=artifact["path"],
                url=artifact["url"],
            )
        )
    db.commit()


def get_task_status_from_db(db: Session, task_id: str) -> TaskStatusResponse | None:
    task = db.get(VisionTask, task_id)
    if task is None:
        return None
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        total=task.total,
        completed=task.completed,
        result_url=f"/api/v1/tasks/{task.id}/result" if task.status == "succeeded" else None,
        error=task.error,
    )


def list_task_artifacts_from_db(db: Session, task_id: str) -> list[dict]:
    task = db.get(VisionTask, task_id)
    if task is None:
        return []
    return [
        {"filename": artifact.filename, "path": artifact.path, "url": artifact.url}
        for artifact in task.artifacts
    ]

