from pathlib import Path

from rq import get_current_job

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.services.detector import YOLODetector
from app.services.storage import write_task_result
from app.services.tasks import replace_task_artifacts, update_task_status
from app.services.visualization import render_detection_artifact


def run_batch_detect(task_id: str, image_paths: list[str]) -> dict:
    init_db()
    settings = get_settings()
    detector = YOLODetector(settings)
    job = get_current_job()
    results = []
    artifacts = []

    with SessionLocal() as db:
        update_task_status(db, task_id, status="running", completed=0)
        try:
            for index, image_path in enumerate(image_paths, start=1):
                path = Path(image_path)
                response = detector.predict(image_bytes=path.read_bytes(), filename=path.name)
                results.append(response.model_dump())
                artifacts.append(render_detection_artifact(settings, task_id, path, response))

                update_task_status(db, task_id, status="running", completed=index)
                if job is not None:
                    job.meta["completed"] = index
                    job.save_meta()

            payload = {
                "task_id": task_id,
                "total": len(results),
                "completed": len(results),
                "results": results,
                "artifacts": artifacts,
            }
            result_path = write_task_result(settings, task_id, payload)
            replace_task_artifacts(db, task_id, artifacts)
            update_task_status(
                db,
                task_id,
                status="succeeded",
                completed=len(results),
                result_path=result_path,
            )
            return payload
        except Exception as exc:
            update_task_status(db, task_id, status="failed", error=str(exc))
            raise
