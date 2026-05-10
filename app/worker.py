from pathlib import Path

from rq import get_current_job

from app.config import get_settings
from app.services.detector import YOLODetector
from app.services.storage import write_task_result


def run_batch_detect(task_id: str, image_paths: list[str]) -> dict:
    settings = get_settings()
    detector = YOLODetector(settings)
    job = get_current_job()
    results = []

    for index, image_path in enumerate(image_paths, start=1):
        path = Path(image_path)
        response = detector.predict(image_bytes=path.read_bytes(), filename=path.name)
        results.append(response.model_dump())

        if job is not None:
            job.meta["completed"] = index
            job.save_meta()

    payload = {
        "task_id": task_id,
        "total": len(results),
        "completed": len(results),
        "results": results,
    }
    write_task_result(settings, task_id, payload)
    return payload

