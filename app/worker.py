from pathlib import Path

from rq import get_current_job

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.services.detector import YOLODetector
from app.services.storage import task_frame_dir, write_task_result
from app.services.tasks import replace_task_artifacts, update_task_status, update_task_total
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


def run_video_detect(task_id: str, video_path: str) -> dict:
    import cv2

    init_db()
    settings = get_settings()
    detector = YOLODetector(settings)
    job = get_current_job()
    video = Path(video_path)
    frame_dir = task_frame_dir(settings, task_id)
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, settings.video_frame_stride)
    estimated_total = min(settings.max_video_frames, (frame_count + stride - 1) // stride) if frame_count else 0

    results = []
    artifacts = []
    processed = 0
    frame_index = 0

    with SessionLocal() as db:
        update_task_status(db, task_id, status="running", completed=0)
        if estimated_total:
            update_task_total(db, task_id, estimated_total)

        try:
            while processed < settings.max_video_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % stride != 0:
                    frame_index += 1
                    continue

                frame_name = f"frame_{frame_index:06d}.jpg"
                frame_path = frame_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise ValueError(f"cannot encode frame {frame_index}")

                response = detector.predict(image_bytes=buffer.tobytes(), filename=frame_name)
                result = response.model_dump()
                result["frame_index"] = frame_index
                results.append(result)
                artifacts.append(render_detection_artifact(settings, task_id, frame_path, response))

                processed += 1
                update_task_status(db, task_id, status="running", completed=processed)
                if job is not None:
                    job.meta["completed"] = processed
                    if estimated_total:
                        job.meta["total"] = estimated_total
                    job.save_meta()

                frame_index += 1

            if processed == 0:
                raise ValueError("video contains no readable frames")

            payload = {
                "task_id": task_id,
                "source_video": str(video),
                "frame_stride": stride,
                "total": processed,
                "completed": processed,
                "results": results,
                "artifacts": artifacts,
            }
            result_path = write_task_result(settings, task_id, payload)
            replace_task_artifacts(db, task_id, artifacts)
            update_task_total(db, task_id, processed)
            update_task_status(
                db,
                task_id,
                status="succeeded",
                completed=processed,
                result_path=result_path,
            )
            return payload
        except Exception as exc:
            update_task_status(db, task_id, status="failed", error=str(exc))
            raise
        finally:
            capture.release()
