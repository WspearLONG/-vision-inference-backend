from typing import Protocol

from fastapi import HTTPException, status
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.job import Job, NoSuchJobError

from app.config import Settings
from app.schemas import TaskStatusResponse


class TaskQueue(Protocol):
    def enqueue_batch_detect(self, task_id: str, image_paths: list[str]) -> str:
        ...

    def enqueue_video_detect(self, task_id: str, video_path: str) -> str:
        ...

    def get_status(self, task_id: str) -> TaskStatusResponse:
        ...


class RedisTaskQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis = Redis.from_url(settings.redis_url)
        self.queue = Queue(settings.queue_name, connection=self.redis)

    def enqueue_batch_detect(self, task_id: str, image_paths: list[str]) -> str:
        try:
            job = self.queue.enqueue(
                "app.worker.run_batch_detect",
                task_id,
                image_paths,
                job_id=task_id,
                meta={"total": len(image_paths), "completed": 0},
                result_ttl=86400,
                failure_ttl=86400,
            )
            return job.id
        except RedisError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis queue is unavailable",
            ) from exc

    def enqueue_video_detect(self, task_id: str, video_path: str) -> str:
        try:
            job = self.queue.enqueue(
                "app.worker.run_video_detect",
                task_id,
                video_path,
                job_id=task_id,
                meta={"total": None, "completed": 0},
                result_ttl=86400,
                failure_ttl=86400,
                job_timeout=3600,
            )
            return job.id
        except RedisError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis queue is unavailable",
            ) from exc

    def get_status(self, task_id: str) -> TaskStatusResponse:
        try:
            job = Job.fetch(task_id, connection=self.redis)
        except NoSuchJobError:
            return TaskStatusResponse(task_id=task_id, status="not_found")
        except RedisError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis queue is unavailable",
            ) from exc

        rq_status = job.get_status(refresh=True)
        status_map = {
            "queued": "pending",
            "started": "running",
            "finished": "succeeded",
            "failed": "failed",
            "deferred": "pending",
            "scheduled": "pending",
        }
        mapped_status = status_map.get(str(rq_status), str(rq_status))
        total = job.meta.get("total")
        completed = job.meta.get("completed")
        error = None
        if mapped_status == "failed":
            error = job.exc_info

        return TaskStatusResponse(
            task_id=task_id,
            status=mapped_status,
            total=total,
            completed=completed,
            result_url=f"/api/v1/tasks/{task_id}/result" if mapped_status == "succeeded" else None,
            error=error,
        )
