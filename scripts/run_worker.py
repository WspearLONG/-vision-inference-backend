import argparse
import os
import sys
from pathlib import Path

from redis import Redis
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ worker.")
    parser.add_argument("--burst", action="store_true", help="Exit after all queued jobs are processed.")
    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.getenv("QUEUE_NAME", "vision-tasks")
    redis = Redis.from_url(redis_url)
    queue = Queue(queue_name, connection=redis)
    worker = SimpleWorker([queue], connection=redis)
    if sys.platform.startswith("win"):
        worker.death_penalty_class = TimerDeathPenalty
    worker.work(burst=args.burst)


if __name__ == "__main__":
    main()
