import os

from celery import Celery

# Configure Celery to use Redis as the broker and result backend.
# In Docker, redis host will be 'redis'. Locally, it would be 'localhost'.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("oskar_tasks", broker=REDIS_URL, backend=REDIS_URL, include=["src.tasks"])

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minute limit for heavy ML tasks
)

if __name__ == "__main__":
    celery_app.start()
