from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "vision-inference-backend"
    app_env: str = "local"
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640
    max_upload_mb: int = 10
    max_video_upload_mb: int = 200
    video_frame_stride: int = 30
    max_video_frames: int = 120
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "vision-tasks"
    upload_dir: str = "uploads"
    output_dir: str = "outputs"
    database_url: str = "sqlite:///./vision.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
