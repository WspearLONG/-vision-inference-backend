from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class RootResponse(BaseModel):
    service: str
    docs: str
    health: str
    endpoints: dict[str, str]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class DetectResponse(BaseModel):
    filename: str
    width: int
    height: int
    detections: list[Detection]
    model: str
    confidence_threshold: float | None = None
    image_size: int | None = None


class ModelResponse(BaseModel):
    id: str
    name: str
    path: str
    task_type: str
    default_confidence: float
    default_image_size: int
    description: str | None = None


class BatchDetectCreateResponse(BaseModel):
    task_id: str
    status: str
    total: int
    model: str | None = None
    confidence_threshold: float | None = None
    image_size: int | None = None


class VideoTaskCreateResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    frame_stride: int
    max_frames: int
    model: str | None = None
    confidence_threshold: float | None = None
    image_size: int | None = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    total: int | None = None
    completed: int | None = None
    result_url: str | None = None
    error: str | None = None


class Artifact(BaseModel):
    filename: str
    path: str
    url: str


class TaskArtifactsResponse(BaseModel):
    task_id: str
    artifacts: list[Artifact]
