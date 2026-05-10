from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


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

