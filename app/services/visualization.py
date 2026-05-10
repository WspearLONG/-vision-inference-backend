from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.config import Settings
from app.schemas import DetectResponse
from app.services.storage import task_artifact_dir


BOX_COLOR = (0, 180, 120)
TEXT_COLOR = (255, 255, 255)
LABEL_BG_COLOR = (0, 120, 90)


def render_detection_artifact(
    settings: Settings,
    task_id: str,
    image_path: Path,
    detection: DetectResponse,
) -> dict:
    output_dir = task_artifact_dir(settings, task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for item in detection.detections:
            box = item.box
            draw.rectangle(
                [(box.x1, box.y1), (box.x2, box.y2)],
                outline=BOX_COLOR,
                width=3,
            )
            label = f"{item.label} {item.confidence:.2f}"
            left, top, right, bottom = draw.textbbox((box.x1, box.y1), label, font=font)
            label_height = bottom - top + 6
            label_width = right - left + 8
            label_y = max(0, box.y1 - label_height)
            draw.rectangle(
                [(box.x1, label_y), (box.x1 + label_width, label_y + label_height)],
                fill=LABEL_BG_COLOR,
            )
            draw.text((box.x1 + 4, label_y + 3), label, fill=TEXT_COLOR, font=font)

        image.save(output_path)

    return {
        "filename": output_path.name,
        "path": str(output_path),
        "url": f"/artifacts/{task_id}/images/{output_path.name}",
    }

