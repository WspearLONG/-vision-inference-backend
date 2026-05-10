import argparse
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a test image to the detection API.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/detect")
    args = parser.parse_args()

    with args.image.open("rb") as image_file:
        response = httpx.post(
            args.url,
            files={"image": (args.image.name, image_file, "image/jpeg")},
            timeout=60,
            trust_env=False,
        )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
