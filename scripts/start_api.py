import uvicorn

from scripts.migrate_db import main as migrate_db


def main() -> None:
    migrate_db()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

