"""Pensive API entry point."""
import logging

import uvicorn

from api.routes import app  # noqa: F401 — exposed for uvicorn main:app
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    Config.validate()
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
    )
