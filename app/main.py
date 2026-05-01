from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.questions import router as questions_router
from app.core.config import get_settings
from app.core.logging import configure_logging

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Local RAG backend for working with a private knowledge base.",
    )

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(documents_router, prefix=settings.api_prefix)
    app.include_router(questions_router, prefix=settings.api_prefix)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def read_ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
