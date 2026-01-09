"""
Main entry point for the RKLLM API Server.
"""
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from app.routers import chat_router
from app.services.chat_service import chat_service
from app.core.config import settings
from app.utils.logger import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for the FastAPI app.
    Handles startup and shutdown events.
    """
    # pylint: disable=redefined-outer-name, unused-argument
    # Startup
    logger = setup_logging()
    logger.info("Starting RKLLM API Server...")
    try:
        chat_service.initialize_model()
    except Exception as e: # pylint: disable=broad-exception-caught
        logger.error("Failed to initialize model: %s", e)
        # We might want to exit or just log, but app will start.
        # Requests will fail if model not loaded.

    yield

    # Shutdown
    logger.info("Shutting down...")
    chat_service.release()

app = FastAPI(
    title="RKLLM OpenAI API",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(chat_router.router)

@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
