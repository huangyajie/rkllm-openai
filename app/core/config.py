"""
Configuration settings for the application.
"""
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings model.
    """
    # Model Config
    MODEL_PATH: str
    TARGET_PLATFORM: str = "rk3588"
    MODEL_NAME: str = "rkllm-model"
    RKLLM_LIB_PATH: str = "lib/librkllmrt.so"
    LORA_MODEL_PATH: Optional[str] = None
    PROMPT_CACHE_PATH: Optional[str] = None

    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    API_KEY: Optional[str] = None

    # Inference Params
    MAX_CONTEXT_LEN: int = 4096
    MAX_NEW_TOKENS: int = 4096
    TOP_K: int = 1
    TOP_P: float = 0.9
    TEMPERATURE: float = 0.8
    REPEAT_PENALTY: float = 1.1
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0

    # Logging Config
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_MAX_BYTES: int = 1048576  # 1MB
    LOG_BACKUP_COUNT: int = 5

    class Config:
        """Pydantic config."""
        env_file = ".env"
        extra = "ignore" # Ignore extra fields in .env if any

settings = Settings()
