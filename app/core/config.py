"""
Configuration settings for the application.
"""
import os
import yaml
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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

    model_config = SettingsConfigDict(
        yaml_file=os.getenv("CONFIG_FILE", "config.yaml"),
        extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads configuration from a YAML file.
    """
    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        encoding = self.config.get("env_file_encoding")
        file_content_json = yaml.safe_load(
            self.file_path.read_text(encoding)
        )
        field_value = file_content_json.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: Any, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        config_file = self.settings_cls.model_config.get("yaml_file")
        if not config_file:
            return d

        path = os.path.abspath(config_file)
        if not os.path.exists(path):
            return d

        with open(path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Flatten the config if it's nested or just use as is
                # Here we assume a flat structure matching the model fields
                d.update(yaml_config)
        return d

settings = Settings()
