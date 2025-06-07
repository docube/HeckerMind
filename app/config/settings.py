# app/config/settings.py

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==== Your environment variables here ====
    OPENAI_API_KEY: str
    DATABASE_URL: str = "sqlite:///./test.db"
    ENVIRONMENT: str = "development"  # dev, prod, staging
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Use caching so settings are only loaded once during runtime
@lru_cache()
def get_settings():
    return Settings()
