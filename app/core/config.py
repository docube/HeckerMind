# app/core/config.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    # Add more configs if needed later

    class Config:
        env_file = ".env"

settings = Settings()
