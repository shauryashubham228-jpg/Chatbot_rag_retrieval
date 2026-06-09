from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    langsmith_api_key: str = ""
    langchain_project: str = "lagorii-agent"
    langchain_tracing_v2: str = "true"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def apply_env(self) -> None:
        os.environ["GROQ_API_KEY"]            = self.groq_api_key
        os.environ["LANGCHAIN_API_KEY"]       = self.langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"]    = self.langchain_tracing_v2
        os.environ["LANGCHAIN_PROJECT"]       = self.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"]      = self.langchain_endpoint


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
settings.apply_env()
