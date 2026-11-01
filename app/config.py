from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str = ""
    database_url: str = "postgresql://srbot:srbot_dev_password@localhost:5432/srbot"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 1800

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""


settings = Settings()
