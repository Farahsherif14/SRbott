from pydantic_settings import BaseSettings, SettingsConfigDict

<<<<<<< HEAD
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_relay_url: str = "http://127.0.0.1:8765/extract"
    telegram_bot_token: str = ""
    database_url: str = "sqlite:///./service_requests.db"
    redis_url: str = "redis://localhost:6379/0"
    whisper_model: str = "small"
    session_ttl_seconds: int = 1800
    classifier_path: str = "data/category_model.joblib"
=======

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str = ""
    database_url: str = "postgresql://srbot:srbot_dev_password@localhost:5432/srbot"
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 1800

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""

>>>>>>> ml-classifier

settings = Settings()
