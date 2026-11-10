from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_relay_url: str = "http://127.0.0.1:8765/extract"
    telegram_bot_token: str = ""
    database_url: str = "sqlite:///./service_requests.db"
    redis_url: str = "redis://localhost:6379/0"
    tawasul_model_path: str = "tawasul_egy_stt.nemo"
    ollama_host: str = "http://localhost:11434"
    session_ttl_seconds: int = 1800
    classifier_path: str = "models/category_clf.joblib"

settings = Settings()
