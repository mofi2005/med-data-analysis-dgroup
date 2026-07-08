from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Medical Data Analysis D-Group (Member 2)"
    debug: bool = True
    database_url: str = "sqlite:///./data/medical_d.db"
    upload_dir: str = "data/uploads"
    export_dir: str = "data/exports"
    member1_base_url: str = "http://127.0.0.1:8001"


settings = Settings()
