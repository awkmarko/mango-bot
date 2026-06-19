from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"
    ollama_timeout: float = 120.0
    max_history_turns: int = 20

    embedding_model: str = "nomic-embed-text"
    chroma_db_path: str = "./chroma_db"
    force_reindex: bool = False

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "clickshop"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
