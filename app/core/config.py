import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL:str = "text-embedding-3-small"
    
    # Milvus Database Configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = os.getenv("MILVUS_PORT", 19530)

    # Rabbit MQ
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USERNAME: str = "root"
    RABBITMQ_PASSWORD:str = "admin1234"
    TENANT_ID:str = "tenant1"

    # Exchange and Queue Names
    TOPIC_EXCHANGE_NAME:str = "amq.topic"  # Topic exchange for customer messages
    AI_MESSAGE_QUEUE:str = f"{TENANT_ID}.ai_message"
    SESSION_QUEUE_TEMPLATE:str = "messages-user{session_id}"  # Pattern for user-specific queues

    # MySQL Configuration (Loaded from .env file)
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_DB: str

    @property
    def database_url(self):
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    class Config:
        env_file = ".env"

# Instantiate the settings object
settings = Settings()