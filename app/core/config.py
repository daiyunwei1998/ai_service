import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_COMPLETION_MODEL: str = "gpt-4o-mini"
    INPUT_TOKEN_PRICE: float = 0.000150 / 1000
    OUTPUT_TOKEN_PRICE: float = 0.000600 / 1000

    
    # Milvus Database Configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = os.getenv("MILVUS_PORT", 19530)

    # Rabbit MQ
    RABBITMQ_HOST: str = "localhost" #"203.204.185.67"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USERNAME: str = "root"
    RABBITMQ_PASSWORD:str = "admin1234"

    # Exchange and Queue Names
    TOPIC_EXCHANGE_NAME:str = "amq.topic"  # Topic exchange for customer messages
    AI_MESSAGE_QUEUE:str = "ai_message"
    SESSION_QUEUE_TEMPLATE:str = "messages-user{session_id}"

    # MySQL Configuration (Loaded from .env file)
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_DB: str

    # MongoDB
    MONGO_HOST:str  = os.getenv('MONGO_HOST', 'localhost')
    MONGO_USERNAME:str = os.getenv('MONGO_USERNAME', 'default_username')
    MONGO_PASSWORD:str = os.getenv('MONGO_PASSWORD', 'default_password')

    MONGODB_URL:str = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:27017/"

    DATABASE_NAME:str = "ai_replies_db"

    @property
    def database_url(self):
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    class Config:
        env_file = ".env"

# Instantiate the settings object
settings = Settings()