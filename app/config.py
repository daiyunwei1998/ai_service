# config.py
RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "root"
RABBITMQ_PASSWORD = "admin1234"
TENANT_ID = "tenant1"

# Exchange and Queue Names
TOPIC_EXCHANGE_NAME = "amq.topic"  # Topic exchange for customer messages
AI_MESSAGE_QUEUE = f"{TENANT_ID}.ai_message"
SESSION_QUEUE_TEMPLATE = "messages-user{session_id}"  # Pattern for user-specific queues
