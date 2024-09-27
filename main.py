import asyncio
import json
from datetime import datetime, timezone
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from aio_pika import connect_robust, ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from pydantic import BaseModel
from typing import Optional

from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.tenant_prompts import router as tenant_prompt_router
from app.api.v1.rag import router as rag_router
from app.core.database import engine, Base
from app.services.rag_service import rag_pipeline
from contextlib import asynccontextmanager
from app.core.prompt import PROMPT_TEMPLATE

RABBITMQ_HOST = settings.RABBITMQ_HOST
RABBITMQ_PORT = settings.RABBITMQ_PORT
RABBITMQ_USERNAME = settings.RABBITMQ_USERNAME
RABBITMQ_PASSWORD = settings.RABBITMQ_PASSWORD
TOPIC_EXCHANGE_NAME = settings.TOPIC_EXCHANGE_NAME
SESSION_QUEUE_TEMPLATE = settings.SESSION_QUEUE_TEMPLATE
AI_MESSAGE_QUEUE = settings.AI_MESSAGE_QUEUE
RAG_PROMPT_TEMPLATE = PROMPT_TEMPLATE

# In-memory store for received messages (for prototype)
received_messages = []


class ReceivedMessage(BaseModel):
    session_id: str
    sender: str
    content: str
    type: str
    tenant_id: str
    user_type: str
    receiver: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Establish robust connection to RabbitMQ
        connection = await connect_robust(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            login=RABBITMQ_USERNAME,
            password=RABBITMQ_PASSWORD,
        )

        # Create a channel
        channel = await connection.channel()

        # Store the connection and channel in the FastAPI app state
        app.state.connection = connection
        app.state.channel = channel

        # Declare or get the queue
        queue = await channel.declare_queue(AI_MESSAGE_QUEUE, durable=True)

        # Start consuming messages from the AI queue
        await queue.consume(on_message_received)
        print(f"[*] Started consuming from queue: {AI_MESSAGE_QUEUE}")

        yield
    finally:
        # Close the RabbitMQ connection gracefully on shutdown
        await app.state.connection.close()
        print("[*] Connection to RabbitMQ closed")


app = FastAPI(title="AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow credentials (e.g., cookies, authorization headers)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, Content-Type, etc.)
)

# Create the tables in the database
Base.metadata.create_all(bind=engine)

app.include_router(tenant_prompt_router, prefix="/api/v1/tenant_prompts", tags=["Tenant Prompts"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])


async def on_message_received(message: AbstractIncomingMessage):
    async with message.process():
        try:
            # Decode and parse the incoming message
            msg_content = message.body.decode()
            msg_json = json.loads(msg_content)
            print(f"[>] Received message: {msg_json}")

            # Validate and store the message
            received_msg = ReceivedMessage(**msg_json)
            received_messages.append(received_msg)

            # Send the AI reply
            await send_acknowledgement_message(received_msg)  
            await send_reply_message(received_msg)

        except json.JSONDecodeError:
            print("[!] Failed to decode message")
        except Exception as e:
            print(f"[!] Error processing message: {e}")

async def publish_message_to_queue(received_msg: ReceivedMessage, message_type: str, content: str = ""):
    """
    Helper method to publish a message to the user's queue.
    This method handles message creation, queue declaration, and message publishing.
    """
    current_timestamp = datetime.now(timezone.utc).isoformat()
    reply_message = {
        "session_id": received_msg.session_id,
        "sender": "ai",
        "content": content,
        "type": message_type,
        "tenant_id": received_msg.tenant_id,
        "user_type": "AI",
        "SourceType": "AI",
        "receiver": received_msg.sender,
        "timestamp": current_timestamp
    }

    # Determine the user queue name based on session ID
    user_queue_name = SESSION_QUEUE_TEMPLATE.format(session_id=received_msg.session_id)
    print(f"Publishing message to default exchange with routing_key: {user_queue_name}")

    # Ensure the user queue exists. If not, declare it.
    user_queue = await app.state.channel.declare_queue(
        user_queue_name, durable=True
    )

    # Publish the message to the default exchange
    await app.state.channel.default_exchange.publish(
        Message(
            body=json.dumps(reply_message).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key=user_queue_name  # Routing key is the queue name
    )

    print(f"[<] Sent {message_type} message to user queue: {user_queue_name}")


async def send_reply_message(received_msg: ReceivedMessage):
    """
    Sends a reply message back to the customer with modified sender and SourceType.
    Utilizes the default exchange for direct messaging to user queues.
    """
    reply_content: str = rag_pipeline(received_msg.content, received_msg.tenant_id, RAG_PROMPT_TEMPLATE)
    await publish_message_to_queue(received_msg, "CHAT", reply_content)


async def send_acknowledgement_message(received_msg: ReceivedMessage):
    """
    Sends an acknowledgement message back to the customer to notify AI processing state.
    Utilizes the default exchange for direct messaging to user queues.
    """
    await publish_message_to_queue(received_msg, "ACKNOWLEDGEMENT")



@app.get("/status", summary="Check if messages have been received")
async def get_status():
    if received_messages:
        return {"status": "received", "message_count": len(received_messages)}
    else:
        return {"status": "no messages received yet"}


@app.get("/messages", summary="Retrieve all received messages")
async def get_messages():
    return received_messages

if __name__ == '__main__':
    uvicorn.run(app="main:app", host="127.0.0.1", port=8001, reload=True)
