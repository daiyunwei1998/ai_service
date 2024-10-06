import time
import json
import logging
from datetime import datetime, timezone

import aioredis
from fastapi import FastAPI, HTTPException
from aio_pika import connect_robust, Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from pydantic import BaseModel, constr, Field
from typing import Optional

from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.tenant_prompts import router as tenant_prompt_router
from app.api.v1.rag import router as rag_router, input_token_price, output_token_price
from app.core.database import engine, Base
from app.schemas.ai_reply import AIReply
from app.schemas.chat_message import ChatMessage, MessageType, SourceType
from app.services.mongodb_service import mongodb_service
from app.services.llm_service import rag_pipeline, summarize
from contextlib import asynccontextmanager
from app.core.prompt import RAG_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE

RABBITMQ_HOST = settings.RABBITMQ_HOST
RABBITMQ_PORT = settings.RABBITMQ_PORT
RABBITMQ_USERNAME = settings.RABBITMQ_USERNAME
RABBITMQ_PASSWORD = settings.RABBITMQ_PASSWORD
TOPIC_EXCHANGE_NAME = settings.TOPIC_EXCHANGE_NAME
SESSION_QUEUE_TEMPLATE = settings.SESSION_QUEUE_TEMPLATE
AGENT_QUEUE_TEMPLATE = settings.AGENT_QUEUE_TEMPLATE
AI_MESSAGE_QUEUE = settings.AI_MESSAGE_QUEUE
RAG_PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE

# In-memory store for received messages (for prototype)
received_messages = []


REDIS_HOST = settings.REDIS_HOST
REDIS_PASSWORD =  settings.REDIS_PASSWORD
REDIS_PORT = settings.REDIS_PORT
# Initialize Redis client (ensure this is done once, e.g., at startup)
redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
redis_client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)

class ReceivedMessage(BaseModel):
    session_id: Optional[str] = None
    sender: str
    content: Optional[str] = None
    type: str
    tenant_id: str
    user_type: Optional[str] = None
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
        logging.info(f"[*] Started consuming from queue: {AI_MESSAGE_QUEUE}")

        yield
    finally:
        # Close the RabbitMQ connection gracefully on shutdown
        await app.state.connection.close()
        logging.info("[*] Connection to RabbitMQ closed")
        await mongodb_service.close_connection()
        logging.info("[*] Connection to mongodb closed")


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
            logging.info(f"[>] Received message: {msg_json}")

            # Validate and store the message
            received_msg = ReceivedMessage(**msg_json)
            received_messages.append(received_msg)


            # Send the AI reply
            logging.info(f"[>] CHAT")
            await reply_with_rag(received_msg)

        except json.JSONDecodeError:
            logging.error("[!] Failed to decode message")
        except Exception as e:
            logging.error(f"[!] Error processing message: {e}")

async def reply_with_rag(received_msg: ReceivedMessage):
    await send_acknowledgement_message(received_msg)
    response = await send_reply_message(received_msg)

    receiver = received_msg.sender
    query = received_msg.content
    tenant_id = received_msg.tenant_id

    ai_reply = AIReply.from_openai_completion(receiver, query, response, tenant_id, input_token_price,
                                              output_token_price)

    await mongodb_service.ensure_index(received_msg.tenant_id)
    await mongodb_service.save_ai_reply(ai_reply)

    timestamp  = time.time()

    # Create ChatMessage instance for the reply
    chat_message = ChatMessage(
        session_id=received_msg.session_id,
        type=MessageType.CHAT,
        content=response.choices[0].message.content,
        sender="AI",
        sender_name="AI",
        receiver=received_msg.sender,
        tenant_id=tenant_id,
        timestamp=timestamp,
        source=SourceType.AI,
        user_type="agent",
        customer_id=received_msg.sender
    )

    logging.info(f"[>] Chat message: {chat_message}")

    # Serialize ChatMessage to JSON
    chat_message_json = chat_message.to_json()

    # Define Redis key following the Java service pattern
    redis_key = f"tenant:{tenant_id}:chat:customer_messages:{received_msg.session_id}"

    # Push the serialized message to Redis
    await redis_client.rpush(redis_key, chat_message_json)
    print(f"Saved message to Redis under key: {redis_key}")


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
    logging.info(f"Publishing message to default exchange with routing_key: {user_queue_name}")

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

    logging.info(f"[<] Sent {message_type} message to user queue: {user_queue_name}")

async def publish_summary_to_queue(received_msg: ReceivedMessage, content: str = ""):
    """
    Helper method to publish a message to the user's queue.
    This method handles message creation, queue declaration, and message publishing.
    """
    current_timestamp = datetime.now(timezone.utc).isoformat()
    reply_message = {
        "session_id": None,
        "sender": "ai",
        "content": content,
        "type": "SUMMARY",
        "tenant_id": received_msg.tenant_id,
        "user_type": "AI",
        "SourceType": "AI",
        "receiver": received_msg.sender,
        "timestamp": current_timestamp
    }

    # Determine the user queue name based on session ID
    agent_queue_name = AGENT_QUEUE_TEMPLATE.format(tenant_id=received_msg.tenant_id)
    logging.info(f"Publishing message to agent with routing_key: {agent_queue_name}")

    # Ensure the user queue exists. If not, declare it.
    agent_queue = await app.state.channel.declare_queue(
        agent_queue_name, durable=True
    )

    # Publish the message to the default exchange
    await app.state.channel.default_exchange.publish(
        Message(
            body=json.dumps(reply_message).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key=agent_queue_name  # Routing key is the queue name
    )

    logging.info(f"[<] Sent summary message to user queue: {agent_queue_name}")


async def send_reply_message(received_msg: ReceivedMessage):
    """
    Sends a reply message back to the customer with modified sender and SourceType.
    Utilizes the default exchange for direct messaging to user queues.
    """
    response = rag_pipeline(received_msg.content, received_msg.tenant_id, RAG_PROMPT_TEMPLATE, received_msg.session_id, received_msg.sender)
    reply_content = response.choices[0].message.content
    await publish_message_to_queue(received_msg, "CHAT", reply_content)
    return response

async def send_acknowledgement_message(received_msg: ReceivedMessage):
    """
    Sends an acknowledgement message back to the customer to notify AI processing state.
    Utilizes the default exchange for direct messaging to user queues.
    """
    await publish_message_to_queue(received_msg, "ACKNOWLEDGEMENT")

async def handle_summary_request(request):
    response = summarize(request.tenant_id, SUMMARY_PROMPT_TEMPLATE, request.customer_id)
    # await publish_summary_to_queue(received_msg, reply_content)
    return response


@app.get("/status", summary="Check if messages have been received")
async def get_status():
    if received_messages:
        return {"status": "received", "message_count": len(received_messages)}
    else:
        return {"status": "no messages received yet"}


@app.get("/messages", summary="Retrieve all received messages")
async def get_messages():
    return received_messages


from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="The tenant ID (numeric)"
    )
    customer_id: str = Field(
        ...,
        min_length=1,
        description="The Customer ID (alphanumeric, underscores, hyphens allowed)"
    )

    class Config:
        from_attributes = True


@app.post("/summary", response_model=dict)
async def get_summary(request: SummaryRequest):
    logging.info(f"[>] Request Summary - tenant_id: {request.tenant_id}, customer_id: {request.customer_id}")
    response = await handle_summary_request(request)

    receiver = "AGENT"
    query = ""
    tenant_id = request.tenant_id

    ai_reply = AIReply.from_openai_completion(receiver, query, response, tenant_id, input_token_price,
                                              output_token_price)

    await mongodb_service.ensure_index(request.tenant_id)
    await mongodb_service.save_ai_reply(ai_reply)

    summary = response.choices[0].message.content

    summary_data = {
        "tenant_id": request.tenant_id,
        "customer_id": request.customer_id,
        "summary": summary
    }

    logging.info(f"[<] Response Summary - {summary_data}")
    return summary_data


