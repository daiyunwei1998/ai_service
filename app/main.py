import asyncio
import json
from fastapi import FastAPI, HTTPException
from aio_pika import connect_robust, ExchangeType, Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage
from pydantic import BaseModel
from typing import Optional
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USERNAME,
    RABBITMQ_PASSWORD,
    TENANT_ID,
    TOPIC_EXCHANGE_NAME,
    AI_MESSAGE_QUEUE,
    SESSION_QUEUE_TEMPLATE
)

app = FastAPI(title="FastAPI Chat Listener Service")

# In-memory store for received messages (for prototype)
received_messages = []

class ReceivedMessage(BaseModel):
    sessionId: str 
    sender: str
    content: str
    type: str
    tenantId: str
    userType: str
    receiver: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    # Establish connection to RabbitMQ
    app.state.connection = await connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USERNAME,
        password=RABBITMQ_PASSWORD,
    )
    app.state.channel = await app.state.connection.channel()
    
    # Declare the topic exchange for customer messages
    app.state.topic_exchange = await app.state.channel.declare_exchange(
        TOPIC_EXCHANGE_NAME, ExchangeType.TOPIC, durable=True
    )
    
    # Declare the AI message queue and bind it to the topic exchange
    app.state.customer_queue = await app.state.channel.declare_queue(
        AI_MESSAGE_QUEUE, durable=True
    )
    await app.state.customer_queue.bind(
        app.state.topic_exchange, routing_key=f"{TENANT_ID}.customer_message"
    )
    
    # Start consuming messages from the AI queue
    await app.state.customer_queue.consume(on_message_received)
    print(f"[*] Started consuming from queue: {AI_MESSAGE_QUEUE}")

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
            await send_reply_message(received_msg)
            
        except json.JSONDecodeError:
            print("[!] Failed to decode message")
        except Exception as e:
            print(f"[!] Error processing message: {e}")

async def send_reply_message(received_msg: ReceivedMessage):
    """
    Sends a reply message back to the customer with modified sender and SourceType.
    Utilizes the default exchange for direct messaging to user queues.
    """
    reply_message = {
        "sessionId": received_msg.sessionId,
        "sender": "ai",
        "content": f"AI Reply: {received_msg.content}", 
        "type": "reply",
        "tenantId": received_msg.tenantId,
        "userType": "AI",
        "SourceType": "AI",
        "receiver": received_msg.sender,  
    }
    
    # Determine the user queue name based on session ID
    user_queue_name = SESSION_QUEUE_TEMPLATE.format(session_id=received_msg.sessionId)
    print(f"Publishing message to default exchange with routing_key: {user_queue_name}")
    
    # Ensure the user queue exists. If not, declare it.
    user_queue = await app.state.channel.declare_queue(
        user_queue_name, durable=True
    )
    # No need to bind to default exchange; it's automatically bound via queue name
    
    # Publish the reply message to the default exchange
    await app.state.channel.default_exchange.publish(
        Message(
            body=json.dumps(reply_message).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
        ),
        routing_key=user_queue_name  # Routing key is the queue name
    )
    
    print(f"[<] Sent reply to user queue: {user_queue_name}")

@app.on_event("shutdown")
async def shutdown_event():
    # Close the RabbitMQ connection gracefully
    await app.state.connection.close()
    print("[*] Connection to RabbitMQ closed")

@app.get("/status", summary="Check if messages have been received")
async def get_status():
    if received_messages:
        return {"status": "received", "message_count": len(received_messages)}
    else:
        return {"status": "no messages received yet"}

@app.get("/messages", summary="Retrieve all received messages")
async def get_messages():
    return received_messages
