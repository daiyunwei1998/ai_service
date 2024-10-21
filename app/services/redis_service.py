import logging

import redis
import json


from app.core.config import settings

# ==============================
# Initialize Redis Client Once
# ==============================

try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True  # Automatically decode bytes to strings
    )

    # Test the connection
    if not redis_client.ping():
        raise ConnectionError("Unable to connect to Redis.")
except redis.AuthenticationError:
    raise ValueError("Authentication failed when connecting to Redis.")
except redis.ConnectionError:
    raise ConnectionError("Could not connect to Redis. Please check the connection parameters.")
except Exception as e:
    raise Exception(f"An error occurred while connecting to Redis: {str(e)}")


# ==============================
# Service Method Implementation
# ==============================

def get_formatted_chat_history(
        user_id: str,
        tenant_id: str
) -> str:
    """
    Retrieves and formats chat history for a given user_id from Redis.

    Args:
        user_id (str): The user ID to retrieve chat history for.
        tenant_id (str): Tenant ID used in Redis keys. Default is 'tenant_72'.

    Returns:
        str: Formatted chat history as "sender: message" pairs separated by newlines.

    Raises:
        ValueError: If session ID or chat history is not found.
        Exception: For other unexpected errors.
    """

    try:
        # Construct the key to retrieve session_id
        session_key = f"tenant:{tenant_id}:user_session:{user_id}"

        # Retrieve session_id from Redis
        session_id = redis_client.get(session_key)

        if session_id:  # Check if session_id is not None
            session_id = session_id.replace('"', '')  # Remove double quotes


        logging.info("session_id found while getting chat_history: " + session_id)
        if not session_id:
            raise ValueError("Session ID not found for the provided user_id.")

        # Construct the key to retrieve chat history
        chat_key = f"tenant:{tenant_id}:chat:customer_messages:{session_id}"

        # Retrieve chat history as a list from Redis
        # Assuming the chat messages are stored as a Redis list
        chat_messages = redis_client.lrange(chat_key, 0, -1)
        if not chat_messages:
            raise ValueError("No chat history found for the session." + chat_key)

        formatted_messages = []
        for message_str in chat_messages:
            try:
                # Parse the JSON string into a Python dictionary
                message = json.loads(message_str)

                # Extract sender and content
                sender = message.get("sender")
                content = message.get("content")

                if sender and content:
                    formatted_messages.append(f"{sender}: {content}")

            except json.JSONDecodeError:
                # Handle invalid JSON formats
                logging.error("Invalid message format.")

        # Join all messages into a single string separated by newlines
        return "\n".join(formatted_messages)

    except ValueError as ve:
        # Re-raise known value errors
        raise ve
    except Exception as e:
        # Catch-all for unexpected errors
        raise Exception(f"An error occurred while retrieving chat history: {str(e)}")
