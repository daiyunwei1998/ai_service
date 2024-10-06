import json
import time
from typing import Union

from openai import OpenAI

import requests
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from openai import OpenAI

from pymilvus import Collection, connections
from app.services.language_service import detect_language
from app.services.redis_service import get_formatted_chat_history
from app.services.tenant_prompt_service import get_template_by_id, search_vectors_in_tenant_db
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)
CHAT_COMPLETION_MODEL = settings.CHAT_COMPLETION_MODEL
API_BASE_URL = "https://flashresponse.net/chat/api/v1/chats"
HANDOVER_ENDPOINT = f"{API_BASE_URL}/handover"

# Initialize connection to Milvus
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# Define the handover function schema
handover_function = {
    "name": "handover_to_agent",
    "description": "Triggers the handover of a customer session to a human agent.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A summary of the conversation so far."
            },
            "reason": {
                "type": "string",
                "description": "The reason for initiating the handover."
            }
        },
        "required": ["summary", "reason"]
    }
}


def trigger_handover(session_id: str, customer_id: str, tenant_id: str, summary: str, reason: str = "") -> bool:
    """
    Triggers the handover to a human agent via the RESTful API.
    """
    payload = {
        "session_id": session_id,
        "customer_id": customer_id,
        "tenant_id": tenant_id,
        "summary": summary,
        "reason": reason
    }

    headers = {
        "Content-Type": "application/json",
        #"Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(HANDOVER_ENDPOINT, json=payload, headers=headers)
        logging.info(f"posting to {HANDOVER_ENDPOINT} with data {payload}")
        if response.status_code == 202:
            logging.info("Handover to human agent initiated successfully.")
            return True
        else:
            logging.error(
                f"Failed to initiate handover. Status Code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Exception occurred while triggering handover: {e}")
        return False


def trigger_handover_with_retry(session_id, customer_id, tenant_id, summary, reason, retries=3, delay=2):
    """
    Implements a retry mechanism for the handover API call.
    """
    for attempt in range(retries):
        success = trigger_handover(session_id, customer_id, tenant_id, summary, reason)
        if success:
            return True
        else:
            logging.warning(f"Handover attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    logging.error("All handover attempts failed.")
    return False



def rag_pipeline(query_string: str, tenant_id: str, prompt_template: str, session_id: str, customer_id: str) -> Union[dict, str]:
    """
    Handles the RAG pipeline with integrated function calling for handover.
    Returns either a ChatCompletion object or a string.
    """
    # Detect the language of the query
    detected_lang = detect_language(query_string)

    # Retrieve relevant documents from the vector database using vector search
    relevant_docs = search_vectors_in_tenant_db(query_string, tenant_id=tenant_id)

    # Combine retrieved documents into a context string
    context = "\n".join(relevant_docs)
    logging.info("Retrieved context: \n" + context)

    # Construct the final prompt by combining template and query data
    prompt = prompt_template.format(document=context, language=detected_lang, question=query_string)

    # Define messages for the AI
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query_string},
    ]

    try:
        # Call OpenAI GPT to generate the response with function definitions
        response = client.chat.completions.create(
            model=CHAT_COMPLETION_MODEL,
            messages=messages,
            functions=[handover_function],
            function_call="auto",  # Let the AI decide whether to call the function
            temperature=0
        )
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        # Trigger handover due to API failure
        summary = f"Session ID: {session_id}\nCustomer ID: {customer_id}\nLast Query: {query_string}\nAI Response: None due to OpenAI API failure."
        reason = "OpenAI API failure."
        handover_success = trigger_handover_with_retry(
            session_id=session_id,
            customer_id=customer_id,
            tenant_id=tenant_id,
            summary=summary,
            reason=reason
        )
        if handover_success:
            # Return a string indicating handover
            return "請稍等，重試轉接中..."
        else:
            return "客服轉接失敗，請重試。"

    # Check if the AI wants to call a function
    choice = response.choices[0]
    if choice.finish_reason == "function_call":
        function_call = choice.message.function_call
        if function_call.name == "handover_to_agent":
            # Extract arguments provided by the AI
            arguments_str = function_call.arguments
            try:
                args = json.loads(arguments_str)
                summary = args.get("summary", "")
                reason = args.get("reason", "")
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing function arguments: {e}")
                summary = "Summary not provided."
                reason = "Invalid function arguments."

            # Append the missing parameters and trigger the handover
            handover_success = trigger_handover_with_retry(
                session_id=session_id,
                customer_id=customer_id,
                tenant_id=tenant_id,
                summary=summary,
                reason=reason
            )

            if handover_success:
                response.choices[0].message.content = "正在為您轉接人工客服，請稍等..."
                return response
            else:
                # Inform the user about the handover failure
                failure_message = "很抱歉，目前無法為您轉接人工客服，請稍後"
                return failure_message

    elif choice.message.content:
        # If the AI provided a regular response, return it as is
        return response  # Return the original ChatCompletion object

    else:
        # Handle case where AI did not generate a response
        summary = f"Session ID: {session_id}\nCustomer ID: {customer_id}\nLast Query: {query_string}\nAI Response: None"
        reason = "No response generated by AI."

        handover_success = trigger_handover_with_retry(
            session_id=session_id,
            customer_id=customer_id,
            tenant_id=tenant_id,
            summary=summary,
            reason=reason
        )

        if handover_success:
            # Return a string indicating handover
            return "Transferring you to a human agent. Please wait..."
        else:
            return "I'm experiencing some issues connecting you to a human agent. Please try again later."

def summarize(tenant_id: str, prompt_template: str, customer_id: str) -> ChatCompletion | str:
    # Get chat history from session (in redis)
    chat_history = get_formatted_chat_history(customer_id, tenant_id)
    logging.info("chat history:" + chat_history)
    # Summary with LLM
    prompt = prompt_template.format(history = chat_history)
    logging.info("prompt:" + prompt)
    messages = [
        {"role": "system", "content": prompt},
    ]
    response = client.chat.completions.create(model=CHAT_COMPLETION_MODEL,
                                              messages=messages, temperature=0)

    if response.choices:
        return response
    else:
        return None;
