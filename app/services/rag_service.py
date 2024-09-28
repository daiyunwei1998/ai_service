from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from openai import OpenAI

from pymilvus import Collection, connections
from app.services.language_service import detect_language
from app.services.tenant_prompt_service import get_template_by_id, search_vectors_in_tenant_db
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)
CHAT_COMPLETION_MODEL = settings.CHAT_COMPLETION_MODEL


# Initialize the OpenAI API client with the API key

# Initialize connection to Milvus
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# # Define the query input model
# class QueryModel(BaseModel):
#     query: str

# Function to handle the RAG pipeline using OpenAI SDK
def rag_pipeline(query_string: str, tenant_id: str, prompt_template: str) -> ChatCompletion | str:
    # Detect the language of the query
    detected_lang = detect_language(query_string)

    # Retrieve relevant documents from the vector database using vector search
    relevant_docs = search_vectors_in_tenant_db(query_string, tenant_id=tenant_id)

    # Combine retrieved documents into a context string
    context = "\n".join(relevant_docs)
    logging.info("Retrieved context: \n" + context)

    # Construct the final prompt by combining template and query data
    prompt = prompt_template.format(document=context, language = detected_lang, question=query_string)

    messages = [
        {"role": "system", "content":prompt },
        {"role": "user", "content": query_string},
    ]
    # Call OpenAI GPT to generate the response
    response = client.chat.completions.create(model = CHAT_COMPLETION_MODEL,
    messages = messages, temperature= 0)

    if response.choices:
        return response
    else:
        #TODO 轉人工客服
        return "Sorry, I couldn't generate a response."

