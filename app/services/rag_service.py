from pydantic import BaseModel
from openai import OpenAI

from pymilvus import Collection, connections
from langdetect import detect
from app.core.database import get_db
from app.services.tenant_prompt_service import get_template_by_id
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)
CHAT_COMPLETION_MODEL = settings.CHAT_COMPLETION_MODEL

PROMPT_TEMPLATE = """
CONTEXT:
You are a customer service AI assistant. Your goal is to provide helpful, accurate, and friendly responses to customer inquiries using the information provided in the DOCUMENT.
You must answer in user's language. User's language: {language}.

DOCUMENT:
{document}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer the QUESTION using information from the DOCUMENT above.
2. Keep your answer grounded in the facts presented in the DOCUMENT.
3. Maintain a professional, friendly, and helpful tone.
4. Provide clear and concise answers.
5. If the DOCUMENT doesn't contain enough information to fully answer the QUESTION:
   a. Share what information you can provide based on the DOCUMENT.
   b. Clearly state that you don't have all the information to fully answer the question.
   c. Suggest where the customer might find additional information if possible.
6. If the QUESTION has multiple parts, address each part separately.
7. Use bullet points or numbering for clarity when appropriate.
8. Offer relevant follow-up questions or additional information that might be helpful.
9. If the inquiry is too complex or requires human intervention, politely explain that you'll need to escalate the issue to a human agent and provide instructions on how to do so.
"""

# Initialize the OpenAI API client with the API key

# Initialize connection to Milvus
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# Define the query input model
class QueryModel(BaseModel):
    query: str

def embed_query(query_string: str) -> list:
    # Use OpenAI API to generate embeddings for the query
    response = client.embeddings.create(model=settings.EMBEDDING_MODEL,
                                        input=query_string)
    return response.data[0].embedding

def search_vectors_in_tenant_db(query_string: str, tenant_id: str) -> list:
    try:
        # Step 1: Generate embedding for the query
        query_embedding = embed_query(query_string)
        logger.info(f"Generated embedding for query: {query_string}")

        # Step 2: Define search parameters with cosine similarity
        search_params = {
            "metric_type": "COSINE",
            "M": 48,
            "params": {"nprobe": 10}
        }
        logger.info(f"Search parameters: {search_params}")

        # Step 3: Perform the search, explicitly requesting the "content" field in the output
        collection = Collection(tenant_id)
        logger.info(f"Searching in collection for tenant: {tenant_id}")
        results = collection.search(
            data=[query_embedding],  # Embedding of the query
            anns_field="embedding",  # Field where vector embeddings are stored
            param=search_params,     # Search parameters using cosine similarity
            limit=5,                 # Limit the number of results
            output_fields=["content"]
        )

        # Step 4: Process search results
        contents = [hit.entity.get("content") for hit in results[0] if hit.entity.get("content")]
        logger.info(f"Search Results: {contents}")
        return contents

    except Exception as e:
        # Log the exception and return an empty list
        logger.error(f"An error occurred during the vector search: {e}")
        return []

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "zh-tw"

# Function to handle the RAG pipeline using OpenAI SDK
def rag_pipeline(query_string: str, tenant_id: str) -> str:
    # Detect the language of the query
    detected_lang = detect_language(query_string)

    # Retrieve relevant documents from the vector database using vector search
    relevant_docs = search_vectors_in_tenant_db(query_string, tenant_id=tenant_id)

    # Combine retrieved documents into a context string
    context = "\n".join(relevant_docs)

    # Fetch the tenant-specific prompt template from the database
    db_session = next(get_db())
    tenant_prompt_template = get_template_by_id(db=db_session, template_id=1)
    db_session.close()

    PROMPT_TEMPLATE = tenant_prompt_template.prompt_template

    # Construct the final prompt by combining template and query data
    prompt = PROMPT_TEMPLATE.format(document=context, language = detected_lang, question=query_string)

    messages = [
        {"role": "system", "content":prompt },
        {"role": "user", "content": query_string},
    ]
    # Call OpenAI GPT to generate the response
    response = client.chat.completions.create(model = CHAT_COMPLETION_MODEL,
    messages = messages, temperature= 0)

    if response.choices:
        return response.choices[0].message.content
    else:
        #TODO 轉人工客服
        return "Sorry, I couldn't generate a response."

# Example usage
if __name__ == "__main__":
    query = "Google Nest mini當機怎麼重設"
    tenant_id = "tenant_1"
    response = rag_pipeline(query, tenant_id)
    print(response)
