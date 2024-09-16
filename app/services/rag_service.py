from pydantic import BaseModel
from langchain_openai import OpenAI, OpenAIEmbeddings
from pymilvus import Collection, connections
from app.core.config import settings
from langchain import PromptTemplate

from app.core.database import get_db
from app.services.tenant_prompt_service import get_template_by_id



PROMPT_TEMPLATE = """
CONTEXT:
You are a customer service AI assistant. Your goal is to provide helpful, accurate, and friendly responses to customer inquiries using the information provided in the DOCUMENT.

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


# Initialize the OpenAI language model
llm = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize connection to Milvus
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# Initialize the embedding model
embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model="text-embedding-ada-002")

# Define the query input model
class QueryModel(BaseModel):
    query: str

def embed_query(query: str):
    return embeddings.embed_query(query)

def search_vectors_in_tenant_db(query: str, tenant_id:str) -> list:
    # Step 1: Generate embedding for the query
    query_embedding = embed_query(query)

    # Step 2: Define search parameters with cosine similarity
    search_params = {
        "metric_type": "COSINE",
        "M": 48,
        "params": {"nprobe": 10}
    }


    # Step 3: Perform the search, explicitly requesting the "content" field in the output
    collection = Collection(tenant_id)
    results = collection.search(
        data=[query_embedding],  # Embedding of the query
        anns_field="embedding",  # Field where vector embeddings are stored
        param=search_params,     # Search parameters using cosine similarity
        limit=5,                 # Limit the number of results
        output_fields=["content"]  # Specify "content" field to retrieve
    )

    # Step 4: Process search results
    contents = [hit.entity.get("content") for hit in results[0] if hit.entity.get("content")]

    print("Search Results: ", contents)
    return contents

# Function to handle the RAG pipeline
def rag_pipeline(query: str, tenant_id: str) -> str:
    # Step 1: Retrieve relevant documents from the vector database using vector search
    relevant_docs = search_vectors_in_tenant_db(query, tenant_id=tenant_id)

    # Step 2: Combine retrieved documents into a context string
    context = "\n".join(relevant_docs)

    # Step 3: Use the LLM to generate a response, incorporating the context
    db_session = next(get_db())
    tenant_prompt_template = get_template_by_id(db=db_session, template_id=1)
    db_session.close()

    PROMPT_TEMPLATE = tenant_prompt_template.prompt_template
    INPUT_VARIABLES = tenant_prompt_template.variables
    print(PROMPT_TEMPLATE)
    print(INPUT_VARIABLES)
    PROMPT = PromptTemplate(
        input_variables=INPUT_VARIABLES,
        template=PROMPT_TEMPLATE,
    )

    print("no\n\n")


    prompt = PROMPT.invoke({"document":context, "question":query})
    print(prompt)
    # Generate the response using the LLM
    response = llm(prompt.to_string())

    return response
