from pymilvus import Collection, connections
from app.core.config import settings
from langchain.embeddings import OpenAIEmbeddings

# Initialize connection to Milvus using the centralized settings
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
collection = Collection("tenant_1")

# Initialize the embedding model using the centralized settings
embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

def embed_query(query: str):
    return embeddings.embed(query)

# def search_vectors_in_tenant_dbctors(query: str) -> list:
#     query_embedding = embed_query(query)
#
#     search_params = {
#         "metric_type": "COSINE",
#         "params": {"nprobe": 10}
#     }
#     results = collection.search([query_embedding], "embedding", search_params, limit=5)
#
#     return [res.entity.get("content") for res in results]