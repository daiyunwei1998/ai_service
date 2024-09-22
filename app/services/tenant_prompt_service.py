from sqlalchemy.orm import Session
import logging
from app.core.config import settings
from app.models.tenant_prompt_model import TenantPromptTemplate as TemplateModel
from app.schemas.tenant_prompt_schema import TenantPromptTemplateCreate
from openai import OpenAI
from pymilvus import Collection

client = OpenAI(api_key=settings.OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_template(db: Session, data: TenantPromptTemplateCreate):
    db_template = TemplateModel(
        tenant_id=data.tenant_id,
        prompt_template=data.prompt_template,
        variables=data.variables,
        type=data.type,
        description=data.description
    )
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template

def get_templates(db: Session, tenant_id: int = None):
    query = db.query(TemplateModel)
    if tenant_id:
        query = query.filter(TemplateModel.tenant_id == tenant_id)
    return query.all()

def get_template_by_id(db: Session, template_id: int):
    return db.query(TemplateModel).filter(TemplateModel.template_id == template_id).first()

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
