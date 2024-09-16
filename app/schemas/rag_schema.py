from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    tenant_id: str
