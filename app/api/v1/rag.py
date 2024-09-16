from fastapi import APIRouter, HTTPException

from app.schemas.rag_schema import SearchRequest
from app.services.rag_service import rag_pipeline

router = APIRouter()

@router.post("/")
async def generate_answer(request: SearchRequest):
    query = request.query
    tenant_id = request.tenant_id
    try:
        response = rag_pipeline(query, tenant_id)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))