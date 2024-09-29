from fastapi import APIRouter, HTTPException
from langchain.chains.summarize.map_reduce_prompt import prompt_template

from app.schemas.rag_schema import SearchRequest
from app.services.rag_service import rag_pipeline
from app.core.prompt import PROMPT_TEMPLATE

prompt_template = PROMPT_TEMPLATE

router = APIRouter()

@router.post("/")
async def generate_answer(request: SearchRequest):
    query = request.query
    tenant_id = request.tenant_id
    try:
        response = rag_pipeline(query, tenant_id, prompt_template)
        return {"data": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))