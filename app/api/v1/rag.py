from fastapi import APIRouter, HTTPException
from langchain.chains.summarize.map_reduce_prompt import prompt_template

from app.core.config import settings
from app.schemas.ai_reply import AIReply
from app.schemas.rag_schema import SearchRequest
from app.services.mongodb_service import mongodb_service
from app.services.llm_service import rag_pipeline
from app.core.prompt import RAG_PROMPT_TEMPLATE

prompt_template = RAG_PROMPT_TEMPLATE

input_token_price = settings.INPUT_TOKEN_PRICE
output_token_price = settings.OUTPUT_TOKEN_PRICE

router = APIRouter()

@router.post("/")
async def generate_answer(request: SearchRequest):
    query = request.query
    tenant_id = request.tenant_id
    try:
        response = rag_pipeline(query, tenant_id, prompt_template)

        ai_reply = AIReply.from_openai_completion("ADMIN", query, response, tenant_id, input_token_price, output_token_price)

        await mongodb_service.ensure_index(tenant_id)
        await mongodb_service.save_ai_reply(ai_reply)
        return {"data": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
