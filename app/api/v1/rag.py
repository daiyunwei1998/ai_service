from fastapi import APIRouter, HTTPException
from langchain.chains.summarize.map_reduce_prompt import prompt_template

from app.schemas.ai_reply import AIReply
from app.schemas.rag_schema import SearchRequest
from app.services.mongodb_service import mongodb_service
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

        ai_reply = AIReply(
            receiver="ADMIN",
            user_query= query,  # Assuming the original message is the user query
            ai_reply= response.choices[0].message.content,
            total_tokens=response.usage.total_tokens,  # Simple token count, replace with actual token counting logic
            tenant_id=tenant_id
        )

        await mongodb_service.ensure_index(tenant_id)
        await mongodb_service.save_ai_reply(ai_reply)
        return {"data": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))