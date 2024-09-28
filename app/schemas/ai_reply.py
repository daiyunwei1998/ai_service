from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel


class AIReply(BaseModel):
    original_message: str
    user_query: str
    ai_reply: str
    total_tokens: int
    customer_feedback: Optional[bool] = None
    tenant_id: str
    created_at: datetime = datetime.now(timezone.utc).isoformat()