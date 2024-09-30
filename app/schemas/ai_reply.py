from datetime import datetime, timezone
from typing import Optional, Dict

from openai.types import Completion
from pydantic import BaseModel

class TokenInfo(BaseModel):
    count: int  # Number of tokens
    price_per_token: float  # Price per token

class AIReply(BaseModel):
    receiver: str
    user_query: str
    ai_reply: str
    tokens: Dict[str, TokenInfo]  # Key could be 'input', 'output', or others
    total_tokens: int  # Can still store the total if needed
    customer_feedback: Optional[bool] = None
    tenant_id: str
    created_at: datetime = datetime.now(timezone.utc)

    @classmethod
    def from_openai_completion(cls, receiver: str, user_query: str, completion: Completion | str, tenant_id: str,
                               input_token_price: float, output_token_price: float):
        # Extract token usage
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Extract the AI's reply from the completion object
        ai_reply = completion.choices[0].message.content

        # Construct the tokens dictionary
        tokens = {
            "input": TokenInfo(count=prompt_tokens, price_per_token=input_token_price),
            "output": TokenInfo(count=completion_tokens, price_per_token=output_token_price)
        }

        # Create the AIReply object
        return cls(
            receiver=receiver,
            user_query=user_query,
            ai_reply=ai_reply,
            tokens=tokens,
            total_tokens=total_tokens,
            tenant_id=tenant_id
        )