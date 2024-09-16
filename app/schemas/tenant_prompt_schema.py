from pydantic import BaseModel
from typing import List, Optional

# Schema for creating a new template (without template_id)
class TenantPromptTemplateCreate(BaseModel):
    tenant_id: str
    prompt_template: str
    variables: List[str]
    type: str
    description: Optional[str] = None

# Schema for returning the template (with template_id)
class TenantPromptTemplate(BaseModel):
    template_id: int  # This will be present in the response
    tenant_id: str
    prompt_template: str
    variables: List[str]
    type: str
    description: Optional[str] = None

    class Config:
        orm_mode = True
