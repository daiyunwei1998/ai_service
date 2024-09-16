from typing import Optional

from sqlalchemy import Column, Integer, Text, JSON, String
from app.core.database import Base


class TenantPromptTemplate(Base):
    __tablename__ = 'tenant_prompt_template'

    template_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tenant_id = Column(String(255), index=True, nullable=False)
    prompt_template = Column(Text, nullable=False)
    variables = Column(JSON, nullable=False)
    type = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
