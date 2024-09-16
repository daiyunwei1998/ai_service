from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.services.tenant_prompt_service import create_template, get_templates, get_template_by_id
from app.schemas.tenant_prompt_schema import TenantPromptTemplate, TenantPromptTemplateCreate
from app.core.database import get_db
from typing import List

router = APIRouter()

@router.post("/", response_model=TenantPromptTemplate)
def store_template(data: TenantPromptTemplateCreate, db: Session = Depends(get_db)):
    try:
        return create_template(db, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[TenantPromptTemplate])
def read_templates(tenant_id: str = None, db: Session = Depends(get_db)):
    try:
        return get_templates(db, tenant_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{template_id}", response_model=TenantPromptTemplate)
def get_template(template_id: str, db: Session = Depends(get_db)):
    template = get_template_by_id(db, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template