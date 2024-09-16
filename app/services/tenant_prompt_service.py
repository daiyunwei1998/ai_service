from sqlalchemy.orm import Session
from app.models.tenant_prompt_model import TenantPromptTemplate as TemplateModel
from app.schemas.tenant_prompt_schema import TenantPromptTemplateCreate


def create_template(db: Session, data: TenantPromptTemplateCreate):
    db_template = TemplateModel(
        tenant_id=data.tenant_id,
        prompt_template=data.prompt_template,
        variables=data.variables,
        type=data.type,
        description=data.description
    )
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template

def get_templates(db: Session, tenant_id: int = None):
    query = db.query(TemplateModel)
    if tenant_id:
        query = query.filter(TemplateModel.tenant_id == tenant_id)
    return query.all()

def get_template_by_id(db: Session, template_id: int):
    return db.query(TemplateModel).filter(TemplateModel.template_id == template_id).first()