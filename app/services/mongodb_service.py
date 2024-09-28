from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from bson import ObjectId

from app.core.config import settings
from app.schemas.ai_reply import AIReply


class MongoDBService:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]

    async def get_tenant_collection(self, tenant_id: str):
        collection_name = f"{tenant_id}_replies"
        return self.db[collection_name]

    async def save_ai_reply(self, ai_reply: AIReply):
        collection = await self.get_tenant_collection(ai_reply.tenant_id)
        result = await collection.insert_one(ai_reply.dict())
        return str(result.inserted_id)

    async def update_feedback(self, reply_id: str, tenant_id: str, feedback: bool):
        collection = await self.get_tenant_collection(tenant_id)
        result = await collection.update_one(
            {"_id": ObjectId(reply_id)},
            {"$set": {"customer_feedback": feedback}}
        )
        return result.modified_count > 0

    async def ensure_indexes(self, tenant_ids: List[str]):
        for tenant_id in tenant_ids:
            collection = await self.get_tenant_collection(tenant_id)
            await collection.create_index("created_at")

    async def ensure_index(self, tenant_id: str):
        collection = await self.get_tenant_collection(tenant_id)
        await collection.create_index("created_at")

    async def close_connection(self):
        self.client.close()

mongodb_service = MongoDBService()