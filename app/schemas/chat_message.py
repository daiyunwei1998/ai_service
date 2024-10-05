# Define Enums to match Java enums
import json
from enum import Enum
from typing import Optional


class MessageType(Enum):
    CHAT = "CHAT"
    JOIN = "JOIN"
    LEAVE = "LEAVE"
    SUMMARY = "SUMMARY"

class SourceType(Enum):
    USER = "USER"
    AGENT = "AGENT"
    AI = "AI"

class ChatMessage:
    def __init__(
        self,
        session_id: str,
        type: MessageType,
        content: str,
        sender: str,
        sender_name: str,
        receiver: Optional[str],
        tenant_id: str,
        timestamp: float,
        source: SourceType,
        user_type: str,
        customer_id: Optional[str],
    ):
        self.session_id = session_id
        self.type = type.value
        self.content = content
        self.sender = sender
        self.sender_name = sender_name
        self.receiver = receiver
        self.tenant_id = tenant_id
        self.timestamp = timestamp
        self.source = source.value
        self.user_type = user_type
        self.customer_id = customer_id

    def to_json(self):
        # Create a copy of the instance's dictionary
        data = self.__dict__.copy()
        # Add the @class field required by Java
        data["@class"] = "org.service.customer.models.ChatMessage"
        # Serialize to JSON with proper encoding
        return json.dumps(data, ensure_ascii=False)