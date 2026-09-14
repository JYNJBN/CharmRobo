from app.models.agent import Agent
from app.models.conversation import Conversation
from app.models.conversation_message import ConversationMessage
from app.models.conversation_summary import ConversationSummary
from app.models.device import Device
from app.models.user import User
from app.models.user_device import UserDevice

# 让alembic可以扫描到对应的表
__all__ = [
    "Conversation",
    "ConversationMessage",
    "Device",
    "User",
    "UserDevice",
    "ConversationSummary",
    "Agent"
]
