from app.models.agent import Agent
from app.models.agent_knowledge_base import AgentKnowledgeBase
from app.models.conversation import Conversation
from app.models.conversation_message import ConversationMessage
from app.models.conversation_summary import ConversationSummary
from app.models.device import Device
from app.models.knowledge_base import KnowledgeBase
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_document import KnowledgeDocument
from app.models.knowledge_document_upload import KnowledgeDocumentUpload
from app.models.knowledge_ingest_job import KnowledgeIngestJob
from app.models.user import User
from app.models.user_device import UserDevice

# 让alembic可以扫描到对应的表
__all__ = [
    "Agent",
    "Conversation",
    "ConversationMessage",
    "ConversationSummary",
    "Device",
    "KnowledgeBase",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "User",
    "UserDevice",
    "AgentKnowledgeBase",
    "KnowledgeIngestJob",
    "KnowledgeDocumentUpload"
]
