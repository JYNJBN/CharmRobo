from app.models.device import Device
from app.models.user import User
from app.models.user_device import UserDevice

# 让alembic可以扫描到对应的表
__all__ = ["Device", "User", "UserDevice"]
