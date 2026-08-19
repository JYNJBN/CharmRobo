import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.dependencies import CurrentUserId
from app.core.config import settings
from app.schemas.common import ApiResponse

router = APIRouter(
    prefix="/upload",
    tags=["上传"],
)

# 允许的图片扩展名（注意：微信小程序真机只稳 PNG/JPG）
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
# 最大 10MB（与前端 chooseImage 的大小限制一致）
MAX_SIZE = 10 * 1024 * 1024


@router.post("/avatar", response_model=ApiResponse)
async def upload_avatar(
    current_user_id: CurrentUserId,
    file: UploadFile = File(..., description="头像图片文件"),
):
    """
    上传头像图片。

    需要登录（JWT）。图片保存在服务器本地 uploads/avatars/{user_id}/ 目录，
    通过 /static 静态路径访问。返回相对 URL，前端拼上自己的 apiBase 即可访问。

    上传成功后，前端再调 PATCH /api/v1/users/info 把 avatar 字段更新为用户头像 URL。
    """

    # 1) 校验扩展名（用 file.filename 防止路径穿越，只取后缀）
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的图片格式，仅支持 {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2) 读取内容并校验大小
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片大小不能超过 10MB",
        )

    # 3) 保存：uploads/avatars/{user_id}/{uuid}{ext}
    #    按用户分目录，避免不同用户互相覆盖/串文件。
    user_dir = Path(settings.upload_dir) / "avatars" / str(current_user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = user_dir / save_name
    save_path.write_bytes(content)

    # 4) 返回相对 URL（前端拼 apiBase，例如 http://192.168.2.132:8001/static/...）
    url = f"/static/avatars/{current_user_id}/{save_name}"
    return ApiResponse(
        message="上传成功",
        data={"url": url},
    )
