import hashlib
import hmac
import secrets


def generate_device_secret() -> str:
    """
    生成设备永久密钥
    这个数据要写入硬件，不写入数据库
    数据库只存摘要
    """
    return secrets.token_urlsafe(32)


def hash_device_secret(device_secret: str) -> str:
    """
    生成摘要
    :param device_secret:str
    :return: str
    """
    return hashlib.sha256(
        device_secret.encode("utf-8"),
    ).hexdigest()


def verify_device_secret(
    device_secret: str,
    expected_hash: str,
) -> bool:
    """
    校验设备上传的密钥是否正确
    :param device_secret: str
    :param expected_hash: str
    :return:boolean
    """
    actual_hash = hash_device_secret(device_secret)
    return  hmac.compare_digest(expected_hash, actual_hash)
