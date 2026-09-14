class BizError(Exception):
    """业务异常：service 层只抛它，翻译成什么由入口决定。
     code 在 HTTP 入口当状态码用（404/403...），
    在 ws/MQTT 入口当业务错误码字段发出去。
    """

    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code
        super().__init__(message)
