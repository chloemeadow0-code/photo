import os
import time
import base64
import asyncio
import json
import requests
import uvicorn
from urllib.parse import parse_qs
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from starlette.types import ASGIApp, Scope, Receive, Send

# ==========================================
# 1. 环境变量与配置 (Zeabur 自动读取)
# ==========================================
MACRODROID_ID   = os.environ.get("MACRODROID_ID",   "你的MacroDroid_ID")
VISION_API_KEY  = os.environ.get("VISION_API_KEY",  "你的大模型KEY")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1")
VISION_MODEL    = os.environ.get("VISION_MODEL",    "gpt-4o-mini")

# 照片缓冲区
photo_buffer: dict[str, str] = {}

# ==========================================
# 2. FastMCP 核心服务
# ==========================================
mcp = FastMCP("VisionNode")

@mcp.tool()
async def take_photo_and_analyze(
    prompt: str = "请详细描述你在这张照片里看到了什么？环境、物品、人物状态等。"
) -> str:
    """
    【视觉感知工具】唤醒手机摄像头静默抓拍，并用大模型分析画面。
    先触发 MacroDroid 拍照，等待手机回传图片后交给视觉模型解析。
    """
    task_id = f"snap_{int(time.time())}"
    webhook_url = (
        f"https://trigger.macrodroid.com/{MACRODROID_ID}/take_photo"
        f"?task_id={task_id}"
    )

    # 1. 触发手机拍照
    try:
        resp = await asyncio.to_thread(
            lambda: requests.get(webhook_url, timeout=10)
        )
        if resp.status_code != 200:
            return f"❌ 无法唤醒手机: HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ 唤醒手机网络失败: {e}"

    # 2. 轮询等待手机传回图片（最多 25 秒）
    base64_image = ""
    for _ in range(25):
        if task_id in photo_buffer:
            base64_image = photo_buffer.pop(task_id)
            break
        await asyncio.sleep(1)

    if not base64_image:
        return "❌ 手机拍照超时，未在 25 秒内收到图片。"

    # 3. 送入视觉模型分析
    try:
        client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)

        def _ask_vision():
            return client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800,
            )

        v_res = await asyncio.to_thread(_ask_vision)
        return f"📸 手机视角解析完成：\n\n{v_res.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ 视觉模型解析失败: {e}"


# ==========================================
# 3. 单端口中间件：合并 /upload_photo 与 MCP 路由
# ==========================================
class SinglePortMiddleware:
    """
    将手机传图接口（POST /upload_photo）与 MCP 的 SSE 接口
    合并到同一个 Zeabur 端口上，无需额外开放端口。
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # 健康检查端点
        if scope["type"] == "http" and scope["path"] == "/health":
            body = b'{"status":"ok"}'
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({"type": "http.response.body", "body": body})
            return

        # 手机图片上传端点
        if (
            scope["type"] == "http"
            and scope["path"] == "/upload_photo"
            and scope["method"] == "POST"
        ):
            await self._handle_upload(scope, receive, send)
            return

        # 其余请求（/sse、/messages 等）全部交给 FastMCP
        await self.app(scope, receive, send)

    async def _handle_upload(self, scope: Scope, receive: Receive, send: Send):
        try:
            qs = scope.get("query_string", b"").decode()
            task_id = parse_qs(qs).get("task_id", [""])[0]

            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break

            if body and task_id:
                photo_buffer[task_id] = base64.b64encode(body).decode()
                resp_body = b'{"status":"success","message":"Photo saved"}'
                status = 200
            else:
                resp_body = b'{"status":"error","message":"Missing body or task_id"}'
                status = 400

        except Exception as e:
            resp_body = json.dumps({"status": "error", "message": str(e)}).encode()
            status = 500

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": resp_body})


# ==========================================
# 4. 启动入口
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Vision MCP 启动于端口 {port}")
    print(f"   MCP SSE  端点: http://0.0.0.0:{port}/sse")
    print(f"   图片上传端点: http://0.0.0.0:{port}/upload_photo?task_id=xxx")
    print(f"   健康检查端点: http://0.0.0.0:{port}/health")

    try:
        raw_app = mcp.get_asgi_app()
    except AttributeError:
        raw_app = mcp.sse_app()

    app = SinglePortMiddleware(raw_app)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=120,
    )
