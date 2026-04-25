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
# 1. 环境变量与配置 (Zeabur 会自动读取这些)
# ==========================================
MACRODROID_ID = os.environ.get("MACRODROID_ID", "你的MacroDroid_ID")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "你的大模型KEY")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")

# 缓冲区，用来暂存手机传来的图片
photo_buffer = {}

# ==========================================
# 2. FastMCP 核心服务
# ==========================================
mcp = FastMCP("VisionNode")

@mcp.tool()
async def take_photo_and_analyze(prompt: str = "请详细描述你在这张照片里看到了什么？环境、物品、人物状态等。") -> str:
    """
    【视觉感知工具】唤醒手机摄像头静默抓拍，并用大模型分析画面。
    """
    task_id = f"snap_{int(time.time())}"
    webhook_url = f"https://trigger.macrodroid.com/{MACRODROID_ID}/take_photo?task_id={task_id}"
    
    try:
        # 1. 触发手机拍照
        resp = await asyncio.to_thread(lambda: requests.get(webhook_url, timeout=10))
        if resp.status_code != 200:
            return f"❌ 无法唤醒手机: {resp.status_code}"
    except Exception as e:
        return f"❌ 唤醒手机网络失败: {e}"

    # 2. 挂起等待手机传回图片
    wait_time = 0
    max_wait = 25 # 防 Rikkahub 断连
    base64_image = ""
    
    while wait_time < max_wait:
        if task_id in photo_buffer:
            base64_image = photo_buffer.pop(task_id)
            break
        await asyncio.sleep(1)
        wait_time += 1

    if not base64_image:
        return "❌ 手机拍照超时或未将图片传回服务器。"

    # 3. 将图片塞进视觉模型分析
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
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=800
            )
        v_res = await asyncio.to_thread(_ask_vision)
        return f"📸 手机视角解析完成：\n\n{v_res.choices[0].message.content.strip()}"
        
    except Exception as e:
        return f"❌ 视觉模型解析崩溃: {str(e)}"

# ==========================================
# 3. Zeabur 单端口中间件拦截器 (核心魔法)
# ==========================================
class SinglePortMiddleware:
    """将手机传图接口与 MCP 通信接口强行合并到同一个 Web 端口上"""
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # 拦截手机的图片上传请求
        if scope["type"] == "http" and scope["path"] == "/upload_photo" and scope["method"] == "POST":
            try:
                # 解析 URL 里的 ?task_id=xxx
                query_string = scope.get("query_string", b"").decode("utf-8")
                params = parse_qs(query_string)
                task_id = params.get("task_id", [""])[0]

                # 读取原生的裸图片数据 (Image raw body)
                body = b""
                while True:
                    msg = await receive()
                    body += msg.get("body", b"")
                    if not msg.get("more_body", False): break
                
                if body and task_id:
                    photo_buffer[task_id] = base64.b64encode(body).decode('utf-8')
                    resp_json = b'{"status": "success", "message": "Photo saved"}'
                else:
                    resp_json = b'{"status": "error", "message": "Missing body or task_id"}'

                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": resp_json})
                return
            except Exception as e:
                err_json = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                await send({"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": err_json})
                return

        # 其他所有请求（比如 /sse 和 /messages）全部放行给 FastMCP 去处理
        await self.app(scope, receive, send)

if __name__ == "__main__":
    # Zeabur 会自动注入 PORT 环境变量，默认兜底 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Vision MCP 正在 Zeabur 端口 {port} 上启动...")
    
    # 提取 FastMCP 底层的 ASGI 引擎并套上我们的护盾
    app = SinglePortMiddleware(mcp.sse_app())
    
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")