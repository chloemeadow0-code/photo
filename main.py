import os
import time
import base64
import asyncio
import json
import requests
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse
from openai import OpenAI

# ==========================================
# 1. 环境变量与配置
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
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }},
                    ],
                }],
                max_tokens=800,
            )

        v_res = await asyncio.to_thread(_ask_vision)
        return f"📸 手机视角解析完成：\n\n{v_res.choices[0].message.content.strip()}"

    except Exception as e:
        return f"❌ 视觉模型解析失败: {e}"


# ==========================================
# 3. 自定义路由
# ==========================================
async def health_endpoint(request: Request):
    return JSONResponse({"status": "ok"})


async def upload_photo_endpoint(request: Request):
    try:
        task_id = request.query_params.get("task_id", "")
        body = await request.body()
        if body and task_id:
            photo_buffer[task_id] = base64.b64encode(body).decode()
            return JSONResponse({"status": "success", "message": "Photo saved"})
        return JSONResponse(
            {"status": "error", "message": "Missing body or task_id"},
            status_code=400
        )
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ==========================================
# 4. 低层级 SSE 路由（兼容 mcp 1.x）
# ==========================================
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp._mcp_server.run(
            streams[0],
            streams[1],
            mcp._mcp_server.create_initialization_options(),
        )


app = Starlette(
    routes=[
        Route("/health",       health_endpoint),
        Route("/upload_photo", upload_photo_endpoint, methods=["POST"]),
        Route("/sse",          handle_sse),
        Mount("/messages/",    app=sse_transport.handle_post_message),
    ]
)


# ==========================================
# 5. 启动
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Vision MCP 启动于端口 {port}")
    print(f"   MCP SSE  端点: http://0.0.0.0:{port}/sse")
    print(f"   图片上传端点: http://0.0.0.0:{port}/upload_photo?task_id=xxx")
    print(f"   健康检查端点: http://0.0.0.0:{port}/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=120,
    )
