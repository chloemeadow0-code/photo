import os
import time
import base64
import asyncio
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

# 最新照片（不再依赖 task_id，MacroDroid 直接 POST 即可）
latest_photo: dict = {"data": None, "ts": 0}

# ==========================================
# 2. FastMCP 工具
# ==========================================
mcp = FastMCP("VisionNode")

@mcp.tool()
async def take_photo_and_analyze(
    prompt: str = "请详细描述你在这张照片里看到了什么？环境、物品、人物状态等。"
) -> str:
    """
    【视觉感知工具】唤醒手机摄像头静默抓拍，并用大模型分析画面。
    触发 MacroDroid 拍照，等待手机回传图片后交给视觉模型解析。
    """
    trigger_url = f"https://trigger.macrodroid.com/{MACRODROID_ID}/take_photo"
    trigger_ts = time.time()

    # 1. 触发手机拍照
    try:
        resp = await asyncio.to_thread(
            lambda: requests.get(trigger_url, timeout=10)
        )
        if resp.status_code != 200:
            return f"❌ 无法唤醒手机: HTTP {resp.status_code}"
    except Exception as e:
        return f"❌ 唤醒手机网络失败: {e}"

    # 2. 等待手机上传新照片（最多 30 秒）
    # 只接受触发之后上传的照片，避免拿到旧图
    base64_image = ""
    for _ in range(30):
        if latest_photo["data"] and latest_photo["ts"] >= trigger_ts:
            base64_image = latest_photo["data"]
            break
        await asyncio.sleep(1)

    if not base64_image:
        return "❌ 手机拍照超时，未在 30 秒内收到图片。"

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
# 3. 路由处理
# ==========================================
async def health_endpoint(request: Request):
    return JSONResponse({"status": "ok"})


async def upload_photo_endpoint(request: Request):
    """
    MacroDroid 拍照后 POST 到此接口。
    不需要任何参数，直接把图片二进制数据放 body 即可。
    """
    try:
        body = await request.body()
        if not body:
            return JSONResponse(
                {"status": "error", "message": "Empty body"},
                status_code=400
            )
        latest_photo["data"] = base64.b64encode(body).decode()
        latest_photo["ts"] = time.time()
        return JSONResponse({"status": "success", "message": "Photo received"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ==========================================
# 4. SSE 路由（兼容 mcp 1.x）
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
    print(f"   图片上传端点: http://0.0.0.0:{port}/upload_photo  (POST, body=图片二进制)")
    print(f"   健康检查端点: http://0.0.0.0:{port}/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=120,
    )
