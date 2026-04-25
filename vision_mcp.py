import os
import time
import base64
import asyncio
import threading
import requests
from fastapi import FastAPI, Request
import uvicorn
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# ==========================================
# 1. 环境变量与配置
# ==========================================
MACRODROID_ID = os.environ.get("MACRODROID_ID", "你的MacroDroid_ID")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "你的大模型KEY")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")

# 双端口配置
RECEIVER_PORT = int(os.environ.get("RECEIVER_PORT", 8123))
SSE_PORT = int(os.environ.get("SSE_PORT", 8000))

# ==========================================
# 2. 手机图片 HTTP 接收器 (端口 8123)
# ==========================================
photo_buffer = {}
app = FastAPI()

@app.post("/upload_photo")
async def upload_photo(task_id: str, request: Request):
    """接收手机端动态提取并通过 HTTP POST 过来的纯图片流"""
    try:
        image_bytes = await request.body()
        if image_bytes:
            photo_buffer[task_id] = base64.b64encode(image_bytes).decode('utf-8')
            return {"status": "success"}
        return {"status": "error", "message": "No image data received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_receiver():
    # 静默运行，避免刷屏
    uvicorn.run(app, host="0.0.0.0", port=RECEIVER_PORT, log_level="warning")

# 丢进后台线程独立运行
threading.Thread(target=run_receiver, daemon=True).start()

# ==========================================
# 3. FastMCP 核心服务 (SSE 模式, 端口 8000)
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

    # 2. 挂起等待接收端口 (8123) 传回图片
    wait_time = 0
    max_wait = 25 # 设为25秒，防 Rikkahub 彻底断连
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

if __name__ == "__main__":
    print(f"🚀 Vision MCP (SSE 模式) 正在启动...")
    print(f"📡 手机图片接收端口: {RECEIVER_PORT}")
    print(f"🔗 供 RikkaHub 连接的 SSE 端口: {SSE_PORT} -> URL 请填: http://127.0.0.1:{SSE_PORT}/sse")
    
    # 核心改动：直接以 SSE 协议运行
    mcp.run(transport="sse", host="0.0.0.0", port=SSE_PORT)