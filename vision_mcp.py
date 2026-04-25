import os
import time
import json
import base64
import asyncio
import threading
import requests
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# ==========================================
# 1. 环境变量配置 (请根据实际情况填入)
# ==========================================
# 手机端 MacroDroid 的 Webhook 触发 ID
MACRODROID_ID = os.environ.get("MACRODROID_ID", "YOUR_MACRODROID_ID")

# 视觉大模型配置
VISION_API_KEY = os.environ.get("VISION_API_KEY", "sk-xxxxxx")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")

# 接收手机图片的端口 (确保手机能访问到该服务器的这个端口)
RECEIVER_PORT = int(os.environ.get("RECEIVER_PORT", 8123))

# ==========================================
# 2. 全局状态缓冲池
# ==========================================
# 存放手机传回来的图片 Base64，格式: {"task_id": "base64_string"}
photo_buffer = {}

# ==========================================
# 3. 伴生 HTTP 接收器 (专接手机发来的图片)
# ==========================================
app = FastAPI()

@app.post("/upload_photo")
async def upload_photo(task_id: str = Form(...), photo: UploadFile = File(...)):
    """手机拍照后，将图片 POST 到这个接口"""
    try:
        image_bytes = await photo.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        photo_buffer[task_id] = base64_image
        return {"status": "success", "message": "Photo received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_receiver():
    # 禁用 uvciorn 的日志避免污染 MCP 的 stdio 通道
    uvicorn.run(app, host="0.0.0.0", port=RECEIVER_PORT, log_level="critical")

# 启动后台接收器线程
threading.Thread(target=run_receiver, daemon=True).start()

# ==========================================
# 4. MCP 核心逻辑
# ==========================================
mcp = FastMCP("MobileCameraVision")

@mcp.tool()
async def take_photo_and_analyze(prompt: str = "请详细描述你在这张照片里看到了什么？环境、物品、人物状态等。") -> str:
    """
    【视觉感知工具】当你需要“看看”用户周围的环境、或者用户让你“看一下”某样东西时调用此工具。
    它会静默唤醒用户的手机摄像头抓拍一张当前视角的照片，并利用视觉大模型进行分析。
    
    参数:
    - prompt: 你希望视觉模型侧重分析的内容，例如"看看屏幕上有什么"或"桌子上有什么吃的"。
    """
    if not MACRODROID_ID:
        return "❌ 缺少 MACRODROID_ID，无法触发手机拍照。"

    # 生成唯一的任务流水号
    task_id = f"snap_{int(time.time())}"
    
    # 1. 向手机发送拍照指令
    webhook_url = f"https://trigger.macrodroid.com/{MACRODROID_ID}/take_photo?task_id={task_id}"
    try:
        resp = await asyncio.to_thread(lambda: requests.get(webhook_url, timeout=10))
        if resp.status_code != 200:
            return f"❌ 无法唤醒手机，状态码: {resp.status_code}"
    except Exception as e:
        return f"❌ 唤醒手机网络请求失败: {e}"

    # 2. 挂起等待手机将图片发到我们的接收端口 (最长等待 30 秒)
    wait_time = 0
    max_wait = 30
    base64_image = ""
    
    while wait_time < max_wait:
        if task_id in photo_buffer:
            base64_image = photo_buffer.pop(task_id)
            break
        await asyncio.sleep(1)
        wait_time += 1

    if not base64_image:
        return "❌ 手机拍照超时或上传失败，没有接收到图像流。"

    # 3. 将抓取到的画面送入视觉大模型
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
        ai_desc = v_res.choices[0].message.content.strip()
        return f"📸 手机摄像头抓拍画面解析完成：\n\n{ai_desc}"
        
    except Exception as e:
        return f"❌ 图片抓拍成功，但视觉大模型解析崩溃了: {e}"

if __name__ == "__main__":
    # 使用 stdio 模式启动，符合主流 MCP 客户端挂载规范
    mcp.run(transport="stdio")