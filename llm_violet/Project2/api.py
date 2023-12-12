# -*- coding: utf-8 -*-
import openai
from callback import callback_handler
from dto import ChatbotRequest
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from samples import basic_card_sample, commerce_card_sample, simple_text_sample

app = FastAPI()


@app.get("/")
async def home():
    page = """
    <html>
        <body>
            <h2>카카오 챗봇빌더 스킬 예제입니다 :)</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=page, status_code=200)


@app.post("/skill/hello")
def skill_hello(req: ChatbotRequest):
    return simple_text_sample


@app.post("/skill/basic-card")
async def skill_basic(req: ChatbotRequest):
    return basic_card_sample


@app.post("/skill/commerce-card")
async def skill_commerce(req: ChatbotRequest):
    return commerce_card_sample


# callback.py 로 연결
@app.post("/callback")
async def skill_callback(req: ChatbotRequest, background_tasks: BackgroundTasks):
    # 핸들러 호출 / background_tasks 변경가능
    background_tasks.add_task(callback_handler, req)
    out = {
        "version": "2.0",
        "useCallback": True,
        "data": {"text": "생각하고 있는 중이에요😘 \n15초 정도 소요될 거 같아요 기다려 주실래요?!"},
    }
    return out
