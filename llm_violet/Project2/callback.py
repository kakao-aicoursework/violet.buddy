import datetime
import logging
import time
from typing import List

import aiohttp
import openai
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from dto import ChatbotRequest
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.schema import SystemMessage
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from samples import list_card

# api key를 파일에서 읽어옴
with open("openai_key.txt", "r") as f:
    openai.api_key = f.read()

SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")


async def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": request.userRequest.utterance},
        ],
        temperature=0,
    )
    # focus
    output_text = response.choices[0].message.content

    # logging
    logger.info(f"output_text: {output_text}")

    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": output_text}}]},
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()
