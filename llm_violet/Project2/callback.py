import datetime
import logging
import os
import time
from typing import List

import openai
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.schema import SystemMessage
from langchain.utilities import DuckDuckGoSearchAPIWrapper

from dto import ChatbotRequest
from samples import list_card

dir_path = os.path.dirname(os.path.realpath(__file__))
channel_txt = os.path.join(dir_path, "project_data_카카오톡채널.txt")


# OpenAI API Key 파일에서 읽어오기
with open("openai_key.txt", "r") as f:
    openai.api_key = f.read()


# read the whole text file
def read_file(file_path=channel_txt):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def send_message(message_log, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
    )

    response_message = response["choices"][0]["message"]
    return response_message.content


SYSTEM_MSG = """
You are the QnA chatbot for a KakaoTalk channel API service.
Your user will be Korean, so communicate in Korean.
At first, greet the user and ask how you can help.
When user asks irrelevant questions, say 'I can't answer that' or 'I don't know'.
"""
logger = logging.getLogger("Callback")

message_log = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "assistant", "content": read_file()},  # 텍스트 파일 통째로 넘기기
]


def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    message_log.append(
        {"role": "user", "content": request.userRequest.utterance}
    )

    output_text = send_message(
        message_log=message_log,
        gpt_model="gpt-3.5-turbo",
        temperature=0,
    )

    message_log.append(
        {"role": "assistant", "content": output_text}
    )

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

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url, json=payload)
