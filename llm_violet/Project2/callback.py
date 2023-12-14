import os
from typing import List

import openai
import requests
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from dto import ChatbotRequest

dir_path = os.path.dirname(os.path.realpath(__file__))
channel_txt = os.path.join(dir_path, "project_data_카카오톡채널.txt")


# OpenAI API Key 파일에서 읽어오기
with open("openai_key.txt", "r") as f:
    k = f.read()
    openai.api_key = k
    os.environ["OPENAI_API_KEY"] = k


# read the whole text file
def read_file(file_path=channel_txt):
    with open(file_path, "r") as f:
        return f.read()


def send_message(message_log, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
    )

    response_message = response["choices"][0]["message"]
    return response_message.content


message_log = []
SYSTEM_MSG = """
You are the QnA chatbot for a KakaoTalk channel API service.
At first, greet the user and ask how you can help.
Refer to the technical document and answer to the question.
Your user will be Korean, so communicate in Korean.
When user asks irrelevant questions, say 'I can't answer that' or 'I don't know'.

document:
{document}
"""


def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    llm = ChatOpenAI(
        temperature=0.1,
        max_tokens=300,
        model="gpt-3.5-turbo",
    )

    global message_log
    if not message_log:
        message_log = [("system", SYSTEM_MSG)]

    message_log.append(("human", request.userRequest.utterance))

    message_log_prompt = ChatPromptTemplate.from_messages(message_log)
    chain = LLMChain(
        llm=llm,
        prompt=message_log_prompt,
        verbose=True,
    )
    ai_response = chain.run(
        document=read_file(),
    )

    message_log.append(("ai", ai_response))

    # logging
    print(f"output_text: {ai_response}")

    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": ai_response}}]},
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url, json=payload)
