import os
from typing import List, Tuple

import openai
import requests
from dto import ChatbotRequest
from langchain import LLMChain
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma

# OpenAI API Key 파일에서 읽어오기
with open("openai_key.txt", "r") as f:
    k = f.read()
    openai.api_key = k
    os.environ["OPENAI_API_KEY"] = k


# read the whole text file
def read_file(file_path: str) -> str:
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


def get_chroma_db() -> Chroma:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PERSIST_DIR = os.path.join(PROJECT_DIR, "chroma-persist")
    CHROMA_COLLECTION_NAME = "dosu-bot"

    _db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    return _db
    # _retriever = _db.as_retriever()


def query_db(db: Chroma, query: str, use_retriever: bool = False) -> list[str]:
    _retriever = db.as_retriever()
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs


message_log: List[Tuple[str, str]] = []


def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================

    db = get_chroma_db()

    llm = ChatOpenAI(
        temperature=0.1,
        max_tokens=300,
        model="gpt-3.5-turbo",
    )

    global message_log
    if not message_log:
        message_log = [("system", read_file("system_message_template.txt"))]

    user_message = request.userRequest.utterance

    # 모든 발화를 message_log에 추가하는 것은 좋지 않다
    message_log.append(("human", user_message))

    message_log_prompt = ChatPromptTemplate.from_messages(message_log)
    chain = LLMChain(
        llm=llm,
        prompt=message_log_prompt,
        verbose=True,
    )
    # ai_response = chain.run(
    #     document=read_file("project_data_카카오톡채널.txt"),
    # )
    ai_response = chain.run(
        document=query_db(db, user_message),
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
    # 참고링크2 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    url = request.userRequest.callbackUrl

    if url:
        requests.post(url, json=payload)
