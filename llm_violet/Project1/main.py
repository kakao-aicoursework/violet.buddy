import json
import os
import sys
import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import scrolledtext

import chromadb
import openai
import pandas as pd

cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
channel_txt = os.path.join(dir_path, "project_data_카카오톡채널.txt")


# OpenAI API Key를 openai_key.txt 파일에서 읽어오기
with open("openai_key.txt", "r") as f:
    openai_key = f.read()
    openai.api_key = openai_key
    print("OpenAI API Key:", openai.api_key)


# read the whole text file
def read_file(file_path=channel_txt):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# read and parse to dict
def get_kakaotalk_doc_dict(file_path=channel_txt):
    kakaotalk_str = read_file(file_path)
    kakaotalk_str = kakaotalk_str.replace("\n\n\n", "\n\n")
    paragraphs = kakaotalk_str.split("\n\n")

    parsed = []
    topic = None

    for paragraph in paragraphs:
        lines = paragraph.split("\n")
        first_line = lines[0].strip()

        if first_line.startswith("#"):
            topic = first_line[1:].strip()
            content = "\n".join(lines[1:]).strip()
        else:
            content = paragraph

        parsed.append({"topic": topic, "content": content})

    return parsed


# parsed = get_kakaotalk_doc_dict()
# for p in parsed:
#     print(p, "\n")
# sys.exit(0)


# ### chromadb init
# client = chromadb.PersistentClient()
# collection = client.get_or_create_collection(
#     name="kakaotalk_API", metadata={"hnsw:space": "l2"}
# )


def send_message(message_log, functions, gpt_model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=message_log,
        temperature=temperature,
        # functions=functions,
        # function_call="auto",
    )

    response_message = response["choices"][0]["message"]
    return response_message.content


def main():
    message_log = [
        {
            "role": "system",
            "content": """
            You are the QnA chatbot for a KakaoTalk channel API service.
            Your user will be Korean, so communicate in Korean.
            At first, greet the user and ask how you can help.
            When user asks irrelevant questions, say 'I can't answer that' or 'I don't know'.
            """,
        },
        {"role": "assistant", "content": read_file()},  # 텍스트 파일 통째로 넘기기
    ]

    def show_popup_message(window, message):
        popup = tk.Toplevel(window)
        popup.title("")

        # 팝업 창의 내용
        label = tk.Label(popup, text=message, font=("맑은 고딕", 20))
        label.pack(expand=True, fill=tk.BOTH)

        # 팝업 창의 크기 조절하기
        window.update_idletasks()
        popup_width = label.winfo_reqwidth() + 20
        popup_height = label.winfo_reqheight() + 20
        popup.geometry(f"{popup_width}x{popup_height}")

        # 팝업 창의 중앙에 위치하기
        window_x = window.winfo_x()
        window_y = window.winfo_y()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        popup_x = window_x + window_width // 2 - popup_width // 2
        popup_y = window_y + window_height // 2 - popup_height // 2
        popup.geometry(f"+{popup_x}+{popup_y}")

        popup.transient(window)
        popup.attributes("-topmost", True)

        popup.update()
        return popup

    def on_send(initial=False):
        user_input = user_entry.get()
        user_entry.delete(0, tk.END)

        if user_input.lower() == "quit":
            window.destroy()
            return

        message_log.append({"role": "user", "content": user_input})
        conversation.config(state=tk.NORMAL)  # 이동

        if not initial:
            conversation.insert(tk.END, f"You: {user_input}\n", "user")  # 이동
        thinking_popup = show_popup_message(window, "처리중...")
        window.update_idletasks()
        # '생각 중...' 팝업 창이 반드시 화면에 나타나도록 강제로 설정하기
        response = send_message(message_log, functions=[])
        thinking_popup.destroy()

        message_log.append({"role": "assistant", "content": response})

        # 태그를 추가한 부분(1)
        conversation.insert(tk.END, f"AI Assistant: {response}\n", "assistant")
        conversation.config(state=tk.DISABLED)
        # conversation을 수정하지 못하게 설정하기
        conversation.see(tk.END)

    window = tk.Tk()
    window.title("GPT AI")

    font = ("맑은 고딕", 20)

    conversation = scrolledtext.ScrolledText(
        window, wrap=tk.WORD, bg="#f0f0f0", font=font
    )
    # width, height를 없애고 배경색 지정하기(2)
    conversation.tag_configure("user", background="#c9daf8")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.tag_configure("assistant", background="#e4e4e4")
    # 태그별로 다르게 배경색 지정하기(3)
    conversation.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # 창의 폭에 맞추어 크기 조정하기(4)

    input_frame = tk.Frame(window)  # user_entry와 send_button을 담는 frame(5)
    input_frame.pack(fill=tk.X, padx=10, pady=10)  # 창의 크기에 맞추어 조절하기(5)

    user_entry = tk.Entry(input_frame)
    user_entry.pack(fill=tk.X, side=tk.LEFT, expand=True)

    send_button = tk.Button(input_frame, text="Send", command=on_send)
    send_button.pack(side=tk.RIGHT)

    on_send(initial=True)

    window.bind("<Return>", lambda event: on_send())
    window.mainloop()


if __name__ == "__main__":
    main()
