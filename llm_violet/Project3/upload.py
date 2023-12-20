# -*- coding: utf-8 -*-
import os

import openai
from langchain.document_loaders import (NotebookLoader, TextLoader,
                                        UnstructuredMarkdownLoader)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# OpenAI API Key 파일에서 읽어오기
with open("openai_key.txt", "r") as f:
    k = f.read()
    openai.api_key = k
    os.environ["OPENAI_API_KEY"] = k

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(PROJECT_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"
LOADER_DICT = {
    "py": TextLoader,
    "txt": TextLoader,
    "md": UnstructuredMarkdownLoader,
    "ipynb": NotebookLoader,
}


def upload_embedding_from_file(file_path):
    loader = LOADER_DICT.get(file_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(
        separator="\n#",
        chunk_size=300,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)
    for d in docs:
        print(d, end="\n\n")
    print("\n\n\n")

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print("db success")


def clear_collection():
    _db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    _db.delete_collection()


if __name__ == "__main__":
    clear_collection()
    upload_embedding_from_file(os.path.join(PROJECT_DIR, "project_data_카카오소셜.txt"))
    upload_embedding_from_file(os.path.join(PROJECT_DIR, "project_data_카카오싱크.txt"))
    upload_embedding_from_file(os.path.join(PROJECT_DIR, "project_data_카카오톡채널.txt"))
