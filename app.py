from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
documents = load_documents()

def main():
    load_dotenv() #looks for a .env file in current directory and loads vars 
    api_key = os.getenv("OPEN_API_KEY")
    chat_model = ChatOpenAI(openai_api_key=api_key)

