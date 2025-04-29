import getpass
import os
import bs4
from langchain.chat_models import init_chat_model
!pip install -U langchain-groq
!pip install -qU langchain-mistralai
from langchain_mistralai import MistralAIEmbedding
!pip install -qU langchain_community beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
!pip install -qU langgraph
from langgraph.graph impo