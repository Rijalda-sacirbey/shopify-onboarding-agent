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
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

os.environ['USER_AGENT'] = 'paai-agent'
os.environ["LANGSMITH_TRACING"] = "true"

# os.environ["LANGSMITH_API_KEY"] = "add-your-key"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API key: ")

if not os.environ.get("GROQ_API_KEY"):
    # os.environ["GROQ_API_KEY"] = "add-your-key"
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# os.environ["HF_TOKEN"] = "add-your-key"
os.environ["HF_TOKEN"] = getpass.getpass("Enter Huggingface token: ")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")
    # os.environ["MISTRAL_API_KEY"] = "add-your-key"

embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = InMemoryVectorStore(embeddings)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), 
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))), 
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(documents=all_splits)

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2) # k is the number of search results
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    system_message = SystemMessage( # Append a clear system instruction
        "You are an assistant. Only use the 'retrieve' tool for domain-specific questions. "
        "For general or conversational queries such as greetings or 'can you help me with something', "
        "provide a direct answer without calling any tools."
    )

    messages = [system_message] + state["messages"]
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(messages)

    # Check if the response includes an empty tool-use message
    tool_calls = getattr(response, "tool_calls", None)

    # If tool_calls attribute exists but is empty, assume no tool was needed.
    if tool_calls is not None and not tool_calls:
        fallback = {"role": "ai", "content": "Hello! How can I assist you today?"}
        return {"messages": [fallback]}
    return {"messages": [response]}

tools = ToolNode([retrieve])
