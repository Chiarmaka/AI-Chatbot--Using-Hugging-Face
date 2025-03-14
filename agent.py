from langchain.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import Graph
from rag_pipeline import query_faiss
from config import GEMINI_API_KEY

# Load Gemini AI Model
gemini_chat = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY)

def chat_with_gemini(user_query):
    """Queries Gemini API for response."""
    return gemini_chat.predict(user_query)

# Define LangGraph flow
graph = Graph()
graph.add_node("faiss_retrieval", query_faiss)
graph.add_node("gemini_chat", chat_with_gemini)

graph.set_entry_point("faiss_retrieval")
graph.add_edge("faiss_retrieval", "gemini_chat")

executor = graph.compile()

def agent_respond(user_input):
    """Runs LangGraph agent to generate response."""
    result = executor.run({"query": user_input})
    return result["gemini_chat"]
