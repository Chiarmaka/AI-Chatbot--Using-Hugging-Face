from rag_pipeline import query_faiss

def agent_respond(user_input):
    """Runs retrieval and generates response."""
    response, confidence = query_faiss(user_input)
    return response
