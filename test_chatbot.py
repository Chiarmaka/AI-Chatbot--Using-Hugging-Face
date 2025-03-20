from rag_pipeline import query_faiss, load_faiss_index

# Load FAISS index and embeddings
vector_store, policy_chunks, embeddings = load_faiss_index()

if vector_store is None or embeddings is None:
    print(" Failed to load FAISS index. Chatbot cannot start.")
    exit()

# Example queries
queries = [
    "What is the refund policy?",
    "Do you offer international shipping?",
    "when will my order arrive",
    "How do I contact you?",
    "What is your return policy?",
    "How long does shipping take?",
    "What if my package is lost?",
    "Do you have a loyalty program?",
    
]

# Run queries if FAISS index is loaded
for query in queries:
    response, confidence_score = query_faiss(query, vector_store, policy_chunks, embeddings)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Confidence: {confidence_score}")
    print("-" * 20)
