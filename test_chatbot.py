from rag_pipeline import query_faiss, load_faiss_index

# Load FAISS index and embeddings
vector_store, embeddings = load_faiss_index()

if vector_store is None or embeddings is None:
    print("❌ Failed to load FAISS index. Chatbot cannot start.")
    exit()

# Example queries
queries = [
    "What is the refund policy?",
    "Do you offer international shipping?",
    "Can I cancel my order?",
    "How do I contact you?",
    "What is your return policy?",
    "How long does shipping take?",
    "What payment methods do you accept?",
    "What if my package is lost?",
    "Do you have a loyalty program?",
    "Are there discounts available?"
]

# Run queries if FAISS index is loaded
for query in queries:
    response, confidence_score = query_faiss(query, vector_store, embeddings)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Confidence: {confidence_score}")
    print("-" * 20)
