from rag_pipeline import query_faiss

def test_rag_pipeline():
    test_queries = [
        "What is the refund policy?",
        "Do you offer international shipping?",
        "Can I cancel my order?",
    ]
    
    for query in test_queries:
        response, confidence = query_faiss(query)
        print(f"Query: {query}\nResponse: {response}\nConfidence: {confidence}\n")

if __name__ == "__main__":
    test_rag_pipeline()
