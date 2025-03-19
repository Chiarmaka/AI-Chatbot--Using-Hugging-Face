from rag_pipeline import load_faiss_index, query_faiss

def main():
    print("✅ Loading FAISS Index...")
    
    # ✅ Ensure correct unpacking of 3 values
    vector_store, documents, embeddings = load_faiss_index()
    
    if vector_store is None or embeddings is None:
        print("❌ Failed to load FAISS index. Exiting...")
        return

    print("✅ Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting chatbot.")
            break

        response, confidence = query_faiss(user_query, vector_store, documents, embeddings)
        print(f"Chatbot: {response}")
        print(f"Confidence: {confidence}\n")

if __name__ == "__main__":
    main()
