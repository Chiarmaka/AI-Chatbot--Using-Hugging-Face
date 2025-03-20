from rag_pipeline import query_faiss, load_faiss_index

def main():
    print(" Chatbot is ready! Type 'exit' to quit.")
    
    vector_store, policy_chunks, embeddings = load_faiss_index()
    if vector_store is None or embeddings is None:
        print(" Failed to load FAISS index. Chatbot cannot start.")
        return

    conversation_history = []  # Stores previous interactions

    while True:
        user_query = input("\nYou: ").strip()

        if user_query.lower() == "exit":
            print("Chatbot: Thank you for chatting! Have a great day. 😊")
            break

        # Append user input to history
        conversation_history.append(f"You: {user_query}")

        # Query FAISS and get response
        response, confidence_score = query_faiss(user_query, vector_store, policy_chunks, embeddings)

        # Append chatbot response to history
        conversation_history.append(f"Chatbot: {response}")

        # Print chatbot response
        print(f"Chatbot: {response} (Confidence: {confidence_score})")

if __name__ == "__main__":
    main()
