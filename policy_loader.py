import os
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import POLICY_FILE_PATH, FAISS_INDEX_PATH

def load_policy():
    """Loads company policy text from file."""
    with open(POLICY_FILE_PATH, "r", encoding="utf-8") as file:
        return file.read()

def build_faiss_index():
    try:
        print("🚀 Building FAISS index...")

        # Load policy text
        policy_text = load_policy()
        print("✅ Policy text loaded.")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        policy_chunks = text_splitter.split_text(policy_text)

        # Load Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create FAISS index
        vector_store = FAISS.from_texts(policy_chunks, embedding=embeddings)
        print("✅ FAISS index created.")

        # Save FAISS index
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump((vector_store, policy_chunks), f)  
        print("✅ FAISS index saved successfully!")

        # Save policy chunks (metadata)
        metadata_path = FAISS_INDEX_PATH.replace(".pkl", "_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(policy_chunks, f)
        print("✅ Policy metadata saved!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    build_faiss_index()
