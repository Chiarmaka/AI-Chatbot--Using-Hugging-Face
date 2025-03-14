import faiss
import pickle
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import POLICY_FILE_PATH, FAISS_INDEX_PATH

def load_policy():
    """Loads company policy text."""
    with open(POLICY_FILE_PATH, "r", encoding="utf-8") as file:
        return file.read()

def build_faiss_index():
    """Creates FAISS vector store from policy text."""
    policy_text = load_policy()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    policy_chunks = text_splitter.split_text(policy_text)

    # Load Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    vector_store = FAISS.from_texts(policy_chunks, embedding=embeddings)

    # Save index
    faiss.write_index(vector_store.index, FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH.replace(".pkl", "_metadata.pkl"), "wb") as f:
        pickle.dump(vector_store, f)

    print("✅ FAISS index built successfully!")

if __name__ == "__main__":
    build_faiss_index()
