import os
import pickle
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_PATH


# Load Hugging Face Authentication Token
HF_AUTH_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_AUTH_TOKEN:
    raise ValueError("❌ Hugging Face API token is missing. Set it as an environment variable.")

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


# Load FAISS Index and Metadata
def load_faiss_index():
    """
    Loads the FAISS index, document texts, and embeddings.
    """
    try:
        # Load FAISS index
        with open(FAISS_INDEX_PATH, "rb") as f:
            vector_store = pickle.load(f)

        # Load policy document texts (ensure this file exists!)
        metadata_path = FAISS_INDEX_PATH.replace(".pkl", "_metadata.pkl")
        with open(metadata_path, "rb") as f:
            documents = pickle.load(f)  # ✅ Load document texts

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("✅ FAISS Index, documents, and embeddings loaded successfully!")
        return vector_store, documents, embeddings  # ✅ Ensure 3 return values
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None, None, None



# Initialize FAISS and Embeddings
vector_store, documents, embeddings = load_faiss_index()


def query_faiss(user_query, vector_store, documents, embeddings):
    """Retrieves the closest matching policy chunk from FAISS and returns relevant information."""
    
    if vector_store is None or embeddings is None:
        return "❌ FAISS index is not loaded correctly.", 0.0

    # Convert the query into an embedding
    query_embedding = embeddings.embed_query(user_query)
    query_embedding = np.array([query_embedding]).astype("float32")  # Ensure FAISS accepts numpy array

    # Retrieve top-k matches from FAISS
    D, I = vector_store.index.search(query_embedding, k=2)  # Get top 2 matches

    # Debugging output
    print(f"🔍 Query: {user_query}")
    print(f"✅ Retrieved indices: {I[0]}")
    print(f"✅ Similarity scores: {D[0]}")

    # **Lower similarity threshold** to ensure results are not ignored
    if len(I[0]) == 0 or D[0][0] > 1.2:  # **Adjust the threshold**
        return "Sorry, I couldn't find relevant information.", 0.5

    # Retrieve document texts based on indices
    retrieved_texts = [documents[idx] for idx in I[0] if idx != -1 and idx < len(documents)]
    
    # **Check if retrieved_texts is empty**
    if not retrieved_texts:
        print("❌ No relevant documents found!")
        return "Sorry, I couldn't find relevant information.", 0.5

    retrieved_text = "\n".join(retrieved_texts)
    print(f"📜 Retrieved Text:\n{retrieved_text}")  # Debugging output

    return retrieved_text, 0.9

