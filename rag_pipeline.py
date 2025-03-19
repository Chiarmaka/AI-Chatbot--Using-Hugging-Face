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
    """Loads the FAISS index and retrieves stored policy text."""
    try:
        with open(FAISS_INDEX_PATH, "rb") as f:
            vector_store, policy_chunks = pickle.load(f)  # ✅ Load both FAISS & text chunks

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("✅ FAISS Index and policy text loaded successfully!")
        return vector_store, policy_chunks, embeddings  # ✅ Return stored policy chunks

    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None, None, None  # Handle errors gracefully



# Initialize FAISS and Embeddings
vector_store, policy_chunks, embeddings = load_faiss_index()


def query_faiss(user_query, vector_store, policy_chunks, embeddings):
    """Retrieves the closest matching policy chunk from FAISS and formats a human-like response."""

    if vector_store is None or embeddings is None or policy_chunks is None:
        return "I'm sorry, I am unable to access the policy database right now. Please try again later.", 0.0

    # Convert query into an embedding
    query_embedding = embeddings.embed_query(user_query)
    query_embedding = np.array([query_embedding]).astype("float32")  # Ensure compatibility

    # Retrieve top 2 matches from FAISS
    D, I = vector_store.index.search(query_embedding, k=2)

    print(f"🔍 Query: {user_query}")
    print(f"✅ Retrieved indices: {I[0]}")
    print(f"✅ Similarity scores: {D[0]}")

    if len(I[0]) == 0 or D[0][0] > 1.0:  # Adjust threshold if needed
        return "I'm sorry, but I couldn't find that information. You can visit our website or contact customer support for further assistance.", 0.5

    # ✅ Retrieve stored policy text using the FAISS index
    retrieved_texts = [policy_chunks[idx] for idx in I[0] if idx != -1]

    if not retrieved_texts:
        return "I'm sorry, but I couldn't find that information. Please check our website or contact support.", 0.5

    # ✅ Format response in a human-like way
    formatted_response = f"Based on our company policy:\n\n{retrieved_texts[0]}\n\nWould you like more details on this?"

    return formatted_response, 0.9


