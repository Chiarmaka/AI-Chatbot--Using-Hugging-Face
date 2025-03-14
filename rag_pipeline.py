import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.text_splitter import RecursiveCharacterTextSplitter

from config import FAISS_INDEX_PATH

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.generation_config.pad_token_id = tokenizer.pad_token_id

def load_faiss():
    """Loads FAISS index from file."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH.replace(".pkl", "_metadata.pkl"), "rb") as f:
        vector_store = pickle.load(f)
    return index, vector_store

index, vector_store = load_faiss()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_faiss(user_query):
    """Retrieves closest policy chunk and generates response."""
    query_embedding = embeddings.embed_query(user_query)
    D, I = index.search([query_embedding], k=1)  # Retrieve top match

    if len(I[0]) == 0 or D[0][0] > 0.5:  # Threshold for match
        return "Sorry, I couldn't find relevant information.", 0.5

    retrieved_text = vector_store[I[0][0]]  # Retrieve relevant policy text

    # Generate AI response using GPT-2
    inputs = tokenizer(f"User: {user_query}\nAI: ", return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=150)
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_response, 0.9  # Assume high confidence

