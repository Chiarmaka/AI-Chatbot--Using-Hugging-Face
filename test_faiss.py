import pickle

FAISS_INDEX_PATH = "faiss_index.pkl"

try:
    with open(FAISS_INDEX_PATH, "rb") as f:
        vector_store = pickle.load(f)
    print("✅ FAISS index exists and loaded successfully!")
    print(f"Number of documents in FAISS: {len(vector_store.docstore._dict)}")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")

docs = vector_store.similarity_search("when will my order arrive")
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)