## AI Chatbot with FAISS & Hugging Face
Conversational AI Chatbot powered by FAISS for retrieval-based responses and Hugging Face embeddings. This chatbot provides human-like responses using a customer service policy knowledge base.

---

##  Features
Retrieval-Augmented Generation (RAG)** with FAISS.  
Hugging Face embeddings for similarity search. 
Conversational responses** (not just raw text dumps).  
Handles queries on orders, refunds, shipping, and more.**  
Multi-turn conversation support. 

---

##  Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Chatbot.git
cd AI-Chatbot
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Keys
Create a `.env` file and add:
```
HUGGINGFACE_TOKEN=your_huggingface_api_key
```
Or set it as an environment variable:
```bash
export HUGGINGFACE_TOKEN=your_huggingface_api_key  # On Windows: set HUGGINGFACE_TOKEN=your_huggingface_api_key
```

---

## Project Structure
```
📦 AI-Chatbot
 ┣ 📂 Company_policy.txt        # Stores the company policy text
 ┣ 📂 faiss_index              # FAISS storage
 ┣ 📜 config.py                # Configuration settings
 ┣ 📜 policy_loader.py         # Loads & processes company policies into FAISS
 ┣ 📜 rag_pipeline.py          # Main RAG model handling FAISS queries
 ┣ 📜 main.py                  # Runs the chatbot in interactive mode
 ┣ 📜 requirements.txt         # Dependencies
 ┣ 📜 README.md                # Project Documentation
```

---

##  How to Use
### 1️⃣ Index the Policy Data
Before using the chatbot, **process & store the company policies** into FAISS:
```bash
python policy_loader.py
```

### 2️⃣ Run the Chatbot
```bash
python main.py
```
You'll see:
```
✅ FAISS Index loaded successfully!
✅ Chatbot is ready! Type 'exit' to quit.
You: When will my order arrive?
Chatbot: Your order should arrive in 5-7 business days for standard shipping. Would you like me to check your tracking details?
```

---

##  Troubleshooting
| Problem | Solution |
|---------|----------|
| `FAISS index not found` | Run `policy_loader.py` to generate the FAISS index. |
| `Hugging Face API key missing` | Set `HUGGINGFACE_TOKEN` in `.env` or as an environment variable. |
| `Chatbot gives generic responses` | Check `policy_loader.py` to confirm policies were indexed correctly. |

---

##  Next Steps
Improve Conversational Flow by integrating GPT-4 for better responses.  
Add a Web UI using FastAPI & React for real-time interactions.  
Log User Queries to improve responses over time.  


 