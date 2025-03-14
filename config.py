# config.py

# import os

# # API Keys and File Paths
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBKsIf_wDcZ0_PEgclEtL6VIqGZT1Km6xM")
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_huggingface_api_key")

# # File Paths
# POLICY_FILE_PATH = "Company_policy.txt"
# FAISS_INDEX_PATH = "faiss_index"

# # Model Names
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # LLM Configuration
# LLM_MODEL_NAME = "google/gemini-pro"

# # Token Configuration
# TOKEN_LIMIT = 4096
import os

# API Keys
GEMINI_API_KEY = os.getenv("AIzaSyBKsIf_wDcZ0_PEgclEtL6VIqGZT1Km6xM")
#HUGGINGFACE_API_KEY = os.getenv("your_huggingface_api_key")  # Load from env variables

# Paths
POLICY_FILE_PATH = "Company_policy.txt"
FAISS_INDEX_PATH = "faiss_index.josn"

# Support Contact Details
SUPPORT_EMAIL = "support@agricultureoption.com"
SUPPORT_PHONE = "+1234567890"
