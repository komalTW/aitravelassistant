# OpenAI API key (set this if using OpenAI)
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HUGGINGFACE_MODEL_ID = os.getenv("HUGGINGFACE_MODEL_ID", "google/flan-t5-base")
FALLBACK_MODEL_PROVIDER = os.getenv("FALLBACK_MODEL_PROVIDER", "huggingface").lower()

# Model provider: 'huggingface' or 'openai'
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
FASTAPI_URL = "http://localhost:8000"