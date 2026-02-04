import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR=BASE_DIR/"data"
DOCS_DIR = DATA_DIR / "insurance_docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
MODELS_DIR = DATA_DIR / "models"

for directory in [DATA_DIR, DOCS_DIR, CHROMA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True,exist_ok=True)


GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API KEY not found in .env file")

MODEL_NAME = os.getenv("MODEL_NAME",default="gemini-1.5-flash-latest")
TEMPERATURE = float(os.getenv("TEMPERATURE",default="0.3")) 
MAX_TOKENS = int(os.getenv("MAX_TOKENS","500"))

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL="models/embedding-001"
NUM_RETRIEVAL_CHUNKS=3


if __name__=="__main__":
    print("Config file loaded")
    print(f" Base directory: {BASE_DIR}")
    print(f" Documents: {DOCS_DIR}")
    print(f" Vector DB: {CHROMA_DIR}")
    print(f" Model Name: {MODEL_NAME}")