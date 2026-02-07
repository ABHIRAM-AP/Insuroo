from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 🔒 Anchor path to project root (NOT CWD)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

class VectorStoreManager:
    def __init__(self):
        self.persist_directory = str(CHROMA_DIR)

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )

    def load_vectorstore(self):
        print("📍 Loading Chroma from:", self.persist_directory)

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name="insurance_docs"
        )