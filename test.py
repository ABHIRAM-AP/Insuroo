import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
CHROMA_DIR = Path("data/chroma_db").resolve()

print("FILES IN CHROMA_DIR:")
for p in CHROMA_DIR.iterdir():
    print(" -", p.name)

emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",api_key=os.environ['GEMINI_API_KEY'])

db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=emb,
    collection_name="insurance_docs"
)

print("COUNT =", db._collection.count())