from dotenv import load_dotenv
from fastapi import FastAPI
import os

from src.rag import VectorStoreManager,InsuranceRAG
# from src.rag import DocumentProcessor
from data.models.qa_model import QuestionRequest,AnswerResponse

load_dotenv()
app = FastAPI()

vector_store_manager = VectorStoreManager()
vector_store = vector_store_manager.load_vectorstore()

print("📦 Vector store count:", vector_store._collection.count())

rag = InsuranceRAG(
    vectorstore=vector_store,
    api_key=os.environ['GEMINI_API_KEY']
)


@app.post("/query/ask",response_model=AnswerResponse)
def ask_question(req:QuestionRequest):

    result=rag.query(req.question)

    return {
        "answer":result["answer"]
    }



@app.get("/health")
def root():
    return {
        "status":"API is running"
    }