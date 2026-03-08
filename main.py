from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

from src.rag import VectorStoreManager, InsuranceRAG
from data.models.qa_model import QuestionRequest, AnswerResponse

load_dotenv()

rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag

    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vectorstore()

    print("📦 Vector store count:", vector_store._collection.count())

    rag = InsuranceRAG(
        vectorstore=vector_store,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    yield  # app runs here

    print("🛑 Shutting down API")


app = FastAPI(lifespan=lifespan)


@app.post("/query/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = rag.query(req.question)
    return {"answer": result["answer"]}


@app.get("/health")
def root():
    return {"status": "API is running"}