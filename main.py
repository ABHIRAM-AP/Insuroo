from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile
import threading

from src.voice.sarvam_tts import SarvamTTS
from src.voice.groq_stt import GroqSTT

from src.rag import VectorStoreManager, InsuranceRAG
from src.recommendation.recommender import PolicyRecommender
from data.models.qa_model import QuestionRequest, AnswerResponse
from data.models.user_profile import UserProfile, RecommendationResponse

load_dotenv()

# ---------------- GLOBALS ---------------- #
rag = None
is_loading = False

stt_cloud: GroqSTT = None
tts: SarvamTTS = None


# ---------------- RAG LOADER ---------------- #
def load_rag():
    global rag, is_loading

    if rag is not None:
        return rag

    if is_loading:
        return None

    is_loading = True
    print("📦 Loading vector store...")

    try:
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.load_vectorstore()

        print("📦 Vector store count:", vector_store._collection.count())

        rag = InsuranceRAG(
            vectorstore=vector_store,
            api_key=os.getenv("GEMINI_API_KEY")
        )

        print("✅ RAG Loaded")

    except Exception as e:
        print("❌ Error loading RAG:", str(e))
        rag = None

    finally:
        is_loading = False

    return rag


# ---------------- BACKGROUND PRELOAD ---------------- #
def preload_rag():
    print("🚀 Background RAG preload started...")
    load_rag()


# ---------------- LIFESPAN ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_local, stt_cloud, tts

    print("⚡ FastAPI starting...")

    # Initialize STT
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("🎙️ Using Groq STT (Cloud)")
        stt_cloud = GroqSTT(api_key=groq_key)
    else:
        print("❌ Groq API Key missing. Cloud STT will not work.")

    # Initialize TTS
    print("🔊 Initializing Sarvam TTS...")
    tts = SarvamTTS()

    # Optimization: Skip preloading in DEV_MODE to speed up reloads
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    if dev_mode:
        print("🚀 [DEV_MODE] Skipping background RAG preload (will lazy load on first request)")
    else:
        # Start background RAG loading (NON-BLOCKING) in production
        threading.Thread(target=preload_rag, daemon=True).start()

    yield

    print("🛑 Shutting down API")


# ---------------- APP INIT ---------------- #
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- HEALTH ---------------- #
@app.get("/health")
def root():
    return {"status": "API is running"}


# ---------------- TEXT RAG ---------------- #
@app.post("/query/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    global rag

    if rag is None:
        rag_instance = load_rag()

        if rag_instance is None:
            return {
                "answer": "⏳ Model is warming up, please try again in a few seconds"
            }
    else:
        rag_instance = rag

    result = rag_instance.query(req.question)
    return {"answer": result["answer"]}


# ---------------- RECOMMENDATION ---------------- #
@app.post("/query/recommend", response_model=RecommendationResponse)
def recommend_policies(profile: UserProfile):
    global rag

    if rag is None:
        rag_instance = load_rag()

        if rag_instance is None:
            return {
                "user_name": profile.name,
                "recommendations": [],
                "summary": "⏳ Thinking... Our recommendation engine is warming up. Please try again in a few seconds."
            }
    else:
        rag_instance = rag

    recommender = PolicyRecommender(rag_instance)
    result = recommender.recommend(profile)

    return result


# ---------------- VOICE STT ---------------- #
@app.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if stt_cloud:
            result = stt_cloud.transcribe(tmp_path, language="hi")
        else:
            result = {"success": False, "error": "Cloud STT not configured"}
    finally:
        os.remove(tmp_path)

    if not result["success"]:
        return {"text": "", "error": result.get("error", "Transcription failed")}

    return {"text": result["text"], "language": result["language"]}


# ---------------- VOICE TTS ---------------- #
@app.post("/voice/speak")
def speak_text(req: dict):
    text = req.get("text", "")

    if not text.strip():
        return Response(content=b"", media_type="audio/wav")

    audio_path = tts.synthesize(text[:450])

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.remove(audio_path)

    return Response(content=audio_bytes, media_type="audio/wav")