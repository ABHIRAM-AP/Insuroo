from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile
import threading

from src.voice.whisper_stt import WhisperSTT
from src.voice.sarvam_tts import SarvamTTS
from src.voice.groq_stt import GroqSTT

from src.rag import VectorStoreManager, InsuranceRAG
from data.models.qa_model import QuestionRequest, AnswerResponse

load_dotenv()

# ---------------- GLOBALS ---------------- #
rag = None
is_loading = False

stt_local: WhisperSTT = None
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
        print("🎙️ Using Whisper STT (Local)")
        stt_local = WhisperSTT(model_name="base")

    # Initialize TTS
    print("🔊 Initializing Sarvam TTS...")
    tts = SarvamTTS()

    # Start background RAG loading (NON-BLOCKING)
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
            result = stt_local.transcribe(tmp_path, language="hi")
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