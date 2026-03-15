from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile

from src.voice.whisper_stt import WhisperSTT
from src.voice.sarvam_tts import SarvamTTS

from src.rag import VectorStoreManager, InsuranceRAG
from data.models.qa_model import QuestionRequest, AnswerResponse

load_dotenv()

rag = None
stt: WhisperSTT = None
tts: SarvamTTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, stt, tts

    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vectorstore()

    print("📦 Vector store count:", vector_store._collection.count())

    rag = InsuranceRAG(
        vectorstore=vector_store,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Initialize voice components
    print("🎙️ Initializing Whisper STT...")
    stt = WhisperSTT(model_name="base")

    print("🔊 Initializing Sarvam TTS...")
    tts = SarvamTTS()

    yield  # app runs here

    print("🛑 Shutting down API")


app = FastAPI(lifespan=lifespan)

# Allow Flutter app (any origin in dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = rag.query(req.question)
    return {"answer": result["answer"]}


@app.get("/health")
def root():
    return {"status": "API is running"}


@app.post("/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts a WAV audio file upload.
    Returns the transcribed text using Whisper STT.
    """
    contents = await file.read()

    # Write to a temp WAV file for Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = stt.transcribe(tmp_path, language="hi")
    finally:
        os.remove(tmp_path)

    if not result["success"]:
        return {"text": "", "error": result.get("error", "Transcription failed")}

    return {"text": result["text"], "language": result["language"]}


@app.post("/voice/speak")
def speak_text(req: dict):
    """
    Accepts {"text": "..."}.
    Returns WAV audio bytes synthesized by Sarvam TTS.
    """
    text = req.get("text", "")
    if not text.strip():
        return Response(content=b"", media_type="audio/wav")

    # Sarvam TTS synthesizes to a temp file — read bytes and return them
    audio_path = tts.synthesize(text[:450])  # first chunk only; Flutter handles long text

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.remove(audio_path)

    return Response(content=audio_bytes, media_type="audio/wav")