# 🛡️ Insuroo

**Insuroo** is an AI-powered insurance assistant backend built with FastAPI. It helps users query insurance policy information through a Retrieval-Augmented Generation (RAG) pipeline and get personalized policy recommendations — all accessible via text or voice (speech-to-text and text-to-speech).

---

## ✨ Features

- 🤖 **RAG-based Q&A** — Ask questions about insurance policies and get accurate, context-aware answers powered by Google Gemini and ChromaDB.
- 📋 **Policy Recommendations** — Submit a user profile and receive tailored insurance policy suggestions.
- 🎙️ **Voice Transcription (STT)** — Upload audio files and get transcriptions via Groq's cloud Speech-to-Text API (supports Hindi and more).
- 🔊 **Text-to-Speech (TTS)** — Convert text answers into audio using Sarvam AI's TTS service.
- ⚡ **Background Preloading** — The RAG vector store loads in a background thread at startup (skippable in dev mode for faster reloads).
- 🌐 **CORS Enabled** — Ready for integration with any frontend.

---

## 🗂️ Project Structure

```
Insuroo/
├── main.py                  # FastAPI app, routes, lifespan management
├── run_dev.py               # Dev-mode runner
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not committed)
├── config/                  # Configuration files
├── data/
│   └── models/
│       ├── qa_model.py      # Pydantic models for Q&A requests/responses
│       └── user_profile.py  # Pydantic models for user profile & recommendations
└── src/
    ├── rag.py               # VectorStoreManager & InsuranceRAG (ChromaDB + Gemini)
    ├── recommendation/
    │   └── recommender.py   # PolicyRecommender logic
    └── voice/
        ├── groq_stt.py      # Groq cloud Speech-to-Text
        └── sarvam_tts.py    # Sarvam AI Text-to-Speech
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- API keys for: **Google Gemini**, **Groq**, and **Sarvam AI**

### 1. Clone the repository

```bash
git clone https://github.com/ABHIRAM-AP/Insuroo.git
cd Insuroo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key   # if required by sarvam_tts
DEV_MODE=false                        # set to true to skip background RAG preload
```

### 4. Run the server

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Development (faster reloads, lazy RAG loading):**
```bash
DEV_MODE=true python run_dev.py
```

---

## 📡 API Endpoints

### `GET /health`
Returns the health status of the API.

```json
{ "status": "API is running" }
```

---

### `POST /query/ask`
Ask a question about insurance policies.

**Request body:**
```json
{ "question": "What does my health insurance cover?" }
```

**Response:**
```json
{ "answer": "Your health insurance covers..." }
```

---

### `POST /query/recommend`
Get personalized insurance policy recommendations based on a user profile.

**Request body** (example fields — see `data/models/user_profile.py` for full schema):
```json
{
  "name": "Ravi Kumar",
  "age": 35,
  "income": 800000,
  "dependents": 2,
  "existing_policies": ["term life"]
}
```

**Response:**
```json
{
  "user_name": "Ravi Kumar",
  "recommendations": [...],
  "summary": "Based on your profile, we recommend..."
}
```

---

### `POST /voice/transcribe`
Transcribe an uploaded audio file to text.

**Request:** `multipart/form-data` with a `file` field (`.wav` audio).

**Response:**
```json
{ "text": "transcribed text here", "language": "hi" }
```

---

### `POST /voice/speak`
Convert text to speech and return a `.wav` audio file.

**Request body:**
```json
{ "text": "Your policy covers hospitalization." }
```

**Response:** Binary `audio/wav` content.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI |
| RAG / LLM | LangChain + Google Gemini |
| Vector Store | ChromaDB |
| PDF Parsing | PyPDF |
| Speech-to-Text | Groq API |
| Text-to-Speech | Sarvam AI |
| Data Validation | Pydantic |
| HTTP Client | httpx, requests |

---

## 📝 Environment Variables Reference

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for RAG | ✅ |
| `GROQ_API_KEY` | Groq API key for cloud STT | ✅ |
| `DEV_MODE` | Set to `true` to skip RAG preload at startup | ❌ |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source. See the repository for license details.
