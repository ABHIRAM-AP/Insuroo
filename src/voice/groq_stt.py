import os
import time
from typing import Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqSTT:
    """
    Cloud-based Speech-to-Text using Groq's Whisper API
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Groq STT
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found. GroqSTT will not work.")
        else:
            self.client = Groq(api_key=self.api_key)
            print("✅ Groq STT ready!")

    def transcribe(self, audio_file_path: str, language: str = "hi") -> Dict:
        """
        Transcribe audio to text using Groq's Whisper API
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (hi, en, etc.)
        
        Returns:
            dict: {
                "text": str,
                "language": str,
                "duration": float,
                "success": bool
            }
        """
        if not self.api_key:
            return {"text": "", "error": "GROQ_API_KEY missing", "success": False}

        print(f"\n🎤 Transcribing audio via Groq...")
        print(f"   File: {audio_file_path}")
        
        start_time = time.time()
        
        try:
            with open(audio_file_path, "rb") as file:
                # Transcribe with Groq
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), file.read()),
                    model="whisper-large-v3",
                    language=language,
                    response_format="json"
                )
            
            duration = time.time() - start_time
            
            print(f"✅ Groq Transcription complete! ({duration:.2f}s)")
            print(f"   Text: {transcription.text[:100]}...")
            
            return {
                "text": transcription.text.strip(),
                "language": language,
                "duration": duration,
                "success": True
            }
        
        except Exception as e:
            print(f"❌ Groq Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
