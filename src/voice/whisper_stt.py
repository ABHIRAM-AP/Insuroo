import time
from typing import Dict
import os
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

class WhisperSTT:
    """
    Speech-to-Text using OpenAI Whisper
    """
    
    def __init__(self, model_name="base"):
        """
        Initialize Whisper STT
        
        Args:
            model_name: Model size (tiny, base, small, medium, large)
        """
        print(f"🔄 Initializing Whisper STT with model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.model_name = model_name
        print(f"✅ Whisper STT ready!")


    def record_audio(self, filename="live_audio.wav", duration=5, sample_rate=16000):

        print(f"\n🎙️ Recording for {duration} seconds...")

        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )

        sd.wait()

        write(filename, sample_rate, recording)

        full_path = os.path.abspath(filename)

        print(f"✅ Audio saved to {full_path}")

        return full_path
    
    def transcribe(self, audio_file_path: str, language: str = "hi") -> Dict:
        """
        Transcribe audio to text
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (hi, en, ta, te, etc.)
        
        Returns:
            dict: {
                "text": str,
                "language": str,
                "segments": list,
                "duration": float
            }
        """
        print(f"\n🎤 Transcribing audio...")
        print(f"   File: {audio_file_path}")
        print(f"   Language: {language}")
        
        start_time = time.time()
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio=audio_file_path,
                language=language,
                fp16=False,  # Use CPU (not GPU)
                verbose=False
            )
            
            duration = time.time() - start_time
            
            print(f"✅ Transcription complete! ({duration:.2f}s)")
            print(f"   Detected language: {result['language']}")
            print(f"   Text: {result['text'][:100]}...")
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result.get("segments", []),
                "duration": duration,
                "success": True
            }
        
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    def transcribe_auto_detect(self, audio_file_path: str) -> Dict:
        """
        Transcribe with automatic language detection
        """
        print(f"\n🔍 Auto-detecting language...")
        
        try:
            result = self.model.transcribe(
                audio=audio_file_path,
                fp16=False,
                verbose=False
            )
            
            print(f"✅ Detected language: {result['language']}")
            
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "success": True
            }
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }

