import requests
from voice.whisper_stt import WhisperSTT
from voice.sarvam_tts import SarvamTTS


class VoiceProcessor:

    def __init__(self):

        self.stt = WhisperSTT()
        self.tts = SarvamTTS()

        self.api_url = "http://127.0.0.1:8000/query/ask"

    def listen(self):

        audio_file = self.stt.record_audio(duration=5)

        result = self.stt.transcribe(audio_file)

        return result["text"]

    def ask_rag(self, query):

        response = requests.post(
            self.api_url,
            json={"question": query}
        )

        data = response.json()

        return data["answer"]

    def run(self):

        print("\n🎙️ Listening...")

        query = self.listen()

        print("👤 User:", query)

        answer = self.ask_rag(query)

        print("🤖 Assistant:", answer)

        self.tts.speak(answer)