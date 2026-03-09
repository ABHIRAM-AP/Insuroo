import base64

import requests
import tempfile
import os
import pygame
from dotenv import load_dotenv

load_dotenv()

class SarvamTTS:

    def __init__(self):
        self.api_key = os.getenv("SARVAM_AI_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_AI_API_KEY not found in environment")
        self.url = "https://api.sarvam.ai/text-to-speech"
        pygame.mixer.init()

    def synthesize(self, text):

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": "karun"
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()

        # Extract base64 audio
        audio_base64 = data["audios"][0]

        audio_bytes = base64.b64decode(audio_base64)

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        with open(temp_audio.name, "wb") as f:
            f.write(audio_bytes)

        return temp_audio.name


    def speak(self, text):

        # Split text into chunks of 450 characters
        chunks = [text[i:i+450] for i in range(0, len(text), 450)]

        for chunk in chunks:

            audio_file = self.synthesize(chunk)

            print("Speaking response...")

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                continue

            os.remove(audio_file)