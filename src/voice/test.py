from whisper_stt import WhisperSTT

stt = WhisperSTT()

audio_file = stt.record_audio(duration=5)

result = stt.transcribe(audio_file, language="hi")

print("\nTranscription:")
print(result["text"])