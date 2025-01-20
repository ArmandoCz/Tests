import whisper

models = ["tiny","base"]

model = whisper.load_model(models[1]).to("cuda")  # Mueve el modelo a la GPU
result = model.transcribe(audio=r"sounds/recorded_audio_3jFqrS1.mp3")
print(result["text"])