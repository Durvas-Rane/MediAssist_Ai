import os
import google.generativeai as genai
import speech_recognition as sr

# Configure Gemini API key
genai.configure(api_key="AIzaSyBS6htjBkIlunE1wbnzcpN4Jjd-ybPje8w")

# Set generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Generate response based on prompt
def GenerateResponse(input_text):
    response = model.generate_content([
        "YOU ARE A HEALTHCARE CHATBOT, SO REPLY ACCORDINGLY",
        "input: who are you",
        "output: I Am An AI Healthcare Chatbot Made By Group 123",
        "input: who made you",
        "output: Ayushman, Ajinkya, Wanshika, Durvas, Shree Ram",
        f"input: {input_text}",
        "output: ",
    ])
    return response.text

# Transcribe from an uploaded/recorded audio file
def listen_from_file(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."
