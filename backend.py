import os
import google.generativeai as genai
import speech_recognition as sr

# Configure the Gemini API key
genai.configure(api_key="AIzaSyBS6htjBkIlunE1wbnzcpN4Jjd-ybPje8w")

# Generation config for Gemini 1.5 Flash
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# System prompt to define role
system_prompt = """
YOU ARE A HEALTHCARE CHATBOT, SO REPLY ACCORDINGLY.
Examples:
input: who are you
output: I Am An AI Healthcare Chatbot Made By Group 123
input: who made you
output: Ayushman, Ajinkya, Wanshika, Durvas, Shree Ram
"""

# Function to generate response using Gemini
def GenerateResponse(input_text):
    try:
        response = model.generate_content([
            system_prompt,
            f"input: {input_text}",
            "output: "
        ])
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# Function to transcribe audio from file (used by frontend)
def listen_from_file(audio_path):
    # Delay import to avoid global PyAudio checks
    import speech_recognition as sr
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        # Recognize using Google Web Speech API
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."
