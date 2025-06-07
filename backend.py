import os
import google.generativeai as genai
import speech_recognition as sr

# Configure the API key
genai.configure(api_key="AIzaSyBS6htjBkIlunE1wbnzcpN4Jjd-ybPje8w")

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Function to generate response
def GenerateResponse(input_text):
    response = model.generate_content([
        "YOU ARE A HEALTHCARE CHATBOT, SO REPLY ACCORDINGLY",
        "input: who are you",
        "output: I Am An AI Healthcare Chatbot Made By Group 123",
        "input: who are you?",
        "output: I Am An AI Healthcare Chatbot Made By Group 123",
        "input: who made you",
        "output: Ayushman, Ajinkya, Wanshika, Durvas, Shree Ram",
        f"input: {input_text}",
        "output: ",
    ])
    return response.text

# Function to capture audio input and convert to text
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak your question.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."



# while True:
#     string= str(input("Enter Your Prompt: "))
#     print("Bot: ", GenerateResponse(string))
