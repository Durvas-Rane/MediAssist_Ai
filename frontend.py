import streamlit as st
from PIL import Image
import time
import backend
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue
import numpy as np
import tempfile
from scipy.io.wavfile import write

# Set page configuration
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1000px; margin: 0 auto; }
    .chat-container {
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #4285F4;
        color: white;
        border-radius: 20px;
        padding: 8px 16px;
        border: none;
    }
    .stButton button:hover { background-color: #3367d6; }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
        padding: 10px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Load logo
logo = Image.open("logo.png")

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=80)
with col2:
    st.title("MediAssist AI")
    st.markdown("Your personal healthcare assistant powered by Gemini 1.5 Flash")

# Define a queue for audio frames
q = queue.Queue()

def audio_callback(frame: av.AudioFrame):
    q.put(frame.to_ndarray().flatten())
    return av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")

# Handle text and voice input

def process_user_query(query):
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        response_placeholder = st.empty()
        with st.spinner("MediAssist is thinking..."):
            try:
                full_response = backend.GenerateResponse(query)
                displayed = ""
                for word in full_response.split():
                    displayed += word + " "
                    response_placeholder.markdown(displayed + "▌")
                    time.sleep(0.02)
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {e}"
                response_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Tabs
chat_tab, about_tab, team_tab = st.tabs(["Chat", "About", "Team"])

with chat_tab:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm MediAssist AI, your healthcare assistant. How can I help you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask your health question...")
    if prompt:
        process_user_query(prompt)

    # Voice input using streamlit-webrtc
    st.subheader("🎤 Voice Input (via browser)")
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        in_audio=True,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_callback
    )

    if ctx and ctx.state.playing:
        st.info("Recording... Speak now!")
        if st.button("Stop & Transcribe"):
            audio_data = []
            while not q.empty():
                audio_data.append(q.get())
            if audio_data:
                audio_np = np.concatenate(audio_data)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    write(f.name, 16000, audio_np.astype(np.int16))
                    voice_prompt = backend.listen_from_file(f.name)
                    st.success(f"You said: {voice_prompt}")
                    if not voice_prompt.startswith("Sorry"):
                        process_user_query(voice_prompt)
            else:
                st.warning("No audio recorded.")

    st.markdown('<div class="disclaimer">DISCLAIMER: This AI assistant provides general information only and is not a substitute for professional medical advice. Always consult a healthcare provider.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with about_tab:
    st.header("About MediAssist AI")
    st.markdown("""
    MediAssist AI is an AI healthcare assistant to provide general health info.

    **What it can do:**
    - Answer general wellness questions
    - Explain medical terms
    - Give healthy lifestyle tips

    **Limitations:**
    - Cannot diagnose
    - Cannot access personal data
    - Not for emergencies
    """)

with team_tab:
    st.header("Our Team")
    st.markdown("""
    - Ayushman: AI Integration
    - Ajinkya: Backend Dev
    - Wanshika: UI/UX
    - Durvas: Speech Recognition
    - Shree Ram: DB Management
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("MediAssist AI v1.0 | Group 123 | Powered by Gemini 1.5 Flash")
