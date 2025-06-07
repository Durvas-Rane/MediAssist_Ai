import streamlit as st
from PIL import Image
import time
import backend
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
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
    .chat-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .chat-header img {
        margin-right: 10px;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
        padding: 10px;
        border-top: 1px solid #eee;
    }
    .health-stats {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Queue to collect audio frames
q = queue.Queue()

def audio_callback(frame: av.AudioFrame):
    q.put(frame.to_ndarray().flatten())
    return frame

# Function to process user queries
def process_user_query(query):
    if not query:
        return
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        response_placeholder = st.empty()
        try:
            with st.spinner("MediAssist is thinking..."):
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

# Load logo
try:
    logo = Image.open("logo.png")
except FileNotFoundError:
    logo = None  # Graceful fallback

# Chat header
col1, col2 = st.columns([1, 5])
with col1:
    if logo:
        st.image(logo, width=80)
with col2:
    st.title("MediAssist AI")
    st.markdown("Your personal healthcare assistant powered by Gemini 1.5 Flash")

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

    # Quick suggestions
    if len(st.session_state.messages) <= 1:
        st.markdown("#### Quick Questions")
        suggestion_cols = st.columns(2)
        suggestions = [
            "What are symptoms of the flu?",
            "How can I improve my sleep?",
            "What's a healthy diet?",
            "How often should I exercise?"
        ]
        if "selected_suggestion" not in st.session_state:
            st.session_state.selected_suggestion = None
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.selected_suggestion = suggestion
                    st.rerun()

    # Check for selected suggestion
    if st.session_state.selected_suggestion:
        process_user_query(st.session_state.selected_suggestion)
        st.session_state.selected_suggestion = None

    # Chat input
    prompt = st.chat_input("Ask your health question...")
    if prompt:
        process_user_query(prompt)

    # Voice input using streamlit-webrtc
    st.subheader("🎤 Voice Input (via browser)")
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
        ),
        audio_receiver_size=256,
        audio_frame_callback=audio_callback,
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

# Sidebar with health tools
st.sidebar.title("Health Tools")

# Emergency resources
emergency_expander = st.sidebar.expander("🚨 Emergency Resources")
with emergency_expander:
    st.markdown("""
    ### Emergency Contacts
    - **Emergency Services**: 102 / 1298
    - **Poison Control**: 1-800-222-1222
    - **Helpline Number**: 112 
    - **National Suicide Prevention Lifeline**: 9152987821
    """)

# Health tracking section
health_tracking = st.sidebar.expander("📊 Health Tracker", expanded=True)
with health_tracking:
    st.markdown('<div class="health-stats">', unsafe_allow_html=True)

    if "water_intake" not in st.session_state:
        st.session_state.water_intake = 0
    if "steps" not in st.session_state:
        st.session_state.steps = 0
    if "medications" not in st.session_state:
        st.session_state.medications = []

    # Water intake
    st.subheader("💧 Water Intake")
    water_col1, water_col2 = st.columns([3, 1])
    with water_col1:
        st.session_state.water_intake = st.slider("Glasses", 0, 8, st.session_state.water_intake)
    with water_col2:
        if st.button("+", key="add_water"):
            if st.session_state.water_intake < 8:
                st.session_state.water_intake += 1
                st.rerun()
    water_percent = (st.session_state.water_intake / 8) * 100
    st.progress(water_percent / 100)
    st.caption(f"{st.session_state.water_intake}/8 glasses")

    # Step counter
    st.subheader("👟 Steps Today")
    steps_col1, steps_col2 = st.columns([3, 1])
    with steps_col1:
        st.session_state.steps = st.number_input("Count", min_value=0, value=st.session_state.steps, step=1000)
    steps_percent = min((st.session_state.steps / 10000) * 100, 100)
    st.progress(steps_percent / 100)
    st.caption(f"{st.session_state.steps}/10,000 steps")

    st.markdown('</div>', unsafe_allow_html=True)

# Appointment scheduler
appointment = st.sidebar.expander("🗓️ Schedule Appointment")
with appointment:
    st.date_input("Select Date")
    st.selectbox("Select Time", ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"])
    st.selectbox("Appointment Type", ["General Checkup", "Specialist Consultation", "Follow-up", "Vaccination"])
    if st.button("Request Appointment", use_container_width=True):
        st.success("Appointment request submitted!")

# Medication reminder
medication = st.sidebar.expander("💊 Medication Reminder")
with medication:
    med_name = st.text_input("Medication Name")
    med_time = st.time_input("Reminder Time")
    med_frequency = st.selectbox("Frequency", ["Daily", "Twice Daily", "Every 8 Hours", "Weekly"])
    if st.button("Add Reminder", use_container_width=True):
        if med_name:
            st.session_state.medications.append({"name": med_name, "time": med_time, "frequency": med_frequency})
            st.success(f"Reminder set for {med_name}")
    if st.session_state.medications:
        st.subheader("Current Medications")
        for i, med in enumerate(st.session_state.medications):
            st.markdown(f"**{med['name']}** - {med['time'].strftime('%I:%M %p')} ({med['frequency']})")
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.medications.pop(i)
                st.rerun()

# Clear chat history
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediAssist AI, your healthcare assistant. How can I help you today?"}
    ]
    if "selected_suggestion" in st.session_state:
        st.session_state.selected_suggestion = None
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("MediAssist AI v1.0 | Group 123 | Powered by Gemini 1.5 Flash")
