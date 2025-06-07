import streamlit as st
from PIL import Image
import time
import backend

# Set page configuration
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for better healthcare look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
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
    .stButton button:hover {
        background-color: #3367d6;
    }
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
    .st-emotion-cache-16txtl3 {
        padding: 15px;
        border-radius: 15px;
    }
    .health-stats {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Function to process all questions - whether from chat input, voice, or predefined
def process_user_query(query):
    if not query:
        return
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        response_placeholder = st.empty()
        
        try:
            # Get response from backend
            with st.spinner("MediAssist is thinking..."):
                full_response = backend.GenerateResponse(query)
                
                # Simple typing animation
                words = full_response.split()
                displayed_response = ""
                
                for i in range(len(words)):
                    displayed_response += words[i] + " "
                    response_placeholder.markdown(displayed_response + "▌")
                    time.sleep(0.02)
                
                # Final response without cursor
                response_placeholder.markdown(full_response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}. Please try again."
            response_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# Load local image
logo = Image.open(r"D:\AI Healthcare Chatbot GP 123 - Copy\AI Healthcare Chatbot GP 123 - Copy\AI Healthcare Chatbot GP 123\logo.png")


# Chat header with logo and team info
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=80)  # Use the loaded logo instead of a URL
with col2:
    st.title("MediAssist AI")
    st.markdown("Your personal healthcare assistant powered by Gemini 1.5 Flash")

# Information tabs
tab1, tab2, tab3 = st.tabs(["Chat", "About", "Team"])

with tab1:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm MediAssist AI, your healthcare assistant. I can answer general health questions, help you understand medical terminology, and provide wellness advice. How can I help you today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # Common health questions suggestions
    if len(st.session_state.messages) <= 1:
        st.markdown("#### Quick Questions")
        suggestion_cols = st.columns(2)
        
        suggestions = [
            "What are symptoms of the flu?",
            "How can I improve my sleep?",
            "What's a healthy diet?",
            "How often should I exercise?"
        ]
        
        # NEW: Use session state to track if a suggestion was selected
        if "selected_suggestion" not in st.session_state:
            st.session_state.selected_suggestion = None
        
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Store the selected suggestion
                    st.session_state.selected_suggestion = suggestion
                    st.rerun()
    
    # Check if a suggestion was selected and process it
    if hasattr(st.session_state, "selected_suggestion") and st.session_state.selected_suggestion:
        process_user_query(st.session_state.selected_suggestion)
        # Reset the selected suggestion
        st.session_state.selected_suggestion = None

    # Accept user input
    input_col1, input_col2 = st.columns([5, 1])

    with input_col1:
        prompt = st.chat_input("Ask your health question...")

    with input_col2:
        if st.button("🎤", key="voice_button"):
            with st.spinner("Listening..."):
                voice_prompt = backend.listen()
                if not voice_prompt.startswith("Sorry"):
                    st.success(f"You said: {voice_prompt}", icon="🎤")
                    # Store the voice input as the prompt
                    prompt = voice_prompt
                else:
                    st.error(voice_prompt, icon="🎤")

    # Process the chat input if we have one
    if prompt:
        process_user_query(prompt)
    
    # Medical disclaimer
    st.markdown('<div class="disclaimer">DISCLAIMER: This AI assistant provides general information only and is not a substitute for professional medical advice. Always consult a healthcare provider for medical concerns.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("About MediAssist AI")
    st.markdown("""
    MediAssist AI is an artificial intelligence healthcare assistant designed to provide general health information and guidance. Our mission is to make healthcare knowledge more accessible to everyone.
    
    ### Technology Behind MediAssist
    This chatbot is powered by Google's Gemini 1.5 Flash model, a state-of-the-art large language model specialized for healthcare conversations.
    
    ### What MediAssist Can Do:
    - Answer general health and wellness questions
    - Explain medical terminology in simple language
    - Provide information about common symptoms and conditions
    - Offer guidance on healthy lifestyle choices
    
    ### What MediAssist Cannot Do:
    - Provide diagnosis or replace medical professionals
    - Prescribe medications or treatments
    - Access or store your personal health records
    - Handle medical emergencies (please call emergency services)
    
    This tool was developed by Group 123 as part of a healthcare innovation project using Google's generative AI technology.
    """)
    
    # Add backend test functionality
    st.subheader("System Status")
    if st.button("Test Backend Connection"):
        try:
            test_response = backend.GenerateResponse("Hello")
            st.success(f"✅ Backend connection successful!")
            st.info(f"Sample response: {test_response[:100]}...")
        except Exception as e:
            st.error(f"❌ Backend connection failed: {str(e)}")

with tab3:
    st.header("Our Team")
    st.markdown("""
    MediAssist AI was developed by Group 123:
    
    - **Ayushman** - AI Model Integration
    - **Ajinkya** - Backend Development
    - **Wanshika** - UI/UX Design
    - **Durvas** - Speech Recognition
    - **Shree Ram** - Database Management
    
    We're passionate about making healthcare information more accessible through cutting-edge AI technology.
    """)
    
    st.subheader("Technical Implementation")
    st.markdown("""
    - **Frontend**: Streamlit
    - **AI Model**: Google Gemini 1.5 Flash
    - **Speech Recognition**: Google Speech Recognition API
    - **Language**: Python
    """)

# Sidebar with additional features
st.sidebar.title("Health Tools")

# Emergency resources section
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
    
    # Initialize trackers if not exist
    if "water_intake" not in st.session_state:
        st.session_state.water_intake = 0
    if "steps" not in st.session_state:
        st.session_state.steps = 0
    if "medications" not in st.session_state:
        st.session_state.medications = []

    # Water intake tracker
    st.subheader("💧 Water Intake")
    water_col1, water_col2 = st.columns([3, 1])
    with water_col1:
        st.session_state.water_intake = st.slider("Glasses", 0, 8, st.session_state.water_intake)
    with water_col2:
        if st.button("+", key="add_water"):
            if st.session_state.water_intake < 8:
                st.session_state.water_intake += 1
                st.rerun()
    
    # Show water progress
    water_percent = (st.session_state.water_intake / 8) * 100
    st.progress(water_percent/100)
    st.caption(f"{st.session_state.water_intake}/8 glasses")

    # Step counter
    st.subheader("👟 Steps Today")
    steps_col1, steps_col2 = st.columns([3, 1])
    with steps_col1:
        st.session_state.steps = st.number_input("Count", min_value=0, value=st.session_state.steps, step=1000)
    
    # Show steps progress
    steps_percent = min((st.session_state.steps / 10000) * 100, 100)
    st.progress(steps_percent/100)
    st.caption(f"{st.session_state.steps}/10,000 steps")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Appointment scheduler
appointment = st.sidebar.expander("🗓️ Schedule Appointment")
with appointment:
    st.date_input("Select Date")
    st.selectbox("Select Time", ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"])
    st.selectbox("Appointment Type", ["General Checkup", "Specialist Consultation", "Follow-up", "Vaccination"])

    if st.button("Request Appointment", use_container_width=True):
        st.success("Appointment request submitted! A healthcare provider will contact you to confirm.")

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
    
    if hasattr(st.session_state, "medications") and st.session_state.medications:
        st.subheader("Current Medications")
        for i, med in enumerate(st.session_state.medications):
            st.markdown(f"**{med['name']}** - {med['time'].strftime('%I:%M %p')} ({med['frequency']})")
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.medications.pop(i)
                st.rerun()

# Clear chat history option
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediAssist AI, your healthcare assistant. I can answer general health questions, help you understand medical terminology, and provide wellness advice. How can I help you today?"}
    ]
    # Also clear any selected suggestion
    if hasattr(st.session_state, "selected_suggestion"):
        st.session_state.selected_suggestion = None
    st.rerun()

# Footer with version info
st.sidebar.markdown("---")
st.sidebar.caption("MediAssist AI v1.0 | Group 123")
st.sidebar.caption("Powered by Google Gemini 1.5 Flash")