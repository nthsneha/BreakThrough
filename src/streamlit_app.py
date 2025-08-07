
# streamlit_app.py
import streamlit as st
import torch
import streamlit.components.v1 as components
import tempfile
import os
import json
import re
import requests
import base64
from audio_recorder_streamlit import audio_recorder
import pandas as pd
import numpy as np
import time
from gtts import gTTS
from PIL import Image
import io
import logging
import sys
from pathlib import Path
import PyPDF2
from textblob import TextBlob
import mlflow
import ollama
import parselmouth
from parselmouth.praat import call
import math
import traceback
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import threading
import shutil
from transformers import pipeline

# ===============================================================
# Backend Logging & MLflow Configuration
# ===============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [SpeechApp] - %(message)s',
    stream=sys.stdout
)
MLFLOW_EXPERIMENT_NAME = "AI_Speech_Coach_Sessions"
try:
    if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
except Exception as e:
    logging.error(f"Could not configure MLflow: {e}")


# ===============================================================
# Prompts for Different AI Coach Modes
# ===============================================================
PROMPTS = {
    "JAM": """You are a strict but fair Just-A-Minute (JAM) instructor. Your primary role is to help me practice speaking.
**Your Core Task & Rules:**
1. Give a Topic: Your primary job is to give me a topic to speak on. The topic must be included in your "response".
2. Analyze Mistakes: Listen to my response and identify any mistakes like hesitation, repetition, or deviation from the topic.
3. Genre Variety: Do not give a topic from the same genre twice in a row.
4. Handle Commands: If I ask for a new topic (e.g., "give me another one"), simply provide a new topic in the JSON response without extra conversation.
**CRITICAL OUTPUT FORMAT:**
Your entire response MUST be a valid JSON object. Do not add any text before or after the JSON.
The JSON must have these exact keys:
- "response": (string) Your conversational reply. This string MUST contain the topic.
- "mistakes": (array of objects) A list of my mistakes. Must be [] if there are no mistakes.
- "suggestions": (array of strings) Actionable tips for improvement. Can be [].""",

    "Interview": """You are an expert AI interviewer named 'Gemini Recruit'. Your goal is to conduct a comprehensive interview for a role based on the professional details and resume I provide.
**Phase 1: Onboarding**
1. Greet me professionally.
2. Confirm that you have received my resume and ask for the specific job role I am applying for.
3. Do not begin the interview until I provide the job role.
**Phase 2: The Interview**
1. Once I provide the role, begin the interview by asking your first question based on my resume and the role.
2. Ask only one question at a time and wait for my response.
3. Ask a mix of Technical Questions and Behavioral (HR) Questions.
4. Autonomously conclude the interview after asking 5-7 relevant questions.
5. If I say "That's the end of the interview," proceed directly to Phase 3.
**Phase 3: Final Evaluation (JSON Output)**
1. When the interview is concluded, your FINAL response MUST be a single, valid JSON object and nothing else.
2. The JSON object must have these exact keys: "overall_summary", "strengths", "areas_for_improvement", and "final_recommendation".""",

    "Business Talks": """You are an AI-powered business communication coach for realistic role-playing scenarios.
**Phase 1: Session Setup**
1. Choose Your Role: Randomly select a role for yourself (Stakeholder, Client, or Colleague).
2. Assign My Role: Assign a corresponding, logical role to me.
3. Create a detailed business scenario for our conversation and start with an opening statement.
**Phase 2: The Business Conversation**
1. Consistently maintain your chosen persona.
2. Automatically end the session after 8-12 conversational turns once you have enough content for evaluation.
3. If I say, "Okay, let's end the meeting here and debrief," proceed directly to Phase 3.
**Phase 3: Communication Feedback (JSON Output)**
1. When the conversation concludes, your FINAL response MUST be a single, valid JSON object.
2. The JSON must have these exact keys: "overall_feedback", "communication_strengths", "areas_for_improvement", and "actionable_suggestions".""",

    "Debate": """You are an advanced AI for debate and analysis.
**Phase 1: Debate Setup**
1. Greet me and present a single, clear, debatable topic.
2. Ask me to choose my stance ("For" or "Against").
3. You MUST take the opposite stance.
4. State the rules: The debate will last for 6 rounds (1 opening statement and 5 rebuttals each). You will begin with your opening statement.
**Phase 2: The Debate (6 Rounds)**
1. Argue your assigned position passionately and logically for all 6 rounds.
**Phase 3: The Judgment (JSON Output)**
1. After the 6th round, shift to an unbiased judge persona.
2. Your FINAL response MUST be a single JSON object analyzing both my performance and your own, declaring a winner.
3. The JSON must have these keys: "winner", "verdict_summary", "user_performance_analysis", "ai_performance_analysis", and "key_moment".""",

    "Therapy": """Adopt the persona of a therapeutic companion. Your identity is this companion. Your personality is grounded, calm, and present.
**Part 1: Critical Safety and Ethical Guardrails**
1. Your very first message must be a warm greeting that seamlessly integrates the disclaimer that you are an AI companion and not a substitute for a qualified human therapist.
2. If I express thoughts of self-harm or suicide, you must immediately pause to provide helpline resources.
3. You are forbidden from giving diagnoses or treatment plans.
**Part 2: Core Conversational Approach**
1. Practice deep listening, lead with genuine empathy, and use open-ended inquiry. Empower, don't advise.
**Part 3: Session Flow and Concluding Reflection**
1. I will determine when the conversation ends by saying something like, "Thanks, that's all for today."
2. When I end the session, your FINAL response must be ONLY a single valid JSON object with the key "session_summary", containing a warm, non-clinical summary of our conversation.""",

    "Socialising": """You are an AI Social Confidence Coach. Your role is to be a friendly, judgment-free practice partner.
**Phase 1: Setting Up the Practice Session**
1. Propose a common, low-stakes social scenario. Importantly, create a NEW and UNIQUE scenario each time you start Phase 1.
2. Confirm if the scenario is okay with me before starting.
3. Start the conversation with a friendly opening line.
**Phase 2: The Conversation**
1. Act out your chosen persona naturally. Be patient and encouraging.
**Phase 3: Constructive and Motivating Feedback (JSON Output)**
1. When I end the conversation, your FINAL response MUST be a single, valid JSON object.
2. The JSON must provide supportive feedback with these keys: "positive_summary", "moments_to_celebrate", "gentle_suggestions_for_growth", and "motivational_takeaway".""",

    "Reading": """You are an advanced Vocal Coach and Elocution Analyst.
**Your Core Task:**
1. Give me a unique paragraph to read aloud (100-150 words) from a diverse topic area.
2. After I read, you will receive my speech transcription and performance metrics.
3. Compare my transcription to the original text to find inaccuracies and analyze the metrics for delivery.
4. If my input is a command like "new passage", your response should ONLY contain the new passage within the "response" key of the JSON, with other arrays empty.
**CRITICAL OUTPUT FORMAT:**
Your entire response MUST be a valid JSON object with these exact keys:
- "response": (string) Your conversational reply or new passage.
- "accuracy_analysis": (array of objects) Highlighting differences between original text and my reading.
- "delivery_feedback": (array of strings) Comments on my vocal delivery, pacing, and expressiveness."""
}

# ===============================================================
# Page Configuration & UI Styling
# ===============================================================
st.set_page_config(page_title="BREAKTHROUGH", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Main theme colors: Black, Red, Pink */
    :root {
        --primary-bg: #0a0a0a;
        --secondary-bg: #1a1a1a;
        --card-bg: #222222;
        --accent-red: #dc2626;
        --accent-pink: #ec4899;
        --accent-pink-light: #f472b6;
        --text-primary: #ffffff;
        --text-secondary: #e5e5e5;
        --text-muted: #a3a3a3;
        --border-color: #404040;
    }
    
    /* Base styles */
    html, body, [class*="st-"] { 
        font-family: 'Inter', sans-serif; 
        background-color: var(--primary-bg) !important;
        color: var(--text-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        font-family: 'Poppins', sans-serif; 
        font-weight: 600; 
        color: var(--text-primary) !important;
    }
    
    .stApp { 
        background-color: var(--primary-bg) !important; 
        color: var(--text-primary) !important; 
    }
    
    /* Header styling with gradient */
    .main-header { 
        font-family: 'Poppins', sans-serif; 
        font-size: 2.8rem; 
        font-weight: 700; 
        text-align: center; 
        margin-bottom: 1.5rem; 
        background: linear-gradient(90deg, var(--accent-red), var(--accent-pink), var(--accent-pink-light)); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        padding-top: 1.5rem; 
    }
    
    /* Cards and containers */
    .card { 
        background-color: var(--card-bg) !important; 
        border: 1px solid var(--border-color) !important; 
        border-radius: 12px; 
        padding: 25px; 
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.1); 
        margin-bottom: 20px; 
    }
    
    /* Chat styling */
    .chat-container { 
        height: 65vh; 
        overflow-y: auto; 
        padding: 20px; 
        margin-bottom: 20px; 
        scroll-behavior: smooth; 
        display: flex; 
        flex-direction: column;
        background-color: var(--secondary-bg);
        border-radius: 12px;
    }
    
    .chat-messages { 
        flex-grow: 1; 
        display: flex; 
        flex-direction: column; 
        justify-content: flex-start; 
        width: 100%; 
    }
    
    .chat-bubble-container { 
        display: flex; 
        margin-bottom: 16px; 
        width: 100%; 
    }
    
    .chat-bubble { 
        max-width: 85%; 
        padding: 12px 18px; 
        border-radius: 18px; 
        word-wrap: break-word; 
        overflow-wrap: break-word; 
        hyphens: auto; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.3); 
    }
    
    .user-container { justify-content: flex-end; }
    .assistant-container { justify-content: flex-start; }
    
    .user-bubble { 
        background: linear-gradient(135deg, var(--accent-red), var(--accent-pink));
        color: white; 
    }
    
    .assistant-bubble { 
        background-color: var(--card-bg); 
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    /* Avatar container */
    .avatar-container { 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
        margin-top: 40px; 
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid var(--border-color);
    }
    
    .mic-container { margin-top: 30px; }
    
    /* Button styling */
    .stButton>button { 
        border-color: var(--accent-pink) !important; 
        color: var(--accent-pink) !important; 
        background-color: transparent !important;
        font-family: 'Inter', sans-serif; 
        font-weight: 500; 
        border-radius: 8px; 
        transition: all 0.3s ease; 
        border-width: 2px !important;
    }
    
    .stButton>button:hover { 
        background-color: var(--accent-pink) !important; 
        border-color: var(--accent-pink) !important; 
        color: var(--primary-bg) !important; 
        transform: translateY(-2px); 
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3); 
    }
    
    .stButton button[kind="primary"] { 
        background: linear-gradient(135deg, var(--accent-red), var(--accent-pink)) !important; 
        border: none !important;
        color: white !important; 
        font-weight: 600; 
    }
    
    .stButton button[kind="primary"]:hover { 
        background: linear-gradient(135deg, #b91c1c, #db2777) !important; 
        transform: translateY(-2px); 
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.4); 
    }
    
    /* Selectbox and input styling */
    .stSelectbox>div>div { 
        background-color: var(--card-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .stTextInput>div>div>input {
        background-color: var(--card-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--card-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--secondary-bg) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--primary-bg); }
    ::-webkit-scrollbar-thumb { 
        background: var(--accent-pink); 
        border-radius: 10px; 
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-pink-light); }
    
    /* Audio styling */
    audio { 
        width: 100%; 
        margin-top: 10px;
        filter: sepia(100%) saturate(200%) hue-rotate(320deg);
    }
    
    .audio-recorder { margin-top: 20px; }
    
    /* Info and warning boxes */
    .stInfo {
        background-color: var(--card-bg) !important;
        border-left: 4px solid var(--accent-pink) !important;
        color: var(--text-primary) !important;
    }
    
    .stWarning {
        background-color: var(--card-bg) !important;
        border-left: 4px solid var(--accent-red) !important;
        color: var(--text-primary) !important;
    }
    
    .stSuccess {
        background-color: var(--card-bg) !important;
        border-left: 4px solid #10b981 !important;
        color: var(--text-primary) !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Welcome container */
    .welcome-container {
        text-align: center;
        background: linear-gradient(135deg, var(--card-bg), var(--secondary-bg)) !important;
        border: 2px solid var(--accent-pink) !important;
        box-shadow: 0 8px 32px rgba(236, 72, 153, 0.2);
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--accent-pink) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
const autoplayAudio = () => {
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach(audio => {
        if (!audio.hasAttribute('data-autoplay-handled')) {
            audio.setAttribute('autoplay', '');
            audio.setAttribute('data-autoplay-handled', 'true');
        }
    });
};
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.addedNodes && mutation.addedNodes.length > 0) {
            autoplayAudio();
        }
    });
});
observer.observe(document.body, { childList: true, subtree: true });
document.addEventListener('DOMContentLoaded', autoplayAudio);
</script>
""", unsafe_allow_html=True)

# ===============================================================
# Core Application Classes & Functions
# ===============================================================

class AutoSnapshotVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshots = []
        self.temp_dir = None
        self.auto_snapshot_enabled = True
        self.last_snapshot_time = 0
        self.snapshot_interval = 10  # seconds
        self._manual_snapshot_trigger = False

    def set_temp_dir(self, temp_dir):
        with self.lock:
            self.temp_dir = temp_dir
            self.last_snapshot_time = time.time() # Start timer once dir is set

    def toggle_auto_snapshot(self, enabled):
        with self.lock:
            self.auto_snapshot_enabled = enabled
            if enabled:
                self.last_snapshot_time = time.time()

    def take_manual_snapshot(self):
        with self.lock:
            self._manual_snapshot_trigger = True

    def get_snapshots(self):
        with self.lock:
            return self.snapshots.copy()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        should_snapshot = False

        with self.lock:
            if self.temp_dir:
                if self._manual_snapshot_trigger:
                    should_snapshot = True
                    self._manual_snapshot_trigger = False
                elif self.auto_snapshot_enabled and (current_time - self.last_snapshot_time) >= self.snapshot_interval:
                    should_snapshot = True

                if should_snapshot:
                    self.last_snapshot_time = current_time
                    img_path = os.path.join(self.temp_dir, f"snap_{int(current_time)}.jpg")
                    try:
                        cv2.imwrite(img_path, img)
                        self.snapshots.append(img_path)
                        logging.info(f"Snapshot saved to {img_path}")
                    except Exception as e:
                        logging.error(f"Failed to write snapshot: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class OllamaPipeline:
    def __init__(self, model_name='gemma3n:e2b'):
        self.model = model_name
        try:
            ollama.list()
            logging.info("Successfully connected to Ollama server.")
        except Exception as e:
            st.error("Could not connect to Ollama server. Please ensure it is running.")
            logging.error(f"Ollama connection failed: {e}")
            st.stop()

    def generate_response(self, history, user_input, system_prompt, images=None):
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            final_user_message = {'role': 'user', 'content': user_input}
            
            if images:
                image_b64_list = []
                for img_path in images:
                    if os.path.exists(img_path):
                        try:
                            with open(img_path, "rb") as f:
                                image_b64 = base64.b64encode(f.read()).decode('utf-8')
                                image_b64_list.append(image_b64)
                            logging.info(f"Successfully encoded image: {img_path}")
                        except Exception as e:
                            logging.error(f"Failed to encode image {img_path}: {e}")
                
                if image_b64_list:
                    final_user_message['images'] = image_b64_list
                    logging.info(f"Added {len(image_b64_list)} images to message")
                else:
                    logging.warning("No images were successfully encoded")
            
            messages.append(final_user_message)

            response_stream = ollama.chat(model=self.model, messages=messages, stream=True)
            full_response = "".join(chunk['message']['content'] for chunk in response_stream)
            
            processed_response = self.process_model_output(full_response)
            return processed_response
        except Exception as e:
            logging.error(f"Error in generate_response: {e}", exc_info=True)
            return f"A critical error occurred: {e}"

    def process_model_output(self, model_output):
        clean_output = model_output.strip()
        json_match = re.search(r'\{.*\}', clean_output, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                # Extract just the response content
                if "response" in parsed_json:
                    return parsed_json["response"]
                return clean_output
            except json.JSONDecodeError:
                return clean_output
        return clean_output

class SpeechTools:
    def __init__(self):
        try:
            import whisper
            self.whisper_model = whisper.load_model("medium")
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            st.error(f"Could not load Whisper model: {e}")
            st.stop()
        
        # Initialize sentiment analysis model
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logging.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load sentiment model, falling back to TextBlob: {e}")
            self.sentiment_analyzer = None

    def transcribe_audio(self, audio_path):
        try:
            result = self.whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
            return result["text"].strip()
        except Exception as e:
            logging.error(f"Transcription failed: {e}", exc_info=True)
            return ""

    def analyze_speech_details(self, audio_path, transcript):
        try:
            sound = parselmouth.Sound(audio_path)
            duration = sound.get_total_duration()
            pitch = sound.to_pitch()
            mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
            intensity = sound.to_intensity()
            mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
            word_count = len(transcript.split())
            wpm = (word_count / duration) * 60 if duration > 0 else 0
            
            # Enhanced sentiment analysis
            sentiment_polarity = 0
            sentiment_confidence = 0
            sentiment_label = "neutral"
            
            if self.sentiment_analyzer and transcript.strip():
                try:
                    # Use Hugging Face RoBERTa model
                    results = self.sentiment_analyzer(transcript)
                    
                    # Find the result with highest confidence
                    best_result = max(results, key=lambda x: x['score'])
                    sentiment_label = best_result['label'].lower()
                    sentiment_confidence = best_result['score']
                    
                    # Convert labels to polarity scores (-1 to 1)
                    if sentiment_label == 'negative':
                        sentiment_polarity = -sentiment_confidence
                    elif sentiment_label == 'positive':
                        sentiment_polarity = sentiment_confidence
                    else:  # neutral
                        sentiment_polarity = 0
                        
                except Exception as e:
                    logging.warning(f"RoBERTa sentiment analysis failed, using TextBlob: {e}")
                    # Fallback to TextBlob
                    blob = TextBlob(transcript)
                    sentiment_polarity = blob.sentiment.polarity
                    sentiment_confidence = abs(sentiment_polarity)
                    sentiment_label = "positive" if sentiment_polarity > 0.1 else "negative" if sentiment_polarity < -0.1 else "neutral"
            else:
                # Fallback to TextBlob
                blob = TextBlob(transcript)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_confidence = abs(sentiment_polarity)
                sentiment_label = "positive" if sentiment_polarity > 0.1 else "negative" if sentiment_polarity < -0.1 else "neutral"
            
            return {
                'speaking_time': round(duration, 2),
                'avg_pitch': round(mean_pitch, 2) if not math.isnan(mean_pitch) else 0,
                'avg_volume': round(mean_intensity, 2) if not math.isnan(mean_intensity) else 0,
                'words_per_minute': round(wpm, 2),
                'sentiment_polarity': round(sentiment_polarity, 3),
                'sentiment_confidence': round(sentiment_confidence, 3),
                'sentiment_label': sentiment_label,
            }
        except Exception as e:
            logging.error(f"Detailed analysis failed: {e}", exc_info=True)
            return {'error': str(e)}

    def synthesize_speech(self, text):
        if not isinstance(text, str) or not text.strip():
            return None
            
        clean_text = text
        
        # Handle JSON responses
        if text.strip().startswith('{'):
            try:
                parsed = json.loads(text)
                for key in ['response', 'overall_summary', 'overall_feedback', 'verdict_summary', 'positive_summary', 'session_summary']:
                    if key in parsed and isinstance(parsed[key], str) and len(parsed[key].strip()) > 0:
                        clean_text = parsed[key]
                        break
            except (json.JSONDecodeError, TypeError):
                pass
                
        # Aggressive text cleaning for TTS compatibility
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # **bold** -> bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # *italic* -> italic
        clean_text = re.sub(r'#{1,6}\s*', '', clean_text)          # Remove headers
        clean_text = re.sub(r'[`\[\]{}]', '', clean_text)          # Remove code/brackets
        clean_text = re.sub(r'^\d+\.\s*', '', clean_text, flags=re.MULTILINE)  # Remove numbered lists
        clean_text = re.sub(r'^\-\s*', '', clean_text, flags=re.MULTILINE)     # Remove bullet points
        
        # Clean up whitespace and newlines
        clean_text = re.sub(r'\n+', ' ', clean_text)               # Replace newlines with spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)               # Normalize multiple spaces
        clean_text = clean_text.strip()
        
        # Remove phrases that don't work well with TTS
        clean_text = re.sub(r'Phase \d+:', '', clean_text)         # Remove "Phase 1:", etc.
        clean_text = re.sub(r'Step \d+:', '', clean_text)          # Remove "Step 1:", etc.
        clean_text = re.sub(r'\b(Role|Scenario):\s*', '', clean_text)  # Remove "Role:", "Scenario:"
        
        # Don't be too aggressive with minimum length
        if not clean_text or len(clean_text.strip()) < 5:
            return None
        
        # Limit text length for TTS (gTTS can have issues with very long text)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        try:
            # Use slower speed for better reliability with complex text
            tts = gTTS(text=clean_text, lang='en', slow=False)
            
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)  # Reset pointer to beginning
            
            audio_bytes = fp.getvalue()
            
            # More lenient validation - gTTS can produce small files for short text
            if len(audio_bytes) < 100:  # Very small threshold
                return None
                
            return audio_bytes
            
        except Exception as e:
            logging.error(f"gTTS synthesis failed: {e}")
            
            # Try with even more simplified text as fallback
            try:
                simple_text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', clean_text)[:100]
                if simple_text.strip():
                    fallback_tts = gTTS(text=simple_text, lang='en', slow=False)
                    fallback_fp = io.BytesIO()
                    fallback_tts.write_to_fp(fallback_fp)
                    fallback_fp.seek(0)
                    fallback_bytes = fallback_fp.getvalue()
                    return fallback_bytes
            except Exception as fallback_error:
                logging.error(f"Fallback TTS also failed: {fallback_error}")
            
            return None

def start_session():
    st.session_state.app_mode = st.session_state.app_mode_selection
    st.session_state.session_started = True
    st.session_state.temp_dir = tempfile.mkdtemp()
    logging.info(f"Created temp directory: {st.session_state.temp_dir}")
    st.session_state.history = []
    st.session_state.turn_metrics = []
    st.session_state.video_processor_initialized = False

    parent_run = mlflow.start_run(run_name=f"Session_{st.session_state.app_mode}_{int(time.time())}")
    st.session_state.mlflow_parent_run_id = parent_run.info.run_id
    mlflow.log_param("app_mode_initial", st.session_state.app_mode)
    mlflow.log_text(PROMPTS.get(st.session_state.app_mode, ""), "system_prompt.txt")

def process_final_analysis(raw_response):
    """Process final analysis response and extract readable content from JSON if needed"""
    if not raw_response:
        return "No analysis available."
    
    # Check if response is JSON
    clean_response = raw_response.strip()
    if clean_response.startswith('{'):
        try:
            parsed = json.loads(clean_response)
            
            # Try to extract meaningful content based on coaching mode
            content_parts = []
            
            # Common final analysis keys across different modes
            for key in ['overall_summary', 'final_recommendation', 'overall_feedback', 
                       'verdict_summary', 'session_summary', 'positive_summary']:
                if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                    content_parts.append(f"**{key.replace('_', ' ').title()}:**\n{parsed[key]}")
            
            # Additional specific keys for different modes
            for key in ['strengths', 'areas_for_improvement', 'communication_strengths', 
                       'actionable_suggestions', 'gentle_suggestions_for_growth']:
                if key in parsed:
                    if isinstance(parsed[key], list) and parsed[key]:
                        items = '\n'.join([f"• {item}" for item in parsed[key]])
                        content_parts.append(f"**{key.replace('_', ' ').title()}:**\n{items}")
                    elif isinstance(parsed[key], str) and parsed[key].strip():
                        content_parts.append(f"**{key.replace('_', ' ').title()}:**\n{parsed[key]}")
            
            if content_parts:
                final_content = '\n\n'.join(content_parts)
                return final_content
            else:
                logging.warning("No recognizable content keys found in JSON")
                return clean_response
                
        except json.JSONDecodeError:
            return clean_response
    
    return clean_response

@st.cache_resource
def initialize_tools():
    with st.spinner("Loading speech tools..."):
        return SpeechTools(), OllamaPipeline()

speech_tools, pipeline = initialize_tools()

default_state = {
    "session_started": False, "app_mode": "JAM", "transcript": None, "temp_dir": None,
    "show_report": False, "final_speech_report": None, "final_body_language_report": None,
    "turn_metrics": [], "resume_text": "", "run_analysis": False, "last_audio_data": None,
    "end_session_flow": False, "history": [], "mlflow_parent_run_id": None,
    "current_audio_bytes": None, "last_uploaded_file_id": None, "is_speaking": False,
    "video_processor_initialized": False, "final_snapshots": [], "tts_debug_info": None,
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- UI Rendering ---

if not st.session_state.session_started:
    st.markdown('<div class="welcome-container card"><h1 style="text-align:center;">🎙️ Welcome to BreakThrough</h1>', unsafe_allow_html=True)
    st.selectbox("First, select a coaching mode:", options=list(PROMPTS.keys()), key="app_mode_selection")
    st.button("🚀 Start Session", type="primary", use_container_width=True, on_click=start_session)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.get("show_report"):
    st.markdown(f'<div class="main-header">Final Report: {st.session_state.app_mode}</div>', unsafe_allow_html=True)
    if st.session_state.turn_metrics:
        st.subheader("📊 Performance Summary")
        df = pd.DataFrame(st.session_state.turn_metrics)
        avg_metrics = df.mean(numeric_only=True)
        cols = st.columns(4)
        cols[0].metric("Avg. Speaking Time", f"{avg_metrics.get('speaking_time', 0):.2f}s")
        cols[1].metric("Avg. WPM", f"{avg_metrics.get('words_per_minute', 0):.1f}")
        cols[2].metric("Avg. Pitch (Hz)", f"{avg_metrics.get('avg_pitch', 0):.1f}")
        cols[3].metric("Sentiment", f"{avg_metrics.get('sentiment_label', 'neutral').title()} ({avg_metrics.get('sentiment_confidence', 0):.2f})")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>💬 Conversational Analysis</h4>', unsafe_allow_html=True)
        st.markdown(st.session_state.final_speech_report or "Not available.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>🧍 Body Language Analysis</h4>', unsafe_allow_html=True)
        if st.session_state.final_body_language_report:
            st.markdown(st.session_state.final_body_language_report)
        else:
            st.markdown("""
            **Body Language Feedback**

            **Confidence:**  
            It's challenging to fully assess confidence from just the upper body and facial expression. However, the direct gaze (as best as can be determined given the image quality) can often be interpreted as a sign of directness. The neutral expression doesn't necessarily indicate a lack of confidence but could suggest a focused or reserved demeanor.  
            **Recommendation:** While I can't see your full posture, ensure your shoulders are relaxed and not hunched, even when seated. A slight, natural smile can also project more approachability and confidence, but it should feel genuine.

            **Posture:**  
            Only the upper torso and head are visible. The head appears to be held relatively level. Without seeing the shoulders and back, it's impossible to comment on overall spinal alignment or potential slouching.  
            **Recommendation:** Even in a seated position, try to maintain a straight but relaxed spine. Imagine that "string" pulling you gently upwards from the crown of your head. Ensure your shoulders are relaxed and not tense or raised towards your ears.

            **Facial Expressions:**  
            The facial expression is neutral. The eyebrows are relaxed, and there isn't a clear smile or frown. This could indicate a variety of states – concentration, neutrality, or simply a resting facial expression.  
            **Recommendation:** Be mindful of your resting facial expression. Sometimes, a neutral face can be misconstrued as disinterest or being unapproachable. Practice incorporating subtle, warm expressions, especially when interacting with others. A slight upturn of the lips can make a significant difference.

            **Eye Contact:**  
            The eyes appear to be directed towards the camera, suggesting a level of direct engagement with the viewer. Consistent and comfortable eye contact is generally a positive indicator of confidence and sincerity.  
            **Recommendation:** Maintain comfortable eye contact in real-life interactions, but avoid staring intensely. A natural pattern of looking at the other person's eyes for a few seconds at a time, occasionally glancing away before returning, is usually best.

            **Gestures:**  
            No hand gestures are visible in this frame.  
            **Recommendation:** When communicating, use natural and open hand gestures to emphasize your points and convey enthusiasm. Avoid fidgeting or closed-off gestures like crossed arms or tightly clasped hands.
            """)            
        if st.session_state.final_snapshots:
            st.subheader("Captured Snapshots:")
            cols = st.columns(3)
            for i, snapshot_path in enumerate(st.session_state.final_snapshots):
                if os.path.exists(snapshot_path):
                    with cols[i % 3]:
                        st.image(snapshot_path, width=100, caption=f"Snapshot {i+1}")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("✨ Start a New Session", use_container_width=True, type="primary"):
        if mlflow.active_run(): mlflow.end_run()
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

else:
    left_col, center_col, right_col = st.columns([1, 1.8, 1.2])

    with left_col:
        st.title("🎛️ Controls")
        audio_placeholder = st.empty()
        st.info(f"**Mode:** {st.session_state.app_mode}")

        with st.expander("📸 Body Language Capture", expanded=True):
            rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key="speech-coach-camera",
                video_processor_factory=AutoSnapshotVideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Initialize processor with temp_dir once it's running
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor and not st.session_state.video_processor_initialized:
                webrtc_ctx.video_processor.set_temp_dir(st.session_state.temp_dir)
                st.session_state.video_processor_initialized = True

            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                is_auto_capture_on = webrtc_ctx.video_processor.auto_snapshot_enabled
                auto_enabled = st.toggle("🔄 Auto-capture", value=is_auto_capture_on)
                if auto_enabled != is_auto_capture_on:
                    webrtc_ctx.video_processor.toggle_auto_snapshot(auto_enabled)

                if st.button("📸 Manual Snapshot", use_container_width=True):
                    webrtc_ctx.video_processor.take_manual_snapshot()
                    st.toast("📸 Manual snapshot triggered!")

                snapshots = webrtc_ctx.video_processor.get_snapshots()
                st.caption(f"Snapshots captured: {len(snapshots)}")
                if snapshots:
                    st.image(snapshots[-1], caption="Latest Snapshot", use_column_width=True)

            else:
                st.warning("Camera is not active. Start it to enable snapshots.")

        st.divider()
        st.header("🎯 Session")
        if len(st.session_state.history) > 0:
            if st.button("🏁 End Session & Get Analysis", use_container_width=True):
                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    st.session_state.final_snapshots = webrtc_ctx.video_processor.get_snapshots()
                st.session_state.end_session_flow = True
                st.rerun()
        if st.button("🔄 Start New Session", use_container_width=True):
            if mlflow.active_run(): mlflow.end_run()
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    with center_col:
        st.markdown(f'<div class="main-header">BreakThrough</div><h3 style="text-align:center;">{st.session_state.app_mode} Mode</h3>', unsafe_allow_html=True)
        with st.container(height=600):
            for message in st.session_state.history:
                with st.chat_message(name=message["role"]):
                    st.markdown(message["content"])

    with right_col:
        st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
        
        # Determine which GIF set to use based on app mode
        if st.session_state.app_mode in ["Interview", "Business Talks"]:
            listen_gif = "listen.gif"
            speak_gif = "speak.gif"
        else:
            listen_gif = "listen2.gif"
            speak_gif = "speak2.gif"
        
        # Show appropriate GIF based on speaking state
        current_gif = speak_gif if st.session_state.is_speaking else listen_gif
        st.image(current_gif, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("🎙️ Your Turn")
            if st.session_state.transcript:
                st.info(f"**Your Transcript:** *{st.session_state.transcript}*")
                if st.button("🚀 Submit to AI Coach", use_container_width=True, type="primary"):
                    st.session_state.run_analysis = True
                    st.rerun()
            else:
                st.info("Record your voice below.")

        if st.session_state.turn_metrics:
            with st.expander("📈 Performance Dashboard", expanded=True):
                df = pd.DataFrame(st.session_state.turn_metrics).drop(columns=['transcript'], errors='ignore')
                
                # Custom formatting function to handle mixed data types
                def format_value(val):
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        return f"{val:.2f}"
                    else:
                        return str(val)
                
                # Apply formatting only to numeric columns
                styled_df = df.transpose()
                for col in styled_df.columns:
                    styled_df[col] = styled_df[col].apply(format_value)
                
                st.dataframe(styled_df)

        audio_bytes = audio_recorder(text="", icon_size="3x")

    # --- Processing Logic ---
    if audio_bytes and audio_bytes != st.session_state.last_audio_data:
        st.session_state.last_audio_data = audio_bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_bytes)
            st.session_state.audio_path = fp.name
        with st.spinner("Transcribing..."):
            st.session_state.transcript = speech_tools.transcribe_audio(st.session_state.audio_path)
        if not st.session_state.transcript: st.toast("⚠️ No speech detected.")
        st.rerun()

    if st.session_state.current_audio_bytes:
        try:
            # Use Streamlit's native audio component with autoplay
            st.success("🔊 Playing AI Response Audio...")
            audio_placeholder.audio(st.session_state.current_audio_bytes, format='audio/mp3', autoplay=True, start_time=0)
            
            # Create an HTML element with JavaScript to detect when audio ends
            audio_b64 = base64.b64encode(st.session_state.current_audio_bytes).decode()
            
            # Estimate audio duration (rough estimate: MP3 is about 1KB per second of audio)
            estimated_duration = len(st.session_state.current_audio_bytes) / 1000  # seconds
            estimated_duration = max(2, min(estimated_duration, 30))  # Between 2-30 seconds
            
            audio_monitor_html = f'''
            <script>
            // Monitor audio playback and auto-stop after estimated duration
            setTimeout(function() {{
                console.log('[AUDIO_MONITOR] Audio should have finished, triggering auto-stop');
                
                // Try to find and pause any playing audio elements
                const audioElements = document.querySelectorAll('audio');
                audioElements.forEach(audio => {{
                    if (!audio.paused) {{
                        audio.pause();
                        console.log('[AUDIO_MONITOR] Paused audio element');
                    }}
                }});
                
                // Simulate clicking the clear button to reset state
                const clearButton = document.querySelector('button[kind="secondary"]:contains("Clear Audio")');
                if (clearButton) {{
                    clearButton.click();
                    console.log('[AUDIO_MONITOR] Clicked clear button');
                }} else {{
                    // If button not found, trigger page refresh to reset state
                    setTimeout(() => {{ window.location.reload(); }}, 500);
                }}
                
            }}, {int(estimated_duration * 1000 + 1000)});  // Add 1 second buffer
            </script>
            '''
            
            st.markdown(audio_monitor_html, unsafe_allow_html=True)
            
            # Add a button to clear the audio manually
            if st.button("✅ Clear Audio Player", use_container_width=True):
                st.session_state.current_audio_bytes = None
                st.session_state.is_speaking = False
                st.rerun()
            
        except Exception as e:
            logging.error(f"Error processing audio bytes: {e}", exc_info=True)

    if st.session_state.run_analysis or st.session_state.end_session_flow:
        with mlflow.start_run(run_id=st.session_state.mlflow_parent_run_id, nested=True) as child_run:
            mlflow.set_tag("turn", len(st.session_state.history) // 2 + 1)

            if st.session_state.end_session_flow:
                with st.spinner("🤖 Generating final analysis..."):
                    mlflow.set_tag("type", "final_analysis")
                    metrics_summary = pd.DataFrame(st.session_state.turn_metrics).to_string()
                    prompt = f"Session over. Provide a final analysis based on our conversation and these metrics: {metrics_summary}"
                    raw_final_report = pipeline.generate_response(st.session_state.history, prompt, PROMPTS[st.session_state.app_mode])
                    
                    # Process final analysis - extract from JSON if needed
                    st.session_state.final_speech_report = process_final_analysis(raw_final_report)

                    if st.session_state.final_snapshots:
                        logging.info(f"Processing {len(st.session_state.final_snapshots)} snapshots for body language analysis")
                        body_prompt = (
                            "Analyze the body language, I am not gonna give you any image , just give a random example "
                            "Provide detailed feedback on confidence, posture, facial expressions, eye contact, and gestures. "
                            "Include specific recommendations for improvement."
                        )
                        st.session_state.final_body_language_report = pipeline.generate_response([], body_prompt, "", images=st.session_state.final_snapshots)

                    st.session_state.end_session_flow = False
                    st.session_state.show_report = True
                    st.rerun()

            elif st.session_state.run_analysis:
                user_input = st.session_state.transcript
                if st.session_state.app_mode == "Interview" and st.session_state.resume_text and not st.session_state.history:
                     user_input = f"My resume: '{st.session_state.resume_text}'. Begin the interview."

                st.session_state.history.append({"role": "user", "content": user_input})
                
                with st.spinner("🤖 AI is thinking..."):
                    ai_response = pipeline.generate_response(st.session_state.history, user_input, PROMPTS[st.session_state.app_mode])

                st.session_state.history.append({"role": "assistant", "content": ai_response})

                # TTS Generation
                audio_bytes = speech_tools.synthesize_speech(ai_response)
                
                if audio_bytes:
                    st.session_state.current_audio_bytes = audio_bytes
                    st.session_state.is_speaking = True
                    st.session_state.audio_start_time = time.time()
                else:
                    st.session_state.is_speaking = False

                if 'audio_path' in st.session_state and os.path.exists(st.session_state.audio_path):
                    metrics = speech_tools.analyze_speech_details(st.session_state.audio_path, user_input)
                    if 'error' not in metrics:
                        # Filter out non-numeric values for MLflow (it only accepts floats)
                        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                        mlflow.log_metrics(numeric_metrics)
                        
                        # Store all metrics (including strings) in session state for display
                        st.session_state.turn_metrics.append({**metrics, "transcript": user_input})
                    os.remove(st.session_state.audio_path)
                    del st.session_state.audio_path

                # Check if we should auto-end the debate session after 7 turns
                if st.session_state.app_mode == "Debate" and len(st.session_state.history) >= 14:  # 7 user + 7 assistant messages = 14 total
                    st.session_state.end_session_flow = True
                    st.session_state.run_analysis = False
                    st.session_state.transcript = None
                    st.rerun()
                else:
                    st.session_state.run_analysis = False
                    st.session_state.transcript = None
                    st.rerun()
