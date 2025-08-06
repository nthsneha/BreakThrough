# BreakThrough
## An AI-driven platform for real-time speech analysis and personalized public speaking feedback.  
This repo demonstrates an AI-powered public speaking coach that analyzes your speech across categories like debates, interviews and therapy sessions using Whisper, Parselmouth and Gemma 3n E2B. Built with Streamlit, this tool provides actionable feedback on your delivery, fluency and confidence.
## Key Features:  
- **AI-Powered Speech Analysis:** Uses Whisper for accurate speech-to-text transcription and Parselmouth for prosody analysis (intonation, pace, clarity).

- **Category-Specific Practice Modes:** Choose from 7 tailored rooms: JAM, Debate, Interview, Business Talks, Reading, Therapy, and Socialising - each simulating real-world speaking scenarios.

- **Session-wise Performance Feedback:** Users receive personalized reports on fluency, filler words, prosody, body language and areas of improvement.

- **Video Snapshots for Non-Verbal Cues:** Captures 10 image frames per minute to analyze posture, facial expression, and engagement (future scope or WIP).

- **Personalized Feedback via RAG:** Maintains a user-specific error history for smarter and adaptive coaching over time.

- **MLOps-Enabled with MLflow:** Tracks experiments and automates fine-tuning to keep models fresh and performance-optimized.

- **Private & Offline-Ready:** Runs entirely on-device using Gemma3n, ensuring full user privacy and enabling usage without internet. 

- **Interactive Streamlit Frontend:** Simple UI built using Streamlit for smooth interaction.  

- **Fine-tuned model:** Fine-tuned Gemma3n E2B model using unsloth.

## Tech Stack: 
### Frontend
- **Streamlit** - For building the interactive web UI and managing session-based inputs.
### Backend
- **Gemma 3n E2B (via Ollama on huggingface)** - Lightweight LLM used for generating AI responses and feedback generation.
- **Parselmouth (Praat)** - For extracting prosodic features like pitch, intensity and speech rate.
- **Whisper ASR (OpenAI)** - For real-time and accurate speech-to-text transcription.
- **Roberta-base (Twitter variant)** - For sentiment analysis of transcribed speech.
- **Silero VAD (Voice Activity Detection)** – Segments voice from silence to isolate meaningful speech for processing.
- **Ollama** – Lightweight LLM runtime used for deploying the main reasoning engine (Gemma 3n).
- **ChromaDB** - Vectore Store; lightweight, fast embedding-based search for response relevance.
- **MongoDB** - For logging session data and enabling MLOps feedback loops.
- **TextRank + Custom Summarizer** – Hybrid summarization combining graph-based ranking with transformer outputs for session wrap-ups.
### MLOps & Experiment Tracking
- **MLflow** - Used to track model performance, log feedback-based metrics, and manage finetuning pipelines.

## System Architecture:

![Title](./includes%20/SystemDiag.png)

## Optimisation:   
- **Text Summarization for Debates:** Integrated TextRank in the Debate Room to reduce transcript length before passing it to Gemma3n. This helps users speak longer per turn by staying within token limits.

- **Voice Activity Detection (VAD):** Applied VAD to remove non-speech audio before Whisper transcription, speeding up processing and improving efficiency.

- **Faster Transcription with Whisper:** Switched from Gemma3n to Whisper for transcription, as it’s significantly faster and more accurate for real-time speech-to-text.

- **Dockerization:** Successfully dockerized to make it lightweight.

## MLOps  
- **Integrated MLflow Tracking:** All experiment runs and metrics like transcription time, response time, time spent in an entire session (to measure user engagement), valid JSON rate and GPU usage are logged using MLflow, ensuring reproducibility and traceability.

- **Scheduled Fine-Tuning:** The system is built to support periodic fine-tuning using newly collected user data, keeping models relevant and adaptive.

- **Future-Ready Pipeline:** The MLOps setup is designed to support continuous integration, scalable deployment and version-controlled model improvement.

## How to use?  
### Method 1: Live Demo
No setup needed. Just click below and start practicing directly in your browser.
https://huggingface.co/spaces/Skaisnehu/BreakThrough
### Method 2: 
```bash
docker compose up -d
```
### (Optional) Start MLflow UI
To track experiments and logs:
```bash
nohup mlflow ui --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
```
Then open http://localhost:8501 in your browser.

## Known Issues:
- **Static Text Summarization Formula:** Instead of using an LLM for dynamic summarization, we're currently relying on a hardcoded formula for text summarization, which may reduce flexibility and accuracy for diverse inputs.

- **Limited Body Language Analysis:** Due to optimization challenges, body language analysis is done using n static images per minute rather than continuous video input. This may miss subtle gestures and microexpressions.
In out current version, Ollama faces issues in image processing so body language analysis is limited.

- **Fixed Debate Rounds:** The debate module is restricted to exactly 8 rounds. Users cannot exit early and still receive partial performance analysis.

- **No Mobile Deployment:** The application is currently not optimized or deployable on mobile phones due to high system requirements and lack of mobile-friendly architecture.

- **Lack of Continuous Session Memory:** Each session is treated independently. There's no memory or context retained across sessions, which limits personalized long-term feedback to an extent.

## Future Scope:
- **Refined UI/UX:** Build a proper front-end interface for smoother user interaction and session navigation.

- **Automated Finetuning Pipelines:** Leverage MLflow logging over time to automate model finetuning for personalization.

- **LLM-Powered Judging:** Integrate an external lightweight LLM to act as debate judge for more dynamic and contextual evaluations and facilitate flexible session length.

- **Smarter Summarization:** Replace the current formulaic summarizer with a hybrid of lightweight AI models and traditional methods like TextRank for more efficient and accurate summaries.

- **Mobile Compatibility:** Convert current models into mobile-optimized formats to enable deployment as a native mobile application.

- **User Feedback Logging:** Log user feedback as an MLOps metric to improve iterative development, model accuracy and user satisfaction.

## UI Screenshots:

![Title](./includes%20/UI1.jpeg)
![Title](./includes%20/UI2.jpeg)

## Impact:
- **Democratizing Public Speaking Coaching:** Provides free, AI-powered feedback across multiple speaking formats, making quality speech training accessible to everyone, regardless of location or resources.

- **Promotes soft-skill development:** Encourages clear articulation, emotional control, and structured thinking - vital for both academic and professional growth.

- **Personalized Growth Through Context-Aware Feedback:** Uses RAG to remember past mistakes and track progress, enabling users to improve meaningfully over time.  

Check out our demo on: https://www.youtube.com/watch?v=uipArL1wPQs

Thank you for going through the README. Hope you achieve a BreakThrough!