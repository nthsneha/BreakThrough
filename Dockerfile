# Use a slim Python base image for smaller size
FROM python:3.10-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_HOST=0.0.0.0:11434 \
    PATH="/usr/local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_GATHER_USAGE_STATS=false

# Install system dependencies in a single RUN command
# build-essential: For compiling some Python packages (e.g., parselmouth, opencv-python)
# ffmpeg: For audio processing (used by whisper)
# curl: For downloading ollama
# git: Sometimes needed by pip for certain packages or dependencies
# libgl1-mesa-glx, libxext6, libsm6, libxrender1: Common dependencies for OpenCV and other graphical/media libraries
# portaudio19-dev: For sounddevice package
# alsa-utils: For audio system support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    git \
    libgl1-mesa-glx \
    libxext6 \
    libsm6 \
    libxrender1 \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Ollama server
# Download the Ollama binary and make it executable
# Create a dedicated user and directory for Ollama data
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama && \
    useradd -r -s /bin/false -m -d /usr/share/ollama ollama && \
    mkdir -p /usr/share/ollama/.ollama && \
    chown -R ollama:ollama /usr/share/ollama/.ollama

# Set working directory for the application
WORKDIR /app

# Change ownership of the /app directory to the 'ollama' user
# This allows the 'ollama' user to write files (like MLflow logs) in /app
RUN chown -R ollama:ollama /app

# Switch to the 'ollama' user from this point onwards for security and permissions
USER ollama

# Add the user's local bin directory to the PATH, as pip might install executables there
ENV PATH="/home/ollama/.local/bin:${PATH}"

# Copy requirements files
COPY --chown=ollama:ollama requirements.txt .

# Install Python dependencies with specific handling for problematic packages
# Install PyTorch and related packages first for better dependency resolution
RUN pip install --no-cache-dir --user torch>=2.0.0 transformers>=4.35.0 accelerate>=0.24.0

# Install parselmouth specifically, ignoring its dependencies to avoid googleads issues
RUN pip install --no-cache-dir --user parselmouth --no-deps

# Install the remaining dependencies from requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Install additional Streamlit-specific packages that are needed based on the code
RUN pip install --no-cache-dir --user \
    streamlit \
    streamlit-webrtc \
    audio-recorder-streamlit \
    gtts \
    opencv-python-headless \
    sounddevice \
    soundfile \
    PyPDF2 \
    textblob \
    mlflow \
    ollama \
    Pillow

# Copy the application files
COPY --chown=ollama:ollama streamlit_app_with_tts_avatar.py .
COPY --chown=ollama:ollama listen.gif .
COPY --chown=ollama:ollama speak.gif .
COPY --chown=ollama:ollama listen2.gif .
COPY --chown=ollama:ollama speak2.gif .

# Create mlruns directory for MLflow
RUN mkdir -p mlruns && chown -R ollama:ollama mlruns

# Expose ports:
# 8501: Default Streamlit port
# 11434: Default Ollama API port
EXPOSE 8501
EXPOSE 11434

# Health check to ensure both services are running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/version && curl -f http://localhost:8501/

# Command to run the application:
# Start the Ollama server in the background using 'ollama serve'.
# Wait a moment for it to initialize, then start Streamlit.
# Use 'exec' to ensure Streamlit becomes the main process for proper signal handling.
CMD ["sh", "-c", "ollama serve > /dev/null 2>&1 & sleep 5 && exec streamlit run streamlit_app_with_tts_avatar.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"]
