# Advanced Speaker Detection System - Setup Guide

## Overview
This upgraded speaker detection system integrates professional ASR APIs and advanced audio processing capabilities as requested by your boss. The system now includes:

- **Professional ASR Integration**: OpenAI Whisper, Google Speech-to-Text, OpenAI Whisper API
- **WebRTC VAD**: Robust voice activity detection instead of browser-based detection
- **Python Backend**: High-performance audio processing with real-time capabilities
- **Advanced Audio Analysis**: MFCC features, spectral analysis, speaker identification
- **WebSocket Communication**: Real-time bidirectional communication
- **Multi-engine Support**: Switch between ASR engines in real-time

## Installation Requirements

### System Requirements
- Python 3.8+
- Audio drivers and microphone access
- Modern web browser with WebSocket support


### Installation Steps

1. **Clone/Setup Project Directory**
```bash
mkdir advanced-speaker-detection
cd advanced-speaker-detection
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install System Audio Dependencies**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install portaudio
brew install ffmpeg
```

**Windows:**
- PyAudio wheels are available via pip
- Install FFmpeg from https://ffmpeg.org/download.html

## Configuration

### 1. Environment Variables
Create a `.env` file in your project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id

# System Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
VAD_MODE=3

# Whisper Model Configuration
WHISPER_MODEL=base  # tiny, base, small, medium, large
```

### 2. Google Cloud Speech-to-Text Setup
1. Create a Google Cloud Project
2. Enable Speech-to-Text API
3. Create a service account and download JSON key
4. Set the path in your `.env` file

### 3. OpenAI API Setup
1. Get API key from https://platform.openai.com/api-keys
2. Add to `.env` file

## Usage

### 1. Start the Backend Server
```bash
python advanced_speaker_detection_backend.py
```

### 2. Open the Frontend
Open `advanced_frontend.html` in your web browser

### 3. System Operation
1. **Connect**: Frontend automatically connects to backend
2. **Add Devices**: Select audio devices from the list
3. **Choose ASR Engine**: Select your preferred transcription engine
4. **Start Detection**: Begin real-time speaker detection and transcription

## Advanced Features

### 1. Multiple ASR Engine Support
- **Local Whisper**: Fast, offline transcription
- **Google Speech-to-Text**: High accuracy, cloud-based
- **OpenAI Whisper API**: Latest model, cloud-based

### 2. WebRTC Voice Activity Detection
- More robust than browser-based VAD
- Adjustable sensitivity modes
- Frame-based processing for real-time performance

### 3. Advanced Audio Processing
- MFCC feature extraction for speaker identification
- Real-time spectral analysis
- Peak frequency detection
- RMS intensity calculation

### 4. Professional Speaker Management
- Multi-device support
- Real-time speaker ranking
- Activity-based transcription assignment
- Color-coded speaker visualization




