#!/usr/bin/env python3
"""
Advanced Real-Time Speaker Detection System with Professional ASR Integration
Features:
- OpenAI Whisper integration for high-quality transcription
- Google Speech-to-Text API support
- WebRTC VAD for robust voice activity detection
- Real-time audio processing with multiple microphones
- Speaker diarization and identification
- WebSocket communication with frontend
"""

import asyncio
import json
import logging
import numpy as np
import wave
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import base64

# Core audio processing
import pyaudio
import webrtcvad
import collections

# ASR APIs
import openai
import whisper
from google.cloud import speech
from google.oauth2 import service_account

# WebSocket server
import websockets
from websockets.server import serve

# Audio analysis
from scipy import signal
from scipy.fft import fft
import librosa

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: int = pyaudio.paInt16
    vad_mode: int = 3  # Most aggressive VAD mode
    frame_duration_ms: int = 30  # WebRTC VAD frame duration
    
@dataclass
class SpeakerData:
    """Speaker information and metrics"""
    id: int
    name: str
    device_id: str
    intensity: float
    is_active: bool
    last_activity: datetime
    peak_frequency: float
    rms_history: List[float]
    color: str
    confidence: float = 0.0
    total_speech_time: float = 0.0

@dataclass
class TranscriptionResult:
    """Transcription result with metadata"""
    id: str
    speaker_id: int
    speaker_name: str
    text: str
    confidence: float
    timestamp: datetime
    audio_duration: float
    language: str
    source: str  # 'whisper', 'google', 'openai'

class WebRTCVAD:
    """Enhanced Voice Activity Detection using WebRTC"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.vad = webrtcvad.Vad(config.vad_mode)
        self.frame_length = int(config.sample_rate * config.frame_duration_ms / 1000)
        
    def is_speech(self, audio_data: bytes) -> bool:
        """Detect if audio frame contains speech"""
        try:
            # Ensure frame is correct length
            if len(audio_data) != self.frame_length * 2:  # 2 bytes per sample for int16
                return False
            return self.vad.is_speech(audio_data, self.config.sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False
    
    def get_speech_segments(self, audio_frames: List[bytes]) -> List[Tuple[int, int]]:
        """Get continuous speech segments from audio frames"""
        speech_frames = []
        for i, frame in enumerate(audio_frames):
            is_speech = self.is_speech(frame)
            speech_frames.append(is_speech)
        
        # Find continuous speech segments
        segments = []
        start_idx = None
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and start_idx is None:
                start_idx = i
            elif not is_speech and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
        
        # Close final segment if needed
        if start_idx is not None:
            segments.append((start_idx, len(speech_frames)))
            
        return segments

class WhisperASR:
    """OpenAI Whisper integration for high-quality transcription"""
    
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        logger.info(f"Loaded Whisper model: {model_size}")
    
    async def transcribe_audio(self, audio_data: np.ndarray, language: str = None) -> TranscriptionResult:
        """Transcribe audio using Whisper"""
        try:
            # Run Whisper in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.model.transcribe(
                    audio_data, 
                    language=language,
                    task="transcribe",
                    fp16=False
                )
            )
            
            return TranscriptionResult(
                id=f"whisper_{int(time.time() * 1000)}",
                speaker_id=0,  # Will be assigned later
                speaker_name="Unknown",
                text=result["text"].strip(),
                confidence=result.get("confidence", 0.0),
                timestamp=datetime.now(),
                audio_duration=len(audio_data) / 16000,
                language=result.get("language", language or "en"),
                source="whisper"
            )
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None

class GoogleSpeechASR:
    """Google Speech-to-Text API integration"""
    
    def __init__(self, credentials_path: str = None):
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = speech.SpeechClient(credentials=credentials)
        else:
            self.client = speech.SpeechClient()
        logger.info("Initialized Google Speech-to-Text client")
    
    async def transcribe_audio(self, audio_data: bytes, language: str = "en-US") -> TranscriptionResult:
        """Transcribe audio using Google Speech-to-Text"""
        try:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                model="latest_long"
            )
            
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Run Google API in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.recognize(config=config, audio=audio)
            )
            
            if response.results:
                result = response.results[0]
                alternative = result.alternatives[0]
                
                return TranscriptionResult(
                    id=f"google_{int(time.time() * 1000)}",
                    speaker_id=0,
                    speaker_name="Unknown",
                    text=alternative.transcript.strip(),
                    confidence=alternative.confidence,
                    timestamp=datetime.now(),
                    audio_duration=len(audio_data) / (16000 * 2),
                    language=language,
                    source="google"
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Google Speech-to-Text error: {e}")
            return None

class OpenAIWhisperAPI:
    """OpenAI Whisper API integration"""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.OpenAI()
        logger.info("Initialized OpenAI Whisper API client")
    
    async def transcribe_audio(self, audio_data: bytes, language: str = None) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Create temporary audio file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write WAV header and data
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_data)
                
                # Transcribe using OpenAI API
                with open(tmp_file.name, 'rb') as audio_file:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language=language,
                            response_format="verbose_json"
                        )
                    )
                
                import os
                os.unlink(tmp_file.name)
                
                return TranscriptionResult(
                    id=f"openai_{int(time.time() * 1000)}",
                    speaker_id=0,
                    speaker_name="Unknown",
                    text=response.text.strip(),
                    confidence=getattr(response, 'confidence', 0.0),
                    timestamp=datetime.now(),
                    audio_duration=getattr(response, 'duration', len(audio_data) / (16000 * 2)),
                    language=getattr(response, 'language', language or "en"),
                    source="openai"
                )
                
        except Exception as e:
            logger.error(f"OpenAI Whisper API error: {e}")
            return None

class AudioProcessor:
    """Advanced audio processing and analysis"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio signal"""
        return np.sqrt(np.mean(audio_data ** 2))
    
    def calculate_intensity_db(self, rms: float) -> float:
        """Convert RMS to decibel intensity"""
        if rms > 0:
            return 20 * np.log10(rms)
        return -100
    
    def find_peak_frequency(self, audio_data: np.ndarray) -> float:
        """Find the peak frequency in audio signal"""
        # Apply window to reduce spectral leakage
        windowed = audio_data * signal.windows.hamming(len(audio_data))
        
        # Compute FFT
        fft_data = np.abs(fft(windowed))
        freqs = np.fft.fftfreq(len(fft_data), 1/self.config.sample_rate)
        
        # Find peak frequency (positive frequencies only)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]
        
        peak_idx = np.argmax(positive_fft)
        return positive_freqs[peak_idx]
    
    def extract_mfcc_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features for speaker identification"""
        try:
            # Convert to float32 for librosa
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=self.config.sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=256
            )
            
            # Return mean of MFCCs across time
            return np.mean(mfccs, axis=1)
        except Exception as e:
            logger.warning(f"MFCC extraction error: {e}")
            return np.zeros(13)

class SpeakerDetectionSystem:
    """Main speaker detection system with advanced ASR integration"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.webrtc_vad = WebRTCVAD(config)
        
        # Audio system
        self.pyaudio = pyaudio.PyAudio()
        self.audio_streams = {}
        self.is_recording = False
        
        # Speaker management
        self.speakers: Dict[int, SpeakerData] = {}
        self.next_speaker_id = 0
        
        # ASR engines
        self.whisper_asr = None
        self.google_asr = None
        self.openai_asr = None
        self.current_asr = "whisper"  # Default ASR engine
        
        # Audio buffers for continuous processing
        self.audio_buffers = {}
        self.speech_segments = {}
        
        # WebSocket clients
        self.websocket_clients = set()
        
        # Processing thread
        self.processing_thread = None
        self.should_stop = False
        
    def initialize_asr_engines(self, whisper_model="base", google_credentials=None, openai_api_key=None):
        """Initialize ASR engines"""
        try:
            # Initialize Whisper (always available)
            self.whisper_asr = WhisperASR(whisper_model)
            
            # Initialize Google Speech-to-Text if credentials provided
            if google_credentials:
                self.google_asr = GoogleSpeechASR(google_credentials)
                
            # Initialize OpenAI Whisper API if key provided
            if openai_api_key:
                self.openai_asr = OpenAIWhisperAPI(openai_api_key)
                
            logger.info("ASR engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ASR engines: {e}")
    
    def get_available_devices(self) -> List[Dict]:
        """Get available audio input devices"""
        devices = []
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'id': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': info['defaultSampleRate']
                })
        return devices
    
    async def add_microphone(self, device_id: int = None, device_name: str = None) -> int:
        """Add a microphone to the system"""
        try:
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            
            # Create audio stream
            stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=None
            )
            
            self.audio_streams[speaker_id] = stream
            
            # Create speaker data
            self.speakers[speaker_id] = SpeakerData(
                id=speaker_id,
                name=device_name or f"Microphone {speaker_id}",
                device_id=str(device_id or "default"),
                intensity=0.0,
                is_active=False,
                last_activity=datetime.now(),
                peak_frequency=0.0,
                rms_history=[],
                color=self.generate_speaker_color(speaker_id)
            )
            
            # Initialize audio buffer
            self.audio_buffers[speaker_id] = collections.deque(maxlen=1000)
            self.speech_segments[speaker_id] = []
            
            await self.broadcast_message({
                'type': 'speaker_added',
                'speaker': asdict(self.speakers[speaker_id])
            })
            
            logger.info(f"Added microphone {speaker_id}: {device_name}")
            return speaker_id
            
        except Exception as e:
            logger.error(f"Error adding microphone: {e}")
            raise
    
    def generate_speaker_color(self, speaker_id: int) -> str:
        """Generate a color for speaker visualization"""
        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
            '#9b59b6', '#1abc9c', '#ff6b6b', '#4ecdc4',
            '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe'
        ]
        return colors[speaker_id % len(colors)]
    
    async def start_detection(self):
        """Start the detection system"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.should_stop = False
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._audio_processing_loop)
        self.processing_thread.start()
        
        await self.broadcast_message({
            'type': 'detection_started',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Started speaker detection system")
    
    def _audio_processing_loop(self):
        """Main audio processing loop (runs in separate thread)"""
        while not self.should_stop and self.is_recording:
            try:
                for speaker_id, stream in self.audio_streams.items():
                    if stream.is_active():
                        # Read audio data
                        try:
                            audio_data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                            asyncio.create_task(self._process_audio_chunk(speaker_id, audio_data))
                        except Exception as e:
                            logger.warning(f"Error reading from stream {speaker_id}: {e}")
                
                time.sleep(0.01)  # Small sleep to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                
    async def _process_audio_chunk(self, speaker_id: int, audio_data: bytes):
        """Process individual audio chunk"""
        try:
            speaker = self.speakers.get(speaker_id)
            if not speaker:
                return
            
            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate audio metrics
            rms = self.audio_processor.calculate_rms(audio_np)
            intensity_db = self.audio_processor.calculate_intensity_db(rms)
            peak_freq = self.audio_processor.find_peak_frequency(audio_np)
            
            # Voice Activity Detection
            is_speech = self.webrtc_vad.is_speech(audio_data)
            
            # Update speaker data
            speaker.intensity = max(0, min(100, ((intensity_db + 60) / 60) * 100))
            speaker.peak_frequency = peak_freq
            speaker.rms_history.append(rms)
            
            if len(speaker.rms_history) > 50:
                speaker.rms_history.pop(0)
            
            current_time = datetime.now()
            
            if is_speech:
                speaker.is_active = True
                speaker.last_activity = current_time
                
                # Add to audio buffer for transcription
                self.audio_buffers[speaker_id].append(audio_data)
                
                # Trigger transcription if we have enough audio
                if len(self.audio_buffers[speaker_id]) >= 50:  # ~1 second of audio
                    await self._trigger_transcription(speaker_id)
            else:
                # Keep active for 500ms after speech stops
                time_since_activity = (current_time - speaker.last_activity).total_seconds()
                speaker.is_active = time_since_activity < 0.5
            
            # Broadcast speaker update
            await self.broadcast_message({
                'type': 'speaker_update',
                'speaker_id': speaker_id,
                'intensity': speaker.intensity,
                'is_active': speaker.is_active,
                'peak_frequency': speaker.peak_frequency
            })
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for speaker {speaker_id}: {e}")
    
    async def _trigger_transcription(self, speaker_id: int):
        """Trigger transcription for accumulated audio"""
        try:
            # Get audio buffer
            audio_chunks = list(self.audio_buffers[speaker_id])
            self.audio_buffers[speaker_id].clear()
            
            if not audio_chunks:
                return
            
            # Combine audio chunks
            combined_audio = b''.join(audio_chunks)
            audio_np = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Choose ASR engine and transcribe
            transcription = None
            
            if self.current_asr == "whisper" and self.whisper_asr:
                transcription = await self.whisper_asr.transcribe_audio(audio_np)
            elif self.current_asr == "google" and self.google_asr:
                transcription = await self.google_asr.transcribe_audio(combined_audio)
            elif self.current_asr == "openai" and self.openai_asr:
                transcription = await self.openai_asr.transcribe_audio(combined_audio)
            
            if transcription and transcription.text:
                # Update transcription with speaker info
                speaker = self.speakers[speaker_id]
                transcription.speaker_id = speaker_id
                transcription.speaker_name = speaker.name
                
                # Broadcast transcription
                await self.broadcast_message({
                    'type': 'transcription',
                    'transcription': asdict(transcription)
                })
                
                logger.info(f"Transcribed from {speaker.name}: {transcription.text}")
                
        except Exception as e:
            logger.error(f"Error in transcription for speaker {speaker_id}: {e}")
    
    async def stop_detection(self):
        """Stop the detection system"""
        self.is_recording = False
        self.should_stop = True
        
        # Stop processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Close audio streams
        for stream in self.audio_streams.values():
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        
        self.audio_streams.clear()
        self.speakers.clear()
        
        await self.broadcast_message({
            'type': 'detection_stopped',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("Stopped speaker detection system")
    
    async def switch_asr_engine(self, engine: str):
        """Switch ASR engine"""
        available_engines = []
        if self.whisper_asr: available_engines.append("whisper")
        if self.google_asr: available_engines.append("google")
        if self.openai_asr: available_engines.append("openai")
        
        if engine in available_engines:
            self.current_asr = engine
            await self.broadcast_message({
                'type': 'asr_switched',
                'engine': engine
            })
            logger.info(f"Switched to ASR engine: {engine}")
        else:
            logger.warning(f"ASR engine {engine} not available")
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if self.websocket_clients:
            # Create tasks for all clients
            tasks = []
            for client in self.websocket_clients.copy():
                tasks.append(self._send_to_client(client, message))
            
            # Send to all clients concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_client(self, websocket, message: dict):
        """Send message to individual WebSocket client"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Error sending to client: {e}")
            # Remove disconnected client
            self.websocket_clients.discard(websocket)
    
    async def handle_websocket_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected from {websocket.remote_address}")
        
        try:
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'connected',
                'available_devices': self.get_available_devices(),
                'current_asr': self.current_asr,
                'available_asr': [
                    engine for engine in ['whisper', 'google', 'openai']
                    if getattr(self, f'{engine}_asr') is not None
                ]
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
                    
        except Exception as e:
            logger.info(f"WebSocket client disconnected: {e}")
        finally:
            self.websocket_clients.discard(websocket)
    
    async def _handle_client_message(self, websocket, data: dict):
        """Handle message from WebSocket client"""
        message_type = data.get('type')
        
        if message_type == 'add_microphone':
            device_id = data.get('device_id')
            device_name = data.get('device_name', f'Device {device_id}')
            speaker_id = await self.add_microphone(device_id, device_name)
            await websocket.send(json.dumps({
                'type': 'microphone_added',
                'speaker_id': speaker_id
            }))
            
        elif message_type == 'start_detection':
            await self.start_detection()
            
        elif message_type == 'stop_detection':
            await self.stop_detection()
            
        elif message_type == 'switch_asr':
            engine = data.get('engine')
            await self.switch_asr_engine(engine)
            
        elif message_type == 'get_system_status':
            await websocket.send(json.dumps({
                'type': 'system_status',
                'is_recording': self.is_recording,
                'speaker_count': len(self.speakers),
                'current_asr': self.current_asr
            }))
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_recording:
            asyncio.create_task(self.stop_detection())
        
        if self.pyaudio:
            self.pyaudio.terminate()

async def main():
    """Main application entry point"""
    # Configuration
    config = AudioConfig()
    
    # Initialize system
    system = SpeakerDetectionSystem(config)
    
    # Initialize ASR engines (configure these with your credentials)
    system.initialize_asr_engines(
        whisper_model="base",  # or "small", "medium", "large"
        google_credentials=None,  # Path to Google service account JSON
        openai_api_key=None  # Your OpenAI API key
    )
    
    # Start WebSocket server
    server = await serve(
        system.handle_websocket_client,
        "localhost",
        8765,
        ping_interval=20,
        ping_timeout=10
    )
    
    logger.info("Advanced Speaker Detection System started on ws://localhost:8765")
    
    try:
        # Keep server running
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        system.cleanup()

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())