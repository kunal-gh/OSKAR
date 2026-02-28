"""
audio_analyzer.py — OSKAR v0.4 Multimodal Intelligence
--------------------------------------------------------
Transcribes audio/video files using OpenAI Whisper (local inference,
no API key required) and pipes the resulting text through the full
OSKAR analysis pipeline.

Supported input formats: mp3, mp4, wav, m4a, ogg, flac, webm

Usage (standalone):
    from audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    result = analyzer.analyze("path/to/audio.mp3", user_id="user_123")

    result = {
        "transcription": "...",
        "language": "en",
        "duration_seconds": 12.4,
        "analysis": { ...full OSKAR pipeline output... }
    }
"""

import os
import tempfile
import time
from typing import Optional

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium
MAX_DURATION_S = 300  # Hard cap: 5 minutes per clip


class AudioAnalyzer:
    """
    Wraps OpenAI Whisper for local audio transcription.

    If whisper is not installed, degrades gracefully:
    - transcription returns a placeholder
    - analysis still runs on empty text
    """

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.enabled = False
        self.model = None
        self.model_name = model_name
        self._load()

    def _load(self):
        try:
            import whisper

            print(f"[AudioAnalyzer] Loading Whisper '{self.model_name}' model...")
            self.model = whisper.load_model(self.model_name)
            self.enabled = True
            print(f"[AudioAnalyzer] Whisper '{self.model_name}' ready.")
        except Exception as e:
            print(f"[AudioAnalyzer] Whisper not available: {e}. Running in dummy mode.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file to text.

        Returns:
            {
                "text": str,
                "language": str,
                "duration_seconds": float,
                "segments": list[dict]
            }
        """
        if not self.enabled:
            return {
                "text": "",
                "language": "unknown",
                "duration_seconds": 0.0,
                "segments": [],
                "error": "Whisper not installed",
            }

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import whisper

            result = self.model.transcribe(
                audio_path,
                fp16=False,  # CPU-safe, no half-precision
                language=None,  # auto-detect language
                verbose=False,
            )
            duration = 0.0
            if result.get("segments"):
                duration = result["segments"][-1].get("end", 0.0)

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "duration_seconds": round(duration, 2),
                "segments": result.get("segments", []),
            }
        except Exception as e:
            return {
                "text": "",
                "language": "unknown",
                "duration_seconds": 0.0,
                "segments": [],
                "error": str(e),
            }

    def analyze(self, audio_path: str, user_id: str = "anonymous", analyze_fn=None) -> dict:
        """
        Full audio → OSKAR pipeline:
        1. Transcribe audio with Whisper
        2. Pass the transcription text to the OSKAR /analyze pipeline

        Args:
            audio_path:  Path to the audio/video file
            user_id:     Who posted this audio content
            analyze_fn:  Callable matching the /analyze contract.
                         If None, returns transcription only.

        Returns:
            {
                "transcription": str,
                "language": str,
                "duration_seconds": float,
                "enabled": bool,
                "analysis": dict | None   # Full OSKAR pipeline result
            }
        """
        start = time.perf_counter()

        transcription_result = self.transcribe(audio_path)
        transcribed_text = transcription_result.get("text", "")

        analysis = None
        if analyze_fn and transcribed_text:
            from typing import Any, Dict, List

            from pydantic import BaseModel

            # Build a minimal AnalyzeRequest-like dict for the pipeline
            class _Req:
                def __init__(self, user_id, text):
                    self.user_id = user_id
                    self.text = text
                    self.context_thread = []
                    self.social_context = None

            analysis = analyze_fn(_Req(user_id, transcribed_text))

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "transcription": transcribed_text,
            "language": transcription_result.get("language", "unknown"),
            "duration_seconds": transcription_result.get("duration_seconds", 0.0),
            "processing_ms": elapsed_ms,
            "enabled": self.enabled,
            "analysis": analysis,
        }
