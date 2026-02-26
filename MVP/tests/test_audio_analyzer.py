"""
test_audio_analyzer.py — OSKAR v0.4
Tests AudioAnalyzer initialization, dummy mode, and schema.
"""

import os
import struct
import wave
import tempfile
import pytest
from src.models.audio_analyzer import AudioAnalyzer


def _make_tiny_wav(path: str, duration_s: float = 0.1, sample_rate: int = 16000):
    """Create a minimal valid WAV file for testing without external libraries."""
    num_samples = int(sample_rate * duration_s)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silence (all zeros)
        wf.writeframes(b'\x00\x00' * num_samples)


def test_audio_analyzer_initialization():
    """AudioAnalyzer should initialize without errors."""
    analyzer = AudioAnalyzer()
    assert hasattr(analyzer, "enabled")
    assert hasattr(analyzer, "model_name")


def test_audio_analyzer_missing_file():
    """Should raise FileNotFoundError for a non-existent path."""
    analyzer = AudioAnalyzer()
    if not analyzer.enabled:
        pytest.skip("Whisper not installed, skipping active transcription test.")
    with pytest.raises(FileNotFoundError):
        analyzer.transcribe("/nonexistent/path/file.wav")


def test_audio_analyzer_schema_no_pipeline():
    """
    analyze() with no analyze_fn should return correct schema
    with analysis=None (transcription-only mode).
    Uses a real WAV file if whisper is available, skips otherwise.
    """
    analyzer = AudioAnalyzer()

    if not analyzer.enabled:
        # Dummy mode: result should still have correct keys with empty transcription
        result = analyzer.analyze("/fake/path.wav", user_id="test_user")
        assert "transcription" in result
        assert "language" in result
        assert "duration_seconds" in result
        assert "processing_ms" in result
        assert "enabled" in result
        assert result["enabled"] is False
        assert result["analysis"] is None
        return

    # Whisper available: use a real silent WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _make_tiny_wav(tmp_path)
        result = analyzer.analyze(tmp_path, user_id="test_user", analyze_fn=None)

        assert isinstance(result["transcription"], str)
        assert isinstance(result["language"], str)
        assert isinstance(result["duration_seconds"], float)
        assert isinstance(result["processing_ms"], float)
        assert result["enabled"] is True
        assert result["analysis"] is None  # No pipeline was passed
    finally:
        os.unlink(tmp_path)


def test_audio_analyzer_latency_transcribe():
    """Transcription of a short silent WAV should complete within 30s."""
    import time
    analyzer = AudioAnalyzer()

    if not analyzer.enabled:
        pytest.skip("Whisper not installed, skipping latency test.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _make_tiny_wav(tmp_path, duration_s=0.5)
        start = time.perf_counter()
        result = analyzer.transcribe(tmp_path)
        elapsed = time.perf_counter() - start
        print(f"[Whisper] Transcription latency: {elapsed*1000:.0f}ms")
        assert elapsed < 30  # Generous initial budget (includes model warmup)
        assert "text" in result
        assert "language" in result
    finally:
        os.unlink(tmp_path)
