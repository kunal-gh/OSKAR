"""
test_ocr_analyzer.py — OSKAR v0.4
Tests OCRAnalyzer initialization, schema, and graceful Tesseract fallback.
"""

import os
import tempfile
import pytest
from PIL import Image, ImageDraw, ImageFont
from ocr_analyzer import OCRAnalyzer


def _make_text_image(path: str, text: str = "Vaccines cause autism"):
    """Create a minimal PNG with text for OCR testing."""
    img = Image.new("RGB", (400, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), text, fill=(0, 0, 0))
    img.save(path)


def test_ocr_analyzer_initialization():
    """OCRAnalyzer should initialize without exceptions."""
    ocr = OCRAnalyzer()
    assert hasattr(ocr, "enabled")


def test_ocr_analyzer_missing_file():
    """Should raise FileNotFoundError for a non-existent image path."""
    ocr = OCRAnalyzer()
    if not ocr.enabled:
        pytest.skip("Tesseract not installed, skipping active OCR test.")
    with pytest.raises(FileNotFoundError):
        ocr.extract_text("/nonexistent/image.png")


def test_ocr_analyzer_schema_dummy_mode():
    """
    In dummy mode (Tesseract not installed), analyze() should still
    return the correct schema keys with empty/default values.
    """
    ocr = OCRAnalyzer()

    if ocr.enabled:
        # If Tesseract IS installed, test with a real image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_text_image(tmp_path, "Hello world")
            result = ocr.analyze(tmp_path, user_id="test_user", analyze_fn=None)
        finally:
            os.unlink(tmp_path)
    else:
        # Dummy mode - still returns proper schema
        result = ocr.analyze("/fake/image.png", user_id="test_user")

    assert "extracted_text" in result
    assert "word_count" in result
    assert "ocr_confidence" in result
    assert "language" in result
    assert "processing_ms" in result
    assert "enabled" in result
    assert "analysis" in result
    assert result["analysis"] is None  # No pipeline passed


def test_ocr_analyzer_extracted_text_type():
    """extracted_text must always be a string with proper types."""
    ocr = OCRAnalyzer()
    if ocr.enabled:
        # Use a real temp image when Tesseract is available
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_text_image(tmp_path, "test text")
            result = ocr.analyze(tmp_path, user_id="test")
        finally:
            os.unlink(tmp_path)
    else:
        result = ocr.analyze("/fake/path.png", user_id="test")
    assert isinstance(result["extracted_text"], str)
    assert isinstance(result["word_count"], int)
    assert isinstance(result["processing_ms"], float)


def test_ocr_analyzer_known_text():
    """If Tesseract is available, OCR should extract recognizable text from a clean image."""
    ocr = OCRAnalyzer()
    if not ocr.enabled:
        pytest.skip("Tesseract not installed, skipping text extraction test.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        test_text = "vaccines cause autism"
        _make_text_image(tmp_path, test_text)
        result = ocr.extract_text(tmp_path)

        # At least some recognizable words should appear
        extracted = result["text"].lower()
        matches = sum(1 for word in test_text.split() if word in extracted)
        assert matches >= 1, f"OCR missed too many words. Got: '{result['text']}'"
    finally:
        os.unlink(tmp_path)
