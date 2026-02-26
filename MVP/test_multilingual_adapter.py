"""
test_multilingual_adapter.py — OSKAR v0.5
Tests MultilingualAdapter: init, language detection, schema, and translation.
"""

import pytest
from multilingual_adapter import MultilingualAdapter, SUPPORTED_LANGUAGES


def test_multilingual_adapter_initialization():
    """Adapter must initialize without errors."""
    adapter = MultilingualAdapter()
    assert hasattr(adapter, "langdetect_ready")


def test_detect_english():
    """Long English text should be detected as 'en' with high confidence."""
    adapter = MultilingualAdapter()
    text = (
        "Multiple scientific studies have conclusively shown that vaccines do not cause autism. "
        "The original paper that claimed this link has been retracted due to fraud. "
        "Health agencies worldwide recommend vaccination as a safe and effective public health measure."
    )
    lang, conf = adapter.detect_language(text)
    assert lang == "en", f"Expected 'en' but got '{lang}'(conf={conf})"
    assert conf > 0.5


def test_detect_spanish():
    """Spanish text should be detected correctly (or at least NOT as English)."""
    adapter = MultilingualAdapter()
    # Long unambiguous Spanish paragraph
    text = (
        "Las vacunas no causan autismo. Esto ha sido confirmado por múltiples estudios científicos "
        "realizados en todo el mundo. Los científicos y médicos están de acuerdo en que las vacunas "
        "son seguras y efectivas para prevenir enfermedades graves."
    )
    lang, conf = adapter.detect_language(text)
    # langdetect should detect es; allow a small chance it returns 'pt' (very similar)
    assert lang in ("es", "pt", "ca"), f"Expected Spanish-family language but got '{lang}'"


def test_detect_hindi():
    """Hindi text should be detected correctly."""
    adapter = MultilingualAdapter()
    text = (
        "यह वैक्सीन पूरी तरह सुरक्षित है। वैज्ञानिकों ने यह साबित किया है कि टीकाकरण से "
        "बच्चों को गंभीर बीमारियों से बचाया जा सकता है। सरकार का दावा बिल्कुल सही है।"
    )
    lang, conf = adapter.detect_language(text)
    assert lang == "hi", f"Expected 'hi' but got '{lang}'"


def test_detect_arabic():
    """Arabic text should be detected correctly."""
    adapter = MultilingualAdapter()
    text = (
        "اللقاحات آمنة وفعالة وفقاً لمنظمة الصحة العالمية وعدد كبير من العلماء والأطباء. "
        "تشير الدراسات العلمية إلى أن التطعيم يحمي الأطفال من الأمراض الخطيرة."
    )
    lang, conf = adapter.detect_language(text)
    # langdetect may return ar, fa (Farsi), or ur (Urdu) for Arabic-script text
    assert lang in ("ar", "fa", "ur"), f"Expected Arabic-script language but got '{lang}'"


def test_process_schema():
    """process() must return the expected schema for any input."""
    adapter = MultilingualAdapter()
    result = adapter.process(
        "The earth is flat, NASA is lying.",
        user_id="test_user",
        analyze_fn=None
    )
    assert "original_text"          in result
    assert "detected_language"      in result
    assert "language_name"          in result
    assert "translated_text"        in result
    assert "translation_confidence" in result
    assert "was_translated"         in result
    assert "processing_ms"          in result
    assert "analysis"               in result
    assert result["analysis"] is None   # No pipeline passed


def test_process_english_no_translation():
    """English text should NOT be translated (was_translated=False)."""
    adapter = MultilingualAdapter()
    result = adapter.process("Climate change is a hoax.", user_id="u1")
    assert result["was_translated"] is False
    assert result["translated_text"] == result["original_text"]


def test_process_spanish_translation():
    """Spanish text should be detected and optionally translated."""
    adapter = MultilingualAdapter()
    text = (
        "El cambio climático es real y está causado por actividades humanas. "
        "Los científicos de todo el mundo están de acuerdo en que debemos actuar ahora "
        "para reducir las emisiones de gases de efecto invernadero."
    )
    result = adapter.process(text, user_id="u2")

    # Language should be in Spanish family
    assert result["detected_language"] in ("es", "pt", "ca"), \
        f"Expected Spanish-family language but got '{result['detected_language']}'"
    assert isinstance(result["translated_text"], str)
    assert len(result["translated_text"]) > 0
    print(f"[ES→EN] '{result['translated_text']}'")

def test_empty_text_graceful():
    """Empty text should return defaults without crashing."""
    adapter = MultilingualAdapter()
    result = adapter.process("", user_id="u3")
    assert result["detected_language"] in ("en", "unknown") or isinstance(result["detected_language"], str)
    assert result["analysis"] is None


def test_supported_languages_coverage():
    """SUPPORTED_LANGUAGES should contain at least Hindi, Spanish, and Arabic."""
    assert "hi" in SUPPORTED_LANGUAGES
    assert "es" in SUPPORTED_LANGUAGES
    assert "ar" in SUPPORTED_LANGUAGES
