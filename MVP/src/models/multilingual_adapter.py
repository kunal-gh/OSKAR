"""
multilingual_adapter.py — OSKAR v0.5 Platform Layer
-----------------------------------------------------
Detects the language of any incoming text and translates it to English
before piping it through the full OSKAR analysis pipeline.

Supported translation pairs (Helsinki-NLP MarianMT):
  * Hindi   (hi) → English
  * Spanish (es) → English
  * Arabic  (ar) → English
  * French  (fr) → English
  * German  (de) → English
  * Portuguese (pt) → English
  * Russian (ru) → English
  * Chinese (zh) → English (Simplified)
  * Japanese (ja) → English
  * Korean  (ko) → English

If the detected language is already English, text passes through unchanged.
If a translation model is not available for the detected language, the
original text is passed through with a warning.

Language detection uses `langdetect` (port of Google's language-detection library).
Translation uses HuggingFace `Helsinki-NLP/opus-mt-{src}-en` models, downloaded
on first use and cached automatically.

Usage:
    from multilingual_adapter import MultilingualAdapter
    adapter = MultilingualAdapter()

    result = adapter.process(
        "यह वैक्सीन खतरनाक है और सरकार झूठ बोल रही है।",
        user_id="user_123"
    )
    # result = {
    #     "original_text": "यह वैक्सीन खतरनाक...",
    #     "detected_language": "hi",
    #     "language_name": "Hindi",
    #     "translated_text": "This vaccine is dangerous and the government is lying.",
    #     "translation_confidence": 0.99,
    #     "was_translated": True,
    #     "analysis": { ...full OSKAR pipeline output... }
    # }
"""

import os
import time
from typing import Callable, Optional

# Supported language codes → display name + Helsinki-NLP model suffix
SUPPORTED_LANGUAGES = {
    "hi": ("Hindi", "Helsinki-NLP/opus-mt-hi-en"),
    "es": ("Spanish", "Helsinki-NLP/opus-mt-es-en"),
    "ar": ("Arabic", "Helsinki-NLP/opus-mt-ar-en"),
    "fr": ("French", "Helsinki-NLP/opus-mt-fr-en"),
    "de": ("German", "Helsinki-NLP/opus-mt-de-en"),
    "pt": ("Portuguese", "Helsinki-NLP/opus-mt-ROMANCE-en"),
    "ru": ("Russian", "Helsinki-NLP/opus-mt-ru-en"),
    "zh-cn": ("Chinese", "Helsinki-NLP/opus-mt-zh-en"),
    "zh-tw": ("Chinese", "Helsinki-NLP/opus-mt-zh-en"),
    "ja": ("Japanese", "Helsinki-NLP/opus-mt-ja-en"),
    "ko": ("Korean", "Helsinki-NLP/opus-mt-ko-en"),
}


class MultilingualAdapter:
    """
    Language-aware text normalizer for OSKAR.

    All translation models are lazy-loaded (downloaded on first use per language).
    The adapter caches loaded models in memory to avoid reloading within a session.
    """

    def __init__(self):
        self._translators: dict = {}  # lang_code → (tokenizer, model)
        self.langdetect_ready = False
        self._init_langdetect()

    def _init_langdetect(self):
        try:
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 42  # Deterministic results
            self.langdetect_ready = True
            print("[MultilingualAdapter] langdetect ready.")
        except ImportError:
            print("[MultilingualAdapter] langdetect not installed. Language detection disabled.")

    def detect_language(self, text: str) -> tuple[str, float]:
        """
        Detect the language of the given text.

        Returns:
            (lang_code: str, confidence: float)
            e.g. ("hi", 0.99) or ("en", 0.99) for English
        """
        if not self.langdetect_ready or not text.strip():
            return ("en", 1.0)

        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                top = results[0]
                return (top.lang, round(top.prob, 4))
            return ("en", 1.0)
        except Exception:
            return ("en", 1.0)

    def _get_translator(self, lang_code: str):
        """
        Lazy-load and cache a Helsinki-NLP MarianMT translation pipeline
        for the given language code.

        Returns:
            (tokenizer, model) tuple, or None if not supported.
        """
        if lang_code in self._translators:
            return self._translators[lang_code]

        if lang_code not in SUPPORTED_LANGUAGES:
            return None

        _, model_name = SUPPORTED_LANGUAGES[lang_code]
        try:
            from transformers import MarianMTModel, MarianTokenizer

            print(f"[MultilingualAdapter] Loading translation model '{model_name}'...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self._translators[lang_code] = (tokenizer, model)
            print(f"[MultilingualAdapter] '{model_name}' ready.")
            return (tokenizer, model)
        except Exception as e:
            print(f"[MultilingualAdapter] Could not load model '{model_name}': {e}")
            return None

    def translate(self, text: str, lang_code: str) -> str:
        """
        Translate text from lang_code → English.
        Returns the original text if translation fails or language is unsupported.
        """
        if lang_code == "en" or not text.strip():
            return text

        translator = self._get_translator(lang_code)
        if translator is None:
            return text

        tokenizer, model = translator
        try:
            inputs = tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            outputs = model.generate(**inputs, max_new_tokens=512)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            print(f"[MultilingualAdapter] Translation failed: {e}")
            return text

    def process(
        self, text: str, user_id: str = "anonymous", analyze_fn: Optional[Callable] = None
    ) -> dict:
        """
        Full multilingual → OSKAR pipeline:
        1. Detect language
        2. Translate to English if needed
        3. Run OSKAR analysis pipeline on English text

        Returns:
            {
                "original_text":          str,
                "detected_language":      str,   # ISO 639-1 code
                "language_name":          str,   # Human-readable, or "Unknown"
                "translated_text":        str,   # Same as original if already English
                "translation_confidence": float,
                "was_translated":         bool,
                "processing_ms":          float,
                "analysis":               dict | None
            }
        """
        start = time.perf_counter()

        lang_code, confidence = self.detect_language(text)

        # Normalize zh-cn / zh-tw
        lookup_code = "zh-cn" if lang_code.startswith("zh") else lang_code

        lang_info = SUPPORTED_LANGUAGES.get(lookup_code)
        language_name = (
            lang_info[0] if lang_info else ("English" if lang_code == "en" else "Unknown")
        )

        was_translated = False
        if lang_code == "en":
            translated_text = text
        else:
            translated_text = self.translate(text, lookup_code)
            was_translated = translated_text != text

        analysis = None
        if analyze_fn and translated_text.strip():

            class _Req:
                def __init__(self, uid, t):
                    self.user_id = uid
                    self.text = t
                    self.context_thread = []
                    self.social_context = None
                    self.temporal_events = None

            analysis = analyze_fn(_Req(user_id, translated_text))

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "original_text": text,
            "detected_language": lang_code,
            "language_name": language_name,
            "translated_text": translated_text,
            "translation_confidence": confidence,
            "was_translated": was_translated,
            "processing_ms": elapsed_ms,
            "analysis": analysis,
        }
