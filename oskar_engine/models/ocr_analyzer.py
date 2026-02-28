"""
ocr_analyzer.py — OSKAR v0.4 Multimodal Intelligence
------------------------------------------------------
Extracts text from images (memes, screenshots, infographics) using
Tesseract OCR via pytesseract + Pillow. The extracted text is then
piped through the full OSKAR analysis pipeline.

Supported input formats: png, jpg, jpeg, gif, bmp, tiff, webp

Prerequisites:
  1. pip install pytesseract Pillow
  2. Install Tesseract binary:
       Windows: https://github.com/UB-Mannheim/tesseract/wiki
       Linux:   sudo apt install tesseract-ocr
       macOS:   brew install tesseract

If Tesseract is not installed on the system, the module degrades
gracefully — OCR returns empty text and analysis continues safely.

Usage:
    from ocr_analyzer import OCRAnalyzer
    ocr = OCRAnalyzer()
    result = ocr.analyze("path/to/meme.png", user_id="user_123")
"""

import os
import time
from typing import Callable, Optional

# Configurable Tesseract binary path (Windows only)
# Set env var TESSERACT_CMD if not on PATH
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)

# Image pre-processing modes for better OCR accuracy
# 6 = Assume a single uniform block of text (good for memes)
# 3 = Fully automatic page segmentation (good for screenshots)
DEFAULT_PSM = 6


class OCRAnalyzer:
    """
    Wraps pytesseract + Pillow for local OCR image text extraction.

    If Tesseract is not installed, degrades gracefully:
    - extract_text returns an empty string
    - analysis returns minimal schema without crashing
    """

    def __init__(self):
        self.enabled = False
        self._configure()

    def _configure(self):
        try:
            import pytesseract
            from PIL import Image  # noqa: F401 — verify Pillow is present

            if TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            # Quick sanity check — will raise if Tesseract binary is missing
            pytesseract.get_tesseract_version()
            self.enabled = True
            print("[OCRAnalyzer] Tesseract ready.")
        except Exception as e:
            print(f"[OCRAnalyzer] Tesseract not available: {e}. Running in dummy mode.")

    def extract_text(
        self, image_path: str, psm: int = DEFAULT_PSM, lang: str = "eng"
    ) -> dict:
        """
        Extract text from an image file using Tesseract OCR.

        Args:
            image_path:  Path to the image file
            psm:         Page-segmentation mode (6 for memes, 3 for screenshots)
            lang:        Tesseract language code (e.g. "eng", "hin", "spa")

        Returns:
            {
                "text":         str,        # Raw extracted text
                "word_count":   int,
                "confidence":   float,      # mean per-word confidence (0-100)
                "language":     str,
                "enabled":      bool
            }
        """
        if not self.enabled:
            return {
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "language": lang,
                "enabled": False,
                "error": "Tesseract not installed",
            }

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            import pytesseract
            from PIL import Image, ImageFilter, ImageOps

            img = Image.open(image_path)

            # Pre-processing for better OCR accuracy on memes:
            # Convert to grayscale → sharpen → increase contrast
            img = ImageOps.grayscale(img)
            img = img.filter(ImageFilter.SHARPEN)

            config = f"--psm {psm}"
            raw_text = pytesseract.image_to_string(img, lang=lang, config=config)

            # Get confidence data
            data = pytesseract.image_to_data(
                img, lang=lang, config=config, output_type=pytesseract.Output.DICT
            )
            confidences = [
                c for c in data["conf"] if isinstance(c, (int, float)) and c > 0
            ]
            mean_conf = (
                round(sum(confidences) / len(confidences), 1) if confidences else 0.0
            )

            cleaned = raw_text.strip()
            words = cleaned.split()

            return {
                "text": cleaned,
                "word_count": len(words),
                "confidence": mean_conf,
                "language": lang,
                "enabled": True,
            }

        except Exception as e:
            return {
                "text": "",
                "word_count": 0,
                "confidence": 0.0,
                "language": lang,
                "enabled": self.enabled,
                "error": str(e),
            }

    def analyze(
        self,
        image_path: str,
        user_id: str = "anonymous",
        psm: int = DEFAULT_PSM,
        lang: str = "eng",
        analyze_fn: Optional[Callable] = None,
    ) -> dict:
        """
        Full image → OSKAR pipeline.
        1. Extract text with Tesseract OCR
        2. Pass extracted text through OSKAR /analyze pipeline

        Returns:
            {
                "extracted_text":  str,
                "word_count":      int,
                "ocr_confidence":  float,
                "language":        str,
                "processing_ms":   float,
                "enabled":         bool,
                "analysis":        dict | None
            }
        """
        start = time.perf_counter()

        ocr_result = self.extract_text(image_path, psm=psm, lang=lang)
        extracted_text = ocr_result.get("text", "")

        analysis = None
        if analyze_fn and extracted_text.strip():

            class _Req:
                def __init__(self, user_id, text):
                    self.user_id = user_id
                    self.text = text
                    self.context_thread = []
                    self.social_context = None

            analysis = analyze_fn(_Req(user_id, extracted_text))

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "extracted_text": extracted_text,
            "word_count": ocr_result.get("word_count", 0),
            "ocr_confidence": ocr_result.get("confidence", 0.0),
            "language": ocr_result.get("language", lang),
            "processing_ms": elapsed_ms,
            "enabled": self.enabled,
            "analysis": analysis,
        }
