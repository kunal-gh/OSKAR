import os
import time
from typing import Any, Dict, List, Optional
from src.celery_app import celery_app

# ML Model Imports
from src.core.burst_detector import BurstDetector
from src.core.cognitive_engine import CognitiveEngine
from src.core.evidence_retrieval import EvidenceRetrieval
from src.core.gnn_detector import GNNDetector
from src.core.narrative_tracker import NarrativeTracker
from src.core.trust_engine import TrustEngine
from src.infra.pii_scrubber import PIIScrubber
from src.models.audio_analyzer import AudioAnalyzer
from src.models.claim_classifier import ClaimClassifier
from src.models.drift_detector import DriftDetector
from src.models.hate_classifier import HateClassifier
from src.models.multilingual_adapter import MultilingualAdapter
from src.models.ocr_analyzer import OCRAnalyzer
from src.models.risk_fusion import RiskFusionEngine

class OSKARDispatcher:
    """
    Singleton-like class to manage ML model instances within a Celery worker.
    """
    def __init__(self):
        print("[Celery Worker] Initializing ML Models...")
        self.hate_clf = HateClassifier()
        self.claim_clf = ClaimClassifier()
        self.er = EvidenceRetrieval(use_neo4j=True)
        # Seed evidence
        self.er.add_evidence([
            "The CDC explicitly states that vaccines do not cause autism.",
            "The WHO confirms that COVID-19 is an airborne virus.",
            "Global warming is primarily caused by human activities.",
            "The central bank lowered interest rates to 4.5% in September 2024.",
            "The First Amendment protects freedom of speech in the United States."
        ])
        
        self.cognitive_engine = CognitiveEngine(temperature=1.5)
        self.trust_engine = TrustEngine()
        self.risk_engine = RiskFusionEngine(num_simulations=100)
        self.gnn_detector = GNNDetector()
        self.audio_analyzer = AudioAnalyzer()
        self.ocr_analyzer = OCRAnalyzer()
        self.burst_detector = BurstDetector()
        self.multilingual_adapter = MultilingualAdapter()
        
        self.narrative_tracker = NarrativeTracker(embedder=self.er.encoder)
        self.drift_detector = DriftDetector(embedder=self.er.encoder)
        self.pii_scrubber = PIIScrubber(mode="redact")
        print("[Celery Worker] All models loaded and ready.")

    def run_pipeline(self, text: str, user_id: str, social_context: Optional[Dict] = None, temporal_events: Optional[List] = None) -> Dict:
        start_time = time.time()
        
        # 1. PII Scrubbing
        scrub_res = self.pii_scrubber.scrub(text)
        safe_text = scrub_res["clean_text"]
        
        # 2. Hate Module
        hate_res = self.hate_clf.predict(safe_text)
        
        # 3. Claim Module
        claim_res = self.claim_clf.predict(safe_text)
        
        # 4. Verification Module
        if claim_res["is_verifiable"]:
            verify_res = self.er.verify_claim(safe_text)
        else:
            verify_res = {"verdict": "uncertain", "confidence": 0.0, "evidence": None}
            
        misinfo_score = 0.5
        if verify_res["verdict"] == "refuted":
            misinfo_score = float(verify_res["confidence"])
        elif verify_res["verdict"] == "supported":
            misinfo_score = max(0.0, 1.0 - verify_res["confidence"])
            
        # 5. GNN Detector
        bot_score = self.gnn_detector.predict(social_context)
        
        # 6. Burst Detector
        burst_result = self.burst_detector.detect(temporal_events or [])
        burst_score = burst_result["anomaly_score"]
        
        # 7. Trust Module
        trust_score = self.trust_engine.get_user_trust(user_id)
        
        # 8. Risk Fusion
        fusion_res = self.risk_engine.calculate_risk(
            hate_score=(hate_res["score"] if hate_res["label"] == "hate" else 1.0 - hate_res["score"]),
            misinfo_score=misinfo_score,
            bot_score=bot_score,
            trust_score=trust_score,
            burst_score=burst_score
        )
        
        # 9. Narrative Tracking & Drift
        narrative_res = self.narrative_tracker.track(text, user_id)
        drift_res = self.drift_detector.track(text)
        
        return {
            "risk_score": fusion_res["mean_risk"],
            "confidence_interval": fusion_res["confidence_interval"],
            "route": fusion_res["route"],
            "components": {
                "hate": hate_res,
                "claim": claim_res,
                "verification": verify_res,
                "bot_swarm": {"probability": bot_score, "enabled": self.gnn_detector.enabled},
                "burst_detect": {"anomaly_score": burst_score, "is_burst": burst_result["is_burst"]},
                "narrative": narrative_res,
                "data_drift": drift_res
            },
            "trust_score": trust_score,
            "processing_ms": round((time.time() - start_time) * 1000, 2)
        }

# Initialize the global worker instance
# Note: In a real production environment, we might use a factory or lazy loading
_worker = None

def get_worker():
    global _worker
    if _worker is None:
        _worker = OSKARDispatcher()
    return _worker

@celery_app.task(name="src.tasks.async_analyze_text")
def async_analyze_text(user_id: str, text: str, social_context: Optional[Dict] = None, temporal_events: Optional[List] = None):
    worker = get_worker()
    return worker.run_pipeline(text, user_id, social_context, temporal_events)

@celery_app.task(name="src.tasks.async_analyze_audio")
def async_analyze_audio(user_id: str, audio_path: str):
    worker = get_worker()
    # Transcription is heavy
    audio_res = worker.audio_analyzer.transcribe(audio_path)
    text = audio_res.get("text", "")
    
    analysis = None
    if text:
        analysis = worker.run_pipeline(text, user_id)
    
    return {
        "transcription": text,
        "language": audio_res.get("language"),
        "duration_seconds": audio_res.get("duration_seconds"),
        "analysis": analysis
    }

@celery_app.task(name="src.tasks.async_analyze_image")
def async_analyze_image(user_id: str, image_path: str):
    worker = get_worker()
    # OCR is heavy
    ocr_res = worker.ocr_analyzer.extract_text(image_path)
    text = ocr_res.get("text", "")
    
    analysis = None
    if text:
        analysis = worker.run_pipeline(text, user_id)
        
    return {
        "extracted_text": text,
        "ocr_confidence": ocr_res.get("confidence"),
        "analysis": analysis
    }
