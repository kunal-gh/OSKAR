import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import fastapi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

# Initialize Prometheus metrics (safe for uvicorn --reload)
from prometheus_client import REGISTRY, Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from src.core.burst_detector import BurstDetector
from src.core.cognitive_engine import CognitiveEngine
from src.core.evidence_retrieval import EvidenceRetrieval
from src.core.gnn_detector import GNNDetector
from src.core.narrative_tracker import NarrativeTracker
from src.core.trust_engine import TrustEngine
from src.core.warning_tracker import WarningTracker
from src.infra.pii_scrubber import PIIScrubber
from src.models.audio_analyzer import AudioAnalyzer
from src.models.claim_classifier import ClaimClassifier
from src.models.drift_detector import DriftDetector
from src.models.hate_classifier import HateClassifier
from src.models.multilingual_adapter import MultilingualAdapter
from src.models.ocr_analyzer import OCRAnalyzer
from src.models.risk_fusion import RiskFusionEngine

try:
    REQUEST_COUNT = Counter(
        "request_count", "App Request Count", ["method", "endpoint", "http_status"]
    )
    REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
except ValueError:
    # Already registered — grab them back from the registry
    REQUEST_COUNT = REGISTRY._names_to_collectors["request_count"]
    REQUEST_LATENCY = REGISTRY._names_to_collectors["request_latency_seconds"]

app = FastAPI(
    title="OSKAR v0.2",
    description="Moderation Decision-Support API — OSKAR 2.0 Ecosystem",
    version="0.2.0",
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add prometheus asgi middleware
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Celery Tasks Import
from src.tasks import async_analyze_text, async_analyze_audio, async_analyze_image
from celery.result import AsyncResult

import threading

models_loaded = threading.Event()

# Global instances
hate_clf = None
claim_clf = None
er = None
cognitive_engine = None
trust_engine = None
risk_engine = None
gnn_detector = None
audio_analyzer = None
ocr_analyzer = None
burst_detector = None
multilingual_adapter = None
narrative_tracker = None
drift_detector = None
pii_scrubber = None


def _bg_load_models():
    global hate_clf, claim_clf, er, cognitive_engine, trust_engine, risk_engine
    global gnn_detector, audio_analyzer, ocr_analyzer, burst_detector
    global multilingual_adapter, narrative_tracker, drift_detector, pii_scrubber

    print("Background ML Load: HATE & CLAIM...")
    hate_clf = HateClassifier()
    claim_clf = ClaimClassifier()

    print("Background ML Load: FAISS & SBERT...")
    er = EvidenceRetrieval(use_neo4j=True)
    er.add_evidence(
        [
            "The CDC explicitly states that vaccines do not cause autism.",
            "The WHO confirms that COVID-19 is an airborne virus.",
            "Global warming is primarily caused by human activities.",
            "The central bank lowered interest rates to 4.5% in September 2024.",
            "The First Amendment protects freedom of speech in the United States.",
        ]
    )

    print("Background ML Load: Engines & Detectors...")
    cognitive_engine = CognitiveEngine(temperature=1.5)
    trust_engine = TrustEngine()
    risk_engine = RiskFusionEngine(num_simulations=100)
    gnn_detector = GNNDetector()
    audio_analyzer = AudioAnalyzer()
    ocr_analyzer = OCRAnalyzer()
    burst_detector = BurstDetector()
    multilingual_adapter = MultilingualAdapter()

    print("Background ML Load: v0.6 Enterprise Modules...")
    narrative_tracker = NarrativeTracker(embedder=er.encoder)
    drift_detector = DriftDetector(embedder=er.encoder)
    pii_scrubber = PIIScrubber(mode="redact")

    print("All ML Models Loaded Successfully. API is fully ready.")
    models_loaded.set()


threading.Thread(target=_bg_load_models, daemon=True).start()


class AnalyzeRequest(BaseModel):
    user_id: str
    text: str
    context_thread: Optional[List[str]] = []
    social_context: Optional[Dict[str, Any]] = None
    temporal_events: Optional[List[Dict[str, Any]]] = None  # v0.4 Burst Detector input


# Security: v0.6 API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = fastapi.security.APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = fastapi.Security(api_key_header)):
    # In production, this would validate against a DB of hashed keys
    # For MVP v0.6, we use an env var or fallback
    expected_key = os.getenv("OSKAR_API_KEY", "REDACTED_USE_ENV_VAR")
    if api_key_header == expected_key:
        return api_key_header
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API Key",
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_content(req: AnalyzeRequest, api_key: str = fastapi.Security(get_api_key)):
    models_loaded.wait()
    start_time = time.time()
    user_hash = req.user_id  # In production, this would be HMAC-SHA256 hashed

    try:
        # Phase 21: PII Scrubbing (Run BEFORE any model processing)
        scrub_res = pii_scrubber.scrub(req.text)
        safe_text = scrub_res["clean_text"]
        pii_metadata = {"pii_found": scrub_res["pii_found"], "redactions": scrub_res["redactions"]}

        # 1. Hate Module
        hate_res = hate_clf.predict(safe_text)

        # 2. Claim Module
        claim_res = claim_clf.predict(safe_text)

        # 3. Verification Module
        if claim_res["is_verifiable"]:
            verify_res = er.verify_claim(safe_text)
        else:
            verify_res = {"verdict": "uncertain", "confidence": 0.0, "evidence": None}

        # Extract misinfo score logic (MVP heuristic):
        # If Supported -> Misinfo=0.0
        # If Refuted -> Misinfo=1.0 (with confidence)
        # If Uncertain -> Misinfo=0.5
        misinfo_score = 0.5
        if verify_res["verdict"] == "refuted":
            misinfo_score = float(verify_res["confidence"])
        elif verify_res["verdict"] == "supported":
            misinfo_score = max(0.0, 1.0 - verify_res["confidence"])

        # 5. OSKAR 2.0 GNNDetector
        bot_score = gnn_detector.predict(req.social_context)

        # 6. v0.4 LSTM Burst Detector — temporal coordination signal
        burst_result = burst_detector.detect(req.temporal_events or [])
        burst_score = burst_result["anomaly_score"]

        # 7. Trust Module
        trust_score = trust_engine.get_user_trust(user_hash)

        # 8. Risk Fusion (now includes burst_score)
        fusion_res = risk_engine.calculate_risk(
            hate_score=(
                hate_res["score"] if hate_res["label"] == "hate" else 1.0 - hate_res["score"]
            ),
            misinfo_score=misinfo_score,
            bot_score=bot_score,
            trust_score=trust_score,
            burst_score=burst_score,
        )

        # 9. v0.6 Narrative Tracking
        narrative_res = narrative_tracker.track(req.text, req.user_id)

        # 10. v0.6 Data Drift Detection
        drift_res = drift_detector.track(req.text)

        REQUEST_COUNT.labels("POST", "/analyze", 200).inc()
        REQUEST_LATENCY.labels("/analyze").observe(time.time() - start_time)

        return {
            "risk_score": fusion_res["mean_risk"],
            "confidence_interval": fusion_res["confidence_interval"],
            "route": fusion_res["route"],
            "components": {
                "hate": hate_res,
                "claim": claim_res,
                "verification": verify_res,
                "bot_swarm": {"probability": bot_score, "enabled": gnn_detector.enabled},
                "burst_detect": {
                    "anomaly_score": burst_score,
                    "is_burst": burst_result["is_burst"],
                },
                "narrative": narrative_res,
                "data_drift": drift_res,
            },
            "trust_score": trust_score,
        }

    except Exception as e:
        REQUEST_COUNT.labels("POST", "/analyze", 500).inc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Audio Endpoint (v0.4 Whisper) ────────────────────────────────
@app.post("/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    user_id: str = Form("anon_audio"),
    api_key: str = fastapi.Security(get_api_key),
):
    models_loaded.wait()
    """
    Whisper audio analysis endpoint. Receives multipart audio, transcribes, 
    and passes text to the main risk pipeline.
    """
    ALLOWED = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format '{suffix}'. Allowed: {', '.join(ALLOWED)}"
        )

    # Save the uploaded file to a temp location for Whisper
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:

        def _run_pipeline(req):
            """Inner pipeline identical to /analyze but accepting a request-like object."""
            hate_res = hate_clf.predict(req.text)
            claim_res = claim_clf.predict(req.text)
            verify_res = (
                er.verify_claim(req.text)
                if claim_res["is_verifiable"]
                else {
                    "verdict": "uncertain",
                    "confidence": 0.0,
                    "evidence": None,
                    "graph_triples": [],
                }
            )
            misinfo = (
                float(verify_res["confidence"])
                if verify_res["verdict"] == "refuted"
                else (
                    max(0.0, 1.0 - verify_res["confidence"])
                    if verify_res["verdict"] == "supported"
                    else 0.5
                )
            )
            bot_score = gnn_detector.predict(req.social_context)
            trust_score = trust_engine.get_user_trust(req.user_id)
            fusion = risk_engine.calculate_risk(
                hate_score=(
                    hate_res["score"] if hate_res["label"] == "hate" else 1.0 - hate_res["score"]
                ),
                misinfo_score=misinfo,
                bot_score=bot_score,
                trust_score=trust_score,
            )
            return {
                "risk_score": fusion["mean_risk"],
                "confidence_interval": fusion["confidence_interval"],
                "route": fusion["route"],
                "components": {
                    "hate": hate_res,
                    "claim": claim_res,
                    "verification": verify_res,
                    "bot_swarm": {"probability": bot_score, "enabled": gnn_detector.enabled},
                },
                "trust_score": trust_score,
            }

        result = audio_analyzer.analyze(tmp_path, user_id=user_id, analyze_fn=_run_pipeline)
        REQUEST_COUNT.labels("POST", "/analyze/audio", 200).inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels("POST", "/analyze/audio", 500).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # Always clean up temp file


# ─── Image OCR Endpoint (v0.4 Tesseract) ──────────────────────────
@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    user_id: str = Form("anon_image"),
    api_key: str = fastapi.Security(get_api_key),
):
    models_loaded.wait()
    """
    Tesseract OCR image analysis endpoint. 
    Multimodal image/meme analysis endpoint.
    Upload any image (png, jpg, jpeg, gif, bmp, tiff, webp).
    Tesseract OCR extracts text, then the full OSKAR pipeline runs on it.
    """
    ALLOWED = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format '{suffix}'. Allowed: {', '.join(ALLOWED)}"
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:

        def _run_pipeline(req):
            hate_res = hate_clf.predict(req.text)
            claim_res = claim_clf.predict(req.text)
            verify_res = (
                er.verify_claim(req.text)
                if claim_res["is_verifiable"]
                else {
                    "verdict": "uncertain",
                    "confidence": 0.0,
                    "evidence": None,
                    "graph_triples": [],
                }
            )
            misinfo = (
                float(verify_res["confidence"])
                if verify_res["verdict"] == "refuted"
                else (
                    max(0.0, 1.0 - verify_res["confidence"])
                    if verify_res["verdict"] == "supported"
                    else 0.5
                )
            )
            bot_score = gnn_detector.predict(req.social_context)
            trust_score = trust_engine.get_user_trust(req.user_id)
            fusion = risk_engine.calculate_risk(
                hate_score=(
                    hate_res["score"] if hate_res["label"] == "hate" else 1.0 - hate_res["score"]
                ),
                misinfo_score=misinfo,
                bot_score=bot_score,
                trust_score=trust_score,
            )
            return {
                "risk_score": fusion["mean_risk"],
                "confidence_interval": fusion["confidence_interval"],
                "route": fusion["route"],
                "components": {
                    "hate": hate_res,
                    "claim": claim_res,
                    "verification": verify_res,
                    "bot_swarm": {"probability": bot_score, "enabled": gnn_detector.enabled},
                },
                "trust_score": trust_score,
            }

        result = ocr_analyzer.analyze(
            tmp_path, user_id=user_id, psm=psm, lang=lang, analyze_fn=_run_pipeline
        )
        REQUEST_COUNT.labels("POST", "/analyze/image", 200).inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels("POST", "/analyze/image", 500).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ─── Multilingual Endpoint (v0.5) ─────────────────────────────────
class MultilingualRequest(BaseModel):
    user_id: str = "anonymous"
    text: str
    source_language: Optional[str] = None  # If None, auto-detect


@app.post("/analyze/multilingual")
def analyze_multilingual(req: AnalyzeRequest, api_key: str = fastapi.Security(get_api_key)):
    models_loaded.wait()
    """
    v0.5 Multilingual Adapter Pipeline
    Native pipeline detects language, translates to English if supported,
    then runs the standard risk evaluation map on the English translation.

    Supported: Hindi, Spanish, Arabic, French, German, Portuguese,
               Russian, Chinese (Simplified/Traditional), Japanese, Korean.
    """

    def _run_pipeline(inner_req):
        hate_res = hate_clf.predict(inner_req.text)
        claim_res = claim_clf.predict(inner_req.text)
        verify_res = (
            er.verify_claim(inner_req.text)
            if claim_res["is_verifiable"]
            else {"verdict": "uncertain", "confidence": 0.0, "evidence": None, "graph_triples": []}
        )
        misinfo = (
            float(verify_res["confidence"])
            if verify_res["verdict"] == "refuted"
            else (
                max(0.0, 1.0 - verify_res["confidence"])
                if verify_res["verdict"] == "supported"
                else 0.5
            )
        )
        bot_score = gnn_detector.predict(inner_req.social_context)
        burst_res = burst_detector.detect(inner_req.temporal_events or [])
        trust_score = trust_engine.get_user_trust(inner_req.user_id)
        fusion = risk_engine.calculate_risk(
            hate_score=(
                hate_res["score"] if hate_res["label"] == "hate" else 1.0 - hate_res["score"]
            ),
            misinfo_score=misinfo,
            bot_score=bot_score,
            trust_score=trust_score,
            burst_score=burst_res["anomaly_score"],
        )
        return {
            "risk_score": fusion["mean_risk"],
            "confidence_interval": fusion["confidence_interval"],
            "route": fusion["route"],
            "components": {
                "hate": hate_res,
                "claim": claim_res,
                "verification": verify_res,
                "bot_swarm": {"probability": bot_score, "enabled": gnn_detector.enabled},
                "burst_detect": {
                    "anomaly_score": burst_res["anomaly_score"],
                    "is_burst": burst_res["is_burst"],
                },
            },
            "trust_score": trust_score,
        }

    try:
        result = multilingual_adapter.process(
            req.text, user_id=req.user_id, analyze_fn=_run_pipeline
        )
        REQUEST_COUNT.labels("POST", "/analyze/multilingual", 200).inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels("POST", "/analyze/multilingual", 500).inc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Warning Efficacy Endpoints (v0.5 A/B) ────────────────────────
warning_tracker = WarningTracker()


class ImpressionRequest(BaseModel):
    user_id: str
    risk_score: float = 0.0
    route: str = "soft_warning"
    content_hash: Optional[str] = None


class FeedbackRequest(BaseModel):
    event_id: str
    action: str  # "ack" | "retraction"


@app.post("/warning/impression")
def log_warning_impression(req: ImpressionRequest, api_key: str = fastapi.Security(get_api_key)):
    """
    Log that a moderation warning was shown to a user.
    Returns the event record including the A/B variant assigned.
    """
    event = warning_tracker.log_impression(
        req.user_id, risk_score=req.risk_score, route=req.route, content_hash=req.content_hash
    )
    return event


@app.post("/warning/feedback")
def log_warning_feedback(req: FeedbackRequest, api_key: str = fastapi.Security(get_api_key)):
    """
    Record a user's response to a moderation warning.
    action: "ack" (dismissed) | "retraction" (deleted/edited post)
    """
    if req.action not in ("ack", "retraction"):
        raise HTTPException(status_code=400, detail="action must be 'ack' or 'retraction'")
    result = warning_tracker.log_feedback(req.event_id, req.action)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/warning/report")
def get_warning_report(api_key: str = fastapi.Security(get_api_key)):
    """Return A/B warning variant efficacy report with CTR and retraction rates."""
    return warning_tracker.get_report()


# ─── Dashboard ────────────────────────────────────
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(DASHBOARD_DIR):

    @app.get("/dashboard")
    def serve_dashboard():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-static")

# ─── Task Management (v1.0 Microservices) ───────────────────────

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str, api_key: str = fastapi.Security(get_api_key)):
    """
    Returns the status and result of a background moderation task.
    """
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "result": result.result if result.ready() else None,
    }


@app.post("/analyze/async")
def analyze_content_async(req: AnalyzeRequest, api_key: str = fastapi.Security(get_api_key)):
    """
    Asynchronous version of /analyze. Returns a task_id immediately.
    """
    task = async_analyze_text.delay(
        req.user_id, req.text, req.social_context, req.temporal_events
    )
    return {"task_id": task.id, "status": "PENDING"}


@app.post("/analyze/audio/async")
async def analyze_audio_async(
    file: UploadFile = File(...),
    user_id: str = Form("anon_audio"),
    api_key: str = fastapi.Security(get_api_key),
):
    """
    Asynchronous version of /analyze/audio. Returns a task_id immediately.
    """
    ALLOWED = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format '{suffix}'. Allowed: {', '.join(ALLOWED)}"
        )

    # Save to a temporary file that the Celery worker can access
    # Note: In production, this would be an S3/GCS bucket URL
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    task = async_analyze_audio.delay(user_id, tmp_path)
    return {"task_id": task.id, "status": "PENDING"}


@app.post("/analyze/image/async")
async def analyze_image_async(
    file: UploadFile = File(...),
    user_id: str = Form("anon_image"),
    api_key: str = fastapi.Security(get_api_key),
):
    """
    Asynchronous version of /analyze/image. Returns a task_id immediately.
    """
    ALLOWED = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format '{suffix}'. Allowed: {', '.join(ALLOWED)}"
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    task = async_analyze_image.delay(user_id, tmp_path)
    return {"task_id": task.id, "status": "PENDING"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
