# OSKAR Advanced Enterprise Roadmap (Phases 4-6)

**Vision:** To evolve OSKAR from a powerful local asynchronous microservice cluster into a true, real-time, planet-scale Content Moderation Engine capable of processing continuous data streams (video, audio, text) from decentralized platforms.

---

## Phase 4: Sub-Second Real-Time Infrastructure (WebSockets & Streaming)

Currently, OSKAR relies on polling (`/tasks/{task_id}`) to retrieve asynchronous results from Celery workers. To reach true enterprise level, the system must process data in real-time.

### Goal 1: WebSocket Integration
- **Objective:** Establish bi-directional WebSocket connections between the API Gateway and the Dashboard.
- **Implementation:** Replace the frontend polling logic in `app.js` with establishing a WebSocket connection. When a Celery worker completes inference, the API will aggressively push the exact JSON payload via the WebSocket, dropping the client wait time by ~800ms.
- **Tools:** `fastapi.websockets`, Redis Pub/Sub.

### Goal 2: Live Audio / Video Moderation Stream
- **Objective:** The ability to moderate an ongoing livestream (e.g., Twitch, Discord voice channels) without waiting for the stream to end.
- **Implementation:** Implement chunked audio ingestion. The API will accept continuous WebRTC audio streams, chunk them into 5-second buffers, and parallelize them through the Whisper/RoBERTa workers to produce a live, rolling "Risk Score" graph on the UI.

---

## Phase 5: Advanced Intelligence & Multilingual Context

The current NLP pipeline is highly effective for English text. To be a global solution, OSKAR must understand nuance across dialects and languages.

### Goal 1: Multi-Agent LLM Orchestration (LangChain/LlamaIndex)
- **Objective:** Replace the static zero-shot `DeBERTa` verification classifier with an active team of LLM Agents (e.g., Llama 3 or Mixtral running locally via Ollama).
- **Implementation:**
  - **Agent 1 (Extractor):** Reads the social graph and post, extracting core entities.
  - **Agent 2 (Researcher):** Actively queries the Neo4j Knowledge Graph and external trusted news APIs (Reuters, AP) in real-time.
  - **Agent 3 (Judge):** Synthesizes the exact proof and outputs a highly detailed, human-readable "Explainable AI" paragraph justifying the risk score.

### Goal 2: Multilayered Multilingual NLP
- **Objective:** Seamlessly moderate Hate and Misinfo across Spanish, Hindi, Arabic, etc.
- **Implementation:** Insert a fast, local translation model (e.g., Meta's `NLLB-200`) explicitly before the inference router, converting all incoming text to a unified embedded space, allowing the core English RoBERTa model to evaluate nuance without training 50 parallel language models.

---

## Phase 6: Autonomous Remediation & Webhooks

A true moderation engine doesn't just calculate scores; it takes action.

### Goal 1: Webhook Auto-Remediation System
- **Objective:** Allow external platforms (Discord, Reddit clones) to plug directly into OSKAR.
- **Implementation:** If OSKAR's `Risk_Fusion_Engine` calculates an Entropy-calibrated Risk Score > 95% (Hate Speech) or detects a GraphSAGE Bot Swarm, OSKAR will automatically fire an outbound HTTP POST (Webhook) to the host platform, instantly deleting the post or shadow-banning the user cluster without human intervention.
- **Tools:** `httpx`, Webhook payload standardization.

### Goal 2: Advanced Dashboard Analytics (Grafana)
- **Objective:** Give the Trust & Safety team high-level monitoring.
- **Implementation:** Deploy a Grafana container linked to an InfluxDB or Prometheus instance that scrapes the FastAPI `/metrics` endpoint. The Dashboard will feature a global map showing localized spikes in Misinformation topics in real-time.
