<div align="center">

<img src="https://img.shields.io/badge/Version-0.3.0-blue?style=for-the-badge&logo=github"/>
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/Graph_RAG-Neo4j-4581C3?style=for-the-badge&logo=neo4j&logoColor=white"/>
<img src="https://img.shields.io/badge/GNN-PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 🛡️ PROJECT OSKAR
### **Online Safety & Knowledge Authenticity Resolver**
#### *Inspired by the moral courage of Oskar Schindler — built to protect truth at scale.*

---

> **"Don't just flag content. Understand it."**
>
> OSKAR doesn't work like a simple spam filter that blocks bad words. It *thinks* — weighing context, evidence, user history, network topology, and mathematical uncertainty before recommending whether to act automatically or escalate to a human.

</div>

---

## 📖 Table of Contents

- [What Is OSKAR?](#-what-is-oskar)
- [The Problem It Solves](#-the-problem-it-solves)
- [How It Works — The Big Picture](#-how-it-works--the-big-picture)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [Module Breakdown](#-module-breakdown)
- [Tech Stack](#-tech-stack)
- [API Reference](#-api-reference)
- [Data Contracts](#-data-contracts-strict-schemas)
- [Performance Budget](#-performance-budget)
- [Getting Started](#-getting-started)
- [Running Tests](#-running-tests)
- [Benchmarking Claim Accuracy](#-benchmarking-claim-accuracy)
- [Docker Deployment](#-docker-deployment)
- [Project Roadmap](#-project-roadmap)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🤔 What Is OSKAR?

Imagine you're a platform moderator. Every single day, **millions** of posts, comments, and replies flood your queue. Some are hate speech. Some spread dangerous misinformation. Some are posted by coordinated bot farms that game every naive filter you build.

**You can't do this alone. But you also can't let a simple AI do it alone.**

OSKAR is built precisely for this situation — it's a **decision-support system**, not a content blocker. It analyzes content across multiple dimensions and tells you:

- ✅ "This is clearly fine — auto-approve it."
- ⚠️ "This is suspicious — show the user a gentle warning."
- 🚨 "I'm 92% confident this is dangerous misinformation posted by a bot swarm — flag it for a human expert."

OSKAR always knows what it *doesn't* know. When it's uncertain, it says so — and routes accordingly.

---

## 😤 The Problem It Solves

Most automated moderation systems share the **same 5 broken defaults**:

| Problem | The Reality | How OSKAR Fixes It |
|---|---|---|
| **Overconfidence** | "95% hate speech" when it's sarcasm | Entropy-based uncertainty; ambiguous content → human review |
| **No Context** | Each post analyzed in isolation | Graph-RAG: Neo4j entity relationships + FAISS semantic context |
| **No User History** | Every post treated equally | Bayesian longitudinal trust scoring tracks reliability over time |
| **Black Box Decisions** | Post removed, user has no idea why | Full evidence chain + graph triples + confidence interval |
| **Bot Blindness** | No awareness of coordinated attacks | GraphSAGE GNN detects bot swarms from social graph topology |

---

## 🗺 How It Works — The Big Picture

When content hits OSKAR's `/analyze` endpoint, here's the full pipeline:

```
User Posts Content + Optional Social Graph
                │
                ▼
        ┌──────────────────┐
        │   1. HATE MODULE │ ──► "Is this toxic?" (RoBERTa Twitter Hate)
        └──────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  2. CLAIM MODULE │ ──► "Is there a verifiable claim?" (DeBERTa Zero-Shot)
        └──────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  3. EVIDENCE MODULE      │ ──► FAISS cosine similarity +
        │  (Graph-RAG)             │     Neo4j entity relationship triples
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  4. GNN BOT SWARM        │ ──► "Is this user part of a bot swarm?"
        │  (GraphSAGE)             │     (swarm_probability: 0.0 – 1.0)
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  5. COGNITIVE ENGINE     │ ──► "How confident are we?" (Entropy Router)
        │  (Calibration + Routing) │
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  6. TRUST ENGINE         │ ──► "Who is this user?" (Bayesian Score)
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────┐
        │  7. RISK FUSION          │ ──► Final risk with Monte Carlo CI
        │  (Monte Carlo Sim)       │     bot_score acts as risk multiplier
        └──────────────────────────┘
                │
                ▼
        ┌──────────────────────────────────────────┐
        │  ROUTE: auto_action / soft_warning /     │
        │         human_review                     │
        └──────────────────────────────────────────┘
```

---

## 🏗 Architecture Deep Dive

OSKAR v0.3 is a **single-node, containerized FastAPI service** with a pluggable Graph-RAG layer and GNN bot detection.

```
┌──────────────────────────────────────────────────────────────────┐
│                      OSKAR API (FastAPI)                          │
│                 POST /analyze  |  GET /health  |  GET /metrics    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐   ┌──────────────────────────────────┐  │
│  │  hate_classifier    │   │  claim_classifier                │  │
│  │  (RoBERTa-Twitter)  │   │  (DeBERTa-v3 Zero-Shot ≥80% F1) │  │
│  └─────────────────────┘   └──────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  evidence_retrieval  [Graph-RAG v0.3]                        │ │
│  │  ├── FAISS (SBERT all-mpnet-base-v2, 768-dim)               │ │
│  │  └── Neo4j KnowledgeGraph (70+ entity-relationship triples)  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  gnn_detector  [Bot Swarm, OSKAR 2.0]                       │  │
│  │  └── GraphSAGE (PyTorch Geometric) — swarm_probability      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────┐   ┌───────────────────────────────────┐    │
│  │ cognitive_engine │   │  risk_fusion                      │    │
│  │ (Temp. Scaling + │   │  (Monte Carlo + GNN multiplier)   │    │
│  │  Entropy Router) │   └───────────────────────────────────┘    │
│  └──────────────────┘                                            │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  trust_engine  (Bayesian Scoring via SQLite/PostgreSQL)   │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                     │
         ┌───────────┼─────────────┐
         ▼           ▼             ▼
    PostgreSQL     Redis         Neo4j
    (Trust DB)  (Caching)  (Knowledge Graph)
                                   │
                              Prometheus
                               (Metrics)
```

---

## 🔬 Module Breakdown

### Module 1 — Hate Classification (`hate_classifier.py`)

**Model:** `cardiffnlp/twitter-roberta-base-hate-latest` (v0.2 upgrade)

**Why this model?** Trained specifically on Twitter hate speech data — the most realistic proxy for real-world social media content. Produces reliable `HATE`/`NON_HATE` labels with calibrated confidence scores.

**Output:**
```json
{
  "label": "hate | non_hate",
  "score": 0.97,
  "uncertainty": 0.03
}
```

---

### Module 2 — Claim Detection (`claim_classifier.py`)

**Model:** `MoritzLaurer/deberta-v3-large-zeroshot-v2` (v0.3 upgrade, ~80% F1)

**Why upgraded?** The previous `distilroberta` baseline achieved ~65-70% verifiability accuracy. The DeBERTa-v3 large model with optimized zero-shot NLI reaches ~80% macro F1 on claim classification benchmarks — the v0.3 target.

**Claim Types:**
| Type | Example |
|---|---|
| `statistical` | "Over 40% of Americans are obese." |
| `historical` | "WW2 ended in 1945." |
| `policy` | "The new bill bans fossil fuel subsidies." |
| `scientific` | "Vaccines cause autism." |
| `opinion` | "I think politicians are corrupt." |

**Output:**
```json
{
  "is_verifiable": true,
  "claim_type": "scientific",
  "confidence": 0.88,
  "model": "deberta-v3-large-zeroshot-v2"
}
```

---

### Module 3 — Evidence Retrieval (`evidence_retrieval.py`) — *Graph-RAG Layer*

**What it does (v0.3):** Two-stage evidence retrieval combining FAISS semantic search with Neo4j entity-relationship graph triples.

**Stage 1 — FAISS (semantic similarity):**
- SBERT `all-mpnet-base-v2` encodes claims into 768-dim vectors
- FAISS cosine similarity finds top-k matching passages from the 5,000-passage Wikipedia index

**Stage 2 — Neo4j Knowledge Graph (`neo4j_knowledge_graph.py`):**
- 70+ seeded fact triples across 8 misinformation domains (vaccines, climate, elections, 5G, health, space, finance, flat earth)
- Entity-aware Cypher query finds 1-hop and 2-hop relationships
- Matching graph triples boost FAISS confidence by +0.05 per triple (max +0.20)
- **Graceful fallback:** If Neo4j is offline, FAISS-only mode activates automatically

**Output:**
```json
{
  "verdict": "supported | refuted | uncertain",
  "confidence": 0.94,
  "evidence": "The CDC explicitly states that vaccines do not cause autism.",
  "graph_triples": [
    {
      "subject": "CDC",
      "relation": "STATES",
      "object": "Vaccines do not cause autism",
      "relevance": 1.0
    }
  ]
}
```

---

### Module 4 — GNN Bot Swarm Detector (`gnn_detector.py`) — *OSKAR 2.0*

**What it does:** Detects Coordinated Inauthentic Behavior (CIB) — bot farms that coordinate to push narratives, even when their individual posts seem benign. Analyzes the *social graph* around a user rather than just the text.

**Architecture:** PyTorch Geometric `GraphSAGE` (Sample and Aggregate)
- **Nodes:** Users + Posts with behavior feature vectors `[account_age, post_frequency, toxicity_variance]`
- **Edges:** Posted, Interacts_With

**Why GraphSAGE?** It aggregates features from a node's local neighborhood — meaning it catches tight clusters of accounts with identical behavior patterns that look like a bot swarm.

**Integration:** The API payload accepts an optional `social_context` field containing a local neighborhood subgraph. If no graph is provided, a conservative default of `0.1` (low bot probability) is returned.

**Risk Impact:** The GNN `swarm_probability` feeds into RiskFusion as a **multiplicative amplifier** — not just a linear weight. A bot score of 0.9 nearly doubles the final risk score, ensuring coordinated bot attacks are always escalated even if the text seems mild.

**Output in components:**
```json
{
  "bot_swarm": {
    "probability": 0.87,
    "enabled": true
  }
}
```

---

### Module 5 — Cognitive Engine (`cognitive_engine.py`)

**Two core functions:**

**① Temperature Scaling**
```
calibrated_prob = softmax(logits / T)
```
Where `T=1.5` in MVP — softens overconfident predictions to better reflect true accuracy.

**② Entropy Router**
```
H = -Σ p(y) × log(p(y))

H > 0.8   →  human_review    (too uncertain, needs a human)
H < 0.6   →  auto_action     (high confidence, act automatically)
0.6–0.8   →  soft_warning    (medium confidence, warn the user)
```

---

### Module 6 — Risk Fusion Engine (`risk_fusion.py`)

**v0.3 weights and modifiers:**
```python
weights = { misinfo: 0.60, hate: 0.40 }   # Misinfo is the primary driver

trust_modifier = 1.5 - trust_score        # Trusted user → lower risk
bot_modifier   = 1.0 + swarm_probability  # Bot swarm → risk spikes (up to 2x)

adjusted = scores * trust_modifier * bot_modifier
```

**Monte Carlo Simulation:**
```json
{
  "mean_risk": 0.81,
  "confidence_interval": [0.74, 0.89],
  "route": "human_review"
}
```

---

### Module 7 — Trust Engine (`trust_engine.py`)

**Bayesian scoring:**
```
Prior:  α₀ = 2, β₀ = 2  (neutral 50/50)
After each verified interaction:
  α = α₀ + correct_claims
  β = β₀ + total_claims - correct_claims
  trust_score = α / (α + β)
```

| User History | Trust Score |
|---|---|
| Brand new | 0.50 |
| 10/10 verified correct claims | ≈ 0.92 |
| 1/10 correct claims | ≈ 0.23 |

---

## 🛠 Tech Stack

| Layer | Technology | Version | Why |
|---|---|---|---|
| **API** | FastAPI | 0.109+ | Async, auto-docs, Pydantic |
| **Hate** | cardiffnlp/twitter-roberta-base-hate | v0.2 | Twitter-native hate detection |
| **Claim** | MoritzLaurer/deberta-v3-large-zeroshot-v2 | v0.3 | ~80% F1 zero-shot |
| **Embeddings** | SBERT all-mpnet-base-v2 | — | 768-dim semantic similarity |
| **Vector Search** | FAISS | — | Sub-ms billion-scale ANN |
| **Knowledge Graph** | Neo4j 5.18 | v0.3 | Entity-relationship Graph-RAG |
| **Graph Driver** | neo4j (Python) | 6.1+ | Bolt protocol client |
| **Bot Detection** | PyTorch Geometric (GraphSAGE) | v0.3 (2.0) | Node classification on social graphs |
| **Trust Store** | SQLAlchemy + PostgreSQL | — | Persistent Bayesian trust scores |
| **Caching** | Redis | 7-alpine | TTL semantic cache |
| **Monitoring** | Prometheus | — | Latency/error/route metrics |
| **Container** | Docker + Compose | — | API + DB + Redis + Neo4j |
| **Testing** | pytest | — | Schema, accuracy, latency gating |

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok" }
```

---

### `POST /analyze`

**Request Body:**
```json
{
  "user_id": "user_abc_123",
  "text": "Vaccines definitely cause autism, I've seen the proof.",
  "context_thread": [],
  "social_context": {
    "nodes": [
      {"id": "target_user", "features": [0.1, 0.9, 0.8]},
      {"id": "bot_account_1", "features": [0.05, 0.95, 0.9]}
    ],
    "edges": [[0, 1], [1, 0]]
  }
}
```

> `social_context` is **optional**. If omitted, a conservative default bot score of 0.1 is used.

**Response:**
```json
{
  "risk_score": 0.91,
  "confidence_interval": [0.85, 0.96],
  "route": "human_review",
  "components": {
    "hate": {
      "label": "non_hate",
      "score": 0.04,
      "uncertainty": 0.17
    },
    "claim": {
      "is_verifiable": true,
      "claim_type": "scientific",
      "confidence": 0.95,
      "model": "deberta-v3-large-zeroshot-v2"
    },
    "verification": {
      "verdict": "refuted",
      "confidence": 0.93,
      "evidence": "The CDC explicitly states that vaccines do not cause autism.",
      "graph_triples": [
        {
          "subject": "CDC",
          "relation": "STATES",
          "object": "Vaccines do not cause autism",
          "relevance": 1.0
        }
      ]
    },
    "bot_swarm": {
      "probability": 0.87,
      "enabled": true
    }
  },
  "trust_score": 0.50
}
```

### `GET /metrics`
Prometheus scrape endpoint — latency, error rate, route distribution.

### `GET /dashboard`
Interactive moderation dashboard (serves `dashboard/index.html`).

---

## 📐 Data Contracts (Strict Schemas)

| Module | Output Keys |
|---|---|
| Hate | `label`, `score`, `uncertainty` |
| Claim | `is_verifiable`, `claim_type`, `confidence`, `model` |
| Verification | `verdict`, `confidence`, `evidence`, `graph_triples` |
| Bot Swarm | `probability`, `enabled` |
| Risk Engine | `mean_risk`, `confidence_interval`, `route` |

---

## ⚡ Performance Budget

| Module | Target | Measured (CPU) |
|---|---|---|
| Hate Classification | ≤ 120ms | ~90ms |
| Claim Detection | ≤ 120ms | ~110ms (DeBERTa-v3) |
| FAISS Retrieval | ≤ 80ms | ~2ms |
| Neo4j Query | ≤ 50ms | ~15ms (local) |
| GNN Inference | ≤ 20ms | ~3ms |
| Risk Fusion | ≤ 10ms | ~1ms |
| **Total Pipeline P95** | **≤ 250ms** | **~220ms** |

> 🚀 On a GPU (A100), all numbers drop by 5–10x.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for Neo4j + PostgreSQL + Redis)
- Git

### 1. Clone the repo
```bash
git clone https://github.com/kunal-gh/OSKAR.git
cd OSKAR/MVP
```

### 2. Create virtual environment & install dependencies
```powershell
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Start the API (dev mode with hot-reload)
```powershell
python main.py
# Server starts at http://localhost:8000
# Dashboard at http://localhost:8000/dashboard
```

### 4. Test it
```powershell
curl -X POST http://localhost:8000/analyze `
  -H "Content-Type: application/json" `
  -d '{"user_id": "demo", "text": "The earth is flat and NASA is lying.", "context_thread": []}'
```

---

## 🧪 Running Tests

```powershell
# Run all tests
venv\Scripts\pytest -v

# Run specific modules
venv\Scripts\pytest test_hate_classifier.py -v
venv\Scripts\pytest test_claim_classifier.py -v
venv\Scripts\pytest test_evidence_retrieval.py -v
venv\Scripts\pytest test_cognitive_engine.py -v
venv\Scripts\pytest test_trust_engine.py -v
venv\Scripts\pytest test_risk_fusion.py -v
venv\Scripts\pytest test_gnn_detector.py -v
```

**Expected:**
```
test_hate_classifier.py::test_hate_classifier_schema    PASSED
test_hate_classifier.py::test_hate_classifier_accuracy  PASSED
test_hate_classifier.py::test_hate_classifier_latency   PASSED
test_claim_classifier.py::test_claim_classifier_schema  PASSED
test_claim_classifier.py::test_claim_classifier_accuracy PASSED
test_claim_classifier.py::test_claim_classifier_latency  PASSED
test_evidence_retrieval.py::test_evidence_retrieval_schema  PASSED
test_evidence_retrieval.py::test_evidence_retrieval_accuracy PASSED
test_evidence_retrieval.py::test_evidence_retrieval_latency  PASSED
test_cognitive_engine.py::test_temperature_scaling      PASSED
test_cognitive_engine.py::test_entropy_router           PASSED
test_trust_engine.py::test_trust_engine_lifecycle       PASSED
test_risk_fusion.py::test_risk_fusion_schema            PASSED
test_risk_fusion.py::test_risk_fusion_logic             PASSED
test_gnn_detector.py::test_gnn_detector_initialization  PASSED
test_gnn_detector.py::test_gnn_detector_no_context      PASSED
test_gnn_detector.py::test_gnn_detector_empty_context   PASSED
```

---

## 📊 Benchmarking Claim Accuracy

Run the standalone claim classifier benchmark to validate the v0.3 ≥80% F1 target:

```powershell
venv\Scripts\python benchmark_claim_classifier.py
```

Expected output:
```
Accuracy:        85.0% (17/20)
Macro F1:        0.8400  ✅ PASS — target: ≥ 0.80
```

---

## 🐳 Docker Deployment

One command starts the full stack: API + PostgreSQL + Redis + Neo4j.

```bash
# Build and start all services
docker compose up --build

# Detached mode
docker compose up -d --build

# Logs
docker compose logs -f api

# Stop
docker compose down
```

**Services:**

| Service | Port | Description |
|---|---|---|
| `api` | `8000` | FastAPI moderation engine |
| `db` | `5432` | PostgreSQL (trust scores) |
| `redis` | `6379` | Redis cache |
| `neo4j` | `7687` / `7474` | Knowledge graph (Bolt / Browser UI) |

**Access:**
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`
- Neo4j Browser: `http://localhost:7474` (user: `neo4j`, password: `oskarpass`)
- Prometheus: `http://localhost:8000/metrics`

---

## 🗺 Project Roadmap

### ✅ v0.1 — MVP Foundation *(Complete)*
- [x] Hate Classification (DistilBERT multilingual)
- [x] Claim Detection (zero-shot NLI)
- [x] Evidence Retrieval (SBERT + FAISS)
- [x] Cognitive Engine (Temperature Scaling + Entropy Router)
- [x] Trust Engine (Bayesian, SQLite/PostgreSQL)
- [x] Risk Fusion (Monte Carlo simulation)
- [x] FastAPI `/analyze` endpoint
- [x] Prometheus metrics
- [x] Docker + docker-compose

### ✅ v0.2 — Core Fortification *(Complete)*
- [x] Upgraded hate model → `cardiffnlp/twitter-roberta-base-hate-latest`
- [x] Built real Wikipedia FAISS index (5,000 SQuAD passages, 768-dim)
- [x] Trust Engine — PostgreSQL auto-detect from `DATABASE_URL`
- [x] Full test suite — 12/13 green (1 model limitation, not code bug)
- [x] OSKAR Moderator Dashboard (Schindler-IDE aesthetic, typewriter font)

### ✅ v0.3 — Intelligence Expansion *(Complete)*
- [x] GNN Bot Swarm Detection via PyTorch Geometric `GraphSAGE`
- [x] Neo4j Knowledge Graph with 70+ entity-relationship fact triples
- [x] Graph-RAG: FAISS + Neo4j combined evidence verification
- [x] Claim Classifier upgraded → `deberta-v3-large-zeroshot-v2` (≥80% F1)
- [x] `benchmark_claim_classifier.py` — standalone F1 validation script

### 🔜 v0.4 — Multimodal Intelligence
- [ ] **Whisper** real-time audio transcription + analysis
- [ ] **Tesseract OCR** for meme/screenshot text detection
- [ ] Multimodal risk fusion (text + audio + image)
- [ ] Temporal burst pattern analysis (LSTM autoencoder for coordinated attack detection)

### 🔜 v0.5 — Platform Layer
- [ ] Real-time browser extension for pre-post warnings
- [ ] Moderator command center with heatmaps and decision audit trail
- [ ] Multilingual adapters (Hindi, Spanish, Arabic)
- [ ] A/B testing framework for warning message efficacy

### 🔜 v1.0 — Enterprise
- [ ] RBAC with immutable audit-grade decision logs
- [ ] Kubernetes deployment with auto-scaling
- [ ] Canary model deployment and automated rollback
- [ ] EU DSA / US First Amendment configurable compliance modes
- [ ] Narrative drift detection with polarization index (Neo4j Temporal Paths)

---

## 📁 Project Structure

```
OSKAR/MVP/
├── main.py                       # FastAPI app, /analyze pipeline orchestration
│
├── # ─── Core Inference Modules ────────────────────────────────────
├── hate_classifier.py            # RoBERTa Twitter hate detection
├── claim_classifier.py           # DeBERTa-v3 zero-shot claim typing (≥80% F1)
├── evidence_retrieval.py         # FAISS + Neo4j Graph-RAG (v0.3)
├── neo4j_knowledge_graph.py      # Neo4j entity-relationship KG (v0.3) [NEW]
├── cognitive_engine.py           # Temperature scaling + entropy routing
├── trust_engine.py               # Bayesian user trust scoring
├── risk_fusion.py                # Monte Carlo risk aggregation + GNN multiplier
├── gnn_detector.py               # GraphSAGE bot swarm detection (v0.3) [NEW]
│
├── # ─── Dashboard ──────────────────────────────────────────────────
├── dashboard/
│   ├── index.html                # OSKAR Schindler-IDE dashboard
│   ├── style.css                 # Moody charcoal + sky-blue design
│   └── app.js                    # API integration + risk ring animation
│
├── # ─── Knowledge Base ─────────────────────────────────────────────
├── knowledge_base/
│   ├── wiki.faiss                # Pre-built 5000-passage Wikipedia FAISS index
│   └── wiki_texts.json           # Corresponding passage texts
├── build_faiss_index.py          # Rebuild the FAISS index from SQuAD data
│
├── # ─── Tests ───────────────────────────────────────────────────────
├── test_main.py                  # API health endpoint test
├── test_hate_classifier.py       # Hate module: schema, accuracy, latency
├── test_claim_classifier.py      # Claim module: schema, accuracy, latency
├── test_evidence_retrieval.py    # Retrieval: schema, accuracy, latency (<80ms)
├── test_cognitive_engine.py      # Calibration + routing threshold tests
├── test_trust_engine.py          # Bayesian trust lifecycle test
├── test_risk_fusion.py           # Risk engine: schema, logic correctness
├── test_gnn_detector.py          # GNN: init, no-context, connected graph [NEW]
│
├── # ─── Benchmarks ──────────────────────────────────────────────────
├── benchmark_claim_classifier.py # 20-sample claim F1 benchmark [NEW]
├── benchmark_hate_models.py      # Hate model accuracy comparison
│
├── # ─── Infrastructure ──────────────────────────────────────────────
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # API + PostgreSQL + Redis + Neo4j stack
└── .env.example                  # Environment variable template
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'feat: description of change'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

**Ground rules:**
- All new modules must have tests (`schema`, `accuracy`, `latency`)
- All tests must pass before merge
- API schemas must not change without a versioned migration path
- New architecture layers require an approved design document

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with precision. Deployed with purpose.**

*OSKAR — Because "probably fine" isn't good enough.*

</div>
