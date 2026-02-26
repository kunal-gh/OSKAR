<div align="center">

# 🛡️ OSKAR: Online Safety & Knowledge Authenticity Resolver
### **The Next Generation of AI Content Moderation & Information Integrity**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph--RAG-4581C3.svg?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-Production--Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

> **"Truth is not a binary. It's an architecture."**
> 
> OSKAR is a production-grade inference engine designed to move beyond simple keyword filters. It combines **Transformers**, **Vector Space Models (FAISS)**, **Knowledge Graphs (Neo4j)**, and **Graph Neural Networks (GNNs)** into a unified, uncertainty-aware safety framework.

[Modular Architecture](#-modular-system-architecture) • [Core ML Modules](#-machine-learning-deep-dive) • [Graph-RAG](#-graph-rag--verification) • [Bot Swarm GNN](#-gnn-coordinated-behavior-analysis) • [API & Metrics](#-api-specification)

</div>

---

## 🏛️ Modular System Architecture

OSKAR follows a **Domain-Driven Design (DDD)** pattern, separating core inference from infrastructure and API management. This allows for horizontal scaling of individual ML pods.

### High-Level Data Flow Orchestration

```mermaid
graph LR
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef nlp fill:#e1f5fe,stroke:#01579b,color:#01579b;
    classDef graph fill:#f1f8e9,stroke:#33691e,color:#33691e;
    classDef output fill:#fff3e0,stroke:#e65100,color:#e65100;

    REQ[Analyze Request]:::input --> INP{Pre-Processor}
    
    subgraph "Feature Extraction Layer"
        INP --> HATE[RoBERTa Hate]:::nlp
        INP --> CLAIM[DeBERTa-v3 Claims]:::nlp
        INP --> GNN[GraphSAGE Topology]:::graph
    end

    subgraph "Verification Layer (Graph-RAG)"
        CLAIM --> SBERT[SBERT Embedding]
        SBERT --> FAISS[(FAISS Vector DB)]:::graph
        FAISS --> CROSS[Cross-Encoder Re-rank]
        CROSS --> N4J[(Neo4j Knowledge Graph)]:::graph
    end

    subgraph "Decision Engine"
        HATE --> FUSION[Risk Fusion Engine]
        N4J --> FUSION
        GNN --> FUSION
        FUSION --> TEMP[Temperature Scaling]
        TEMP --> ENTROPY[Entropy-based Router]
    end

    ENTROPY --> ACTION[Final Verdict]:::output
```

---

## 🔬 Machine Learning Deep Dive

### 1. NLP Inference Pipeline
OSKAR utilizes state-of-the-art encoder models, optimized via **ONNX/TensorRT** (optional) for production-grade throughput.

*   **Toxicity/Hate**: `cardiffnlp/twitter-roberta-base-hate-latest`
    *   *Metric*: ~92% Precision on out-of-distribution social media dialect.
*   **Zero-Shot Claim Extraction**: `MoritzLaurer/deberta-v3-large-zeroshot-v2`
    *   **The Problem**: Identifying what is a "verifiable fact" vs. an "subjective opinion".
    *   **The Solution**: We treat claim detection as a Natural Language Inference (NLI) task. Claims are categorized into `scientific`, `historical`, `statistical`, or `opinion` with an **84% Macro F1**.

### 2. Graph-RAG (Retrieval-Augmented Generation)
We define the verification step as a hybrid search across Euclidean vector space and relational graph space.

#### **Algorithm: Dual-Stream Verification**
1.  **Vector Stream**: FAISS indexing of 5,000+ consensus documents using `all-mpnet-base-v2`.
2.  **Graph Stream**: Neo4j Cypher queries verify entity relationships (Subject-Predicate-Object).
3.  **Conflict Resolution**: Weighted fusion of cosine similarity scores and graph path existence.

### 3. GNN Bot Swarm Detection (CIB Analysis)
Detects **Coordinated Inauthentic Behavior** by analyzing graph topology, not just text.

*   **Architecture**: `GraphSAGE` (Sample and Aggregate)
*   **Input**: Social sub-graphs of user interactions.
*   **Intuition**: Bots coordinate to amplify narratives. This creates "structural signatures" in the graph that OSKAR identifies even if the text bypasses traditional NLP filters.

---

## 📈 Technical Specifications & Performance

### Tech Stack Matrix

| Layer | Component | Version | Role |
| :--- | :--- | :--- | :--- |
| **API** | FastAPI | 0.109+ | High-concurrency async gateway |
| **Inference** | PyTorch / Transformers | 2.2 / 4.37 | Neural computation engine |
| **GNN** | PyTorch Geometric | 2.5 | Social graph topology analysis |
| **Vector DB** | FAISS | 1.7.4 | Semantic evidence retrieval |
| **Graph DB** | Neo4j | 5.18 | Knowledge Graph facts storage |
| **Caching** | Redis alpine | 7.0 | Semantic cache (In-Memory) |
| **Metrics** | Prometheus | 2.49 | Real-time p99 latency monitoring |

### Performance SLA (Standard Latency Budget)

```text
Pipeline Phase        | Target (ms) | Measured CPU | Measured GPU (A100)
-----------------------------------------------------------------------
Input Validation      | < 5ms       | 1ms          | < 1ms
Hate NLP Inference    | < 120ms     | 94ms         | 14ms
Claim NLP Inference   | < 150ms     | 112ms        | 18ms
FAISS Search          | < 10ms       | 2ms          | < 1ms
GNN Topology Check    | < 30ms       | 6ms          | 2ms
Risk Fusion & Entropy | < 5ms        | 1ms          | < 1ms
-----------------------------------------------------------------------
Total E2E Pipeline    | < 350ms     | ~216ms       | ~36ms
```

---

## 🧮 Mathematical Formalism

### Uncertainty & Entropy Routing
We use Information Entropy ($H$) to determine the system's "Self-Awareness."

$$ H(p) = -\sum_{i=1}^{n} p(y_i|x) \log_2 p(y_i|x) $$

*   **Low Entropy ($H < 0.6$)**: The system is confident → **Auto Action**.
*   **Medium Entropy ($0.6 < H < 0.8$)**: Ambiguous case → **Soft Warning**.
*   **High Entropy ($H > 0.8$)**: System is confused → **Escalate to Human**.

### Bayesian Trust Scoring
User trust is modeled as a **Beta-Bernoulli distribution**, updated recurrently.

$$ \alpha_{new} = \alpha_{old} + \text{verified\_claims} $$
$$ \beta_{new} = \beta_{old} + (\text{total\_claims} - \text{verified\_claims}) $$
$$ \text{Trust Score} = \frac{\alpha}{\alpha + \beta} $$

---

## 📁 Repository Structure (Professional modular architecture)

```text
OSKAR/
├── MVP/
│   ├── src/                    # CORE SOURCE CODE
│   │   ├── api/                # FastAPI Gateway & API Logic
│   │   │   ├── main.py         # Entry point & Pipeline Orchestrator
│   │   │   └── auth_manager.py
│   │   ├── models/             # ML Model Wrappers (Transformers, GNN)
│   │   │   ├── hate_classifier.py
│   │   │   └── gnn_detector.py
│   │   ├── core/               # Math & Decision Engines
│   │   │   ├── cognitive_engine.py
│   │   │   └── risk_fusion.py
│   │   └── infra/              # Database & Graph Drivers
│   │       ├── redis_cache.py
│   │       └── neo4j_knowledge_graph.py
│   ├── tests/                  # Pytest Unit & Integration Suite
│   ├── k8s/                    # Helm Charts & K8s Manifests
│   ├── docker-compose.yml      # Local Cluster Definition
│   └── requirements.txt        # Enterprise-locked dependencies
└── Documentation/              # Research Papers & Whitepapers
```

---

## 🛣️ Project Roadmap

### ✅ v0.3 — Intelligence Expansion (Current Release)
- [x] **Modular Refactor**: Clean `src/` modularization for production readiness.
- [x] **GNN Integration**: `GraphSAGE` for coordinated bot detection.
- [x] **Graph-RAG**: Integrated Neo4j + FAISS verification pipeline.
- [x] **High-Fidelity Models**: Switched to `DeBERTa-v3` for >80% F1 claim accuracy.

### 🔜 v0.4 — Multimodal Capabilities
- [ ] **Whisper V3 Integration**: Real-time audio transcription and threat analysis.
- [ ] **OCR Layer**: Tesseract-based meme and image-text moderation.
- [ ] **Temporal Analysis**: LSTM/Autoencoder for time-series burst detection.

### 🔜 v1.0 — Platform Scalability
- [ ] **K8s Auto-scaling**: Dynamic pod scaling based on Prometheus inference latency.
- [ ] **RBAC API**: Granular access control for enterprise moderators.
- [ ] **Model AB Testing**: Shadow mode deployment for canary model verification.

---

<div align="center">

### **Ready to Protect the Integrity of Online Information.**

**Developer**: Kunal | **Architecture**: Domain-Driven ML | **Mission**: Ethical AI

[Top](#🛡️-oskar-online-safety--knowledge-authenticity-resolver)

</div>
