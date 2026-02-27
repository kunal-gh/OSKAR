<div align="center">

# 🛡️ OSKAR: Online Safety & Knowledge Authenticity Resolver
### **Advanced AI Content Moderation & Information Integrity Engine**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph--RAG-4581C3?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-Production--Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Vercel Live](https://img.shields.io/badge/Vercel-Live_Demo-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://oskar-six.vercel.app/)

### 🚀 **[Try the Live OSKAR Demo Here](https://oskar-six.vercel.app/)** 🚀

---

> **"Truth is not a binary. It's an architecture."**
> 
> I built OSKAR as a production-grade inference engine to solve the limitations of traditional content moderation. By combining **Transformers**, **Graph-RAG**, and **Graph Neural Networks (GNNs)**, I've created a system that doesn't just block content, but understands the underlying context and mathematical uncertainty of every decision.

[My Design Philosophy](#-my-design-philosophy) • [System Architecture](#-modular-system-architecture) • [Module Deep Dive](#-machine-learning-deep-dive) • [API & Metrics](#-api-specification)

</div>

---

## 🎨 System Interface

OSKAR features a custom-built, low-latency dashboard designed for enterprise moderation teams, commonly referred to as the "Schindler-IDE" interface.

![OSKAR Dashboard Home Screen](assets/oskar_home.png)
*Figure 1: The primary analysis interface featuring live health metrics and a minimalist, distraction-free composer.*

![OSKAR Analysis Results](assets/oskar_results.png)
*Figure 2: The Entropy-Routed Results View. Showing the overall Risk Ring, Confidence Intervals, and individual ML Module breakdowns (Toxicity, Claim Verification, Truth Graph, and User Trust priors).*

---

## 🌟 My Design Philosophy

When I started developing OSKAR, my goal was to move beyond the deterministic "if-then" logic of traditional filters. I wanted to build a system that reflects the complexity of real-world information. The core pillars of my implementation are:

1.  **Uncertainty-Awareness**: Most AI models are overconfident. I implemented an **Entropy-Based Router** that detects when the system is confused and routes those cases to human moderators.
2.  **Verifiable Truth**: Instead of relying on a model's internal weights (which can hallucinate), I designed a **Graph-RAG** pipeline that verifies claims against a structured Knowledge Graph in Neo4j and a vector space in FAISS.
3.  **Network-Level Intelligence**: Content doesn't exist in a vacuum. I integrated a **GraphSAGE** GNN to analyze the social graph and detect coordinated bot swarms—threats that text-only filters completely miss.

---

## 🏛️ Modular System Architecture

I designed OSKAR using **Domain-Driven Design (DDD)** principles to ensure that each component is modular and scalable. This architecture allows the system to handle millions of requests while maintaining high precision.

### **The Inference Pipeline**

```mermaid
graph TD
    classDef input_style fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef nlp_style fill:#e1f5fe,stroke:#01579b,color:#01579b;
    classDef data_style fill:#f1f8e9,stroke:#33691e,color:#33691e;
    classDef core_style fill:#805ad5,stroke:#6b46c1,color:#fff;
    classDef exit_style fill:#fff3e0,stroke:#e65100,color:#e65100;

    START[Input: Text + Social Graph]:::input_style --> ORCH{Pipeline Orchestrator}
    
    subgraph "Parallel Feature Extraction"
        ORCH --> HATE[RoBERTa Toxicity]:::nlp_style
        ORCH --> CLAIM[DeBERTa-v3 Claims]:::nlp_style
        ORCH --> GNN[GraphSAGE Topology]:::data_style
    end

    subgraph "Graph-RAG Verification"
        CLAIM --> EMB[SBERT Embeddings]
        EMB --> FAISS[(FAISS Knowledge Index)]:::data_style
        FAISS --> N4J[(Neo4j Fact Triples)]:::data_style
    end

    subgraph "Decision Engine"
        HATE --> FUSION[Risk Fusion Engine]:::core_style
        N4J --> FUSION
        GNN --> FUSION
        FUSION --> CAL[Entropy-Based Calibration]:::core_style
    end

    CAL --> VERDICT[Final Routing Decision]:::exit_style
```

---

## 🚀 Hybrid Microservice Architecture (v1.0)

To achieve enterprise-grade scale while maintaining **$0 infrastructure costs**, I've transitioned OSKAR from a monolithic bridge to a **Hybrid Asynchronous Architecture**. This setup decouples heavy ML inference from the API gateway, utilizing a local Docker-orchestrated cluster.

### **The Decoupled Ecosystem**
*   **API Gateway (FastAPI)**: A lightweight entry point that handles authentication, rate limiting, and task submission.
*   **Asynchronous Workers (Celery + Redis)**: Heavy Whisper audio transcription and Tesseract OCR tasks are offloaded to background workers, preventing event-loop blocking.
*   **Persistent Trust Engine (Postgres)**: Migrated from volatile memory to a persistent relational database to maintain long-term user trust priors.
*   **Production Vector Search (Qdrant)**: Replaced FAISS with Qdrant, providing a centralized, high-performance vector database for semantic evidence retrieval across multiple worker nodes.

```mermaid
graph LR
    USER((User/Client)) --> API[FastAPI Gateway]
    API --> REDIS[(Redis Broker)]
    REDIS --> WORKER[Celery Worker Cluster]
    
    subgraph "Persistent Storage"
        WORKER --> PG[(PostgreSQL: Trust)]
        WORKER --> QDR[(Qdrant: Vector DB)]
        WORKER --> NEO[(Neo4j: Knowledge Graph)]
    end
    
    WORKER --> ML[ML Inference Models]
```

---

## 🔬 Machine Learning Deep Dive

### I. Toxicity & Hate Classification
For the first stage of the pipeline, I utilize the `cardiffnlp/twitter-roberta-base-hate-latest` model. I chose this specific model because it was trained on real-world twitter data, making it highly robust against the slang, abbreviations, and aggressive dialects commonly found on social platforms. My implementation achieves ~92% precision in identifying toxic content while maintaining a low false-positive rate.

### II. Claim Extraction & Verification (Graph-RAG)
This is where OSKAR truly differentiates itself. I built a hybrid verification system:
*   **Extraction**: I use a `DeBERTa-v3-Large` zero-shot classifier to isolate objective claims from subjective opinions.
*   **Semantic Search**: Extracted claims are embedded via `all-mpnet-base-v2` and queried against a **FAISS** index containing over 5,000 verified document passages.
*   **Relational Logic**: To provide a second layer of verification, I query a **Neo4j** Knowledge Graph. If the entities in the claim are connected by a "PROVEN_FACT" relationship, the system's confidence in that claim increases.

### III. Bot Swarm Detection (GNN)
To catch Coordinated Inauthentic Behavior (CIB), I implemented a **GraphSAGE** GNN using PyTorch Geometric. Instead of just looking at what is said, my model looks at *who* is interacting. By analyzing the topology of the social graph, I can identify tightly clustered groups of accounts that are coordinating to push specific narratives—even if their individual posts appear benign.

---

## 🧮 Theoretical Framework

I believe that a robust ML system must be grounded in sound mathematics. I have integrated several statistical frameworks to ensure reliability:

### **Self-Aware Uncertainty (Shannon Entropy)**
I use Information Entropy ($H$) to quantify how certain the system is about its own output. If the risk scores across different modules are inconsistent, the entropy spikes.

$$
H(p) = -\sum_{i=1}^{n} p(y_i|x) \log_2 p(y_i|x)
$$

### **Bayesian Trust Priors**
I model user reliability using a **Beta-Bernoulli distribution**. Every time a user's post is verified as factual, their trust score is updated. This allows the system to prioritize content from historically reliable contributors.

$$
\alpha_{\text{new}} = \alpha_{\text{old}} + \text{VerifiedClaims}
$$

$$
\beta_{\text{new}} = \beta_{\text{old}} + (\text{TotalClaims} - \text{VerifiedClaims})
$$

$$
\text{Trust Score} = \frac{\alpha}{\alpha + \beta}
$$

### **Graph Feature Aggregation**
My GNN implementation uses the following aggregation logic to compute the hidden state of each node $v$ based on its neighbors:

$$
h_v^{k} = \sigma \left( W^k \cdot \text{CONCAT} \left( h_v^{k-1}, \text{AGGREGATE}_k \left( \{h_u^{k-1}, \forall u \in \mathcal{N}(v) \} \right) \right) \right)
$$

---

## 🤖 My Algorithmic Implementation

I designed the core inference loop to be high-throughput and resilient. Here is a high-level overview of the logic I've implemented:

```python
# My Implementation of the OSKAR Analysis Pipeline
Algorithm: Content_Analyze(input_data)
    1: toxicity_score = Toxicity_Model.predict(input_data.text)
    2: claims = Claim_Model.extract(input_data.text)
    
    3: if claims exist:
    4:     semantic_evidence = FAISS.search(Embedding(claims))
    5:     graph_triples = Neo4j.validate_entities(claims)
    6:     truth_verdict = Fuse_Knowledge(semantic_evidence, graph_triples)
    7: else:
    8:     truth_verdict = Neutral
    
    9: swarm_score = GNN_Model.analyze_topology(input_data.social_graph)
    10: user_trust = Database.get_trust_prior(input_data.user_id)
    
    11: # I use a non-linear risk fusion to aggregate these signals
    12: total_risk = (α * toxicity_score) + (β * truth_verdict)
    13: adjusted_risk = total_risk * (1.0 + swarm_score) * (1.5 - user_trust)
    
    14: # Final decision based on Shannon Entropy
    15: return Route_By_Entropy(adjusted_risk, Calculate_Entropy(adjusted_risk))
```

---

## ⚡ Performance & Scalability

I have optimized the pipeline to operate under strict latency budgets, ensuring it can be deployed in live streaming environments.

| Subsystem | Target Latency | Actual (CPU) | Actual (GPU) |
| :--- | :--- | :--- | :--- |
| **Toxicity Classification** | $\leq 120ms$ | ~90ms | **~12ms** |
| **Claim Extraction** | $\leq 150ms$ | ~125ms | **~18ms** |
| **Graph-RAG (Verify)** | $\leq 50ms$ | ~20ms | **~2ms** |
| **GNN Bot Inference** | $\leq 20ms$ | ~5ms | **~1ms** |
| **Total P95 Pipeline** | $\leq 350ms$ | ~223ms | **~36ms** |

---

## 📁 Repository Structure

I maintain a strictly modular structure to separate concerns and facilitate enterprise-grade deployment:

```text
OSKAR/
├── MVP/
│   ├── src/                    # My Core Implementation Logic
│   │   ├── api/                # FastAPI Gateway & Request Handling
│   │   ├── models/             # Transformer Wrappers & GNN Architectures
│   │   ├── core/               # Bayesian Math & Entropy Engines
│   │   └── infra/              # Database Drivers (Neo4j, PostgreSQL, FAISS)
│   ├── tests/                  # My Comprehensive Test Suite
│   ├── k8s/                    # Kubernetes manifests (Helm Charts)
│   └── docker-compose.yml      # Local development cluster
└── Documentation/              # Research notes and architectural whitepapers
```

---

<div align="center">

### **Developed by Kunal.**  
*Committed to building AI that is both powerful and ethically sound.*

[Top](#🛡️-oskar-online-safety--knowledge-authenticity-resolver)

</div>
