# OSKAR: Cloud Scaling Roadmap (Path B - True Production)

*If you are reading this, congratulations! You have secured the funding to host OSKAR at true enterprise scale. This document outlines the architecture and estimated costs for a highly-available, public internet deployment.*

---

## 🏗️ Target Scale Architecture

When transitioning from the local Docker/CI testing environment to a public cloud, OSKAR must decouple its monolithic inference pipeline. Currently, FastAPI loads the multi-GB Transformer models directly into memory. At scale, this causes fatal event-loop blocking.

The true cloud architecture involves three isolated layers:

### 1. The Gateway Layer (FastAPI)
*   **Role**: Handles authentication, rate-limiting, and synchronous I/O.
*   **Hosting**: AWS ECS (Fargate) or Google Cloud Run.
*   **Cost**: ~$20 - $50 / month (scales to zero).

### 2. The Model Serving Layer (Triton Inference Server)
*   **Role**: Hosts the heavy PyTorch/ONNX models (RoBERTa, DeBERTa, Whisper). It batches incoming requests dynamically to maximize GPU throughput.
*   **Hosting**: AWS EKS or RunPod Clusters with dedicated NVIDIA A10G or L4 GPUs.
*   **Cost**: ~$150 - $400 / month per node.

### 3. The Asynchronous Worker Layer (Celery)
*   **Role**: Processes long-running tasks like audio transcription and large OCR batch jobs to prevent the Gateway API from timing out.
*   **Hosting**: AWS EC2 (t3.medium) pulling from an SQS queue or Redis.
*   **Cost**: ~$30 - $60 / month.

### 4. Database & Knowledge Graph Layer
*   **Role**: Stores persistent Trust scores and verified vectors.
*   **Hosting**: 
    *   **PostgreSQL** (AWS RDS): ~$30 / month.
    *   **Vector DB** (Managed Pinecone or Milvus Serverless): ~$70 / month.
    *   **Graph DB** (Neo4j AuraDB Professional): ~$65 / month.

---

## 💰 Total Estimated Live Production Cost

To keep OSKAR running 24/7 on the internet, capable of processing hundreds of requests per second:

| Service | Estimated Monthly Cost | 
| :--- | :--- | 
| API Gateway (Fargate) | $30 | 
| Triton GPU Node (A10G) | $250 | 
| Redis Cache / Broker | $20 | 
| Managed Vector DB | $70 | 
| Managed Postgres | $30 | 
| **Total Baseline Cost** | **~$400 / Month** | 

---

## 🚀 Migration Steps (From Local to Cloud)

When the funds are available, execute this migration plan:

1.  **Extract Models**: Export the local PyTorch models to ONNX or TensorRT format.
2.  **Deploy Triton**: Spin up a RunPod or AWS EC2 instance with the NVIDIA Triton Docker image. Mount the ONNX models.
3.  **Refactor FastAPI**: Rewrite `src/api/main.py` to strip out the model initialization logic. Instead, use the `tritonclient` gRPC library to send text features to the Triton IP address.
4.  **Migrate DBs**: Export the local Docker Postgres and Neo4j data using `pg_dump` and Neo4j export tools. Import the data into the managed RDS and AuraDB instances.
5.  **Provision K8s**: Use Terraform (or the provided Helm charts in `MVP/k8s/`) to spin up the EKS cluster and bind the domain.

---
*Note: Until this migration happens, continue to use the Zero-Cost Hybrid approach: run tests via GitHub Actions and execute the heavy models exclusively on your local workstation's Docker daemon.*
