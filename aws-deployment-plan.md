# Phoenice-PolyRAG — AWS Deployment Plan (Full-Feature Edition)

## Problem Statement
Deploy the full Phoenice-PolyRAG stack (FastAPI backend + React frontend) on AWS with ALL of the
following enabled: all 4 server vector stores (Qdrant + Weaviate + Milvus + PGVector), SPLADE sparse
retrieval, and LLM-powered features (query rewriting, HyDE, multi-query). Budget target was ≤ $10/month
but enabling all 3 feature groups makes that impossible — realistic minimum is **~$22–42/month**
depending on instance size and risk tolerance (see tiers below).

---

## Why $10/month Is No Longer Achievable

### RAM budget breakdown with all features enabled

| Component | RAM |
|-----------|-----|
| FastAPI + uvicorn + system | ~400 MB |
| MiniLM-L6-v2 embedding model | ~500 MB |
| **SPLADE (naver/splade-v3)** | **~1,500 MB** |
| **Qdrant** (Docker container) | **~500 MB** |
| **Weaviate** (Docker container) | **~1,000 MB** |
| **Milvus** (milvus + etcd + MinIO = 3 containers) | **~2,000 MB** |
| **PGVector** (PostgreSQL 16) | **~400 MB** |
| Docker daemon overhead | ~300 MB |
| **Total** | **~6.6 GB RAM** |

The smallest EC2 instance that can hold this stack is a **t3.large (8 GB RAM)** — which costs
~$18/mo on Spot alone, before storage and networking. $10/month is simply not feasible with all 4
vector stores + SPLADE + LLM enabled simultaneously.

### LM Studio → OpenAI API (gpt-4o-mini)
LM Studio runs a local LLM and cannot run in the cloud without a GPU instance (>$100/mo).
The easiest replacement: **OpenAI API** — the project already uses the OpenAI Python SDK;
you just change `base_url` from `localhost:1234` to `api.openai.com`. At gpt-4o-mini pricing
($0.15/1M input, $0.60/1M output), low personal usage costs **$0.10–$1/month**.

---

## Revised Architecture

```
Internet
   │
   ▼
CloudFront (CDN) ──► S3 Bucket (React static build)           ~$1/mo
   │
   ▼ /api/*  (proxied via EC2 Nginx)
EC2 t3.large SPOT (2 vCPU, 8 GB RAM)                         ~$18/mo
  ├── Nginx reverse proxy (80/443 → 8000)
  ├── FastAPI + uvicorn (port 8000)
  ├── Docker containers via docker-compose.polyrag.yml:
  │     ├── Qdrant        (port 6333)
  │     ├── Weaviate      (port 8088)
  │     ├── Milvus + etcd + MinIO  (port 19530)
  │     └── PGVector      (port 5433)
  ├── ChromaDB + FAISS    (in-process)
  ├── SPLADE model        (in-process, loaded with app)
  └── Kuzu graph          (in-process)
         │
         ▼
     EBS gp3 50 GB (Docker images + all vector data)          ~$4/mo
         │
         ▼
     OpenAI API (gpt-4o-mini) — LLM for query rewriting/HyDE  ~$0.10–1/mo
```

---

## Cost Tiers — Pick Your Comfort Level

### Tier A — Budget Stretch (t3.large Spot, 8 GB RAM) — ~$23/month
Tight on RAM; handles the stack if you don't run all 4 vector stores simultaneously.
Only use 1–2 vector stores at a time.

| Service | Spec | Cost |
|---------|------|------|
| EC2 t3.large Spot | 2 vCPU, 8 GB RAM | ~$18.00 |
| EBS gp3 40 GB | Docker images + data | ~$3.20 |
| S3 | Static frontend (~200 MB) | ~$0.01 |
| CloudFront | 10 GB transfer/mo | ~$0.85 |
| OpenAI API (gpt-4o-mini) | ~500K tokens/mo | ~$0.20 |
| Data Transfer | <1 GB extra | ~$0.09 |
| **TOTAL** | | **~$22–23/mo** |

> **Risk**: Running all 4 vector store containers simultaneously on 8 GB may OOM under load.
> Mitigate by enabling only the backends you actively use in `config.yaml`.

### Tier B — Stable Full-Feature (t3.xlarge Spot, 16 GB RAM) — ~$38/month
All 4 vector stores + SPLADE + LLM with comfortable headroom.

| Service | Spec | Cost |
|---------|------|------|
| EC2 t3.xlarge Spot | 4 vCPU, 16 GB RAM | ~$34.00 |
| EBS gp3 50 GB | Docker images + data | ~$4.00 |
| S3 | Static frontend | ~$0.01 |
| CloudFront | 10 GB transfer/mo | ~$0.85 |
| OpenAI API (gpt-4o-mini) | ~500K tokens/mo | ~$0.20 |
| Data Transfer | ~1 GB | ~$0.09 |
| **TOTAL** | | **~$38–40/mo** |

### Tier C — Original $10 plan (no server vector stores, no SPLADE, no LLM)
See the original plan. ChromaDB + FAISS + BM25 hybrid still work. ~$6–8/mo.

---

## Feature Status (Full-Feature Plan)

| Feature | Status | Notes |
|---------|--------|-------|
| **Qdrant** | ✅ Active | Docker container, EBS-backed |
| **Weaviate** | ✅ Active | Docker container, EBS-backed |
| **Milvus** | ✅ Active | 3-container stack (milvus+etcd+MinIO) |
| **PGVector** | ✅ Active | PostgreSQL 16 + pgvector |
| **ChromaDB + FAISS** | ✅ Active | In-process (default backend) |
| **SPLADE sparse retrieval** | ✅ Active | ~1.5 GB RAM at load time |
| **Query rewriting / HyDE / multi-query** | ✅ Active | Via OpenAI API (gpt-4o-mini) |
| **Knowledge graph (Kuzu)** | ✅ Active | Embedded, no extra RAM |
| **MiniLM-L6-v2 embeddings** | ✅ Active | CPU, ~80 MB |
| **React frontend (S3+CloudFront)** | ✅ Active | Static, very cheap |
| **All 13 API routers** | ✅ Active | No changes needed |
| LM Studio (localhost:1234) | ❌ Replaced | → OpenAI API (1-line config change) |
| BGE-large embeddings | ❌ Disabled | 1.3 GB RAM — too costly on CPU |
| RAPTOR, contextual reranker | ⚠️ LLM-dependent | Works if OpenAI API key set |

---

## Workplan

### Phase 0 — Prerequisites
- [ ] Create AWS account (or use existing); enable billing alerts at $25 and $40
- [ ] Install AWS CLI locally (`aws configure` with IAM user having EC2/S3/CloudFront permissions)
- [ ] Obtain OpenAI API key (gpt-4o-mini) for LLM features — replaces LM Studio
- [ ] Install Docker + Docker Compose locally to test the full stack before deploying

### Phase 1 — Prepare & Build the Project

- [ ] **1.1** Create `Dockerfile.api` for the FastAPI backend:
  - Base: `python:3.11-slim`
  - Install CPU-only torch (avoids 2 GB CUDA download)
  - Copy `core/`, `orchestrator/`, `api/`, `config/`, `requirements.txt`
  - Pre-download MiniLM + SPLADE models at build time
  - Expose port 8000

- [ ] **1.2** Create `cloud-config.yaml` override file:
  ```yaml
  store:
    backend: chromadb           # default; switch to qdrant/weaviate/pgvector/milvus as needed
  llm:
    base_url: https://api.openai.com/v1    # replaces LM Studio
    model: gpt-4o-mini
    enable_rewrite: true
    enable_hyde: true
    enable_multi_query: true
  retrieval:
    splade:
      enabled: true
      persist_dir: ./data/splade
  embedding:
    model: all-MiniLM-L6-v2
    device: cpu
  ```

- [ ] **1.3** Add `OPENAI_API_KEY` to EC2 environment (never commit to git):
  ```bash
  echo "OPENAI_API_KEY=sk-..." | sudo tee -a /etc/environment
  ```

- [ ] **1.4** Build React frontend:
  - Set `VITE_API_URL=http://<EC2-ELASTIC-IP>` in `frontend/.env.production`
  - Run `npm run build` → generates `frontend/dist/`

### Phase 2 — AWS Infrastructure Setup

- [ ] **2.1** Create S3 bucket + enable static website hosting:
  ```bash
  aws s3 mb s3://polyrag-frontend --region us-east-1
  aws s3 website s3://polyrag-frontend --index-document index.html --error-document index.html
  aws s3 sync frontend/dist/ s3://polyrag-frontend --delete
  ```

- [ ] **2.2** Create CloudFront distribution (S3 origin, PriceClass_100, SPA 404 → index.html)

- [ ] **2.3** Create EC2 Security Group:
  - Inbound: SSH (22) from your IP, HTTP (80), HTTPS (443)
  - Outbound: All (pip, HuggingFace, OpenAI API, Docker Hub)

- [ ] **2.4** Launch EC2 Spot Instance — **choose tier**:
  ```bash
  # Tier A (budget): t3.large, 8 GB RAM, ~$18/mo
  # Tier B (stable): t3.xlarge, 16 GB RAM, ~$34/mo
  aws ec2 request-spot-instances \
    --spot-price "0.05" \
    --instance-count 1 \
    --type "persistent" \
    --launch-specification '{
      "ImageId": "ami-0c02fb55956c7d316",
      "InstanceType": "t3.large",
      "KeyName": "your-key-pair",
      "SecurityGroupIds": ["sg-xxxx"],
      "BlockDeviceMappings": [{
        "DeviceName": "/dev/xvda",
        "Ebs": {"VolumeSize": 50, "VolumeType": "gp3", "DeleteOnTermination": false}
      }]
    }'
  ```
  > Use `"type": "persistent"` — Spot fleet auto-relaunches after interruption.
  > Set `DeleteOnTermination: false` so Docker volumes + vector data survive interruptions.

- [ ] **2.5** Attach Elastic IP to the instance

### Phase 3 — Server Setup (SSH into EC2)

- [ ] **3.1** System bootstrap:
  ```bash
  sudo dnf update -y
  sudo dnf install -y python3.11 python3.11-pip nginx git docker
  sudo systemctl enable docker && sudo systemctl start docker
  sudo usermod -aG docker ec2-user
  ```

- [ ] **3.2** Install Docker Compose plugin:
  ```bash
  sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  ```

- [ ] **3.3** Add swap space (4 GB — critical buffer for SPLADE + vector stores):
  ```bash
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
  ```

- [ ] **3.4** Clone repo and install Python dependencies:
  ```bash
  git clone https://github.com/your-org/Phoenice-PolyRAG.git /opt/polyrag
  cd /opt/polyrag
  pip3.11 install torch --index-url https://download.pytorch.org/whl/cpu
  pip3.11 install -r requirements.txt
  python3.11 -m spacy download en_core_web_sm
  ```

- [ ] **3.5** Pre-download models (run once; baked into disk cache):
  ```python
  from sentence_transformers import SentenceTransformer, SparseEncoder
  SentenceTransformer("all-MiniLM-L6-v2")          # ~80 MB
  SparseEncoder("naver/splade-v3")                  # ~440 MB
  ```

- [ ] **3.6** Start vector store containers:
  ```bash
  cd /opt/polyrag
  docker compose -f docker-compose.polyrag.yml up -d
  # Verify all healthy:
  docker compose -f docker-compose.polyrag.yml ps
  ```

- [ ] **3.7** Create systemd service for FastAPI:
  ```ini
  [Unit]
  Description=Phoenice-PolyRAG FastAPI
  After=network.target docker.service
  Requires=docker.service

  [Service]
  User=ec2-user
  WorkingDirectory=/opt/polyrag
  EnvironmentFile=/etc/environment
  ExecStart=/usr/local/bin/gunicorn api.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
  Restart=always
  RestartSec=15
  Environment=PYTHONPATH=/opt/polyrag

  [Install]
  WantedBy=multi-user.target
  ```
  > `-w 1` = 1 worker only; prevents loading SPLADE model twice (would OOM on t3.large)

- [ ] **3.8** Configure Nginx reverse proxy (port 80 → 8000):
  ```nginx
  server {
      listen 80;
      client_max_body_size 50M;
      location /api/ {
          proxy_pass http://127.0.0.1:8000;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_read_timeout 120s;
      }
  }
  ```

- [ ] **3.9** Enable and start all services:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable polyrag nginx
  sudo systemctl start polyrag nginx
  ```

### Phase 4 — Wire Frontend to Backend

- [ ] **4.1** Redeploy frontend with EC2 Elastic IP set in `VITE_API_URL`:
  ```bash
  # In frontend/.env.production:
  VITE_API_URL=http://<ELASTIC-IP>
  npm run build
  aws s3 sync frontend/dist/ s3://polyrag-frontend --delete
  aws cloudfront create-invalidation --distribution-id <CF-ID> --paths "/*"
  ```

### Phase 5 — Verify & Monitor

- [ ] **5.1** Health check: `curl http://<EC2-IP>/api/health`
- [ ] **5.2** Verify Qdrant: `curl http://<EC2-IP>:6333/readyz`
- [ ] **5.3** Verify Weaviate: `curl http://<EC2-IP>:8088/v1/.well-known/ready`
- [ ] **5.4** Test SPLADE: ingest a doc and run a hybrid search
- [ ] **5.5** Test LLM: run a query and confirm query rewriting fires (check logs)
- [ ] **5.6** Set AWS Billing Alert at $25 and $40 via SNS → Email
- [ ] **5.7** Monitor RAM: `free -h` — if used > 7 GB on t3.large, upgrade to t3.xlarge

### Phase 6 — Ongoing Cost Control

- [ ] **6.1** Run only the vector stores you're actively using; comment out the rest in docker-compose
- [ ] **6.2** Add EC2 auto-stop via EventBridge if not running 24/7 (saves up to 60%)
- [ ] **6.3** Set OpenAI usage cap in OpenAI dashboard ($5/mo hard limit)
- [ ] **6.4** Review AWS Cost Explorer weekly for first month

---

## Key Configuration Changes Required

### `cloud-config.yaml` (place in /opt/polyrag, override defaults)
```yaml
store:
  backend: chromadb               # Switch to qdrant/weaviate/milvus/pgvector as needed
llm:
  base_url: https://api.openai.com/v1   # Replaces LM Studio
  model: gpt-4o-mini
  enable_rewrite: true
  enable_hyde: true
  enable_multi_query: true
retrieval:
  splade:
    enabled: true
embedding:
  model: all-MiniLM-L6-v2
  device: cpu
```

### `frontend/.env.production`
```env
VITE_API_URL=http://<EC2-ELASTIC-IP>
```

---

## Notes & Caveats

1. **LM Studio replacement**: The app uses OpenAI SDK (`openai` Python package). Changing
   `base_url` in config is the only code-level change needed to point to OpenAI API.

2. **Milvus on t3.large**: Milvus alone (3 containers) uses ~2 GB RAM. On t3.large (8 GB),
   you cannot run Milvus + Weaviate + SPLADE simultaneously without OOMing. On t3.xlarge (16 GB),
   all 4 run comfortably.

3. **Spot interruption**: Persistent Spot requests auto-relaunch. Docker volumes + EBS survive.
   Expect ~2–5 min downtime per interruption (rare for t3 in us-east-1).

4. **Free Tier note**: If your account is < 12 months old, t2.micro (free) + 30 GB EBS (free)
   gives you only 1 GB RAM — nowhere near enough for this full-feature plan.

5. **Scale-down path**: If cost is too high, disable Milvus + Weaviate (heaviest) and run only
   Qdrant + PGVector + SPLADE. RAM drops to ~4.5 GB; t3.large handles it easily.

