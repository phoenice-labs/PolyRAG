# Phoenice-PolyRAG — ISV Self-Hosted Licensing & IP Protection Plan

## Business Model Summary

| Dimension | Decision |
|-----------|----------|
| **Delivery model** | Self-hosted — customer installs on their own infra |
| **Licensing unit** | Per named user (or concurrent session — see tiers) |
| **Billing cadence** | Annual subscription |
| **LLM flexibility** | Customer chooses: local LLM (LM Studio, Ollama) or commercial API (OpenAI, Azure OpenAI, etc.) |
| **Connectivity** | Both: online-connected AND air-gapped (offline) deployments |
| **Source code delivery** | ❌ Never — compiled binaries + Docker images only |

---

## The Core IP Protection Challenge

When software is installed on-premises, the customer controls the machine.
Without active protections, they can:
- Copy and redistribute the software beyond the licensed seat count
- Run it past the license expiry without paying
- Reverse-engineer the Python source (trivially — `.pyc` files are decompilable)
- Deploy in multiple environments beyond what was purchased

**Solution: 5 layered defenses — Legal + Code + License + Distribution + Audit**

---

## Layer 1 — Legal Protection

### Contracts Required
Every customer signs before receiving software:

1. **Master Subscription Agreement (MSA)**
   - Grants a non-transferable, non-sublicensable license to use the software
   - Defines permitted number of named users (matches license file)
   - Prohibits reverse engineering, decompilation, redistribution, sublicensing
   - Grants vendor **right to audit** (remote or on-site, 30-day notice)
   - Auto-terminates on non-payment; customer must delete all copies
   - Governing law, dispute resolution, liability caps

2. **End-User License Agreement (EULA)**
   - Covers individual user behavior
   - Acceptable use policy (no sharing credentials)
   - Anti-circumvention clause (prohibits bypassing license checks)

3. **Data Processing Agreement (DPA)** — if the software processes customer data
   - GDPR/CCPA compliance for vendors selling to EU/California customers

4. **Audit Rights Clause** (key IP protection):
   - Vendor may audit customer's deployment logs on 30-day notice
   - If audit reveals > 10% over-usage, customer pays back-fees + audit cost
   - This clause alone deters most accidental over-deployment

> **Practical note**: Without a signed MSA, none of the technical controls below are legally enforceable.

---

## Layer 2 — Code Obfuscation & Compilation

### The Problem with Plain Python
Python `.py` files are human-readable. Even compiled `.pyc` bytecode can be decompiled
in minutes using tools like `uncompyle6` or `decompile3`. The core IP lives in:
- `core/retrieval/hybrid.py` — 3-way RRF fusion algorithm
- `core/confidence/` — 7-signal confidence aggregator
- `orchestrator/pipeline.py` — the 11-phase orchestration logic
- `core/graph/` — entity extraction + Kuzu integration

### Solution: Compile Python → Native Binary (Nuitka)

**Nuitka** compiles Python to C source, then to a native `.so` (Linux) or `.pyd` (Windows) binary.
The result is:
- No readable source code
- No `.pyc` to decompile
- Runs as fast or faster than interpreted Python
- Fully compatible with FastAPI, torch, sentence-transformers

#### What to Compile (Core IP Modules)
```
core/retrieval/      → compiled .so/.pyd
core/confidence/     → compiled .so/.pyd
core/chunking/       → compiled .so/.pyd
orchestrator/        → compiled .so/.pyd
api/routers/rag.py   → compiled .so/.pyd  (the main agentic endpoint)
```

#### What to Leave as-is (Not Core IP)
```
config/config.yaml          — customer needs to edit this
api/main.py                 — startup boilerplate, visible
frontend/ (React build)     — already minified by Vite; source maps excluded
requirements.txt            — standard OSS deps, no IP here
```

#### Build Command (in CI/CD)
```bash
# Compile core module to standalone .so
python -m nuitka \
  --module core/ \
  --include-package=core \
  --output-dir=dist/compiled \
  --remove-output \
  --no-pyi-file

# Compile orchestrator
python -m nuitka \
  --module orchestrator/ \
  --include-package=orchestrator \
  --output-dir=dist/compiled
```

#### Alternative: PyArmor (simpler, less strong)
If Nuitka build complexity is too high initially, **PyArmor** wraps Python bytecode with
an encrypted runtime. Easier to set up; weaker than Nuitka but catches casual inspection.
```bash
pip install pyarmor
pyarmor gen --recursive core/ orchestrator/
```

---

## Layer 3 — License Enforcement (The Technical Heart)

### License Architecture

Two enforcement modes — same codebase, different config:

```
┌─────────────────────────────────────────────────────┐
│              License Validation Engine               │
│                                                      │
│  ┌──────────────────┐    ┌────────────────────────┐  │
│  │   ONLINE MODE    │    │  OFFLINE / AIR-GAP MODE│  │
│  │                  │    │                        │  │
│  │ Heartbeat every  │    │ Signed .lic file       │  │
│  │ 24h to vendor    │    │ validated locally with │  │
│  │ license server   │    │ vendor's RSA public key│  │
│  │                  │    │ (baked into binary)    │  │
│  │ 7-day grace on   │    │                        │  │
│  │ connectivity loss│    │ 90-day validity window │  │
│  └──────────────────┘    └────────────────────────┘  │
│                                                      │
│  Both modes enforce: user count, expiry, features   │
└─────────────────────────────────────────────────────┘
```

### License File Format (JWT-based)

The license is a **signed JWT** (RS256 — RSA 2048-bit):
- Vendor holds the **private key** (signs licenses)
- Vendor's **public key is compiled into the binary** (verifies signatures)
- Customer cannot forge a valid license without the private key

```json
{
  "header": { "alg": "RS256", "typ": "JWT" },
  "payload": {
    "iss": "phoenice-polyrag-licensing",
    "iat": 1711238400,
    "exp": 1742774400,
    "customer_id": "ACME-CORP-001",
    "customer_name": "Acme Corporation",
    "max_named_users": 25,
    "max_concurrent_sessions": 10,
    "features": {
      "splade": true,
      "graph": true,
      "all_vector_backends": true,
      "llm_rewriting": true,
      "ragas_eval": false
    },
    "install_fingerprint": "sha256-of-machine-ids",   // optional: hardware binding
    "license_tier": "professional",
    "support_level": "standard",
    "air_gapped": true
  },
  "signature": "...RSA-SHA256 signature..."
}
```

### License Validation Code (baked into compiled binary)
```python
# core/licensing/validator.py  ← this module gets compiled with Nuitka
import jwt
from datetime import datetime, timezone
from pathlib import Path

VENDOR_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...  # baked in at compile time
-----END PUBLIC KEY-----"""

class LicenseValidator:
    def __init__(self, license_path: str = "/etc/polyrag/license.lic"):
        self._license_path = Path(license_path)
        self._cached_claims = None

    def validate(self) -> dict:
        token = self._license_path.read_text().strip()
        try:
            claims = jwt.decode(token, VENDOR_PUBLIC_KEY, algorithms=["RS256"])
        except jwt.ExpiredSignatureError:
            raise RuntimeError("License expired. Contact licensing@phoenice.ai to renew.")
        except jwt.InvalidTokenError as e:
            raise RuntimeError(f"Invalid license file: {e}")

        # User count enforcement (checked at login)
        self._cached_claims = claims
        return claims

    def check_user_allowed(self, active_users: int) -> bool:
        claims = self._cached_claims or self.validate()
        return active_users < claims["max_named_users"]

    def is_feature_enabled(self, feature: str) -> bool:
        claims = self._cached_claims or self.validate()
        return claims.get("features", {}).get(feature, False)
```

### Startup Enforcement
On every `uvicorn` start, before accepting any request:
```python
# api/main.py (startup event)
@app.on_event("startup")
async def enforce_license():
    validator = LicenseValidator()
    claims = validator.validate()           # Raises if expired or tampered
    app.state.license = claims
    logger.info(f"License valid. Customer: {claims['customer_name']}, "
                f"Expires: {claims['exp']}, Users: {claims['max_named_users']}")
```

### Online Heartbeat (Internet-connected customers)
```python
# Runs every 24 hours in background
async def license_heartbeat():
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://licensing.phoenice.ai/api/v1/heartbeat",
            json={
                "customer_id": app.state.license["customer_id"],
                "token": license_token,
                "active_users": get_active_user_count(),
                "version": APP_VERSION,
            },
            timeout=10
        )
    if r.status_code == 402:  # Payment required
        app.state.license_grace_expires = now() + timedelta(days=7)
```

### Hardware Fingerprinting (Optional — for highest-value customers)
Binds the license to specific machine(s). Prevents copying the license to unauthorized servers.
```python
import hashlib, platform, uuid

def get_machine_fingerprint() -> str:
    components = [
        platform.node(),           # hostname
        str(uuid.getnode()),       # MAC address
        platform.machine(),        # CPU arch
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
```
- Include `install_fingerprint` in the license JWT at issuance time
- Validator checks fingerprint on startup
- For Docker/cloud deployments: fingerprint the instance ID or ECS task ID instead

---

## Layer 4 — Vendor License Server (Small SaaS Backend)

A simple FastAPI app you host (tiny EC2 t3.micro, ~$8/mo):

### Endpoints
```
POST /api/v1/issue          ← Internal: generate and sign a new license JWT
POST /api/v1/heartbeat      ← Receive pings from connected installations
GET  /api/v1/usage/:cust_id ← Internal dashboard: see active users per customer
POST /api/v1/revoke         ← Immediately invalidate a license (non-payment, fraud)
GET  /api/v1/portal         ← Customer self-service: download license, see seat usage
```

### License Issuance Flow
```
Sales closes deal
      │
      ▼
Vendor creates license in admin portal
  (customer_id, user count, expiry, features)
      │
      ▼
License server signs JWT with RSA private key
      │
      ▼
License file (.lic) emailed to customer + downloadable from portal
      │
      ▼
Customer places file at /etc/polyrag/license.lic
      │
      ▼
Software validates on startup ✅
```

### Revocation
- **Online mode**: Heartbeat receives HTTP 402 → software enters read-only mode after 7-day grace
- **Air-gapped mode**: Issue a short-validity replacement license (90-day window forces re-issuance)
  - Non-renewing customer simply doesn't receive the next license file → software expires naturally

---

## Layer 5 — Distribution Control

### Never Ship Source Code
Customers receive **only**:

```
polyrag-v14.2.0-linux-x86_64.tar.gz
├── bin/
│   └── polyrag-server          ← Nuitka standalone binary (FastAPI + all compiled modules)
├── frontend/
│   └── dist/                   ← Pre-built React (minified, no source maps)
├── config/
│   └── config.yaml             ← Template config (customer edits this)
├── docker-compose.polyrag.yml  ← For vector store backends
├── install.sh                  ← Automated setup script
└── LICENSE.lic.example         ← Placeholder — real .lic file delivered separately
```

### Private Docker Registry (AWS ECR)
```
# Customer pulls images using time-limited ECR credentials
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/polyrag:14.2.0
```
- Credentials are per-customer, scoped to `pull` only (no push)
- Credentials expire and are rotated at renewal time
- Non-renewing customers have ECR access revoked → can't pull future versions

### Version Control (Anti-Downgrade)
```json
// In license JWT
"min_version": "14.0.0",   // prevents using new license with old unpatch'd version
"max_version": "14.x.x"    // optional: prevents using under a specific major
```

---

## Pricing Tier Recommendations

| Tier | Named Users | Key Features | Annual Price (suggested) |
|------|-------------|--------------|--------------------------|
| **Starter** | Up to 5 | ChromaDB + FAISS + BM25, 1 LLM backend | $4,000/yr |
| **Professional** | Up to 25 | All vector backends, SPLADE, graph, LLM rewriting | $15,000/yr |
| **Enterprise** | Up to 100 | All features, RAPTOR, RAGAS eval, audit logs, SLA | $40,000/yr |
| **Unlimited** | Unlimited users | Full stack + custom model support + source escrow | $100,000+/yr |

> **Per-user enforcement options:**
> - **Named users**: Specific email addresses enumerated in license. Simplest to enforce.
> - **Concurrent sessions**: Maximum active sessions at any moment. Requires session tracking middleware.
> - **Active users (rolling 30-day)**: Users who logged in within the past 30 days. Common in SaaS.

---

## Workplan (Implementation Order)

### Phase 0 — Legal Foundation
- [ ] Engage IP attorney to draft MSA + EULA template
- [ ] Add audit rights clause with 10% overage penalty
- [ ] Register copyright for the software (US Copyright Office — $65, 6-month process)
- [ ] Consider patent search for the 3-way RRF fusion + 7-signal confidence aggregator (novel algorithms)
- [ ] Set up licensing@phoenice.ai email for renewals/support

### Phase 1 — License Enforcement Module
- [ ] Generate RSA-2048 key pair (private key stored in HSM or AWS KMS — never on disk)
  ```bash
  openssl genrsa -out vendor_private.pem 2048
  openssl rsa -in vendor_private.pem -pubout -out vendor_public.pem
  ```
- [ ] Create `core/licensing/validator.py` with public key baked in
- [ ] Add license validation to `api/main.py` startup event
- [ ] Add per-request user count enforcement middleware
- [ ] Add feature flag checks in relevant routers (SPLADE, graph, etc.)
- [ ] Add 7-day grace period logic for connectivity loss
- [ ] Write tests: expired license, tampered license, over-user-count, feature disabled

### Phase 2 — Online Heartbeat Service
- [ ] Create `core/licensing/heartbeat.py` — async background task, 24-hour interval
- [ ] Set up vendor license server (FastAPI, separate repo)
  - Issue, revoke, heartbeat, portal endpoints
  - PostgreSQL backend (customer_id, seats, expiry, usage logs)
  - Deploy on EC2 t3.micro Spot (~$8/mo)
- [ ] Implement revocation: HTTP 402 → read-only grace mode → shutdown after 7 days

### Phase 3 — Air-Gap / Offline Support
- [ ] Offline license: 90-day JWT expiry (no heartbeat needed)
- [ ] License renewal workflow: vendor issues new .lic file, customer replaces it
- [ ] Add license hot-reload: app detects file change and revalidates without restart
- [ ] Document the air-gap installation guide for regulated-industry customers

### Phase 4 — Code Compilation
- [ ] Add Nuitka to CI/CD pipeline (GitHub Actions)
- [ ] Compile `core/` and `orchestrator/` to `.so` binaries
- [ ] Test compiled binaries: all 13 API routers, embedding, SPLADE, graph
- [ ] Verify no source leakage in compiled output (`strings binary | grep def ` should return nothing critical)
- [ ] Build React with `--no-sourcemap` flag (remove source maps from production build)

### Phase 5 — Distribution Infrastructure
- [ ] Set up private AWS ECR registry (`123456789.dkr.ecr.us-east-1.amazonaws.com/polyrag`)
- [ ] Per-customer ECR IAM credentials (pull-only, scoped policy)
- [ ] Build release pipeline: compile → package → push to ECR → notify customer
- [ ] Create install.sh that:
  1. Validates OS (Ubuntu 20.04+, Amazon Linux 2023, RHEL 8+)
  2. Checks license file is present
  3. Pulls Docker images
  4. Configures systemd services
  5. Runs smoke test

### Phase 6 — Admin Portal (Customer Self-Service)
- [ ] License portal UI (simple React app):
  - Customer login → view seat usage, expiry date, download .lic renewal
  - Vendor login → issue licenses, view all customers, revoke, see usage
- [ ] Usage reporting: "You are using 18 of 25 licensed seats"
- [ ] Renewal reminder emails at 90, 60, 30, 14, 7 days before expiry

### Phase 7 — Source Code Escrow (Optional, for Enterprise)
- [ ] Engage a software escrow provider (Iron Mountain, NCC Group)
- [ ] Deposit: annotated source code, build instructions, cryptographic keys
- [ ] Release conditions: vendor insolvency, end of business, failure to maintain
- [ ] Include escrow option as an Enterprise-tier upsell ($5,000–$10,000/yr extra)
- [ ] This closes the deal with large enterprise procurement teams who require it

---

## Air-Gapped vs. Online — Feature Comparison

| | Online | Air-Gapped |
|--|--------|-----------|
| **License validation** | JWT + 24h heartbeat | Signed JWT, local only |
| **Revocation speed** | Immediate (next heartbeat) | Next renewal cycle (≤ 90 days) |
| **LLM** | Cloud API (OpenAI, Azure OpenAI) | Local (LM Studio, Ollama) |
| **HuggingFace models** | Downloaded on first use | Pre-baked into Docker image |
| **Updates** | Pull from ECR | Vendor ships new tarball/image |
| **Usage reporting** | Automatic (heartbeat) | Honor-system + annual audit |
| **License file validity** | 1 year (renewed via heartbeat) | 90 days (requires new .lic file) |

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Customer runs more users than licensed | Medium | User count middleware + audit rights |
| Customer copies binary to unlicensed server | Low | Hardware fingerprinting (optional) + audit rights |
| Customer decompiles Nuitka binary | Very Low | Nuitka C compilation is very hard to reverse; legal deterrent |
| Customer shares license file | Low | Fingerprint binding + audit; file contains customer_id (traceable) |
| Customer forges a license JWT | Near zero | Requires vendor RSA private key (stored in KMS, never shipped) |
| Customer refuses audit | Contractual breach | MSA gives right to suspend access; last resort: legal action |
| Vendor license server goes down | Medium | 7-day offline grace period; license server is simple, low traffic |

---

## Summary

```
IP Protection = Legal (MSA + EULA) 
             + Code (Nuitka compilation — no readable source)
             + License (RSA-signed JWT — unforgeable)
             + Distribution (ECR pull-only — no source repo access)
             + Audit rights (contractual backstop)
```

The combination makes it **economically and technically impractical** for a customer to steal IP
or over-deploy. No single layer is perfect, but all five together create strong deterrence.
