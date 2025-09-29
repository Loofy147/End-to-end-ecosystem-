# NEXUS — ARCHITECTURE.md

**One-line decision summary:**  
Build NEXUS as a modular, microservices-based adaptive knowledge-driven AI orchestrator using Graph-RAG, multi-tier memory, an LLM adapter layer, and a secure execution sandbox (E2B) to enable safe, auditable, and scalable AI workflows.

---

## Project Overall Summary
The NEXUS Proof of Concept (PoC) demonstrates a modular microservices architecture for an adaptive knowledge-driven AI orchestrator. It orchestrates interactions between specialized components to intelligently process user queries, combining knowledge retrieval, memory management, agent-based handling, and safe code execution, while laying the groundwork for observability and security.

### Key Goals
- Explore service-oriented architecture patterns for AI systems.
- Implement core functionalities: hybrid knowledge retrieval (Graph-RAG), memory tiers, LLM adapter (mock), and execution sandbox (E2B prototype).
- Demonstrate an agent-oriented processing pipeline for dynamic query routing.
- Establish baseline observability, governance, and security practices.

---

## Core Components (Implemented in PoC)
1. **Orchestrator**
   - Central request handler and workflow coordinator.
   - Retrieves context from Memory, calls Hybrid Retriever, selects agents from Agent Manager, interacts with the LLM adapter and Sandbox, and returns consolidated responses.
2. **Agent Manager**
   - Mocked for PoC (keyword/context matching).
   - Responsible for selecting processing paths (agents) and managing agent metadata.
3. **Knowledge Graph (Graph-RAG Mock)**
   - Hybrid search stub that returns document hits and optional graph expansions.
4. **Memory Layer (File-based Mock)**
   - Demonstrates ephemeral, episodic, and semantic tiers via file persistence.
5. **Execution Sandbox (Basic Security)**
   - Executes code in an isolated process with keyword/regex checks and timeout protections (PoC-level only).
6. **LLM Abstraction Layer (Mock Adapter)**
   - Simulates LLM interactions, returning generated answers or suggested actions.
7. **Infra & Developer Tooling**
   - Dockerfiles, docker-compose, placeholder Kubernetes manifests, basic CI, and developer integration stubs (CopilotKit/Aider).

---

## High-Level Architecture
```
    +--------------------+        +-------------------+
    |      Client UI     | <----> |   Orchestrator    |
    +--------------------+        +-------------------+
                                         |
            +----------------------------+---------------------------+
            |                            |                           |
  +------------------+        +--------------------+        +-------------------+
  | Knowledge Graph  | <----> |   Hybrid Retriever | <----> |    Vector Store   |
  |   (Neo4j)        |        |   (Graph-RAG)      |        |  (pgvector/FAISS) |
  +------------------+        +--------------------+        +-------------------+
            |                            |
            |                            v
            |                      +-----------+
            |                      |  Memory   |
            |                      | (Mem0/GPTCache)|
            |                      +-----------+
            |                            |
            v                            v
      +-------------+            +----------------+
      | Agent Mgmt  | <--------> |   Execution    |
      | (agents)    |            |   Sandbox E2B  |
      +-------------+            +----------------+
```

---

## Data Flow (PoC)
1. Client sends `query` to Orchestrator.
2. Orchestrator loads session/user context from Memory.
3. Hybrid Retriever performs vector search and graph expansion (Graph-RAG).
4. Orchestrator/Agent Manager selects agent(s) and composes prompt.
5. LLM Adapter (mock) returns an answer or an action plan.
6. If execution is required, Execution Agent dispatches code to the Sandbox (E2B) for controlled run.
7. Results, provenance, and telemetry are stored back into Graph and Memory; cache is updated.

---

## APIs (PoC Endpoints)
- `POST /api/v1/query` — body: `{user_id, session_id, query, priority, constraints}`
- `POST /api/v1/execute` — body: `{agent, payload, resource_limits, sandbox}`
- `POST /hybrid_search` — body: `{query}`
- `POST /memory/store` — body: `{user_id, session_id, event}`

---

## Deployment & Infra (PoC vs Production)
- **PoC:** Docker Compose orchestration with local Neo4j, file-based memory, and basic sandbox process execution.
- **Production recommendations:**
  - Kubernetes (K8s) with HPA, NetworkPolicies, and Pod Security Policies.
  - Managed or highly available Postgres + pgvector or production vector DB (or vector-engine like Milvus/Weaviate with sharding).
  - Neo4j or JanusGraph cluster with backups and monitoring.
  - Wasm runtimes (Wasmtime) or microVMs (Firecracker) for secure execution; avoid raw subprocess execution.
  - Vault/Secret Manager for LLM keys and credentials.

---

## Observability & Security Baselines
- **Observability:** Prometheus metrics, OpenTelemetry tracing, structured logs (ELK/Loki), dashboarding (Grafana).
- **Security:** TLS for all services, secret management (Vault/KMS), sandbox syscall filtering, resource limits, PII redaction, and audit trails for provenance and data access.
- **Governance:** Policy engine to enforce privacy levels, allowable APIs, and execution permissions.

---

## Testing Strategy
- **Unit tests:** service-level tests (pytest).
- **Integration tests:** end-to-end RAG flow with mocked LLM and vector store.
- **Security tests:** sandbox fuzzing, static analysis, and controlled pentests.
- **Load tests:** Locust/k6 to validate P95 latency targets.

---

## KPIs & Targets
- P50 latency for common queries: < 300 ms (target).
- Cache hit rate: ≥ 60% (after warm-up).
- Retrieval recall@10: improve by ~15% over vector-only baseline using Graph-RAG.
- Sandbox failure rate: <<1% in normal operations (target).

---

## Limitations & Notes
- The current PoC sandbox is **not** production secure. Replace with robust Wasm or microVM-based execution for any production use.
- Memory persistence in PoC is file-based for demonstration; replace with Redis/Postgres + vector store for production.
- LLM Adapter is a mock; integrate real providers via an adapter layer supporting streaming, function-calling, and fallbacks.

---

## Next Steps (Recommended)
1. Replace file-based memory with Redis + Postgres (pgvector) for semantic tier.
2. Implement a secure WASM-based sandbox and run a security evaluation.
3. Integrate a real LLM provider via LLM Adapter (OpenAI/Anthropic/Gemini) with a fallback strategy.
4. Implement provenance recording in the Graph for every external data source used in answers.
5. Add Prometheus/Grafana dashboards and OpenTelemetry tracing for end-to-end observability.
6. Prepare a hardened K8s deployment manifest and run production load testing.

---

## Appendix / References
- Design: Graph-RAG, Memory tiers (ephemeral/episodic/semantic), E2B sandbox design, LLM adapter patterns.
- For detailed ADRs, diagrams, and implementation tickets, see `/docs/` directory and the repository `README.md`.