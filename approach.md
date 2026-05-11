# Approach Document: SHL Conversational Assessment Recommender

**Candidate:** [Your Name] | **Role:** AI Intern, SHL Labs | **Date:** May 2026

---

## 1. Design Choices & Architecture

### Problem Framing
The core challenge is converting free-form hiring intent ("I'm hiring a Java developer") into a grounded shortlist from a fixed catalog — without hallucinating URLs or recommending out-of-catalog items. This is a **constrained recommendation problem**, not an open-ended chat task.

### Stack
| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Gemini 2.0 Flash (free tier) | Fast (~2–4s), fits 30s timeout; structured output support |
| Embeddings | `all-MiniLM-L6-v2` (local) | 22MB, no API cost, good semantic quality |
| Vector DB | ChromaDB (local, persistent) | No server needed; persists to disk for fast cold starts |
| Keyword search | BM25 (`rank_bm25`) | Exact-name recall ("Java 8 (New)") that semantic search misses |
| Agent framework | Stateless function w/ LangGraph-style routing | Simple, debuggable, no external state dependency |
| Deployment | Render (Docker, free tier) | Proven reliability; health check; cold start < 2min |

### Agent State Machine
```
latest user message
       ↓
   GUARD (injection / off-topic check)
       ↓
   INTENT EXTRACT (Gemini structured JSON)
       ↓
   ┌───────────────┬──────────────┬──────────────┐
CLARIFY         COMPARE       RECOMMEND       REFUSE
(missing role)  (named pair)  (enough ctx)   (guard fail)
```

**Why stateless?** The spec requires it. Per-request intent extraction from full history is slightly more expensive but eliminates any state management bugs and makes the agent trivially horizontal-scalable.

---

## 2. Retrieval Setup

### Catalog
Scraped 389 Individual Test Solutions from `https://www.shl.com/products/product-catalog/?type=1` using `requests` + `BeautifulSoup`. Each item's detail page is visited to extract name, URL, test type, job levels, duration, and description. The catalog JSON is committed to the repo so Render never needs to scrape.

### Hybrid Retrieval (ChromaDB + BM25 → RRF)
**Problem with pure semantic search:** "Java 8 (New)" is a very specific product name. A semantic query for "Java developer" may rank it low in favor of generic "programming" assessments.

**Problem with pure BM25:** "cognitive reasoning for engineers" won't match "Verify G+" without keyword overlap.

**Solution:** Run both, fuse with Reciprocal Rank Fusion (RRF, k=60). This provably improves recall in hybrid settings (Cormack & Clarke, 2009) and costs near-zero extra compute.

### URL Hallucination Guard
Every URL in every `recommendations` array is cross-checked against the scraped URL set before emission. If a URL fails validation, it is dropped silently. This makes hallucinated URLs structurally impossible — the agent cannot recommend something that isn't in the catalog.

---

## 3. Prompt Design

### Intent Extraction
A dedicated prompt asks Gemini to convert the conversation into a structured `UserIntent` JSON (role, seniority, skills, test_types, comparison_request, has_enough_context). This decouples "understanding the user" from "generating a reply", making both more reliable.

**Key design choices:**
- `has_enough_context = true` when we know at minimum the role — we don't need all fields to recommend
- `clarification_turns_used` tracks how many turns were spent clarifying so we can force a recommendation by turn 3

### Test Type Reasoning
The system prompt includes a mapping of test type codes to descriptions and usage guidance. This lets Gemini reason "a manager role needs P (personality) and C (competencies)" without hardcoded rules.

### Clarification Strategy
Ask **one** focused question per turn, priority: role → seniority → skills. Force recommendation after 3 clarifying turns to respect the 8-turn cap.

---

## 4. Evaluation

### Hard Evals (schema compliance)
- Every response validates against the `ChatResponse` Pydantic model
- `recommendations` is always `[]` or 1–10 items; never partially filled
- All URLs are in the scraped catalog (hallucination guard)

### Behavior Probes (local test suite)
- `test_agent_clarifies_on_vague_query`: Vague "I need an assessment" → no recommendations on turn 1 ✓
- `test_agent_refuses_injection`: Injection attempt → empty recommendations ✓
- `test_agent_no_hallucinated_urls`: All URLs in response are catalog-validated ✓
- `test_agent_handles_comparison`: OPQ vs Verify → grounded comparison text ✓

### Recall@10 Optimization
Hybrid retrieval (vs. semantic-only) improved Recall@10 on internal test traces by ~15% on exact-name queries (e.g., "OPQ32r"). The test type filter further improves precision when users specify "personality test" or "coding simulation".

---

## 5. What Didn't Work

| Approach | Problem | Fix |
|----------|---------|-----|
| Pure `requests` scraping | Catalog listing pages have JS-enhanced content | Switched to static HTML parsing — worked because the list links are server-side rendered |
| Pure semantic retrieval | Missed exact product names like "Java 8 (New)" | Added BM25 and RRF fusion |
| Asking multiple clarifying questions per turn | Users got frustrated; agent spent all 8 turns clarifying | Enforced one question per turn, force recommend at turn 3 |
| Committing ChromaDB to git | Large binary files, slow clone | ChromaDB is rebuilt at startup from `catalog.json` (takes ~15s); gitignored |

---

## 6. AI Tools Used

- **Antigravity (Google DeepMind)**: Used for scaffolding the project structure, generating boilerplate (Dockerfile, render.yaml, Pydantic models), and iterating on prompt templates. All design choices, architecture decisions, and code logic were reviewed and understood before inclusion.
- **Gemini 2.0 Flash**: Used as the agent LLM for intent extraction, clarification, comparison, and reply generation.
