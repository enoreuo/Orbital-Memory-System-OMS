# Orbital Memory System (OMS)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Provider](https://img.shields.io/badge/LLM-Claude%20%7C%20Gemini-orange)

A local-first, physics-inspired memory engine for LLMs. Memories orbit a center of relevance — drifting outward over time (decay) and snapping inward when retrieved (momentum). Retrieval is ranked by a gravity score combining semantic similarity with orbital proximity.

---

## How It Works

```
                    ┌─────────────────────────────────────┐
  Raw Text ──►      │  LLM: generate semantic summary      │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼──────────┐    ┌─────────────────┐
                    │      ChromaDB             │    │    SQLite        │
                    │   (The Search Head)       │    │  (The Drawer)    │
                    │                           │    │                  │
                    │  summary + vector         │◄──►│  orb_id (FK)     │
                    │  orbital_radius           │    │  full_text       │
                    │  last_accessed            │    │                  │
                    └───────────────────────────┘    └─────────────────┘
```

### The Physics

**Decay** — memories drift outward over time:
```
R_new = R_old + ln(1 + Δt) × λ
```

**Momentum** — memories snap inward when retrieved:
```
R_new = R_old × 0.9
```

**Gravity** — ranking score combining relevance + accessibility:
```
G_t = (semantic_similarity × 0.7) + ((1 / radius) × 0.3)
```

---

## Features

- **Hybrid storage**: ChromaDB for vector search, SQLite for full-text archive
- **LLM summarization**: summaries are embedded (not raw text) — better search signal
- **Multi-provider**: Claude (Anthropic) and Gemini (Google) — switch via `.env`
- **Orbital physics**: decay, momentum, and gravity scoring built into retrieval
- **Standalone decay worker**: optional background process to sync radii
- **CLI interface**: `add`, `search`, `chat`, `list` commands
- **14 unit tests**: pure physics engine tests, no API calls needed

---

## Project Structure

```
OMS/
├── engine.py              # Pure physics math — MemoryOrb + decay/momentum/gravity
├── config.py              # All settings: model names, physics constants, paths
├── llm_client.py          # Multi-provider LLM client (Anthropic + Google)
├── memory_agent.py        # LLM↔OMS bridge: ingest() + chat()
├── searcher.py            # ChromaDB ANN search → gravity rerank → momentum
├── decay_worker.py        # Standalone background decay sync
├── cli.py                 # CLI: add / search / chat / list
├── storage/
│   ├── chroma_store.py    # ChromaDB operations
│   └── sqlite_store.py    # SQLite full-text archive
├── tests/
│   └── test_engine.py     # 14 unit tests for physics engine
├── .env.example           # API key template
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
OMS_LLM_PROVIDER=anthropic        # or: google
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### 3. Add memories

```bash
python cli.py add "The Earth orbits the Sun at 150 million km"
python cli.py add "Python was created by Guido van Rossum in 1991"
python cli.py add "Transformers use self-attention to process sequences"
```

### 4. Search

```bash
python cli.py search "solar system"
```

Output:
```
Rank 1 [G=0.8921]  "The Earth orbits the Sun at 150 million km"
         id=3f2a1b...  summary: "Earth orbits Sun at 150M km distance..."
```

### 5. Chat with memory

```bash
python cli.py chat
python cli.py chat --provider google
```

### 6. List all memories

```bash
python cli.py list
```

### 7. Run the decay worker (optional, separate terminal)

```bash
python decay_worker.py               # sync every 30 minutes
python decay_worker.py --interval 60 # sync every 60 minutes
python decay_worker.py --once        # run once and exit
```

---

## CLI Reference

```
python cli.py add "text"              Store a new memory
python cli.py add "text" --provider google

python cli.py search "query"          Search by gravity score
python cli.py search "query" --top-k 10

python cli.py chat                    Interactive chat with memory context
python cli.py chat --provider google

python cli.py list                    List all memories sorted by radius
```

---

## Switching LLM Providers

Change one line in `.env`:

```env
# Use Claude (Anthropic)
OMS_LLM_PROVIDER=anthropic

# Use Gemini (Google)
OMS_LLM_PROVIDER=google
```

### Model Templates (edit `config.py`)

| Task | Anthropic | Google |
|---|---|---|
| Summarization | `claude-haiku-4-5-20251001` | `gemini-2.0-flash-lite` |
| Chat | `claude-sonnet-4-6` | `gemini-2.5-flash` |

---

## Physics Tuning

All constants are in `config.py`:

```python
DECAY_RATE      = 0.01   # How fast memories fade (higher = faster)
MOMENTUM_FACTOR = 0.9    # How much radius shrinks on retrieval (lower = stronger pull)
GRAVITY_W_S     = 0.7    # Weight for semantic similarity in ranking
GRAVITY_W_R     = 0.3    # Weight for orbital proximity in ranking
```

---

## OMS vs RAG vs MemGPT

| | Standard RAG | MemGPT | **OMS** |
|---|---|---|---|
| Search method | Cosine similarity | Semantic + recency | **Gravity = similarity + orbital proximity** |
| Memory priority | All equal | Recency-based | **Physics-based: usage reinforces, time fades** |
| Forgetting | None | Manual/TTL | **Natural logarithmic decay** |
| Reinforcement | None | Explicit flags | **Automatic momentum on retrieval** |
| LLM calls/query | 1 | 2 | **1** |
| Explainability | Low | Medium | **High — gravity score is transparent** |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 14 tests cover the physics engine (no API calls, no database):
- **TestDecay**: radius increases with time, monotonic, pure function
- **TestMomentum**: radius decreases on access, factor respected, pure function
- **TestGravity**: formula correctness, distance ranking, zero-radius guard

---

## Architecture Deep Dive

### Why two databases?

**ChromaDB** (search head): stores the LLM-generated summary + its vector embedding + orbital metadata (`orbital_radius`, `last_accessed`). This is what gets searched — the compressed semantic essence of each memory.

**SQLite** (drawer): stores the original full text. Fetched only after search identifies the top results. Keeps the search layer lean and fast.

Both are linked by the same `orb_id` UUID.

### Why lazy decay?

Decay is computed on-the-fly at query time: `effective_radius = stored_radius + ln(1 + Δt) × λ`. The stored radius is never stale for *search purposes* — it's always recomputed from `last_accessed`. The decay worker is optional and only exists to sync stored values for display/analytics.

### Why embed summaries instead of full text?

Embedding a 3-sentence summary produces a cleaner, more focused semantic vector than embedding a long passage. The LLM distills the semantic core — noise is removed before the embedding step. Better signal → better search.

---

## Roadmap

- [ ] Chat UI (Streamlit or FastAPI)
- [ ] FastAPI layer for Flutter integration
- [ ] Per-orb decay rate (different memories fade at different rates)
- [ ] Momentum applied to top-3 results (not just top-1)
- [ ] Auto-archive orbs with `radius > threshold`
- [ ] Async ingestion queue (non-blocking `add`)
- [ ] Export/import memory store

---

## Contributing

Contributions are welcome — this project is intentionally modular so it's easy to extend.

**Good first issues:**
- Add a new LLM provider (OpenAI, Ollama, Mistral) in `llm_client.py`
- Build a Streamlit or FastAPI chat UI
- Add per-orb decay rate (some memories should fade faster than others)
- Implement auto-archiving when `radius > threshold`

**How to contribute:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Make your changes — keep physics logic in `engine.py`, storage in `storage/`
4. Add or update tests in `tests/`
5. Open a Pull Request with a clear description

**Questions or ideas?** Open an [Issue](../../issues) — all feedback is welcome.

---

## License

MIT License — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.
