# Plan: Orbital Memory System (OMS)

## Context
Building a local-first dynamic memory engine from scratch in `c:\Users\HP\Downloads\MSD\`. The system uses vector embeddings and orbital physics metaphors: memories drift further from "center" over time (decay) and snap back when recalled (momentum). Retrieval ranks by a gravity formula combining semantic similarity and orbital proximity. The spec is in `oms.md`.

---

## Project Structure

```
MSD/
├── requirements.txt
├── engine.py       # Pure physics math — MemoryOrb + all 3 physics functions
├── storage.py      # SQLite CRUD — serialize/deserialize orbs
├── searcher.py     # Orchestration — embed, score, momentum update
├── cli.py          # argparse CLI: add / search / list
└── tests/
    ├── __init__.py
    └── test_engine.py
```

---

## Implementation Steps (in order)

### 1. `requirements.txt`
```
numpy
scikit-learn
sentence-transformers
```

### 2. `engine.py` — Pure math, zero I/O
- `@dataclass MemoryOrb`: `content`, `vector` (np.ndarray), `orbital_radius=1.0`, `last_accessed` (datetime), `metadata={}`, `id=None`
- `apply_decay(orb, decay_rate=0.01) -> float`  
  Formula: `R_new = R_old + ln(1 + Δt) × λ` (Δt in hours). Pure — does NOT mutate orb.
- `apply_momentum(orb, factor=0.9) -> float`  
  Formula: `R_new = R_old × factor`. Pure — does NOT mutate orb.
- `compute_gravity(orb, semantic_similarity, w_s=0.7, w_r=0.3) -> float`  
  Formula: `G_t = (sim × w_s) + (1/radius × w_r)`. Epsilon guard on radius.

### 3. `storage.py` — SQLite only
- `DB_PATH = "oms.db"`
- `initialize_db()` — CREATE TABLE IF NOT EXISTS `orbs` (id, content, vector_blob BLOB, orbital_radius REAL, last_accessed TEXT ISO8601, metadata_blob BLOB)
- `save_orb(orb) -> int` — INSERT, return `lastrowid`
- `load_all_orbs() -> list[MemoryOrb]` — SELECT *, deserialize blobs via `pickle.loads`
- `update_orb(orb) -> None` — UPDATE radius + last_accessed by id
- Vector stored as BLOB: `pickle.dumps(np.ndarray)` / `pickle.loads`

### 4. `searcher.py` — Orchestration layer
- Lazy singleton: `_get_model() -> SentenceTransformer("all-MiniLM-L6-v2")`
- `embed_text(text) -> np.ndarray`
- `add_memory(text, metadata=None) -> MemoryOrb` — embed → build orb → `storage.save_orb`
- `query(text, top_k=5) -> list[tuple[MemoryOrb, float]]`
  1. Embed query text
  2. `load_all_orbs()`
  3. For each orb: apply decay in-memory (for scoring only, not persisted), compute `cosine_similarity`, compute `compute_gravity`
  4. Sort by gravity descending, take top_k
  5. Apply momentum + update `last_accessed` on top result → `storage.update_orb`
  6. Return `[(orb, gravity_score), ...]`
- `list_all() -> list[MemoryOrb]` — load all, sort by radius ascending

### 5. `cli.py` — Thin argparse wrapper
- Subcommands: `add "text"`, `search "query"`, `list`
- `main()`: calls `storage.initialize_db()` first, then dispatches
- Print user-friendly message on first model load
- Output format for search:
  ```
  Rank 1 [G=0.847]  "the quick brown fox..."   (radius=1.02, id=3)
  ```

### 6. `tests/test_engine.py` — Unit tests (no DB, no model)
- `TestDecay`: radius increases with time, ~zero decay for fresh orb, monotonically increasing, pure (no mutation)
- `TestMomentum`: radius decreases on access, factor respected, pure (no mutation)
- `TestGravity`: formula correctness (hand-calc: `G = 0.8×0.7 + (1/2)×0.3 = 0.71`), closer orb ranks higher, higher similarity ranks higher, zero-radius guard

---

## Critical Files
- `c:\Users\HP\Downloads\MSD\engine.py`
- `c:\Users\HP\Downloads\MSD\storage.py`
- `c:\Users\HP\Downloads\MSD\searcher.py`
- `c:\Users\HP\Downloads\MSD\cli.py`
- `c:\Users\HP\Downloads\MSD\tests\test_engine.py`

---

## Verification
1. `pip install -r requirements.txt`
2. `python cli.py add "The Earth orbits the Sun"`
3. `python cli.py add "The Moon orbits the Earth"`
4. `python cli.py search "planetary orbits"` → should rank Earth/Sun memory higher
5. `python cli.py list` → shows orbs sorted by radius
6. Run again after waiting → radius of non-accessed orbs should have grown (decay visible)
7. `python -m pytest tests/ -v` → all unit tests pass
