# 🛠 Project Specification: Orbital Memory System (OMS) Implementation

هذا الملف مخصص لتوجيه "كلود" (Claude Code) لبناء النواة البرمجية لنظام الذاكرة المداري. المشروع مفتوح المصدر وموجه للاستخدام العام كمحرك ذاكرة ديناميكي.

## 1. الأهداف التقنية (Technical Goals)
- بناء محرك ذاكرة محلي (Local-First) يعتمد على المتجهات (Vectors).
- تنفيذ منطق "الجاذبية المدارية" لاسترجاع البيانات.
- تنفيذ نظام "الاضمحلال الزمني" (Decay) والزخم (Momentum) للأجرام المعلوماتية.

## 2. الهيكل البرمجي (Core Components)

### أ. فئة الجرم المعلوماتي (MemoryOrb Class)
يجب أن يحتوي كل جرم على:
- content: النص الأصلي.
- vector: تمثيل عددي (Embedding) للنص.
- orbital_radius: قيمة عائمة تبدأ من 1.0 (تمثل القرب من المركز).
- last_accessed: طابع زمني.
- metadata: بيانات إضافية اختيارية.

### ب. المعادلات الرياضية للتنفيذ
يجب استخدام المعادلات التالية في منطق البرمجة:

1. الجاذبية الكلية ($G_t$):
$$G_t = (Semantic\_Similarity \cdot w_s) + \left( \frac{1}{Radius} \cdot w_r \right)$$

2. الاضمحلال المداري (Decay):
$$R_{new} = R_{old} + \ln(1 + \Delta t) \cdot \lambda$$

## 3. مراحل التنفيذ المطلوبة من "كلود" (Implementation Steps)

### المرحلة الأولى: النواة والفيزياء (The Core & Physics)
1. إنشاء بيئة Python افتراضية وتثبيت numpy و scikit-learn (للحسابات) و sentence-transformers (للمتجهات).
2. كتابة ملف engine.py يحتوي على MemoryOrb ودالة تحديث المدارات.

### المرحلة الثانية: قاعدة البيانات (Storage Layer)
1. استخدام SQLite لتخزين البيانات.
2. إنشاء جدول orbs يحتوي على أعمدة للبيانات والخصائص الفيزيائية (Radius, Mass, etc).
3. تنفيذ دالة save و load لضمان استمرارية الذاكرة.

### المرحلة الثالثة: محرك البحث المداري (The Searcher)
1. بناء دالة query(text) تقوم بـ:
    - تحويل النص إلى Vector.
    - حساب $G_t$ لكل الأجرام في القاعدة.
    - ترتيب النتائج وإعادة الأقرب جاذبية.
    - تحديث orbital_radius للجرم المختار ليقترب من المركز (Momentum).

### المرحلة الرابعة: الواجهة البرمجية (CLI/API)
1. بناء واجهة سطر أوامر (CLI) بسيطة لإضافة المعلومات والبحث عنها.
2. (اختياري لاحقاً) بناء FastAPI لربط النظام بواجهات Flutter.

## 4. تعليمات خاصة لـ "كلود"
- Clean Code: اتبع معايير PEP 8.
- Modularity: اجعل منطق الحسابات الفيزيائية منفصلاً عن منطق تخزين البيانات.
- Documentation: أضف تعليقات تشرح كل دالة، خاصة الأجزاء الرياضية.
- Testing: اكتب اختبارات وحدة (Unit Tests) للتأكد من أن نصف القطر (Radius) يزداد مع الوقت وينقص عند الاستخدام.

---
جاهز للتنفيذ؟ ابدأ ببناء المرحلة الأولى الآن.


Great. Let me think through both the hybrid architecture and the decay question together, because they're actually connected.

The Hybrid Architecture — Confirmed Understanding
```
                    ┌─────────────────────────────────────┐
  Raw Text ──►      │  Claude API: generate summary        │
                    └────────────────┬────────────────────┘
                                     │ "Key facts in 2-3 sentences"
                    ┌────────────────▼──────────┐    ┌─────────────────┐
                    │      ChromaDB             │    │    SQLite        │
                    │  (The Search Head)        │    │  (The Drawer)    │
                    │                           │    │                  │
                    │  orb_id (UUID)            │◄──►│  orb_id (FK)     │
                    │  summary_vector (384-dim) │    │  full_text       │
                    │  summary (text)           │    │  created_at      │
                    │  metadata: {              │    │                  │
                    │    orbital_radius,        │    └─────────────────┘
                    │    last_accessed,         │
                    │    decay_rate             │
                    │  }                        │
                    └───────────────────────────┘
Key insight: ChromaDB holds the "searchable brain", SQLite holds the "full archive". They're linked by the same orb_id UUID.
```
The Decay Background Process — 4 Options
Option 1: Lazy Decay (No background process at all)
This is actually the most correct approach mathematically:


# The radius is always computable from stored_radius + last_accessed
# You never need to "run" decay — you just compute it when needed

def get_effective_radius(stored_radius: float, last_accessed: datetime) -> float:
    delta_t = (datetime.utcnow() - last_accessed).total_seconds() / 3600
    return stored_radius + math.log(1 + delta_t) * 0.01
Why it works: R_new = R_old + ln(1 + Δt) × λ is deterministic. If you know last_accessed, you can always compute the exact current radius. Zero background process, zero extra writes, 100% accurate.

The stored orbital_radius becomes just a baseline — it only changes when momentum is applied.

Option 2: threading.Timer daemon (lightweight in-process)

# decay_worker.py
import threading, math
from datetime import datetime

def _decay_tick(interval: int):
    """Fetch all metadata, recompute radii, batch-update ChromaDB."""
    # Only fetches metadata (no vectors) — very cheap
    _sync_decay_to_db()
    t = threading.Timer(interval, _decay_tick, args=[interval])
    t.daemon = True   # <-- dies automatically when main process exits
    t.start()

def start(interval_seconds: int = 1800):  # every 30 min
    threading.Timer(interval_seconds, _decay_tick, args=[interval_seconds]).start()
Pros: Zero extra deps, zero CPU between ticks, dies with the process.

Cons: Only runs while your Python app is alive. If app is off, radii in DB are stale (but lazy decay still gives correct scores at query time).

Option 3: APScheduler (robust in-process)

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(_sync_decay_to_db, 'interval', minutes=30)
scheduler.start()
More robust — handles missed runs, has error recovery. Good if you're building this into a long-running FastAPI server later.

Option 4: Separate script + OS scheduler
A standalone python decay_worker.py run by Windows Task Scheduler every 30 minutes. Fully independent of the main process. Best for production but requires OS setup.

My Recommendation: Lazy + Optional Background Sync

Query time:    Always use lazy decay for scoring (always accurate)
Background:    Optional threading.Timer to sync radii to DB (for analytics/visualization)
This means:

Search is always correct even if the background process hasn't run
Background process is optional — it just keeps the DB values fresh for display in cli.py list
Zero wasted computation — decay only runs when you actually search or the timer ticks
New Module Structure
```
MSD/
├── engine.py              # unchanged — pure physics math
├── storage/
│   ├── __init__.py
│   ├── chroma_store.py    # ChromaDB: add, query, update_radius
│   └── sqlite_store.py    # SQLite: save/fetch full text by orb_id
├── memory_agent.py        # Claude API: summarize + chat with memory
├── searcher.py            # ANN search → gravity rerank → momentum
├── decay_worker.py        # Optional background radius sync
├── cli.py                 # add / search / chat / list
└── requirements.txt
```
The Key Question Before We Code
The decay worker needs to call ChromaDB to batch-update metadata. ChromaDB's metadata update API is limited — you can't update just one metadata field; you have to upsert the whole document. This means:


_sync_decay_to_db():
    1. collection.get(include=["metadatas"])  # no vectors, lightweight
    2. For each orb: compute new radius from last_accessed
    3. collection.update(ids=[...], metadatas=[{...updated...}])
This is fine. No vectors involved = fast.

Two things to confirm before I start building:

For the Claude API summarization — do you have an Anthropic API key already? And should the summary be generated using claude-haiku-4-5 (fast/cheap) for ingestion, and claude-sonnet-4-6 for the chat responses?

For the background decay — do you want it running automatically when cli.py starts, or as a separate opt-in process?




```


Good — those two answers clarify everything. Let me map out the full design before we touch code.

Multi-Provider Design ("Template for All Models")
The core idea: one abstract interface, two implementations, all model names in one config file.


config.py          ← model names, weights, paths (change one file to swap models)
    │
llm_client.py      ← abstract LLMClient + AnthropicClient + GoogleClient
    │
memory_agent.py    ← calls llm_client, never knows which provider is active
config.py will look like:


MODELS = {
    "summarize": {
        "anthropic": "claude-haiku-4-5-20251001",   # fast + cheap for ingestion
        "google":    "gemini-1.5-flash",
    },
    "chat": {
        "anthropic": "claude-sonnet-4-6",            # powerful for conversation
        "google":    "gemini-1.5-pro",
    },
}
PROVIDER = os.getenv("OMS_LLM_PROVIDER", "anthropic")  # switch via .env
Switch between Claude and Gemini by just changing one line in .env.

The Hybrid Storage — Two DBs, One orb_id

                        orb_id (UUID) — the shared key
                              │
          ┌───────────────────┴──────────────────┐
          ▼                                       ▼
     ChromaDB                                  SQLite
  (The Search Head)                         (The Drawer)
  ─────────────────                         ───────────
  summary (text)                            orb_id (PK)
  summary_vector                            full_text
  orbital_radius  ◄── decay/momentum        created_at
  last_accessed
  decay_rate
The Decay Worker — Standalone, Open Source Friendly

python decay_worker.py              # runs every 30 min forever
python decay_worker.py --interval 60 # every 60 min
python decay_worker.py --once        # run once and exit (good for cron)
Why lazy decay + sync worker together is the right pattern:

searcher.py always uses lazy decay for scoring → always accurate at query time
The worker just writes the computed values back to ChromaDB periodically
This keeps the DB values fresh for list and analytics, without the searcher depending on the worker
Complete New File Structure

MSD/
├── .env.example              # API key template (open source safe)
├── config.py                 # All model names, physics constants, paths
├── engine.py                 # UNCHANGED — pure physics math
│
├── llm_client.py             # NEW — abstract LLMClient, AnthropicClient, GoogleClient
├── memory_agent.py           # NEW — ingest() + chat() — the LLM↔OMS bridge
│
├── storage/
│   ├── __init__.py
│   ├── chroma_store.py       # NEW — ChromaDB: add, ANN search, update radius
│   └── sqlite_store.py       # NEW — SQLite: save/fetch full text by orb_id
│
├── searcher.py               # UPDATED — ChromaDB ANN → gravity rerank → momentum
├── cli.py                    # UPDATED — add chat command + --provider flag
├── decay_worker.py           # NEW — standalone background decay sync
│
├── tests/
│   ├── __init__.py
│   └── test_engine.py        # UNCHANGED — pure unit tests still pass
│
└── requirements.txt          # UPDATED — add chromadb, anthropic, google-generativeai
The Full Data Flow
Ingestion (oms add "text" or via memory_agent.ingest()):


raw_text
  → Claude/Gemini: "summarize into key semantic facts"
  → summary text
  → SentenceTransformer("all-MiniLM-L6-v2"): embed(summary)
  → summary_vector
  → ChromaDB.add(orb_id, summary_vector, summary, {radius=1.0, last_accessed=now})
  → SQLite.save(orb_id, raw_text)
Search (oms search "query" or memory_agent.chat()):


query_text
  → embed(query)
  → ChromaDB.query(query_vector, n=20)   # fast ANN
  → for each candidate:
      effective_radius = stored_radius + ln(1 + Δt) × λ  # lazy decay
      sim = 1 - cosine_distance
      G_t = sim × 0.7 + (1/effective_radius) × 0.3
  → sort by G_t → top-K
  → top result: apply momentum → update ChromaDB
  → fetch full_text from SQLite
  → return results
Chat (oms chat):


user_message
  → query_memories(user_message, top_k=3)
  → memory_context = format(top memories)
  → llm.chat(messages, memory_context)
  → response
  → ingest("User: ...\nAssistant: ...")  # store exchange as new memory
What Stays the Same for Open Source
engine.py — pure math, zero deps, easy to understand and contribute to
tests/test_engine.py — all 14 tests still pass unchanged
.env.example committed, .env in .gitignore
decay_worker.py is a standalone script — anyone can run it independently
```