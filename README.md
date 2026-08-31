# Moodle Course-Content Chatbot (RAG + Gemini)

A minimal, working scaffold for a chatbot that answers student questions
from your Moodle course content, using Google's Gemini API for embeddings
and generation.

## Architecture

```
Moodle (Web Services API)
        │
        ▼
scripts/ingest.py   ──►  pulls course content, chunks it, embeds it,
                          stores it in a local SQLite vector store
        │
        ▼
   data/vectors.db
        │
        ▼
app/main.py (FastAPI)  ──►  /chat endpoint: embeds question, retrieves
                             relevant chunks, calls Gemini, returns answer
        │
        ▼
Moodle HTML block (widget.html)  ──►  floating chat widget students use
```

This uses a **local SQLite + numpy vector store** to keep the prototype
dependency-free and easy to run anywhere. It's a drop-in swap to pgvector
later if you need to scale beyond a few courses — see "Scaling up" below.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

2. Set environment variables:
   ```bash
   export GEMINI_API_KEY="AQ.Ab8RN6L8VCx_RX61BXLa9RNczCWz1Wjqbg_IUQuUw8upXSEH_g"
   export MOODLE_URL="https://academy.ikusasasolutions.co.za/"
   export MOODLE_TOKEN="637b78cb4be39cc66b2ffe56224f83cb"
   ```

   Getting a Moodle token: Site Administration → Server → Web services →
   Manage tokens. The token's associated user needs a role with permission
   to read course content (`webservice/rest:use` capability, plus the
   relevant `core_course_get_contents` etc. functions enabled in the
   external service).

3. Run ingestion for a course:
   ```bash
   python scripts/ingest.py --course-id 123
   ```

4. Start the backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. Test it:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"course_id": 123, "question": "When is the midterm?"}'
   ```

6. Embed `widget/widget.html` in a Moodle HTML block (edit the `API_URL`
   and `COURSE_ID` constants at the top of the file first).

## Files

- `app/main.py` — FastAPI backend, `/chat` endpoint
- `app/gemini_client.py` — thin wrapper around Gemini embeddings + generation
- `app/vector_store.py` — SQLite-backed vector store (cosine similarity search)
- `app/config.py` — environment/config loading
- `scripts/ingest.py` — pulls Moodle content, chunks, embeds, stores
- `scripts/moodle_client.py` — Moodle Web Services API wrapper
- `widget/widget.html` — drop-in chat widget for a Moodle HTML block

## Scaling up (when you outgrow the prototype)

- **Vector store**: swap `vector_store.py` for pgvector-backed Postgres.
  The interface (`add_chunks`, `search`) is the same — only the
  implementation needs to change.
- **Auth**: currently the `/chat` endpoint takes `course_id` at face value.
  Before real deployment, verify the caller is an enrolled student — e.g.
  validate a Moodle session/token and call `core_enrol_get_enrolled_users`
  server-side to confirm enrollment.
- **Caching**: cache embeddings for unchanged content, and consider an
  FAQ cache for repeat questions to cut Gemini API costs.
- **Rate limiting**: add per-user/IP limits so a single student (or bug)
  can't run up unbounded API costs.
- **Moodle plugin**: replace the HTML-block widget with a proper
  `block_` or `local_` Moodle plugin once this proves useful — it can
  auto-pass the logged-in user's course context instead of hardcoding it.

## Known gaps in this scaffold (intentional, for a first pass)

- No auth/enrollment check yet (see above) — don't deploy publicly as-is.
- No re-ranking of retrieved chunks beyond cosine similarity + threshold.
- No file-type parsing for PDFs/DOCX linked as Moodle resources yet —
  `moodle_client.py` has a TODO marker where you'd add `pypdf`/`python-docx`
  extraction for file resources.
