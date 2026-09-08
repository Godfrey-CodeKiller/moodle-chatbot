from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Config
from app.gemini_client import embed_text, generate_answer
from app.vector_store import VectorStore

app = FastAPI(title="Moodle Course Chatbot")

# CORS: allow your Moodle domain to call this API from the browser widget.
# Tighten this to your actual Moodle origin before going anywhere near production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace with your Moodle site's origin
    allow_methods=["POST"],
    allow_headers=["*"],
)

store = VectorStore(Config.VECTOR_DB_PATH)


class ConversationTurn(BaseModel):
    role: str  # "user" or "model"
    text: str


class ChatRequest(BaseModel):
    course_id: int
    question: str
    conversation_history: Optional[List[ConversationTurn]] = None


class Source(BaseModel):
    module_name: str
    source_url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # NOTE: no enrollment/auth check yet. Before deploying beyond a local
    # test, verify the caller is actually enrolled in course_id — e.g. by
    # validating a Moodle session token and calling
    # core_enrol_get_enrolled_users server-side. See README "Scaling up".

    if store.count_for_course(req.course_id) == 0:
        return ChatResponse(
            answer=(
                "This course hasn't been indexed yet, so I don't have any "
                "material to search. Please contact your instructor or admin."
            ),
            sources=[],
        )

    query_embedding = embed_text(req.question, task_type="RETRIEVAL_QUERY")

    results = store.search(
        course_id=req.course_id,
        query_embedding=query_embedding,
        top_k=Config.TOP_K,
        min_score=Config.SIMILARITY_THRESHOLD,
    )

    context_chunks = [
        {"module_name": r.module_name, "text": r.text, "source_url": r.source_url}
        for r in results
    ]

    history = (
        [{"role": t.role, "text": t.text} for t in req.conversation_history]
        if req.conversation_history
        else None
    )

    answer = generate_answer(req.question, context_chunks, history)

    sources = [
        Source(module_name=r.module_name, source_url=r.source_url, score=round(r.score, 3))
        for r in results
    ]

    return ChatResponse(answer=answer, sources=sources)