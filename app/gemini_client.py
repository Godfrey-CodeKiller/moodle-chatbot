"""
Thin wrapper around the Gemini API for the two things this project needs:
embedding text, and generating a grounded answer from retrieved context.

Uses the `google-genai` SDK: pip install google-genai
"""
from typing import List

from google import genai
from google.genai import types

from app.config import Config

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    task_type should be RETRIEVAL_DOCUMENT when embedding course content
    at ingestion time, and RETRIEVAL_QUERY when embedding a student's
    question at query time. Gemini's embedding model uses this to
    optimize the vector for its role in the search.
    """
    client = get_client()
    result = client.models.embed_content(
        model=Config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values


def embed_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed multiple texts. Simple sequential loop for the prototype;
    batch this properly (single API call with multiple contents, or
    concurrent requests) if ingesting large courses."""
    return [embed_text(t, task_type=task_type) for t in texts]


SYSTEM_INSTRUCTION = """You are a course assistant for a Moodle course. \
Answer the student's question using ONLY the course material excerpts \
provided below as context. Follow these rules strictly:

1. If the answer is not contained in the provided context, say clearly \
that you couldn't find that in the course materials, and suggest the \
student check with their instructor. Do not guess or use outside knowledge.
2. Keep answers concise and direct — students want the answer, not an essay.
3. When you use information from a specific excerpt, mention which module \
or resource it came from (using the module name given with each excerpt).
4. Do not fabricate module names, dates, or facts that aren't in the context.
"""


def generate_answer(question: str, context_chunks: List[dict], conversation_history: List[dict] = None) -> str:
    """
    context_chunks: list of {"module_name": str, "text": str, "source_url": str}
    conversation_history: list of {"role": "user"|"model", "text": str}
    """
    client = get_client()

    context_block = "\n\n".join(
        f"[Source: {c['module_name']}]\n{c['text']}" for c in context_chunks
    )

    if not context_chunks:
        context_block = "(No relevant course material was found for this question.)"

    prompt_parts = []
    if conversation_history:
        for turn in conversation_history[-6:]:  # keep last few turns only
            prompt_parts.append(f"{turn['role'].upper()}: {turn['text']}")

    prompt_parts.append(f"COURSE MATERIAL EXCERPTS:\n{context_block}")
    prompt_parts.append(f"STUDENT QUESTION: {question}")

    full_prompt = "\n\n".join(prompt_parts)

    response = client.models.generate_content(
        model=Config.GENERATION_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return response.text
