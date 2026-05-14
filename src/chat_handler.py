import os
from openai import OpenAI
from typing import Optional, Generator
from src.rag_pipeline import query_knowledge_graph

CHAT_SYSTEM_PROMPT = """You are SilverAI, an intelligent assistant that answers questions \
strictly based on the documents the user has uploaded.

Guidelines:
- Ground every answer in the retrieved source material provided to you.
- If the answer is not found in the sources, say so clearly — do not hallucinate.
- Cite the source document when referencing specific information, e.g. [Source: filename.pdf].
- Be concise and precise for factual questions; elaborate only when depth is asked for.
- If the user asks you to generate a handbook, tell them to use the sidebar option instead.
"""

def _get_client() -> OpenAI:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROK_API_KEY is not set. Please add it to your .env file."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )


def _build_messages(
    user_query: str,
    history: list[dict],
    context_chunks: list[str],
) -> list[dict]:
    """Construct the full messages list for the Grok API call."""
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

    # Inject RAG context as a system-level context block
    if context_chunks:
        context_text = "\n\n---\n\n".join(context_chunks)
        messages.append({
            "role": "system",
            "content": (
                "The following excerpts were retrieved from the user's uploaded documents. "
                "Use them to answer the next question.\n\n"
                f"=== RETRIEVED CONTEXT ===\n{context_text}\n=== END CONTEXT ==="
            ),
        })

    # Append conversation history (last N turns to stay within context limits)
    for turn in history[-10:]:
        messages.append(turn)

    # Append the new user message
    messages.append({"role": "user", "content": user_query})
    return messages


def get_chat_response(
    user_query: str,
    history: Optional[list[dict]] = None,
    top_k: int = 5,
    model: str = "grok-3",
    stream: bool = True,
) -> Generator[str, None, None] | str:
    """
    Generate a RAG-grounded chat response for the user's query.

    Args:
        user_query: The user's question.
        history:    Conversation history as a list of {"role": ..., "content": ...} dicts.
        top_k:      Number of context chunks to retrieve from the knowledge graph.
        model:      Grok model to use.
        stream:     If True, returns a generator that yields response tokens.
                    If False, returns the complete response string.

    Returns:
        A generator (stream=True) or a string (stream=False).
    """
    if history is None:
        history = []

    # Retrieve relevant context from vector store
    context_chunks = query_knowledge_graph(user_query, top_k=top_k)

    if not context_chunks:
        no_context_msg = (
            "I couldn't find any relevant information in your uploaded documents. "
            "Please upload some PDFs first, then ask your question again."
        )
        if stream:
            def _no_context_gen():
                yield no_context_msg
            return _no_context_gen()
        return no_context_msg

    client = _get_client()
    messages = _build_messages(user_query, history, context_chunks)

    if stream:
        def _stream_gen() -> Generator[str, None, None]:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.3,   # Lower temp = more factual / grounded answers
                max_tokens=2048,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _stream_gen()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content


def build_history_turn(role: str, content: str) -> dict:
    """Helper to create a single history turn dict."""
    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")
    return {"role": role, "content": content}
