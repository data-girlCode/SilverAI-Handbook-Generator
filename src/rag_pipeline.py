import os
import asyncio
import numpy as np
from typing import Optional
from openai import AsyncOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

load_dotenv()

import concurrent.futures

def _run(coro):
    """Run async code in a fresh thread with its own event loop — avoids Streamlit conflicts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()

WORKING_DIR = "./rag_storage"
XAI_BASE_URL = "https://api.x.ai/v1"
LLM_MODEL = "grok-4-1"
EMBED_MODEL = "text-embedding-ada-002"

async def _grok_llm_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GROK_API_KEY"),
        base_url=XAI_BASE_URL,
        **kwargs,
    )
 
async def _grok_embed_func(texts: list[str]) -> np.ndarray:
    client = AsyncOpenAI(
        api_key=os.getenv("GROK_API_KEY"),
        base_url=XAI_BASE_URL,
    )
    response = await client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
    )
    return np.array([item.embedding for item in response.data])

 
_rag_instance: Optional[LightRAG] = None


def _get_rag() -> LightRAG:
    global _rag_instance
    if _rag_instance is None:
        os.makedirs(WORKING_DIR, exist_ok=True)
        _rag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=_grok_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=_grok_embed_func,
            ),
        )
    return _rag_instance
 
async def _async_initialize() -> None:
    """Initialize LightRAG storage backends (must be called before first use)."""
    rag = _get_rag()
    await rag.initialize_storages()
    print("[rag_pipeline] LightRAG storages initialised.")


def initialize() -> None:
    """Synchronous wrapper to initialise LightRAG — call once at app startup."""
    _run(_async_initialize())

 
async def _async_ingest(text: str) -> None:
    rag = _get_rag()
    await rag.ainsert(text)


def ingest_chunks(
    chunks: list[str],
    source_filename: str = "uploaded_file.pdf",
) -> int:
    """
    Build/update the LightRAG knowledge graph from text chunks.

    Args:
        chunks:          List of text strings extracted from a PDF.
        source_filename: Used for logging only.

    Returns:
        Number of chunks successfully ingested.
    """
    if not chunks:
        print("[rag_pipeline] No chunks to ingest.")
        return 0

    # LightRAG works best with larger text blocks — join all chunks into one doc
    full_text = "\n\n".join(chunks)
    print(f"[rag_pipeline] Ingesting '{source_filename}' into LightRAG knowledge graph...")

    try:
        _run(_async_ingest(full_text))
        print(f"[rag_pipeline] Done — {len(chunks)} chunks ingested from '{source_filename}'.")
        return len(chunks)
    except Exception as e:
        print(f"[rag_pipeline] Ingestion failed: {e}")
        return 0


async def _async_query(query: str, mode: str) -> str:
    rag = _get_rag()
    return await rag.aquery(query, param=QueryParam(mode=mode))


def query_knowledge_graph(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",   # "local" | "global" | "hybrid" | "naive"
) -> list[str]:
    """
    Query the LightRAG knowledge graph for context relevant to the query.

    Args:
        query:  The user's question or topic.
        top_k:  Kept for interface compatibility (LightRAG manages retrieval depth).
        mode:   LightRAG query mode:
                  "local"  — entity-focused, great for specific facts
                  "global" — relationship-focused, great for broad topics
                  "hybrid" — combines both (recommended default)
                  "naive"  — plain vector search, no graph reasoning

    Returns:
        A list containing the retrieved answer/context as a single string item,
        ready for use in chat_handler or handbook_gen.
    """
    print(f"[rag_pipeline] Querying LightRAG (mode={mode}): '{query}'")
    try:
        result = _run(_async_query(query, mode))
        if result:
            print(f"[rag_pipeline] Retrieved context ({len(result.split())} words).")
            return [result]
        print("[rag_pipeline] No result returned from LightRAG.")
        return []
    except Exception as e:
        print(f"[rag_pipeline] Query failed: {e}")
        return []


async def _async_finalize() -> None:
    rag = _get_rag()
    await rag.finalize_storages()


def finalize() -> None:
    """Gracefully close LightRAG storage connections. Call at app shutdown."""
    _run(_async_finalize())
    print("[rag_pipeline] LightRAG storages finalised.")
