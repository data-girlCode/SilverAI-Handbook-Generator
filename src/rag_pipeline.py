import os
import json
import numpy as np
from typing import Optional
from supabase import create_client, Client

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("[rag_pipeline] Loading embedding model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[rag_pipeline] Embedding model loaded.")
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors."""
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()

def _get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(url, key)

_in_memory_store: list[dict] = []  

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)

def ingest_chunks(
    chunks: list[str],
    source_filename: str = "uploaded_file.pdf",
    use_supabase: bool = True,
) -> int:
    """
    Embed and store text chunks.

    Args:
        chunks:          List of text strings to embed and store.
        source_filename: Name of the source PDF (stored as metadata).
        use_supabase:    If True, stores in Supabase; otherwise in-memory only.

    Returns:
        Number of chunks successfully stored.
    """
    if not chunks:
        print("[rag_pipeline] No chunks to ingest.")
        return 0

    print(f"[rag_pipeline] Embedding {len(chunks)} chunks from '{source_filename}'...")
    embeddings = embed_texts(chunks)

    if use_supabase:
        try:
            client = _get_supabase_client()
            rows = [
                {
                    "content": chunk,
                    "embedding": embedding,
                    "source": source_filename,
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]
            # Supabase table: 'documents' with columns (id, content, embedding, source)
            response = client.table("documents").insert(rows).execute()
            count = len(response.data) if response.data else 0
            print(f"[rag_pipeline] Stored {count} chunks in Supabase.")
            return count
        except Exception as e:
            print(f"[rag_pipeline] Supabase insert failed ({e}). Falling back to in-memory store.")

    # In-memory fallback
    for chunk, embedding in zip(chunks, embeddings):
        _in_memory_store.append({
            "content": chunk,
            "embedding": embedding,
            "source": source_filename,
        })
    print(f"[rag_pipeline] Stored {len(chunks)} chunks in memory.")
    return len(chunks)

def query_knowledge_graph(
    query: str,
    top_k: int = 5,
    use_supabase: bool = True,
) -> list[str]:
    """
    Retrieve the top-k most relevant text chunks for a given query.

    Args:
        query:        The user's question or topic.
        top_k:        Number of chunks to return.
        use_supabase: If True, runs vector search on Supabase; otherwise in-memory.

    Returns:
        List of relevant text chunks (strings), most relevant first.
    """
    print(f"[rag_pipeline] Querying for: '{query}'")
    query_embedding = embed_texts([query])[0]

    if use_supabase:
        try:
            client = _get_supabase_client()
            # Uses Supabase's match_documents RPC (pgvector cosine similarity)
            response = client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                },
            ).execute()
            results = [row["content"] for row in response.data] if response.data else []
            if results:
                print(f"[rag_pipeline] Retrieved {len(results)} chunks from Supabase.")
                return results
            print("[rag_pipeline] No Supabase results, falling back to in-memory search.")
        except Exception as e:
            print(f"[rag_pipeline] Supabase query failed ({e}). Falling back to in-memory search.")

    # In-memory fallback: cosine similarity
    if not _in_memory_store:
        print("[rag_pipeline] In-memory store is empty. Please ingest PDFs first.")
        return []

    scored = [
        (_cosine_similarity(query_embedding, item["embedding"]), item["content"])
        for item in _in_memory_store
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [content for _, content in scored[:top_k]]
    print(f"[rag_pipeline] Retrieved {len(results)} chunks from in-memory store.")
    return results

def clear_memory_store() -> None:
    """Clear all chunks from the in-memory vector store."""
    global _in_memory_store
    _in_memory_store = []
    print("[rag_pipeline] In-memory store cleared.")
