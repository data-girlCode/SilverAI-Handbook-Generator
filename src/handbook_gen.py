import os
from openai import OpenAI
from typing import Optional 

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


LONGWRITER_SYSTEM_PROMPT = """You are SilverAI, an expert technical writer and knowledge synthesiser.
Your task is to produce a comprehensive, well-structured handbook of approximately 20,000 words.

Follow the LongWriter technique:
1. First, silently plan a detailed outline with chapters, sub-sections, and estimated word counts.
2. Then write every section in full — do NOT truncate, summarise, or use placeholders.
3. Each chapter must be thorough, detailed, and complete before moving to the next.
4. Use clear markdown headings (# ## ###), bullet points, numbered lists, code blocks where relevant.
5. Cite source documents using the format [Source: <filename>] whenever drawing from them.
6. The output must feel like a professional reference handbook — not a blog post or summary.

Structure your handbook as follows:
- Title Page
- Table of Contents (with estimated page references)
- Introduction (500–800 words)
- 6–10 in-depth chapters (~2,000–3,000 words each)
- Conclusion & Key Takeaways (~500 words)
- Glossary of Terms (~500 words)
"""


def _build_user_prompt(topic: str, context_chunks: Optional[list[str]] = None) -> str:
    """Construct the user prompt, optionally injecting RAG context."""
    prompt = f"Write a comprehensive 20,000-word handbook on the topic: **{topic}**\n\n"

    if context_chunks:
        # Inject retrieved context — limit to avoid exceeding context window
        combined_context = "\n\n---\n\n".join(context_chunks[:30])
        prompt += (
            "Use the following excerpts from the user's uploaded documents as your "
            "primary knowledge source. Ground your writing in this material and cite "
            "the relevant passages where appropriate.\n\n"
            "=== SOURCE MATERIAL ===\n"
            f"{combined_context}\n"
            "=== END OF SOURCE MATERIAL ===\n\n"
        )

    prompt += (
        "Now write the full 20,000-word handbook. Begin immediately with the Title Page. "
        "Do not stop until the Glossary is complete."
    )
    return prompt

 
def generate_handbook(
    topic: str,
    context_chunks: Optional[list[str]] = None,
    model: str = "grok-4-1",
    stream: bool = True,
) -> str:
    """
    Generate a ~20,000-word handbook on the given topic using the Grok API.

    Args:
        topic:          The subject/title of the handbook.
        context_chunks: Optional list of text chunks retrieved from uploaded PDFs.
        model:          The Grok model to use (default: grok-3).
        stream:         If True, streams and prints progress; returns full text.

    Returns:
        The complete handbook as a string.
    """
    client = _get_client()
    user_prompt = _build_user_prompt(topic, context_chunks)

    messages = [
        {"role": "system", "content": LONGWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"[handbook_gen] Starting handbook generation on: '{topic}'")
    print(f"[handbook_gen] Model: {model} | Context chunks: {len(context_chunks) if context_chunks else 0}")

    handbook_text = ""

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=16000,   # Increase if your plan supports it
            temperature=0.7,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                handbook_text += delta
                print(delta, end="", flush=True)
        print()  # newline after streaming finishes
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=16000,
            temperature=0.7,
        )
        handbook_text = response.choices[0].message.content

    print(f"\n[handbook_gen] Done — {len(handbook_text.split()):,} words generated.")
    return handbook_text
 
def save_handbook(handbook_text: str, topic: str, output_dir: str = "output") -> str:
    """Save the handbook to a .md file and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = topic.lower().replace(" ", "_").replace("/", "-")[:60]
    file_path = os.path.join(output_dir, f"{safe_name}_handbook.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(handbook_text)
    print(f"[handbook_gen] Handbook saved to: {file_path}")
    return file_path
