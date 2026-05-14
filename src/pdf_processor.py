import pdfplumber
import pypdf
import os
from typing import Union
from io import BytesIO

def extract_text_from_pdf(file_source: Union[str, BytesIO]) -> str:
    """Extract all text from a PDF file path or a file-like BytesIO object."""
    text = ""
    try:
        with pdfplumber.open(file_source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber failed ({e}), falling back to pypdf...")
        reader = pypdf.PdfReader(file_source)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def process_pdf(file_source: Union[str, BytesIO], filename: str = "uploaded_file.pdf") -> list[str]:
    """Full pipeline: extract text and return chunks.
    
    Args:
        file_source: A file path string or a BytesIO object (e.g. from Streamlit uploader).
        filename: Optional display name used for logging (useful when file_source is BytesIO).
    """
    display_name = filename if isinstance(file_source, BytesIO) else os.path.basename(file_source)
    print(f"Processing: {display_name}")
    text = extract_text_from_pdf(file_source)
    chunks = chunk_text(text)
    print(f"Done: {len(chunks)} chunks extracted from {display_name}")
    return chunks