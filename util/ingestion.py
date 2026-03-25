"""
Document ingestion: extract text from PDF, DOCX, or TXT files,
then tokenize into sentences for downstream processing.
"""

import io
from pathlib import Path

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in {".docx", ".doc"}:
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .docx, .doc, .txt")


def extract_sentences(text: str) -> list[str]:
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
