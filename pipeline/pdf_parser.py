"""Page-aware PDF extraction and overlapping word chunks."""
import hashlib
import time
import pymupdf
from pipeline.configuration import positive_integer


class PDFParser:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        positive_integer(chunk_size, "chunk_size")
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, file_bytes: bytes, filename: str) -> dict:
        started = time.perf_counter()
        if not file_bytes:
            raise ValueError(f"{filename}: the file is empty.")
        try:
            with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ValueError("Password-protected PDFs are not supported.")
                pages = [{"page": i + 1, "text": page.get_text("text", sort=True)}
                         for i, page in enumerate(doc)]
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"{filename}: cannot read PDF ({exc}).") from exc
        doc_id = hashlib.sha256(file_bytes).hexdigest()
        chunks = self._create_chunks(pages, filename, doc_id)
        if not chunks:
            raise ValueError(f"{filename}: no extractable text. Scanned PDFs need OCR first.")
        return {
            "doc_id": doc_id, "filename": filename, "total_pages": len(pages),
            "pages": pages,
            "raw_text": "\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in pages),
            "chunks": chunks,
            "parse_duration_sec": round(time.perf_counter() - started, 4),
        }

    def _create_chunks(self, pages: list[dict], filename: str, doc_id: str = "") -> list[dict]:
        chunks = []
        for page in pages:
            words = page["text"].split()
            for offset in range(0, len(words), self.chunk_size - self.chunk_overlap):
                text = " ".join(words[offset:offset + self.chunk_size])
                if text:
                    chunks.append({
                        "chunk_id": f"{doc_id or filename}_p{page['page']}_w{offset}",
                        "doc_id": doc_id or filename, "filename": filename,
                        "page": page["page"], "text": text,
                    })
                if offset + self.chunk_size >= len(words):
                    break
        return chunks
