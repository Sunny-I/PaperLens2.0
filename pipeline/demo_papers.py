"""Load the user's PDFs from the project's sample_papers directory."""
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_papers"


def load_demo_papers():
    paths = sorted(p for p in SAMPLE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf") if SAMPLE_DIR.is_dir() else []
    if not paths:
        raise ValueError("No PDF files found in sample_papers. Add PDFs to that folder and try again.")
    return [(path.name, path.read_bytes()) for path in paths]
