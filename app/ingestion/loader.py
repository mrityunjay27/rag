from pathlib import Path


def load_document(file_path: str) -> str:
    """Take a file path and return the content of the document as a string."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    return path.read_text(encoding="utf-8")

