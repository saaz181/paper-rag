import re
import json

def clean_text(text):
    # Fix spaces/tabs but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove citation patterns such as [23]
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    return text.strip()

def is_noisy_chunk(text):
    text_lower = text.lower()
    words = text.split()

    if len(words) < 20:
        return True

    # Explicit section names
    if text_lower.startswith("references"):
        return True

    if text_lower.startswith("acknowledgments"):
        return True

    # Too many URLs / arXiv references
    url_count = len(
        re.findall(
            r"https?://|arxiv|doi:",
            text_lower
        )
    )

    if url_count >= 2:
        return True

    # Reference-like publication patterns
    citation_patterns = len(
        re.findall(
            r"\b(19|20)\d{2}\b",
            text
        )
    )

    if citation_patterns >= 6:
        return True

    # Table-heavy / numerical chunks
    digit_ratio = (
        sum(c.isdigit() for c in text)
        / max(len(text), 1)
    )

    if digit_ratio > 0.12:
        return True

    return False


def chunk_paragraph(text, max_words=150, overlap=25):

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = []

    for paragraph in paragraphs:

        words = paragraph.split()

        # If one paragraph itself is huge,
        # split it into smaller overlapping pieces
        if len(words) > max_words:

            if current:
                chunks.append(" ".join(current))
                current = []

            step = max_words - overlap

            for i in range(0, len(words), step):
                piece = words[i:i + max_words]

                if piece:
                    chunks.append(" ".join(piece))

            continue

        # Normal paragraph
        if len(current) + len(words) <= max_words:
            current.extend(words)

        else:
            if current:
                chunks.append(" ".join(current))

            current = words

    if current:
        chunks.append(" ".join(current))

    return chunks



def chunk_docs(documents, overlap=25, max_words=180):

    all_chunks = []

    for doc_name, doc in documents.items():

        for page in doc['pages']:

            page_number = page['page']
            text = page['text']
            text = clean_text(text)

            # chunk word wise
            # chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

            # chunk paragraph-wise
            chunks = chunk_paragraph(text, max_words=max_words, overlap=overlap)

            clean_chunks = [chunk for chunk in chunks if not is_noisy_chunk(chunk) ]

            for chunk_idx, chunk in enumerate(clean_chunks):
                all_chunks.append({
                    "document": doc_name,
                    "page": page_number,
                    "chunk_id": chunk_idx,
                    "text": chunk
                })


    return all_chunks


def save_chunks(
    chunks,
    path="chunks.json"
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )