"""Stage 2: PDFs on disk turn into leaf nodes.

A leaf node is one chunk of text plus where it came from. These are the
bottom layer of the RAPTOR tree, and they are also the entire index used by
the flat baseline, so both systems read exactly the same text.
"""

import re
from pathlib import Path

from pypdf import PdfReader

from . import config


def clean_text(text: str) -> str:
    text = text.replace("­", "")               # soft hyphen
    text = text.replace("\t", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # rejoin words split at a line end
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        lines.append(line.strip())
    return "\n".join(lines).strip()


def furniture_key(line: str) -> str:
    """A line with digits removed, so "Page 3" and "Page 4" compare as equal."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip().lower()


def strip_page_furniture(pages: list[dict]) -> tuple[list[dict], int]:
    """Remove running headers, footers and bare page numbers.

    A line that appears on at least half the pages (and on three or more) is
    page furniture, not content. Left in, it sits inside every chunk, and the
    extractive summarizer picks it as the most "representative" sentence of
    the whole document precisely because it is everywhere. Short lines are
    never treated as furniture, so a heading like "Step 1" or "Given:" that
    happens to recur is kept.
    """
    if len(pages) < 3:
        return pages, 0

    pages_containing = {}
    for page in pages:
        seen = set()
        for line in page["text"].split("\n"):
            key = furniture_key(line)
            if len(key) >= 12 and key not in seen:
                seen.add(key)
                pages_containing[key] = pages_containing.get(key, 0) + 1

    threshold = max(3, len(pages) // 2)
    repeated = set()
    for key, count in pages_containing.items():
        if count >= threshold:
            repeated.add(key)

    cleaned = []
    for page in pages:
        kept = []
        for line in page["text"].split("\n"):
            if furniture_key(line) in repeated:
                continue
            if re.fullmatch(r"\s*\d{1,4}\s*", line):
                continue
            kept.append(line)
        cleaned.append({"page": page["page"], "text": "\n".join(kept).strip()})
    return cleaned, len(repeated)


def load_pdf_pages(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    page_number = 1
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append({"page": page_number, "text": clean_text(raw)})
        page_number = page_number + 1
    pages, _ = strip_page_furniture(pages)
    return pages


def recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    """Split on the largest natural boundary that keeps pieces under max_chars."""
    if len(text) <= max_chars:
        return [text]

    if len(separators) == 0:
        pieces = []
        start = 0
        while start < len(text):
            pieces.append(text[start:start + max_chars])
            start = start + max_chars
        return pieces

    separator = separators[0]
    finer = separators[1:]
    parts = text.split(separator)

    chunks = []
    current = ""
    for part in parts:
        if current == "":
            candidate = part
        else:
            candidate = current + separator + part

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current != "":
                chunks.append(current)
            if len(part) > max_chars:
                for sub in recursive_split(part, max_chars, finer):
                    chunks.append(sub)
                current = ""
            else:
                current = part

    if current != "":
        chunks.append(current)
    return chunks


def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Start each chunk with the tail of the one before it."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    joined = [chunks[0]]
    position = 1
    while position < len(chunks):
        tail = chunks[position - 1][-overlap:]
        # Start the overlap at a word boundary, so no chunk opens mid-word
        # ("pensive or slow..." instead of "expensive or slow...").
        first_space = tail.find(" ")
        if 0 <= first_space < len(tail) - 1:
            tail = tail[first_space + 1:]
        joined.append(tail + " " + chunks[position])
        position = position + 1
    return joined


def chunk_document(pages: list[dict], source: str) -> list[dict]:
    """Chunk a document, remembering which pages each chunk covers.

    Pages are joined first so a chunk can run across a page break, then each
    chunk's pages are recovered by looking up where it started and ended.
    """
    full_text = ""
    page_spans = []                       # (start, end, page) in full_text
    for page in pages:
        if page["text"] == "":
            continue
        start = len(full_text)
        full_text = full_text + page["text"] + "\n\n"
        page_spans.append((start, len(full_text), page["page"]))

    pieces = recursive_split(full_text, config.CHUNK_CHARS, config.SPLIT_SEPARATORS)
    pieces = add_overlap(pieces, config.CHUNK_OVERLAP)

    records = []
    cursor = 0
    for piece in pieces:
        stripped = piece.strip()
        found = full_text.find(stripped[:60], cursor) if len(stripped) >= 60 else -1
        if found == -1:
            found = cursor
        start = found
        end = start + len(stripped)
        cursor = max(cursor, start + 1)

        pages_touched = []
        for span_start, span_end, page_number in page_spans:
            if start < span_end and end > span_start:
                pages_touched.append(page_number)
        if len(pages_touched) == 0:
            pages_touched = [page_spans[0][2]] if len(page_spans) > 0 else [1]

        if len(stripped) < 80:            # skip near-empty fragments
            continue

        records.append({
            "text": stripped,
            "source": source,
            "pages": pages_touched,
        })
    return records


def build_leaves(raw_dir: Path = None) -> list[dict]:
    """Every PDF in data/raw becomes a list of leaf nodes."""
    directory = raw_dir if raw_dir is not None else config.RAW_DIR
    leaves = []
    counter = 0

    paths = sorted(directory.glob("*.pdf"))
    for path in paths:
        pages = load_pdf_pages(path)
        for record in chunk_document(pages, path.name):
            counter = counter + 1
            leaves.append({
                "id": "L" + str(counter).zfill(4),
                "level": 0,
                "kind": "leaf",
                "text": record["text"],
                "source": record["source"],
                "pages": record["pages"],
                "children": [],
                "member_count": 1,
            })
    return leaves
