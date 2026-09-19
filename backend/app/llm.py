"""One small interface over four ways of getting text generated.

    complete(prompt, system)         -> str
    stream_complete(prompt, system)  -> yields str pieces

Providers: "offline" (no model at all), "ollama" (local), "anthropic", "openai".
The offline provider is extractive: it picks the most representative sentences
out of the text it is given. It is deterministic, free, and always available,
which is what makes the project runnable with zero setup.
"""

import json
import re
import urllib.error
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

SUMMARY_SYSTEM = (
    "You summarize study material for a student revising for an exam. "
    "Write plainly and keep every technical term, number, formula and unit exactly as given. "
    "Never invent facts that are not in the text."
)

SUMMARY_INSTRUCTION = (
    "Below are several passages from the same course material. Write a single summary of "
    "about 150 words that captures what topics they cover and the key facts, formulas and "
    "results in them. Start directly with the content, with no preamble.\n\n"
)

# Higher levels summarise summaries. Without a different instruction a small
# model tends to copy the longest child verbatim, which makes the upper levels
# of the tree worthless: a level-2 node must describe the whole span, not
# repeat one section of it.
HIGHER_LEVEL_INSTRUCTION = (
    "Below are summaries of DIFFERENT sections of one course. Write a single overview of "
    "about 150 words that covers ALL of the sections together. Name the topics each "
    "section deals with. Do not copy any sentence from the input, and do not focus on "
    "one section at the expense of the others. Start directly with the content.\n\n"
)


# ---------------------------------------------------------------- offline ----

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        cleaned = part.strip()
        if len(cleaned) > 25:
            sentences.append(cleaned)
    return sentences


def looks_like_code(sentence: str) -> bool:
    """Code and formulas make poor summary sentences: they say what, not about what."""
    stripped = sentence.strip()
    for prefix in ["#", "def ", "import ", "from ", "class ", "return ", ">>>", "$ "]:
        if stripped.startswith(prefix):
            return True
    symbols = 0
    for character in stripped:
        if character in "=(){}[]<>_;|\\":
            symbols = symbols + 1
    return symbols > 0.08 * max(1, len(stripped))


def unique_prose_sentences(text: str, skip_covers: bool = True) -> list[str]:
    """Sentences in order, without repeats and without code.

    Chunk overlap puts the same sentence into two neighbouring chunks, so a
    cluster's text contains duplicates; without this the summary can quote
    the same line twice.
    """
    seen = set()
    kept = []
    for sentence in split_sentences(text):
        key = re.sub(r"\s+", " ", sentence).strip().lower()
        if key in seen or looks_like_code(sentence):
            continue
        # A child summary's own "Covers: ..." line; the parent writes a fresh one.
        if skip_covers and key.startswith("covers:"):
            continue
        seen.add(key)
        kept.append(sentence)
    return kept


def key_terms(sentences: list[str], count: int = 8) -> list[str]:
    """The terms that carry the most weight across these sentences."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[A-Za-z][A-Za-z-]{2,}\b")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(weights)[::-1]

    terms = []
    for position in order:
        term = names[position]
        # Skip a word already covered by a chosen phrase, and the reverse.
        overlaps = False
        for chosen in terms:
            if term in chosen or chosen in term:
                overlaps = True
                break
        if not overlaps:
            terms.append(term)
        if len(terms) >= count:
            break
    return terms


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    """A summary made only of the text's own words, with no model.

    Starts with the cluster's key terms, which is what makes the node match a
    broad question like "what does this cover?", then the sentences closest to
    the cluster's average meaning. Page furniture, repeated sentences and code
    are excluded, since each of those is "central" for the wrong reason.
    """
    sentences = unique_prose_sentences(text)
    if len(sentences) == 0:
        return text[:600].strip()

    terms = key_terms(sentences)
    header = ("Covers: " + ", ".join(terms) + ". ") if len(terms) > 0 else ""

    if len(sentences) <= max_sentences:
        return header + " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return header + " ".join(sentences[:max_sentences])

    centroid = np.asarray(matrix.mean(axis=0))
    scores = (matrix @ centroid.T).ravel()

    best_positions = np.argsort(scores)[::-1][:max_sentences]
    chosen = sorted(best_positions.tolist())      # keep the original order

    picked = []
    for position in chosen:
        picked.append(sentences[position])
    return header + " ".join(picked)


REFUSAL = "I could not find this in the provided documents."


def parse_passages(prompt: str) -> tuple[list[tuple], str]:
    """Recover (rank, text) for each passage, and the question, from a prompt
    built by generate.build_prompt."""
    if "CONTEXT:" not in prompt or "QUESTION:" not in prompt:
        return [], ""
    after_context = prompt.split("CONTEXT:", 1)[1]
    context, question = after_context.split("QUESTION:", 1)

    passages = []
    for block in context.split("\n\n---\n\n"):
        block = block.strip()
        if block == "":
            continue
        parts = block.split("\n", 1)
        header = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        match = re.match(r"\[(\d+)\]", header)
        rank = int(match.group(1)) if match else len(passages) + 1
        passages.append((rank, body))
    return passages, question.strip()


def offline_answer(prompt: str, retrieval_score: float = None) -> str:
    """An answer with no model, built only from the retrieved passages.

    1. Refuse when retrieval found nothing convincing (see OFFLINE_MIN_SCORE).
    2. If sentences share words with the question, show the best of them,
       each tagged with the passage it came from, so the answer is cited.
    3. If none do - typical for broad questions like "what does this cover?",
       which share no vocabulary with any sentence - say so, and show how the
       closest passages begin. For a summary node that is its "Covers:" line,
       which is exactly what a broad question is asking for.
    """
    passages, question = parse_passages(prompt)
    if len(passages) == 0:
        return REFUSAL
    if retrieval_score is not None and retrieval_score < config.OFFLINE_MIN_SCORE:
        return REFUSAL

    tagged = []
    for rank, body in passages:
        for sentence in unique_prose_sentences(body, skip_covers=False):
            tagged.append((rank, sentence))
    if len(tagged) == 0:
        return REFUSAL

    if question != "":
        sentences = []
        for rank, sentence in tagged:
            sentences.append(sentence)
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            matrix = vectorizer.fit_transform(sentences + [question])
            scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        except ValueError:
            scores = np.zeros(len(sentences))

        if float(scores.max()) > 0.0:
            best = np.argsort(scores)[::-1][:5]
            chosen = sorted(best.tolist())
            lines = ["The most relevant sentences (extractive, no language model):"]
            for position in chosen:
                if scores[position] <= 0.0:
                    continue
                rank, sentence = tagged[position]
                lines.append("- " + sentence + " [" + str(rank) + "]")
            return "\n".join(lines)

    lines = ["No sentence shares your wording, so here is how the closest passages "
             "begin (extractive, no language model):"]
    for rank, body in passages[:3]:
        opening = unique_prose_sentences(body, skip_covers=False)
        if len(opening) > 0:
            lines.append("- " + " ".join(opening[:2]) + " [" + str(rank) + "]")
    return "\n".join(lines)


# ----------------------------------------------------------------- ollama ----

def ollama_request(payload: dict, stream: bool):
    url = config.OLLAMA_URL + "/api/generate"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT)


def ollama_complete(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with ollama_request(payload, stream=False) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "").strip()


def ollama_stream(prompt: str, system: str, max_tokens: int, temperature: float):
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with ollama_request(payload, stream=True) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line == "":
                continue
            data = json.loads(line)
            piece = data.get("response", "")
            if piece != "":
                yield piece
            if data.get("done") is True:
                return


# -------------------------------------------------------------- anthropic ----

def anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def anthropic_complete(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    response = anthropic_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text = text + block.text
    return text.strip()


def anthropic_stream(prompt: str, system: str, max_tokens: int, temperature: float):
    with anthropic_client().messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for piece in stream.text_stream:
            yield piece


# ----------------------------------------------------------------- openai ----

def openai_client():
    import openai

    return openai.OpenAI()


def openai_messages(prompt: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def openai_complete(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    response = openai_client().chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=openai_messages(prompt, system),
    )
    return (response.choices[0].message.content or "").strip()


def openai_stream(prompt: str, system: str, max_tokens: int, temperature: float):
    stream = openai_client().chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=openai_messages(prompt, system),
        stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece is not None and piece != "":
            yield piece


# --------------------------------------------------------------- dispatch ----

def provider() -> str:
    return config.LLM_PROVIDER


def complete(prompt: str, system: str = "", max_tokens: int = 700,
             temperature: float = 0.0, use_provider: str = None,
             retrieval_score: float = None) -> str:
    name = use_provider if use_provider is not None else provider()
    if name == "ollama":
        return ollama_complete(prompt, system, max_tokens, temperature)
    if name == "anthropic":
        return anthropic_complete(prompt, system, max_tokens, temperature)
    if name == "openai":
        return openai_complete(prompt, system, max_tokens, temperature)
    return offline_answer(prompt, retrieval_score)


def stream_complete(prompt: str, system: str = "", max_tokens: int = 700,
                    temperature: float = 0.0, use_provider: str = None,
                    retrieval_score: float = None):
    name = use_provider if use_provider is not None else provider()
    try:
        if name == "ollama":
            for piece in ollama_stream(prompt, system, max_tokens, temperature):
                yield piece
            return
        if name == "anthropic":
            for piece in anthropic_stream(prompt, system, max_tokens, temperature):
                yield piece
            return
        if name == "openai":
            for piece in openai_stream(prompt, system, max_tokens, temperature):
                yield piece
            return
    except (urllib.error.URLError, OSError) as error:
        yield "[" + name + " unavailable: " + str(error) + "] falling back to extractive mode.\n\n"

    # offline, or a provider that failed
    for word in offline_answer(prompt, retrieval_score).split(" "):
        yield word + " "


# Summaries the model could not write, with the reason. build_index reports
# these at the end: a silent fallback once hid two broken summaries inside a
# 57-minute build.
fallbacks = []


def summarize(texts: list[str], level: int = 1, use_provider: str = None) -> str:
    """Summarize the passages of one cluster into a parent node's text.

    `level` is the level of the node being written: 1 means its children are
    original chunks, 2 and above means its children are themselves summaries,
    which needs a different instruction.
    """
    joined = "\n\n---\n\n".join(texts)
    if len(joined) > config.SUMMARY_INPUT_CHARS:
        joined = joined[: config.SUMMARY_INPUT_CHARS]

    chosen = use_provider if use_provider is not None else provider()
    if chosen == "offline":
        return extractive_summary(joined, max_sentences=6)

    instruction = SUMMARY_INSTRUCTION if level <= 1 else HIGHER_LEVEL_INSTRUCTION

    summary = ""
    reason = ""
    try:
        summary = complete(
            instruction + joined,
            system=SUMMARY_SYSTEM,
            max_tokens=400,
            temperature=0.0,
        )
        if summary.strip() == "":
            reason = "model returned an empty response"
    except Exception as error:
        reason = type(error).__name__ + ": " + str(error)

    if reason != "":
        fallbacks.append({"chars_sent": len(joined), "reason": reason})
        print("    ! summary fell back to extractive - " + reason, flush=True)
        return extractive_summary(joined, max_sentences=6)

    return summary.strip()


def health() -> dict:
    """Is the configured provider actually reachable right now?"""
    name = provider()
    if name == "offline":
        return {"provider": name, "ready": True, "detail": "extractive mode, no model needed"}
    if name == "ollama":
        try:
            with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            names = []
            for model in data.get("models", []):
                names.append(model.get("name", ""))
            ready = config.OLLAMA_MODEL in names
            detail = "models installed: " + (", ".join(names) if len(names) > 0 else "none")
            if not ready:
                detail = detail + " (run: ollama pull " + config.OLLAMA_MODEL + ")"
            return {"provider": name, "ready": ready, "detail": detail}
        except Exception as error:
            return {"provider": name, "ready": False, "detail": "cannot reach Ollama: " + str(error)}

    key = "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENAI_API_KEY"
    import os

    if os.getenv(key):
        return {"provider": name, "ready": True, "detail": key + " is set"}
    return {"provider": name, "ready": False, "detail": key + " is not set"}
