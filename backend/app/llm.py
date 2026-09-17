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


# ---------------------------------------------------------------- offline ----

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        cleaned = part.strip()
        if len(cleaned) > 25:
            sentences.append(cleaned)
    return sentences


def extractive_summary(text: str, max_sentences: int = 6) -> str:
    """Pick the sentences closest to the average meaning of the whole text."""
    sentences = split_sentences(text)
    if len(sentences) == 0:
        return text[:600].strip()
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:max_sentences])

    centroid = np.asarray(matrix.mean(axis=0))
    scores = (matrix @ centroid.T).ravel()

    best_positions = np.argsort(scores)[::-1][:max_sentences]
    chosen = sorted(best_positions.tolist())      # keep the original order

    picked = []
    for position in chosen:
        picked.append(sentences[position])
    return " ".join(picked)


def offline_answer(prompt: str) -> str:
    """A readable answer with no model: the most relevant lines of the context.

    The prompt we build always puts the passages between CONTEXT: and QUESTION:,
    so we can pull them back out and rank their sentences against the question.
    """
    context = prompt
    question = ""
    if "CONTEXT:" in prompt and "QUESTION:" in prompt:
        after_context = prompt.split("CONTEXT:", 1)[1]
        context, question = after_context.split("QUESTION:", 1)

    sentences = split_sentences(context)
    if len(sentences) == 0:
        return "I could not find this in the provided documents."

    question = question.strip()
    if question == "":
        return extractive_summary(context, max_sentences=5)

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences + [question])
    except ValueError:
        return extractive_summary(context, max_sentences=5)

    question_vector = matrix[-1]
    sentence_matrix = matrix[:-1]
    scores = (sentence_matrix @ question_vector.T).toarray().ravel()

    if float(scores.max()) <= 0.0:
        return "I could not find this in the provided documents."

    best_positions = np.argsort(scores)[::-1][:5]
    chosen = sorted(best_positions.tolist())

    lines = ["Based on the retrieved passages (extractive mode, no language model):"]
    for position in chosen:
        lines.append("- " + sentences[position])
    lines.append(
        "\nSet LLM_PROVIDER to ollama, anthropic or openai in .env for a written answer."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------- ollama ----

def ollama_request(payload: dict, stream: bool):
    url = config.OLLAMA_URL + "/api/generate"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=300)


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


def complete(prompt: str, system: str = "", max_tokens: int = 700, temperature: float = 0.0) -> str:
    name = provider()
    if name == "ollama":
        return ollama_complete(prompt, system, max_tokens, temperature)
    if name == "anthropic":
        return anthropic_complete(prompt, system, max_tokens, temperature)
    if name == "openai":
        return openai_complete(prompt, system, max_tokens, temperature)
    return offline_answer(prompt)


def stream_complete(prompt: str, system: str = "", max_tokens: int = 700, temperature: float = 0.0):
    name = provider()
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
    for word in offline_answer(prompt).split(" "):
        yield word + " "


# Summaries the model could not write, with the reason. build_index reports
# these at the end: a silent fallback once hid two broken summaries inside a
# 57-minute build.
fallbacks = []


def summarize(texts: list[str]) -> str:
    """Summarize the passages of one cluster into a parent node's text."""
    joined = "\n\n---\n\n".join(texts)
    if len(joined) > config.SUMMARY_INPUT_CHARS:
        joined = joined[: config.SUMMARY_INPUT_CHARS]

    if provider() == "offline":
        return extractive_summary(joined, max_sentences=6)

    summary = ""
    reason = ""
    try:
        summary = complete(
            SUMMARY_INSTRUCTION + joined,
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
