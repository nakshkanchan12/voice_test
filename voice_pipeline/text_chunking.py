from __future__ import annotations

import re

SENTENCE_MODE = "sentence"
AGGRESSIVE_MODE = "aggressive"


def normalize_stream_mode(mode: str | None) -> str:
    if not mode:
        return SENTENCE_MODE

    token = mode.strip().lower()
    if token in {SENTENCE_MODE, AGGRESSIVE_MODE}:
        return token
    return SENTENCE_MODE


def split_ready_chunks(
    buffer: str,
    mode: str = SENTENCE_MODE,
    aggressive_min_tokens: int = 5,
) -> tuple[list[str], str]:
    normalized_mode = normalize_stream_mode(mode)

    chunks, remainder = _emit_complete_sentences(buffer)
    if normalized_mode != AGGRESSIVE_MODE:
        return chunks, remainder

    aggressive_chunks, remainder = _emit_aggressive_chunks(
        remainder,
        min_tokens=max(2, aggressive_min_tokens),
    )
    chunks.extend(aggressive_chunks)
    return chunks, remainder


def finalize_tail(buffer: str) -> str | None:
    candidate = buffer.strip()
    if not candidate:
        return None
    if candidate.endswith((".", "!", "?")):
        return candidate
    return f"{candidate}."


def _emit_complete_sentences(buffer: str) -> tuple[list[str], str]:
    chunks: list[str] = []
    start = 0

    for index, char in enumerate(buffer):
        if char not in ".!?\n":
            continue

        candidate = buffer[start : index + 1].strip()
        if candidate:
            chunks.append(candidate)
        start = index + 1

    return chunks, buffer[start:]


def _emit_aggressive_chunks(buffer: str, min_tokens: int) -> tuple[list[str], str]:
    chunks: list[str] = []
    remainder = buffer

    while True:
        stripped = remainder.lstrip()
        if stripped != remainder:
            remainder = stripped

        token_count = _count_tokens(remainder)
        if token_count < min_tokens:
            break

        punctuation_cut = _find_punctuation_cut(remainder)
        if punctuation_cut is not None:
            candidate = remainder[:punctuation_cut].strip()
            if candidate:
                chunks.append(candidate)
            remainder = remainder[punctuation_cut:]
            continue

        token_cut = _find_token_cut(remainder, min_tokens=min_tokens)
        if token_cut is None:
            break

        candidate = remainder[:token_cut].strip()
        if candidate:
            chunks.append(candidate)
        remainder = remainder[token_cut:]

    return chunks, remainder


def _count_tokens(value: str) -> int:
    return len(re.findall(r"\S+", value))


def _find_punctuation_cut(value: str) -> int | None:
    for index, char in enumerate(value):
        if char in ",;:-\n" and index >= 8:
            return index + 1
    return None


def _find_token_cut(value: str, min_tokens: int) -> int | None:
    count = 0
    for match in re.finditer(r"\S+\s*", value):
        count += 1
        if count < min_tokens:
            continue

        boundary = match.end()
        if boundary >= len(value):
            return None
        return boundary

    return None
