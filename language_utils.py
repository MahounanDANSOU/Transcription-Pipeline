from __future__ import annotations

import re

FRENCH_GUESS_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "que", "qui",
    "pour", "avec", "dans", "pas", "ne", "ce", "cette", "ces", "par", "sur",
    "entre", "tout", "tous", "mais", "est", "sont", "comme", "plus", "ou", "où",
}

ENGLISH_GUESS_WORDS = {
    "the", "and", "that", "this", "with", "for", "from", "have", "you", "are",
    "not", "was", "were", "what", "when", "where", "there", "their", "they",
    "your", "one", "all", "also", "but", "is", "of", "to", "in", "on",
}

ACCENTED_FR_REGEX = re.compile(r"[àâäéèêëîïôöùûüœç]")

def guess_language_from_text(text: str, default: str = "fr") -> str:
    normalized = (text or "").lower().strip()
    if not normalized:
        return default
    if ACCENTED_FR_REGEX.search(normalized):
        return "fr"
    words = re.findall(r"[a-z']+", normalized)
    if not words:
        return default
    fr_score = sum(1 for word in words if word in FRENCH_GUESS_WORDS)
    en_score = sum(1 for word in words if word in ENGLISH_GUESS_WORDS)
    if fr_score > en_score:
        return "fr"
    if en_score > fr_score:
        return "en"
    if "n't" in normalized or re.search(r"\b(don't|can't|won't|isn't|aren't|they're|you're|we're)\b", normalized):
        return "en"
    if re.search(r"\b(d'|l'|qu')", normalized):
        return "fr"
    return default

def smooth_language_labels(segments: list[dict], default_language: str) -> list[dict]:
    if not segments:
        return []
    items = [dict(seg) for seg in segments]
    for item in items:
        item["language"] = guess_language_from_text(str(item.get("text", "")), default_language)

    for i in range(1, len(items) - 1):
        if items[i].get("language") != items[i-1].get("language") and items[i-1].get("language") == items[i+1].get("language"):
            if len(str(items[i].get("text", "")).strip().split()) <= 4:
                items[i]["language"] = items[i-1].get("language")
    return items