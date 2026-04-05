import re


SEMANTIC_GROUP_NAMES = ("detail", "scene", "attribute")

_WRAPPER_PATTERNS = (
    r"^(?:a|an|the)\s+(?:photo|photograph|picture|image)\s+of\s+",
    r"^(?:a|an|the)\s+(?:close-up|close up)\s+(?:photo|portrait)\s+of\s+",
    r"^(?:a|an|the)\s+portrait\s+of\s+",
)

_SCENE_PREPOSITIONS = (
    " on ",
    " in ",
    " at ",
    " under ",
    " over ",
    " against ",
    " near ",
    " by ",
    " beside ",
    " next to ",
    " inside ",
    " outside ",
    " behind ",
    " around ",
    " amid ",
    " among ",
)

_SCENE_HINTS = (
    "lighting",
    "background",
    "indoors",
    "outdoors",
    "sunny",
    "night",
    "day",
    "studio",
    "bokeh",
    "depth of field",
    "35mm",
    "top-down",
    "top down",
    "close-up",
    "close up",
    "wide shot",
    "soft natural light",
    "warm light",
    "warm lighting",
)

_ATTRIBUTE_PHRASES = (
    "highly detailed",
    "soft natural light",
    "warm lighting",
    "studio lighting",
    "shallow depth of field",
    "top-down",
    "top down",
    "sunny day",
    "soft light",
    "cinematic lighting",
)

_ATTRIBUTE_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "golden",
    "silver",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "cozy",
    "warm",
    "soft",
    "natural",
    "cinematic",
    "detailed",
    "dramatic",
    "close-up",
    "close",
    "up",
    "top-down",
    "top",
    "down",
    "wooden",
    "brick",
    "shallow",
    "sunny",
    "studio",
}


def _compress_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = text.strip(" ,")
    return text


def _strip_wrapper(clause: str) -> str:
    res = clause.strip()
    for pattern in _WRAPPER_PATTERNS:
        res = re.sub(pattern, "", res, flags=re.IGNORECASE)
    return _compress_text(res)


def _split_clauses(prompt: str) -> list[str]:
    return [_compress_text(part) for part in prompt.split(",") if _compress_text(part)]


def _find_scene_split(clause: str) -> int | None:
    lowered = f" {clause.lower()} "
    split_idx = None
    for prep in _SCENE_PREPOSITIONS:
        idx = lowered.find(prep)
        if idx == -1:
            continue
        raw_idx = max(0, idx - 1)
        if split_idx is None or raw_idx < split_idx:
            split_idx = raw_idx
    return split_idx


def _drop_scene_from_clause(clause: str) -> str:
    split_idx = _find_scene_split(clause)
    if split_idx is None:
        return _compress_text(clause)
    stripped = clause[:split_idx]
    return _compress_text(stripped) or _compress_text(clause)


def _looks_like_scene_clause(clause: str) -> bool:
    lowered = clause.lower()
    return any(hint in lowered for hint in _SCENE_HINTS)


def _drop_attributes_from_clause(clause: str) -> str:
    stripped = f" {_compress_text(clause)} "
    for phrase in _ATTRIBUTE_PHRASES:
        stripped = re.sub(rf"\b{re.escape(phrase)}\b", " ", stripped, flags=re.IGNORECASE)
    words = []
    for token in stripped.split():
        if token.lower().strip(",.") in _ATTRIBUTE_WORDS:
            continue
        words.append(token)
    cleaned = " ".join(words)
    cleaned = re.sub(r"\b(a|an|the)\s+(?=(?:on|in|at|under|over|against|near|by)\b)", "", cleaned, flags=re.IGNORECASE)
    return _compress_text(cleaned) or _compress_text(clause)


def build_semantic_degraded_prompts(prompt: str) -> dict[str, str]:
    clauses = _split_clauses(prompt)
    if not clauses:
        prompt = _compress_text(prompt)
        return {name: prompt for name in SEMANTIC_GROUP_NAMES}

    main_clause = _strip_wrapper(clauses[0]) or _compress_text(clauses[0])
    detail_clauses = clauses[1:]

    detail_drop = main_clause

    scene_free_main = _drop_scene_from_clause(main_clause)
    kept_detail_clauses = [clause for clause in detail_clauses if not _looks_like_scene_clause(clause)]
    scene_drop = _compress_text(", ".join([scene_free_main, *kept_detail_clauses])) or main_clause

    attribute_free_main = _drop_attributes_from_clause(main_clause)
    attribute_drop = _compress_text(", ".join([attribute_free_main, *detail_clauses])) or main_clause

    degraded = {
        "detail": detail_drop,
        "scene": scene_drop,
        "attribute": attribute_drop,
    }
    return {name: _compress_text(text) or _compress_text(prompt) for name, text in degraded.items()}
