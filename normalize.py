"""
postprocessing/normalize.py

Responsibilities:
  - Text cleaning / normalization functions used both during generation
    (output post-processing) and during evaluation (string comparison).
"""


# =============================================================================
# OUTPUT NORMALIZATION (post-generation)
# =============================================================================

def strip_prefixes(text: str, strip_list: list) -> str:
    """
    Remove known prefix artefacts introduced by the prompt template or
    occasional decoder bleed-through.

    Args:
        text       (str):       Raw generated text.
        strip_list (list[str]): Ordered list of prefix strings to remove.

    Returns:
        str: Text with the first matching prefix stripped.
    """
    for prefix in strip_list:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break  # Only strip one prefix per call.
    return text


def strip_leading_punctuation(text: str) -> str:
    """
    Remove stray leading punctuation characters that sometimes appear
    after prefix stripping (e.g. ':' or ',').

    Args:
        text (str): Text after prefix stripping.

    Returns:
        str: Text with leading ':,. ' characters removed.
    """
    return text.lstrip(":,. ")


def is_empty_or_trivial(text: str) -> bool:
    """
    Return True if the text is empty or contains fewer than two tokens,
    which is treated as an unanswerable / degenerate output.

    Args:
        text (str): Candidate answer text.

    Returns:
        bool
    """
    return not text or len(text.split()) < 2


def is_cant_answer(text: str, cant_answer_phrases: list) -> bool:
    """
    Return True if the text contains any of the canonical "cannot answer"
    phrases (case-insensitive).

    Args:
        text                (str):       Candidate answer text.
        cant_answer_phrases (list[str]): Phrases that indicate the model
                                         could not find the answer.

    Returns:
        bool
    """
    lower = text.lower()
    return any(phrase in lower for phrase in cant_answer_phrases)


def normalize_output(gen_text: str, config: dict) -> str:
    """
    Full post-processing pipeline applied to each model output:
      1. Strip known prefix artefacts.
      2. Remove leading punctuation.
      3. Treat empty / single-token outputs as unanswerable.
      4. Normalise all "cannot answer" variants to a single canonical string.

    Args:
        gen_text (str):  Raw decoded model output.
        config   (dict): Loaded configuration dict.

    Returns:
        str: Cleaned, normalised answer string.
    """
    strip_list          = config["strip_prefixes"]
    cant_answer_phrases = config["cant_answer_phrases"]
    canonical           = config["cant_answer_canonical"]

    gen_text = strip_prefixes(gen_text, strip_list)
    gen_text = strip_leading_punctuation(gen_text)

    if is_empty_or_trivial(gen_text):
        return canonical

    if is_cant_answer(gen_text, cant_answer_phrases):
        return canonical

    return gen_text


# =============================================================================
# EVALUATION NORMALIZATION (string comparison)
# =============================================================================

def safe_normalize(text) -> str:
    """
    Lowercase, strip leading/trailing whitespace, and collapse internal
    whitespace for consistent ROUGE-L string comparison.

    Returns "" for non-string inputs to avoid scorer crashes.

    Args:
        text: Input value (expected str; any other type -> "").

    Returns:
        str: Normalised string, or "" if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())