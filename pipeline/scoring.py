"""Scoring: does a model's answer match the gold answer? Every method uses these, so they live in one place.

    Press Run to see a few answers scored.

Answers are compared after HotpotQA's official clean-up (lowercase, no punctuation, no "a/an/the", single spaces),
so "The Yoruba." matches "Yoruba".
"""

import re
import string


def normalize_answer(text: str) -> str:
    """An answer cleaned the way HotpotQA's official scoring cleans it: "The  Yoruba!" -> "yoruba"."""
    text = text.lower()
    # Here we remove every punctuation mark.
    text = "".join(character for character in text if character not in string.punctuation)
    # Here we remove the words "a", "an" and "the" (only as whole words, so "theatre" stays).
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Here we squeeze any run of spaces into one, and trim both ends.
    return " ".join(text.split())


def exact_match(predicted: str, gold: str) -> bool:
    """True if the predicted answer and the gold answer are the same once both are cleaned."""
    return normalize_answer(predicted) == normalize_answer(gold)


if __name__ == "__main__":
    examples = [("The Yoruba.", "The Yoruba"), ("yoruba people", "The Yoruba"), ("World War I", "World War I"),
                ("WWI", "World War I"), ("yes", "Yes")]
    for predicted, gold in examples:
        print(f"{predicted!r:18} vs {gold!r:16} -> {exact_match(predicted, gold)}")
