"""Scoring: does a model's answer match the gold answer? Every method uses these, so they live in one place.

    Press Run to see a few answers scored (it starts this project's Ollama server for the cosine similarity).

Answers are compared after HotpotQA's official clean-up (lowercase, no punctuation, no "a/an/the", single spaces),
so "The Yoruba." matches "Yoruba".

Scores per question:
  correct         exact match: the cleaned answers are identical (yes or no)
  precision       shared words / words in the model's answer   (did it add extra words?)
  recall          shared words / words in the gold answer      (did it leave words out?)
  f1              2 x precision x recall / (precision + recall), partial credit between 0 and 1
  cosine          how alike the two answers are in meaning: the cosine similarity of their embeddings (1 = same)
  correct_cosine  correct by meaning: an exact match, or a cosine at or above config.yaml's scoring.cosine_threshold
                  (a yes/no answer still needs an exact match)
The first four are HotpotQA's official scores; the last two need config.yaml's embedding model (scoring.embedder).
"""

import math
import re
import string
from collections import Counter

from pipeline import llm2, load_config, settings

# Here are the answers HotpotQA only accepts word for word: a "yes" never gets partial credit against "yes it is".
EXACT_ONLY_ANSWERS = {"yes", "no", "noanswer"}


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


def is_yes_no_answer(predicted: str, gold: str) -> bool:
    """True if either answer, once cleaned, is one HotpotQA only accepts word for word (yes, no or noanswer)."""
    return normalize_answer(predicted) in EXACT_ONLY_ANSWERS or normalize_answer(gold) in EXACT_ONLY_ANSWERS


def precision_recall_f1(predicted: str, gold: str) -> dict:
    """How many words the two cleaned answers share, as precision, recall and F1 (HotpotQA's official word F1)."""
    no_overlap = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # Here we apply HotpotQA's rule: a yes/no answer scores only if it matches exactly.
    if is_yes_no_answer(predicted, gold) and not exact_match(predicted, gold):
        return no_overlap

    predicted_words = normalize_answer(predicted).split()
    gold_words = normalize_answer(gold).split()
    # Here we count the shared words; a word repeated in one answer counts only as often as it is in both.
    shared_word_counts = Counter(predicted_words) & Counter(gold_words)
    shared_count = sum(shared_word_counts.values())
    if shared_count == 0:
        return no_overlap

    precision = shared_count / len(predicted_words)
    recall = shared_count / len(gold_words)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def cosine_similarity(first_embedding: list[float], second_embedding: list[float]) -> float:
    """Return how similar two embeddings are: 1 means the same meaning, around 0 means unrelated."""
    dot_product = 0.0
    for first_value, second_value in zip(first_embedding, second_embedding):
        dot_product += first_value * second_value

    # Here we divide by both lengths, so only the direction of the embeddings counts, not their size.
    first_length = math.sqrt(sum(value * value for value in first_embedding))
    second_length = math.sqrt(sum(value * value for value in second_embedding))
    return dot_product / (first_length * second_length)


def trim_ends(text: str) -> str:
    """The text without spaces or punctuation at its two ends: '"The Yoruba." ' -> 'The Yoruba'.
    Punctuation inside stays, so "J.R.R. Tolkien" keeps its dots."""
    return text.strip().strip(string.punctuation).strip()


def answer_similarity(predicted: str, gold: str) -> float:
    """How alike the two answers are in meaning: the cosine similarity of their embeddings.

    Both answers are trimmed first (trim_ends), because the embedding model builds the embedding from the text's last
    token: a full stop at the end changed "The Yoruba." vs "The Yoruba" from about 1 to 0.849.
    An empty answer gets 0, since there is nothing to compare. Needs this project's Ollama server running.
    """
    predicted = trim_ends(predicted)
    gold = trim_ends(gold)
    if not predicted or not gold:
        return 0.0
    embedder = load_config()["scoring"]["embedder"]
    # Here both answers go to the embedding model in one call, and come back in the same order.
    predicted_embedding, gold_embedding = llm2.embed_texts(embedder, [predicted, gold])
    return cosine_similarity(predicted_embedding, gold_embedding)


def correct_by_cosine(predicted: str, gold: str, cosine: float) -> bool:
    """True if the answer counts as correct by meaning: an exact match always does; otherwise its cosine must reach
    config.yaml's scoring.cosine_threshold, and a yes/no answer can only count by exact match."""
    if exact_match(predicted, gold):
        return True
    if is_yes_no_answer(predicted, gold):
        return False
    return cosine >= load_config()["scoring"]["cosine_threshold"]


def score_answer(predicted: str, gold: str) -> dict:
    """Every score for one answer: correct, precision, recall, f1, cosine and correct_cosine (see the top of this file)."""
    cosine = answer_similarity(predicted, gold)
    return {
        "correct": exact_match(predicted, gold),
        **precision_recall_f1(predicted, gold),
        "cosine": cosine,
        "correct_cosine": correct_by_cosine(predicted, gold, cosine),
    }


def average_score(results: list[dict], score_name: str) -> float:
    """The average of one score ("precision", "recall" or "f1") over a run's results; 0 if there are none."""
    if not results:
        return 0.0
    return sum(result[score_name] for result in results) / len(results)


if __name__ == "__main__":
    embedder = load_config()["scoring"]["embedder"]
    if not settings.model_is_downloaded(embedder):
        raise SystemExit(f"{embedder} isn't in {settings.MODELS_FOLDER}. Run project_set_up.py first.")

    examples = [("The Yoruba.", "The Yoruba"), ("Yoruba people", "The Yoruba"), ("Antonio Vivaldi", "Antonio Lucio Vivaldi"),
                ("J.R.R. Tolkien", "J. R. R. Tolkien"), ("Ken Flach", "Kenneth Eliot Flach"),
                ("David Diamond", "David Weissman"), ("Scanian dialect", "Scanian dialects"), ("no", "yes")]
    server_started_here = llm2.start_ollama_server()
    try:
        print(f"Embedding model: {embedder}   Threshold: {load_config()['scoring']['cosine_threshold']}\n")
        for predicted, gold in examples:
            scores = score_answer(predicted, gold)
            print(f"{predicted!r:18} vs {gold!r:24} -> correct {scores['correct']!s:5}  "
                  f"P {scores['precision']:.2f}  R {scores['recall']:.2f}  F1 {scores['f1']:.2f}  "
                  f"cosine {scores['cosine']:.3f}  correct (cosine) {scores['correct_cosine']}")
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            llm2.stop_ollama_server()
