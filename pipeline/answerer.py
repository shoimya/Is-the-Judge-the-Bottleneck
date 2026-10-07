"""Answerer: question + Evidence -> a short answer.

The prompt here is a temporary draft so closed_book and single_turn can run end to end now.
T06 replaces it with the real, version-numbered prompt (CONTEXT.md, Roles).

    Press Debug on this file to answer one question from memory (needs `ollama serve` running on the Mac).
"""

from pipeline.llm import generate

PROMPT_VERSION = "draft"   # T06 sets "v0"


def build_answer_prompt(question_text: str, evidence: list[dict]) -> str:
    """The Answerer's prompt: the Evidence paragraphs (if any), the question, and the short-answer instruction."""
    evidence_lines = [f"{paragraph['title']}: {paragraph['text']}" for paragraph in evidence]
    evidence_section = "Evidence:\n" + "\n".join(evidence_lines) + "\n\n" if evidence else ""
    return f"{evidence_section}Question: {question_text}\nAnswer with a short phrase only."


def answer(question_text: str, evidence: list[dict], config: dict) -> dict:
    """Ask the model to answer from the Evidence; with no Evidence it answers from memory (Closed-book).

    Returns the model's full reply (text, token counts, probabilities); the answer itself is reply["text"].
    """
    answer_prompt = build_answer_prompt(question_text, evidence)
    return generate("answerer", answer_prompt, config)


if __name__ == "__main__":
    from pipeline.config import load_config

    reply = answer("Were Scott Derrickson and Ed Wood of the same nationality?", evidence=[], config=load_config())
    print("Answer from memory:", reply["text"].strip())
