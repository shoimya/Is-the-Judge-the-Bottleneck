"""Read the Pilot set and Test set questions. (They are made once by setup/get_questions.py, T02.)

Each question is a dictionary:
    {"id", "question", "answer", "type", "level", "gold_titles"}

    Press Debug on this file to load the Pilot set and print its first question.
"""

import json
from pathlib import Path

from pipeline.config import QUESTIONS_DIR


def load_question_set(set_name: str, questions_dir: Path = QUESTIONS_DIR) -> list[dict]:
    """Read one question set ("pilot" or "test") from its .jsonl file, one question per line."""
    question_file_path = questions_dir / f"{set_name}.jsonl"
    with open(question_file_path) as question_file:
        return [json.loads(line) for line in question_file]


def find_question(question_id: str, question_set: list[dict]) -> dict:
    """Find one question in a question set by its id."""
    for question in question_set:
        if question["id"] == question_id:
            return question
    raise KeyError(f"No question with id {question_id} in this question set.")


if __name__ == "__main__":
    pilot_set = load_question_set("pilot")
    print(f"Loaded {len(pilot_set)} Pilot questions. The first one:")
    for field_name, field_value in pilot_set[0].items():
        print(f"  {field_name}: {field_value}")
