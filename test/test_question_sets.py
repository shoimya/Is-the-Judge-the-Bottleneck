"""Checks for pipeline/question_sets.py (T02). Run `pytest` from the repo folder, or run this file on its own:

    python test/test_question_sets.py        (or press Run / Debug on this file)

These tests cover the functions agreed with the owner. They write only to a temporary folder, never to data/ or results/.
"""

import socket
import sys

import pytest

from pipeline.question_sets import (count_questions_without_two_gold_titles, download_hotpotqa,
                                    question_ids_of, raw_row_to_question, read_hotpotqa_questions,
                                    split_pilot_and_test, write_question_ids)


def make_raw_row(question_id: str) -> dict:
    """A small HotpotQA row shaped like the real download, with a Gold title repeated once per supporting sentence."""
    return {
        "id": question_id,
        "question": "Which city was the director of Inception born in?",
        "answer": "London",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["Inception", "Christopher Nolan", "Inception"], "sent_id": [0, 0, 1]},
        "context": {"title": ["Inception"], "sentences": [["Inception is a 2010 film."]]},
    }


def test_a_raw_row_keeps_only_the_project_fields_and_lists_each_gold_title_once_in_order():
    """The record has the six fields the project uses, and a Gold title repeated in the source appears once."""
    question = raw_row_to_question(make_raw_row("question-1"))

    assert question == {
        "id": "question-1",
        "question": "Which city was the director of Inception born in?",
        "answer": "London",
        "type": "bridge",
        "level": "hard",
        "gold_titles": ["Inception", "Christopher Nolan"],
    }


def make_questions(question_count: int) -> list[dict]:
    """A list of question records with ids question-0, question-1, ..."""
    return [raw_row_to_question(make_raw_row(f"question-{number}")) for number in range(question_count)]


def test_the_split_has_the_asked_sizes_shares_no_question_and_is_the_same_for_the_same_seed():
    """Two splits with the same seed pick the same questions, the sets have the asked sizes, and no id is in both."""
    questions = make_questions(50)

    pilot_set, test_set = split_pilot_and_test(questions, pilot_size=5, test_size=20, seed=0)
    pilot_set_again, test_set_again = split_pilot_and_test(questions, pilot_size=5, test_size=20, seed=0)

    assert len(pilot_set) == 5
    assert len(test_set) == 20
    assert set(question_ids_of(pilot_set)).isdisjoint(question_ids_of(test_set))
    assert question_ids_of(pilot_set) == question_ids_of(pilot_set_again)
    assert question_ids_of(test_set) == question_ids_of(test_set_again)


def test_the_id_list_has_one_id_per_line_in_order_and_writing_it_again_gives_the_same_file(tmp_path):
    """The id file lists each question's id on its own line, and a second write leaves it byte-identical."""
    question_set = make_questions(3)
    id_file_path = tmp_path / "question_ids" / "hotpotqa_pilot.txt"

    write_question_ids(question_set, id_file_path)
    first_write = id_file_path.read_bytes()
    write_question_ids(question_set, id_file_path)

    assert first_write == b"question-0\nquestion-1\nquestion-2\n"
    assert id_file_path.read_bytes() == first_write


def hugging_face_is_reachable() -> bool:
    """True if this machine can open a connection to huggingface.co, where HotpotQA is downloaded from."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


def test_the_real_hotpotqa_download_has_every_dev_question_each_with_two_gold_titles(tmp_path):
    """Downloads the real file into a temporary folder: 7,405 dev questions, each with exactly 2 Gold titles."""
    if not hugging_face_is_reachable():
        pytest.skip("huggingface.co can't be reached (no internet?), so the real HotpotQA download wasn't checked.")

    hotpotqa_file_path = download_hotpotqa(tmp_path)
    questions = read_hotpotqa_questions(hotpotqa_file_path)

    assert len(questions) == 7405
    assert count_questions_without_two_gold_titles(questions) == 0


if __name__ == "__main__":
    # Running this file directly runs its tests, listing each one by name.
    sys.exit(pytest.main([__file__, "-v"]))
