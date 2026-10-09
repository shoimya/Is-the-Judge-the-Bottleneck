"""Checks for retriever.py and build_index.py (T03). Run `pytest` from the repo folder, or run this file on its own:

    python test/test_retriever.py        (or press Run / Debug on this file)

Most tests build a real BM25 index from five short paragraphs in a temporary folder, so they are quick and never
touch data/ or results/. The last test uses the real Wikipedia index and skips if it isn't on this machine.
"""

import sys
from pathlib import Path

import pytest

# The scripts being tested live one folder up. pytest finds them through pytest.ini; this line lets the file
# also run on its own with plain `python`. The imports below have to come after it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import build_index
from retriever import INDEX_DIR, count_paragraphs, load_index, measure_gold_recall, search

# Five short paragraphs, shaped like the Wikipedia ones: a title and a text.
SMALL_WIKIPEDIA = [
    {"title": "Paris", "text": "Paris is the capital of France."},
    {"title": "Berlin", "text": "Berlin is the capital of Germany."},
    {"title": "France", "text": "France is a country in Europe. Its capital is Paris."},
    {"title": "Rome", "text": "Rome is the capital of Italy."},
    {"title": "Germany", "text": "Germany is a country in Europe."},
]


@pytest.fixture
def small_index(tmp_path):
    """A real BM25 index of the five paragraphs above, built and saved in a temporary folder, then loaded."""
    index_dir = tmp_path / "bm25_index"
    build_index(SMALL_WIKIPEDIA, index_dir)
    return load_index(index_dir)


def titles_of(search_results: list[dict]) -> list[str]:
    """The titles of the search results, best first."""
    return [result["title"] for result in search_results]


def test_a_search_returns_k_paragraphs_with_title_text_and_score_best_first(small_index):
    """Asking for 2 results gives 2 paragraphs, each with its title, text and score, highest score first."""
    search_results = search(small_index, "capital of France", k=2)

    assert len(search_results) == 2
    assert set(titles_of(search_results)) == {"Paris", "France"}
    assert search_results[0]["score"] >= search_results[1]["score"]
    assert search_results[0]["text"] in {"Paris is the capital of France.",
                                         "France is a country in Europe. Its capital is Paris."}


def test_a_search_skips_titles_already_in_the_evidence_and_still_returns_k_paragraphs(small_index):
    """With "Paris" already in the Evidence, asking for 2 results gives 2 others: France and the next best."""
    search_results = search(small_index, "capital of France", k=2, exclude_titles={"Paris"})

    assert len(search_results) == 2
    assert "Paris" not in titles_of(search_results)
    assert titles_of(search_results)[0] == "France"


def test_a_query_with_no_searchable_words_returns_nothing(small_index):
    """An empty query, or one made only of common words like "the", finds no paragraphs at all."""
    assert search(small_index, "", k=2) == []
    assert search(small_index, "the of is", k=2) == []


def test_recall_counts_the_share_of_questions_with_all_and_with_at_least_one_gold_paragraph_found(small_index):
    """In the top 1, each question finds one Gold paragraph. In the top 2, only the first finds both.

    The second question's Gold title "Spain" isn't in the index, so it can never have all its Gold paragraphs found.
    """
    questions = [
        {"question": "capital of France", "gold_titles": ["Paris", "France"]},
        {"question": "capital of Italy", "gold_titles": ["Rome", "Spain"]},
    ]

    recall_rows = measure_gold_recall(small_index, questions, cutoffs=[1, 2])

    assert recall_rows == [
        {"k": 1, "all_found": 0.0, "at_least_one_found": 1.0},
        {"k": 2, "all_found": 0.5, "at_least_one_found": 1.0},
    ]


def test_the_real_wikipedia_index_holds_every_paragraph_and_answers_a_real_question():
    """The real index has all 5,233,329 paragraphs, and the first Pilot question gets 5 results."""
    if not (INDEX_DIR / "params.index.json").exists():
        pytest.skip(f"The real index isn't in {INDEX_DIR}, so it wasn't checked. Setup explains how to get it.")

    real_index = load_index(INDEX_DIR)
    search_results = search(real_index, "Who set Nietzche's philosophical novel to music?", k=5)

    assert count_paragraphs(real_index) == 5233329
    assert len(search_results) == 5


if __name__ == "__main__":
    # Running this file directly runs its tests, listing each one by name.
    sys.exit(pytest.main([__file__, "-v"]))
