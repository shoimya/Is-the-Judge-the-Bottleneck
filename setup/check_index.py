"""Check search quality: how often BM25 finds the Gold paragraphs for the Pilot questions. (T03)

    python setup/check_index.py        (or press Debug on this file)

Searches once with each Pilot question and counts, for the top 2 / 5 / 10 / 20 results, the share of questions
where ALL Gold paragraphs were found and where AT LEAST ONE was. Saves results/pilot/bm25_recall.csv.
Expected (Oct 4): all found 11 / 24 / 34 / 44 %, at least one 77 / 83 / 84 / 90 %.
"""

import csv
from pathlib import Path

from pipeline.config import RECALL_CSV
from pipeline.dataset import load_question_set
from pipeline.retriever import Retriever

RECALL_CUTOFFS = [2, 5, 10, 20]   # how many top results to look at


def ranked_titles_for_each_question(questions: list[dict], retriever: Retriever, how_many: int) -> list[list[str]]:
    """Search once with each question and keep the titles of the top results, best first."""
    return [[result["title"] for result in retriever.search(question["question"], k=how_many)]
            for question in questions]


def count_gold_found(gold_titles: list[str], top_titles: list[str]) -> int:
    """How many of a question's Gold paragraphs are among the given top titles."""
    return sum(gold_title in top_titles for gold_title in gold_titles)


def gold_recall_at_k(questions: list[dict], retriever: Retriever, cutoffs: list[int]) -> list[dict]:
    """For each cutoff k: the share of questions with all, and with at least one, Gold paragraph in the top k."""
    # Here we search once with the largest cutoff; a smaller cutoff is just the top of the same list.
    ranked_titles = ranked_titles_for_each_question(questions, retriever, how_many=max(cutoffs))
    recall_rows = []
    for k in cutoffs:
        all_found_count = 0
        at_least_one_found_count = 0
        for question, top_titles in zip(questions, ranked_titles):
            gold_found_count = count_gold_found(question["gold_titles"], top_titles[:k])
            all_found_count += gold_found_count == len(question["gold_titles"])
            at_least_one_found_count += gold_found_count >= 1
        recall_rows.append({"k": k,
                            "all_found": all_found_count / len(questions),
                            "at_least_one_found": at_least_one_found_count / len(questions)})
    return recall_rows


def write_recall_csv(recall_rows: list[dict], csv_path: Path) -> None:
    """Save the recall table as a CSV file for the paper."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["k", "all_found", "at_least_one_found"])
        csv_writer.writeheader()
        csv_writer.writerows(recall_rows)


def print_recall(recall_rows: list[dict]) -> None:
    """Print the recall table as percentages."""
    for row in recall_rows:
        print(f"k={row['k']:>2}  all Gold paragraphs found: {row['all_found']:.0%}   "
              f"at least one: {row['at_least_one_found']:.0%}")


if __name__ == "__main__":
    pilot_set = load_question_set("pilot")
    recall_rows = gold_recall_at_k(pilot_set, Retriever.load(), RECALL_CUTOFFS)
    write_recall_csv(recall_rows, RECALL_CSV)
    print_recall(recall_rows)
    print(f"Saved to {RECALL_CSV}")
