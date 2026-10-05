"""Small checks for the pipeline. Run with `pytest`."""

import bz2
import json
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline import load_config
from pipeline.dataset import build_question_sets, load_question_set, split_pilot_and_test, to_question
from pipeline.llm import generate, generate_many, model_for
from pipeline.retriever import Retriever, build_index, gold_recall_at_k, read_wiki_paragraphs


def test_config_loads():
    config = load_config()
    assert config["loop"]["rounds"] == 3
    assert set(config["models"]) == {"judge", "rewriter", "answerer"}


# --- dataset (T02) ---

# One HotpotQA row as Hugging Face stores it: supporting_facts lists a title
# once per supporting sentence, so the same title can repeat.
RAW_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood", "Ed Wood"], "sent_id": [0, 0, 1]},
    "context": {"title": ["Ed Wood"], "sentences": [["Edward Davis Wood Jr. was an American filmmaker."]]},
}


def test_raw_row_becomes_question_with_gold_titles_in_order_without_repeats():
    assert to_question(RAW_ROW) == {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "gold_titles": ["Scott Derrickson", "Ed Wood"],
    }


def make_questions(question_count):
    return [{"id": f"q{number}", "gold_titles": ["A", "B"]} for number in range(question_count)]


def test_split_gives_disjoint_pilot_and_test_sets_of_the_requested_sizes():
    questions = make_questions(20)
    pilot, test = split_pilot_and_test(questions, pilot_size=3, test_size=5, seed=0)
    pilot_ids = {question["id"] for question in pilot}
    test_ids = {question["id"] for question in test}
    assert len(pilot) == 3 and len(test) == 5
    assert pilot_ids.isdisjoint(test_ids)
    assert pilot_ids | test_ids <= {question["id"] for question in questions}


def test_split_is_the_same_every_time_for_the_same_seed():
    questions = make_questions(20)
    assert split_pilot_and_test(questions, 3, 5, seed=0) == split_pilot_and_test(questions, 3, 5, seed=0)
    assert split_pilot_and_test(questions, 3, 5, seed=0) != split_pilot_and_test(questions, 3, 5, seed=1)


def write_fake_hotpotqa(path, row_count):
    rows = [dict(RAW_ROW, id=f"q{row_number}") for row_number in range(row_count)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_build_writes_question_files_and_id_lists_identically_on_every_run(tmp_path):
    parquet_path = tmp_path / "dev.parquet"
    write_fake_hotpotqa(parquet_path, 10)
    questions_dir, question_ids_dir = tmp_path / "data", tmp_path / "ids"

    build_question_sets(parquet_path, questions_dir, question_ids_dir, pilot_size=2, test_size=4, seed=0)
    first = {written_file.name: written_file.read_bytes() for written_file in [*questions_dir.iterdir(), *question_ids_dir.iterdir()]}
    build_question_sets(parquet_path, questions_dir, question_ids_dir, pilot_size=2, test_size=4, seed=0)
    second = {written_file.name: written_file.read_bytes() for written_file in [*questions_dir.iterdir(), *question_ids_dir.iterdir()]}
    assert first == second

    pilot = [json.loads(line) for line in (questions_dir / "pilot.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (questions_dir / "test.jsonl").read_text().splitlines()]
    assert len(pilot) == 2 and len(test) == 4
    assert pilot[0]["gold_titles"] == ["Scott Derrickson", "Ed Wood"]
    assert (question_ids_dir / "hotpotqa_pilot.txt").read_text().split() == [question["id"] for question in pilot]
    assert (question_ids_dir / "hotpotqa_test.txt").read_text().split() == [question["id"] for question in test]


# --- retriever (T03) ---


def write_fake_wiki_dump(path, articles):
    """A dump file like HotpotQA's: bz2-compressed, one JSON article per line, text as a list of sentences."""
    lines = [json.dumps({"id": str(article_number), "title": title, "text": sentences})
             for article_number, (title, sentences) in enumerate(articles)]
    path.write_bytes(bz2.compress("\n".join(lines).encode()))


def test_reading_the_dump_gives_one_paragraph_per_article_with_sentences_joined(tmp_path):
    dump_file = tmp_path / "wiki_00.bz2"
    write_fake_wiki_dump(dump_file, [
        ("Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker.", " He made Plan 9."]),
        ("Scott Derrickson", ["Scott Derrickson is an American director."]),
    ])
    assert list(read_wiki_paragraphs([dump_file])) == [
        {"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker. He made Plan 9."},
        {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director."},
    ]


TINY_WIKI = [
    {"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker known for low-budget films."},
    {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director of horror films."},
    {"title": "Plan 9 from Outer Space", "text": "Plan 9 from Outer Space is a science fiction film by Ed Wood."},
    {"title": "Doctor Strange (2016 film)", "text": "Doctor Strange is a superhero film directed by Scott Derrickson."},
    {"title": "Lighthouse", "text": "A lighthouse is a tower that emits light to guide ships at sea."},
    {"title": "Volcano", "text": "A volcano is a rupture in the crust of a planet where lava escapes."},
]


def test_search_ranks_the_matching_paragraph_first_and_survives_save_and_load(tmp_path):
    build_index(TINY_WIKI, tmp_path / "index")
    retriever = Retriever.load(tmp_path / "index")

    results = retriever.search("lighthouse ships", k=3)

    assert len(results) == 3
    assert results[0]["title"] == "Lighthouse"
    assert results[0]["text"] == TINY_WIKI[4]["text"]
    assert results[0]["score"] > results[1]["score"]


def test_search_never_returns_paragraphs_already_in_the_evidence(tmp_path):
    build_index(TINY_WIKI, tmp_path / "index")
    retriever = Retriever.load(tmp_path / "index")

    results = retriever.search("Ed Wood film", k=3, exclude_titles={"Ed Wood"})

    assert len(results) == 3
    assert "Ed Wood" not in [result["title"] for result in results]
    assert results[0]["title"] == "Plan 9 from Outer Space"


def test_gold_recall_counts_questions_with_both_or_at_least_one_gold_paragraph_in_the_top_k(tmp_path):
    build_index(TINY_WIKI, tmp_path / "index")
    retriever = Retriever.load(tmp_path / "index")
    questions = [
        {"question": "Were Scott Derrickson and Ed Wood of the same nationality?",
         "gold_titles": ["Scott Derrickson", "Ed Wood"]},
        {"question": "Which tower guides ships at sea?", "gold_titles": ["Lighthouse", "Volcano"]},
    ]

    # k=1: one result can hold at most one of the two Gold paragraphs; each question finds one.
    # k=6: the whole corpus comes back, so both are always found.
    assert gold_recall_at_k(questions, retriever, ks=[1, 6]) == [
        {"k": 1, "both_found": 0.0, "at_least_one_found": 1.0},
        {"k": 6, "both_found": 1.0, "at_least_one_found": 1.0},
    ]


# --- model backend (T04) ---

def test_each_role_uses_its_own_model_name_for_the_active_backend():
    config = {
        "backend": "ollama",
        "models": {
            "judge": {"ollama": "qwen3:14b", "vllm": "Qwen/Qwen3-14B-AWQ"},
            "answerer": {"ollama": "qwen3:4b-instruct", "vllm": "Qwen/Qwen3-4B-Instruct-2507"},
        },
    }
    assert model_for("judge", config) == "qwen3:14b"
    assert model_for("answerer", config) == "qwen3:4b-instruct"

    config["backend"] = "vllm"
    assert model_for("judge", config) == "Qwen/Qwen3-14B-AWQ"


def ollama_is_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except OSError:
        return False


needs_ollama = pytest.mark.skipif(load_config()["backend"] != "ollama" or not ollama_is_running(),
                                  reason="needs the Ollama server running (see README: Ollama on the Mac)")


@needs_ollama
def test_ollama_answers_with_token_counts_and_probabilities_for_every_generated_token():
    prompt = "Answer with one word: what is the capital of France?"
    generation = generate("answerer", prompt)

    assert "Paris" in generation["text"]
    assert generation["prompt_tokens"] > 0
    # completion_tokens can include the end-of-answer token, which has no probability entry
    assert 0 < len(generation["logprobs"]) <= generation["completion_tokens"]
    for token_entry in generation["logprobs"]:
        assert token_entry["logprob"] <= 0
        assert len(token_entry["top"]) == 5
        assert token_entry["token"] == token_entry["top"][0]["token"]   # greedy: the chosen token is the most likely
    assert generate("answerer", prompt)["text"] == generation["text"]  # greedy decoding repeats exactly


@needs_ollama
def test_a_batch_of_prompts_comes_back_as_one_answer_per_prompt_in_order():
    generations = generate_many("answerer", [
        "Answer with one word: what is the capital of France?",
        "Answer with one word: what is the capital of Japan?",
    ])
    assert "Paris" in generations[0]["text"]
    assert "Tokyo" in generations[1]["text"]


# --- review fixes (Oct 4) ---

def test_a_built_question_set_loads_back_unchanged(tmp_path):
    parquet_path = tmp_path / "dev.parquet"
    write_fake_hotpotqa(parquet_path, 10)
    questions_dir = tmp_path / "data"
    build_question_sets(parquet_path, questions_dir, tmp_path / "ids", pilot_size=2, test_size=4, seed=0)

    pilot_set = load_question_set("pilot", questions_dir)
    assert len(pilot_set) == 2
    assert pilot_set[0]["gold_titles"] == ["Scott Derrickson", "Ed Wood"]


def test_search_asking_for_more_results_than_paragraphs_returns_them_all(tmp_path):
    build_index(TINY_WIKI, tmp_path / "index")
    retriever = Retriever.load(tmp_path / "index")

    assert len(retriever.search("film", k=10, exclude_titles={"Volcano"})) == 5


def test_gold_recall_needs_every_gold_paragraph_when_there_are_more_than_two(tmp_path):
    build_index(TINY_WIKI, tmp_path / "index")
    retriever = Retriever.load(tmp_path / "index")
    three_hop_question = {"question": "Ed Wood Plan 9 Lighthouse",
                          "gold_titles": ["Ed Wood", "Plan 9 from Outer Space", "Lighthouse"]}

    # Top 2 can hold at most two of the three Gold paragraphs, so "all found" must be 0, not 1.
    assert gold_recall_at_k([three_hop_question], retriever, ks=[2])[0]["both_found"] == 0.0
