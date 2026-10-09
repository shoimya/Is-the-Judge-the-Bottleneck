"""Checks for every piece of the pipeline, in pipeline order. Open this file and press Run to run them all.

Tests use tiny made-up data, so they run in seconds. Tests marked needs_ollama talk to the real model and are
skipped (not failed) when the Ollama server isn't running.
"""

import bz2
import http.server
import json
import shutil
import tarfile
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.answerer import build_answer_prompt
from pipeline.config import load_config
from pipeline.dataset import find_question, load_question_set
from pipeline.llm import generate, generate_many, model_for, ollama_is_running, ollama_name_for
from pipeline.loop import (SCENARIOS, closed_book, make_answerer_entry, make_result, make_retriever_entry,
                           make_round, never_stop, single_turn)
from pipeline.retriever import Retriever
from pipeline.run_log import (SCENARIO_FOLDER_NAMES, read_meta, read_question_records, run_scenario,
                              scenario_log_folder)
from pipeline.server_launcher import start_ollama_server, stop_ollama_server
from project_set_up import missing_libraries, ollama_model_is_downloaded, python_is_new_enough, question_sets_exist
from run import check_setup
from setup.build_index import build_index, download_and_unpack_dump, download_with_resume, read_wiki_paragraphs
from setup.check_index import gold_recall_at_k
from setup.get_questions import build_question_sets, raw_row_to_question, split_pilot_and_test

# Tests marked with this run only when the Ollama server is running on this Mac.
needs_ollama = pytest.mark.skipif(not ollama_is_running(),
                                  reason="needs the Ollama server running (see README: Ollama on the Mac)")


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# One HotpotQA row as Hugging Face stores it: supporting_facts lists a title once per supporting sentence,
# so the same title can repeat.
RAW_HOTPOTQA_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood", "Ed Wood"], "sent_id": [0, 0, 1]},
    "context": {"title": ["Ed Wood"], "sentences": [["Edward Davis Wood Jr. was an American filmmaker."]]},
}

# A six-paragraph "Wikipedia", small enough to know the right search results by eye.
TINY_WIKI = [
    {"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker known for low-budget films."},
    {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director of horror films."},
    {"title": "Plan 9 from Outer Space", "text": "Plan 9 from Outer Space is a science fiction film by Ed Wood."},
    {"title": "Doctor Strange (2016 film)", "text": "Doctor Strange is a superhero film directed by Scott Derrickson."},
    {"title": "Lighthouse", "text": "A lighthouse is a tower that emits light to guide ships at sea."},
    {"title": "Volcano", "text": "A volcano is a rupture in the crust of a planet where lava escapes."},
]


@pytest.fixture(scope="module")
def tiny_retriever(tmp_path_factory):
    """A Retriever over TINY_WIKI, built and saved once for all tests in this file, then loaded like the real one."""
    index_dir = tmp_path_factory.mktemp("tiny_index")
    build_index(TINY_WIKI, index_dir)
    return Retriever.load(index_dir)


def write_fake_hotpotqa(parquet_path, row_count):
    """A small HotpotQA-style Parquet file: copies of RAW_HOTPOTQA_ROW with ids q0, q1, ..."""
    rows = [dict(RAW_HOTPOTQA_ROW, id=f"q{row_number}") for row_number in range(row_count)]
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)


def make_fake_questions(question_count):
    """Stand-in questions q0, q1, ... for testing the split; only their ids matter."""
    return [{"id": f"q{number}", "gold_titles": ["A", "B"]} for number in range(question_count)]


def read_all_files(*folders):
    """Every file in the given folders as {file name: file contents}."""
    return {written_file.name: written_file.read_bytes() for folder in folders for written_file in folder.iterdir()}


def write_fake_wiki_dump(dump_file_path, articles):
    """A dump file like HotpotQA's: bz2-compressed, one JSON article per line, text as a list of sentences."""
    article_lines = [json.dumps({"id": str(article_number), "title": title, "text": sentences})
                     for article_number, (title, sentences) in enumerate(articles)]
    dump_file_path.write_bytes(bz2.compress("\n".join(article_lines).encode()))


# ---------------------------------------------------------------------------
# 0. Setting up a machine: project_set_up.py
# ---------------------------------------------------------------------------

def test_python_312_or_newer_is_new_enough_and_311_is_not():
    """The project needs Python 3.12 or newer."""
    assert python_is_new_enough((3, 12, 0))
    assert python_is_new_enough((3, 13, 1))
    assert not python_is_new_enough((3, 11, 9))


def test_only_libraries_that_are_missing_or_the_wrong_version_are_reported():
    """Comments, extras and the project's own `-e .` line are understood; only what needs installing is listed."""
    requirement_lines = [
        "# a comment line",
        "",
        "PyYAML==6.0.3",
        "bm25s[core]==0.3.11              # extras in brackets are part of the name",
        "huggingface_hub==1.33.0",
        "pytest==9.1.1",
        "-e .",
    ]
    # Here pip reports names in its own spelling ("huggingface-hub"), pytest is the wrong version,
    # and the project's own package isn't installed.
    installed_versions = {"pyyaml": "6.0.3", "bm25s": "0.3.11", "huggingface-hub": "1.33.0", "pytest": "8.0.0"}
    assert missing_libraries(requirement_lines, installed_versions) == ["pytest==9.1.1", "-e ."]


def test_question_sets_exist_only_when_both_the_pilot_and_test_files_are_there(tmp_path):
    """Both pilot.jsonl and test.jsonl are needed."""
    assert not question_sets_exist(tmp_path)
    (tmp_path / "pilot.jsonl").write_text("{}\n")
    assert not question_sets_exist(tmp_path)
    (tmp_path / "test.jsonl").write_text("{}\n")
    assert question_sets_exist(tmp_path)


def test_an_ollama_model_counts_as_downloaded_only_when_its_manifest_is_in_the_models_folder(tmp_path):
    """Ollama keeps one manifest file per model: manifests/registry.ollama.ai/library/<name>/<tag>."""
    assert not ollama_model_is_downloaded(tmp_path, "qwen3:4b-instruct")
    manifest_file = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b-instruct"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}")
    assert ollama_model_is_downloaded(tmp_path, "qwen3:4b-instruct")
    # Here a model asked for without a tag means Ollama's "latest" tag, which isn't downloaded.
    assert not ollama_model_is_downloaded(tmp_path, "qwen3")


# ---------------------------------------------------------------------------
# 1. Settings (T01)
# ---------------------------------------------------------------------------

def test_config_holds_the_experiment_settings():
    """config.yaml loads and holds the settings the pipeline relies on."""
    config = load_config()
    # Here we check the settings the pipeline relies on are present and sensible.
    assert config["backend"] in {"ollama", "vllm"}
    assert config["rounds"] == 3
    assert set(config["max_tokens"]) == {"judge", "rewriter", "answerer"}


def test_setup_check_passes_on_this_machine(capsys):
    """run.py's setup check passes here (Python version, config, writable runs/)."""
    check_setup()
    assert "Setup OK." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. Questions: setup/get_questions.py and pipeline/dataset.py (T02)
# ---------------------------------------------------------------------------

def test_a_raw_row_becomes_a_question_with_each_gold_title_once_in_order():
    """A HotpotQA row keeps only our fields, and each Gold title appears once."""
    # Here we check the extra raw fields are dropped and "Ed Wood" appears once in gold_titles, not twice.
    assert raw_row_to_question(RAW_HOTPOTQA_ROW) == {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "gold_titles": ["Scott Derrickson", "Ed Wood"],
    }


def test_the_split_gives_pilot_and_test_sets_of_the_asked_sizes_with_no_question_in_both():
    """The Pilot and Test sets have the asked sizes and never share a question."""
    questions = make_fake_questions(20)
    pilot_set, test_set = split_pilot_and_test(questions, pilot_size=3, test_size=5, seed=0)
    pilot_ids = {question["id"] for question in pilot_set}
    test_ids = {question["id"] for question in test_set}
    assert len(pilot_set) == 3 and len(test_set) == 5
    assert pilot_ids.isdisjoint(test_ids)
    assert pilot_ids | test_ids <= {question["id"] for question in questions}


def test_the_same_seed_always_gives_the_same_split():
    """Every machine picks the same questions, because the seed fixes the random choice."""
    questions = make_fake_questions(20)
    assert split_pilot_and_test(questions, 3, 5, seed=0) == split_pilot_and_test(questions, 3, 5, seed=0)
    assert split_pilot_and_test(questions, 3, 5, seed=0) != split_pilot_and_test(questions, 3, 5, seed=1)


def test_building_the_question_sets_twice_writes_identical_files(tmp_path):
    """Building twice gives byte-identical files, and the id list matches the questions."""
    parquet_path = tmp_path / "dev.parquet"
    write_fake_hotpotqa(parquet_path, 10)
    questions_dir, question_ids_dir = tmp_path / "data", tmp_path / "ids"

    build_question_sets(parquet_path, questions_dir, question_ids_dir, pilot_size=2, test_size=4, seed=0)
    files_after_first_build = read_all_files(questions_dir, question_ids_dir)
    build_question_sets(parquet_path, questions_dir, question_ids_dir, pilot_size=2, test_size=4, seed=0)
    files_after_second_build = read_all_files(questions_dir, question_ids_dir)

    assert files_after_first_build == files_after_second_build
    # Here we check the id list matches the questions written.
    test_ids_written = (question_ids_dir / "hotpotqa_test.txt").read_text().split()
    assert test_ids_written == [question["id"] for question in load_question_set("test", questions_dir)]


def test_a_built_question_set_loads_back_and_a_question_can_be_found_by_id(tmp_path):
    """A written question set reads back unchanged, and find_question finds (or rejects) an id."""
    parquet_path = tmp_path / "dev.parquet"
    write_fake_hotpotqa(parquet_path, 10)
    build_question_sets(parquet_path, tmp_path / "data", tmp_path / "ids", pilot_size=2, test_size=4, seed=0)

    pilot_set = load_question_set("pilot", tmp_path / "data")
    assert len(pilot_set) == 2
    assert pilot_set[0]["gold_titles"] == ["Scott Derrickson", "Ed Wood"]
    assert find_question(pilot_set[1]["id"], pilot_set) == pilot_set[1]
    with pytest.raises(KeyError):
        find_question("not-a-real-id", pilot_set)


# ---------------------------------------------------------------------------
# 3. Search: pipeline/retriever.py and setup/build_index.py (T03)
# ---------------------------------------------------------------------------

def test_search_ranks_the_matching_paragraph_first(tiny_retriever):
    """Search puts the best-matching paragraph first, with its full text and the highest score."""
    results = tiny_retriever.search("lighthouse ships", k=3)
    assert len(results) == 3
    assert results[0]["title"] == "Lighthouse"
    assert results[0]["text"] == TINY_WIKI[4]["text"]
    assert results[0]["score"] > results[1]["score"]


def test_search_never_returns_paragraphs_already_in_the_evidence(tiny_retriever):
    """Search skips paragraphs already in the Evidence and still returns k new ones."""
    # Here we pretend "Ed Wood" is already in the Evidence: it must not come back, and we still get 3 results.
    results = tiny_retriever.search("Ed Wood film", k=3, exclude_titles={"Ed Wood"})
    assert len(results) == 3
    assert "Ed Wood" not in [result["title"] for result in results]
    assert results[0]["title"] == "Plan 9 from Outer Space"


def test_a_query_with_no_searchable_words_returns_no_paragraphs(tiny_retriever):
    """An empty query, or one of only common words, finds nothing instead of random paragraphs."""
    assert tiny_retriever.search("", k=3) == []
    assert tiny_retriever.search("the of and", k=3) == []


def test_asking_for_more_results_than_there_are_paragraphs_returns_them_all(tiny_retriever):
    """Asking for more results than exist returns what there is, not an error."""
    # 6 paragraphs, 1 excluded: we should get the 5 that are left, not an error.
    assert len(tiny_retriever.search("film", k=10, exclude_titles={"Volcano"})) == 5


def test_reading_the_dump_gives_one_paragraph_per_article_with_sentences_joined(tmp_path):
    """The Wikipedia dump reads as one paragraph per article, sentences joined."""
    dump_file = tmp_path / "wiki_00.bz2"
    write_fake_wiki_dump(dump_file, [
        ("Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker.", " He made Plan 9."]),
        ("Scott Derrickson", ["Scott Derrickson is an American director."]),
    ])
    assert list(read_wiki_paragraphs([dump_file])) == [
        {"title": "Ed Wood", "text": "Edward Davis Wood Jr. was an American filmmaker. He made Plan 9."},
        {"title": "Scott Derrickson", "text": "Scott Derrickson is an American director."},
    ]


class RangeAwareFileServer(http.server.BaseHTTPRequestHandler):
    """A tiny web server for tests: serves FILE_BYTES and, like a real server, honours "Range: bytes=<start>-"."""

    FILE_BYTES = bytes(range(256)) * 400   # 100 KB of known bytes

    def do_GET(self):
        """Answer a download request, honouring Range like a real server."""
        # Here we send only the bytes from <start> on (status 206), or the whole file if no Range was asked for.
        start = int(self.headers.get("Range", "bytes=0-").removeprefix("bytes=").removesuffix("-"))
        body = self.FILE_BYTES[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments):
        """Keep test output quiet (the server would otherwise print every request)."""
        pass


def test_an_interrupted_download_resumes_and_ends_with_the_complete_file(tmp_path):
    """A download cut off halfway continues where it stopped and ends complete."""
    test_server = http.server.HTTPServer(("127.0.0.1", 0), RangeAwareFileServer)
    threading.Thread(target=test_server.serve_forever, daemon=True).start()
    file_url = f"http://127.0.0.1:{test_server.server_port}/dump.tar.bz2"
    # Here we pretend an earlier download stopped after the first 30 KB.
    destination = tmp_path / "dump.tar.bz2"
    (tmp_path / "dump.tar.bz2.part").write_bytes(RangeAwareFileServer.FILE_BYTES[:30_000])

    download_with_resume(file_url, destination)
    test_server.shutdown()

    assert destination.read_bytes() == RangeAwareFileServer.FILE_BYTES
    assert not (tmp_path / "dump.tar.bz2.part").exists()


def test_an_interrupted_unpack_is_redone_so_no_dump_file_is_missing(tmp_path):
    """An unpack cut off halfway is redone, so the index is never built on part of Wikipedia."""
    # Here we make a small archive holding two dump files, as if it had already been downloaded.
    source_dir = tmp_path / "source" / "AA"
    source_dir.mkdir(parents=True)
    write_fake_wiki_dump(source_dir / "wiki_00.bz2", [("Ed Wood", ["Ed Wood was a filmmaker."])])
    write_fake_wiki_dump(source_dir / "wiki_01.bz2", [("Volcano", ["A volcano erupts."])])
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    with tarfile.open(dump_dir / "dump.tar.bz2", "w:bz2") as archive:
        archive.add(source_dir, arcname="AA")
    # Here we pretend an earlier unpack stopped after only one of the two files.
    (dump_dir / "unpacked.part" / "AA").mkdir(parents=True)
    (dump_dir / "unpacked.part" / "AA" / "wiki_00.bz2").write_bytes(b"")

    dump_files = download_and_unpack_dump("http://unused/dump.tar.bz2", dump_dir)

    assert [dump_file.name for dump_file in dump_files] == ["wiki_00.bz2", "wiki_01.bz2"]
    assert [paragraph["title"] for paragraph in read_wiki_paragraphs(dump_files)] == ["Ed Wood", "Volcano"]


# ---------------------------------------------------------------------------
# 4. Search quality: setup/check_index.py (T03)
# ---------------------------------------------------------------------------

def test_gold_recall_counts_questions_with_all_or_at_least_one_gold_paragraph_in_the_top_k(tiny_retriever):
    """The recall table counts all-found and at-least-one-found correctly."""
    questions = [
        {"question": "Were Scott Derrickson and Ed Wood of the same nationality?",
         "gold_titles": ["Scott Derrickson", "Ed Wood"]},
        {"question": "Which tower guides ships at sea?", "gold_titles": ["Lighthouse", "Volcano"]},
    ]
    # k=1: one result can hold at most one of two Gold paragraphs; each question finds one.
    # k=6: the whole corpus comes back, so every Gold paragraph is found.
    assert gold_recall_at_k(questions, tiny_retriever, cutoffs=[1, 6]) == [
        {"k": 1, "all_found": 0.0, "at_least_one_found": 1.0},
        {"k": 6, "all_found": 1.0, "at_least_one_found": 1.0},
    ]


def test_gold_recall_needs_every_gold_paragraph_when_there_are_more_than_two(tiny_retriever):
    """"All found" means every Gold paragraph, even when a question has three."""
    three_hop_question = {"question": "Ed Wood Plan 9 Lighthouse",
                          "gold_titles": ["Ed Wood", "Plan 9 from Outer Space", "Lighthouse"]}
    # The top 2 can hold at most two of the three, so "all found" must be 0.
    assert gold_recall_at_k([three_hop_question], tiny_retriever, cutoffs=[2])[0]["all_found"] == 0.0


# ---------------------------------------------------------------------------
# 5. The model: pipeline/llm.py (T04)
# ---------------------------------------------------------------------------

def test_every_role_uses_the_main_model_unless_a_judge_model_is_set():
    """Every role uses `model`; setting judge_model changes only the Judge (Tier 3)."""
    config = {"model": "Qwen/Qwen3-4B-Instruct-2507", "judge_model": None}
    assert model_for("judge", config) == "Qwen/Qwen3-4B-Instruct-2507"
    # Tier 3: only the Judge changes.
    config["judge_model"] = "Qwen/Qwen3-14B-AWQ"
    assert model_for("judge", config) == "Qwen/Qwen3-14B-AWQ"
    assert model_for("answerer", config) == "Qwen/Qwen3-4B-Instruct-2507"


def test_each_model_has_an_ollama_name_and_an_unknown_model_says_where_to_add_it():
    """Each model's Ollama name is known; an unknown model says where to add it."""
    assert ollama_name_for("Qwen/Qwen3-4B-Instruct-2507") == "qwen3:4b-instruct"
    with pytest.raises(KeyError, match="OLLAMA_MODEL_NAMES"):
        ollama_name_for("Some/Unknown-Model")


@pytest.mark.skipif(shutil.which("ollama") is None or ollama_is_running(),
                    reason="needs Ollama installed and not already running (a running server is left alone)")
def test_the_launcher_opens_the_server_and_closes_it_again(tmp_path):
    """start_ollama_server opens a server that keeps running; stop_ollama_server closes it."""
    # Here the server's details and log go in a temporary folder, not the real data/ and runs/ folders.
    server_file = tmp_path / "server.json"
    assert start_ollama_server(server_file=server_file, logs_dir=tmp_path) is True
    assert ollama_is_running()
    # Here a second start sees the server is already open and starts nothing.
    assert start_ollama_server(server_file=server_file, logs_dir=tmp_path) is False

    assert stop_ollama_server(server_file=server_file) is True
    assert not ollama_is_running()
    assert not server_file.exists()


def test_closing_does_nothing_when_the_launcher_did_not_open_a_server(tmp_path):
    """With no server opened by the launcher, closing stops nothing (a server started by hand is left alone)."""
    assert stop_ollama_server(server_file=tmp_path / "server.json") is False


@needs_ollama
def test_ollama_answers_with_token_counts_and_probabilities_for_every_generated_token():
    """Ollama answers correctly, with token counts and the top 5 alternatives for every token."""
    config = load_config() | {"backend": "ollama"}
    prompt = "Answer with one word: what is the capital of France?"
    generation = generate("answerer", prompt, config)

    assert "Paris" in generation["text"]
    assert generation["prompt_tokens"] > 0
    # completion_tokens can include the end-of-answer token, which has no probability entry.
    assert 0 < len(generation["logprobs"]) <= generation["completion_tokens"]
    for token_entry in generation["logprobs"]:
        assert token_entry["logprob"] <= 0
        assert len(token_entry["top"]) == 5
        assert token_entry["token"] == token_entry["top"][0]["token"]   # the chosen token is the most likely
    assert generate("answerer", prompt, config)["text"] == generation["text"]   # same prompt, same reply


@needs_ollama
def test_a_batch_of_prompts_comes_back_as_one_answer_per_prompt_in_order():
    """Several prompts at once come back as one answer each, in the same order."""
    config = load_config() | {"backend": "ollama"}
    generations = generate_many("answerer", ["Answer with one word: what is the capital of France?",
                                             "Answer with one word: what is the capital of Japan?"], config)
    assert "Paris" in generations[0]["text"]
    assert "Tokyo" in generations[1]["text"]


# ---------------------------------------------------------------------------
# 6. Answerer and the loop: pipeline/answerer.py and pipeline/loop.py (skeleton until T06-T08)
# ---------------------------------------------------------------------------

def test_the_answer_prompt_shows_the_evidence_only_when_there_is_some():
    """The Answerer's prompt lists the Evidence when there is some, and none for Closed-book."""
    lighthouse = {"title": "Lighthouse", "text": "A lighthouse guides ships.", "score": 1.0}
    with_evidence = build_answer_prompt("What guides ships?", [lighthouse])
    from_memory = build_answer_prompt("What guides ships?", [])
    assert "Lighthouse: A lighthouse guides ships." in with_evidence
    assert "Evidence" not in from_memory
    assert "Question: What guides ships?" in from_memory


def test_every_scenario_has_a_function():
    """Every scenario in the study has a function run.py can call."""
    assert set(SCENARIOS) == {"closed-book", "single-turn", "always-loop", "coin-flip", "A", "B", "C", "D"}


def test_always_loop_never_asks_to_stop():
    """Always-loop's stop rule always says keep going."""
    assert never_stop(question={}, evidence=[], config={}) is False


def test_scenarios_not_built_yet_say_which_ticket_builds_them(tiny_retriever):
    """Scenarios not built yet stop with a message naming the ticket that builds them."""
    question = {"id": "q1", "question": "Which tower guides ships?", "gold_titles": ["Lighthouse"]}
    for unbuilt_setting in ["always-loop", "coin-flip", "A", "B", "C", "D"]:
        with pytest.raises(NotImplementedError, match="T0"):
            SCENARIOS[unbuilt_setting](question, tiny_retriever, load_config())


@needs_ollama
def test_single_turn_searches_once_and_answers_from_what_it_found(tiny_retriever):
    """Single-turn searches once, uses those paragraphs, and answers from them."""
    config = load_config() | {"backend": "ollama", "paragraphs_per_round": 2}
    question = {"id": "q1", "question": "Which tower emits light to guide ships at sea?", "answer": "a lighthouse",
                "gold_titles": ["Lighthouse"]}
    result = single_turn(question, tiny_retriever, config)
    only_round = result["rounds"][0]
    assert result["rounds_used"] == 1
    assert only_round["retriever"]["query"] == question["question"]
    assert only_round["retriever"]["retrieved"][0]["title"] == "Lighthouse"
    assert only_round["answerer"]["paragraphs_seen"][0] == "Lighthouse"
    assert only_round["judge"] is None and only_round["rewriter"] is None
    assert "lighthouse" in result["final_answer"].lower()


@needs_ollama
def test_closed_book_answers_from_memory_without_searching(tiny_retriever):
    """Closed-book answers from the model's memory, with no search."""
    config = load_config() | {"backend": "ollama"}
    question = {"id": "q1", "question": "What is the capital of France?", "answer": "Paris", "gold_titles": []}
    result = closed_book(question, tiny_retriever, config)
    assert result["rounds_used"] == 0
    assert result["rounds"][0]["retriever"] is None
    assert result["rounds"][0]["answerer"]["paragraphs_seen"] == []
    assert "Paris" in result["final_answer"]


# ---------------------------------------------------------------------------
# 7. Run logs: pipeline/run_log.py (T05)
# ---------------------------------------------------------------------------

def test_every_scenario_has_its_own_log_folder_name():
    """Each of the 8 scenarios logs into its own folder; A-D names say who stops and who steers."""
    assert set(SCENARIO_FOLDER_NAMES) == set(SCENARIOS)
    assert SCENARIO_FOLDER_NAMES["A"] == "A_llm-stops_llm-steers"
    assert SCENARIO_FOLDER_NAMES["B"] == "B_llm-stops_oracle-steers"
    assert SCENARIO_FOLDER_NAMES["C"] == "C_oracle-stops_llm-steers"
    assert SCENARIO_FOLDER_NAMES["D"] == "D_oracle-stops_oracle-steers"


def test_a_scenario_log_folder_is_created_the_first_time_and_reused_after(tmp_path):
    """The first run of a scenario creates runs/<scenario folder>/; later runs reuse it."""
    assert not (tmp_path / "single-turn").exists()
    first_folder = scenario_log_folder("single-turn", runs_dir=tmp_path)
    assert first_folder == tmp_path / "single-turn" and first_folder.is_dir()
    assert scenario_log_folder("single-turn", runs_dir=tmp_path) == first_folder
    # Here we check no other scenario's folder was made along the way.
    assert [folder.name for folder in tmp_path.iterdir()] == ["single-turn"]


# A stand-in model reply, with token counts, for building records without a model.
FAKE_REPLY = {"text": " Lighthouse ", "prompt_tokens": 40, "completion_tokens": 3, "logprobs": []}
LIGHTHOUSE_PARAGRAPH = {"title": "Lighthouse", "text": "A lighthouse guides ships.", "score": 2.5}


def make_fake_question_set(question_count):
    """Stand-in questions with everything a record copies: id, English question, gold answer, Gold titles."""
    return [{"id": f"q{number}", "question": f"Question number {number}?", "answer": "Lighthouse",
             "gold_titles": ["Lighthouse"]} for number in range(question_count)]


def fake_single_turn(question, retriever, config):
    """A Single-turn stand-in that returns a real-shaped record without searching or calling a model."""
    rounds = [make_round(1, make_retriever_entry(question["question"], [LIGHTHOUSE_PARAGRAPH]),
                         make_answerer_entry([LIGHTHOUSE_PARAGRAPH], FAKE_REPLY))]
    return make_result(question, rounds, final_answer=FAKE_REPLY["text"].strip())


def test_a_record_holds_the_question_what_was_retrieved_and_what_the_answerer_saw():
    """One question's record: the question, the query and full paragraphs retrieved, the answerer's view, totals."""
    question = make_fake_question_set(1)[0]
    record = fake_single_turn(question, retriever=None, config={})
    only_round = record["rounds"][0]
    assert record["question_id"] == "q0" and record["question"] == "Question number 0?"
    assert record["gold_answer"] == "Lighthouse" and record["gold_titles"] == ["Lighthouse"]
    assert only_round["retriever"]["retrieved"] == [LIGHTHOUSE_PARAGRAPH]   # full text and score, not just the title
    assert only_round["answerer"]["paragraphs_seen"] == ["Lighthouse"]
    assert only_round["answerer"]["prompt_version"] == "draft"
    assert only_round["answerer"]["answer"] == "Lighthouse"
    assert record["rounds_used"] == 1 and record["llm_calls"] == 1
    assert record["total_prompt_tokens"] == 40 and record["total_completion_tokens"] == 3


def test_a_run_writes_its_meta_and_one_record_per_question_in_the_scenario_folder(tmp_path):
    """A 5-question run writes meta.json and 5 timed records under runs/<scenario>/<run id>/."""
    questions = make_fake_question_set(5)
    run_folder = run_scenario("single-turn", fake_single_turn, questions, "pilot", load_config(),
                              retriever=None, runs_dir=tmp_path)

    assert run_folder.parent == tmp_path / "single-turn"
    meta = read_meta(run_folder)
    assert meta["scenario"] == "single-turn" and meta["question_set"] == "pilot"
    assert meta["question_ids"] == ["q0", "q1", "q2", "q3", "q4"]
    assert meta["config"] == load_config() and meta["finished_at"] is not None
    records = read_question_records(run_folder)
    assert [record["question_id"] for record in records] == ["q0", "q1", "q2", "q3", "q4"]
    assert all(record["started_at"] <= record["finished_at"] for record in records)


def test_a_stopped_run_resumes_and_finishes_with_no_duplicates(tmp_path):
    """A run that crashes partway (even mid-line) continues where it stopped when started again."""
    questions = make_fake_question_set(5)

    def crash_on_the_third_question(question, retriever, config):
        """Stand-in that crashes at q2, like a Kaggle session ending."""
        if question["id"] == "q2":
            raise RuntimeError("session ended")
        return fake_single_turn(question, retriever, config)

    with pytest.raises(RuntimeError):
        run_scenario("single-turn", crash_on_the_third_question, questions, "pilot", load_config(),
                     retriever=None, runs_dir=tmp_path)
    [stopped_run] = (tmp_path / "single-turn").iterdir()
    assert read_meta(stopped_run)["finished_at"] is None
    # Here we pretend the crash also cut the next line off halfway through writing it.
    with open(stopped_run / "questions.jsonl", "a") as questions_file:
        questions_file.write('{"question_id": "q2", "quest')

    resumed_run = run_scenario("single-turn", fake_single_turn, questions, "pilot", load_config(),
                               retriever=None, runs_dir=tmp_path)

    assert resumed_run == stopped_run
    assert [record["question_id"] for record in read_question_records(resumed_run)] == ["q0", "q1", "q2", "q3", "q4"]
    assert read_meta(resumed_run)["finished_at"] is not None


def test_a_finished_run_is_kept_and_running_again_starts_a_new_run(tmp_path):
    """Running the same scenario again after it finished starts a fresh run folder instead of adding to the old one."""
    questions = make_fake_question_set(2)
    first_run = run_scenario("single-turn", fake_single_turn, questions, "pilot", load_config(),
                             retriever=None, runs_dir=tmp_path)
    second_run = run_scenario("single-turn", fake_single_turn, questions, "pilot", load_config(),
                              retriever=None, runs_dir=tmp_path)
    assert second_run != first_run
    assert len(read_question_records(first_run)) == 2 and len(read_question_records(second_run)) == 2


if __name__ == "__main__":
    # Here pressing Run on this file runs every test in it and reports passed / skipped / failed.
    raise SystemExit(pytest.main([__file__]))
