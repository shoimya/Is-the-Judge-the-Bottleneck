"""Small checks for the pipeline. Run with `pytest`."""

import json
import math
from datetime import datetime

import pandas as pd
import pytest

from evaluate import load_run
from pipeline import dataset, llm, load_config, retriever
from run import make_run_id, run_questions, step_fake


def test_config_loads():
    cfg = load_config()
    assert cfg["loop"]["rounds"] == 3
    assert set(cfg["models"]) == {"judge", "rewriter", "answerer"}


def make_item(qid, question="q", answer="a", qtype="bridge", level="hard", titles=None):
    """A minimal HotpotQA dev item, like the parquet rows dataset.py reads."""
    titles = titles or ["Title A", "Title B"]
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": qtype,
        "level": level,
        "supporting_facts": {"title": titles, "sent_id": [0] * len(titles)},
    }


def test_gold_titles():
    assert dataset.gold_titles(make_item("x", titles=["A", "B"])) == ["A", "B"]
    # several fact sentences across the same two paragraphs -> still 2 titles
    assert dataset.gold_titles(make_item("x", titles=["A", "B", "A", "B"])) == ["A", "B"]
    # all facts in one paragraph -> no second hop, drop it
    assert dataset.gold_titles(make_item("x", titles=["A", "A"])) is None
    assert dataset.gold_titles(make_item("x", titles=["A"])) is None


def test_to_record():
    item = make_item("abc", question="Who?", answer="yes", qtype="comparison", titles=["P", "Q"])
    assert dataset.to_record(item, ["P", "Q"]) == {
        "id": "abc",
        "question": "Who?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "gold_titles": ["P", "Q"],
    }


def test_sample_ids_deterministic_and_disjoint():
    pool = [f"id{i:04d}" for i in range(3000)]
    pilot_a, test_a = dataset.sample_ids(pool, seed=0)
    pilot_b, test_b = dataset.sample_ids(pool, seed=0)
    assert pilot_a == pilot_b
    assert test_a == test_b
    assert len(pilot_a) == 100 and len(test_a) == 1000
    assert not set(pilot_a) & set(test_a)  # disjoint
    assert set(pilot_a) | set(test_a) <= set(pool)


def test_build_writes_identical_files(tmp_path):
    items = [make_item(f"id{i:04d}") for i in range(200)]
    items.append(make_item("id-same-paragraph", titles=["X", "X"]))  # must be dropped
    kwargs = dict(
        items=items,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        seed=7,
        pilot_size=20,
        test_size=50,
    )

    dataset.build(**kwargs)
    outputs = {
        p: p.read_bytes()
        for p in [
            tmp_path / "data" / "pilot.jsonl",
            tmp_path / "data" / "test.jsonl",
            tmp_path / "results" / "question_ids" / "pilot_ids.txt",
            tmp_path / "results" / "question_ids" / "test_ids.txt",
        ]
    }

    # every question has 2 gold titles; pilot and test share no ids
    pilot = [json.loads(line) for line in outputs[tmp_path / "data" / "pilot.jsonl"].decode().splitlines()]
    test = [json.loads(line) for line in outputs[tmp_path / "data" / "test.jsonl"].decode().splitlines()]
    assert len(pilot) == 20 and len(test) == 50
    assert all(len(q["gold_titles"]) == 2 for q in pilot + test)
    assert "id-same-paragraph" not in {q["id"] for q in pilot + test}
    assert not {q["id"] for q in pilot} & {q["id"] for q in test}
    # the committed id lists match the jsonl files
    assert outputs[tmp_path / "results" / "question_ids" / "pilot_ids.txt"].decode().splitlines() == [q["id"] for q in pilot]
    assert outputs[tmp_path / "results" / "question_ids" / "test_ids.txt"].decode().splitlines() == [q["id"] for q in test]

    # running it again gives byte-identical files
    dataset.build(**kwargs)
    for p, expected in outputs.items():
        assert p.read_bytes() == expected


def build_tiny_index(tmp_path, n_docs=25, batch_size=7):
    """A tiny BM25 index built through the real build_index() path."""
    import bz2

    words = ["alpha", "beta", "gamma", "delta", "crimson", "quasar", "nebula", "orbit"]
    abstracts = [
        {
            "title": f"Title {i:03d}",
            "text": [
                " ".join(words[(i + j) % len(words)] for j in range(8)),
                f"zymurgy mention {i}" if i == 3 else "plain sentence",
            ],
        }
        for i in range(n_docs)
    ]
    # two json-lines bz2 shards, like the real wiki_*.bz2 corpus
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    half = n_docs // 2
    for shard_i, chunk in enumerate([abstracts[:half], abstracts[half:]]):
        with bz2.open(shards_dir / f"wiki_{shard_i:02d}.bz2", "wt", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item) + "\n")
    index_dir = tmp_path / "index"
    summary = retriever.build_index(shards_dir, index_dir, batch_size=batch_size)
    assert summary["docs"] == n_docs
    return retriever.load_retriever(index_dir)


def test_search_returns_k_sorted_results(tmp_path):
    r = build_tiny_index(tmp_path)
    hits = retriever.search("alpha beta", k=5, retriever=r)
    assert len(hits) == 5
    assert all(set(h) == {"title", "text", "score"} for h in hits)
    assert [h["score"] for h in hits] == sorted([h["score"] for h in hits], reverse=True)


def test_search_finds_the_right_paragraph(tmp_path):
    r = build_tiny_index(tmp_path)
    hits = retriever.search("zymurgy", k=3, retriever=r)
    assert hits[0]["title"] == "Title 003"


def test_search_excludes_titles_and_tops_up(tmp_path):
    r = build_tiny_index(tmp_path)
    top = retriever.search("alpha beta", k=5, retriever=r)[0]["title"]
    hits = retriever.search("alpha beta", k=5, exclude_titles=[top], retriever=r)
    assert len(hits) == 5
    assert all(h["title"] != top for h in hits)


def test_search_handles_any_query(tmp_path):
    r = build_tiny_index(tmp_path)
    assert len(retriever.search("zzzqqq unheard of", k=4, retriever=r)) == 4
    assert len(retriever.search("the of", k=4, retriever=r)) == 4


def test_sanity_recall_csv_matches_its_header(tmp_path):
    import csv

    r = build_tiny_index(tmp_path)
    questions = [
        {"id": "q1", "question": "zymurgy mention", "gold_titles": ["Title 003", "Title 001"]},
        {"id": "q2", "question": "alpha beta", "gold_titles": ["Title 000", "Title 024"]},
    ]
    pilot = tmp_path / "pilot.jsonl"
    with open(pilot, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    out_csv = tmp_path / "recall.csv"

    recalls = retriever.sanity_recall(pilot, out_csv, retriever=r)

    rows = list(csv.DictReader(open(out_csv)))
    assert [row["id"] for row in rows] == ["q1", "q2"]
    # gold hits are monotone in k (a hit at k=2 implies hits at 5, 10, 20)
    for row in rows:
        for g in (1, 2):
            for k_small, k_big in ((2, 5), (5, 10), (10, 20)):
                assert row[f"gold{g}@{k_small}"] <= row[f"gold{g}@{k_big}"], row
    # the zymurgy paragraph is the top hit for the zymurgy question
    assert rows[0]["gold1@2"] == "1"
    # the returned recalls match what the CSV holds
    assert rows[0]["gold1@20"] == ("1" if recalls["gold1@20"] else "0")
    assert all(0.0 <= v <= 1.0 for v in recalls.values())


def make_llm_config(backend="fake", models=None):
    """A config with a fake backend: every role/model/max_tokens is distinct."""
    cfg = load_config()
    cfg["models"] = {"judge": "judge-model", "rewriter": "rewriter-model",
                     "answerer": "answerer-model", **(models or {})}
    cfg["llm"] = {"backend": backend, "temperature": 0,
                  "max_tokens": {"judge": 111, "rewriter": 222, "answerer": 333}}
    return cfg


def test_llm_plan_routes_roles_to_models(monkeypatch):
    monkeypatch.setitem(llm._BACKENDS, "fake", lambda *args: None)
    cfg = make_llm_config()
    assert llm.plan("judge", cfg) == ("fake", "judge-model", 111, 0)
    assert llm.plan("rewriter", cfg)[1] == "rewriter-model"
    assert llm.plan("answerer", cfg)[2] == 333
    # changing models.judge changes what the Judge gets, without touching code
    cfg["models"]["judge"] = "qwen3:14b"
    assert llm.plan("judge", cfg)[1] == "qwen3:14b"
    assert llm.plan("rewriter", cfg)[1] == "rewriter-model"  # other roles untouched


def test_generate_dispatches_to_configured_backend(monkeypatch):
    calls = []
    def fake_backend(model, prompt, max_tokens, temperature):
        calls.append((model, prompt, max_tokens, temperature))
        return {"text": "ok"}

    monkeypatch.setitem(llm._BACKENDS, "fake", fake_backend)
    out = llm.generate("rewriter", "hello", config=make_llm_config())
    assert out == {"text": "ok"}
    assert calls == [("rewriter-model", "hello", 222, 0)]


def test_llm_rejects_unknown_role_and_backend():
    cfg = make_llm_config()
    with pytest.raises(ValueError):
        llm.plan("narrator", cfg)
    cfg["llm"]["backend"] = "telepathy"
    with pytest.raises(ValueError):
        llm.plan("judge", cfg)


def test_p_yes_sums_yes_spellings():
    table = [
        {"token": "Yes", "logprob": -0.3},
        {"token": "no", "logprob": -0.1},
        {"token": " yes", "logprob": -1.5},
    ]
    assert llm.p_yes(table) == pytest.approx(math.exp(-0.3) + math.exp(-1.5))
    assert llm.p_yes([{"token": "No", "logprob": -0.2}]) is None
    assert llm.p_yes(None) is None


# ------------------------------------------------------------------ T05 run log


def fake_questions(n=5):
    return [{"id": f"q{i}", "question": f"question {i}?", "answer": f"answer {i}",
             "gold_titles": [f"Title {i}a", f"Title {i}b"]} for i in range(n)]


def read_log(folder):
    return [json.loads(line) for line in
            (folder / "questions.jsonl").read_text(encoding="utf-8").splitlines()]


def test_make_run_id_is_time_plus_condition():
    assert make_run_id("A", now=datetime(2026, 10, 14, 15, 30)) == "2026-10-14T1530_A"


def test_fake_run_writes_all_files_and_load_run_reads_them_back(tmp_path):
    runs_dir = tmp_path / "runs"
    summary = run_questions(fake_questions(), step_fake, condition="fake",
                            dataset="pilot", run_dir=runs_dir)
    folder = summary["run_dir"]
    assert folder.name == summary["run_id"]

    meta = json.loads((folder / "meta.json").read_text())
    for key in ("run_id", "datetime", "condition", "dataset", "question_set",
                "models", "backend", "git_commit", "config"):
        assert key in meta
    assert meta["condition"] == "fake" and meta["dataset"] == "pilot"
    assert meta["models"] == load_config()["models"]          # config snapshot
    assert meta["config"]["llm"]["backend"] == meta["backend"]
    assert meta["prompt_versions"] == {"judge": "v0", "rewriter": "v0", "answerer": "v0"}

    records = read_log(folder)
    assert len(records) == 5
    first = records[0]
    for key in ("id", "question", "gold_answer", "gold_titles", "rounds_used",
                "final_answer", "em", "f1", "total_prompt_tokens",
                "total_completion_tokens", "llm_calls", "rounds"):
        assert key in first
    assert first["llm_calls"] == {"main": 6, "shadow": 2}     # shadow counted separately
    assert len(first["rounds"]) == 3
    for key in ("query", "retrieved_titles", "judge_raw", "stop", "p_yes",
                "steer", "shadow_answer", "shadow_em", "tokens"):
        assert key in first["rounds"][0]

    # load_run reads the folder back, meta attached, one row per question
    df = load_run(folder)
    assert len(df) == 5
    assert set(df["id"]) == {f"q{i}" for i in range(5)}
    assert df.attrs["meta"]["run_id"] == folder.name
    assert all(len(rounds) == 3 for rounds in df["rounds"])


def test_run_resumes_after_death_without_duplicates(tmp_path):
    runs_dir = tmp_path / "runs"
    processed = {"n": 0}

    def dying_step(question):
        processed["n"] += 1
        if processed["n"] > 2:
            raise KeyboardInterrupt("fake session death")
        return step_fake(question)

    with pytest.raises(KeyboardInterrupt):
        run_questions(fake_questions(), dying_step, condition="fake",
                      dataset="pilot", run_dir=runs_dir)

    # restart: same condition+dataset resumes the same folder and finishes
    summary = run_questions(fake_questions(), step_fake, condition="fake",
                            dataset="pilot", run_dir=runs_dir)
    assert summary["new"] == 3 and summary["skipped"] == 2
    ids = [record["id"] for record in read_log(summary["run_dir"])]
    assert len(ids) == 5 and len(set(ids)) == 5                  # no duplicates
    assert [folder.name for folder in runs_dir.iterdir()] == [summary["run_id"]]


def test_fresh_run_starts_a_new_folder(tmp_path):
    runs_dir = tmp_path / "runs"
    first = run_questions(fake_questions(), step_fake, condition="fake",
                          dataset="pilot", run_dir=runs_dir)
    second = run_questions(fake_questions(), step_fake, condition="fake",
                           dataset="pilot", run_dir=runs_dir, resume=False)
    assert second["run_dir"] != first["run_dir"]
    assert second["new"] == 5 and len(read_log(second["run_dir"])) == 5


# ------------------------------------------------------ T06 roles and parsing


def test_p_yes_normalized_is_conditional_probability():
    table = [
        {"token": " yes", "logprob": -0.1},
        {"token": " no", "logprob": -0.4},
        {"token": "No", "logprob": -1.2},
    ]
    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-0.4) + math.exp(-1.2))
    assert llm.p_yes_normalized(table) == pytest.approx(expected)
    assert llm.p_yes_normalized([{"token": "maybe", "logprob": -0.1}]) is None
    assert llm.p_yes_normalized(None) is None


def judge_out(text, sampled_tokens, tables=None):
    """A generate() result whose sampled tokens spell `text` position by
    position; `tables` replaces the table at the given indexes."""
    tables = tables or {}
    logprobs = []
    for i, token in enumerate(sampled_tokens):
        logprobs.append(tables.get(i, [{"token": token, "logprob": -0.1, "sampled": True}]))
    return {"text": text, "prompt_tokens": 1, "completion_tokens": 1,
            "logprobs": logprobs}


def test_judge_parse_reads_fields_and_p_yes_at_enough():
    from pipeline import judge

    verdict = [{"token": " no", "logprob": -0.3, "sampled": True},
               {"token": " yes", "logprob": -0.8}]
    tokens = (["REASONING", ":", " the", " evidence", " fits", ".\n",
               "ENOUGH", ":", " no", "\n", "MISSING", ":", " nothing"])
    parsed = judge.parse(judge_out(
        "REASONING: the evidence fits.\nENOUGH: no\nMISSING: nothing",
        tokens, tables={8: verdict}))

    assert parsed["reasoning"] == "the evidence fits."
    assert parsed["enough"] == "no"
    assert parsed["missing"] == "nothing"
    assert not parsed["parse_error"]
    # p_yes comes from the ENOUGH position's table, normalized
    assert parsed["p_yes"] == pytest.approx(llm.p_yes_normalized(verdict))


def test_judge_p_yes_ignores_yes_no_in_reasoning():
    from pipeline import judge

    reasoning_yes = [{"token": "Yes", "logprob": -0.01, "sampled": True},
                     {"token": "No", "logprob": -2.0}]
    verdict = [{"token": " no", "logprob": -0.2, "sampled": True},
               {"token": " yes", "logprob": -1.1}]
    tokens = (["REASONING", ":", " Yes", ",", " the", " date", " is", " missing", ".\n",
               "ENOUGH", ":", " no"])
    parsed = judge.parse(judge_out(
        "REASONING: Yes, the date is missing.\nENOUGH: no",
        tokens, tables={2: reasoning_yes, 11: verdict}))

    assert parsed["enough"] == "no"
    # the "Yes" in REASONING must not win over the " no" at ENOUGH
    assert parsed["p_yes"] == pytest.approx(llm.p_yes_normalized(verdict))


def test_judge_p_yes_falls_back_when_header_is_not_a_token():
    from pipeline import judge

    verdict = [{"token": " yes", "logprob": -0.4, "sampled": True},
               {"token": " no", "logprob": -0.9}]
    parsed = judge.parse(judge_out(  # "ENOUGH" split across tokens
        "ENOUGH: yes", ["EN", "OUGH", ":", " yes"], tables={3: verdict}))
    assert parsed["p_yes"] == pytest.approx(llm.p_yes_normalized(verdict))


def test_judge_malformed_counts_as_no_and_flags():
    from pipeline import judge

    parsed = judge.parse({"text": "I think the evidence is enough.",
                          "prompt_tokens": 1, "completion_tokens": 1, "logprobs": []})
    assert parsed["enough"] == "no"
    assert parsed["parse_error"] is True
    assert parsed["reasoning"] == "" and parsed["missing"] == ""
    assert parsed["p_yes"] is None


def test_judge_ask_builds_prompt_and_parses(monkeypatch):
    from pipeline import judge

    captured = {}

    def fake_backend(model, prompt, max_tokens, temperature):
        captured.update(model=model, prompt=prompt)
        return {"text": "REASONING: it fits.\nENOUGH: yes\nMISSING: nothing",
                "prompt_tokens": 11, "completion_tokens": 7, "logprobs": []}

    monkeypatch.setitem(llm._BACKENDS, "fake", fake_backend)
    result = judge.ask("Which film?", [{"title": "Titanic", "text": "A 1997 film."}],
                       config=make_llm_config())
    assert captured["model"] == "judge-model"
    assert "Which film?" in captured["prompt"]
    assert "[1] Titanic\nA 1997 film." in captured["prompt"]
    assert result["enough"] == "yes" and result["reasoning"] == "it fits."
    assert result["prompt_version"] == "v0"


def test_render_evidence_numbers_paragraphs():
    from pipeline.judge import render_evidence

    evidence = [{"title": "Title A", "text": "Text A."},
                {"title": "Title B", "text": "Text B."}]
    assert render_evidence(evidence) == "[1] Title A\nText A.\n\n[2] Title B\nText B."
    assert render_evidence([]) == ""


def test_rewriter_parse_takes_first_clean_line():
    from pipeline import rewriter

    parsed = rewriter.parse({"text": ' "Titanic 1997 Oscars" \n\nextra nonsense\n'})
    assert parsed == {"query": "Titanic 1997 Oscars", "parse_error": False}
    assert rewriter.parse({"text": "\n \n"}) == {"query": "", "parse_error": True}


def test_rewriter_ask_sees_question_queries_and_steer(monkeypatch):
    from pipeline import rewriter

    captured = {}

    def fake_backend(model, prompt, max_tokens, temperature):
        captured.update(model=model, prompt=prompt)
        return {"text": "Titanic 1997", "prompt_tokens": 5, "completion_tokens": 2,
                "logprobs": []}

    monkeypatch.setitem(llm._BACKENDS, "fake", fake_backend)
    result = rewriter.ask("Which film?", ["Which 1997 film"], "its director",
                          config=make_llm_config())
    assert captured["model"] == "rewriter-model"
    assert "Which film?" in captured["prompt"]
    assert "- Which 1997 film" in captured["prompt"]
    assert "its director" in captured["prompt"]
    assert result["query"] == "Titanic 1997" and result["prompt_version"] == "v0"


def test_answerer_parse_cleans_answer():
    from pipeline import answerer

    assert answerer.parse({"text": "  The Yoruba. \n"}) == {
        "answer": "The Yoruba", "parse_error": False}
    assert answerer.parse({"text": "Paris"}) == {"answer": "Paris", "parse_error": False}
    assert answerer.parse({"text": "   "}) == {"answer": "", "parse_error": True}


def test_answerer_ask_sees_question_and_evidence(monkeypatch):
    from pipeline import answerer

    captured = {}

    def fake_backend(model, prompt, max_tokens, temperature):
        captured.update(model=model, prompt=prompt)
        return {"text": "Titanic", "prompt_tokens": 5, "completion_tokens": 1,
                "logprobs": []}

    monkeypatch.setitem(llm._BACKENDS, "fake", fake_backend)
    result = answerer.ask("Which film?", [{"title": "Titanic", "text": "A film."}],
                          config=make_llm_config())
    assert captured["model"] == "answerer-model"
    assert "Which film?" in captured["prompt"] and "[1] Titanic" in captured["prompt"]
    assert result["answer"] == "Titanic" and result["prompt_version"] == "v0"


def test_role_prompt_versions_are_v0():
    from pipeline import answerer, judge, rewriter

    versions = (judge.PROMPT_VERSION, rewriter.PROMPT_VERSION, answerer.PROMPT_VERSION)
    assert versions == ("v0", "v0", "v0")


# ------------------------------------------------------------- T07 oracle judge


def test_oracle_stop_and_steer_cover_the_combinations():
    from pipeline import judge

    gold = ["Ida (sword)", "Yoruba people"]
    # nothing retrieved: not enough, and the first gold title is the steer
    assert judge.oracle_stop([], gold) is False
    assert judge.oracle_steer([], gold) == "Missing information about: Ida (sword)"
    # one of two found: still not enough; steer names the second, in order
    assert judge.oracle_stop(["Ida (sword)"], gold) is False
    assert judge.oracle_steer(["Ida (sword)"], gold) == \
        "Missing information about: Yoruba people"
    # both found (any order): enough, nothing missing
    assert judge.oracle_stop(["Yoruba people", "Ida (sword)"], gold) is True
    assert judge.oracle_steer(["Yoruba people", "Ida (sword)"], gold) == "nothing"


def test_oracle_matches_capitalization_and_whitespace_variants():
    from pipeline import judge

    gold = ["Yoruba people"]
    assert judge.oracle_stop(["  yoruba   People "], gold) is True
    assert judge.oracle_steer(["YORUBA PEOPLE"], gold) == "nothing"


def test_oracle_handles_duplicate_evidence_titles():
    from pipeline import judge

    gold = ["Ida (sword)", "Yoruba people"]
    evidence = ["Ida (sword)", "Ida (sword)", "yoruba people"]
    assert judge.oracle_stop(evidence, gold) is True
    assert judge.oracle_steer(evidence, gold) == "nothing"


# -------------------------------------------------- T08 loop and the settings


class ScriptedLoop:
    """A fake LLM backend + fake search that drive the loop deterministically.

    `judge_verdicts`: enough values per Judge call ("no", "no", "yes", ...).
    `hits_by_round`: the paragraphs each Round's search returns.
    `answer`: the Answerer's output (all Rounds).
    """

    def __init__(self, judge_verdicts, hits_by_round, answer="answer", missing="the date"):
        self.judge_verdicts = list(judge_verdicts)
        self.hits_by_round = list(hits_by_round)
        self.answer = answer
        self.missing = missing
        self.calls = []          # every LLM call: (model, prompt)
        self.queries = []        # every search query
        self.excludes = []       # exclude_titles per search

    def backend(self, model, prompt, max_tokens, temperature):
        self.calls.append((model, prompt))
        if model == "judge-model":
            verdict = self.judge_verdicts.pop(0)
            return {"text": f"REASONING: because.\nENOUGH: {verdict}\nMISSING: {self.missing}",
                    "prompt_tokens": 100, "completion_tokens": 20, "logprobs": []}
        if model == "rewriter-model":
            return {"text": f"rewritten query {len(self.queries)}",
                    "prompt_tokens": 50, "completion_tokens": 5, "logprobs": []}
        return {"text": self.answer,
                "prompt_tokens": 80, "completion_tokens": 4, "logprobs": []}

    def search(self, query, k, exclude_titles=(), retriever=None):
        self.queries.append(query)
        self.excludes.append(set(exclude_titles))
        hits = self.hits_by_round[len(self.queries) - 1]
        # honor exclusion, topping up with nothing (tests use small hit lists)
        return [hit for hit in hits if hit["title"] not in exclude_titles]

    def install(self, monkeypatch):
        monkeypatch.setitem(llm._BACKENDS, "fake", self.backend)
        from pipeline import loop as loop_module
        monkeypatch.setattr(loop_module.retriever_mod, "search", self.search)
        return make_llm_config()  # backend fake, k comes from this config below

    def question(self, gold_titles=("Gold A", "Gold B"), qid="q1"):
        return {"id": qid, "question": "the question?", "answer": "gold answer",
                "gold_titles": list(gold_titles)}


def loop_cfg(script, monkeypatch, rounds=3, k=2):
    cfg = script.install(monkeypatch)
    cfg["retrieval"] = {"k": k}
    cfg["loop"] = {"rounds": rounds, "token_budget": None}
    return cfg


HIT = lambda title, text="text": {"title": title, "text": text, "score": 1.0}


def test_loop_condition_a_llm_stop_and_steer(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["no", "no", "yes"], [
        [HIT("T1"), HIT("T2")], [HIT("T3")], [HIT("T4")]])
    cfg = loop_cfg(script, monkeypatch)
    record = loop.run_question(script.question(), "A", None, config=cfg)

    assert record["rounds_used"] == 3
    assert [r["stop"] for r in record["rounds"]] == ["no", "no", "yes"]
    # steer is the LLM judge's MISSING every Round
    assert all(r["steer"] == "the date" for r in record["rounds"])
    assert [r["retrieved_titles"] for r in record["rounds"]] == \
        [["T1", "T2"], ["T3"], ["T4"]]
    # judge, shadow answer every Round; rewriter only while continuing
    assert script.queries[0] == "the question?"           # raw question opens
    assert script.queries[1:] == ["rewritten query 1", "rewritten query 2"]
    assert record["llm_calls"] == {"main": 6, "shadow": 2}
    assert record["total_prompt_tokens"] == 3 * 100 + 2 * 50 + 80
    assert record["total_completion_tokens"] == 3 * 20 + 2 * 5 + 4
    # evidence accumulates: Round 2's search must exclude Round 1's titles
    assert script.excludes[1] == {"T1", "T2"}


def test_loop_stops_when_llm_judge_says_yes(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["yes"], [[HIT("T1")]])
    record = loop.run_question(script.question(), "A", None, config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 1
    assert record["llm_calls"] == {"main": 2, "shadow": 0}  # judge + final answerer
    assert record["final_answer"] == "answer"


def test_loop_condition_b_oracle_stops_when_all_gold_titles_are_in(monkeypatch):
    from pipeline import loop

    # the LLM judge says no, but the oracle sees both Gold titles -> stop
    script = ScriptedLoop(["no"], [[HIT("Gold A"), HIT("Gold B")]])
    record = loop.run_question(script.question(), "B", None, config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 1
    assert record["rounds"][0]["stop"] == "yes"
    # steer still comes from the LLM judge
    assert record["rounds"][0]["steer"] == "the date"


def test_loop_condition_c_oracle_steers_to_the_missing_title(monkeypatch):
    from pipeline import loop

    # the LLM judge says no twice (so the loop continues), oracle steers
    script = ScriptedLoop(["no", "no", "no"], [
        [HIT("Gold A")], [HIT("T2")], [HIT("T3")]])
    record = loop.run_question(script.question(), "C", None, config=loop_cfg(script, monkeypatch))
    assert record["rounds"][0]["steer"] == "Missing information about: Gold B"
    assert record["rounds"][1]["steer"] == "Missing information about: Gold B"
    assert record["rounds_used"] == 3  # LLM judge never said yes; Round 3 forces


def test_loop_condition_d_oracle_stop_and_steer(monkeypatch):
    from pipeline import loop

    # oracle finds Gold B at Round 2 -> stop at 2 even though the judge says no
    script = ScriptedLoop(["no", "no"], [[HIT("Gold A")], [HIT("Gold B")]])
    record = loop.run_question(script.question(), "D", None, config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 2
    assert [r["steer"] for r in record["rounds"]] == \
        ["Missing information about: Gold B", "nothing"]


def test_loop_single_turn_stops_after_round_1(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["no"], [[HIT("T1")]])
    record = loop.run_question(script.question(), "single-turn", None,
                               config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 1
    assert record["rounds"][0]["stop"] == "yes"    # the judge's no is overridden
    assert record["rounds"][0]["steer"] is None
    assert record["llm_calls"] == {"main": 2, "shadow": 0}


def test_loop_always_loop_runs_all_three_rounds(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["yes", "yes", "yes"], [[HIT("T1")], [HIT("T2")], [HIT("T3")]])
    record = loop.run_question(script.question(), "always-loop", None,
                               config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 3
    assert [r["stop"] for r in record["rounds"]] == ["no", "no", "yes"]
    assert record["llm_calls"] == {"main": 6, "shadow": 2}
    # the judge was called every Round even though it never got to stop
    assert sum(1 for model, _ in script.calls if model == "judge-model") == 3


def test_loop_closed_book_answers_without_retrieval(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop([], [], answer="gold answer")
    record = loop.run_question(script.question(), "closed-book", None,
                               config=loop_cfg(script, monkeypatch))
    assert record["rounds_used"] == 0 and record["rounds"] == []
    assert record["final_answer"] == "gold answer" and record["em"] == 1
    assert record["llm_calls"] == {"main": 1, "shadow": 0}
    assert script.queries == [] and script.calls[0][0] == "answerer-model"


def test_loop_em_and_shadow_em_follow_the_answerer(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["yes"], [[HIT("T1")]], answer="gold answer")
    record = loop.run_question(script.question(), "A", None, config=loop_cfg(script, monkeypatch))
    assert record["em"] == 1
    assert record["rounds"][0]["shadow_em"] == 1


def test_loop_unknown_setting_raises(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop([], [])
    with pytest.raises(ValueError):
        loop.run_question(script.question(), "Z", None, config=loop_cfg(script, monkeypatch))


def test_loop_requires_k(monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["yes"], [[HIT("T1")]])
    cfg = loop_cfg(script, monkeypatch)
    cfg["retrieval"]["k"] = None
    with pytest.raises(ValueError):
        loop.run_question(script.question(), "A", None, config=cfg)


def test_loop_record_round_trips_through_load_run(tmp_path, monkeypatch):
    from pipeline import loop

    script = ScriptedLoop(["no", "yes"] * 2, [[HIT("T1")], [HIT("T2")]] * 2)
    cfg = loop_cfg(script, monkeypatch)
    questions = [script.question(qid=f"q{i}") for i in range(2)]

    def step(question):
        return loop.run_question(question, "A", None, config=cfg)

    summary = run_questions(questions, step, condition="A", dataset="pilot",
                            run_dir=tmp_path / "runs", config=cfg)
    df = load_run(summary["run_dir"])
    assert len(df) == 2
    assert df.attrs["meta"]["condition"] == "A"
    assert all(len(rounds) == 2 for rounds in df["rounds"])


def test_gold_coverage_measures_evidence_completeness():
    from evaluate import gold_coverage

    df = pd.DataFrame({
        "gold_titles": [["Gold A", "Gold B"], ["Gold C", "Gold D"]],
        "rounds": [
            [{"retrieved_titles": ["Gold A"]}, {"retrieved_titles": ["Gold B"]}],  # complete
            [{"retrieved_titles": ["Gold C"]}],                                     # incomplete
        ],
    })
    assert gold_coverage(df) == 0.5
    assert gold_coverage(pd.DataFrame({"gold_titles": [], "rounds": []})) == 0.0
