"""Condition B: the LLM judge decides when to stop; the Oracle judge says what is missing. B - A = the cost of steering.

    Set the settings below, then press Run.

Round 1 searches with the question. After each Round's search, the LLM judge reads the Evidence and stops the loop as
soon as it says the Evidence is enough. If it says not enough, the Oracle judge's Steer signal ("Missing information
about: <first missing Gold title>") goes to the Rewriter, which writes the next query. The Oracle names only the
title, never the text, so the Rewriter and BM25 still have to find the paragraph. When every Gold paragraph is already
in the Evidence but the LLM judge still says not enough, the Oracle's steer is "nothing". At most config.yaml's
loop.rounds Rounds, k new paragraphs per search (retrieval.k). Then the Answerer answers once from all the Evidence.

Only the steering is made perfect here, so comparing with Condition A (same stopping, LLM steering) measures what the
LLM judge's steering costs (CONTEXT.md, "The 2x2"). The LLM judge's own steer is still logged, never used.

Per question: one LLM judge call per Round used, one Rewriter call per Round after the first, one Answerer call.
Each run writes runs/condition_b/<date>/<HHMMSS>.txt: a header with the run's settings, one block per question
(written as soon as it is answered, so a stopped run keeps what it finished), and a summary at the end.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

from pipeline import REPO_ROOT, answerer, judge, llm2, load_config, rewriter
from pipeline.dataset import load_questions
from pipeline.retriever import Retriever
from pipeline.scoring import exact_match

# --- Settings: edit them, then press Run ---
QUESTION_SET = "pilot"       # "pilot" (100 questions) or "test" (1,000)
HOW_MANY_QUESTIONS = 10    # how many from the start of the set; None = all of them
LOG_FULL_PROMPTS = False     # True = also log every call's full prompt and reply (use it for small test runs)


def new_log_file(start_time: datetime) -> Path:
    """Create this run's log file, runs/condition_b/<date>/<HHMMSS>.txt: one file per run, one folder per day."""
    date_folder = REPO_ROOT / load_config()["paths"]["runs_dir"] / "condition_b" / start_time.strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)
    log_file = date_folder / f"{start_time:%H%M%S}.txt"
    # Here exist_ok=False stops with an error rather than mixing two runs into one file.
    log_file.touch(exist_ok=False)
    return log_file


def write_to_log(log_file: Path, text: str) -> None:
    """Add text to the end of the log file straight away, so it is kept even if the run stops later."""
    with open(log_file, "a", encoding="utf-8") as opened_log:
        opened_log.write(text + "\n")


def log_header(log_file: Path, start_time: datetime, question_count: int) -> None:
    """Write the run's settings at the top of the log, so the log always says what produced it."""
    config = load_config()
    write_to_log(log_file, "\n".join([
        "Method:          Condition B (the LLM judge stops; the Oracle judge steers)",
        f"Started:         {start_time:%Y-%m-%d %H:%M:%S}",
        f"Questions:       the first {question_count} of the {QUESTION_SET} set",
        f"Rounds:          at most {config['loop']['rounds']}",
        f"Retriever:       BM25, k = {config['retrieval']['k']} new paragraphs per Round",
        f"Models:          judge {llm2.model_for_role('judge')}, rewriter {llm2.model_for_role('rewriter')}, "
        f"answerer {llm2.model_for_role('answerer')}",
        f"Prompt versions: judge {judge.PROMPT_VERSION}, rewriter {rewriter.PROMPT_VERSION}, "
        f"answerer {answerer.PROMPT_VERSION}",
        f"Temperature:     {config['llm']['temperature']}   Seed: {config['seeds'][0]}",
        f"Full prompts:    {'logged' if LOG_FULL_PROMPTS else 'not logged'}",
        "",
    ]))


def answer_one_question(question: dict, retriever: Retriever) -> dict:
    """Search Round by Round until the LLM judge says the Evidence is enough (or the Rounds run out), with the Oracle
    judge's Steer signal behind each new query. Then answer once from all the Evidence, and check the answer against
    the gold answer."""
    config = load_config()
    gold_titles = question["gold_titles"]
    evidence = []
    past_queries = []
    rounds = []
    for round_number in range(1, config["loop"]["rounds"] + 1):
        # Here Round 1 searches with the question; later Rounds search with the Rewriter's query,
        # built from the Oracle judge's steer in the Round before.
        rewrite_result = None
        query = question["question"]
        if round_number > 1:
            previous_steer = rounds[-1]["oracle_steer"]
            rewrite_result = rewriter.rewrite_query(question["question"], past_queries, previous_steer)
            query = rewrite_result["query"]
        past_queries.append(query)

        # Here the search skips paragraphs already in the Evidence, so each Round brings new ones.
        titles_held = {paragraph["title"] for paragraph in evidence}
        found = retriever.search(query, k=config["retrieval"]["k"], exclude_titles=titles_held)
        evidence += found

        # Here the LLM judge decides when to stop; its own steer is logged but not used.
        judgment = judge.judge_evidence(question["question"], evidence)

        # Here the Oracle judge names the first missing Gold title, which steers the next query.
        evidence_titles = [paragraph["title"] for paragraph in evidence]
        oracle_enough = judge.oracle_stop(evidence_titles, gold_titles)
        rounds.append({"round_number": round_number, "rewrite": rewrite_result, "query": query, "found": found,
                       "judgment": judgment, "oracle_enough": oracle_enough,
                       "oracle_steer": judge.oracle_steer(evidence_titles, gold_titles)})
        if judgment["enough"] == "yes":
            break

    answer_result = answerer.answer_question(question["question"], evidence)
    # Here we gather every model call this question made, to add up its costs. The Oracle judge makes none.
    calls = [round_record["judgment"] for round_record in rounds]
    calls += [round_record["rewrite"] for round_record in rounds if round_record["rewrite"]]
    calls.append(answer_result)
    return {
        "rounds": rounds,
        "rounds_used": len(rounds),
        "all_gold_found": rounds[-1]["oracle_enough"],
        "answer_result": answer_result,
        "answer": answer_result["answer"],
        "correct": exact_match(answer_result["answer"], question["answer"]),
        "calls": len(calls),
        "attempts": sum(call["attempts"] for call in calls),
        "tokens_read": sum(call["tokens_read"] for call in calls),
        "tokens_written": sum(call["tokens_written"] for call in calls),
        "seconds": round(sum(call["seconds"] for call in calls), 1),
    }


def call_warnings(role_name: str, call_result: dict) -> list[str]:
    """WARNING lines for one model call: a reply that never fitted its shape, a possibly cut prompt, a cut reply."""
    warnings = []
    if call_result["answer_problems"]:
        warnings.append(f"WARNING  {role_name}: the reply never fitted its JSON shape: "
                        f"{'; '.join(call_result['answer_problems'])}")
    if call_result["prompt_may_be_cut"]:
        warnings.append(f"WARNING  {role_name}: the prompt may have been cut to fit the context window")
    if call_result["reply_was_cut"]:
        warnings.append(f"WARNING  {role_name}: the reply was cut at the reply limit")
    return warnings


def p_yes_text(p_yes: float | None) -> str:
    """The Judge's confidence for the log, to two decimals ("n/a" when the odds weren't there)."""
    return "n/a" if p_yes is None else f"{p_yes:.2f}"


def full_prompt_lines(system_prompt: str, call_result: dict) -> list[str]:
    """One call's full prompt and full reply, indented, for when LOG_FULL_PROMPTS is True."""
    return [
        f"          [system] {system_prompt}",
        f"          [user]   {call_result['prompt']}",
        f"          Full reply: {call_result['reply']}",
    ]


def log_condition_b(log_file: Path, question_number: int, question_count: int, question: dict, result: dict) -> None:
    """Write one question's block, in the order things happened: each Round's rewrite, search, the LLM judge's stop
    decision and the Oracle judge's steer, then the final answer and the totals."""
    lines = [
        f"===== Question {question_number} of {question_count} (id {question['id']}) =====",
        f"Question:    {question['question']}",
        f"Gold answer: {question['answer']}",
        f"Gold titles: {'   '.join(question['gold_titles'])}",
    ]
    last_round_allowed = load_config()["loop"]["rounds"]
    for round_record in result["rounds"]:
        lines += ["", f"--- Round {round_record['round_number']} ---"]

        # Here a later Round opens with the Oracle steer the Rewriter read and the query it wrote from it.
        rewrite_result = round_record["rewrite"]
        if rewrite_result:
            previous_steer = result["rounds"][round_record["round_number"] - 2]["oracle_steer"]
            lines.append(f'Rewriter: "{previous_steer}"  ->  "{rewrite_result["query"]}"')
            if LOG_FULL_PROMPTS:
                lines += full_prompt_lines(rewriter.REWRITER_SYSTEM_PROMPT, rewrite_result)
            lines += call_warnings("Rewriter", rewrite_result)

        lines.append(f"Retriever query: {round_record['query']}")
        scores_and_titles = [f"[{paragraph['score']:.1f}] {paragraph['title']}" for paragraph in round_record["found"]]
        lines.append(f"Retriever found: {'   '.join(scores_and_titles) if scores_and_titles else 'nothing'}")
        if not scores_and_titles:
            lines.append("WARNING  the retriever found nothing new this Round")

        # Here the LLM judge's stop decision, which this method acts on; its own steer is logged only.
        judgment = round_record["judgment"]
        if judgment["enough"] == "yes":
            decision = "stop"
        elif round_record["round_number"] == last_round_allowed:
            decision = "no Rounds left"
        else:
            decision = "go on"
        lines += [
            f"LLM judge: enough = {judgment['enough']} (p_yes {p_yes_text(judgment['p_yes'])})   {decision}",
            f"           reasoning: {judgment['reasoning']}",
            f"           steer: {judgment['missing']}   (logged only, not used)",
        ]
        if LOG_FULL_PROMPTS:
            lines += full_prompt_lines(judge.JUDGE_SYSTEM_PROMPT, judgment)
        lines += call_warnings("LLM judge", judgment)

        # Here the Oracle judge's steer, which this method acts on when the loop goes on.
        steer_used = judgment["enough"] != "yes" and round_record["round_number"] < last_round_allowed
        agrees = "agrees" if (judgment["enough"] == "yes") == round_record["oracle_enough"] else "DISAGREES"
        lines.append(f"Oracle:    enough = {'yes' if round_record['oracle_enough'] else 'no'} ({agrees} with the "
                     f"LLM judge)   steer: {round_record['oracle_steer']}"
                     f"{'' if steer_used else '   (not used: the loop ends here)'}")

    answer_result = result["answer_result"]
    paragraph_count = sum(len(round_record["found"]) for round_record in result["rounds"])
    lines += ["", f"--- Answer (from all {paragraph_count} paragraphs, after {result['rounds_used']} Round(s)) ---"]
    if LOG_FULL_PROMPTS:
        lines += full_prompt_lines(answerer.ANSWERER_SYSTEM_PROMPT, answer_result)
    lines += [
        f"Answer:          {result['answer']}",
        f"Gold answer:     {question['answer']}",
        f"Correct:         {'yes' if result['correct'] else 'no'}",
        f"All Gold found:  {'yes' if result['all_gold_found'] else 'no'}",
        f"Calls: {result['calls']}   Attempts: {result['attempts']}   Tokens read: {result['tokens_read']}, "
        f"written: {result['tokens_written']}, in {result['seconds']} s",
    ]
    lines += call_warnings("Answerer", answer_result)
    write_to_log(log_file, "\n".join(lines) + "\n")


def log_summary(log_file: Path, results: list[dict], question_count: int, start_time: datetime) -> None:
    """Write the run's totals at the end of the log: answered and correct, how often every Gold paragraph was found,
    Rounds used, calls, tokens, and time."""
    answered_count = len(results)

    def share(count: int) -> str:
        return f"{count} of {answered_count} ({count / answered_count if answered_count else 0:.0%})"

    rounds_used = Counter(result["rounds_used"] for result in results)
    rounds_text = "   ".join(f"{rounds} Round(s): {rounds_used[rounds]}" for rounds in sorted(rounds_used))
    minutes_taken = (datetime.now() - start_time).total_seconds() / 60
    write_to_log(log_file, "\n".join([
        "===== Summary =====",
        f"Answered:       {answered_count} of {question_count}",
        f"Correct:        {share(sum(result['correct'] for result in results))}",
        f"All Gold found: {share(sum(result['all_gold_found'] for result in results))}",
        f"Rounds used:    {rounds_text or 'none'}",
        f"Model calls: {sum(result['calls'] for result in results)}",
        f"Tokens read: {sum(result['tokens_read'] for result in results)}, "
        f"written: {sum(result['tokens_written'] for result in results)}",
        f"Time: {minutes_taken:.1f} minutes",
    ]))


def run_condition_b() -> Path:
    """Run Condition B on the chosen questions one after another, logging each as soon as it is answered.
    Returns the log file."""
    # Here [:None] keeps every question, so HOW_MANY_QUESTIONS = None means the whole set.
    questions = load_questions(QUESTION_SET)[:HOW_MANY_QUESTIONS]
    start_time = datetime.now()
    log_file = new_log_file(start_time)
    log_header(log_file, start_time, len(questions))

    # Here the search index is opened once, and the server starts once, so every question reuses both.
    retriever = Retriever.load()
    server_started_here = llm2.start_ollama_server()
    results = []
    try:
        for question_number, question in enumerate(questions, start=1):
            result = answer_one_question(question, retriever)
            results.append(result)
            log_condition_b(log_file, question_number, len(questions), question, result)
            print(f"{question_number:>4}/{len(questions)}  {'✓' if result['correct'] else '✗'}  "
                  f"Rounds {result['rounds_used']}  Gold {'all' if result['all_gold_found'] else 'not all'}  "
                  f"{result['answer']}")
    finally:
        # Here the summary is written even if the run is stopped early, and only our own server is stopped.
        log_summary(log_file, results, len(questions), start_time)
        if server_started_here:
            llm2.stop_ollama_server()
    return log_file


if __name__ == "__main__":
    finished_log_file = run_condition_b()
    print(f"\nLog: {finished_log_file}")
