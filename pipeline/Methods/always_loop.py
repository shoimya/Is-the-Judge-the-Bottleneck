"""Always-loop: three Rounds of search for every question, steered by the Judge, then the Answerer answers once.

    Set the settings below, then press Run.

Round 1 searches with the question. In Rounds 2 and 3 the Judge's Steer signal ("missing") from the Round before leads
to the query. How the steer becomes a query is config.yaml's loop.pipeline: under Pipeline 1 the Rewriter writes the
query from it, under Pipeline 2 the steer is the query as written (choose_query_for_round in pipeline/retriever.py).
The new paragraphs are added to the Evidence. The Judge is asked in every Round: its Steer signal guides the next
search, and its yes/no is logged but never stops the loop. After Round 3 the Answerer answers from all the Evidence.

Per question: 3 Judge calls, 1 Answerer call, and under Pipeline 1 one Rewriter call per Round after the first. The
number of Rounds comes from config.yaml's loop.rounds and the paragraphs per search from retrieval.k. Each run writes
runs/always_loop/P<pipeline>-k<k>/<date>/<HHMMSS>.txt: a header with the run's settings, one block per question
(written as soon as it is answered, so a stopped run keeps what it finished), and a summary at the end.
"""

from datetime import datetime
from pathlib import Path

from pipeline import answerer, judge, llm2, load_config, run_log
from pipeline.dataset import load_questions
from pipeline.retriever import Retriever, choose_query_for_round
from pipeline.scoring import average_score, score_answer

# --- Settings: edit them, then press Run ---
QUESTION_SET = "pilot"       # "pilot" (100 questions) or "test" (1,000)
HOW_MANY_QUESTIONS = 5        # how many from the start of the set; None = all of them
LOG_FULL_PROMPTS = False     # True = also log every call's full prompt and reply (use it for small test runs)


def log_header(log_file: Path, start_time: datetime, question_count: int) -> None:
    """Write the run's settings at the top of the log, so the log always says what produced it."""
    config = load_config()
    run_log.write_to_log(log_file, "\n".join([
        "Method:          Always-loop (every Round runs; the Judge steers each new search, but never stops the loop)",
        f"Started:         {start_time:%Y-%m-%d %H:%M:%S}",
        f"Questions:       the first {question_count} of the {QUESTION_SET} set",
        f"Rounds:          {config['loop']['rounds']}",
        f"Retriever:       BM25, k = {config['retrieval']['k']} new paragraphs per Round",
        *run_log.pipeline_header_lines(),
        f"Models:          judge {llm2.model_for_role('judge')}, answerer {llm2.model_for_role('answerer')}",
        f"Prompt versions: judge {judge.PROMPT_VERSION}, answerer {answerer.PROMPT_VERSION}",
        f"Temperature:     {config['llm']['temperature']}   Seed: {config['seeds'][0]}",
        f"Cosine scoring:  {config['scoring']['embedder']}, threshold {config['scoring']['cosine_threshold']}",
        f"Full prompts:    {'logged' if LOG_FULL_PROMPTS else 'not logged'}",
        "",
    ]))


def answer_one_question(question: dict, retriever: Retriever) -> dict:
    """Run every Round for one question (search, then Judge), then answer once from all the Evidence, and check the
    answer against the gold answer."""
    config = load_config()
    evidence = []
    past_queries = []
    rounds = []
    for round_number in range(1, config["loop"]["rounds"] + 1):
        # Here Round 1 searches with the question; later Rounds start from the Judge's steer of the Round before,
        # which the pipeline setting turns into the query (Rewriter on or off).
        previous_steer = rounds[-1]["judgment"]["missing"] if rounds else None
        query_choice = choose_query_for_round(question["question"], previous_steer, past_queries)
        query = query_choice["query"]
        past_queries.append(query)

        # Here the search skips paragraphs already in the Evidence, so each Round brings new ones.
        titles_held = {paragraph["title"] for paragraph in evidence}
        found = retriever.search(query, k=config["retrieval"]["k"], exclude_titles=titles_held)
        evidence += found

        judgment = judge.judge_evidence(question["question"], evidence)
        # Here we keep everything this Round did, one item per line, for the log and the cost count.
        rounds.append({
            "round_number": round_number,
            "steer": previous_steer,
            "rewrite": query_choice["rewrite"],
            "query": query,
            "found": found,
            "judgment": judgment,
        })

    answer_result = answerer.answer_question(question["question"], evidence)
    # Here we gather every model call this question made, to add up its costs. The Rewriter's calls exist only under
    # Pipeline 1.
    calls = [round_record["judgment"] for round_record in rounds]
    calls += [round_record["rewrite"] for round_record in rounds if round_record["rewrite"]]
    calls.append(answer_result)
    return {
        "rounds": rounds,
        "answer_result": answer_result,
        "answer": answer_result["answer"],
        # Here we add the four scores: correct (exact match), precision, recall and f1.
        **score_answer(answer_result["answer"], question["answer"]),
        "calls": len(calls),
        "attempts": sum(call["attempts"] for call in calls),
        "tokens_read": sum(call["tokens_read"] for call in calls),
        "tokens_written": sum(call["tokens_written"] for call in calls),
        "seconds": round(sum(call["seconds"] for call in calls), 1),
    }


def log_always_loop(log_file: Path, question_number: int, question_count: int, question: dict, result: dict) -> None:
    """Write one question's block, in the order things happened: each Round's Rewriter step (Pipeline 1 only), search
    and judgment, then the final answer and the totals."""
    lines = [
        f"===== Question {question_number} of {question_count} (id {question['id']}) =====",
        f"Question:    {question['question']}",
        f"Gold answer: {question['answer']}",
    ]
    last_round_number = len(result["rounds"])
    for round_record in result["rounds"]:
        lines += ["", f"--- Round {round_record['round_number']} ---"]
        # Here a later Round under Pipeline 1 opens with the steer the Rewriter read and the query it wrote from it.
        lines += run_log.rewriter_log_lines(round_record, LOG_FULL_PROMPTS)
        lines.append(f"Retriever query: {round_record['query']}")
        scores_and_titles = [f"[{paragraph['score']:.1f}] {paragraph['title']}" for paragraph in round_record["found"]]
        lines.append(f"Retriever found: {'   '.join(scores_and_titles) if scores_and_titles else 'nothing'}")
        if not scores_and_titles:
            lines.append("WARNING  the retriever found nothing new this Round")

        judgment = round_record["judgment"]
        steer_note = "   (last Round: steer not used)" if round_record["round_number"] == last_round_number else ""
        lines += [
            f"Judge:     enough = {judgment['enough']} (p_yes {run_log.p_yes_text(judgment['p_yes'])})",
            f"           reasoning: {judgment['reasoning']}",
            f"           steer: {judgment['missing']}{steer_note}",
        ]
        if LOG_FULL_PROMPTS:
            lines += run_log.full_prompt_lines(judge.JUDGE_SYSTEM_PROMPT, judgment)
        lines += run_log.call_warnings("Judge", judgment)

    answer_result = result["answer_result"]
    paragraph_count = sum(len(round_record["found"]) for round_record in result["rounds"])
    lines += ["", f"--- Answer (from all {paragraph_count} paragraphs) ---"]
    if LOG_FULL_PROMPTS:
        lines += run_log.full_prompt_lines(answerer.ANSWERER_SYSTEM_PROMPT, answer_result)
    lines += [
        f"Answer:      {result['answer']}",
        f"Gold answer: {question['answer']}",
        f"Correct:     {'yes' if result['correct'] else 'no'}",
        f"Precision:   {result['precision']:.2f}",
        f"Recall:      {result['recall']:.2f}",
        f"F1:          {result['f1']:.2f}",
        f"Cosine similarity: {result['cosine']:.3f}",
        f"Calls: {result['calls']}   Attempts: {result['attempts']}   Tokens read: {result['tokens_read']}, "
        f"written: {result['tokens_written']}, in {result['seconds']} s",
    ]
    lines += run_log.call_warnings("Answerer", answer_result)
    run_log.write_to_log(log_file, "\n".join(lines) + "\n")


def log_summary(log_file: Path, results: list[dict], question_count: int, start_time: datetime) -> None:
    """Write the run's totals at the end of the log: how many were answered and correct, calls, tokens, and time."""
    answered_count = len(results)
    # Here we count the answers correct by meaning (exact match, or cosine at or above the threshold).
    cosine_correct_count = sum(result["correct_cosine"] for result in results)
    cosine_threshold = load_config()["scoring"]["cosine_threshold"]
    minutes_taken = (datetime.now() - start_time).total_seconds() / 60
    run_log.write_to_log(log_file, "\n".join([
        "===== Summary =====",
        f"Answered: {answered_count} of {question_count}",
        f"Correct:  {run_log.share_text(sum(result['correct'] for result in results), answered_count)}",
        f"Correct (cosine ≥ {cosine_threshold}): {run_log.share_text(cosine_correct_count, answered_count)}",
        f"Precision: {average_score(results, 'precision'):.2f} (average)",
        f"Recall:    {average_score(results, 'recall'):.2f} (average)",
        f"F1:        {average_score(results, 'f1'):.2f} (average)",
        f"Model calls: {sum(result['calls'] for result in results)}",
        f"Tokens read: {sum(result['tokens_read'] for result in results)}, "
        f"written: {sum(result['tokens_written'] for result in results)}",
        f"Time: {minutes_taken:.1f} minutes",
    ]))


def run_always_loop() -> Path:
    """Run every Round for the chosen questions one after another, logging each as soon as it is answered.
    Returns the log file."""
    # Here [:None] keeps every question, so HOW_MANY_QUESTIONS = None means the whole set.
    questions = load_questions(QUESTION_SET)[:HOW_MANY_QUESTIONS]
    start_time = datetime.now()
    # Here the log goes under a folder named after the pipeline and k, so runs with different settings never mix.
    log_file = run_log.new_log_file("always_loop", start_time, run_log.pipeline_and_k_folder())
    log_header(log_file, start_time, len(questions))

    # Here the search index is opened once, and the server starts once, so every question reuses both.
    retriever = Retriever.load()
    server_started_here = llm2.start_ollama_server()
    results = []
    try:
        for question_number, question in enumerate(questions, start=1):
            result = answer_one_question(question, retriever)
            results.append(result)
            log_always_loop(log_file, question_number, len(questions), question, result)
            print(f"{question_number:>4}/{len(questions)}  {'✓' if result['correct'] else '✗'}  {result['answer']}")
    finally:
        # Here the summary is written even if the run is stopped early, and only our own server is stopped.
        log_summary(log_file, results, len(questions), start_time)
        if server_started_here:
            llm2.stop_ollama_server()
    return log_file


if __name__ == "__main__":
    finished_log_file = run_always_loop()
    print(f"\nLog: {finished_log_file}")
