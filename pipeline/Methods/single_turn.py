"""Single-turn: search once with the question, then the Answerer answers from those paragraphs. No Judge, no loop.

    Set the two settings below, then press Run.

It is the floor every looping method must beat: one search, no second chance. The number of paragraphs per search
(k) comes from config.yaml's retrieval.k, so every method uses the same k. Each run writes
runs/single_turn/<date>/<HHMMSS>.txt: a header with the run's settings, one block per question (written as soon as
it is answered, so a stopped run keeps what it finished), and a summary at the end.
"""

from datetime import datetime
from pathlib import Path

from pipeline import REPO_ROOT, llm2, load_config
from pipeline.answerer import ANSWERER_SYSTEM_PROMPT, PROMPT_VERSION, answer_question
from pipeline.dataset import load_questions
from pipeline.retriever import Retriever
from pipeline.scoring import average_score, score_answer

# --- Settings: edit them, then press Run ---
QUESTION_SET = "pilot"       # "pilot" (100 questions) or "test" (1,000)
HOW_MANY_QUESTIONS = 5    # how many from the start of the set; None = all of them


def new_log_file(start_time: datetime) -> Path:
    """Create this run's log file, runs/single_turn/<date>/<HHMMSS>.txt: one file per run, one folder per day."""
    date_folder = REPO_ROOT / load_config()["paths"]["runs_dir"] / "single_turn" / start_time.strftime("%Y-%m-%d")
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
        "Method:         Single-turn (one search with the question, then the Answerer answers from it)",
        f"Started:        {start_time:%Y-%m-%d %H:%M:%S}",
        f"Questions:      the first {question_count} of the {QUESTION_SET} set",
        f"Retriever:      BM25, k = {config['retrieval']['k']} paragraphs, query = the question as written",
        f"Model:          {llm2.model_for_role('answerer')}",
        f"Prompt version: {PROMPT_VERSION}",
        f"Temperature:    {config['llm']['temperature']}   Seed: {config['seeds'][0]}",
        f"Cosine scoring: {config['scoring']['embedder']}, threshold {config['scoring']['cosine_threshold']}",
        "",
    ]))


def answer_one_question(question: dict, retriever: Retriever) -> dict:
    """Search once with the question, answer from the paragraphs found, and check the answer against the gold answer."""
    # Here the question itself is the search query, and k comes from config.yaml.
    evidence = retriever.search(question["question"], k=load_config()["retrieval"]["k"])
    result = answer_question(question["question"], evidence)
    result["evidence"] = evidence
    # Here we add the four scores to the result: correct (exact match), precision, recall and f1.
    result.update(score_answer(result["answer"], question["answer"]))
    return result


def log_single_turn(log_file: Path, question_number: int, question_count: int, question: dict, result: dict) -> None:
    """Write one question's block: the question, what the retriever found, the full prompt, the full reply,
    the gold answer and the score."""
    lines = [
        f"===== Question {question_number} of {question_count} (id {question['id']}) =====",
        f"Question: {question['question']}",
        "",
        "Retriever (query = the question):",
    ]
    # Here we list each paragraph's rank, score and title; their full text is in the prompt below.
    for rank, paragraph in enumerate(result["evidence"], start=1):
        lines.append(f"  {rank}. [{paragraph['score']:.1f}] {paragraph['title']}")
    lines += [
        "",
        "Full prompt sent to the model:",
        f"[system] {ANSWERER_SYSTEM_PROMPT}",
        f"[user]   {result['prompt']}",
        "",
        f"Full reply:  {result['reply']}",
        f"Answer:      {result['answer']}",
        f"Gold answer: {question['answer']}",
        f"Correct:     {'yes' if result['correct'] else 'no'}",
        f"Precision:   {result['precision']:.2f}",
        f"Recall:      {result['recall']:.2f}",
        f"F1:          {result['f1']:.2f}",
        f"Cosine similarity: {result['cosine']:.3f}",
        f"Attempts: {result['attempts']}   Tokens read: {result['tokens_read']}, "
        f"written: {result['tokens_written']}, in {result['seconds']} s",
    ]
    # Here we warn loudly when something may have gone wrong silently.
    if not result["evidence"]:
        lines.append("WARNING  the retriever found nothing, so the Answerer had no Evidence")
    if result["answer_problems"]:
        lines.append(f"WARNING  the reply never fitted its JSON shape: {'; '.join(result['answer_problems'])}")
    if result["prompt_may_be_cut"]:
        lines.append("WARNING  the prompt may have been cut to fit the context window")
    if result["reply_was_cut"]:
        lines.append("WARNING  the reply was cut at the reply limit")
    write_to_log(log_file, "\n".join(lines) + "\n")


def log_summary(log_file: Path, results: list[dict], question_count: int, start_time: datetime) -> None:
    """Write the run's totals at the end of the log: how many were answered and correct, tokens, and time."""
    correct_count = sum(result["correct"] for result in results)
    answered_count = len(results)
    share_correct = correct_count / answered_count if answered_count else 0
    # Here we count the answers correct by meaning (exact match, or cosine at or above the threshold).
    cosine_correct_count = sum(result["correct_cosine"] for result in results)
    share_cosine_correct = cosine_correct_count / answered_count if answered_count else 0
    cosine_threshold = load_config()["scoring"]["cosine_threshold"]
    minutes_taken = (datetime.now() - start_time).total_seconds() / 60
    write_to_log(log_file, "\n".join([
        "===== Summary =====",
        f"Answered: {answered_count} of {question_count}",
        f"Correct:  {correct_count} of {answered_count} ({share_correct:.0%})",
        f"Correct (cosine ≥ {cosine_threshold}): {cosine_correct_count} of {answered_count} ({share_cosine_correct:.0%})",
        f"Precision: {average_score(results, 'precision'):.2f} (average)",
        f"Recall:    {average_score(results, 'recall'):.2f} (average)",
        f"F1:        {average_score(results, 'f1'):.2f} (average)",
        f"Tokens read: {sum(result['tokens_read'] for result in results)}, "
        f"written: {sum(result['tokens_written'] for result in results)}",
        f"Time: {minutes_taken:.1f} minutes",
    ]))


def run_single_turn() -> Path:
    """Search and answer the chosen questions one after another, logging each as soon as it is answered.
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
            log_single_turn(log_file, question_number, len(questions), question, result)
            print(f"{question_number:>4}/{len(questions)}  {'✓' if result['correct'] else '✗'}  {result['answer']}")
    finally:
        # Here the summary is written even if the run is stopped early, and only our own server is stopped.
        log_summary(log_file, results, len(questions), start_time)
        if server_started_here:
            llm2.stop_ollama_server()
    return log_file


if __name__ == "__main__":
    finished_log_file = run_single_turn()
    print(f"\nLog: {finished_log_file}")
