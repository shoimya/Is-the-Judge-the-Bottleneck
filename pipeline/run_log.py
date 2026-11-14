"""The pieces of run logging every method shares: where the log file goes, how a line is written, and the log lines
that look the same in every method (the pipeline header, the Rewriter's step, a call's warnings).

    Press Run to see where a Condition A run started now would write its log, and its pipeline header lines.

Each run writes one file, runs/<method>/<settings>/<date>/<HHMMSS>.txt. The <settings> folder names the pipeline and
k the run used (for example P2-k3: Pipeline 2, 3 new paragraphs per search), so runs of the k study and the pipeline
study sit apart from one another, and the header repeats both. Methods that don't loop have a shorter folder:
Single-turn names only k (k3), and Closed-book, which never searches, has no settings folder.
"""

from datetime import datetime
from pathlib import Path

from pipeline import REPO_ROOT, llm2, load_config, pipeline_number, rewriter


def pipeline_and_k_folder() -> str:
    """The settings folder for a looping method, such as "P2-k3": the pipeline and the paragraphs per search."""
    return f"P{pipeline_number()}-{k_folder()}"


def k_folder() -> str:
    """The settings folder for a method that searches but doesn't loop, such as "k3"."""
    return f"k{load_config()['retrieval']['k']}"


def log_file_path(method_name: str, start_time: datetime, settings_folder: str | None = None) -> Path:
    """Where a run's log file goes, runs/<method>/<settings>/<date>/<HHMMSS>.txt (no <settings> part when the method
    has none), without creating anything."""
    method_folder = REPO_ROOT / load_config()["paths"]["runs_dir"] / method_name
    if settings_folder:
        method_folder = method_folder / settings_folder
    date_folder = method_folder / start_time.strftime("%Y-%m-%d")
    return date_folder / f"{start_time:%H%M%S}.txt"


def new_log_file(method_name: str, start_time: datetime, settings_folder: str | None = None) -> Path:
    """Create this run's log file at log_file_path(...): one file per run, one folder per day."""
    log_file = log_file_path(method_name, start_time, settings_folder)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Here exist_ok=False stops with an error rather than mixing two runs into one file.
    log_file.touch(exist_ok=False)
    return log_file


def write_to_log(log_file: Path, text: str) -> None:
    """Add text to the end of the log file straight away, so it is kept even if the run stops later."""
    with open(log_file, "a", encoding="utf-8") as opened_log:
        opened_log.write(text + "\n")


def pipeline_header_lines() -> list[str]:
    """The header lines that depend on the pipeline: which one runs, and the Rewriter's model and prompt version
    when it is on. Every looping method puts these under its Retriever line."""
    if pipeline_number() == 1:
        return [
            "Pipeline:        1 (Rewriter on: later Rounds search with the Rewriter's query, written from the steer)",
            f"Rewriter:        {llm2.model_for_role('rewriter')}, prompt {rewriter.PROMPT_VERSION}",
        ]
    return ["Pipeline:        2 (Rewriter off: later Rounds search with the Steer signal as written)"]


def rewriter_log_lines(round_record: dict, log_full_prompts: bool) -> list[str]:
    """The Rewriter's step in one Round's block: the steer it read and the query it wrote, then its full prompt (if
    asked for) and its warnings. Empty when the Rewriter did not run in that Round."""
    rewrite_result = round_record["rewrite"]
    if not rewrite_result:
        return []
    lines = [f'Rewriter: "{round_record["steer"]}"  ->  "{rewrite_result["query"]}"']
    if log_full_prompts:
        lines += full_prompt_lines(rewriter.REWRITER_SYSTEM_PROMPT, rewrite_result)
    return lines + call_warnings("Rewriter", rewrite_result)


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


def share_text(count: int, total: int) -> str:
    """A count as "count of total (percent)", for the summary: 3 of 10 (30%)."""
    return f"{count} of {total} ({count / total if total else 0:.0%})"


if __name__ == "__main__":
    # Here we only show the path a Condition A log started now would get, without creating it.
    demo_path = log_file_path("condition_a", datetime.now(), pipeline_and_k_folder())
    print(f"A Condition A run started now would log to:\n  {demo_path.relative_to(REPO_ROOT)}\n")
    print("Its pipeline header lines:")
    for line in pipeline_header_lines():
        print(f"  {line}")
