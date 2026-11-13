"""Every call to a language model goes through this file, on the Mac and on Kaggle alike.

    Press Run to start this project's Ollama server (if it isn't running), ask the Judge one made-up question,
    and print its answer, its confidence (p_yes) and what the call cost.

send_to_ollama() is the only code that talks to a model. It never checks which machine it runs on: it always sends
to settings.OLLAMA_ADDRESS, and settings.py holds every difference between the machines. Each reply comes back in a
JSON shape the caller chooses, so no reply ever needs to be picked apart by hand.
"""

import json
import math
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

from pipeline import load_config, settings

# --- Settings ---
# Here we switch the models' thinking on (True) or off (False) for every call.
THINKING_ON = False

# Here we add room for the model's thinking to every reply limit when thinking is on, because its thinking counts
# towards the reply limit: without it, a long think would be cut off before the JSON answer.
# (If thinking is ever turned on, also raise the Mac's OLLAMA_CONTEXT_WINDOW in settings.py to 32,768.)
ROOM_FOR_THINKING_IN_TOKENS = 12288 if THINKING_ON else 0

# Here we wait longer for a reply when the model thinks, because thinking can write thousands of tokens.
SECONDS_TO_WAIT_FOR_A_REPLY = 7200 if THINKING_ON else 1800

# Here we set how long to wait for an embedding: a short answer embeds in well under a second, but the first call
# also loads the embedding model.
SECONDS_TO_WAIT_FOR_EMBEDDINGS = 300

# Here we round every context window up to whole blocks of 8,192 tokens, so most calls share a few sizes:
# Ollama reloads the model each time the window size changes.
CONTEXT_WINDOW_BLOCK_IN_TOKENS = 8192

# Here we estimate a prompt's length in tokens from its characters (about 4 per token in English), and add 20%
# in case the estimate is low.
CHARACTERS_PER_TOKEN = 4
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.2

# Here we set how many alternative words, with their odds, the model reports at each position of its reply.
# p_yes reads them at the Judge's yes/no word.
TOP_ALTERNATIVES = 10

# Here we set how long to wait for the server to start or stop answering.
SECONDS_TO_WAIT_FOR_THE_SERVER = 30

# Here we set how many times one question may be sent before we give up on a reply in the right format.
MAXIMUM_ATTEMPTS = 3

# Here is the note added to the prompt on a retry. With temperature 0 the model always picks its most likely word,
# so resending the same prompt would bring back the same reply: the note changes the prompt, and so the reply.
RETRY_NOTE = ("\n\nYour last answer was empty or didn't match the required format ({problems}). "
              "Answer again, using exactly the JSON shape you were given.")

# --- Demo question (used only when you press Run on this file) ---
DEMO_SYSTEM_PROMPT = ("You are the Judge in a search loop. Decide whether the evidence is enough to answer the "
                      "question. Reason briefly first, then decide.")
DEMO_PROMPT = ("Question: Which river flows through the capital city of Hungary?\n\n"
               "Evidence:\n[1] Budapest: Budapest is the capital and the most populous city of Hungary.\n\n"
               "Answer as JSON with keys reasoning (a few sentences), enough (yes or no), "
               "and missing (what is still needed, or nothing).")
DEMO_ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "enough": {"type": "string", "enum": ["yes", "no"]},
        "missing": {"type": "string"},
    },
    "required": ["reasoning", "enough", "missing"],
}


# ---------------------------------------------------------------------------
# This project's Ollama server
# ---------------------------------------------------------------------------

def ollama_server_is_running() -> bool:
    """True if this project's Ollama server answers on its port. It never touches any other Ollama on the machine."""
    # Here we ask the server for its version, the smallest question it answers; no answer means it isn't running.
    try:
        with urllib.request.urlopen(settings.OLLAMA_ADDRESS + "/api/version", timeout=2):
            return True
    except OSError:
        return False


_started_server_process = None


def start_ollama_server() -> bool:
    """Start this project's Ollama server in the background, with this machine's models folder and port.

    Returns True if this call started the server, so the same code knows to stop it again when it has finished.
    Returns False if the server was already running: whoever started it stops it.
    """
    global _started_server_process
    if ollama_server_is_running():
        print(f"This project's Ollama server is already running at {settings.OLLAMA_ADDRESS}.")
        return False
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama isn't installed: on Windows install from https://ollama.com/download/windows; "
                           "on the Mac run `brew install ollama`; on Kaggle, run the "
                           "notebook's install cell first.")

    # Here the server's own messages go to a file next to the models, to read if it fails to start.
    # The server keeps writing to it after we close our copy of the file.
    settings.MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(settings.MODELS_FOLDER / "ollama-server.log", "a") as server_messages_file:
        # Here start_new_session lets the server keep running after this file finishes.
        background_options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                              else {"start_new_session": True})
        _started_server_process = subprocess.Popen(
            ["ollama", "serve"], env=settings.ollama_environment(),
            stdout=server_messages_file, stderr=server_messages_file, **background_options,
        )

    # Here we wait until the server answers, so a call can use it straight away.
    for _ in range(SECONDS_TO_WAIT_FOR_THE_SERVER):
        if ollama_server_is_running():
            print(f"Started this project's Ollama server at {settings.OLLAMA_ADDRESS}.")
            return True
        time.sleep(1)
    raise RuntimeError(f"The Ollama server didn't answer within {SECONDS_TO_WAIT_FOR_THE_SERVER} seconds. "
                       f"See {settings.MODELS_FOLDER / 'ollama-server.log'} for its messages.")


def server_process_ids() -> list[int]:
    """The process ids of whatever program is listening on this project's Ollama port."""
    # Here we ask the system which programs listen on the port; -t prints only their process ids.
    lsof_result = subprocess.run(["lsof", "-t", "-i", f"TCP:{settings.OLLAMA_PORT}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True)
    return [int(process_id) for process_id in lsof_result.stdout.split()]


def stop_ollama_server() -> None:
    """Stop this project's Ollama server, and wait until it no longer answers. Works the same on the Mac and Kaggle."""
    if not ollama_server_is_running():
        print(f"No Ollama server is running at {settings.OLLAMA_ADDRESS}. Nothing to stop.")
        return
    if os.name == "nt":
        # Windows has no lsof; only stop the process this Python session started.
        if _started_server_process is None or _started_server_process.poll() is not None:
            print("WARNING  this session did not start the running Ollama server; leaving it running.")
            return
        _started_server_process.terminate()
        _started_server_process.wait(timeout=SECONDS_TO_WAIT_FOR_THE_SERVER)
    else:
        for process_id in server_process_ids():
            os.kill(process_id, signal.SIGTERM)

    # Here we wait until the server stops answering, so we only report success once it has really stopped.
    for _ in range(SECONDS_TO_WAIT_FOR_THE_SERVER):
        if not ollama_server_is_running():
            print(f"Stopped this project's Ollama server at {settings.OLLAMA_ADDRESS}.")
            return
        time.sleep(1)
    print(f"WARNING  the Ollama server at {settings.OLLAMA_ADDRESS} is still running.")


def download_model(model_name: str) -> None:
    """Download a model, such as "qwen3:8b", into this machine's models folder, unless it is already there.

    This project's server must be running (start_ollama_server), because the server is what stores the model.
    """
    if settings.model_is_downloaded(model_name):
        print(f"{model_name} is already in {settings.MODELS_FOLDER}.")
        return
    if not ollama_server_is_running():
        raise RuntimeError(f"Start this project's Ollama server first (start_ollama_server), "
                           f"then download {model_name}.")

    print(f"Downloading {model_name} into {settings.MODELS_FOLDER} ...")
    # Here ollama_environment() points `ollama pull` at our server's port, so the model lands in our models folder.
    subprocess.run(["ollama", "pull", model_name], env=settings.ollama_environment(), check=True)


# ---------------------------------------------------------------------------
# One call to the model
# ---------------------------------------------------------------------------

def model_for_role(role: str) -> str:
    """The model a role ("judge", "rewriter" or "answerer") uses, from config.yaml's `models` section."""
    models = load_config()["models"]
    if not models.get(role):
        raise ValueError(f"config.yaml has no model for the role {role!r}; it has: {', '.join(models)}")
    return models[role]


def reply_limit_for_role(role: str) -> int:
    """The most tokens a role's reply may have, from config.yaml's `llm.max_tokens` (set in the pilot, then frozen)."""
    return load_config()["llm"]["max_tokens"][role]


def context_window_for(system_prompt: str, prompt: str, reply_room_in_tokens: int) -> int:
    """A context window just big enough for this call's prompt and its reply, in whole blocks of 8,192 tokens,
    and never bigger than the largest window this machine holds (settings.py)."""
    # Here we estimate the prompt's tokens from its characters, then add the safety margin and the room for the reply.
    estimated_prompt_tokens = (len(system_prompt) + len(prompt)) / CHARACTERS_PER_TOKEN
    needed_tokens = estimated_prompt_tokens * TOKEN_ESTIMATE_SAFETY_FACTOR + reply_room_in_tokens

    # Here we round up to the next whole block, so the window is never smaller than what is needed.
    number_of_blocks = math.ceil(needed_tokens / CONTEXT_WINDOW_BLOCK_IN_TOKENS)
    rounded_window = number_of_blocks * CONTEXT_WINDOW_BLOCK_IN_TOKENS

    # Here we never ask for more than this machine holds; a longer prompt is then flagged as possibly cut.
    return min(rounded_window, settings.OLLAMA_CONTEXT_WINDOW)


def json_or_none(reply: str) -> dict | None:
    """The reply as a Python dictionary, or None if it isn't complete JSON (which happens when the reply was cut)."""
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        return None


def send_to_ollama(model: str, system_prompt: str, prompt: str, answer_format: dict,
                   temperature: float, seed: int, reply_limit_in_tokens: int) -> dict:
    """Make one call: send a role's rules and a prompt to the model, and return its answer and what the call cost.

    One call = one prompt sent to one model and its one reply. This is the only code that sends a prompt to a model
    (embed_texts, below, is the only code that asks for embeddings), and it writes no log: each method's own log function records what it needs from the returned dictionary.
    Methods normally call send_to_ollama_with_retries(), which uses this function.

    Inputs:
        model                  which Ollama model answers, e.g. "qwen3:8b"
        system_prompt          the role's rules
        prompt                 the question, Evidence and instructions for this call
        answer_format          the JSON shape the reply must have
        temperature            how random the reply is (0 = always the most likely word)
        seed                   fixes the randomness, so a call can be repeated exactly
        reply_limit_in_tokens  the most tokens the answer may have (reply_limit_for_role gives a role's limit)

    Returns a dictionary with:
        answer             the reply as a dictionary (None if the reply was cut before its JSON ended)
        reply              the reply exactly as the model wrote it
        thinking           the model's thinking ("" when THINKING_ON is False)
        word_odds          for each word written: the word, and the TOP_ALTERNATIVES words it weighed, with odds
        tokens_read        prompt tokens the model read
        tokens_written     reply tokens the model wrote
        seconds            how long the call took
        context_window     the window this call asked for
        prompt_may_be_cut  True if the prompt left too little room for the reply: Ollama may have dropped part of it
        reply_was_cut      True if the reply stopped at the reply limit instead of ending by itself
    """
    # Here the reply room is the answer's limit, plus room for thinking when THINKING_ON is True.
    reply_room_in_tokens = reply_limit_in_tokens + ROOM_FOR_THINKING_IN_TOKENS
    context_window = context_window_for(system_prompt, prompt, reply_room_in_tokens)

    # Here we build the message in Ollama's chat format: the role's rules as the system prompt, then the prompt.
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        # Here we ask for the whole reply at once, and switch thinking on or off for this call.
        "stream": False,
        "think": THINKING_ON,
        # Here we make the model answer in the JSON shape the caller asks for.
        "format": answer_format,
        # Here we ask for the odds of the words the model weighed at each position (p_yes reads them).
        "logprobs": True,
        "top_logprobs": TOP_ALTERNATIVES,
        # Here num_ctx lets the model read this call's whole window; without it Ollama reads only about 2,000 tokens and
        # silently drops the rest of the prompt. num_predict stops the reply at the room kept free for it.
        # The temperature and seed are the caller's, so the call's randomness is fixed and can be repeated.
        "options": {
            "num_ctx": context_window,
            "num_predict": reply_room_in_tokens,
            "temperature": temperature,
            "seed": seed,
        },
    }
    # Here we address the message to this project's server as JSON.
    request = urllib.request.Request(settings.OLLAMA_ADDRESS + "/api/chat",
                                     data=json.dumps(request_body).encode(),
                                     headers={"Content-Type": "application/json"})

    # Here we wait for the reply and time it.
    start_time = time.time()
    try:
        with urllib.request.urlopen(request, timeout=SECONDS_TO_WAIT_FOR_A_REPLY) as response:
            server_answer = json.load(response)
    except urllib.error.HTTPError as http_error:
        # Here we pass on Ollama's own explanation (e.g. "model 'x' not found"), not just "HTTP Error 404".
        raise RuntimeError(f"Ollama refused the call to {model}: {http_error.read().decode()}") from http_error
    seconds = round(time.time() - start_time, 1)

    # Here we read Ollama's token counts: prompt_eval_count = tokens read, eval_count = tokens written.
    tokens_read = server_answer.get("prompt_eval_count", 0)
    reply = server_answer["message"]["content"]
    return {
        "answer": json_or_none(reply),
        "reply": reply,
        "thinking": server_answer["message"].get("thinking", ""),
        "word_odds": server_answer.get("logprobs") or [],
        "tokens_read": tokens_read,
        "tokens_written": server_answer.get("eval_count", 0),
        "seconds": seconds,
        "context_window": context_window,
        # Here a prompt that ate into the room kept for the reply may have been cut to fit the window.
        "prompt_may_be_cut": tokens_read + reply_room_in_tokens > context_window,
        # Here Ollama says "length" when it stopped the reply at the limit, and "stop" when the reply ended by itself.
        "reply_was_cut": server_answer.get("done_reason") == "length",
    }


def answer_problems(answer: dict | None, answer_format: dict) -> list[str]:
    """What is wrong with an answer, compared with its JSON shape (an empty list means it fits).

    Problems: the reply isn't complete JSON, a required field is missing, a field with a fixed list of values
    (such as yes or no) has another value, or a required text field is empty.
    """
    if not isinstance(answer, dict):
        return ["the reply isn't complete JSON"]
    problems = []
    for field_name in answer_format.get("required", []):
        field_rules = answer_format["properties"][field_name]
        if field_name not in answer:
            problems.append(f'"{field_name}" is missing')
            continue
        value = answer[field_name]
        if "enum" in field_rules and value not in field_rules["enum"]:
            problems.append(f'"{field_name}" must be one of: {", ".join(field_rules["enum"])}')
        elif isinstance(value, str) and not value.strip():
            problems.append(f'"{field_name}" is empty')
    return problems


def send_to_ollama_with_retries(model: str, system_prompt: str, prompt: str, answer_format: dict,
                                temperature: float, seed: int, reply_limit_in_tokens: int) -> dict:
    """Send a call, and ask again (up to MAXIMUM_ATTEMPTS in all) while the answer doesn't fit its JSON shape.

    Takes the same inputs as send_to_ollama. Returns the last attempt's dictionary, with three changes:
        tokens_read, tokens_written, seconds   added up over every attempt, because every attempt costs
        attempts                               how many calls it took (1 = the first answer fitted)
        answer_problems                        what is still wrong with the last answer ([] = it fits)
    """
    attempt_prompt = prompt
    total_tokens_read = 0
    total_tokens_written = 0
    total_seconds = 0.0
    for attempt_number in range(1, MAXIMUM_ATTEMPTS + 1):
        result = send_to_ollama(model, system_prompt, attempt_prompt, answer_format,
                                temperature, seed, reply_limit_in_tokens)
        total_tokens_read += result["tokens_read"]
        total_tokens_written += result["tokens_written"]
        total_seconds += result["seconds"]
        problems = answer_problems(result["answer"], answer_format)
        if not problems:
            break
        # Here the next attempt gets the original prompt plus a note saying what was wrong with this answer.
        attempt_prompt = prompt + RETRY_NOTE.format(problems="; ".join(problems))

    result["tokens_read"] = total_tokens_read
    result["tokens_written"] = total_tokens_written
    result["seconds"] = round(total_seconds, 1)
    result["attempts"] = attempt_number
    result["answer_problems"] = problems
    return result


# ---------------------------------------------------------------------------
# Embeddings (scoring.py uses them to compare answers by meaning)
# ---------------------------------------------------------------------------

def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Turn each text into its embedding (a long list of numbers that stands for its meaning), in one call.

    model is an Ollama embedding model, such as config.yaml's scoring.embedder ("qwen3-embedding:0.6b").
    Returns one embedding per text, in the same order as the texts.
    """
    request_body = {"model": model, "input": texts}
    # Here we address the texts to this project's server, at Ollama's embedding address.
    request = urllib.request.Request(settings.OLLAMA_ADDRESS + "/api/embed",
                                     data=json.dumps(request_body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=SECONDS_TO_WAIT_FOR_EMBEDDINGS) as response:
            server_answer = json.load(response)
    except urllib.error.HTTPError as http_error:
        # Here we pass on Ollama's own explanation (e.g. "model 'x' not found"), not just "HTTP Error 404".
        raise RuntimeError(f"Ollama refused the embedding call to {model}: {http_error.read().decode()}") from http_error
    return server_answer["embeddings"]


# ---------------------------------------------------------------------------
# The Judge's confidence
# ---------------------------------------------------------------------------

def odds_at_field(word_odds: list[dict], field_name: str) -> list[dict] | None:
    """The alternatives the model weighed for the first word of one JSON field's value, e.g. the "yes" after
    "enough":, each as {token, logprob}. None if the field isn't in the reply."""
    field_key = f'"{field_name}":'
    text_so_far = ""
    for word in word_odds:
        # Here we skip punctuation such as ' "' between the key and its value: the value starts at the first letters.
        if field_key in text_so_far and word["token"].strip(' "\n\t'):
            return word.get("top_logprobs", [])
        text_so_far += word["token"]
    return None


def spelling_as_word(token: str) -> str:
    """A token reduced to its plain word, ignoring spaces, quotes and capitals: ' "Yes' -> "yes"."""
    return token.strip().strip('"').lower()


def p_yes_normalized(alternatives: list[dict] | None) -> float | None:
    """How sure the Judge was: the odds of every spelling of "yes", divided by those of "yes" plus "no".

    For example 0.9 = fairly sure the Evidence is enough. None if neither word is among the alternatives.
    """
    if not alternatives:
        return None
    yes_odds = 0.0
    no_odds = 0.0
    for alternative in alternatives:
        # Here a logprob is a log of the odds, so math.exp turns it back into odds between 0 and 1.
        odds = math.exp(alternative["logprob"])
        word = spelling_as_word(alternative["token"])
        if word == "yes":
            yes_odds += odds
        elif word == "no":
            no_odds += odds
    if yes_odds + no_odds == 0:
        return None
    return yes_odds / (yes_odds + no_odds)


if __name__ == "__main__":
    config = load_config()
    judge_model = model_for_role("judge")
    if not settings.model_is_downloaded(judge_model):
        raise SystemExit(f"{judge_model} isn't in {settings.MODELS_FOLDER}. Run project_set_up.py first.")

    server_started_here = start_ollama_server()
    # This example shows a demo of how the judge will behave. 
    try:
        # Here we use the config's temperature, first seed and the Judge's reply limit, as the methods will.
        result = send_to_ollama_with_retries(judge_model, DEMO_SYSTEM_PROMPT, DEMO_PROMPT, DEMO_ANSWER_FORMAT,
                                             temperature=config["llm"]["temperature"], seed=config["seeds"][0],
                                             reply_limit_in_tokens=reply_limit_for_role("judge"))
        print(f"\nModel: {judge_model}")
        print(f"Answer: {json.dumps(result['answer'], indent=2)}")
        print(f"Attempts: {result['attempts']}   Problems left: {result['answer_problems'] or 'none'}")
        print(f"p_yes: {p_yes_normalized(odds_at_field(result['word_odds'], 'enough'))}")
        print(f"Tokens read: {result['tokens_read']}, written: {result['tokens_written']}, "
              f"in {result['seconds']} s (context window {result['context_window']:,})")
        print(f"Prompt may be cut: {result['prompt_may_be_cut']}   Reply was cut: {result['reply_was_cut']}\n")
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            stop_ollama_server()
