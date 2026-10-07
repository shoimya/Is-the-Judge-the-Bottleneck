"""The one place that talks to a model. (T04)

Every role calls generate(role, prompt, config). config.yaml's `backend` picks the program that runs the model:
Ollama on the Mac (development only) or vLLM on Kaggle's GPU (every reported number). Both return the same shape:

    {"text": str, "prompt_tokens": int, "completion_tokens": int,
     "logprobs": [{"token": str, "logprob": float, "top": [{"token": str, "logprob": float}, ...]}, ...]}

`logprobs` has one entry per generated token: the token chosen and the 5 most likely alternatives at that position
(the Judge's P("yes") is read from these in T06). completion_tokens counts the end-of-answer token too, but that
token has no logprobs entry on either backend. Ollama's prompt_tokens can be lower when it reuses a cached prompt;
only vLLM counts are reported.

    Press Debug on this file to ask the model one question (needs `ollama serve` running on the Mac).
"""

import json
import os
import urllib.error
import urllib.request

# Decoding rules, the same for every role and both backends (CONTEXT.md, Roles).
TEMPERATURE = 0          # always pick the most likely token, so the same prompt always gives the same reply
TOP_ALTERNATIVES = 5     # how many likely alternatives to keep for each generated token

# Ollama, on the Mac.
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"
OLLAMA_TIMEOUT_SECONDS = 300
START_OLLAMA_HINT = ('Ollama isn\'t running. In a second terminal, from the repo folder, run:\n'
                     '    OLLAMA_MODELS="$PWD/data/ollama" ollama serve')
# The name Ollama uses for each model in config.yaml (which uses the Hugging Face name).
OLLAMA_MODEL_NAMES = {
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3:4b-instruct",
    "Qwen/Qwen3-8B-AWQ": "qwen3:8b",
    "Qwen/Qwen3-14B-AWQ": "qwen3:14b",
}

# vLLM, on Kaggle's T4 GPUs.
VLLM_NUMBER_FORMAT = "float16"    # the T4 has no bfloat16
VLLM_MAX_TOKENS_PER_REQUEST = 8192   # prompt + reply; 3 Rounds of Evidence need ~2K
VLLM_GPU_MEMORY_SHARE = 0.90      # share of the GPU's memory vLLM may use


def model_for(role: str, config: dict) -> str:
    """The model (Hugging Face name) a role uses: `model`, except the Judge uses `judge_model` when one is set."""
    if role == "judge" and config.get("judge_model"):
        return config["judge_model"]
    return config["model"]


def ollama_name_for(model_name: str) -> str:
    """Ollama's name for a model, e.g. Qwen/Qwen3-4B-Instruct-2507 -> qwen3:4b-instruct."""
    if model_name not in OLLAMA_MODEL_NAMES:
        raise KeyError(f"Add {model_name}'s Ollama name to OLLAMA_MODEL_NAMES in pipeline/llm.py.")
    return OLLAMA_MODEL_NAMES[model_name]


def generate(role: str, prompt: str, config: dict) -> dict:
    """Send one prompt to the role's model and return its reply, token counts and per-token probabilities."""
    return generate_many(role, [prompt], config)[0]


def generate_many(role: str, prompts: list[str], config: dict) -> list[dict]:
    """Send several prompts for one role; one reply per prompt, in the same order. vLLM runs them together (faster)."""
    if config["backend"] == "ollama":
        return [generate_with_ollama(role, prompt, config) for prompt in prompts]
    if config["backend"] == "vllm":
        return generate_with_vllm(role, prompts, config)
    raise ValueError(f"backend in config.yaml must be ollama or vllm, not {config['backend']!r}")


def make_token_entry(token: str, logprob: float, alternatives: list[tuple[str, float]]) -> dict:
    """One logprobs entry: the chosen token, its log probability, and the top alternatives (same shape on both backends)."""
    return {
        "token": token,
        "logprob": logprob,
        "top": [{"token": alternative_token, "logprob": alternative_logprob}
                for alternative_token, alternative_logprob in alternatives],
    }


# --- Ollama (the Mac) ---

def ollama_is_running() -> bool:
    """True if the Ollama server answers on this Mac."""
    try:
        with urllib.request.urlopen(OLLAMA_VERSION_URL, timeout=2):
            return True
    except OSError:
        return False


def generate_with_ollama(role: str, prompt: str, config: dict) -> dict:
    """Send one prompt to Ollama and return the reply in the shared shape."""
    request_body = build_ollama_request(role, prompt, config)
    ollama_reply = send_to_ollama(request_body)
    return ollama_reply_to_generation(ollama_reply)


def build_ollama_request(role: str, prompt: str, config: dict) -> dict:
    """The request Ollama expects: which model, the prompt as a chat message, and how to generate."""
    return {
        "model": ollama_name_for(model_for(role, config)),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,      # one complete reply instead of word-by-word pieces
        "think": False,       # Qwen3: answer directly; the Judge writes its reasoning in its REASONING field instead
        "logprobs": True,     # also return how likely each generated token was
        "top_logprobs": TOP_ALTERNATIVES,
        "options": {"temperature": TEMPERATURE, "num_predict": config["max_tokens"][role]},
    }


def send_to_ollama(request_body: dict) -> dict:
    """Post the request to the Ollama server and return its JSON reply; stop with a clear hint if it isn't running."""
    http_request = urllib.request.Request(OLLAMA_CHAT_URL, data=json.dumps(request_body).encode(),
                                          headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(http_request, timeout=OLLAMA_TIMEOUT_SECONDS) as http_response:
            return json.load(http_response)
    except urllib.error.URLError as connection_error:
        raise RuntimeError(START_OLLAMA_HINT) from connection_error


def ollama_reply_to_generation(ollama_reply: dict) -> dict:
    """Copy the parts we need from Ollama's reply into the shared shape."""
    token_entries = [
        make_token_entry(token_info["token"], token_info["logprob"],
                         [(alternative["token"], alternative["logprob"]) for alternative in token_info["top_logprobs"]])
        for token_info in ollama_reply["logprobs"]
    ]
    return {
        "text": ollama_reply["message"]["content"],
        "prompt_tokens": ollama_reply.get("prompt_eval_count", 0),   # missing when Ollama reused a cached prompt
        "completion_tokens": ollama_reply["eval_count"],
        "logprobs": token_entries,
    }


# --- vLLM (Kaggle) ---

# One loaded engine per model name: loading takes minutes, so each model is loaded once and reused.
loaded_vllm_engines: dict = {}


def get_vllm_engine(model_name: str):
    """Load the model onto the GPU the first time it is asked for; after that, return the loaded one."""
    if model_name not in loaded_vllm_engines:
        # Here we make vLLM start its worker as a fresh process: copying (forking) a program that already
        # touched the GPU (e.g. JAX, after loading the index) fails with "CUDA driver initialization failed".
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM   # imported here: vLLM only installs on Kaggle's GPUs, never on the Mac

        loaded_vllm_engines[model_name] = LLM(model=model_name, dtype=VLLM_NUMBER_FORMAT,
                                              max_model_len=VLLM_MAX_TOKENS_PER_REQUEST,
                                              gpu_memory_utilization=VLLM_GPU_MEMORY_SHARE)
    return loaded_vllm_engines[model_name]


def generate_with_vllm(role: str, prompts: list[str], config: dict) -> list[dict]:
    """Run a batch of prompts through vLLM on the GPU and return the replies in the shared shape."""
    from vllm import SamplingParams

    engine = get_vllm_engine(model_for(role, config))
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=config["max_tokens"][role],
                                     logprobs=TOP_ALTERNATIVES)
    # Here we wrap each prompt as a one-message chat and send the whole batch to the GPU at once.
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    vllm_outputs = engine.chat(chats, sampling_params, use_tqdm=False,
                               chat_template_kwargs={"enable_thinking": False})   # Qwen3: answer directly
    return [vllm_output_to_generation(vllm_output) for vllm_output in vllm_outputs]


def vllm_output_to_generation(vllm_output) -> dict:
    """Copy the parts we need from one vLLM reply into the shared shape."""
    completion = vllm_output.outputs[0]
    token_entries = vllm_token_entries(completion)
    # Here we drop the entry for the end-of-answer token: Ollama has no entry for it, so neither do we.
    if completion.finish_reason == "stop" and token_entries:
        token_entries.pop()
    return {
        "text": completion.text,
        "prompt_tokens": len(vllm_output.prompt_token_ids),
        "completion_tokens": len(completion.token_ids),
        "logprobs": token_entries,
    }


def vllm_token_entries(completion) -> list[dict]:
    """One logprobs entry per generated token, from vLLM's {token id: probability info} for each position."""
    token_entries = []
    for chosen_token_id, logprobs_by_token_id in zip(completion.token_ids, completion.logprobs):
        chosen_token = logprobs_by_token_id[chosen_token_id]
        alternatives_by_rank = sorted(logprobs_by_token_id.values(), key=lambda alternative: alternative.rank)
        token_entries.append(make_token_entry(
            chosen_token.decoded_token, chosen_token.logprob,
            [(alternative.decoded_token, alternative.logprob) for alternative in alternatives_by_rank[:TOP_ALTERNATIVES]]))
    return token_entries


def shut_down_vllm_engines() -> None:
    """Stop every loaded vLLM engine and free the GPU, so the program can exit instead of hanging."""
    for engine in loaded_vllm_engines.values():
        engine.llm_engine.engine_core.shutdown()
    loaded_vllm_engines.clear()


if __name__ == "__main__":
    from pipeline.config import load_config

    config = load_config()
    reply = generate("answerer", "Answer with one word: what is the capital of France?", config)
    print(f"[{config['backend']}] {model_for('answerer', config)}: {reply['text']!r}")
    print(f"prompt_tokens={reply['prompt_tokens']}  completion_tokens={reply['completion_tokens']}")
    shut_down_vllm_engines()
