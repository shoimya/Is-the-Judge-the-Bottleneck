# T04 · Model backend (plug-and-play roles)

Tier: 1 · Owner: _ · Depends: T02, research [02](../specification.md#12-research-findings-behind-the-decisions)

## What

- `pipeline/llm.py`: one function all code calls, `generate(role, prompt) -> {text, prompt_tokens, completion_tokens, logprobs}`.
- Two backends behind it: **vLLM** (Kaggle, for every reported number) and **Ollama** (Mac, development only). The backend is picked in the config.
- `config.yaml` gets `backend: ollama | vllm` and one model setting per role (`models.judge`, `models.rewriter`, `models.answerer`), each holding **both backend names**, e.g. `{ollama: qwen3:4b-instruct, vllm: Qwen/Qwen3-4B-Instruct-2507}`. Switching machines changes only `backend`. Greedy decoding (temperature 0), max output tokens per role and `top_logprobs` also live there. The model names are development values until T10 picks the final model.
- A cell in `kaggle.ipynb` that loads the model and runs 3 test prompts.
- **Setup facts from [research 02](../specification.md#12-research-findings-behind-the-decisions):**
  - Kaggle's free GPU is 2× T4 (fp16 only, no bf16). Pin `vllm==0.30.0` (fallback 0.28.0) in a separate `requirements-gpu.txt`, because vLLM doesn't install on the Mac.
  - Run one model copy per T4 (data parallel) for the 8B AWQ model; don't use FP8 KV cache or act-order GPTQ checkpoints.
  - Qwen3 models: pass `enable_thinking=False` in the chat template. The Judge's reasoning goes in its REASONING field instead.
  - `logprobs` holds, for **every** generated token, the chosen token's log probability and the top 5 alternatives (both backends). T06 reads P("yes") at the token after `ENOUGH:`.
  - Ollama returns logprobs since v0.12.11, but only for development.
- **Measure throughput:** run 20 Pilot set questions through a fake 3-Round loop on Kaggle (padding prompts with filler paragraphs to realistic Evidence length, since BM25 may not be ready yet) and record prompt tok/s, generated tok/s and wall time in the Throughput section of `NOTES.md`.

## Done when

- The same 3 prompts run on both backends and return text and token counts.
- On both backends, `logprobs` has one entry per generated token with the chosen token and its top 5 alternatives, so P("yes") can be read at any position later.
- Changing `models.judge` in the config changes which model the Judge uses without touching code.
- The Throughput section of `NOTES.md` has the measured numbers.

## Description

Every role goes through one door. That makes swapping a model a config change, which is exactly what Tier 3 needs (swap only the Judge). The Mac is for fast development. Numbers in the paper only come from vLLM on Kaggle, because the Mac's compressed models can answer slightly differently.

## Notes
- Oct 4: Mac part done test-first. `pipeline/llm.py`: `generate`/`generate_many`, `model_for`, Ollama and vLLM backends, `smoke` and `benchmark` commands; `requirements-gpu.txt` pins vLLM. Ollama runs from `data/ollama/` on the SSD (README). Ollama returns no logprobs entry for the end-of-answer token; vLLM does (documented in the module).
- Oct 4 code review fixes: clearer names (`check_setup`, `arguments`, `evidence_prompt`, `config_file`, `work_dir`, `repo_dir`, test helpers); safe fallback when Ollama omits the prompt token count; the yes/no adding-up moved out of `llm.py` (now T06's job, see T06); docstring corrected for vLLM; `kaggle.ipynb` T04 cell installs `requirements-gpu.txt` and runs `smoke` and `benchmark`.
- Remaining for Done: on Kaggle (GPU T4 x2, index attached) run the T04 cell; paste the benchmark numbers into the Throughput section of `NOTES.md`.
- Open review findings, not yet fixed (owner's call): settings still written in code — the Ollama address and `timeout=300`, `dtype="float16"`, the benchmark's 20 questions / 5 paragraphs per Round, the `hotpotqa` data folder name, and the tests' Ollama address and `5`. The benchmark's GPU-hour estimate multiplies by 5 loop settings; spec §8.3 budgets 7 (Single-turn and Closed-book also use the GPU), so it understates cost.
