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
- Oct 6, first Kaggle run (GPU T4 x2, branch SC): vLLM 0.30.0 loads Qwen3-4B-Instruct-2507 in fp16 (7.6 GB of the T4, ~30.8K tokens of KV cache) and the smoke test answered all three roles correctly ("Paris", "Where was Scott Derrickson born?", "ENOUGH: no"). Three problems found and fixed:
  - Kaggle's preinstalled TorchAudio is built for CUDA 12.8 but vLLM installs a CUDA 13 PyTorch, so importing vLLM crashed. The notebook's T04 cell now uninstalls TorchAudio (unused).
  - vLLM returned a probability entry for the end-of-answer marker `<|im_end|>`, so the smoke test showed the marker instead of the last word. `llm.py` now drops it, so both backends return the same format.
  - The benchmark failed with "CUDA driver initialization failed": loading the BM25 index first wakes JAX on the GPU, and vLLM then forks its worker. `llm.py` now starts the worker with `spawn`.
- Kaggle environment changes seen on Oct 6, to settle before the main runs: Kaggle now runs **Python 3.13** (project targets 3.12); installing vLLM changed numpy to 2.4.6 and numba to 0.65.0 (we pin 2.5.3 / 0.68.0); `transformers`/`tokenizers` want `huggingface_hub<2.0` while we pin 2.1.1. Regenerate the pins on Kaggle (planned) and pick a `huggingface_hub` 1.x.
- Oct 6, second Kaggle run: the smoke test now shows the judge's verdict correctly (`' no' 0.00, ' yes' -18.91`, i.e. P(no) ≈ 1). The benchmark then failed with only 3.4 of 14.6 GB GPU memory free: loading the BM25 index starts JAX (preinstalled on Kaggle), which reserves 75% of GPU memory by default. Fixed in `pipeline/__init__.py`: `JAX_PLATFORMS=cpu` keeps JAX off the GPU (the search doesn't need it).
- Oct 6: pins aligned with what vLLM needs on Kaggle: `huggingface_hub==1.33.0` (was 2.1.1; `transformers` needs < 2.0), `numpy==2.4.6`, `numba==0.65.0`. Verified on the Mac: 13 tests pass, Pilot recall unchanged, question sets byte-identical.
- Oct 6: `kaggle.ipynb` rebuilt from scratch into 11 numbered sections, each with a plain-language note (what it does, which files it uses). Every command goes through `run()`, which stops the notebook on failure. Handles every problem seen so far: internet off, branch selection, Kaggle's dependency warnings, the attached index (or an optional rebuild), checks that questions and search results match the Mac, TorchAudio removal, and a GPU check before vLLM.
- Oct 6, third Kaggle run (rebuilt notebook): every section passed. Smoke test correct on all three roles; benchmark 20 questions × 3 Rounds in 110 s, 887 prompt tok/s, 118 generated tok/s on one T4 → ≈ 10.7 GPU-hours for Tier 1 main runs (×7 settings, upper bound). Numbers in `NOTES.md` Throughput. **T04 Done.**
- Oct 6 full code review (since the first commit, incl. uncommitted work) — **suggestions, not yet applied; owner decides after the next check:**
  - A. Notebook: section 5 deletes the tracked `runs/.gitkeep` (dirty clone; a same-session re-run of section 3 can fail at `git checkout`); section 8 should use `os.path.lexists` for a dangling index link; section 3 should discard stray local changes before switching branch; section 8's text hard-codes 11/24/34/44% and says rebuild "~40 min" (T03/spec say ~12).
  - B. Spec: record Python 3.13 on Kaggle (or move the project to 3.13); record JAX on CPU, vLLM `spawn`, TorchAudio removal and the dropped end-marker logprob in §6.3/§8.5 and T04 "Done when"; update the spec date and T04 status; remove the superseded pins note above; number the §12 research rows; add `requirements-gpu.txt` and `.python-version` to the layouts (spec, T01, README; README also misses CONTEXT.md and specification.md).
  - C. Code: ~~rename `both_found` → `all_found`~~ (done Oct 6, see below); one shared builder for logprob entries in `llm.py`; group tests by pipeline step, not by date; tests should read the Ollama address and `top_logprobs` instead of repeating them.
  - D. Move remaining settings into `config.yaml`: Ollama address and timeout, `dtype`, benchmark sizes, ×7 (not ×5) settings in the GPU-hour estimate, folder names (`hotpotqa`, `raw`, `question_ids`, `pilot`, `cache/huggingface`).
  - E. Delete `results/.gitkeep` (`results/` now has real files).
  - Later tickets: data parallel one copy per T4 (T10); `gold_answer` vs `answer` field name (T05); Colab can't use the saved index; no automated test for the vLLM path; pass `hotpotqa_config` instead of 3 params (T14); parent-folder CLAUDE.md still points to `.scratch/` and `docs/agents/` (outside the repo).

- Oct 6, readability pass (for the paper's readers): plain "Here we ..." comments above every block that does something, in all code files, the tests and the notebook; docstrings added to `load_config`, `check_setup`, `Retriever.load` and both backend functions. Renames: `to_question` → `raw_row_to_question`, `sampled` → `pilot_and_test_sample`, `as_ids` → `return_word_ids`, `ks` → `cutoffs`, `both_found` → `all_found` (also the header of `results/pilot/bm25_recall.csv`), `sampling` → `sampling_params`, `alternatives`/`ranked` → `logprobs_by_token_id`/`alternatives_by_rank`, `last_word` → `last_visible_token`, `seconds` → `elapsed_seconds`, `make_questions` → `make_fake_questions`, notebook `run()` → `run_or_stop()`, `attached_index` → `attached_index_matches`. Two dense lines split (`build_evidence_prompt()` in `llm.py`, `read_written_files()` in the tests). No behaviour changed: 13 tests pass. Bugs from the same review are not fixed yet.
