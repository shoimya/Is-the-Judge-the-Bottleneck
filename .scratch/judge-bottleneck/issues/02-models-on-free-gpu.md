# Which 3–9B models run on Kaggle's free GPU and give token probabilities?

Type: research
Status: resolved
Blocked by: none

## Question

Kaggle's free GPUs are 2× T4 (16 GB each) or 1× P100. Which open-weight instruct models of about 3–9B (for example Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 9B, Mistral 7B) run under vLLM there, in what precision or quantization, and does vLLM support the T4 for them? Does vLLM return per-token log probabilities (needed for P("yes"))? Does Ollama on the Mac return log probabilities too? Rank 2–3 candidates for the Tier 1 model, and note which larger (~12–14B) or other-family models could serve as Tier 3 judges on the same hardware.

Feeds backlog: [T04](../../../backlog/T04-model-backend.md), [T10](../../../backlog/T10-pilot-and-freeze.md), Tier 3.

## Answer

- **Hardware:** Kaggle's free GPU is now **2× T4 only** (the P100 was retired Sep 15, 2026). The T4 has no bf16, so everything runs in fp16 or 4-bit AWQ.
- **vLLM 0.30.0** still supports the T4, using the Triton attention backend. 0.28.0 is the fallback. Avoid FP8 KV cache and act-order GPTQ checkpoints.
- **Tier 1 candidates:** (1) **Qwen3-8B-AWQ**, one copy per T4, run with `enable_thinking=False`; (2) **Qwen3-4B-Instruct-2507** in fp16; (3) Llama-3.1-8B-Instruct. The Qwen3.5 family and Gemma 4 are ruled out on the T4.
- **Log probabilities:** vLLM returns them, and `logprob_token_ids` can ask for "yes"/"no" directly. Ollama has returned them since v0.12.11, but its quantized model gives different values, so it stays dev-only.
- **Tier 3 candidates:** larger, same family: **Qwen3-14B-AWQ** (fits on one T4). Other family: **Llama-3.1-8B-Instruct** or **Granite-4.2-8B**.
- **Throughput:** no measured batched numbers exist for 8B on a T4. The source-free estimate is ~200–350 output tok/s and 1–2k prompt tok/s per T4. T04 must measure it.
