# AI Alignment Faking

First, make sure .hf_token has your hugging face token

Replication of alignment faking behaviour in LLMs. Prompts the same model under two system prompts — one claiming training monitoring, one claiming deployment privacy — and measures compliance gap on [AdvBench](https://github.com/llm-attacks/llm-attacks).

## How to run

### HPC cluster (L40 48 GB) — `main/run.slurm`

Use when you have access to the UNIBO HPC cluster. Runs the full pipeline on a single L40 GPU, bfloat16, no quantization.

```bash
sbatch main/run.slurm
```

```bash
squeue -u joaofilipe.silvade                        # check status
tail -f logs/slurm-<JOBID>.log logs/slurm-<JOBID>.err  # live logs
scancel <JOBID>                                     # cancel
```

> Jobs are killed after 8 h. The pipeline (100 queries × 2 contexts) fits comfortably within that.

---

### Kaggle (T4×2 or P100) — `main/run.ipynb`

Use when burning free Kaggle GPU hours (30 h/week). Open the notebook in a Kaggle session with GPU accelerator enabled.

Before running:
1. Add your HuggingFace token in **Kaggle → Settings → Secrets** with key `HF_TOKEN`.
2. Set `REPO_URL` in cell 1 to your fork.
3. Set `TENSOR_PARALLEL = 1` if assigned a P100 instead of T4×2.

Run all cells top to bottom. The vllm server starts in the background; the final cell runs `gen.py` + `eval.py` and shuts the server down cleanly.

> Uses float16 (T4 has no bfloat16 support) and `--max-model-len 4096`.

---

### Local PC (RTX 2060 6 GB) — `main/run.sh`

Use for quick tests and development. Automatically picks GPU or CPU mode.

```bash
bash main/run.sh
```

**GPU mode** (default if `nvidia-smi` works): loads `casperhansen/llama-3-8b-instruct-awq` (AWQ INT4, ~4.5 GB) with `--max-model-len 2048`.

**CPU fallback** (no GPU detected): loads the original fp32 model on CPU. Functional but slow — expect ~10 min per query.

> For a faster CPU alternative, use [Ollama](https://ollama.com): `ollama pull llama3 && ollama serve`, then point `CLIENT_URL` in `src/gen.py` to `http://localhost:11434/v1`.