# Improving Generative AI Systems through Prompt Optimization and Efficient Processing

**CS5202 — GenAI and LLM | Spring 2026 | Department of CS & AI**

---

## Team Members

| Name | Roll Number |
|------|------------|
| Laasya Sri | SE23UARI044 |
| Pradham Reddy | SE23UARI055 |
| Akshitha Kothapally | SE23UARI057 |
| Vivek TKS | SE23UECM060 |

---

## Problem Summary

Transformer-based language models (T5, GPT, etc.) are highly sensitive to how prompts are phrased — a minor rewording can produce drastically different outputs. Additionally, long documents cause context degradation, where models lose track of earlier information. This project addresses both issues without modifying the underlying model.

**Our two-part approach:**
1. **Prompt Optimization** — restructures inputs using role-priming, chain-of-thought hints, and filler removal to reduce output variance.
2. **Chunk-based Processing** — splits long documents into overlapping segments, summarizes each independently, and merges results to preserve full-document context.

---

## Repository Structure

```
project-<id>-SE23UARI044/
├── README.md                    ← This file
├── domain_note.pdf              ← Milestone 1 domain note
├── requirements.txt             ← Pinned Python dependencies
├── data/
│   └── sample.jsonl             ← Cached CNN/DailyMail test split (auto-generated)
├── src/
│   ├── pipeline.py              ← Main end-to-end pipeline (entry point)
│   ├── prompt_optimizer.py      ← Prompt restructuring logic + ablation variants
│   ├── chunk_processor.py       ← Overlapping chunk splitter + merge strategies
│   ├── data_loader.py           ← CNN/DailyMail dataset loading + preprocessing
│   ├── evaluate.py              ← ROUGE + BLEU scoring + JSONL logging
│   ├── visualize_results.py     ← Generates all result plots
│   └── test_modules.py          ← Unit tests (no model needed)
├── notebooks/
│   └── exploration.ipynb        ← Data exploration and preliminary analysis
└── results/
    ├── eval_log_<run_id>.jsonl  ← Per-sample evaluation log
    ├── summary_<run_id>.json    ← Aggregated scores per variant
    ├── bar_chart_rouge.png      ← ROUGE comparison chart
    ├── bar_chart_bleu.png       ← BLEU comparison chart
    ├── heatmap_metrics.png      ← Full metric heatmap
    └── long_vs_short.png        ← Chunk vs. direct on long docs
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/laasyagummadi/<repo-name>.git
cd <repo-name>

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt')"
```

---

## How to Run

### Run unit tests first (no GPU / model download needed)

```bash
python src/test_modules.py
```

### Run the full pipeline

```bash
# Standard run — 100 articles from CNN/DailyMail test split
python src/pipeline.py --split test --sample 100 --max_new_tokens 150

# Faster run for quick experiments
python src/pipeline.py --split test --sample 20 --max_new_tokens 100

# Use cached data (skip re-downloading)
python src/pipeline.py --use_cached --sample 100
```

### Generate plots from results

```bash
python src/visualize_results.py \
    --summary results/summary_<run_id>.json \
    --log results/eval_log_<run_id>.jsonl
```

---

## Ablation Study Design

The pipeline tests **5 prompt variants** for every article:

| Variant | Role Prefix | CoT Suffix | Filler Removal |
|---------|:-----------:|:----------:|:--------------:|
| `baseline` | ✗ | ✗ | ✗ |
| `role_only` | ✓ | ✗ | ✗ |
| `cot_only` | ✗ | ✓ | ✗ |
| `no_fillers` | ✗ | ✗ | ✓ |
| `full` | ✓ | ✓ | ✓ |

For long documents (>300 words), an additional `chunk_full` variant applies chunk-based processing on top of the full optimizer.

---

## Dataset

**CNN/DailyMail 3.0.0** (publicly available)
- Source: [huggingface.co/datasets/cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)
- Fields used: `article` (input), `highlights` (reference summary)
- Split: `test` (used to avoid data leakage)
- Sample size: 100–200 articles (configurable)

---

## Model

**T5-small** from HuggingFace Transformers
- Chosen for: lightweight, seq2seq architecture, publicly available, fast iteration
- Swap to `t5-base` or `google/flan-t5-base` in `pipeline.py` for stronger baselines

---

## Reproducibility

All random seeds are fixed to `42` across Python, NumPy, and PyTorch. Every evaluation result is logged with a timestamp to `results/eval_log_<run_id>.jsonl`. The exact dataset sample is cached to `data/sample.jsonl` on first run.

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| ROUGE-1 | Unigram overlap between generated and reference summary |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence (fluency proxy) |
| BLEU | n-gram precision with brevity penalty |

---

## Milestone Status

| Milestone | Due | Status |
|-----------|-----|--------|
| Domain Note + Data Pipeline + Initial Results | Apr 30, 2026 | ✅ Submitted |
| Full System + Ablation + 4–6 page Report | May 15, 2026 | 🔄 In Progress |
