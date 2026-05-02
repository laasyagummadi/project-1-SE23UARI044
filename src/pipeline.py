"""
pipeline.py
-----------
Main end-to-end pipeline for the CS5202 project:
"Improving Generative AI Systems through Prompt Optimization
 and Efficient Processing"

Workflow
--------
1. Load CNN/DailyMail articles (data_loader.py)
2. For each article:
   a. Build ablation prompt variants (prompt_optimizer.py)
   b. Route to either direct summarization or chunk-based processing
      depending on article length (chunk_processor.py)
   c. Generate summaries with T5-small (huggingface transformers)
   d. Evaluate ROUGE + BLEU vs gold reference (evaluate.py)
3. Aggregate and save results to results/

Usage
-----
    python src/pipeline.py --split test --sample 100 --max_new_tokens 150
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import List, Dict

# Fix seeds for reproducibility
random.seed(42)

import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from transformers import T5ForConditionalGeneration, T5Tokenizer

from data_loader import CNNDailyMailLoader, LONG_DOC_THRESHOLD
from prompt_optimizer import PromptOptimizer, get_ablation_variants
from chunk_processor import ChunkProcessor
from evaluate import SummaryEvaluator, compare_variants

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "t5-small"  # swap to "t5-base" or "google/flan-t5-base" for better results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_PROMPT = (
    "Please just basically summarize the following article very simply:"
)  # intentionally filler-heavy to show optimizer's effect


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class T5Summarizer:
    """
    Wraps T5 tokenizer + model for seq2seq summarization.
    """

    def __init__(self, model_name: str = MODEL_NAME, max_new_tokens: int = 150):
        print(f"[Model] Loading {model_name} …")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[Model] Running on {self.device}")

    def summarize(self, text: str) -> str:
        """Generate a summary for *text*. Truncates input to 512 tokens."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    split: str = "test",
    sample_size: int = 100,
    max_new_tokens: int = 150,
    use_cached: bool = False,
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"  CS5202 Pipeline  |  run_id={run_id}")
    print(f"{'='*60}\n")

    # 1. Load data
    cache_path = "data/sample.jsonl"
    if use_cached and os.path.exists(cache_path):
        records = CNNDailyMailLoader.load_from_file(cache_path)
    else:
        loader = CNNDailyMailLoader(split=split, sample_size=sample_size)
        records = loader.load()
        loader.save_sample(cache_path)

    # 2. Load model
    model = T5Summarizer(max_new_tokens=max_new_tokens)

    # 3. Evaluator
    log_path = os.path.join(RESULTS_DIR, f"eval_log_{run_id}.jsonl")
    evaluator = SummaryEvaluator(log_path=log_path)

    # 4. Chunk processor (uses the model's summarize fn)
    chunk_proc = ChunkProcessor(
        summarize_fn=model.summarize,
        max_sentences=8,
        overlap=2,
        merge_strategy="recursive",
    )

    # 5. Standard prompt optimizer (full config)
    full_optimizer = PromptOptimizer(
        use_role_prefix=True, use_cot_suffix=True, remove_fillers=True
    )

    # Aggregate results
    all_results = []          # per-record, per-variant
    variant_scores = {}       # variant_name → list of metric dicts

    for rec_idx, record in enumerate(records):
        article = record["article"]
        reference = record["reference"]
        doc_id = record.get("id", str(rec_idx))
        is_long = record["is_long"]

        print(
            f"\n[{rec_idx+1}/{len(records)}] id={doc_id} | "
            f"words={record['word_count']} | long={is_long}"
        )

        # --- Ablation: 5 prompt variants × direct summarization ---
        variants_input = get_ablation_variants(RAW_PROMPT, article)
        variant_outputs = {}

        for vname, full_input in variants_input.items():
            start = time.time()
            gen = model.summarize(full_input)
            elapsed = time.time() - start
            variant_outputs[vname] = gen
            scores = evaluator.evaluate(
                gen, reference,
                variant_name=vname,
                extra_meta={"doc_id": doc_id, "is_long": is_long, "elapsed_s": round(elapsed, 2)},
            )
            variant_scores.setdefault(vname, []).append(scores)
            print(
                f"  [{vname:12s}] R1={scores['rouge1']:.3f} "
                f"R2={scores['rouge2']:.3f} RL={scores['rougeL']:.3f} "
                f"BLEU={scores['bleu']:.3f}  ({elapsed:.1f}s)"
            )

        # --- Chunk-based processing (long docs only, using full optimizer) ---
        if is_long:
            opt_article = full_optimizer.build_full_input(RAW_PROMPT, article)
            chunk_result = chunk_proc.process(opt_article)
            gen_chunk = chunk_result["final_summary"]
            scores_chunk = evaluator.evaluate(
                gen_chunk, reference,
                variant_name="chunk_full",
                extra_meta={
                    "doc_id": doc_id,
                    "is_long": is_long,
                    "num_chunks": chunk_result["num_chunks"],
                },
            )
            variant_scores.setdefault("chunk_full", []).append(scores_chunk)
            print(
                f"  [chunk_full  ] R1={scores_chunk['rouge1']:.3f} "
                f"R2={scores_chunk['rouge2']:.3f} RL={scores_chunk['rougeL']:.3f} "
                f"BLEU={scores_chunk['bleu']:.3f}  "
                f"(chunks={chunk_result['num_chunks']})"
            )

        all_results.append(
            {
                "doc_id": doc_id,
                "is_long": is_long,
                "variant_outputs": variant_outputs,
                "reference": reference,
            }
        )

    # 6. Aggregate and save summary table
    summary_rows = []
    for vname, scores_list in variant_scores.items():
        n = len(scores_list)
        avg_r1 = sum(s["rouge1"] for s in scores_list) / n
        avg_r2 = sum(s["rouge2"] for s in scores_list) / n
        avg_rl = sum(s["rougeL"] for s in scores_list) / n
        avg_bl = sum(s["bleu"]   for s in scores_list) / n
        summary_rows.append(
            {
                "variant": vname,
                "n": n,
                "avg_rouge1": round(avg_r1, 4),
                "avg_rouge2": round(avg_r2, 4),
                "avg_rougeL": round(avg_rl, 4),
                "avg_bleu":   round(avg_bl, 4),
            }
        )

    summary_rows.sort(key=lambda x: x["avg_rougeL"], reverse=True)

    summary_path = os.path.join(RESULTS_DIR, f"summary_{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump({"run_id": run_id, "results": summary_rows}, f, indent=2)

    print(f"\n{'='*60}")
    print("  FINAL RESULTS (sorted by avg ROUGE-L)")
    print(f"{'='*60}")
    print(f"{'Variant':<14} {'N':>4} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BLEU':>6}")
    print("-" * 45)
    for row in summary_rows:
        print(
            f"{row['variant']:<14} {row['n']:>4} "
            f"{row['avg_rouge1']:>6.3f} {row['avg_rouge2']:>6.3f} "
            f"{row['avg_rougeL']:>6.3f} {row['avg_bleu']:>6.3f}"
        )

    print(f"\n[Pipeline] Logs  → {log_path}")
    print(f"[Pipeline] Table → {summary_path}")
    return summary_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS5202 Summarization Pipeline")
    parser.add_argument("--split",       default="test",  help="Dataset split")
    parser.add_argument("--sample",      type=int, default=100, help="Number of articles")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Max output tokens")
    parser.add_argument("--use_cached",  action="store_true", help="Use data/sample.jsonl")
    args = parser.parse_args()

    run_pipeline(
        split=args.split,
        sample_size=args.sample,
        max_new_tokens=args.max_new_tokens,
        use_cached=args.use_cached,
    )
