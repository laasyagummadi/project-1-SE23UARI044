"""
evaluate.py
-----------
Evaluation utilities for generated summaries.

Metrics implemented:
  - ROUGE-1, ROUGE-2, ROUGE-L  (via rouge-score)
  - BLEU                        (via nltk)

All results are logged to results/evaluation_log.jsonl for reproducibility.
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Fix random seed
random.seed(42)

# Ensure NLTK tokenizer data is present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------

LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "evaluation_log.jsonl"
)


class SummaryEvaluator:
    """
    Computes ROUGE and BLEU scores between generated and reference summaries.

    Parameters
    ----------
    log_path : str | None
        Path to the JSONL file where each evaluation result is appended.
        Set to None to disable logging.
    """

    def __init__(self, log_path: str = LOG_PATH):
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smoother = SmoothingFunction().method4
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Return dict of ROUGE F1 scores."""
        scores = self.rouge.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def bleu_score(self, generated: str, reference: str) -> float:
        """Return sentence-level BLEU score (smoothed)."""
        ref_tokens = nltk.word_tokenize(reference.lower())
        gen_tokens = nltk.word_tokenize(generated.lower())
        score = sentence_bleu(
            [ref_tokens], gen_tokens, smoothing_function=self.smoother
        )
        return round(score, 4)

    def evaluate(
        self,
        generated: str,
        reference: str,
        variant_name: str = "unnamed",
        extra_meta: dict = None,
    ) -> Dict:
        """
        Compute all metrics and log the result.

        Parameters
        ----------
        generated    : model-generated summary
        reference    : human / gold-standard reference summary
        variant_name : label for the prompt variant being evaluated
        extra_meta   : any extra fields to store in the log (e.g., doc_id)

        Returns
        -------
        dict with keys: variant, rouge1, rouge2, rougeL, bleu, timestamp
        """
        rouge = self.rouge_scores(generated, reference)
        bleu = self.bleu_score(generated, reference)

        result = {
            "variant": variant_name,
            **rouge,
            "bleu": bleu,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra_meta:
            result.update(extra_meta)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        generated_list: List[str],
        reference_list: List[str],
        variant_name: str = "unnamed",
    ) -> Dict[str, float]:
        """
        Evaluate multiple (generated, reference) pairs and return averages.
        """
        assert len(generated_list) == len(reference_list), (
            "generated_list and reference_list must have the same length"
        )
        records = []
        for i, (gen, ref) in enumerate(zip(generated_list, reference_list)):
            r = self.evaluate(gen, ref, variant_name=variant_name, extra_meta={"idx": i})
            records.append(r)

        avg = {
            "variant": variant_name,
            "n": len(records),
            "avg_rouge1": round(sum(r["rouge1"] for r in records) / len(records), 4),
            "avg_rouge2": round(sum(r["rouge2"] for r in records) / len(records), 4),
            "avg_rougeL": round(sum(r["rougeL"] for r in records) / len(records), 4),
            "avg_bleu":   round(sum(r["bleu"]   for r in records) / len(records), 4),
        }
        return avg


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compare_variants(
    variant_outputs: Dict[str, str],
    reference: str,
    evaluator: SummaryEvaluator = None,
) -> List[Dict]:
    """
    Given a dict of {variant_name: generated_summary}, evaluate each
    against *reference* and return a sorted list (best ROUGE-L first).
    """
    if evaluator is None:
        evaluator = SummaryEvaluator()
    results = []
    for name, gen in variant_outputs.items():
        r = evaluator.evaluate(gen, reference, variant_name=name)
        results.append(r)
    return sorted(results, key=lambda x: x["rougeL"], reverse=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    gen = "Transformer models are large neural networks that use attention."
    ref = "Large transformer models rely on attention mechanisms for NLP tasks."
    ev = SummaryEvaluator(log_path=None)
    r = ev.evaluate(gen, ref, variant_name="smoke_test")
    print("Evaluation result:", json.dumps(r, indent=2))
