"""
test_modules.py
---------------
Unit tests for prompt_optimizer, chunk_processor, and evaluate modules.

Run with:
    python src/test_modules.py
or:
    pytest src/test_modules.py -v
"""

import sys
import os
import random

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

random.seed(42)

from prompt_optimizer import PromptOptimizer, get_ablation_variants
from chunk_processor import SentenceSplitter, ChunkProcessor
from evaluate import SummaryEvaluator

# ---------------------------------------------------------------------------
# PromptOptimizer tests
# ---------------------------------------------------------------------------

def test_filler_removal():
    opt = PromptOptimizer(use_role_prefix=False, use_cot_suffix=False, remove_fillers=True)
    raw = "Please kindly just summarize this."
    result = opt.optimize(raw)
    assert "please" not in result.lower(), "Filler 'please' should be removed"
    assert "kindly" not in result.lower(), "Filler 'kindly' should be removed"
    assert "just" not in result.lower(), "Filler 'just' should be removed"
    print("[PASS] test_filler_removal")


def test_role_prefix_added():
    opt = PromptOptimizer(use_role_prefix=True, use_cot_suffix=False, remove_fillers=False)
    result = opt.optimize("Summarize this.")
    assert result.startswith("You are"), "Role prefix should start with 'You are'"
    print("[PASS] test_role_prefix_added")


def test_cot_suffix_added():
    opt = PromptOptimizer(use_role_prefix=False, use_cot_suffix=True, remove_fillers=False)
    result = opt.optimize("Summarize this.")
    assert "step by step" in result.lower(), "CoT suffix should contain 'step by step'"
    print("[PASS] test_cot_suffix_added")


def test_full_optimizer():
    opt = PromptOptimizer(use_role_prefix=True, use_cot_suffix=True, remove_fillers=True)
    full_input = opt.build_full_input("Please just summarize this.", "Some article text.")
    assert "You are" in full_input
    assert "step by step" in full_input.lower()
    assert "please" not in full_input.lower()
    assert "Some article text." in full_input
    print("[PASS] test_full_optimizer")


def test_ablation_variants():
    variants = get_ablation_variants("Summarize.", "Article text here.")
    expected = {"baseline", "role_only", "cot_only", "no_fillers", "full"}
    assert set(variants.keys()) == expected, f"Expected variants {expected}"
    for name, text in variants.items():
        assert "Article text here." in text, f"Article missing from variant '{name}'"
    print("[PASS] test_ablation_variants")


# ---------------------------------------------------------------------------
# SentenceSplitter tests
# ---------------------------------------------------------------------------

def test_splitter_basic():
    splitter = SentenceSplitter(max_sentences=3, overlap=1)
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = splitter.split(text)
    assert len(chunks) >= 2, "Should produce at least 2 chunks"
    print(f"[PASS] test_splitter_basic ({len(chunks)} chunks)")


def test_splitter_overlap():
    splitter = SentenceSplitter(max_sentences=3, overlap=1)
    text = (
        "First sentence. Second sentence. Third sentence. "
        "Fourth sentence. Fifth sentence."
    )
    chunks = splitter.split(text)
    # Second chunk should contain the last sentence of the first chunk
    assert len(chunks) >= 2
    # The overlap means the last sentence of chunk[0] should appear in chunk[1]
    last_of_first = chunks[0].split(".")[-2].strip()
    assert last_of_first in chunks[1] or len(last_of_first) == 0
    print("[PASS] test_splitter_overlap")


def test_splitter_empty():
    splitter = SentenceSplitter()
    chunks = splitter.split("")
    assert chunks == [], "Empty input should return empty list"
    print("[PASS] test_splitter_empty")


# ---------------------------------------------------------------------------
# ChunkProcessor tests
# ---------------------------------------------------------------------------

def test_chunk_processor_concat():
    def mock_fn(text):
        return "Summary of: " + text[:20]

    proc = ChunkProcessor(mock_fn, max_sentences=3, overlap=1, merge_strategy="concat")
    result = proc.process(
        "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta."
    )
    assert result["num_chunks"] >= 1
    assert result["final_summary"] != ""
    print(f"[PASS] test_chunk_processor_concat ({result['num_chunks']} chunks)")


def test_chunk_processor_recursive():
    calls = []

    def mock_fn(text):
        calls.append(text)
        return "SUMMARY"

    proc = ChunkProcessor(mock_fn, max_sentences=2, overlap=0, merge_strategy="recursive")
    result = proc.process("A. B. C. D. E. F.")
    # recursive: chunk summaries are joined and summarized again
    assert "SUMMARY" in result["final_summary"]
    print(f"[PASS] test_chunk_processor_recursive ({len(calls)} model calls)")


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

def test_rouge_scores():
    ev = SummaryEvaluator(log_path=None)
    gen = "Transformers use attention mechanisms for NLP."
    ref = "Attention mechanisms power transformer models in NLP."
    scores = ev.rouge_scores(gen, ref)
    assert "rouge1" in scores
    assert 0.0 <= scores["rouge1"] <= 1.0
    print(f"[PASS] test_rouge_scores (R1={scores['rouge1']:.3f})")


def test_bleu_score():
    ev = SummaryEvaluator(log_path=None)
    gen = "The model generates text using attention."
    ref = "Attention helps the model generate text."
    score = ev.bleu_score(gen, ref)
    assert 0.0 <= score <= 1.0
    print(f"[PASS] test_bleu_score (BLEU={score:.3f})")


def test_evaluate_no_log():
    ev = SummaryEvaluator(log_path=None)
    result = ev.evaluate(
        "Generated summary here.",
        "Reference summary here.",
        variant_name="test",
    )
    assert result["variant"] == "test"
    assert all(k in result for k in ["rouge1", "rouge2", "rougeL", "bleu"])
    print("[PASS] test_evaluate_no_log")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CS5202 Unit Tests")
    print("="*50)

    test_filler_removal()
    test_role_prefix_added()
    test_cot_suffix_added()
    test_full_optimizer()
    test_ablation_variants()

    test_splitter_basic()
    test_splitter_overlap()
    test_splitter_empty()

    test_chunk_processor_concat()
    test_chunk_processor_recursive()

    test_rouge_scores()
    test_bleu_score()
    test_evaluate_no_log()

    print("\n" + "="*50)
    print("  All tests passed!")
    print("="*50)
