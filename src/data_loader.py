"""
data_loader.py
--------------
Dataset loading and preprocessing pipeline.

Dataset used: CNN/DailyMail (publicly available via HuggingFace)
  - URL: huggingface.co/datasets/cnn_dailymail
  - 3.0.0 split: train / validation / test
  - Fields: article, highlights (reference summary)

Long articles (> 512 words) are flagged for chunk processing.
Short articles go through the standard single-pass pipeline.
"""

import random
import os
from typing import List, Tuple, Dict

# Fix seed
random.seed(42)

# HuggingFace datasets (install: pip install datasets)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARNING] 'datasets' package not found. Install with: pip install datasets")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"

# Articles above this word count are considered "long" → chunk processing
LONG_DOC_THRESHOLD = 300  # words


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------

class CNNDailyMailLoader:
    """
    Loads and pre-processes a sample from CNN/DailyMail.

    Parameters
    ----------
    split        : "train" | "validation" | "test"
    sample_size  : number of articles to sample (for faster experiments)
    min_words    : skip articles shorter than this (removes stubs)
    """

    def __init__(
        self,
        split: str = "test",
        sample_size: int = 200,
        min_words: int = 50,
    ):
        self.split = split
        self.sample_size = sample_size
        self.min_words = min_words
        self._data = None

    def load(self) -> List[Dict]:
        """
        Download (first call) or use cached dataset, then return samples.

        Returns
        -------
        list of dicts with keys: id, article, reference, is_long
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Install HuggingFace datasets: pip install datasets"
            )

        print(f"[DataLoader] Loading {DATASET_NAME} ({DATASET_VERSION}) — {self.split} split …")
        raw = load_dataset(DATASET_NAME, DATASET_VERSION, split=self.split)

        # shuffle deterministically, then sample
        indices = list(range(len(raw)))
        random.shuffle(indices)
        selected = indices[: self.sample_size * 3]  # over-sample to allow filtering

        records = []
        for idx in selected:
            row = raw[idx]
            article = row["article"].strip()
            reference = row["highlights"].strip()

            word_count = len(article.split())
            if word_count < self.min_words:
                continue

            records.append(
                {
                    "id": row["id"],
                    "article": article,
                    "reference": reference,
                    "word_count": word_count,
                    "is_long": word_count > LONG_DOC_THRESHOLD,
                }
            )

            if len(records) >= self.sample_size:
                break

        print(
            f"[DataLoader] Loaded {len(records)} articles "
            f"({sum(r['is_long'] for r in records)} long, "
            f"{sum(not r['is_long'] for r in records)} short)"
        )
        self._data = records
        return records

    def get_long_articles(self) -> List[Dict]:
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        return [r for r in self._data if r["is_long"]]

    def get_short_articles(self) -> List[Dict]:
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        return [r for r in self._data if not r["is_long"]]

    def save_sample(self, path: str = "data/sample.jsonl"):
        """Persist loaded samples so the exact same set can be re-used."""
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self._data:
                f.write(json.dumps(rec) + "\n")
        print(f"[DataLoader] Saved {len(self._data)} records → {path}")

    @staticmethod
    def load_from_file(path: str) -> List[Dict]:
        """Load previously saved sample from a JSONL file."""
        import json
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        print(f"[DataLoader] Loaded {len(records)} records from {path}")
        return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = CNNDailyMailLoader(split="test", sample_size=50)
    data = loader.load()
    loader.save_sample("data/sample.jsonl")
    print("First article preview:")
    print(data[0]["article"][:300], "…")
    print("Reference:", data[0]["reference"])
