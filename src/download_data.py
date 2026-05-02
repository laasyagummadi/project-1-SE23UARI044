"""
download_data.py
----------------
Downloads CNN/DailyMail dataset and saves it as CSV files.
Run this FIRST before pipeline.py.

Usage:
    python src/download_data.py

Outputs:
    data/articles_full.csv      ← all 200 sampled articles
    data/articles_long.csv      ← articles with >300 words
    data/articles_short.csv     ← articles with <=300 words
    data/sample_preview.csv     ← first 10 rows for quick check
"""

import os
import random
import csv

random.seed(42)

print("="*55)
print("  CS5202 — Data Downloader")
print("  Dataset: CNN/DailyMail 3.0.0 (HuggingFace)")
print("="*55)

# Step 1: Install datasets if missing
try:
    from datasets import load_dataset
    print("[OK] datasets package found")
except ImportError:
    print("[!] Installing datasets package...")
    os.system("pip install datasets")
    from datasets import load_dataset

LONG_THRESHOLD = 300   # words
SAMPLE_SIZE    = 200   # articles to download
SPLIT          = "test"

os.makedirs("data", exist_ok=True)

print(f"\n[1/4] Downloading CNN/DailyMail ({SPLIT} split)...")
print("      This may take 1-2 minutes on first run (caches after).")

dataset = load_dataset("cnn_dailymail", "3.0.0", split=SPLIT)
print(f"[OK] Total available in test split: {len(dataset)} articles")

# Deterministic shuffle + sample
indices = list(range(len(dataset)))
random.shuffle(indices)

records = []
for idx in indices:
    row = dataset[idx]
    article   = row["article"].strip()
    reference = row["highlights"].strip()
    word_count = len(article.split())

    if word_count < 50:   # skip very short stubs
        continue

    records.append({
        "id":         row["id"],
        "article":    article,
        "reference":  reference,
        "word_count": word_count,
        "is_long":    word_count > LONG_THRESHOLD,
    })

    if len(records) >= SAMPLE_SIZE:
        break

print(f"[OK] Sampled {len(records)} articles")
long_count  = sum(1 for r in records if r["is_long"])
short_count = len(records) - long_count
print(f"     Long  (>{LONG_THRESHOLD} words): {long_count}")
print(f"     Short (<={LONG_THRESHOLD} words): {short_count}")

# CSV writer helper
FIELDNAMES = ["id", "word_count", "is_long", "article", "reference"]

def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    size_kb = os.path.getsize(path) / 1024
    print(f"[OK] Saved {len(rows):>4} rows → {path}  ({size_kb:.0f} KB)")

print("\n[2/4] Saving CSV files...")
save_csv(records,                                    "data/articles_full.csv")
save_csv([r for r in records if r["is_long"]],       "data/articles_long.csv")
save_csv([r for r in records if not r["is_long"]],   "data/articles_short.csv")
save_csv(records[:10],                               "data/sample_preview.csv")

# Also save as JSONL for the pipeline
import json
jsonl_path = "data/sample.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"[OK] Saved {len(records):>4} rows → {jsonl_path}  (for pipeline.py)")

print("\n[3/4] Preview of first 3 articles:")
print("-"*55)
for i, r in enumerate(records[:3]):
    print(f"  Article {i+1}:  {r['word_count']} words  |  long={r['is_long']}")
    print(f"  Preview:  {r['article'][:120].strip()}...")
    print(f"  Summary:  {r['reference'][:100].strip()}...")
    print()

print("[4/4] Done! Your data files:")
for fname in ["articles_full.csv", "articles_long.csv",
              "articles_short.csv", "sample_preview.csv", "sample.jsonl"]:
    path = os.path.join("data", fname)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  data/{fname:<28} {size_kb:>7.1f} KB")

print("\n✓ Now run:  python src/pipeline.py --use_cached --sample 50")
