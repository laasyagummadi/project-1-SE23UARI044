"""
chunk_processor.py
------------------
Chunk-based processing module for long documents.

Strategy:
  1. Split document into overlapping chunks (preserves boundary context).
  2. Summarize each chunk independently.
  3. Merge chunk summaries into a final coherent output.

This prevents the "lost-in-the-middle" problem where models ignore
text that appears far from the start or end of a long input.
"""

import re
import random
from typing import List

# Fix random seed for reproducibility
random.seed(42)


# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------

class SentenceSplitter:
    """
    Splits text into sentences and groups them into chunks of at most
    *max_sentences* each, with *overlap* sentences of overlap between
    consecutive chunks.
    """

    def __init__(self, max_sentences: int = 8, overlap: int = 2):
        if overlap >= max_sentences:
            raise ValueError("overlap must be less than max_sentences")
        self.max_sentences = max_sentences
        self.overlap = overlap

    def _sentence_tokenize(self, text: str) -> List[str]:
        """Naive sentence splitter (no NLTK dependency required)."""
        # split on .  !  ? followed by whitespace or end-of-string
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        # filter empty strings
        return [s.strip() for s in raw if s.strip()]

    def split(self, text: str) -> List[str]:
        """
        Return a list of chunk strings.

        Each chunk contains at most *max_sentences* sentences.
        Consecutive chunks share *overlap* sentences to preserve context.
        """
        sentences = self._sentence_tokenize(text)
        if not sentences:
            return []

        chunks = []
        step = self.max_sentences - self.overlap
        i = 0
        while i < len(sentences):
            chunk_sents = sentences[i : i + self.max_sentences]
            chunks.append(" ".join(chunk_sents))
            i += step

        return chunks


# ---------------------------------------------------------------------------
# Chunk processor
# ---------------------------------------------------------------------------

class ChunkProcessor:
    """
    Applies a *summarize_fn* callable to each chunk and merges results.

    Parameters
    ----------
    summarize_fn : callable
        Function with signature summarize_fn(text: str) -> str.
        Can be a local T5 call, a GPT API call, or a mock for testing.
    max_sentences : int
        Max sentences per chunk.
    overlap : int
        Overlapping sentences between adjacent chunks.
    merge_strategy : str
        "concat"    — join chunk summaries with newlines
        "recursive" — summarize the joined chunk summaries again (two-pass)
    """

    def __init__(
        self,
        summarize_fn,
        max_sentences: int = 8,
        overlap: int = 2,
        merge_strategy: str = "concat",
    ):
        self.summarize_fn = summarize_fn
        self.splitter = SentenceSplitter(max_sentences=max_sentences, overlap=overlap)
        self.merge_strategy = merge_strategy

    def process(self, document: str) -> dict:
        """
        Summarize *document* using chunk-based processing.

        Returns
        -------
        dict with keys:
            chunks          : list[str]  — the raw text chunks
            chunk_summaries : list[str]  — per-chunk summaries
            final_summary   : str        — merged / re-summarized result
            num_chunks      : int
        """
        chunks = self.splitter.split(document)
        if not chunks:
            return {
                "chunks": [],
                "chunk_summaries": [],
                "final_summary": "",
                "num_chunks": 0,
            }

        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            summary = self.summarize_fn(chunk)
            chunk_summaries.append(summary)

        joined = "\n".join(chunk_summaries)

        if self.merge_strategy == "recursive":
            final_summary = self.summarize_fn(joined)
        else:
            final_summary = joined

        return {
            "chunks": chunks,
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "num_chunks": len(chunks),
        }


# ---------------------------------------------------------------------------
# Quick test / demo (no model needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Transformer models have changed NLP dramatically. "
        "They rely on attention mechanisms to relate every token to every other token. "
        "BERT was introduced by Google in 2018. "
        "GPT-2 followed in 2019, and GPT-3 in 2020. "
        "These models grow exponentially in size. "
        "Handling very long texts remains a challenge. "
        "Chunking is one practical solution. "
        "The idea is to split text, summarize each piece, and combine. "
        "This avoids losing early context in long documents. "
        "Overlap between chunks preserves boundary information. "
    )

    # mock summarizer: return first sentence of chunk
    def mock_summarizer(text):
        first = text.split(".")[0].strip()
        return first + "."

    proc = ChunkProcessor(
        summarize_fn=mock_summarizer,
        max_sentences=4,
        overlap=1,
        merge_strategy="concat",
    )
    result = proc.process(sample)
    print(f"Num chunks: {result['num_chunks']}")
    print("Final summary:", result["final_summary"])
