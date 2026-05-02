"""
prompt_optimizer.py
-------------------
Prompt optimization module for improving consistency and clarity
of inputs to transformer-based generative models.

Techniques:
  - Role-priming (adds structured persona prefix)
  - Instruction clarity rewriting
  - Noise/stopword reduction
  - Structured format injection (chain-of-thought hint)
"""

import re
import random

# Fix random seed for reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_PREFIX = (
    "You are a precise and concise summarization assistant. "
    "Your task is to carefully read the following passage and produce a "
    "clear, factual, and well-structured summary.\n\n"
)

COT_SUFFIX = (
    "\n\nThink step by step: identify the key ideas first, then write "
    "the summary in your own words."
)

FILLER_PATTERNS = [
    r"\bplease\b", r"\bkindly\b", r"\bjust\b",
    r"\bbasically\b", r"\bactually\b", r"\bvery\b",
    r"\breally\b", r"\bsimply\b",
]

# ---------------------------------------------------------------------------
# Core optimizer class
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """
    Wraps raw user prompts with structure to reduce output variance.

    Parameters
    ----------
    use_role_prefix : bool
        Prepend a role-priming instruction.
    use_cot_suffix : bool
        Append a chain-of-thought nudge.
    remove_fillers : bool
        Strip filler words that add no information.
    """

    def __init__(
        self,
        use_role_prefix: bool = True,
        use_cot_suffix: bool = True,
        remove_fillers: bool = True,
    ):
        self.use_role_prefix = use_role_prefix
        self.use_cot_suffix = use_cot_suffix
        self.remove_fillers = remove_fillers

    def _strip_fillers(self, text: str) -> str:
        for pat in FILLER_PATTERNS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        # collapse extra whitespace
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    def optimize(self, raw_prompt: str) -> str:
        """
        Return an optimized version of *raw_prompt*.

        Parameters
        ----------
        raw_prompt : str
            The original user-supplied prompt / instruction.

        Returns
        -------
        str
            The restructured prompt ready to be prepended to an article.
        """
        prompt = raw_prompt

        if self.remove_fillers:
            prompt = self._strip_fillers(prompt)

        if self.use_role_prefix:
            prompt = ROLE_PREFIX + prompt

        if self.use_cot_suffix:
            prompt = prompt + COT_SUFFIX

        return prompt

    def build_full_input(self, raw_prompt: str, article_text: str) -> str:
        """
        Combine the optimized prompt with the article text.
        """
        opt_prompt = self.optimize(raw_prompt)
        return f"{opt_prompt}\n\n---\n\n{article_text}"


# ---------------------------------------------------------------------------
# Ablation helper — generate variants for ablation study
# ---------------------------------------------------------------------------

def get_ablation_variants(raw_prompt: str, article: str):
    """
    Return a dict of {variant_name: full_input} for ablation experiments.

    Variants
    --------
    baseline     : no optimization
    role_only    : role prefix only
    cot_only     : chain-of-thought suffix only
    no_fillers   : filler removal only
    full         : all three combined (the main system)
    """
    variants = {}

    # baseline — raw prompt as-is
    variants["baseline"] = f"{raw_prompt}\n\n{article}"

    # role only
    opt = PromptOptimizer(use_role_prefix=True, use_cot_suffix=False, remove_fillers=False)
    variants["role_only"] = opt.build_full_input(raw_prompt, article)

    # cot only
    opt = PromptOptimizer(use_role_prefix=False, use_cot_suffix=True, remove_fillers=False)
    variants["cot_only"] = opt.build_full_input(raw_prompt, article)

    # filler removal only
    opt = PromptOptimizer(use_role_prefix=False, use_cot_suffix=False, remove_fillers=True)
    variants["no_fillers"] = opt.build_full_input(raw_prompt, article)

    # full system
    opt = PromptOptimizer(use_role_prefix=True, use_cot_suffix=True, remove_fillers=True)
    variants["full"] = opt.build_full_input(raw_prompt, article)

    return variants
