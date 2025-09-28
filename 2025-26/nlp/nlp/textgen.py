import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

class ConditionalTextGenerator:
    """
    Generate texts from a vocabulary with user-defined conditional probabilities.
    Supports n-gram contexts (order >= 1), interpolation with a base unigram
    distribution, and simple backoff to shorter contexts.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        base_probs: Optional[Dict[str, float]] = None,
        order: int = 1,
        mix: float = 0.8,   # weight for conditional dist vs base unigram
        seed: Optional[int] = None,
    ):
        assert order >= 1, "order must be >= 1"
        self.vocab = tuple(vocab)
        self.V = set(self.vocab)
        self.order = order
        self.mix = float(mix)
        self.rng = random.Random(seed)

        # Base (unigram) distribution — defaults to uniform over vocab
        if base_probs is None:
            w = 1.0 / len(self.vocab)
            self.base_unigram = {t: w for t in self.vocab}
        else:
            self.base_unigram = self._normalize_dist(base_probs)

        # Rules: mapping from context tuple (len <= order) -> next-token prob dict
        self.rules: Dict[Tuple[str, ...], Dict[str, float]] = {}

    # ---------- Public API ----------

    def set_rule(self, context: Sequence[str], next_token_probs: Dict[str, float]) -> None:
        """
        Define P(next | context). Context length should be <= order.
        next_token_probs is a dict token->prob (not necessarily normalized).
        Tokens not in vocab are ignored; missing tokens implicitly get prob 0.
        """
        ctx = tuple(context)[-self.order:]  # keep last `order` items
        self._validate_context(ctx)
        dist = {t: p for t, p in next_token_probs.items() if t in self.V}
        if not dist:
            raise ValueError("No valid tokens from vocab in next_token_probs.")
        self.rules[ctx] = self._normalize_dist(dist)

    def set_rules(self, mapping: Dict[Tuple[str, ...], Dict[str, float]]) -> None:
        """Bulk version of set_rule."""
        for ctx, dist in mapping.items():
            self.set_rule(ctx, dist)

    def generate(
        self,
        length: int,
        start_context: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """
        Generate a sequence of `length` tokens.
        - If start_context is given, it's used as initial context (truncated to order).
        - Sampling uses longest-available context rule; backs off to shorter context,
          finally to base unigram. Interpolates with base using `mix`.
        """
        if length <= 0:
            return []

        history: List[str] = []
        if start_context:
            history.extend([t for t in start_context if t in self.V])
            history[:] = history[-self.order:]  # truncate

        while len(history) < length:
            ctx = tuple(history[-self.order:]) if history else tuple()
            dist = self._interpolated_distribution(ctx)
            nxt = self._sample_from_dist(dist)
            history.append(nxt)

        return history[:length]

    # ---------- Helpers ----------

    def _interpolated_distribution(self, context: Tuple[str, ...]) -> Dict[str, float]:
        """
        Build the distribution to sample from:
        1) Try longest context rule, backing off to shorter suffixes.
        2) If none found, use base unigram.
        3) Interpolate: mix*rule + (1-mix)*base.
        """
        # Backoff: try full context down to empty tuple
        for k in range(len(context), -1, -1):
            ctx = context[-k:] if k > 0 else tuple()
            if ctx in self.rules:
                rule_dist = self.rules[ctx]
                break
        else:
            rule_dist = None  # shouldn't happen

        base = self.base_unigram

        if rule_dist is None:
            return base

        # Interpolate without assuming full support overlap
        merged_keys = self.V
        mix = self.mix
        out = {}
        for t in merged_keys:
            p_rule = rule_dist.get(t, 0.0)
            p_base = base.get(t, 0.0)
            out[t] = mix * p_rule + (1.0 - mix) * p_base

        return self._normalize_dist(out)

    def _sample_from_dist(self, dist: Dict[str, float]) -> str:
        r = self.rng.random()
        cum = 0.0
        for t, p in dist.items():
            cum += p
            if r <= cum:
                return t
        # Numerical fallback
        return next(iter(dist.keys()))

    def _normalize_dist(self, dist: Dict[str, float]) -> Dict[str, float]:
        total = sum(p for t, p in dist.items() if t in self.V and p > 0)
        if total <= 0:
            raise ValueError("Distribution has non-positive total probability.")
        return {t: (dist.get(t, 0.0) / total) for t in self.vocab}

    def _validate_context(self, ctx: Tuple[str, ...]) -> None:
        if len(ctx) > self.order:
            raise ValueError(f"Context longer than order={self.order}.")
        bad = [t for t in ctx if t not in self.V]
        if bad:
            raise ValueError(f"Context contains tokens not in vocab: {bad}")

    # ---------- Optional: learn rules from counts ----------

    @classmethod
    def from_counts(
        cls,
        vocab: Sequence[str],
        ngrams: Iterable[Tuple[Tuple[str, ...], str]],
        order: int = 1,
        base_probs: Optional[Dict[str, float]] = None,
        mix: float = 0.8,
        seed: Optional[int] = None,
    ) -> "ConditionalTextGenerator":
        """
        Helper to build rules from (context, next) pairs (e.g., extracted from a corpus).
        ngrams: iterable of ((tok_{i-order}..tok_{i-1}), tok_i)
        """
        gen = cls(vocab, base_probs=base_probs, order=order, mix=mix, seed=seed)
        counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        V = set(vocab)
        for ctx, nxt in ngrams:
            ctx = tuple(ctx)[-order:]
            if all(t in V for t in ctx) and nxt in V:
                counts[ctx][nxt] += 1
        for ctx, nxt_counts in counts.items():
            gen.set_rule(ctx, nxt_counts)
        return gen
