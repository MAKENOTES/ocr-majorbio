from __future__ import annotations

from typing import List


def load_vocab(vocab_path: str) -> List[str]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip("\n") for line in f if line.strip("\n")]
    return vocab


def ctc_greedy_decode(logits, vocab: List[str], blank_id: int = 0) -> str:
    """Greedy CTC decode for a single sample."""
    max_ids = logits.argmax(dim=2).squeeze(1).tolist()
    prev = None
    out_chars = []
    for idx in max_ids:
        if idx == blank_id or idx == prev:
            prev = idx
            continue
        if idx < len(vocab):
            out_chars.append(vocab[idx])
        prev = idx
    return "".join(out_chars)
