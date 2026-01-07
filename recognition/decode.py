from __future__ import annotations

from typing import List


def load_vocab(vocab_path: str) -> List[str]:
    """加载字符表文件，返回按行排列的字符列表。

    Args:
        vocab_path: 字符表文件路径。

    Returns:
        字符列表。
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip("\n") for line in f if line.strip("\n")]
    return vocab


def ctc_greedy_decode(logits, vocab: List[str], blank_id: int = 0) -> str:
    """对单样本进行 CTC 贪心解码。

    Args:
        logits: 模型输出的序列 logits。
        vocab: 字符表列表。
        blank_id: CTC blank 的索引。

    Returns:
        解码后的字符串。
    """
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
