"""Matching: cosine similarity of identity embeddings."""
import numpy as np


def similarity(a, b):
    """Cosine similarity of raw embeddings, -1 to 1; higher means more likely the same finger.

    a, b: embeddings from `AFID.embed`, [768] or [N, 768]. The embeddings are normalised to
    unit length here (their raw length is the quality signal, not part of the match score).
        [768] and [768]    -> float
        [N, 768] and [768] -> [N]
        [N, 768] and [M, 768] -> [N, M], every row of a against every row of b
    """
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    scores = _unit(a) @ _unit(b).T
    return float(scores) if scores.ndim == 0 else scores


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)
