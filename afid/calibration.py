#!/usr/bin/env python3
"""
AFID-Q: fusion of the two quality signals and normalisation to the ISO/IEC 29794-1 integer 0-100 scale.

Inputs:
    Q_n = L2 norm of the AFID identity embedding (MagFace)
    Q_e = mean of the 10 expert heads of the AFID-Q regressor, in [0, 1]

Step 1 - fusion:

    F = max(Q_n - 10, 0) * Q_e

    10 is the MagFace norm lower bound, images without friction ridge content
    (pushed to Q_n ~ 5 by the background regulariser) get F = 0.

Step 2 - normalisation (monotone, so it does not change the ranking):

    Q = round(100 * ln(1 + min(F, F_MAX) / K) / ln(1 + F_MAX / K))
    K = Q0² / (F_MAX - 2 Q0)

    A log map followed by min-max scaling rounded to the nearest integer per ISO/IEC 29794-1.

    Q0 = 2 is the utility anchor: Q(Q0) = 50 where P(rank-1 hit) ~ 0.5.

    F_MAX = 18.9514 is the 99th percentile of F over the 69,734 real training
    images; F >= F_MAX maps to 100.
"""
import math

QN_FLOOR = 10.0
Q0 = 2.0
F_MAX = 18.9514
K = Q0 ** 2 / (F_MAX - 2 * Q0)          # 0.26753
_NORM = math.log1p(F_MAX / K)


def fuse(q_n: float, q_e: float) -> float:
    """Step 1"""
    return max(float(q_n) - QN_FLOOR, 0.0) * float(q_e)


def calibrate(fusion: float) -> int:
    """Step 2"""
    f = min(max(float(fusion), 0.0), F_MAX)
    return int(round(100.0 * math.log1p(f / K) / _NORM))


if __name__ == "__main__":
    print(f"{'F = max(Q_n-10,0)*Q_e':<24}{'AFID-Q':<8}note")
    print("-" * 64)
    for f, note in [(0.0, "no usable ridges (Q_n <= 10)"),
                    (0.5, "very poor fingermark"),
                    (2.0, "~50% rank-1 identification probability"),
                    (6.35, "good fingermark / poor fingerprint"),
                    (18.95, "F_MAX and above")]:
        print(f"{f:<24.2f}{calibrate(f):<8d}{note}")
