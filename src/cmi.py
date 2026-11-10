"""Code-Mixing Index (CMI), Das and Gambäck (2014).

    CMI = 100 * (1 - max(w_i) / (n - u))      if n > u, else 0

n   = number of tokens in the tweet
u   = number of language-independent tokens (tags in `independent`)
w_i = number of tokens of language i; max(w_i) is the most frequent language

0 means a single language; larger values mean more even mixing.
"""
from collections import Counter

LANGS = ("Hin", "Eng")


def cmi(langs, independent=("O", "EMT")):
    """CMI of one tweet given its list of language tags."""
    n = len(langs)
    ind = set(independent)
    u = sum(1 for l in langs if l in ind)
    if n == 0 or n == u:
        return 0.0
    counts = Counter(l for l in langs if l not in ind)
    return 100.0 * (1.0 - max(counts.values()) / (n - u))
