"""Code-Mixing Index (CMI) of one tweet, Gambäck and Das (2014).

    CMI = 100 * (1 - max(w_i) / (n - u))      if n > u, else 0

n   = number of tokens in the tweet
u   = number of language-independent tokens (tags in `independent`)
w_i = number of tokens of language i; max(w_i) is the most frequent language

0 means a single language; larger values mean more even mixing.

Purpose:
    Gives every tweet one number that says how evenly Hindi and English are mixed. It is the measure behind
    our research question "does the amount of code-mixing change accuracy?".

Why this file exists:
    The same formula must be used everywhere, otherwise the CMI buckets of different analyses would not
    agree. ``cmi`` is the single implementation.

Place in the pipeline:
    per-token language tags (from src/data.py, column ``langs``)  ->  cmi  ->  results/cmi_per_tweet.csv
    (written by src/eda_part2.py)  ->  read by src/cmi_evaluation.py for the bucket analysis.

Inputs:
    A list of language tags for one tweet, for example ``["Hin", "Hin", "Eng", "O"]``.

Output:
    A float. With only Hindi and English counted it lies between 0 (one language) and 50 (an even split).

Used by:
    src/eda_part2.py and tests/test_cmi.py. (src/cmi_evaluation.py does not import this module; it reads the
    saved per-tweet values.)

Depends on:
    The standard library only.

Design decisions and notes:
    * Tags in ``independent`` (default ``O`` = other and ``EMT`` = emoticon) are language-independent: they
      are removed from both the count of the most frequent language and from the denominator.
    * Our mean CMI on the training set is 29.8, but the task paper reports 25.3 (docs/eda_part2.md). Several
      variants of the formula were tried and none reproduced the paper's value, so we use this standard
      formula consistently and compare tweets by rank only. Open question: the exact definition the
      organisers used.
    * The citation order (Gambäck and Das) should be checked against the paper before publication.
    * ``LANGS`` is defined below but is not used anywhere in the repository.
"""
from collections import Counter

LANGS = ("Hin", "Eng")  # not referenced by any code in the repository; kept as documentation of the two languages


def cmi(langs, independent=("O", "EMT")):
    """Compute the Code-Mixing Index of one tweet from its language tags.

    Purpose:
        Turn a tweet's per-token language tags into one number between 0 and 50 (for two languages).

    How it works:
        1. n = number of tags; u = number of tags in ``independent``.
        2. If there is no token left to judge (empty tweet, or every token independent) the result is 0.
        3. Otherwise count the remaining tags per language, take the largest count, and return
           ``100 * (1 - largest / (n - u))``. A tweet in one language gives 0; an even split between two
           languages gives 50.

    Args:
        langs: List (or other iterable that supports ``len``) of language tags, one per token.
        independent: Tags that do not count as a language. Default ``("O", "EMT")``.

    Returns:
        The index as a float.

    Edge cases:
        * Empty list -> 0.0; only independent tags -> 0.0.
        * A tag that is not in ``independent`` counts as a language, whatever it is. With the project's tags
          (Hin, Eng, O, EMT) only two languages remain, so the maximum is 50. A third non-independent tag could
          push the value above 50.
        * Passing ``independent=("O",)`` makes EMT count as its own language (a test covers this).

    Example:
        cmi(["Hin"] * 5 + ["Eng"] * 3 + ["O", "EMT"])  ->  37.5   (10 tokens, 2 independent, 5 of 8 Hindi)
    """
    n = len(langs)
    ind = set(independent)
    u = sum(1 for l in langs if l in ind)
    # Nothing to compare: no tokens at all, or all tokens are language-independent. Returning 0 also avoids a
    # division by zero below.
    if n == 0 or n == u:
        return 0.0
    counts = Counter(l for l in langs if l not in ind)
    return 100.0 * (1.0 - max(counts.values()) / (n - u))
