from __future__ import annotations

import re
from typing import Dict, Any

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_levenshtein",
    "name_char3_jaccard",
    "name_token_jaccard",

    "suffix_name_ratio",
    "suffix_name_token_sort",
    "suffix_name_token_set",
    "suffix_name_jaro_winkler",

    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_jaro_winkler",
    "address_levenshtein",
    "address_char3_jaccard",
    "address_token_jaccard",

    "digit_overlap",
    "pin_overlap",

    "exact_name",
    "exact_suffix_name",
    "exact_address",
    "same_country",

    "name_length_difference",
    "address_length_difference",
]


def clean(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def token_set(value: str) -> set[str]:
    value = clean(value)

    if not value:
        return set()

    return set(value.split())


def char_ngrams(value: str, n: int = 3) -> set[str]:
    value = clean(value)

    if len(value) < n:
        return {value} if value else set()

    return {
        value[i:i + n]
        for i in range(len(value) - n + 1)
    }


def jaccard(a: set[str], b: set[str]) -> float:

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def digit_set(value: str) -> set[str]:
    return set(re.findall(r"\d", clean(value)))


def pin_set(value: str) -> set[str]:
    """
    Extract 5 or 6 digit postal/PIN-like values.
    Works for both Indian PINs and 5-digit postal codes.
    """

    return set(
        re.findall(
            r"\b\d{5,6}\b",
            str(value or "")
        )
    )


def normalized_similarity(func, a: str, b: str) -> float:

    a = clean(a)
    b = clean(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return float(func(a, b))


def make_features(
    source1: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, float]:

    # --------------------------------------------------
    # NAME
    # --------------------------------------------------

    name1 = clean(
        source1.get("business_name_normalized")
        or source1.get("business_name")
    )

    name2 = clean(
        candidate.get("business_name_normalized")
        or candidate.get("business_name")
    )

    # --------------------------------------------------
    # SUFFIX-STRIPPED NAME
    # --------------------------------------------------

    suffix1 = clean(
        source1.get("business_name_suffix_stripped")
    )

    suffix2 = clean(
        candidate.get("business_name_suffix_stripped")
    )

    # --------------------------------------------------
    # ADDRESS
    # --------------------------------------------------

    address1 = clean(
        source1.get("business_address_normalized")
        or source1.get("business_address")
    )

    address2 = clean(
        candidate.get("business_address_normalized")
        or candidate.get("business_address")
    )

    # --------------------------------------------------
    # NAME TOKEN / CHARACTER FEATURES
    # --------------------------------------------------

    name_tokens1 = token_set(name1)
    name_tokens2 = token_set(name2)

    name_grams1 = char_ngrams(name1)
    name_grams2 = char_ngrams(name2)

    # --------------------------------------------------
    # ADDRESS TOKEN / CHARACTER FEATURES
    # --------------------------------------------------

    address_tokens1 = token_set(address1)
    address_tokens2 = token_set(address2)

    address_grams1 = char_ngrams(address1)
    address_grams2 = char_ngrams(address2)

    # --------------------------------------------------
    # DIGITS / PIN
    # --------------------------------------------------

    digits1 = digit_set(address1)
    digits2 = digit_set(address2)

    pins1 = pin_set(address1)
    pins2 = pin_set(address2)

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    features = {

        # ============================
        # NAME
        # ============================

        "name_ratio":
            fuzz.ratio(name1, name2) / 100.0,

        "name_token_sort":
            fuzz.token_sort_ratio(name1, name2) / 100.0,

        "name_token_set":
            fuzz.token_set_ratio(name1, name2) / 100.0,

        "name_jaro_winkler":
            JaroWinkler.normalized_similarity(
                name1,
                name2
            ),

        "name_levenshtein":
            Levenshtein.normalized_similarity(
                name1,
                name2
            ),

        "name_char3_jaccard":
            jaccard(
                name_grams1,
                name_grams2
            ),

        "name_token_jaccard":
            jaccard(
                name_tokens1,
                name_tokens2
            ),

        # ============================
        # SUFFIX-STRIPPED NAME
        # ============================

        "suffix_name_ratio":
            fuzz.ratio(suffix1, suffix2) / 100.0,

        "suffix_name_token_sort":
            fuzz.token_sort_ratio(
                suffix1,
                suffix2
            ) / 100.0,

        "suffix_name_token_set":
            fuzz.token_set_ratio(
                suffix1,
                suffix2
            ) / 100.0,

        "suffix_name_jaro_winkler":
            JaroWinkler.normalized_similarity(
                suffix1,
                suffix2
            ),

        # ============================
        # ADDRESS
        # ============================

        "address_ratio":
            fuzz.ratio(address1, address2) / 100.0,

        "address_token_sort":
            fuzz.token_sort_ratio(
                address1,
                address2
            ) / 100.0,

        "address_token_set":
            fuzz.token_set_ratio(
                address1,
                address2
            ) / 100.0,

        "address_jaro_winkler":
            JaroWinkler.normalized_similarity(
                address1,
                address2
            ),

        "address_levenshtein":
            Levenshtein.normalized_similarity(
                address1,
                address2
            ),

        "address_char3_jaccard":
            jaccard(
                address_grams1,
                address_grams2
            ),

        "address_token_jaccard":
            jaccard(
                address_tokens1,
                address_tokens2
            ),

        # ============================
        # DIGITS / PIN
        # ============================

        "digit_overlap":
            jaccard(
                digits1,
                digits2
            ),

        "pin_overlap":
            jaccard(
                pins1,
                pins2
            ),

        # ============================
        # EXACT MATCH FEATURES
        # ============================

        "exact_name":
            float(
                bool(name1)
                and name1 == name2
            ),

        "exact_suffix_name":
            float(
                bool(suffix1)
                and suffix1 == suffix2
            ),

        "exact_address":
            float(
                bool(address1)
                and address1 == address2
            ),

        "same_country":
            float(
                clean(source1.get("country"))
                == clean(candidate.get("country"))
            ),

        # ============================
        # LENGTH DIFFERENCES
        # ============================

        "name_length_difference":
            abs(len(name1) - len(name2)),

        "address_length_difference":
            abs(len(address1) - len(address2)),
    }

    return features