import csv
import os
import re
from rapidfuzz.fuzz import ratio
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def norm(x):
    if not x:
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def suffix_name(x):
    x = re.sub(
        r"\b(private|pvt|limited|ltd|incorporated|inc|corporation|corp|company|co)\b",
        " ",
        x
    )
    return re.sub(r"\s+", " ", x).strip()


def load_records(path):
    records = {}

    print("Loading:", path)

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for r in reader:

            eid = r["entity_id"].strip()
            name = norm(r.get("business_name", ""))
            address = norm(r.get("business_address", ""))

            records[eid] = (
                name,
                address,
                suffix_name(name),
                r.get("country", "").strip().lower()
            )

    return records


print("Loading test Source 1...")
s1 = load_records(
    os.path.join(DATA_DIR, "test_source1.tsv")
)

print("Loading test Source 2...")
s2 = load_records(
    os.path.join(DATA_DIR, "test_source2.tsv")
)

print("Loading test Source 3...")
s3 = load_records(
    os.path.join(DATA_DIR, "test_source3.tsv")
)

targets = {}
targets.update(s2)
targets.update(s3)

print("Total target records:", len(targets))

candidate_file = os.path.join(
    DATA_DIR,
    "candidate_pairs.tsv"
)

output_file = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

print("Reading candidates and generating matches...")


def choose_matches(s1_id, candidate_ids):

    if s1_id not in s1:
        return []

    s1_name, s1_address, s1_suffix, s1_country = s1[s1_id]

    candidates = []

    for cid in candidate_ids:

        if cid not in targets:
            continue

        name, address, suffix, country = targets[cid]

        # Country must agree
        if s1_country and country and s1_country != country:
            continue

        # -------------------------------
        # STRONG EXACT MATCHES
        # -------------------------------

        if s1_name and name and s1_name == name:
            candidates.append((1000, cid))
            continue

        if s1_suffix and suffix and s1_suffix == suffix:
            candidates.append((950, cid))
            continue

        if s1_address and address and s1_address == address:
            candidates.append((900, cid))
            continue

        # -------------------------------
        # FUZZY SCORE
        # -------------------------------

        name_score = 0

        if s1_name and name:
            name_score = ratio(
                s1_name,
                name
            )

        address_score = 0

        if s1_address and address:
            address_score = ratio(
                s1_address,
                address
            )

        suffix_score = 0

        if s1_suffix and suffix:
            suffix_score = ratio(
                s1_suffix,
                suffix
            )

        # Name gets highest weight
        score = (
            0.55 * name_score
            + 0.30 * address_score
            + 0.15 * suffix_score
        )

        candidates.append((score, cid))

    if not candidates:
        return []

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # ------------------------------------------------
    # EXACT MATCHES
    # ------------------------------------------------

    exact = [
        cid
        for score, cid in candidates
        if score >= 900
    ]

    if exact:
        return sorted(set(exact))

    # ------------------------------------------------
    # FUZZY MATCHES
    # ------------------------------------------------

    best_score = candidates[0][0]

    # Conservative threshold because F0.5
    # weights precision more heavily.
    if best_score < 78:
        return []

    # Keep candidates close to the best candidate.
    selected = [
        cid
        for score, cid in candidates
        if score >= max(78, best_score - 8)
        and score >= 78
    ]

    # Avoid excessive predictions
    selected = selected[:10]

    return sorted(set(selected))


with open(
    candidate_file,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as fin, open(
    output_file,
    "w",
    encoding="utf-8",
    newline=""
) as fout:

    reader = csv.DictReader(
        fin,
        delimiter="\t"
    )

    writer = csv.writer(
        fout,
        delimiter="\t"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    for row in tqdm(
        reader,
        desc="Matching"
    ):

        s1_id = row["source1_entity_id"]

        raw = row.get(
            "candidate_entity_ids",
            ""
        )

        if not raw:
            matches = []
        else:
            candidate_ids = [
                x.strip()
                for x in raw.split(",")
                if x.strip()
            ]

            matches = choose_matches(
                s1_id,
                candidate_ids
            )

        writer.writerow([
            s1_id,
            ",".join(matches)
        ])


print()
print("=" * 60)
print("MATCHING COMPLETE")
print("=" * 60)
print("Output:", output_file)