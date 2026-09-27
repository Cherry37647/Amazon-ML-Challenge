# Amazon Business Entity Resolution Challenge — Blocking Handoff

## 1. Purpose

This folder contains the supplied challenge data, the verified test candidate file, and small supporting scripts from the preprocessing and blocking stages. The next stage is pairwise entity matching and model development.

## 2. What Has Been Completed

- Original raw train and test files were preserved; the handoff copies are byte-for-byte copies.
- Normalized train and test records already exist in the project workspace. The handoff includes the preprocessing code needed to recreate those normalized representations.
- Test retrieval/blocking was performed using a compact SQLite lookup built from the normalized test Source 2 and Source 3 records.
- Test candidates were generated and the final `data/candidate_pairs.tsv` passed the schema, row-count, duplicate, and candidate-ID checks listed below.
- **Training blocking recall was not finalized.**

The 8.8 GB training retrieval database and all other index/cache artifacts were intentionally left out of this handoff.

## 3. Final Candidate File

`data/candidate_pairs.tsv` has exactly two tab-separated columns:

- `source1_entity_id`
- `candidate_entity_ids`

The second column contains comma-separated Source 2 and Source 3 IDs. Candidate IDs are deduplicated and sorted within each row. Empty candidate lists are allowed. This is the candidate set produced by the blocking stage for the downstream matching stage; the downstream model should score only these candidate pairs.

## 4. How the Teammate Should Use It

Use `data/test_source1.tsv`, `data/test_source2.tsv`, `data/test_source3.tsv`, and `data/candidate_pairs.tsv` together:

```text
Source 1 + Source 2 + Source 3 + candidate_pairs.tsv
                         ↓
             Join IDs to source records
                         ↓
       Feature engineering / similarity scores
                         ↓
                 Matching model
                         ↓
             matching_results.tsv
```

`candidate_pairs.tsv` contains IDs only. Join each candidate ID back to Source 2 or Source 3 to obtain business names, addresses, countries, and other supplied fields.

## 5. Training Data

- `train/train_source1.tsv`, `train/train_source2.tsv`, and `train/train_source3.tsv` contain the training entities and their business fields.
- `train/train_ground_truth.tsv` provides known Source 1 to Source 2/3 matches.

Use the ground truth to label candidate pairs for training and validation. Blocking recall was not finalized, so unmatched ground-truth pairs may be absent from the candidate set if the teammate regenerates training candidates.

## 6. Preprocessing

`preprocessing/normalize.py` is a portable copy of the project’s streaming normalization logic. It processes TSVs incrementally and writes seven columns: the four original fields, normalized name, normalized address, and suffix-stripped name.

Name normalization applies Unicode NFKC, lowercasing, converts `&` and `+` to “and”, replaces `/` with a space, removes punctuation, expands common company suffix abbreviations, and derives a suffix-stripped name using the suffix list in the script. Address normalization lowercases, converts `&` to “and”, removes `#`, expands common street/address abbreviations, and removes punctuation. The script does not add external business information.

## 7. Blocking / Candidate Generation

The final test candidate file was generated from the existing normalized test data using country-scoped exact lookup on three fields: normalized business name, suffix-stripped business name, and normalized address. The three result sets are unioned, deduplicated, and sorted. Matching Source 2 and Source 3 IDs are both eligible.

This final test run did **not** use BM25, lexical token ranking, character n-gram ranking, or approximate similarity. Those channels appear in other project code but were not used for this delivered test file. No all-pairs comparison was performed.

`blocking/generate_candidates.py` reproduces the delivered blocking method from the handoff’s normalized files and writes candidates incrementally. It builds a compact SQLite exact-lookup index if one is not already available.

## 8. Important Constraints

- Preserve the supplied raw data; do not modify it.
- Use only the supplied challenge data. External identity lookup and data augmentation are outside this handoff’s scope.
- The downstream matcher should evaluate candidate pairs listed in `data/candidate_pairs.tsv`.

## 9. File Map

| File | Purpose |
|---|---|
| `data/test_source1.tsv` | Test Source 1 entities |
| `data/test_source2.tsv` | Test Source 2 entities |
| `data/test_source3.tsv` | Test Source 3 entities |
| `data/candidate_pairs.tsv` | Final blocked test pairs, represented as Source 1 ID plus comma-separated candidate IDs |
| `train/train_source1.tsv` | Training Source 1 entities |
| `train/train_source2.tsv` | Training Source 2 entities |
| `train/train_source3.tsv` | Training Source 3 entities |
| `train/train_ground_truth.tsv` | Known training matches |
| `preprocessing/normalize.py` | Streaming normalization for the supplied raw TSVs |
| `blocking/generate_candidates.py` | Country-scoped exact name, suffix-stripped name, and address candidate generation |
| `requirements/requirements.txt` | Notes that only the Python standard library is required |
| `MANIFEST.txt` | File sizes and purposes for every handoff file |

## 10. Running the Code

Run these commands from the handoff folder with Python installed:

```powershell
python preprocessing\normalize.py --input-root . --output-dir work\normalized
python blocking\generate_candidates.py --normalized-dir work\normalized --index work\indexes\test_exact_retrieval.sqlite3 --output data\candidate_pairs.tsv
```

The first command creates normalized train and test files under `work/normalized`. The second indexes normalized test Source 2/3 and writes the candidate file. The scripts use Python’s standard library, including SQLite; no package installation is needed. These commands reproduce the exact blocking channels described above. They do not run the unfinished training recall evaluation.

## 11. Validation

Verified for `data/candidate_pairs.tsv`:

| Check | Result |
|---|---:|
| Test Source 1 records | 1,732,544 |
| Candidate rows | 1,732,544 |
| Total candidate pairs | 17,398,356 |
| Average candidates per Source 1 | 10.042086 |
| Candidate file size | 248,704,044 bytes |
| Exact schema | PASS |
| Duplicate Source 1 IDs | 0 |
| Rows with duplicate candidate IDs | 0 |
| Candidate IDs outside test Source 2/3 | 0 |
| Blank/NaN or malformed values | 0 |
| Row count matches test Source 1 | PASS |

Training blocking recall was not finalized. The challenge-provided validator was not run during the urgent test-output pass.

## 12. Next Stage

1. Load `data/candidate_pairs.tsv`.
2. Join candidate IDs to Source 2/3 records.
3. Generate pairwise matching features.
4. Train and validate the matching model using the training ground truth.
5. Apply the model to the test candidate pairs.
6. Generate `matching_results.tsv`.
