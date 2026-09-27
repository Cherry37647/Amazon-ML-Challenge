

## 1. Project Overview

This project solves the Business Entity Resolution problem from the Amazon ML Challenge.

The objective is to identify which records from Source 2 and Source 3 refer to the same real-world business as each record in Source 1.

The solution follows a multi-stage entity resolution pipeline:

1. Data preprocessing and normalization
2. Candidate generation and blocking
3. Pairwise feature engineering
4. LightGBM binary classification
5. Validation-based threshold selection using macro F0.5
6. Final entity matching
7. Submission validation

## 2. Problem Definition

The challenge provides three independent business data sources:

- Source 1 — reference/deduplicated source
- Source 2 — candidate business records
- Source 3 — candidate business records

Each record contains:

- `entity_id`
- `business_name`
- `business_address`
- `country`

The task is to identify all Source 2 and Source 3 records that refer to the same real-world business as each Source 1 record.

A Source 1 entity may have:

- No matching records
- One matching record
- Multiple matching records

The final prediction contains one row for every Source 1 test entity.

---

## 3. Project Structure

Solution Pipeline

The complete solution follows this pipeline:

```text
Input Data
    |
    v
Preprocessing & Normalization
    |
    v
Candidate Generation / Blocking
    |
    v
Candidate Pairs
    |
    v
Feature Engineering
    |
    v
LightGBM Matching Model
    |
    v
Matching Probability
    |
    v
Validation-Based Threshold Selection
    |
    v
Final Matching Results
    |
    v
Submission Validation



## 4. Important Existing Data

```text
data/
├── candidate_pairs.tsv
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv

train/
├── train_source1.tsv
├── train_source2.tsv
├── train_source3.tsv
└── train_ground_truth.tsv
```

The existing `data/candidate_pairs.tsv` is the current fallback candidate file. It contains approximately 1.73M Source 1 rows and 17.4M candidate pairs.

**Do not regenerate it unless specifically required.**

## 5. Matching Code

The intended model pipeline is:

```text
matching/
├── features.py
├── train_model.py
├── predict.py
└── build_train_candidates.py
```

If the training candidate file does not exist, run:

```powershell
python matching\build_train_candidates.py
```

Expected output:

```text
work\train_candidate_pairs.tsv
```

Then:

```powershell
python matching\train_model.py
```

Finally:

```powershell
python matching\predict.py
```

Expected final prediction:

```text
output\matching_results.tsv
```

## 6. Feature Engineering

The pairwise features include:

### Business name
- normalized name similarity
- token-sort similarity
- token-set similarity
- Jaro-Winkler similarity
- Levenshtein similarity
- character/token overlap
- exact-name indicators
- suffix-stripped name similarity

### Address
- normalized address similarity
- token-based similarity
- Jaro-Winkler similarity
- Levenshtein similarity
- exact-address indicator
- address length differences

### Numeric/address information
- digit overlap
- PIN/postal-code matching
- extracted numeric information

### Other
- country match
- string length differences

## 7. Model

The intended matcher is a LightGBM binary classifier.

```text
1 = true match
0 = non-match
```

Training uses the provided training sources and ground truth.

## 8. Threshold Selection

The challenge uses macro F0.5, which weights precision more heavily than recall.

The threshold should therefore be selected using validation macro F0.5 rather than automatically assuming 0.50.

The selected threshold should be documented after training.

## 9. Final Prediction Format

The final file is:

```text
output/matching_results.tsv
```

with exactly:

```text
source1_entity_id
matched_entity_ids
```

There must be exactly one row for every test Source 1 entity.

Examples:

```text
S1-12345
S1-12346    S2-54321
S1-12347    S2-54321,S2-67890,S3-11111
```

Requirements:
- one row per test S1
- no duplicate S1 IDs
- no duplicate matched IDs
- only valid S2/S3 IDs
- empty match list for no-match cases
- final matches must belong to the candidate set used by the model

## 10. Candidate/Matching Consistency

The final matching IDs must be a subset of the candidate IDs available for that Source 1 entity.

Do not manually add external entity IDs.

## 11. Official Validation

Run the official validator supplied by the organizers after the final files are ready.

Expected command, subject to the exact organizer directory layout:

```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate data\candidate_pairs.tsv --test-dir data\test
```

If the organizer package uses `dataset/test` instead, use that official path.

Do not modify the official validator.

## 12. Final Checklist

### Files
- [ ] `output/matching_results.tsv`
- [ ] candidate file used by the final model
- [ ] source code
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `Documentation_template.md`

### Output
- [ ] exactly one row per test S1
- [ ] no duplicate S1 IDs
- [ ] no duplicate matched IDs
- [ ] no invalid S2/S3 IDs
- [ ] correct empty-match representation
- [ ] all final matches are from the candidate set

### Compliance
- [ ] no external business identity lookup
- [ ] no geocoding/API augmentation
- [ ] no external business databases
- [ ] only challenge-provided data used

## 13. Current Diagnostic Status

The original exact blocking approach was measured on training data and achieved:

```text
Pair-level blocking recall: 45.7088%
Entity-level complete recall: 11.6957%
```

This diagnostic showed that the original exact blocker does not recover all true training matches.

Because the submission deadline is close, the existing `data/candidate_pairs.tsv` is the fallback candidate set. Experimental candidate-generation scripts should not be run automatically during final packaging.

## 14. Validation and Compliance

Before submission, the final output is checked for:

Exactly one row for every Source 1 test entity
Duplicate Source 1 IDs
Duplicate matched entity IDs
Invalid Source 2 or Source 3 IDs
Source 1 self-matches
Matches that are not present in the candidate set
Correct TSV structure
Correct handling of entities with no predicted match

The local validation code is provided in:

validation/validate.py

The official challenge validator should also be run when the official submission package is available.

The final submission should pass all required validation checks before packaging.

## 15. Country Handling

The pipeline treats country as a general string attribute.

The implementation does not hardcode the training countries or restrict processing to only the countries present in the training data.

This allows previously unseen country labels in the test data to be processed without being discarded.

## 16. Reproducibility

The pipeline can be reproduced from the provided data and source code.

Step 1: Preprocessing
python preprocessing/normalize.py --input-root . --output-dir work/normalized
Step 2: Candidate Generation
python blocking/generate_candidates.py --normalized-dir work/normalized --index work/indexes/test_exact_retrieval.sqlite3 --output data/candidate_pairs.tsv

The existing final candidate file should be retained for the submitted run unless candidate generation is intentionally being reproduced.

Step 3: Training Candidate Generation

If a training candidate file is required:

python matching/build_train_candidates.py
Step 4: Model Training
python matching/train_model.py
Step 5: Prediction
python matching/predict.py

The final prediction is written to:

output/matching_results.tsv

## 17. Dependencies

The required Python dependencies are listed in:

requirements/requirements.txt

The matching pipeline uses LightGBM together with the libraries required for data processing and feature engineering.

## 18. Evaluation Metric

The challenge evaluates the solution using macro F0.5.

The F0.5 score is calculated using:

F0.5 = (1.25 × Precision × Recall) /
       (0.25 × Precision + Recall)

The score is calculated per Source 1 entity and then macro-averaged across the evaluation set.

F0.5 gives greater importance to precision than recall. Correctly identifying entities with no matching records is also part of the evaluation.

## 17. Fair-Play and Data Usage

The solution uses only the datasets supplied as part of the challenge.

No external data is used for entity resolution or data augmentation, including:

External business databases
Government business-registration databases
Commercial entity-resolution APIs
Geocoding APIs
Internet-based business lookup
External identity data

The solution is designed to perform entity resolution using only the challenge-provided data.
## 18. Team Responsibilities

### Palak
- Preprocessing/normalization
- Original blocking/candidate generation
- Existing `data/candidate_pairs.tsv`

### Geetanjali
- Pairwise feature engineering
- RapidFuzz/string similarity features
- Digit/PIN features
- LightGBM matching model
- Threshold calibration
- Prediction pipeline

### Suman
- Validation/compliance
- Final documentation
- Packaging
- Final ZIP submission
