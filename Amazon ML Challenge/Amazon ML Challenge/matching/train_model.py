from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from features import (
    FEATURE_NAMES,
    make_features,
)


def load_records(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    ).fillna("")

    return {
        row["entity_id"]: row.to_dict()
        for _, row in df.iterrows()
    }


def load_ground_truth(path):

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    ).fillna("")

    truth = {}

    for _, row in gt.iterrows():

        truth[
            row["source1_entity_id"]
        ] = {
            x.strip()
            for x in row[
                "matched_entity_ids"
            ].split(",")
            if x.strip()
        }

    return truth


def candidate_rows(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    ).fillna("")

    for _, row in df.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        candidates = [
            x.strip()
            for x in row[
                "candidate_entity_ids"
            ].split(",")
            if x.strip()
        ]

        for candidate_id in candidates:

            yield (
                s1_id,
                candidate_id
            )


def make_dataset(
    candidate_path,
    s1_lookup,
    s23_lookup,
    truth,
    max_negatives_per_positive=10
):

    rows = []

    positives_by_s1 = {}

    # First pass: collect positives.
    for s1_id, candidate_id in candidate_rows(
        candidate_path
    ):

        label = int(
            candidate_id
            in truth.get(
                s1_id,
                set()
            )
        )

        if label == 1:

            positives_by_s1.setdefault(
                s1_id,
                []
            ).append(
                candidate_id
            )

    # Second pass: construct balanced-ish dataset.
    negative_count = {}

    for s1_id, candidate_id in candidate_rows(
        candidate_path
    ):

        s1 = s1_lookup.get(s1_id)

        candidate = s23_lookup.get(
            candidate_id
        )

        if s1 is None or candidate is None:
            continue

        label = int(
            candidate_id
            in truth.get(
                s1_id,
                set()
            )
        )

        if label == 0:

            pos_count = max(
                len(
                    positives_by_s1.get(
                        s1_id,
                        []
                    )
                ),
                1
            )

            limit = (
                pos_count
                * max_negatives_per_positive
            )

            current = negative_count.get(
                s1_id,
                0
            )

            if current >= limit:
                continue

            negative_count[
                s1_id
            ] = current + 1

        features = make_features(
            s1,
            candidate
        )

        features[
            "source1_entity_id"
        ] = s1_id

        features[
            "candidate_entity_id"
        ] = candidate_id

        features[
            "label"
        ] = label

        rows.append(features)

    return pd.DataFrame(rows)


def macro_f05(
    s1_ids,
    truth,
    predictions
):

    scores = []

    for s1_id in s1_ids:

        actual = set(
            truth.get(
                s1_id,
                set()
            )
        )

        predicted = set(
            predictions.get(
                s1_id,
                set()
            )
        )

        # Correct singleton.
        if not actual and not predicted:

            scores.append(1.0)
            continue

        # False merge on singleton.
        if not actual and predicted:

            scores.append(0.0)
            continue

        true_positive = len(
            actual & predicted
        )

        precision = (
            true_positive
            / len(predicted)
            if predicted
            else 0.0
        )

        recall = (
            true_positive
            / len(actual)
            if actual
            else 0.0
        )

        denominator = (
            0.25 * precision
            + recall
        )

        if denominator == 0:

            scores.append(0.0)

        else:

            scores.append(
                (
                    1.25
                    * precision
                    * recall
                )
                / denominator
            )

    return float(
        np.mean(scores)
    )


def predictions_from_scores(
    pairs,
    probabilities,
    threshold
):

    predictions = {}

    selected = (
        pairs[
            probabilities
            >= threshold
        ]
    )

    for _, row in selected.iterrows():

        predictions.setdefault(
            row[
                "source1_entity_id"
            ],
            set()
        ).add(
            row[
                "candidate_entity_id"
            ]
        )

    return predictions


def find_best_threshold(
    model,
    validation,
    feature_columns,
    truth,
    validation_s1_ids
):

    X = validation[
        feature_columns
    ]

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    best_threshold = 0.50
    best_score = -1.0

    for threshold in np.arange(
        0.50,
        0.991,
        0.01
    ):

        predictions = (
            predictions_from_scores(
                validation,
                probabilities,
                threshold
            )
        )

        score = macro_f05(
            validation_s1_ids,
            truth,
            predictions
        )

        if score > best_score:

            best_score = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_score
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        default="train/train_source1.tsv"
    )

    parser.add_argument(
        "--s2",
        default="train/train_source2.tsv"
    )

    parser.add_argument(
        "--s3",
        default="train/train_source3.tsv"
    )

    parser.add_argument(
        "--ground-truth",
        default="train/train_ground_truth.tsv"
    )

    parser.add_argument(
        "--candidates",
        default="work/train_candidate_pairs.tsv"
    )

    parser.add_argument(
        "--model-output",
        default="model/lightgbm_entity_match.pkl"
    )

    parser.add_argument(
        "--threshold-output",
        default="model/threshold.json"
    )

    args = parser.parse_args()

    print("Loading training records...")

    s1_lookup = load_records(
        args.s1
    )

    s2_lookup = load_records(
        args.s2
    )

    s3_lookup = load_records(
        args.s3
    )

    s23_lookup = {
        **s2_lookup,
        **s3_lookup
    }

    truth = load_ground_truth(
        args.ground_truth
    )

    print(
        f"S1 records: {len(s1_lookup):,}"
    )

    print(
        f"S2/S3 records: {len(s23_lookup):,}"
    )

    print("Building training features...")

    dataset = make_dataset(
        args.candidates,
        s1_lookup,
        s23_lookup,
        truth
    )

    print(
        f"Training pairs: {len(dataset):,}"
    )

    print(
        f"Positive pairs: "
        f"{dataset['label'].sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(dataset['label'] == 0).sum():,}"
    )

    # ---------------------------------------------
    # Source-1-level split
    # ---------------------------------------------

    unique_s1 = dataset[
        "source1_entity_id"
    ].unique()

    train_ids, validation_ids = (
        train_test_split(
            unique_s1,
            test_size=0.20,
            random_state=42
        )
    )

    train_ids = set(train_ids)
    validation_ids = set(
        validation_ids
    )

    train = dataset[
        dataset[
            "source1_entity_id"
        ].isin(train_ids)
    ].copy()

    validation = dataset[
        dataset[
            "source1_entity_id"
        ].isin(validation_ids)
    ].copy()

    X_train = train[
        FEATURE_NAMES
    ]

    y_train = train[
        "label"
    ].astype(int)

    X_validation = validation[
        FEATURE_NAMES
    ]

    y_validation = validation[
        "label"
    ].astype(int)

    print("Training LightGBM...")

    model = LGBMClassifier(

        objective="binary",

        n_estimators=500,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=30,

        subsample=0.9,

        colsample_bytree=0.9,

        reg_alpha=0.1,

        reg_lambda=2.0,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1,

        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    if len(
        np.unique(y_validation)
    ) > 1:

        auc = roc_auc_score(
            y_validation,
            validation_probability
        )

        print(
            f"Validation ROC-AUC: {auc:.5f}"
        )

    threshold, f05 = (
        find_best_threshold(
            model,
            validation,
            FEATURE_NAMES,
            truth,
            validation_ids
        )
    )

    print(
        f"Best threshold: {threshold:.2f}"
    )

    print(
        f"Validation macro F0.5: {f05:.5f}"
    )

    # ---------------------------------------------
    # Save model
    # ---------------------------------------------

    model_path = Path(
        args.model_output
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with model_path.open(
        "wb"
    ) as f:

        pickle.dump(
            {
                "model": model,
                "feature_names":
                    FEATURE_NAMES
            },
            f
        )

    threshold_path = Path(
        args.threshold_output
    )

    threshold_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    threshold_path.write_text(
        json.dumps(
            {
                "threshold":
                    threshold,
                "validation_macro_f05":
                    f05,
                "feature_count":
                    len(FEATURE_NAMES)
            },
            indent=2
        )
    )

    print(
        f"Model saved to {model_path}"
    )

    print(
        f"Threshold saved to {threshold_path}"
    )


if __name__ == "__main__":
    main()