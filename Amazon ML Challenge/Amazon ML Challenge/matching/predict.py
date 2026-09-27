from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import pandas as pd

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


def load_model(path):

    with open(
        path,
        "rb"
    ) as f:

        bundle = pickle.load(f)

    return (
        bundle["model"],
        bundle["feature_names"]
    )


def load_threshold(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return float(
        data["threshold"]
    )


def process_candidates(
    candidate_file,
    s1_lookup,
    s23_lookup,
    model,
    feature_names,
    threshold,
    matching_output,
    probability_output,
):

    matching_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    probability_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Temporary files make the output safer if the
    # process is interrupted.
    matching_tmp = matching_output.with_suffix(
        ".tmp.tsv"
    )

    probability_tmp = probability_output.with_suffix(
        ".tmp.tsv"
    )

    with (
        candidate_file.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as fin,

        matching_tmp.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as match_file,

        probability_tmp.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as probability_file
    ):

        reader = csv.DictReader(
            fin,
            delimiter="\t"
        )

        match_writer = csv.writer(
            match_file,
            delimiter="\t"
        )

        probability_writer = csv.writer(
            probability_file,
            delimiter="\t"
        )

        match_writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids"
            ]
        )

        probability_writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_id",
                "match_probability"
            ]
        )

        current_s1 = None
        current_predictions = []

        feature_rows = []
        id_rows = []

        processed = 0

        def flush_batch():

            nonlocal feature_rows
            nonlocal id_rows
            nonlocal current_s1
            nonlocal current_predictions

            if not feature_rows:
                return

            X = pd.DataFrame(
                feature_rows
            )[
                feature_names
            ]

            probabilities = (
                model.predict_proba(X)[:, 1]
            )

            for (
                (sid, cid),
                probability
            ) in zip(
                id_rows,
                probabilities
            ):

                probability_writer.writerow(
                    [
                        sid,
                        cid,
                        f"{probability:.8f}"
                    ]
                )

                if probability >= threshold:

                    current_predictions.append(
                        cid
                    )

            feature_rows.clear()
            id_rows.clear()

        for row in reader:

            sid = (
                row.get(
                    "source1_entity_id"
                )
                or ""
            ).strip()

            candidate_ids = [
                x.strip()
                for x in (
                    row.get(
                        "candidate_entity_ids"
                    )
                    or ""
                ).split(",")
                if x.strip()
            ]

            # The candidate file is ordered by S1,
            # so when S1 changes we write its result.
            if (
                current_s1 is not None
                and sid != current_s1
            ):

                flush_batch()

                match_writer.writerow(
                    [
                        current_s1,
                        ",".join(
                            sorted(
                                set(
                                    current_predictions
                                )
                            )
                        )
                    ]
                )

                current_predictions = []

            current_s1 = sid

            s1 = s1_lookup.get(
                sid
            )

            if s1 is None:
                continue

            for candidate_id in candidate_ids:

                candidate = s23_lookup.get(
                    candidate_id
                )

                if candidate is None:
                    continue

                feature_rows.append(
                    make_features(
                        s1,
                        candidate
                    )
                )

                id_rows.append(
                    (
                        sid,
                        candidate_id
                    )
                )

                processed += 1

                # Keep memory bounded.
                if len(feature_rows) >= 10000:

                    flush_batch()

                    if (
                        processed
                        % 100000
                        < 10000
                    ):

                        print(
                            f"Processed "
                            f"{processed:,} candidate pairs",
                            flush=True
                        )

        # Final S1.
        flush_batch()

        if current_s1 is not None:

            match_writer.writerow(
                [
                    current_s1,
                    ",".join(
                        sorted(
                            set(
                                current_predictions
                            )
                        )
                    )
                ]
            )

    matching_tmp.replace(
        matching_output
    )

    probability_tmp.replace(
        probability_output
    )

    print(
        f"Processed {processed:,} candidate pairs"
    )

    print(
        f"Wrote {matching_output}"
    )

    print(
        f"Wrote {probability_output}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1",
        default="data/test_source1.tsv"
    )

    parser.add_argument(
        "--s2",
        default="data/test_source2.tsv"
    )

    parser.add_argument(
        "--s3",
        default="data/test_source3.tsv"
    )

    parser.add_argument(
        "--candidates",
        default="data/candidate_pairs.tsv"
    )

    parser.add_argument(
        "--model",
        default="model/lightgbm_entity_match.pkl"
    )

    parser.add_argument(
        "--threshold",
        default="model/threshold.json"
    )

    parser.add_argument(
        "--matching-output",
        default="output/matching_results.tsv"
    )

    parser.add_argument(
        "--probability-output",
        default="output/candidate_match_probabilities.tsv"
    )

    args = parser.parse_args()

    print("Loading test records...")

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

    print(
        f"S1 records: "
        f"{len(s1_lookup):,}"
    )

    print(
        f"S2/S3 records: "
        f"{len(s23_lookup):,}"
    )

    model, feature_names = (
        load_model(
            args.model
        )
    )

    threshold = load_threshold(
        args.threshold
    )

    print(
        f"Using threshold: "
        f"{threshold:.2f}"
    )

    process_candidates(
        Path(args.candidates),
        s1_lookup,
        s23_lookup,
        model,
        feature_names,
        threshold,
        Path(
            args.matching_output
        ),
        Path(
            args.probability_output
        ),
    )


if __name__ == "__main__":
    main()