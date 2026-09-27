from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def read_tsv(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        yield from csv.DictReader(
            f,
            delimiter="\t"
        )


def load_ground_truth(path: Path):

    truth = {}

    for row in read_tsv(path):

        s1 = (row.get("source1_entity_id") or "").strip()

        ids = {
            x.strip()
            for x in
            (row.get("matched_entity_ids") or "").split(",")
            if x.strip()
        }

        truth[s1] = ids

    return truth


def build_index(
    normalized_dir: Path,
    db_path: Path
):

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if db_path.exists():
        print(
            f"Using existing SQLite index: {db_path}"
        )
        return

    conn = sqlite3.connect(db_path)

    conn.execute(
        "PRAGMA journal_mode=OFF"
    )

    conn.execute(
        "PRAGMA synchronous=OFF"
    )

    conn.execute(
        "PRAGMA temp_store=FILE"
    )

    conn.execute(
        """
        CREATE TABLE post(
            kind TEXT,
            country TEXT,
            term TEXT,
            eid TEXT,
            PRIMARY KEY(
                kind,
                country,
                term,
                eid
            )
        ) WITHOUT ROWID
        """
    )

    batch = []

    for source in (
        "source2",
        "source3"
    ):

        file_path = (
            normalized_dir
            / f"normalized_train_{source}.tsv"
        )

        for row in read_tsv(file_path):

            eid = (
                row.get("entity_id")
                or ""
            ).strip()

            country = (
                row.get("country")
                or ""
            ).strip()

            if not eid or not country:
                continue

            for kind, column in (
                (
                    "n",
                    "business_name_normalized"
                ),
                (
                    "s",
                    "business_name_suffix_stripped"
                ),
                (
                    "a",
                    "business_address_normalized"
                ),
            ):

                term = (
                    row.get(column)
                    or ""
                ).strip()

                if term:

                    batch.append(
                        (
                            kind,
                            country,
                            term,
                            eid
                        )
                    )

            if len(batch) >= 50000:

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO post
                    VALUES (?, ?, ?, ?)
                    """,
                    batch
                )

                conn.commit()

                batch.clear()

        print(
            f"Indexed training {source}"
        )

    if batch:

        conn.executemany(
            """
            INSERT OR IGNORE INTO post
            VALUES (?, ?, ?, ?)
            """,
            batch
        )

        conn.commit()

    conn.close()


def generate_candidates(
    normalized_dir: Path,
    db_path: Path,
    ground_truth_path: Path,
    output_path: Path
):

    truth = load_ground_truth(
        ground_truth_path
    )

    conn = sqlite3.connect(
        db_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".partial"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids"
            ]
        )

        for i, row in enumerate(
            read_tsv(
                normalized_dir
                / "normalized_train_source1.tsv"
            ),
            1
        ):

            s1_id = (
                row.get("entity_id")
                or ""
            ).strip()

            country = (
                row.get("country")
                or ""
            ).strip()

            candidates = set()

            for kind, column in (
                (
                    "n",
                    "business_name_normalized"
                ),
                (
                    "s",
                    "business_name_suffix_stripped"
                ),
                (
                    "a",
                    "business_address_normalized"
                ),
            ):

                term = (
                    row.get(column)
                    or ""
                ).strip()

                if not country or not term:
                    continue

                cursor = conn.execute(
                    """
                    SELECT eid
                    FROM post
                    WHERE kind=?
                      AND country=?
                      AND term=?
                    """,
                    (
                        kind,
                        country,
                        term
                    )
                )

                candidates.update(
                    x[0]
                    for x in cursor
                )

            # Make sure known positives are available
            # for supervised learning.
            candidates.update(
                truth.get(
                    s1_id,
                    set()
                )
            )

            writer.writerow(
                [
                    s1_id,
                    ",".join(
                        sorted(candidates)
                    )
                ]
            )

            if i % 100000 == 0:

                print(
                    f"Processed {i:,} training S1 records"
                )

    conn.close()

    temp_path.replace(
        output_path
    )

    print(
        f"Wrote {output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path(
            "work/normalized"
        )
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "work/indexes/train_exact_retrieval.sqlite3"
        )
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "train/train_ground_truth.tsv"
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "work/train_candidate_pairs.tsv"
        )
    )

    args = parser.parse_args()

    build_index(
        args.normalized_dir,
        args.index
    )

    generate_candidates(
        args.normalized_dir,
        args.index,
        args.ground_truth,
        args.output
    )


if __name__ == "__main__":
    main()