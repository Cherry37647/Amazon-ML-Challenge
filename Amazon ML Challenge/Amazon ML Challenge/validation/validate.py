import csv
import os
import argparse


def read_ids(filepath, column_name):
    ids = set()

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if column_name not in reader.fieldnames:
            raise ValueError(
                f"{filepath} does not contain required column: {column_name}"
            )

        for row in reader:
            value = row[column_name].strip()

            if value:
                ids.add(value)

    return ids


def split_ids(value):
    if not value:
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def validate_matching_results(
    matching_file,
    test_source1_ids,
    valid_source23_ids
):
    errors = {
        "duplicate_source1": 0,
        "invalid_matches": 0,
        "self_matches": 0,
        "duplicate_matched_ids": 0,
        "missing_source1": 0,
        "extra_source1": 0
    }

    seen_source1 = set()
    matching_rows = 0

    with open(matching_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        expected_columns = {
            "source1_entity_id",
            "matched_entity_ids"
        }

        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(
                "matching_results.tsv must contain exactly these columns:\n"
                "source1_entity_id\tmatched_entity_ids"
            )

        for row in reader:
            matching_rows += 1

            source1_id = row["source1_entity_id"].strip()
            matched_ids = split_ids(row["matched_entity_ids"].strip())

            if not source1_id:
                continue

            if source1_id in seen_source1:
                errors["duplicate_source1"] += 1

            seen_source1.add(source1_id)

            # Check duplicate matched IDs
            if len(matched_ids) != len(set(matched_ids)):
                errors["duplicate_matched_ids"] += 1

            # Check every matched ID
            for matched_id in matched_ids:

                # Source 1 self-match
                if matched_id == source1_id:
                    errors["self_matches"] += 1

                # Must be a valid Source 2 or Source 3 test ID
                if matched_id not in valid_source23_ids:
                    errors["invalid_matches"] += 1

    # Missing Source 1 entities
    errors["missing_source1"] = len(test_source1_ids - seen_source1)

    # Extra Source 1 entities
    errors["extra_source1"] = len(seen_source1 - test_source1_ids)

    return matching_rows, errors, seen_source1


def validate_candidate_pairs(
    candidate_file,
    test_source1_ids,
    valid_source23_ids
):
    errors = {
        "duplicate_source1": 0,
        "invalid_candidate_ids": 0,
        "duplicate_candidate_ids": 0,
        "missing_source1": 0,
        "extra_source1": 0
    }

    seen_source1 = set()
    candidate_rows = 0

    with open(candidate_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        expected_columns = {
            "source1_entity_id",
            "candidate_entity_ids"
        }

        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(
                "candidate_pairs.tsv must contain exactly these columns:\n"
                "source1_entity_id\tcandidate_entity_ids"
            )

        for row in reader:
            candidate_rows += 1

            source1_id = row["source1_entity_id"].strip()
            candidate_ids = split_ids(row["candidate_entity_ids"].strip())

            if not source1_id:
                continue

            if source1_id in seen_source1:
                errors["duplicate_source1"] += 1

            seen_source1.add(source1_id)

            # Check duplicate candidate IDs
            if len(candidate_ids) != len(set(candidate_ids)):
                errors["duplicate_candidate_ids"] += 1

            # Check candidate IDs
            for candidate_id in candidate_ids:

                if candidate_id not in valid_source23_ids:
                    errors["invalid_candidate_ids"] += 1

    errors["missing_source1"] = len(test_source1_ids - seen_source1)
    errors["extra_source1"] = len(seen_source1 - test_source1_ids)

    return candidate_rows, errors, seen_source1


def validate_matches_are_candidates(
    matching_file,
    candidate_file
):
    """
    Checks that every final matched ID also exists in the
    candidate list for the same Source 1 entity.
    """

    matching_data = {}

    with open(matching_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            source1_id = row["source1_entity_id"].strip()
            matched_ids = set(
                split_ids(row["matched_entity_ids"].strip())
            )

            matching_data[source1_id] = matched_ids

    invalid_final_matches = 0

    with open(candidate_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            source1_id = row["source1_entity_id"].strip()

            if source1_id not in matching_data:
                continue

            candidate_ids = set(
                split_ids(row["candidate_entity_ids"].strip())
            )

            matched_ids = matching_data[source1_id]

            invalid_final_matches += len(
                matched_ids - candidate_ids
            )

            del matching_data[source1_id]

    # Any remaining Source 1 entities were missing from candidate file
    return invalid_final_matches


def print_result(name, value):
    print(f"{name}: {value}")


def main():

    parser = argparse.ArgumentParser(
        description="Validate Amazon ML Challenge entity resolution outputs."
    )

    parser.add_argument(
        "--matching",
        default="output/matching_results.tsv",
        help="Path to matching_results.tsv"
    )

    parser.add_argument(
        "--candidate",
        default="output/candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing test_source1.tsv, test_source2.tsv and test_source3.tsv"
    )

    args = parser.parse_args()

    matching_file = args.matching
    candidate_file = args.candidate

    test_source1_file = os.path.join(
        args.data_dir,
        "test_source1.tsv"
    )

    test_source2_file = os.path.join(
        args.data_dir,
        "test_source2.tsv"
    )

    test_source3_file = os.path.join(
        args.data_dir,
        "test_source3.tsv"
    )

    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION VALIDATION")
    print("=" * 60)
    print()

    # Check files exist
    required_files = [
        matching_file,
        candidate_file,
        test_source1_file,
        test_source2_file,
        test_source3_file
    ]

    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"ERROR: File not found -> {filepath}")
            return

    # ---------------------------------------------------------
    # Read valid test IDs
    # ---------------------------------------------------------

    print("Reading test entity IDs...")

    test_source1_ids = read_ids(
        test_source1_file,
        "entity_id"
    )

    test_source2_ids = read_ids(
        test_source2_file,
        "entity_id"
    )

    test_source3_ids = read_ids(
        test_source3_file,
        "entity_id"
    )

    valid_source23_ids = test_source2_ids | test_source3_ids

    print_result(
        "Total Source 1 entities",
        len(test_source1_ids)
    )

    print_result(
        "Total valid Source 2/3 IDs",
        len(valid_source23_ids)
    )

    print()

    # ---------------------------------------------------------
    # Validate matching_results.tsv
    # ---------------------------------------------------------

    print("Checking matching_results.tsv...")
    print()

    matching_rows, matching_errors, _ = validate_matching_results(
        matching_file,
        test_source1_ids,
        valid_source23_ids
    )

    print_result(
        "Matching result rows",
        matching_rows
    )

    print_result(
        "Duplicate Source 1 IDs",
        matching_errors["duplicate_source1"]
    )

    print_result(
        "Invalid candidate matches",
        matching_errors["invalid_matches"]
    )

    print_result(
        "Missing Source 1 IDs",
        matching_errors["missing_source1"]
    )

    print_result(
        "Extra Source 1 IDs",
        matching_errors["extra_source1"]
    )

    print_result(
        "Self matches",
        matching_errors["self_matches"]
    )

    print_result(
        "Duplicate matched IDs",
        matching_errors["duplicate_matched_ids"]
    )

    print()

    # ---------------------------------------------------------
    # Validate candidate_pairs.tsv
    # ---------------------------------------------------------

    print("Checking candidate_pairs.tsv...")
    print()

    candidate_rows, candidate_errors, _ = validate_candidate_pairs(
        candidate_file,
        test_source1_ids,
        valid_source23_ids
    )

    print_result(
        "Candidate rows",
        candidate_rows
    )

    print_result(
        "Duplicate Source 1 IDs",
        candidate_errors["duplicate_source1"]
    )

    print_result(
        "Invalid candidate IDs",
        candidate_errors["invalid_candidate_ids"]
    )

    print_result(
        "Duplicate candidate IDs",
        candidate_errors["duplicate_candidate_ids"]
    )

    print_result(
        "Missing Source 1 IDs",
        candidate_errors["missing_source1"]
    )

    print_result(
        "Extra Source 1 IDs",
        candidate_errors["extra_source1"]
    )

    print()

    # ---------------------------------------------------------
    # Check final matches are inside candidate set
    # ---------------------------------------------------------

    print("Checking final matches against candidate pairs...")

    invalid_final_matches = validate_matches_are_candidates(
        matching_file,
        candidate_file
    )

    print_result(
        "Final matches outside candidate set",
        invalid_final_matches
    )

    print()

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    all_errors = (
        sum(matching_errors.values())
        + sum(candidate_errors.values())
        + invalid_final_matches
    )

    print("=" * 60)

    if all_errors == 0:
        print("VALIDATION PASSED")
        print("All required checks completed successfully.")
    else:
        print("VALIDATION FAILED")
        print(f"Total issues found: {all_errors}")

    print("=" * 60)


if __name__ == "__main__":
    main()