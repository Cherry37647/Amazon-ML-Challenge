"""Generate test candidates using country-scoped exact normalized fields."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


def iter_rows(path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def build_index(normalized_dir, db):
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        c = sqlite3.connect(db)
        try:
            c.execute("SELECT eid FROM post LIMIT 1").fetchone()
            c.close()
            print(f"Reusing {db}", flush=True)
            return
        except sqlite3.DatabaseError:
            c.close()
            raise RuntimeError(f"Existing index is unreadable; preserving it: {db}")
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=OFF")
    c.execute("PRAGMA synchronous=OFF")
    c.execute("PRAGMA temp_store=FILE")
    c.execute("CREATE TABLE post(kind TEXT,country TEXT,term TEXT,eid TEXT,PRIMARY KEY(kind,country,term,eid)) WITHOUT ROWID")
    batch = []
    for source in ("source2", "source3"):
        for row in iter_rows(normalized_dir / f"normalized_test_{source}.tsv"):
            eid, country = (row.get("entity_id") or "").strip(), (row.get("country") or "").strip()
            if not eid or not country:
                continue
            for kind, column in (("n", "business_name_normalized"), ("s", "business_name_suffix_stripped"), ("a", "business_address_normalized")):
                term = (row.get(column) or "").strip()
                if term:
                    batch.append((kind, country, term, eid))
            if len(batch) >= 50000:
                c.executemany("INSERT OR IGNORE INTO post VALUES(?,?,?,?)", batch)
                c.commit()
                batch.clear()
        print(f"Indexed test {source}", flush=True)
    if batch:
        c.executemany("INSERT OR IGNORE INTO post VALUES(?,?,?,?)", batch)
        c.commit()
    c.close()


def generate(normalized_dir, db, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    c = sqlite3.connect(db)
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for i, row in enumerate(iter_rows(normalized_dir / "normalized_test_source1.tsv"), 1):
            eid, country = (row.get("entity_id") or "").strip(), (row.get("country") or "").strip()
            candidates = set()
            for kind, column in (("n", "business_name_normalized"), ("s", "business_name_suffix_stripped"), ("a", "business_address_normalized")):
                term = (row.get(column) or "").strip()
                if country and term:
                    candidates.update(r[0] for r in c.execute("SELECT eid FROM post WHERE kind=? AND country=? AND term=?", (kind, country, term)))
            writer.writerow([eid, ",".join(sorted(candidates))])
            if i % 100000 == 0:
                print(f"Generated {i:,} records", flush=True)
    c.close()
    temp.replace(output)
    print(f"Wrote {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", type=Path, default=Path("work/normalized"))
    ap.add_argument("--index", type=Path, default=Path("work/indexes/test_exact_retrieval.sqlite3"))
    ap.add_argument("--output", type=Path, default=Path("data/candidate_pairs.tsv"))
    args = ap.parse_args()
    build_index(args.normalized_dir, args.index)
    generate(args.normalized_dir, args.index, args.output)


if __name__ == "__main__":
    main()
