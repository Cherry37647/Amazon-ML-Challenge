"""Normalize the supplied challenge TSVs into chunk-written handoff artifacts."""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

FIELDS = ["entity_id", "business_name", "business_address", "country",
          "business_name_normalized", "business_address_normalized",
          "business_name_suffix_stripped"]


def normalize_unicode(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def collapse_spaces(value):
    return re.sub(r"\s+", " ", value).strip()


def normalize_business_name(raw):
    text = normalize_unicode(raw).lower().replace("&", " and ").replace("+", " and ").replace("/", " ")
    text = collapse_spaces(re.sub(r"[^\w\s]", " ", text))
    stripped = text
    for suffix in ["private limited", "limited company", "private company", "pvt ltd", "llc", "corporation", "incorporated", "company", "limited", "private", "pvt", "ltd", "corp", "inc", "co"]:
        if text.endswith(suffix):
            stripped = text[:-len(suffix)].strip()
            break
    normalized = re.sub(r"\bpvt\.?\b", " private ", text)
    normalized = re.sub(r"\bltd\.?\b", " limited ", normalized)
    normalized = re.sub(r"\bcorp\.?\b", " corporation ", normalized)
    normalized = re.sub(r"\binc\.?\b", " incorporated ", normalized)
    normalized = re.sub(r"\bco\.?\b", " company ", normalized)
    return collapse_spaces(normalized), collapse_spaces(stripped)


def normalize_address(raw):
    text = normalize_unicode(raw).lower().replace("&", " and ").replace("#", " ")
    text = re.sub(r"[^\w\s,.-]", " ", text)
    for abbr, repl in [(" st ", " street "), (" rd ", " road "), (" ave ", " avenue "), (" blvd ", " boulevard "), (" ln ", " lane "), (" dr ", " drive "), (" hwy ", " highway "), (" ct ", " court "), (" apt ", " apartment "), (" unit ", " unit "), (" fl ", " floor ")]:
        text = re.sub(rf"\b{re.escape(abbr.strip())}\b", repl.strip(), text)
    return collapse_spaces(re.sub(r"[,.-]", " ", text))


def normalize_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace", newline="") as fin, target.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        next(reader, None)
        writer = csv.DictWriter(fout, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        batch = []
        for row in reader:
            if len(row) < 4 or row[0] == "entity_id":
                continue
            eid, name, address, country = row[:4]
            name_norm, suffix = normalize_business_name(name)
            batch.append({"entity_id": eid.strip(), "business_name": name.strip(),
                          "business_address": address.strip(), "country": country.strip(),
                          "business_name_normalized": name_norm,
                          "business_address_normalized": normalize_address(address),
                          "business_name_suffix_stripped": suffix})
            if len(batch) == 20000:
                writer.writerows(batch)
                batch.clear()
        if batch:
            writer.writerows(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("."), help="Handoff root containing train/ and data/")
    ap.add_argument("--output-dir", type=Path, default=Path("work/normalized"))
    args = ap.parse_args()
    for split, folder, prefix in (("train", "train", "train"), ("test", "data", "test")):
        for source in ("source1", "source2", "source3"):
            raw = args.input_root / folder / f"{prefix}_{source}.tsv"
            out = args.output_dir / f"normalized_{split}_{source}.tsv"
            normalize_file(raw, out)
            print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
