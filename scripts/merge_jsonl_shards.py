#!/usr/bin/env python3
"""Merge JSONL shard outputs and optionally validate against a manifest."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path, rows, expected_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row.get("status", "unknown") for row in rows)
    tts_counts = Counter(row.get("tts_model", "unknown") for row in rows)
    condition_counts = Counter(row.get("condition_id", "unknown") for row in rows)
    type_counts = Counter(row.get("perturbation_type", "unknown") for row in rows)
    present_ids = {row.get("id") for row in rows}
    missing_ids = sorted(expected_ids - present_ids) if expected_ids else []

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["overall", "rows", len(rows)])
        writer.writerow(["overall", "unique_ids", len(present_ids)])
        writer.writerow(["overall", "expected_ids", len(expected_ids) if expected_ids else ""])
        writer.writerow(["overall", "missing_ids", len(missing_ids)])
        for key, value in sorted(status_counts.items()):
            writer.writerow(["status", key, value])
        for key, value in sorted(tts_counts.items()):
            writer.writerow(["tts_model", key, value])
        for key, value in sorted(condition_counts.items()):
            writer.writerow(["condition_id", key, value])
        for key, value in sorted(type_counts.items()):
            writer.writerow(["perturbation_type", key, value])
        for missing_id in missing_ids[:100]:
            writer.writerow(["missing_id", missing_id, ""])


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSONL shard files.")
    parser.add_argument("--input-glob", required=True, help="Glob pattern for shard files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest for expected id validation.")
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--prefer-ok",
        action="store_true",
        help="If duplicate ids exist, prefer a row with status=ok over a non-ok row.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    shard_paths = sorted(Path().glob(args.input_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched: {args.input_glob}")

    expected_ids = set()
    if args.manifest:
        expected_ids = {row[args.id_field] for row in load_jsonl(args.manifest)}

    rows_by_id = {}
    duplicates = 0
    raw_rows = 0
    for path in shard_paths:
        for row in load_jsonl(path):
            raw_rows += 1
            row_id = row.get(args.id_field)
            if not row_id:
                continue
            if row_id in rows_by_id:
                duplicates += 1
                if args.prefer_ok and rows_by_id[row_id].get("status") != "ok" and row.get("status") == "ok":
                    rows_by_id[row_id] = row
            else:
                rows_by_id[row_id] = row

    rows = [rows_by_id[row_id] for row_id in sorted(rows_by_id)]
    write_jsonl(args.output, rows)
    if args.summary:
        write_summary(args.summary, rows, expected_ids)

    print(f"matched_shards={len(shard_paths)}")
    print(f"raw_rows={raw_rows}")
    print(f"merged_rows={len(rows)}")
    print(f"duplicates={duplicates}")
    if expected_ids:
        print(f"expected_ids={len(expected_ids)}")
        print(f"missing_ids={len(expected_ids - set(rows_by_id))}")
    print(f"output={args.output}")
    if args.summary:
        print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
