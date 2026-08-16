#!/usr/bin/env python3
"""Create Phase 3 narrative-wrapped prompts from the existing seed dataset."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "seed_prompts_en_v0_1_kimi_labeled.jsonl"
DEFAULT_WRAPPERS = PROJECT_ROOT / "data" / "phase3_narrative_wrappers_v0_1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "phase3_narrative_wrapped_prompts_v0_1.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "phase3_narrative_wrapped_prompts_v0_1_summary.csv"


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL at {}:{}: {}".format(path, line_no, exc)) from exc
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def load_wrappers(path):
    with path.open("r", encoding="utf-8") as f:
        wrappers = json.load(f)
    by_category = {}
    for wrapper in wrappers:
        category = wrapper.get("risk_category")
        if not category:
            raise ValueError("Wrapper missing risk_category: {}".format(wrapper))
        if category in by_category:
            raise ValueError("Duplicate wrapper for risk_category={}".format(category))
        template = wrapper.get("template", "")
        if "{harmful_request}" not in template:
            raise ValueError("Wrapper {} missing {{harmful_request}} placeholder".format(category))
        by_category[category] = wrapper
    return by_category


def build_record(record, wrapper):
    original_text = record["tts_text"]
    wrapped_text = wrapper["template"].format(harmful_request=original_text)
    source_id = record["id"]

    new_record = {
        "id": "phase3_{}".format(source_id),
        "source_id": source_id,
        "language": record.get("language", "en"),
        "risk_category": record["risk_category"],
        "intent_level": record.get("intent_level"),
        "tts_text": wrapped_text,
        "original_tts_text": original_text,
        "wrapped_tts_text": wrapped_text,
        "wrapper_id": wrapper["wrapper_id"],
        "wrapper_family": wrapper["wrapper_family"],
        "wrapper_template": wrapper["template"],
        "public_prompt": record.get("public_prompt"),
        "benign_rewrite": record.get("benign_rewrite"),
        "expected_safe_behavior": record.get("expected_safe_behavior"),
        "source": "phase3_narrative_wrapped_v0.1",
        "parent_source": record.get("source"),
        "release_tier": record.get("release_tier", "restricted"),
        "notes": (
            "Narrative-wrapped Phase 3 prompt for authorized audio safety evaluation; "
            "the harmful core request is preserved in original_tts_text for comparability."
        ),
        "needs_kimi_seed_recheck": True,
    }
    if "kimi_security" in record:
        new_record["original_kimi_security"] = record["kimi_security"]
    return new_record


def write_summary(path, records):
    counts = {}
    for record in records:
        key = (record["risk_category"], record["wrapper_family"], record["wrapper_id"])
        counts[key] = counts.get(key, 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("risk_category,wrapper_family,wrapper_id,count\n")
        for (category, family, wrapper_id), count in sorted(counts.items()):
            f.write("{},{},{},{}\n".format(category, family, wrapper_id, count))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--wrappers", type=Path, default=DEFAULT_WRAPPERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit("Output exists. Use --overwrite to replace: {}".format(args.output))
    if args.summary.exists() and not args.overwrite:
        raise SystemExit("Summary exists. Use --overwrite to replace: {}".format(args.summary))

    source_records = read_jsonl(args.input)
    wrappers = load_wrappers(args.wrappers)

    missing = sorted({r["risk_category"] for r in source_records} - set(wrappers))
    if missing:
        raise ValueError("Missing wrappers for categories: {}".format(", ".join(missing)))

    wrapped_records = [build_record(record, wrappers[record["risk_category"]]) for record in source_records]
    write_jsonl(args.output, wrapped_records)
    write_summary(args.summary, wrapped_records)

    print("input={}".format(args.input))
    print("wrappers={}".format(args.wrappers))
    print("output={}".format(args.output))
    print("summary={}".format(args.summary))
    print("source_records={}".format(len(source_records)))
    print("wrapped_records={}".format(len(wrapped_records)))
    print("categories={}".format(len({r["risk_category"] for r in wrapped_records})))


if __name__ == "__main__":
    main()
