#!/usr/bin/env python3
"""Use Step-Audio2 as an ASR model and compute WER for benchmark TTS audio.

The script reads a TTS manifest, transcribes each wav file with Step-Audio2, and
compares the ASR hypothesis against the original ``tts_text``. Outputs are:

- JSONL: one row per sample with transcript, edit counts, and WER.
- CSV: overall/category WER summary for quick reporting.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "matcha_seed_prompts_v0_1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "stepaudio2_asr_matcha_wer.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "stepaudio2_asr_matcha_wer_summary.csv"
DEFAULT_STEP_AUDIO_ROOT = Path("/hpc_stor03/sjtu_home/yi.yang/Step-Audio2")
DEFAULT_MODEL_DIR = Path(
    "/hpc_stor03/sjtu_home/yi.yang/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini"
)
DEFAULT_ASR_PROMPT = "请记录下你所听到的语音内容。"
PASSTHROUGH_FIELDS = [
    "source_id",
    "source_manifest",
    "original_wav_path",
    "perturbation_type",
    "condition_id",
    "speed",
    "volume_db",
    "snr_db",
    "front_silence_sec",
    "back_silence_sec",
    "duration_sec",
    "perturbation_config",
    "perturbation_stats",
]


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


def load_done_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and row.get("status") == "ok":
                done.add(row["id"])
    return done


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def resolve_wav_path(path):
    path = Path(path)
    if path.exists():
        return path

    text = str(path)
    prefix_map = {
        "/mnt/cloudstorfs/sjtu_home": "/hpc_stor03/sjtu_home",
        "/hpc_stor03/sjtu_home": "/mnt/cloudstorfs/sjtu_home",
    }
    for old_prefix, new_prefix in prefix_map.items():
        if text.startswith(old_prefix):
            candidate = Path(new_prefix + text[len(old_prefix) :])
            if candidate.exists():
                return candidate
    return path


def build_messages(wav_path, asr_prompt):
    return [
        {"role": "system", "content": asr_prompt},
        {"role": "human", "content": [{"type": "audio", "audio": str(wav_path)}]},
        {"role": "assistant", "content": None},
    ]


def normalize_for_wer(text):
    text = (text or "").lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_tokens(text):
    normalized = normalize_for_wer(text)
    return normalized.split() if normalized else []


def edit_counts(reference_words, hypothesis_words):
    """Return Levenshtein distance and S/D/I counts at word level."""
    n = len(reference_words)
    m = len(hypothesis_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                best = (dp[i - 1][j - 1], "M")
            else:
                best = (dp[i - 1][j - 1] + 1, "S")
            candidates = [
                best,
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], back[i][j] = min(candidates, key=lambda item: item[0])

    i, j = n, m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            if op == "S":
                substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            break

    distance = substitutions + deletions + insertions
    return {
        "edit_distance": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def compute_wer(reference, hypothesis):
    ref_words = word_tokens(reference)
    hyp_words = word_tokens(hypothesis)
    counts = edit_counts(ref_words, hyp_words)
    ref_word_count = len(ref_words)
    wer = counts["edit_distance"] / ref_word_count if ref_word_count else 0.0
    return {
        "wer": wer,
        "ref_word_count": ref_word_count,
        "hyp_word_count": len(hyp_words),
        "normalized_reference": " ".join(ref_words),
        "normalized_hypothesis": " ".join(hyp_words),
        **counts,
    }


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_rows(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    scopes = [("overall", "all", ok_rows)]
    categories = sorted({row.get("risk_category") or "unknown" for row in ok_rows})
    for category in categories:
        scopes.append(
            (
                "category",
                category,
                [row for row in ok_rows if (row.get("risk_category") or "unknown") == category],
            )
        )

    summary = []
    for scope, category, scoped_rows in scopes:
        wers = [float(row.get("wer", 0.0)) for row in scoped_rows]
        total_ref_words = sum(int(row.get("ref_word_count", 0)) for row in scoped_rows)
        total_edits = sum(int(row.get("edit_distance", 0)) for row in scoped_rows)
        weighted_wer = total_edits / total_ref_words if total_ref_words else 0.0
        summary.append(
            {
                "scope": scope,
                "category": category,
                "total": len(scoped_rows),
                "avg_wer": f"{statistics.mean(wers):.6f}" if wers else "0.000000",
                "weighted_wer": f"{weighted_wer:.6f}",
                "median_wer": f"{statistics.median(wers):.6f}" if wers else "0.000000",
                "p90_wer": f"{percentile(wers, 0.9):.6f}",
                "max_wer": f"{max(wers):.6f}" if wers else "0.000000",
                "total_ref_words": total_ref_words,
                "total_edits": total_edits,
            }
        )
    return summary


def write_summary(path, rows):
    summary = summarize_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "category",
        "total",
        "avg_wer",
        "weighted_wer",
        "median_wer",
        "p90_wer",
        "max_wer",
        "total_ref_words",
        "total_edits",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step-Audio2 ASR on TTS benchmark audio and compute WER."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--step-audio-root", type=Path, default=DEFAULT_STEP_AUDIO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asr-prompt", default=DEFAULT_ASR_PROMPT)
    parser.add_argument("--reference-field", default="tts_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--print-transcripts", action="store_true")
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Only check imports, model path, manifest, and CUDA availability.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=8.0,
        help="Minimum free GPU memory required before loading Step-Audio2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.step_audio_root.exists():
        raise FileNotFoundError(f"Step-Audio2 root not found: {args.step_audio_root}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Step-Audio2 model dir not found: {args.model_dir}")

    sys.path.insert(0, str(args.step_audio_root))
    from stepaudio2 import StepAudio2  # pylint: disable=import-error,import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    records = load_jsonl(args.manifest)

    print(f"manifest={args.manifest}")
    print(f"output={args.output}")
    print(f"summary={args.summary}")
    print(f"model_dir={args.model_dir}")
    print(f"step_audio_root={args.step_audio_root}")
    print(f"manifest_records={len(records)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        print(f"cuda_device_0_name={torch.cuda.get_device_name(0)}")
        print(f"cuda_device_0_free_gb={free_gb:.2f}")
        print(f"cuda_device_0_total_gb={total_gb:.2f}")

    if args.check_env_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Step-Audio2 calls .cuda() during model loading, "
            "so run this script on a GPU-visible node/session."
        )

    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Not enough free GPU memory before loading Step-Audio2: {free_gb:.2f} GiB free, "
            f"but --min-free-gb requires {args.min_free_gb:.2f} GiB."
        )

    if args.overwrite and args.output.exists():
        args.output.unlink()

    done_ids = load_done_ids(args.output)
    pending = [row for row in records if row.get("id") not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"total_records={len(records)} done={len(done_ids)} pending_this_run={len(pending)}")
    if not pending:
        existing_rows = load_jsonl(args.output) if args.output.exists() else []
        write_summary(args.summary, existing_rows)
        print(f"Nothing to do. Summary written to {args.summary}")
        return

    model = StepAudio2(str(args.model_dir))
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }

    for idx, row in enumerate(pending, 1):
        sample_id = row["id"]
        wav_path = resolve_wav_path(row["wav_path"])
        reference_text = row.get(args.reference_field) or ""
        print(f"[{idx}/{len(pending)}] ASR {sample_id}", flush=True)
        started = time.time()

        result = {
            "id": sample_id,
            "risk_category": row.get("risk_category"),
            "tts_model": row.get("tts_model"),
            "wav_path": str(wav_path),
            "original_wav_path": row.get("wav_path"),
            "reference_field": args.reference_field,
            "reference_text": reference_text,
            "stepaudio2_model_dir": str(args.model_dir),
            "asr_prompt": args.asr_prompt,
            "generation": generation_kwargs,
        }
        for field in PASSTHROUGH_FIELDS:
            if field in row:
                result[field] = row[field]

        if not wav_path.exists():
            result.update(
                {
                    "status": "error",
                    "asr_text": "",
                    "wer": None,
                    "error": f"wav_path not found: {wav_path}",
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            append_jsonl(args.output, result)
            print(f"[{idx}/{len(pending)}] error {sample_id}: wav_path missing", flush=True)
            continue

        try:
            messages = build_messages(wav_path, args.asr_prompt)
            _, asr_text, _ = model(messages, **generation_kwargs)
            asr_text = (asr_text or "").strip()
            wer_info = compute_wer(reference_text, asr_text)
            result.update(
                {
                    "status": "ok",
                    "asr_text": asr_text,
                    "error": "",
                    "elapsed_sec": round(time.time() - started, 3),
                    **wer_info,
                }
            )
            append_jsonl(args.output, result)
            if args.print_transcripts:
                print(f"{sample_id}: WER={wer_info['wer']:.4f} ASR={asr_text}", flush=True)
        except Exception as exc:  # Keep long batch jobs resumable.
            result.update(
                {
                    "status": "error",
                    "asr_text": "",
                    "wer": None,
                    "error": repr(exc),
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            append_jsonl(args.output, result)
            print(f"[{idx}/{len(pending)}] error {sample_id}: {exc}", flush=True)

    all_rows = load_jsonl(args.output)
    write_summary(args.summary, all_rows)
    ok_count = sum(1 for row in all_rows if row.get("status") == "ok")
    error_count = sum(1 for row in all_rows if row.get("status") == "error")
    overall = summarize_rows(all_rows)[0] if ok_count else {}
    print(f"summary_written={args.summary}")
    print(
        "done "
        f"rows={len(all_rows)} ok={ok_count} error={error_count} "
        f"weighted_wer={overall.get('weighted_wer', '0.000000')}"
    )


if __name__ == "__main__":
    main()
