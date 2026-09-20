#!/usr/bin/env python3
"""Evaluate WER of adversarial audio using Step-Audio2 ASR."""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP_AUDIO_ROOT = Path(os.environ.get("STEP_AUDIO_ROOT", "/opt/Step-Audio2"))
DEFAULT_MODEL_DIR = Path(os.environ.get("STEP_AUDIO_MODEL", "/models/Step-Audio-2-mini"))
DEFAULT_ASR_PROMPT = "请记录下你所听到的语音内容。"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "data" / "phase3_narrative_wrapped_prompts_v0_2.jsonl"


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def normalize_for_wer(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_tokens(text):
    return normalize_for_wer(text).split()


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[n][m]


def compute_wer(reference, hypothesis):
    ref_words = word_tokens(reference)
    hyp_words = word_tokens(hypothesis)
    dist = edit_distance(ref_words, hyp_words)
    wer = dist / len(ref_words) if ref_words else 0.0
    return {
        "wer": wer,
        "ref_word_count": len(ref_words),
        "hyp_word_count": len(hyp_words),
        "distance": dist,
    }


def summarize_rows(rows):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    
    scopes = [("overall", "all", ok_rows)]
    categories = sorted({r.get("risk_category", "unknown") for r in ok_rows})
    for cat in categories:
        scopes.append(("category", cat, [r for r in ok_rows if r.get("risk_category") == cat]))

    summary = []
    for scope, cat, scoped in scopes:
        orig_wers = [r["original_wer"] for r in scoped]
        adv_wers = [r["adversarial_wer"] for r in scoped]
        wer_deltas = [adv - orig for orig, adv in zip(orig_wers, adv_wers)]
        
        summary.append({
            "scope": scope,
            "category": cat,
            "total": len(scoped),
            "avg_orig_wer": f"{statistics.mean(orig_wers):.6f}" if orig_wers else "0.000000",
            "avg_adv_wer": f"{statistics.mean(adv_wers):.6f}" if adv_wers else "0.000000",
            "avg_wer_delta": f"{statistics.mean(wer_deltas):.6f}" if wer_deltas else "0.000000",
            "median_orig_wer": f"{statistics.median(orig_wers):.6f}" if orig_wers else "0.000000",
            "median_adv_wer": f"{statistics.median(adv_wers):.6f}" if adv_wers else "0.000000",
        })
    return summary


def write_summary(path, rows):
    summary = summarize_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "category", "total", "avg_orig_wer", "avg_adv_wer", "avg_wer_delta", "median_orig_wer", "median_adv_wer"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def build_messages(wav_path, asr_prompt):
    return [
        {"role": "system", "content": asr_prompt},
        {"role": "human", "content": [{"type": "audio", "audio": str(wav_path)}]},
        {"role": "assistant", "content": None},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Directory containing original audio files")
    parser.add_argument("--adv-audio-dir", required=True, help="Directory containing adversarial audio files")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--step-audio-root", type=Path, default=DEFAULT_STEP_AUDIO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--asr-prompt", default=DEFAULT_ASR_PROMPT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"audio_dir={args.audio_dir}")
    print(f"adv_audio_dir={args.adv_audio_dir}")

    if args.overwrite and args.output.exists():
        args.output.unlink()

    prompt_rows = load_jsonl(args.prompt_file)
    prompt_map = {r["id"]: r for r in prompt_rows}
    print(f"Loaded {len(prompt_map)} prompts")

    adv_audio_files = sorted(glob.glob(os.path.join(args.adv_audio_dir, "*_adversarial.wav")))
    print(f"Found {len(adv_audio_files)} adversarial audio files")

    if not adv_audio_files:
        print("No adversarial audio files found!")
        return

    if args.limit:
        adv_audio_files = adv_audio_files[:args.limit]

    sys.path.insert(0, str(args.step_audio_root))
    from stepaudio2 import StepAudio2
    import torch

    print(f"Loading Step-Audio2 model...")
    model = StepAudio2(str(args.model_dir))
    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
    }

    all_rows = []

    for idx, adv_path in enumerate(adv_audio_files, 1):
        audio_name = os.path.basename(adv_path).replace("_adversarial.wav", "")
        original_audio_path = os.path.join(args.audio_dir, f"{audio_name}.wav")
        
        print(f"\n[{idx}/{len(adv_audio_files)}] {audio_name}", flush=True)
        started = time.time()

        if not os.path.exists(original_audio_path):
            print(f"  Original audio not found")
            continue

        if audio_name not in prompt_map:
            print(f"  Prompt not found")
            continue

        reference_text = prompt_map[audio_name].get("tts_text", "")
        risk_category = audio_name.split("_")[2] if len(audio_name.split("_")) > 2 else "unknown"

        messages = build_messages(original_audio_path, args.asr_prompt)
        _, orig_asr_text, _ = model(messages, **generation_kwargs)
        orig_asr_text = (orig_asr_text or "").strip()

        messages = build_messages(adv_path, args.asr_prompt)
        _, adv_asr_text, _ = model(messages, **generation_kwargs)
        adv_asr_text = (adv_asr_text or "").strip()

        orig_wer_info = compute_wer(reference_text, orig_asr_text)
        adv_wer_info = compute_wer(reference_text, adv_asr_text)

        result = {
            "id": audio_name,
            "risk_category": risk_category,
            "reference_text": reference_text[:200],
            "original_asr_text": orig_asr_text[:200],
            "adversarial_asr_text": adv_asr_text[:200],
            "original_wer": orig_wer_info["wer"],
            "adversarial_wer": adv_wer_info["wer"],
            "wer_delta": adv_wer_info["wer"] - orig_wer_info["wer"],
            "ref_word_count": orig_wer_info["ref_word_count"],
            "elapsed_sec": round(time.time() - started, 3),
            "status": "ok",
        }

        all_rows.append(result)
        append_jsonl(args.output, result)

        print(f"  Orig WER: {orig_wer_info['wer']:.4f}, Adv WER: {adv_wer_info['wer']:.4f}, Delta: {result['wer_delta']:.4f}")

    write_summary(args.summary, all_rows)
    
    overall = summarize_rows(all_rows)[0] if all_rows else {}
    print(f"\n=== WER Evaluation Complete ===")
    print(f"Total: {len(all_rows)}")
    print(f"Avg Orig WER: {overall.get('avg_orig_wer', 'N/A')}")
    print(f"Avg Adv WER: {overall.get('avg_adv_wer', 'N/A')}")
    print(f"Avg WER Delta: {overall.get('avg_wer_delta', 'N/A')}")


if __name__ == "__main__":
    main()