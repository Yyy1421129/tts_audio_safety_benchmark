#!/usr/bin/env python3
"exec" "python3" "$0" "$@"
"""Batch synthesize seed-prompt JSONL records with Matcha-TTS.

This wrapper adapts the benchmark JSONL format to Matcha-TTS's CLI, then
renames generated wav files to stable sample ids and writes a manifest for
downstream Step-Audio2 evaluation.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "seed_prompts_en_v0_1_kimi_labeled.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "audio" / "matcha_seed_prompts_v0_1"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "matcha_seed_prompts_v0_1.jsonl"
DEFAULT_MATCHA_ROOT = Path("/hpc_stor03/sjtu_home/yi.yang/Matcha-TTS")


def read_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_completed_ids(manifest_path):
    completed = set()
    if not manifest_path.exists():
        return completed
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and row.get("wav_path") and Path(row["wav_path"]).exists():
                completed.add(row["id"])
    return completed


def append_manifest(manifest_path, row):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_records(records, args):
    if args.only_attack_definition_met:
        records = [
            r for r in records
            if r.get("kimi_security", {}).get("attack_definition_met") is True
        ]
    completed = set()
    if args.resume and not args.overwrite:
        completed = load_completed_ids(args.manifest)
    selected = []
    for record in records:
        wav_path = args.output_dir / ("%s.wav" % record["id"])
        if not args.overwrite and (record["id"] in completed or wav_path.exists()):
            continue
        selected.append(record)
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def write_text_file(records, path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            text = " ".join(record["tts_text"].split())
            f.write(text + "\n")


def build_matcha_command(args, text_file, output_folder):
    cli_path = args.matcha_root / "matcha" / "cli.py"
    # Some existing matcha-tts environments have a torchvision version that
    # expects torch.library.register_fake, while the installed torch is older.
    # Patch only the child CLI process so we do not mutate the conda env.
    launcher = r"""
import runpy
import sys

import torch

if hasattr(torch, "library") and not hasattr(torch.library, "register_fake"):
    def register_fake(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    torch.library.register_fake = register_fake

cli_path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(cli_path, run_name="__main__")
"""
    command = [
        sys.executable,
        "-c",
        launcher,
        str(cli_path),
        "--file",
        str(text_file),
        "--output_folder",
        str(output_folder),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--steps",
        str(args.steps),
        "--batched",
        "--batch_size",
        str(args.batch_size),
    ]
    if args.checkpoint_path:
        command.extend(["--checkpoint_path", str(args.checkpoint_path)])
    if args.vocoder:
        command.extend(["--vocoder", args.vocoder])
    if args.speaking_rate is not None:
        command.extend(["--speaking_rate", str(args.speaking_rate)])
    if args.spk is not None:
        command.extend(["--spk", str(args.spk)])
    if args.cpu:
        command.append("--cpu")
    return command


def expected_matcha_base(index, total_records, spk):
    if total_records == 1:
        base_index = 1
    else:
        base_index = index
    if spk is None:
        return "utterance_%03d" % base_index
    return "utterance_%03d_speaker_%03d" % (base_index, spk)


def collect_outputs(records, tmp_audio_dir, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, record in enumerate(records):
        base = expected_matcha_base(idx, len(records), args.spk)
        src_wav = tmp_audio_dir / ("%s.wav" % base)
        if not src_wav.exists():
            raise RuntimeError("Expected Matcha output not found: %s" % src_wav)
        dst_wav = args.output_dir / ("%s.wav" % record["id"])
        if dst_wav.exists() and args.overwrite:
            dst_wav.unlink()
        shutil.move(str(src_wav), str(dst_wav))
        row = {
            "id": record["id"],
            "wav_path": str(dst_wav),
            "tts_text": record["tts_text"],
            "risk_category": record.get("risk_category"),
            "kimi_security": record.get("kimi_security"),
            "tts_model": "matcha-tts",
            "matcha_model": args.model,
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
            "vocoder": args.vocoder,
            "spk": args.spk,
            "temperature": args.temperature,
            "speaking_rate": args.speaking_rate,
            "steps": args.steps,
            "sample_rate": 22050,
            "source_jsonl": str(args.input),
        }
        for key in [
            "source_id",
            "original_tts_text",
            "wrapped_tts_text",
            "wrapper_id",
            "wrapper_family",
            "wrapper_template",
            "original_kimi_security",
        ]:
            if key in record:
                row[key] = record[key]
        manifest_rows.append(row)
    return manifest_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize seed prompts with Matcha-TTS.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--matcha-root", type=Path, default=DEFAULT_MATCHA_ROOT)
    parser.add_argument("--model", default="matcha_ljspeech", choices=["matcha_ljspeech", "matcha_vctk"])
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--spk", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--speaking-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--only-attack-definition-met", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit("Input JSONL not found: %s" % args.input)
    if not args.matcha_root.exists():
        raise SystemExit("Matcha-TTS root not found: %s" % args.matcha_root)
    if not (args.matcha_root / "matcha" / "cli.py").exists():
        raise SystemExit("Matcha CLI not found under: %s" % args.matcha_root)
    if args.checkpoint_path is not None and not args.checkpoint_path.exists():
        raise SystemExit("Checkpoint not found: %s" % args.checkpoint_path)

    records = read_jsonl(args.input)
    if args.overwrite and args.manifest.exists():
        args.manifest.unlink()
    selected = select_records(records, args)
    print("input=%s" % args.input)
    print("output_dir=%s" % args.output_dir)
    print("manifest=%s" % args.manifest)
    print("matcha_root=%s" % args.matcha_root)
    print("model=%s total_records=%d pending_this_run=%d" % (args.model, len(records), len(selected)))
    if not selected:
        return

    with tempfile.TemporaryDirectory(prefix="matcha_seed_prompts_") as tmp:
        tmp_dir = Path(tmp)
        text_file = tmp_dir / "texts.txt"
        tmp_audio_dir = tmp_dir / "audio"
        tmp_audio_dir.mkdir(parents=True, exist_ok=True)
        write_text_file(selected, text_file)

        command = build_matcha_command(args, text_file, tmp_audio_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.matcha_root) + os.pathsep + env.get("PYTHONPATH", "")
        print("running=%s" % " ".join(command), flush=True)
        subprocess.check_call(command, cwd=str(tmp_dir), env=env)

        rows = collect_outputs(selected, tmp_audio_dir, args)
        for row in rows:
            append_manifest(args.manifest, row)
    print("done")


if __name__ == "__main__":
    main()
