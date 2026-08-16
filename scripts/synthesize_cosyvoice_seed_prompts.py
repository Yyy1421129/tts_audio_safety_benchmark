#!/usr/bin/env python3
"exec" "python3" "$0" "$@"
"""Batch synthesize seed-prompt JSONL records with CosyVoice.

The script reads the benchmark JSONL format, uses `tts_text` as TTS input,
saves one wav per sample, and writes a manifest for downstream Step-Audio2
evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "seed_prompts_en_v0_1_kimi_labeled.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "audio" / "cosyvoice_seed_prompts_v0_1"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "cosyvoice_seed_prompts_v0_1.jsonl"
DEFAULT_COSYVOICE_ROOT = Path("/hpc_stor03/sjtu_home/yi.yang/CosyVoice")
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/yi.yang/.cache/modelscope/hub/models/iic/CosyVoice2-0.5B")
DEFAULT_PROMPT_WAV = DEFAULT_COSYVOICE_ROOT / "asset" / "zero_shot_prompt.wav"
DEFAULT_PROMPT_TEXT = "I hope that in the future you can do even better than me."


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


def import_cosyvoice(cosyvoice_root):
    sys.path.insert(0, str(cosyvoice_root))
    sys.path.insert(0, str(cosyvoice_root / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel
    import torch
    import torchaudio
    return AutoModel, torch, torchaudio


def synthesize_one(cosyvoice, torch, torchaudio, record, args):
    text = record["tts_text"]
    pieces = []

    if args.mode == "sft":
        iterator = cosyvoice.inference_sft(
            text,
            args.spk_id,
            stream=False,
            speed=args.speed,
            text_frontend=not args.disable_text_frontend,
        )
    elif args.mode == "zero_shot":
        iterator = cosyvoice.inference_zero_shot(
            text,
            args.prompt_text,
            str(args.prompt_wav),
            zero_shot_spk_id=args.zero_shot_spk_id,
            stream=False,
            speed=args.speed,
            text_frontend=not args.disable_text_frontend,
        )
    elif args.mode == "cross_lingual":
        iterator = cosyvoice.inference_cross_lingual(
            text,
            str(args.prompt_wav),
            zero_shot_spk_id=args.zero_shot_spk_id,
            stream=False,
            speed=args.speed,
            text_frontend=not args.disable_text_frontend,
        )
    elif args.mode == "instruct":
        iterator = cosyvoice.inference_instruct(
            text,
            args.spk_id,
            args.instruct_text,
            stream=False,
            speed=args.speed,
            text_frontend=not args.disable_text_frontend,
        )
    elif args.mode == "instruct2":
        iterator = cosyvoice.inference_instruct2(
            text,
            args.instruct_text,
            str(args.prompt_wav),
            zero_shot_spk_id=args.zero_shot_spk_id,
            stream=False,
            speed=args.speed,
            text_frontend=not args.disable_text_frontend,
        )
    else:
        raise ValueError("Unsupported mode: %s" % args.mode)

    for output in iterator:
        pieces.append(output["tts_speech"].cpu())

    if not pieces:
        raise RuntimeError("CosyVoice produced no audio for %s" % record["id"])

    waveform = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
    wav_path = args.output_dir / ("%s.wav" % record["id"])
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(wav_path), waveform, cosyvoice.sample_rate)
    return wav_path


def build_manifest_row(record, wav_path, args, sample_rate):
    return {
        "id": record["id"],
        "wav_path": str(wav_path),
        "tts_text": record["tts_text"],
        "risk_category": record.get("risk_category"),
        "kimi_security": record.get("kimi_security"),
        "tts_model": "cosyvoice",
        "cosyvoice_mode": args.mode,
        "model_dir": str(args.model_dir),
        "prompt_wav": str(args.prompt_wav) if args.prompt_wav else None,
        "spk_id": args.spk_id,
        "speed": args.speed,
        "sample_rate": sample_rate,
        "source_jsonl": str(args.input),
    }


def detect_cosyvoice_model_type(model_dir):
    if (model_dir / "cosyvoice.yaml").exists():
        return "cosyvoice"
    if (model_dir / "cosyvoice2.yaml").exists():
        return "cosyvoice2"
    if (model_dir / "cosyvoice3.yaml").exists():
        return "cosyvoice3"
    return "auto"


def load_cosyvoice_model(AutoModel, args):
    model_type = detect_cosyvoice_model_type(args.model_dir)
    if model_type == "auto":
        return AutoModel(model_dir=str(args.model_dir))
    kwargs = {
        "model_dir": str(args.model_dir),
        "load_trt": args.load_trt,
        "fp16": args.fp16,
    }
    if model_type in ("cosyvoice", "cosyvoice2"):
        kwargs["load_jit"] = args.load_jit
    if model_type in ("cosyvoice2", "cosyvoice3"):
        kwargs["load_vllm"] = args.load_vllm
    return AutoModel(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize seed prompts with CosyVoice.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cosyvoice-root", type=Path, default=DEFAULT_COSYVOICE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--mode",
        choices=["sft", "zero_shot", "cross_lingual", "instruct", "instruct2"],
        default="cross_lingual",
    )
    parser.add_argument("--spk-id", default="中文女", help="Speaker id for sft/instruct modes.")
    parser.add_argument("--prompt-wav", type=Path, default=DEFAULT_PROMPT_WAV)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--zero-shot-spk-id", default="")
    parser.add_argument("--instruct-text", default="Please speak naturally in English.<|endofprompt|>")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--only-attack-definition-met", action="store_true")
    parser.add_argument("--disable-text-frontend", action="store_true")
    parser.add_argument("--load-jit", action="store_true")
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--load-vllm", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit("Input JSONL not found: %s" % args.input)
    if not args.cosyvoice_root.exists():
        raise SystemExit("CosyVoice root not found: %s" % args.cosyvoice_root)
    if args.mode in ("zero_shot", "cross_lingual", "instruct2") and not args.prompt_wav.exists():
        raise SystemExit("Prompt wav not found: %s" % args.prompt_wav)

    records = read_jsonl(args.input)
    selected = select_records(records, args)
    print("input=%s" % args.input)
    print("output_dir=%s" % args.output_dir)
    print("manifest=%s" % args.manifest)
    print("model_dir=%s" % args.model_dir)
    print("mode=%s total_records=%d pending_this_run=%d" % (args.mode, len(records), len(selected)))
    if not selected:
        return

    AutoModel, torch, torchaudio = import_cosyvoice(args.cosyvoice_root)
    cosyvoice = load_cosyvoice_model(AutoModel, args)

    for idx, record in enumerate(selected, 1):
        print("[%d/%d] synthesizing %s" % (idx, len(selected), record["id"]), flush=True)
        wav_path = synthesize_one(cosyvoice, torch, torchaudio, record, args)
        row = build_manifest_row(record, wav_path, args, cosyvoice.sample_rate)
        append_manifest(args.manifest, row)
    print("done")


if __name__ == "__main__":
    main()

# vc submit -p pdgpu-a10  -i docker.v2.aispeech.com/sjtu/sjtu_yukai-zhihanli-cosyvoice3:v1.0 -c 4 -m 24G -g 1  -n 1  --cmd "cd /hpc_stor03/sjtu_home/yi.yang/tts_audio_safety_benchmark_plan/scripts && python3 synthesize_cosyvoice_seed_prompts.py"