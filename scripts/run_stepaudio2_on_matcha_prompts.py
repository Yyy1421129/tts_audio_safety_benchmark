#!/usr/bin/env python3
"""Run Step-Audio2 on synthesized Matcha-TTS seed-prompt audio.

The script reads the benchmark manifest, sends each wav file to Step-Audio2 as
audio-only user input, and writes model responses to JSONL for later safety
judging. It intentionally does not print response text by default.
"""

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "matcha_seed_prompts_v0_1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "stepaudio2_matcha_seed_responses.jsonl"
DEFAULT_STEP_AUDIO_ROOT = Path("/hpc_stor03/sjtu_home/yi.yang/Step-Audio2")
DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/yi.yang/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-mini")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
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
    with path.open("r", encoding="utf-8") as f:
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


def select_shard(records, num_shards, shard_index):
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [row for idx, row in enumerate(records) if idx % num_shards == shard_index]


def build_messages(wav_path, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": [{"type": "audio", "audio": str(wav_path)}]},
        {"role": "assistant", "content": None},
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step-Audio2 responses for Matcha-TTS seed prompt audio."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step-audio-root", type=Path, default=DEFAULT_STEP_AUDIO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending records.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output and rerun all records.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index, zero-based.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--print-responses", action="store_true", help="Print response text to stdout.")
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Only check Step-Audio2 imports, model path, manifest, and CUDA availability.",
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
    shard_records = select_shard(records, args.num_shards, args.shard_index)

    print(f"manifest={args.manifest}")
    print(f"output={args.output}")
    print(f"model_dir={args.model_dir}")
    print(f"step_audio_root={args.step_audio_root}")
    print(f"manifest_records={len(records)}")
    print(f"num_shards={args.num_shards}")
    print(f"shard_index={args.shard_index}")
    print(f"shard_records={len(shard_records)}")
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
            "CUDA is not available in this session. Step-Audio2 currently calls .cuda() "
            "during model loading, so run this script on a GPU-visible node/session."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"Not enough free GPU memory before loading Step-Audio2: {free_gb:.2f} GiB free "
            f"on {torch.cuda.get_device_name(0)}, but --min-free-gb requires {args.min_free_gb:.2f} GiB. "
            "Choose a freer GPU with CUDA_VISIBLE_DEVICES or stop unrelated GPU jobs."
        )

    if args.overwrite and args.output.exists():
        args.output.unlink()
    done_ids = load_done_ids(args.output)

    pending = [row for row in shard_records if row.get("id") not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"total_records={len(records)} shard_records={len(shard_records)} "
        f"done={len(done_ids)} pending_this_run={len(pending)}"
    )
    if not pending:
        print("Nothing to do.")
        return

    model = StepAudio2(str(args.model_dir))

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty

    for idx, row in enumerate(pending, 1):
        sample_id = row["id"]
        wav_path = resolve_wav_path(row["wav_path"])
        print(f"[{idx}/{len(pending)}] running {sample_id}", flush=True)

        started = time.time()
        result = {
            "id": sample_id,
            "risk_category": row.get("risk_category"),
            "tts_model": row.get("tts_model"),
            "wav_path": str(wav_path),
            "tts_text": row.get("tts_text"),
            "kimi_security": row.get("kimi_security"),
            "stepaudio2_model_dir": str(args.model_dir),
            "system_prompt": args.system_prompt,
            "generation": generation_kwargs,
        }
        for field in PASSTHROUGH_FIELDS:
            if field in row:
                result[field] = row[field]

        if not wav_path.exists():
            result.update(
                {
                    "status": "error",
                    "response_text": "",
                    "error": f"wav_path not found: {wav_path}",
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            append_jsonl(args.output, result)
            print(f"[{idx}/{len(pending)}] error {sample_id}: wav_path missing", flush=True)
            continue

        try:
            messages = build_messages(wav_path, args.system_prompt)
            _, response_text, _ = model(messages, **generation_kwargs)
            result.update(
                {
                    "status": "ok",
                    "response_text": response_text.strip(),
                    "error": "",
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            append_jsonl(args.output, result)
            if args.print_responses:
                print(f"{sample_id}: {response_text.strip()}", flush=True)
        except Exception as exc:  # Keep long batch runs resumable.
            result.update(
                {
                    "status": "error",
                    "response_text": "",
                    "error": repr(exc),
                    "elapsed_sec": round(time.time() - started, 3),
                }
            )
            append_jsonl(args.output, result)
            print(f"[{idx}/{len(pending)}] error {sample_id}: {exc}", flush=True)

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
