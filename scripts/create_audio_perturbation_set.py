#!/usr/bin/env python3
"""Create controlled audio perturbation sets for phase-2 benchmark experiments.

This script reads one or more existing TTS manifests, applies reproducible audio
edits, and writes a new manifest that can be fed into Step-Audio2 or Qwen audio
evaluation scripts.

Implemented perturbations:
- baseline copy
- speed change via ffmpeg atempo
- volume change via ffmpeg volume
- white-noise mixing at a target SNR via numpy/scipy
- front/back silence padding via numpy/scipy

The default configuration uses single-variable changes to keep the experiment
easy to interpret. A JSON config can be provided to override the defaults.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "audio" / "phase2_perturbations"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "manifests" / "phase2_audio_perturbations.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "manifests" / "phase2_audio_perturbations_summary.csv"


DEFAULT_CONFIG = {
    "conditions": [
        {"condition_id": "orig", "type": "baseline"},
        {"condition_id": "speed_0p8", "type": "speed", "speed": 0.8},
        {"condition_id": "speed_1p2", "type": "speed", "speed": 1.2},
        {"condition_id": "speed_1p4", "type": "speed", "speed": 1.4},
        {"condition_id": "volume_m6db", "type": "volume", "volume_db": -6.0},
        {"condition_id": "volume_p6db", "type": "volume", "volume_db": 6.0},
        {"condition_id": "noise_snr20db", "type": "noise", "snr_db": 20.0},
        {"condition_id": "noise_snr10db", "type": "noise", "snr_db": 10.0},
        {
            "condition_id": "silence_0p5s",
            "type": "silence",
            "front_silence_sec": 0.5,
            "back_silence_sec": 0.5,
        },
        {
            "condition_id": "silence_1p0s",
            "type": "silence",
            "front_silence_sec": 1.0,
            "back_silence_sec": 1.0,
        },
    ],
    "noise_seed": 20260713,
    "target_sample_rate": None,
    "output_subtype": "pcm_s16le",
}


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


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    if not path:
        return DEFAULT_CONFIG
    with Path(path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg or load an environment with ffmpeg.")


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


def sanitize_id(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def get_tts_model(row, fallback):
    return sanitize_id(row.get("tts_model") or fallback or "unknown_tts")


def read_audio_float(path):
    sample_rate, data = wavfile.read(path)
    original_dtype = data.dtype
    if data.ndim == 1:
        data = data[:, None]

    if np.issubdtype(original_dtype, np.integer):
        max_abs = float(np.iinfo(original_dtype).max)
        audio = data.astype(np.float32) / max_abs
    else:
        audio = data.astype(np.float32)
    return sample_rate, audio, original_dtype


def write_audio_float(path, sample_rate, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    int_audio = (audio * 32767.0).astype(np.int16)
    if int_audio.ndim == 2 and int_audio.shape[1] == 1:
        int_audio = int_audio[:, 0]
    wavfile.write(path, sample_rate, int_audio)


def audio_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def peak_abs(audio):
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def duration_sec(path):
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate) if rate else 0.0


def ffmpeg_run(input_path, output_path, audio_filter, target_sample_rate=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        audio_filter,
    ]
    if target_sample_rate:
        cmd.extend(["-ar", str(target_sample_rate)])
    cmd.extend(["-acodec", "pcm_s16le", str(output_path)])
    subprocess.run(cmd, check=True)


def unlink_if_exists(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def decode_for_python(input_path, tmp_path, target_sample_rate=None):
    ffmpeg_run(input_path, tmp_path, "anull", target_sample_rate=target_sample_rate)
    return read_audio_float(tmp_path)


def copy_baseline(input_path, output_path, target_sample_rate=None):
    if target_sample_rate:
        ffmpeg_run(input_path, output_path, "anull", target_sample_rate=target_sample_rate)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)


def apply_speed(input_path, output_path, speed, target_sample_rate=None):
    # atempo supports 0.5-100 in chained filters; keep chaining for robustness.
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    factors = []
    remaining = float(speed)
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    audio_filter = ",".join(f"atempo={factor:.8g}" for factor in factors)
    ffmpeg_run(input_path, output_path, audio_filter, target_sample_rate=target_sample_rate)


def apply_volume(input_path, output_path, volume_db, target_sample_rate=None):
    ffmpeg_run(input_path, output_path, f"volume={float(volume_db):.8g}dB", target_sample_rate=target_sample_rate)


def apply_noise(input_path, output_path, snr_db, rng, target_sample_rate=None):
    tmp_path = Path(output_path).with_suffix(".decode.tmp.wav")
    sample_rate, audio, _ = decode_for_python(input_path, tmp_path, target_sample_rate=target_sample_rate)
    unlink_if_exists(tmp_path)

    speech_rms = audio_rms(audio)
    if speech_rms == 0:
        noisy = audio
        noise_rms = 0.0
    else:
        noise = rng.normal(loc=0.0, scale=1.0, size=audio.shape).astype(np.float32)
        current_noise_rms = audio_rms(noise)
        target_noise_rms = speech_rms / (10.0 ** (float(snr_db) / 20.0))
        noise = noise * (target_noise_rms / max(current_noise_rms, 1e-12))
        noisy = audio + noise
        noise_rms = audio_rms(noise)

    peak = peak_abs(noisy)
    clipped = peak > 1.0
    if clipped:
        noisy = noisy / peak * 0.999
    write_audio_float(output_path, sample_rate, noisy)
    return {"speech_rms": speech_rms, "noise_rms": noise_rms, "peak_before_clip": peak, "clipped": clipped}


def apply_silence(input_path, output_path, front_sec, back_sec, target_sample_rate=None):
    tmp_path = Path(output_path).with_suffix(".decode.tmp.wav")
    sample_rate, audio, _ = decode_for_python(input_path, tmp_path, target_sample_rate=target_sample_rate)
    unlink_if_exists(tmp_path)

    channels = audio.shape[1]
    front = np.zeros((int(round(float(front_sec) * sample_rate)), channels), dtype=np.float32)
    back = np.zeros((int(round(float(back_sec) * sample_rate)), channels), dtype=np.float32)
    padded = np.concatenate([front, audio, back], axis=0)
    write_audio_float(output_path, sample_rate, padded)
    return {"front_silence_samples": len(front), "back_silence_samples": len(back)}


def make_output_path(output_dir, row, source_tts_model, condition_id):
    sample_id = sanitize_id(row["id"])
    suffix = Path(row["wav_path"]).suffix or ".wav"
    filename = f"{sample_id}__{condition_id}{suffix}"
    return Path(output_dir) / source_tts_model / condition_id / filename


def build_manifest_row(source_manifest, row, condition, output_path, duration, extra=None):
    condition_type = condition["type"]
    new_id = f"{row['id']}__{condition['condition_id']}"
    manifest_row = dict(row)
    manifest_row.update(
        {
            "id": new_id,
            "source_id": row["id"],
            "source_manifest": str(source_manifest),
            "original_wav_path": row.get("wav_path"),
            "wav_path": str(output_path),
            "perturbation_type": condition_type,
            "condition_id": condition["condition_id"],
            "speed": condition.get("speed", 1.0),
            "volume_db": condition.get("volume_db", 0.0),
            "snr_db": condition.get("snr_db"),
            "front_silence_sec": condition.get("front_silence_sec", 0.0),
            "back_silence_sec": condition.get("back_silence_sec", 0.0),
            "duration_sec": round(duration, 4),
            "perturbation_config": condition,
        }
    )
    if extra:
        manifest_row["perturbation_stats"] = extra
    return manifest_row


def write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    for row in rows:
        key = (row.get("tts_model", "unknown"), row.get("condition_id", "unknown"), row.get("perturbation_type", "unknown"))
        counts[key] = counts.get(key, 0) + 1

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tts_model", "condition_id", "perturbation_type", "count"])
        writer.writeheader()
        for (tts_model, condition_id, perturbation_type), count in sorted(counts.items()):
            writer.writerow(
                {
                    "tts_model": tts_model,
                    "condition_id": condition_id,
                    "perturbation_type": perturbation_type,
                    "count": count,
                }
            )


def validate_conditions(conditions):
    seen = set()
    allowed = {"baseline", "speed", "volume", "noise", "silence"}
    for condition in conditions:
        condition_id = condition.get("condition_id")
        condition_type = condition.get("type")
        if not condition_id:
            raise ValueError(f"Condition missing condition_id: {condition}")
        if condition_id in seen:
            raise ValueError(f"Duplicate condition_id: {condition_id}")
        if condition_type not in allowed:
            raise ValueError(f"Unsupported condition type {condition_type!r} in {condition_id}")
        seen.add(condition_id)


def process_condition(input_path, output_path, condition, rng, target_sample_rate):
    condition_type = condition["type"]
    if condition_type == "baseline":
        copy_baseline(input_path, output_path, target_sample_rate=target_sample_rate)
        return {}
    if condition_type == "speed":
        apply_speed(input_path, output_path, float(condition["speed"]), target_sample_rate=target_sample_rate)
        return {}
    if condition_type == "volume":
        apply_volume(input_path, output_path, float(condition["volume_db"]), target_sample_rate=target_sample_rate)
        return {}
    if condition_type == "noise":
        return apply_noise(input_path, output_path, float(condition["snr_db"]), rng, target_sample_rate=target_sample_rate)
    if condition_type == "silence":
        return apply_silence(
            input_path,
            output_path,
            float(condition.get("front_silence_sec", 0.0)),
            float(condition.get("back_silence_sec", 0.0)),
            target_sample_rate=target_sample_rate,
        )
    raise ValueError(f"Unsupported condition type: {condition_type}")


def stable_seed(base_seed, *parts):
    text = "::".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def parse_args():
    parser = argparse.ArgumentParser(description="Create controlled audio perturbation manifests.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Input JSONL manifest. Can be specified multiple times.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config overriding default conditions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=None, help="Limit records per input manifest.")
    parser.add_argument("--condition", action="append", default=None, help="Only run selected condition_id(s).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--tts-model-fallback",
        default="",
        help="Fallback tts_model value when a manifest row does not contain tts_model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_ffmpeg()

    config = load_config(args.config)
    conditions = config.get("conditions", [])
    validate_conditions(conditions)
    if args.condition:
        wanted = set(args.condition)
        conditions = [condition for condition in conditions if condition["condition_id"] in wanted]
        missing = wanted - {condition["condition_id"] for condition in conditions}
        if missing:
            raise ValueError(f"Unknown condition_id(s): {sorted(missing)}")

    target_sample_rate = config.get("target_sample_rate")
    noise_seed = int(config.get("noise_seed", DEFAULT_CONFIG["noise_seed"]))

    print(f"manifests={args.manifest}")
    print(f"output_dir={args.output_dir}")
    print(f"output_manifest={args.output_manifest}")
    print(f"summary={args.summary}")
    print(f"conditions={len(conditions)} {[c['condition_id'] for c in conditions]}")
    print(f"target_sample_rate={target_sample_rate}")
    print(f"noise_seed={noise_seed}")

    if args.overwrite and args.output_manifest.exists() and not args.dry_run:
        args.output_manifest.unlink()

    output_rows = []
    errors = []

    for manifest_path in args.manifest:
        manifest_path = Path(manifest_path)
        records = load_jsonl(manifest_path)
        if args.limit is not None:
            records = records[: args.limit]
        print(f"processing_manifest={manifest_path} records={len(records)}")

        for row_idx, row in enumerate(records, 1):
            source_wav = resolve_wav_path(row["wav_path"])
            source_tts_model = get_tts_model(row, args.tts_model_fallback or manifest_path.stem)
            if not source_wav.exists():
                message = f"Missing wav for {row.get('id')}: {source_wav}"
                errors.append(message)
                print(f"ERROR: {message}", file=sys.stderr)
                if args.fail_fast:
                    raise FileNotFoundError(message)
                continue

            for condition in conditions:
                condition_id = sanitize_id(condition["condition_id"])
                output_path = make_output_path(args.output_dir, row, source_tts_model, condition_id)
                print(f"[{manifest_path.name} {row_idx}/{len(records)}] {row['id']} -> {condition_id}")
                if args.dry_run:
                    continue

                rng = np.random.default_rng(stable_seed(noise_seed, row["id"], condition_id))
                try:
                    extra = process_condition(source_wav, output_path, condition, rng, target_sample_rate)
                    row_duration = duration_sec(output_path)
                    manifest_row = build_manifest_row(manifest_path, row, condition, output_path, row_duration, extra)
                    append_jsonl(args.output_manifest, manifest_row)
                    output_rows.append(manifest_row)
                except Exception as exc:
                    message = f"{row.get('id')} {condition_id}: {repr(exc)}"
                    errors.append(message)
                    print(f"ERROR: {message}", file=sys.stderr)
                    if args.fail_fast:
                        raise

    if not args.dry_run:
        if output_rows:
            write_summary(args.summary, output_rows)
        print(f"written_rows={len(output_rows)}")
        print(f"summary_written={args.summary if output_rows else ''}")
    else:
        planned = sum(
            min(len(load_jsonl(Path(manifest))), args.limit or math.inf) * len(conditions)
            for manifest in args.manifest
        )
        print(f"dry_run_planned_rows={planned}")

    if errors:
        print(f"errors={len(errors)}", file=sys.stderr)
        for message in errors[:20]:
            print(f"ERROR: {message}", file=sys.stderr)
        if len(errors) > 20:
            print(f"ERROR: ... {len(errors) - 20} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
