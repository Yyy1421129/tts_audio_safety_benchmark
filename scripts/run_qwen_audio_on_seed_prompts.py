#!/usr/bin/env python3
"""Run a Bailian/DashScope Qwen audio model on synthesized seed-prompt audio.

The output JSONL intentionally mirrors the Step-Audio2 response format used by
this benchmark: each row keeps the manifest metadata and stores the target model
answer in ``response_text`` so the existing Kimi judge script can be reused.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "matcha_seed_prompts_v0_1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qwen3_5_omni_matcha_seed_responses.jsonl"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-flash"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_TEXT = (
    "Please listen to the audio and answer the spoken user request directly."
)


def normalize_api_key(key):
    key = (key or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        key = key[1:-1].strip()
    return key


def normalize_api_base(api_base):
    api_base = (api_base or "").strip()
    api_base = api_base.replace("`", "").strip()
    api_base = re.sub(r"\s+", "", api_base)
    return api_base.rstrip("/")


def mask_api_key(key):
    if not key:
        return "<empty>"
    if len(key) <= 12:
        return key[:2] + "***"
    return key[:6] + "***" + key[-4:]


def load_api_key(args):
    if args.api_key:
        return normalize_api_key(args.api_key), "--api-key"
    if args.api_key_file:
        return normalize_api_key(Path(args.api_key_file).read_text(encoding="utf-8")), "--api-key-file"
    key = normalize_api_key(os.environ.get("DASHSCOPE_API_KEY", ""))
    if key:
        return key, "DASHSCOPE_API_KEY"
    raise SystemExit(
        "Missing DashScope/Bailian API key. Set DASHSCOPE_API_KEY, pass --api-key, "
        "or pass --api-key-file /path/to/key.txt"
    )


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
    old_prefix = "/mnt/cloudstorfs/sjtu_home"
    new_prefix = "/hpc_stor03/sjtu_home"
    if text.startswith(old_prefix):
        candidate = Path(new_prefix + text[len(old_prefix) :])
        if candidate.exists():
            return candidate
    return path


def audio_data_url(wav_path, audio_format, audio_encoding):
    raw = Path(wav_path).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    if audio_encoding == "raw-base64":
        return encoded
    mime_type = mimetypes.guess_type(str(wav_path))[0]
    if not mime_type:
        mime_type = f"audio/{audio_format}"
    return f"data:{mime_type};base64,{encoded}"


def build_messages(wav_path, args):
    input_audio = {
        "data": audio_data_url(wav_path, args.audio_format, args.audio_encoding)
    }
    if args.include_audio_format:
        input_audio["format"] = args.audio_format

    user_content = [{"type": "input_audio", "input_audio": input_audio}]
    if args.user_text:
        user_content.append({"type": "text", "text": args.user_text})

    return [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_extra_body(extra_body_json):
    if not extra_body_json:
        return {}
    try:
        parsed = json.loads(extra_body_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --extra-body-json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    return parsed


def extract_message_content(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return ""


def call_chat_completions(api_key, args, messages, extra_body):
    url = normalize_api_base(args.api_base) + "/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_exc = None
    for attempt in range(1, args.retries + 2):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read().decode("utf-8")
            result = json.loads(raw)
            message = result["choices"][0]["message"]
            return extract_message_content(message), result
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_exc = exc
            if attempt > args.retries:
                raise RuntimeError(f"Request failed after {attempt} attempts: {exc}") from exc
            time.sleep(args.retry_sleep * attempt)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected API response shape: {exc}") from exc
    raise RuntimeError(f"Request failed: {last_exc}")


def auth_check(api_key, args, extra_body):
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": "Reply with OK."},
    ]
    content, raw = call_chat_completions(api_key, args, messages, extra_body)
    return {"content": content, "raw_model": raw.get("model")}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Qwen audio-capable Bailian model on benchmark audio."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="", help="API key string. Prefer --api-key-file.")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-text", default=DEFAULT_USER_TEXT)
    parser.add_argument("--audio-format", default="wav")
    parser.add_argument(
        "--audio-encoding",
        choices=["data-url", "raw-base64"],
        default="data-url",
        help="How to put local audio into input_audio.data.",
    )
    parser.add_argument(
        "--include-audio-format",
        action="store_true",
        help="Also include input_audio.format. Useful for models expecting raw base64.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-responses", action="store_true")
    parser.add_argument("--auth-check-only", action="store_true")
    parser.add_argument(
        "--extra-body-json",
        default="",
        help="Extra JSON object merged into the chat/completions request body.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api_key, key_source = load_api_key(args)
    extra_body = parse_extra_body(args.extra_body_json)

    print(f"api_key_source={key_source} api_key_masked={mask_api_key(api_key)}")
    print(f"api_base={normalize_api_base(args.api_base)}")
    print(f"model={args.model}")

    if args.auth_check_only:
        result = auth_check(api_key, args, extra_body)
        print(f"auth_check=OK raw_model={result.get('raw_model')} content={result.get('content')!r}")
        return

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    records = load_jsonl(args.manifest)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    done_ids = load_done_ids(args.output)
    pending = [row for row in records if row.get("id") not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"manifest={args.manifest}")
    print(f"output={args.output}")
    print(f"manifest_records={len(records)} done={len(done_ids)} pending_this_run={len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    request_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "audio_encoding": args.audio_encoding,
        "include_audio_format": args.include_audio_format,
        "extra_body": extra_body,
    }

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
            "original_wav_path": row.get("wav_path"),
            "tts_text": row.get("tts_text"),
            "kimi_security": row.get("kimi_security"),
            "qwen_model": args.model,
            "api_base": normalize_api_base(args.api_base),
            "system_prompt": args.system_prompt,
            "user_text": args.user_text,
            "generation": request_config,
        }

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
            messages = build_messages(wav_path, args)
            response_text, raw = call_chat_completions(api_key, args, messages, extra_body)
            result.update(
                {
                    "status": "ok",
                    "response_text": response_text.strip(),
                    "error": "",
                    "elapsed_sec": round(time.time() - started, 3),
                    "raw_model": raw.get("model"),
                    "usage": raw.get("usage"),
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
