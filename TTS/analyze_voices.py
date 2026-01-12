#!/usr/bin/env python3
"""Synthesize audio for Kokoro voice packs and analyze with OpenAI."""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional
from urllib import request

import numpy as np
import soundfile as sf
import torch

try:
    from kokoro import KModel, KPipeline
except Exception as exc:
    raise SystemExit(
        "Missing dependency: kokoro. Install with "
        "`pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"`."
    ) from exc


SAMPLE_RATE = 24000


def build_en_callable(repo_id: str):
    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

    def en_callable(text: str) -> str:
        return next(en_pipeline(text)).phonemes

    return en_callable


def build_pipeline(repo_id: str, lang_code: str, model: KModel) -> KPipeline:
    try:
        return KPipeline(
            lang_code=lang_code,
            repo_id=repo_id,
            model=model,
            en_callable=build_en_callable(repo_id),
        )
    except TypeError:
        return KPipeline(lang_code=lang_code, repo_id=repo_id, model=model)


def synthesize_voice(
    pipeline: KPipeline,
    voice_pack: torch.FloatTensor,
    text: str,
    speed: float,
    silence: int,
) -> np.ndarray:
    wavs = []
    for result in pipeline(text, voice=voice_pack, speed=speed):
        wav = result.audio
        if wav is None:
            continue
        wav = wav.squeeze().cpu().numpy().astype(np.float32)
        if wavs and silence > 0:
            wavs.append(np.zeros(silence, dtype=np.float32))
        wavs.append(wav)
    if not wavs:
        raise RuntimeError("No audio produced for voice.")
    return np.concatenate(wavs)


def guess_gender_from_filename(name: str) -> str:
    lower = name.lower()
    if lower.startswith("zf_"):
        return "female"
    if lower.startswith("zm_"):
        return "male"
    return "unknown"


def call_openai(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    wav_path: Path,
    timeout: int,
) -> str:
    audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/responses",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return raw


def extract_output_text(response_json: str) -> str:
    data = json.loads(response_json)
    outputs = data.get("output", [])
    chunks = []
    for output in outputs:
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    if chunks:
        return "\n".join(chunks).strip()
    return data.get("output_text", "").strip()


def parse_json_response(text: str) -> Optional[Dict[str, str]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize voice samples and analyze with OpenAI."
    )
    parser.add_argument(
        "--voices-dir",
        default="hexgrad/Kokoro-82M-v1.1-zh/voices",
        help="Directory containing .pt voice packs.",
    )
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh")
    parser.add_argument("--lang-code", default="z")
    parser.add_argument(
        "--text",
        default="Kokoro 是一系列体积虽小但功能强大的 TTS 模型，该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。",
        help="Text used to synthesize each voice.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--silence", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="voice_samples",
        help="Directory to write synthesized wavs.",
    )
    parser.add_argument(
        "--analysis-out",
        default="voice_analysis.jsonl",
        help="Output JSONL with analysis results.",
    )
    parser.add_argument("--model", default="google/gemini-3-pro-preview")
    parser.add_argument(
        "--prompt",
        default=(
            r"""You are analyzing a short TTS voice sample.

Task:
1. Classify the speaker's perceived gender based on voice characteristics only.
2. Provide a concise Chinese timbre-style description suitable for voice selection.

Output format:
Return JSON only, no extra text:
{
  "gender": "female|male|unknown",
  "description": "<4-6 Chinese characters>"
}

Rules:
- The description must be 4–6 Chinese characters.
- No punctuation, no emojis, no explanations.
- Describe timbre and age impression only (e.g. youthful, mature, soft, deep).
- Do NOT describe emotion, content, language, or speaking style.
- Do NOT include gender words in the description.
- Use common TTS selection terms understandable by non-experts.
- If the voice is ambiguous, set gender to "unknown" but still give a best-effort description.
"""
        ),
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-voices", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Model device: auto|cpu|cuda",
    )
    args = parser.parse_args()

    voices_dir = Path(args.voices_dir)
    if not voices_dir.is_dir():
        raise SystemExit(f"Missing voices directory: {voices_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = Path(args.analysis_out)

    api_key = args.api_key or ("" if args.skip_analysis else os.getenv("OPENAI_API_KEY", ""))
    if not args.skip_analysis and not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = KModel(repo_id=args.repo_id).to(device).eval()
    pipeline = build_pipeline(args.repo_id, args.lang_code, model=model)

    completed = set()
    if analysis_path.exists():
        with analysis_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                voice_id = record.get("voice")
                if voice_id:
                    completed.add(voice_id)

    pt_files = sorted(voices_dir.glob("*.pt"))
    if args.max_voices > 0:
        pt_files = pt_files[: args.max_voices]

    with analysis_path.open("a", encoding="utf-8") as out_f:
        for voice_path in pt_files:
            voice_id = voice_path.stem
            if voice_id in completed:
                continue
            voice_pack = torch.load(voice_path, weights_only=True).to(device)
            with torch.no_grad():
                wav = synthesize_voice(
                    pipeline=pipeline,
                    voice_pack=voice_pack,
                    text=args.text,
                    speed=args.speed,
                    silence=args.silence,
                )
            wav_path = output_dir / f"{voice_id}.wav"
            sf.write(wav_path.as_posix(), wav, SAMPLE_RATE)

            record = {
                "voice": voice_id,
                "voice_path": voice_path.as_posix(),
                "wav_path": wav_path.as_posix(),
                "filename_gender": guess_gender_from_filename(voice_id),
                "analysis": None,
                "raw_response": None,
            }

            if not args.skip_analysis:
                raw = call_openai(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=args.prompt,
                    wav_path=wav_path,
                    timeout=args.timeout,
                )
                record["raw_response"] = raw
                output_text = extract_output_text(raw)
                record["analysis"] = parse_json_response(output_text) or {
                    "unparsed_text": output_text
                }
                if args.sleep > 0:
                    time.sleep(args.sleep)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Wrote analysis to {analysis_path}")


if __name__ == "__main__":
    main()
