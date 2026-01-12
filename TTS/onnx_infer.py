#!/usr/bin/env python
# Run Kokoro ONNX inference using KPipeline for G2P and voice packs.

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit("Missing dependency: onnxruntime. Install with `pip install onnxruntime`.") from exc

try:
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
except Exception as exc:
    raise SystemExit(
        "Missing dependency: kokoro. Install with "
        "`pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"`."
    ) from exc


SAMPLE_RATE = 24000


def load_vocab(repo_id):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["vocab"], config["plbert"]["max_position_embeddings"]


def phonemes_to_input_ids(vocab, phonemes, context_length):
    input_ids = list(filter(lambda i: i is not None, map(vocab.get, phonemes)))
    if len(input_ids) + 2 > context_length:
        raise ValueError(
            f"Input too long: {len(input_ids)+2} > {context_length}"
        )
    return np.asarray([[0, *input_ids, 0]], dtype=np.int64)


def ort_dtype(type_str):
    if "bfloat16" in type_str:
        return "bf16"
    if "float16" in type_str:
        return "fp16"
    return "fp32"


def to_ort_array(value, dtype_name):
    if dtype_name == "bf16":
        try:
            import ml_dtypes
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: ml_dtypes (required for bfloat16). "
                "Install with `pip install ml_dtypes`."
            ) from exc
        return np.asarray(value, dtype=ml_dtypes.bfloat16)
    if dtype_name == "fp16":
        return np.asarray(value, dtype=np.float16)
    return np.asarray(value, dtype=np.float32)


def build_pipeline(repo_id, lang_code):
    def en_callable(text):
        return next(en_pipeline(text)).phonemes

    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)
    try:
        return KPipeline(
            lang_code=lang_code,
            repo_id=repo_id,
            model=False,
            en_callable=en_callable,
        )
    except TypeError:
        return KPipeline(lang_code=lang_code, repo_id=repo_id, model=False)


def build_en_callable(repo_id):
    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)
    def en_callable(text):
        return next(en_pipeline(text)).phonemes
    return en_callable


def text_to_phonemes(text, repo_id):
    zh_pipeline = KPipeline(
        lang_code="z",
        repo_id=repo_id,
        model=False,
        en_callable=build_en_callable(repo_id),
    )
    parts = []
    for result in zh_pipeline(text):
        if result.phonemes:
            parts.append(result.phonemes)
    if not parts:
        raise ValueError("G2P produced empty phonemes.")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Run Kokoro ONNX inference.")
    parser.add_argument("--onnx", default="TTS/kokoro.onnx")
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh")
    parser.add_argument("--lang-code", default="z")
    parser.add_argument("--voice", default="zf_001")
    parser.add_argument("--voice-path", default="")
    parser.add_argument("--ref-s-index", type=int, default=-1)
    parser.add_argument("--text", default="Kokoro ONNX inference example.")
    parser.add_argument("--g2p", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", default="TTS/onnx_output.wav")
    parser.add_argument("--silence", type=int, default=0)
    args = parser.parse_args()

    vocab, context_length = load_vocab(args.repo_id)
    pipeline = build_pipeline(args.repo_id, args.lang_code)
    if args.voice_path:
        voice_pack = torch.load(args.voice_path, weights_only=True).cpu()
    else:
        voice_pack = pipeline.load_voice(args.voice).cpu()

    if args.g2p:
        args.text = text_to_phonemes(args.text, args.repo_id)
        print(f"G2P phonemes: {args.text}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_types = {i.name: ort_dtype(i.type) for i in sess.get_inputs()}
    output_names = [o.name for o in sess.get_outputs()]

    wavs = []
    if args.g2p:
        phoneme_chunks = [args.text]
    else:
        phoneme_chunks = [r.phonemes for r in pipeline(args.text) if r.phonemes]

    for phonemes in phoneme_chunks:
        input_ids = phonemes_to_input_ids(vocab, phonemes, context_length)
        if args.ref_s_index >= 0:
            ref_s = voice_pack[args.ref_s_index]
        else:
            ref_s = voice_pack[len(phonemes) - 1]

        ref_s_np = to_ort_array(ref_s.numpy(), input_types.get("ref_s", "fp32"))
        speed_np = to_ort_array(args.speed, input_types.get("speed", "fp32"))

        ort_inputs = {
            "input_ids": input_ids,
            "ref_s": ref_s_np,
            "speed": speed_np,
        }
        outputs = sess.run(output_names, ort_inputs)
        waveform = outputs[0]
        if waveform.ndim > 1:
            waveform = np.squeeze(waveform)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)
        if wavs and args.silence > 0:
            wavs.append(np.zeros(args.silence, dtype=np.float32))
        wavs.append(waveform)

    if not wavs:
        raise SystemExit("No audio produced. Check input text and G2P output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path.as_posix(), np.concatenate(wavs), SAMPLE_RATE)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
