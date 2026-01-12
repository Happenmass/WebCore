#!/usr/bin/env python
# Export a Kokoro KModel to ONNX by capturing real inputs from KPipeline.

import argparse
import inspect
from pathlib import Path

import torch

try:
    from kokoro import KModel, KPipeline
except Exception as exc:
    raise SystemExit(
        "Missing dependency: kokoro. Install with "
        "`pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"`."
    ) from exc


def build_pipeline(repo_id, model):
    def en_callable(text):
        return next(en_pipeline(text)).phonemes

    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)
    try:
        return KPipeline(
            lang_code="z",
            repo_id=repo_id,
            model=model,
            en_callable=en_callable,
        )
    except TypeError:
        return KPipeline(lang_code="z", repo_id=repo_id, model=model)


def get_first_phonemes(pipeline, text):
    for result in pipeline(text):
        if result.phonemes:
            return result.phonemes
    raise RuntimeError("No phonemes produced from input text.")


def phonemes_to_input_ids(model, phonemes):
    input_ids = list(filter(lambda i: i is not None, map(model.vocab.get, phonemes)))
    if len(input_ids) + 2 > model.context_length:
        raise ValueError(
            f"Input too long: {len(input_ids)+2} > {model.context_length}"
        )
    return torch.LongTensor([[0, *input_ids, 0]])


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    return value


def force_custom_stft(model):
    try:
        from kokoro.custom_stft import CustomSTFT
        from kokoro.istftnet import TorchSTFT
    except Exception:
        return

    for module in model.modules():
        if hasattr(module, "stft") and isinstance(module.stft, TorchSTFT):
            stft = module.stft
            module.stft = CustomSTFT(
                filter_length=stft.filter_length,
                hop_length=stft.hop_length,
                win_length=stft.win_length,
            )


def main():
    parser = argparse.ArgumentParser(description="Export Kokoro KModel to ONNX.")
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh")
    parser.add_argument("--voice", default="zf_001")
    parser.add_argument("--text", default="Kokoro export example.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="kokoro.onnx")
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--enable-complex", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    disable_complex = not args.enable_complex

    pipeline = build_pipeline(args.repo_id, model=False)
    phonemes = get_first_phonemes(pipeline, args.text)
    voice_pack = pipeline.load_voice(args.voice).cpu()
    ref_s = voice_pack[len(phonemes) - 1]
    speed = args.speed

    try:
        model = KModel(repo_id=args.repo_id, disable_complex=disable_complex).to(device).eval()
        if disable_complex:
            try:
                force_custom_stft(model)
            except Exception:
                pass
        input_ids = phonemes_to_input_ids(model, phonemes)
        speed_tensor = torch.tensor(float(speed), dtype=torch.float32)
        example_args = (to_cpu(input_ids), to_cpu(ref_s), to_cpu(speed_tensor))
        example_kwargs = {}
        model = model.to("cpu").eval()
        example_args = (
            example_args[0],
            example_args[1].to(torch.float32),
            example_args[2].to(torch.float32),
        )

        output_names = ["waveform", "duration"]

        input_names = ["input_ids", "ref_s", "speed"]

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class OnnxWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *a, **kw):
                return self.inner.forward_with_tokens(*a, **kw)

        wrapper = OnnxWrapper(model)

        dynamic_axes = {
            "input_ids": {1: "text_len"},
            "waveform": {0: "audio_len"},
            "duration": {0: "text_len"},
        }

        if args.dynamo:
            try:
                export_options = torch.onnx.ExportOptions(
                    opset_version=args.opset
                )
            except Exception:
                export_options = None
            try:
                onnx_program = torch.onnx.dynamo_export(
                    wrapper,
                    *example_args,
                    **example_kwargs,
                    export_options=export_options,
                )
            except TypeError:
                onnx_program = torch.onnx.dynamo_export(
                    wrapper,
                    *example_args,
                    **example_kwargs,
                )
            onnx_program.save(output_path.as_posix())
        else:
            torch.onnx.export(
                wrapper,
                example_args,
                output_path.as_posix(),
                kwargs=example_kwargs,
                opset_version=args.opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        sig = inspect.signature(model.forward)
        print(f"Exported ONNX to {output_path}")
        print(f"Model forward signature: {sig}")
    finally:
        pass


if __name__ == "__main__":
    main()
