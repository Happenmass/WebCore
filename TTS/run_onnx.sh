#!/usr/bin/env bash
# set -euo pipefail

# Export fp32 ONNX
# python TTS/export_kokoro_onnx.py --output TTS/kokoro_fp32.onnx

# Run ONNX inference (edit --text/--voice as needed)
python TTS/onnx_infer.py \
  --onnx TTS/kokoro_fp32.onnx \
  --voice-path TTS/hexgrad/Kokoro-82M-v1.1-zh/voices/zf_001.pt \
  --g2p \
  --text "Kokoro 是一系列体积虽小但功能强大的 TTS 模型。" \
  --output TTS/onnx_output.wav
