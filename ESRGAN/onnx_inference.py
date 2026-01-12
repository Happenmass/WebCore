import argparse
import os

import numpy as np
from PIL import Image
import torch

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc

from RealESRGAN import RealESRGAN
from RealESRGAN.model import (
    pad_reflect,
    split_image_into_overlapping_patches,
    stich_together,
    unpad_image,
)


def export_onnx(onnx_path, weights_path, scale=4, opset=11):
    device = torch.device("cpu")
    model = RealESRGAN(device, scale=scale)
    model.load_weights(weights_path, download=False)
    net = model.model
    net.eval()

    dummy = torch.randn(1, 3, 64, 64, device=device)
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height_out", 3: "width_out"},
    }

    export_kwargs = dict(
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    try:
        # Prefer legacy exporter to avoid onnxscript version conversion issues.
        torch.onnx.export(net, dummy, onnx_path, **export_kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(net, dummy, onnx_path, **export_kwargs)


def run_onnx_inference(
    session,
    image,
    scale=4,
    batch_size=4,
    patches_size=192,
    padding=24,
    pad_size=15,
):
    lr_image = np.array(image)
    lr_image = pad_reflect(lr_image, pad_size)

    patches, p_shape = split_image_into_overlapping_patches(
        lr_image, patch_size=patches_size, padding_size=padding
    )
    img = (patches.astype(np.float32) / 255.0).transpose((0, 3, 1, 2))

    outputs = []
    for i in range(0, img.shape[0], batch_size):
        batch = img[i : i + batch_size]
        out = session.run(None, {"input": batch})[0]
        outputs.append(out)

    res = np.concatenate(outputs, axis=0)
    res = np.clip(res, 0, 1).transpose((0, 2, 3, 1))

    padded_size_scaled = (p_shape[0] * scale, p_shape[1] * scale, 3)
    scaled_image_shape = (lr_image.shape[0] * scale, lr_image.shape[1] * scale, 3)
    np_sr_image = stich_together(
        res,
        padded_image_shape=padded_size_scaled,
        target_shape=scaled_image_shape,
        padding_size=padding * scale,
    )
    sr_img = (np_sr_image * 255.0).astype(np.uint8)
    sr_img = unpad_image(sr_img, pad_size * scale)
    return Image.fromarray(sr_img)


def build_session(onnx_path):
    providers = ["CPUExecutionProvider"]
    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ort.InferenceSession(onnx_path, providers=providers)


def parse_args():
    parser = argparse.ArgumentParser(description="Export ESRGAN to ONNX and run inference.")
    parser.add_argument("--input", default="inputs/lr_image.jpeg", help="Path to input image")
    parser.add_argument("--output", default="results/sr_image_onnx.png", help="Path to output image")
    parser.add_argument("--weights", default="weights/RealESRGAN_x4.pth", help="Path to .pth weights")
    parser.add_argument("--onnx", default="weights/RealESRGAN_x4.onnx", help="Path to output ONNX model")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--export-only", action="store_true", help="Only export ONNX model")
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force re-export even if ONNX already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_export or not os.path.exists(args.onnx):
        export_onnx(args.onnx, args.weights, scale=args.scale, opset=args.opset)

    if args.export_only:
        return

    session = build_session(args.onnx)
    image = Image.open(args.input).convert("RGB")
    sr_image = run_onnx_inference(session, image, scale=args.scale)
    sr_image.save(args.output)


if __name__ == "__main__":
    main()
