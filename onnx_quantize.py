"""Quantize a Mamba ONNX model to INT8 using dynamic quantization.

Usage:
    python onnx_quantize.py --input mamba.onnx --output mamba_quantized.onnx

INT8 dynamic quantization keeps activations in float32 and quantizes
weights only, so no calibration data is needed. Typical size reduction
on a model this size is 3-4x.
"""

import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(input_path: str, output_path: str) -> None:
    size_before = os.path.getsize(input_path)

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    size_after = os.path.getsize(output_path)
    print(f"Original : {size_before / 1e6:.2f} MB")
    print(f"Quantized: {size_after / 1e6:.2f} MB  ({size_before / size_after:.1f}x smaller)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="mamba.onnx")
    parser.add_argument("--output", default="mamba_quantized.onnx")
    args = parser.parse_args()

    quantize(args.input, args.output)
