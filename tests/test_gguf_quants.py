"""Unit tests for the extended GGUF dequantizers (legacy / K / i-quants).

The block layouts are checked against hand-built blocks whose dequantized values
are known, so a wrong bit layout (which produces silently garbage weights) is
caught without needing a real model file.

Run: .venv\\Scripts\\python.exe tests\\test_gguf_quants.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gguf  # noqa: E402

from engine import gguf as gguf_reader  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def close(name, got, want):
    if got.shape != want.shape or not torch.allclose(got.float(), want.float(), atol=1e-3):
        FAILED.append(f"{name}: values differ got {got.tolist()} want {want.tolist()}")
    else:
        print(f"ok  {name}")


class FakeTensor:
    """The slice of the gguf tensor object ``engine.gguf.dequantize`` reads."""

    def __init__(self, raw_bytes, gguf_shape, ttype):
        self.data = raw_bytes
        self.shape = gguf_shape
        self.tensor_type = ttype


def _q4_0_block(nibble, scale):
    d = np.float16(scale).tobytes()
    packed = bytes([nibble | (nibble << 4)] * 16)
    return np.frombuffer(d + packed, dtype=np.uint8)


def _q8_0_block(values, scale):
    d = np.float16(scale).tobytes()
    body = np.array(values, dtype=np.int8).tobytes()
    return np.frombuffer(d + body, dtype=np.uint8)


def test_q4_0_layout():
    t = FakeTensor(_q4_0_block(9, 1.0), (32, 1), gguf.GGMLQuantizationType.Q4_0)
    out = gguf_reader.dequantize(t)
    close("q4_0 nibble 9 scale 1 -> ones", out, torch.ones(1, 32))

    t = FakeTensor(_q4_0_block(8, 1.0), (32, 1), gguf.GGMLQuantizationType.Q4_0)
    close("q4_0 nibble 8 -> zeros", gguf_reader.dequantize(t), torch.zeros(1, 32))

    t = FakeTensor(_q4_0_block(15, 2.0), (32, 1), gguf.GGMLQuantizationType.Q4_0)
    close("q4_0 nibble 15 scale 2 -> 14s", gguf_reader.dequantize(t),
          torch.full((1, 32), 14.0))


def test_q5_1_layout():
    # 32 values, d=1.0, m=0.5: byte layout is d(2) m(2) qh(4) qs(16).
    d = np.float16(1.0).tobytes()
    m = np.float16(0.5).tobytes()
    qh = bytes([0xFF] * 4)          # every high bit set -> +16
    qs = bytes([0x00] * 16)         # every low nibble 0
    raw = np.frombuffer(d + m + qh + qs, dtype=np.uint8)
    t = FakeTensor(raw, (32, 1), gguf.GGMLQuantizationType.Q5_1)
    out = gguf_reader.dequantize(t)
    close("q5_1 high bits add 16, plus min", out, torch.full((1, 32), 16.5))


def test_q8_0_layout():
    t = FakeTensor(_q8_0_block([4] * 32, 0.5), (32, 1),
                   gguf.GGMLQuantizationType.Q8_0)
    close("q8_0 -> 2.0", gguf_reader.dequantize(t), torch.full((1, 32), 2.0))


def test_f32_round_trip():
    values = np.array([[1.5, -2.0, 3.25]], dtype=np.float32)
    t = FakeTensor(values, (3, 1), gguf.GGMLQuantizationType.F32)
    out = gguf_reader.dequantize(t)
    check("f32 shape", tuple(out.shape), (1, 3))
    close("f32 values", out, torch.tensor([[1.5, -2.0, 3.25]]))


def test_int_type_passthrough():
    values = np.array([1, -2, 3, -4], dtype=np.int8)
    t = FakeTensor(values, (4,), gguf.GGMLQuantizationType.I8)
    out = gguf_reader.dequantize(t, dtype=torch.float32)
    close("i8 cast", out, torch.tensor([1.0, -2.0, 3.0, -4.0]))


def test_registry_has_new_quants():
    names = {t.name for t in gguf_reader._DEQUANT}
    for wanted in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "IQ4_NL", "IQ4_XS"):
        check(f"registered {wanted}", wanted in names, True)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()