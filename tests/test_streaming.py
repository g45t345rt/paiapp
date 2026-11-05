"""Unit tests for the disk-streaming facility (engine.streaming).

A tiny transformer is built on the meta device and its blocks are materialised
from a fake, in-memory "checkpoint" source, exercising the exact hook, prefetch
and evict path the real GGUF loader uses -- without a model file or a GPU.

Run: .venv\\Scripts\\python.exe tests\\test_streaming.py
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import streaming  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.lin(x))


class Tiny(nn.Module):
    def __init__(self, dim=8, blocks=3):
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(blocks)])
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


class FakeSource:
    """The subset of the streaming reader interface ``DiskStreamer`` calls."""

    def __init__(self, state):
        self.state = {name: tensor.clone() for name, tensor in state.items()}
        self.names = set(self.state)
        self.reads = []

    def has(self, model_name):
        return model_name in self.state

    def tensor(self, model_name, dtype=torch.float32):
        value = self.state.get(model_name)
        return None if value is None else value.to(dtype)

    def get_many(self, names, dtype=torch.float32):
        names = list(names)
        self.reads.append(tuple(names))
        return {name: self.state[name].to(dtype) for name in names if name in self.state}

    def tensor_bytes(self, model_name):
        value = self.state.get(model_name)
        return 0 if value is None else value.numel() * value.element_size()

    def block_bytes(self, names):
        return sum(self.tensor_bytes(name) for name in names)

    def close(self):
        pass


def _dense_reference():
    torch.manual_seed(0)
    model = Tiny().eval()
    state = {name: param.detach().clone() for name, param in model.named_parameters()}
    return model, state


def test_shard_order_finds_blocks():
    model, _state = _dense_reference()
    order = streaming.shard_order(model)
    prefixes = [name for name, _module in order]
    check("three blocks", prefixes, ["blocks.0", "blocks.1", "blocks.2"])


def test_rate_control_helpers():
    check("no loras", streaming.can_stream("model.gguf", loras=[]), True)
    check("with lora", streaming.can_stream("model.gguf", loras=["x"]), False)
    check("safetensors", streaming.can_stream("model.safetensors"), True)
    check("unsupported", streaming.can_stream("model.ckpt"), False)


def test_qwen_fused_name_map():
    from engine import qwen_image

    name_map = qwen_image._stream_name_map("model.diffusion_model.")
    fused = name_map("model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight")
    check("fused maps to two params", len(fused), 2)
    check("fused names", [name for name, _slice in fused], [
        "transformer_blocks.0.img_mlp.gate_layer.weight",
        "transformer_blocks.0.img_mlp.proj.weight",
    ])
    full = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    gate = fused[0][1](full)
    up = fused[1][1](full)
    check("gate takes first half", torch.equal(gate, full[:2]), True)
    check("proj takes second half", torch.equal(up, full[2:]), True)
    check("identity after prefix", name_map("model.diffusion_model.img_in.weight"), "img_in.weight")
    check("foreign prefix skipped", name_map("other.img_in.weight"), None)


def test_streaming_matches_dense():
    dense, state = _dense_reference()
    source = FakeSource(state)
    x = torch.randn(2, 8)

    with torch.no_grad():
        want = dense(x)

    model, streamer = streaming.build_streaming_model(
        lambda: Tiny(), source, device="cpu", dtype=torch.float32,
    )

    # Before the first forward every parameter is still on the meta device.
    on_meta = all(p.device.type == "meta" for p in model.parameters())
    check("built on meta", on_meta, True)
    check("armed for place_dit", model._streaming_source is source, True)

    with torch.no_grad():
        got = model(x)
    if not torch.allclose(got, want, atol=1e-5):
        FAILED.append("streamed forward differs from dense")
    else:
        print("ok  streamed forward matches dense")

    # Resident weights stay; block weights are evicted back to meta after use.
    blocks_meta = all(
        p.device.type == "meta"
        for _name, module in streamer.order
        for p in module.parameters()
    )
    check("blocks evicted after pass", blocks_meta, True)
    resident_on_cpu = all(
        dict(model.named_parameters())[name].device.type == "cpu"
        for name in ("inp.weight", "out.weight")
    )
    check("resident weights present", resident_on_cpu, True)

    # A second pass re-materialises and gives the same result.
    with torch.no_grad():
        again = model(x)
    if not torch.allclose(again, want, atol=1e-5):
        FAILED.append("second streamed pass differs")
    else:
        print("ok  second pass re-materialises correctly")


def test_callable_source_reader():
    dense, state = _dense_reference()
    x = torch.randn(2, 8)
    with torch.no_grad():
        want = dense(x)

    reader = streaming.CallableSource(
        state.keys(),
        lambda name: state[name].clone(),
        bytes_of=lambda name: state[name].numel() * state[name].element_size(),
    )
    model, streamer = streaming.build_streaming_model(
        lambda: Tiny(), reader, device="cpu", dtype=torch.float32,
    )
    with torch.no_grad():
        got = model(x)
    check("callable source forward matches dense", torch.allclose(got, want, atol=1e-5), True)
    streamer.close()


def test_evict_and_release():
    _dense, state = _dense_reference()
    source = FakeSource(state)
    model, streamer = streaming.build_streaming_model(
        lambda: Tiny(), source, device="cpu", dtype=torch.float32,
    )
    with torch.no_grad():
        model(torch.randn(1, 8))
    released = streamer.release_cached_blocks()
    check("release reports a count", released >= 0, True)
    streamer.close()


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