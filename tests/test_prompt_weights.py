"""Unit tests for engine.prompt_weights (run: .venv\\Scripts\\python.exe tests\\test_prompt_weights.py)."""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import prompt_weights as pw  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_parse():
    check("plain", pw.parse("a photo of a cat"), [("a photo of a cat", 1.0)])
    check("set weight", pw.parse("(anime:2)"), [("anime", 2.0)])
    check("float weight", pw.parse("(red:1.5)"), [("red", 1.5)])
    check("default emphasis", pw.parse("(cat)"), [("cat", 1.1)])
    check("nested default", pw.parse("((cat))"), [("cat", 1.1 * 1.1)])
    check("mixed", pw.parse("a (red:1.5) car"),
          [("a ", 1.0), ("red", 1.5), (" car", 1.0)])
    check("escape open", pw.parse("\\(literal\\)"), [("(literal)", 1.0)])
    check("text around set weight", pw.parse("(a red:2) car"),
          [("a red", 2.0), (" car", 1.0)])
    check("plain()", pw.plain(pw.parse("a (red:1.5) car")), "a red car")


def test_has_weights():
    check("no weights", pw.has_weights(pw.parse("a cat")), False)
    check("one weight", pw.has_weights(pw.parse("a (cat:1.2)")), True)
    check("explicit one", pw.has_weights(pw.parse("a (cat)")), True)


def test_token_weights_qwen():
    from transformers import Qwen2Tokenizer

    tok = Qwen2Tokenizer.from_pretrained(
        os.path.join(ROOT, "engine", "tokenizer_data")
    )
    pairs = pw.parse("a (red:1.5) car")
    clean = pw.plain(pairs)
    ids = list(tok(clean)["input_ids"])
    weights = pw.token_weights(tok, clean, pairs, target_ids=ids)
    check("qwen len", len(weights), len(ids))
    # "red" carries its leading space, so it is a single " red" token.
    check("qwen tokens", [tok.decode([t]) for t in ids], ["a", " red", " car"])
    check("qwen weights", weights, [1.0, 1.5, 1.0])


def test_token_weights_clip_padding():
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    pairs = pw.parse("a (red:3) car")
    clean = pw.plain(pairs)
    ids = list(tok(clean, padding="max_length", max_length=77,
                   truncation=True)["input_ids"])
    weights = pw.token_weights(tok, clean, pairs, target_ids=ids)
    check("clip padded len", len(weights), 77)
    check("clip bos weight", weights[0], 1.0)
    check("clip pad weight", weights[-1], 1.0)
    red_at = pw._find(ids, list(tok("red", add_special_tokens=False)["input_ids"]))
    check("clip red weight", weights[red_at], 3.0)


def test_interpolate():
    out = torch.ones(1, 3, 2)
    empty = torch.zeros(1, 3, 2)
    blended = pw.interpolate(out, empty, [1.0, 2.0, 1.0])
    check("blend values", blended.flatten().tolist(), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0])

    # weight 1 is a no-op even when out and empty differ.
    other = torch.full((1, 3, 1), 7.0)
    check("weight-1 no-op", pw.interpolate(out, other, [1.0, 1.0, 1.0]).flatten().tolist(),
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # short weight list is padded with 1.0.
    check("short weights", pw.interpolate(out, empty, [2.0]).flatten().tolist(),
          [2.0, 2.0, 1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    test_parse()
    test_has_weights()
    test_token_weights_qwen()
    test_token_weights_clip_padding()
    test_interpolate()
    if FAILED:
        print("\nFAILURES:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall prompt-weight tests passed")
