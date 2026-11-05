"""Prompt emphasis weights: ``(text:weight)``, ``(text)``.

Ported from ComfyUI's ``token_weights`` / ``parse_parentheses``
(``comfy/sd1_clip.py``), so a prompt written for ComfyUI or A1111 means the same
thing here:

* ``(text)`` raises emphasis by 1.1,
* ``(text:2)`` sets the weight to 2 (any float),
* ``\\`` ``\\(`` ``\\)`` escape a literal backslash or round bracket,
* nesting multiplies, and outer text keeps its weight.

Emphasis does not change *which* tokens the tokenizer produces: the prompt text
is tokenized with the brackets stripped, and each token's hidden state is pulled
away from the empty-prompt hidden state at the same sequence position, matching
ComfyUI's ``ClipTokenWeightEncoder.encode_token_weights``::

    out = empty + (out - empty) * weight

Anima is the exception: like ComfyUI, it keeps the Qwen3 token weights at 1.0 and
multiplies the returned weights onto the LLM-adapter output instead (see
:func:`engine.anima.preprocess_text_embeds`).

``weight == 1`` is therefore a no-op, so a prompt with no emphasis encodes
exactly as it did before (the empty pass is skipped entirely).

The public helpers are:

* :func:`parse` / :func:`plain` / :func:`has_weights` -- split a prompt into
  ``(text, weight)`` pairs, rejoin it, or test whether any weight is set;
* :func:`token_weights` -- a per-token weight list aligned to a tokenizer's
  output (``add_special_tokens=False`` content ids, placed inside the ids the
  encoder actually feeds the model);
* :func:`fit_ids` -- pad/truncate an id list to a fixed length (the empty pass);
* :func:`interpolate` -- apply the ComfyUI blend to an encoder output tensor.
"""

import torch

_ESCAPED_CLOSE = "\x00\x01"
_ESCAPED_OPEN = "\x00\x02"


def _escape(text):
    return text.replace("\\)", _ESCAPED_CLOSE).replace("\\(", _ESCAPED_OPEN)


def _unescape(text):
    return text.replace(_ESCAPED_CLOSE, ")").replace(_ESCAPED_OPEN, "(")


def _parse_parentheses(string):
    """Group top-level ``(...)`` spans, tracking nesting (ComfyUI verbatim)."""
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _token_weights(string, current_weight):
    """``str -> [(text, weight), ...]`` (ComfyUI ``token_weights`` verbatim)."""
    out = []
    for item in _parse_parentheses(string):
        weight = current_weight
        if len(item) >= 2 and item[-1] == ")" and item[0] == "(":
            item = item[1:-1]
            colon = item.rfind(":")
            weight *= 1.1
            if colon > 0:
                try:
                    weight = float(item[colon + 1:])
                    item = item[:colon]
                except ValueError:
                    pass
            out += _token_weights(item, weight)
        else:
            out += [(item, current_weight)]
    return out


def parse(text):
    """Split ``text`` into ``[(plain_text, weight), ...]``.

    The pairs concatenate back to the prompt with the emphasis syntax removed;
    escaped brackets are restored to literal ``(`` / ``)``.
    """
    pairs = _token_weights(_escape(text or ""), 1.0)
    return [(_unescape(segment), weight) for segment, weight in pairs]


def plain(pairs):
    """The prompt text with every ``(text:weight)`` wrapper stripped."""
    return "".join(segment for segment, _ in pairs)


def has_weights(pairs):
    """True when any pair carries a weight other than 1.0."""
    return any(weight != 1.0 for _, weight in pairs)


def _find(haystack, needle):
    """Index of the first contiguous ``needle`` in ``haystack``, or None."""
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return None
    first = needle[0]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] == first and haystack[start:start + len(needle)] == needle:
            return start
    return None


def _boundary(tokenizer, ids, full_text, char_pos):
    """Number of ``ids`` tokens that lie before character ``char_pos``.

    Tokenizing a prefix on its own can split a trailing space into its own token
    that merges with the next word in the full string (``" red"``), which makes a
    plain length difference skip that word. Comparing the prefix's token ids
    against the full ids instead reports the true shared prefix, so the boundary
    survives that merge.
    """
    prefix = list(tokenizer(full_text[:char_pos], add_special_tokens=False)["input_ids"])
    index = 0
    while index < len(prefix) and index < len(ids) and prefix[index] == ids[index]:
        index += 1
    return index


def token_weights(tokenizer, full_text, pairs, prompt_start=0, target_ids=None):
    """Per-token weights for ``pairs`` within ``full_text``.

    ``full_text`` is the exact string handed to the encoder (a chat template plus
    the plain prompt, for example); ``prompt_start`` is the character offset where
    the weighted prompt begins. ``target_ids`` are the ids the encoder feeds the
    model (specials and padding included); when omitted the tokenizer is asked for
    the plain ``add_special_tokens=False`` ids.

    Returns one weight per entry of ``target_ids`` (1.0 outside the prompt). The
    prompt's own tokens are found inside ``target_ids`` as a contiguous run, so
    BOS/EOS/padding and any template markers stay at weight 1.0.
    """
    content = list(tokenizer(full_text, add_special_tokens=False)["input_ids"])

    weights = [1.0] * len(content)
    position = prompt_start
    for segment, weight in pairs:
        start, end = position, position + len(segment)
        position = end
        if weight == 1.0 or not segment:
            continue
        lead = _boundary(tokenizer, content, full_text, start)
        tail = _boundary(tokenizer, content, full_text, end)
        for index in range(lead, min(tail, len(weights))):
            weights[index] *= weight

    if target_ids is None:
        target_ids = content
    # Locate the content run inside the ids actually fed to the model. Trimming
    # the tail covers prompts the encoder truncated.
    for length in range(len(content), -1, -1):
        offset = _find(list(target_ids), content[:length])
        if offset is None:
            continue
        out = [1.0] * len(target_ids)
        for index in range(length):
            out[offset + index] = weights[index]
        return out
    return [1.0] * len(target_ids)


def fit_ids(ids, length, pad):
    """Return ``ids`` resized to exactly ``length`` with ``pad`` appended."""
    ids = list(ids)[:length]
    return ids + [pad] * (length - len(ids))


def interpolate(out, empty, weights):
    """``empty + (out - empty) * weights`` over the sequence dimension.

    ``out`` and ``empty`` are ``[B, T, C]`` (or any ``[B, T, ...]``) tensors of
    matching shape; ``weights`` is the per-token list from :func:`token_weights`.
    Missing positions default to 1.0 (a no-op).
    """
    width = out.shape[1]
    w = torch.as_tensor(weights, device=out.device, dtype=out.dtype).reshape(1, -1, 1)
    if w.shape[1] < width:
        w = torch.cat([
            w,
            torch.ones(1, width - w.shape[1], 1, device=out.device, dtype=out.dtype),
        ], dim=1)
    else:
        w = w[:, :width]
    return empty + (out - empty) * w
