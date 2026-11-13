"""Hugging Face Hub access for browsing and downloading models.

The Model Settings dialog lets the user point the app at their own model files;
this module is the optional network half, so a model can be found and fetched
without leaving the app. It talks to the public Hub REST API
(``https://huggingface.co/api``) with only the standard library, so nothing here
adds a dependency: search repos, list the files in a repo, then stream one file
into a local folder.

Public repos need no credentials. When ``HF_TOKEN`` (or the older
``HUGGING_FACE_HUB_TOKEN``) is set in the environment it is sent as a bearer
token, which also unlocks gated repos the user already has access to.

Only :func:`search_models`, :func:`model_summary`, :func:`list_files`,
:func:`download_file` and the name helpers (:func:`suggest_filename`,
:func:`safe_name`, :func:`unique_name`, :func:`is_generic_filename`) are meant to
be called from the UI; everything else is an implementation detail. Failures
raise :class:`HubError` with a short, user-readable message.
"""

import json
import os
import re
import shutil
import zipfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .cancel import CancelToken, Cancelled

ENDPOINT = "https://huggingface.co"

USER_AGENT = "Pai/1.0 (model browser)"

# Sort keys the Hub accepts, paired with the label the dialog shows. ``lastModified``
# and ``createdAt`` arrive as ISO timestamps; the rest are plain counts.
SORTS = (
    ("Downloads", "downloads"),
    ("Likes", "likes"),
    ("Trending", "trendingScore"),
    ("Recently updated", "lastModified"),
    ("Newest", "createdAt"),
)

DEFAULT_SORT = "downloads"

# The Hub only honours an ascending ``direction`` for the "Newest" sort; every
# other key rejects ``direction=1`` with a 400 ("only descending sort is
# supported for ..."). Descending is the server default, so we simply omit the
# parameter unless the user asked for it.
ASCENDING_SORTS = frozenset({"createdAt"})

# Optional tag filters, paired with their display label. ``None`` means "no
# filter". These are the tags that line up with the families the engine runs.
FILTERS = (
    ("Any model", None),
    ("Text to image", "text-to-image"),
    ("Diffusers", "diffusers"),
    ("GGUF", "gguf"),
    ("Safetensors", "safetensors"),
    ("LoRA", "lora"),
    ("ControlNet", "controlnet"),
)

# File suffixes the picker treats as model files, used by the dialog's "model
# files only" filter. ``.zip`` is included because several model families (the
# SAM segmentation exports, for one) publish their weights only as an archive
# bundle; the dialog unpacks such a download after fetching it.
MODEL_SUFFIXES = (
    ".safetensors", ".ckpt", ".gguf", ".onnx", ".bin", ".pt", ".pth", ".zip",
)

# Weight file names Hugging Face uses for the "real" weights, so unrelated repos
# commonly both offer ``model.safetensors``. A file whose stem is one of these
# (optionally numbered, e.g. ``pytorch_model_00001``) is treated as generic and
# renamed from its repo id so a download stays identifiable in a full folder.
_GENERIC_STEM = re.compile(
    r"(?:model|models|pytorch_model|tf_model|flax_model|diffusion_pytorch_model|"
    r"diffusion_model|model_state_dict|model_weights|consolidated|unet|weights|"
    r"checkpoint)(?:_\d+)?",
    re.IGNORECASE,
)

# Characters Windows (and, for the most part, POSIX) disallow in a file name.
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_CHUNK = 1024 * 1024


class HubError(Exception):
    """A Hugging Face request failed in a way worth showing the user."""


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def _request(url):
    """A ``Request`` with the headers the Hub expects (and a token, if any)."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    token = _token()
    if token:
        headers["Authorization"] = "Bearer " + token
    return Request(url, headers=headers)


def _get_json(path, timeout=30):
    """GET ``path`` from the Hub and decode it as JSON.

    ``path`` may be relative (``/api/models``) or absolute (a resolve URL) so the
    same error handling covers the API and the download probe.
    """
    url = path if path.startswith("http") else ENDPOINT + path
    try:
        with urlopen(_request(url), timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        if exc.code == 401:
            raise HubError("This repo needs a token (set HF_TOKEN).") from exc
        if exc.code == 403:
            raise HubError("Access to this repo is gated.") from exc
        if exc.code == 404:
            raise HubError("Repo or file not found (404).") from exc
        raise HubError(f"{exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise HubError(f"Network error: {exc.reason}") from exc
    try:
        return json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HubError("Unexpected response from Hugging Face.") from exc


def search_models(query="", sort=DEFAULT_SORT, descending=True, tag=None,
                  limit=50, cursor=None):
    """Search repos and return the raw Hub result list.

    ``query`` is a substring match on the repo id, ``sort`` is one of the keys in
    :data:`SORTS`, ``descending`` flips the direction, and ``tag`` is an optional
    tag filter from :data:`FILTERS`. ``cursor`` continues a previous page. Gated
    repos are excluded, so every result can be downloaded without an account.
    The raw dicts are passed to :func:`model_summary` for display.

    The Hub only sorts ascending for the "Newest" key (:data:`ASCENDING_SORTS`);
    the other keys reject ``direction=1`` with a 400, so ascending is refused
    here with a :class:`HubError` instead of surfacing the server error.
    """
    params = {"limit": int(limit), "sort": sort, "gated": "false"}
    if descending:
        params["direction"] = -1
    elif sort in ASCENDING_SORTS:
        params["direction"] = 1
    else:
        raise HubError(
            "Hugging Face only sorts ascending by Newest; use descending for"
            " this sort."
        )
    if query:
        params["search"] = query
    if tag:
        params["filter"] = tag
    if cursor:
        params["cursor"] = cursor
    data = _get_json("/api/models?" + urlencode(params))
    if not isinstance(data, list):
        return []
    # ``gated=false`` already filters server-side; drop any gated row anyway so a
    # change in that behavior cannot surface a repo the user cannot fetch.
    return [item for item in data if not item.get("gated")]


def model_summary(item):
    """Normalise one raw Hub result into the fields the dialog shows."""
    return {
        "id": item.get("id") or item.get("modelId") or "",
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "trending": int(item.get("trendingScore") or 0),
        "updated": item.get("lastModified") or item.get("createdAt") or "",
        "pipeline": item.get("pipeline_tag") or "",
        "library": item.get("library_name") or "",
        "tags": [str(tag) for tag in (item.get("tags") or [])],
        "gated": item.get("gated"),
    }


def list_files(repo, revision="main"):
    """Return ``[{"path", "size"}, ...]`` for the files in ``repo``.

    Only files are returned (directories are dropped). Sizes come from the LFS
    metadata when present, so a weight file reports its real size rather than the
    few bytes of its pointer.
    """
    path = f"/api/models/{quote(repo, safe='/')}/tree/{quote(revision, safe='')}"
    data = _get_json(path + "?recursive=true", timeout=60)
    files = []
    for entry in data if isinstance(data, list) else []:
        if entry.get("type") != "file":
            continue
        if "lfs" in entry and isinstance(entry["lfs"], dict):
            size = entry["lfs"].get("size")
        else:
            size = entry.get("size")
        files.append({"path": entry.get("path") or "", "size": int(size or 0)})
    return files


def download_url(repo, filename, revision="main"):
    """The resolve URL a browser would download a repo file from."""
    return (
        f"{ENDPOINT}/{quote(repo, safe='/')}"
        f"/resolve/{quote(revision, safe='')}/{quote(filename, safe='/')}?download=true"
    )


def safe_name(name):
    """A single file-name segment with no directory parts or illegal characters.

    Used for every name that reaches disk, so a manual rename (or a crafted repo
    path) cannot escape the destination folder. Returns ``""`` when nothing
    usable is left.
    """
    base = os.path.basename(os.fspath(name or "").strip())
    return _INVALID_NAME.sub("_", base).strip(" .")


def is_generic_filename(filename):
    """True when ``filename`` is a generic weight name (see :data:`_GENERIC_STEM`)."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    return bool(_GENERIC_STEM.fullmatch(stem))


def suggest_filename(repo, filename):
    """A more identifiable destination name for a repo file.

    A generic weight name (``model.safetensors``) is replaced with
    ``<repo>.safetensors`` so the saved file is recognizable in a folder full of
    models and unlike-named repos do not collide. When the repo's own name is
    generic too the owner is prefixed. Any other file name is returned unchanged.
    """
    base = os.path.basename(filename or "")
    if not is_generic_filename(base):
        return base
    ext = os.path.splitext(base)[1]
    repo_name = (repo or "").rsplit("/", 1)[-1]
    if is_generic_filename(repo_name + ext):
        owner = repo.split("/", 1)[0] if "/" in repo else ""
        repo_name = f"{owner}_{repo_name}" if owner else repo_name
    stem = safe_name(repo_name)
    return stem + ext if stem else base


def unique_name(name, taken=()):
    """``name``, or ``stem (n).ext``, not already in ``taken`` (case-insensitive)."""
    used = {os.fspath(item).lower() for item in taken}
    if os.fspath(name).lower() not in used:
        return name
    stem, ext = os.path.splitext(name)
    index = 1
    while f"{stem} ({index}){ext}".lower() in used:
        index += 1
    return f"{stem} ({index}){ext}"


def download_file(repo, filename, dest_dir, name=None, progress=None, cancel=None,
                  revision="main"):
    """Stream one repo file into ``dest_dir`` and return the saved path.

    The bytes are written to ``<name>.part`` and renamed on success, so an
    interrupted or cancelled download never leaves a half file where the engine
    would scan it. ``name`` overrides the on-disk file name (a rename); without
    it the repo file's own base name is used. ``progress(frac, label, read,
    total)`` is called as data arrives (``frac`` is 0..1, or 0.0 when the server
    gives no length; ``read`` and ``total`` are byte counts, ``total`` 0 when
    unknown) and ``cancel`` is a :class:`~engine.cancel.CancelToken` checked
    between chunks.
    """
    if cancel is None:
        cancel = CancelToken()
    url = download_url(repo, filename, revision)
    name = safe_name(name) or safe_name(filename) or "download"
    target = os.path.join(dest_dir, name)
    os.makedirs(dest_dir, exist_ok=True)
    part = target + ".part"
    try:
        with urlopen(_request(url), timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            with open(part, "wb") as handle:
                while True:
                    cancel.throw_if_cancelled()
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    read += len(chunk)
                    if progress is not None:
                        progress(read / total if total else 0.0, name, read, total)
        if progress is not None:
            progress(1.0, name, read, total)
        os.replace(part, target)
    except Cancelled:
        _discard(part)
        raise
    except HTTPError as exc:
        _discard(part)
        if exc.code in (401, 403):
            raise HubError("This file needs a token or gated access.") from exc
        raise HubError(f"Download failed: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        _discard(part)
        raise HubError(f"Download failed: {exc.reason}") from exc
    except OSError as exc:
        _discard(part)
        raise HubError(f"Could not write the file: {exc}") from exc
    return target


def _discard(path):
    try:
        os.remove(path)
    except OSError:
        pass


def is_archive(filename):
    """Whether ``filename`` is an archive :func:`extract_archive` can unpack."""
    return os.fspath(filename).lower().endswith(".zip")


def extract_archive(path, dest_dir, name=None):
    """Unpack a ``.zip`` into ``dest_dir/<stem>/`` and return the file paths.

    Used for model repos that ship their weights as an archive (the SAM ONNX
    segmentation exports, say). Member directory parts are stripped and each
    base name sanitised through :func:`safe_name`, so a crafted archive cannot
    write outside the destination folder. Non-zip files (or an archive that
    yields no files) return an empty list, so the caller can keep the original
    download instead.
    """
    if not is_archive(path):
        return []
    stem = safe_name(os.path.splitext(name or os.path.basename(path))[0])
    folder = os.path.join(dest_dir, stem or "archive")
    os.makedirs(folder, exist_ok=True)
    extracted = []
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                base = safe_name(os.path.basename(member.filename.replace("\\", "/")))
                if not base:
                    continue
                target = os.path.join(folder, base)
                with archive.open(member) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted.append(target)
    except (zipfile.BadZipFile, OSError):
        return []
    return extracted
