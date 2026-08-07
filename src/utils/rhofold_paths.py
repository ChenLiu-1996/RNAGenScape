"""Portable RhoFold code + checkpoint resolution (no lab-local paths required).

Defaults live under ``<repo>/external_src/``. Override with ``RHOFOLD_DIR`` /
``RHOFOLD_CKPT`` or CLI flags. Missing weights are downloaded from Hugging Face.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

RHOFOLD_GITHUB = "https://github.com/ml4bio/RhoFold.git"
RHOFOLD_HF_REPO = "cuhkaih/rhofold"
RHOFOLD_HF_FILE = "rhofold_pretrained_params.pt"
RHOFOLD_HF_URL = (
    f"https://huggingface.co/{RHOFOLD_HF_REPO}/resolve/main/{RHOFOLD_HF_FILE}"
)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_rhofold_dir() -> str:
    return os.path.join(repo_root(), "external_src", "RhoFold")


def default_ckpt_path() -> str:
    return os.path.join(
        repo_root(), "external_src", "RhoFold_pretrained", "RhoFold_pretrained.pt"
    )


def resolve_rhofold_dir(explicit: Optional[str] = None) -> str:
    """Resolve RhoFold source tree: CLI/env → importable package → ``external_src``."""
    candidates = []
    if explicit and str(explicit).strip():
        candidates.append(os.path.abspath(str(explicit).strip()))
    env = os.environ.get("RHOFOLD_DIR", "").strip()
    if env:
        candidates.append(os.path.abspath(env))
    candidates.append(default_rhofold_dir())

    for path in candidates:
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "rhofold")):
            return path

    # Already installed into the active environment?
    try:
        import rhofold  # type: ignore

        pkg = os.path.dirname(os.path.abspath(rhofold.__file__))
        parent = os.path.dirname(pkg)
        if os.path.isdir(os.path.join(parent, "rhofold")):
            return parent
        return pkg
    except Exception:
        pass

    hint = default_rhofold_dir()
    raise FileNotFoundError(
        "RhoFold source not found.\n"
        f"Tried: {candidates}\n"
        "Install once with:\n"
        "  bash bash/setup_rhofold.sh\n"
        f"or: git clone {RHOFOLD_GITHUB} {hint}\n"
        "Then optionally: pip install -e that directory."
    )


def ensure_rhofold_checkpoint(ckpt: Optional[str] = None) -> str:
    """Return a local checkpoint path, downloading from Hugging Face if needed."""
    path = os.path.abspath((ckpt or "").strip() or default_ckpt_path())
    env = os.environ.get("RHOFOLD_CKPT", "").strip()
    if (not (ckpt or "").strip()) and env:
        path = os.path.abspath(env)

    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"RhoFold checkpoint missing; downloading to {path}")
    print(f"  source: {RHOFOLD_HF_URL}")

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=RHOFOLD_HF_REPO,
            filename=RHOFOLD_HF_FILE,
            local_dir=os.path.dirname(path),
        )
        # hf_hub_download may keep the upstream filename; normalize to our path.
        if os.path.abspath(downloaded) != path:
            if os.path.isfile(path):
                os.remove(path)
            os.replace(downloaded, path)
    except Exception as hf_err:
        print(f"  huggingface_hub failed ({hf_err}); falling back to urllib")
        tmp = path + ".partial"
        try:
            urllib.request.urlretrieve(RHOFOLD_HF_URL, tmp)
            os.replace(tmp, path)
        except Exception as url_err:
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise FileNotFoundError(
                f"Could not download RhoFold weights to {path}.\n"
                f"Manual download:\n"
                f"  mkdir -p {os.path.dirname(path)}\n"
                f"  curl -L -o {path} '{RHOFOLD_HF_URL}'\n"
                f"Errors: hf={hf_err!r}; urllib={url_err!r}"
            ) from url_err

    if not os.path.isfile(path):
        raise FileNotFoundError(f"RhoFold checkpoint still missing after download: {path}")
    return path
