#!/bin/bash
# Fetch public RhoFold code + pretrained weights into this repo (no lab paths).
# Also installs RhoFold Python deps into the active env (or .venv via uv).
# Usage (from repo root):
#   bash bash/setup_rhofold.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RHOFOLD_DIR="${ROOT_DIR}/external_src/RhoFold"
CKPT_DIR="${ROOT_DIR}/external_src/RhoFold_pretrained"
CKPT_PATH="${CKPT_DIR}/RhoFold_pretrained.pt"
HF_URL="https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt"

mkdir -p "${ROOT_DIR}/external_src" "${CKPT_DIR}"

if [[ ! -d "${RHOFOLD_DIR}/.git" && ! -d "${RHOFOLD_DIR}/rhofold" ]]; then
  echo "Cloning ml4bio/RhoFold → ${RHOFOLD_DIR}"
  git clone --depth 1 https://github.com/ml4bio/RhoFold.git "${RHOFOLD_DIR}"
else
  echo "RhoFold source already present: ${RHOFOLD_DIR}"
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Downloading RhoFold weights → ${CKPT_PATH}"
  curl -L --fail -o "${CKPT_PATH}" "${HF_URL}"
else
  echo "Checkpoint already present: ${CKPT_PATH}"
fi

# RhoFold runtime deps (setup.py lists 'Bio' which is the biopython package).
install_rhofold_deps() {
  echo "Ensuring RhoFold Python deps ..."
  if command -v uv >/dev/null 2>&1; then
    UV_HTTP_TIMEOUT=600 uv sync --extra rhofold --python 3.12
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]] && "${ROOT_DIR}/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    "${ROOT_DIR}/.venv/bin/python" -m pip install -q \
      "biopython>=1.83" \
      "python-box>=7.0" \
      "dm-tree>=0.1.8" \
      "ml-collections>=0.1.1"
  else
    echo "ERROR: need uv (preferred) or pip to install RhoFold deps" >&2
    return 1
  fi
  local py="${ROOT_DIR}/.venv/bin/python"
  if [[ ! -x "${py}" ]]; then
    py="$(command -v python || command -v python3)"
  fi
  if ! "${py}" -c "import Bio, box, tree, ml_collections" >/dev/null 2>&1; then
    echo "ERROR: RhoFold deps still not importable after install" >&2
    return 1
  fi
  echo "RhoFold deps OK (Bio / box / tree / ml_collections)."
}

install_rhofold_deps

echo
echo "Done. Defaults used by eval_rhofold.py:"
echo "  RHOFOLD_DIR=${RHOFOLD_DIR}"
echo "  RHOFOLD_CKPT=${CKPT_PATH}"
echo
echo "Override anytime with env vars or --rhofold_dir / --ckpt."
