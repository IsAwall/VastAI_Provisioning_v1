#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai  --  LTX-Video 2.3 (LTXV 2.3)
#
# SCRIPT_VERSION: 2   (previous: 1)
# CHANGELOG:
#   v2 (2026-06-21):
#     - FIX: distilled LoRA  -> pinned to ...-384-1.1 (was unversioned 1.0-era)
#     - FIX: spatial upscaler -> x2-1.1 (was x2-1.0)
#     - ADD: temporal upscaler x2-1.0 (two-stage pipeline requires it; the
#            comfy.org docs table omits it, which is why nodes reported a
#            missing/mismatched model)
#     - ADD: unquantized Gemma-3-12B text-encoder FOLDER for the Lightricks
#            two-stage / IC-LoRA graphs (kept the fp4 single-file encoder for
#            the native templates). Requires HF_TOKEN (gated google repo).
#     - ADD: prints SCRIPT_VERSION into provisioning.log at run start.
#
# HOW TO USE:
#   1. Put this file somewhere it can be fetched as RAW plain text:
#        - GitHub Gist  -> use the "Raw" button URL
#        - Pastebin     -> use the raw URL
#   2. On your vast.ai instance, set the environment variable:
#        PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Re)start the instance. ai-dock runs this on every boot.
#
#   It is idempotent: present nodes get their deps reinstalled, missing nodes
#   get cloned; models already on disk are skipped, missing ones are downloaded.
#   Individual failures are logged but do NOT abort the rest of provisioning.
#
#   Targets the NATIVE ComfyUI LTX-2.3 workflows (Template Library > Video).
#   Those run on core nodes; the Lightricks ComfyUI-LTXVideo node is included
#   below only for the advanced two-stage / IC-LoRA / audio pipelines.
# =============================================================================

# Note: deliberately NOT using `set -e` -- one failed node should not stop the
# model downloads (and vice versa).
set -o pipefail

# ---------------------------------------------------------------------------
# Persistent log
#
# Vast's "Logs" button only shows a recent SNAPSHOT (it's request-based, not a
# live stream), so the early output scrolls off. Mirror everything to a file in
# /workspace so the full run is readable afterward -- and can be tailed live by
# SSHing in (or opening the Instance Portal / Jupyter web terminal) and running:
#   tail -f /workspace/provisioning.log
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
SCRIPT_VERSION=2
echo ""
echo "########## provisioning run: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"
echo "[provisioning] script version: ${SCRIPT_VERSION}"

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
# ai-dock exposes $WORKSPACE (defaults to /workspace) and installs ComfyUI under it.
COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

# Resolve the Python interpreter ComfyUI actually uses, trying the common image
# layouts in order. Installing via "$PY -m pip" guarantees packages land in
# ComfyUI's environment rather than a stray system pip -- a bare `pip` may not
# exist, or may hit PEP 668's "externally-managed-environment" error, which is
# the usual reason node requirements fail to install.
[[ -f /opt/ai-dock/etc/environment.sh ]] && source /opt/ai-dock/etc/environment.sh
[[ -f /opt/ai-dock/bin/venv-set.sh ]] && source /opt/ai-dock/bin/venv-set.sh comfyui

if [[ -n "${COMFYUI_VENV_PYTHON:-}" && -x "${COMFYUI_VENV_PYTHON}" ]]; then
  PY="$COMFYUI_VENV_PYTHON"                                  # ai-dock (helper-provided)
elif [[ -x /venv/main/bin/python ]]; then
  PY="/venv/main/bin/python"                                 # vast.ai native base image
elif [[ -x /opt/environments/python/comfyui/bin/python ]]; then
  PY="/opt/environments/python/comfyui/bin/python"           # older ai-dock layout
else
  # Last resort: the interpreter actually running ComfyUI, else python on PATH.
  PY="$(ps -eo args 2>/dev/null | grep '[m]ain.py' | grep -oE '^[^ ]*python[^ ]*' | head -1)"
  [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
fi
echo "[provisioning] using python: ${PY:-<none found>}"

pip_install() {
  # Install through ComfyUI's python; retry under PEP 668 if pip refuses.
  "$PY" -m pip install "$@" && return 0
  echo "[pip] first attempt failed, retrying with --break-system-packages"
  "$PY" -m pip install --break-system-packages "$@"
}

# ---------------------------------------------------------------------------
# System packages the script ITSELF needs.
#
# apt-installed binaries live in the container's image layer, NOT in your
# persistent /workspace -- so on a fresh instance they are gone even though
# your storage came back. This is the same reason aria2c kept disappearing on
# fresh RunPod pods, and the same reason pip packages vanish. So we re-check on
# every boot and reinstall only what's missing (apt-get update runs only then,
# so no penalty when nothing is missing). Needs root + normal outbound network,
# which vast.ai / RunPod provisioning both have.
# ---------------------------------------------------------------------------
ensure_pkg() {
  # ensure_pkg <command-to-check> <apt-package-to-install>
  command -v "$1" >/dev/null 2>&1 && return 0
  echo "[provisioning] '$1' missing -> installing '$2'"
  apt-get update -qq && apt-get install -y -qq "$2" \
    || echo "[provisioning] WARNING: failed to install '$2'"
}
ensure_pkg aria2c aria2   # for model downloads
ensure_pkg git    git     # for cloning custom nodes

# ---------------------------------------------------------------------------
# Custom nodes (folder present -> reinstall deps; missing -> clone, then deps)
#
# The native LTX-2.3 i2v / t2v templates run on ComfyUI core nodes alone.
# VideoHelperSuite / KJNodes / Easy-Use are general-purpose carry-overs;
# RIFE-TensorRT-Auto is your frame-interp node (drives the cuda cap below);
# ComfyUI-LTXVideo (Lightricks) is OPTIONAL -- only the two-stage upscaler,
# IC-LoRA control, and audio-VAE-decode pipelines need it. Comment it out if
# you're only running the native templates.
# ---------------------------------------------------------------------------
NODES=(
  "https://github.com/kijai/ComfyUI-KJNodes"
  "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
  "https://github.com/yolain/ComfyUI-Easy-Use"
  "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
  "https://github.com/Lightricks/ComfyUI-LTXVideo"
)

install_node() {
  local url="$1"
  local name path
  name="$(basename "$url" .git)"
  path="${NODES_DIR}/${name}"

  if [[ -d "$path" ]]; then
    echo "[node] $name present"
    # To keep nodes updated on every boot, uncomment the next line:
    # ( cd "$path" && git pull --ff-only ) || echo "[node] git pull failed: $name"
  else
    echo "[node] cloning $name"
    git clone --recursive "$url" "$path" || { echo "[node] CLONE FAILED: $name"; return 0; }
  fi

  # Reinstall deps into ComfyUI's env (this is what a fresh environment loses):
  if [[ -f "${path}/requirements.txt" ]]; then
    pip_install --no-cache-dir -r "${path}/requirements.txt" \
      || echo "[node] requirements.txt FAILED: $name"
  fi
  # Some nodes (e.g. TensorRT ones) put setup logic in install.py instead:
  if [[ -f "${path}/install.py" ]]; then
    ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
  fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
for n in "${NODES[@]}"; do install_node "$n"; done

# ---------------------------------------------------------------------------
# CUDA reconciliation
#
# ComfyUI-RIFE-TensorRT-Auto's requirements pull cuda-python 13.x (CUDA 13),
# which makes pip replace cuda-bindings 12.9.x with 13.3.1 -- but the image's
# torch (a cu12 build) pins cuda-bindings==12.9.4 and reports the 13.x binding
# as incompatible. Cap cuda-python to the CUDA-12 line to restore the version
# torch wants. TensorRT itself uses tensorrt-cu12 and is unaffected.
# If you ever drop the TensorRT node, this line is a harmless no-op.
# NOTE: if the image's torch changes to a CUDA-13 build, remove this cap.
# ---------------------------------------------------------------------------
echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# Models (skip if already fully downloaded; resume partial downloads)
#
# Native ComfyUI LTX-2.3 layout (note: checkpoints/, not diffusion_models/):
#   models/checkpoints/           -> the 22B transformer
#   models/loras/                 -> distilled LoRA (two-stage pipeline)
#   models/latent_upscale_models/ -> spatial + temporal upscalers (two-stage)
#   models/text_encoders/         -> Gemma 3 12B encoder
#
# Two encoder variants are fetched, because the two graph families want
# different layouts:
#   - native templates  -> single-file fp4-mixed encoder (text_encoders/*.safetensors)
#   - Lightricks node    -> unquantized Gemma FOLDER at
#                           text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/
# ---------------------------------------------------------------------------
CKPT="${COMFY}/models/checkpoints"
LORA="${COMFY}/models/loras"
UPSCALE="${COMFY}/models/latent_upscale_models"
TE="${COMFY}/models/text_encoders"

dl() {
  # dl <dest_dir> <filename> <url>
  local dir="$1" name="$2" url="$3"
  mkdir -p "$dir"
  # A finished aria2c download removes its ".aria2" control file. So:
  #   file exists AND no control file ==> complete, skip
  #   otherwise                       ==> (re)download with --continue
  if [[ -f "${dir}/${name}" && ! -f "${dir}/${name}.aria2" ]]; then
    echo "[model] $name present, skipping"
    return 0
  fi
  echo "[model] downloading $name"
  aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=5 -c \
    -d "$dir" -o "$name" "$url" \
    || echo "[model] DOWNLOAD FAILED: $name (will retry next boot)"
}

dl_hf_folder() {
  # dl_hf_folder <parent_dir> <repo_id> <sentinel_file>
  # Snapshots a whole HF repo into <parent_dir>/<repo-basename>/ via
  # huggingface_hub. Skips if <sentinel_file> already exists in the target.
  # NOTE: google/gemma-* repos are GATED -- export HF_TOKEN on the instance
  # (Vast: add it under env vars) or this 401s. We don't abort on failure;
  # the native fp4 encoder above still covers the core templates.
  local parent="$1" repo="$2" sentinel="$3"
  local target="${parent}/$(basename "$repo")"
  if [[ -f "${target}/${sentinel}" ]]; then
    echo "[model] $(basename "$repo") folder present, skipping"
    return 0
  fi
  echo "[model] fetching HF folder: $repo"
  "$PY" -c "import huggingface_hub" 2>/dev/null || pip_install --no-cache-dir huggingface_hub
  "$PY" - "$repo" "$target" <<'PYEOF' \
    || echo "[model] HF FOLDER FAILED: ${repo} (gated? set HF_TOKEN. Native fp4 encoder still works.)"
import os, sys
from huggingface_hub import snapshot_download
repo, target = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=target,
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
)
PYEOF
}

echo "=================== MODELS ==================="

# --- Checkpoint: full BF16 dev (~42 GB) ---
# Default for your 96 GB card -- fits with huge headroom, best quality / trainable.
dl "$CKPT" "ltx-2.3-22b-dev.safetensors" \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors"

# --- OPTIONAL checkpoint: FP8 quantized (~half the size, ~4x faster) ---
# Lives in a SEPARATE repo (LTX-2.3-fp8). Uncomment if you'd rather run FP8;
# you generally wouldn't need both on disk.
# dl "$CKPT" "ltx-2.3-22b-dev-fp8.safetensors" \
#   "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors"

# --- Distilled LoRA (used by the two-stage pipeline) ---
dl "$LORA" "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

# --- Spatial upscaler (two-stage pipeline) ---
dl "$UPSCALE" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# --- Temporal upscaler (two-stage pipeline; REQUIRED, easy to miss) ---
dl "$UPSCALE" "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

# --- Text encoder (A): Gemma 3 12B fp4-mixed, single file -> native templates ---
dl "$TE" "gemma_3_12B_it_fp4_mixed.safetensors" \
  "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"

# --- Text encoder (B): unquantized Gemma 3 12B FOLDER -> Lightricks node / best quality ---
# Lands at: $TE/gemma-3-12b-it-qat-q4_0-unquantized/  (the layout that node expects)
# Gated google repo: needs HF_TOKEN. ~24 GB; comment out if you only run the
# native templates (encoder A above is enough for those).
dl_hf_folder "$TE" "google/gemma-3-12b-it-qat-q4_0-unquantized" "config.json"

echo "=================== PROVISIONING COMPLETE ==================="
