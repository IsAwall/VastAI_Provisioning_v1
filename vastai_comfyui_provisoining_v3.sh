#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
#
# HOW TO USE:
#   1. Put this file somewhere it can be fetched as RAW plain text:
#        - GitHub Gist  -> use the "Raw" button URL
#        - Pastebin     -> use the raw URL
#   2. On your vast.ai instance, set the environment variable:
#        PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Re)start the instance. ai-dock runs this on every boot.
#
# It is idempotent: present nodes get their deps reinstalled, missing nodes
# get cloned; models already on disk are skipped, missing ones are downloaded.
# Individual failures are logged but do NOT abort the rest of provisioning.
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
#     tail -f /workspace/provisioning.log
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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
[[ -f /opt/ai-dock/bin/venv-set.sh    ]] && source /opt/ai-dock/bin/venv-set.sh comfyui

if   [[ -n "${COMFYUI_VENV_PYTHON:-}" && -x "${COMFYUI_VENV_PYTHON}" ]]; then
    PY="$COMFYUI_VENV_PYTHON"                          # ai-dock (helper-provided)
elif [[ -x /venv/main/bin/python ]]; then
    PY="/venv/main/bin/python"                         # vast.ai native base image
elif [[ -x /opt/environments/python/comfyui/bin/python ]]; then
    PY="/opt/environments/python/comfyui/bin/python"   # older ai-dock layout
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
# Custom nodes  (folder present -> reinstall deps; missing -> clone, then deps)
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
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
# Models  (skip if already fully downloaded; resume partial downloads)
# ---------------------------------------------------------------------------
DM="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
VAE="${COMFY}/models/vae"
TE="${COMFY}/models/text_encoders"

dl() {
    # dl <dest_dir> <filename> <url>
    local dir="$1" name="$2" url="$3"
    mkdir -p "$dir"
    # A finished aria2c download removes its ".aria2" control file. So:
    #   file exists AND no control file  ==> complete, skip
    #   otherwise                        ==> (re)download with --continue
    if [[ -f "${dir}/${name}" && ! -f "${dir}/${name}.aria2" ]]; then
        echo "[model] $name present, skipping"
        return 0
    fi
    echo "[model] downloading $name"
    aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=5 -c \
        -d "$dir" -o "$name" "$url" \
        || echo "[model] DOWNLOAD FAILED: $name (will retry next boot)"
}

echo "=================== MODELS ==================="

# --- Diffusion models ---
dl "$DM" "wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
dl "$DM" "wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# --- SVI / Lightx2v LoRAs ---
dl "$LORA" "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
dl "$LORA" "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"
dl "$LORA" "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

# --- VAE ---
dl "$VAE" "wan_2.1_vae.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

# --- OPTIONAL: MMAudio VAE ---
# In your .txt this URL had no aria2c command, no filename and no destination,
# so the name/folder below are my best guess (it sat under the VAE header).
# None of your three nodes use it -- uncomment only if a workflow needs it.
# dl "$VAE" "mmaudio_vae_44k_fp16.safetensors" \
#     "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_vae_44k_fp16.safetensors"

# --- Text encoder ---
dl "$TE" "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "https://huggingface.co/chatpig/encoder/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

echo "=================== PROVISIONING COMPLETE ==================="
