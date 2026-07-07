#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai   ---  v4
#
# HOW TO USE:
#   1. Put this file somewhere it can be fetched as RAW plain text:
#        - GitHub  -> use the "Raw" button URL (raw.githubusercontent.com/...)
#        - Gist    -> use the "Raw" button URL
#        - Pastebin-> use the raw URL
#   2. On your vast.ai instance, set the environment variable:
#        PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Re)start the instance. ai-dock runs this on every boot.
#
# It is idempotent: present nodes get their deps reinstalled, missing nodes
# get cloned; models already on disk are skipped, missing ones are downloaded.
# Individual failures are logged but do NOT abort the rest of provisioning.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED IN v4 (download logic):
#   * Completeness is now verified by SIZE, not by "is the .aria2 file gone?".
#     Before downloading, we HEAD the URL for the true remote size and compare
#     it to the local file. A truncated file (killed instance, full disk, saved
#     error page) no longer counts as "complete" and get skipped forever -- it
#     gets re-fetched. This is the fix for the poisoned-cache failure mode.
#   * Disk pre-flight: sum the sizes still to download, compare to free space,
#     and SKIP the model phase (with a loud message) if there isn't headroom --
#     instead of half-filling the disk and corrupting a model.
#   * Script-level retries around aria2 (in addition to aria2's own retries),
#     plus connect/stall timeouts so a dead mirror fails fast instead of hanging
#     provisioning (which is what left ComfyUI "not starting" behind a 502).
#   * Post-download validation: re-check final size, and catch an HTML/JSON
#     error page saved under a .safetensors name.
#   * Optional HF_TOKEN  -> authenticated downloads (gated repos + fewer 429s).
#   * Optional HF_ENDPOINT (e.g. https://hf-mirror.com) -> route HF through a
#     mirror when huggingface.co is slow/blocked. Unset => HF direct (unchanged).
#   * Models are now a declarative manifest (dir|file|url) so the pre-flight can
#     iterate them and adding/removing a model is a one-line edit.
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
echo "########## provisioning run (v4): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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
ensure_pkg curl   curl    # for HEAD size checks (v4)

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

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE (v4)
# ===========================================================================

# --- Optional HuggingFace auth ---
# Public files work anonymously, but HF increasingly 429-rate-limits anonymous
# traffic and gated repos require a token. Set HF_TOKEN in the instance env to
# use one. Empty => no header => anonymous (v3 behaviour, unchanged).
HF_ARIA_HDR=()          # header args for aria2c
CURL_AUTH=()            # header args for curl
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ARIA_HDR=(--header="Authorization: Bearer ${HF_TOKEN}")
    CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    echo "[provisioning] HF_TOKEN detected -> authenticated downloads"
fi

map_url() {
    # Optionally reroute huggingface.co through a mirror set via HF_ENDPOINT
    # (e.g. https://hf-mirror.com). No-op when HF_ENDPOINT is unset/empty.
    local u="$1"
    if [[ -n "${HF_ENDPOINT:-}" ]]; then
        u="${u/https:\/\/huggingface.co/${HF_ENDPOINT%/}}"
    fi
    printf '%s' "$u"
}

remote_size() {
    # Echo the expected byte size of a URL, or nothing if it can't be determined.
    # HF /resolve/ URLs 302-redirect LFS files to a CDN; the resolve response
    # carries the true object size in x-linked-size, and the CDN in content-length.
    # Follow redirects (-L), prefer x-linked-size, else the LAST content-length.
    # A failed HEAD echoes nothing so callers fall back to the resume-based check
    # rather than needlessly re-pulling a good multi-GB file.
    local url; url="$(map_url "$1")"
    local headers val
    headers="$(curl -sIL --connect-timeout 15 --max-time 60 "${CURL_AUTH[@]}" "$url" 2>/dev/null)" || return 0
    val="$(printf '%s' "$headers" | tr -d '\r' \
           | awk -F': ' 'tolower($1)=="x-linked-size"{v=$2} END{if(v!="")print v}')"
    [[ -z "$val" ]] && val="$(printf '%s' "$headers" | tr -d '\r' \
           | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{if(v!="")print v}')"
    printf '%s' "${val//[^0-9]/}"       # digits only, safe for arithmetic
}

dl() {
    # dl <dest_dir> <filename> <url>
    local dir="$1" name="$2" url; url="$(map_url "$3")"
    local dest="${dir}/${name}"
    mkdir -p "$dir"

    local want have=0
    want="$(remote_size "$3")"          # "" if the HEAD failed
    [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"

    # ---- decide: skip / re-fetch / resume / fresh ----
    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        if [[ -n "$want" ]]; then
            if (( have == want )); then
                echo "[model] $name complete (${have} bytes), skipping"
                return 0
            fi
            # Size disagrees and there's no control file: the local copy is
            # truncated/corrupt. Delete it so aria2 starts clean instead of
            # "resuming" onto a bad file (the v3 poisoned-cache bug).
            echo "[model] $name size mismatch (local ${have} != remote ${want}) -> re-fetching"
            rm -f "$dest"; have=0
        else
            # Can't verify (HEAD failed). Trust v3's heuristic: no control file
            # => assume complete. Avoids re-pulling GBs on a flaky HEAD.
            echo "[model] $name present, size unverifiable, assuming complete"
            return 0
        fi
    fi

    # ---- download with resume + script-level retries ----
    local tries=3 n=1
    while (( n <= tries )); do
        echo "[model] downloading $name (attempt ${n}/${tries})"
        if aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 \
                  --continue=true --auto-file-renaming=false \
                  --max-tries=5 --retry-wait=5 \
                  --connect-timeout=30 --timeout=600 --max-file-not-found=2 \
                  "${HF_ARIA_HDR[@]}" -d "$dir" -o "$name" "$url"; then
            break
        fi
        echo "[model] attempt ${n} failed for $name"
        (( n++ )); sleep 5
    done

    # ---- verify ----
    have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [[ ! -f "$dest" ]]; then
        echo "[model] DOWNLOAD FAILED: $name (no file; will retry next boot)"; return 0
    fi
    if [[ -n "$want" ]] && (( have != want )); then
        echo "[model] WARNING: $name incomplete (${have}/${want} bytes) -- left for resume next boot"
        return 0
    fi
    # Belt-and-suspenders for the unverifiable case: a tiny "*.safetensors" that
    # is actually an HTML/JSON error page (404, gated login, mirror hiccup).
    if [[ -z "$want" && "$name" == *.safetensors && "$have" -lt 1048576 ]]; then
        if head -c 64 "$dest" | grep -qiE '<!doctype|<html|\{"error'; then
            echo "[model] ERROR: $name looks like an error page, not a model -> removing"
            rm -f "$dest"; return 0
        fi
    fi
    echo "[model] $name OK (${have} bytes)"
}

# ---------------------------------------------------------------------------
# Model manifest        format:  dest_dir | filename | url
# (Sizes are HEAD-checked at run time, so nothing to hard-code here.)
# ---------------------------------------------------------------------------
DM="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
VAE="${COMFY}/models/vae"
TE="${COMFY}/models/text_encoders"

MODELS=(
    # --- Diffusion models (Wan 2.2 I2V 14B, fp16 ~ big) ---
    "$DM|wan2.2_i2v_high_noise_14B_fp16.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    "$DM|wan2.2_i2v_low_noise_14B_fp16.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

    # --- SVI / Lightx2v LoRAs ---
    "$LORA|SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors|https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
    "$LORA|SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors|https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"
    "$LORA|lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors|https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

    # --- VAE ---
    "$VAE|wan_2.1_vae.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

    # --- Text encoder ---
    "$TE|umt5_xxl_fp8_e4m3fn_scaled.safetensors|https://huggingface.co/chatpig/encoder/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

    # --- OPTIONAL: MMAudio VAE (none of your 4 nodes use it; uncomment if a
    #     workflow needs it; filename/dest are a best guess from your original) ---
    # "$VAE|mmaudio_vae_44k_fp16.safetensors|https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_vae_44k_fp16.safetensors"
)

# ---------------------------------------------------------------------------
# Disk pre-flight
#
# Sum the bytes still to fetch (remote size minus whatever's already on disk),
# compare to free space on the models filesystem, and SKIP the model phase if
# there isn't enough headroom. Half a downloaded model is worse than none: it
# gets loaded and crashes ComfyUI. Better to stop loudly and tell you to resize.
# NOTE: we HEAD each URL here and again in dl(). HEADs are tiny, so for a
# handful of files that's fine; kept separate for readability.
# ---------------------------------------------------------------------------
preflight_disk() {
    local need=0 dir f u want have
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r dir f u <<< "$entry"
        have=0; [[ -f "${dir}/${f}" ]] && have="$(stat -c%s "${dir}/${f}" 2>/dev/null || echo 0)"
        want="$(remote_size "$u")"
        [[ -z "$want" ]] && continue                 # unknown -> leave out of estimate
        (( want > have )) && need=$(( need + want - have ))
    done

    mkdir -p "$DM"
    local avail; avail="$(df -PB1 "$DM" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))           # keep 5 GiB headroom
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] models still to fetch: ${h_need};  free on models FS: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Skipping model downloads to avoid truncated files."
        echo "[provisioning] !!! Resize the instance disk (Vast: edit -> disk space) and reboot."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch models
# ---------------------------------------------------------------------------
echo "=================== MODELS ==================="
if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r d f u <<< "$entry"
        dl "$d" "$f" "$u"
    done
else
    echo "[provisioning] model phase skipped (see disk warning above)"
fi

echo "=================== PROVISIONING COMPLETE ==================="
