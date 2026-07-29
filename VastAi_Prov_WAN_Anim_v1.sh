#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# WAN 2.2 ANIMATE 14B  ---  v1   (max-quality build)
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text (GitHub "Raw"
#      button URL, Gist raw URL, pastebin raw URL).
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Optional but recommended) set  HF_TOKEN=<your token>  in the instance
#      env -- authenticated HF transfers are prioritised and avoid 429s.
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# Idempotent: nodes cloned if missing / deps reinstalled; models skipped if
# already complete, resumed if partial. Individual failures are logged, not fatal.
#
# -----------------------------------------------------------------------------
# WHAT THIS IS (and how it differs from the I2V script, v5):
#
#   Wan 2.2 Animate is a DIFFERENT model family from Wan 2.2 I2V. It is built on
#   the single-expert Wan-I2V foundation, NOT the 2.2 MoE backbone -- so there is
#   no high-noise/low-noise pair here. One monolithic checkpoint, one UNETLoader.
#
#   It also pulls in three dependencies I2V never touches:
#     * CLIP-vision  (clip_vision_h)      -- reference-image conditioning
#     * DWPose       (controlnet_aux)     -- whole-body skeleton extraction
#     * SAM2         (segment-anything-2) -- character mask for replacement mode
#
#   The pose and mask models are NOT on huggingface.co under a path this script
#   can treat as a normal model file -- they are lazily fetched by the custom
#   nodes on first use, into node-local ckpt directories. On an ephemeral Vast
#   box that means your FIRST generate stalls several minutes on an unattended,
#   unlogged download. The AUX MODEL PRE-WARM phase below fetches them during
#   provisioning instead, into the exact paths the nodes look in.
#
# -----------------------------------------------------------------------------
# MAX-QUALITY MODEL SELECTION (this is the "highest quality only" build):
#
#     wan2.2_animate_14B_bf16.safetensors        34.5 GB   (not fp8, not GGUF)
#     umt5_xxl_fp16.safetensors                  11.4 GB   (not the fp8_scaled)
#     clip_vision_h.safetensors                   1.26 GB
#     wan2.2_animate_14B_relight_lora_bf16        1.44 GB
#     wan_2.1_vae.safetensors                    ~0.25 GB
#     SAM2.1 hiera large + DWPose bundle         ~1.5 GB
#                                             ----------
#                                               ~50 GB  (decimal; ~47 GiB)
#
#   The lightx2v step-distill LoRA is DELIBERATELY OMITTED. It collapses
#   sampling to ~4 steps and is the single largest quality regression available
#   in this pipeline -- it is a speed tool, not a quality one. It is left in the
#   manifest below, commented, if you want it for fast preview iterations.
#
#   >>> DISK NOTE: ~50 GB of models, plus Xet's transient chunk cache during the
#   >>> run, plus video I/O. Rent with >=150 GiB. The pre-flight below will
#   >>> refuse the model phase rather than half-fill the disk, but a disk that
#   >>> only just fits will bite you later on output frames.
# =============================================================================

# Note: deliberately NOT using `set -e` -- one failed node/model should not stop
# the rest of provisioning.
set -o pipefail

# ---------------------------------------------------------------------------
# Persistent log  (Vast's "Logs" button only shows a recent snapshot, so mirror
# everything to a file; tail it live with:  tail -f /workspace/provisioning.log)
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (WAN-Animate v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

# ---------------------------------------------------------------------------
# Paths & the Python interpreter ComfyUI actually uses.
# ---------------------------------------------------------------------------
COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

[[ -f /opt/ai-dock/etc/environment.sh ]] && source /opt/ai-dock/etc/environment.sh
[[ -f /opt/ai-dock/bin/venv-set.sh    ]] && source /opt/ai-dock/bin/venv-set.sh comfyui

if   [[ -n "${COMFYUI_VENV_PYTHON:-}" && -x "${COMFYUI_VENV_PYTHON}" ]]; then
    PY="$COMFYUI_VENV_PYTHON"
elif [[ -x /venv/main/bin/python ]]; then
    PY="/venv/main/bin/python"
elif [[ -x /opt/environments/python/comfyui/bin/python ]]; then
    PY="/opt/environments/python/comfyui/bin/python"
else
    PY="$(ps -eo args 2>/dev/null | grep '[m]ain.py' | grep -oE '^[^ ]*python[^ ]*' | head -1)"
    [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
fi
echo "[provisioning] using python: ${PY:-<none found>}"

pip_install() {
    "$PY" -m pip install "$@" && return 0
    echo "[pip] first attempt failed, retrying with --break-system-packages"
    "$PY" -m pip install --break-system-packages "$@"
}

ensure_pkg() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "[provisioning] '$1' missing -> installing '$2'"
    apt-get update -qq && apt-get install -y -qq "$2" \
        || echo "[provisioning] WARNING: failed to install '$2'"
}

ensure_pkg git  git    # cloning custom nodes
ensure_pkg curl curl   # HEAD size checks for the skip/pre-flight logic
# aria2 is installed later, only if the manifest actually has non-HF entries.

# ---------------------------------------------------------------------------
# Custom nodes
#
#   KJNodes                 Points Editor (the green-point mask seeds), utils
#   VideoHelperSuite        video load/save, frame batching
#   comfyui_controlnet_aux  DWPreprocessor -- whole-body pose extraction
#   segment-anything-2      SAM2 character masking for replacement mode
#   RIFE-TensorRT-Auto      frame interpolation on the output side
#
#   Dropped vs v5: ComfyUI-Easy-Use (nothing in the Animate graph needs it).
#   Not included: ComfyUI-WanVideoWrapper + ComfyUI-WanAnimatePreprocess. Those
#   are the alternative (kijai) execution path -- they use ViTPose instead of
#   DWPose and require a working SageAttention + Triton build. The native
#   ComfyUI Animate graph needs none of that. Uncomment only if you're
#   deliberately switching paths, and expect to debug attention kernels.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/kijai/ComfyUI-segment-anything-2"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    # "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    # "https://github.com/kijai/ComfyUI-WanAnimatePreprocess"
)

install_node() {
    local url="$1" name path
    name="$(basename "$url" .git)"
    path="${NODES_DIR}/${name}"
    if [[ -d "$path" ]]; then
        echo "[node] $name present"
        # ( cd "$path" && git pull --ff-only ) || echo "[node] git pull failed: $name"
    else
        echo "[node] cloning $name"
        git clone --recursive "$url" "$path" || { echo "[node] CLONE FAILED: $name"; return 0; }
    fi
    if [[ -f "${path}/requirements.txt" ]]; then
        pip_install --no-cache-dir -r "${path}/requirements.txt" \
            || echo "[node] requirements.txt FAILED: $name"
    fi
    if [[ -f "${path}/install.py" ]]; then
        ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
    fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
for n in "${NODES[@]}"; do install_node "$n"; done

# ---------------------------------------------------------------------------
# CUDA reconciliation (carried over from v3/v4/v5)
# RIFE-TensorRT-Auto's deps pull cuda-python 13.x, which swaps cuda-bindings to
# 13.x and breaks the image's cu12 torch (pins cuda-bindings==12.9.x). Cap it.
# If torch ever moves to a CUDA-13 build, remove this line.
# ---------------------------------------------------------------------------
echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# onnxruntime on Blackwell (sm_120 / RTX PRO 6000, 5090, etc.)
#
# controlnet_aux's requirements pull a generic onnxruntime. On Blackwell the
# CUDA execution provider commonly fails to register (QuickGelu / provider
# init errors), and DWPose silently falls back to the CPU provider -- which
# still works but turns pose extraction into the slowest stage in the graph.
#
# We do NOT force a version here, because the right pin moves with the driver
# and the image's CUDA line. Instead we report what's installed and which
# providers actually register, so a bad combination is visible in the log
# rather than showing up as a mysteriously slow first run.
#
# If the CUDA provider is missing below, either pin onnxruntime (1.20.1 is the
# commonly-cited Blackwell fix) or sidestep onnxruntime entirely by setting the
# DWPose node's bbox_detector to "yolox_l.torchscript.pt" -- the torchscript
# detector and the default torchscript pose estimator use torch, not ORT. Both
# torchscript variants are pre-warmed below, so that switch costs no download.
# ---------------------------------------------------------------------------
"$PY" - <<'PYEOF' || true
try:
    import onnxruntime as ort
    print("[onnx] onnxruntime %s; providers: %s" % (ort.__version__, ort.get_available_providers()))
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[onnx] WARNING: no CUDAExecutionProvider -> DWPose onnx path will run on CPU.")
        print("[onnx] WARNING: use the .torchscript.pt detector/estimator, or pin onnxruntime.")
except Exception as e:
    print("[onnx] onnxruntime not importable yet (%s) -- controlnet_aux will handle it." % e)
PYEOF

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE  (unchanged from v5 -- Xet-native transport)
#
# Reminder of why this exists: the Comfy-Org / Kijai repos are served from HF's
# Xet backend, which hands out a separate signed URL per byte-range chunk. aria2
# follows one redirect then fires N ranged requests at that single URL; every
# request outside its authorised range 403s and the connection dies. No aria2
# flag fixes it. hf_xet queries the CAS for the reconstruction manifest and
# fetches xorb ranges with adaptive concurrency instead.
# ===========================================================================

"$PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
if ! "$PY" -c "import hf_xet" 2>/dev/null; then
    echo "[provisioning] installing hf_xet for fast HF (Xet) downloads"
    pip_install hf_xet || echo "[provisioning] WARNING: hf_xet install failed -> HF downloads will fall back to the (slower) LFS bridge"
fi

# --- HF environment ---
export HF_HOME="${WORKSPACE:-/workspace}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0        # deprecated + silently ignored; keep off
mkdir -p "$HF_HOME"
mem_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$mem_gb" ]] && (( mem_gb >= 64 )); then
    export HF_XET_HIGH_PERFORMANCE=1
    echo "[provisioning] ${mem_gb} GB RAM -> HF_XET_HIGH_PERFORMANCE=1"
fi

CURL_AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    echo "[provisioning] HF_TOKEN detected -> authenticated downloads"
fi

map_url() {
    local u="$1"
    [[ -n "${HF_ENDPOINT:-}" ]] && u="${u/https:\/\/huggingface.co/${HF_ENDPOINT%/}}"
    printf '%s' "$u"
}

hf_resolve_url() {
    map_url "https://huggingface.co/${1}/resolve/main/${2}"
}

remote_size() {
    local url; url="$(map_url "$1")"
    local headers val
    headers="$(curl -sIL --connect-timeout 15 --max-time 60 "${CURL_AUTH[@]}" "$url" 2>/dev/null)" || return 0
    val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-size"{v=$2} END{if(v!="")print v}')"
    [[ -z "$val" ]] && val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{if(v!="")print v}')"
    printf '%s' "${val//[^0-9]/}"
}

# --- Python helper: Xet-native single-file download, then flatten to ComfyUI's
#     layout. Written once here so quoting stays sane. ---
HF_GET="/tmp/hf_get.py"
cat > "$HF_GET" <<'PYEOF'
import sys, os, shutil, traceback
try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    sys.stderr.write("huggingface_hub import failed: %s\n" % e); sys.exit(3)

def main():
    if len(sys.argv) < 4:
        sys.stderr.write("usage: hf_get.py <repo_id> <repo_path> <dest_file>\n"); return 2
    repo, path, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    token = os.environ.get("HF_TOKEN") or None
    dest_dir = os.path.dirname(dest) or "."
    # Stage under the destination's own volume so the final move is an instant
    # rename; the stage dir also holds hf's resume metadata across boots.
    stage = os.path.join(dest_dir, ".hf_stage")
    os.makedirs(stage, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=path, local_dir=stage, token=token)
    shutil.move(got, dest)
    print(dest)
    return 0

try:
    sys.exit(main())
except Exception:
    traceback.print_exc(); sys.exit(1)
PYEOF

dl_hf() {
    # dl_hf <dest_dir> <dest_name> <repo_id> <repo_path>
    local dir="$1" name="$2" repo="$3" rpath="$4"
    local dest="${dir}/${name}"
    local check_url; check_url="$(hf_resolve_url "$repo" "$rpath")"
    mkdir -p "$dir"

    local want have=0
    want="$(remote_size "$check_url")"
    [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [[ -f "$dest" ]]; then
        if [[ -n "$want" ]] && (( have == want )); then
            echo "[model] $name complete (${have} bytes), skipping"; return 0
        elif [[ -n "$want" ]]; then
            echo "[model] $name size mismatch (local ${have} != remote ${want}) -> re-fetching"
            rm -f "$dest"
        else
            echo "[model] $name present, size unverifiable, assuming complete"; return 0
        fi
    fi

    echo "[model] downloading $name via hf_xet (${repo})"
    if "$PY" "$HF_GET" "$repo" "$rpath" "$dest"; then
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        if [[ -n "$want" ]] && (( have != want )); then
            echo "[model] WARNING: $name size after download ${have} != expected ${want} (kept for resume next boot)"
        else
            echo "[model] $name OK (${have} bytes)"
        fi
    else
        echo "[model] DOWNLOAD FAILED: $name (hf_xet; will retry next boot)"
    fi
}

dl_aria2() {
    # dl_aria2 <dest_dir> <filename> <url>   -- for NON-HF ("url") entries only.
    local dir="$1" name="$2" url; url="$(map_url "$3")"
    local dest="${dir}/${name}"
    mkdir -p "$dir"
    local want have=0
    want="$(remote_size "$3")"
    [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        if [[ -n "$want" ]] && (( have == want )); then
            echo "[model] $name complete (${have} bytes), skipping"; return 0
        elif [[ -n "$want" ]]; then
            echo "[model] $name size mismatch -> re-fetching"; rm -f "$dest"; have=0
        else
            echo "[model] $name present, size unverifiable, assuming complete"; return 0
        fi
    fi
    local tries=3 n=1
    while (( n <= tries )); do
        echo "[model] downloading $name via aria2 (attempt ${n}/${tries})"
        if aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 \
                  --continue=true --auto-file-renaming=false \
                  --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600 \
                  --max-file-not-found=2 -d "$dir" -o "$name" "$url"; then
            break
        fi
        echo "[model] attempt ${n} failed for $name"; (( n++ )); sleep 5
    done
    have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    [[ -f "$dest" ]] || { echo "[model] DOWNLOAD FAILED: $name"; return 0; }
    if [[ -n "$want" ]] && (( have != want )); then
        echo "[model] WARNING: $name incomplete (${have}/${want}) -- kept for resume"; return 0
    fi
    echo "[model] $name OK (${have} bytes)"
}

# ---------------------------------------------------------------------------
# Model manifest
#   HF  entries:  hf  | dest_dir | dest_filename | repo_id            | repo_path
#   URL entries:  url | dest_dir | dest_filename | https://...              (non-HF)
# ---------------------------------------------------------------------------
DM="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
VAE="${COMFY}/models/vae"
TE="${COMFY}/models/text_encoders"
CLIPV="${COMFY}/models/clip_vision"

MODELS=(
    # --- Diffusion model: Wan 2.2 Animate 14B, bf16, ~34.5 GB -------------
    # Single checkpoint. There is no high/low noise pair for Animate.
    "hf|$DM|wan2.2_animate_14B_bf16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors"
    #   fp8 alternative (~17 GB). Kijai's v2 build is what the WanVideoWrapper
    #   example graphs expect; only relevant if you switch execution paths.
    # "hf|$DM|Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors|Kijai/WanVideo_comfy_fp8_scaled|Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"

    # --- Relight LoRA, bf16, ~1.44 GB ------------------------------------
    # Trained on IC-Light-synthesised lighting pairs. This is what matches the
    # inserted character's illumination to the destination scene -- the
    # difference between an integrated swap and a pasted cut-out. Not optional
    # for replacement mode.
    "hf|$LORA|wan2.2_animate_14B_relight_lora_bf16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors"
    #   Identical weights, Kijai's mirror, if the Comfy-Org path ever moves:
    # "hf|$LORA|WanAnimate_relight_lora_fp16.safetensors|Kijai/WanVideo_comfy|LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"

    #   SPEED, NOT QUALITY -- omitted from this build on purpose. The step-distill
    #   LoRA drops sampling to ~4 steps at a real cost in detail and motion
    #   fidelity. Enable only for fast preview passes, then disable to render.
    # "hf|$LORA|lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors|Kijai/WanVideo_comfy|Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

    # --- CLIP vision, ~1.26 GB -------------------------------------------
    # New dependency vs the I2V stack. Reference-image conditioning; the Animate
    # graph will not load without it. Only lives in the 2.1 repackaged repo.
    "hf|$CLIPV|clip_vision_h.safetensors|Comfy-Org/Wan_2.1_ComfyUI_repackaged|split_files/clip_vision/clip_vision_h.safetensors"

    # --- VAE, ~0.25 GB ---------------------------------------------------
    # Animate reuses the Wan 2.1 VAE unchanged.
    "hf|$VAE|wan_2.1_vae.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/vae/wan_2.1_vae.safetensors"

    # --- Text encoder: umt5-xxl fp16, ~11.4 GB ---------------------------
    # fp16, not the 6.74 GB fp8_scaled, per the max-quality brief. If you'd
    # rather spend the 4.7 GB elsewhere, the fp8 line is below -- the text
    # encoder runs once per prompt, so the quality delta here is the smallest
    # of any swap in this manifest.
    "hf|$TE|umt5_xxl_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp16.safetensors"
    # "hf|$TE|umt5_xxl_fp8_e4m3fn_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)

# Install aria2 only if the manifest actually has non-HF entries.
if printf '%s\n' "${MODELS[@]}" | grep -q '^url|'; then
    ensure_pkg aria2c aria2
fi

# ---------------------------------------------------------------------------
# Disk pre-flight: sum bytes still to fetch vs free space; skip the model phase
# (loudly) rather than half-fill the disk and corrupt a model.
# ---------------------------------------------------------------------------
preflight_disk() {
    local need=0 kind a b c d url dest have want
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        case "$kind" in
            hf)  url="$(hf_resolve_url "$c" "$d")"; dest="${a}/${b}" ;;
            url) url="$c"; dest="${a}/${b}" ;;
            *)   continue ;;
        esac
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        want="$(remote_size "$url")"
        [[ -z "$want" ]] && continue
        (( want > have )) && need=$(( need + want - have ))
    done
    # Aux models (SAM2 + DWPose) aren't in the manifest; reserve for them.
    need=$(( need + 2 * 1024*1024*1024 ))

    mkdir -p "$DM"
    local avail; avail="$(df -PB1 "$DM" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] models still to fetch: ${h_need};  free on models FS: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Skipping model downloads. Resize the instance disk and reboot,"
        echo "[provisioning] !!! or switch to the fp8 Animate + fp8 text encoder lines (~-23 GB)."
        return 1
    fi
    if (( avail - need < 20 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~20GiB). Xet's chunk cache"
        echo "[provisioning] NOTE: uses transient space, and Animate writes large frame batches."
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch models
# ---------------------------------------------------------------------------
echo "=================== MODELS ==================="
if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        case "$kind" in
            hf)  dl_hf    "$a" "$b" "$c" "$d" ;;
            url) dl_aria2 "$a" "$b" "$c"      ;;
            *)   echo "[model] unknown manifest kind: '$kind' in: $entry" ;;
        esac
    done
else
    echo "[provisioning] model phase skipped (see disk warning above)"
fi

# ===========================================================================
# AUX MODEL PRE-WARM  (the part that doesn't exist in the I2V script)
#
# SAM2 and DWPose are fetched by their nodes at first use, into node-specific
# directories, with no progress reporting inside ComfyUI. Pull them now.
#
#   SAM2    -> ComfyUI/models/sam2/                    (repo Kijai/sam2-safetensors)
#   DWPose  -> comfyui_controlnet_aux/ckpts/<repo_id>/ (that layout is
#              custom_hf_download's, i.e. ckpts/yzd-v/DWPose/yolox_l.onnx)
#
# Both use snapshot_download with the same allow_patterns the nodes use, rather
# than hardcoding filenames -- so an upstream rename degrades to "node fetches
# it at runtime like it used to", not a broken script.
# ===========================================================================
echo "=================== AUX MODELS (SAM2 / DWPose) ==================="

SAM2_DIR="${COMFY}/models/sam2"
AUX_CKPTS="${NODES_DIR}/comfyui_controlnet_aux/ckpts"

"$PY" - "$SAM2_DIR" "$AUX_CKPTS" <<'PYEOF' || echo "[aux] pre-warm had failures -- nodes will fetch at first use instead"
import os, sys, traceback
from huggingface_hub import snapshot_download

sam2_dir, aux_ckpts = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None

# (repo_id, allow_patterns, local_dir, label)
JOBS = [
    # SAM2.1 hiera LARGE -- the highest-quality variant the node offers.
    # The node globs *<model>* against this repo, so mirror that.
    ("Kijai/sam2-safetensors", ["*sam2.1_hiera_large*"],
     sam2_dir, "SAM2.1 hiera large"),

    # DWPose defaults: yolox_l.onnx detector + torchscript bs5 pose estimator.
    ("yzd-v/DWPose", ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
     os.path.join(aux_ckpts, "yzd-v", "DWPose"), "DWPose onnx (detector + pose)"),
    ("hr16/DWPose-TorchScript-BatchSize5", ["dw-ll_ucoco_384_bs5.torchscript.pt"],
     os.path.join(aux_ckpts, "hr16", "DWPose-TorchScript-BatchSize5"), "DWPose torchscript pose"),

    # torchscript detector -- the onnxruntime-free path on Blackwell.
    ("hr16/yolox-onnx", ["yolox_l.torchscript.pt"],
     os.path.join(aux_ckpts, "hr16", "yolox-onnx"), "yolox torchscript detector"),
]

failed = 0
for repo, patterns, local_dir, label in JOBS:
    try:
        os.makedirs(local_dir, exist_ok=True)
        print("[aux] fetching %s (%s)" % (label, repo))
        snapshot_download(repo_id=repo, allow_patterns=patterns,
                          local_dir=local_dir, token=token)
        print("[aux] %s OK" % label)
    except Exception:
        failed += 1
        print("[aux] FAILED: %s (%s)" % (label, repo))
        traceback.print_exc()

sys.exit(1 if failed else 0)
PYEOF

# ---------------------------------------------------------------------------
# Workflow template
# The native Animate graph ships inside ComfyUI (Workflow -> Browse Templates ->
# Video), so this is belt-and-braces: drop a copy where the UI lists saved
# workflows, in case the bundled templates lag the model release.
#
# NOTE: the template's UNETLoader is pre-set to the fp8 KJ filename. With the
# bf16 checkpoint this manifest installs, that node will show red on first load
# until you repoint the dropdown. That is not a missing-node error.
# ---------------------------------------------------------------------------
WF_DIR="${COMFY}/user/default/workflows"
mkdir -p "$WF_DIR"
if [[ ! -f "${WF_DIR}/video_wan2_2_14B_animate.json" ]]; then
    echo "[workflow] fetching native Wan 2.2 Animate template"
    curl -sL --max-time 60 -o "${WF_DIR}/video_wan2_2_14B_animate.json" \
        "https://comfy.org/workflows/download/c2c4b45b7df5.json?filename=video_wan2_2_14B_animate" \
        && echo "[workflow] OK" \
        || echo "[workflow] fetch failed -- use the in-app template browser instead"
else
    echo "[workflow] template already present"
fi

echo "=================== PROVISIONING COMPLETE ==================="
