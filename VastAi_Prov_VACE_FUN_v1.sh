#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# WAN 2.2 VACE-FUN A14B  ---  v1   (max-quality build)
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text (GitHub "Raw"
#      button URL, Gist raw URL, pastebin raw URL).
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Optional but recommended) set  HF_TOKEN=<your token>  in the instance
#      env -- authenticated HF transfers avoid anonymous-IP rate limiting.
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# Idempotent: nodes cloned if missing / deps reinstalled; models skipped if
# already complete, resumed if partial. Individual failures are logged, not fatal.
#
# -----------------------------------------------------------------------------
# WHAT THIS PROVISIONS -- and why it's ONE stage, not two:
#
#   reference image (UNPOSED) + DWPose control video  ->  motion-matched clip
#
#   VACE-Fun takes the reference as CONDITIONING rather than as frame one.
#   That's the whole reason to prefer it here: plain Fun Control needs its
#   start_image pre-posed to match the driving video's first frame, which is
#   what forced the Qwen pre-posing stage. VACE-Fun applies spatially-aligned
#   pose conditioning to every frame including the first, so the handoff -- and
#   the seam, the pose-matching search, and Qwen's liberal interpretation of
#   your intent -- all disappear.
#
#   No Qwen models here. No Animate, no SAM2, no relight LoRA.
#
# -----------------------------------------------------------------------------
# TWO ROUTES EXIST FOR VACE-FUN. This script takes the first:
#
#   ROUTE A (default here) -- standalone full models from alibaba-pai.
#       Self-contained: VACE layers are baked into the weights, ~34.7 GB per
#       expert, bf16, Apache 2.0. Loads straight into models/diffusion_models
#       via two UNETLoaders, driven by the native WanVaceToVideo node. No base
#       T2V model needed, no wrapper required.
#
#   ROUTE B -- VACE module applied over base Wan 2.2 T2V A14B.
#       Kijai publishes Wan2_2_Fun_VACE_module_A14B_HIGH/LOW as separate small
#       modules (~3 GB each at fp8, ~6 GB at bf16) that get layered onto the
#       base T2V high/low pair. Total disk works out about the same as Route A
#       once you count the base model, and it needs WanVideoWrapper -- but the
#       best-documented 2.2 VACE-Fun reference graphs are built this way.
#       Both module lines are in the manifest below, commented.
#
#   Honest caveat: ComfyUI's native VACE support landed with Wan 2.1 VACE, and
#   the 2.1 native path is the more heavily travelled one. Route A should work
#   natively -- WanVaceToVideo emits conditioning plus a latent and is not
#   model-specific -- but if the native graph misbehaves, Route B is the
#   fallback with more community mileage behind it. WanVideoWrapper is
#   installed below precisely so that option is one download away, and its
#   example_workflows/ folder is where the reference graphs live.
#
# -----------------------------------------------------------------------------
# MAX-QUALITY MODEL SELECTION:
#
#     wan2.2 vace-fun high noise, bf16 (PAI)     ~34.7 GB
#     wan2.2 vace-fun low  noise, bf16 (PAI)     ~34.7 GB
#     umt5_xxl_fp16                               11.4 GB
#     clip_vision_h                                1.26 GB
#     wan_2.1_vae                                 ~0.25 GB
#     DWPose bundle                               ~1 GB
#                                              ----------
#                                                ~83 GB  (decimal; ~78 GiB)
#
#   Lightning / step-distill LoRAs are omitted on purpose -- 4-step sampling
#   costs motion dynamics, which is the entire point of pose-driven generation.
#   Left commented in the manifest for preview passes only.
#
#   >>> DISK: rent >=200 GiB. The pre-flight below refuses the model phase
#   >>> rather than half-filling the volume.
#
# -----------------------------------------------------------------------------
# THE ONE PARAMETER THAT MATTERS MOST once you're running:
#
#   WanVaceToVideo has a `strength` input. VACE stretches limb proportions to
#   match the pose skeleton's bone lengths, so if your reference character is
#   stockier or shorter than the driving performer, the output comes out
#   elongated. Raising strength to roughly 1.10-1.25 pulls the result back
#   toward the reference's proportions. That is the fix for the proportion
#   bleed you'd otherwise blame on the reference image.
#
#   Working numbers carried over from the Fun Control build: 16 fps (set
#   force_rate=16 on Load Video -- the control clip MUST be resampled or you
#   get slow motion), 81 frames for ~5 s, 768x1344 portrait or 1344x768
#   landscape, control clip cut to ~6 s so the final frames still have pose
#   reference. Frame counts follow 4n+1; 80 silently truncates to 77.
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
echo "########## provisioning run (VACE-FUN v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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
#   comfyui_controlnet_aux  DWPreprocessor -- builds the pose control video.
#                           This is the load-bearing node: VACE follows the
#                           skeleton, so skeleton quality IS output quality.
#   VideoHelperSuite        video load/save; Load Video's force_rate=16 is the
#                           slow-motion fix
#   KJNodes                 ImageResizeKJv2, plus GGUFLoaderKJ if you ever go
#                           the quantised module route
#   WanVideoWrapper         ACTIVE in this build (unlike the Fun Control one).
#                           Reason: the best-documented Wan 2.2 VACE-Fun
#                           reference graphs ship in its example_workflows/
#                           folder, and it's the Route B execution path.
#                           SageAttention is OPTIONAL -- the wrapper falls back
#                           to sdpa, so this install should not be fragile.
#                           PatchSageAttentionKJ nodes in example graphs can be
#                           bypassed with Ctrl+B if SageAttention isn't built.
#   RIFE-TensorRT-Auto      output-side frame interpolation, 16 -> 32 fps
#
#   Deliberately NOT installed: ComfyUI-segment-anything-2 (Animate-only --
#   VACE takes an optional mask but this pipeline doesn't need SAM2 to make it).
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
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
#
# NOTE: WanVideoWrapper's requirements are also installed above and are the
# other likely source of dependency churn in this build. Run this AFTER all
# node requirements, which is what the ordering here does.
# If torch ever moves to a CUDA-13 build, remove this line.
# ---------------------------------------------------------------------------
echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# Report torch / attention backend state.
#
# WanVideoWrapper graphs often reference SageAttention. It is optional -- the
# wrapper falls back to sdpa -- but knowing at provisioning time whether it's
# available saves you loading an example graph and misreading a bypassable
# node as a broken install.
# ---------------------------------------------------------------------------
"$PY" - <<'PYEOF' || true
try:
    import torch
    print("[torch] %s | cuda %s | device: %s" % (
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
except Exception as e:
    print("[torch] not importable: %s" % e)
for mod, label in (("sageattention", "SageAttention"), ("triton", "Triton"),
                   ("flash_attn", "FlashAttention")):
    try:
        __import__(mod); print("[attn] %s available" % label)
    except Exception:
        print("[attn] %s NOT available (optional -- wrapper falls back to sdpa)" % label)
PYEOF

# ---------------------------------------------------------------------------
# onnxruntime provider check (Blackwell / sm_120)
#
# DWPose is the single most important preprocessing step in this pipeline --
# VACE follows the skeleton, so a silent CPU fallback here is the difference
# between seconds and minutes per clip. We report rather than pin, because the
# right pin moves with the driver and CUDA line.
#
# If CUDAExecutionProvider is missing below: either pin onnxruntime (1.20.1 is
# the commonly-cited Blackwell fix), or set the DWPose node's bbox_detector to
# "yolox_l.torchscript.pt" -- torchscript detector plus the default torchscript
# pose estimator avoid ORT entirely. Both are pre-warmed below, so that switch
# costs no download.
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
# The alibaba-pai, Comfy-Org and Kijai repos are all Xet-backed. aria2 follows
# one redirect then fires N ranged requests at a single signed URL; requests
# outside its authorised range 403 and the connection dies. No aria2 flag fixes
# it. hf_xet queries the CAS for the reconstruction manifest and fetches xorb
# ranges with adaptive concurrency instead.
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
#     layout. The rename is REQUIRED for the PAI repos: both experts are named
#     diffusion_pytorch_model.safetensors upstream and differ only by folder,
#     so without renaming the second download overwrites the first. ---
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
    # =====================================================================
    # ROUTE A -- WAN 2.2 VACE-FUN A14B, standalone bf16 (MoE high + low)
    # =====================================================================
    # Self-contained: VACE layers baked in, no base T2V required.
    # Both experts are diffusion_pytorch_model.safetensors upstream and differ
    # only by folder, so the rename here is load-bearing, not cosmetic.
    "hf|$DM|wan2.2_vace_fun_high_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-VACE-Fun-A14B|high_noise_model/diffusion_pytorch_model.safetensors"
    "hf|$DM|wan2.2_vace_fun_low_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-VACE-Fun-A14B|low_noise_model/diffusion_pytorch_model.safetensors"

    # =====================================================================
    # ROUTE B -- VACE module over base Wan 2.2 T2V A14B  (commented)
    # =====================================================================
    #   The module is small; the cost is that you ALSO need the base T2V
    #   high/low pair, which this manifest does not fetch. Only worth enabling
    #   if the native Route A graph gives you trouble, or if you specifically
    #   want the WanVideoWrapper example workflows to run unmodified.
    #
    #   fp8 modules (verified paths, ~3.05 GB each):
    # "hf|$DM|Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors|Kijai/WanVideo_comfy_fp8_scaled|VACE/Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
    # "hf|$DM|Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors|Kijai/WanVideo_comfy_fp8_scaled|VACE/Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors"
    #
    #   bf16 modules live in Kijai/WanVideo_comfy rather than the fp8 repo.
    #   Check the current folder layout there before enabling -- Kijai
    #   reorganises paths periodically (several files have been moved into
    #   LoRAs/ subfolders), so I've left these unwritten rather than guess:
    #     Wan2_2_Fun_VACE_module_A14B_HIGH_bf16.safetensors
    #     Wan2_2_Fun_VACE_module_A14B_LOW_bf16.safetensors

    #   SPEED, NOT QUALITY -- omitted on purpose. 4-step sampling costs motion
    #   dynamics, and motion is the whole point here. Pair high-with-high and
    #   low-with-low if you enable them for preview passes.
    # "hf|$LORA|wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    # "hf|$LORA|wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

    # --- VAE, ~0.25 GB ---------------------------------------------------
    # A14B models use the 2.1 VAE unchanged. (wan2.2_vae is for the 5B TI2V
    # line only -- don't cross them.)
    "hf|$VAE|wan_2.1_vae.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/vae/wan_2.1_vae.safetensors"

    # --- Text encoder: umt5-xxl fp16, 11.4 GB ----------------------------
    # Taking this from Comfy-Org sidesteps a well-known footgun: Kijai's repo
    # also ships umt5-xxl-enc-* variants whose names look nearly identical, and
    # loading an -enc- file in a native graph throws a cryptic
    # "mat1 and mat2 shapes cannot be multiplied" error.
    "hf|$TE|umt5_xxl_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp16.safetensors"
    # "hf|$TE|umt5_xxl_fp8_e4m3fn_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

    # --- CLIP vision, ~1.26 GB -------------------------------------------
    # The native WanVaceToVideo path does NOT require this. Several
    # WanVideoWrapper example graphs do (WanVideoClipVisionEncode). Cheap
    # insurance against a first-run stall if you try a wrapper graph.
    "hf|$CLIPV|clip_vision_h.safetensors|Comfy-Org/Wan_2.1_ComfyUI_repackaged|split_files/clip_vision/clip_vision_h.safetensors"
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
    # DWPose aux models aren't in the manifest; reserve for them.
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
        echo "[provisioning] !!! or switch to the fp8 text encoder / Route B module lines."
        return 1
    fi
    if (( avail - need < 20 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~20GiB). Xet's chunk cache"
        echo "[provisioning] NOTE: uses transient space, and video frame batches are large."
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
# AUX MODEL PRE-WARM -- DWPose
#
# controlnet_aux lazily fetches these on first use into ckpts/<repo_id>/, with
# no progress reporting inside ComfyUI. VACE follows the skeleton literally, so
# this is the most consequential preprocessing step in the pipeline -- worth
# having on disk before your first queue.
#
# Node defaults: yolox_l.onnx detector + dw-ll_ucoco_384_bs5.torchscript.pt
# pose estimator. We also pull the onnx pose model and the torchscript
# detector so the fully ORT-free path needs no extra download.
# ===========================================================================
echo "=================== AUX MODELS (DWPose) ==================="

AUX_CKPTS="${NODES_DIR}/comfyui_controlnet_aux/ckpts"

"$PY" - "$AUX_CKPTS" <<'PYEOF' || echo "[aux] pre-warm had failures -- nodes will fetch at first use instead"
import os, sys, traceback
from huggingface_hub import snapshot_download

aux_ckpts = sys.argv[1]
token = os.environ.get("HF_TOKEN") or None

# (repo_id, allow_patterns, local_dir, label)
JOBS = [
    ("yzd-v/DWPose", ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
     os.path.join(aux_ckpts, "yzd-v", "DWPose"), "DWPose onnx (detector + pose)"),
    ("hr16/DWPose-TorchScript-BatchSize5", ["dw-ll_ucoco_384_bs5.torchscript.pt"],
     os.path.join(aux_ckpts, "hr16", "DWPose-TorchScript-BatchSize5"), "DWPose torchscript pose"),
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
# Reference workflows
#
# WanVideoWrapper ships its own example graphs, including the Wan 2.2 VACE-Fun
# ones, inside custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/. Those
# are the best-documented starting point for this model, so just point at them
# rather than fetching anything.
#
# NOTE on loader dropdowns: this build installs bf16 standalone weights, while
# most published graphs are pre-set to fp8 or module filenames. Expect red
# loader nodes on first load until you repoint them. That is not a
# missing-node error.
# ---------------------------------------------------------------------------
WF_DIR="${COMFY}/user/default/workflows"
mkdir -p "$WF_DIR"
WRAPPER_EXAMPLES="${NODES_DIR}/ComfyUI-WanVideoWrapper/example_workflows"
if [[ -d "$WRAPPER_EXAMPLES" ]]; then
    echo "[workflow] WanVideoWrapper example graphs available at:"
    echo "[workflow]   ${WRAPPER_EXAMPLES}"
    ls -1 "$WRAPPER_EXAMPLES" 2>/dev/null | grep -iE "vace" | sed 's/^/[workflow]   - /' \
        || echo "[workflow]   (no filename matched 'vace' -- list the folder manually)"
else
    echo "[workflow] wrapper example folder not found -- check the clone above"
fi
echo "[workflow] Native VACE graphs: Workflow -> Browse Templates -> Video"

echo "=================== PROVISIONING COMPLETE ==================="
