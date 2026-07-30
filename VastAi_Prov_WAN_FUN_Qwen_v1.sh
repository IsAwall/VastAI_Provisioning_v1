#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# WAN 2.2 FUN CONTROL + QWEN-IMAGE-EDIT 2509  ---  v1   (max-quality build)
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
# THE PIPELINE THIS PROVISIONS (two stages, two models):
#
#   STAGE 1 -- Qwen-Image-Edit 2509  (still image)
#       reference character + scene + DWPose skeleton  ->  correctly-posed still
#       Three image inputs: image1 = character, image2 = scene,
#       image3 = DWPose render of the driving video's chosen start frame.
#       Prompting follows that structure literally, e.g.
#         "the person in image1 changes pose to image3, in the scene in image2"
#
#   STAGE 2 -- Wan 2.2 Fun Control  (video)
#       posed still (start frame) + pose control video  ->  final clip
#       Node: Wan22FunControlToVideo
#
#   The point of the split: stage 1 is where you get scene / outfit / camera
#   latitude, cheaply and iterably. Stage 2 is where you get exact motion.
#   Asking either model to do both jobs is what fails.
#
#   NO WAN ANIMATE, no SAM2, no CLIP-vision, no relight LoRA -- none of that
#   is used by this pipeline. If you also want Animate, run the other script.
#
# -----------------------------------------------------------------------------
# MODEL SELECTION -- and an important caveat about "highest quality":
#
#   Comfy-Org's repackaged repo only publishes the Wan 2.2 Fun models in
#   fp8_scaled. There is no bf16 repackage. So for max quality this script
#   pulls the ORIGINAL alibaba-pai weights instead -- single unsharded
#   safetensors, Apache 2.0, ~29.5 GB per expert -- and renames them into
#   ComfyUI's naming convention on the way in. ComfyUI loads these directly
#   from models/diffusion_models; the diffusers key names are compatible.
#   The fp8_scaled repackages are left in the manifest, commented, as the
#   ~27 GB-cheaper fallback.
#
#   Qwen's text encoder is the one place this build is NOT max precision:
#   qwen_2.5_vl_7b_fp8_scaled is what Comfy-Org publishes and what every
#   reference workflow loads. Check the repo for an fp16/bf16 variant if you
#   want it -- but the text encoder runs once per prompt, so this is the
#   smallest-impact quantisation in the whole stack.
#
#     wan2.2 fun control high noise, bf16 (PAI)      ~29.5 GB
#     wan2.2 fun control low  noise, bf16 (PAI)      ~29.5 GB
#     qwen_image_edit_2509_bf16                      ~40 GB
#     umt5_xxl_fp16 (Wan text encoder)                11.4 GB
#     qwen_2.5_vl_7b_fp8_scaled (Qwen text encoder)   ~9 GB
#     wan_2.1_vae + qwen_image_vae                    ~0.5 GB
#     DWPose bundle                                   ~1 GB
#                                                  ----------
#                                                    ~121 GB  (decimal)
#
#   >>> DISK: rent >=250 GiB. This is a much heavier build than the Animate
#   >>> one -- two full model families. The pre-flight below will refuse the
#   >>> model phase rather than half-fill the volume. If you need it smaller,
#   >>> switch to the fp8_scaled Fun lines (-27 GB) and/or the fp8 Qwen line
#   >>> (-20 GB), both commented in the manifest.
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
echo "########## provisioning run (FUN+QWEN v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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
#   comfyui_controlnet_aux  DWPreprocessor -- used TWICE in this pipeline:
#                           once to make the skeleton image for Qwen's image3
#                           slot, once to preprocess the control video for
#                           Fun Control. This is the load-bearing node here.
#   VideoHelperSuite        video load/save, frame batching
#   KJNodes                 ImageResizeKJv2 and friends -- keeping the still
#                           and the control video at one identical resolution
#                           is what stops the stage-1/stage-2 seam mismatch
#   RIFE-TensorRT-Auto      frame interpolation on the output side
#
#   Deliberately NOT installed (Animate-only, unused by this pipeline):
#     ComfyUI-segment-anything-2, ComfyUI-WanAnimatePreprocess
#
#   ComfyUI-GGUF is not needed -- this build is bf16/fp8 safetensors only.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    # "https://github.com/kijai/ComfyUI-WanVideoWrapper"   # alt execution path; needs SageAttention+Triton
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
# onnxruntime provider check (Blackwell / sm_120)
#
# DWPose is on the critical path twice in this pipeline, so a silent CPU
# fallback here costs more than it did in the Animate build. On Blackwell the
# ORT CUDA provider commonly fails to register and DWPose quietly runs on CPU.
#
# We report rather than pin, because the correct pin moves with the driver and
# CUDA line. If CUDAExecutionProvider is missing below: either pin onnxruntime
# (1.20.1 is the commonly-cited Blackwell fix), or set the DWPose node's
# bbox_detector to "yolox_l.torchscript.pt" -- the torchscript detector and the
# default torchscript pose estimator go through torch, not ORT. Both
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
# Every repo touched below (Comfy-Org, alibaba-pai, Kijai) is Xet-backed. aria2
# follows one redirect then fires N ranged requests at a single signed URL;
# requests outside its authorised range 403 and the connection dies. No aria2
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
#     layout. Also handles the PAI repos, where the file lives in a
#     high_noise_model/ or low_noise_model/ subfolder and must be RENAMED on
#     the way in -- both experts are called diffusion_pytorch_model.safetensors
#     upstream, so without renaming they would collide. ---
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

MODELS=(
    # =====================================================================
    # STAGE 2 -- WAN 2.2 FUN CONTROL 14B (MoE: high + low noise experts)
    # =====================================================================
    # bf16 originals from alibaba-pai. Both experts are named
    # diffusion_pytorch_model.safetensors upstream and differ only by folder,
    # so the rename below is REQUIRED -- otherwise they overwrite each other.
    "hf|$DM|wan2.2_fun_control_high_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-Fun-A14B-Control|high_noise_model/diffusion_pytorch_model.safetensors"
    "hf|$DM|wan2.2_fun_control_low_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-Fun-A14B-Control|low_noise_model/diffusion_pytorch_model.safetensors"

    #   fp8_scaled fallback (~16 GB each, -27 GB total). These are what the
    #   native ComfyUI template's dropdowns are pre-set to, so if you'd rather
    #   not repoint two loader nodes, use these instead of the bf16 pair above.
    # "hf|$DM|wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"
    # "hf|$DM|wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"

    #   Fun InP -- first+last frame. This is the deliberate scene-transition
    #   tool: start image = scene A, end image = scene B, and a large enough
    #   gap resolves as a cut rather than a morph. ~28.6 GB per expert.
    #   Uncomment if you want that stage as well (+57 GB).
    # "hf|$DM|wan2.2_fun_inpaint_high_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-Fun-A14B-InP|high_noise_model/diffusion_pytorch_model.safetensors"
    # "hf|$DM|wan2.2_fun_inpaint_low_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-Fun-A14B-InP|low_noise_model/diffusion_pytorch_model.safetensors"

    #   VACE-Fun -- adds a reference-image slot alongside the control video, so
    #   identity is anchored across the whole clip instead of only frame one.
    #   ~34.7 GB per expert (+69 GB). Worth it if you see identity decay over
    #   5 s; unnecessary if the Qwen-generated start frame holds up.
    # "hf|$DM|wan2.2_vace_fun_high_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-VACE-Fun-A14B|high_noise_model/diffusion_pytorch_model.safetensors"
    # "hf|$DM|wan2.2_vace_fun_low_noise_14B_bf16.safetensors|alibaba-pai/Wan2.2-VACE-Fun-A14B|low_noise_model/diffusion_pytorch_model.safetensors"

    #   SPEED, NOT QUALITY -- omitted on purpose. The 4-step Lightning pair
    #   costs real motion dynamics, which is the whole point of Fun Control.
    #   Enable for preview passes only; note they must be paired high-with-high
    #   and low-with-low against the matching expert.
    # "hf|$LORA|wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    # "hf|$LORA|wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

    # --- Wan VAE (2.2 Fun reuses the 2.1 VAE unchanged), ~0.25 GB ---------
    "hf|$VAE|wan_2.1_vae.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/vae/wan_2.1_vae.safetensors"

    # --- Wan text encoder: umt5-xxl fp16, 11.4 GB ------------------------
    "hf|$TE|umt5_xxl_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp16.safetensors"
    # "hf|$TE|umt5_xxl_fp8_e4m3fn_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

    # =====================================================================
    # STAGE 1 -- QWEN-IMAGE-EDIT 2509  (20B image edit model, Apache 2.0)
    # =====================================================================
    # bf16, ~40 GB. Three image inputs + native ControlNet keypoint support,
    # which is what lets the DWPose skeleton go straight into the image3 slot
    # with no separate ControlNet model to download.
    "hf|$DM|qwen_image_edit_2509_bf16.safetensors|Comfy-Org/Qwen-Image-Edit_ComfyUI|split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"
    #   fp8 fallback (~20 GB), and what the stock template's dropdown expects:
    # "hf|$DM|qwen_image_edit_2509_fp8_e4m3fn.safetensors|Comfy-Org/Qwen-Image-Edit_ComfyUI|split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"

    #   NOTE: a newer 2511 revision exists (Dec 2025) in the same repo, and the
    #   text encoder and VAE below are shared across both. You asked for 2509;
    #   swapping is a one-line change if you want to A/B them.

    # --- Qwen text encoder (Qwen2.5-VL-7B), ~9 GB ------------------------
    # fp8_scaled is what Comfy-Org publishes here -- see the header caveat.
    "hf|$TE|qwen_2.5_vl_7b_fp8_scaled.safetensors|Comfy-Org/Qwen-Image_ComfyUI|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

    # --- Qwen VAE, ~0.25 GB ----------------------------------------------
    "hf|$VAE|qwen_image_vae.safetensors|Comfy-Org/Qwen-Image_ComfyUI|split_files/vae/qwen_image_vae.safetensors"
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
        echo "[provisioning] !!! or switch to the fp8_scaled Fun / fp8 Qwen lines (~-47 GB)."
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
# no progress reporting inside ComfyUI. DWPose sits on the critical path twice
# in this pipeline (skeleton for Qwen image3, and control-video preprocessing),
# so an unattended first-run download is worth avoiding.
#
# Node defaults: yolox_l.onnx detector + dw-ll_ucoco_384_bs5.torchscript.pt
# pose estimator. We also pull the onnx pose model and the torchscript
# detector, so the fully ORT-free path is available with no extra download.
#
# No SAM2 here -- that was an Animate dependency and this pipeline never
# masks a subject.
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
# Workflow templates
#
# Both Fun templates and the Qwen ImageEdit template ship inside ComfyUI
# (Workflow -> Browse Templates -> Video / Image), which is the authoritative
# source. These fetches are belt-and-braces and non-fatal: the Comfy-Org
# template filenames are not stable enough to depend on, and the ComfyUI Wiki's
# Fun Control page currently links the fun_inpaint JSON by mistake -- so treat
# a 404 here as expected, not as a problem.
#
# NOTE on loader dropdowns: this build installs bf16 weights, while the stock
# templates are pre-set to the fp8 filenames. Expect red loader nodes on first
# load until you repoint them. That is not a missing-node error.
# ---------------------------------------------------------------------------
WF_DIR="${COMFY}/user/default/workflows"
TPL_BASE="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates"
mkdir -p "$WF_DIR"
for tpl in video_wan2_2_14B_fun_control video_wan2_2_14B_fun_inpaint image_qwen_image_edit_2509; do
    if [[ -f "${WF_DIR}/${tpl}.json" ]]; then
        echo "[workflow] ${tpl} already present"; continue
    fi
    if curl -fsSL --max-time 60 -o "${WF_DIR}/${tpl}.json" "${TPL_BASE}/${tpl}.json"; then
        echo "[workflow] ${tpl} OK"
    else
        rm -f "${WF_DIR}/${tpl}.json"
        echo "[workflow] ${tpl} not fetched -- use the in-app template browser"
    fi
done

echo "=================== PROVISIONING COMPLETE ==================="
