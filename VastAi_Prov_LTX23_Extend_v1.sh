#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# LTX-2.3 EXTEND stack (RuneXX split-model workflows)  ---  v1
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text (GitHub "Raw"
#      button URL, Gist raw URL, pastebin raw URL).
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Optional env vars:
#        HF_TOKEN=<token>        authenticated HF transfers, avoids 429s
#        CIVITAI_TOKEN=<token>   REQUIRED for the Civitai LoRA section below
#        UPDATE_NODES=1          git pull existing custom nodes on every boot
#                                (see the CUSTOM NODES section -- KJNodes and
#                                 ComfyUI-GGUF must be current for LTX-2.3)
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# Idempotent: nodes cloned if missing / deps reinstalled; models skipped if
# already complete, resumed if partial. Individual failures are logged, not fatal.
#
# -----------------------------------------------------------------------------
# >>> WHAT CHANGED vs VastAi_Prov_LTX23_v1.sh  <<<
#
#   THE MODEL LAYOUT IS COMPLETELY DIFFERENT. The old script fetched MONOLITHIC
#   checkpoints (transformer + VAE + encoder glue in one file) into
#   models/checkpoints/, because that is what ComfyUI's six native templates
#   expect. The RuneXX extend workflows are built on Kijai's SPLIT extraction
#   instead -- transformer, video VAE, audio VAE, text projection and text
#   encoder are all separate files in separate folders.
#
#   You cannot mix these. A split workflow will not load a monolithic
#   checkpoint and vice versa. If you want to keep running your existing
#   checkpoint-based template as well, both model sets can coexist on disk --
#   uncomment the MONOLITHIC block near the bottom of the manifest and budget
#   another ~60 GB.
#
#   Split layout this script builds:
#
#     models/diffusion_models/   transformer only (dev + distilled 1.1)
#     models/text_encoders/      gemma_3_12B_it + ltx-2.3_text_projection
#     models/vae/                LTX23_video_vae, LTX23_audio_vae, taeltx2_3
#     models/loras/              distilled 1.1 LoRA (+ your own)
#     models/latent_upscale_models/  spatial upscaler x2 1.1
#     models/onnx/               RIFE ONNX (pre-fetched, see RIFE section)
#     models/tensorrt/rife/      RIFE .trt engines (built on first use)
#
#   >>> taeltx2_3.safetensors IS needed here. The old script correctly noted it
#   >>> belongs to the split pipeline and commented it out. This IS the split
#   >>> pipeline, so it is now active -- it drives the sampler preview.
#
#   Other changes:
#     * ComfyUI-GGUF (city96) added -- required by the RuneXX workflow set.
#     * UPDATE_NODES opt-in, because KJNodes and ComfyUI-GGUF both carry an
#       explicit "must be up to date for LTX-2 support" warning and the old
#       script's git pull line was commented out.
#     * RIFE ONNX files pre-fetched (new section, details there).
#     * Workflow JSONs installed into the ComfyUI sidebar (new section).
#
# -----------------------------------------------------------------------------
# DISK BUDGET
#
#     ltx-2.3-22b-dev_transformer_only_fp8_scaled          23.5 GB
#     ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled 25.2 GB
#     gemma_3_12B_it (bf16)                                24.4 GB
#     ltx-2.3_text_projection_bf16                          2.31 GB
#     distilled 1.1 LoRA                                    2.74 GB
#     LTX23_video_vae / audio_vae / taeltx2_3                1.84 GB
#     spatial upscaler x2 1.1                                ~2 GB
#     RIFE ONNX x3                                          0.06 GB
#                                                         ----------
#                                                            ~82 GB
#
#   >>> DISK: rent >=200 GiB. That leaves room for the Xet chunk cache, your
#   >>> own LoRAs, generated video, and the TensorRT engines.
#
# -----------------------------------------------------------------------------
# GEOMETRY RULES (unchanged, still bites)
#     width and height must be divisible by 32
#     frame count must be divisible by 8, plus 1  -> 9, 17, 25 ... 121, 161
#
# LICENSE: LTX-2 Community License Agreement, not Apache 2.0. Read the terms
# before anything commercial.
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
echo "########## provisioning run (LTX-2.3 EXTEND v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

# ---------------------------------------------------------------------------
# Paths & the Python interpreter ComfyUI actually uses.
# ---------------------------------------------------------------------------
COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"
WORKFLOW_DIR="${COMFY}/user/default/workflows"

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

ensure_pkg git    git      # cloning custom nodes
ensure_pkg curl   curl     # HEAD size checks for the skip/pre-flight logic
ensure_pkg aria2c aria2    # ALWAYS installed -- Civitai section needs it, and
                           # so does any ad-hoc download you do over SSH later.

# ---------------------------------------------------------------------------
# Custom nodes
#
# KJNODES AND COMFYUI-GGUF ARE HARD REQUIREMENTS for the RuneXX workflow set,
# and both carry an explicit "must be up to date for LTX-2 support" warning
# upstream. KJNodes in particular supplies the LTXVAudioVideoMask node that the
# whole extend mechanism is built on -- an old copy will load the workflow with
# red missing-node boxes.
#
# The old script had its `git pull` line commented out, so an existing clone
# was never refreshed. Set UPDATE_NODES=1 in the instance env to enable it.
# Leave unset for reproducible boots once you have a build that works.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/kijai/ComfyUI-KJNodes"              # REQUIRED: LTXVAudioVideoMask etc.
    "https://github.com/city96/ComfyUI-GGUF"                # REQUIRED by the RuneXX set
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    #   Lightricks' own node pack. The RuneXX workflows do not need it; it adds
    #   LTX-specific utilities and their non-native pipelines. Enable only if
    #   you hit a workflow that asks for LTXV* nodes.
    # "https://github.com/Lightricks/ComfyUI-LTXVideo"
)

install_node() {
    local url="$1" name path
    name="$(basename "$url" .git)"
    path="${NODES_DIR}/${name}"
    if [[ -d "$path" ]]; then
        if [[ "${UPDATE_NODES:-0}" == "1" ]]; then
            echo "[node] $name present -> updating (UPDATE_NODES=1)"
            ( cd "$path" && git pull --ff-only ) || echo "[node] git pull FAILED (kept existing): $name"
        else
            echo "[node] $name present (set UPDATE_NODES=1 to git pull)"
            return 0
        fi
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
# CUDA reconciliation (carried over -- still required)
# ComfyUI-RIFE-TensorRT-Auto lists an UNPINNED `cuda-python` in its base
# requirements.txt and auto-installs TensorRT on first node load. Unpinned
# cuda-python resolves to 13.x, which swaps cuda-bindings to 13.x and breaks
# the image's cu12 torch (pins cuda-bindings==12.9.x). Cap it.
# If torch ever moves to a CUDA-13 build, remove this line.
# ---------------------------------------------------------------------------
echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

"$PY" - <<'PYEOF' || true
try:
    import torch
    print("[torch] %s | cuda %s | device: %s" % (
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
except Exception as e:
    print("[torch] not importable: %s" % e)
PYEOF

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE  (unchanged -- Xet-native transport)
#
# Lightricks, Kijai and Comfy-Org repos are Xet-backed. aria2 follows one
# redirect then fires N ranged requests at a single signed URL; requests outside
# its authorised range 403 and the connection dies. hf_xet queries the CAS for
# the reconstruction manifest and fetches xorb ranges instead.
#
# aria2 is still installed and IS used -- for Civitai, which is plain HTTP with
# proper range support and where multi-connection genuinely helps.
# ===========================================================================

"$PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
if ! "$PY" -c "import hf_xet" 2>/dev/null; then
    echo "[provisioning] installing hf_xet for fast HF (Xet) downloads"
    pip_install hf_xet || echo "[provisioning] WARNING: hf_xet install failed -> HF downloads will fall back to the (slower) LFS bridge"
fi

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

hf_resolve_url() { map_url "https://huggingface.co/${1}/resolve/main/${2}"; }

remote_size() {
    local url; url="$(map_url "$1")"
    local headers val
    headers="$(curl -sIL --connect-timeout 15 --max-time 60 "${CURL_AUTH[@]}" "$url" 2>/dev/null)" || return 0
    val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-size"{v=$2} END{if(v!="")print v}')"
    [[ -z "$val" ]] && val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{if(v!="")print v}')"
    printf '%s' "${val//[^0-9]/}"
}

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
    # dl_aria2 <dest_dir> <filename> <url> [extra aria2 args...]
    local dir="$1" name="$2" url; url="$(map_url "$3")"; shift 3
    local dest="${dir}/${name}"
    mkdir -p "$dir"
    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        echo "[model] $name already present, skipping"; return 0
    fi
    local tries=3 n=1
    while (( n <= tries )); do
        echo "[model] downloading $name via aria2 (attempt ${n}/${tries})"
        if aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 \
                  --continue=true --auto-file-renaming=false \
                  --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600 \
                  --max-file-not-found=2 "$@" -d "$dir" -o "$name" "$url"; then
            break
        fi
        echo "[model] attempt ${n} failed for $name"; (( n++ )); sleep 5
    done
    [[ -f "$dest" && ! -f "${dest}.aria2" ]] \
        && echo "[model] $name OK ($(stat -c%s "$dest") bytes)" \
        || echo "[model] DOWNLOAD FAILED / incomplete: $name"
    return 0
}

# ===========================================================================
# CIVITAI DOWNLOADS
#
# Civitai is NOT Xet-backed -- it's plain HTTPS with proper range support, so
# aria2's 16 connections are the right tool here (the opposite of the HF case).
#
# You need a token: civitai.com -> Account Settings -> API Keys. Set it on the
# Vast instance as CIVITAI_TOKEN. Without it most downloads return an HTML
# login page that aria2 will happily save AS your .safetensors -- which then
# fails to load in ComfyUI with a confusing header error. The size sanity check
# below catches exactly that case.
#
# The ID is the MODEL VERSION id, not the model id. On a Civitai model page,
# pick the version, and the download button's URL ends in
#   /api/download/models/<THIS NUMBER>
# ===========================================================================
dl_civitai() {
    # dl_civitai <dest_dir> <dest_filename> <model_version_id>
    local dir="$1" name="$2" vid="$3"
    local dest="${dir}/${name}"
    mkdir -p "$dir"

    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        echo "[civitai] $name already present, skipping"; return 0
    fi
    if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
        echo "[civitai] SKIP $name -- CIVITAI_TOKEN not set in the instance env"
        return 0
    fi

    local url="https://civitai.com/api/download/models/${vid}?token=${CIVITAI_TOKEN}"
    echo "[civitai] downloading $name (version ${vid})"
    aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 \
           --continue=true --auto-file-renaming=false \
           --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600 \
           --max-file-not-found=2 --allow-overwrite=true \
           -d "$dir" -o "$name" "$url" \
        || { echo "[civitai] DOWNLOAD FAILED: $name"; return 0; }

    # Sanity check: an auth failure yields a small HTML body saved under the
    # .safetensors name. Anything under 1 MiB is almost certainly that.
    local sz; sz="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if (( sz < 1048576 )); then
        echo "[civitai] WARNING: $name is only ${sz} bytes -- almost certainly an"
        echo "[civitai] WARNING: error/login page, not a model. Check CIVITAI_TOKEN"
        echo "[civitai] WARNING: and whether the version id is correct. Removing."
        head -c 200 "$dest" 2>/dev/null | tr -d '\0'; echo
        rm -f "$dest"
    else
        echo "[civitai] $name OK (${sz} bytes)"
    fi
}

# ---------------------------------------------------------------------------
# Model manifest
#   HF entries:  hf | dest_dir | dest_filename | repo_id | repo_path
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
LORA="${COMFY}/models/loras"
UPSCALE="${COMFY}/models/latent_upscale_models"
ONNX="${COMFY}/models/onnx"
TRT="${COMFY}/models/tensorrt/rife"
CKPT="${COMFY}/models/checkpoints"

MODELS=(
    # --- Transformers (SPLIT: transformer weights only, no VAE/encoder) ----
    # dev: the quality path. Pair with the distilled LoRA below for the
    # two-stage HQ pipeline, or run full CFG steps on its own.
    "hf|$DIFF|ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors|Kijai/LTX2.3_comfy|diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"
    # distilled 1.1: the fast path, 8 steps at CFG=1 baked in. This is the
    # CURRENT revision -- 1.1 improved aesthetics and audio over the original
    # distilled release, and Lightricks re-cut their IC-LoRAs to match it. Do
    # not pair a 1.1 model with pre-1.1 IC-LoRAs.
    "hf|$DIFF|ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors|Kijai/LTX2.3_comfy|diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
    #   bf16 variants are 42 GB each if you have the disk and want them:
    # "hf|$DIFF|ltx-2.3-22b-dev_transformer_only_bf16.safetensors|Kijai/LTX2.3_comfy|diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors"
    # "hf|$DIFF|ltx-2.3-22b-distilled-1.1_transformer_only_bf16.safetensors|Kijai/LTX2.3_comfy|diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_bf16.safetensors"

    # --- Text encoder: TWO FILES, both mandatory --------------------------
    # Gemma 3 12B at full precision (24.4 GB bf16). The workflows may default
    # to gemma_3_12B_it_fp4_mixed (9.45 GB) -- repoint the loader dropdown.
    # Intermediate options in the same folder: gemma_3_12B_it_fp8_scaled
    # (13.2 GB), gemma_3_12B_it_fpmixed (13.7 GB).
    "hf|$TE|gemma_3_12B_it.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it.safetensors"
    # >>> The text projection is NOT optional and NOT interchangeable. It is a
    # >>> learned linear layer mapping Gemma's output dimension to the one the
    # >>> LTX transformer expects. Omit it and the loader either errors, or --
    # >>> worse -- appears to load while the text conditioning is silently
    # >>> garbage. Both files go in text_encoders/, never models/clip/.
    "hf|$TE|ltx-2.3_text_projection_bf16.safetensors|Kijai/LTX2.3_comfy|text_encoders/ltx-2.3_text_projection_bf16.safetensors"

    # --- VAEs: separate video and audio, plus the preview VAE -------------
    "hf|$VAE|LTX23_video_vae_bf16.safetensors|Kijai/LTX2.3_comfy|vae/LTX23_video_vae_bf16.safetensors"
    "hf|$VAE|LTX23_audio_vae_bf16.safetensors|Kijai/LTX2.3_comfy|vae/LTX23_audio_vae_bf16.safetensors"
    # Tiny VAE (madebyollin). Drives the live sampler preview. Without it you
    # still get previews via KJNodes latentrgb, just much lower resolution.
    # 23.5 MB -- no reason to skip it.
    "hf|$VAE|taeltx2_3.safetensors|Kijai/LTX2.3_comfy|vae/taeltx2_3.safetensors"

    # --- LoRAs ------------------------------------------------------------
    # Distilled 1.1 LoRA -- applies distillation to the DEV transformer for the
    # two-stage HQ path. Must match the 1.1 generation of the model above.
    "hf|$LORA|ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors|Kijai/LTX2.3_comfy|loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"

    # --- Spatial upscaler -------------------------------------------------
    # >>> Use the 1.1 file specifically. The original release of this upscaler
    # >>> had a defect that stamped a splash-logo artifact onto the last few
    # >>> frames of every generation -- which makes extension workflows
    # >>> particularly miserable, since that artifact becomes the reference
    # >>> frames for the next segment.
    "hf|$UPSCALE|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    # --- RIFE ONNX (see the RIFE section below for why these are here) ----
    "hf|$ONNX|rife49_ensemble_True_scale_1_sim.onnx|yuvraj108c/rife-onnx|rife49_ensemble_True_scale_1_sim.onnx"
    "hf|$ONNX|rife48_ensemble_True_scale_1_sim.onnx|yuvraj108c/rife-onnx|rife48_ensemble_True_scale_1_sim.onnx"
    "hf|$ONNX|rife47_ensemble_True_scale_1_sim.onnx|yuvraj108c/rife-onnx|rife47_ensemble_True_scale_1_sim.onnx"

    # --- MONOLITHIC checkpoints (for your OLD native-template workflows) ---
    #   Uncomment to keep the checkpoint-based pipeline working alongside the
    #   split one. ~59 GB extra. These are NOT used by the extend workflows.
    # "hf|$CKPT|ltx-2.3-22b-dev-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-dev-fp8.safetensors"
    # "hf|$CKPT|ltx-2.3-22b-distilled-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-distilled-fp8.safetensors"
)

# ---------------------------------------------------------------------------
# Civitai LoRAs -- fill this in.  Format: dest_dir|dest_filename|version_id
# ---------------------------------------------------------------------------
CIVITAI_LORAS=(
    # "$LORA|my_ltx_style_lora.safetensors|1234567"
    # "$LORA|another_lora.safetensors|7654321"
)

# ---------------------------------------------------------------------------
# Disk pre-flight
# ---------------------------------------------------------------------------
preflight_disk() {
    local need=0 kind a b c d url dest have want
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        [[ "$kind" == "hf" ]] || continue
        url="$(hf_resolve_url "$c" "$d")"; dest="${a}/${b}"
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        want="$(remote_size "$url")"
        [[ -z "$want" ]] && continue
        (( want > have )) && need=$(( need + want - have ))
    done

    mkdir -p "$DIFF"
    local avail; avail="$(df -PB1 "$DIFF" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] models still to fetch: ${h_need};  free on models FS: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Skipping model downloads. Resize the disk and reboot, or drop"
        echo "[provisioning] !!! one transformer (-24 GB) / use a quantised Gemma (-11 to -15 GB)."
        return 1
    fi
    if (( avail - need < 20 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~20GiB). Xet's chunk cache"
        echo "[provisioning] NOTE: uses transient space, TensorRT engines are built on disk, and"
        echo "[provisioning] NOTE: extend workflows write several intermediate video files."
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch
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

echo "=================== CIVITAI LORAS ==================="
if (( ${#CIVITAI_LORAS[@]} == 0 )); then
    echo "[civitai] none configured -- add entries to CIVITAI_LORAS above"
elif [[ -z "${CIVITAI_TOKEN:-}" ]]; then
    echo "[civitai] ${#CIVITAI_LORAS[@]} configured but CIVITAI_TOKEN is unset -- skipping all"
else
    for entry in "${CIVITAI_LORAS[@]}"; do
        IFS='|' read -r cdir cname cvid <<< "$entry"
        dl_civitai "$cdir" "$cname" "$cvid"
    done
fi

# ===========================================================================
# RIFE  --  how the "Auto" loader actually behaves
#
# ComfyUI-RIFE-TensorRT-Auto's loader node is genuinely auto-downloading, but
# only halfway, and it is worth knowing exactly where the line falls:
#
#   1. It looks for a prebuilt engine in  models/tensorrt/rife/
#   2. Not found -> it looks for the ONNX in  models/onnx/
#   3. Not found -> it downloads the ONNX from yuvraj108c/rife-onnx
#   4. Then it BUILDS the TensorRT engine from that ONNX.
#
# Step 3 is what this script pre-empts (above, in the model manifest) so your
# first generation does not stall on a download.
#
# Step 4 CANNOT be pre-empted from a provisioning script. The engine filename
# encodes the model, precision, resolution profile, batch dimensions AND the
# TensorRT version, and the engine itself is compiled for the specific GPU
# architecture it is built on. A .trt built on an A100 is useless on a 4090.
# So: EXPECT THE FIRST RUN AFTER EACH NEW INSTANCE TO PAUSE FOR SEVERAL
# MINUTES while the engine compiles. It is not hung. Subsequent runs load it
# instantly, and it persists on the volume across reboots of the same machine.
#
# >>> GOTCHA WORTH KNOWING: the node's model dropdown offers six options --
# >>> rife47, rife48, rife49, rife417, rife426 and sudo_rife4_269.662. The
# >>> ONNX repo it downloads from only contains THREE: rife47, rife48, rife49.
# >>> Picking any of the other three gives you a 404 at step 3 and a failed
# >>> node. Stick to rife47/48/49. The node's own default (rife49) is fine and
# >>> is what this script pre-fetches first.
#
# Resolution profile note: 'small' (384-1080) is the node default and is the
# wrong choice if you are working at 768x1344 like the LTX geometry rules
# encourage -- 1344 exceeds the profile max. Use 'medium' (672-1312) or
# 'large' (720-1920) and rebuild. Each profile builds its own engine file.
# ===========================================================================
echo "=================== RIFE ==================="
mkdir -p "$ONNX" "$TRT"
echo "[rife] onnx dir:   $ONNX"
echo "[rife] engine dir: $TRT"
if compgen -G "${ONNX}/rife*.onnx" > /dev/null; then
    echo "[rife] pre-fetched ONNX present:"
    for f in "${ONNX}"/rife*.onnx; do echo "[rife]   $(basename "$f") ($(stat -c%s "$f") bytes)"; done
else
    echo "[rife] WARNING: no ONNX files present -- the node will download on first use"
fi
if compgen -G "${TRT}/*.trt" > /dev/null; then
    echo "[rife] existing engines (will load instantly):"
    for f in "${TRT}"/*.trt; do echo "[rife]   $(basename "$f")"; done
else
    echo "[rife] no engines yet -- first generation will spend several minutes building one"
fi

# ===========================================================================
# WORKFLOWS
#
# Cloning node packs does NOT make workflows appear in ComfyUI's sidebar. They
# have to be copied into user/default/workflows/ explicitly, which is what this
# does. Existing files are never overwritten, so your edits survive a reboot --
# to pull a fresh upstream copy, delete the local file and re-provision.
#
# WHICH ONE TO USE for a first segment + independently-LoRA'd extension:
#
#   Extend_Any_Video          <- the one to start with. Takes ANY video in and
#                                extends it once. Because it is a separate run
#                                from whatever generated the input, its LoRA
#                                stack is independent BY CONSTRUCTION. Run your
#                                I2V workflow with LoRA set A, feed the result
#                                here with LoRA set B. Repeat as needed.
#
#   Extend_Any_Video_Multi-Extend_long_video
#                             <- three chained extend groups in one graph, for
#                                roughly a minute of video in one click. Note
#                                its LoRA loader is GLOBAL across all three
#                                groups, so it does NOT give you per-segment
#                                LoRAs without modification.
#
#   Extend_Any_Video_towards_Last-Frame-image
#                             <- extension that targets a supplied end image.
#
#   I2V_T2V_Dev_Full-Steps / _Basic
#                             <- split-model image-to-video for segment one.
#
#   I2V_T2V_Basic_for_checkpoint_models
#                             <- the bridge back to MONOLITHIC checkpoints, if
#                                you re-enable those in the manifest.
#
# Add your own self-hosted workflows to EXTRA_WORKFLOW_URLS at the bottom.
# ===========================================================================
WF_REPO="RuneXX/LTX-2.3-Workflows"
WF_SUBDIR="${WORKFLOW_DIR}/LTX23-Extend"

WORKFLOWS=(
    "Video-2-Video/Extend-Any-Video/LTX-2.3_-_V2V_Extend_Any_Video.json"
    "Video-2-Video/Extend-Any-Video/LTX-2.3_-_V2V_Extend_Any_Video_Multi-Extend_long_video.json"
    "Video-2-Video/Extend-Any-Video/LTX-2.3_-_V2V_Extend_Any_Video_towards_Last-Frame-image.json"
    "LTX-2.3_-_I2V_T2V_Dev_Full-Steps.json"
    "LTX-2.3_-_I2V_T2V_Basic.json"
    # "LTX-2.3_-_I2V_T2V_Basic_for_checkpoint_models.json"
)

# Plain-HTTP workflow URLs (your own gists, S3, etc). Format: filename|url
EXTRA_WORKFLOW_URLS=(
    # "my_custom_extend.json|https://example.com/raw/my_custom_extend.json"
)

install_workflows() {
    mkdir -p "$WF_SUBDIR" || { echo "[workflow] cannot create $WF_SUBDIR"; return 0; }
    local rpath base
    for rpath in "${WORKFLOWS[@]}"; do
        base="$(basename "$rpath")"
        if [[ -f "${WF_SUBDIR}/${base}" ]]; then
            echo "[workflow] $base present (local edits preserved; delete to refresh)"
            continue
        fi
        dl_hf "$WF_SUBDIR" "$base" "$WF_REPO" "$rpath"
    done

    local entry name url
    for entry in "${EXTRA_WORKFLOW_URLS[@]}"; do
        IFS='|' read -r name url <<< "$entry"
        [[ -z "$name" || -z "$url" ]] && continue
        if [[ -f "${WF_SUBDIR}/${name}" ]]; then
            echo "[workflow] $name present (local edits preserved)"
            continue
        fi
        echo "[workflow] fetching $name"
        curl -fsSL --connect-timeout 20 --max-time 300 -o "${WF_SUBDIR}/${name}" "$url" \
            && echo "[workflow] $name OK" \
            || { echo "[workflow] FAILED: $name"; rm -f "${WF_SUBDIR}/${name}"; }
    done

    # ai-dock runs provisioning as root; ComfyUI may not. Keep the tree writable
    # so the UI can save edits back over these files.
    chmod -R a+rwX "${WORKFLOW_DIR}" 2>/dev/null || true
}

echo "=================== WORKFLOWS ==================="
install_workflows
echo "[workflow] installed into: ${WF_SUBDIR}"
echo "[workflow] visible in the ComfyUI sidebar under Workflows -> LTX23-Extend"

# ---------------------------------------------------------------------------
# Layout check
# ---------------------------------------------------------------------------
echo "=================== LAYOUT CHECK ==================="
for p in "${DIFF}/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors" \
         "${DIFF}/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" \
         "${TE}/gemma_3_12B_it.safetensors" \
         "${TE}/ltx-2.3_text_projection_bf16.safetensors" \
         "${VAE}/LTX23_video_vae_bf16.safetensors" \
         "${VAE}/LTX23_audio_vae_bf16.safetensors" \
         "${VAE}/taeltx2_3.safetensors" \
         "${LORA}/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors" \
         "${UPSCALE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
         "${ONNX}/rife49_ensemble_True_scale_1_sim.onnx"; do
    [[ -f "$p" ]] && echo "[layout] OK      ${p#${COMFY}/}" || echo "[layout] MISSING ${p#${COMFY}/}"
done

for d in "${NODES_DIR}/ComfyUI-KJNodes" "${NODES_DIR}/ComfyUI-GGUF"; do
    [[ -d "$d" ]] && echo "[layout] OK      custom_nodes/$(basename "$d")" \
                  || echo "[layout] MISSING custom_nodes/$(basename "$d")  <-- REQUIRED"
done

echo ""
echo "[notes] Workflows -> LTX23-Extend in the ComfyUI sidebar."
echo "[notes] Start with LTX-2.3_-_V2V_Extend_Any_Video.json for a single extension."
echo "[notes] Reminder: width & height divisible by 32; frame count = 8n+1"
echo "[notes] Reminder: repoint the Gemma loader to gemma_3_12B_it.safetensors (bf16)"
echo "[notes] Reminder: RIFE first run builds a TensorRT engine -- several minutes, not a hang"
echo "[notes] Reminder: RIFE dropdown only works for rife47/rife48/rife49"

echo "=================== PROVISIONING COMPLETE ==================="
