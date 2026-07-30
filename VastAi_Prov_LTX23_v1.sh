#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# LTX-2.3 (22B audio-video)  ---  v1   (max-quality build)
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text (GitHub "Raw"
#      button URL, Gist raw URL, pastebin raw URL).
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Optional env vars:
#        HF_TOKEN=<token>        authenticated HF transfers, avoids 429s
#        CIVITAI_TOKEN=<token>   REQUIRED for the Civitai LoRA section below
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# Idempotent: nodes cloned if missing / deps reinstalled; models skipped if
# already complete, resumed if partial. Individual failures are logged, not fatal.
#
# -----------------------------------------------------------------------------
# WHAT'S DIFFERENT ABOUT LTX-2.3 vs THE WAN SCRIPTS
#
#   * It is NATIVE. ComfyUI ships six LTX-2.3 templates (T2V, I2V, FLF2V,
#     IA2V, IC-LoRA Union Control, ID-LoRA) and needs no custom nodes for any
#     of them. The node list below is your usual working set, not a dependency.
#
#   * Checkpoints are MONOLITHIC. Unlike Wan's split diffusion_models /
#     text_encoders / vae layout, these are full checkpoints -- transformer,
#     VAE and encoder glue in one file -- so they go in models/checkpoints/
#     and load through CheckpointLoaderSimple. There is NO separate VAE file
#     to fetch for the native workflows. (taeltx2_3.safetensors, which every
#     LTX guide tells you is mandatory, belongs to Kijai's SPLIT workflows.
#     Commented at the bottom if you go that way.)
#
#   * Text encoder is Gemma 3 12B, not umT5.
#
#   * It generates SYNCHRONISED AUDIO natively. That's the headline feature
#     and there's nothing equivalent in the Wan stack.
#
#   * >>> DIFFERENT GEOMETRY RULES FROM WAN. Do not carry your Wan numbers
#     >>> over. Per the Lightricks model card:
#     >>>     width and height must be divisible by 32   (Wan: 16)
#     >>>     frame count must be divisible by 8, plus 1 (Wan: 4n+1)
#     >>> So valid lengths are 9, 17, 25 ... 121, 161. Your Wan habit of 81
#     >>> frames is NOT valid here -- 81 = 8*10+1, so it happens to work, but
#     >>> 77 does not. Check before assuming.
#     >>> 768x1344 carries over fine (both divisible by 32).
#
#   * LICENSE: LTX-2 Community License Agreement, NOT Apache 2.0 like the Wan
#     weights. Open weights, but read the terms before anything commercial.
#
# -----------------------------------------------------------------------------
# TWO CHECKPOINTS, because the six templates don't agree:
#
#     dev-fp8        T2V, I2V, IA2V, ID-LoRA     full CFG sampling
#     distilled-fp8  FLF2V, IC-LoRA              8 steps, CFG=1
#
#   Both are fetched. If you only care about one branch, comment the other and
#   save ~29 GB.
#
#   Precision note: Lightricks publishes the ComfyUI-ready checkpoints as fp8,
#   and the templates are pre-set to those filenames -- so that's what this
#   fetches. The bf16 dev weights exist in Lightricks/LTX-2.3 but I could not
#   verify the exact filename, so it's commented rather than guessed. The text
#   encoder IS taken at full precision (24.4 GB bf16 rather than the 9.45 GB
#   fp4_mixed the templates default to), which is the one place the quality
#   choice was actually available.
#
#     ltx-2.3-22b-dev-fp8                        ~29 GB
#     ltx-2.3-22b-distilled-fp8                   29.5 GB
#     gemma_3_12B_it (bf16)                       24.4 GB
#     distilled 1.1 LoRA + gemma abliterated LoRA  ~5 GB
#     spatial upscaler x2 1.1                      ~2 GB
#     IC-LoRA union control + MoGe + ID-LoRA       ~4 GB
#                                               ----------
#                                                 ~94 GB
#
#   >>> DISK: rent >=200 GiB.
#
#   >>> IC-LoRA Union Control is the one to look at first given what you've
#   >>> been building -- it takes depth, pose, or edge guidance from a driving
#   >>> video, which is the same job as Wan Fun Control but on a model that
#   >>> also produces audio. Worth an A/B against VACE-Fun on the same clip.
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
echo "########## provisioning run (LTX-2.3 v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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

ensure_pkg git    git      # cloning custom nodes
ensure_pkg curl   curl     # HEAD size checks for the skip/pre-flight logic
ensure_pkg aria2c aria2    # ALWAYS installed -- Civitai section needs it, and
                           # so does any ad-hoc download you do over SSH later.

# ---------------------------------------------------------------------------
# Custom nodes
#
# LTX-2.3's six native templates need NONE of these -- they're your standard
# working set for video I/O, resizing, control-signal preprocessing (for the
# IC-LoRA branch) and output interpolation.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    #   Lightricks' own node pack. The six native templates do not need it;
    #   it adds LTX-specific utilities and their non-native pipelines. Enable
    #   only if you hit a workflow that asks for LTXV* nodes.
    # "https://github.com/Lightricks/ComfyUI-LTXVideo"
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
# DOWNLOAD INFRASTRUCTURE  (unchanged from v5 -- Xet-native transport)
#
# Lightricks and Comfy-Org repos are Xet-backed. aria2 follows one redirect
# then fires N ranged requests at a single signed URL; requests outside its
# authorised range 403 and the connection dies. hf_xet queries the CAS for the
# reconstruction manifest and fetches xorb ranges instead.
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
CKPT="${COMFY}/models/checkpoints"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
UPSCALE="${COMFY}/models/latent_upscale_models"
GEOM="${COMFY}/models/geometry_estimation"

MODELS=(
    # --- Checkpoints (monolithic: transformer + VAE + encoder glue) --------
    # dev: used by T2V, I2V, IA2V, ID-LoRA. Full CFG sampling.
    "hf|$CKPT|ltx-2.3-22b-dev-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-dev-fp8.safetensors"
    # distilled: used by FLF2V and IC-LoRA. 8 steps, CFG=1 -- baked in, so
    # don't "fix" the low step count when you open those templates.
    "hf|$CKPT|ltx-2.3-22b-distilled-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-distilled-fp8.safetensors"
    #   bf16 dev weights exist in Lightricks/LTX-2.3 but I could not confirm the
    #   exact filename -- check the repo file list and fill this in rather than
    #   trusting a guess:
    # "hf|$CKPT|ltx-2.3-22b-dev.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-dev.safetensors"

    # --- Text encoder: Gemma 3 12B, FULL PRECISION ------------------------
    # 24.4 GB bf16. The templates are pre-set to gemma_3_12B_it_fp4_mixed
    # (9.45 GB), so you WILL need to repoint the loader dropdown. fp4 is a very
    # aggressive quantisation for a text encoder and you have the VRAM not to
    # take it. Intermediate options in the same folder if you want them:
    # gemma_3_12B_it_fp8_scaled (13.2 GB), gemma_3_12B_it_fpmixed (13.7 GB).
    "hf|$TE|gemma_3_12B_it.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it.safetensors"
    # "hf|$TE|gemma_3_12B_it_fp4_mixed.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"

    # --- LoRAs ------------------------------------------------------------
    # Distilled 1.1 LoRA -- applies distillation to the dev checkpoint. This is
    # the "speed" LoRA of the LTX stack, but unlike lightx2v it's part of the
    # stock template wiring rather than an optional add-on, so it's active here.
    "hf|$LORA|ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors|Comfy-Org/ltx-2.3|split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
    # Gemma abliterated LoRA -- applied to the TEXT ENCODER, not the video
    # model. Loosens Gemma's instruction-tuned refusal behaviour so prompts
    # get encoded as written. Used by every template except FLF2V.
    "hf|$LORA|gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors|Comfy-Org/ltx-2|split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
    # IC-LoRA Union Control -- depth / pose / edge guidance from a driving
    # video. The closest thing here to Wan Fun Control, on a model that also
    # generates audio. Pairs with the DISTILLED checkpoint.
    "hf|$LORA|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    # ID-LoRA TalkVid -- identity + voice from a reference image and audio clip.
    "hf|$LORA|ltx-2.3-id-lora-talkvid-3k.safetensors|Comfy-Org/ltx-2.3|split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors"

    # --- Spatial upscaler (multi-stage / multiscale pipelines) ------------
    "hf|$UPSCALE|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    # --- Geometry estimation for the IC-LoRA control branch ---------------
    "hf|$GEOM|moge_2_vitl_normal_fp16.safetensors|Comfy-Org/MoGe|geometry_estimation/moge_2_vitl_normal_fp16.safetensors"

    # --- Kijai SPLIT-workflow components (NOT needed for native templates) --
    #   Only if you switch to Kijai's split pipeline, which loads a
    #   transformer-only file plus these separately. This is where
    #   taeltx2_3.safetensors actually belongs.
    # "hf|${COMFY}/models/vae|taeltx2_3.safetensors|Kijai/LTX2.3_comfy|vae/taeltx2_3.safetensors"
    # "hf|$TE|ltx-2.3_text_projection_bf16.safetensors|Kijai/LTX2.3_comfy|text_encoders/ltx-2.3_text_projection_bf16.safetensors"
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

    mkdir -p "$CKPT"
    local avail; avail="$(df -PB1 "$CKPT" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] models still to fetch: ${h_need};  free on models FS: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Skipping model downloads. Resize the disk and reboot, or drop"
        echo "[provisioning] !!! one checkpoint (-29 GB) / use the fp4 text encoder (-15 GB)."
        return 1
    fi
    if (( avail - need < 20 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~20GiB). Xet's chunk cache"
        echo "[provisioning] NOTE: uses transient space, and LTX writes audio+video output."
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

# ---------------------------------------------------------------------------
# Layout check
# ---------------------------------------------------------------------------
echo "=================== LAYOUT CHECK ==================="
for p in "${CKPT}/ltx-2.3-22b-dev-fp8.safetensors" \
         "${CKPT}/ltx-2.3-22b-distilled-fp8.safetensors" \
         "${TE}/gemma_3_12B_it.safetensors" \
         "${LORA}/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" \
         "${LORA}/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
         "${LORA}/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
         "${LORA}/ltx-2.3-id-lora-talkvid-3k.safetensors" \
         "${UPSCALE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
         "${GEOM}/moge_2_vitl_normal_fp16.safetensors"; do
    [[ -f "$p" ]] && echo "[layout] OK      ${p#${COMFY}/}" || echo "[layout] MISSING ${p#${COMFY}/}"
done

echo "[workflow] Native LTX-2.3 templates: Workflow -> Browse Templates -> Video"
echo "[workflow]   T2V / I2V / FLF2V / IA2V / IC-LoRA Union Control / ID-LoRA"
echo "[workflow] Reminder: width & height divisible by 32; frame count = 8n+1"
echo "[workflow] Reminder: repoint the Gemma loader to gemma_3_12B_it.safetensors (bf16)"

echo "=================== PROVISIONING COMPLETE ==================="
