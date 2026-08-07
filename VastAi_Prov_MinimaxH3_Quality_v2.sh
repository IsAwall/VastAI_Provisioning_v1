#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# MiniMax H3 (Hailuo 3.0) -- MAXIMUM QUALITY BUILD  ---  v2
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          recommended (rate limits on a ~190 GB pull)
#        CIVITAI_TOKEN=<token>     only if you add Civitai entries below
#        COMFY_AUTO_UPDATE=1       opt-in: git pull ComfyUI to reach >= 0.30.0
#        WANT_REF2VA=0             skip the R2V checkpoint (saves 61.7 GB)
#        WANT_TURBO=0              skip the Turbo LoRA draft tier (saves 1.5 GB)
#        WANT_SEEDVR2=0            skip the SeedVR2 restore stage (saves ~15 GB)
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY -- see below)
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~193 GB with everything on. Provision a 250 GB volume minimum.
#         WANT_REF2VA=0 brings it to ~131 GB -> 180 GB volume.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v1, AND WHY
#
#   v1 was a "get it running" build. This one is configured for the quality
#   ceiling, which changes four things:
#
#   * TEXT ENCODER IS NOW bf16 (48.0 GB) instead of int8_convrot (25.3 GB).
#     Honest caveat: this is the SMALLEST of the quality levers here. Qwen3-VL
#     is an encoder, not a generator -- int8 convrot is close to lossless on
#     it, and what you'd notice is marginal prompt-adherence drift on long
#     multi-shot prompts, not per-frame fidelity. If disk or RAM is the binding
#     constraint, this is the first thing to give back, not the last.
#
#   * TURBO LoRA IS DEMOTED TO A DRAFT TIER. It is a step-distillation LoRA:
#     it trains the model to take much larger jumps along the flow trajectory
#     so 4 steps land where ~20 would. That is a fidelity trade by
#     construction, not an implementation defect, and the author currently
#     flags plastic skin and over-sharp grain on ckpt850. Same call you made
#     cutting Lightning from the Qwen 2509 graph. It is still fetched, because
#     it is genuinely good at cheap prompt iteration -- see THE LADDER below.
#
#   * SAGEATTENTION IS OFF BY DEFAULT. "~2x faster, minimal quality loss" --
#     minimal is not none, and it is an approximate attention kernel. Fine on
#     drafts, off for finals.
#
#   * SeedVR2 IS ADDED AS A RESTORE STAGE. This is the single biggest quality
#     lever available to you, because it is the only way past H3's hard 768px
#     local ceiling. Detail below.
#
# -----------------------------------------------------------------------------
# WHERE THE QUALITY ACTUALLY IS (ranked, largest lever first)
#
#   1. THE RESTORE STAGE. H3-Regenerate-2K was withheld from the weight
#      release; local output is capped at a 768px short edge, full stop. A
#      temporal restorer is the only route past that. SeedVR2 is a
#      diffusion-transformer video restorer -- it reconstructs texture from a
#      generative prior rather than interpolating pixels, and it is temporally
#      conditioned, so it does not flicker the way per-frame ESRGAN does on
#      water, fabric, and fine detail. 7B fp16, not fp8: the repo itself
#      reports quality problems on 7B fp8.
#
#   2. STEP COUNT AND SAMPLER. res_multistep is a second-order integrator --
#      local truncation error per step scales as the square of a first-order
#      method's, so it buys real accuracy per unit of compute rather than just
#      more of it. Baseline is 20 steps; 25-30 gives measurably better content
#      and motion. Below ~15 quality falls off visibly.
#
#   3. RESOLUTION. Run at the native canvas (1344x768 at 16:9, ~1.0 MP), not
#      the template's 0.4 MP preview. Note the model has a 384p FLOOR -- 256p
#      fails outright.
#
#   4. DIFFUSION PRECISION. bf16 (61.7 GB) over int8_convrot (31.7 GB) over
#      pruned_int8 (19.5 GB). Real but smaller than 1-3.
#
#   5. TEXT ENCODER PRECISION. See caveat above. Smallest lever on the list.
#
# -----------------------------------------------------------------------------
# THE LADDER (how to actually work, given the above)
#
#   DRAFT   -> pruned_int8 or Turbo LoRA, 0.4 MP, 4-8 steps, SageAttention on.
#              Iterate the prompt here. Cheap and fast.
#   FINAL   -> fl2va_bf16, bf16 encoder, res_multistep + simple, 25-30 steps,
#              1344x768, no Turbo LoRA, no Sage.
#   RESTORE -> SeedVR2 7B fp16 to 1440p or 2160p.
#   INTERP  -> RIFE TensorRT, 24 -> 48/60 fps.
#
#   CAVEAT ON SEED TRANSFER: locking the seed from a draft does NOT reproduce
#   the shot at final settings. Changing precision, sampler, or step count
#   changes the integration trajectory, so the same seed lands somewhere
#   related but not identical. Use drafts to settle PROMPT and COMPOSITION,
#   then expect to re-roll a few seeds at final settings.
#
# -----------------------------------------------------------------------------
# LICENSE -- READ THIS BEFORE YOU PICK A DATACENTER
#
#   The MiniMax H3 Community License (Aug 2 2026) defines "Applicable
#   Territory" as worldwide EXCLUDING the EU, UK, South Korea, and the USA.
#   The exclusion covers running the weights AND using their outputs.
#   Canada is not on the list -- but a Vast.ai host in a US datacenter is a
#   question worth answering before you commit a 250 GB volume to this.
#
#   Attribution ("MiniMax H3" shown in-product) is required for commercial
#   use; >$20M revenue needs separate written authorization. The Turbo LoRA
#   and SeedVR2 are both Apache-2.0 -- separate licenses, base weights still
#   governed by the above.
#
# -----------------------------------------------------------------------------
# MEMORY: CHECK SYSTEM RAM BEFORE YOU TRUST THIS CONFIG
#
#   bf16 DiT (61.7) + bf16 encoder (48.0) = 109.7 GB of weights against 96 GB
#   of VRAM. That works ONLY because the encoder runs once up front and is
#   evicted before the DiT loads -- but the evicted copy has to land in system
#   RAM or it gets re-read from disk every run. Budget 128 GB RAM minimum,
#   192 GB comfortable. The script prints both and warns if RAM is short.
#
#   Also: on ComfyUI 0.30.x there is a pinned-memory regression that makes
#   model loading pathologically slow. If load times look wrong, launch with
#   --disable-pinned-memory.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (MiniMax H3 QUALITY v2): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

WANT_REF2VA="${WANT_REF2VA:-1}"
WANT_TURBO="${WANT_TURBO:-1}"
WANT_SEEDVR2="${WANT_SEEDVR2:-1}"
echo "[provisioning] tiers: ref2va=${WANT_REF2VA} turbo=${WANT_TURBO} seedvr2=${WANT_SEEDVR2}"

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

ensure_pkg git    git
ensure_pkg curl   curl
ensure_pkg aria2c aria2

# ---------------------------------------------------------------------------
# Host memory check
#
# The bf16 pairing is the whole point of this build and it is RAM-gated, not
# VRAM-gated. Fail loudly here rather than after a 190 GB download.
# ---------------------------------------------------------------------------
echo "=================== HOST MEMORY ==================="
RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$RAM_GB" ]]; then
    echo "[mem] system RAM: ${RAM_GB} GB"
    if (( RAM_GB < 128 )); then
        echo "[mem] !!! Under 128 GB. The bf16 DiT + bf16 encoder pairing will thrash:"
        echo "[mem] !!! the evicted text encoder cannot stay resident and gets re-read"
        echo "[mem] !!! from disk on every run. Either move to a higher-RAM machine, or"
        echo "[mem] !!! switch the text encoder to int8_convrot in the manifest below."
    elif (( RAM_GB < 192 )); then
        echo "[mem] adequate. 192 GB+ would give comfortable headroom for offload."
    else
        echo "[mem] comfortable."
    fi
fi

# ---------------------------------------------------------------------------
# ComfyUI version gate
# Native H3 nodes landed in 0.30.0 (Comfy-Org/ComfyUI PR #15224). On an older
# image the graph opens with red nodes and no obvious cause.
# ---------------------------------------------------------------------------
comfy_version() {
    local v=""
    if [[ -f "${COMFY}/comfyui_version.py" ]]; then
        v="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${COMFY}/comfyui_version.py" | head -1)"
    fi
    if [[ -z "$v" && -d "${COMFY}/.git" ]]; then
        v="$(git -C "$COMFY" describe --tags --abbrev=0 2>/dev/null | tr -d 'v')"
    fi
    printf '%s' "$v"
}

echo "=================== COMFYUI VERSION ==================="
CV="$(comfy_version)"
MIN_CV="0.30.0"
if [[ -z "$CV" ]]; then
    echo "[comfy] version undetectable -- verify manually that you are on >= ${MIN_CV}"
elif [[ "$(printf '%s\n%s\n' "$MIN_CV" "$CV" | sort -V | head -1)" == "$MIN_CV" ]]; then
    echo "[comfy] version ${CV} -- OK (native H3 support present)"
else
    echo "[comfy] !!! version ${CV} is BELOW ${MIN_CV} -- H3 nodes will not exist."
    if [[ "${COMFY_AUTO_UPDATE:-0}" == "1" && -d "${COMFY}/.git" ]]; then
        echo "[comfy] COMFY_AUTO_UPDATE=1 -> pulling"
        git -C "$COMFY" fetch --all --tags --prune && git -C "$COMFY" pull --ff-only \
            && pip_install --no-cache-dir -r "${COMFY}/requirements.txt"
        echo "[comfy] now at: $(comfy_version)"
    else
        echo "[comfy] !!! Set COMFY_AUTO_UPDATE=1 and reboot, or pick a newer image."
        echo "[comfy] !!! Auto-update is opt-in because a pull can drag the venv"
        echo "[comfy] !!! out from under pinned torch/cu128 on Blackwell."
    fi
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>]
# SeedVR2 needs a specific directory name -- its own docs and CLI paths assume
# custom_nodes/seedvr2_videoupscaler, not the repo basename.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
)
[[ "$WANT_TURBO"   == "1" ]] && NODES+=( "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo" )
[[ "$WANT_SEEDVR2" == "1" ]] && NODES+=( "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler|seedvr2_videoupscaler" )

# Optional: explicit dual-sigma sampler with manual video/audio shift control
# (12 / 3) and a 50-sigma-point schedule. Genuinely more control than the
# stock graph, BUT it overlaps the native H3 nodes and the Turbo nodes, and
# three H3 node packs in one install is a recipe for node-name collisions.
# Add it deliberately, after the stock path works.
#   "https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3"

# Packages where staleness silently breaks compatibility -> pull every boot.
NODES_TRACK_HEAD=(
    "ComfyUI-MiniMax-H3-Turbo"
    "seedvr2_videoupscaler"
)

install_node() {
    local spec="$1" url name path
    url="${spec%%|*}"
    if [[ "$spec" == *"|"* ]]; then name="${spec##*|}"; else name="$(basename "$url" .git)"; fi
    path="${NODES_DIR}/${name}"
    if [[ -d "$path" ]]; then
        if [[ " ${NODES_TRACK_HEAD[*]} " == *" ${name} "* ]]; then
            echo "[node] $name present -> pulling (tracked at HEAD)"
            git -C "$path" pull --ff-only || echo "[node] pull failed: $name (keeping local)"
        else
            echo "[node] $name present"
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

echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# SageAttention -- opt-in, and DRAFTS ONLY on this build
#
# It is an approximate attention kernel: quantized QK^T with a smoothing
# correction. ~2x throughput, and the error is small but nonzero. That is a
# fine trade while you are iterating prompts and a bad one on a final render,
# so it is not installed unless you ask.
#
# Note you do NOT need KJNodes' patch node for this -- recent ComfyUI takes
# --use-sage-attention as a launch flag directly.
# ---------------------------------------------------------------------------
if [[ "${INSTALL_SAGE:-0}" == "1" ]]; then
    echo "=================== SAGEATTENTION ==================="
    if "$PY" -c "import sageattention" 2>/dev/null; then
        echo "[sage] already installed"
    else
        echo "[sage] attempting source build (slow -- may fail)"
        pip_install --no-cache-dir sageattention \
            || echo "[sage] FAILED -- grab a wheel matching your torch/CUDA from github.com/woct0rdho/SageAttention/releases"
    fi
    echo "[sage] REMINDER: launch with --use-sage-attention for drafts, and"
    echo "[sage] REMINDER: relaunch without it for final renders."
fi

"$PY" - <<'PYEOF' || true
try:
    import torch
    print("[torch] %s | cuda %s | device: %s" % (
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print("[torch] vram: %.1f GB total" % (total / 1024**3))
except Exception as e:
    print("[torch] not importable: %s" % e)
PYEOF

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE
# HF via hf_xet (Xet-backed repos break aria2's ranged requests);
# Civitai via aria2 (plain HTTPS, proper range support).
# ===========================================================================

"$PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
if ! "$PY" -c "import hf_xet" 2>/dev/null; then
    echo "[provisioning] installing hf_xet for fast HF (Xet) downloads"
    pip_install hf_xet || echo "[provisioning] WARNING: hf_xet install failed -> slower LFS bridge"
fi

export HF_HOME="${WORKSPACE:-/workspace}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"
mem_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$mem_gb" ]] && (( mem_gb >= 64 )); then
    export HF_XET_HIGH_PERFORMANCE=1
    echo "[provisioning] ${mem_gb} GB RAM -> HF_XET_HIGH_PERFORMANCE=1"
fi

CURL_AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    echo "[provisioning] HF_TOKEN detected -> authenticated HF downloads"
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
            echo "[model] WARNING: $name size ${have} != expected ${want} (kept for resume)"
        else
            echo "[model] $name OK (${have} bytes)"
        fi
    else
        echo "[model] DOWNLOAD FAILED: $name (will retry next boot)"
    fi
}

# ===========================================================================
# CIVITAI (carried over from the LTX script -- manifest is empty by default)
#
# H3 finetunes and style LoRAs are already landing on Civitai. Auth goes in a
# header so the token never reaches provisioning.log; falls back to the
# query-string form only if the header is rejected, and says so when it does.
# ===========================================================================
CIVITAI_RESERVE_GB="${CIVITAI_RESERVE_GB:-20}"

dl_civitai() {
    # dl_civitai <dest_dir> <dest_filename> <full_url>
    local dir="$1" name="$2" url="$3"
    local dest="${dir}/${name}"
    mkdir -p "$dir"

    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        echo "[civitai] $name already present ($(stat -c%s "$dest") bytes), skipping"; return 0
    fi
    if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
        echo "[civitai] SKIP $name -- CIVITAI_TOKEN not set in the instance env"
        return 0
    fi

    local common=(-x 16 -s 16 -k 1M --file-allocation=none --summary-interval=15
                  --continue=true --auto-file-renaming=false --allow-overwrite=true
                  --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600
                  --max-file-not-found=2)

    echo "[civitai] downloading $name"
    echo "[civitai]   from: ${url}"          # token is NOT in this URL
    if ! aria2c "${common[@]}" \
                --header="Authorization: Bearer ${CIVITAI_TOKEN}" \
                -d "$dir" -o "$name" "$url"; then
        echo "[civitai] header auth failed -- retrying with query-string token"
        echo "[civitai] NOTE: this form puts the token in aria2's URL output, which"
        echo "[civitai] NOTE: lands in ${WORKSPACE:-/workspace}/provisioning.log."
        echo "[civitai] NOTE: Rotate the key afterwards if that logfile is shared."
        local sep="?"; [[ "$url" == *\?* ]] && sep="&"
        aria2c "${common[@]}" -d "$dir" -o "$name" "${url}${sep}token=${CIVITAI_TOKEN}" \
            || { echo "[civitai] DOWNLOAD FAILED: $name"; return 0; }
    fi

    local sz; sz="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if (( sz < 1048576 )); then
        echo "[civitai] WARNING: $name is only ${sz} bytes -- almost certainly an"
        echo "[civitai] WARNING: auth/error response, not a model. First 200 bytes:"
        head -c 200 "$dest" 2>/dev/null | tr -d '\0'; echo
        rm -f "$dest"
        return 0
    fi
    echo "[civitai] $name OK (${sz} bytes)"
}

# ---------------------------------------------------------------------------
# Manifest
#
# Sizes below are the actual Comfy-Org repo figures, not estimates.
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
WF="${COMFY}/user/default/workflows"
SVR="${COMFY}/models/SEEDVR2"

# --- Civitai:  dest_dir | dest_filename | full_url ---
CIVITAI_FILES=(
    # "$LORA|some_h3_style_lora.safetensors|https://civitai.com/api/download/models/1234567"
)

# --- Hugging Face:  hf | dest_dir | dest_filename | repo_id | repo_path ---
MODELS=(
    # === DIFFUSION MODEL -- FULL bf16 ====================================
    # fl2va = t2v / i2v / first-last-frame.  61.7 GB.
    # Alternatives if you need the disk back: int8_convrot (31.7 GB) is a
    # genuinely small step down; pruned_int8 (19.5 GB) is the template
    # default and a visible one.
    "hf|$DIFF|minimax_h3_fl2va_bf16.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    # "hf|$DIFF|minimax_h3_fl2va_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"

    # === TEXT ENCODER -- FULL bf16 =======================================
    # Qwen3-VL-32B, 48.0 GB. See the header caveat: this is the smallest
    # quality lever in the build and the first thing to trade away.
    # int8_convrot is 25.3 GB; nvfp4_awq is 14.6 GB and the real step down.
    "hf|$TE|qwen3vl_32b_minimax_h3_bf16.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
    # "hf|$TE|qwen3vl_32b_minimax_h3_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

    # === VAEs -- BOTH REQUIRED (4.9 + 0.6 GB) ============================
    # The audio VAE is not optional even for silent output: the audio stream
    # is generated in the same pass and has to be decoded. Missing it is the
    # usual cause of "my output has no sound" -- along with a missing
    # VAEDecodeAudio node feeding SaveVideo.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"
)

# === REF2VA -- reference-driven, SEPARATE WEIGHTS (61.7 GB) ==============
# Identity / style / motion / camera / voice from up to 9 images, 3 videos,
# 3 audio clips. The R2V template will NOT run on fl2va.
# Worth knowing given your pose-control thread: this is appearance and
# identity conditioning, the same class as Wan 2.7 R2V. It is not spatially
# aligned pose control, so expect the same mismatch you hit there. Useful for
# locking a character across shots; not a DWPose substitute.
if [[ "$WANT_REF2VA" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|minimax_h3_ref2va_bf16.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_ref2va_bf16.safetensors"
)
fi

# === TURBO LoRA -- DRAFT TIER ONLY (~1.5 GB) =============================
# Not on the quality path. Kept because iterating a prompt at 4 steps instead
# of 25 is worth a lot when you are still deciding what the shot is.
if [[ "$WANT_TURBO" == "1" ]]; then
MODELS+=(
    "hf|$LORA|minimax_h3_turbo_4step_ema_ckpt850.safetensors|larryvrh/MiniMax-H3-Turbo-Lora|minimax_h3_turbo_4step_ema_ckpt850.safetensors"
    "hf|$WF|minimax_h3_t2v_turbo.json|larryvrh/MiniMax-H3-Turbo-Lora|minimax_h3_t2v_turbo.json"
)
fi

# === SeedVR2 RESTORE STAGE (~15 GB) ======================================
# 7B fp16 deliberately: the node's own README reports quality problems with
# 7B fp8 and says to use fp16 whenever it fits. At 96 GB it fits easily.
# Best-effort prefetch -- if a filename has moved, the node auto-downloads on
# first use with SHA256 validation, so a miss here costs a stall, not a break.
if [[ "$WANT_SEEDVR2" == "1" ]]; then
MODELS+=(
    "hf|$SVR|seedvr2_ema_7b_fp16.safetensors|numz/SeedVR2_comfyUI|seedvr2_ema_7b_fp16.safetensors"
    "hf|$SVR|ema_vae_fp16.safetensors|numz/SeedVR2_comfyUI|ema_vae_fp16.safetensors"
)
fi

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

    local cdir cname curl_ pending=0
    for entry in "${CIVITAI_FILES[@]}"; do
        IFS='|' read -r cdir cname curl_ <<< "$entry"
        [[ -f "${cdir}/${cname}" && ! -f "${cdir}/${cname}.aria2" ]] || (( pending++ ))
    done
    (( pending > 0 )) && need=$(( need + pending * CIVITAI_RESERVE_GB * 1024*1024*1024 ))

    mkdir -p "$DIFF"
    local avail; avail="$(df -PB1 "$DIFF" | awk 'NR==2{print $4}')"
    local margin=$(( 15 * 1024*1024*1024 ))   # bf16 loads + SeedVR2 both spill
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Cheapest give-backs, in the order you should make them:"
        echo "[provisioning] !!!   WANT_REF2VA=0                     -61.7 GB"
        echo "[provisioning] !!!   text encoder -> int8_convrot      -22.7 GB"
        echo "[provisioning] !!!   diffusion    -> int8_convrot      -30.0 GB"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
if (( ${#CIVITAI_FILES[@]} > 0 )); then
    echo "=================== CIVITAI ==================="
    if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
        echo "[civitai] CIVITAI_TOKEN unset -- skipping ALL civitai downloads."
    else
        for entry in "${CIVITAI_FILES[@]}"; do
            IFS='|' read -r cdir cname curl_ <<< "$entry"
            dl_civitai "$cdir" "$cname" "$curl_"
        done
    fi
fi

echo "=================== HUGGING FACE ==================="
echo "[provisioning] NOTE: full build is ~193 GB. First boot on a fresh volume"
echo "[provisioning] NOTE: is a long pull -- the bf16 DiT alone is 61.7 GB."
if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        case "$kind" in
            hf) dl_hf "$a" "$b" "$c" "$d" ;;
            *)  echo "[model] unknown manifest kind: '$kind' in: $entry" ;;
        esac
    done
else
    echo "[provisioning] HF model phase skipped (see disk warning above)"
fi

# ---------------------------------------------------------------------------
# Layout check
# ---------------------------------------------------------------------------
echo "=================== LAYOUT CHECK ==================="
for entry in "${CIVITAI_FILES[@]}"; do
    IFS='|' read -r cdir cname curl_ <<< "$entry"
    if [[ -f "${cdir}/${cname}" ]]; then
        echo "[layout] OK      ${cdir#${COMFY}/}/${cname}  ($(numfmt --to=iec "$(stat -c%s "${cdir}/${cname}")" 2>/dev/null))"
    else
        echo "[layout] MISSING ${cdir#${COMFY}/}/${cname}"
    fi
done
for entry in "${MODELS[@]}"; do
    IFS='|' read -r kind a b c d <<< "$entry"
    [[ "$kind" == "hf" ]] || continue
    if [[ -f "${a}/${b}" ]]; then
        echo "[layout] OK      ${a#${COMFY}/}/${b}  ($(numfmt --to=iec "$(stat -c%s "${a}/${b}")" 2>/dev/null))"
    else
        echo "[layout] MISSING ${a#${COMFY}/}/${b}"
    fi
done

for nd in ComfyUI-MiniMax-H3-Turbo seedvr2_videoupscaler ComfyUI-RIFE-TensorRT-Auto; do
    if [[ -d "${NODES_DIR}/${nd}" ]]; then
        echo "[layout] OK      custom_nodes/${nd} @ $(git -C "${NODES_DIR}/${nd}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    else
        echo "[layout] absent  custom_nodes/${nd}"
    fi
done

# ---------------------------------------------------------------------------
# Operating notes
# ---------------------------------------------------------------------------
cat <<'NOTES'

=================== FINAL-RENDER SETTINGS ===================

MODELS
  diffusion    minimax_h3_fl2va_bf16.safetensors        (ref2va for R2V)
  text encoder qwen3vl_32b_minimax_h3_bf16.safetensors
  vae          minimax_h3_video_vae_fp16 + minimax_h3_audio_vae_fp32
  LoRA         none. Bypass or delete the Turbo nodes for finals.

SAMPLER
  T2V / I2V    res_multistep + simple scheduler, 25-30 steps
  R2V          res_multistep + beta (or normal) scheduler -- reported to beat
               simple on reference-heavy prompts
  Below ~15 steps quality drops visibly. 20 is the community baseline; the
  25-30 band is where you stop getting much for the wall-clock.

RESOLUTION -- both constraints are hard, and there is a FLOOR
  Resolution Selector: 16:9, Megapixels ~1.0, Multiple 32  ->  1344x768
  That is the native canvas: 768px short edge, capped 768x1344.
  Minimum is 384p. 256p fails outright rather than degrading.
  frames: 17k+5 grid at 24 fps. 124 ~ 5 s, 362 ~ 15 s. Validated 124-362.

R2V EXTRAS
  ref_image_size = max   (keeps references up to 2048px short edge -- stronger
                          identity fidelity, slower. "match" is the speed option.)
  Tag every reference in prompt order: <Picture 1>, <Video 1>, <Audio 1>, and
  state explicitly which one drives identity vs style vs motion vs voice.
  Reference ORDER is significant -- reordering changes the conditioning.

RESTORE STAGE -- the biggest quality lever you have
  SeedVR2 Load DiT Model: seedvr2_ema_7b_fp16.safetensors   (NOT fp8)
  SeedVR2 Load VAE Model: ema_vae_fp16.safetensors
  SeedVR2 Video Upscaler: resolution 1440 or 2160
                          batch_size must follow 4n+1 (5, 9, ..., 81, ...)
  Batch size is the temporal-consistency dial: it is the window the restorer
  sees at once, so larger = less drift between segments. At 96 GB push it high
  -- 81 is a sane starting point. Ideally match it to shot length.
  Optionally chain SeedVR2 Torch Compile Settings (max-autotune / inductor) --
  that is pure speed, no quality cost.
  Do NOT apply the downscale-first trick here. That exists to give a restorer
  clean input from dirty compressed footage; H3 output at native canvas is
  already clean, so feed it directly.

ORDER OF OPERATIONS
  generate -> SeedVR2 restore -> RIFE interpolate -> encode
  Restore before interpolation: RIFE synthesises intermediate frames from what
  it is given, so interpolating first just asks SeedVR2 to restore invented
  frames, and doubles its workload for nothing.

AUDIO -- 32 kHz stereo, and the weakest part of the release
  Expect repeated syllables and occasionally unrelated audio on dialogue.
  Audio problems are usually a SCHEDULER mismatch, not a broken model: video
  and audio ride separate flow schedules (video shift 12, audio shift 3) and
  H3 is unusually sensitive to sampler configuration. Before assuming a file
  is bad, check the scheduler.

IF MODEL LOADING IS PATHOLOGICALLY SLOW
  ComfyUI 0.30.x has a pinned-memory regression. Launch with
  --disable-pinned-memory.

STILL NOT AVAILABLE LOCALLY
  H3-Regenerate-2K and H3-Context-IR. 2K and hosted prompt restructuring are
  API-only. SeedVR2 is your substitute for the first; there is no substitute
  for the second beyond writing better prompts yourself.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
