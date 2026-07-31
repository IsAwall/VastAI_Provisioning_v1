#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# LTX-2.3 with CIVITAI CHECKPOINT (10Eros v1.4)  ---  v2
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables to set on the instance:
#        CIVITAI_TOKEN=<token>   REQUIRED -- nothing in the Civitai section
#                                downloads without it
#        HF_TOKEN=<token>        optional but recommended for the HF files
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v1
#
#   * The two official LTX-2.3 checkpoints (dev-fp8, distilled-fp8) are now
#     COMMENTED OUT, replaced by the Civitai 10Eros v1.4 checkpoint.
#     They're left in place rather than deleted -- see "A/B" below.
#
#   * dl_civitai() was rewritten to take a FULL URL instead of a bare version
#     id. Your link is on civitai.red rather than civitai.com and carries a
#     fileId query parameter, neither of which the v1 helper could express.
#
#   * Auth now goes in an Authorization header rather than the query string.
#     This matters: aria2 echoes the URL it's fetching into stdout, and this
#     script tees stdout to /workspace/provisioning.log, so a ?token= URL
#     would write your API key into a logfile that persists on the volume and
#     shows up in Vast's log viewer. The header keeps it out of both. There's
#     an automatic fallback to the query-string form if the header is
#     rejected, with a warning so you know it happened.
#
#   * Disk pre-flight reserves a flat allowance for the Civitai file, because
#     Civitai won't reliably answer a HEAD with a content-length the way HF's
#     x-linked-size does. Tune CIVITAI_RESERVE_GB if the checkpoint is larger
#     than the default 60 GB assumption.
#
# -----------------------------------------------------------------------------
# WHAT IS STILL FETCHED FROM HUGGING FACE, AND WHY
#
#   A checkpoint finetune replaces the transformer (and usually carries its own
#   VAE). It does NOT bundle:
#       - the Gemma 3 text encoder
#       - the distilled / abliterated LoRAs
#       - the spatial upscaler
#       - MoGe geometry estimation
#   So those stay in the manifest. If 10Eros turns out to ship its own encoder
#   or VAE, you'll have a redundant copy rather than a broken install.
#
# -----------------------------------------------------------------------------
# THINGS I CANNOT VERIFY ABOUT THE CIVITAI FILE -- check these on first run
#
#   1. FILE TYPE. If 10Eros v1.4 is a LoRA rather than a full checkpoint, it
#      belongs in models/loras/ and you should re-enable the base checkpoints
#      to load it against. The destination below assumes a full checkpoint
#      because you described it as replacing the base model. The script prints
#      the downloaded size, which tells you immediately: a full LTX-2.3
#      checkpoint is tens of GB, a LoRA is hundreds of MB to a few GB.
#
#   2. DISTILLED-LORA COMPATIBILITY. Community finetunes don't always accept
#      the official distilled LoRA -- the same way LTX-2 LoRAs frequently
#      don't attach cleanly to 2.3. If motion comes out stiff or colours
#      drift, bypass the LoRA loader (Ctrl+B) and use the reduced-strength or
#      full-CFG settings from the v1 notes instead of assuming the checkpoint
#      is bad.
#
#   3. WHETHER THE TOKEN CARRIES ACROSS DOMAINS. civitai.red and civitai.com
#      share an account system, but if auth fails on .red specifically, try
#      generating a fresh key while logged into that domain.
#
#   4. A/B. Keep ~60 GB free and uncomment the dev-fp8 line to have both. When
#      a finetune misbehaves, having the base model on the same box to compare
#      against saves a lot of guessing.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (LTX-2.3 civitai v2): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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

ensure_pkg git    git
ensure_pkg curl   curl
ensure_pkg aria2c aria2    # load-bearing in this build -- the checkpoint comes
                           # through it, not just the optional LoRAs.

# ---------------------------------------------------------------------------
# Custom nodes
# ComfyUI-LTXVideo is now ACTIVE (it was commented in v1). You need it for the
# audio-only voice pipeline, video extension, and the Lightricks-pipeline
# workflows -- three separate things all gated behind this one package.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/Lightricks/ComfyUI-LTXVideo"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
)

install_node() {
    local url="$1" name path
    name="$(basename "$url" .git)"
    path="${NODES_DIR}/${name}"
    if [[ -d "$path" ]]; then
        echo "[node] $name present"
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
# DOWNLOAD INFRASTRUCTURE
# HF via hf_xet (Xet-backed repos break aria2's ranged requests);
# Civitai via aria2 (plain HTTPS, proper range support, 16 connections help).
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
if [[ -n "${CIVITAI_TOKEN:-}" ]]; then
    echo "[provisioning] CIVITAI_TOKEN detected (${#CIVITAI_TOKEN} chars)"
else
    echo "[provisioning] WARNING: CIVITAI_TOKEN is NOT set -- the checkpoint will be skipped"
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
# CIVITAI
#
# Takes a full URL so any civitai domain and any query parameters (fileId,
# type, format) pass through untouched. Auth goes in a header so the token
# never reaches the logfile; falls back to the query-string form only if the
# header is rejected, and says so when it does.
#
# Get a key at: Account Settings -> API Keys. Set it on the Vast instance as
# CIVITAI_TOKEN. Without it Civitai returns a small HTML or JSON error body
# that aria2 will happily save under your .safetensors name -- which then
# fails in ComfyUI with an unhelpful header error. The size check below
# catches that.
# ===========================================================================
CIVITAI_RESERVE_GB="${CIVITAI_RESERVE_GB:-60}"

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
    # Tell the user what they actually got, since destination depends on it.
    if (( sz < 8589934592 )); then
        echo "[civitai] NOTE: under 8 GB. That's LoRA-sized, not full-checkpoint-sized."
        echo "[civitai] NOTE: If this is a LoRA, move it to ${COMFY}/models/loras/ and"
        echo "[civitai] NOTE: re-enable a base checkpoint in the manifest to load it against."
    fi
}

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
CKPT="${COMFY}/models/checkpoints"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
UPSCALE="${COMFY}/models/latent_upscale_models"
GEOM="${COMFY}/models/geometry_estimation"

# --- Civitai:  dest_dir | dest_filename | full_url ---
CIVITAI_FILES=(
    "$CKPT|ltx-2.3-10eros-v1.4.safetensors|https://civitai.red/api/download/models/3109610?fileId=2989669"
    # add your other Civitai LoRAs here, same format:
    # "$LORA|some_style_lora.safetensors|https://civitai.com/api/download/models/1234567"
)

# --- Hugging Face ---
MODELS=(
    # --- Official LTX-2.3 checkpoints: REPLACED by the Civitai model above ---
    #   Uncomment dev-fp8 to keep a baseline on the box for A/B comparison
    #   (+~29 GB). Recommended at least for the first session.
    # "hf|$CKPT|ltx-2.3-22b-dev-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-dev-fp8.safetensors"
    # "hf|$CKPT|ltx-2.3-22b-distilled-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-distilled-fp8.safetensors"

    # --- Text encoder: Gemma 3 12B, full precision (24.4 GB) --------------
    # Not bundled in a transformer finetune. Templates default to the 9.45 GB
    # fp4_mixed, so you'll repoint this dropdown.
    "hf|$TE|gemma_3_12B_it.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it.safetensors"
    # "hf|$TE|gemma_3_12B_it_fp8_scaled.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors"

    # --- LoRAs ------------------------------------------------------------
    # Step-distillation LoRA. May or may not attach cleanly to a community
    # finetune -- bypass it if motion goes stiff (see header note 2).
    "hf|$LORA|ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors|Comfy-Org/ltx-2.3|split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
    # Applied to the TEXT ENCODER, not the video model.
    "hf|$LORA|gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors|Comfy-Org/ltx-2|split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
    # Structural control from a driving video (depth / pose / canny).
    # Officially pairs with the DISTILLED checkpoint -- untested against a finetune.
    "hf|$LORA|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

    # --- Spatial upscaler (1.1 -- the 1.0 build had a splash-logo artifact) -
    "hf|$UPSCALE|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    # --- Geometry estimation for the IC-LoRA control branch ---------------
    "hf|$GEOM|moge_2_vitl_normal_fp16.safetensors|Comfy-Org/MoGe|geometry_estimation/moge_2_vitl_normal_fp16.safetensors"
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

    # Civitai sizes aren't reliably queryable pre-auth -- reserve a flat amount
    # per configured file that isn't already on disk.
    local cdir cname curl_ pending=0
    for entry in "${CIVITAI_FILES[@]}"; do
        IFS='|' read -r cdir cname curl_ <<< "$entry"
        [[ -f "${cdir}/${cname}" && ! -f "${cdir}/${cname}.aria2" ]] || (( pending++ ))
    done
    (( pending > 0 )) && need=$(( need + pending * CIVITAI_RESERVE_GB * 1024*1024*1024 ))

    mkdir -p "$CKPT"
    local avail; avail="$(df -PB1 "$CKPT" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need} (incl. ${pending} civitai file(s)"
    echo "[provisioning] reserved at ${CIVITAI_RESERVE_GB} GB each);  free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Resize the disk and reboot, or lower CIVITAI_RESERVE_GB if you"
        echo "[provisioning] !!! know the checkpoint is smaller than ${CIVITAI_RESERVE_GB} GB."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
echo "=================== CIVITAI (checkpoint) ==================="
if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
    echo "[civitai] CIVITAI_TOKEN unset -- skipping ALL civitai downloads."
    echo "[civitai] Set it in the Vast instance environment and reboot."
else
    for entry in "${CIVITAI_FILES[@]}"; do
        IFS='|' read -r cdir cname curl_ <<< "$entry"
        dl_civitai "$cdir" "$cname" "$curl_"
    done
fi

echo "=================== HUGGING FACE (support models) ==================="
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
for p in "${TE}/gemma_3_12B_it.safetensors" \
         "${LORA}/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" \
         "${LORA}/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
         "${LORA}/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
         "${UPSCALE}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
         "${GEOM}/moge_2_vitl_normal_fp16.safetensors"; do
    [[ -f "$p" ]] && echo "[layout] OK      ${p#${COMFY}/}" || echo "[layout] MISSING ${p#${COMFY}/}"
done

echo "[workflow] Native templates: Workflow -> Browse Templates -> Video"
echo "[workflow] Point CheckpointLoaderSimple at ltx-2.3-10eros-v1.4.safetensors"
echo "[workflow] Reminder: w/h divisible by 32 (64 for two-stage); frames = 8n+1"

echo "=================== PROVISIONING COMPLETE ==================="
