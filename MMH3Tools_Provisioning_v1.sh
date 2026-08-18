#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning for vast.ai  ---  MMH3Tools_Provisioning_v1
#
# Target workflow: ckinpdx_MMH3_I2V_2K.json
#   (ckinpdx/ComfyUI-MMH3Tools -- three-stage MiniMax H3 I2V to 2K)
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Recommended: set HF_TOKEN=<your token> in the instance env.
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# DISK: the four H3 files total ~54 GiB (21 + 27.1 + 5.2 + 0.6). Xet's on-disk
# chunk cache needs transient headroom on top. Rent >=150 GiB.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v5
#
#   1. UPDATES ACTUALLY HAPPEN. This is the fix for "I still have to update by
#      hand". Two separate bugs in v5:
#        (a) install_node()'s `git pull --ff-only` was COMMENTED OUT, so any
#            node that already existed on the persistent volume was reported
#            "present" and never touched again. Every reboot after the first
#            was a no-op for node code.
#        (b) v5 never updated ComfyUI core at all -- it only ever installed
#            custom nodes and models.
#      v1 adds git_sync(), which handles the cases a bare `git pull` chokes on:
#      detached HEAD (ai-dock images often pin core to a commit, and `pull`
#      silently does nothing there), shallow clones (`--unshallow` first, or
#      pins are unreachable), dirty worktrees (warn and skip rather than
#      destroy your edits -- set FORCE_CLEAN=1 to hard-reset), and repos whose
#      default branch is master vs main.
#
#   2. Dependency reinstalls are now CONDITIONAL. v5 ran pip against every
#      node's requirements.txt on every boot, which cost minutes and could
#      re-resolve a working env. v1 hashes requirements.txt and only reinstalls
#      when it changed, the node was freshly cloned, or FORCE_DEPS=1.
#
#   3. ComfyUI core requirements are reinstalled when core moves. This is the
#      usual cause of "I pulled but the UI is the old one" -- the frontend is a
#      pip package (comfyui-frontend-package) pinned in core's requirements.txt,
#      so a git pull without a pip install leaves you on the old frontend.
#
#   4. Version gate: MMH3Tools requires ComfyUI >= 0.30.0 (native H3 support).
#      The script reports the version after updating and warns loudly if short.
#
#   5. Optional service restart. If core or a node actually changed AND the
#      comfyui service is already running, the script restarts it so the new
#      code is live without a manual bounce. Set RESTART_COMFY_ON_UPDATE=0 to
#      disable.
#
#   6. Model manifest replaced: Wan 2.2 -> the four MiniMax H3 files this
#      workflow's loaders actually name. Node list extended to the five packs
#      the workflow needs beyond your v5 set.
#
#   Retained unchanged from v5: hf_xet Xet-native transport, HF caches pinned
#   under $WORKSPACE, size-verified skip logic, disk pre-flight.
#
# -----------------------------------------------------------------------------
# ENV TOGGLES
#   UPDATE_ON_BOOT=1            pull core + nodes each boot (default 1)
#   FORCE_DEPS=0                reinstall all pip deps regardless of change
#   FORCE_CLEAN=0               hard-reset dirty repos instead of skipping them
#   RESTART_COMFY_ON_UPDATE=1   bounce the comfyui service if code changed
#   SKIP_MODELS=0               nodes only, no model downloads
# =============================================================================

set -o pipefail

UPDATE_ON_BOOT="${UPDATE_ON_BOOT:-1}"
FORCE_DEPS="${FORCE_DEPS:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-1}"
SKIP_MODELS="${SKIP_MODELS:-0}"

COMFY_MIN_VERSION="0.30.0"

# ---------------------------------------------------------------------------
# Persistent log  (tail -f /workspace/provisioning.log)
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## MMH3Tools provisioning v1: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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

ensure_pkg git  git
ensure_pkg curl curl

# ai-dock runs provisioning as root but the persistent volume may be owned by
# another uid; without this every git call fails "dubious ownership" and the
# update phase silently degrades to no-ops.
git config --global --add safe.directory '*' 2>/dev/null || true

# ===========================================================================
# GIT SYNC  --  the core of the update fix
# ===========================================================================
# git_sync <path> [pin]
#   Returns 0 if HEAD moved (i.e. something changed), 1 otherwise.
#   With a pin (commit sha or tag) it checks that out and stays detached.
#   Without one it fast-forwards the tracking branch.
git_sync() {
    local path="$1" pin="${2:-}"
    [[ -d "${path}/.git" ]] || { echo "[git] $(basename "$path"): not a git repo, skipping"; return 1; }

    local before after name
    name="$(basename "$path")"
    before="$(git -C "$path" rev-parse HEAD 2>/dev/null)"

    # Dirty worktree: a pull would fail or clobber. Decide explicitly.
    if [[ -n "$(git -C "$path" status --porcelain 2>/dev/null)" ]]; then
        if [[ "$FORCE_CLEAN" == "1" ]]; then
            echo "[git] $name: dirty worktree -> hard reset (FORCE_CLEAN=1)"
            git -C "$path" reset --hard >/dev/null 2>&1
            git -C "$path" clean -fd     >/dev/null 2>&1
        else
            echo "[git] $name: WORKTREE DIRTY -> skipping update (set FORCE_CLEAN=1 to override)"
            return 1
        fi
    fi

    # Shallow clones can't reach arbitrary commits, and ai-dock clones core
    # shallow. Deepen once; it's cheap relative to the model downloads.
    if [[ "$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null)" == "true" ]]; then
        echo "[git] $name: shallow -> unshallowing"
        git -C "$path" fetch --unshallow --tags origin >/dev/null 2>&1 \
            || git -C "$path" fetch --depth=100 --tags origin >/dev/null 2>&1
    fi

    git -C "$path" fetch --prune --tags origin >/dev/null 2>&1 \
        || { echo "[git] $name: fetch FAILED (offline? rate-limited?)"; return 1; }

    if [[ -n "$pin" ]]; then
        if git -C "$path" checkout --quiet "$pin" 2>/dev/null; then
            echo "[git] $name: pinned at ${pin:0:12}"
        else
            echo "[git] $name: PIN NOT FOUND ($pin) -- leaving HEAD as is"
        fi
    else
        # A detached HEAD is the trap: `git pull` there exits 0 and does nothing.
        # Get back onto the remote's default branch first.
        if ! git -C "$path" symbolic-ref -q HEAD >/dev/null 2>&1; then
            local defb="" cand
            # Some clones leave origin/HEAD unset; repair it before asking.
            git -C "$path" remote set-head origin -a >/dev/null 2>&1 || true
            defb="$(git -C "$path" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
            # `git remote show` prints the literal "(unknown)" when it can't tell;
            # checking that out fails and used to strand the repo detached.
            if [[ -z "$defb" || "$defb" == "(unknown)" ]]; then
                defb="$(git -C "$path" remote show origin 2>/dev/null | awk '/HEAD branch/{print $NF}')"
            fi
            if [[ -z "$defb" || "$defb" == "(unknown)" ]]; then
                for cand in main master; do
                    git -C "$path" show-ref --verify --quiet "refs/remotes/origin/${cand}" \
                        && { defb="$cand"; break; }
                done
            fi
            if [[ -z "$defb" || "$defb" == "(unknown)" ]]; then
                defb="$(git -C "$path" for-each-ref --format='%(refname:short)' refs/remotes/origin 2>/dev/null \
                        | grep -v '/HEAD$' | head -1 | sed 's|^origin/||')"
            fi
            if [[ -n "$defb" ]]; then
                echo "[git] $name: detached HEAD -> checking out ${defb}"
                git -C "$path" checkout --quiet -B "$defb" "origin/${defb}" 2>/dev/null \
                    || echo "[git] $name: could not check out ${defb}"
            else
                echo "[git] $name: detached and no remote branch found -- left alone"
            fi
        fi

        # merge --ff-only '@{u}' needs an upstream; a -B checkout may not have one.
        local br up
        br="$(git -C "$path" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        up="$(git -C "$path" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"
        if [[ -z "$up" && -n "$br" && "$br" != "HEAD" ]]; then
            git -C "$path" branch --set-upstream-to="origin/${br}" "$br" >/dev/null 2>&1 \
                && up="origin/${br}"
        fi
        if [[ -n "$up" ]]; then
            git -C "$path" merge --ff-only "$up" >/dev/null 2>&1 \
                || echo "[git] $name: fast-forward not possible (diverged?) -- left alone"
        else
            echo "[git] $name: no upstream tracking ref -- cannot fast-forward"
        fi
    fi

    after="$(git -C "$path" rev-parse HEAD 2>/dev/null)"
    if [[ "$before" != "$after" ]]; then
        echo "[git] $name: ${before:0:8} -> ${after:0:8}  (updated)"
        return 0
    fi
    echo "[git] $name: already current (${after:0:8})"
    return 1
}

# reqs_changed <path> <reqfile>  -- 0 if the file is new or its hash moved.
reqs_changed() {
    local path="$1" req="$2" marker sum old
    [[ -f "$req" ]] || return 1
    marker="${path}/.prov_reqs.sha256"
    sum="$(sha256sum "$req" | awk '{print $1}')"
    old="$(cat "$marker" 2>/dev/null)"
    if [[ "$sum" != "$old" ]]; then
        printf '%s' "$sum" > "$marker"
        return 0
    fi
    return 1
}

CHANGED_ANY=0

# ===========================================================================
# COMFYUI CORE
# ===========================================================================
echo "=================== COMFYUI CORE ==================="
if [[ "$UPDATE_ON_BOOT" == "1" && -d "${COMFY}/.git" ]]; then
    if git_sync "$COMFY" "${COMFY_PIN:-}"; then
        CHANGED_ANY=1
        # Core moved -> its requirements may pin a new comfyui-frontend-package.
        # Skipping this is why a pulled ComfyUI can still serve the old UI.
        if [[ -f "${COMFY}/requirements.txt" ]]; then
            echo "[core] reinstalling core requirements after update"
            pip_install --no-cache-dir -r "${COMFY}/requirements.txt" \
                || echo "[core] requirements.txt FAILED"
        fi
    fi
elif [[ "$UPDATE_ON_BOOT" != "1" ]]; then
    echo "[core] UPDATE_ON_BOOT=0 -> skipping core update"
else
    echo "[core] ${COMFY} is not a git checkout -> cannot self-update"
fi

comfy_version() {
    local vf="${COMFY}/comfyui_version.py"
    [[ -f "$vf" ]] && sed -n 's/^__version__ *= *"\(.*\)"/\1/p' "$vf" | head -1
}
CV="$(comfy_version)"
if [[ -n "$CV" ]]; then
    echo "[core] ComfyUI version: ${CV}"
    if [[ "$(printf '%s\n%s\n' "$COMFY_MIN_VERSION" "$CV" | sort -V | head -1)" != "$COMFY_MIN_VERSION" ]]; then
        echo "[core] !!! WARNING: ComfyUI ${CV} < ${COMFY_MIN_VERSION}."
        echo "[core] !!! MMH3Tools and the native MiniMax H3 nodes will NOT load."
    fi
else
    echo "[core] could not read comfyui_version.py -- version gate not checked"
fi

# ===========================================================================
# CUSTOM NODES
# ===========================================================================
# Format: "url|pin"  -- leave the pin empty to track the default branch.
# The commit hashes in the comments are the versions ckinpdx's workflow JSON
# records as the ones it was authored against. Fill them into the pin field if
# you want reproducibility instead of updates; you cannot have both.
NODES=(
    # --- required by ckinpdx_MMH3_I2V_2K.json ---
    "https://github.com/ckinpdx/ComfyUI-MMH3Tools|"       # workflow built at d43152979fdbd737f51f78a7819a1d5dfe974e63
    "https://github.com/ClownsharkBatwing/RES4LYF|"       # Clown samplers, Sigmas Resample/Rescale/Split, Linear Quadratic Advanced
    "https://github.com/kijai/ComfyUI-KJNodes|"           # VAELoaderKJ, SageAttention patch, MiniMax low-VRAM attn + chunked FF
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite|"
    "https://github.com/rgthree/rgthree-comfy|"           # Label nodes + RES4LYF's nested sampler menus

    # --- optional, workflow will load without them ---
    "https://github.com/kijai/ComfyUI-SolAttn_triton|"    # SolAttnPatch. Triton 3.3+; TMA path needs SM90+. Bypass the node if the kernel falls back.
    "https://github.com/ckinpdx/ComfyUI-LlamaOmni|"       # LlamaGenerate/Options/Connectivity -- SEE THE NOTE AT THE BOTTOM

    # --- your standing pipeline, not used by this workflow ---
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto|"
    # "https://github.com/yolain/ComfyUI-Easy-Use|"       # dropped: no Easy-Use nodes in this graph
)

install_node() {
    local entry="$1" url pin name path fresh=0 changed=0
    IFS='|' read -r url pin <<< "$entry"
    name="$(basename "$url" .git)"
    path="${NODES_DIR}/${name}"

    if [[ -d "$path" ]]; then
        if [[ "$UPDATE_ON_BOOT" == "1" ]]; then
            git_sync "$path" "$pin" && { changed=1; CHANGED_ANY=1; }
        else
            echo "[node] $name present (updates disabled)"
        fi
    else
        echo "[node] cloning $name"
        git clone --recursive "$url" "$path" || { echo "[node] CLONE FAILED: $name"; return 0; }
        [[ -n "$pin" ]] && git -C "$path" checkout --quiet "$pin" 2>/dev/null
        fresh=1; changed=1; CHANGED_ANY=1
    fi

    local req="${path}/requirements.txt"
    if [[ -f "$req" ]]; then
        if (( fresh )) || [[ "$FORCE_DEPS" == "1" ]] || reqs_changed "$path" "$req"; then
            echo "[node] $name: installing requirements"
            pip_install --no-cache-dir -r "$req" || echo "[node] requirements.txt FAILED: $name"
        else
            echo "[node] $name: requirements unchanged, skipping pip"
        fi
    fi

    if [[ -f "${path}/install.py" ]] && { (( fresh )) || (( changed )); }; then
        ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
    fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
for n in "${NODES[@]}"; do install_node "$n"; done

# RIFE-TensorRT-Auto's deps pull cuda-python 13.x, which swaps cuda-bindings to
# 13.x and breaks the image's cu12 torch (pins cuda-bindings==12.9.x). Only
# needed while that node is in the list; remove if torch moves to a CUDA-13 build.
if [[ -d "${NODES_DIR}/ComfyUI-RIFE-TensorRT-Auto" ]]; then
    echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
    pip_install "cuda-python<13"
fi

# Drop the pack's example workflows where the UI can see them.
WF_SRC="${NODES_DIR}/ComfyUI-MMH3Tools/workflows"
WF_DST="${COMFY}/user/default/workflows/MMH3Tools"
if [[ -d "$WF_SRC" ]]; then
    mkdir -p "$WF_DST" && cp -f "$WF_SRC"/*.json "$WF_DST"/ 2>/dev/null \
        && echo "[provisioning] example workflows -> user/default/workflows/MMH3Tools"
fi

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE  (unchanged from v5)
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
            echo "[model] WARNING: $name size after download ${have} != expected ${want} (kept for resume next boot)"
        else
            echo "[model] $name OK (${have} bytes)"
        fi
    else
        echo "[model] DOWNLOAD FAILED: $name (hf_xet; will retry next boot)"
    fi
}

dl_aria2() {
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
# Model manifest -- exactly the files this workflow's loaders name.
#   HF  entries:  hf  | dest_dir | dest_filename | repo_id | repo_path
#   URL entries:  url | dest_dir | dest_filename | https://...
#
#   UNETLoader   -> minimax_h3_fl2va_pruned_int8_convrot.safetensors
#   CLIPLoader   -> qwen3vl_32b_minimax_h3_int8_convrot.safetensors  (type: minimax)
#   VAELoaderKJ  -> minimax_h3_video_vae_fp16.safetensors
#   VAELoaderKJ  -> minimax_h3_audio_vae_fp32.safetensors
#
#   FL2VA is the first/last-frame variant -- correct for this I2V graph. If you
#   move to omni-references, swap in minimax_h3_ref2va_pruned_int8_convrot and
#   repoint the UNETLoader.
#
#   No LoRA: this workflow gets its step reduction from RES4LYF sigma shaping
#   (Sigmas Resample / Rescale / Split + Linear Quadratic Advanced), not from a
#   lightning/turbo LoRA. Nothing to fetch into models/loras.
# ---------------------------------------------------------------------------
DM="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
VAE="${COMFY}/models/vae"
TE="${COMFY}/models/text_encoders"

MODELS=(
    # --- Diffusion model (~21 GiB) ---
    "hf|$DM|minimax_h3_fl2va_pruned_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

    # --- Text encoder: Qwen3-VL-32B, int8 convrot (~27 GiB) ---
    "hf|$TE|qwen3vl_32b_minimax_h3_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

    # --- VAEs: video (~5.2 GiB) + audio (~605 MiB). The audio VAE is not
    #     optional here -- without it VAEDecodeAudio produces a silent clip.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"

    # --- Optional: bf16 encoder if you'd rather not run the quantised one.
    #     ~66 GiB, so mind the pre-flight.
    # "hf|$TE|qwen3vl_32b_minimax_h3_bf16.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
)

if printf '%s\n' "${MODELS[@]}" | grep -q '^url|'; then
    ensure_pkg aria2c aria2
fi

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

    mkdir -p "$DM"
    local avail; avail="$(df -PB1 "$DM" | awk 'NR==2{print $4}')"
    local margin=$(( 5 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] models still to fetch: ${h_need};  free on models FS: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 5GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Skipping model downloads. Resize the instance disk and reboot."
        return 1
    fi
    if (( avail - need < 15 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~15GiB). Xet's chunk"
        echo "[provisioning] NOTE: cache uses transient space; consider a bigger disk."
    fi
    return 0
}

echo "=================== MODELS ==================="
if [[ "$SKIP_MODELS" == "1" ]]; then
    echo "[provisioning] SKIP_MODELS=1 -> model phase skipped"
elif preflight_disk; then
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
# RESTART IF CODE CHANGED
# ===========================================================================
# ai-dock brings services up around provisioning rather than strictly after it,
# so a core/node update applied here is on disk but not in the running process.
# That is the other half of "I still have to update manually" -- the pull worked,
# the process was just already loaded. Bounce it only if something moved.
echo "=================== FINALISE ==================="
if (( CHANGED_ANY )) && [[ "$RESTART_COMFY_ON_UPDATE" == "1" ]] && command -v supervisorctl >/dev/null 2>&1; then
    if supervisorctl status comfyui 2>/dev/null | grep -q RUNNING; then
        echo "[provisioning] code changed and comfyui is running -> restarting service"
        supervisorctl restart comfyui || echo "[provisioning] restart FAILED -- bounce it manually"
    else
        echo "[provisioning] code changed; comfyui not running yet -> it will start on new code"
    fi
elif (( CHANGED_ANY )); then
    echo "[provisioning] code changed. Restart ComfyUI to load it (RESTART_COMFY_ON_UPDATE=${RESTART_COMFY_ON_UPDATE})"
else
    echo "[provisioning] nothing changed this boot"
fi

cat <<'NOTE'

--------------------------------------------------------------------------
NOT PROVISIONED BY THIS SCRIPT: the LlamaOmni prompt server
--------------------------------------------------------------------------
The workflow's LlamaConnectivity node points at http://127.0.0.1:8000 and asks
for model "qwen3-omni-30b". That is an OpenAI-compatible inference server you
run separately (vLLM/sglang + a Qwen3-Omni-30B checkout) -- another large
download and a second GPU resident alongside a 21 GiB DiT and a 27 GiB encoder.

In this graph the omni model exists to transcribe the uploaded song's lyrics so
the character lip-syncs. If you are not doing lip-sync, or you want to write the
H3 prompt yourself, bypass LlamaGenerate/LlamaOptions/LlamaConnectivity and feed
the PrimitiveStringMultiline straight into CLIPTextEncode -- the canvas Note says
the prompt nodes are meant to be swappable.

--------------------------------------------------------------------------
NOTE

echo "=================== PROVISIONING COMPLETE ==================="
