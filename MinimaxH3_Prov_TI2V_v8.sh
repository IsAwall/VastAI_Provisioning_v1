#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning for vast.ai -- MiniMax H3 (Hailuo 3.0) I2V -- v8
#
# USE
#   Host this file as raw text and set PROVISIONING_SCRIPT=<raw-url> on the
#   instance. It runs on every boot. ~131 GB of weights: 180 GB volume minimum.
#
# ENV
#   HF_TOKEN=<token>            recommended (never put the token in this file)
#   CIVITAI_TOKEN=<token>       only needed if CIVITAI_FILES has entries
#   COMFY_UPDATE=1              update ComfyUI core                 (default 1)
#   COMFY_TRACK=release|master  what "latest" means for core      (default release)
#   COMFY_PIN=<tag|sha>         hold core at this ref (overrides COMFY_TRACK)
#   NODE_UPDATE=1               update custom nodes                (default 1)
#   GIT_FORCE_RESET=1           discard local edits/commits that block an update
#   RESTART_COMFY_ON_UPDATE=1   restart comfyui when code moved    (default 1)
#   FORCE_DEPS=1                reinstall pip requirements even if unchanged
#   WANT_SEEDVR2=1              fetch SeedVR2 restore weights (+15 GB)
#   WANT_TURBO=1                add Larryvrh's Turbo node pack (NOT needed for
#                               the lightx2v LoRA, which loads via LoraLoader)
#   PURGE_LEGACY_SEEDVR2=1      delete the old third-party SeedVR2 pack + weights
#   FIX_FLASH_ATTN=0            skip the half-installed flash-attn guard
#   INSTALL_SAGE=1              install SageAttention (drafts only)
#
# v8 CHANGES
#   - Core updates to the LATEST RELEASE TAG by default. v6/v7 left an
#     image-pinned (detached) checkout alone, so on vast.ai images core never
#     moved regardless of COMFY_UPDATE. COMFY_TRACK=master tracks the branch.
#   - Custom nodes on a detached HEAD are moved onto their default branch and
#     fast-forwarded. Append @<sha|tag> to a NODES entry to hold one.
#   - git fetch now passes --force for tags. A moved upstream tag made fetch
#     fail ("would clobber existing tag") and the update aborted quietly.
#   - Dirty working trees are detected before checkout, not after a failed merge.
#   - Restart after a code change is ON by default (skipped if crash-looping).
#   - Kijai's lightx2v 4-step Turbo LoRA is fetched unconditionally. The v7
#     entry was missing its repo id and could never download.
#   - Civitai manifest emptied; the download machinery is unchanged.
#   - Comments cut down to what you need while operating.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (MiniMax H3 I2V v8): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

COMFY_UPDATE="${COMFY_UPDATE:-1}"
COMFY_TRACK="${COMFY_TRACK:-release}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_TURBO="${WANT_TURBO:-0}"
WANT_SEEDVR2="${WANT_SEEDVR2:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-1}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} track=${COMFY_TRACK} pin=${COMFY_PIN:-<none>} node_update=${NODE_UPDATE}"
echo "[provisioning] turbo_nodes=${WANT_TURBO} seedvr2=${WANT_SEEDVR2} force_deps=${FORCE_DEPS} restart=${RESTART_COMFY_ON_UPDATE} force_reset=${GIT_FORCE_RESET:-0}"

[[ -f /opt/ai-dock/etc/environment.sh ]] && source /opt/ai-dock/etc/environment.sh
[[ -f /opt/ai-dock/bin/venv-set.sh    ]] && source /opt/ai-dock/bin/venv-set.sh comfyui

# ---------------------------------------------------------------------------
# Python / pip helpers
# ---------------------------------------------------------------------------
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
    echo "[pip] retrying with --break-system-packages"
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

# Provisioning runs as root; the volume is often owned by another uid. Without
# this every git call fails with "dubious ownership" and updates silently no-op.
git config --global --add safe.directory '*' 2>/dev/null \
    && echo "[git] safe.directory configured" \
    || echo "[git] WARNING: could not set safe.directory -- git ops may fail"

# ---------------------------------------------------------------------------
# flash-attn guard
# A half-installed flash_attn makes xformers take the flash path and die, which
# kills every node that imports diffusers. Nothing here needs flash-attn, so
# the fix is removal. Must run before torch pins are written.
# ---------------------------------------------------------------------------
fix_flash_attn() {
    local rc out pkgdir detail

    out="$("$PY" - <<'PYEOF'
import importlib, os, sys
try:
    import flash_attn
except Exception:
    print("ABSENT\t\t"); sys.exit(0)

paths = list(getattr(flash_attn, "__path__", []) or [])
d = paths[0] if paths else os.path.dirname(getattr(flash_attn, "__file__", "") or "")
try:
    importlib.import_module("flash_attn.flash_attn_interface")
except Exception as e:
    print("BROKEN\t%s\t%s" % (d, e)); sys.exit(7)
print("OK\t%s\t%s" % (d, getattr(flash_attn, "__version__", "?"))); sys.exit(0)
PYEOF
)"
    rc=$?
    pkgdir="$(printf '%s' "$out" | awk -F'\t' 'NR==1{print $2}')"
    detail="$(printf '%s' "$out" | awk -F'\t' 'NR==1{print $3}')"

    case "$(printf '%s' "$out" | awk -F'\t' 'NR==1{print $1}')" in
        ABSENT) echo "[flash] not installed (fine)"; return 0 ;;
        OK)     echo "[flash] healthy (${detail:-?})"; return 0 ;;
    esac
    (( rc == 7 )) || { echo "[flash] unexpected probe state, leaving alone"; return 0; }

    echo "[flash] !!! half-installed flash_attn: ${detail}"
    echo "[flash] !!! removing it (breaks every diffusers-based node otherwise)"
    "$PY" -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
    if [[ -n "$pkgdir" && "$pkgdir" == */flash_attn ]]; then
        rm -rf "$pkgdir"
        rm -rf "${pkgdir%/*}"/flash_attn-*.dist-info "${pkgdir%/*}"/flash_attn-*.egg-info
    fi
    if "$PY" -c "import xformers.ops" 2>/dev/null; then
        echo "[flash] xformers imports cleanly again"
    else
        echo "[flash] xformers still broken -> removing it too"
        "$PY" -m pip uninstall -y xformers >/dev/null 2>&1 || true
    fi
    "$PY" - <<'PYEOF'
import importlib.util as u
if u.find_spec("diffusers") is None:
    print("[flash] diffusers not installed -- nothing further to verify")
else:
    try:
        import diffusers.models.embeddings  # noqa: F401
        print("[flash] diffusers imports cleanly -- fixed")
    except Exception as e:
        print("[flash] WARNING: diffusers still fails: %s" % e)
PYEOF
}

# ---------------------------------------------------------------------------
# Torch pins
# Snapshot the installed CUDA stack into a constraints file so no requirements
# install can swap the pinned torch wheel for a generic one.
# ---------------------------------------------------------------------------
CONSTRAINTS="/tmp/torch-pins.txt"

write_torch_pins() {
    : > "$CONSTRAINTS"
    "$PY" - >> "$CONSTRAINTS" <<'PYEOF' || true
import importlib.metadata as md
import importlib.util as u
for p in ("torch", "torchvision", "torchaudio", "torchsde",
          "triton", "pytorch-triton", "xformers",
          "sageattention", "flash-attn"):
    mod = p.replace("-", "_")
    try:
        if u.find_spec(mod) is None:
            continue
    except Exception:
        continue
    try:
        print("%s==%s" % (p, md.version(p)))
    except Exception:
        pass
PYEOF
    if [[ -s "$CONSTRAINTS" ]]; then
        echo "[pins] torch stack pinned:"
        sed 's/^/[pins]   /' "$CONSTRAINTS"
    else
        echo "[pins] nothing to pin (torch not importable yet?)"
    fi
}

# pip_reqs <requirements-file> <label>
pip_reqs() {
    local req="$1" label="$2"
    [[ -f "$req" ]] || return 0
    if [[ -s "$CONSTRAINTS" ]]; then
        pip_install --no-cache-dir -c "$CONSTRAINTS" -r "$req" && return 0
        echo "[pip] ${label}: constrained install failed (something wants to move torch)"
        echo "[pip] ${label}: retrying unconstrained; CUDA is verified afterwards"
    fi
    pip_install --no-cache-dir -r "$req" || { echo "[pip] ${label}: requirements FAILED"; return 1; }
}

# reqs_changed <repo_path> <requirements_file> -> 0 when new or content changed
reqs_changed() {
    local path="$1" req="$2" marker sum old
    [[ -f "$req" ]] || return 1
    marker="${path}/.prov_reqs.sha256"
    sum="$(sha256sum "$req" | awk '{print $1}')"
    old="$(cat "$marker" 2>/dev/null)"
    [[ "$sum" == "$old" ]] && return 1
    printf '%s' "$sum" > "$marker"
    return 0
}

CHANGED_ANY=0

verify_torch() {
    "$PY" - <<'PYEOF' || true
try:
    import torch
    ok = torch.cuda.is_available()
    print("[torch] %s | cuda %s | device: %s" % (
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if ok else "NONE"))
    if ok:
        free, total = torch.cuda.mem_get_info(0)
        print("[torch] vram: %.1f GB total" % (total / 1024**3))
    else:
        print("[torch] !!! CUDA UNAVAILABLE -- torch wheel was probably replaced.")
        print("[torch] !!! Reinstall the pinned build for your CUDA line.")
except Exception as e:
    print("[torch] not importable: %s" % e)
PYEOF
}

echo "=================== FLASH-ATTN GUARD ==================="
if [[ "${FIX_FLASH_ATTN:-1}" == "1" ]]; then fix_flash_attn; else echo "[flash] skipped (FIX_FLASH_ATTN=0)"; fi

echo "=================== TORCH PINS ==================="
write_torch_pins

# bf16 DiT (61.7 GB) + bf16 encoder (48 GB) only works if the evicted encoder
# can sit in system RAM. 128 GB minimum.
echo "=================== HOST MEMORY ==================="
RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$RAM_GB" ]]; then
    echo "[mem] system RAM: ${RAM_GB} GB"
    if (( RAM_GB < 128 )); then
        echo "[mem] !!! under 128 GB: the bf16/bf16 pairing will re-read the text encoder"
        echo "[mem] !!! from disk every run. Use int8_convrot for the encoder, or more RAM."
    fi
fi

# ---------------------------------------------------------------------------
# git_sync <repo_path> <label> <mode> [pin]
#   mode: release = latest semver tag   branch = default branch, fast-forward
#   pin overrides mode.
# Returns: 0 = HEAD moved   1 = could not update   2 = already current
# ---------------------------------------------------------------------------
git_default_branch() {
    local path="$1" def
    def="$(git -C "$path" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
    if [[ -z "$def" ]]; then
        git -C "$path" remote set-head origin --auto >/dev/null 2>&1
        def="$(git -C "$path" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
    fi
    if [[ -z "$def" ]]; then
        local c
        for c in master main; do
            git -C "$path" show-ref --verify --quiet "refs/remotes/origin/${c}" && { def="$c"; break; }
        done
    fi
    printf '%s' "$def"
}

git_sync() {
    local path="$1" label="$2" mode="$3" pin="${4:-}"
    local before after target branch def
    [[ -d "${path}/.git" ]] || { echo "[git] ${label}: not a git checkout, skipping"; return 1; }

    before="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"

    if [[ "$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null)" == "true" ]]; then
        echo "[git] ${label}: shallow clone -> unshallowing"
        git -C "$path" fetch --unshallow --tags --force --prune 2>/dev/null \
            || git -C "$path" fetch --depth=2147483647 --tags --force --prune 2>/dev/null \
            || echo "[git] ${label}: unshallow failed, continuing"
    fi

    # --force: a moved upstream tag otherwise fails the whole fetch.
    if ! git -C "$path" fetch --all --tags --force --prune; then
        echo "[git] ${label}: fetch FAILED (remote: $(git -C "$path" remote get-url origin 2>/dev/null || echo '?'))"
        return 1
    fi

    # A dirty tree blocks both checkout and merge.
    if [[ -n "$(git -C "$path" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
        if [[ "${GIT_FORCE_RESET:-0}" == "1" ]]; then
            echo "[git] ${label}: dirty tree -> discarding local edits (GIT_FORCE_RESET=1)"
            git -C "$path" reset --hard HEAD >/dev/null 2>&1
        else
            echo "[git] ${label}: dirty tree (local edits) -- NOT updating. GIT_FORCE_RESET=1 to discard."
            return 1
        fi
    fi

    # --- resolve the target ---
    if [[ -n "$pin" ]]; then
        target="$pin"; mode="pin"
    elif [[ "$mode" == "release" ]]; then
        target="$(git -C "$path" tag -l | grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)"
        if [[ -z "$target" ]]; then
            echo "[git] ${label}: no release tags -> tracking default branch instead"
            mode="branch"
        fi
    fi

    # --- pinned / release: detached checkout of a ref ---
    if [[ "$mode" == "pin" || "$mode" == "release" ]]; then
        if ! git -C "$path" checkout --quiet --detach "$target" 2>/dev/null; then
            echo "[git] ${label}: ref not found: ${target} -- leaving HEAD as is"
            return 1
        fi
        after="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"
        if [[ "$before" == "$after" ]]; then
            echo "[git] ${label}: already at ${target} (${after})"
            return 2
        fi
        echo "[git] ${label}: ${before} -> ${after} (${mode}: ${target})"
        return 0
    fi

    # --- branch: get onto the default branch and fast-forward ---
    branch="$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -z "$branch" ]] || ! git -C "$path" show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
        def="$(git_default_branch "$path")"
        [[ -n "$def" ]] || { echo "[git] ${label}: cannot resolve default branch"; return 1; }
        echo "[git] ${label}: ${branch:-detached HEAD} -> checking out ${def}"
        git -C "$path" checkout -B "$def" "origin/${def}" >/dev/null 2>&1 \
            || { echo "[git] ${label}: checkout ${def} FAILED"; return 1; }
        branch="$def"
    fi

    if ! git -C "$path" merge --ff-only "origin/${branch}" >/dev/null 2>&1; then
        if [[ "${GIT_FORCE_RESET:-0}" == "1" ]]; then
            echo "[git] ${label}: fast-forward blocked -> hard reset to origin/${branch}"
            git -C "$path" reset --hard "origin/${branch}" >/dev/null 2>&1 || return 1
        else
            echo "[git] ${label}: fast-forward blocked (local commits). GIT_FORCE_RESET=1 to discard."
            return 1
        fi
    fi

    after="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"
    if [[ "$before" == "$after" ]]; then
        echo "[git] ${label}: already current (${branch} @ ${after})"
        return 2
    fi
    echo "[git] ${label}: ${before} -> ${after} (${branch})"
    return 0
}

# ---------------------------------------------------------------------------
# ComfyUI core update. Native H3 nodes need >= 0.30.0.
# ---------------------------------------------------------------------------
comfy_version() {
    local v=""
    [[ -f "${COMFY}/comfyui_version.py" ]] && \
        v="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${COMFY}/comfyui_version.py" | head -1)"
    [[ -z "$v" && -d "${COMFY}/.git" ]] && \
        v="$(git -C "$COMFY" describe --tags --abbrev=0 2>/dev/null | tr -d 'v')"
    printf '%s' "$v"
}

# Force a pip package to the version ComfyUI's requirements.txt pins.
# Frontend/backend skew produces phantom link errors and stale templates.
sync_pinned_pkg() {
    local pkg="$1" spec have want
    spec="$(grep -iE "^[[:space:]]*${pkg}[[:space:]]*==" "${COMFY}/requirements.txt" 2>/dev/null | head -1 | tr -d ' \r')"
    [[ -n "$spec" ]] || { echo "[deps] ${pkg}: no pin in requirements.txt"; return 0; }
    want="${spec##*==}"
    have="$("$PY" -c "import importlib.metadata as m; print(m.version('${pkg}'))" 2>/dev/null || echo "")"
    [[ "$have" == "$want" ]] && { echo "[deps] ${pkg}: ${have} OK"; return 0; }
    echo "[deps] ${pkg}: ${have:-<absent>} -> ${want}"
    pip_install --no-cache-dir "$spec" || echo "[deps] WARNING: failed to sync ${spec}"
}

echo "=================== COMFYUI UPDATE ==================="
if [[ -d "${COMFY}/.git" ]]; then
    _head="$(git -C "$COMFY" rev-parse --short HEAD 2>/dev/null)"
    _br="$(git -C "$COMFY" symbolic-ref --short -q HEAD 2>/dev/null || echo '(detached)')"
    _tag="$(git -C "$COMFY" describe --tags --abbrev=0 2>/dev/null)"
    echo "[comfy] git: ${_head} on ${_br}; nearest tag ${_tag:-none}; remote $(git -C "$COMFY" remote get-url origin 2>/dev/null || echo '?')"
else
    echo "[comfy] !!! ${COMFY} is not a git checkout -- core cannot be updated by this script"
fi

echo "[comfy] before: $(comfy_version)"
if [[ "$COMFY_UPDATE" == "1" ]]; then
    _mode="release"; [[ "$COMFY_TRACK" == "master" || "$COMFY_TRACK" == "branch" ]] && _mode="branch"
    git_sync "$COMFY" "ComfyUI" "$_mode" "${COMFY_PIN:-}"; rc=$?
    if (( rc == 0 )); then
        CHANGED_ANY=1
        echo "[comfy] source moved -> reinstalling requirements (torch pinned)"
        pip_reqs "${COMFY}/requirements.txt" "ComfyUI"
    elif (( rc == 2 )); then
        echo "[comfy] source unchanged"
    else
        echo "[comfy] !!! update did not complete -- see the [git] lines above"
    fi
    for p in comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs; do
        sync_pinned_pkg "$p"
    done
else
    echo "[comfy] COMFY_UPDATE=0 -> skipping"
fi

CV="$(comfy_version)"
MIN_CV="0.30.0"
echo "[comfy] after: ${CV:-<undetectable>}"
if [[ -z "$CV" ]]; then
    echo "[comfy] version undetectable -- verify >= ${MIN_CV} manually"
elif [[ "$(printf '%s\n%s\n' "$MIN_CV" "$CV" | sort -V | head -1)" == "$MIN_CV" ]]; then
    echo "[comfy] OK -- native H3 and SeedVR2 nodes present"
else
    echo "[comfy] !!! ${CV} is BELOW ${MIN_CV} -- H3 nodes will not exist. Read the [git] lines."
fi

# ---------------------------------------------------------------------------
# Custom nodes.  Entry: <git-url>[|<dir-name>][@<pin>]
# All track their default branch on every boot unless pinned.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop"
)
# Larryvrh's pack is only needed for larryvrh's own turbo LoRA; the lightx2v
# LoRA below is a plain LoRA and needs no extra nodes.
[[ "$WANT_TURBO" == "1" ]] && NODES+=( "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo" )

install_node() {
    local spec="$1" url name path pin rc fresh=0
    if [[ "$spec" == *"@"* && "$spec" != *"@"*"/"* ]]; then
        pin="${spec##*@}"; spec="${spec%@*}"
    else
        pin=""
    fi
    url="${spec%%|*}"
    if [[ "$spec" == *"|"* ]]; then name="${spec##*|}"; else name="$(basename "$url" .git)"; fi
    path="${NODES_DIR}/${name}"

    if [[ -d "$path" ]]; then
        if [[ "$NODE_UPDATE" == "1" ]]; then
            git_sync "$path" "$name" branch "$pin"; rc=$?
            (( rc == 0 )) && CHANGED_ANY=1
        else
            echo "[node] $name present (NODE_UPDATE=0)"
        fi
    else
        echo "[node] cloning $name"
        git clone --recursive "$url" "$path" || { echo "[node] CLONE FAILED: $name"; return 0; }
        [[ -n "$pin" ]] && git -C "$path" checkout --quiet "$pin" 2>/dev/null
        fresh=1; CHANGED_ANY=1
    fi

    local req="${path}/requirements.txt"
    if [[ -f "$req" ]]; then
        if (( fresh )) || [[ "$FORCE_DEPS" == "1" ]] || reqs_changed "$path" "$req"; then
            pip_reqs "$req" "$name"
        else
            echo "[node] $name: requirements unchanged"
        fi
    fi
    if [[ -f "${path}/install.py" ]] && { (( fresh )) || [[ "$FORCE_DEPS" == "1" ]]; }; then
        ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
    fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
for n in "${NODES[@]}"; do install_node "$n"; done

# Legacy third-party SeedVR2 pack (superseded by core nodes).
LEGACY_NODE="${NODES_DIR}/seedvr2_videoupscaler"
LEGACY_MODELS="${COMFY}/models/SEEDVR2"
if [[ -d "$LEGACY_NODE" || -d "$LEGACY_MODELS" ]]; then
    echo "=================== LEGACY SEEDVR2 ==================="
    if [[ "${PURGE_LEGACY_SEEDVR2:-0}" == "1" ]]; then
        [[ -d "$LEGACY_NODE"   ]] && { echo "[legacy] removing ${LEGACY_NODE}";   rm -rf "$LEGACY_NODE"; }
        [[ -d "$LEGACY_MODELS" ]] && { echo "[legacy] removing ${LEGACY_MODELS}"; rm -rf "$LEGACY_MODELS"; }
    else
        echo "[legacy] old SeedVR2 pack/weights present. PURGE_LEGACY_SEEDVR2=1 to delete."
    fi
fi

echo "[provisioning] reconciling cuda-python to the CUDA-12 line"
pip_install "cuda-python<13"

# SageAttention: approximate attention, ~2x throughput. Drafts only.
# Launch with --use-sage-attention (no KJNodes patch node needed).
if [[ "${INSTALL_SAGE:-0}" == "1" ]]; then
    echo "=================== SAGEATTENTION ==================="
    if "$PY" -c "import sageattention" 2>/dev/null; then
        echo "[sage] already installed"
    else
        pip_install --no-cache-dir sageattention \
            || echo "[sage] FAILED -- get a wheel from github.com/woct0rdho/SageAttention/releases"
    fi
fi

echo "=================== TORCH VERIFY ==================="
verify_torch

# ===========================================================================
# DOWNLOADS -- HF via hf_xet, Civitai via aria2
# ===========================================================================
"$PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
"$PY" -c "import hf_xet" 2>/dev/null || pip_install hf_xet || echo "[provisioning] WARNING: hf_xet install failed"

export HF_HOME="${WORKSPACE:-/workspace}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"
mem_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
[[ -n "$mem_gb" ]] && (( mem_gb >= 64 )) && export HF_XET_HIGH_PERFORMANCE=1

CURL_AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && { CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}"); echo "[provisioning] HF_TOKEN detected"; }

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
            echo "[model] $name complete, skipping"; return 0
        elif [[ -n "$want" ]]; then
            echo "[model] $name size mismatch (${have} != ${want}) -> re-fetching"
            rm -f "$dest"
        else
            echo "[model] $name present, size unverifiable, assuming complete"; return 0
        fi
    fi

    echo "[model] downloading $name (${repo})"
    if "$PY" "$HF_GET" "$repo" "$rpath" "$dest"; then
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        if [[ -n "$want" ]] && (( have != want )); then
            echo "[model] WARNING: $name size ${have} != expected ${want}"
        else
            echo "[model] $name OK (${have} bytes)"
        fi
    else
        echo "[model] DOWNLOAD FAILED: $name (retries next boot)"
    fi
}

CIVITAI_RESERVE_GB="${CIVITAI_RESERVE_GB:-20}"

# dl_civitai <dest_dir> <dest_filename> <full_url>
dl_civitai() {
    local dir="$1" name="$2" url="$3"
    local dest="${dir}/${name}"
    mkdir -p "$dir"

    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        echo "[civitai] $name present, skipping"; return 0
    fi
    [[ -n "${CIVITAI_TOKEN:-}" ]] || { echo "[civitai] SKIP $name -- CIVITAI_TOKEN unset"; return 0; }

    local common=(-x 16 -s 16 -k 1M --file-allocation=none --summary-interval=15
                  --continue=true --auto-file-renaming=false --allow-overwrite=true
                  --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600
                  --max-file-not-found=2)

    echo "[civitai] downloading $name"
    if ! aria2c "${common[@]}" --header="Authorization: Bearer ${CIVITAI_TOKEN}" -d "$dir" -o "$name" "$url"; then
        echo "[civitai] header auth failed -> retrying with query-string token (lands in provisioning.log; rotate if shared)"
        local sep="?"; [[ "$url" == *\?* ]] && sep="&"
        aria2c "${common[@]}" -d "$dir" -o "$name" "${url}${sep}token=${CIVITAI_TOKEN}" \
            || { echo "[civitai] DOWNLOAD FAILED: $name"; return 0; }
    fi

    local sz; sz="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if (( sz < 1048576 )); then
        echo "[civitai] WARNING: $name is ${sz} bytes -- an error page, not a model. Removing."
        rm -f "$dest"; return 0
    fi
    echo "[civitai] $name OK (${sz} bytes)"
}

# ---------------------------------------------------------------------------
# Manifest  (~131 GB)
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
WF="${COMFY}/user/default/workflows"

# --- Civitai:  "dest_dir|dest_filename|full_url"  (empty by default) ---
CIVITAI_FILES=(
)

# --- Hugging Face:  "hf|dest_dir|dest_filename|repo_id|repo_path" ---
MODELS=(
    # Diffusion model, bf16 (61.7 GB). fl2va serves I2V, T2V and first-last.
    # int8_convrot (31.7 GB) is the small step down if disk binds.
    "hf|$DIFF|minimax_h3_fl2va_bf16.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_bf16.safetensors"

    # Text encoder, bf16 (48.0 GB). Smallest quality lever; first thing to
    # trade for int8_convrot (25.3 GB) if RAM or disk binds.
    "hf|$TE|qwen3vl_32b_minimax_h3_bf16.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"

    # VAEs -- both required, even for silent output.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"

    # Turbo / lightning LoRA -- Kijai's lightx2v 4-step (draft tier).
    "hf|$LORA|minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors|Kijai/MiniMax-H3_comfy|loras/minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors"

    # Other LoRAs
    "hf|$LORA|H3_K3NK_Side_Deepthroat.safetensors|Ghiladden/7801|7800.safetensors"

    # Workflows
    "hf|$WF|Mike MiniMax H3 I2V Context Extend v1.json|Ghiladden/7801|Mike MiniMax H3 I2V Context Extend v1.json"
)

# SeedVR2 restore weights (~15 GB) -- native Comfy-Org conversions. 7B fp16,
# not sharp: at the 768 ceiling you want texture overwritten, not sharpened.
if [[ "$WANT_SEEDVR2" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|seedvr2_7b_fp16.safetensors|Comfy-Org/SeedVR2|diffusion_models/seedvr2_7b_fp16.safetensors"
    "hf|$VAE|seedvr2_ema_vae_fp16.safetensors|Comfy-Org/SeedVR2|vae/seedvr2_ema_vae_fp16.safetensors"
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
    local margin=$(( 15 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15 GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Give-backs: PURGE_LEGACY_SEEDVR2=1 (-15 GB), encoder int8_convrot (-22.7 GB),"
        echo "[provisioning] !!!            diffusion int8_convrot (-30 GB), WANT_SEEDVR2=0 (-15 GB)"
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
        echo "[civitai] CIVITAI_TOKEN unset -- skipping all civitai downloads"
    else
        for entry in "${CIVITAI_FILES[@]}"; do
            IFS='|' read -r cdir cname curl_ <<< "$entry"
            dl_civitai "$cdir" "$cname" "$curl_"
        done
    fi
fi

echo "=================== HUGGING FACE ==================="
if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        case "$kind" in
            hf) dl_hf "$a" "$b" "$c" "$d" ;;
            *)  echo "[model] unknown manifest kind '$kind' in: $entry" ;;
        esac
    done
else
    echo "[provisioning] HF model phase skipped (disk)"
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
for nd in ComfyUI-VideoHelperSuite ComfyUI-KJNodes ComfyUI-Olm-DragCrop \
          ComfyUI-RIFE-TensorRT-Auto ComfyUI-MiniMaxH3-Contex-Loop \
          ComfyUI-MiniMax-H3-Turbo; do
    if [[ -d "${NODES_DIR}/${nd}" ]]; then
        echo "[layout] OK      custom_nodes/${nd} @ $(git -C "${NODES_DIR}/${nd}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    else
        echo "[layout] absent  custom_nodes/${nd}"
    fi
done

# ===========================================================================
# FINALISE -- log discovery, health check, restart
# ===========================================================================
echo "=================== FINALISE ==================="

# Read the comfyui log path and launch command out of the supervisor block.
COMFY_LOG=""; COMFY_CMD=""
for d in /etc/supervisor/conf.d /etc/supervisor/supervisord/conf.d /etc/supervisord.d /etc/supervisor; do
    [[ -d "$d" ]] || continue
    for f in "$d"/*.conf "$d"/*.ini; do
        [[ -f "$f" ]] || continue
        grep -q '^\[program:comfyui\]' "$f" 2>/dev/null || continue
        [[ -z "$COMFY_LOG" ]] && COMFY_LOG="$(awk '/^\[program:comfyui\]/{p=1;next} /^\[/{p=0} p&&/^[[:space:]]*stdout_logfile[[:space:]]*=/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "$f")"
        [[ -z "$COMFY_CMD" ]] && COMFY_CMD="$(awk '/^\[program:comfyui\]/{p=1;next} /^\[/{p=0} p&&/^[[:space:]]*command[[:space:]]*=/{sub(/^[^=]*=[[:space:]]*/,"");print;exit}' "$f")"
    done
done
if [[ -z "$COMFY_LOG" ]]; then
    for c in /var/log/portal/comfyui.log /var/log/supervisor/comfyui.log /var/log/comfyui.log; do
        [[ -f "$c" ]] && { COMFY_LOG="$c"; break; }
    done
fi
[[ -n "$COMFY_LOG" ]] && echo "[provisioning] comfyui log: ${COMFY_LOG}" \
                      || echo "[provisioning] comfyui log: not found (ls /var/log/portal /var/log/supervisor)"

# Sample the pid twice: supervisor says RUNNING after 5 s even in a crash-loop.
comfy_pid() { supervisorctl status comfyui 2>/dev/null | grep -oE 'pid [0-9]+' | awk '{print $2}'; }
COMFY_STATE="unknown"
if command -v supervisorctl >/dev/null 2>&1; then
    _p1="$(comfy_pid)"; sleep 12; _p2="$(comfy_pid)"
    _st="$(supervisorctl status comfyui 2>/dev/null | awk '{print $2}')"
    if [[ -n "$_p1" && -n "$_p2" && "$_p1" != "$_p2" ]]; then
        COMFY_STATE="flapping"
        echo "[provisioning] !!! comfyui is CRASH-LOOPING (pid ${_p1} -> ${_p2} in 12 s)"
        echo "[provisioning] !!! supervisorctl stop comfyui, then run it in the foreground:"
        [[ -n "$COMFY_CMD" ]] && echo "[provisioning] !!!   cd ${COMFY} && ${COMFY_CMD}" \
                              || echo "[provisioning] !!!   cd ${COMFY} && ${PY} main.py"
    elif [[ "$_st" == "RUNNING" && -n "$_p2" ]]; then
        COMFY_STATE="stable"; echo "[provisioning] comfyui stable (pid ${_p2})"
    else
        COMFY_STATE="${_st:-absent}"; echo "[provisioning] comfyui state: ${COMFY_STATE}"
    fi
fi

if (( CHANGED_ANY )); then
    if [[ "$RESTART_COMFY_ON_UPDATE" != "1" ]]; then
        echo "[provisioning] code changed. Restart to load it:  supervisorctl restart comfyui"
    elif [[ "$COMFY_STATE" == "flapping" ]]; then
        echo "[provisioning] code changed but comfyui is flapping -> NOT restarting; fix the crash first"
    elif command -v supervisorctl >/dev/null 2>&1; then
        echo "[provisioning] code changed -> restarting comfyui"
        supervisorctl restart comfyui || echo "[provisioning] restart FAILED -- bounce it manually"
    fi
else
    echo "[provisioning] nothing changed this boot"
fi

# ---------------------------------------------------------------------------
# Operating notes
# ---------------------------------------------------------------------------
cat <<'NOTES'

=================== FINAL RENDER ===================
  diffusion     minimax_h3_fl2va_bf16          text encoder  qwen3vl_32b_minimax_h3_bf16
  vae           video_vae_fp16 + audio_vae_fp32 (both wired, even for silent output)
  sampler       res_multistep + simple, 25-30 steps. No LoRA. No Sage.
  canvas        1344x768 (16:9, ~1.0 MP, multiple of 32). 768 short edge is both
                the floor and the cap; draft with fewer frames/steps, never lower res.
  frames        17k+5 at 24 fps: 124 = 5 s, 362 = 15 s.
  shift         12 (video) / 3 (audio). Fixed constants, not dials.

=================== DRAFT TIER -- lightx2v turbo LoRA ===================
  LoRA          minimax_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16
  load          plain LoraLoaderModelOnly between model loader and sampler, strength 1.0
  sampler       euler, cfg 1.0, 4 steps (up to 8), shift 12/3 unchanged
  use           silent drafts for prompt/composition only. Audio is fragile on
                distilled tiers; shift and LoRA experiments belong on the base model.
  seeds         do not transfer to final settings -- expect to re-roll.

=================== PROMPTING ===================
  BasicGuider at cfg 1 has no negative path: in-prompt negation backfires.
  Suppress an unwanted motion axis by over-specifying the wanted one.
  LoRA trigger words go at the FRONT of the prompt.
  Fill the audio fields even for muted output -- joint AV attention means an
  empty audio path degrades motion timing in the video.
  Same failure across three seeds = prompt/schema bug, not variance.

=================== RESTORE -- native SeedVR2 ===================
  VAEDecode (H3) -> Resize x1.875 lanczos -> Pre-Process SeedVR2 Input
    -> VAEEncodeTiled (SeedVR2 VAE, 512/128) -> [Split SeedVR2 Latent, overlap 3]
    -> KSampler 1 step, cfg 1, euler, simple, denoise 1 -> [Merge SeedVR2 Latents]
    -> VAEDecodeTiled -> Post-Process SeedVR2 Output -> RIFE -> CreateVideo
  Loaders: UNETLoader seedvr2_7b_fp16 (feeds Conditioning AND KSampler);
           VAELoader seedvr2_ema_vae_fp16 (feeds both tiled VAE nodes).
  Wire the Resize output to Post-Process original_resized_images as well.
  Audio bypasses the restore chain straight into CreateVideo.
  Build this as a separate graph loading frames from disk so you can iterate.

=================== INTERPOLATION (RIFE TensorRT) ===================
  Order: generate -> SeedVR2 -> RIFE -> encode. 24 -> 48/60 fps.
  Set CreateVideo fps to match or the video plays at half speed.
  First run compiles the TRT engine (slow, not a hang; per GPU architecture).

=================== TROUBLESHOOTING ===================
  Core/nodes not updating: read the [git] lines in provisioning.log.
    "fetch FAILED"          network or remote URL problem
    "dirty tree"            local edits; GIT_FORCE_RESET=1
    "fast-forward blocked"  local commits; GIT_FORCE_RESET=1
    "ref not found"         bad COMFY_PIN or @pin
    "not a git checkout"    the image installed ComfyUI without git; reclone
    If none of those appear and version still stays old: something else is
    checking core back out before this runs -- grep the supervisor scripts:
      grep -rn 'checkout\|COMFYUI_REF' /etc/supervisor /opt/supervisor-scripts 2>/dev/null
  Version in the UI is the frontend package, not core. Core: comfyui_version.py.
  Slow start: check the last [provisioning] comfyui line for CRASH-LOOPING.
  Node import failures: read the LAST traceback line. flash_attn errors are
    handled by the [flash] block near the top of this log.
  Pathologically slow model load on 0.30.x: launch with --disable-pinned-memory.
  Phantom links on autogrow nodes: frontend skew; delete and re-add the node.
  CUDA gone after an update: torch was replaced; reinstall the pinned build.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
