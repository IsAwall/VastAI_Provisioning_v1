#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai   ---  v6
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text (GitHub "Raw"
#      button URL, Gist raw URL, pastebin raw URL).
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. (Optional but recommended) set  HF_TOKEN=<your token>  in the instance
#      env -- authenticated HF transfers are prioritised and avoid 429s.
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
# Idempotent: nodes cloned if missing / updated if present; models skipped if
# already complete, resumed if partial. Individual failures are logged, not fatal.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED IN v7  (v6's update fix was too aggressive -- this reins it in)
#
#   v6 fixed "nothing ever updates" but introduced two defects of its own. Both
#   were found on a Vast ComfyUI template (NOT ai-dock) where they combined to
#   put ComfyUI into a supervisor crash-loop.
#
#   DEFECT 1: detached HEAD was treated as damage to repair.
#     v6's git_sync() saw a detached HEAD and checked out the remote default
#     branch to "recover" it. But most images pin ComfyUI core to a release tag
#     ON PURPOSE -- ai-dock via COMFYUI_REF, Vast via its own template. v6 read
#     that deliberate pin as breakage and dragged core onto bleeding-edge master
#     (observed: 23 commits past v0.33.0). ComfyUI's own README warns that
#     commits outside stable release tags may be very unstable and break many
#     custom nodes, and the launch flags baked into a template are written
#     against the release it pinned.
#
#     v7: a detached HEAD with no explicit pin is LEFT ALONE and logged. Branch
#     recovery now requires ALLOW_BRANCH_RECOVERY=1. Core is not updated at all
#     unless you opt in with UPDATE_COMFY_CORE=1 or name a COMFY_PIN.
#
#   DEFECT 2: the auto-restart was gated on a meaningless condition.
#     v6 restarted comfyui if supervisorctl reported RUNNING. Supervisor reports
#     RUNNING once the process has merely survived `startsecs` (5s by default) --
#     it says nothing about whether ComfyUI finished importing nodes and bound
#     its port. Against a service that was already flapping, `restart` just added
#     another cycle to the loop:
#
#         INFO success: comfyui entered RUNNING state ... (startsecs)
#         INFO exited: comfyui (exit status 0; expected)
#         INFO spawned: 'comfyui' with pid <next>      <-- every ~6 seconds
#
#     v7: RESTART_COMFY_ON_UPDATE now DEFAULTS TO 0. When enabled it samples the
#     service pid twice before acting and refuses to restart a flapping service.
#
#   ALSO NEW IN v7:
#     - Post-run health check. Samples the comfyui pid twice; a changed pid means
#       crash-looping, and the script says so loudly instead of reporting success.
#     - Log path discovery. The comfyui log is NOT at a fixed path -- ai-dock uses
#       /var/log/supervisor/comfyui.log, Vast templates use /var/log/portal/.
#       v7 reads stdout_logfile out of the supervisor program block and prints it,
#       along with the exact foreground command to reproduce a failed startup.
#     - Core git state (branch/tag/commit) is reported every run, so a drift like
#       defect 1 is visible in the log rather than silent.
#
#   RETAINED FROM v6 (these were the actual fixes and they were correct):
#     - git_sync() replacing v5's commented-out `git pull --ff-only`, which meant
#       any already-present node was logged "present" and never updated again.
#     - Shallow-clone deepening, dirty-worktree detection, and the "(unknown)"
#       guard on default-branch lookup.
#     - Core requirements reinstalled when core moves (the web UI ships as the
#       pip package comfyui-frontend-package pinned in core's requirements.txt,
#       so pulling core without a pip install leaves you on the old frontend).
#     - Conditional node dependency installs, hashed on requirements.txt.
#     - Optional "|<commit>" pins on NODES entries.
#
# -----------------------------------------------------------------------------
# WHY hf_xet (retained from v5)
#
#   The Comfy-Org Wan repos are served from HF's Xet backend, not classic Git-LFS.
#   Xet hands out a SEPARATE CloudFront signed URL per byte-range chunk, each
#   locked to its own range by policy. aria2's multi-connection range-splitting
#   fundamentally conflicts with that: it follows one redirect, then fires N
#   connections at that single URL with different Range headers -- every request
#   outside the URL's authorised range gets 403 and the connection dies. Result
#   was connection attrition (6 -> 1) and a ~7x slowdown even though the instance
#   pipe peaked >500 MiB/s. No aria2 flag fixes this. huggingface_hub + hf_xet
#   queries the Xet CAS for the chunk-reconstruction manifest and fetches xorb
#   ranges with adaptive concurrency (auto-scales to 64 streams). aria2 is kept
#   ONLY for non-HF ("url") entries and is not installed if there are none.
#
#   >>> DISK NOTE: two fp16 14B Wan files are ~26 GiB EACH. With Xet's on-disk
#   >>> chunk cache adding transient overhead, ~64 GiB of models on a 72 GiB disk
#   >>> is uncomfortably tight. Rent with ~150 GiB, or switch to the fp8 Wan
#   >>> variants (~half the size). See the fp8 note in the manifest below.
#
# -----------------------------------------------------------------------------
# ENV TOGGLES
#   UPDATE_NODES=1              update custom nodes each boot (default 1)
#   UPDATE_COMFY_CORE=0         also update ComfyUI core (default 0 -- OFF).
#                               Most images pin core deliberately; leave this off
#                               unless you manage core yourself.
#   ALLOW_BRANCH_RECOVERY=0     let a detached HEAD be moved onto the remote
#                               default branch. Off by default: detached usually
#                               means image-pinned, not broken.
#   UPDATE_ON_BOOT              legacy v6 name; still honoured, seeds UPDATE_NODES
#   FORCE_DEPS=0                reinstall all pip deps regardless of change
#   FORCE_CLEAN=0               hard-reset dirty repos instead of skipping them
#   RESTART_COMFY_ON_UPDATE=0   bounce the comfyui service if code changed
#                               (default 0; refuses to restart a flapping service)
#   SKIP_MODELS=0               nodes only, no model downloads
#   COMFY_PIN=<sha|tag>         check core out to this ref and hold it there.
#                               Setting this implies core management is wanted.
#   COMFY_MIN_VERSION=<x.y.z>   warn if core is below this (unset = no check)
# =============================================================================

# Note: deliberately NOT using `set -e` -- one failed node/model should not stop
# the rest of provisioning.
set -o pipefail

UPDATE_NODES="${UPDATE_NODES:-${UPDATE_ON_BOOT:-1}}"   # legacy UPDATE_ON_BOOT seeds this
UPDATE_COMFY_CORE="${UPDATE_COMFY_CORE:-0}"
ALLOW_BRANCH_RECOVERY="${ALLOW_BRANCH_RECOVERY:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-0}"
SKIP_MODELS="${SKIP_MODELS:-0}"

# ---------------------------------------------------------------------------
# Persistent log  (Vast's "Logs" button only shows a recent snapshot, so mirror
# everything to a file; tail it live with:  tail -f /workspace/provisioning.log)
# ---------------------------------------------------------------------------
mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (v6): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

# ---------------------------------------------------------------------------
# Paths & the Python interpreter ComfyUI actually uses.
# Installing via "$PY -m pip" guarantees packages land in ComfyUI's env rather
# than a stray system pip (which may not exist / may hit PEP 668).
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

# ---------------------------------------------------------------------------
# System packages the script itself needs (image-layer binaries vanish on a
# fresh instance even though /workspace persists, so re-check every boot).
# ---------------------------------------------------------------------------
ensure_pkg() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "[provisioning] '$1' missing -> installing '$2'"
    apt-get update -qq && apt-get install -y -qq "$2" \
        || echo "[provisioning] WARNING: failed to install '$2'"
}

ensure_pkg git  git    # cloning + updating custom nodes and core
ensure_pkg curl curl   # HEAD size checks for the skip/pre-flight logic
# aria2 is installed later, only if the manifest actually has non-HF entries.

# ai-dock runs provisioning as root but the persistent volume may be owned by
# another uid. Without this, every git call fails "detected dubious ownership"
# and the whole update phase degrades to silent no-ops.
git config --global --add safe.directory '*' 2>/dev/null || true

# ===========================================================================
# GIT SYNC  --  the core of the v6 update fix
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
        # A detached HEAD is ambiguous. `git pull` there exits 0 and does nothing,
        # so it LOOKS updated -- but detached usually means the image pinned this
        # checkout on purpose (ai-dock COMFYUI_REF, Vast templates, node pins).
        # v6 "recovered" it onto the default branch and dragged core onto master.
        # v7 leaves it alone unless told otherwise.
        if ! git -C "$path" symbolic-ref -q HEAD >/dev/null 2>&1 \
             && [[ "$ALLOW_BRANCH_RECOVERY" != "1" ]]; then
            echo "[git] $name: detached HEAD -- looks image-pinned, leaving alone"
            echo "[git] $name: (set ALLOW_BRANCH_RECOVERY=1 to move it onto a branch,"
            echo "[git] $name:  or pass an explicit pin to move it deliberately)"
            return 1
        fi
        if ! git -C "$path" symbolic-ref -q HEAD >/dev/null 2>&1; then
            local defb="" cand
            # Some clones leave origin/HEAD unset; repair it before asking.
            git -C "$path" remote set-head origin -a >/dev/null 2>&1 || true
            defb="$(git -C "$path" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')"
            # `git remote show` prints the literal "(unknown)" when it can't tell;
            # checking that out fails and would strand the repo detached.
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
# COMFYUI CORE  (new in v6 -- v5 never touched this)
# ===========================================================================
echo "=================== COMFYUI CORE ==================="
# Report core git state every run, so drift is visible instead of silent.
if [[ -d "${COMFY}/.git" ]]; then
    _head="$(git -C "$COMFY" rev-parse --short HEAD 2>/dev/null)"
    _br="$(git -C "$COMFY" symbolic-ref --short -q HEAD 2>/dev/null || echo '(detached)')"
    _tag="$(git -C "$COMFY" describe --tags --abbrev=0 2>/dev/null)"
    _ahead="$(git -C "$COMFY" rev-list --count "${_tag}..HEAD" 2>/dev/null || echo '?')"
    echo "[core] git: ${_head} on ${_br}; nearest tag ${_tag:-none} (+${_ahead} commits)"
    if [[ "$_br" != "(detached)" && "$_ahead" != "0" && "$_ahead" != "?" ]]; then
        echo "[core] NOTE: ${_ahead} commits past ${_tag}. Templates pin a release for a"
        echo "[core] NOTE: reason -- launch flags and custom nodes are tested against it."
        echo "[core] NOTE: To return:  git -C ${COMFY} checkout ${_tag} && \\"
        echo "[core] NOTE:             ${PY} -m pip install -r ${COMFY}/requirements.txt"
    fi
fi

if [[ -n "${COMFY_PIN:-}" || "$UPDATE_COMFY_CORE" == "1" ]] && [[ -d "${COMFY}/.git" ]]; then
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
elif [[ ! -d "${COMFY}/.git" ]]; then
    echo "[core] ${COMFY} is not a git checkout -> nothing to manage"
else
    echo "[core] not managing core (UPDATE_COMFY_CORE=0, no COMFY_PIN) -- this is the"
    echo "[core] safe default; the image's own pin is left intact"
fi

comfy_version() {
    local vf="${COMFY}/comfyui_version.py"
    [[ -f "$vf" ]] && sed -n 's/^__version__ *= *"\(.*\)"/\1/p' "$vf" | head -1
}
CV="$(comfy_version)"
if [[ -n "$CV" ]]; then
    echo "[core] ComfyUI version: ${CV}"
    if [[ -n "${COMFY_MIN_VERSION:-}" ]]; then
        if [[ "$(printf '%s\n%s\n' "$COMFY_MIN_VERSION" "$CV" | sort -V | head -1)" != "$COMFY_MIN_VERSION" ]]; then
            echo "[core] !!! WARNING: ComfyUI ${CV} is below the required ${COMFY_MIN_VERSION}"
        fi
    fi
else
    echo "[core] could not read comfyui_version.py"
fi

# ---------------------------------------------------------------------------
# Custom nodes  (missing -> clone; present -> update; deps only when they change)
# Format: "url|pin". Leave the pin empty to track the default branch.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/kijai/ComfyUI-KJNodes|"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite|"
    "https://github.com/yolain/ComfyUI-Easy-Use|"
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto|"
)

install_node() {
    local entry="$1" url pin name path fresh=0 changed=0
    IFS='|' read -r url pin <<< "$entry"
    name="$(basename "$url" .git)"
    path="${NODES_DIR}/${name}"

    if [[ -d "$path" ]]; then
        if [[ "$UPDATE_NODES" == "1" ]]; then
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

# ---------------------------------------------------------------------------
# CUDA reconciliation (from v3/v4, now conditional)
# RIFE-TensorRT-Auto's deps pull cuda-python 13.x, which swaps cuda-bindings to
# 13.x and breaks the image's cu12 torch (pins cuda-bindings==12.9.x). Cap it.
# Only needed while that node is installed. If torch ever moves to a CUDA-13
# build, remove this block.
# ---------------------------------------------------------------------------
if [[ -d "${NODES_DIR}/ComfyUI-RIFE-TensorRT-Auto" ]]; then
    echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
    pip_install "cuda-python<13"
fi

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE  (unchanged from v5)
# ===========================================================================

# --- HF client: ensure hf_xet is available for Xet-native transfers ---
# huggingface_hub is already present (ComfyUI depends on it); >=0.32 bundles
# hf_xet. We only add hf_xet if it's not importable, so ComfyUI's hub pin is
# left untouched in the common case. (hf_transfer is deprecated; hf_xet replaces it.)
"$PY" -c "import huggingface_hub" 2>/dev/null || pip_install huggingface_hub
if ! "$PY" -c "import hf_xet" 2>/dev/null; then
    echo "[provisioning] installing hf_xet for fast HF (Xet) downloads"
    pip_install hf_xet || echo "[provisioning] WARNING: hf_xet install failed -> HF downloads will fall back to the (slower) LFS bridge"
fi

# --- HF environment ---
# Keep every HF cache on the persistent volume (default is ~/.cache on the
# ephemeral container root, which is small and lost on reboot).
export HF_HOME="${WORKSPACE:-/workspace}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0        # make sure the deprecated path is off
mkdir -p "$HF_HOME"
# HF_XET_HIGH_PERFORMANCE raises concurrency bounds + buffer sizes; HF recommends
# it only for boxes with >=64 GB RAM. Enable adaptively.
mem_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$mem_gb" ]] && (( mem_gb >= 64 )); then
    export HF_XET_HIGH_PERFORMANCE=1
    echo "[provisioning] ${mem_gb} GB RAM -> HF_XET_HIGH_PERFORMANCE=1"
fi

# Optional HF token (also used by the curl HEAD checks below).
CURL_AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    CURL_AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    echo "[provisioning] HF_TOKEN detected -> authenticated downloads"
fi

map_url() {
    # Optional HF mirror via HF_ENDPOINT (e.g. https://hf-mirror.com). No-op if unset.
    local u="$1"
    [[ -n "${HF_ENDPOINT:-}" ]] && u="${u/https:\/\/huggingface.co/${HF_ENDPOINT%/}}"
    printf '%s' "$u"
}

hf_resolve_url() {
    # hf_resolve_url <repo_id> <repo_path> -> /resolve/ URL (used ONLY for the
    # size check; the actual bytes come via hf_xet). x-linked-size on this
    # endpoint gives the true file size even for Xet-backed files.
    map_url "https://huggingface.co/${1}/resolve/main/${2}"
}

remote_size() {
    # Echo expected size in bytes, or nothing if it can't be determined.
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

MODELS=(
    # --- Diffusion models (Wan 2.2 I2V 14B, fp16 ~26 GiB each) ---
    "hf|$DM|wan2.2_i2v_high_noise_14B_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    "hf|$DM|wan2.2_i2v_low_noise_14B_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
    #   fp8 alternative (~half the size + disk; swap in if your workflows allow):
    # "hf|$DM|wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    # "hf|$DM|wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

    # --- SVI / Lightx2v LoRAs ---
    "hf|$LORA|SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors|Kijai/WanVideo_comfy|LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
    "hf|$LORA|SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors|Kijai/WanVideo_comfy|LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"
    "hf|$LORA|lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors|Kijai/WanVideo_comfy|Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

    # --- VAE ---
    "hf|$VAE|wan_2.1_vae.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/vae/wan_2.1_vae.safetensors"

    # --- Text encoder ---
    "hf|$TE|umt5_xxl_fp8_e4m3fn_scaled.safetensors|chatpig/encoder|umt5_xxl_fp8_e4m3fn_scaled.safetensors"
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
        echo "[provisioning] !!! or switch to the fp8 Wan variants in the manifest (~half the size)."
        return 1
    fi
    # Soft warning: Xet's chunk cache needs transient headroom during the run.
    if (( avail - need < 15 * 1024*1024*1024 )); then
        echo "[provisioning] NOTE: tight headroom after download (< ~15GiB). Xet's chunk cache"
        echo "[provisioning] NOTE: uses transient space; consider a bigger disk or the fp8 models."
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch models
# ---------------------------------------------------------------------------
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
# FINALISE: health check, log discovery, optional restart
# ===========================================================================
echo "=================== FINALISE ==================="

# --- where does this image actually log ComfyUI? ---
# There is no fixed path. ai-dock: /var/log/supervisor/comfyui.log.
# Vast templates: /var/log/portal/. Read it out of the supervisor program block.
COMFY_LOG=""
COMFY_CMD=""
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
                      || echo "[provisioning] comfyui log: not found (check: ls /var/log/portal /var/log/supervisor)"

# --- is the service healthy, or flapping? ---
# supervisor reports RUNNING once the process survives `startsecs` (5s). That
# says nothing about whether ComfyUI imported its nodes and bound its port. A
# crash-loop therefore reads as a rapid succession of healthy RUNNING states.
# Sampling the pid twice is what actually distinguishes them.
comfy_pid() {
    supervisorctl status comfyui 2>/dev/null | grep -oE 'pid [0-9]+' | awk '{print $2}'
}
COMFY_STATE="unknown"
if command -v supervisorctl >/dev/null 2>&1; then
    _p1="$(comfy_pid)"; sleep 12; _p2="$(comfy_pid)"
    _st="$(supervisorctl status comfyui 2>/dev/null | awk '{print $2}')"
    if [[ -n "$_p1" && -n "$_p2" && "$_p1" != "$_p2" ]]; then
        COMFY_STATE="flapping"
        echo "[provisioning] !!! comfyui is CRASH-LOOPING (pid ${_p1} -> ${_p2} in 12s)"
        echo "[provisioning] !!! It is being respawned, not starting slowly."
        echo "[provisioning] !!! Stop the loop and run it in the foreground to see why:"
        echo "[provisioning] !!!   supervisorctl stop comfyui"
        [[ -n "$COMFY_CMD" ]] && echo "[provisioning] !!!   cd ${COMFY} && ${COMFY_CMD}" \
                              || echo "[provisioning] !!!   cd ${COMFY} && ${PY} main.py"
    elif [[ "$_st" == "RUNNING" && -n "$_p2" ]]; then
        COMFY_STATE="stable"
        echo "[provisioning] comfyui stable (pid ${_p2})"
    else
        COMFY_STATE="${_st:-absent}"
        echo "[provisioning] comfyui state: ${COMFY_STATE}"
    fi
fi

# --- restart only if it is both wanted and safe ---
if (( CHANGED_ANY )); then
    if [[ "$RESTART_COMFY_ON_UPDATE" != "1" ]]; then
        echo "[provisioning] code changed. Restart to load it:  supervisorctl restart comfyui"
        echo "[provisioning] (auto-restart is off by default in v7; RESTART_COMFY_ON_UPDATE=1 to enable)"
    elif [[ "$COMFY_STATE" == "flapping" ]]; then
        echo "[provisioning] code changed, but comfyui is flapping -> NOT restarting"
        echo "[provisioning] (restarting a crash-loop just adds a cycle; fix the crash first)"
    elif command -v supervisorctl >/dev/null 2>&1; then
        echo "[provisioning] code changed and service is stable -> restarting comfyui"
        supervisorctl restart comfyui || echo "[provisioning] restart FAILED -- bounce it manually"
    fi
else
    echo "[provisioning] nothing changed this boot"
fi

echo "=================== PROVISIONING COMPLETE ==================="
