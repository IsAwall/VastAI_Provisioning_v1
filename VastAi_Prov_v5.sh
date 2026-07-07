#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai   ---  v5
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
# WHAT CHANGED IN v5  (the real fix -- transport, not just correctness):
#
#   The Comfy-Org Wan repos (and most HF repos now) are served from HF's *Xet*
#   backend, not classic Git-LFS. Xet hands out a SEPARATE CloudFront signed URL
#   per byte-range chunk, each locked to its own range by policy. aria2's multi-
#   connection range-splitting fundamentally conflicts with that: it follows one
#   redirect, then fires N connections at that single URL with different Range
#   headers -- every request outside the URL's authorised range gets 403 and the
#   connection dies. Result was connection attrition (6 -> 1) and a ~7x slowdown,
#   even though the instance pipe peaked >500 MiB/s. No aria2 flag fixes this.
#
#   v5 therefore downloads HF files with huggingface_hub + hf_xet, the Xet-native
#   client. It queries the Xet CAS for the chunk-reconstruction manifest and
#   fetches the xorb ranges with adaptive concurrency (auto-scales to 64 streams
#   based on live bandwidth -- no tuning needed). aria2 is kept ONLY for any
#   non-HF ("url") entries, and is not even installed if there are none.
#
#   Also: HF_XET_HIGH_PERFORMANCE is auto-enabled on boxes with >=64 GB RAM
#   (raises concurrency + buffers); HF caches are pinned under $WORKSPACE so they
#   land on the persistent volume, not the ephemeral container root; the size-
#   verified skip logic and disk pre-flight from v4 are retained unchanged.
#
#   >>> DISK NOTE: two fp16 14B Wan files are ~26 GiB EACH. With Xet's on-disk
#   >>> chunk cache adding transient overhead, ~64 GiB of models on a 72 GiB disk
#   >>> is uncomfortably tight. Rent with ~150 GiB, or switch to the fp8 Wan
#   >>> variants (~half the size). See the fp8 note in the manifest below.
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
echo "########## provisioning run (v5): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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

ensure_pkg git  git    # cloning custom nodes
ensure_pkg curl curl   # HEAD size checks for the skip/pre-flight logic
# aria2 is installed later, only if the manifest actually has non-HF entries.

# ---------------------------------------------------------------------------
# Custom nodes  (present -> reinstall deps; missing -> clone, then deps)
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/yolain/ComfyUI-Easy-Use"
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
# CUDA reconciliation (unchanged from v3/v4)
# RIFE-TensorRT-Auto's deps pull cuda-python 13.x, which swaps cuda-bindings to
# 13.x and breaks the image's cu12 torch (pins cuda-bindings==12.9.x). Cap it.
# If torch ever moves to a CUDA-13 build, remove this line.
# ---------------------------------------------------------------------------
echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE (v5)
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
        echo "[mod
