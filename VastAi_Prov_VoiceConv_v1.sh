#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# VOICE CONVERSION + TTS  ---  v1
# (same infrastructure as VastAi_Prov_LTX23_*.sh; safe to run on the same box)
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Optional env vars:
#        HF_TOKEN=<token>        authenticated HF transfers, avoids 429s
#        CIVITAI_TOKEN=<token>   only if you add Civitai entries below
#   4. (Re)start the instance.
#
# -----------------------------------------------------------------------------
# WHAT THIS INSTALLS AND WHY
#
# One node pack does all of it: diodiogod/TTS-Audio-Suite. It is a unified
# multi-engine TTS + voice-conversion suite with a single "Voice Changer" node
# that is engine-agnostic, so you can A/B conversion engines without rewiring.
#
# TWO CONVERSION ENGINES, AND THE DIFFERENCE MATTERS FOR YOUR ACCENT PROBLEM:
#
#   RVC       Loads a .pth model TRAINED ON ONE TARGET VOICE. It swaps timbre
#             over the pronunciation and prosody of YOUR source recording.
#             There is no generative language prior deciding how words should
#             sound, so there is no accent to leak in. This is the structural
#             fix, not a workaround. Cost: you need a .pth per voice -- either
#             a community model or ~10 min of clean target audio to train one
#             (the suite has an integrated trainer; see bottom of this file).
#
#   ChatterBox VC   Zero-shot, like seed-vc: give it source + target audio, no
#             training. From Resemble AI and English-first, so it is a
#             reasonable zero-shot fallback when you have no .pth. Supports
#             iterative refinement passes (1-30, 1-5 useful) that push the
#             output progressively closer to the target voice.
#
#   Rule of thumb: RVC when you will reuse a voice. ChatterBox VC for one-offs.
#
# NOT INSTALLED ON PURPOSE: CosyVoice3 and IndexTTS-2 also do voice conversion
# and are good models, but both are Alibaba/Bilibili-trained and Chinese-first,
# which is the same data-prior situation you are trying to get away from. They
# auto-download if you ever select them -- nothing here blocks that.
#
# -----------------------------------------------------------------------------
# ON THE ACCENT BIAS YOU HIT
#
# The mechanism is real but worth stating precisely: in zero-shot VC, the
# reference clip supplies timbre, while the model's own decoder still decides
# pronunciation. Accent lives largely in pronunciation, not timbre -- so a
# model whose training data skews toward one accent family will pull output
# that way regardless of your reference. Seed-VC is trained substantially on
# Mandarin + English data, so your read is plausible.
#
# Before blaming the model entirely, though, these produce the same symptom:
#   * Reference clip too short or noisy -> speaker embedding underspecified,
#     model falls back on its prior. Use 10-30 s of clean, single-speaker audio.
#   * Too few diffusion steps -> under-converged toward the reference.
#   * Wrong checkpoint for the job (e.g. the singing/f0 model on speech).
# Worth one control test on seed-vc with a long clean reference before you
# write it off. But if you want the problem to structurally not exist: RVC.
#
# -----------------------------------------------------------------------------
# DOWNLOAD BUDGET
#
#     RVC base: hubert_base.pt + rmvpe.pt + content-vec-best     ~700 MB
#     RVC pretrained_v2 (training init, 40k pair only)           ~200 MB
#     ChatterBox English model set                                ~2  GB
#     UVR vocal-separation weights (source cleanup)               ~120 MB
#                                                              ----------
#                                                                ~3  GB
#
#   Trivial next to the LTX checkpoints. Everything else in the suite
#   (F5-TTS, VibeVoice, Higgs, Qwen3...) auto-downloads only if you select it.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (VoiceConv v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

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
ensure_pkg aria2c aria2
ensure_pkg ffmpeg ffmpeg     # the suite uses it for high-quality time-stretch;
                             # falls back to a phase vocoder without it

# ---------------------------------------------------------------------------
# System libs the suite's installer explicitly asks for on Linux.
# libsamplerate0-dev -> resampy / soxr resampling (REQUIRED in practice)
# portaudio19-dev    -> mic capture nodes (only if you record in-browser)
# Not command-line tools, so ensure_pkg's `command -v` test can't see them.
# ---------------------------------------------------------------------------
echo "[provisioning] installing audio system libraries"
apt-get update -qq && apt-get install -y -qq libsamplerate0-dev portaudio19-dev \
    || echo "[provisioning] WARNING: audio system libs failed -- resampy/recording may break"

# ---------------------------------------------------------------------------
# Custom nodes
# ---------------------------------------------------------------------------
NODES=(
    #   The whole toolkit: RVC + ChatterBox VC + ~13 TTS engines + vocal
    #   removal + RVC training. Has its own install.py that resolves some
    #   genuinely nasty dependency conflicts (numpy/librosa/s3tokenizer) --
    #   install_node() runs it automatically below.
    "https://github.com/diodiogod/TTS-Audio-Suite"
    #   Audio/video I/O, already present if you ran the LTX script.
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
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
    # install.py FIRST for this suite: it is the intelligent installer and
    # handles conflicts that a bare requirements.txt install would create.
    if [[ -f "${path}/install.py" ]]; then
        echo "[node] running install.py for $name (this one takes a while)"
        ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
    elif [[ -f "${path}/requirements.txt" ]]; then
        pip_install --no-cache-dir -r "${path}/requirements.txt" \
            || echo "[node] requirements.txt FAILED: $name"
    fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
"$PY" -m pip install --upgrade pip setuptools wheel \
    || echo "[provisioning] WARNING: pip/setuptools upgrade failed (s3tokenizer may fail to build)"
for n in "${NODES[@]}"; do install_node "$n"; done

# ---------------------------------------------------------------------------
# CUDA reconciliation -- carried over. Harmless if nothing pulled cuda-python.
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
# DOWNLOAD INFRASTRUCTURE  (identical to the LTX scripts -- Xet-native)
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
        sys.stderr.write("usage: hf_get.py <repo_id> <repo_path> <dest_file> [repo_type]\n"); return 2
    repo, path, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    repo_type = sys.argv[4] if len(sys.argv) > 4 else "model"
    token = os.environ.get("HF_TOKEN") or None
    dest_dir = os.path.dirname(dest) or "."
    stage = os.path.join(dest_dir, ".hf_stage")
    os.makedirs(stage, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=path, local_dir=stage,
                          token=token, repo_type=repo_type)
    shutil.move(got, dest)
    print(dest)
    return 0

try:
    sys.exit(main())
except Exception:
    traceback.print_exc(); sys.exit(1)
PYEOF

dl_hf() {
    # dl_hf <dest_dir> <dest_name> <repo_id> <repo_path> [repo_type]
    local dir="$1" name="$2" repo="$3" rpath="$4" rtype="${5:-model}"
    local dest="${dir}/${name}"
    mkdir -p "$dir"

    local check_url want have=0
    if [[ "$rtype" == "model" ]]; then
        check_url="$(hf_resolve_url "$repo" "$rpath")"
        want="$(remote_size "$check_url")"
    fi
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
    if "$PY" "$HF_GET" "$repo" "$rpath" "$dest" "$rtype"; then
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
# CIVITAI  (RVC .pth voice models are sometimes distributed there)
# ===========================================================================
dl_civitai() {
    local dir="$1" name="$2" vid="$3"
    local dest="${dir}/${name}"
    mkdir -p "$dir"
    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        echo "[civitai] $name already present, skipping"; return 0
    fi
    if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
        echo "[civitai] SKIP $name -- CIVITAI_TOKEN not set"; return 0
    fi
    local url="https://civitai.com/api/download/models/${vid}?token=${CIVITAI_TOKEN}"
    echo "[civitai] downloading $name (version ${vid})"
    aria2c -x 16 -s 16 -k 1M --file-allocation=none --summary-interval=10 \
           --continue=true --auto-file-renaming=false \
           --max-tries=5 --retry-wait=5 --connect-timeout=30 --timeout=600 \
           --max-file-not-found=2 --allow-overwrite=true \
           -d "$dir" -o "$name" "$url" \
        || { echo "[civitai] DOWNLOAD FAILED: $name"; return 0; }
    local sz; sz="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if (( sz < 102400 )); then
        echo "[civitai] WARNING: $name is only ${sz} bytes -- probably a login/error page. Removing."
        head -c 200 "$dest" 2>/dev/null | tr -d '\0'; echo
        rm -f "$dest"
    else
        echo "[civitai] $name OK (${sz} bytes)"
    fi
}

# ---------------------------------------------------------------------------
# Model manifest
#   hf | dest_dir | dest_filename | repo_id | repo_path | [repo_type]
#
# NOTE: this suite auto-downloads everything on first use. Pre-fetching here
# just means your first run isn't a 10-minute stall, and that a headless box
# has the files even if a download endpoint is flaky later.
# ---------------------------------------------------------------------------
RVC="${COMFY}/models/TTS/RVC"
UVR="${COMFY}/models/TTS/UVR"

MODELS=(
    # --- RVC inference base models ---------------------------------------
    # HuBERT content encoder: extracts WHAT is being said from your source,
    # stripped of speaker identity. This is the piece that makes RVC preserve
    # your source pronunciation instead of re-synthesising it.
    "hf|$RVC|hubert_base.pt|lj1995/VoiceConversionWebUI|hubert_base.pt"
    # RMVPE pitch extraction -- the good f0 estimator; use it over pm/harvest.
    "hf|$RVC|rmvpe.pt|lj1995/VoiceConversionWebUI|rmvpe.pt"
    # ContentVec, the suite's listed voice-feature model.
    "hf|$RVC|content-vec-best.safetensors|lengyue233/content-vec-best|pytorch_model.bin"

    # --- RVC training init checkpoints (40k pair) -------------------------
    # Only needed if you TRAIN your own voice. 40k is the standard sample rate
    # for speech; add the 32k/48k pairs if you pick those in the trainer.
    "hf|$RVC/pretrained_v2|f0G40k.pth|lj1995/VoiceConversionWebUI|pretrained_v2/f0G40k.pth"
    "hf|$RVC/pretrained_v2|f0D40k.pth|lj1995/VoiceConversionWebUI|pretrained_v2/f0D40k.pth"
    # "hf|$RVC/pretrained_v2|f0G48k.pth|lj1995/VoiceConversionWebUI|pretrained_v2/f0G48k.pth"
    # "hf|$RVC/pretrained_v2|f0D48k.pth|lj1995/VoiceConversionWebUI|pretrained_v2/f0D48k.pth"

    # --- UVR vocal separation --------------------------------------------
    # Clean the SOURCE before conversion. Music/noise under the voice degrades
    # VC badly -- same principle as the MelBand step in the LTX workflow.
    "hf|$UVR|HP5-vocals+instrumentals.pth|lj1995/VoiceConversionWebUI|uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth"

    # --- Example RVC character models (optional) --------------------------
    # The suite auto-downloads a default character pack (Claire, Sayano,
    # Mae_v2, Fuji, Monika) from the SayanoAI/RVC-Studio DATASET repo on first
    # use. Uncomment to pre-fetch one as a smoke test -- note repo_type=dataset.
    # "hf|$RVC|Claire.pth|SayanoAI/RVC-Studio|RVC/Claire.pth|dataset"
)

# ---------------------------------------------------------------------------
# Your own RVC voice models
#   HF format:      "$RVC|myvoice.pth|<repo_id>|<path/in/repo>"
#   Civitai format: dest_dir|dest_filename|version_id  (in CIVITAI_MODELS)
# Drop matching .index files in the same folder; RVC picks them up
# automatically and they measurably improve similarity.
# ---------------------------------------------------------------------------
EXTRA_RVC_HF=(
    # "$RVC|my_narrator.pth|someuser/some-rvc-repo|my_narrator.pth"
)
CIVITAI_MODELS=(
    # "$RVC|my_voice.pth|1234567"
)

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
echo "=================== MODELS ==================="
mkdir -p "$RVC" "$UVR" "$RVC/pretrained_v2"
avail="$(df -PB1 "$RVC" | awk 'NR==2{print $4}')"
echo "[provisioning] free on models FS: $(numfmt --to=iec "$avail" 2>/dev/null || echo "$avail B")"
if (( avail < 10 * 1024*1024*1024 )); then
    echo "[provisioning] WARNING: <10 GiB free. Audio models are small, but the"
    echo "[provisioning] WARNING: suite auto-downloads multi-GB TTS engines on demand."
fi

for entry in "${MODELS[@]}"; do
    IFS='|' read -r kind a b c d e <<< "$entry"
    case "$kind" in
        hf) dl_hf "$a" "$b" "$c" "$d" "${e:-model}" ;;
        *)  echo "[model] unknown manifest kind: '$kind' in: $entry" ;;
    esac
done

for entry in "${EXTRA_RVC_HF[@]}"; do
    IFS='|' read -r a b c d <<< "$entry"
    dl_hf "$a" "$b" "$c" "$d"
done

echo "=================== CIVITAI ==================="
if (( ${#CIVITAI_MODELS[@]} == 0 )); then
    echo "[civitai] none configured"
else
    for entry in "${CIVITAI_MODELS[@]}"; do
        IFS='|' read -r cdir cname cvid <<< "$entry"
        dl_civitai "$cdir" "$cname" "$cvid"
    done
fi

# ---------------------------------------------------------------------------
# Layout check
# ---------------------------------------------------------------------------
echo "=================== LAYOUT CHECK ==================="
for p in "${RVC}/hubert_base.pt" \
         "${RVC}/rmvpe.pt" \
         "${RVC}/content-vec-best.safetensors" \
         "${RVC}/pretrained_v2/f0G40k.pth" \
         "${RVC}/pretrained_v2/f0D40k.pth" \
         "${UVR}/HP5-vocals+instrumentals.pth"; do
    [[ -f "$p" ]] && echo "[layout] OK      ${p#${COMFY}/}" || echo "[layout] MISSING ${p#${COMFY}/}"
done
echo "[layout] RVC .pth voice models found: $(ls -1 "${RVC}"/*.pth 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# Usage crib
# ---------------------------------------------------------------------------
cat <<'CRIB'
=================== HOW TO USE ===================
[vc] Example workflows ship INSIDE the node pack. Load this one first:
[vc]   ComfyUI/custom_nodes/TTS-Audio-Suite/example_workflows/
[vc]   "Unified <VoiceChanger> Voice Changer - RVC X ChatterBox.json"
[vc] It has both engines pre-wired so you can A/B them on one source clip.
[vc] (Filename has an emoji in it -- use tab-completion, or open the folder
[vc]  in the ComfyUI workflow browser.)
[vc]
[vc] RVC path (recommended for repeat voices):
[vc]   1. Put a .pth in models/TTS/RVC/  (+ matching .index if you have one)
[vc]   2. Node: "Load RVC Character Model" -> pick the .pth
[vc]   3. Node: "Voice Changer" -> engine = RVC -> connect source audio
[vc]   4. Set f0 method to rmvpe. Start refinement passes at 1.
[vc]
[vc] ChatterBox path (zero-shot, no .pth needed):
[vc]   Voice Changer -> engine = ChatterBox -> source audio + target audio.
[vc]   Refinement passes 1-5; results are cached, so changing 5->3->4 is
[vc]   instant. More passes = closer to target but more artefacts.
[vc]
[vc] TRAINING YOUR OWN RVC VOICE (~10 min of clean target audio):
[vc]   RVC Engine -> RVC Dataset Prep -> RVC Training Config -> Model Training
[vc]   Example workflow: "RVC <grad> Model Training.json" in the same folder.
[vc]   Checkpoints/logs: ComfyUI/output/tts_audio_suite_training/rvc/
[vc]   Finished .pth + .index land in models/TTS/RVC/ automatically.
[vc]   Note: save_best_model picks the lowest-loss candidate, which is not
[vc]   the same as the best-sounding one. Listen to a few checkpoints.
[vc]
[vc] FEEDING THE RESULT INTO LTX-2.3:
[vc]   Voice Changer output -> Save Audio -> use that file as the LoadAudio
[vc]   input in the IA2V workflow. Or wire it directly if both node packs are
[vc]   installed on the same instance. Trim to whole seconds first.
[vc]
[vc] SOURCE QUALITY: run noisy/musical source through the vocal-removal node
[vc] before conversion. Garbage in stays garbage in, for VC as much as LTX.
CRIB

echo "=================== PROVISIONING COMPLETE ==================="
