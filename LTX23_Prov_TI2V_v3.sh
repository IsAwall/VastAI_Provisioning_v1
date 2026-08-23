#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# LTX-2.3 (Lightricks, 22B audio-video DiT) -- T2V / I2V / FLF2V BUILD  ---  v1
#
# Derived from MinimaxH3_Prov_TI2V_v7.sh. All of the update, torch-pinning,
# flash-attn, disk-preflight and finalise machinery is carried over unchanged
# in behaviour. What moved is the manifest, the version gate, the node list,
# and every operating note -- because LTX-2.3 is laid out fundamentally
# differently from H3. Read WHAT IS STRUCTURALLY DIFFERENT below before you
# assume anything transfers.
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          STRONGLY recommended -- see GATED REPOS
#        CIVITAI_TOKEN=<token>     only if you add Civitai entries below
#        COMFY_UPDATE=0            skip the ComfyUI git update (default: ON)
#        NODE_UPDATE=0             skip the custom-node git updates (default: ON)
#        GIT_FORCE_RESET=1         discard local commits blocking a fast-forward
#        WANT_LORA_384=0           drop the rank-384 distilled LoRA (-7.6 GB)
#        WANT_DISTILLED_CKPT=1     add the distilled 1.1 checkpoint (+46.1 GB)
#        WANT_SPLIT=1              add Kijai's split component files (+46 GB)
#        WANT_TEMPORAL=0           drop the temporal upscaler (-262 MB)
#        WANT_TAE=0                drop the tiny preview VAE (-24 MB)
#        WANT_LTXV_NODES=1         add Lightricks/ComfyUI-LTXVideo (see NODES)
#        WANT_SEEDVR2=1            add SeedVR2 restore weights (+~15 GB)
#        PURGE_H3=1                delete MiniMax H3 weights to reclaim disk
#        PURGE_LEGACY_SEEDVR2=1    delete the v3-era third-party pack + weights
#        FIX_FLASH_ATTN=0          skip the half-installed flash-attn guard
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY)
#        COMFY_PIN=<sha|tag>       hold core at this ref
#        ALLOW_BRANCH_RECOVERY=1   permit moving a detached HEAD onto a branch
#        FORCE_DEPS=1              reinstall pip requirements even if unchanged
#        RESTART_COMFY_ON_UPDATE=1 bounce the comfyui service if code changed
#        SKIP_PROBE=1              skip the pre-download manifest HEAD probe
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~84 GB of weights with the defaults below. Provision 130 GB minimum,
#   160 comfortable. Turning on WANT_DISTILLED_CKPT or WANT_SPLIT adds ~46 GB
#   EACH -- at that point you want 220 GB.
#
#   IF YOU ARE REUSING THE H3 VOLUME: 131 GB of H3 weights are still on it and
#   LTX will not fit alongside them. The disk pre-flight detects this and tells
#   you exactly what to set. PURGE_H3=1 removes them.
#
# -----------------------------------------------------------------------------
# WHAT IS STRUCTURALLY DIFFERENT FROM THE H3 BUILD
#
#   Four things, and every one of them will bite you if you carry an H3 habit
#   across.
#
#   1. LTX-2.3 SHIPS AS AN ALL-IN-ONE CHECKPOINT, NOT SPLIT COMPONENTS.
#      ltx-2.3-22b-dev.safetensors contains the transformer, the video VAE,
#      the audio VAE, AND the text projection/connector. It loads through
#      CheckpointLoaderSimple into models/checkpoints/ -- NOT UNETLoader into
#      diffusion_models/. There are no separate VAE files to fetch.
#
#      Confirm this by looking at what the stock template actually wires:
#        CheckpointLoaderSimple  ckpt_name = ltx-2.3-22b-dev-fp8.safetensors
#        LTXVAudioVAELoader      ckpt_name = <the same checkpoint>
#        LTXAVTextEncoderLoader  ckpt_name = <the same checkpoint>
#                                text_encoder = gemma_3_12B_it_*.safetensors
#      Three loaders, one checkpoint. The audio VAE and the text projection
#      are read out of it. The ONLY separate weight you must supply is Gemma.
#
#      (The stock subgraph proxies all three ckpt_name widgets to a single
#      input, so in the template you change it once. If you rebuild the graph
#      flat you must change it in all three places or you get a confusing
#      partial load.)
#
#   2. THE TEXT ENCODER IS EXTERNAL AND IS NOT PART OF THE LTX REPO.
#      Gemma 3 12B IT lives in Comfy-Org/ltx-2 (note: ltx-2, not ltx-2.3 --
#      the encoder did not change between releases). Full bf16 is 24.4 GB.
#      The template ships pointing at the 9.45 GB fp4_mixed build, so after
#      this script runs the dropdown will NOT match what you downloaded.
#      Repoint it. This is the single most common "why is my node red".
#
#   3. THE STOCK TEMPLATE IS A DISTILLED 8+3 STEP GRAPH, NOT A BASE-MODEL
#      GRAPH. It runs the dev checkpoint with the distilled LoRA at strength
#      0.5, euler, CFG 1, and a hand-written ManualSigmas list. You are not
#      looking at "the base model with a speedup bolted on" -- you are looking
#      at a distilled schedule that happens to load base weights. See THE
#      LADDER for what the dial actually is.
#
#   4. THERE IS NO SHIFT CALIBRATION HERE. H3's coupled video/audio sigma
#      shifts (12/3) have no analogue. LTX-2.3 uses explicit sigma LISTS via
#      ManualSigmas. Do not import shift reasoning; there is nothing to shift.
#
# -----------------------------------------------------------------------------
# GATED REPOS -- THE FAILURE MODE THAT LOOKS LIKE A BROKEN SCRIPT
#
#   Lightricks gates its weight repos behind license acceptance. A gated repo
#   with no token, or with a token belonging to an account that has not
#   accepted the license, returns 401/403 on the resolve URL. Left unhandled
#   that surfaces as a generic "DOWNLOAD FAILED" on every boot forever, with
#   nothing in the log distinguishing it from a network blip.
#
#   v1 probes every manifest entry with a HEAD before downloading anything and
#   classifies the result:
#       200      fine
#       401/403  gated or wrong token -> open the repo page, accept the
#                license WITH THE SAME ACCOUNT the token belongs to
#       404/410  the file moved or was renamed -> the manifest is stale, and
#                no amount of retrying will fix it
#       000/5xx  transient, will retry next boot
#
#   This is the same class of bug that ate the Kijai H3 LoRA when it moved
#   into a loras/ subdirectory: a 404 on every boot that read like a network
#   problem. Set SKIP_PROBE=1 if you want the old behaviour.
#
# -----------------------------------------------------------------------------
# MANIFEST FORMAT -- AND A REAL BUG IN THE H3 SCRIPT
#
#   Entries are FIVE pipe-separated fields:
#       hf|<dest_dir>|<dest_filename>|<repo_id>|<path_within_repo>
#
#   The H3 script's WANT_TURBO block has a FOUR-field entry:
#       "hf|$LORA|minimax_..._bf16.safetensors|loras/minimax_..._bf16.safetensors"
#   `IFS='|' read -r kind a b c d` then binds c="loras/..." and d="", so it
#   requests repo_id "loras/<filename>" with an empty path. That is a
#   guaranteed 404 that reads like a moved file. v1 validates the field count
#   on every entry and says so out loud rather than failing downstream.
#
# -----------------------------------------------------------------------------
# THE UPDATE PATH -- unchanged from the H3 script, still the thing that breaks
#
#   1. DUBIOUS OWNERSHIP. ai-dock runs provisioning as root; a persistent
#      volume is often owned by another uid; modern git then refuses EVERY
#      operation with "detected dubious ownership" and git_sync's error paths
#      absorb it individually, so updates degrade to silent no-ops while the
#      log still looks plausible. safe.directory is set before the first git
#      call. If that line says WARNING, nothing in this run updated.
#
#   2. DETACHED HEAD. Images pin core to a release tag on purpose. A detached
#      HEAD with no explicit pin is LEFT ALONE and logged. Pass COMFY_PIN, or
#      ALLOW_BRANCH_RECOVERY=1, to move it deliberately.
#
#   3. SHALLOW CLONE. --depth=1 has no tag history, so version detection
#      returns nothing and a fast-forward has no merge base. Unshallow first.
#
#   4. TORCH GETTING CLOBBERED. Any requirements.txt can resolve a fresh torch
#      off PyPI over a pinned cu128 Blackwell build. Every pip call that
#      touches a requirements file runs against a constraints file pinning the
#      installed torch stack to its exact current versions, +cuXXX suffix
#      included. Failure there is the correct outcome, and it is reported.
#
#   FRONTEND/BACKEND SKEW. comfyui-frontend-package, comfyui-workflow-templates
#   and comfyui-embedded-docs are pinned by ComfyUI's requirements.txt. Skew
#   produces phantom link errors on autogrow sockets and stale templates. All
#   three are force-synced after a core update. This matters more here than on
#   H3: the LTX-2.3 templates ARE the workflow-templates package, so a stale
#   pin means you are reading last quarter's graph.
#
# -----------------------------------------------------------------------------
# WHERE THE QUALITY IS, IN THIS BUILD (ranked, largest lever first)
#
#   1. THE DISTILLED-LoRA STRENGTH, AND WHICH DISTILLED LoRA. This is the
#      whole ballgame and it is not a speed/quality toggle in the usual sense.
#      Lightricks' own two-stage HQ pipeline runs the DEV checkpoint with the
#      distilled LoRA at 0.8; the ComfyUI template runs it at 0.5. So the
#      real dial is LoRA strength on a dev checkpoint, not "distilled vs not".
#      0.5 is fast, 0.8 is the published HQ setting. Sweep between them before
#      you touch anything else.
#
#   2. THE SECOND STAGE. The template generates at HALF the requested canvas,
#      runs LTXVLatentUpsampler with the x2 spatial upscaler IN LATENT SPACE,
#      then refines with a short 3-sigma pass. That is a genuinely better
#      position than any post-decode restorer, because the upscaler sees
#      latents and the refine pass re-samples them. Do not disable stage two
#      to "save time" -- you are removing most of the resolution.
#
#   3. TEXT ENCODER PRECISION. Bigger lever here than on H3, because Gemma 3
#      12B is much smaller than Qwen3-VL 32B and quantization bites sooner.
#      Full bf16 (24.4 GB) is the default in this manifest. fp8_scaled is
#      13.2 GB, fp4_mixed is 9.45 GB and is what the template ships pointing
#      at. On 96 GB there is no reason to run either.
#
#   4. CHECKPOINT PRECISION. bf16 (46.1 GB) over fp8 (~25 GB). Real, but on
#      an already-distilled schedule it is a smaller effect than 1 or 2.
#
#   5. RESOLUTION AND DURATION. Native up to 4K / 50 fps, but the template
#      default is 1280x720 at 25 fps for 5 s. Unlike H3 there is no hard 768
#      floor and no off-distribution cliff -- LTX degrades gracefully, so
#      drafting at reduced resolution is legitimate here in a way it never
#      was on H3. That is a real habit to unlearn.
#
# -----------------------------------------------------------------------------
# THE LADDER
#
#   DRAFT  -> dev bf16 + distilled LoRA 0.5, 8+3 steps, half canvas, Sage on.
#             Cut the canvas freely. LTX has no 768 floor and no RoPE cliff;
#             composition holds at reduced resolution, which is the opposite
#             of the H3 rule. Frame count is also free to cut.
#   FINAL  -> dev bf16 + distilled LoRA 0.8, full gemma bf16, two-stage with
#             the x2 spatial upscaler, no Sage, audio described in the prompt.
#   INTERP -> optional. LTX generates natively up to 50 fps, so ask for 50 at
#             the source before you reach for RIFE. RIFE remains useful going
#             past 50 or when you already have a 24/25 fps render.
#
#   SIGMA LISTS ARE NOT A DIAL YOU TURN CASUALLY. The template's two
#   ManualSigmas lists were tuned against the distilled LoRA at 0.5:
#     stage 1 (8 steps): 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375,
#                        0.725, 0.421875, 0.0
#     stage 2 (3 steps): 0.85, 0.7250, 0.4219, 0.0
#   Note the shape of stage 1: four near-zero decrements at the top of the
#   schedule, then three large jumps. That front-loaded plateau is doing
#   structural work early and is exactly what distillation memorised. If you
#   change LoRA or strength, re-derive the schedule or use the one the LoRA
#   ships with -- do not carry these across and then blame the LoRA. Same
#   lesson as the H3 turbo tier, different mechanism.
#
#   CAVEAT ON SEED TRANSFER: unchanged from H3. Changing precision, sigma
#   list, or LoRA strength changes the integration trajectory, so a locked
#   seed lands somewhere related but not identical. Use drafts to settle
#   PROMPT and COMPOSITION, then re-roll at final settings.
#
# -----------------------------------------------------------------------------
# LICENSE -- DIFFERENT SHAPE FROM H3, READ IT BEFORE YOU BUILD ON IT
#
#   LTX-2.x Community License Agreement. The gate is REVENUE, not territory:
#   entities with annual revenue of at least $10,000,000 (aggregated across
#   affiliates under common control) must obtain a paid license for any use.
#   Under that, self-hosting, fine-tuning and commercial use are permitted.
#   There is no "Applicable Territory" exclusion list of the kind H3 has, so
#   the US-datacenter question that constrains your H3 host does not apply
#   the same way here.
#
#   TWO THINGS THAT DO BIND YOU AND ARE EASY TO MISS:
#     - Attachment A is an enforceable part of the agreement and lists
#       prohibited uses. It is not boilerplate. If you are adding LoRAs to
#       the manifest below, read it first and decide deliberately -- accepting
#       the HF gate is accepting Attachment A.
#     - Attachment A item 20 is a non-compete: using LTX-2.x or derivatives in
#       a product that competes with, replaces, or substitutes Lightricks'
#       commercial offerings needs a separate commercial license.
#   Derivatives (including LoRAs you train) inherit the terms, and if you pass
#   them to a Commercial Entity that entity needs its own paid license.
#
#   I am not a lawyer and this is a summary, not advice. The authoritative
#   text is the LICENSE file in Lightricks/LTX-2.3 and in the LTX-2 repo.
#
# -----------------------------------------------------------------------------
# MEMORY -- THE GOOD NEWS RELATIVE TO H3
#
#   dev bf16 (46.1) + gemma bf16 (24.4) = 70.5 GB against 96 GB of VRAM. That
#   fits RESIDENT. There is no eviction dance, no encoder-runs-first-then-gets-
#   swapped-out choreography, and therefore no hard 128 GB system-RAM floor.
#   64 GB RAM is workable, 128 comfortable. This build is much less fragile
#   than the H3 one on exactly the axis that made H3 fragile.
#
#   The pinned-memory regression note still applies to whatever ComfyUI you
#   land on: if model loading is pathologically slow, launch with
#   --disable-pinned-memory.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (LTX-2.3 v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

# ---------------------------------------------------------------------------
# Env toggles. Every one of these reads from the environment -- no literals.
# ---------------------------------------------------------------------------
COMFY_UPDATE="${COMFY_UPDATE:-1}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_LORA_384="${WANT_LORA_384:-1}"
WANT_DISTILLED_CKPT="${WANT_DISTILLED_CKPT:-0}"
WANT_SPLIT="${WANT_SPLIT:-0}"
WANT_TEMPORAL="${WANT_TEMPORAL:-1}"
WANT_TAE="${WANT_TAE:-1}"
WANT_LTXV_NODES="${WANT_LTXV_NODES:-0}"
WANT_SEEDVR2="${WANT_SEEDVR2:-0}"
ALLOW_BRANCH_RECOVERY="${ALLOW_BRANCH_RECOVERY:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-0}"
SKIP_PROBE="${SKIP_PROBE:-0}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} node_update=${NODE_UPDATE} lora384=${WANT_LORA_384} distilled_ckpt=${WANT_DISTILLED_CKPT}"
echo "[provisioning] split=${WANT_SPLIT} temporal=${WANT_TEMPORAL} tae=${WANT_TAE} ltxv_nodes=${WANT_LTXV_NODES} seedvr2=${WANT_SEEDVR2}"
echo "[provisioning] branch_recovery=${ALLOW_BRANCH_RECOVERY} force_deps=${FORCE_DEPS} restart=${RESTART_COMFY_ON_UPDATE} pin=${COMFY_PIN:-<none>}"

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
# safe.directory -- must run before the first git invocation. See item 1 in
# THE UPDATE PATH. If this warns, nothing else in this run is updating.
# ---------------------------------------------------------------------------
git config --global --add safe.directory '*' 2>/dev/null \
    && echo "[git] safe.directory configured (prevents 'dubious ownership' no-ops)" \
    || echo "[git] WARNING: could not set safe.directory -- git ops may fail silently"

# ---------------------------------------------------------------------------
# Broken flash-attn guard  --  RUNS BEFORE THE PIN FILE IS WRITTEN
#
#   <any custom node> -> diffusers -> xformers.ops -> flash_attn.flash_attn_interface
#   ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
#
# flash_attn is present enough that `import flash_attn` succeeds (often as a
# namespace package -- a directory with no __init__.py, the residue of a
# failed source build), so xformers concludes flash attention is available and
# dies on the submodule. Cleanly absent is FINE. The fix is removal.
#
# Must run before write_torch_pins(): a half-installed package still carries
# dist-info metadata, so it would otherwise get pinned into the constraints
# file and held in place for the whole run.
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
        ABSENT)
            echo "[flash] not installed -- fine, xformers falls back to its own kernels"
            return 0 ;;
        OK)
            echo "[flash] healthy (${detail:-?})"
            return 0 ;;
    esac

    (( rc == 7 )) || { echo "[flash] probe returned an unexpected state, leaving alone"; return 0; }

    echo "[flash] !!! HALF-INSTALLED flash_attn detected."
    echo "[flash] !!!   ${detail}"
    echo "[flash] !!! This breaks EVERY custom node that imports diffusers, not"
    echo "[flash] !!! just the one you noticed. Removing it."

    "$PY" -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true

    if [[ -n "$pkgdir" && "$pkgdir" == */flash_attn ]]; then
        echo "[flash] removing residue: ${pkgdir}"
        rm -rf "$pkgdir"
        rm -rf "${pkgdir%/*}"/flash_attn-*.dist-info "${pkgdir%/*}"/flash_attn-*.egg-info
    fi

    if "$PY" -c "import xformers.ops" 2>/dev/null; then
        echo "[flash] xformers imports cleanly again"
    else
        echo "[flash] xformers still broken -> removing it too (nothing here needs it)"
        "$PY" -m pip uninstall -y xformers >/dev/null 2>&1 || true
    fi

    "$PY" - <<'PYEOF'
import importlib.util as u
if u.find_spec("diffusers") is None:
    print("[flash] diffusers not installed on this venv -- nothing further to verify")
else:
    try:
        import diffusers.models.embeddings          # noqa: F401
        print("[flash] diffusers imports cleanly -- fixed")
    except Exception as e:
        print("[flash] WARNING: diffusers STILL fails to import: %s" % e)
        print("[flash] WARNING: read the full traceback in the ComfyUI log; the")
        print("[flash] WARNING: root cause is something other than flash-attn.")
PYEOF
}

# ---------------------------------------------------------------------------
# Torch stack guard. Snapshot the exact installed versions of the fragile CUDA
# cluster into a pip constraints file, including local +cuXXX suffixes. A
# constrained install that FAILS is the correct outcome: it means something
# genuinely wanted to move torch, and you want to know that.
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
    # Do not pin a package whose metadata exists but whose module is gone:
    # that is the half-installed state fix_flash_attn() cleans up, and pinning
    # it would hold the breakage in place for the whole run.
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
        echo "[pins] torch stack pinned for this run:"
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
        if pip_install --no-cache-dir -c "$CONSTRAINTS" -r "$req"; then
            return 0
        fi
        echo "[pip] ${label}: constrained install failed -- something wants to move torch."
        echo "[pip] ${label}: retrying UNCONSTRAINED, will verify CUDA afterwards."
    fi
    pip_install --no-cache-dir -r "$req" || { echo "[pip] ${label}: requirements FAILED"; return 1; }
}

# reqs_changed <repo_path> <requirements_file>
# Returns 0 when the file is new or its content hash moved. Without this every
# boot spends minutes re-resolving dependencies and gives itself another
# chance to break a working environment.
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
        print("[torch] !!! CUDA UNAVAILABLE. If this worked before an update, the")
        print("[torch] !!! torch wheel was replaced. Reinstall the pinned build for")
        print("[torch] !!! your CUDA line before rendering anything.")
except Exception as e:
    print("[torch] not importable: %s" % e)
PYEOF
}

echo "=================== FLASH-ATTN GUARD ==================="
if [[ "${FIX_FLASH_ATTN:-1}" == "1" ]]; then
    fix_flash_attn
else
    echo "[flash] FIX_FLASH_ATTN=0 -> skipping (import failures in diffusers-based"
    echo "[flash] packs are on you)"
fi

echo "=================== TORCH PINS ==================="
write_torch_pins

# ---------------------------------------------------------------------------
# Host memory check. Much less load-bearing than on the H3 build -- see the
# MEMORY note in the header. 70.5 GB of weights fit resident on 96 GB VRAM.
# ---------------------------------------------------------------------------
echo "=================== HOST MEMORY ==================="
RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$RAM_GB" ]]; then
    echo "[mem] system RAM: ${RAM_GB} GB"
    if (( RAM_GB < 48 )); then
        echo "[mem] !!! Under 48 GB. bf16 checkpoint loading stages through host RAM;"
        echo "[mem] !!! at 46 GB for the checkpoint alone this will swap or OOM during"
        echo "[mem] !!! load even though the model fits in VRAM once resident."
    elif (( RAM_GB < 128 )); then
        echo "[mem] adequate. Unlike the H3 build there is no eviction dance here --"
        echo "[mem] adequate. 70.5 GB of weights stay resident on a 96 GB card."
    else
        echo "[mem] comfortable."
    fi
fi

# ---------------------------------------------------------------------------
# git_sync <repo_path> <label> [pin]
# Return codes:  0 = HEAD moved   1 = could not update   2 = already current
#
# A detached HEAD is NOT "repaired" by default -- images pin core to a release
# tag on purpose. Pass a pin to move it, or ALLOW_BRANCH_RECOVERY=1.
# ---------------------------------------------------------------------------
git_sync() {
    local path="$1" label="$2" pin="${3:-}" branch def c before after
    [[ -d "${path}/.git" ]] || { echo "[git] ${label}: not a git checkout, skipping update"; return 1; }

    before="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"

    if [[ "$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null)" == "true" ]]; then
        echo "[git] ${label}: shallow clone -> unshallowing (needed for tags and merge base)"
        git -C "$path" fetch --unshallow --tags --prune 2>/dev/null \
            || git -C "$path" fetch --depth=2147483647 --tags --prune 2>/dev/null \
            || echo "[git] ${label}: unshallow failed, continuing anyway"
    fi

    git -C "$path" fetch --all --tags --prune || { echo "[git] ${label}: fetch FAILED"; return 1; }

    # --- explicit pin wins over everything ---
    if [[ -n "$pin" ]]; then
        if git -C "$path" checkout --quiet "$pin" 2>/dev/null; then
            echo "[git] ${label}: pinned at ${pin}"
        else
            echo "[git] ${label}: PIN NOT FOUND (${pin}) -- leaving HEAD as is"
            return 1
        fi
        after="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"
        [[ "$before" == "$after" ]] && { echo "[git] ${label}: already at pin (${after})"; return 2; }
        echo "[git] ${label}: ${before} -> ${after} (pinned)"
        return 0
    fi

    branch="$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -z "$branch" ]]; then
        if [[ "$ALLOW_BRANCH_RECOVERY" != "1" ]]; then
            echo "[git] ${label}: detached HEAD -- looks image-pinned, leaving alone"
            echo "[git] ${label}: (pass a pin to move it, or ALLOW_BRANCH_RECOVERY=1)"
            return 2
        fi
        def="$(git -C "$path" remote show origin 2>/dev/null | awk '/HEAD branch/{print $NF}')"
        if [[ -z "$def" || "$def" == "(unknown)" ]]; then
            def="$(git -C "$path" rev-parse --abbrev-ref origin/HEAD 2>/dev/null | sed 's|^origin/||')"
        fi
        if [[ -z "$def" || "$def" == "(unknown)" ]]; then
            for c in master main; do
                git -C "$path" show-ref --verify --quiet "refs/remotes/origin/${c}" && { def="$c"; break; }
            done
        fi
        [[ -z "$def" || "$def" == "(unknown)" ]] && { echo "[git] ${label}: detached HEAD and no default branch resolvable"; return 1; }
        echo "[git] ${label}: detached HEAD -> checking out ${def} (ALLOW_BRANCH_RECOVERY=1)"
        git -C "$path" checkout -B "$def" "origin/${def}" || { echo "[git] ${label}: checkout FAILED"; return 1; }
        branch="$def"
    fi

    if ! git -C "$path" merge --ff-only "origin/${branch}" >/dev/null 2>&1; then
        echo "[git] ${label}: fast-forward blocked (local commits or a dirty tree)"
        if [[ "${GIT_FORCE_RESET:-0}" == "1" ]]; then
            echo "[git] ${label}: GIT_FORCE_RESET=1 -> hard reset to origin/${branch}"
            git -C "$path" reset --hard "origin/${branch}" || return 1
        else
            echo "[git] ${label}: keeping local state. Set GIT_FORCE_RESET=1 to discard it."
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
# ComfyUI update
#
# LTX-2.3 got day-0 NATIVE support -- Comfy Org's announcement points at
# 0.16.1. That is the hard floor: below it the LTXV/LTXAV node family does not
# exist and the template opens as a wall of red.
#
# The floor is not where you want to sit, though. Everything the current
# templates reference is newer than the launch set: the distilled 1.1 LoRA,
# the 1.1 spatial upscaler, and the LTXAV fix for sampling without an audio
# latent. Track the current release line.
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

# Force a pip package to exactly the version ComfyUI's requirements.txt pins.
# comfyui-workflow-templates matters more on this build than on H3: the
# LTX-2.3 graphs you are going to load ARE that package.
sync_pinned_pkg() {
    local pkg="$1" spec have want
    spec="$(grep -iE "^[[:space:]]*${pkg}[[:space:]]*==" "${COMFY}/requirements.txt" 2>/dev/null \
            | head -1 | tr -d ' \r')"
    if [[ -z "$spec" ]]; then
        echo "[deps] ${pkg}: no == pin in requirements.txt, leaving alone"
        return 0
    fi
    want="${spec##*==}"
    have="$("$PY" -c "import importlib.metadata as m; print(m.version('${pkg}'))" 2>/dev/null || echo "")"
    if [[ "$have" == "$want" ]]; then
        echo "[deps] ${pkg}: ${have} matches pin"
        return 0
    fi
    echo "[deps] ${pkg}: ${have:-<absent>} -> ${want}"
    pip_install --no-cache-dir "$spec" || echo "[deps] WARNING: failed to sync ${spec}"
}

echo "=================== COMFYUI UPDATE ==================="
if [[ -d "${COMFY}/.git" ]]; then
    _head="$(git -C "$COMFY" rev-parse --short HEAD 2>/dev/null)"
    _br="$(git -C "$COMFY" symbolic-ref --short -q HEAD 2>/dev/null || echo '(detached)')"
    _tag="$(git -C "$COMFY" describe --tags --abbrev=0 2>/dev/null)"
    _ahead="$(git -C "$COMFY" rev-list --count "${_tag}..HEAD" 2>/dev/null || echo '?')"
    echo "[comfy] git: ${_head} on ${_br}; nearest tag ${_tag:-none} (+${_ahead} commits)"
    if [[ "$_br" != "(detached)" && "$_ahead" != "0" && "$_ahead" != "?" && -n "$_tag" ]]; then
        echo "[comfy] NOTE: ${_ahead} commits past ${_tag}. Templates pin a release for a"
        echo "[comfy] NOTE: reason -- launch flags and custom nodes are tested against it."
        echo "[comfy] NOTE: To return:  git -C ${COMFY} checkout ${_tag} && \\"
        echo "[comfy] NOTE:             ${PY} -m pip install -r ${COMFY}/requirements.txt"
    fi
fi

echo "[comfy] before: $(comfy_version)"
if [[ "$COMFY_UPDATE" == "1" ]]; then
    git_sync "$COMFY" "ComfyUI" "${COMFY_PIN:-}"; rc=$?
    if (( rc == 0 )); then
        CHANGED_ANY=1
        echo "[comfy] source moved -> reinstalling requirements (torch pinned)"
        pip_reqs "${COMFY}/requirements.txt" "ComfyUI"
    elif (( rc == 2 )); then
        echo "[comfy] source unchanged -> verifying pinned packages only"
    else
        echo "[comfy] !!! update did not complete. See the [git] lines above."
    fi
    for p in comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs; do
        sync_pinned_pkg "$p"
    done
else
    echo "[comfy] COMFY_UPDATE=0 -> skipping git update"
fi

CV="$(comfy_version)"
MIN_CV="0.16.1"     # native LTX-2.3 support landed here (Comfy Org, day 0)
REC_CV="0.32.0"     # current line at time of writing; also carries LTX-2.5
echo "[comfy] after: ${CV:-<undetectable>}"
if [[ -z "$CV" ]]; then
    echo "[comfy] version undetectable -- verify manually that you are on >= ${MIN_CV}"
elif [[ "$(printf '%s\n%s\n' "$MIN_CV" "$CV" | sort -V | head -1)" != "$MIN_CV" ]]; then
    echo "[comfy] !!! ${CV} is BELOW ${MIN_CV} -- the LTXV/LTXAV nodes will not exist"
    echo "[comfy] !!! and the template will open as red nodes. The update above did"
    echo "[comfy] !!! not take. Check the [git] lines, or pick a newer base image."
elif [[ "$(printf '%s\n%s\n' "$REC_CV" "$CV" | sort -V | head -1)" != "$REC_CV" ]]; then
    echo "[comfy] OK -- native LTX-2.3 support present (${CV})."
    echo "[comfy] NOTE: below ${REC_CV}. You have the nodes, but the current templates"
    echo "[comfy] NOTE: reference the distilled 1.1 LoRA and the 1.1 spatial upscaler,"
    echo "[comfy] NOTE: and the LTXAV no-audio-latent crash fix is recent. If a graph"
    echo "[comfy] NOTE: references a widget your build does not have, this is why."
else
    echo "[comfy] OK -- ${CV} is current-line, everything in this manifest is supported"
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>][@<pin>]
#
# NOTE WHAT IS NOT HERE: the H3 packs. ComfyUI-MiniMaxH3-Contex-Loop,
# ComfyUI-MiniMax-H3-Turbo and MMH3Tools are all H3-specific and do nothing
# for LTX. If they are on a shared volume they will keep loading and keep
# occupying the node menu; that is harmless but noisy.
#
# NOTE ALSO: the LTX nodes are CORE. You do not need a custom pack to run any
# of the six shipped templates -- T2V, I2V, FLF2V, IA2V, IC-LoRA, ID-LoRA.
# ---------------------------------------------------------------------------
NODES=(
    # Video load/save, frame extraction, the VHS_* family.
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    # Utility layer: resolution selector, torch compile, graph plumbing, and
    # the latent preview path. MUST be current -- KJNodes needed an explicit
    # update to understand the LTX-2.3 audio VAE, and a stale copy is a known
    # cause of preview/VAE errors on 2.3 graphs specifically.
    "https://github.com/kijai/ComfyUI-KJNodes"
    # Interactive drag-to-crop on the image preview. Still the right tool for
    # framing an I2V source still. No pip dependencies.
    "https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop"
    # Frame interpolation. Demoted on this build -- LTX generates natively up
    # to 50 fps, so ask the model for the frame rate before interpolating.
    # Still useful past 50, or on an already-rendered 24/25 fps clip.
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
)

# Lightricks' own node pack. OPT-IN, and think before you turn it on.
# It is not required -- every shipped template runs on core nodes. What it
# adds is the example_workflows/2.3 collection and some loaders (the IC-LoRA
# reference-downscale loader among them). What it risks is exactly what the
# third H3 pack risked: node-name overlap with the core LTX nodes. Add it
# deliberately, after the stock path works.
[[ "$WANT_LTXV_NODES" == "1" ]] && NODES+=( "https://github.com/Lightricks/ComfyUI-LTXVideo" )

install_node() {
    local spec="$1" url name path pin rc fresh=0
    # optional @pin suffix, stripped before the |dirname split
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
            git_sync "$path" "$name" "$pin"; rc=$?
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

    # Only resolve pip when the requirements file actually moved.
    local req="${path}/requirements.txt"
    if [[ -f "$req" ]]; then
        if (( fresh )) || [[ "$FORCE_DEPS" == "1" ]] || reqs_changed "$path" "$req"; then
            pip_reqs "$req" "$name"
        else
            echo "[node] $name: requirements unchanged, skipping pip"
        fi
    fi

    if [[ -f "${path}/install.py" ]] && { (( fresh )) || [[ "$FORCE_DEPS" == "1" ]]; }; then
        ( cd "$path" && "$PY" install.py ) || echo "[node] install.py FAILED: $name"
    fi
}

echo "=================== CUSTOM NODES ==================="
mkdir -p "$NODES_DIR"
for n in "${NODES[@]}"; do install_node "$n"; done

# ---------------------------------------------------------------------------
# Legacy SeedVR2 pack from the v3-era H3 build
# ---------------------------------------------------------------------------
LEGACY_NODE="${NODES_DIR}/seedvr2_videoupscaler"
LEGACY_MODELS="${COMFY}/models/SEEDVR2"
if [[ -d "$LEGACY_NODE" || -d "$LEGACY_MODELS" ]]; then
    echo "=================== LEGACY SEEDVR2 ==================="
    if [[ "${PURGE_LEGACY_SEEDVR2:-0}" == "1" ]]; then
        [[ -d "$LEGACY_NODE"   ]] && { echo "[legacy] removing ${LEGACY_NODE}";   rm -rf "$LEGACY_NODE"; }
        [[ -d "$LEGACY_MODELS" ]] && { echo "[legacy] removing ${LEGACY_MODELS}"; rm -rf "$LEGACY_MODELS"; }
        echo "[legacy] done"
    else
        [[ -d "$LEGACY_NODE" ]] && \
            echo "[legacy] custom_nodes/seedvr2_videoupscaler present -- superseded by core nodes"
        [[ -d "$LEGACY_MODELS" ]] && \
            echo "[legacy] models/SEEDVR2 present ($(du -sh "$LEGACY_MODELS" 2>/dev/null | cut -f1)) -- unreadable by the native path"
        echo "[legacy] set PURGE_LEGACY_SEEDVR2=1 to delete both on the next boot."
    fi
fi

echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# SageAttention -- opt-in, and DRAFTS ONLY.
#
# Approximate attention: quantized QK^T with a smoothing correction. ~2x
# throughput, small but nonzero error. Fine while iterating prompts, bad on a
# final render. Recent ComfyUI takes --use-sage-attention as a launch flag; no
# KJNodes patch node needed.
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

echo "=================== TORCH VERIFY ==================="
verify_torch

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
else
    echo "[provisioning] !!! NO HF_TOKEN. Lightricks gates its weight repos behind"
    echo "[provisioning] !!! license acceptance -- without a token every Lightricks"
    echo "[provisioning] !!! entry below will return 401/403 and fetch nothing."
fi

map_url() {
    local u="$1"
    [[ -n "${HF_ENDPOINT:-}" ]] && u="${u/https:\/\/huggingface.co/${HF_ENDPOINT%/}}"
    printf '%s' "$u"
}
hf_resolve_url() { map_url "https://huggingface.co/${1}/resolve/main/${2}"; }

# http_probe <url>  ->  prints "<http_code>\t<bytes>"
# One HEAD, two answers. The status code is what turns "DOWNLOAD FAILED" into
# something actionable; the size is what makes resume-vs-refetch decidable.
http_probe() {
    local url; url="$(map_url "$1")"
    local out code size
    out="$(curl -sIL --connect-timeout 15 --max-time 60 -w '\n__CODE__%{http_code}' \
           "${CURL_AUTH[@]}" "$url" 2>/dev/null)" || { printf '000\t'; return 0; }
    code="$(printf '%s' "$out" | awk -F'__CODE__' '/__CODE__/{print $2}' | tail -1)"
    size="$(printf '%s' "$out" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-size"{v=$2} END{if(v!="")print v}')"
    [[ -z "$size" ]] && size="$(printf '%s' "$out" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{if(v!="")print v}')"
    printf '%s\t%s' "${code:-000}" "${size//[^0-9]/}"
}

remote_size() { http_probe "$1" | cut -f2; }

# classify_code <code> <label> -> prints a human verdict, returns 0 if fetchable
classify_code() {
    local code="$1" label="$2"
    case "$code" in
        200|206)
            return 0 ;;
        401|403)
            echo "[probe] GATED   ${label}  (HTTP ${code})"
            echo "[probe]   The repo requires license acceptance, or your HF_TOKEN"
            echo "[probe]   belongs to an account that has not accepted it. Open the"
            echo "[probe]   model page, accept, and make sure it is the SAME account"
            echo "[probe]   the token was issued from. Retrying will not help."
            return 1 ;;
        404|410)
            echo "[probe] MISSING ${label}  (HTTP ${code})"
            echo "[probe]   The file is not at that path. Either it was renamed or it"
            echo "[probe]   moved into a subdirectory. This manifest entry is stale --"
            echo "[probe]   fix the path, do not wait for it to come back."
            return 1 ;;
        000)
            echo "[probe] NO REPLY ${label} -- network or DNS. Transient, will retry."
            return 1 ;;
        *)
            echo "[probe] HTTP ${code} ${label} -- treating as transient."
            return 1 ;;
    esac
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

# Nested-duplicate detection.
#
# `hf download --local-dir` PRESERVES the repo's subdirectory prefix, so a
# manual pull of "loras/foo.safetensors" lands at <dir>/loras/foo.safetensors,
# while dl_hf writes flat at <dir>/foo.safetensors. Both then appear in the
# ComfyUI dropdown as different strings pointing at identical weights, and you
# can spend an afternoon A/B testing a file against itself.
dup_check() {
    local dir="$1" name="$2" hit
    hit="$(find "$dir" -mindepth 2 -name "$name" -print -quit 2>/dev/null)"
    if [[ -n "$hit" ]]; then
        echo "[dup] WARNING: ${name} also exists at ${hit}"
        echo "[dup] WARNING: that is the same weights under a second dropdown entry"
        echo "[dup] WARNING: (an 'hf download --local-dir' pull keeps the repo prefix)."
        echo "[dup] WARNING: Delete one before you compare them."
    fi
}

dl_hf() {
    local dir="$1" name="$2" repo="$3" rpath="$4"
    local dest="${dir}/${name}"
    local check_url; check_url="$(hf_resolve_url "$repo" "$rpath")"
    mkdir -p "$dir"

    local probe code want have=0
    probe="$(http_probe "$check_url")"
    code="$(printf '%s' "$probe" | cut -f1)"
    want="$(printf '%s' "$probe" | cut -f2)"

    [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [[ -f "$dest" ]]; then
        dup_check "$dir" "$name"
        if [[ -n "$want" ]] && (( have == want )); then
            echo "[model] $name complete (${have} bytes), skipping"; return 0
        elif [[ -n "$want" ]]; then
            echo "[model] $name size mismatch (local ${have} != remote ${want}) -> re-fetching"
            rm -f "$dest"
        else
            echo "[model] $name present, size unverifiable, assuming complete"; return 0
        fi
    fi

    classify_code "$code" "$name" || { echo "[model] SKIPPING $name (see [probe] above)"; return 0; }

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
# MANIFEST IS EMPTY BY DEFAULT -- the whole block is inert until you add
# entries. The transport is kept intact because that is where LTX style and
# identity LoRAs land, and rebuilding this later is annoying.
#
# Auth goes in a header so the token never reaches provisioning.log; it falls
# back to the query-string form only if the header is rejected, and says so.
#
# Before you fill this in: the LTX-2.x license's Attachment A is an
# enforceable use-restriction list, not boilerplate, and accepting the HF gate
# is accepting it. Worth five minutes of reading given what you point this at.
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
# Manifest  --  ~84 GB with the defaults
#
# Sizes are the actual repo figures, not estimates.
#
# NOTE THE DIRECTORIES. checkpoints/ is not a mistake and not an H3 habit
# carried over -- LTX-2.3 genuinely loads through CheckpointLoaderSimple. See
# item 1 in WHAT IS STRUCTURALLY DIFFERENT.
# ---------------------------------------------------------------------------
CKPT="${COMFY}/models/checkpoints"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
DIFF="${COMFY}/models/diffusion_models"
LATUP="${COMFY}/models/latent_upscale_models"
WF="${COMFY}/user/default/workflows"

# --- Civitai:  dest_dir | dest_filename | full_url ---
# Deliberately empty. Add your entries here; the block above is fully wired.
CIVITAI_FILES=()

# --- Hugging Face:  hf | dest_dir | dest_filename | repo_id | repo_path ---
# FIVE fields. The loop validates this and complains loudly if you miscount.
MODELS=(
    # === BASE CHECKPOINT -- FULL bf16 (46.1 GB) ==========================
    # All-in-one: transformer + video VAE + audio VAE + text projection.
    # This one file satisfies CheckpointLoaderSimple, LTXVAudioVAELoader and
    # the ckpt_name input on LTXAVTextEncoderLoader.
    #
    # The template ships pointing at ltx-2.3-22b-dev-fp8.safetensors (~25 GB)
    # from Lightricks/LTX-2.3-fp8. You will need to repoint the dropdown.
    "hf|$CKPT|ltx-2.3-22b-dev.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-dev.safetensors"

    # === TEXT ENCODER -- FULL bf16 (24.4 GB) =============================
    # Gemma 3 12B IT. Note the repo is Comfy-Org/ltx-2, NOT ltx-2.3 -- the
    # encoder did not change between releases and was never re-uploaded.
    # Give-backs if disk binds: gemma_3_12B_it_fp8_scaled (13.2 GB),
    # gemma_3_12B_it_fp4_mixed (9.45 GB, the template default).
    "hf|$TE|gemma_3_12B_it.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it.safetensors"

    # === DISTILLED LoRA v1.1, rank ~111 (2.74 GB) ========================
    # THE one the stock templates reference, at strength 0.5. Not optional if
    # you want the shipped graphs to run unmodified.
    "hf|$LORA|ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors|Comfy-Org/ltx-2.3|split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"

    # === GEMMA ABLITERATED LoRA rank64 bf16 (small) ======================
    # Applied to the TEXT ENCODER, not the transformer, via a plain LoraLoader
    # at 1.0/1.0 in the stock graph. Gemma is instruction-tuned and will
    # refuse or sanitise prompts; this is the release's answer to that. Part
    # of the official template model set, so fetch it or the graph is red.
    "hf|$LORA|gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors|Comfy-Org/ltx-2|split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"

    # === SPATIAL UPSCALER x2 v1.1 (996 MB) ===============================
    # Latent-space, and it is stage two of the stock two-stage graph. Small
    # file, large effect. v1.1 over v1.0 -- the older one is still floating
    # around in third-party workflow JSONs.
    "hf|$LATUP|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)

# === RANK-384 DISTILLED LoRA v1.1 (7.61 GB) -- ON BY DEFAULT =============
# The official Lightricks distillation LoRA, and the heavier of the two. Pairs
# with the dev checkpoint at 8 steps / CFG 1. On a 96 GB card there is no
# reason not to have both on disk and A/B them; the rank-111 file is what the
# template wires by default, this is the higher-fidelity option.
#
# BOTH are dev-model LoRAs. Neither belongs on the distilled CHECKPOINT --
# that is already distilled and stacking a distillation LoRA on it does not
# compose.
if [[ "$WANT_LORA_384" == "1" ]]; then
MODELS+=(
    "hf|$LORA|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
)
fi

# === TEMPORAL UPSCALER x2 (262 MB) ======================================
# Doubles frame count in latent space. Cheap, and a real alternative to RIFE:
# it interpolates before decode with the model's own prior rather than after
# decode from finished pixels.
if [[ "$WANT_TEMPORAL" == "1" ]]; then
MODELS+=(
    "hf|$LATUP|ltx-2.3-temporal-upscaler-x2-1.0.safetensors|Lightricks/LTX-2.3|ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
)
fi

# === TINY VAE for sampler previews (23.5 MB) ============================
# taeltx2_3, madebyollin's TAE for the 2.3 latent space. Without it you get
# latent-RGB previews via KJNodes at much lower fidelity. 24 MB to be able to
# see what a 5-minute sample is actually doing before it finishes.
if [[ "$WANT_TAE" == "1" ]]; then
MODELS+=(
    "hf|$VAE|taeltx2_3.safetensors|Kijai/LTX2.3_comfy|vae/taeltx2_3.safetensors"
)
fi

# === DISTILLED CHECKPOINT v1.1 (46.1 GB) -- OPT-IN ======================
# A separate all-in-one checkpoint, natively few-step, no LoRA required. The
# FLF2V template uses this rather than the dev checkpoint. Only worth the disk
# if you actually want first-last-frame work or a no-LoRA fast path.
if [[ "$WANT_DISTILLED_CKPT" == "1" ]]; then
MODELS+=(
    "hf|$CKPT|ltx-2.3-22b-distilled-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-distilled-1.1.safetensors"
)
fi

# === SPLIT COMPONENT FILES (Kijai) -- OPT-IN, ~46 GB ====================
# The same weights as the all-in-one, pulled apart into transformer / video
# VAE / audio VAE / text projection. This is the layout your H3 build uses,
# and the reason to want it is the same: per-component memory control, the
# ability to swap a VAE without touching the transformer, and compatibility
# with the community workflows built around Kijai's split files.
#
# THE COST IS REAL: it is a full duplicate of weights you already have, the
# six SHIPPED TEMPLATES DO NOT USE IT, and you will need workflows built for
# split loading (RuneXX/LTX-2.3-Workflows is the usual source) or you will be
# rewiring loaders by hand.
#
# Do not turn this on "just in case". Turn it on when a specific workflow
# needs it.
if [[ "$WANT_SPLIT" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|ltx-2.3-22b-dev_transformer_only_bf16.safetensors|Kijai/LTX2.3_comfy|diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors"
    "hf|$VAE|LTX23_video_vae_bf16.safetensors|Kijai/LTX2.3_comfy|vae/LTX23_video_vae_bf16.safetensors"
    "hf|$VAE|LTX23_audio_vae_bf16.safetensors|Kijai/LTX2.3_comfy|vae/LTX23_audio_vae_bf16.safetensors"
    "hf|$TE|ltx-2.3_text_projection_bf16.safetensors|Kijai/LTX2.3_comfy|text_encoders/ltx-2.3_text_projection_bf16.safetensors"
)
fi

# === SeedVR2 RESTORE STAGE -- OPT-IN, AND DEMOTED ON THIS BUILD =========
# On the H3 build SeedVR2 was the single largest quality lever, because H3 was
# hard-capped at a 768 short edge and a temporal restorer was the only route
# past it. That reasoning does not carry.
#
# LTX-2.3 generates natively to 4K and ships a LATENT spatial upscaler that
# runs before decode and is followed by a real refine pass. That is strictly
# better positioned than a post-decode pixel restorer: the upscaler sees
# latents, SeedVR2 sees finished frames and has to infer backwards.
#
# SeedVR2 still earns its place for restoring footage that has ALREADY been
# decoded -- salvaging an old render, upscaling source material, cleaning up
# something that came out of another pipeline. It is no longer stage two of
# the default path.
if [[ "$WANT_SEEDVR2" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|seedvr2_7b_fp16.safetensors|Comfy-Org/SeedVR2|diffusion_models/seedvr2_7b_fp16.safetensors"
    "hf|$VAE|seedvr2_ema_vae_fp16.safetensors|Comfy-Org/SeedVR2|vae/seedvr2_ema_vae_fp16.safetensors"
)
fi

# ---------------------------------------------------------------------------
# Manifest validation -- catches the four-field bug class before it becomes a
# mystery 404 an hour into a boot.
# ---------------------------------------------------------------------------
validate_manifest() {
    local bad=0 entry kind a b c d extra
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d extra <<< "$entry"
        if [[ "$kind" != "hf" ]]; then
            echo "[manifest] BAD KIND '$kind': $entry"; bad=1; continue
        fi
        if [[ -z "$a" || -z "$b" || -z "$c" || -z "$d" ]]; then
            echo "[manifest] MALFORMED (need 5 |-separated fields, got fewer): $entry"
            echo "[manifest]   format: hf|<dir>|<filename>|<repo_id>|<path_in_repo>"
            bad=1; continue
        fi
        if [[ -n "$extra" ]]; then
            echo "[manifest] MALFORMED (more than 5 fields): $entry"; bad=1; continue
        fi
        if [[ "$c" != */* || "$c" == */*/* ]]; then
            echo "[manifest] SUSPECT repo_id '$c' -- expected exactly one slash (owner/name)"
            echo "[manifest]   in: $entry"
            bad=1
        fi
    done
    (( bad == 0 )) && echo "[manifest] ${#MODELS[@]} entries, all well-formed"
    return 0
}

# ---------------------------------------------------------------------------
# Pre-download probe. One HEAD per entry, before a single byte is fetched, so
# a gated repo or a stale path is visible in the first thirty seconds instead
# of at the end of a long pull.
# ---------------------------------------------------------------------------
probe_manifest() {
    local entry kind a b c d url probe code n_ok=0 n_bad=0
    echo "[probe] checking ${#MODELS[@]} manifest entries..."
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        [[ "$kind" == "hf" && -n "$d" ]] || continue
        url="$(hf_resolve_url "$c" "$d")"
        probe="$(http_probe "$url")"
        code="$(printf '%s' "$probe" | cut -f1)"
        if classify_code "$code" "${c}/${d}"; then
            (( n_ok++ ))
        else
            (( n_bad++ ))
        fi
    done
    echo "[probe] ${n_ok} reachable, ${n_bad} not"
    if (( n_bad > 0 )); then
        echo "[probe] !!! ${n_bad} entries will not download. Read the verdicts above:"
        echo "[probe] !!!   GATED   -> accept the license, check which account the token is from"
        echo "[probe] !!!   MISSING -> the path is stale, edit the manifest"
        echo "[probe] !!! Everything reachable still downloads; this is not fatal."
    fi
}

# ---------------------------------------------------------------------------
# Disk pre-flight
#
# Extra job on this build: detect a leftover H3 model set. 131 GB of H3
# weights and 84 GB of LTX weights do not both fit on a 180 GB volume, and the
# failure mode without this check is a download that dies two thirds of the
# way through the 46 GB checkpoint.
# ---------------------------------------------------------------------------
H3_FILES=(
    "${COMFY}/models/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    "${COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
    "${COMFY}/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
    "${COMFY}/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    "${COMFY}/models/vae/minimax_h3_video_vae_fp16.safetensors"
    "${COMFY}/models/vae/minimax_h3_audio_vae_fp32.safetensors"
)

h3_bytes() {
    local total=0 f sz
    for f in "${H3_FILES[@]}"; do
        [[ -f "$f" ]] || continue
        sz="$(stat -c%s "$f" 2>/dev/null || echo 0)"
        total=$(( total + sz ))
    done
    printf '%s' "$total"
}

check_h3_residue() {
    local n=0 f used
    for f in "${H3_FILES[@]}"; do [[ -f "$f" ]] && (( n++ )); done
    (( n == 0 )) && return 0
    used="$(h3_bytes)"
    echo "=================== H3 RESIDUE ==================="
    echo "[h3] ${n} MiniMax H3 weight files present, $(numfmt --to=iec "$used" 2>/dev/null || echo "${used} B") total."
    if [[ "${PURGE_H3:-0}" == "1" ]]; then
        for f in "${H3_FILES[@]}"; do
            [[ -f "$f" ]] && { echo "[h3] removing $(basename "$f")"; rm -f "$f"; }
        done
        echo "[h3] done. The H3 custom-node packs are left alone -- they are small"
        echo "[h3] and harmless. Remove them by hand if you want the menu back."
    else
        echo "[h3] Leaving them in place. Set PURGE_H3=1 to reclaim that space on the"
        echo "[h3] next boot, and keep an offline copy first if the Kijai 4-step v0.1"
        echo "[h3] turbo LoRA is among what you would be deleting -- that file is a"
        echo "[h3] superseded preview with no upstream retention guarantee."
    fi
}

preflight_disk() {
    local need=0 kind a b c d url dest have want
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        [[ "$kind" == "hf" && -n "$d" ]] || continue
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

    mkdir -p "$CKPT"
    local avail; avail="$(df -PB1 "$CKPT" | awk 'NR==2{print $4}')"
    local margin=$(( 15 * 1024*1024*1024 ))   # bf16 loads spill to disk
    local h_need h_avail h_h3
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Cheapest give-backs, in the order you should make them:"
        local used; used="$(h3_bytes)"
        if (( used > 0 )); then
            h_h3="$(numfmt --to=iec "$used" 2>/dev/null || echo "${used} B")"
            echo "[provisioning] !!!   PURGE_H3=1                       -${h_h3}   <- start here"
        fi
        echo "[provisioning] !!!   PURGE_LEGACY_SEEDVR2=1 (if models/SEEDVR2 exists)  -15.0 GB"
        echo "[provisioning] !!!   WANT_SPLIT=0                       -46.0 GB"
        echo "[provisioning] !!!   WANT_DISTILLED_CKPT=0              -46.1 GB"
        echo "[provisioning] !!!   WANT_SEEDVR2=0                     -15.0 GB"
        echo "[provisioning] !!!   WANT_LORA_384=0                     -7.6 GB"
        echo "[provisioning] !!!   text encoder -> fp8_scaled         -11.2 GB"
        echo "[provisioning] !!!   text encoder -> fp4_mixed          -15.0 GB"
        echo "[provisioning] !!!   checkpoint   -> dev-fp8            -21.0 GB"
        echo "[provisioning] !!!     (Lightricks/LTX-2.3-fp8, ltx-2.3-22b-dev-fp8.safetensors)"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
echo "=================== MANIFEST ==================="
validate_manifest
check_h3_residue

for d in "$CKPT" "$LORA" "$TE" "$VAE" "$LATUP" "$WF"; do mkdir -p "$d"; done
# latent_upscale_models is a newer ComfyUI folder. If the dropdown on
# LatentUpscaleModelLoader is empty after this run despite the file being on
# disk, the folder is not registered in your build -- that is a ComfyUI
# version problem, not a download problem.

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
else
    echo "[civitai] manifest empty -- nothing to do (add entries in CIVITAI_FILES)"
fi

echo "=================== HUGGING FACE ==================="
echo "[provisioning] NOTE: default build is ~84 GB. First boot on a fresh volume"
echo "[provisioning] NOTE: is a long pull -- the bf16 checkpoint alone is 46.1 GB."
if [[ "$SKIP_PROBE" != "1" ]]; then
    probe_manifest
else
    echo "[probe] SKIP_PROBE=1 -> skipping the pre-download HEAD sweep"
fi

if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d extra <<< "$entry"
        if [[ "$kind" != "hf" || -z "$a" || -z "$b" || -z "$c" || -z "$d" || -n "$extra" ]]; then
            echo "[model] SKIPPING malformed entry: $entry"
            continue
        fi
        dl_hf "$a" "$b" "$c" "$d"
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
    [[ "$kind" == "hf" && -n "$d" ]] || continue
    if [[ -f "${a}/${b}" ]]; then
        echo "[layout] OK      ${a#${COMFY}/}/${b}  ($(numfmt --to=iec "$(stat -c%s "${a}/${b}")" 2>/dev/null))"
    else
        echo "[layout] MISSING ${a#${COMFY}/}/${b}"
    fi
done

for nd in ComfyUI-VideoHelperSuite ComfyUI-KJNodes ComfyUI-Olm-DragCrop \
          ComfyUI-RIFE-TensorRT-Auto ComfyUI-LTXVideo; do
    if [[ -d "${NODES_DIR}/${nd}" ]]; then
        echo "[layout] OK      custom_nodes/${nd} @ $(git -C "${NODES_DIR}/${nd}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    else
        echo "[layout] absent  custom_nodes/${nd}"
    fi
done

# ===========================================================================
# FINALISE -- log discovery, health check, optional restart
# ===========================================================================
echo "=================== FINALISE ==================="

# There is no fixed path for the ComfyUI log. ai-dock writes
# /var/log/supervisor/comfyui.log; Vast templates write under /var/log/portal/.
# Read stdout_logfile straight out of the supervisor program block instead of
# guessing, and grab `command=` too so a failed startup can be reproduced in
# the foreground -- the only reliable way to see an import traceback.
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
                      || echo "[provisioning] comfyui log: not found (try: ls /var/log/portal /var/log/supervisor)"

# supervisor reports RUNNING once a process survives `startsecs` (5s default).
# That says nothing about whether ComfyUI imported its nodes and bound its
# port, so a crash-loop reads as a rapid succession of healthy RUNNING states.
# Sampling the pid twice is what actually distinguishes them.
comfy_pid() { supervisorctl status comfyui 2>/dev/null | grep -oE 'pid [0-9]+' | awk '{print $2}'; }
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
        COMFY_STATE="stable"; echo "[provisioning] comfyui stable (pid ${_p2})"
    else
        COMFY_STATE="${_st:-absent}"; echo "[provisioning] comfyui state: ${COMFY_STATE}"
    fi
fi

if (( CHANGED_ANY )); then
    if [[ "$RESTART_COMFY_ON_UPDATE" != "1" ]]; then
        echo "[provisioning] code changed. Restart to load it:  supervisorctl restart comfyui"
        echo "[provisioning] (auto-restart off by default; RESTART_COMFY_ON_UPDATE=1 to enable)"
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

# ---------------------------------------------------------------------------
# Operating notes
# ---------------------------------------------------------------------------
cat <<'NOTES'

=================== FIRST RUN: THE THREE DROPDOWNS ===================

Load Template Library > Video > "LTX-2.3: Image to Video". It will open
pointing at the fp8 build and the fp4 encoder, because that is what the
template ships with. Three widgets to change:

  ckpt_name      ltx-2.3-22b-dev-fp8   ->  ltx-2.3-22b-dev.safetensors
  text_encoder   gemma_3_12B_it_fp4_mixed -> gemma_3_12B_it.safetensors
  (the distilled_lora and latent_upscale_model dropdowns already match what
   this script fetched -- rank-111 v1.1 and spatial-upscaler-x2-1.1)

The stock graph is a subgraph and proxies ckpt_name to a single input, so one
change covers all three consumers. If you flatten it or rebuild by hand, the
checkpoint has to be set in THREE places:
    CheckpointLoaderSimple    (transformer + video VAE)
    LTXVAudioVAELoader        (audio VAE, read from the same file)
    LTXAVTextEncoderLoader    (text projection, read from the same file)
Set two of three and you get a partial load with an unhelpful error.

=================== WHAT THE STOCK I2V GRAPH ACTUALLY DOES ===================

Read off the shipped template, not from memory:

  MODEL     CheckpointLoaderSimple  -> LoraLoaderModelOnly
                                       distilled 1.1 LoRA, strength 0.50
  ENCODER   LTXAVTextEncoderLoader(gemma, ckpt, "default")
                                    -> LoraLoader(gemma abliterated, 1.0/1.0)
                                    -> CLIPTextEncode  x2
  NEGATIVE  "pc game, console game, video game, cartoon, childish, ugly"
            (yes, there IS a negative prompt here -- unlike H3's BasicGuider
             at CFG 1, this graph uses CFGGuider with a real negative path.
             In-prompt negation is not the only tool you have on LTX.)
  COND      LTXVConditioning(frame_rate) -> LTXVImgToVideoInplace
  LATENT    EmptyLTXVLatentVideo + LTXVEmptyLatentAudio
                                    -> LTXVConcatAVLatent
  STAGE 1   SamplerCustomAdvanced, euler, CFGGuider cfg=1
            ManualSigmas: 1.0, 0.99375, 0.9875, 0.98125, 0.975,
                          0.909375, 0.725, 0.421875, 0.0        (8 steps)
            AT HALF THE REQUESTED CANVAS -- there are ComfyMathExpression
            nodes computing a/2 on both width and height.
  UPSCALE   LTXVLatentUpsampler with ltx-2.3-spatial-upscaler-x2-1.1
            (latent space, before any decode)
  STAGE 2   SamplerCustomAdvanced, euler, CFGGuider cfg=1
            ManualSigmas: 0.85, 0.7250, 0.4219, 0.0             (3 steps)
  DECODE    LTXVSeparateAVLatent -> VAEDecodeTiled(768/64/4096/4)
                                 -> LTXVAudioVAEDecode
  OUT       CreateVideo(images, audio, fps) -> SaveVideo

THE WIDTH/HEIGHT WIDGETS ARE THE FINAL OUTPUT, NOT THE SAMPLED CANVAS.
Ask for 1280x720 and stage one samples 640x360. Ask for 1920x1088 and stage
one samples 960x544. This is the most common misreading of the graph and it
is why "it looks soft at high res" reports are usually someone who disabled
stage two.

DEFAULTS: 1280 x 720, 25 fps, 5 s. Frame count is computed as
fps * duration + 1 (there is a ComfyMathExpression doing exactly `a * b + 1`),
so 25 x 5 + 1 = 126 frames. The +1 is the anchor frame; if you hand-set a
frame count, keep it on that grid.

=================== THE ONE DIAL THAT MATTERS ===================

DISTILLED LoRA STRENGTH.

  0.50   what the ComfyUI template ships. Fast.
  0.80   what Lightricks' own two-stage HQ pipeline uses, on the same dev
         checkpoint. This is the published quality setting.

Sweep 0.5 -> 0.8 before you touch sigmas, steps, resolution, or the encoder.
It is a single-variable change and it is where the visible difference lives.

WHAT NOT TO DO: run the dev checkpoint with NO distilled LoRA and the stock
ManualSigmas. Those sigma lists are a distilled schedule -- eight steps with
four near-zero decrements at the top. Feed them to an undistilled model and
you are asking it to do in eight steps what it was trained to do in fifty,
with a schedule shaped for a different velocity field. If you genuinely want
the undistilled path, replace ManualSigmas with a normal scheduler and raise
the step count; do not mix.

THE TWO LoRAs, AND WHICH TO REACH FOR:
  rank ~111 (2.74 GB)  ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_...
                       The template default. Lighter, and what every shipped
                       graph and most community JSONs reference.
  rank 384 (7.61 GB)   ltx-2.3-22b-distilled-lora-384-1.1
                       Lightricks' official distillation LoRA, heavier and
                       higher fidelity. 8 steps, CFG 1.
Both are DEV-model LoRAs. Neither goes on the distilled checkpoint.

If you swap between them, do not carry the sigma list across without checking
it. Distillation memorises specific sigma values -- the same reason shift
experiments do not belong on an H3 turbo tier, arriving by a different route.

=================== WHAT DOES NOT TRANSFER FROM THE H3 BUILD ===================

  NO 768 FLOOR. H3 fell off a cliff below a 768 short edge: RoPE went
  off-distribution and shift 12 was calibrated for ~1.0 MP, so low-res drafts
  changed structure rather than coarsening it. LTX-2.3 has no equivalent
  cliff. Drafting at reduced resolution is legitimate here and what you learn
  there does transfer. This is a habit worth actively unlearning.

  NO SHIFT CALIBRATION. There is no ModelSamplingAV analogue, no coupled
  video/audio shift pair, nothing to tune. Sigma LISTS replace it.

  THERE IS A NEGATIVE PROMPT. H3's stock graph used BasicGuider with no
  negative path, which is why in-prompt negation backfired and the fix was
  over-specifying in positive space. LTX-2.3's template uses CFGGuider with
  a populated negative. Suppression by negative conditioning works here.

  AUDIO IS DESCRIBED, NOT CONDITIONED FROM A FILE (in T2V/I2V). Write the
  sounds and dialogue into the prompt; the model generates synchronized audio
  in the same pass. If you want to DRIVE from an audio file, that is a
  different template -- IA2V -- and it does lip-sync from a supplied clip.

  TRIGGER-WORD POSITION IS NOT THE SAME PROBLEM. The H3 rule (front-load
  trigger words, because a tail-positioned bare token sits next to the audio
  slot and gets vocalised) came out of H3's caption-format priors. Gemma 3 is
  a different encoder with a different prompt format. Test it rather than
  assuming the rule carries.

=================== PROMPTING ===================

Lightricks' own guidance, in their order:
  1. CORE ACTIONS   -- events and actions as they occur over time
  2. VISUAL DETAILS -- everything you want to appear
  3. AUDIO          -- sounds and dialogue for the scene

For I2V specifically: describe what happens NEXT. Do not re-describe what is
already visible in the source still -- you are spending tokens re-stating the
conditioning image back to a model that already has it.

The abliterated Gemma LoRA in the graph exists because Gemma 3 is
instruction-tuned and will otherwise sanitise or refuse. If prompts are coming
back visibly softened, check that LoRA is actually applied before you rewrite
the prompt.

PROMPT ENHANCER: there is a TextGenerateLTX2Prompt node in the graph, gated
off by default behind a boolean. It expands a short prompt into a structured
one. Off is the right default while you are learning what the model responds
to -- you cannot debug a prompt you did not write.

=================== FRAMING THE INPUT IMAGE (Olm DragCrop) ===================

Same reasoning as the H3 build: I2V conditions on the source still, so a
source at the wrong aspect gets letterboxed or stretched into the canvas and
the model inherits that. Load Image -> Olm DragCrop -> the LTX image input.
The graph does not evaluate until you run it, so you can frame freely without
burning a render.

=================== RESOLUTION, FRAME RATE, DURATION ===================

Native up to 4K and up to 50 fps with synchronized audio. Remember the halving:
  final 1280 x  720  -> stage 1 samples  640 x 360
  final 1920 x 1088  -> stage 1 samples  960 x 544
  final 2560 x 1440  -> stage 1 samples 1280 x 720
That last one is the interesting configuration on a 96 GB card: stage one at
what most people call "full res", stage two at 1440p.

ASK FOR THE FRAME RATE YOU WANT AT THE SOURCE. LTX generates up to 50 fps
natively. Generating at 25 and interpolating to 50 with RIFE gives you
synthesised intermediate frames; generating at 50 gives you sampled ones.
Reach for RIFE when you want to go past 50, or on footage you already have.

TEMPORAL UPSCALER x2 is the middle option: it doubles frame count in LATENT
space using the model's own prior, before decode. Cheaper than re-rendering,
better positioned than RIFE. 262 MB, fetched by default.

=================== SeedVR2, IF YOU TURNED IT ON ===================

The chain from the H3 build still works and the wiring has not changed:

    (decoded frames)
      -> Resize Image (lanczos)
      -> Pre-Process SeedVR2 Input
      -> VAEEncodeTiled            (SeedVR2 VAE)
      -> [Split SeedVR2 Latent]
      -> KSampler                  (1 step, cfg 1, euler, simple, denoise 1)
      -> [Merge SeedVR2 Latents]
      -> VAEDecodeTiled            (SeedVR2 VAE)
      -> Post-Process SeedVR2 Output

  Loaders are stock: UNETLoader -> seedvr2_7b_fp16, VAELoader ->
  seedvr2_ema_vae_fp16. Wire the resized images to BOTH the pre-processor and
  Post-Process's original_resized_images socket -- that second input is the
  colour-matching reference and the node has nothing to match against without
  it. The KSampler is not a dial: 1 step is the formulation the model was
  trained for.

  BUT ASK FIRST WHETHER YOU NEED IT. On this build the answer is usually no.
  LTX's spatial upscaler runs in latent space with a refine pass behind it;
  SeedVR2 runs on finished pixels and has to infer backwards. Use SeedVR2 for
  material that is ALREADY decoded and cannot be re-rendered.

=================== TROUBLESHOOTING ===================

IF EVERY LIGHTRICKS FILE FAILS TO DOWNLOAD
  Look for [probe] GATED lines near the top of the HUGGING FACE section. The
  repos are license-gated. Open the model page, accept, and confirm the
  account you accepted with is the one HF_TOKEN was issued from -- that
  mismatch is the usual cause and it produces an identical 403 to "never
  accepted at all". No amount of retrying changes either.

IF A SINGLE FILE 404s EVERY BOOT
  [probe] MISSING means the path is stale, not that the network flaked. Files
  get renamed and moved into subdirectories; the manifest has to follow. Open
  the repo's file tree and compare.

IF UPDATES DO NOT SEEM TO LAND
  Read the [git] lines, in this order:
    1. "safe.directory configured". If that says WARNING, every git call in
       the run is failing on dubious ownership and nothing updated, whatever
       else the log claims.
    2. "detached HEAD -- looks image-pinned, leaving alone". That is correct
       behaviour. COMFY_PIN moves it deliberately; ALLOW_BRANCH_RECOVERY=1
       tracks a branch instead.
    3. "fast-forward blocked". Local commits or a dirty tree.
       GIT_FORCE_RESET=1 discards them.

IF THE LTX NODES ARE MISSING ENTIRELY
  Check the [comfy] version line. Native LTX-2.3 support starts at 0.16.1.
  Below that the nodes do not exist and there is nothing to fix in the graph.

IF A NODE EXISTS BUT A WIDGET DOES NOT
  You are on a build between the floor and current. The templates track core;
  a widget added after your pinned release will not be there. Update, or load
  an older workflow JSON.

IF latent_upscale_models IS EMPTY IN THE DROPDOWN
  The file is on disk (check the [layout] lines) but the folder is not
  registered in your ComfyUI build. That is a version problem. It is also
  worth checking extra_model_paths.yaml if you have one -- a custom paths
  file that predates this folder will not list it.

IF CUSTOM NODES FAIL TO IMPORT
  Read the LAST line of the traceback, not the first. A pack dying on
    ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
  is not itself broken -- it imported diffusers, diffusers probed xformers,
  xformers found a half-installed flash_attn and took the flash path. This
  script detects and removes that on every boot; see the [flash] lines. If
  they say "healthy" or "not installed" and a pack still fails, look
  elsewhere.

  ComfyUI-Manager's SECURITY LEVEL does not cause import failures. It gates
  whether Manager may run install scripts. "Try fix" only re-runs that pack's
  requirements.txt, which cannot repair a broken package not listed in it.

IF PREVIEWS ARE BROKEN OR THE AUDIO VAE ERRORS
  Update KJNodes. It needed an explicit change to understand the LTX-2.3
  audio VAE, and a stale copy on a persistent volume is a known cause of
  exactly this on 2.3 graphs. This script pulls it every boot when
  NODE_UPDATE=1 -- check the [git] line for it.

IF COMFYUI SEEMS SLOW TO START
  Check the [provisioning] comfyui line at the end of this log. CRASH-LOOPING
  means it is being respawned every few seconds and there is no startup to
  wait for. Stop the service and run the printed command in the foreground;
  the traceback only appears there. supervisor reporting RUNNING means only
  that the process survived five seconds.

IF MODEL LOADING IS PATHOLOGICALLY SLOW
  Launch with --disable-pinned-memory.

IF CUDA GOES MISSING AFTER AN UPDATE
  Something moved torch despite the constraints file. The [torch] block said
  so loudly. Reinstall the pinned cu128 build for your card before rendering.

=================== NOT IN THIS BUILD ===================

  LTX-2.5. It is the current release and it is a different model: Gemma 4 12B
  encoder instead of Gemma 3, a Diffusion Video Decoder, native multishot,
  split component files rather than an all-in-one checkpoint, and a 0.32.0
  ComfyUI floor. Checkpoints are NOT interchangeable with 2.3. Most 2.3 LoRAs
  and IC-LoRAs reportedly carry over, but validate before you rely on it. If
  you want 2.5 this manifest is the wrong shape -- it needs its own script.

  IC-LoRA control (union canny+depth), ID-LoRA personalisation, and the
  ingredients IC-LoRA. All exist for 2.3 and all have shipped templates; none
  are fetched here because they are workflow-specific. IC-LoRAs load through
  a reference-downscale loader rather than a plain LoraLoader, so read the
  template before adding them by hand.

  The GGUF quantisations. Irrelevant at 96 GB.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
