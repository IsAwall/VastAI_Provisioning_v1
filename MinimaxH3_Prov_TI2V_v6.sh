#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# MiniMax H3 (Hailuo 3.0) -- I2V BUILD  ---  v6
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          recommended (rate limits on a ~131 GB pull)
#        CIVITAI_TOKEN=<token>     only if you add Civitai entries below
#        COMFY_UPDATE=0            skip the ComfyUI git update (default: ON)
#        NODE_UPDATE=0             skip the custom-node git updates (default: ON)
#        GIT_FORCE_RESET=1         discard local commits blocking a fast-forward
#        WANT_TURBO=1              add the Turbo LoRA draft tier (+1.5 GB)
#        WANT_SEEDVR2=1            add the SeedVR2 restore weights (+~15 GB)
#        PURGE_LEGACY_SEEDVR2=1    delete the v3-era third-party pack + weights
#        FIX_FLASH_ATTN=0          skip the half-installed flash-attn guard
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY -- see below)
#        COMFY_PIN=<sha|tag>       hold core at this ref (see UPDATE PATH below)
#        ALLOW_BRANCH_RECOVERY=1   permit moving a detached HEAD onto a branch
#        FORCE_DEPS=1              reinstall pip requirements even if unchanged
#        RESTART_COMFY_ON_UPDATE=1 bounce the comfyui service if code changed
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~131 GB of weights. Provision a 180 GB volume minimum, 200 comfortable.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v5 -- ALL OF IT IS ABOUT UPDATES ACTUALLY LANDING
#
#   v5's update machinery was structurally sound (git_sync already handled
#   detached HEAD, shallow clones and ff-only merges; the frontend pins were
#   already force-synced). Five specific things stopped it working in practice.
#
#   1. THE ENV TOGGLES WERE HARDCODED, so none of them worked. v5 documented
#      COMFY_UPDATE / NODE_UPDATE / WANT_TURBO / WANT_SEEDVR2 as environment
#      variables and then immediately overwrote all four:
#
#          COMFY_UPDATE="1"        <- not ${COMFY_UPDATE:-1}
#          NODE_UPDATE="1"
#          WANT_TURBO="0"
#          WANT_SEEDVR2="0"
#
#      Setting WANT_SEEDVR2=1 on the instance did nothing at all, and the two
#      update flags could not be turned off to freeze a working state. All four
#      now read from the environment with the documented defaults.
#
#   2. NO safe.directory, WHICH SILENTLY DISABLES EVERY GIT OPERATION. ai-dock
#      runs provisioning as root, but a persistent volume is frequently owned by
#      a different uid. Modern git then refuses with "detected dubious ownership
#      in repository" on EVERY call -- fetch, merge, rev-parse. git_sync's own
#      error paths swallow that, so updates degrade to silent no-ops while the
#      log still looks plausible. This is the single most likely reason core and
#      nodes were not updating on boot. One line fixes it, added before any git
#      call runs.
#
#   3. DETACHED HEAD WAS TREATED AS DAMAGE TO REPAIR. v5 resolved the remote's
#      default branch and did `git checkout -B "$def" "origin/${def}"`. But
#      images pin ComfyUI core to a release tag ON PURPOSE -- ai-dock via
#      COMFYUI_REF, Vast templates via their own config. Recovering that pin
#      walks core onto bleeding-edge master, and ComfyUI's own README warns that
#      commits outside stable release tags may be very unstable and break many
#      custom nodes. The launch flags baked into a template are written against
#      the release it pinned, so this can crash-loop the service.
#
#      v6: a detached HEAD with no explicit pin is LEFT ALONE and logged as
#      such. To move it, either pass a pin (COMFY_PIN for core, or "|<sha>" on a
#      NODES entry) or set ALLOW_BRANCH_RECOVERY=1. Deliberate beats automatic.
#
#   4. REQUIREMENTS WERE REINSTALLED ON EVERY BOOT. v5 skipped pip only when
#      git_sync returned "unchanged" (rc 2); a FAILED update (rc 1) still fell
#      through to a full constrained pip run. Every boot therefore spent minutes
#      re-resolving dependencies and gave itself another chance to break a
#      working environment. v6 hashes each requirements.txt and installs only
#      when the file actually changed, the node was freshly cloned, or
#      FORCE_DEPS=1.
#
#   5. NOTHING EVER RESTARTED COMFYUI, so code updated during provisioning sat
#      on disk while the running process kept the old modules loaded -- which
#      reads exactly like "the update didn't work". v6 adds an optional restart,
#      OFF by default, and it refuses to fire at a service that is crash-looping
#      (restarting a flapping service just adds a cycle). It also samples the pid
#      twice at the end and says plainly whether ComfyUI is stable or looping.
#
#   Also new: core's git state (branch / nearest tag / commits ahead) is printed
#   every run so drift is visible in the log rather than silent, and the comfyui
#   log path is discovered from the supervisor program block rather than assumed
#   -- ai-dock uses /var/log/supervisor/comfyui.log, Vast templates use
#   /var/log/portal/, and guessing wrong costs you an hour.
#
#   UNCHANGED FROM v5: the flash-attn guard, the torch constraints file, the
#   frontend/backend pin sync, the model manifest, the disk pre-flight, and all
#   operating notes. Nothing about what gets downloaded has moved.
#
# -----------------------------------------------------------------------------
# THE UPDATE PATH -- the things that break unattended ComfyUI pulls
#
#   1. DUBIOUS OWNERSHIP. See item 2 above. Fixed unconditionally now.
#
#   2. DETACHED HEAD. ai-dock images frequently check out a release tag rather
#      than a branch. `git pull` on a detached HEAD exits 0 and does nothing, so
#      it looks like it worked. v6 detects it and, by default, respects it --
#      because a detached core is usually intentional, not broken. If you want
#      to manage core yourself, set COMFY_PIN to the tag you want.
#
#   3. SHALLOW CLONE. A --depth=1 image has no tag history, so version detection
#      returns nothing and a fast-forward has no merge base. git_sync unshallows
#      first.
#
#   4. TORCH GETTING CLOBBERED. This is the one that actually costs you the
#      instance. `pip install -r requirements.txt` -- ComfyUI's own, or any
#      custom node's -- can resolve a fresh torch off PyPI and replace a pinned
#      cu128 Blackwell build with a generic wheel. Every pip call that touches a
#      requirements file runs against a constraints file pinning the installed
#      torch stack to its exact current versions, local +cuXXX suffix included.
#      If a constrained install fails, the script retries unconstrained and then
#      verifies CUDA is still alive, loudly.
#
#   FRONTEND/BACKEND SKEW. comfyui-frontend-package, comfyui-workflow-templates
#   and comfyui-embedded-docs are pip packages pinned by ComfyUI's
#   requirements.txt. A mismatch between frontend and backend produces phantom
#   link errors on nodes with autogrow sockets (the ref_audio_0 class of bug) and
#   stale built-in templates. After a core update all three are force-synced to
#   exactly what the new requirements.txt asks for.
#
# -----------------------------------------------------------------------------
# WHERE THE QUALITY IS, IN THIS BUILD (ranked, largest lever first)
#
#   1. THE RESTORE STAGE. H3-Regenerate-2K was withheld from the weight
#      release; local output is capped at a 768px short edge, full stop. A
#      temporal restorer is the only route past that. SeedVR2 is a
#      diffusion-transformer video restorer -- it reconstructs texture from a
#      generative prior rather than interpolating pixels, and it is temporally
#      conditioned, so it does not flicker the way per-frame ESRGAN does on
#      water, fabric, and fine detail. 7B fp16, not fp8 and not the sharp
#      variant: sharp crisps up whatever is already there, which is the wrong
#      move when what is already there is mottling.
#
#   2. STEP COUNT AND SAMPLER. res_multistep is a second-order integrator --
#      local truncation error per step scales as the square of a first-order
#      method's, so it buys real accuracy per unit of compute rather than just
#      more of it. Baseline is 20 steps; 25-30 gives measurably better content
#      and motion. Below ~15 quality falls off visibly.
#
#   3. RESOLUTION. Run at the native canvas (1344x768 at 16:9, ~1.0 MP), not
#      the template's 0.4 MP preview. The model has a 384p FLOOR -- 256p fails
#      outright rather than degrading.
#
#   4. DIFFUSION PRECISION. bf16 (61.7 GB) over int8_convrot (31.7 GB) over
#      pruned_int8 (19.5 GB). Real but smaller than 1-3.
#
#   5. TEXT ENCODER PRECISION. Smallest lever here. Qwen3-VL is an encoder,
#      not a generator -- int8_convrot is close to lossless on it, and what
#      you would notice is marginal prompt-adherence drift on long multi-shot
#      prompts, not per-frame fidelity. If disk or RAM binds, this is the
#      first thing to give back, not the last.
#
# -----------------------------------------------------------------------------
# THE LADDER
#
#   DRAFT  -> Turbo v4-600, 6-8 steps, NATIVE CANVAS, 124 frames, Sage on,
#             SILENT. Buy the speedup with fewer frames and fewer steps, NOT
#             with a smaller canvas: dropping below the 768 short edge moves
#             the latent off-distribution and shift 12 is calibrated for ~1.0
#             MP, so low-res drafts change composition and structure rather
#             than just coarsening them. What you learn there does not
#             transfer. Frame count is free to cut; resolution is not.
#   FINAL  -> fl2va_bf16, bf16 encoder, res_multistep + simple, 25-30 steps,
#             1344x768, Turbo LoRA bypassed, no Sage, audio fields populated.
#   RESTORE-> SeedVR2 7B fp16, native nodes, resize multiplier 1.875 -> 1440.
#   INTERP -> RIFE TensorRT, 24 -> 48/60 fps. Last stage before encode.
#
#   TURBO IS A DRAFT TIER, NOT A FAST FINAL. It is a step-distillation LoRA:
#   it trains the model to take much larger jumps along the flow trajectory
#   so a handful of steps land where ~20 would. That is a fidelity trade by
#   construction.
#
#   WHAT IS NO LONGER TRUE: the plastic-skin / over-sharp-grain complaint was
#   a property of the v1 (~850) line, and the author reports it fully resolved
#   in v4-600. If you demoted turbo on those grounds, re-test before assuming.
#
#   WHAT IS STILL TRUE: audio is one of two areas the author lists as still
#   being improved (the other is behaviour under fast, intense motion). Since
#   H3 runs joint video-audio attention, degraded audio conditioning
#   propagates back into the video as motion-timing and boundary artifacts.
#   Keep drafting silent until you have tested v4's audio yourself.
#
#   4-8 STEPS IS THE USEFUL RANGE, and this is a real range, not a floor with
#   graceful decay. 4 is the recommended minimum, 6-8 looks noticeably better,
#   and past 8 it stops helping and starts introducing over-sharp artifacts.
#   Do not push a distilled LoRA to 15 steps expecting base-model behaviour.
#
#   DO NOT RUN SHIFT EXPERIMENTS ON THE TURBO TIER. Distillation memorises
#   specific sigma values; moving shift off the calibrated point breaks the
#   schedule rather than testing anything. Shift work belongs on the base
#   model.
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
#   question worth answering before you commit a 180 GB volume to this.
#
#   Attribution ("MiniMax H3" shown in-product) is required for commercial
#   use; >$20M revenue needs separate written authorization. The Turbo LoRA is
#   Apache-2.0 and SeedVR2 is Apache-2.0 -- separate licenses, base weights
#   still governed by the above.
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
echo "########## provisioning run (MiniMax H3 I2V v6): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

# ---------------------------------------------------------------------------
# CHANGED IN v6: these read from the environment now.
# v5 assigned literals here, which silently overrode everything the header
# documented -- WANT_SEEDVR2=1 on the instance had no effect whatsoever.
# ---------------------------------------------------------------------------
COMFY_UPDATE="${COMFY_UPDATE:-1}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_TURBO="${WANT_TURBO:-0}"
WANT_SEEDVR2="${WANT_SEEDVR2:-0}"
ALLOW_BRANCH_RECOVERY="${ALLOW_BRANCH_RECOVERY:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-0}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} node_update=${NODE_UPDATE} turbo=${WANT_TURBO} seedvr2=${WANT_SEEDVR2}"
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
# NEW IN v6, AND THE MOST LIKELY REASON NOTHING WAS UPDATING
#
# ai-dock runs provisioning as root, but a persistent /workspace volume is often
# owned by a different uid. Modern git refuses to operate on a repo it considers
# foreign:
#
#     fatal: detected dubious ownership in repository at '/workspace/ComfyUI'
#
# EVERY git call fails -- fetch, merge, rev-parse, describe. git_sync's error
# paths absorb those failures individually, so the run completes and the log
# looks broadly normal while nothing has actually been updated. This must run
# before the first git invocation.
# ---------------------------------------------------------------------------
git config --global --add safe.directory '*' 2>/dev/null \
    && echo "[git] safe.directory configured (prevents 'dubious ownership' no-ops)" \
    || echo "[git] WARNING: could not set safe.directory -- git ops may fail silently"

# ---------------------------------------------------------------------------
# Broken flash-attn guard  --  RUNS BEFORE THE PIN FILE IS WRITTEN
#
# Failure mode this exists for:
#
#   <any custom node> -> diffusers -> xformers.ops -> flash_attn.flash_attn_interface
#   ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
#
# The flash_attn directory is present enough that `import flash_attn` succeeds
# (often as a namespace package, i.e. a directory with no __init__.py -- the
# residue of a failed source build or an interrupted uninstall), so xformers
# concludes flash attention is available, takes that code path, and dies on
# the submodule. Every pack downstream of diffusers dies with it.
#
# Cleanly absent is FINE -- xformers falls back to its own kernels. So the fix
# is removal, not repair. Nothing in this build uses flash-attn.
#
# This must run before write_torch_pins(): a half-installed package still
# carries dist-info metadata, so importlib.metadata.version() succeeds and the
# broken build would otherwise be pinned into the constraints file and held in
# place for the rest of the run.
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
# Torch stack guard
#
# Snapshot the exact installed versions of the fragile CUDA cluster into a pip
# constraints file, including local +cuXXX suffixes. Every requirements.txt
# install below runs with -c "$CONSTRAINTS", so a transitive dependency cannot
# silently pull a generic torch wheel over a pinned Blackwell build. A
# constrained install that fails is the correct outcome: it means something
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
    # that is exactly the half-installed state fix_flash_attn() cleans up,
    # and pinning it would hold the breakage in place for the whole run.
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

# ---------------------------------------------------------------------------
# NEW IN v6: reqs_changed <repo_path> <requirements_file>
# Returns 0 when the file is new or its content hash moved.
#
# v5 ran a full constrained pip resolve on every boot for every node whose
# git_sync did not return exactly "unchanged" -- including nodes whose update
# FAILED. Minutes of wall clock per boot, and a fresh chance each time to
# re-resolve a working environment into a broken one.
# ---------------------------------------------------------------------------
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
# Host memory check
#
# The bf16 pairing is the whole point of this build and it is RAM-gated, not
# VRAM-gated. Fail loudly here rather than after a 131 GB download.
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
# git_sync <repo_path> <label> [pin]
#
# Return codes:  0 = HEAD moved   1 = could not update   2 = already current
#
# CHANGED IN v6: a detached HEAD is no longer "repaired" by default. See item 3
# in the changelog. Pass a pin to move it deliberately, or set
# ALLOW_BRANCH_RECOVERY=1 to restore the old v5 behaviour.
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
        # v5 checked out the default branch here. That walks an intentionally
        # pinned checkout onto master. Respect the pin unless told otherwise.
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
# Native H3 nodes landed in 0.30.0 (Comfy-Org/ComfyUI PR #15224). Native
# SeedVR2 landed earlier, in PR #14424. On an older image the graph opens with
# red nodes and no obvious cause, which is why this runs by default.
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
# These three carry the web frontend, the built-in workflow templates, and the
# node help docs. Skew here is invisible until a node with autogrow sockets
# starts reporting links that do not exist.
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
# NEW IN v6: report core git state every run, so drift is visible in the log
# instead of silent. If this ever shows a large "+N commits", something moved
# core off the release the image pinned.
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
MIN_CV="0.30.0"
echo "[comfy] after: ${CV:-<undetectable>}"
if [[ -z "$CV" ]]; then
    echo "[comfy] version undetectable -- verify manually that you are on >= ${MIN_CV}"
elif [[ "$(printf '%s\n%s\n' "$MIN_CV" "$CV" | sort -V | head -1)" == "$MIN_CV" ]]; then
    echo "[comfy] OK -- native H3 and native SeedVR2 support both present"
else
    echo "[comfy] !!! ${CV} is BELOW ${MIN_CV} -- the H3 nodes will not exist."
    echo "[comfy] !!! The update above did not take. Check the [git] lines, or pick"
    echo "[comfy] !!! a newer base image."
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>][@<pin>]
# Everything here is pulled on every boot when NODE_UPDATE=1 (the default).
# Append @<sha-or-tag> to hold a node at a known-good commit instead.
#
# NOTE what is NOT here: the SeedVR2 upscaler pack. SeedVR2 is core now.
# See RESTORE STAGE in the notes at the end.
# ---------------------------------------------------------------------------
NODES=(
    # Video load/save, frame extraction, the VHS_* family.
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    # Utility layer: resolution selector, torch compile, misc graph plumbing.
    "https://github.com/kijai/ComfyUI-KJNodes"
    # Interactive drag-to-crop on the image preview. No pip dependencies.
    "https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop"
    # Frame interpolation. Pulls the TensorRT stack; builds its engine on
    # first use, per GPU architecture. See the header note.
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
    # Native Context Loop for extending videos
    "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop"
)
[[ "$WANT_TURBO" == "1" ]] && NODES+=( "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo" )

# Optional: explicit dual-sigma sampler with manual video/audio shift control
# (12 / 3) and a 50-sigma-point schedule. Genuinely more control than the
# stock graph, BUT it overlaps the native H3 nodes and the Turbo nodes, and
# three H3 node packs in one install is a recipe for node-name collisions.
# Add it deliberately, after the stock path works.
#   "https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3"

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

    # CHANGED IN v6: only resolve pip when the requirements file actually moved.
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
# Legacy SeedVR2 pack from v3
#
# On a persistent volume the old third-party clone is still sitting there,
# failing to import on every boot and cluttering the node-manager list, and
# models/SEEDVR2 is holding ~15 GB of weights the native path cannot read.
# Warn by default; only delete when explicitly asked.
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
# CIVITAI (manifest is empty by default -- inert unless you add entries)
#
# Kept because H3 style and identity LoRAs land here. Auth goes in a header so
# the token never reaches provisioning.log; falls back to the query-string
# form only if the header is rejected, and says so when it does.
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
# Manifest  --  ~131 GB total
#
# Sizes below are the actual Comfy-Org repo figures, not estimates.
# UNCHANGED FROM v5.
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
WF="${COMFY}/user/default/workflows"

# --- Civitai:  dest_dir | dest_filename | full_url ---
CIVITAI_FILES=(
    "$LORA|H3_Epic_Cumshots.safetensors|https://civitai.red/api/download/models/3202064?fileId=3083352"
    "$LORA|H3_HMNSFW_AIO_Sex_v2.safetensors|https://civitai.red/api/download/models/3206518?fileId=3088013"
    "$LORA|H3_Mini_Dick_Fix.safetensors|https://civitai.red/api/download/models/3207332?fileId=3088892"
    "$LORA|H3_K3NK_Side_View_Deepthroat.safetensors|https://civitai.red/api/download/models/3216591?fileId=3098396"
)

# --- Hugging Face:  hf | dest_dir | dest_filename | repo_id | repo_path ---
MODELS=(
    # === DIFFUSION MODEL -- FULL bf16 (61.7 GB) ==========================
    # fl2va is the I2V checkpoint: it serves image-to-video, text-to-video,
    # and first-last-frame from one set of weights. There is no separate
    # "i2v-only" file to fetch.
    # Alternatives if you need the disk back: int8_convrot (31.7 GB) is a
    # genuinely small step down; pruned_int8 (19.5 GB) is the template
    # default and a visible one.
    "hf|$DIFF|minimax_h3_fl2va_bf16.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_bf16.safetensors"
    # "hf|$DIFF|minimax_h3_fl2va_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"

    # === TEXT ENCODER -- FULL bf16 (48.0 GB) =============================
    # Qwen3-VL-32B. See the header: smallest quality lever in the build and
    # the first thing to trade away. int8_convrot is 25.3 GB; nvfp4_awq is
    # 14.6 GB and the real step down.
    "hf|$TE|qwen3vl_32b_minimax_h3_bf16.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
    # "hf|$TE|qwen3vl_32b_minimax_h3_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

    # === H3 VAEs -- BOTH REQUIRED (4.9 + 0.6 GB) =========================
    # The audio VAE is not optional even for silent output: the audio stream
    # is generated in the same pass and has to be decoded. Missing it is the
    # usual cause of "my output has no sound" -- along with a missing
    # VAEDecodeAudio node feeding SaveVideo.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"

    # === LoRAs via HuggingFace ===========================================
    "hf|$LORA|H3_K3NK_Side_Deepthroat.safetensors|Ghiladden/7801|7800.safetensors"

    # === Workflows =======================================================
    "hf|$WF|Mike MiniMax H3 I2V Context Extend v1.json|Ghiladden/7801|Mike MiniMax H3 I2V Context Extend v1.json"

)

# === TURBO LoRA -- Kijai's 4-step kijai_minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16 =======
if [[ "$WANT_TURBO" == "1" ]]; then
MODELS+=(
    "hf|$LORA|minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors|lightx2v/Minimax-h3-Turbo|minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
)

fi

# === SeedVR2 RESTORE STAGE -- NATIVE WEIGHTS (~15 GB) ====================
# These are the Comfy-Org conversions for ComfyUI's native SeedVR2 nodes. They
# are NOT the numz/SeedVR2_comfyUI files v3 fetched, and the two are not
# interchangeable -- different key naming, different loaders, different
# directories. If you have the old files in models/SEEDVR2 they are dead
# weight; see the LEGACY SEEDVR2 block above.
#
# The DiT goes in diffusion_models and loads through a stock UNETLoader; the
# VAE goes in vae and loads through a stock VAELoader. That is the whole
# reason to prefer this path: ComfyUI's memory manager owns the model and
# evicts the H3 DiT for you.
#
# 7B fp16 deliberately. Not fp8 (quality), and NOT the sharp variant -- sharp
# crisps whatever texture is already present, and at the 768 ceiling what is
# present is mottling. You want it overwritten, not sharpened.
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
    local margin=$(( 15 * 1024*1024*1024 ))   # bf16 loads spill to disk
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Cheapest give-backs, in the order you should make them:"
        echo "[provisioning] !!!   PURGE_LEGACY_SEEDVR2=1 (if models/SEEDVR2 exists)  -15.0 GB"
        echo "[provisioning] !!!   text encoder -> int8_convrot      -22.7 GB"
        echo "[provisioning] !!!   text encoder -> nvfp4_awq         -33.4 GB"
        echo "[provisioning] !!!   diffusion    -> int8_convrot      -30.0 GB"
        echo "[provisioning] !!!   WANT_SEEDVR2=0                    -15.0 GB"
        echo "[provisioning] !!!   WANT_TURBO=0                       -1.5 GB"
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
echo "[provisioning] NOTE: full build is ~131 GB. First boot on a fresh volume"
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
# FINALISE -- log discovery, health check, optional restart   (NEW IN v6)
# ===========================================================================
echo "=================== FINALISE ==================="

# There is no fixed path for the ComfyUI log. ai-dock writes
# /var/log/supervisor/comfyui.log; Vast templates write under /var/log/portal/.
# Read stdout_logfile straight out of the supervisor program block instead of
# guessing, and grab `command=` too so a failed startup can be reproduced in the
# foreground (which is the only reliable way to see an import traceback).
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
# That says nothing about whether ComfyUI imported its nodes and bound its port,
# so a crash-loop reads as a rapid succession of healthy RUNNING states.
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

=================== FINAL-RENDER SETTINGS ===================

MODELS
  diffusion    minimax_h3_fl2va_bf16.safetensors
  text encoder qwen3vl_32b_minimax_h3_bf16.safetensors
  vae          minimax_h3_video_vae_fp16 + minimax_h3_audio_vae_fp32
  LoRA         none. Bypass or delete the Turbo nodes for finals.

SAMPLER
  res_multistep + simple scheduler, 25-30 steps.
  Below ~15 steps quality drops visibly. 20 is the community baseline; the
  25-30 band is where you stop getting much for the wall-clock.

TURBO DRAFT TIER (larryvrh v4-600)
  LoRA      minimax_h3_turbo_v4_step600_ema.safetensors
  steps     6-8. 4 is the minimum, 6-8 is where v4 looks its best, past 8
            stops helping and starts introducing over-sharp artifacts.
  strength  1.0. It is tuned for 1.0 and holds across the whole 4-8 range.
            Only touch the dial for a specific misbehaving clip:
              blurry ghosting / smear  -> nudge UP   (~1.05-1.20)
              over-sharp grain         -> nudge DOWN (~0.80-0.95)
  scheduler simple.
  low_vram  OFF. Off applies the LoRA at run time and is sharpest; on merges
            it into the weights for lower peak VRAM and is softer. At 96 GB
            there is no reason to turn it on.

  WIRING -- two changes to the official graph, nothing else moves:
    1. insert MiniMax-H3 Turbo LoRA between the model loader and the sampler
    2. feed SamplerCustomAdvanced from MiniMax-H3 Turbo Sampler
  The Turbo Sampler auto-adapts to your ComfyUI version -- video and audio
  ride two different flow schedules, recent ComfyUI handles that natively via
  ModelSamplingAV and older ComfyUI does not, and the node detects which.
  That matters here because this script updates ComfyUI on every boot: the
  node absorbs the change, so keep it updated and do not pin either side.

  WHEN TO REACH FOR ckpt850 INSTEAD: one case only -- 4 steps AND large, fast
  motion, where v4 can produce motion-smear and trailing ghosting. At 6-8
  steps that mostly goes away and v4 wins outright. Everywhere else ckpt850
  is the older over-sharpened line and you do not want it.

RESOLUTION -- both constraints are hard, and there is a FLOOR
  Resolution Selector: 16:9, Megapixels ~1.0, Multiple 32  ->  1344x768
  That is the native canvas: 768px short edge, capped 768x1344.
  Minimum is 384p. 256p fails outright rather than degrading.
  frames: 17k+5 grid at 24 fps. 124 ~ 5 s, 362 ~ 15 s. Validated 124-362.

  Mottled or patchy skin at this canvas is the resolution ceiling showing
  through, not a sampler misconfiguration. Do not chase it with sigma tuning.
  SeedVR2 is the lever.

FRAMING THE INPUT IMAGE (Olm DragCrop)
  I2V conditions on the source still, so a source at the wrong aspect gets
  letterboxed or stretched into the canvas and the model inherits that.
  Load Image -> Olm DragCrop -> the H3 image input. Drag the box on the
  preview to 16:9, fine-tune with the crop_left/right/top/bottom widgets,
  and note that the graph does not evaluate until you run it -- you can frame
  freely without burning a render.

AUDIO -- 32 kHz stereo, and the weakest part of the release
  The audio fields are NOT optional decoration. H3 runs joint video-audio
  attention, so an empty or broken audio conditioning path perturbs the video
  itself: motion timing drifts and boundary coherence degrades. Fill them
  even when you intend to mute the result.
  Expect repeated syllables and occasionally unrelated audio on dialogue.
  Audio problems are usually a SCHEDULER mismatch, not a broken model: video
  and audio ride separate flow schedules (video shift 12, audio shift 3) and
  H3 is unusually sensitive to sampler configuration. Before assuming a file
  is bad, check the scheduler.

PROMPTING AT CFG 1
  The stock graph uses BasicGuider, which has no negative conditioning path.
  In-prompt negation ("no camera movement") therefore primes the concept with
  nothing to subtract it, and reliably backfires. Suppress an unwanted motion
  axis by over-specifying the wanted one in positive space instead -- name the
  rig type, state the frame invariants.
  Put LoRA trigger words at the FRONT of the prompt. Qwen3-VL is decoder-only
  with causal attention, so a front-positioned token propagates conditioning
  across the whole sequence; a trailing one sits next to the audio slot and
  gets vocalised as speech.

=================== RESTORE STAGE -- NATIVE SeedVR2 ===================

The SeedVR2 nodes are core, so they are always present and always match your
ComfyUI version -- but nothing ships pre-wired into the H3 template, so the
graph is still yours to build.

  NODE SEARCH: type "SeedVR2". You should see five nodes:
    Pre-Process SeedVR2 Input      (SeedVR2Preprocess)
    Apply SeedVR2 Conditioning     (SeedVR2Conditioning)
    Split SeedVR2 Latent           (temporal chunking)
    Merge SeedVR2 Latents
    Post-Process SeedVR2 Output    (SeedVR2PostProcessing)
  If those five are missing, ComfyUI is too old. If OTHER packs are missing
  from the menu, that is a different problem -- see IF CUSTOM NODES FAIL TO
  IMPORT at the bottom.

  THE CHAIN, spliced after the H3 video VAEDecode:

    VAEDecode (H3 video)
      -> Resize Image (multiplier 1.875, lanczos)
      -> Pre-Process SeedVR2 Input
      -> VAEEncodeTiled            (SeedVR2 VAE)
      -> [Split SeedVR2 Latent]
      -> KSampler                  (1 step, cfg 1, euler, simple, denoise 1)
      -> [Merge SeedVR2 Latents]
      -> VAEDecodeTiled            (SeedVR2 VAE)
      -> Post-Process SeedVR2 Output
      -> RIFE -> CreateVideo

    VAEDecodeAudio ----------------AUDIO----------------> CreateVideo
      Audio still bypasses the whole restore path.

  LOADERS -- stock nodes, not SeedVR2-branded ones:
    UNETLoader -> seedvr2_7b_fp16.safetensors
                  feeds BOTH Apply SeedVR2 Conditioning AND the KSampler
    VAELoader  -> seedvr2_ema_vae_fp16.safetensors
                  feeds BOTH VAEEncodeTiled and VAEDecodeTiled
  These are SeedVR2's DiT and VAE, not H3's. Two VAELoaders in one graph is
  correct and expected -- do not try to share H3's video VAE here.

  THE SECOND INPUT ON POST-PROCESS. Post-Process SeedVR2 Output takes
  original_resized_images as well as images. Wire the Resize Image output to
  BOTH the pre-processor and that socket. It is the reference for colour
  matching; without it the node has nothing to match against.

  THERE IS NO resolution WIDGET. You upscale in pixel space FIRST and SeedVR2
  restores at whatever it is handed. From the 768 short edge:
      multiplier 1.875 -> 2520 x 1440     (recommended)
      multiplier 2.0   -> 2688 x 1536
  Use lanczos on the resize node.

  THE KSAMPLER IS NOT A DIAL. 1 step, cfg 1.0, euler, simple, denoise 1.0.
  This is the one-step formulation the model was trained for, not a starting
  point. Raising steps does not improve it.

  VAE TILING IS DELIBERATE. VAEEncodeTiled / VAEDecodeTiled, 512 tile / 128
  overlap. SeedVR2's VAE is the throughput bottleneck at 1440p, not the DiT.

  SPLIT / MERGE LATENT are the temporal-chunking pair and the replacement for
  the old batch_size + temporal_overlap dials. Bracket the KSampler with them
  for anything past a few seconds, and set the overlap to 3 to blend the
  chunk boundaries. Read the exact widget names off the node -- they live on
  the split node, not the sampler.

  COLOUR CORRECTION lives on Post-Process (color_correction_method) and
  defaults to none. Check what your build's dropdown actually offers.

  YOU DO NOT NEED A VRAM FLUSH NODE. The DiT loads through UNETLoader, so
  ComfyUI's own model manager owns it and evicts the 61.7 GB H3 DiT when it
  needs the room. The KJNodes VRAM-flush passthrough the old third-party pack
  required is unnecessary -- delete it if you have one wired in.

  ITERATE THIS AS A SEPARATE GRAPH. Restore is deterministic given input
  frames, so build a second workflow that loads frames from disk (VHS or
  LoadVideo) straight into the chain above. You can then A/B the multiplier,
  chunk size, and colour method without re-running H3 every time.

  STARTING POINT: load the built-in template "SeedVR2 3B Int8: Upscale Video"
  from the template library to read the wiring off a working graph. It is
  wrapped in a subgraph with VIDEO in and out, so enter the subgraph to get
  at the IMAGE-level chain rather than the video boundary.

INTERPOLATION (RIFE TensorRT)
  Order of operations: generate -> SeedVR2 -> RIFE -> encode. Interpolation
  is always last. RIFE synthesises intermediate frames from what it is given,
  so restoring after interpolating just asks SeedVR2 to reconstruct invented
  frames, and doubles its workload for nothing.

  Native H3 output is 24 fps. 2x -> 48, 2.5x -> 60. SET CreateVideo TO MATCH:
  RIFE doubles the frame count but audio duration is fixed, so leaving
  CreateVideo at 24 plays the video at half speed against correct-length
  audio.
  First run on a new instance stalls while TensorRT compiles the engine. That
  is expected, it is not a hang, and the cached engine is tied to the GPU
  architecture it was built on -- a different card type rebuilds from scratch.
  Interpolating a 5 s shot to 48 fps roughly doubles the frames the encoder
  writes; if you are already near the disk margin, encode before you batch.

=================== TROUBLESHOOTING ===================

IF UPDATES DO NOT SEEM TO LAND
  Read the [git] lines. Three things to look for, in order:
    1. "safe.directory configured" near the top. If that line says WARNING,
       every git call in the run is failing on dubious ownership and nothing
       is updating regardless of what else the log says.
    2. "detached HEAD -- looks image-pinned, leaving alone". That is v6 doing
       the right thing: the image pinned core deliberately. If you actually
       want to move it, set COMFY_PIN to a tag, or ALLOW_BRANCH_RECOVERY=1
       to track a branch (and read the note about master below first).
    3. "fast-forward blocked". You have local commits or a dirty tree in that
       checkout. GIT_FORCE_RESET=1 discards them.
  Also check the [comfy] git line: if it reports a large "+N commits" past the
  nearest tag, something has walked core off the release the image pinned.
  ComfyUI's own README warns that commits outside stable release tags may be
  very unstable and break many custom nodes, and template launch flags are
  written against the pinned release.

IF COMFYUI SEEMS SLOW TO START
  Check the [provisioning] comfyui line at the end of this log first. If it
  says CRASH-LOOPING, it is not slow -- it is being respawned every few
  seconds and there is no startup to wait for. Stop the service and run the
  printed command in the foreground; the traceback only appears there.
  Note that supervisor reporting RUNNING means only that the process survived
  five seconds, not that ComfyUI finished importing nodes.

IF CUSTOM NODES FAIL TO IMPORT
  Read the LAST line of the traceback, not the first. A pack that dies on
    ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
  is not itself broken: it imported diffusers, diffusers probed xformers,
  xformers found a half-installed flash_attn and took the flash path. This
  script detects and removes that on every boot -- look for the [flash] lines
  near the top of provisioning.log. If they say "healthy" or "not installed"
  and a pack still fails, the cause is something else.

  ComfyUI-Manager's SECURITY LEVEL does not cause import failures. It gates
  whether Manager may run install scripts or install from arbitrary git URLs.
  And "Try fix" only re-runs that pack's requirements.txt, which by
  definition cannot repair a broken package that is not listed in it. Reading
  the traceback beats toggling the security level every time.

IF MODEL LOADING IS PATHOLOGICALLY SLOW
  ComfyUI 0.30.x has a pinned-memory regression. Launch with
  --disable-pinned-memory.

IF NODES SHOW LINKS THAT DO NOT EXIST
  Frontend/backend skew on comfyui-frontend-package. This script force-syncs
  it to the pin in requirements.txt on every boot, so if it still happens the
  affected node's slot serialisation is stale in the saved workflow -- delete
  and re-add that node rather than rewiring it.

IF CUDA GOES MISSING AFTER AN UPDATE
  Something moved torch despite the constraints file. The [torch] block in
  this log will have said so loudly. Reinstall the pinned cu128 build for
  your card before rendering anything.

NOT IN THIS BUILD
  ref2va (reference-to-video) and controlnet_aux.
  H3-Regenerate-2K and H3-Context-IR remain API-only. SeedVR2 substitutes for
  the first. For the second there is a partial local option: lightx2v's
  MiniMax-H3 Prompt Rewriter LoRA, a Qwen3.6-27B adapter that expands a short
  prompt into a structured H3 prompt. Not fetched here -- it is another 27B
  model to host -- but prompt restructuring is no longer strictly API-only.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
