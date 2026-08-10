#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# MiniMax H3 (Hailuo 3.0) -- I2V BUILD  ---  v3
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          recommended (rate limits on a ~132 GB pull)
#        CIVITAI_TOKEN=<token>     only if you add Civitai entries below
#        COMFY_UPDATE=0            skip the ComfyUI git update (default: ON)
#        NODE_UPDATE=0             skip the custom-node git updates (default: ON)
#        GIT_FORCE_RESET=1         discard local commits blocking a fast-forward
#        WANT_TURBO=0              skip the Turbo LoRA draft tier (saves 1.5 GB)
#        WANT_SEEDVR2=0            skip the SeedVR2 restore stage (saves ~15 GB)
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY -- see below)
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~132 GB of weights. Provision a 180 GB volume minimum, 200 comfortable.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v2, AND WHY
#
#   * SCOPE CUT TO I2V. The ref2va checkpoint (-61.7 GB) is gone. The build
#     is now the fl2va checkpoint (which serves t2v, i2v, and first-last-frame
#     from one set of weights), its two VAEs, the bf16 text encoder, the Turbo
#     LoRA, and the SeedVR2 restore stage. No reference-to-video.
#
#   * SEEDVR2 RETAINED, and it is the largest quality lever in the build.
#     H3's local weights cap the short edge at 768px, H3-Regenerate-2K was
#     withheld from the release, and a temporal restorer is the only route
#     past that ceiling. Set WANT_SEEDVR2=0 to drop it and save ~15 GB.
#     NOTE: it is a custom node pack, so nothing in the stock H3 template
#     wires it up for you. See RESTORE STAGE in the notes at the end.
#
#   * COMFYUI NOW UPDATES ON EVERY BOOT, BY DEFAULT. v2 gated this behind
#     COMFY_AUTO_UPDATE=1 and only triggered it when the version was below
#     0.30.0, so in practice it never ran. It now runs unconditionally, with
#     three specific pieces of hardening -- see THE UPDATE PATH below.
#
#   * CUSTOM NODES NOW UPDATE ON EVERY BOOT TOO. v2 pulled only the two
#     packages in NODES_TRACK_HEAD and printed "present" for everything else,
#     which is how you end up on a six-week-old VideoHelperSuite against a
#     current frontend.
#
#   * ComfyUI-Olm-DragCrop ADDED. Interactive drag-to-crop on the image
#     preview. Relevant here because I2V wants its source still framed to the
#     native canvas aspect before it enters the graph, and doing that by
#     hand-computing pixel offsets is miserable. No extra pip dependencies.
#
#   * DROPPED: comfyui_controlnet_aux. Nothing in the I2V graph touches it.
#
#   * RIFE IS KEPT (ComfyUI-RIFE-TensorRT-Auto). Two things to know about it:
#     it pulls a TensorRT stack (tensorrt, onnx, polygraphy), which makes it
#     comfortably the most likely requirements.txt in this build to trip the
#     torch constraint guard -- if you see a constrained-install failure in
#     the log, look here first. And no weights are fetched for it: the node
#     builds its engine on first use. That build takes minutes and is
#     compiled for the exact GPU architecture it runs on, so it does not
#     survive a move to a different card type. Budget for paying it again on
#     every new instance shape.
#
# -----------------------------------------------------------------------------
# THE UPDATE PATH -- the three things that break unattended ComfyUI pulls
#
#   1. DETACHED HEAD. ai-dock images frequently check out a release tag rather
#      than a branch, and `git pull` on a detached HEAD fails with an error
#      that reads like a network problem. git_sync() detects it, resolves the
#      remote's default branch, and checks out properly before merging.
#
#   2. SHALLOW CLONE. A --depth=1 image has no tag history, so version
#      detection returns nothing and a fast-forward has no merge base.
#      git_sync() unshallows first.
#
#   3. TORCH GETTING CLOBBERED. This is the one that actually costs you the
#      instance. `pip install -r requirements.txt` -- ComfyUI's own, or any
#      custom node's -- can resolve a fresh torch off PyPI and replace a
#      pinned cu128 Blackwell build with a generic wheel. Every pip call that
#      touches a requirements file in this script runs against a constraints
#      file pinning the installed torch stack to its exact current versions,
#      local +cuXXX suffix included. If a constrained install fails, the
#      script retries unconstrained and then verifies CUDA is still alive,
#      loudly, rather than letting you discover it at first render.
#
#   FRONTEND/BACKEND SKEW. Separately from the git state: comfyui-frontend-
#   package, comfyui-workflow-templates, and comfyui-embedded-docs are pip
#   packages pinned by ComfyUI's requirements.txt, and a mismatch between the
#   frontend and the backend is what produces phantom link errors on nodes
#   with autogrow sockets (the ref_audio_0 class of bug) and stale built-in
#   templates. After the git update the script force-syncs all three to
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
#   RESTORE-> SeedVR2 7B fp16 to 1440 or 2160.
#   INTERP -> RIFE TensorRT, 24 -> 48/60 fps. Last stage before encode.
#
#   TURBO IS A DRAFT TIER, NOT A FAST FINAL -- but the reason has changed.
#   It is a step-distillation LoRA: it trains the model to take much larger
#   jumps along the flow trajectory so a handful of steps land where ~20
#   would. That is a fidelity trade by construction.
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
#   Apache-2.0 -- separate license, base weights still governed by the above.
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
echo "########## provisioning run (MiniMax H3 I2V v3): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

COMFY_UPDATE="${COMFY_UPDATE:-1}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_TURBO="${WANT_TURBO:-1}"
WANT_SEEDVR2="${WANT_SEEDVR2:-1}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} node_update=${NODE_UPDATE} turbo=${WANT_TURBO} seedvr2=${WANT_SEEDVR2}"

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
for p in ("torch", "torchvision", "torchaudio", "torchsde",
          "triton", "pytorch-triton", "xformers",
          "sageattention", "flash-attn"):
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

write_torch_pins

# ---------------------------------------------------------------------------
# Host memory check
#
# The bf16 pairing is the whole point of this build and it is RAM-gated, not
# VRAM-gated. Fail loudly here rather than after a 132 GB download.
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
# git_sync <repo_path> <label>
#
# Update a git checkout in the three states an ai-dock image actually hands
# you: shallow, detached, or a normal branch. Never destroys local work unless
# GIT_FORCE_RESET=1 is set explicitly.
# ---------------------------------------------------------------------------
git_sync() {
    local path="$1" label="$2" branch def c before after
    [[ -d "${path}/.git" ]] || { echo "[git] ${label}: not a git checkout, skipping update"; return 1; }

    before="$(git -C "$path" rev-parse --short HEAD 2>/dev/null || echo '?')"

    if [[ "$(git -C "$path" rev-parse --is-shallow-repository 2>/dev/null)" == "true" ]]; then
        echo "[git] ${label}: shallow clone -> unshallowing (needed for tags and merge base)"
        git -C "$path" fetch --unshallow --tags --prune 2>/dev/null \
            || git -C "$path" fetch --depth=2147483647 --tags --prune 2>/dev/null \
            || echo "[git] ${label}: unshallow failed, continuing anyway"
    fi

    git -C "$path" fetch --all --tags --prune || { echo "[git] ${label}: fetch FAILED"; return 1; }

    branch="$(git -C "$path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
    if [[ -z "$branch" ]]; then
        def="$(git -C "$path" remote show origin 2>/dev/null | awk '/HEAD branch/{print $NF}')"
        if [[ -z "$def" || "$def" == "(unknown)" ]]; then
            def="$(git -C "$path" rev-parse --abbrev-ref origin/HEAD 2>/dev/null | sed 's|^origin/||')"
        fi
        if [[ -z "$def" ]]; then
            for c in master main; do
                git -C "$path" show-ref --verify --quiet "refs/remotes/origin/${c}" && { def="$c"; break; }
            done
        fi
        [[ -z "$def" ]] && { echo "[git] ${label}: detached HEAD and no default branch resolvable"; return 1; }
        echo "[git] ${label}: detached HEAD -> checking out ${def}"
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
# Native H3 nodes landed in 0.30.0 (Comfy-Org/ComfyUI PR #15224). On an older
# image the graph opens with red nodes and no obvious cause, which is why this
# now runs by default instead of waiting to be asked.
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
echo "[comfy] before: $(comfy_version)"
if [[ "$COMFY_UPDATE" == "1" ]]; then
    git_sync "$COMFY" "ComfyUI"; rc=$?
    if (( rc == 0 )); then
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
    echo "[comfy] OK -- native H3 support present"
else
    echo "[comfy] !!! ${CV} is BELOW ${MIN_CV} -- the H3 nodes will not exist."
    echo "[comfy] !!! The update above did not take. Check the [git] lines, or pick"
    echo "[comfy] !!! a newer base image."
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>]
# Everything here is pulled on every boot when NODE_UPDATE=1 (the default).
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
)
[[ "$WANT_TURBO" == "1" ]] && NODES+=( "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo" )

# Restore stage. It needs this exact clone directory name -- its own docs and
# CLI paths assume custom_nodes/seedvr2_videoupscaler, not the repo basename.
[[ "$WANT_SEEDVR2" == "1" ]] && NODES+=( "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler|seedvr2_videoupscaler" )

# Optional: explicit dual-sigma sampler with manual video/audio shift control
# (12 / 3) and a 50-sigma-point schedule. Genuinely more control than the
# stock graph, BUT it overlaps the native H3 nodes and the Turbo nodes, and
# three H3 node packs in one install is a recipe for node-name collisions.
# Add it deliberately, after the stock path works.
#   "https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3"

install_node() {
    local spec="$1" url name path rc
    url="${spec%%|*}"
    if [[ "$spec" == *"|"* ]]; then name="${spec##*|}"; else name="$(basename "$url" .git)"; fi
    path="${NODES_DIR}/${name}"

    if [[ -d "$path" ]]; then
        if [[ "$NODE_UPDATE" == "1" ]]; then
            git_sync "$path" "$name"; rc=$?
            if (( rc == 2 )); then
                return 0    # unchanged, requirements already satisfied
            fi
        else
            echo "[node] $name present (NODE_UPDATE=0)"
            return 0
        fi
    else
        echo "[node] cloning $name"
        git clone --recursive "$url" "$path" || { echo "[node] CLONE FAILED: $name"; return 0; }
    fi

    pip_reqs "${path}/requirements.txt" "$name"
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
# Manifest  --  ~132 GB total
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

    # === VAEs -- BOTH REQUIRED (4.9 + 0.6 GB) ============================
    # The audio VAE is not optional even for silent output: the audio stream
    # is generated in the same pass and has to be decoded. Missing it is the
    # usual cause of "my output has no sound" -- along with a missing
    # VAEDecodeAudio node feeding SaveVideo.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"
)

# === TURBO LoRA -- DRAFT TIER (~1.5 GB, two files at ~744 MB each) =======
# Iterating a prompt at 6-8 steps instead of 25 is worth a lot while you are
# still deciding what the shot is.
#
# BOTH checkpoints are fetched deliberately -- they are not redundant:
#   v4_step600_ema   DEFAULT. Current best. Much better static and small-
#                    motion shots, better micro-detail on faces/fingers/fine
#                    texture, and the over-sharpening + plastic look of the
#                    v1 (~850) line is resolved. Tolerates high step counts
#                    better than v1. Its one weakness is motion-smear and
#                    trailing ghosting at 4 steps with large fast motion.
#   4step_ema_ckpt850  FALLBACK for exactly one case: 4 steps AND heavy
#                    motion, where v1 is still the friendlier pick. Do not
#                    use it otherwise -- it is the over-sharpened line.
#
# One LoRA file covers every base. The node auto-detects a pruned base and
# re-injects time-conditioning at run time, so bf16 and pruned both work.
if [[ "$WANT_TURBO" == "1" ]]; then
MODELS+=(
    "hf|$LORA|minimax_h3_turbo_v4_step600_ema.safetensors|larryvrh/MiniMax-H3-Turbo-Lora|minimax_h3_turbo_v4_step600_ema.safetensors"
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
    local margin=$(( 15 * 1024*1024*1024 ))   # bf16 loads spill to disk
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Cheapest give-backs, in the order you should make them:"
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
echo "[provisioning] NOTE: full build is ~132 GB. First boot on a fresh volume"
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
          ComfyUI-RIFE-TensorRT-Auto ComfyUI-MiniMax-H3-Turbo seedvr2_videoupscaler; do
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
  That matters here because this script now updates ComfyUI on every boot:
  the node absorbs the change, so keep it updated and do not pin either side.

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

RESTORE STAGE (SeedVR2) -- YOU MUST WIRE THIS IN YOURSELF
  Nothing ships pre-wired. The built-in H3 template is core nodes only --
  templates cannot depend on packages that may not be installed -- so
  SeedVR2, RIFE, and DragCrop are all absent from it, as they are from the
  Turbo workflow JSON this script downloads.

  Insertion point is between VAEDecode and CreateVideo:

    VAEDecode ---IMAGE--> SeedVR2 Video Upscaler ---IMAGE--> RIFE ---IMAGE--\
                                                                             > CreateVideo -> SaveVideo
    VAEDecodeAudio ------------------------------------------AUDIO----------/

  Nodes to add:
    SeedVR2 Load DiT Model   -> seedvr2_ema_7b_fp16.safetensors   (NOT fp8)
    SeedVR2 Load VAE Model   -> ema_vae_fp16.safetensors
    SeedVR2 Video Upscaler   -> takes the decoded IMAGE batch plus both
                                loaders; resolution 1440 or 2160
    SeedVR2 Torch Compile Settings  (optional: max-autotune / inductor.
                                     Pure speed, no quality cost.)

  batch_size must follow 4n+1 (5, 9, ..., 81, ...). It is the temporal-
  consistency dial, not a throughput knob: it is the window the restorer sees
  at once, so larger means less drift between segments. At 96 GB push it high
  -- 81 is a sane starting point. Ideally match it to shot length.

  Audio bypasses the restorer entirely. Keep VAEDecodeAudio wired straight to
  the save node; do not try to route it through SeedVR2.

  Do NOT apply the downscale-first trick here. That exists to give a restorer
  clean input from dirty compressed footage; H3 output at native canvas is
  already clean, so feed it directly.

INTERPOLATION (RIFE TensorRT)
  Order of operations: generate -> SeedVR2 -> RIFE -> encode. Interpolation
  is always last. RIFE synthesises intermediate frames from what it is given,
  so restoring after interpolating just asks SeedVR2 to reconstruct invented
  frames, and doubles its workload for nothing.

  Native H3 output is 24 fps. 2x -> 48, 2.5x -> 60.
  First run on a new instance stalls while TensorRT compiles the engine. That
  is expected, it is not a hang, and the cached engine is tied to the GPU
  architecture it was built on -- a different card type rebuilds from scratch.
  Interpolating a 5 s shot to 48 fps roughly doubles the frames the encoder
  writes; if you are already near the disk margin, encode before you batch.

IF MODEL LOADING IS PATHOLOGICALLY SLOW
  ComfyUI 0.30.x has a pinned-memory regression. Launch with
  --disable-pinned-memory.

IF NODES SHOW LINKS THAT DO NOT EXIST
  Frontend/backend skew on comfyui-frontend-package. This script force-syncs
  it to the pin in requirements.txt on every boot, so if it still happens the
  affected node's slot serialisation is stale in the saved workflow -- delete
  and re-add that node rather than rewiring it.

NOT IN THIS BUILD
  ref2va (reference-to-video) and controlnet_aux.
  H3-Regenerate-2K and H3-Context-IR remain API-only. SeedVR2 substitutes for
  the first. For the second there is now a partial local option that did not
  exist at v2: lightx2v's MiniMax-H3 Prompt Rewriter LoRA, a Qwen3.6-27B
  adapter that expands a short prompt into a structured H3 prompt. Not
  fetched here -- it is another 27B model to host -- but it is no longer
  true that prompt restructuring has no local path.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
