#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# ByteDance Bernini-R -- REFERENCE-GUIDED VIDEO EDITING (rv2v) BUILD  ---  v1
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          recommended (rate limits on a ~75 GB pull)
#        COMFY_UPDATE=0            skip the ComfyUI git update (default: ON)
#        NODE_UPDATE=0             skip the custom-node git updates (default: ON)
#        GIT_FORCE_RESET=1         discard local commits blocking a fast-forward
#        WANT_FP8=1                also fetch the fp8_scaled expert pair (+31 GB)
#        WANT_FULL_BERNINI=1       add the FULL Bernini renderer swap (+31 GB)
#        WANT_1_3B=1               add the 1.3B single-expert draft model (+5 GB)
#        WANT_PROMPT_ENHANCER=1    add RH-Bernini prompt-enhancer nodes
#        WANT_BERNINIR_PACK=1      add neuregex/ComfyUI-BerniniR (APG path)
#        WANT_RIFE=1               add RIFE TensorRT for 16->24 fps finishing
#        FIX_FLASH_ATTN=0          skip the half-installed flash-attn guard
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY -- see below)
#        MANAGER_SECURITY=weak     set ComfyUI-Manager security_level
#                                  (strong|normal|normal-|weak; unset = leave alone)
#        MANAGER_DOWNGRADE_BLACKLIST=torch,torchvision,torchaudio
#                                  stop Manager pip-downgrading the torch stack
#                                  (recommended even if you never lower security)
#        COMFY_PIN=<sha|tag>       hold core at this ref (see UPDATE PATH below)
#        ALLOW_BRANCH_RECOVERY=1   permit moving a detached HEAD onto a branch
#        FORCE_DEPS=1              reinstall pip requirements even if unchanged
#        RESTART_COMFY_ON_UPDATE=1 bounce the comfyui service if code changed
#        HF_PATH_RESOLVE=0         disable the repo-path resolver (see below)
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~75 GB base. 120 GB volume minimum, 150 comfortable. Every optional
#   tier above adds to that -- the preflight prints the real number.
#
# -----------------------------------------------------------------------------
# WHAT THIS IS
#
#   Bernini is ByteDance's unified video generation/editing framework: an
#   MLLM-based semantic planner (Qwen2.5-VL) feeding a DiT-based renderer built
#   on Wan2.2-T2V-A14B. Paper arXiv:2605.22344, Apache-2.0.
#
#   WHAT COMFYUI ACTUALLY RUNS IS THE RENDERER ONLY. ByteDance open-sourced
#   Bernini-R -- the renderer trained to work standalone -- and Kijai added
#   in-context conditioning support to core in Comfy-Org/ComfyUI PR #14216.
#   The planner is not in this path and this script does not pretend otherwise.
#   That gap is the single largest quality lever in the build and it is item 1
#   below.
#
#   Target task is rv2v: source video + reference image(s) -> reference-guided
#   video editing. The same weights also serve v2v, r2v, ads2v, and the image
#   tasks; nothing extra to fetch for those.
#
# -----------------------------------------------------------------------------
# LINEAGE -- WHAT WAS INHERITED AND WHAT IS NEW
#
#   The update machinery here is lifted wholesale from the MiniMax H3 v6/v7
#   build and is not reinvented: safe.directory, the unshallow/ff-only git_sync
#   with 0/1/2 return codes, the detached-HEAD-is-deliberate rule, the torch
#   constraints file, the requirements-hash gate, the frontend/backend pin sync,
#   the flash-attn guard, the disk preflight, the supervisor log discovery, and
#   the double-pid crash-loop check. Those lessons cost someone a lot of boots
#   and none of them are model-specific. They are carried over unchanged in
#   behaviour.
#
#   THREE THINGS ARE NEW, all forced by this model rather than by taste:
#
#   1. HF REPO-PATH RESOLVER. The H3 build fetched everything from one
#      Comfy-Org repo with a stable layout. This build pulls from four repos,
#      and one of them -- Kijai/WanVideo_comfy -- reorganises constantly. As of
#      writing, ComfyUI's own Bernini-R documentation links the lightx2v LoRA at
#      the repo ROOT while the file actually lives under Lightx2v/. A hardcoded
#      repo_path is therefore a guaranteed future 404, and the failure is silent
#      in the sense that you get a MISSING line at the end of a 40-minute pull
#      rather than an error when it matters.
#
#      dl_hf now takes a PIPE-SEPARATED CANDIDATE LIST for the repo path, and
#      if every candidate misses it calls list_repo_files() and locates the file
#      by basename anywhere in the repo. It prints the path it resolved to, so
#      the log tells you what to hardcode next time. Set HF_PATH_RESOLVE=0 to
#      turn the fallback off and fail fast instead.
#
#   2. CAPABILITY DETECTION INSTEAD OF A VERSION FLOOR. The H3 build checked
#      comfyui_version >= 0.30.0 because PR #15224 landed in a known release.
#      I could not establish which release PR #14216 shipped in, and inventing a
#      number would be worse than useless -- it would either block a working
#      install or wave through a broken one. This script looks for the Bernini
#      node module in comfy_extras/ and the registered node class instead. That
#      is the thing you actually care about, and it does not go stale.
#
#   3. NODE-PACK COLLISION GUARD. There are at least five community Bernini
#      packs and several of them BACKPORT PR #14216's runtime patches. Once core
#      has native support, a backport pack is patching code that already does
#      the thing, and you get duplicate node names or a patched-twice runtime.
#      Every Bernini pack here is opt-in, and the script refuses to be quiet
#      about installing a backport alongside native support.
#
# -----------------------------------------------------------------------------
# THE UPDATE PATH -- the things that break unattended ComfyUI pulls
#
#   1. DUBIOUS OWNERSHIP. ai-dock provisions as root; a persistent volume is
#      often owned by another uid; modern git then refuses every operation with
#      "detected dubious ownership" and git_sync's error paths absorb it, so
#      updates become silent no-ops while the log still reads plausibly. Fixed
#      unconditionally, before the first git call.
#
#   2. DETACHED HEAD. Images pin core to a release tag ON PURPOSE. `git pull` on
#      a detached HEAD exits 0 and does nothing. Detached-with-no-pin is LEFT
#      ALONE and logged. Pass COMFY_PIN to move it deliberately.
#
#   3. SHALLOW CLONE. No tag history, no merge base. git_sync unshallows first.
#
#   4. TORCH GETTING CLOBBERED. This is the one that costs you the instance, and
#      it is WORSE on this build than on H3. See the BLACKWELL section: a
#      generic torch wheel resolved off PyPI has no sm_120 kernels at all, so
#      the failure is not "slower", it is "CUDA unavailable". Every pip call
#      touching a requirements file runs against a constraints file pinning the
#      installed torch stack including its local +cuXXX suffix.
#
#   FRONTEND/BACKEND SKEW. comfyui-frontend-package, comfyui-workflow-templates
#   and comfyui-embedded-docs are pinned by ComfyUI's requirements.txt. Skew
#   produces phantom link errors on autogrow sockets and stale built-in
#   templates -- and the Bernini-R templates ARE built-in templates, so a stale
#   comfyui-workflow-templates is a real way to "not have" the workflow this
#   whole script exists to run. All three are force-synced after a core update.
#
# -----------------------------------------------------------------------------
# WHERE THE QUALITY IS, IN THIS BUILD (ranked, largest lever first)
#
#   1. THE MISSING PLANNER, AND WHAT SUBSTITUTES FOR IT. Bernini's whole
#      argument is that an MLLM should do semantic planning before the DiT
#      renders anything. ComfyUI has the renderer and not the planner, so the
#      structured semantic target the renderer was trained to receive is simply
#      absent. ByteDance's own inference code recommends --use_pe (prompt
#      enhancement through a vision-capable LLM) for best quality even WITH the
#      planner present. Without it, a good enhanced prompt is not a nicety, it
#      is the closest available stand-in for the missing stage.
#
#      WANT_PROMPT_ENHANCER=1 installs RH-Bernini's enhancer nodes, which carry
#      the official per-task templates and emit system/user prompts for an
#      external LLM node plus a parser for the response. Point it at any
#      OpenAI-compatible endpoint. Send it your reference images if the LLM node
#      supports vision -- the templates are written expecting the model can see
#      what image0 and image1 actually are.
#
#      This is first on the list because it is the only item that addresses a
#      MISSING COMPONENT rather than a precision or step-count trade.
#
#   2. THE DISTILL LoRA, AND BYPASSING IT. The stock Bernini-R template ships
#      with lightx2v_T2V_14B_cfg_step_distill_v2 wired in. That is a
#      step-distillation LoRA: it trains the model to take much larger jumps
#      along the flow trajectory so a handful of steps land where ~30 would, and
#      it distills away the CFG pass as well. Both are fidelity trades by
#      construction. It is fetched here because it makes iteration bearable, and
#      it is BYPASSED for finals. If you only change one thing about the stock
#      template, change this one.
#
#   3. RESOLUTION AND FRAME COUNT -- AND THESE TRADE AGAINST EACH OTHER, NOT
#      AGAINST NOTHING. ByteDance's own high-quality rv2v case runs
#      --num_frames 121 --fps 24 --max_image_size 1280. That is 720p/24fps/5s
#      and it is the configuration to beat, not a demo setting.
#
#      Wan's training horizon is 81 frames; past it, dense attention tends
#      toward frozen or looping motion. 81@16fps and 121@24fps are both ~5s of
#      content -- the horizon is a DURATION, and fps is how you spend it.
#
#      rv2v is tighter than t2v on top of that: the source video is VAE-encoded
#      and injected as in-context tokens ALONGSIDE the generated latent, so the
#      sequence is roughly double a t2v of the same length, plus reference
#      tokens. At 1280x720 that is ~3,600 tokens per latent frame, so 121 frames
#      is ~220k tokens of context and 241 frames is ~440k. Attention is
#      quadratic in that. Ten seconds in one pass is not 2x the cost of five, it
#      is ~4x, and it is also 2x outside the training window.
#
#   4. TEXT ENCODER PRECISION. The stock template specifies umt5_xxl fp8_scaled.
#      This build fetches fp16 (11.4 GB) instead. UMT5-XXL is an encoder, not a
#      generator, so this is a smaller lever than 1-3 -- what you notice is
#      prompt-adherence drift on long multi-clause edit instructions, not
#      per-frame fidelity. But at 96 GB the fp16 copy costs you nothing you were
#      using, so there is no reason to take the fp8.
#
#   5. DIFFUSION PRECISION. fp16 expert pair (~31 GB each) over fp8_scaled
#      (~15.5 GB each) over int8_convrot. Real, and at 96 GB free. Note that
#      Comfy-Org also publishes mxfp8 and int8_convrot variants of both experts;
#      those exist for cards that cannot hold the pair, which is not your
#      problem.
#
# -----------------------------------------------------------------------------
# THE LADDER
#
#   DRAFT   -> fp8_scaled pair (or 1.3B), lightx2v LoRA ON, 4-8 steps, 480p,
#              81 frames @ 16fps, Sage on. Settle PROMPT, REFERENCE FRAMING and
#              TASK TYPE here. Cheap and fast, and unlike a text-to-video draft
#              it is fairly faithful -- the source video pins composition for
#              you, so a low-res draft is still telling you the truth about
#              whether the edit lands.
#   FINAL   -> fp16 pair, fp16 encoder, LoRA BYPASSED, real CFG, 30-40 steps
#              split across the two experts, 1280x720, 121 frames @ 24fps,
#              no Sage.
#   SEGMENT -> anything past ~5s. Two 121-frame passes, same seed, same prompt,
#              same reference images, a few frames of overlap. See the LENGTH
#              section in the operating notes for why this is much safer for an
#              EDIT than the equivalent chaining is for a generation.
#   FINISH  -> optional RIFE if you shot 16fps for token budget and want 24.
#
#   THE DRAFT TIER DOES NOT TRANSFER SEEDS. Changing precision, sampler, step
#   count or CFG changes the integration trajectory. The same seed lands
#   somewhere related, not identical. Drafts settle prompt and framing; expect
#   to re-roll seeds at final settings.
#
#   DO NOT PUSH THE DISTILL LoRA PAST ~8 STEPS expecting base-model behaviour.
#   Distillation memorises a schedule. If you want more steps, drop the LoRA.
#
# -----------------------------------------------------------------------------
# BLACKWELL / sm_120 -- READ THIS BEFORE YOU DEBUG ANYTHING ELSE
#
#   This build is written for an RTX PRO 6000 Blackwell (96 GB, sm_120).
#
#   THE OFFICIAL BYTEDANCE REPO IS THE WRONG PATH ON THIS CARD, and not because
#   of preference. bytedance/Bernini pins torch==2.5.1+cu124 and recommends a
#   Hopper GPU so FlashAttention-3 can be used. torch 2.5.1+cu124 ships no
#   sm_120 kernels, and the FA3 hopper build is Hopper-only, so you would be
#   unpinning the environment and rebuilding attention before generating a
#   single frame. ComfyUI's native path avoids all of it. This script does not
#   clone bytedance/Bernini and does not install its requirements.
#
#   ATTENTION: FA3 is unavailable to you. SageAttention 2.2 has sm_120 wheels
#   but they must match your torch minor version and CUDA line EXACTLY, and
#   there is a well-documented failure where the GLOBAL --use-sage-attention
#   flag produces BLACK OUTPUT on Wan-family models. That is why KJNodes is a
#   non-optional dependency here rather than a convenience: use its
#   "Patch Sage Attention KJ" node after the model loader, and leave the launch
#   flag off. If you see black frames, this is the first thing to check and it
#   is almost always this.
#
#   DO NOT let anything install xformers. It force-downgrades torch through its
#   dependency chain. The constraints file guards this; the guard is not a
#   reason to test it.
#
# -----------------------------------------------------------------------------
# MEMORY
#
#   fp16 high expert (~31) + fp16 low expert (~31) + fp16 UMT5 (11.4) = ~73 GB
#   of weights against 96 GB of VRAM. Unlike the H3 build this is NOT a
#   knife-edge: the text encoder runs once up front and is evicted before
#   sampling, so peak residency is the expert pair plus activations, and both
#   experts can stay resident across the high/low handoff. That handoff staying
#   in VRAM is most of the reason to buy the 96 GB card for this model.
#
#   Budget 64 GB system RAM minimum so the evicted encoder lands in RAM rather
#   than being re-read from disk each run. 128 GB is comfortable.
#
#   Launch with --highvram. If model loading is pathologically slow, ComfyUI
#   0.30.x has a pinned-memory regression: add --disable-pinned-memory.
#
# -----------------------------------------------------------------------------
# VERIFY BEFORE YOUR FIRST BOOT -- things I could not confirm
#
#   Stated plainly rather than buried, because a wrong assumption here costs a
#   pull cycle:
#
#   * fp16 expert sizes (~31 GB each) are DERIVED from the published fp8_scaled
#     sizes (~15.5 GB), not read off the repo. The preflight queries the real
#     remote sizes at runtime, so the disk check is correct regardless; only the
#     header figure is an estimate.
#   * The ComfyUI release that first shipped native Bernini-R nodes. Detected at
#     runtime instead of asserted -- see CAPABILITY DETECTION above.
#   * attashe/Bernini-Wan2.2-fp8-scaled's in-repo file paths. The model card
#     lists bare filenames, which usually means repo root; the resolver covers
#     it if that is wrong.
#   * Whether the FULL Bernini renderer actually beats Bernini-R inside ComfyUI.
#     It was jointly trained WITH the planner that ComfyUI does not run, so it
#     may be out of distribution here. Bernini-R is the variant explicitly
#     trained to stand alone. WANT_FULL_BERNINI is an A/B, not an upgrade, and
#     it is off by default for that reason.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (Bernini-R rv2v v1): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

# ---------------------------------------------------------------------------
# All toggles read from the environment with documented defaults.
# (The H3 v5 lesson: never assign literals here. Documenting an env var and
# then overwriting it means the instance setting silently does nothing.)
# ---------------------------------------------------------------------------
COMFY_UPDATE="${COMFY_UPDATE:-1}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_FP8="${WANT_FP8:-0}"
WANT_FULL_BERNINI="${WANT_FULL_BERNINI:-0}"
WANT_1_3B="${WANT_1_3B:-0}"
WANT_PROMPT_ENHANCER="${WANT_PROMPT_ENHANCER:-0}"
WANT_BERNINIR_PACK="${WANT_BERNINIR_PACK:-0}"
WANT_RIFE="${WANT_RIFE:-0}"
ALLOW_BRANCH_RECOVERY="${ALLOW_BRANCH_RECOVERY:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-0}"
HF_PATH_RESOLVE="${HF_PATH_RESOLVE:-1}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} node_update=${NODE_UPDATE} fp8=${WANT_FP8} full_bernini=${WANT_FULL_BERNINI} 1.3b=${WANT_1_3B}"
echo "[provisioning] enhancer=${WANT_PROMPT_ENHANCER} berninir_pack=${WANT_BERNINIR_PACK} rife=${WANT_RIFE} path_resolve=${HF_PATH_RESOLVE}"
echo "[provisioning] branch_recovery=${ALLOW_BRANCH_RECOVERY} force_deps=${FORCE_DEPS} restart=${RESTART_COMFY_ON_UPDATE} pin=${COMFY_PIN:-<none>}"
echo "[provisioning] manager_security=${MANAGER_SECURITY:-<leave alone>} downgrade_blacklist=${MANAGER_DOWNGRADE_BLACKLIST:-<unset>}"

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
# safe.directory -- must run before the first git invocation.
# Without it every git call fails on dubious ownership, git_sync's error paths
# absorb each failure individually, and the run "completes" having updated
# nothing while the log still looks normal.
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
# The flash_attn directory is present enough that `import flash_attn` succeeds
# (often as a namespace package -- a directory with no __init__.py, the residue
# of a failed source build), so xformers concludes flash attention is available,
# takes that path, and dies on the submodule. Everything downstream of diffusers
# dies with it.
#
# Cleanly absent is FINE. The fix is removal, not repair. NOTHING IN THIS BUILD
# USES FLASH-ATTN -- FA3 is Hopper-only and unavailable on sm_120 anyway.
#
# Must run before write_torch_pins(): a half-installed package still carries
# dist-info metadata, so it would otherwise be pinned into the constraints file
# and held in place for the whole run.
# ---------------------------------------------------------------------------
fix_flash_attn() {
    local out state pkgdir

    out="$("$PY" - <<'PYEOF'
import os, sys
try:
    import flash_attn
except Exception:
    print("ABSENT\t"); sys.exit(0)

paths = list(getattr(flash_attn, "__path__", []) or [])
d = paths[0] if paths else os.path.dirname(getattr(flash_attn, "__file__", "") or "")
try:
    import flash_attn.flash_attn_interface  # noqa: F401
    print("HEALTHY\t%s" % d)
except Exception:
    print("BROKEN\t%s" % d)
PYEOF
)" || out="ABSENT\t"

    state="${out%%$'\t'*}"
    pkgdir="${out#*$'\t'}"

    case "$state" in
        ABSENT)  echo "[flash] not installed -- fine, nothing here needs it" ;;
        HEALTHY) echo "[flash] healthy (${pkgdir})" ;;
        BROKEN)
            echo "[flash] BROKEN half-install detected at ${pkgdir}"
            echo "[flash] removing it -- xformers falls back to its own kernels"
            "$PY" -m pip uninstall -y flash-attn 2>/dev/null || true
            [[ -n "$pkgdir" && -d "$pkgdir" ]] && rm -rf "$pkgdir"
            for meta in "$(dirname "$pkgdir")"/flash_attn-*.dist-info; do
                [[ -d "$meta" ]] && rm -rf "$meta"
            done
            echo "[flash] removed"
            ;;
        *) echo "[flash] indeterminate state, leaving alone" ;;
    esac
}

# ---------------------------------------------------------------------------
# Torch stack guard
#
# Snapshot the exact installed versions of the fragile CUDA cluster into a pip
# constraints file, local +cuXXX suffix included. Every requirements.txt install
# runs with -c "$CONSTRAINTS". A constrained install that FAILS is the correct
# outcome: it means something genuinely wanted to move torch and you want to
# know that -- on sm_120 a generic PyPI wheel does not degrade performance, it
# removes CUDA entirely.
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
# Returns 0 when the file is new or its content hash moved. Without this, every
# boot spends minutes re-resolving dependencies and gives itself another chance
# to break a working environment.
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
        cap = torch.cuda.get_device_capability(0)
        print("[torch] vram: %.1f GB total | compute capability sm_%d%d" % (
            total / 1024**3, cap[0], cap[1]))
        archs = torch.cuda.get_arch_list()
        print("[torch] built for: %s" % ", ".join(archs))
        want = "sm_%d%d" % (cap[0], cap[1])
        if want not in archs:
            print("[torch] !!! THIS WHEEL HAS NO KERNELS FOR %s." % want)
            print("[torch] !!! Blackwell needs a cu128+ build. A generic PyPI torch")
            print("[torch] !!! will import fine and then fail at the first kernel.")
    else:
        print("[torch] !!! CUDA UNAVAILABLE. If this worked before an update, the")
        print("[torch] !!! torch wheel was replaced. Reinstall the pinned cu128+ build")
        print("[torch] !!! for your card before rendering anything.")
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

echo "=================== HOST MEMORY ==================="
RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$RAM_GB" ]]; then
    echo "[mem] system RAM: ${RAM_GB} GB"
    if (( RAM_GB < 64 )); then
        echo "[mem] !!! Under 64 GB. The fp16 text encoder cannot stay resident once"
        echo "[mem] !!! evicted and gets re-read from disk on every run. Either move to"
        echo "[mem] !!! a higher-RAM machine, or switch the encoder to fp8_scaled in the"
        echo "[mem] !!! manifest below (it is the cheapest give-back in this build)."
    elif (( RAM_GB < 128 )); then
        echo "[mem] adequate. 128 GB+ would give comfortable offload headroom."
    else
        echo "[mem] comfortable."
    fi
fi

# ---------------------------------------------------------------------------
# git_sync <repo_path> <label> [pin]
# Return codes:  0 = HEAD moved   1 = could not update   2 = already current
#
# A detached HEAD is NOT "repaired" by default -- images pin core to a release
# tag deliberately, and walking it onto master breaks custom nodes and the
# launch flags a template was written against.
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
# comfyui-workflow-templates matters more here than in most builds: the
# Bernini-R templates ARE built-in templates, so a stale package is
# indistinguishable from "the workflow does not exist".
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

echo "[comfy] after: $(comfy_version)"

# ---------------------------------------------------------------------------
# CAPABILITY DETECTION -- does this core actually have Bernini support?
#
# Deliberately NOT a version-number floor. I could not establish which ComfyUI
# release PR #14216 shipped in, and a wrong floor either blocks a working
# install or waves through a broken one. Look for the thing itself.
#
# Checks, in order of how much they prove:
#   1. comfy_extras/nodes_bernini.py on disk (what the PR added)
#   2. "bernini" appearing in the workflow-templates package (the built-in
#      Bernini-R templates this build is designed around)
# ---------------------------------------------------------------------------
echo "=================== BERNINI SUPPORT CHECK ==================="
BERNINI_NATIVE=0
if compgen -G "${COMFY}/comfy_extras/*bernini*" > /dev/null 2>&1; then
    BERNINI_NATIVE=1
    for f in "${COMFY}"/comfy_extras/*bernini*; do echo "[bernini] core module: ${f#${COMFY}/}"; done
else
    echo "[bernini] !!! No bernini module found in comfy_extras/."
    echo "[bernini] !!! Native Bernini-R conditioning (Comfy-Org/ComfyUI PR #14216) is"
    echo "[bernini] !!! NOT present in this core. The weights will download fine and the"
    echo "[bernini] !!! workflow will open with red nodes."
    echo "[bernini] !!! Fix: let COMFY_UPDATE=1 run (it is the default), or move to a"
    echo "[bernini] !!! newer base image. If core is detached at an old tag, that pin is"
    echo "[bernini] !!! why -- see the [git] lines above."
fi

TPL_DIR="$("$PY" -c "import comfyui_workflow_templates,os;print(os.path.dirname(comfyui_workflow_templates.__file__))" 2>/dev/null || echo "")"
if [[ -n "$TPL_DIR" && -d "$TPL_DIR" ]]; then
    _n="$(find "$TPL_DIR" -iname '*bernini*' 2>/dev/null | wc -l)"
    if (( _n > 0 )); then
        echo "[bernini] built-in templates: ${_n} bernini file(s) present"
    else
        echo "[bernini] WARNING: no bernini templates in comfyui-workflow-templates."
        echo "[bernini] WARNING: Search the Template Library for 'Bernini-R'; if it is"
        echo "[bernini] WARNING: absent, that package is stale -- see the [deps] lines."
    fi
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>][@<pin>]
#
# WHAT IS AND IS NOT HERE, AND WHY:
#   VideoHelperSuite -- load/save video, frame extraction. Non-negotiable.
#   KJNodes          -- NOT optional on Blackwell. "Patch Sage Attention KJ" is
#                       the supported way to use Sage without the global flag
#                       that black-frames Wan models. Also carries torch.compile
#                       and resolution helpers.
#
# Everything Bernini-branded is OPT-IN because the packs overlap each other AND
# overlap core. See the collision guard below.
# ---------------------------------------------------------------------------
NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-KJNodes"
)

# Prompt-enhancer nodes: official per-task Bernini templates -> external LLM ->
# parser. This is the stand-in for the missing MLLM planner and it is the single
# highest-value optional addition in the build. NOTE it also backports PR #14216
# runtime patches -- see the guard below.
[[ "$WANT_PROMPT_ENHANCER" == "1" ]] && NODES+=( "https://github.com/RH-RunningHub/ComfyUI-RH-Bernini" )

# Alternative sampling path: reimplements Bernini's own source-id RoPE and
# multi-condition APG guidance, switching experts by timestep. Arguably closer
# to the paper than core's conditioning. Add it AFTER the stock template works,
# so you have something to A/B against.
[[ "$WANT_BERNINIR_PACK" == "1" ]] && NODES+=( "https://github.com/neuregex/ComfyUI-BerniniR" )

# Frame interpolation, for finishing a 16fps token-budget render at 24. Pulls
# the TensorRT stack and builds its engine on first use, per GPU architecture.
[[ "$WANT_RIFE" == "1" ]] && NODES+=( "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto" )

# Deliberately NOT in the list:
#   AIMixer/ComfyUI-Bernini   -- standalone WanVideoWrapper-derived engine with
#     a timeline "director" for segmented long-video edits. Genuinely useful for
#     the >5s case, but it is a SECOND full inference path that duplicates core.
#     Add it on purpose, alone, once the stock path is known-good.
#   CCpt5/ComfyUI-BerniniStudio -- single-node wrapper + Ollama enhancement.
#     Overlaps RH-Bernini. Pick one enhancer, not both.
#   bytedance/Bernini itself  -- see the BLACKWELL section. Its pinned
#     torch==2.5.1+cu124 has no sm_120 kernels. Do not let it near this venv.

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
# Bernini node-pack collision guard
#
# Several community packs BACKPORT PR #14216's runtime patches so they can be
# used before the PR reached the target runtime. Once core has native support,
# a backport is patching code that already does the thing. Symptoms are
# duplicate node names in the search, a graph that loads with the wrong node
# bound, or a double-patched sampler that behaves subtly differently from the
# template it was built for -- none of which announce themselves.
# ---------------------------------------------------------------------------
BACKPORT_PACKS=( ComfyUI-RH-Bernini ComfyUI-Bernini ComfyUI-BerniniStudio )
_present=()
for p in "${BACKPORT_PACKS[@]}"; do
    [[ -d "${NODES_DIR}/${p}" ]] && _present+=( "$p" )
done
if (( ${#_present[@]} > 0 )); then
    echo "=================== BERNINI PACK GUARD ==================="
    echo "[packs] Bernini-related packs installed: ${_present[*]}"
    if (( BERNINI_NATIVE == 1 )); then
        echo "[packs] NOTE: core already has native Bernini support. These packs"
        echo "[packs] NOTE: backport the same runtime patches. Use them for their"
        echo "[packs] NOTE: PROMPT-ENHANCER nodes and leave the conditioning to core."
        echo "[packs] NOTE: If a graph misbehaves, check which pack a Bernini node"
        echo "[packs] NOTE: actually came from before you start tuning anything."
    fi
    if (( ${#_present[@]} > 1 )); then
        echo "[packs] !!! MORE THAN ONE Bernini pack is installed. Node-name"
        echo "[packs] !!! collisions are near-certain. Keep one and remove the rest."
    fi
fi

# ---------------------------------------------------------------------------
# ComfyUI-Manager configuration  --  security level and downgrade protection
#
# WHY THIS IS OPT-IN RATHER THAN JUST SET: "deliberate beats automatic" is the
# rule the rest of this script follows, and this one lowers a security control.
# Set MANAGER_SECURITY=weak once in the Vast template environment and it applies
# on every boot forever, which is the automation you actually want -- without
# the script silently weakening an instance nobody asked it to.
#
# WHAT THE LEVEL ACTUALLY GATES. Not node imports. A pack that fails to import
# fails for a Python reason and the traceback says which -- see the
# troubleshooting section. What the level gates is whether MANAGER may act:
# installing from an arbitrary git URL (i.e. anything not in its database), and
# the "Try fix" action. Those are exactly the operations you reach for when a
# pack did not land, which is why it FEELS like it fixes import errors.
#
# NOTE THAT NODES IN THIS SCRIPT'S ${NODES} ARRAY ARE GIT-CLONED DIRECTLY AND
# NEVER TOUCH MANAGER. Manager's security level has no bearing on them at all.
# If one of those is failing, the [git] and [pip] lines say why and this setting
# will not change it. This block is for the ad-hoc installs you do in the UI
# afterwards.
#
# WHY 'weak' AND NOT 'normal-' ON THIS BOX. Manager computes:
#     is_local_mode = args.listen.startswith('127.') or args.listen.startswith('local.')
# ai-dock and Vast templates launch ComfyUI with --listen 0.0.0.0 so the portal
# can reach it. is_local_mode is therefore False, and 'normal-' -- the sane
# middle setting on a laptop -- is not honoured. The script detects the listen
# address and tells you which levels are actually available to you.
#
# THE RISK, STATED ONCE AND PLAINLY. security_level=weak lets anyone who can
# reach the ComfyUI web UI install arbitrary code from arbitrary URLs and run it
# as root on this instance, with your HF_TOKEN in the environment. On a rented
# box with a forwarded port that is a real exposure, not a theoretical one. Keep
# the instance's portal auth enabled, do not put the raw port on a public IP,
# and treat the token as burnable. This is a single-tenant throwaway GPU box --
# that is what makes weak defensible here and not on a machine you keep.
#
# DOWNGRADE_BLACKLIST is the reason to touch this file even if you never lower
# the security level. Manager runs custom-node requirements installs outside
# this script, which means outside the constraints file that is the only thing
# standing between your cu128 torch and a generic PyPI wheel with no sm_120
# kernels. Manager has its own guard for exactly this and it is empty by
# default. Setting it costs nothing and closes the one hole in the torch story.
# ---------------------------------------------------------------------------
comfy_listen_addr() {
    local a f d
    a="$(ps -eo args 2>/dev/null | grep '[m]ain.py' \
         | grep -oE -- '--listen([= ]+[^ ]+)?' | head -1 \
         | sed -E 's/^--listen[= ]*//')"
    if [[ -z "$a" ]]; then
        for d in /etc/supervisor/conf.d /etc/supervisor/supervisord/conf.d /etc/supervisord.d /etc/supervisor; do
            [[ -d "$d" ]] || continue
            for f in "$d"/*.conf "$d"/*.ini; do
                [[ -f "$f" ]] || continue
                grep -q '^\[program:comfyui\]' "$f" 2>/dev/null || continue
                a="$(grep -oE -- '--listen([= ]+[^ ]+)?' "$f" | head -1 | sed -E 's/^--listen[= ]*//')"
                [[ -n "$a" ]] && break 2
            done
        done
    fi
    # bare --listen with no value means 0.0.0.0
    [[ -z "$a" || "$a" == "--"* ]] && a="0.0.0.0"
    printf '%s' "$a"
}

if [[ -n "${MANAGER_SECURITY:-}" || -n "${MANAGER_DOWNGRADE_BLACKLIST:-}" ]]; then
    echo "=================== COMFYUI-MANAGER CONFIG ==================="

    # Path moved at Manager 3.0 and the user directory is relocatable with
    # --user-directory, so this is discovered, not assumed. Manager prints the
    # path it is using in its own startup log if you need to confirm.
    MGR_CFG=""
    _user_dir="$(ps -eo args 2>/dev/null | grep '[m]ain.py' \
                 | grep -oE -- '--user-directory[= ]+[^ ]+' | head -1 \
                 | sed -E 's/^--user-directory[= ]*//')"
    for c in "${_user_dir:+${_user_dir}/default/ComfyUI-Manager/config.ini}" \
             "${COMFY}/user/default/ComfyUI-Manager/config.ini" \
             "${NODES_DIR}/ComfyUI-Manager/config.ini" \
             "${NODES_DIR}/comfyui-manager/config.ini"; do
        [[ -n "$c" && -f "$c" ]] && { MGR_CFG="$c"; break; }
    done

    if [[ -z "$MGR_CFG" ]]; then
        if [[ -d "${NODES_DIR}/ComfyUI-Manager" || -d "${NODES_DIR}/comfyui-manager" ]]; then
            # Manager is installed but has never run, so it has not written a
            # config yet. Create the modern path -- Manager reads it on start
            # and fills in the rest of the defaults itself.
            MGR_CFG="${COMFY}/user/default/ComfyUI-Manager/config.ini"
            mkdir -p "$(dirname "$MGR_CFG")"
            printf '[default]\n' > "$MGR_CFG"
            echo "[manager] no existing config -- created ${MGR_CFG#${COMFY}/}"
        else
            echo "[manager] ComfyUI-Manager not installed -- nothing to configure"
            MGR_CFG=""
        fi
    fi

    if [[ -n "$MGR_CFG" ]]; then
        echo "[manager] config: ${MGR_CFG#${COMFY}/}"

        _listen="$(comfy_listen_addr)"
        if [[ "$_listen" == 127.* || "$_listen" == local.* ]]; then
            echo "[manager] listen=${_listen} -> Manager sees this as LOCAL."
            echo "[manager] 'normal-' is honoured here and is the safer choice."
        else
            echo "[manager] listen=${_listen} -> Manager sees this as REMOTE."
            echo "[manager] 'normal-' will NOT be honoured; only 'weak' unblocks"
            echo "[manager] git-URL installs and 'Try fix' in this configuration."
        fi

        [[ -f "${MGR_CFG}.prov.bak" ]] || cp -f "$MGR_CFG" "${MGR_CFG}.prov.bak" 2>/dev/null

        "$PY" - "$MGR_CFG" "${MANAGER_SECURITY:-}" "${MANAGER_DOWNGRADE_BLACKLIST:-}" <<'PYEOF'
import configparser, os, sys

path, level, blacklist = sys.argv[1], sys.argv[2], sys.argv[3]
VALID = ("strong", "normal", "normal-", "weak")

# interpolation=None: values like git_exe can contain '%' and would otherwise
# blow up ConfigParser on read. optionxform=str: do not lowercase keys.
cp = configparser.ConfigParser(interpolation=None)
cp.optionxform = str
try:
    if os.path.exists(path):
        cp.read(path, encoding="utf-8")
except Exception as e:
    print("[manager] !!! could not parse config.ini: %s" % e)
    print("[manager] !!! leaving it alone rather than rewriting a file I cannot read")
    sys.exit(1)

if not cp.has_section("default"):
    cp.add_section("default")

changed = False

if level:
    if level not in VALID:
        print("[manager] !!! '%s' is not a valid security_level." % level)
        print("[manager] !!! Valid: %s" % " | ".join(VALID))
        sys.exit(1)
    old = cp.get("default", "security_level", fallback="<unset>")
    if old == level:
        print("[manager] security_level already %s" % level)
    else:
        cp.set("default", "security_level", level)
        changed = True
        print("[manager] security_level: %s -> %s" % (old, level))

if blacklist:
    old = cp.get("default", "downgrade_blacklist", fallback="")
    merged = [p.strip() for p in (old + "," + blacklist).split(",") if p.strip()]
    seen, out = set(), []
    for p in merged:
        if p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    new = ",".join(out)
    if new == old:
        print("[manager] downgrade_blacklist already covers: %s" % new)
    else:
        cp.set("default", "downgrade_blacklist", new)
        changed = True
        print("[manager] downgrade_blacklist: %s -> %s" % (old or "<empty>", new))

if changed:
    tmp = path + ".prov.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        cp.write(fh)
    os.replace(tmp, path)
    print("[manager] written")
else:
    print("[manager] no change needed")
PYEOF
        _mrc=$?

        if [[ "${MANAGER_SECURITY:-}" == "weak" ]]; then
            echo "[manager] !!! security_level=weak is ACTIVE on this instance."
            echo "[manager] !!! Anyone who can reach the ComfyUI UI can install and"
            echo "[manager] !!! run arbitrary code here, as root, with HF_TOKEN in env."
            echo "[manager] !!! Keep the portal auth on. Treat the token as burnable."
        fi

        # Manager holds config in memory and rewrites the file on exit, so a
        # change applied while ComfyUI is running gets clobbered on shutdown.
        # This must be followed by a restart to take effect AND to stick.
        if (( _mrc == 0 )); then
            echo "[manager] NOTE: takes effect on RESTART. Manager rewrites this file"
            echo "[manager] NOTE: from memory when it exits, so without a bounce the"
            echo "[manager] NOTE: running process will overwrite what was just set."
            CHANGED_ANY=1
        fi
    fi
fi

echo "[provisioning] reconciling cuda-python to the CUDA-12 line for torch"
pip_install "cuda-python<13"

# ---------------------------------------------------------------------------
# SageAttention -- opt-in, DRAFTS ONLY, and USE THE NODE NOT THE FLAG
#
# It is an approximate attention kernel: quantized QK^T with a smoothing
# correction. Roughly 2x throughput with small but nonzero error -- a fine
# trade while iterating and a bad one on a final render.
#
# CRITICAL, AND DIFFERENT FROM THE H3 BUILD: do NOT launch with
# --use-sage-attention. The global flag routes through a backend that produces
# BLACK OUTPUT on Wan-family models, which is exactly what Bernini-R is. Use
# KJNodes' "Patch Sage Attention KJ" node after the model loader instead.
#
# On sm_120 a source build usually fails; you want a prebuilt wheel matching
# your torch minor version AND CUDA line exactly.
# ---------------------------------------------------------------------------
if [[ "${INSTALL_SAGE:-0}" == "1" ]]; then
    echo "=================== SAGEATTENTION ==================="
    if "$PY" -c "import sageattention" 2>/dev/null; then
        echo "[sage] already installed"
    else
        echo "[sage] attempting install (source build on sm_120 usually fails)"
        pip_install --no-cache-dir sageattention \
            || echo "[sage] FAILED -- grab an sm_120 wheel matching your torch/CUDA from github.com/woct0rdho/SageAttention/releases"
    fi
    echo "[sage] REMINDER: do NOT add --use-sage-attention to the launch flags."
    echo "[sage] REMINDER: wire KJNodes 'Patch Sage Attention KJ' after the model"
    echo "[sage] REMINDER: loader, for DRAFTS only, and bypass it for finals."
fi

echo "=================== TORCH VERIFY ==================="
verify_torch

# ===========================================================================
# DOWNLOAD INFRASTRUCTURE
# Everything here is Hugging Face. No Civitai tier in this build -- there is no
# meaningful Bernini LoRA ecosystem yet. Add one back from the H3 script if that
# changes; dl_civitai lifts across unmodified.
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

# ---------------------------------------------------------------------------
# NEW IN THIS BUILD: repo-path resolver
#
# Locate a file by basename anywhere in a repo. Exists because Kijai's
# WanVideo_comfy repo reorganises files into subfolders regularly and ComfyUI's
# own documentation currently links at least one Bernini-R dependency at a path
# it no longer occupies. A hardcoded repo_path is a future 404 that only shows
# up as a MISSING line after a long pull.
# ---------------------------------------------------------------------------
HF_FIND="/tmp/hf_find.py"
cat > "$HF_FIND" <<'PYEOF'
import os, sys
try:
    from huggingface_hub import list_repo_files
except Exception as e:
    sys.stderr.write("huggingface_hub import failed: %s\n" % e); sys.exit(3)

repo, base = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None
try:
    files = list_repo_files(repo_id=repo, token=token)
except Exception as e:
    sys.stderr.write("list_repo_files failed: %s\n" % e); sys.exit(1)

hits = [f for f in files if os.path.basename(f) == base]
if not hits:
    sys.exit(2)
# Shortest path wins: prefer repo root over a nested archive/ or old/ copy.
hits.sort(key=lambda f: (f.count("/"), len(f)))
print(hits[0])
PYEOF

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

# resolve_rpath <repo_id> <candidate1|candidate2|...>
# Echoes the first candidate that exists remotely, else the resolver's answer,
# else nothing. Caches per repo+basename so the preflight and the fetch phase do
# not each pay for the lookup.
declare -A RPATH_CACHE=()
resolve_rpath() {
    local repo="$1" cands="$2" c base key found
    base="$(basename "${cands%%|*}")"
    key="${repo}::${base}"
    [[ -n "${RPATH_CACHE[$key]:-}" ]] && { printf '%s' "${RPATH_CACHE[$key]}"; return 0; }

    local IFS='|'
    for c in $cands; do
        [[ -z "$c" ]] && continue
        if [[ -n "$(remote_size "$(hf_resolve_url "$repo" "$c")")" ]]; then
            RPATH_CACHE[$key]="$c"; printf '%s' "$c"; return 0
        fi
    done
    unset IFS

    if [[ "$HF_PATH_RESOLVE" != "1" ]]; then
        echo "[resolve] ${repo}: no candidate matched ${base} and HF_PATH_RESOLVE=0" >&2
        return 1
    fi

    found="$("$PY" "$HF_FIND" "$repo" "$base" 2>/dev/null)"
    if [[ -n "$found" ]]; then
        echo "[resolve] ${repo}: ${base} moved -> ${found}  (hardcode this)" >&2
        RPATH_CACHE[$key]="$found"; printf '%s' "$found"; return 0
    fi
    echo "[resolve] ${repo}: ${base} NOT FOUND anywhere in the repo" >&2
    return 1
}

# dl_hf <dest_dir> <dest_filename> <repo_id> <candidate_paths>
dl_hf() {
    local dir="$1" name="$2" repo="$3" cands="$4"
    local dest="${dir}/${name}" rpath check_url
    mkdir -p "$dir"

    rpath="$(resolve_rpath "$repo" "$cands")" || {
        echo "[model] SKIP $name -- could not resolve a path in ${repo}"; return 0; }
    check_url="$(hf_resolve_url "$repo" "$rpath")"

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

    echo "[model] downloading $name via hf_xet (${repo}/${rpath})"
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

# ---------------------------------------------------------------------------
# Manifest  --  base build ~75 GB
#
# Format:  hf | dest_dir | dest_filename | repo_id | candidate_path[|more...]
#
# The candidate list is the point. Put the path the docs claim first and any
# known alternative second; the resolver handles the case where both are stale
# and tells you what to write down.
#
# Expert-pair sizes are ESTIMATED at ~31 GB each (derived from the published
# ~15.5 GB fp8_scaled figures). The preflight reads real remote sizes, so the
# disk arithmetic below is not affected by the estimate.
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"

MODELS=(
    # === BERNINI-R EXPERT PAIR -- fp16 (~31 GB each) =====================
    # Wan2.2 A14B is a two-expert MoE: the high-noise expert sets composition,
    # the low-noise expert refines detail, and the sampler hands off between
    # them partway through the schedule. BOTH ARE REQUIRED -- this is not a
    # pick-one. The 96 GB card exists so that handoff never touches disk.
    # Comfy-Org also publishes fp8_scaled / int8_convrot / mxfp8 variants of
    # each; those are for cards that cannot hold the pair.
    "hf|$DIFF|wan2.2_bernini_r_high_noise_fp16.safetensors|Comfy-Org/Bernini-R|diffusion_models/wan2.2_bernini_r_high_noise_fp16.safetensors"
    "hf|$DIFF|wan2.2_bernini_r_low_noise_fp16.safetensors|Comfy-Org/Bernini-R|diffusion_models/wan2.2_bernini_r_low_noise_fp16.safetensors"

    # === TEXT ENCODER -- UMT5-XXL fp16 (11.4 GB) =========================
    # The stock template specifies fp8_e4m3fn_scaled (6.74 GB). fp16 instead:
    # see quality lever 4. Same file exists in the Wan 2.1 repackage; the 2.2
    # repo is listed first because Bernini-R is Wan2.2-based.
    "hf|$TE|umt5_xxl_fp16.safetensors|Comfy-Org/Wan_2.2_ComfyUI_Repackaged|split_files/text_encoders/umt5_xxl_fp16.safetensors"

    # === VAE -- Wan 2.1 VAE (~250 MB) ====================================
    # Yes, 2.1. The A14B line uses the Wan 2.1 VAE; only the 5B TI2V model has
    # the newer one. Do not substitute a "Wan 2.2 VAE" here.
    "hf|$VAE|Wan2_1_VAE_bf16.safetensors|Kijai/WanVideo_comfy|Wan2_1_VAE_bf16.safetensors|VAE/Wan2_1_VAE_bf16.safetensors"

    # === DISTILL LoRA -- DRAFT TIER ONLY (631 MB) ========================
    # ComfyUI's Bernini-R doc links this at the repo ROOT. It currently lives
    # under Lightx2v/. Both are listed; the resolver covers a third move.
    # Fetched for iteration, BYPASSED for finals -- quality lever 2.
    "hf|$LORA|lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors|Kijai/WanVideo_comfy|Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors|lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors|LoRAs/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
)

# === fp8_scaled PAIR -- draft tier / fallback (~15.5 GB each) ============
# Not needed to run, but a fast draft tier that leaves the fp16 pair untouched,
# and a fallback if you ever run this on a smaller card.
if [[ "$WANT_FP8" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|wan2.2_bernini_r_high_noise_fp8_scaled.safetensors|Comfy-Org/Bernini-R|diffusion_models/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"
    "hf|$DIFF|wan2.2_bernini_r_low_noise_fp8_scaled.safetensors|Comfy-Org/Bernini-R|diffusion_models/wan2.2_bernini_r_low_noise_fp8_scaled.safetensors"
)
fi

# === FULL BERNINI RENDERER -- AN A/B, NOT AN UPGRADE (~15.5 GB each) =====
# These are the renderer transformers from the FULL Bernini pipeline, extracted
# from ByteDance/Bernini-Diffusers and quantized to the same fp8_scaled layout
# as Comfy-Org's Bernini-R files -- byte-for-byte identical key structure, so
# they drop into the same nodes.
#
# READ THIS BEFORE YOU ASSUME IT IS BETTER: the full renderer was trained
# JOINTLY WITH THE MLLM PLANNER. ComfyUI does not run the planner, so you are
# feeding it a conditioning signal it was not trained to see alone. Bernini-R is
# the variant explicitly trained to stand alone. This may be better, worse, or
# task-dependent. Only fp8 is published, so you are also comparing fp8-vs-fp16
# unless you fetch WANT_FP8=1 and compare like with like -- which you should.
if [[ "$WANT_FULL_BERNINI" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|wan2.2_bernini_FULL_high_noise_fp8_scaled.safetensors|attashe/Bernini-Wan2.2-fp8-scaled|wan2.2_bernini_high_noise_fp8_scaled.safetensors|diffusion_models/wan2.2_bernini_high_noise_fp8_scaled.safetensors"
    "hf|$DIFF|wan2.2_bernini_FULL_low_noise_fp8_scaled.safetensors|attashe/Bernini-Wan2.2-fp8-scaled|wan2.2_bernini_low_noise_fp8_scaled.safetensors|diffusion_models/wan2.2_bernini_low_noise_fp8_scaled.safetensors"
)
fi

# === 1.3B SINGLE-EXPERT -- fast structural draft ==========================
# Fine-tuned from Wan2.1-T2V-1.3B, so it is single-expert: load it alone and
# leave the low-noise slot empty. ByteDance report it close to the 14B on
# simple tasks (style transfer, watermark removal, local edits) and behind on
# complex ones, notably human generation. Use it to check whether an edit
# INSTRUCTION parses at all before spending 14B time on it.
if [[ "$WANT_1_3B" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|wan2.1_bernini_1.3B_fp16.safetensors|Comfy-Org/Bernini-R|diffusion_models/wan2.1_bernini_1.3B_fp16.safetensors"
)
fi

# ---------------------------------------------------------------------------
# Disk pre-flight
# ---------------------------------------------------------------------------
preflight_disk() {
    local need=0 kind a b c d url dest have want rpath
    for entry in "${MODELS[@]}"; do
        # read assigns the ENTIRE remainder to the last variable, so d holds the
        # full pipe-separated candidate list, not just the first candidate.
        IFS='|' read -r kind a b c d <<< "$entry"
        [[ "$kind" == "hf" ]] || continue
        rpath="$(resolve_rpath "$c" "$d")" || continue
        url="$(hf_resolve_url "$c" "$rpath")"; dest="${a}/${b}"
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        want="$(remote_size "$url")"
        [[ -z "$want" ]] && continue
        (( want > have )) && need=$(( need + want - have ))
    done

    mkdir -p "$DIFF"
    local avail; avail="$(df -PB1 "$DIFF" | awk 'NR==2{print $4}')"
    local margin=$(( 15 * 1024*1024*1024 ))
    local h_need h_avail
    h_need="$(numfmt --to=iec "$need"  2>/dev/null || echo "${need} B")"
    h_avail="$(numfmt --to=iec "$avail" 2>/dev/null || echo "${avail} B")"
    echo "[provisioning] estimated to fetch: ${h_need}   free: ${h_avail}"

    if (( need + margin > avail )); then
        echo "[provisioning] !!! INSUFFICIENT DISK: need ~${h_need} + 15GiB headroom, have ${h_avail}"
        echo "[provisioning] !!! Cheapest give-backs, in the order you should make them:"
        echo "[provisioning] !!!   WANT_FULL_BERNINI=0                   -31.0 GB"
        echo "[provisioning] !!!   WANT_FP8=0                            -31.0 GB"
        echo "[provisioning] !!!   WANT_1_3B=0                            -5.0 GB"
        echo "[provisioning] !!!   text encoder -> umt5_xxl_fp8_e4m3fn_scaled   -4.7 GB"
        echo "[provisioning] !!!   experts -> fp8_scaled instead of fp16       -31.0 GB"
        echo "[provisioning] !!! The last one is the real quality cut. Take the others first."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
echo "=================== HUGGING FACE ==================="
echo "[provisioning] NOTE: base build is ~75 GB. First boot on a fresh volume is"
echo "[provisioning] NOTE: a long pull -- the fp16 expert pair alone is ~62 GB."
if preflight_disk; then
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"   # d = full candidate list
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
for entry in "${MODELS[@]}"; do
    IFS='|' read -r kind a b c d <<< "$entry"
    [[ "$kind" == "hf" ]] || continue
    if [[ -f "${a}/${b}" ]]; then
        echo "[layout] OK      ${a#${COMFY}/}/${b}  ($(numfmt --to=iec "$(stat -c%s "${a}/${b}")" 2>/dev/null))"
    else
        echo "[layout] MISSING ${a#${COMFY}/}/${b}"
    fi
done

for nd in ComfyUI-VideoHelperSuite ComfyUI-KJNodes ComfyUI-RH-Bernini \
          ComfyUI-BerniniR ComfyUI-RIFE-TensorRT-Auto; do
    if [[ -d "${NODES_DIR}/${nd}" ]]; then
        echo "[layout] OK      custom_nodes/${nd} @ $(git -C "${NODES_DIR}/${nd}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    else
        echo "[layout] absent  custom_nodes/${nd}"
    fi
done

# The pair check is worth calling out separately: one expert present and the
# other missing produces a graph that loads, samples, and outputs garbage,
# because the handoff hands off to nothing.
if [[ -f "${DIFF}/wan2.2_bernini_r_high_noise_fp16.safetensors" && ! -f "${DIFF}/wan2.2_bernini_r_low_noise_fp16.safetensors" ]] \
|| [[ -f "${DIFF}/wan2.2_bernini_r_low_noise_fp16.safetensors" && ! -f "${DIFF}/wan2.2_bernini_r_high_noise_fp16.safetensors" ]]; then
    echo "[layout] !!! EXPERT PAIR INCOMPLETE. You have one of the two fp16 experts."
    echo "[layout] !!! Bernini-R needs BOTH. Re-run provisioning to fetch the other."
fi

# ===========================================================================
# FINALISE -- log discovery, health check, optional restart
# ===========================================================================
echo "=================== FINALISE ==================="

# No fixed path for the ComfyUI log: ai-dock writes /var/log/supervisor/,
# Vast templates write under /var/log/portal/. Read stdout_logfile out of the
# supervisor program block instead of guessing, and grab command= too so a
# failed startup can be reproduced in the foreground -- which is the only
# reliable way to see an import traceback.
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

# supervisor reports RUNNING once a process survives startsecs (5s default),
# which says nothing about whether ComfyUI imported its nodes and bound its
# port -- a crash-loop reads as a rapid succession of healthy RUNNING states.
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

=================== FIRST RUN ===================

Do NOT open the final config first. Two runs, in this order.

RUN 1 -- VALIDATE THE PIPELINE, IGNORE QUALITY
  Template Library -> search "Bernini-R" -> Video Editing template.
  480p, 81 frames @ 16fps, distill LoRA ON, fp8 experts if you fetched them.
  A couple of minutes. You are checking three things and nothing else:
    - the conditioning wires up and the graph runs to completion
    - the reference image is actually being READ (change it, see the output
      change -- a disconnected ref socket fails silently, it does not error)
    - the output is not black frames (that is SageAttention; see below)

RUN 2 -- THE REAL CONFIG
  fp16 expert pair, fp16 UMT5, distill LoRA BYPASSED, real CFG,
  30-40 steps split across the two experts, 1280x720, 121 frames @ 24fps.
  That is ByteDance's own max-quality rv2v setting.

The delta between those two runs tells you what the distill LoRA actually costs
on YOUR footage, which is worth more than any general claim about it.

=================== TASK TYPES ===================

  t2v    text prompt                       text-to-video
  v2v    source video                      restyle / edit
  rv2v   source video + reference image(s) reference-guided edit  <- this build
  r2v    reference image(s)                reference-to-video
  img    source image + prompt             image editing
  ads2v  source video + reference video    insert content into source

REFERENCE IMAGES: each batched image becomes its OWN in-context token. Address
them in the prompt as image0, image1, ... when they play different roles
("replace the jacket with image0, keep the background"). If the subject is small
in the reference frame, crop in before you feed it -- the model sees the token,
not your intent.

=================== FINAL-RENDER SETTINGS ===================

MODELS
  diffusion    wan2.2_bernini_r_high_noise_fp16 + wan2.2_bernini_r_low_noise_fp16
               BOTH. High sets composition, low refines detail, sampler hands
               off between them. One alone produces garbage, not half-quality.
  text encoder umt5_xxl_fp16
  vae          Wan2_1_VAE_bf16   (2.1, not 2.2 -- see the manifest note)
  LoRA         none. Bypass the lightx2v node for finals.

RESOLUTION AND LENGTH -- these trade against each other
  1280x720, 121 frames, 24 fps. ~5.0 s. This is the reference configuration.
  Frame counts must be 4n+1 (the VAE compresses 4x temporally):
      25=1s  49=2s  73=3s  97=4s  121=5s  (at 24fps)
  Wan's training horizon is 81 frames. 81@16fps and 121@24fps are both ~5s of
  content -- the limit is a DURATION and fps is how you spend it.

  If you are token-bound, shoot 161 frames @ 16fps (10.06s) rather than 241 @
  24fps. Same duration, ~33% fewer tokens, and RIFE back up to 24 afterward.

SAMPLER
  res_multistep + simple is the reported starting point for this model. It is a
  second-order integrator, so it buys accuracy per unit compute rather than
  just more compute. Treat it as a starting point and not gospel -- A/B it
  against your own footage once the rest is settled.
  30-40 steps total, split across the two experts. Real CFG (start ~4-5) once
  the distill LoRA is bypassed.

VAE
  Tiling OFF. You have 96 GB; tiled decode buys nothing here and produces
  seams on exactly the kind of large flat gradients that video edits expose.

LAUNCH FLAGS
  --highvram
  NOT --use-sage-attention  (see below)
  --disable-pinned-memory   only if model loading is pathologically slow
                            (ComfyUI 0.30.x pinned-memory regression)

=================== LENGTH PAST 5 SECONDS ===================

SEGMENT IT. Two 121-frame passes beats one 241-frame pass, and the reason is
specific to editing rather than general caution:

The looping / motion-reverting failure mode past the training horizon is a
GENERATIVE artifact -- it is what happens when a model has to invent motion
beyond what it was trained to plan. In rv2v the source video's latents are
in-context at every step, continuously re-anchoring structure and motion. Your
reference images are identical across both passes. So the realistic drift
between segments is a slow shift in the EDITED ATTRIBUTE (colour, lighting,
identity detail) rather than the subject undoing its own motion.

  - same seed, same prompt, same reference images, both passes
  - overlap a few frames and check the seam before you commit to the approach
  - AIMixer/ComfyUI-Bernini has a timeline node that splits a source video into
    N segments with per-segment prompts, if you are doing this a lot. It is a
    second full inference path -- install it alone, not alongside the others.

CAVEAT, STATED PLAINLY: this reasoning is from how in-context conditioning
works, not from published rv2v length benchmarks. I did not find anyone
reporting systematic tests past 121 frames. Run a cheap 480p comparison of
one-pass-161f against two-pass-121f on your actual source before you commit
GPU hours at 720p. It is an hour that could save you ten.

=================== PROMPT ENHANCEMENT (WANT_PROMPT_ENHANCER=1) ===================

This is the stand-in for the MLLM planner ComfyUI does not run, and it is the
largest single quality lever in the build. Wiring:

  Bernini Prompt Enhancer  -> system_prompt + user_prompt -> your LLM node
  your LLM node            -> response                    -> Bernini Prompt Result Parser
  Parser (enhanced_prompt) -> text encode

  - set json_mode consistently on BOTH the enhancer and the parser for r2v,
    r2i, rv2v and vrc2v; the parser looks for rewritten_text in JSON mode
  - if your LLM node takes only one prompt input, use llm_prompt instead of the
    system/user pair
  - SEND IT THE REFERENCE IMAGES if the node supports vision. The official
    templates are written assuming the model can see what image0 actually is;
    without that it is guessing at the thing it is supposed to be planning.
  - point BERNINI_PE_BASE_URL / OPENAI_BASE_URL at any OpenAI-compatible
    endpoint, local or hosted

=================== TROUBLESHOOTING ===================

BLACK OUTPUT FRAMES
  SageAttention, ~always. The global --use-sage-attention flag routes through a
  backend that black-frames Wan-family models, and Bernini-R is a Wan-family
  model. Remove the flag; use KJNodes "Patch Sage Attention KJ" after the model
  loader instead. If you are not using Sage at all and still get black frames,
  check the [torch] arch line -- a torch wheel with no sm_120 kernels can
  produce silent garbage rather than an error.

RED NODES / "BERNINI CONDITIONING" MISSING
  Read the [bernini] block near the top of provisioning.log first. If it says
  no bernini module in comfy_extras/, core does not have the support and no
  amount of node-installing will fix it -- see the [git] lines for why the
  update did not land. If core HAS it but the template is absent, it is
  comfyui-workflow-templates being stale; see the [deps] lines.

A MODEL SHOWS AS MISSING AFTER A CLEAN RUN
  Read the [resolve] lines. If one says a file "moved -> <path>", the resolver
  found it and the download should have proceeded; hardcode that path into the
  manifest so the next boot skips the lookup. If it says NOT FOUND anywhere in
  the repo, the file was renamed or removed upstream and the manifest entry
  needs a real edit.

IF UPDATES DO NOT SEEM TO LAND
  Read the [git] lines, in this order:
    1. "safe.directory configured". If that says WARNING, every git call in the
       run is failing on dubious ownership and nothing updated, regardless of
       what the rest of the log says.
    2. "detached HEAD -- looks image-pinned, leaving alone". That is correct
       behaviour: the image pinned core deliberately. COMFY_PIN to move it.
    3. "fast-forward blocked". Local commits or a dirty tree.
       GIT_FORCE_RESET=1 discards them.

IF COMFYUI SEEMS SLOW TO START
  Check the [provisioning] comfyui line at the end. If it says CRASH-LOOPING it
  is not slow -- it is being respawned and there is no startup to wait for.
  Stop the service and run the printed command in the foreground; the traceback
  only appears there. supervisor reporting RUNNING means only that the process
  survived five seconds.

IF CUSTOM NODES FAIL TO IMPORT
  Read the LAST line of the traceback, not the first. A pack dying on
    ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
  is not itself broken: it imported diffusers, diffusers probed xformers,
  xformers found a half-installed flash_attn and took the flash path. Look for
  the [flash] lines near the top. If they say healthy or not installed and a
  pack still fails, the cause is elsewhere.

  MANAGER'S SECURITY LEVEL DOES NOT CAUSE IMPORT FAILURES, and lowering it will
  not repair one. What it gates is whether Manager may ACT: install from an
  arbitrary git URL, and run "Try fix". Those are the operations you reach for
  after a pack failed, which is why the two get conflated. Note also that "Try
  fix" only re-runs that pack's requirements.txt, so it cannot repair a broken
  package the file does not list. Reading the traceback beats toggling the
  level every time.

  Nodes in this script's NODES array are git-cloned directly and never go
  through Manager at all, so the level is irrelevant to those. If one of them
  did not land, the [git] and [pip] lines say why.

IF MANAGER REFUSES TO INSTALL FROM A GIT URL
  "This action is not allowed with this security level configuration."
  Set MANAGER_SECURITY=weak in the instance environment and restart. Read the
  [manager] lines: they print the config path actually in use (it moved at
  Manager 3.0 and moves again with --user-directory), the listen address, and
  the before/after value.

  'normal-' is the safer setting and it will NOT work here. Manager computes
  is_local_mode from whether --listen starts with 127. or local., and ai-dock
  launches with --listen 0.0.0.0 so the portal can reach the UI. On this box
  Manager considers itself remote, and only 'weak' unblocks git-URL installs.

  IF THE SETTING DOES NOT STICK: Manager keeps config in memory and rewrites
  the file when it exits, so a change written underneath a running process is
  overwritten on shutdown. Provisioning sets CHANGED_ANY when it edits the
  file; either run with RESTART_COMFY_ON_UPDATE=1 or bounce it yourself. The
  same applies in reverse -- changing the level in the Manager UI wins until
  the next boot, when this script sets it back.

  The original file is preserved once as config.ini.prov.bak.

IF CUDA GOES MISSING AFTER AN UPDATE
  Something moved torch despite the constraints file. The [torch] block says so
  loudly, including whether the wheel has sm_120 kernels at all. On Blackwell a
  generic PyPI torch does not degrade performance, it removes CUDA. Reinstall
  the pinned cu128+ build before rendering anything.

TWO BERNINI NODES WITH THE SAME NAME
  Read the [packs] block. More than one Bernini pack is installed and they are
  colliding. Keep one.

NOT IN THIS BUILD
  The MLLM planner (not open-sourced -- only the renderer was).
  bytedance/Bernini's own inference scripts (torch 2.5.1+cu124, no sm_120).
  Multi-GPU Ulysses sequence parallel -- that is how ByteDance gets 8xH100
  speed, and it is a speed story, not a VRAM requirement. One 96 GB card runs
  the same configuration, just slower per clip.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
