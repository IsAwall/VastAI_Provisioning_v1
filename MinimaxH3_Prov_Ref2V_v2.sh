#!/bin/bash
# =============================================================================
# ai-dock / ComfyUI provisioning script for vast.ai
# MiniMax H3 (Hailuo 3.0) -- REF2V BUILD  ---  v2
#
# Derived from MinimaxH3_Prov_TI2V_v7.sh. The update machinery, flash-attn
# guard, torch constraints, git handling and supervisor health check are
# carried over so the scripts diff cleanly. What moved is the manifest, the
# node set, and a set of bugs fixed on the way through.
#
# HOW TO USE:
#   1. Host this file where it can be fetched as RAW plain text.
#   2. On the vast.ai instance set:  PROVISIONING_SCRIPT=<that-raw-url>
#   3. Environment variables:
#        HF_TOKEN=<token>          recommended (rate limits on a ~121 GB pull)
#        CIVITAI_TOKEN=<token>     only if you add Civitai entries below
#        COMFY_UPDATE=0            skip the ComfyUI git update (default: ON)
#        NODE_UPDATE=0             skip the custom-node git updates (default: ON)
#        GIT_FORCE_RESET=1         discard local commits blocking a fast-forward
#        WANT_ACCEL=0              skip both acceleration adapters (default: ON)
#        ACCEL_PDD=0               skip the PDD Acc 8-step adapter (default: ON)
#        ACCEL_LIGHTX2V=0          skip the lightx2v ref2v v0.1 (default: ON)
#        WANT_EMBEDDINGS=1         add the 10 community style embeddings
#        WANT_CONTEXT_LOOP=1       add ethanfel's context-loop pack (see note)
#        WANT_SEEDVR2=1            add the SeedVR2 restore weights (+~15 GB)
#        PURGE_LEGACY_SEEDVR2=1    delete the v3-era third-party pack + weights
#        FIX_FLASH_ATTN=0          skip the half-installed flash-attn guard
#        INSTALL_SAGE=1            opt-in: SageAttention (DRAFTS ONLY)
#        COMFY_PIN=<sha|tag>       hold core at this ref
#        ALLOW_BRANCH_RECOVERY=1   permit moving a detached HEAD onto a branch
#        FORCE_DEPS=1              reinstall pip requirements even if unchanged
#        RESTART_COMFY_ON_UPDATE=1 bounce the comfyui service if code changed
#   4. (Re)start the instance. ai-dock runs this on every boot.
#
#   DISK: ~121 GB of weights before SeedVR2, ~136 GB with it. Provision a
#   200 GB volume. The ref2va DiT is 66.3 GB -- 4.6 GB LARGER than the fl2va
#   file the I2V build pulls, so a 180 GB volume that fit v7 is now tight.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM v1 OF THIS SCRIPT -- THE TURBO ANSWER MOVED
#
#   v1 fetched larryvrh's v4-600 because you asked for it, with a warning
#   attached. That warning is now confirmed empirically, and a better option
#   shipped four days ago. Both facts change the manifest.
#
#   1. larryvrh v4-600 IS REMOVED. Not on inference -- on evidence. There is a
#      thread on the LoRA's own model card titled "Does it support ref2v?" in
#      which users report it failing when a reference VIDEO is connected, and a
#      separate audio-reference failure tracked as issue #9 on the node repo.
#      Combined with the model card scoping itself to t2v/i2v and the
#      standalone script hardcoding an fl2va base, that is enough. The node
#      pack goes with it -- without the LoRA it has no reason to be installed,
#      and it was a second H3 node family competing with core for names.
#
#   2. THE NEW PRIMARY: MiniMax-H3-Ref2VA-Acc-8Step (Alibaba PAI, 2026-08-26).
#      Parallel Decoding Distillation applied to H3, published as TWO trunk-
#      specific adapters -- one for FL2VA, one for Ref2VA. This is the first
#      ref2v acceleration artifact from a major lab rather than a community
#      preview, and its own model card ships side-by-side Ref2VA comparisons
#      against both the undistilled baseline and the lightx2v ref2v v0.1.
#
#      Why it fits this build specifically:
#        - It is trunk-matched. Pair the Ref2VA adapter with a ref2va UNET;
#          bf16 originals and int8_convrot builds both work.
#        - Its trained sigma shift is EXACTLY 12.0 / 3.0 -- your calibration,
#          not a value you have to reconcile. The ComfyUI node FAILS CLOSED if
#          the shift is anything else, so a mis-set schedule is an error rather
#          than a silent quality loss. That is a meaningful change from every
#          other LoRA on this model.
#        - CFG 1.0 with BasicGuider and euler: the stock H3 graph already.
#        - 8 steps is the trained block size, 4 is an officially sanctioned
#          regrouping, 6 works via a non-uniform partition. Any other count is
#          REJECTED by the node rather than silently degraded.
#
#      THE CATCH, AND IT IS A REAL ONE: these are not plain LoRAs. Alongside a
#      rank-64 trunk LoRA each file carries a Parallel Decoding Distillation
#      head bank -- 32 per-interval copies of the final-layer video and audio
#      projections, fused into one mean-block-velocity head per sampler step.
#      A NORMAL LoRA LOADER READS THE TRUNK AND SILENTLY DROPS THE HEAD BANK.
#      You get a model that loads, runs, and has lost the distillation. This is
#      the same silent-failure shape as the fl2va/ref2va key match, and it
#      needs the same discipline: use the dedicated node, never LoraLoader.
#
#      Requirements: ComfyUI-MiniMax-H3-PDD-Acc (installed below, and it has NO
#      pip requirements file, so it cannot touch your torch stack), and ComfyUI
#      >= 0.33.0 for the carried-audio mechanics from PR #15243. The node fails
#      closed with an update message on older cores.
#
#      Weights go in models/pdd_acc/, NOT models/loras/. Different folder.
#
#   3. THE lightx2v REF2V v0.1 STAYS, as the no-custom-node fallback. It loads
#      through the native Lightning LoRA checkbox on the R2V node, and the
#      official ComfyUI R2V template scans for it by name -- so it is worth
#      having on disk whether or not you draft with it.
#
#   4. THERE IS NO NEWER lightx2v REF2V. I checked the upstream model-spec
#      table. The fl2v line has moved to 4-step and 8-step v1.0 768p variants
#      and there is a v1.1 768p file referenced in the Alibaba comparisons, but
#      the ref2v line is still at 4-step v0.1 from 2026-08-13. Upstream calls
#      v0.1 a preview whose image detail still needs improvement, and
#      "improve the visual quality and consistency of Ref2VA and FL2VA Turbo"
#      is an open roadmap item. So: no, you have not missed a release.
#
#   5. LICENSE DIVERGENCE BETWEEN THE TWO ADAPTERS -- CHECK THIS BEFORE YOU
#      SHIP ANYTHING. The lightx2v ref2v v0.1 is Apache-2.0. The Alibaba PAI
#      Acc LoRAs are published under the minimax-h3-community-license-agreement
#      -- the same license as the base weights, with the same excluded
#      territories. The third-party node pack's README describes them as
#      Apache-2.0, which does not match the Hugging Face model card. The model
#      card governs. Given you were already weighing datacenter jurisdiction
#      for the base weights, this does not add a new constraint, but it does
#      remove a get-out: the accelerator does not come with a looser license
#      than the model it accelerates.
#
#   6. TRAINING-RESOLUTION CORRECTION, AND IT CUTS AGAINST THE HOUSE RULE.
#      The lightx2v ref2v v0.1 was distilled at 544p mixed aspect ratio, and
#      upstream's own ref2v example graph defaults to 960x544 (0.5 MP). Your
#      standing rule -- draft at native canvas, cut frames and steps rather
#      than resolution -- is a statement about the BASE model's RoPE and shift
#      calibration. It is not automatically true of a LoRA distilled somewhere
#      else. The strongest evidence for that: when lightx2v retrained the fl2v
#      turbo AT 768p they had to change the shift from 12/3 to 6/3. Shift is
#      coupled to training resolution, so a 544p-trained adapter carrying 12/3
#      is calibrated for 544p, and running it at 1344x768 is off ITS
#      distribution even while it is on the base model's.
#      There is no 768p ref2v turbo. If you want a 768p draft tier on this
#      branch, the PDD adapter is the only candidate -- and note that Alibaba
#      label their FL2VA demos 768p while leaving the Ref2VA demos unlabelled,
#      so confirm that yourself rather than taking it from me.
#
#   7. BUG FIXES -- see the BUGS FIXED section below.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED FROM THE I2V v7 SCRIPT
#
#   1. THE DIFFUSION MODEL. minimax_h3_ref2va_bf16.safetensors (66.3 GB), not
#      fl2va. These are genuinely different weights -- ComfyUI's own docs call
#      this out explicitly on the R2V template. You cannot drive
#      MiniMaxH3ReferenceToVideo from an fl2va checkpoint.
#
#      Give-back ladder if disk or VRAM binds:
#        minimax_h3_ref2va_int8_convrot         34 GB
#        minimax_h3_ref2va_pruned_int8_convrot  21 GB   (the template default)
#      NOTE the Comfy-Org model card's own caveat: prefer int8_convrot only if
#      you can run torch on cu130. This build pins a cu128 stack, so int8 is
#      not a free swap here -- verify before you rely on it. fp8_scaled is the
#      fallback for when int8 is unavailable, not a quality choice.
#
#   2. THE ACCELERATION TIER. Two trunk-matched adapters, no fl2va imports.
#      See the section above. v7's turbo entry was also structurally broken --
#      see item 4.
#
#   3. NODE-CLASS PROBE REPLACES THE VERSION-STRING GUESS. v7 compared a
#      version string against 0.30.0 and inferred capability from it. That is
#      a proxy. This script greps comfy_extras/nodes_minimax_h3.py for the
#      class names it actually needs, which is the fact rather than a stand-in
#      for it:
#        MiniMaxH3ReferenceToVideo  -- required, landed in 0.30.0
#        MiniMaxH3AddGuide          -- optional, landed in 0.34.0 (PR #15439)
#      The version comparison is kept as a secondary signal, not the gate.
#
#   4. MANIFEST FIELD-COUNT VALIDATION -- because v7 shipped with a silent
#      manifest bug. Line 1051 of v7:
#
#        "hf|$LORA|minimax_h3_fl2v_lightx2v_..._bf16.safetensors|loras/minimax_h3_fl2v_lightx2v_..._bf16.safetensors"
#
#      Four fields, not five. The repo_id is missing entirely. `IFS='|' read`
#      does not care: it binds the repo PATH into the repo_ID slot and leaves
#      the path empty, so hf_resolve_url builds
#
#        https://huggingface.co/loras/minimax_h3_.../resolve/main/
#
#      which 404s. WANT_TURBO=1 on v7 downloads nothing and says only
#      "DOWNLOAD FAILED", with no hint that the manifest itself is malformed.
#      validate_manifest() now rejects wrong-arity entries before the fetch
#      phase starts, naming the offending line.
#
#   5. HTTP STATUS DISAMBIGUATION on the pre-download probe. v7's remote_size()
#      returned an empty string for every failure mode, so a typo'd repo path
#      (404), an expired token (401), a gated repo (403) and a transient 5xx
#      all produced the same "size unverifiable, assuming complete" or a bare
#      failure. Those want different responses from you, so they now print
#      different things.
#
#   6. EMBEDDINGS DIRECTORY. ComfyUI added `embedding:` syntax support for H3
#      prompts (PR #15697) and Comfy-Org hosts 10 community style embeddings.
#      models/embeddings is created unconditionally; the files are opt-in via
#      WANT_EMBEDDINGS=1 (they are small). These are community contributions
#      via silveroxides, NOT produced by MiniMax or Comfy-Org.
#
#   7. THE CONTEXT-LOOP PACK IS NOW OPT-IN. ethanfel's
#      ComfyUI-MiniMaxH3-Contex-Loop is built around fl2va first/last-frame
#      extension. I have no evidence either way about ref2va, and on a ref2v
#      build the native MiniMaxH3AddGuide covers arbitrary-frame anchoring
#      including the "feed the tail of the previous clip back in" pattern.
#      There is an official example graph for exactly this:
#      video_minimax_h3_r2v_addguides_v1.json. Set WANT_CONTEXT_LOOP=1 if you
#      want the pack anyway; it is not installed by default here.
#
#   8. RESOLUTION NOTE CORRECTED. v7 said "Megapixels ~1.0". ComfyUI's docs
#      are explicit that 1.0 MP yields 1376x768, which is ABOVE the model's
#      768x1344 pixel-area cap. Use 0.98 for the native canvas, or bypass the
#      selector and type 1344 x 768 into the node directly.
#
# -----------------------------------------------------------------------------
# BUGS FIXED IN v2 (all of these were in v1 of this script; two came from v7)
#
#   A. safe.directory GREW THE GITCONFIG ON EVERY BOOT. `git config --global
#      --add safe.directory '*'` appends a NEW line each run -- it does not
#      deduplicate. On a long-lived persistent volume that file accumulates
#      one identical entry per boot forever. Now checked before adding.
#      (Inherited from v7.)
#
#   B. EVERY REMOTE FILE WAS PROBED TWICE. preflight_disk() issued a HEAD for
#      each manifest entry to size the pull, then dl_hf() issued the SAME HEAD
#      again moments later. With the full manifest that is ~24 requests where
#      12 would do, each with a 60s ceiling, and it is a good way to earn a
#      429 on an unauthenticated run. Probes are now memoised for the run.
#      (Inherited from v7.)
#
#   C. THE HF STAGING DIRECTORY WAS NEVER CLEANED. hf_hub_download writes into
#      <dest_dir>/.hf_stage/ and leaves its .cache metadata tree behind after
#      the file is moved out. Harmless per file, but it accumulates inside
#      models/diffusion_models over rebuilds. Now pruned after a VERIFIED
#      download only -- on failure the partial is deliberately left in place,
#      because that is what makes the next boot resume rather than restart a
#      66 GB transfer.
#
#   D. THE NESTED-DUPLICATE SWEEP ONLY RAN ON ALREADY-COMPLETE FILES. If a
#      file had just been downloaded, or its probe failed, the check was
#      skipped -- so the case it exists to catch (a 66 GB duplicate from a
#      previous `hf download --local-dir` run) could sit there unreported.
#      Moved into the layout phase where it runs over every manifest entry.
#
#   E. preflight_disk() CLOBBERED THE CALLER'S LOOP VARIABLE. `entry` was not
#      declared local, so the function wrote to the same global the fetch loop
#      uses. Harmless today because the fetch loop reassigns it first, but it
#      is the kind of thing that turns into a real bug the moment someone adds
#      a second loop. Declared local. (Inherited from v7.)
#
#   F. NO GATE FOR THE 0.33.0 PDD REQUIREMENT. The node probe checked for the
#      ref2v node and AddGuide but had nothing to say about the core version
#      the PDD pack needs. Added, and it only fires when ACCEL_PDD is on.
#
#   G. models/pdd_acc WAS NOT CREATED. The PDD node creates it on first
#      launch, but the manifest writes into it before ComfyUI ever starts.
#      Created up front alongside the other model directories.
#
#   H. ref_image_size ADVICE WAS WRONG FOR THE DISTILLED PATH. v1 said "max
#      for finals". Upstream is explicit that `match` is the policy used in
#      distillation training and is what they recommend for distilled models.
#      `max` remains right for undistilled finals. The notes now split the
#      advice by tier instead of giving one answer for both.
#
# -----------------------------------------------------------------------------
# WHAT REF2V CHANGES ABOUT HOW YOU WORK
#
#   THE PROMPT IS A REFERENCE-ASSIGNMENT DOCUMENT, NOT A SHOT DESCRIPTION.
#   Everything you already know about H3 being a schema parser rather than a
#   natural-language model applies harder here, because ref2v adds a whole
#   second axis the schema has to carry: which reference drives which property
#   of the target shot.
#
#     - Address every reference by tag, in CONNECTION ORDER:
#         <Picture 1>, <Picture 2>, <Video 1>, <Audio 1>
#       The ordinal is the socket index, not a name you choose. Rewiring a
#       reference input renumbers every tag downstream of it and silently
#       invalidates a prompt that used to work. This is the ref2v equivalent
#       of a structural prompt bug: it will reproduce identically across
#       seeds, and rerolling is the wrong instinct.
#     - Give each reference an explicit JOB: identity, style, motion, camera,
#       voice. "Use <Picture 1>" underspecifies and the model gap-fills.
#     - MiniMax publishes a SEPARATE prompt guide for reference mode:
#         VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
#       It is not the base-mode guide with extra paragraphs. It defines its own
#       rewrite output structure -- subject definitions, reference labels,
#       retention analysis. If you have been working from the base guide, the
#       ref guide is a different schema and the base one will not transfer.
#
#   REFERENCE LIMITS: 9 images, 3 videos (each may carry its own soundtrack),
#   3 standalone audio clips.
#
#   ref_image_size IS THREE THINGS AT ONCE: QUALITY, VRAM, AND TRAINING MATCH.
#     match -> matches reference pixel area to the target canvas, preserving
#              aspect, never upscaling. Scale = min(1, sqrt(target/ref area)).
#     max   -> preserves aspect, only scales down references whose short edge
#              exceeds 2048px. Scale = min(1, 2048 / ref_short_edge).
#     USE match ON ANY ACCELERATED TIER. Upstream distills with match and
#     explicitly recommends it for distilled models, so it is the
#     training-matched policy, not merely the fast one. v1 of this script told
#     you to use max for finals without that qualifier; that advice is right
#     for the undistilled 25-30 step path and wrong for the adapters.
#   On this build `max` is where the VRAM math gets interesting: the bf16 DiT
#   is 66.3 GB against 96 GB of VRAM, leaving under 30 GB for latents plus
#   reference conditioning. Nine references at a 2048px short edge is a real
#   allocation against that. If you OOM on ref2v where I2V was comfortable,
#   ref_image_size is the first dial to look at, before you touch precision.
#
#   THE Context-IR GAP IS WIDER HERE THAN IT WAS ON I2V. The withheld
#   preprocessor's job is relationship resolution across inputs. With one
#   image and a well-formed prompt there is not much for it to resolve. With
#   six references that each need a role assigned and reconciled against each
#   other, there is a great deal. Expect the local-vs-API delta you measured
#   on I2V to understate the ref2v case.
#
# -----------------------------------------------------------------------------
# WHERE THE QUALITY IS, IN THIS BUILD (ranked, largest lever first)
#
#   1. PROMPT STRUCTURE AND REFERENCE ASSIGNMENT. On ref2v this outranks the
#      restore stage, which it did not on I2V. A mis-tagged reference is not a
#      fidelity problem you can restore your way out of -- the wrong thing got
#      generated. Fix the schema before you touch anything else.
#   2. THE RESTORE STAGE. H3-Regenerate-2K is API-only; local output is capped
#      at a 768px short edge. SeedVR2 is a diffusion-transformer video
#      restorer -- generative prior, temporally conditioned, so it does not
#      flicker the way per-frame ESRGAN does. 7B fp16, not fp8, not sharp.
#   3. STEP COUNT AND SAMPLER. res_multistep is second-order; local truncation
#      error per step scales as the square of a first-order method's. Baseline
#      20 steps; 25-30 measurably better.
#   4. ref_image_size = max, where VRAM allows. This is a ref2v-only lever and
#      it buys identity fidelity that nothing downstream can recover.
#   5. DIFFUSION PRECISION. bf16 (66.3) over int8_convrot (34) over
#      pruned_int8 (21). Real, but smaller than 1-4.
#   6. TEXT ENCODER PRECISION. Smallest lever. Qwen3-VL is an encoder, not a
#      generator. First thing to give back, not the last.
#
# -----------------------------------------------------------------------------
# THE LADDER
#
#   DRAFT  -> PDD Acc 8-step (or 4), ref_image_size match, Sage on, SILENT.
#             On the lightx2v v0.1 fallback, draft at 960x544 rather than
#             1344x768 -- that adapter was distilled at 544p and its 12/3
#             shift is calibrated for that canvas, not for yours.
#   FINAL  -> ref2va_bf16, bf16 encoder, res_multistep + simple, 25-30 steps,
#             1344x768, ref_image_size max, no acceleration adapter, no Sage,
#             audio fields populated.
#   RESTORE-> SeedVR2 7B fp16, native nodes, resize multiplier 1.875 -> 1440.
#   INTERP -> RIFE TensorRT, 24 -> 48/60 fps. Last stage before encode.
#
#   ACCELERATION IS A DRAFT TIER, NOT A FAST FINAL. Distillation trains the
#   model to take larger jumps along the flow trajectory. That is a fidelity
#   trade by construction, and it is true of PDD as much as of a turbo LoRA.
#
#   DISTILLATIONS DO NOT STACK. One adapter at a time -- never PDD plus the
#   lightx2v turbo. Character and style LoRAs stack with either normally.
#
#   DO NOT RUN SHIFT EXPERIMENTS ON AN ACCELERATED TIER. Distillation
#   memorises specific sigma values; moving shift off the calibrated point
#   breaks the schedule rather than testing anything. On the PDD path this is
#   not even possible -- the node refuses anything but 12.0/3.0 -- which is a
#   useful property rather than a limitation. Shift work belongs on the base
#   model, at 50 steps, with recommended weights.
#
#   ON THE AUDIO QUESTION YOU ARE STILL CHASING: the mechanism is documented
#   upstream. H3 runs video and audio on two flow schedules (12 and 3), stock
#   samplers step both on one schedule, and at low step counts that badly
#   over-steps the audio. That is a sampler-path claim, not a LoRA-strength
#   one, and it lines up with the ModelSamplingAV carry-path theory rather
#   than with turning the strength dial down. The PDD path is a clean test
#   surface for this: its head bank is armed per step by sigma on the shift-12
#   video and shift-3 audio grids separately, so if audio behaves there and
#   not elsewhere, the schedule was the variable.
#
# -----------------------------------------------------------------------------
# LICENSE -- UNCHANGED, AND STILL WORTH RE-READING BEFORE YOU PICK A HOST
#
#   The MiniMax H3 Community License defines "Applicable Territory" as
#   worldwide EXCLUDING the EU, UK, South Korea, and the USA. The exclusion
#   covers running the weights AND using their outputs. Canada is not on the
#   list -- but a Vast.ai host in a US datacenter is a question worth
#   answering before you commit a 200 GB volume.
#
#   Attribution ("MiniMax H3" shown in-product) is required for commercial
#   use; >$20M revenue needs separate written authorization. The larryvrh
#   Turbo LoRA is Apache-2.0 and SeedVR2 is Apache-2.0 -- separate licenses,
#   base weights still governed by the above.
#
# -----------------------------------------------------------------------------
# MEMORY
#
#   bf16 DiT (66.3) + bf16 encoder (48.0) = 114.3 GB of weights against 96 GB
#   of VRAM. That works ONLY because the encoder runs once up front and is
#   evicted before the DiT loads -- but the evicted copy has to land in system
#   RAM or it gets re-read from disk every run. Budget 128 GB RAM minimum,
#   192 GB comfortable. This is 4.6 GB tighter than the I2V build.
#
#   On ComfyUI 0.30.x there is a pinned-memory regression that makes model
#   loading pathologically slow. If load times look wrong, launch with
#   --disable-pinned-memory.
# =============================================================================

set -o pipefail

mkdir -p "${WORKSPACE:-/workspace}"
exec > >(tee -a "${WORKSPACE:-/workspace}/provisioning.log") 2>&1
echo ""
echo "########## provisioning run (MiniMax H3 REF2V v2): $(date -u '+%Y-%m-%d %H:%M:%S UTC') ##########"

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
NODES_DIR="${COMFY}/custom_nodes"

COMFY_UPDATE="${COMFY_UPDATE:-1}"
NODE_UPDATE="${NODE_UPDATE:-1}"
WANT_ACCEL="${WANT_ACCEL:-1}"
ACCEL_PDD="${ACCEL_PDD:-1}"
ACCEL_LIGHTX2V="${ACCEL_LIGHTX2V:-1}"
WANT_EMBEDDINGS="${WANT_EMBEDDINGS:-0}"
WANT_CONTEXT_LOOP="${WANT_CONTEXT_LOOP:-0}"
WANT_SEEDVR2="${WANT_SEEDVR2:-0}"
ALLOW_BRANCH_RECOVERY="${ALLOW_BRANCH_RECOVERY:-0}"
FORCE_DEPS="${FORCE_DEPS:-0}"
RESTART_COMFY_ON_UPDATE="${RESTART_COMFY_ON_UPDATE:-0}"
echo "[provisioning] comfy_update=${COMFY_UPDATE} node_update=${NODE_UPDATE} seedvr2=${WANT_SEEDVR2}"
echo "[provisioning] accel=${WANT_ACCEL} (pdd=${ACCEL_PDD} lightx2v=${ACCEL_LIGHTX2V}) embeddings=${WANT_EMBEDDINGS} ctxloop=${WANT_CONTEXT_LOOP}"
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
# safe.directory -- must run before the first git invocation.
# ai-dock provisions as root; a persistent volume is often owned by another
# uid; modern git then refuses every operation with "dubious ownership" and
# git_sync's error paths absorb each failure individually, so the run looks
# normal while nothing updates.
# ---------------------------------------------------------------------------
# BUG FIX A: --add appends unconditionally, so v7/v1 grew this file by one
# identical line on every single boot. Check before writing.
if git config --global --get-all safe.directory 2>/dev/null | grep -qx '\*'; then
    echo "[git] safe.directory already configured"
elif git config --global --add safe.directory '*' 2>/dev/null; then
    echo "[git] safe.directory configured (prevents 'dubious ownership' no-ops)"
else
    echo "[git] WARNING: could not set safe.directory -- git ops may fail silently"
fi

# ---------------------------------------------------------------------------
# Broken flash-attn guard  --  RUNS BEFORE THE PIN FILE IS WRITTEN
#
#   <any custom node> -> diffusers -> xformers.ops -> flash_attn.flash_attn_interface
#   ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
#
# The flash_attn directory is present enough that `import flash_attn` succeeds
# (often as a namespace package -- the residue of a failed source build), so
# xformers concludes flash attention is available, takes that path, and dies on
# the submodule. Every pack downstream of diffusers dies with it. Cleanly
# absent is FINE, so the fix is removal, not repair.
#
# Must run before write_torch_pins(): a half-installed package still carries
# dist-info metadata, so it would otherwise be pinned into the constraints file
# and held in place for the whole run.
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

echo "=================== HOST MEMORY ==================="
RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [[ -n "$RAM_GB" ]]; then
    echo "[mem] system RAM: ${RAM_GB} GB"
    if (( RAM_GB < 128 )); then
        echo "[mem] !!! Under 128 GB. The bf16 DiT (66.3) + bf16 encoder (48.0)"
        echo "[mem] !!! pairing will thrash: the evicted text encoder cannot stay"
        echo "[mem] !!! resident and gets re-read from disk on every run. Either"
        echo "[mem] !!! move to a higher-RAM machine, or switch the text encoder"
        echo "[mem] !!! to int8_convrot in the manifest below."
    elif (( RAM_GB < 192 )); then
        echo "[mem] adequate. Note ref2va is 4.6 GB larger than fl2va, so this is"
        echo "[mem] tighter than the I2V build at the same RAM."
    else
        echo "[mem] comfortable."
    fi
fi

# ---------------------------------------------------------------------------
# git_sync <repo_path> <label> [pin]
# Return codes:  0 = HEAD moved   1 = could not update   2 = already current
#
# A detached HEAD is NOT "repaired" by default -- images pin ComfyUI core to a
# release tag on purpose, and walking it onto master breaks launch flags and
# custom nodes. Pass a pin to move it deliberately, or ALLOW_BRANCH_RECOVERY=1.
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

# ---------------------------------------------------------------------------
# H3 NODE-CLASS PROBE  --  the fact, not a proxy for it
#
# v7 gated on a version string. This greps the extension module for the class
# names actually required. If ComfyUI ships the node under a name this misses,
# the probe says INCONCLUSIVE rather than asserting a failure.
# ---------------------------------------------------------------------------
CV="$(comfy_version)"
MIN_CV="0.30.0"
echo "[comfy] after: ${CV:-<undetectable>}"

H3_NODES_FILE="${COMFY}/comfy_extras/nodes_minimax_h3.py"
echo "=================== H3 NODE PROBE ==================="
if [[ -f "$H3_NODES_FILE" ]]; then
    if grep -q 'MiniMaxH3ReferenceToVideo' "$H3_NODES_FILE"; then
        echo "[h3] OK      MiniMaxH3ReferenceToVideo present -- ref2v graphs will build"
    else
        echo "[h3] !!! MiniMaxH3ReferenceToVideo NOT FOUND in ${H3_NODES_FILE}."
        echo "[h3] !!! This is a REF2V build; without that node there is no graph to"
        echo "[h3] !!! run. It landed in ComfyUI 0.30.0. Detected: ${CV:-<unknown>}."
        echo "[h3] !!! Fix the core update before fetching 121 GB of weights."
    fi
    if grep -q 'MiniMaxH3AddGuide' "$H3_NODES_FILE"; then
        echo "[h3] OK      MiniMaxH3AddGuide present -- arbitrary-frame anchoring available"
        echo "[h3]         (this is the native path for clip extension on ref2v:"
        echo "[h3]          anchor the tail of the previous clip at frame_idx 0)"
    else
        echo "[h3] absent  MiniMaxH3AddGuide -- landed in ComfyUI 0.34.0 (PR #15439)."
        echo "[h3]         Without it, guides anchor only at first/last frame."
    fi
    grep -q 'denoise_mask' "$H3_NODES_FILE" \
        && echo "[h3] OK      per-token noise masks present (PR #15375)" \
        || echo "[h3] absent  per-token noise masks (PR #15375) -- no latent inpainting"
else
    echo "[h3] INCONCLUSIVE: ${H3_NODES_FILE} does not exist."
    echo "[h3] Either core is below 0.30.0, or the module was renamed upstream."
    echo "[h3] Check the node menu for 'MiniMaxH3' before trusting this build."
fi

if [[ -n "$CV" ]]; then
    if [[ "$(printf '%s\n%s\n' "$MIN_CV" "$CV" | sort -V | head -1)" == "$MIN_CV" ]]; then
        echo "[comfy] version ${CV} >= ${MIN_CV} (secondary signal; the probe above is the gate)"
    else
        echo "[comfy] !!! ${CV} is BELOW ${MIN_CV}."
    fi
fi

# BUG FIX F: the PDD pack needs a newer core than the ref2v node does. Only
# worth saying when that path is actually enabled.
PDD_MIN_CV="0.33.0"
if [[ "$WANT_ACCEL" == "1" && "$ACCEL_PDD" == "1" ]]; then
    if [[ -z "$CV" ]]; then
        echo "[h3] PDD: core version undetectable -- verify you are on >= ${PDD_MIN_CV}"
    elif [[ "$(printf '%s\n%s\n' "$PDD_MIN_CV" "$CV" | sort -V | head -1)" == "$PDD_MIN_CV" ]]; then
        echo "[h3] OK      core ${CV} >= ${PDD_MIN_CV} -- PDD Acc adapter supported"
    else
        echo "[h3] !!! core ${CV} is below ${PDD_MIN_CV}. The PDD Acc node needs the"
        echo "[h3] !!! carried-audio mechanics from PR #15243 and will fail closed."
        echo "[h3] !!! The 1.4 GB adapter still downloads; it just will not load."
        echo "[h3] !!! Either update core or set ACCEL_PDD=0 and use the lightx2v"
        echo "[h3] !!! ref2v v0.1 file, which needs no custom node."
    fi
fi

# ---------------------------------------------------------------------------
# Custom nodes
#
# Entry format:  <git-url>[|<clone-dir-name>][@<pin>]
#
# NOT here by default:
#   - the SeedVR2 upscaler pack (SeedVR2 is core now)
#   - ethanfel's context-loop pack (fl2va-oriented; WANT_CONTEXT_LOOP=1)
# ---------------------------------------------------------------------------
NODES=(
    # Video load/save, frame extraction, the VHS_* family.
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    # Utility layer: resolution selector, torch compile, misc graph plumbing.
    # Also carries Patch Sage Attention KJ if you prefer the node to the flag.
    "https://github.com/kijai/ComfyUI-KJNodes"
    # Interactive drag-to-crop on the image preview. No pip dependencies.
    # More useful on ref2v than on I2V: you are framing up to 9 references.
    "https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop"
    # Frame interpolation. Pulls the TensorRT stack; builds its engine on
    # first use, per GPU architecture.
    "https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto"
)

# The PDD Acc pack. Required to load MiniMax-H3-Ref2VA-Acc-8Step at all: a
# plain LoraLoader reads the rank-64 trunk and silently discards the 32-entry
# PDD head bank, leaving you with a model that runs and has lost the distill.
#
# Two things make this a low-risk install despite being a third H3 node family:
#   - it ships NO requirements.txt, so it cannot move your torch stack;
#   - its Apply node fails closed on a wrong sigma shift, a wrong step count,
#     an unpatched model, or a core older than 0.33.0, rather than degrading.
# Nodes it registers: MiniMaxH3PDDAccApply, MiniMaxH3PDDAccScheduler,
# MiniMaxH3PDDAccWarmupScheduler, MiniMaxH3AVLatentUpscaleBy. None of those
# names collide with core's MiniMaxH3* set.
[[ "$WANT_ACCEL" == "1" && "$ACCEL_PDD" == "1" ]] && \
    NODES+=( "https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc" )

# ethanfel's context loop. Built around fl2va first/last-frame extension. I
# have no evidence about its behaviour on ref2va, and MiniMaxH3AddGuide covers
# the same ground natively on 0.34.0+. Opt-in only.
[[ "$WANT_CONTEXT_LOOP" == "1" ]] && \
    NODES+=( "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop" )

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
# SageAttention -- opt-in, DRAFTS ONLY.
#
# Approximate attention: quantized QK^T with a smoothing correction. ~2x
# throughput, small but nonzero error. Fine while iterating prompts and
# reference assignments, bad on a final render.
#
# Expect console noise on H3: Sage needs fp16/bf16 tensors and H3 runs some
# layers in other dtypes, so you will see "using pytorch attention instead"
# messages. Those are expected, not a misconfiguration.
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

# ---------------------------------------------------------------------------
# probe_url <url>  ->  prints "<status> <size>" on one line
#
# NEW: v7's remote_size() collapsed every failure into an empty string, so a
# malformed manifest entry, an expired token, a gated repo and a transient 5xx
# were indistinguishable in the log. They need different responses from you.
# ---------------------------------------------------------------------------
# BUG FIX B: memoise probes for the run. preflight_disk() and dl_hf() each
# probed every manifest entry, doubling the HEAD count for no benefit and
# making a 429 materially more likely on an unauthenticated pull.
declare -A _PROBE_CACHE 2>/dev/null || echo "[probe] NOTE: bash < 4, probe cache disabled"

probe_url() {
    local key="$1"
    if [[ -n "${_PROBE_CACHE[$key]+x}" ]]; then
        printf '%s' "${_PROBE_CACHE[$key]}"
        return 0
    fi
    local out; out="$(_probe_url_uncached "$1")"
    _PROBE_CACHE["$key"]="$out"
    printf '%s' "$out"
}

_probe_url_uncached() {
    local url; url="$(map_url "$1")"
    local headers status val
    headers="$(curl -sIL -w '\nHTTPSTATUS:%{http_code}\n' --connect-timeout 15 --max-time 60 \
                    "${CURL_AUTH[@]}" "$url" 2>/dev/null)" || { printf '000 '; return 0; }
    status="$(printf '%s' "$headers" | awk -F: '/^HTTPSTATUS:/{s=$2} END{gsub(/[^0-9]/,"",s); print s}')"
    val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-size"{v=$2} END{if(v!="")print v}')"
    [[ -z "$val" ]] && val="$(printf '%s' "$headers" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{if(v!="")print v}')"
    printf '%s %s' "${status:-000}" "${val//[^0-9]/}"
}

# classify_status <status> <label> -> echoes a human diagnosis
classify_status() {
    local st="$1" label="$2"
    case "$st" in
        2*) return 0 ;;
        404|410)
            echo "[probe] ${label}: HTTP ${st} -- the file is not at that path."
            echo "[probe] ${label}: This is a MANIFEST bug, not an auth or network"
            echo "[probe] ${label}: problem. Check repo_id and repo_path against the"
            echo "[probe] ${label}: repo's file tree. A token will not fix it."
            return 1 ;;
        401)
            echo "[probe] ${label}: HTTP 401 -- HF_TOKEN missing, expired, or wrong scope."
            return 1 ;;
        403)
            echo "[probe] ${label}: HTTP 403 -- gated repo, or the token lacks access."
            echo "[probe] ${label}: Accept the license on the model page with the same"
            echo "[probe] ${label}: account the token belongs to."
            return 1 ;;
        429)
            echo "[probe] ${label}: HTTP 429 -- rate limited. Set HF_TOKEN, or wait."
            return 1 ;;
        000)
            echo "[probe] ${label}: no response (DNS, egress block, or timeout)."
            return 1 ;;
        5*)
            echo "[probe] ${label}: HTTP ${st} -- upstream error, transient. Retries next boot."
            return 1 ;;
        *)
            echo "[probe] ${label}: HTTP ${st} -- unexpected."
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

# ---------------------------------------------------------------------------
# Nested-duplicate sweep.
#
# hf download --local-dir writes <dir>/<repo_path>, i.e. a nested tree, while
# dl_hf writes <dir>/<name> flat. If a file was ever fetched both ways there
# are two copies of a 66 GB model on the volume and only one is visible to
# ComfyUI. Report, do not delete.
# ---------------------------------------------------------------------------
sweep_nested_dupes() {
    local dir="$1" name="$2" nested
    for nested in "${dir}/diffusion_models/${name}" "${dir}/loras/${name}" \
                  "${dir}/vae/${name}" "${dir}/text_encoders/${name}" \
                  "${dir}/embeddings/${name}" "${dir}/pdd_acc/${name}"; do
        [[ -f "$nested" ]] || continue
        echo "[dupe] ${nested}"
        echo "[dupe]   is a nested copy of ${name} ($(numfmt --to=iec "$(stat -c%s "$nested")" 2>/dev/null))."
        echo "[dupe]   ComfyUI reads the flat path only. Remove it to reclaim the space."
    done
}

dl_hf() {
    local dir="$1" name="$2" repo="$3" rpath="$4"
    local dest="${dir}/${name}"
    local check_url; check_url="$(hf_resolve_url "$repo" "$rpath")"
    mkdir -p "$dir"

    local probe status want have=0
    probe="$(probe_url "$check_url")"
    status="${probe%% *}"; want="${probe##* }"

    if ! classify_status "$status" "$name"; then
        if [[ -f "$dest" ]]; then
            echo "[model] $name present locally; probe failed, leaving it alone"
            return 0
        fi
        echo "[model] SKIPPING $name -- probe failed and no local copy exists"
        return 0
    fi

    [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
    if [[ -f "$dest" ]]; then
        if [[ -n "$want" ]] && (( have == want )); then
            echo "[model] $name complete (${have} bytes), skipping"
            return 0
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
            # BUG FIX C: prune the staging tree ONLY on a verified download.
            # On failure the partial must survive -- that is what lets the next
            # boot resume instead of restarting a 66 GB transfer.
            rm -rf "${dir}/.hf_stage" 2>/dev/null || true
        fi
    else
        echo "[model] DOWNLOAD FAILED: $name (partial kept in .hf_stage for resume)"
    fi
}

CIVITAI_RESERVE_GB="${CIVITAI_RESERVE_GB:-20}"

dl_civitai() {
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
# Manifest  --  ~121 GB base, ~136 GB with SeedVR2
#
# Format:  hf | dest_dir | dest_filename | repo_id | repo_path      (5 fields)
# Sizes are the actual repo figures where verified, not estimates.
# ---------------------------------------------------------------------------
DIFF="${COMFY}/models/diffusion_models"
LORA="${COMFY}/models/loras"
TE="${COMFY}/models/text_encoders"
VAE="${COMFY}/models/vae"
EMB="${COMFY}/models/embeddings"
PDD="${COMFY}/models/pdd_acc"
WF="${COMFY}/user/default/workflows"
# BUG FIX G: the PDD node creates models/pdd_acc on first launch, but the
# manifest writes into it before ComfyUI has ever started on a fresh volume.
mkdir -p "$EMB" "$PDD" "$DIFF" "$LORA" "$TE" "$VAE"

# --- Civitai:  dest_dir | dest_filename | full_url      (3 fields) ---
# Carried over from the I2V build. NOTE these are fl2va-trained H3 LoRAs; the
# same branch caveat in the header applies to every one of them on ref2va.
CIVITAI_FILES=(
    "$LORA|H3_Epic_Cumshots.safetensors|https://civitai.red/api/download/models/3202064?fileId=3083352"
    "$LORA|H3_HMNSFW_AIO_Sex_v2.safetensors|https://civitai.red/api/download/models/3206518?fileId=3088013"
    "$LORA|H3_Mini_Dick_Fix.safetensors|https://civitai.red/api/download/models/3207332?fileId=3088892"
    "$LORA|H3_K3NK_Side_View_Deepthroat.safetensors|https://civitai.red/api/download/models/3216591?fileId=3098396"
    "$LORA|H3_HMPenis_Cock.safetensors|https://civitai.red/api/download/models/3247473?fileId=3130327"
)

MODELS=(
    # === DIFFUSION MODEL -- ref2va, FULL bf16 (66.3 GB) ==================
    # THIS IS THE LINE THAT MAKES IT A REF2V BUILD. ref2va and fl2va are
    # different weights, not different modes of one file. The R2V template
    # and MiniMaxH3ReferenceToVideo both require this checkpoint.
    #
    # Give-backs, in order (see the header on the cu130 caveat):
    #   int8_convrot        34 GB
    #   pruned_int8_convrot 21 GB   <- ComfyUI's own R2V template default
    "hf|$DIFF|minimax_h3_ref2va_bf16.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_ref2va_bf16.safetensors"
    # "hf|$DIFF|minimax_h3_ref2va_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors"
    # "hf|$DIFF|minimax_h3_ref2va_pruned_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"

    # === TEXT ENCODER -- FULL bf16 (48.0 GB) =============================
    # Qwen3-VL-32B. Shared between branches -- same encoder for fl2va and
    # ref2va, so if the I2V volume already has this file, it is the same file.
    # int8_convrot is 25.3 GB; nvfp4_awq is 14.6 GB and does NOT require a
    # Blackwell GPU despite the name.
    "hf|$TE|qwen3vl_32b_minimax_h3_bf16.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
    # "hf|$TE|qwen3vl_32b_minimax_h3_int8_convrot.safetensors|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

    # === H3 VAEs -- BOTH REQUIRED (4.9 + 0.6 GB) =========================
    # The audio VAE is not optional even for silent output: the audio stream
    # is generated in the same pass and has to be decoded. On ref2v this
    # matters more, not less -- reference VIDEOS can carry their own
    # soundtracks into the conditioning.
    "hf|$VAE|minimax_h3_video_vae_fp16.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_fp16.safetensors"
    "hf|$VAE|minimax_h3_audio_vae_fp32.safetensors|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"
)

# === ACCELERATION TIER -- BOTH ADAPTERS ARE ref2va-TRUNK-MATCHED ==========
if [[ "$WANT_ACCEL" == "1" ]]; then

    # --- PRIMARY: Alibaba PAI PDD Acc 8-step, Ref2VA trunk (1.4 GB) -------
    # Goes in models/pdd_acc/, NOT models/loras/. Loaded by
    # MiniMaxH3PDDAccApply from the ComfyUI-MiniMax-H3-PDD-Acc pack above.
    #
    # DO NOT LOAD THIS WITH LoraLoader. It will appear to work. The rank-64
    # trunk will apply, the 32-entry PDD head bank will be dropped without a
    # warning, and you will be sampling an undistilled model at 8 steps and
    # wondering why it looks soft.
    #
    # Trained recipe, all of it enforced by the node rather than suggested:
    #   sampler   euler (KSamplerSelect)
    #   guidance  CFG 1.0, BasicGuider
    #   shift     12.0 / 3.0 exactly -- the node fails closed otherwise
    #   steps     8 (trained block size) | 4 (sanctioned) | 6 (8,8,4,4,4,4)
    #   strength  lora_strength 1.0 / head_strength 1.0
    #
    # License is the MiniMax H3 Community License, NOT Apache-2.0 -- see the
    # header. Same excluded territories as the base weights.
    if [[ "$ACCEL_PDD" == "1" ]]; then
    MODELS+=(
        "hf|$PDD|MiniMax-H3-Ref2VA-Acc-8Step.safetensors|alibaba-pai/MiniMax-H3-Acc-LoRAs|MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
    )
    # Pre-converted ComfyUI-key redistribution (1.7 GB). The node auto-detects
    # either format, so this is only worth fetching if you want to skip the
    # in-memory key conversion at load. Third-party repack, not upstream.
    # MODELS+=(
    #     "hf|$PDD|minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors|aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI|minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors"
    # )
    fi

    # --- FALLBACK: lightx2v / ModelTC Ref2VA Turbo 4-step v0.1 ------------
    # The branch-native community adapter, and still the only one from that
    # line -- the fl2v side has moved to v1.0/v1.1 768p variants, ref2v has
    # not moved since 2026-08-13. Upstream calls it a preview whose image
    # detail needs improvement.
    #
    # Worth keeping on disk regardless of whether you draft with it: the
    # official ComfyUI R2V template scans for this filename and flags it
    # missing. It loads through the native Lightning LoRA checkbox, so it is
    # the option that needs no custom node at all.
    #
    #   trained at  544p, mixed aspect ratio  <- NOT 768p; see header item 6
    #   shift       12 / 3
    #   steps       4 (distillation NFE and recommended NFE)
    #   Apache-2.0
    #
    # Sourced from Comfy-Org rather than lightx2v because that is the copy
    # ComfyUI's own template references. Upstream original, if you prefer it:
    #   lightx2v/Minimax-h3-Turbo | minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
    if [[ "$ACCEL_LIGHTX2V" == "1" ]]; then
    MODELS+=(
        "hf|$LORA|minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors|Comfy-Org/MiniMax-H3|loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    fi
fi

# NOT FETCHED, and why -- so this does not get re-litigated in three weeks:
#
#   larryvrh minimax_h3_turbo_v4_step600_ema
#     fl2va-distilled. Model card scopes itself to t2v/i2v, the standalone
#     script hardcodes an fl2va base, and there is a thread on its own model
#     card ("Does it support ref2v?") reporting failures with reference video,
#     plus an audio-reference bug filed against the node repo. Keys match
#     ref2va, so it applies silently. Removed in v2.
#
#   fl2v 8-step turbo v1.0 + Kijai minimax_h3_ref_lora_rank_256_bf16
#     A community workaround that several people report working at 8-10 steps:
#     drive the ref2va NODE from fl2va weights plus a Kijai "ref" LoRA. Other
#     users in the same threads report it erroring on the Kijai file. It is two
#     stacked adapters standing in for a trunk that already exists as real
#     weights, and stacking a distill LoRA with anything else is exactly what
#     you have already found does not work. If you want to try it, it lives at
#     Kijai/MiniMax-H3-experimental in loras/ -- but test the two adapters
#     above first.

# === STYLE EMBEDDINGS (opt-in, small) ====================================
# Invoked in a prompt as `embedding:<filename-without-extension>`. Community
# contributions via silveroxides, merged into the Comfy-Org repo by PR -- NOT
# produced by MiniMax or Comfy-Org. Requires a core with PR #15697.
if [[ "$WANT_EMBEDDINGS" == "1" ]]; then
    for _e in art_is_explosion blooming_flowers bullet_time dark_magic \
              fire_breath four_seasons kiss_camera spiral_ascent \
              storm_magic truman_show; do
        MODELS+=( "hf|$EMB|minimaxh3_${_e}.safetensors|Comfy-Org/MiniMax-H3|embeddings/minimaxh3_${_e}.safetensors" )
    done
fi

# === SeedVR2 RESTORE STAGE -- NATIVE WEIGHTS (~15 GB) ====================
# Comfy-Org conversions for ComfyUI's native SeedVR2 nodes. NOT the
# numz/SeedVR2_comfyUI files -- different key naming, different loaders,
# different directories, not interchangeable.
#
# The DiT loads through a stock UNETLoader and the VAE through a stock
# VAELoader, which is the whole reason to prefer this path: ComfyUI's memory
# manager owns the model and evicts the 66.3 GB H3 DiT for you.
#
# 7B fp16 deliberately. Not fp8 (quality), and NOT the sharp variant -- sharp
# crisps whatever texture is present, and at the 768 ceiling what is present
# is mottling. You want it overwritten, not sharpened.
if [[ "$WANT_SEEDVR2" == "1" ]]; then
MODELS+=(
    "hf|$DIFF|seedvr2_7b_fp16.safetensors|Comfy-Org/SeedVR2|diffusion_models/seedvr2_7b_fp16.safetensors"
    "hf|$VAE|seedvr2_ema_vae_fp16.safetensors|Comfy-Org/SeedVR2|vae/seedvr2_ema_vae_fp16.safetensors"
)
fi

# ---------------------------------------------------------------------------
# NEW: manifest field-count validation.
#
# `IFS='|' read -r kind a b c d` silently accepts a 4-field entry, binding the
# repo PATH into the repo_ID slot and leaving the path empty. The resulting URL
# 404s and the only symptom is "DOWNLOAD FAILED", which reads like a network
# problem. v7 shipped with exactly this bug in its WANT_TURBO block, so
# WANT_TURBO=1 downloaded nothing. Catch it here, name the entry, and refuse
# to start a 121 GB pull against a malformed manifest.
# ---------------------------------------------------------------------------
validate_manifest() {
    local bad=0 entry n i=0
    for entry in "${MODELS[@]}"; do
        i=$(( i + 1 ))
        n="$(awk -F'|' '{print NF}' <<< "$entry")"
        if (( n != 5 )); then
            echo "[manifest] !!! MODELS entry ${i} has ${n} fields, expected 5:"
            echo "[manifest] !!!   ${entry}"
            echo "[manifest] !!! Format: hf|dest_dir|dest_filename|repo_id|repo_path"
            bad=1
        fi
        [[ "${entry%%|*}" == "hf" ]] || {
            echo "[manifest] !!! MODELS entry ${i} kind is '${entry%%|*}', expected 'hf'"
            bad=1
        }
    done
    i=0
    for entry in "${CIVITAI_FILES[@]}"; do
        i=$(( i + 1 ))
        n="$(awk -F'|' '{print NF}' <<< "$entry")"
        if (( n != 3 )); then
            echo "[manifest] !!! CIVITAI_FILES entry ${i} has ${n} fields, expected 3:"
            echo "[manifest] !!!   ${entry}"
            bad=1
        fi
    done
    if (( bad )); then
        echo "[manifest] !!! Refusing to run the fetch phase against a malformed manifest."
        return 1
    fi
    echo "[manifest] OK -- ${#MODELS[@]} HF entries, ${#CIVITAI_FILES[@]} civitai entries, all well-formed"
    return 0
}

# ---------------------------------------------------------------------------
# Disk pre-flight
# ---------------------------------------------------------------------------
preflight_disk() {
    local need=0 kind a b c d url dest have probe status want entry
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r kind a b c d <<< "$entry"
        [[ "$kind" == "hf" ]] || continue
        url="$(hf_resolve_url "$c" "$d")"; dest="${a}/${b}"
        have=0; [[ -f "$dest" ]] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        probe="$(probe_url "$url")"; status="${probe%% *}"; want="${probe##* }"
        classify_status "$status" "$b" >/dev/null 2>&1 || continue
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
        echo "[provisioning] !!!   an fl2va DiT left over from the I2V build          -61.7 GB"
        echo "[provisioning] !!!     (ref2v cannot use it -- check models/diffusion_models)"
        echo "[provisioning] !!!   text encoder -> int8_convrot                       -22.7 GB"
        echo "[provisioning] !!!   text encoder -> nvfp4_awq                          -33.4 GB"
        echo "[provisioning] !!!   diffusion    -> ref2va_int8_convrot                -32.3 GB"
        echo "[provisioning] !!!   diffusion    -> ref2va_pruned_int8_convrot         -45.3 GB"
        echo "[provisioning] !!!   WANT_SEEDVR2=0                                     -15.0 GB"
        echo "[provisioning] !!!   WANT_ACCEL=0                                        -2.9 GB"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# An fl2va DiT sitting on this volume is 61.7 GB that ref2v cannot use.
# Report it -- do not delete it, you may be dual-booting both builds.
# ---------------------------------------------------------------------------
echo "=================== CROSS-BRANCH DISK CHECK ==================="
_fl_found=0
for _f in "$DIFF"/minimax_h3_fl2va_*.safetensors; do
    [[ -f "$_f" ]] || continue
    _fl_found=1
    echo "[branch] fl2va checkpoint present: $(basename "$_f") ($(numfmt --to=iec "$(stat -c%s "$_f")" 2>/dev/null))"
done
if (( _fl_found )); then
    echo "[branch] These are I2V/T2V weights. MiniMaxH3ReferenceToVideo cannot use"
    echo "[branch] them. If this volume is ref2v-only, that is reclaimable space."
    echo "[branch] If you dual-boot both provisioning scripts, leave them."
else
    echo "[branch] no fl2va checkpoints on this volume"
fi

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
echo "=================== MANIFEST VALIDATION ==================="
if ! validate_manifest; then
    echo "[provisioning] !!! ABORTING the model phase. Fix the manifest above."
    MANIFEST_OK=0
else
    MANIFEST_OK=1
fi

if (( MANIFEST_OK )) && (( ${#CIVITAI_FILES[@]} > 0 )); then
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

if (( MANIFEST_OK )); then
    echo "=================== HUGGING FACE ==================="
    echo "[provisioning] NOTE: base build is ~121 GB. First boot on a fresh volume"
    echo "[provisioning] NOTE: is a long pull -- the ref2va bf16 DiT alone is 66.3 GB."
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

# BUG FIX D: run the nested-duplicate sweep over EVERY manifest entry, not
# just the ones that happened to be complete when dl_hf looked at them.
for entry in "${MODELS[@]}"; do
    IFS='|' read -r kind a b c d <<< "$entry"
    [[ "$kind" == "hf" ]] || continue
    sweep_nested_dupes "$a" "$b"
done

for nd in ComfyUI-VideoHelperSuite ComfyUI-KJNodes ComfyUI-Olm-DragCrop \
          ComfyUI-RIFE-TensorRT-Auto ComfyUI-MiniMax-H3-PDD-Acc \
          ComfyUI-MiniMaxH3-Contex-Loop; do
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
# Read stdout_logfile out of the supervisor program block rather than guessing,
# and grab `command=` so a failed startup can be reproduced in the foreground.
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

# supervisor reports RUNNING once a process survives startsecs (5s default).
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

=================== FINAL-RENDER SETTINGS (REF2V) ===================

MODELS
  diffusion    minimax_h3_ref2va_bf16.safetensors      <- NOT fl2va
  text encoder qwen3vl_32b_minimax_h3_bf16.safetensors
  vae          minimax_h3_video_vae_fp16 + minimax_h3_audio_vae_fp32
  adapters     none for finals. Bypass MiniMaxH3PDDAccApply, and clear the
               Lightning LoRA checkbox on the R2V node.

  The node is MiniMaxH3ReferenceToVideo, not MiniMaxH3ImageToVideo. If you
  built this graph by editing an I2V one, check you swapped BOTH the node and
  the checkpoint. Swapping only one is the most likely way to get a graph that
  runs, produces something, and ignores your references.

SAMPLER
  res_multistep + simple scheduler, 25-30 steps.
  Below ~15 steps quality drops visibly. 20 is the community baseline; 25-30
  is where you stop getting much for the wall-clock.

REFERENCE INPUTS -- the part that is genuinely new
  Limits: 9 images, 3 videos (each may carry its own soundtrack), 3 audio.

  TAGS ARE POSITIONAL. Address references as <Picture 1>, <Video 1>,
  <Audio 1>, numbered in CONNECTION order, not by any name you assign. Rewire
  an input and every tag downstream of it renumbers. A prompt that worked
  yesterday will silently address the wrong asset. When a shot comes out
  wrong on ref2v, check the tag-to-socket mapping BEFORE you touch the
  prompt text -- and certainly before you reroll. Wrong-socket failures
  reproduce identically across seeds, exactly like a schema bug.

  ASSIGN EACH REFERENCE A JOB. identity / style / motion / camera / voice.
  "Use <Picture 1> and <Picture 2>" leaves the relationship unresolved and
  the model gap-fills it. This is the same CFG-1 gap-filling behaviour you
  already know from the I2V prompts, applied to reference roles instead of
  motion axes: whatever is most recently primed fills the underspecified
  slot.

  ref_image_size -- SPLIT THE ANSWER BY TIER
    match -> matches reference pixel area to the target canvas, preserving
             aspect, never upscaling.  min(1, sqrt(target_area / ref_area))
    max   -> preserves aspect, scales down only when the reference short edge
             exceeds 2048px.           min(1, 2048 / ref_short_edge)

    On an ACCELERATED tier use match. Upstream distills with match and
    recommends it for distilled models, so it is the training-matched policy
    rather than just the cheap one.
    On the UNDISTILLED 25-30 step final, max buys identity fidelity.

    It is also the first dial to reach for on an OOM, before precision: the
    bf16 DiT leaves under 30 GB, and nine references at a 2048px short edge is
    a real allocation against that.

  PROMPT GUIDE: reference mode has its OWN guide,
  VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, not the base-mode one. It defines a
  different rewrite structure -- subject definitions, reference labels,
  retention analysis. The base guide does not transfer.

ACCELERATION TIER -- TWO OPTIONS, BOTH ref2va-TRUNK-MATCHED

  PRIMARY: MiniMax-H3-Ref2VA-Acc-8Step        (models/pdd_acc/)
    Alibaba PAI, Parallel Decoding Distillation, published 2026-08-26 as two
    trunk-specific adapters. This is the ref2va one -- pair it with a ref2va
    UNET, bf16 or int8_convrot.

    LOAD IT WITH MiniMaxH3PDDAccApply. NOT LoraLoader. The file carries a
    rank-64 trunk LoRA AND a 32-entry PDD head bank. A plain LoRA loader reads
    the trunk, drops the head bank, reports nothing, and leaves you sampling an
    undistilled model at 8 steps. Same silent-failure shape as an fl2va LoRA on
    ref2va keys, different mechanism.

    The recipe is enforced, not suggested -- the node fails closed:
      sampler    euler, via KSamplerSelect
      guidance   CFG 1.0, BasicGuider
      shift      12.0 / 3.0 exactly. Anything else is an error.
      steps      8 default (the trained block size)
                 4 officially sanctioned regrouping
                 6 via the non-uniform 8,8,4,4,4,4 partition
                 anything else is rejected rather than degraded
      strength   lora_strength 1.0, head_strength 1.0
    More than 8 steps is not "closer to the base model" -- read the pack's
    README before you reach for it.

    The pack also ships a Warmup Scheduler that splits the sigmas into a
    two-phase pass specifically for better reference likeness. That is a
    ref2v-shaped feature and worth trying before you conclude the adapter is
    costing you identity.

    Needs ComfyUI >= 0.33.0 (PR #15243, carried-audio mechanics).
    License: MiniMax H3 Community License, same excluded territories as the
    base weights. Not Apache-2.0, whatever the node README says.

  FALLBACK: minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16   (models/loras/)
    lightx2v / ModelTC, 2026-08-13. Loads through the native Lightning LoRA
    checkbox on the R2V node -- no custom node, no version floor beyond 0.30.0.
    The official R2V template scans for this filename, so keep it on disk even
    if you never draft with it.
      shift      12 / 3
      steps      4
      trained    544p, MIXED ASPECT RATIO
      Apache-2.0

    THE 544p FACT MATTERS AND IT CUTS AGAINST THE HOUSE RULE. Your standing
    position -- draft at native canvas, cut frames and steps rather than
    resolution -- is a claim about the BASE model's RoPE and shift
    calibration. It does not automatically transfer to an adapter distilled
    somewhere else. The proof is in the fl2v line: when the same team
    retrained that turbo at 768p they had to move the shift from 12/3 to 6/3.
    Shift is coupled to training resolution. A 544p adapter carrying 12/3 is
    calibrated for 544p.

    So on this fallback, draft at 960x544 (16:9, 0.5 MP) -- which is what
    upstream's own ref2v example graph defaults to -- and accept that
    composition will shift when you go to 1344x768 for the final. That is the
    opposite of your I2V draft discipline, and it is a property of this
    specific adapter rather than a change of principle.

    There is no 768p ref2v turbo in that line. If you want an accelerated tier
    at native canvas, PDD is the only candidate. Note that Alibaba label their
    FL2VA demo comparisons 768p and leave the Ref2VA ones unlabelled, so
    confirm the canvas behaviour yourself rather than assuming it.

  WHAT IS DELIBERATELY NOT HERE
    larryvrh v4-600. It is the stronger LoRA in absolute terms and it is
    fl2va-distilled. Its own model card has a thread titled "Does it support
    ref2v?" reporting failure with reference video, and an audio-reference bug
    filed against the node pack. The keys match ref2va so it applies in
    silence. v1 of this script fetched it with a warning; v2 does not fetch it.

    The fl2v-8step + Kijai ref-LoRA workaround. Several users report it works
    at 8-10 steps; others in the same threads report it erroring. It is two
    stacked adapters emulating a trunk that exists as real weights, and
    stacking a distill with anything else is the failure mode you already know.

  HOW TO JUDGE THESE
    Judge on reference adherence, not sharpness. The failure you are looking
    for is reference identity drifting while the frame still looks good, which
    is exactly what a fidelity-focused eyeball test misses. Fix a seed, fix a
    reference set, vary only the adapter, and compare identity retention.
    Set ref_image_size to match on both -- see below.

  DO NOT STACK the two adapters, or either one with any other distillation.
  Character and style LoRAs stack with either normally.

RESOLUTION -- and a correction to the I2V notes
  Resolution Selector: 16:9, Megapixels 0.98, Multiple 32  ->  1344x768.
  NOT 1.0. Megapixels 1.0 yields 1376x768, which is above the model's
  768x1344 pixel-area cap. The I2V script said ~1.0; that was one rounding
  step into off-distribution territory. Or bypass the selector entirely and
  type 1344 x 768 into the node.

  frames: 17k+5 grid at 24 fps. 124 ~ 5 s, 362 ~ 15 s. Validated 124-362.

  Mottled or patchy skin at this canvas is the resolution ceiling showing
  through, not a sampler misconfiguration. Do not chase it with sigma tuning.
  SeedVR2 is the lever.

FRAMING REFERENCE IMAGES (Olm DragCrop)
  More load-bearing here than on I2V, because you may be feeding nine of
  them. A reference at the wrong aspect gets letterboxed or stretched, and
  the model reads the letterboxing as part of the reference. Load Image ->
  Olm DragCrop -> the reference input. The graph does not evaluate until you
  run it, so you can frame all nine without burning a render.

AUDIO -- 32 kHz stereo
  The audio fields are NOT optional decoration. H3 runs joint video-audio
  attention, so an empty or broken audio conditioning path perturbs the video
  itself: motion timing drifts and boundary coherence degrades. Fill them
  even when you intend to mute the result.

  On ref2v you have a further path: reference VIDEOS carry their own
  soundtracks into conditioning, and standalone reference AUDIO can drive
  voice. If audio is misbehaving, check whether a reference is contributing
  audio you did not intend before you blame the sampler.

  Video and audio ride separate flow schedules (video shift 12, audio shift
  3). Those values are coupled and move together. At low step counts a
  single-schedule sampler over-steps the audio -- which is the mechanism
  behind most "the turbo LoRA broke my audio" reports, and it is a sampler
  path issue rather than a LoRA strength one.

PROMPTING AT CFG 1
  BasicGuider has no negative conditioning path. In-prompt negation ("no
  camera movement") primes the concept with nothing to subtract it and
  reliably backfires. Suppress an unwanted axis by over-specifying the wanted
  one in positive space -- name the rig type, state the frame invariants.
  Put LoRA trigger words at the FRONT. Qwen3-VL is decoder-only with causal
  attention, so a front-positioned token propagates conditioning across the
  whole sequence; a trailing one sits next to the audio slot and gets
  vocalised as speech.

=================== EXTENDING A CLIP ON REF2V ===================

  The native path is MiniMaxH3AddGuide (ComfyUI 0.34.0+, PR #15439), not a
  custom pack. It anchors an image or audio guide at any frame_idx on a
  continuous time axis; negative values count from the end.

    - Feed the last N frames of the previous clip (N on the 17k+5 grid: 5,
      22, 39...) plus their audio into an AddGuide at frame_idx 0. The model
      generates the continuation of BOTH streams.
    - Batches shorter than 5 frames use only the first image.
    - Wire the video VAE to `vae` for image guides, audio VAE to `audio_vae`
      for audio guides. Chain several AddGuide nodes for multiple anchors.
    - There is an official example graph: video_minimax_h3_r2v_addguides_v1.json

  CONTEXT FRAMES MUST COME FROM NATIVE-CANVAS DECODE, not from upscaled or
  restored output. Feeding a SeedVR2'd frame back in as context puts the
  conditioning off-distribution.

  ethanfel's ComfyUI-MiniMaxH3-Contex-Loop is NOT installed by default in
  this build. It is built around fl2va first/last-frame extension and I have
  no evidence about its behaviour on ref2va weights. WANT_CONTEXT_LOOP=1 if
  you want it anyway -- but test the native AddGuide path first, and be aware
  that running it alongside core H3 nodes and the larryvrh pack means three
  H3 node families in one install.

=================== RESTORE STAGE -- NATIVE SeedVR2 ===================

The SeedVR2 nodes are core, so they always match your ComfyUI version -- but
nothing ships pre-wired into the H3 templates, so the graph is yours to build.

  NODE SEARCH: type "SeedVR2". Five nodes:
    Pre-Process SeedVR2 Input      (SeedVR2Preprocess)
    Apply SeedVR2 Conditioning     (SeedVR2Conditioning)
    Split SeedVR2 Latent           (temporal chunking)
    Merge SeedVR2 Latents
    Post-Process SeedVR2 Output    (SeedVR2PostProcessing)

  THE CHAIN, spliced after the H3 video VAEDecode:

    VAEDecode (H3 video)
      -> Resize Image (multiplier 1.875, lanczos)
      -> Pre-Process SeedVR2 Input
      -> VAEEncodeTiled            (SeedVR2 VAE, 512 tile / 128 overlap)
      -> [Split SeedVR2 Latent]
      -> KSampler                  (1 step, cfg 1, euler, simple, denoise 1)
      -> [Merge SeedVR2 Latents]
      -> VAEDecodeTiled            (SeedVR2 VAE)
      -> Post-Process SeedVR2 Output
      -> RIFE -> CreateVideo

    VAEDecodeAudio ----------------AUDIO----------------> CreateVideo

  LOADERS -- stock nodes, not SeedVR2-branded ones:
    UNETLoader -> seedvr2_7b_fp16.safetensors
                  feeds BOTH Apply SeedVR2 Conditioning AND the KSampler
    VAELoader  -> seedvr2_ema_vae_fp16.safetensors
                  feeds BOTH VAEEncodeTiled and VAEDecodeTiled
  Two VAELoaders in one graph is correct. Do not share H3's video VAE here.

  THE SECOND INPUT ON POST-PROCESS. Wire Resize Image to BOTH the
  pre-processor and the original_resized_images socket. It is the reference
  for colour matching; without it the node has nothing to match against.

  THERE IS NO resolution WIDGET. Upscale in pixel space FIRST:
      multiplier 1.875 -> 2520 x 1440     (recommended)
      multiplier 2.0   -> 2688 x 1536
  lanczos on the resize node.

  THE KSAMPLER IS NOT A DIAL. 1 step, cfg 1.0, euler, simple, denoise 1.0.
  This is the one-step formulation the model was trained for. Raising steps
  does not improve it.

  SPLIT / MERGE LATENT bracket the KSampler for anything past a few seconds.
  Overlap 3 to blend chunk boundaries. Widget names live on the split node.

  YOU DO NOT NEED A VRAM FLUSH NODE. The DiT loads through UNETLoader, so
  ComfyUI's model manager owns it and evicts the 66.3 GB H3 DiT itself.

  ITERATE THIS AS A SEPARATE GRAPH. Restore is deterministic given input
  frames. Build a second workflow that loads frames from disk straight into
  the chain above, and A/B multiplier, chunk size and colour method without
  re-running H3.

INTERPOLATION (RIFE TensorRT)
  Order: generate -> SeedVR2 -> RIFE -> encode. Interpolation is always last.
  RIFE synthesises intermediate frames from what it is given, so restoring
  after interpolating asks SeedVR2 to reconstruct invented frames and doubles
  its workload for nothing.

  Native H3 output is 24 fps. 2x -> 48, 2.5x -> 60. SET CreateVideo TO MATCH:
  RIFE doubles the frame count but audio duration is fixed, so leaving
  CreateVideo at 24 plays the video at half speed against correct-length
  audio.
  First run on a new instance stalls while TensorRT compiles the engine. Not
  a hang. The cached engine is tied to the GPU architecture it was built on.

=================== TROUBLESHOOTING ===================

IF THE REFERENCES SEEM TO BE IGNORED
  Check these in order, and do not reroll until all four are clear:
    1. Is the checkpoint ref2va? An fl2va file in a ref2v graph is the
       single most likely cause. Read the [layout] lines above.
    2. Is the node MiniMaxH3ReferenceToVideo, not MiniMaxH3ImageToVideo?
    3. Do the <Picture N> tags in the prompt match the actual socket order?
       Count the sockets. Do not trust your memory of the wiring.
    4. Is an acceleration adapter applied, and loaded the right way? If the
       PDD file went through LoraLoader instead of MiniMaxH3PDDAccApply, the
       head bank was silently dropped. Bypass the adapter entirely and re-run
       at 25 steps before concluding anything about the prompt.
    5. Is ref_image_size set to match while you are running an undistilled
       final? References scaled to canvas carry less identity than max.
  Three seeds failing identically means a structural problem, not variance.

IF UPDATES DO NOT SEEM TO LAND
  Read the [git] lines. In order:
    1. "safe.directory configured" near the top. If that says WARNING, every
       git call is failing on dubious ownership and nothing is updating.
    2. "detached HEAD -- looks image-pinned, leaving alone". That is correct
       behaviour: the image pinned core deliberately. Set COMFY_PIN to move
       it, or ALLOW_BRANCH_RECOVERY=1 to track a branch.
    3. "fast-forward blocked". Local commits or a dirty tree.
       GIT_FORCE_RESET=1 discards them.
  Also check the [comfy] git line for a large "+N commits" past the nearest
  tag -- something has walked core off the release the image pinned.

IF A MODEL WILL NOT DOWNLOAD
  Read the [probe] line, not just the [model] one. They say different things:
    HTTP 404/410 -> the manifest is wrong. A token will not help.
    HTTP 401     -> HF_TOKEN missing, expired, or wrong scope.
    HTTP 403     -> gated repo; accept the license with the token's account.
    HTTP 429     -> rate limited; set HF_TOKEN.
    HTTP 5xx/000 -> transient; it retries next boot.
  And check [manifest] near the top: a wrong-arity entry aborts the whole
  fetch phase by design, because v7 shipped with exactly that bug and it
  presented as a download failure.

IF CUSTOM NODES FAIL TO IMPORT
  Read the LAST line of the traceback, not the first. A pack dying on
    ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
  is not itself broken: it imported diffusers, diffusers probed xformers,
  xformers found a half-installed flash_attn and took the flash path. This
  script removes that on every boot -- see the [flash] lines. If they say
  "healthy" or "not installed" and a pack still fails, the cause is elsewhere.

  ComfyUI-Manager's SECURITY LEVEL does not cause import failures. And "Try
  fix" only re-runs that pack's requirements.txt, which cannot repair a
  broken package that is not listed in it.

IF MODEL LOADING IS PATHOLOGICALLY SLOW
  ComfyUI 0.30.x has a pinned-memory regression. Launch with
  --disable-pinned-memory.

IF NODES SHOW LINKS THAT DO NOT EXIST
  Frontend/backend skew on comfyui-frontend-package. This script force-syncs
  it to the requirements.txt pin every boot, so if it persists the affected
  node's slot serialisation is stale in the saved workflow -- delete and
  re-add that node rather than rewiring it. Ref2v nodes have autogrow
  reference sockets, which is exactly the class of node this bites.

IF CUDA GOES MISSING AFTER AN UPDATE
  Something moved torch despite the constraints file. The [torch] block said
  so loudly. Reinstall the pinned cu128 build before rendering anything.

NOT IN THIS BUILD
  fl2va (t2v / i2v / first-last-frame) -- that is the other script. Note the
  PDD release also ships MiniMax-H3-FL2VA-Acc-8Step for that trunk; if you
  move the I2V build onto PDD, fetch that file, not this one.
  H3-Regenerate-2K and H3-Context-IR remain API-only. SeedVR2 substitutes for
  the first. The second matters more on ref2v than it did on I2V: its job is
  relationship resolution across inputs, and ref2v is where the input
  relationships get complicated.

NOTES

echo "=================== PROVISIONING COMPLETE ==================="
