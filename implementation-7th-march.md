# Implementation Plan - 7th March

## Purpose

This document defines the new reconstruction path for generating a substantially better CBCT-derived panoramic image than the current CPR pano strip.

This plan replaces the "keep tuning the legacy curved slab CPR" strategy as the main quality path.

The current codebase audit, repeated experiments, and multiple external research reviews all point to the same conclusion:

- the legacy CPR pano path has reached a practical quality ceiling
- the display pipeline is not the hidden problem
- the reconstruction model itself must change

This document is the working plan for a new branch dedicated only to the new architecture.

## Branching Rule

Do this work on a separate branch only.

Recommended branch name:

`feature/cpr-virtual-pano-dp`

Do not continue mixing this work into the stable legacy CPR path until the new architecture has proven itself.

## Hard Findings

### Confirmed from codebase audit

- The pano viewport is showing the actual selected worker output.
- There is no hidden file elsewhere in the codebase post-processing or overriding the pano after worker output.
- The main problem remains inside the CPR reconstruction and selection pipeline.
- The current stable output path is still basically the legacy CPR strip path.
- The current legacy path is not failing because of one bad threshold. It is failing because it reduces the wrong anatomy before making a correct support-surface decision.

### Confirmed from repeated experiments

- Slab thickness tuning, MEAN vs MIP tuning, VOI changes, denoise changes, lower-band suppression, one-pass gating, two-pass eligibility, and score tuning did not produce acceptable images.
- The best stable legacy results are still too noisy, too contaminated, and too anatomically unreliable.
- The first experimental virtual-pano output attempt failed because the calibration was wrong, not because the overall architecture direction was wrong.

### Confirmed from the research reviews

Two independent research-style reviews converged on the same architecture class:

- build an unrolled ribbon volume around the arch
- estimate a patient-specific focal trough / support surface with dynamic programming or shortest-path optimization
- reconstruct narrowly around that surface

This matches the strongest algorithmic direction for the current constraints.

## Current Codebase Context

Primary files:

- `Viewers/modes/cpr/src/cprWorker.ts`
- `Viewers/modes/cpr/src/cprMath.ts`
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`
- `Viewers/modes/cpr/src/panoImageLoader.ts`

Display path files already audited:

- `Viewers/extensions/cornerstone/src/services/ViewportService/CornerstoneViewportService.ts`
- `Viewers/extensions/cornerstone/src/Viewport/OHIFCornerstoneViewport.tsx`
- `Viewers/extensions/default/src/getHangingProtocolModule.js`

Important operational fact:

- The display/loader path is not the source of the middle-pano corruption.
- The loader and viewport should remain mostly unchanged for this new architecture.

## Problem Statement

We need a CPU-only panoramic reconstruction that is materially more readable and more anatomically faithful than the current CPR strip.

The target is not an exact duplicate of a real panoramic radiograph. That is unrealistic from CBCT alone. The target is:

- cleaner background
- much lower lower-band contamination
- better tooth/background separation
- much less anterior distortion
- diagnostically usable overall image

## Constraints

- CPU only
- no GPU renderer
- no fake generative enhancement
- no "just denoise/CLAHE/VOI" strategy
- preserve real anatomy
- implementation must fit the existing TS worker-based viewer
- runtime must remain practical
- legacy CPR path must stay available as fallback until the new path is proven

## Architectural Decision

### Approved new direction

Implement a patient-specific focal-trough virtual pano based on:

1. an unrolled ribbon volume around the arch
2. a tooth-likelihood / support cost map
3. dynamic-programming depth path selection
4. narrow local reconstruction around the selected support path

### Rejected as primary path

Do not continue using these as the main quality strategy:

- more curved-slab tuning
- more MEAN vs MIP tuning
- more conditional masking inside the current slab-collapse path
- more VOI-only fixes
- more denoise-only fixes

## Why the Legacy Path Saturated

The current CPR pano path is still a curved slab reduction model.

That means:

- it samples a slab around the arch
- it mixes multiple incompatible anatomical depths into the same output pixel
- it then reduces them with a mean-like or MIP-like operator

Once wrong anatomy has already been mixed into a pixel, no later score, VOI, denoise, or gating heuristic can fully recover the correct structure.

This is the architectural ceiling we hit.

## New Reconstruction Model

### Core representation

Construct an unrolled ribbon volume:

- `s`: position along the arch
- `d`: bucco-lingual depth offset from the arch
- `z`: vertical row / height

Represent the volume as:

`V[s, d, z]`

This ribbon keeps the depth dimension available until after support selection.

### Support-surface selection

Build a tooth-likelihood or support score map:

`L[s, d]`

This score should come from structured anatomy indicators, not raw brightest-value selection.

Recommended components:

- robust intensity term favoring tooth/bone-support ranges
- gradient term favoring structured edges
- optional local variance / local structure term
- depth-edge penalty so the path does not hug the boundary of the sampled trough

Then compute a smooth optimal depth path:

`d_star[s]`

using dynamic programming / shortest path with smoothness penalties.

### Final reconstruction

For each output pano pixel `(s, z)`:

- sample only a narrow neighborhood around `d_star[s]`
- aggregate with a normalized local kernel
- do not use a bright-emphasizing reducer in the first stable version

This is the key architectural change:

- select depth first
- reduce locally second

## What We Are Not Doing First

Do not start with:

- full multi-surface blending
- ray-traced panoramic scan simulation
- tomosynthesis-style virtual scanner geometry
- ML or generative postprocessing
- fully automatic arch extraction replacement

Those may be future options, but they are not phase-1.

## Implementation Scope

### Keep unchanged initially

- `panoImageLoader.ts`
- pano viewport / layout code
- hanging protocol
- general display plumbing

### Primary implementation file

- `Viewers/modes/cpr/src/cprWorker.ts`

### Secondary support files

- `Viewers/modes/cpr/src/cprMath.ts`
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`

The new architecture should be added as a separate path, not by overwriting the stable legacy path.

## Phased Execution Plan

## Phase 0 - Stable Baseline and Isolation

Goal:

- keep the current stable legacy path available
- isolate the new work to a separate branch and a separate reconstruction mode

Requirements:

- legacy path remains selectable and untouched
- all new work is behind a flag or new mode label

Success:

- safe iteration without breaking the currently usable CPR path

## Phase 1 - Ribbon Volume and Support Diagnostics Only

Goal:

- build the unrolled ribbon volume and support-cost infrastructure without changing the displayed image yet

Implement:

- ribbon volume sampling `V[s, d, z]`
- score map `L[s, d]`
- dynamic-programming path `d_star[s]`
- diagnostics/logs for support path quality

Do not yet:

- replace the displayed pano

Diagnostics to log:

- depth range used
- score map thresholds
- DP path min/max/std
- DP path jump percentiles
- edge-boundary occupancy
- score-margin statistics

Success criteria:

- path is smooth
- path is not stuck to boundaries
- support metrics are finite and sane

## Phase 2 - Narrow Local Reconstruction Behind Flag

Goal:

- render an actual virtual pano from the selected support path

Implement:

- narrow local Gaussian-weighted mean around `d_star[s]`
- normalized kernel so brightness remains stable
- no bright-tail emphasis in the first stable render
- strict fallback gate back to legacy output

Important:

- first render version should be conservative
- if any sanity check fails, do not replace legacy output

Diagnostics to log:

- whether virtual render was used
- support depth clamp fraction
- tooth-band mean
- lower-band mean
- lower-band bright fraction
- off-trough energy ratio
- fallback reason if virtual output is rejected

Success criteria:

- lower band becomes materially darker/cleaner
- tooth/background separation improves
- no white blown-out failure

## Phase 3 - Scoring and Selection Integration

Goal:

- make the orchestrator correctly compare legacy vs virtual outputs

Implement:

- summary metrics specific to the new virtual pano
- selection logic that prefers diagnostically better outputs
- avoid rewarding pathological dark lower bands alone

Success criteria:

- virtual output only wins when actually cleaner
- legacy output remains fallback when virtual path is not yet trustworthy

## Phase 4 - Runtime Optimization

Goal:

- reduce the new architecture to practical runtime

Likely strategies:

- keep the persistent worker model
- cache arch/ribbon sampling structures where possible
- compute cost map on slightly reduced vertical support if needed
- reuse support buffers across attempts
- avoid repeated worker reinitialization

Target:

- single-digit seconds on typical scans

## Phase 5 - Optional Refinements

Only after the single-surface model is stable:

- multi-surface support model
- row-dependent support tilt or blended lower-root surface
- better edentulous handling
- optional adaptive support priors by region

## Data Structures

Suggested worker-side buffers:

- `Float32Array ribbonVolume`
- `Float32Array supportScoreMap`
- `Float32Array dpCost`
- `Int16Array dpBackpointer`
- `Float32Array selectedDepthMm`
- `Float32Array virtualPanoPixelData`

Optional later:

- `Float32Array selectedRootDepthMm`
- `Float32Array supportConfidence`

## Initial Algorithm Defaults

These are starting points, not final truths.

Ribbon:

- depth half-range: roughly `5-6 mm`
- depth step: roughly `0.2-0.3 mm`
- depth samples: roughly `40-60`

DP:

- local predecessor window: small, around `2-4` bins
- first-order smoothness penalty
- optional second-order curvature penalty
- explicit penalty near depth-range boundaries

Render:

- narrow support kernel only
- normalized Gaussian-weighted mean
- no MIP
- no upper-tail emphasis in first production attempt

## Metrics That Must Improve

The earliest metrics that should improve if the new model is correct:

- lower-band bright fraction should drop substantially
- lower-band median should get darker / less positive
- tooth-band spread should become more coherent
- support path should stop hugging depth boundaries
- off-trough energy ratio should drop

The earliest image characteristics that should improve:

- less background static
- less double-contour contamination
- more consistent anterior tooth shape
- better root continuity

## Failure Modes To Watch

- support path hugs depth edges
- virtual output becomes blown-out or posterized
- metal/high-density structures dominate support scoring
- edentulous regions become unstable
- runtime becomes unacceptable

## Validation Protocol

For every phase:

1. compare legacy and experimental output on the same scan
2. collect worker diagnostics
3. verify the experimental path is not silently replacing the stable output unless intended
4. review screenshot plus metrics together

Do not trust metrics alone.

## Definition of Success

Success does not mean matching a true panoramic radiograph perfectly.

Success means:

- significantly cleaner than current CPR strip
- materially lower contamination
- anatomy is readable and defensible
- teeth and roots are visibly interpretable
- stable enough to compare favorably against the current best legacy result

## Final Decision

Proceed with this new architecture.

Do not spend more major effort on the current slab-collapse CPR path as the main quality solution.

The next development branch should focus on the focal-trough / support-surface virtual pano path only.


-----------------------------------------



HOW TO PROCEED WITH THE CHANGES 
ROADMAP:

How we should proceed

We should do it in controlled checkpoints, not all at once.

Branch baseline

Keep the current legacy path intact inside this branch as the fallback/reference.
New architecture must live behind a separate reconstruction mode/flag.
No “replace stable output immediately” mistake again.
Phase 1: Support-surface diagnostics only

Implement:
ribbon volume V[s,d,z]
support score map L[s,d]
DP path d*[s]
Do not render it to the viewer yet.
Only log diagnostics and verify the selected path is sane.
Validation checkpoint 1

You regenerate CPR once.
I inspect the new diagnostics:
path range
path smoothness
depth-edge occupancy
score margin
If the path looks unstable, we fix that before ever using it for rendering.
Phase 2: Narrow local virtual render behind fallback

Render around d*[s] using a normalized narrow Gaussian-weighted mean.
No bright-tail/white-emphasis tricks.
Keep strict fallback to legacy if sanity checks fail.
Validation checkpoint 2

Compare legacy vs new virtual output on the same scan.
Only if the virtual path is clearly better do we let it win.
Phase 3: Selection integration

Only after the virtual renderer is visually sane.
Then integrate summary/scoring so selection can choose between legacy and virtual properly.
Phase 4: Runtime

Once quality is real.
Then optimize ribbon sampling, cache reuse, and support-map computation.
What I would do first now

Start with Phase 1 only in this branch.

That is the safest and smartest next move because:

it gives us a real signal whether the architecture is working
it does not risk corrupting the viewer output again
it lets us debug the support surface before we debate image quality
So the concrete answer
