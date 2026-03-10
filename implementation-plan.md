# CPU Virtual Pano Implementation Plan

## Goal

Replace the current CPU curved-slab CPR panorama with a CPU virtual panoramic reconstruction that:

- selects the tooth-support depth before reduction
- suppresses lower/background contamination at reconstruction time
- preserves teeth and roots as the brightest, sharpest structures
- stays compatible with the current arch/worker pipeline

This plan is based on the current codebase and the failures we have already measured in local CPR logs. It is intentionally implementation-oriented and does not assume GPU rendering or post-hoc image beautification.

## Current Diagnosis

The current pipeline is still fundamentally:

`curve -> vertical/depth sampling -> slab reduction -> try to reject junk later`

That architecture is why the output still shows:

- high lower-band contamination
- full-field speckled "TV static" background
- poor tooth/background separation
- residual anterior tooth distortion

Representative failure signals from recent logs:

- `lowerBandBrightFraction` stays around `0.80+`
- `lowerBandP50` stays strongly positive instead of dark
- `selectedQualityScore` stays strongly negative
- `twoPassEligibilityDiagnostics.lowerBandEligibleFraction` is still too high

These values show that the current worker still admits the wrong depth support into the pano before any real anatomy-aware decision is made.

## Constraints

- CPU only
- no GPU renderer
- no GAN / image-generation cleanup
- no "just denoise more" strategy
- no global CLAHE as the primary fix
- keep current orchestrator scoring and VOI logic unchanged in the first implementation phase
- preserve the existing legacy CPR path until the new path is proven better

## High-Level Strategy

Stop treating the pano as a wide depth slab that is reduced directly.

Instead:

1. Build a curved reformat volume that keeps the depth axis.
2. Score which depth is tooth-supporting for each pano column.
3. Select a smooth depth path across columns.
4. Add a small per-column row tilt so crowns and roots can live at slightly different depths.
5. Flatten the curved volume around that support surface.
6. Render the final pano from only a narrow local neighborhood around the support surface.

The key shift is:

`sample wide slab and reduce`

becomes

`estimate support surface first, then reduce narrowly around it`

## Scope By File

### Primary file

- `Viewers/modes/cpr/src/cprWorker.ts`

This file will carry the first implementation.

### Secondary files

- `Viewers/modes/cpr/src/cprMath.ts`
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`

These should stay unchanged in phase 1 unless a small feature-flag plumbing change is needed for A/B selection.

## What We Keep From The Current Pipeline

- current arch centerline from `useCPROrchestrator.ts`
- current frame generation from `cprMath.ts`
- current worker launch shape and volume sampling machinery
- current modality-LUT and stored-value normalization behavior
- current output image loading path
- current logging pattern

## What We Replace

Inside `generatePanorama()` in `cprWorker.ts`, replace the current:

- slab-mean / slab-MIP style reconstruction path
- one-pass/two-pass sample gating attempts

with a new virtual pano path driven by:

- curved volume stack
- depth scoring
- dynamic-programming path selection
- support-surface flattening
- narrow local reduction

## Core Data Model

Define a curved volume stack:

`R[c, d, r] = V(C(c) + u(d) * N(c) + y(r) * Z)`

Where:

- `c` = pano column along the arch
- `d` = depth index along the arch-normal direction
- `r` = pano row
- `C(c)` = frame position along the arch
- `N(c)` = current slab normal for that column
- `Z` = effective vertical direction already used by the worker
- `u(d)` = depth offset in mm
- `y(r)` = row offset in mm

### Initial phase-1 depth stack parameters

- `depthHalfRangeMm = 6.0`
- `depthStepMm = 0.25`
- `depthSamples = 49`

Reason:

- large enough to span real tooth-root support and nearby contaminating structures
- still feasible in a worker on CPU for current pano sizes

## Row Bands

Use row bands relative to the current pano center and half-height, not fixed row numbers.

For each column:

- `r_mid = panoHeight / 2`
- `halfHeight = panoHeight / 2`
- `yNorm = (r - r_mid) / halfHeight`

Initial band definitions:

- `B_tooth`: `yNorm in [-0.35, 0.55]`
- `B_top`: `yNorm in [-0.35, 0.05]`
- `B_bot`: `yNorm in [0.15, 0.65]`
- `B_low`: `yNorm in [0.65, 1.15]`

These match the structure of the current CPR strip and are consistent with the current local-center conventions already used in the worker.

## Phase 1: Add A New Worker Reconstruction Mode

### Objective

Introduce a new CPU path beside the current legacy CPR path so we can A/B compare.

### Implementation

Inside `cprWorker.ts`:

- keep the current legacy `generatePanorama()` logic available
- add a new branch or helper set for:
  - `buildCurvedVolumeStack()`
  - `computeDepthScores()`
  - `selectDepthPathDP()`
  - `estimateColumnTilt()`
  - `renderVirtualPanoFromSurface()`

If needed later:

- add `reconstructionMode: 'legacyCpr' | 'virtualPanoCpu'` to the worker input
- wire it from `useCPROrchestrator.ts`

### Guardrail

Do not delete the legacy path until the new path is measurably better on the target scan.

## Phase 2: Build The Curved Volume Stack

### Function

Add in `cprWorker.ts`:

`buildCurvedVolumeStack(frames, verticalDir, depthSamples, depthHalfRangeMm, depthStepMm, panoWidth, panoHeight)`

### Output

- `Float32Array curvedVolume`
- layout: `[depth][row][col]` or `[col][depth][row]`

Recommended layout for the worker:

- physical storage: `[depth][pixelIndex]`
- `pixelIndex = row * panoWidth + col`
- `stackIndex = depth * planeSize + pixelIndex`

Why:

- this matches the way the current worker already stores stack-style buffers
- it keeps pass-by-depth operations cache-friendly enough

### Behavior

For each `(c, r, d)`:

- compute world position using current frame position
- move by row offset along `effectiveVerticalDir`
- move by depth offset along `slabDir`
- trilinearly sample the volume
- apply current modality LUT / normalization path exactly as today

### Success criteria

- curved volume builds without changing legacy path
- sampled values look numerically consistent with current worker output

## Phase 3: Compute Depth Scores

### Objective

For each column `c` and depth `d`, score how likely that depth is to represent useful tooth-bearing anatomy.

### Thresholds

Use scan-adaptive thresholds, but not a single global threshold over the full volume.

For each column `c`, compute thresholds from a local column window:

- neighborhood columns: `[c - 8, c + 8]`, clamped
- use only rows in `B_tooth`

Define:

- `T_soft[c] = percentile(values in local tooth-band window, 60)`
- `T_hard[c] = percentile(values in local tooth-band window, 85)`
- `G_cap[c] = percentile(gradient magnitude in local tooth-band window, 90)`

Then smooth these three arrays across columns with a short radius, e.g. radius 5.

### Responses

For each `(c, d)`:

- `Hard(c,d) = mean over r in B_tooth of clamp((R[c,d,r] - T_soft[c]) / max(T_hard[c] - T_soft[c], eps), 0, 1)`
- `Grad(c,d) = mean over r in B_tooth of clamp(grad(c,d,r) / max(G_cap[c], eps), 0, 1)`
- `Low(c,d) = mean over r in B_low of clamp((R[c,d,r] - T_soft[c]) / max(T_hard[c] - T_soft[c], eps), 0, 1)`

Where:

- `grad(c,d,r) = abs(R[c,d+1,r] - R[c,d-1,r]) + 0.5 * abs(R[c,d,r+1] - R[c,d,r-1])`

### Initial score function

- `Score(c,d) = 0.55 * Hard(c,d) + 0.30 * Grad(c,d) - 0.40 * Low(c,d)`

Reasoning:

- `Hard` is the strongest primary signal
- `Grad` supports sharp tooth boundaries
- `Low` directly penalizes inferior contamination

### Candidate pruning

For each column:

- keep top `K = 5` depth candidates by score

This reduces DP cost without losing plausible support depths.

## Phase 4: Select A Smooth Depth Path

### Objective

Choose one depth path `D[c]` across columns that is both high-scoring and smooth.

### Initial dynamic-programming formulation

Use first-order DP in the first implementation for minimal risk:

`Cost(c, i) = -Score(c, d_i) + min over j of [Cost(c-1, j) + P_jump(d_i, d_j)]`

Where:

- `d_i`, `d_j` are candidate depths in mm
- `P_jump = 0.22 * abs(d_i - d_j) + 0.12 * max(0, abs(d_i - d_j) - 0.75)`

Why first-order first:

- simpler and less bug-prone than second-order DP
- still enough to prevent jitter and wild jumps

### Output

- `D_mm[c]` = selected depth in mm

### Post step

After backtracking:

- apply a short 1D smoothing pass radius 2-3
- clamp movement so it does not violate `pathJumpP95Mm` goals

## Phase 5: Estimate Simple Per-Column Tilt

### Objective

Allow crowns and roots to sit at slightly different depths within the same column.

### Method

For each column:

- recompute depth selection using only `B_top` -> `D_top_mm[c]`
- recompute depth selection using only `B_bot` -> `D_bot_mm[c]`

Then define:

- `A_mm_per_row[c] = clamp((D_bot_mm[c] - D_top_mm[c]) / max(r_bot_center - r_top_center, 1), -0.03, 0.03)`

Where:

- `r_top_center` = center row of `B_top`
- `r_bot_center` = center row of `B_bot`

Then define support surface:

- `S_mm[c,r] = D_mm[c] + A_mm_per_row[c] * (r - r_mid)`

This is the minimal support surface we should build first.

## Phase 6: Flatten And Reconstruct

### Objective

Render the final pano from a narrow neighborhood around the support surface, not from the full depth slab.

### Flattened depth coordinate

For each `(c, r, d)`:

- `delta_mm = depthOffsetMm[d] - S_mm[c,r]`

Only samples with small `|delta_mm|` may contribute.

### Initial row-dependent trough width

First implementation should be row-dependent only.

Define:

- if `yNorm < -0.10`: `tau_mm = 0.8`, `sigma_mm = 0.45`
- if `-0.10 <= yNorm <= 0.55`: `tau_mm = 1.2`, `sigma_mm = 0.60`
- if `yNorm > 0.55`: `tau_mm = 0.45`, `sigma_mm = 0.25`

Later, if needed, add posterior widening by column region.

### Eligible set

Eligible if:

- `abs(delta_mm) <= tau_mm`

### Reducer

Do not use plain mean.

Use weighted upper-tail local reduction:

1. collect eligible samples
2. weight each by:
   - `w = exp(-0.5 * (delta_mm / sigma_mm)^2)`
3. sort eligible samples by intensity descending
4. keep the highest-intensity set until cumulative weight reaches `q = 0.35` of total eligible weight
5. return the weighted mean of that retained set

### Fallback order

If no sample is eligible:

1. use the nearest-to-surface depth sample
2. if unavailable, use the central depth sample
3. if unavailable, use `-1000`

This is safer than reverting to wide-slab mean.

## Phase 7: Diagnostics

### Existing logs that must improve

These should move in the following direction:

- `lowerBandBrightFraction`: down materially
- `lowerBandP50`: down materially
- `detailBandHorizontalEdgeMean`: down somewhat if background grain is removed
- `detailBandVerticalEdgeMean`: up or hold
- `selectedQualityScore`: improve toward zero or positive

### New logs to add

#### Support surface quality

- `virtualPano.depthMinMm`
- `virtualPano.depthMaxMm`
- `virtualPano.depthStdMm`
- `virtualPano.pathJumpP95Mm`
- `virtualPano.scoreMarginMean`

#### Surface tilt quality

- `virtualPano.tiltMinMmPerRow`
- `virtualPano.tiltMaxMmPerRow`
- `virtualPano.tiltStdMmPerRow`

#### Flattening / suppression quality

- `virtualPano.eligibleFraction`
- `virtualPano.lowerBandEligibleFraction`
- `virtualPano.offTroughEnergyRatio`
- `virtualPano.emptyFallbackFraction`

#### Anatomy/readability quality

- `virtualPano.rootBandContrast`
- `virtualPano.toothBandEdgeEnergy`
- `virtualPano.columnSharpnessP50`
- `virtualPano.lowerSuppressionRatio`

### Success targets on the current problem scan

Initial phase-1 target values:

- `lowerBandBrightFraction < 0.45`
- `lowerBandP50 < 40`
- `virtualPano.lowerBandEligibleFraction < 0.25`
- `detailBandHorizontalEdgeMean / detailBandVerticalEdgeMean < 2.5`
- visible reduction of full-field background speckle

These are success targets, not hard correctness guarantees.

## Rollout Order

### Step 1

Add the new virtual pano path beside the legacy path in `cprWorker.ts`.

### Step 2

Implement curved volume stack + depth scoring + DP path only.

At this point:

- log the selected path
- do not replace final output yet

### Step 3

Add tilt estimation and surface flattening.

### Step 4

Add narrow local upper-tail reconstruction and compare against legacy.

### Step 5

Only after clear improvement:

- consider wiring the mode through orchestrator for default use

## Rollback Strategy

- keep legacy CPR path alive until the new path clearly wins
- isolate the new path behind a reconstruction mode switch
- add enough logs to compare both reconstructions on the same scan
- if the new path regresses badly:
  - switch back to legacy immediately
  - keep curved volume + scoring helpers for debugging

## What Not To Do In The First Implementation

- do not add CLAHE
- do not add GAN / learned cleanup
- do not replace the current arch extraction path
- do not jump directly to a full 2D support surface
- do not keep tuning the current slab-mean path once this plan starts

## Expected Outcome

If this plan works as intended, the first visible improvement should be:

- the lower/background region stops looking like full-field static
- the teeth separate more clearly from the background
- roots remain bright without the strip being globally contaminated

After that, if anterior distortion still remains, the next upgrade should be:

- coarse 2D support surface instead of `D[c] + tilt`
- then curvature-adaptive spline sampling if still needed

This is the plan we should implement next.
