# CPR Remediation Plan - March 8

No code changes are included in this step. This file records the implementation plan for review before any CPR fixes are applied.

## Phase 1: Fix Intensity-Domain Truth First

Files:
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`
- `Viewers/modes/cpr/src/crossSectionImageLoader.ts`
- `Viewers/modes/cpr/src/cprSyntheticDisplay.ts`

Changes:
- Replace the current loose `huDomain` logic with an explicit domain decision: `hu`, `native`, or `unknown`.
- Remove the two false triggers that are poisoning the output:
  - `identity slope/intercept => HU`
  - `range inside [-5000, 7000] => HU`
- Make VOI selection depend on that domain.
  - `hu`: allow CT-style dental window constraints.
  - `native` or `unknown`: use percentile/range-based VOI from actual CPR pixels, without HU clamping.
- Keep synthetic pano/cross images as identity-rescale display images, but stop pretending native CBCT values are HU.

Why this is first:
- Right now the app can take raw CBCT values and display them as if they were CT HU. That alone can create the white fog.

Success criteria:
- The failing scan no longer gets classified as HU unless the source really supports it.
- `windowWidth/windowCenter` stop collapsing into the washed-out white presentation.

## Phase 2: Make CPR Sample the Real Source Volume

Files:
- `Viewers/extensions/cornerstone/src/services/CornerstoneCacheService/CornerstoneCacheService.ts`
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`

Changes:
- Disable the `>400 slices => every second slice` decimation for CPR source volumes.
- Before running CPR generation, require the source volume to be fully loaded.
- If the volume is still hydrating, block generation until it is complete instead of sampling partial scalar data.

Why this is second:
- Even a correct CPR renderer will fail if the source volume is half-loaded or decimated before sampling.

Success criteria:
- CPR always samples a full-resolution, fully loaded source volume.
- No CPR run starts from a partially hydrated cache volume.

## Phase 3: Re-anchor the Sampling Geometry

Files:
- `Viewers/modes/cpr/src/useCPROrchestrator.ts`
- `Viewers/modes/cpr/src/cprWorker.ts`
- `Viewers/modes/cpr/src/cprMath.ts`
- `Viewers/modes/cpr/src/crossSectionImageLoader.ts`

Changes:
- Reduce the initial vertical sampling band so pano does not start by swallowing a huge column of noisy in-volume tissue.
- Increase adaptive center-search and allowable center correction so the worker can actually move toward the tooth/root band.
- Remove the world-XY slab-normal fallback in `toSlabNormal` and derive slab direction from actual frame/volume orientation.
- Tighten cross-section extents so they match the corrected pano geometry.

Why this is third:
- The current output reads like the renderer is sampling too much real anatomy outside the intended dental band, not just displaying the right anatomy badly.

Success criteria:
- Lower-band brightness drops.
- Cross-sections look centered on the tooth band rather than flooded with surrounding tissue.

## Phase 4: Validate With Existing Logs

Files:
- Same files as above, plus the existing runtime logs

Changes:
- Rebuild and test the same failing study.
- Compare before/after using:
  - `[CPR-PHASE2-JSON]`
  - `[CPR-DIAGNOSTIC-JSON]`
  - `[CPR-LOADER-METADATA-JSON]`
  - `[CPR-CROSSSECTION-INTENSITY-JSON]`
- Specifically verify:
  - source volume was not decimated
  - source volume was fully loaded
  - domain was classified correctly
  - lower-band brightness decreased
  - pano and cross-section are both readable

## What This Plan Does Not Do

- It does not replace OHIF or add a new library first.
- It does not change reducers again before fixing domain and source-volume integrity.
- It does not treat this as a pure sharpness problem.

## Implementation Order After Approval

1. Phase 1 and Phase 2 together.
2. Rebuild and test once.
3. If contamination remains, apply Phase 3.
4. Rebuild and test again.
5. Report exact before/after findings.
