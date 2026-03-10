# CPR Panoramic Reconstruction — Architecture Analysis & Recommendation

## 1. Executive Verdict

### Salvage vs. Pivot

**The legacy CPR slab path cannot be salvaged.** It must be replaced with a virtual panoramic architecture.

The fundamental issue is structural, not parametric: the current pipeline samples a wide depth slab first, reduces it to a single value, and *then* tries to reject contamination in 2D. This order of operations is backwards. No amount of Winsorized-mean tuning, background suppression, or two-pass eligibility gating can fix it because by the time those run, the depth information is already lost.

**The good news:** the codebase already has ~75% of the virtual-pano infrastructure built as diagnostics-only code (lines 1851–2220 _within_ `generatePanorama`). The Phase 1/2 diagnostics compute a curved volume stack, depth scoring, candidate pruning, and DP path selection — they just never use the result to produce output pixels.

---

## 2. Root Cause Analysis — Why the Current Output Fails

### 2.1 The core architectural flaw

The current pipeline in `generatePanorama()` (line 785–2372, `cprWorker.ts`) follows this flow:

```
for each (col, row):
    for each slab_depth:
        trilinearly sample volume → slabValueBuffer[s]
        weight by Gaussian focal trough → slabWeightBuffer[s]
    sort samples ascending
    reduce to single value via Winsorized-weighted-mean OR weighted-high-band-mean
    → pass1ProvisionalPano[pixelIdx]
```

**The problem:** Every pixel reduces `slabSampleCount` (typically 7–13) depth samples into one value *before* any understanding of whether those samples belong to tooth anatomy, soft tissue, or air. The reducer (lines 364–428 `computeWinsorizedWeightedMean`) tries to Winsorize bright outliers, but:

- A tooth (bright, ~1500–3000 HU) surrounded by air (−1000 HU) and soft tissue (~50 HU) will have the mean dragged down
- A MIP approach (lines 430–456 `computeWeightedHighBandMean`) picks the highest values, which catches metal artifacts, ridge edges, and cross-talk from adjacent teeth

The two-pass eligibility gating (lines 1644–1815) adds a BFS flood-fill through a 3D `stackSize = planeSize × slabSampleCount` mask. This is conceptually the right idea — it tries to keep only "connected foreground" slab samples — but:
- It operates on a mask that already contains mixed anatomy/background samples because the seed thresholds (lines 1651-1652: `floor + 0.35 * bandSpan`) are not depth-aware
- The BFS flood propagates in (col, row) within a single depth plane — it doesn't track which *depth* represents the tooth surface
- Even with the mask, the final reducer (line 1791) still averages eligible samples across *all* depths, not around the support surface

### 2.2 Why `lowerBandBrightFraction` stays at 0.75–0.86

The lower band (below the tooth roots, `yNorm > 0.5`) receives the same wide-slab reduction as the tooth band. At those rows, the slab normal sweeps through bone, soft tissue, and air. The `computeWinsorizedWeightedMean` averages these, landing at ~100–300 HU: bright enough to look "contaminated" rather than dark like the reference pano.

The `suppressLowerBackground` function (lines 533–754) tries to fix this retroactively by darkening pixels *after* reduction — but it's too weak (the `structureProtect` clause on line 707 preserves too many edge-adjacent pixels).

### 2.3 Why anterior teeth look sheared/alien

The slab normal `N_slab` is computed in [`cprMath.ts`](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L150-L162) by `toSlabNormal(nCamera, s)`:

```typescript
function toSlabNormal(nCamera: Point3, s: Point3): Point3 {
  const zZeroedN: Point3 = [nCamera[0], nCamera[1], 0];
  // ...
}
```

This projects the camera normal to the XY plane (zeroing Z). For posterior teeth where the arch is nearly in the XY plane, this works. For anterior teeth where the arch curves sharply in XY *and* the tangent direction changes rapidly, the Z-zeroed slab normal rotates quickly column-to-column, sweeping the depth slab through a different angle than neighboring columns. Combined with the coarse `slabSampleCount ≤ 13`, this creates:

- Spatial aliasing at sharp curvature points
- Depth samples from adjacent teeth cross-talking into the same pixel
- Visible shearing when the slab normal changes faster than one voxel spacing per column step

### 2.4 Runtime bottleneck: 40–50 seconds

The worker performs **three full passes** through the entire output grid:

| Pass | Work | Lines |
|------|------|-------|
| Adaptive center search | `panoWidth × adaptiveCandidateCount(5-7) × adaptiveProfileSampleCount(13)` calls to `sampleReducedPoint`, each doing `slabSampleCount` trilinear samples, sorting, reducing | 1324–1461 |
| Pass 1 | Full `panoWidth × panoHeight × slabSampleCount` trilinear samples into `pass1IntensityStack` | 1619–1644 |
| Virtual Pano Phase 1/2 (diagnostics-only) | Full `panoWidth × panoHeight × virtualPanoDepthSamples(49!)` trilinear samples + score computation + DP | 1895–2160 |
| Pass 2 | Full `panoWidth × panoHeight` re-reduction from `pass1IntensityStack` with eligibility mask | 1740–1815 |
| Background suppression | Full `panoWidth × panoHeight` multi-pass | 2222–2242 |

The **virtualPanoPhase12** diagnostics (pass 3) **dominates runtime**. It samples `panoWidth × panoHeight × 49 = ~19.6M` voxels with trilinear interpolation, then runs per-column score computation and DP. This was supposed to be diagnostic-only but was never gated behind a flag or removed after the revert. It is the primary reason runtime jumped to 40–50 seconds.

---

## 3. Why Alternative Approaches Won't Work

### 3.1 More slab-mean tuning

The Winsorized mean (lines 364–428) already has bright-tail capping, coherent bright cluster detection, and conditional preservation. The parameters (`clusterTolerance = max(110, robustSpan * 0.16)`, etc.) have been tuned. Further tuning cannot fix the fact that the slab mixes tooth, air, and soft tissue before reduction. **The signal-to-noise ratio of the reducer input is fundamentally too low.**

### 3.2 MIP-like approaches

The `computeWeightedHighBandMean` (lines 430–456) already implements a soft-MIP variant — it keeps the top N values weighted. But MIP:
- Amplifies metal artifacts and ridge edges
- Cannot distinguish a tooth surface from a bone surface at a different depth
- Produces the "blown out white" appearance that was the previous failure mode with the virtual pano attempt

### 3.3 One-pass or two-pass sample gating only

The existing two-pass eligibility mask (lines 1644–1815) is the closest attempt to a correct solution, but it gating operates in `(col, row)` per depth plane. It does not reason about *which depth* is the right one — it just asks "is this slab sample connected to a seed?" That's necessary but insufficient because:
- Seeds are placed using a fixed intensity threshold per-column, not by depth-coherent surface estimation
- The BFS flood can propagate through soft tissue bridges between teeth, admitting wrong-depth samples
- The final reducer still averages across all eligible depths instead of centering the trough on the support surface

---

## 4. Recommended Architecture: Depth-First Virtual Pano

### 4.1 The key insight

A clinical panoramic radiograph works by sweeping a narrow focal trough through the jaw — the image shows anatomy only where the trough intersects tissue. The current CPR pipeline uses a *wide, fixed-depth slab* instead. The fix is to make the focal trough **narrow and anatomy-adaptive**: estimate where the tooth surface is, then reduce only around that depth.

### 4.2 Architecture summary

```
┌─────────────────────────┐
│ Spline + RMF frames     │  ← KEEP (useCPROrchestrator + cprMath)
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│ Curved Volume Stack     │  ← BUILD (exists as diagnostics, promote to primary)
│ R[col, depth, row]      │     ~6mm half-range, 0.25mm steps, 49 depths
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│ Per-Column Depth Scoring│  ← ALREADY BUILT (lines 1920–2069)
│ Score(c,d) = 0.55·Hard  │     Threshold from tooth-band percentiles
│   + 0.30·Grad           │     Gradient response in depth direction
│   - 0.40·Low            │     Lower-band contamination penalty
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│ DP Path Selection       │  ← ALREADY BUILT (lines 2084–2160)
│ D[c] = smooth depth path│     First-order DP with jump penalty
│                         │     Backtrack + smoothing
└─────────┬───────────────┘
          ▼
┌────────────────────────────┐
│ Support Surface + Tilt     │  ← NEW: minimal addition
│ S[c,r] = D[c] + tilt(c)·r │     Per-column top/bottom depth offset
└─────────┬──────────────────┘
          ▼
┌──────────────────────────────┐
│ Narrow Focal Trough Reducer  │  ← NEW: the critical change
│ For each (c,r):              │
│   eligible = samples where   │
│     |depth - S[c,r]| < τ(r)  │
│   weight by Gaussian ∝       │
│     exp(-(Δ/σ)²)             │
│   upper-tail weighted mean   │
│   of eligible set            │
└──────────────────────────────┘
```

### 4.3 Why this architecture is better

| Property | Current pipeline | Virtual pano |
|----------|-----------------|--------------|
| Depth selection | After reduction (lost) | Before reduction (preserved) |
| Background in tooth-band | Mixed in during slab mean | Excluded by narrow trough |
| Lower-band contamination | Retroactive suppression | Never sampled beyond surface |
| Anterior distortion | Slab sweeps through wrong depths | Surface-adaptive per column |
| Runtime | 3 full passes + diagnostics pass | 1 volume stack + 1 render pass |

---

## 5. True Bottlenecks in Current Code

| Bottleneck | Location | Impact | Fix |
|------------|----------|--------|-----|
| **virtualPanoPhase12 diagnostics** | Lines 1851–2220 | **Dominant runtime (~60%)** — samples 49 depths × full grid but never writes output | Remove or gate behind flag; promote to primary path only when outputs use the result |
| **Double trilinear sampling** | Pass 1 (lines 1619–1644) + virtual pano pass (1895–1918) | Volume is sampled twice at similar positions | Unify into single curved volume stack pass |
| **Adaptive center search** | Lines 1324–1461 | `panoWidth × 5-7 candidates × 13 profile samples × slabSampleCount` trilinear calls | Move to virtual-pano depth scoring (already scores depths) |
| **InsertionSort in reducer** | `sortSamplePairsAscending` (lines 345–362) called per-pixel | O(n²) sort called ~400K times | For n≤13 this is acceptable; for n=49, use a partial sort or running statistics |
| **BFS flood-fill** | Lines 1680–1735 | Allocates `planeSize × slabSampleCount` Uint8Array masks (~10MB per pass) | Eliminated by surface-centric architecture |
| **Background suppression** | Lines 533–754 | Per-column percentile sorts + gradient computation | Partially eliminated if lower band is already dark from surface reduction |

---

## 6. Phased Implementation Plan

### Phase 0: Clean Up (Diagnostic-Only, 0 regression risk)

**Goal:** Remove the runtime bottleneck without changing any output.

- **File:** `cprWorker.ts`
- **Change:** Gate the entire `virtualPanoPhase12` block (lines 1851–2220) behind a `const ENABLE_VIRTUAL_PANO_DIAGNOSTICS = false` flag. Log that it was skipped.
- **Expected metric movement:** Runtime drops from ~45s to ~15–20s. Zero pixel-data change.
- **Regression risk:** None — output identical.
- **Runtime impact:** −25–30s.

### Phase 1: Promote Virtual Pano to Output Path (Behind Flag)

**Goal:** Use the existing curved volume stack + depth scoring + DP path to produce actual output pixels, behind a `reconstructionMode` flag.

- **Files:** `cprWorker.ts`, minor wiring in `useCPROrchestrator.ts`
- **Functions to add/modify:**
  - `renderVirtualPanoFromSurface()` — new function in `cprWorker.ts` (~150 lines)
  - Narrow focal-trough Gaussian reducer (replaces Winsorized mean for the virtual path)
  - `reconstructionMode: 'legacy' | 'virtualPano'` input field, default 'legacy'
- **Expected metric movement:**
  - `lowerBandBrightFraction` drops from 0.75–0.86 → target <0.50
  - `lowerBandP50` drops from positive → target <50 HU
  - `detailBandHorizontalEdgeMean / detailBandVerticalEdgeMean` ratio drops
  - `selectedQualityScore` moves toward zero
- **Regression risk:** Low — legacy path untouched; new path behind flag.
- **Runtime impact:** Single curved-volume pass (similar to current virtualPanoPhase12) + single render pass. Expected ~12–18s total. Faster than current because no redundant passes.

### Phase 2: Per-Column Tilt + Row-Adaptive Trough Width

**Goal:** Allow crowns and roots to sit at slightly different depths; tighten the trough above and below the tooth band.

- **File:** `cprWorker.ts`
- **Functions to add:**
  - `estimateColumnTilt()` — score top-band vs. bottom-band depth preference (~80 lines)
  - Row-dependent `τ` and `σ` parameters for the focal trough
- **Expected metric movement:**
  - Root visibility improves (separate depth from crown in posterior regions)
  - Further lower-band contamination reduction
  - `virtualPano.tiltStdMmPerRow` appears in logs
- **Regression risk:** Medium — incorrect tilt could shift root positions. Guard with max tilt clamp (±0.03 mm/row).
- **Runtime impact:** Negligible (per-column, O(panoWidth)).

### Phase 3: Curvature-Adaptive Slab Normal for Anteriors

**Goal:** Fix the anterior tooth distortion.

- **File:** `cprMath.ts` (modify `toSlabNormal`)
- **Change:** At high curvature points, widen the slab normal blend window or use a locally averaged tangent over a ±3 frame neighborhood before computing the normal. This smooths the depth sampling direction at the expense of minor depth resolution.
- **Expected metric movement:**
  - Anterior teeth stop looking sheared
  - `turnAngleMaxDeg` in diagnostics correlates with regions that change
- **Regression risk:** Medium — could affect posterior teeth if blend window is too aggressive. Guard with curvature factor threshold.
- **Runtime impact:** Negligible.

### Phase 4: Single-Pass Architecture (Performance Phase)

**Goal:** Eliminate redundant passes.

- **File:** `cprWorker.ts`
- **Changes:**
  - Remove the legacy adaptive-center-search pass (lines 1324–1461)
  - Remove the pass-1 provisional pano + BFS eligibility (lines 1619–1815)
  - The DP path selection already identifies the correct vertical center
  - Unify all sampling into: curved volume stack → depth score → DP → render
- **Expected metric movement:** Runtime under 10s for typical scans.
- **Regression risk:** Medium-high — full pipeline restructure. Must have Phase 1 proven first.

---

## 7. What the Smallest Worth-Doing Phase Is

**Phase 0** (gate virtualPanoPhase12) is the absolute smallest useful change — it halves runtime with zero output risk.

**Phase 1** is the smallest *image-quality* improvement worth doing — it actually uses the support surface to render.

I recommend doing Phase 0 immediately, then Phase 1 as the primary effort.

---

## 8. Achievability of Reference-Quality Output

### What is realistically achievable

The reference image is a dedicated panoramic radiograph — it was acquired by a machine designed specifically for that modality, with a rotating source and detector, and a focal trough engineered for the dental arch. A CBCT volume is a different modality entirely: isotropic resolution, cone-beam geometry, lower signal-to-noise ratio per voxel.

**Realistically achievable from this pipeline:**
- ✅ Clean, dark background (not TV-static)
- ✅ Clear tooth/background separation
- ✅ Individual teeth identifiable without shearing
- ✅ Roots visible as continuous structures
- ✅ Correct left-right ordering and approximate anatomical proportions
- ✅ Clinically readable (a dentist can orient themselves and identify pathology)

**Not achievable from this pipeline:**
- ❌ Same spatial resolution as a dedicated OPG (the CBCT voxel spacing sets the floor)
- ❌ Same contrast as a detector-based radiograph (no scatter rejection, no anti-scatter grid)
- ❌ Identical soft-tissue rendering (CBCT pano is a virtual reformat, not a transmission image)
- ❌ Dynamic range matching (CBCT values are CT numbers, not detector log-exposure)
- ❌ Artifact-free metal implant visualization (CBCT inherent limitation)

### Bottom line

The output should look significantly closer to the reference than the current speckled/sheared mess. Expect maybe **60–70% of the way to the reference** in terms of clinical readability. The biggest remaining gap will be spatial resolution and contrast detail — those are physical limitations of the CBCT modality, not software limitations.

---

## 9. Summary of Recommended Changes by File

| File | Change | Phase |
|------|--------|-------|
| [cprWorker.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts) | Gate `virtualPanoPhase12` | 0 |
| [cprWorker.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts) | Add `renderVirtualPanoFromSurface()` using existing stack + DP path | 1 |
| [cprWorker.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts) | Add `estimateColumnTilt()` | 2 |
| [cprMath.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts) | Curvature-adaptive `toSlabNormal` | 3 |
| [useCPROrchestrator.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/useCPROrchestrator.ts) | Wire `reconstructionMode` flag in `launchCPRWorker` | 1 |
| [cprWorker.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts) | Unify to single-pass architecture | 4 |
