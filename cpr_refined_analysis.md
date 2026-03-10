# CPR Pipeline — Refined Analysis & Exact Specs

This document answers the five specific questions against the actual current code.

---

## 1. Phase 0: Exact Patch to Disable virtualPanoPhase12 Runtime

### What to skip

The virtualPanoPhase12 work spans [lines 1851–2220](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1851-L2220) inside `generatePanorama()`. It has four logical segments:

| Segment | Lines | What it does | CPU cost |
|---------|-------|-------------|----------|
| **Allocations** | 1851–1877 | Creates `virtualPanoStack` (`Float32Array(planeSize × 49)`), `virtualScoreByColDepth`, threshold arrays, candidate arrays, DP arrays | Memory only (~15 MB for 800×400), no compute |
| **Volume sampling** | 1895–1918 | Triple-nested loop: `panoWidth × panoHeight × 49` trilinear samples via `sampleWorldIntensityForVirtualPano` | **Dominant cost** — ~15.7M trilinear interpolations |
| **Threshold + scoring + DP** | 1920–2160 | Per-column thresholds, gradient, scoring, candidate pruning, DP path + backtrack | Moderate — all in-memory, no volume access |
| **Diagnostics assembly** | 2162–2220 | Computes stats from DP path, builds `virtualPanoPhase12Diagnostics` object | Negligible |

The helper function `sampleWorldIntensityForVirtualPano` (lines 1823–1849) only exists for this phase. It has zero runtime cost as a function definition, so leave it in place.

### Exact minimal patch

Add a single constant at the top of `generatePanorama()` and wrap the runtime work:

```typescript
// --- Add near line 838 (after destructuring input) ---
const ENABLE_VIRTUAL_PANO_PHASE12 = false;
```

Then wrap the segments:

```typescript
// --- At line 1851, before virtualPanoDepthHalfRangeMm ---
let virtualPanoPhase12Diagnostics: Record<string, unknown>;

if (ENABLE_VIRTUAL_PANO_PHASE12) {
  // === existing lines 1851–2220 stay exactly as-is ===
  // (all allocations, sampling, thresholds, scoring, DP, diagnostics assembly)
} else {
  virtualPanoPhase12Diagnostics = { enabled: false, skippedReason: 'PHASE0_GATE' };
}
```

The `virtualPanoPhase12Diagnostics` variable is referenced exactly once, on [line 2340](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2340):

```typescript
virtualPanoPhase12: virtualPanoPhase12Diagnostics,
```

The stub assignment ensures the diagnostic payload remains valid JSON.

### Row-band definitions and `rowFromNormalizedOffset`

`toothBandStartRow` through `lowerBandEndRow` ([lines 1884–1891](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1884-L1891)) and `rowFromNormalizedOffset` ([lines 1879–1882](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1879-L1882)) are **only consumed** within lines 1895–2069. They should move inside the `if` block.

### Arrays that become dead when gated

These allocations on lines 1857–1877 should move inside the `if` block:

| Line | Variable | Size |
|------|----------|------|
| 1857 | `virtualPanoDepthOffsetsMm` | `Float32Array(49)` |
| 1862 | `virtualPanoStack` | `Float32Array(planeSize × 49)` ≈ 6.3 MB |
| 1863 | `virtualScoreByColDepth` | `Float32Array(panoWidth × 49)` |
| 1864–1866 | threshold arrays | 3× `Float32Array(panoWidth)` |
| 1867 | `virtualThresholdScratch` | `Float32Array(panoWidth)` |
| 1868 | `virtualCandidateDepthIndices` | `Int16Array(panoWidth × 5)` |
| 1870 | `virtualCandidateScores` | `Float32Array(panoWidth × 5)` |
| 1874 | `virtualDpBackPointers` | `Int16Array(panoWidth × 5)` |
| 1876 | `virtualSelectedDepthMm` | `Float32Array(panoWidth)` |
| 1877 | `virtualSelectedDepthScratch` | `Float32Array(panoWidth)` |

Moving them inside the `if` saves ~7 MB of allocation when disabled.

### Diagnostic log confirming Phase 0 is off

The existing log on [line 2348](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2348):

```
console.warn('[CPR-DIAGNOSTIC]', diagnosticPayload);
```

will emit `virtualPanoPhase12: { enabled: false, skippedReason: 'PHASE0_GATE' }`. No additional log line needed.

### Summary of changes

| Action | Lines |
|--------|-------|
| Add `ENABLE_VIRTUAL_PANO_PHASE12 = false` | Near 838 (1 line) |
| Hoist `let virtualPanoPhase12Diagnostics` | Before 1851 (1 line) |
| Wrap lines 1851–2220 in `if (ENABLE_VIRTUAL_PANO_PHASE12)` | 2 lines of control flow |
| Add `else` stub assignment | 3 lines |

**Total: ~6 lines of net new code. Zero pixel-data change.**

---

## 2. Runtime Instrumentation Points

Add `performance.now()` markers at five seams inside `generatePanorama()`:

| Marker variable | Insert after line | Captures |
|----------------|-------------------|----------|
| `_t0_start` | 838 (after input destructuring) | Start of all computation |
| `_t1_afterAdaptiveCenter` | 1555 (after final continuity enforcement pass) | End of adaptive center search |
| `_t2_afterTwoPassRender` | 1821 (after `lowerBandEligibleFraction` computation) | End of pass-1 + BFS + pass-2 render |
| `_t3_afterVirtualPano` | 2220 (after virtualPanoPhase12 diagnostics assembly, right before `backgroundSuppressionResult`) | End of virtualPanoPhase12 |
| `_t4_afterSuppressDenoise` | 2247 (after denoise range update) | End of suppression + denoise |
| `_t5_beforePayload` | 2274 (right before `const diagnosticPayload`) | End of all pixel work |

Add to `diagnosticPayload` object literal (inside the block starting at [line 2274](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2274)):

```typescript
timingMs: {
  adaptiveCenterSearch: Math.round(_t1_afterAdaptiveCenter - _t0_start),
  pass1And2TwoPassRender: Math.round(_t2_afterTwoPassRender - _t1_afterAdaptiveCenter),
  virtualPanoPhase12: Math.round(_t3_afterVirtualPano - _t2_afterTwoPassRender),
  suppressionAndDenoise: Math.round(_t4_afterSuppressDenoise - _t3_afterVirtualPano),
  diagnosticAssembly: Math.round(_t5_beforePayload - _t4_afterSuppressDenoise),
  total: Math.round(_t5_beforePayload - _t0_start),
},
```

This adds 6 `performance.now()` calls (each <1μs) and one diagnostic object field. The existing `[CPR-DIAGNOSTIC-JSON]` log on [line 2349](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2349) will automatically carry these timings.

**Removal:** Delete the 6 marker lines and the `timingMs` block.

---

## 3. Phase 1: Virtual-Pano Render Path Behind Flag

### What stays untouched

All legacy paths remain **completely unchanged**:

- Adaptive center search loop ([lines 1324–1555](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1324-L1555))
- Pass-1 provisional pano + intensity stack ([lines 1619–1644](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1619-L1644))
- Two-pass eligibility BFS ([lines 1644–1735](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1644-L1735))
- Pass-2 re-reduction into `pixelData` ([lines 1740–1815](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1740-L1815))
- Background suppression ([lines 2222–2229](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2222-L2229))
- Bilateral denoise ([lines 2231–2247](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2231-L2247))
- All diagnostic assembly and logging

### New input field

Add to `CPRWorkerInput` ([line 12](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L12)):

```typescript
reconstructionMode?: 'legacy' | 'virtualPano';
```

Wire from `useCPROrchestrator.ts` in `launchCPRWorker` params and `worker.postMessage` ([line 1155](file:///d:/diacom%20demo/Viewers/modes/cpr/src/useCPROrchestrator.ts#L1155)). Default to `'legacy'`.

### New render function

Add after `suppressLowerBackground` (after [line 754](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L754)):

```typescript
function renderFromSupportSurface(
  curvedStack: Float32Array,          // virtualPanoStack
  depthOffsetsMm: Float32Array,       // virtualPanoDepthOffsetsMm
  selectedDepthMm: Float32Array,      // virtualSelectedDepthMm (from DP)
  panoWidth: number,
  panoHeight: number,
  planeSize: number,
  depthSamples: number,
  panoCenterRow: number,
): { pixelData: Float32Array; minValue: number; maxValue: number }
```

### Reducer: why previous attempt blew out and what to use instead

The previous virtual-pano attempt likely used MIP or wide mean across all 49 depths. MIP always selects the brightest structure → blow-out. Wide mean averages 49 depths of mixed anatomy → contaminated mid-range.

**The correct first reducer** is a narrow-trough weighted upper-percentile mean around the DP-selected depth:

```
For each pixel (col, row):
  supportDepthMm = selectedDepthMm[col]
  yNorm = (row - panoCenterRow) / panoCenterRow

  // Row-adaptive trough width
  tau = yNorm < -0.10 ? 1.0 : yNorm <= 0.55 ? 1.4 : 0.5   // mm
  sigma = tau * 0.55                                         // mm

  // Collect eligible samples within trough
  for each depth d:
    delta = depthOffsetsMm[d] - supportDepthMm
    if |delta| > tau: skip
    value = curvedStack[d * planeSize + pixelIndex]
    if !isFinite(value): skip
    weight = exp(-0.5 * (delta / sigma)²)
    → eligibleBuffer[eligibleCount++]

  if eligibleCount == 0:
    → nearest-to-surface single sample fallback

  // Weighted upper-40% mean:
  sort eligible by value ascending
  totalWeight = sum weights
  walk from top, accumulate until cumulativeWeight >= 0.40 * totalWeight
  return weighted mean of accumulated set
```

**Why this won't blow out:**
- `tau ≤ 1.4 mm` → at most ~11 of 49 depth samples participate
- Upper-40% weighted mean is brighter than full mean but avoids pure MIP
- Gaussian depth weighting biases toward the DP surface
- NaN/OOB excluded

### Brightness normalization

No additional normalization is needed. `sampleWorldIntensityForVirtualPano` ([line 1823](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1823)) already applies the same modality LUT as the legacy path. The orchestrator's `computeAdaptivePanoVoi` ([line 451](file:///d:/diacom%20demo/Viewers/modes/cpr/src/useCPROrchestrator.ts#L451)) will auto-adapt to the new value distribution.

### Fallback to legacy output

After the Phase-1 render, evaluate a quick sanity check:

```typescript
if (reconstructionMode === 'virtualPano') {
  const vpResult = renderFromSupportSurface(...);
  const vpRange = vpResult.maxValue - vpResult.minValue;
  const vpUsable = vpRange > 200 && vpResult.minValue < 0 && vpResult.maxValue > 400;

  if (vpUsable) {
    pixelData.set(vpResult.pixelData);
    minValue = vpResult.minValue;
    maxValue = vpResult.maxValue;
    // diagnosticPayload.virtualPanoUsedAsOutput = true;
  } else {
    // pixelData already has legacy pass-2 output — keep it
    // diagnosticPayload.virtualPanoUsedAsOutput = false;
  }
}
```

The legacy pass-2 output is already in `pixelData` at this point. The virtual-pano render only *overwrites* if it passes sanity. Fallback is zero-cost.

### Files to touch for Phase 1

| File | Change |
|------|--------|
| [cprWorker.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts) | Add `renderFromSupportSurface()` (~120 lines). Set `ENABLE_VIRTUAL_PANO_PHASE12 = true` when `reconstructionMode === 'virtualPano'`. Add render + fallback block after line 2220. Add `reconstructionMode` to `CPRWorkerInput`. |
| [useCPROrchestrator.ts](file:///d:/diacom%20demo/Viewers/modes/cpr/src/useCPROrchestrator.ts) | Add `reconstructionMode` param to `launchCPRWorker` and pass through to `worker.postMessage` (line 1155). |

---

## 4. Geometry Claim Correction: N_slab Transport

### What I previously claimed (incorrect)

> `toSlabNormal` zeroes the Z component, causing rapid slab normal rotation at high-curvature points.

### What the current code actually does

The current `buildRMFFrames` ([lines 218–276](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L218-L276)) uses **Rotation Minimizing Frame (RMF) transport** for `N_slab`:

**Step 1 — Initialize** ([lines 196–207](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L196-L207)): `prevNslab` starts from `toSlabNormal(N, S)`. This is the **only point** where Z-zeroing directly defines the slab normal (the seed).

**Step 2 — Transport** ([lines 226–232](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L226-L232)): For each frame `i > 0`:
```typescript
transportedNslab = rotateAroundAxis(prevNslab, axis, angle);
```
This is Rodrigues rotation carrying `prevNslab` forward by the exact same rotation from `prevT → T`. **No Z-zeroing is re-applied.**

**Step 3 — Fallback** ([lines 244–246](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L244-L246)): `toSlabNormal(Ni, Si)` is used **only when** `norm(Nslab) < EPS` — a degenerate case, not the normal path.

**Step 4 — Vertical bias** ([lines 248–258](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L248-L258)):
```typescript
const blendedU = normalize(
  add(scale(currentU, 0.95), scale(normalizedTargetU, 0.05)),
  normalizedTargetU
);
Nslab = normalize(cross(T, blendedU), Nslab);
```
A gentle 5% per-frame pull toward volume vertical. Over N frames this accumulates as `1 - 0.95^N`. After ~50 frames through the anterior curve, `N_slab` is 92% biased toward vertical — **which is desirable** for a dental pano (slab normal should point toward the patient).

**Step 5 — Continuity** ([lines 260–262](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprMath.ts#L260-L262)): Sign flip prevention.

### Corrected assessment

The transported RMF N_slab with 5% vertical bias is a well-designed frame propagation. It does **not** cause the rapid slab-direction changes I previously attributed to it.

**Residual concern:** At the very beginning of the anterior turn (first ~5–10 columns entering the curve), the slab normal hasn't been biased much yet, and Rodrigues transport can cause the normal to rotate through an intermediate direction. But this is a **localized effect** covering at most ~5–10 columns.

**Revised recommendation:** Phase 3 (curvature-adaptive slab normal) should be **deprioritized to rank 5**. If anterior teeth still look wrong after Phase 1, the cause is more likely the depth slab not tracking the tooth surface — which Phase 1's virtual-pano render addresses directly.

---

## 5. Re-Ranked Phase Prioritization

| Rank | Phase | Image Quality | Runtime | Regression Risk |
|------|-------|--------------|---------|-----------------|
| **1** | **Phase 0: Gate virtualPanoPhase12** | None (identical) | **High** (~55–65% eliminated) | **Zero** |
| **2** | **Timing instrumentation** | None (identical) | None | **Zero** |
| **3** | **Phase 1: Virtual-pano render behind flag** | **High** (primary quality fix) | **Moderate positive** | **Low** (fallback to legacy) |
| **4** | **Phase 2: Per-column tilt + row-adaptive trough** | **Medium** (root visibility) | Negligible | **Medium** (tilt clamp guards) |
| **5** | **Phase 4: Single-pass architecture** | None (identical to 1+2) | **High** (under 10s total) | **Medium-high** (removes fallback) |
| **6** | **Phase 3: Curvature-adaptive slab normal** | **Low** (after correction) | Negligible | **Medium** |

### Key change from original

Phase 3 moved from rank 3 → rank 6. The RMF transport is adequate. Anterior distortion is primarily a depth-selection issue (Phase 1), not geometry.

### Recommended execution order

```
Phase 0  →  Instrument timing  →  Measure  →  Phase 1  →  Evaluate  →  Phase 2  →  Phase 4
                                                                ↓
                                                          (if anterior still bad)
                                                                ↓
                                                           Phase 3
```

---

## Appendix: Key Code Locations

| Reference | File | Lines |
|-----------|------|-------|
| `generatePanorama` entry | cprWorker.ts | 785–808 |
| Input destructuring | cprWorker.ts | 809–834 |
| Slab direction setup | cprWorker.ts | 1080–1091 |
| Adaptive center search | cprWorker.ts | 1324–1555 |
| Pass-1 intensity stack | cprWorker.ts | 1619–1644 |
| Two-pass eligibility BFS | cprWorker.ts | 1644–1735 |
| Pass-2 re-reduction | cprWorker.ts | 1740–1815 |
| `sampleWorldIntensityForVirtualPano` | cprWorker.ts | 1823–1849 |
| virtualPanoPhase12 full block | cprWorker.ts | 1851–2220 |
| Background suppression | cprWorker.ts | 2222–2229 |
| Bilateral denoise | cprWorker.ts | 2231–2247 |
| Diagnostic payload + `virtualPanoPhase12` ref | cprWorker.ts | 2274–2340 |
| `buildRMFFrames` N_slab init | cprMath.ts | 196–207 |
| `buildRMFFrames` N_slab transport | cprMath.ts | 226–232 |
| `buildRMFFrames` vertical bias | cprMath.ts | 248–258 |
| `toSlabNormal` (fallback only) | cprMath.ts | 150–162 |
| `launchCPRWorker` + `worker.postMessage` | useCPROrchestrator.ts | 897–1186 |
