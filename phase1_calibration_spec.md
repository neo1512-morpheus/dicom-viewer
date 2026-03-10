# Phase 1 Calibration Spec — Exact Decisions

Based on the actual failed virtual-render metrics and current scoring/DP code.

---

## 1. Path-Edge Failure Fix

### Root cause

The current depth scoring ([line 2052](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2052)):

```typescript
const score = 0.55 * hardMean + 0.3 * gradMean - 0.4 * lowMean;
```

has **no distance-to-boundary penalty**. The `Hard` response rewards the brightest anatomy regardless of whether it's at depth index 0 or 48 (i.e., at ±6mm boundary). When the brightest bone/tooth structure happens to sit near the edge of the ±6mm search range, the DP selects it freely — explaining the `selectedDepthFirst8Mm` values of 5.25, 5.313, 5.438, 5.5 (hugging the +6mm edge).

The DP transition cost ([line 2125](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L2125)):

```typescript
const transitionCost = 0.22 * jumpMm + 0.12 * Math.max(0, jumpMm - 0.75);
```

penalizes **column-to-column jumps** but not absolute edge proximity. A path that smoothly glides along the +6mm boundary pays zero boundary penalty.

### Fix: add edge-distance penalty in `Score(c,d)`

Add a boundary penalty term to the score formula. Place it in the **score** (not the DP transition), because the problem is that edge candidates get high scores and enter the top-K candidate set intact:

```typescript
// After line 2051 (after lowMean is computed), replace line 2052:
const depthMm = virtualPanoDepthOffsetsMm[depth];
const edgeDistanceMm = virtualPanoDepthHalfRangeMm - Math.abs(depthMm);
const edgePenalty = edgeDistanceMm < 1.5
  ? 0.35 * (1 - edgeDistanceMm / 1.5)   // ramps from 0.35 at boundary to 0 at 1.5mm in
  : 0;
const score = 0.55 * hardMean + 0.3 * gradMean - 0.4 * lowMean - edgePenalty;
```

**Why these values:**
- `1.5mm` fade zone: at 0.25mm depth step, this affects the outer 6 depth indices on each side (indices 0–5 and 43–48). This is narrow enough not to penalize legitimate mid-range teeth.
- `0.35` max penalty: the `Hard` response is clamped [0,1], weighted 0.55, so max hard contribution = 0.55. A 0.35 penalty at the very edge is ~64% of the max hard contribution. This makes an edge candidate need substantially better `Hard + Grad - Low` to compete with a mid-range one.
- Linear ramp (not smoothstep): avoids a sharp cutoff that could cause path oscillation near the 1.5mm zone boundary.

### Should DP transition also have edge penalty?

**No, not in the first implementation.** The score-level penalty is sufficient to prevent edge candidates from entering the top-K. Adding a second penalty in the DP would double-penalize and could push the path too aggressively toward depth=0 (center). Keep the DP transition cost exactly as-is for Phase 1. If paths still hug boundaries after the scoring penalty, add a small DP term in a follow-up iteration.

---

## 2. Reducer Choice: Not Upper-40%

### Reassessment against failed metrics

The previous blown-out output had:
- `toothBandMean ≈ 1550–1607` (extremely bright)
- `p50 ≈ 1455`
- `lowerBandMean ≈ 685–835` (far too bright)
- `lowerBandBrightFraction ≈ 0.90`
- `offTroughEnergyRatio ≈ 4.8–5.4`

An upper-40% weighted mean emphasizes the brightest samples — it would produce values **above** the weighted-mean baseline. With a tooth band already at 1550+, this would land even higher. The failed output was already blown-out white; upper-tail emphasis **makes it worse**, not better.

### Correct Phase-1 reducer: centered weighted mean (no upper-tail bias)

Use a **plain Gaussian-weighted mean** around the support surface for Phase 1:

```
For each pixel (col, row):
  supportDepthMm = selectedDepthMm[col]
  yNorm = (row - panoCenterRow) / panoCenterRow

  // Row-adaptive trough width
  tau = yNorm < -0.10 ? 1.0 : yNorm <= 0.55 ? 1.4 : 0.5   // mm eligibility radius
  sigma = tau * 0.55                                         // Gaussian sigma

  eligibleCount = 0
  for each depth d:
    delta = depthOffsetsMm[d] - supportDepthMm
    if |delta| > tau: skip
    value = curvedStack[d * planeSize + pixelIndex]
    if !isFinite(value): skip
    weight = exp(-0.5 * (delta / sigma)²)
    eligibleValues[eligibleCount] = value
    eligibleWeights[eligibleCount] = weight
    eligibleCount++

  if eligibleCount == 0:
    → nearest-to-surface single sample fallback
    → if unavailable, -1000

  // Plain weighted mean — no sorting, no upper-tail selection
  weightedSum = 0
  weightTotal = 0
  for i = 0 to eligibleCount-1:
    weightedSum += eligibleValues[i] * eligibleWeights[i]
    weightTotal += eligibleWeights[i]

  pixelValue = weightTotal > 0 ? weightedSum / weightTotal : -1000
```

**Why plain weighted mean first:**
- It is strictly less likely to blow out than upper-tail
- The Gaussian centering on the DP surface already provides anatomy selectivity
- The narrow trough (`tau ≤ 1.4mm` → max ~11 of 49 samples) already excludes most contamination
- If the plain weighted mean looks too flat/soft after proving stability, adding upper-tail emphasis is a ~5-line change to the reducer — easy to layer on later
- The failed render's `toothBandMean ≈ 1550–1607` was high because it was likely MIP-like or upper-weighted; a centered mean will land lower and closer to the legacy 600–800 range

### When to add upper-tail emphasis

Only after Phase 1 plain-mean render produces:
- `lowerBandBrightFraction < 0.55`
- `toothBandMean / lowerBandMean > 2.0`
- Stable output with no white blow-out

Then switch to upper-30% (not 40%) weighted mean in a follow-up.

---

## 3. Threshold Calibration

### Why current thresholds are too high

The threshold computation ([lines 1920–1969](file:///d:/diacom%20demo/Viewers/modes/cpr/src/cprWorker.ts#L1920-L1969)) collects `toothThresholdSamples` from **all depths** within the tooth-band rows:

```typescript
for (let depth = 0; depth < virtualPanoDepthSamples; depth += virtualThresholdDepthStep) {
  for (let row = toothBandStartRow; row <= toothBandEndRow; row += virtualThresholdRowStep) {
    // ...
    toothThresholdSamples.push(value);
  }
}
```

With `depthStep = 2` and 49 total depths, this collects ~25 depth planes × ~90 rows = ~2250 samples per column. These span air (−1000), soft tissue (~50), bone (~500–1200), and tooth enamel (~1500–3000). The 60th percentile of 2250 samples from such a mixed population easily reaches ~1000+, and the 85th percentile reaches ~1800+.

**Result:** The `Hard` response `clamp((value - T_soft) / (T_hard - T_soft), 0, 1)` only fires for the very brightest structures. This selects the depth with the densest bone/tooth, which is often at the search boundary because the arch curve has the jaw bone at the edge of the slab.

### Fix: restrict threshold computation to central depths only

Instead of pooling all depths, compute thresholds from only the **inner 60% of the depth range** (indices ~10–39 out of 0–48):

```typescript
const thresholdDepthMargin = Math.max(3, Math.floor(virtualPanoDepthSamples * 0.2));
const thresholdDepthStart = thresholdDepthMargin;
const thresholdDepthEnd = virtualPanoDepthSamples - 1 - thresholdDepthMargin;

for (let depth = thresholdDepthStart; depth <= thresholdDepthEnd; depth += virtualThresholdDepthStep) {
  // ... same row loop
}
```

And additionally, **lower the percentile ranks** to be less aggressive:

```typescript
virtualSoftThresholdByCol[col] =
  toothThresholdSamples.length > 0 ? percentile(toothThresholdSamples, 0.35) : -200;
virtualHardThresholdByCol[col] =
  toothThresholdSamples.length > 0 ? percentile(toothThresholdSamples, 0.70) : 300;
```

**Why `0.35` and `0.70` (down from `0.60` and `0.85`):**
- After excluding edge depths, the sample pool is ~60% as large and contains fewer extreme-boundary bone values
- The 35th percentile of central depths will land closer to the tissue/bone transition (~300–600 HU), not deep into bone territory
- The 70th percentile will land around ~800–1200 HU, which is the true boundary where tooth structure begins to dominate
- The `Hard` response `(value - T_soft) / (T_hard - T_soft)` will now produce nonzero responses for a wider range of anatomy, not just the brightest structures

**Do not switch to rank-based thresholds** in Phase 1. Rank-based approaches need more careful calibration across different scanners and require additional infrastructure. The intensity-percentile approach is fine once thresholds are computed from the right sample pool.

---

## 4. Stricter Fallback Gate

### Why the previous gate was too weak

The suggested check `range > 200 && min < 0 && max > 400` would pass almost any output that isn't pure NaN or pure zero. The blown-out virtual render had range ~3500, min ~−1000, max ~2500 — it passes trivially.

### Exact `virtualPanoUsable` gate

Compute these metrics from the virtual-pano render's `pixelData` before deciding to use it:

```typescript
// Compute from the virtual-pano render output (before overwriting legacy pixelData)
const vpSummary = summarizeVirtualPanoOutput(vpPixelData, panoWidth, panoHeight, panoCenterRow);
// (this is a lightweight function, similar to existing summarizeFloatBufferForDebug)

const virtualPanoUsable =
  // 1. Lower band must not be excessively bright
  vpSummary.lowerBandMean < vpSummary.toothBandMean * 0.65 &&

  // 2. Lower band brightness fraction must be materially better than legacy baseline
  vpSummary.lowerBandBrightFraction < 0.70 &&

  // 3. Tooth band must have meaningful contrast (not flat white)
  vpSummary.toothBandP90 - vpSummary.toothBandP10 > 300 &&

  // 4. Support path must not be clamped to boundary for majority of columns
  vpSummary.supportDepthClampFraction < 0.40 &&

  // 5. Overall range sanity
  vpSummary.range > 400 &&
  vpSummary.minValue < 200;
```

### Metric definitions

```typescript
function summarizeVirtualPanoOutput(
  pixelData: Float32Array,
  panoWidth: number,
  panoHeight: number,
  panoCenterRow: number,
): {
  minValue: number;
  maxValue: number;
  range: number;
  toothBandMean: number;
  toothBandP10: number;
  toothBandP90: number;
  lowerBandMean: number;
  lowerBandBrightFraction: number;
  supportDepthClampFraction: number;
}
```

Where:
- **toothBand**: rows where `yNorm ∈ [-0.35, 0.55]` (same as scoring bands)
- **lowerBand**: rows where `yNorm ∈ [0.65, 1.15]`
- **lowerBandBrightFraction**: fraction of lower-band pixels where `value > -200`
- **supportDepthClampFraction**: `supportDepthClampCount / panoWidth`, where `supportDepthClampCount` counts columns where `|selectedDepthMm[col]| > depthHalfRangeMm - 0.5`

### Threshold justification against failed metrics

| Gate | Failed render value | Threshold | Would it catch failure? |
|------|-------------------|-----------|------------------------|
| `lowerBandMean / toothBandMean < 0.65` | 685/1550 = 0.44; 835/1607 = 0.52 | 0.65 | ❌ **No** — this ratio was actually not terrible. But combined with other gates: |
| `lowerBandBrightFraction < 0.70` | 0.90 | 0.70 | ✅ **Yes** |
| `toothBandP90 - toothBandP10 > 300` | Unknown, but blown-out suggests compressed | 300 | Likely ✅ |
| `supportDepthClampFraction < 0.40` | `selectedDepthFirst8Mm` all near 5.25–5.5; likely >50% clamped | 0.40 | ✅ **Yes** |
| `range > 400` | ~3500 | 400 | ❌ Would pass |
| `minValue < 200` | ~−1000 | 200 | ❌ Would pass |

The **two primary catches** are `lowerBandBrightFraction < 0.70` and `supportDepthClampFraction < 0.40`. These are the strongest discriminators against the observed failure mode.

> [!IMPORTANT]
> The lowerBandMean/toothBandMean ratio gate alone would NOT have caught the failure (0.44 and 0.52 are both < 0.65). The multi-gate approach is essential.

### Log the gate decision

```typescript
diagnosticPayload.virtualPanoGate = {
  usedAsOutput: virtualPanoUsable,
  lowerBandMean: vpSummary.lowerBandMean,
  toothBandMean: vpSummary.toothBandMean,
  lbToTbRatio: vpSummary.toothBandMean > 0
    ? Math.round((vpSummary.lowerBandMean / vpSummary.toothBandMean) * 1000) / 1000 : null,
  lowerBandBrightFraction: vpSummary.lowerBandBrightFraction,
  toothBandContrastRange: vpSummary.toothBandP90 - vpSummary.toothBandP10,
  supportDepthClampFraction: vpSummary.supportDepthClampFraction,
  range: vpSummary.range,
  minValue: vpSummary.minValue,
};
```

---

## 5. Phase-1 Render Scope: No Tilt

**Use `D[c]` only. No tilt in Phase 1.**

Rationale:
- Tilt estimation requires running depth scoring again with separate top-band and bottom-band scoring. This doubles the scoring code surface area and introduces a new failure mode (incorrect tilt pushing roots out of the trough).
- The failed metrics don't suggest tilt as the primary problem. The failure was: wrong depth selection (boundary hugging) → wrong anatomy sampled → blown-out output.
- Phase 1's job: prove that the narrow trough around a correct support path produces stable, non-blown-out output with clean background.
- Tilt is Phase 2, after Phase 1 is proven stable.

**Concretely:** `supportDepthMm` for each `(col, row)` pixel is simply `selectedDepthMm[col]` — the same depth for all rows in that column. The row-adaptive trough width (`tau`) provides enough per-row flexibility via eligibility radius without geometric tilt.

---

## 6. Expected Metric Movement If Calibrated Correctly

### Metrics that should move first (in priority order)

| Metric | Failed value | Target after Phase 1 | Direction | Why |
|--------|-------------|---------------------|-----------|-----|
| **supportDepthClampFraction** | ~0.50+ (estimated from first8) | **< 0.20** | ↓ strongly | Edge penalty directly addresses boundary hugging |
| **lowerBandBrightFraction** | 0.90 | **< 0.55** | ↓ strongly | Narrow trough excludes off-surface bone from lower band; lower thresholds stop selecting bone-dominated depths |
| **lowerBandP50** | ~926 HU | **< 100 HU** | ↓ strongly | Lower band now samples air/soft tissue near the surface, not bone at the slab edge |
| **lowerBandMean** | 685–835 | **< 200** | ↓ strongly | Same mechanism as lowerBandP50 |
| **toothBandMean / lowerBandMean ratio** | ~2.0 (1550/750) | **> 3.5** | ↑ | Tooth band stays bright (narrow trough centers on teeth); lower band drops |
| **offTroughEnergyRatio** | 4.8–5.4 | **< 2.0** | ↓ | With correct support path, off-trough energy is genuinely low (the bright anatomy is *on* the surface, not off it) |
| **p50** | ~1455 | **~400–700** | ↓ | Plain weighted mean lands lower than MIP/upper-tail; this is expected and correct — the adaptive VOI will compensate |
| **toothBandMean** | 1550–1607 | **~800–1200** | ↓ somewhat | Plain weighted mean is softer; acceptable for Phase 1 |
| **detailBandHorizontalEdgeMean / detailBandVerticalEdgeMean** | 399/161 ≈ 2.5 | **< 2.0** | ↓ | Background grain (horizontal noise) reduces; vertical tooth edges preserved |

### What should NOT change (or should stay stable)

| Metric | Current value | Expected | Note |
|--------|--------------|----------|------|
| `minValue` | ~−1000 | ~−1000 | OOB/air still −1000 |
| `maxValue` | ~2500+ | ~1800–2500 | Teeth still bright, but capped by mean not MIP |
| `selectedQualityScore` (orchestrator) | Strongly negative | Should improve toward 0 | Computed from the summary in orchestrator |

### First visual indicator of success

The lower half of the pano should go **dark** (near −1000 to 0 HU) instead of speckled gray/white. This is the most obvious visual signal. Teeth in the tooth band should remain bright and individually distinguishable, but possibly slightly less crisp than MIP — acceptable for Phase 1, improvable in Phase 2 (tilt) and a future upper-tail reducer.

---

## Summary of All Calibration Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Edge-distance penalty | In `Score(c,d)`, linear 0.35 ramp below 1.5mm from boundary | Prevents top-K candidates from being dominated by edge depths |
| DP boundary penalty | None in Phase 1 | Score-level penalty is sufficient; avoid double-penalizing |
| Reducer | Plain Gaussian-weighted mean (no upper-tail) | Failed render was blown-out; upper-tail makes it worse |
| Threshold percentiles | `p35` (soft), `p70` (hard), from central 60% of depth range | Current p60/p85 across all depths inflates thresholds above bone level |
| Fallback gate | 5-criterion multi-gate with `lowerBandBrightFraction < 0.70` and `supportDepthClampFraction < 0.40` as primary catches | No single criterion would catch all failure modes |
| Tilt | No tilt in Phase 1 | Reduce scope; not the primary failure cause |
| Upper-tail emphasis | Deferred to post-Phase-1 follow-up | Only after stability proven |
