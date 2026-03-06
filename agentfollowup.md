Below is the **tightened implementation spec** for the three patches, with the current call architecture in mind.

I’ll assume `sampleRow` increases downward. If your row axis is inverted, flip the sign in the `yNorm` formulas.

---

# 1) `sampleReducedPoint()` one-pass contamination gate

## A. Extra parameters to add

Do **not** add several new positional args. Add one optional context object so the existing two call sites stay readable.

### New signature

```ts
sampleReducedPoint(
  bx, by, bz,
  slabDirX, slabDirY, slabDirZ,
  recordOobStats,
  debugCaptureRow,
  gateCtx?: {
    mode: 'centerSearch' | 'finalRender';

    // only used in finalRender mode
    sampleRow?: number;
    localCenterRow?: number;
    effectiveVerticalHalfHeight?: number;

    // derived from the already-computed local vertical profile at this column
    profileValueAtRow?: number;
    profilePeakValue?: number;
    profileFloorValue?: number;
  }
)
```

## B. What runs in center-search vs final-render

### 1. During center-search calls

Use **no center-dependent anatomy gate**.

Keep exactly:

* in-bounds filtering
* existing Gaussian slab-depth weighting
* current MEAN / `computeWinsorizedWeightedMean()`

Reason: during center search, any gate that depends on `candidateCenterRow` or vertical position relative to center will bias the search toward its own prior.

So for `gateCtx.mode === 'centerSearch'`:

* **do not** use `sampleRow`
* **do not** use `localCenterRow`
* **do not** use `profileConf`
* only increment a call counter and return the current behavior

### 2. During final pano render

Apply the anatomy-aware gate.

This is safe because the center has already been selected, so the gate is no longer participating in center selection.

---

## C. How to avoid circular dependence

Use this rule:

* **Center-search path:** no center-dependent gate
* **Final-render path:** gate allowed, using the already-chosen `localCenterRow` and the already-computed vertical profile for that column

That breaks the circular loop cleanly.

In other words:

* center search chooses the center from the raw/current profile logic
* final render uses that chosen center to suppress lower/background slab samples

Do **not** feed the gated reducer back into candidate scoring in this first patch.

---

## D. Exact `profileConf` formula from current worker signals

Use the already-computed local vertical profile for the current column.

Let:

* `P(r)` = smoothed local vertical profile value at row `r`
* `c` = chosen local center row
* `sampleRow` = pano row currently being rendered
* `Ppeak = P(c)`
* `Pfloor = q25(P over the current center-search row window)`
  computed once per column and reused for all rows in that column
* `Prow = P(sampleRow)` using linear interpolation if needed

Then:

```ts
profileConf = clamp(
  (Prow - Pfloor) / max(Ppeak - Pfloor, 1e-6),
  0,
  1
)
```

### What to pass into `sampleReducedPoint()`

In the final-render caller, pass:

* `profileValueAtRow = Prow`
* `profilePeakValue = Ppeak`
* `profileFloorValue = Pfloor`

Then inside `sampleReducedPoint()`:

```ts
profileConf =
  clamp(
    (profileValueAtRow - profileFloorValue) /
    Math.max(profilePeakValue - profileFloorValue, 1e-6),
    0,
    1
  )
```

This uses only signals already present in the worker and keeps `sampleReducedPoint()` simple.

---

## E. Exact gating formula in final-render mode

For each slab sample, keep the existing slab-depth Gaussian:

```ts
wDepth = exp(-0.5 * (slabOffset / sigmaSlab) ** 2)
```

Compute normalized vertical position:

```ts
yNorm =
  (sampleRow - localCenterRow) /
  Math.max(effectiveVerticalHalfHeight, 1)
```

Where:

* `yNorm <= 0`: at/above center
* `yNorm > 0`: below center

### Lower-band attenuation

```ts
wLower =
  yNorm <= 0
    ? 1
    : 1 - 0.80 * smoothstep(0.20, 0.95, yNorm)
```

So:

* near/above center: unchanged
* moderately below center: progressively reduced
* deep lower band: strongly reduced, not automatically zero

### Profile confidence weight

```ts
wProfile = 0.25 + 0.75 * profileConf
```

This ensures low-confidence rows are suppressed, but never annihilated by profile term alone.

### Base anatomy weight

```ts
wAnatomy = wLower * wProfile
```

---

## F. What to do when slab sample count is too small for stable `q75`

Let `n` be the number of in-bounds slab samples collected for this reduce point.

### Bright reference rule

Use a brightness-rescue reference only in final render:

* `n >= 5`
  `brightRef = q75(values)`
* `n == 4`
  `brightRef = secondHighest(values)`
* `n == 3`
  `brightRef = max(values)` and **disable hard reject**
* `n <= 2`
  **disable hard reject and brightness rescue thresholding entirely**

This is the safest small-`n` rule.

Why:

* with `3–6` samples, `q75` is noisy
* for `n <= 3`, hard rejection becomes too brittle
* the first patch should prefer **downweighting** over aggressive deletion when evidence is thin

---

## G. Exact reject vs downweight logic

### Bright rescue

If the sample is bright enough, keep it from being crushed:

```ts
isBrightRescue =
  n >= 5 ? value >= brightRef
  : n == 4 ? value >= brightRef
  : n == 3 ? value >= brightRef   // only the max sample
  : false
```

If:

* `yNorm > 0.20`
* and `isBrightRescue`

then:

```ts
wAnatomy = Math.max(wAnatomy, 0.50)
```

This is the main protection for true root/cortical signal.

### Hard reject

Only allow hard reject when evidence is strong enough:

```ts
allowHardReject = n >= 4
```

Then reject only if all are true:

```ts
allowHardReject &&
yNorm > 1.00 &&
profileConf < 0.15 &&
value < brightRef
```

That targets:

* deep below-center samples
* weak tooth-band support
* not among the brighter local slab samples

### Strong downweight

If not rejected, but still suspicious:

```ts
if (
  yNorm > 0.55 &&
  profileConf < 0.30 &&
  (!isBrightRescue)
) {
  wAnatomy *= 0.25
}
```

### Final weight

```ts
wFinal = wDepth * wAnatomy
```

If every sample ends up rejected or near-zero weighted:

* fall back to the **original ungated reducer result** for that point
* increment a fallback counter

That keeps the first patch safe.

---

## H. How true root / cortical signal is preserved

Three explicit safeguards:

### 1. No hard reject for small `n`

For `n <= 3`, never hard-reject.

### 2. Bright rescue floor

Any bright lower sample keeps a nontrivial weight floor:

```ts
wAnatomy >= 0.50
```

That preserves:

* bright root cortex
* strong cortical boundaries
* real bright dental structure dipping below the center band

### 3. Upper/central band untouched

For `yNorm <= 0.20`, lower suppression barely engages.

That avoids flattening the main tooth body.

---

## I. Exact debug counters to add

Add these to the worker diagnostics.

### Call-path counters

* `sampleReduceCenterSearchCallCount`
* `sampleReduceFinalRenderCallCount`

### Gate action counters

* `anatGateAppliedCount`
* `anatGateRejectedSampleCount`
* `anatGateDownweightedSampleCount`
* `anatGateBrightRescueCount`
* `anatGateSmallNSkipHardRejectCount`
* `anatGateNoValidWeightedSampleFallbackCount`

### Region counters

* `anatGateBelowCenterSampleCount`
* `anatGateDeepLowerRejectCount`

### Small scalar accumulators

* `anatGateWeightSumBefore`
* `anatGateWeightSumAfter`
* `anatGateRejectedYNormSum`

From those you can derive:

* mean retained weight ratio
* mean `yNorm` of rejected samples

---

## J. What existing logs should improve first

If this patch is working, the first metrics that should move are:

* `lowerBandBrightFraction`
  should drop first and most
* `lowerBandP50`
  should drop
* `qualityScore`
  should improve materially

Likely secondary:

* `detailBandHorizontalEdgeMean` should come down somewhat
* `detailBandVerticalEdgeMean` may hold or rise slightly

If `lowerBandBrightFraction` drops but teeth/roots thin out, the gate is too aggressive.

---

# 2) `buildRMFFrames()` transported slab-frame patch

## A. Frame 0 initialization

Replace the first-frame setup with:

```ts
T0 = normalize(T[0])

Useed = normalizedVerticalDir - dot(normalizedVerticalDir, T0) * T0

if (length(Useed) < eps) {
  // pick fallback axis least aligned with T0, then project it off T0
  fallback = leastAlignedCardinalAxis(T0)
  Useed = fallback - dot(fallback, T0) * T0
}

U0 = normalize(Useed)

// preserve current handedness convention
N0 = normalize(cross(T0, U0))

// re-orthogonalize once
U0 = normalize(cross(N0, T0))
```

This keeps the initial orientation compatible with the old convention while only using global vertical once.

---

## B. Exact transport update from frame `i-1` to `i`

For each `i >= 1`:

```ts
Tprev = normalize(T[i - 1])
Tcur  = normalize(T[i])

Nprev = N[i - 1]
Uprev = U[i - 1]

axis = cross(Tprev, Tcur)
s = length(axis)
c = clamp(dot(Tprev, Tcur), -1, 1)
```

### If the tangent changed meaningfully

If `s > 1e-6`:

```ts
k = axis / s
theta = atan2(s, c)

Ncand = rodriguesRotate(Nprev, k, theta)
Ucand = rodriguesRotate(Uprev, k, theta)
```

### If the tangent barely changed

Else:

```ts
Ncand = Nprev
Ucand = Uprev
```

Transporting both `N` and `U` is slightly safer than transporting only `N`.

---

## C. Exact re-orthogonalization steps

After transport:

```ts
Nproj = Ncand - dot(Ncand, Tcur) * Tcur
if (length(Nproj) < eps) {
  // rare fallback
  Useed = normalizedVerticalDir - dot(normalizedVerticalDir, Tcur) * Tcur
  if (length(Useed) < eps) {
    fallback = leastAlignedCardinalAxis(Tcur)
    Useed = fallback - dot(fallback, Tcur) * Tcur
  }
  Utmp = normalize(Useed)
  Nproj = normalize(cross(Tcur, Utmp))
}

Ncur = normalize(Nproj)
Ucur = normalize(cross(Ncur, Tcur))
```

This restores an orthonormal frame even after numeric drift.

---

## D. Soft bias to global vertical

Yes, keep it, but very small.

### Recommended alpha

```ts
alpha = 0.05
```

Do **not** recompute `N` from `cross(T, globalVertical)`.

Bias only the vertical-like axis `Ucur` toward the global-up projection:

```ts
Utarget = normalizedVerticalDir - dot(normalizedVerticalDir, Tcur) * Tcur

if (length(Utarget) >= eps) {
  Utarget = normalize(Utarget)

  Ublend = normalize((1 - alpha) * Ucur + alpha * Utarget)
  Nblend = normalize(cross(Tcur, Ublend))
  Ublend = normalize(cross(Nblend, Tcur))

  Ncur = Nblend
  Ucur = Ublend
}
```

That limits long-run roll drift without reintroducing the old frame re-anchoring bug.

---

## E. Preserve sign continuity of `N_slab`

After re-orthogonalization and soft bias, enforce sign continuity:

```ts
if (dot(Ncur, Nprev) < 0) {
  Ncur = -Ncur
  Ucur = -Ucur
}
```

Flip both axes together so handedness stays consistent.

This is important because a numerically valid frame can still flip sign from one sample to the next, which looks like shear/twist.

---

## F. Diagnostics to log

Add these to confirm the transported frame is stable:

* `rmfTurnAngleMeanDeg`
* `rmfTurnAngleMaxDeg`
* `rmfNormalDeltaMeanDeg`
* `rmfNormalDeltaMaxDeg`
* `rmfBiasCorrectionMeanDeg`
* `rmfBiasCorrectionMaxDeg`
* `rmfFlipCorrectionCount`
* `rmfFallbackCount`
* `rmfOrthoErrorMax`

Where:

* `normalDeltaDeg = acos(clamp(dot(Ncur, Nprev), -1, 1)) * 180 / PI`
* `biasCorrectionDeg = acos(clamp(dot(UcurBeforeBias, UcurAfterBias), -1, 1)) * 180 / PI`
* `orthoErrorMax` is the max absolute value of:

  * `dot(Tcur, Ncur)`
  * `dot(Tcur, Ucur)`
  * `dot(Ncur, Ucur)`

If you want one targeted metric for the bug region, also log the same `normalDelta` stats for the central/anterior third of the arch.

---

# 3) Curvature-aware local-center priors

## A. Exact turn-angle computation

Use a small local average so the curvature estimate does not flicker.

For column `j`:

```ts
a0 = angleDeg(T[j - 1], T[j])       // if j > 0
a1 = angleDeg(T[j], T[j + 1])       // if j + 1 < count
turnAngleDeg = mean(of available a0, a1)
```

Where:

```ts
angleDeg(A, B) =
  acos(clamp(dot(normalize(A), normalize(B)), -1, 1)) * 180 / Math.PI
```

This is more stable than using only one side.

If only one neighbor exists, use the one-sided angle.

---

## B. Recommended `smoothstep` range

For dental CPR, keep the onset conservative:

```ts
kRaw = smoothstep(2.0, 7.0, turnAngleDeg)
```

Then smooth it across columns to avoid wobble:

```ts
k =
  0.25 * kRaw[j - 1] +
  0.50 * kRaw[j] +
  0.25 * kRaw[j + 1]
```

Use edge-safe handling at the ends.

This is the simplest way to avoid per-column oscillation.

---

## C. Exact new penalty formulas

Keep the same score structure. Only replace the fixed weights.

### New global-center penalty

```ts
wGlobal = 26 * (1 - 0.65 * k)
```

Range:

* straight zones: `26`
* high-curvature zones: about `9.1`

### New previous-column continuity penalty

```ts
wPrev = 18 * (1 - 0.35 * k)
```

Range:

* straight zones: `18`
* high-curvature zones: about `11.7`

This is intentionally asymmetric:

* relax global-center prior more
* keep continuity prior stronger to prevent wobble

---

## D. Should `searchHalfRange` and `maxDeviationMm` widen?

Yes, but modestly.

### Search half-range

```ts
searchHalfRange =
  Math.round(baseSearchHalfRange * (1 + 0.20 * k))
```

### Max deviation

```ts
maxDeviationMm =
  baseMaxDeviationMm * (1 + 0.15 * k)
```

Do not widen more than this in the first patch.

The main effect should come from lighter penalties, not from a huge search expansion.

---

## E. Which current log values should improve first

If this patch is working:

* `detailBandVerticalEdgeMean` should improve first
* `qualityScore` should improve
* the anterior should look less flattened / less off-center

Possible smaller secondary changes:

* `lowerBandBrightFraction` may improve a little if the center stops drifting downward
* `lowerBandP50` may improve slightly

But the main observable gain should be **vertical tooth structure**.

---

## F. How to avoid making posterior columns wobble

Use all four safeguards together:

### 1. Smoothed curvature factor

Use the 3-column smoothing on `k`.

### 2. Conservative continuity relaxation

Do not relax `wPrev` more than 35%.

### 3. Modest search widening only

Keep the widening to:

* `+20%` for `searchHalfRange`
* `+15%` for `maxDeviationMm`

### 4. Keep posterior nearly unchanged

Because `k` is near zero in straighter regions, posterior columns stay very close to the old behavior.

That is the minimal safe way to free the anterior without destabilizing the posterior.

---

# Recommended patch order

1. `sampleReducedPoint()`
   one-pass gate in **final-render mode only**
2. `buildRMFFrames()`
   true transported frames + sign continuity
3. local-center priors
   curvature-aware relaxation with smoothed `k`

That is still the smallest safe sequence.

