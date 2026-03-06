Yes — here is the **minimal CPU-only implementation spec** for all three patches.

I’ll keep this tightly scoped to the smallest safe algorithmic changes in the existing flow.

---

# 1) Minimal one-pass contamination fix inside `sampleReducedPoint()` in `cprWorker.ts`

## Goal

Suppress **lower/background anatomy before reduction** while keeping true tooth, root, and cortical structure.

The key is to add a **per-sample anatomy gate** before the MEAN reducer, not to retune winsorization again.

## 1. Exact gating signals to use

Use only signals already available during render:

### A. Normalized vertical position relative to current local center

For each slab sample, compute:

```ts
yNorm = (sampleRow - localCenterRow) / max(effectiveVerticalHalfHeight, 1)
```

Assumption: **positive `yNorm` means below the local center**.
If your row convention is opposite, flip the sign.

This is the most important signal. The current failure is overwhelmingly **lower-band contamination**, so this signal should dominate the gate.

### B. Existing slab-depth Gaussian weight

Keep the existing depth weighting:

```ts
wDepth = exp(-0.5 * (slabOffset / sigmaSlab)²)
```

Do not remove it. Just stop letting it be the only thing.

### C. Local vertical-profile tooth-likeness / band confidence

Use the same local vertical profile behavior already available in the worker to derive a simple confidence in `[0, 1]`:

```ts
profileConf = clamp(localToothBandConfidenceAtThisRow, 0, 1)
```

If you do not have a per-row confidence directly, use a cheaper proxy:

```ts
profileConf = clamp(1 - smoothstep(0.45, 1.10, abs(yNorm)), 0, 1)
```

That fallback is weaker, but still useful.

### D. Sample intensity relative to the slab’s own robust bright level

To preserve true roots/cortex, use a simple local brightness rescue.
For the current sample set, compute a cheap robust threshold such as `q75` of the in-bounds sample intensities.

You do **not** need a fancy model. Just:

* collect in-bounds slab sample values
* compute `q75`
* optionally also compute `q90` if easy

This is used only as a **rescue**, not as the main gate.

---

## 2. Weighting formula for each slab sample

Use this structure:

```ts
w = wDepth * wAnatomy
```

where:

```ts
wLower = yNorm <= 0
  ? 1
  : 1 - 0.85 * smoothstep(0.20, 0.90, yNorm)

wProfile = 0.20 + 0.80 * profileConf
```

Then apply a brightness rescue so real bright root/cortical structure is not over-suppressed:

```ts
isBrightRescue = value >= q75
```

Final anatomy weight:

```ts
wAnatomy = wLower * wProfile
```

Then enforce a rescue floor for bright structure below center:

```ts
if (yNorm > 0.20 && isBrightRescue) {
  wAnatomy = max(wAnatomy, 0.50)
}
```

Final per-sample weight:

```ts
w = wDepth * wAnatomy
```

This is intentionally simple. It is the smallest patch that directly attacks the real failure mode.

---

## 3. What should be rejected vs only downweighted

### Hard reject

Reject only obviously bad lower/background samples:

```ts
if (
  yNorm > 1.00 &&
  profileConf < 0.20 &&
  value < q75
) reject sample
```

That means:

* deep below the current local center
* poor tooth-band evidence
* not one of the brighter local slab samples

Those are the samples currently polluting the pano.

### Strong downweight

If not bad enough for rejection, but still suspicious:

```ts
if (
  yNorm > 0.55 &&
  profileConf < 0.35 &&
  value < q75
) w *= 0.25
```

### Leave mostly intact

Do **not** aggressively suppress:

* samples at or above local center
* bright samples even slightly below center
* anything with decent profile confidence

That is how you avoid killing real teeth and roots.

---

## 4. How to preserve true root / cortical signal

This is the main place people overdo the gate and destroy anatomy.

Use these rules:

### A. Never hard-reject based on “below center” alone

Roots often extend below the center estimate.
So **depth below center is not enough**.

### B. Bright rescue wins over lower suppression

If a sample is in the upper quartile of the local slab (`value >= q75`), keep a meaningful weight floor:

```ts
wAnatomy = max(wAnatomy, 0.50)
```

This preserves:

* root cortex
* lamina-like bright boundaries
* strong enamel/dentin interfaces that dip low locally

### C. Keep the central band untouched

For `yNorm <= 0.20`, do not suppress at all except normal slab Gaussian.

That avoids flattening the main tooth body.

### D. Use hard reject only for deep-lower + low-confidence + non-bright

That combination is the contamination, not the anatomy.

---

## 5. Debug counters to log

Add counters that tell you whether the gate is doing useful work and whether it is too aggressive.

### Sample counters

* `gateInBoundsSampleCount`
* `gateRejectedSampleCount`
* `gateDownweightedSampleCount`
* `gateBrightRescueCount`
* `gateBelowCenterSampleCount`
* `gateBelowCenterRejectedCount`
* `gateBelowCenterRescuedCount`

### Weight diagnostics

* `gateMeanAnatomyWeight`
* `gateMeanFinalWeight`
* `gateMeanWeightBelowCenter`
* `gateMeanWeightAboveCenter`

### Decision-shape diagnostics

* `gateDeepLowerLowConfCount`
* `gateDeepLowerBrightRescueCount`
* `gateNoSurvivorColumnCount`
  This one is important. If it rises, the gate is too aggressive.

### Optional percentile diagnostics

* `gateSlabQ75Mean`
* `gateRejectedBelowQ75Count`

---

## 6. What log values should improve first if this works

### Should improve first

These are the primary acceptance metrics:

* `lowerBandBrightFraction` should drop first and most

  * from ~`0.75` toward something much lower
* `lowerBandP50` should drop

  * from ~`16` toward low single digits or clearly lower than current
* `qualityScore` should improve materially

### Likely secondary movement

* `detailBandHorizontalEdgeMean` should drop somewhat

  * because polluted background texture is contributing a lot of false horizontal edge energy right now
* `detailBandVerticalEdgeMean` may stay similar at first or rise slightly

### Failure sign the gate is too aggressive

If this happens, you overdid it:

* `lowerBandBrightFraction` improves, **but**
* teeth or roots start disappearing
* `detailBandVerticalEdgeMean` drops
* `gateNoSurvivorColumnCount` rises
* anterior/posterior root thickness visibly collapses

That would mean the reject rule is too hard or the rescue floor is too low.

---

## Minimal pseudocode sketch

```ts
collect inBounds samples: { value, slabOffset, sampleRow, profileConf }

compute q75 over sample values

for each sample:
  wDepth = gaussian(slabOffset)

  yNorm = (sampleRow - localCenterRow) / effectiveVerticalHalfHeight

  wLower =
    yNorm <= 0
      ? 1
      : 1 - 0.85 * smoothstep(0.20, 0.90, yNorm)

  wProfile = 0.20 + 0.80 * profileConf
  wAnatomy = wLower * wProfile

  if (yNorm > 0.20 && value >= q75) {
    wAnatomy = max(wAnatomy, 0.50)
    brightRescueCount++
  }

  if (yNorm > 1.00 && profileConf < 0.20 && value < q75) {
    reject
    continue
  }

  if (yNorm > 0.55 && profileConf < 0.35 && value < q75) {
    wAnatomy *= 0.25
    downweightedCount++
  }

  finalWeight = wDepth * wAnatomy
  keep sample with finalWeight
```

That is the smallest useful one-pass gate.

---

# 2) Minimal safe transported-frame patch in `buildRMFFrames()` in `cprMath.ts`

## Goal

Stop re-anchoring every frame to one global vertical.
Transport the slab frame from one point to the next with **minimal rotation**.

This is the smallest real fix for the anterior shear.

---

## 1. Exact algorithm step-by-step

For each spline sample `i`, you already have tangent `T[i]`.

### Step 1: initialize frame 0

Use the global vertical only once to seed the first frame.

Let:

```ts
T0 = normalize(T[0])
Utarget0 = normalize(verticalDir - dot(verticalDir, T0) * T0)
```

If that degenerates, use a fallback axis least aligned with `T0`.

Then:

```ts
N0 = normalize(cross(T0, Utarget0))
U0 = normalize(cross(N0, T0))
```

Use the handedness that matches your current conventions.

### Step 2: for each next frame, transport previous normal

For `i > 0`:

```ts
Tprev = normalize(T[i - 1])
Tcur  = normalize(T[i])
Nprev = N[i - 1]
```

Compute the minimal rotation from `Tprev` to `Tcur`:

```ts
axis = cross(Tprev, Tcur)
s = length(axis)
c = clamp(dot(Tprev, Tcur), -1, 1)
```

If `s > eps`, rotate `Nprev` around `axis / s` by angle `atan2(s, c)` using Rodrigues rotation.

If `s <= eps`, just carry `Nprev` forward.

That gives:

```ts
Ntransport
```

### Step 3: re-orthogonalize to current tangent

Project out any tangent component:

```ts
Ncur = normalize(Ntransport - dot(Ntransport, Tcur) * Tcur)
```

If degenerate, fall back once to a seed from global vertical for that frame only.

### Step 4: rebuild vertical axis from transported normal

```ts
Ucur = normalize(cross(Ncur, Tcur))
```

Again, match your handedness.

### Step 5: optional tiny soft bias to global vertical

Only after transport, apply a **small** drift correction, not a re-anchoring.

Compute:

```ts
Utarget = normalize(verticalDir - dot(verticalDir, Tcur) * Tcur)
```

Then:

```ts
alpha = 0.05
Ubiased = normalize((1 - alpha) * Ucur + alpha * Utarget)
Nbiased = normalize(cross(Tcur, Ubiased))
```

Store `Nbiased`, `Ubiased`.

This keeps long-run roll drift under control without reintroducing the old shear bug.

---

## 2. How to initialize the first frame

Use global vertical **only at frame 0**:

```ts
U0 = orthogonalized global vertical against T0
N0 = cross(T0, U0)
```

That is correct and safe.

The bug is not using global vertical at all.
The bug is using it to **rebuild every frame**.

---

## 3. How to transport `N_slab` from frame i-1 to i

Use minimal rotation between neighboring tangents.

In words:

* compute the rotation that aligns `T[i-1]` to `T[i]`
* apply that same rotation to `N[i-1]`
* then re-orthogonalize against `T[i]`

That is the actual transported frame.

Do **not** compute:

```ts
N_slab = normalize(cross(T, normalizedVerticalDir), fallback)
```

per frame anymore.

That is exactly what is shearing the anterior.

---

## 4. Whether to keep any soft bias to global vertical

Yes, but tiny.

Recommended:

```ts
alpha = 0.03 to 0.07
```

I would start at:

```ts
alpha = 0.05
```

Rules:

* apply it only **after** transport
* bias the vertical axis `U`, not rebuild `N` from scratch from global vertical
* keep it constant and small for the first patch

This keeps posterior regression risk low.

---

## 5. Diagnostics to log

You want diagnostics that prove the frame is transported and not flipping/twisting.

### Geometry diagnostics

* `rmfTurnAngleMean`
* `rmfTurnAngleMax`
* `rmfTransportedNormalDeltaMean`
* `rmfTransportedNormalDeltaMax`
* `rmfVerticalBiasCorrectionMeanDeg`
* `rmfVerticalBiasCorrectionMaxDeg`

### Stability diagnostics

* `rmfNormalFlipCount`

  * count if `dot(N[i], N[i-1]) < 0`
* `rmfOrthoErrorMax`

  * max of `abs(dot(T,N))`, `abs(dot(T,U))`, `abs(dot(N,U))`
* `rmfFallbackCount`

### Optional regional diagnostic

Log the same deltas in the anterior third only:

* `rmfAnteriorNormalDeltaMean`
* `rmfAnteriorBiasCorrectionMeanDeg`

That is where the bug is most obvious.

---

## 6. What existing image/log metrics should improve if this fixes the anterior

### Should improve

* `detailBandVerticalEdgeMean` should rise
* horizontal/vertical edge imbalance should shrink
* `qualityScore` should improve
* visible anterior teeth should stop looking stretched/sheared/alien

### Should not move much

* `lowerBandBrightFraction`
* `lowerBandP50`

Those are contamination metrics, not frame metrics.

So if the frame patch is working, the image should look better in the anterior even if lower-band pollution is still partly present.

---

# 3) Minimal curvature-aware replacement for local-center candidate scoring in `cprWorker.ts`

## Goal

Let the center tracker breathe in curved anterior regions without destabilizing straighter posterior regions.

The safest change is:

* compute a curvature factor from tangent turn angle
* reduce the priors only where curvature is high
* widen search only modestly with the same factor

---

## 1. Exact formula using tangent delta / turn angle

For column `i`, compute:

```ts
turnAngleRad = acos(clamp(dot(T[i - 1], T[i]), -1, 1))
turnAngleDeg = turnAngleRad * 180 / Math.PI
```

Then convert to a curvature-relaxation factor:

```ts
k = smoothstep(2.0, 8.0, turnAngleDeg)
```

So:

* `k = 0` in nearly straight zones
* `k -> 1` in higher-curvature anterior zones

Then keep your existing evidence term, but replace fixed penalties with:

```ts
wGlobal = 26 * (1 - 0.70 * k)
wPrev   = 18 * (1 - 0.45 * k)
```

So at high curvature:

* global-center penalty drops from `26` to about `7.8`
* previous-column penalty drops from `18` to about `9.9`

That is the right asymmetry: relax global more than continuity.

Then score as:

```ts
score(candidate) =
  evidenceScore(candidate)
  - wGlobal * abs(candidateCenter - globalCenter)
  - wPrev   * abs(candidateCenter - prevCenter)
```

If your current code uses squared distances, keep squared distances.
The important patch is changing the **weights**, not the distance form.

---

## 2. How much to reduce each penalty in high-curvature zones

Recommended first patch:

### Global-center penalty

Reduce by **70% at full curvature factor**

```ts
26 -> 7.8
```

### Previous-column penalty

Reduce by **45% at full curvature factor**

```ts
18 -> 9.9
```

Why this split:

* global prior is the more harmful one in the anterior
* previous-column continuity is still useful and should not be weakened as much

This is the smallest safe bias change.

---

## 3. Should search half-range and maxDeviation widen with the same factor?

Yes, but only modestly.

Use the same `k`, with smaller multipliers:

```ts
searchHalfRange = baseSearchHalfRange * (1 + 0.30 * k)
maxDeviation    = baseMaxDeviation    * (1 + 0.25 * k)
```

So in straight posterior zones, nothing changes.
In the anterior, the tracker gets a bit more room.

Do **not** widen aggressively on the first patch.
The main fix is relaxed penalties, not a huge search explosion.

---

## 4. What log fields should move if the change is working

### Add these if you can

* `centerTurnAngleMeanDeg`
* `centerTurnAngleMaxDeg`
* `centerCurvatureRelaxationMean`
* `centerCurvatureRelaxationMax`
* `centerChosenAbsGlobalDeviationMean`
* `centerChosenAbsGlobalDeviationAnteriorMean`
* `centerChosenPrevDeltaMean`
* `centerChosenPrevDeltaAnteriorMean`
* `centerBoundaryHitCount`
* `centerMaxDeviationClampCount`

### Existing metrics likely to improve

* `detailBandVerticalEdgeMean` should rise
* `qualityScore` should improve
* anterior teeth should look less flattened/sheared
* `lowerBandBrightFraction` may improve slightly if the band stops drifting low, but that is secondary

### Good sign

* chosen center deviates more from the global center in the anterior
* boundary/clamp hits go down
* vertical edge strength goes up

### Bad sign

* posterior starts wobbling
* center jitter rises everywhere
* boundary hits increase instead of decrease

If that happens, the search widening is too much or the continuity penalty was relaxed too hard.

---

# Recommended implementation order

1. **One-pass anatomy-aware gating in `sampleReducedPoint()`**
2. **Transported frame in `buildRMFFrames()`**
3. **Curvature-aware center-prior relaxation**
4. Only then revisit spline sampling if residual softness remains

That order is still the right one.

