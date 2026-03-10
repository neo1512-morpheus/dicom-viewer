Below is the reconstruction change I would make.

## Core shift

Stop treating panorama generation as:

`curved slab sample -> wide depth reduction -> try to reject junk afterward`

and change it to:

`curved volume sample -> estimate tooth support surface first -> flatten around that surface -> do only narrow local reduction`

That is the key difference between the current CPR-style strip and a more proper CPU virtual panoramic reconstruction.

---

## 1. Estimate a patient-specific focal trough / support surface around the arch

### A. Build a curved reformat volume, not just a reduced strip

Using the existing arch curve and current CPR sampling machinery, resample the CBCT into a curved coordinate system:

* `c`: pano column along arch centerline
* `d`: depth across the arch normal (buccal↔lingual / inside↔outside of trough)
* `r`: pano row (superior↔inferior)

Define:

`R[c,d,r] = V( C(c) + u(d) * N(c) + y(r) * Z )`

Where:

* `C(c)` = 3D point on arch
* `N(c)` = in-plane arch normal
* `Z` = volume superior/inferior axis
* `u(d)` = depth offset in mm
* `y(r)` = row height in mm

This is still CPU-only trilinear interpolation.
It is the same geometry you already have, but you keep the depth dimension instead of reducing it immediately.

### B. Score which depth actually corresponds to the tooth-bearing zone

For each `(c,d)`, score how “tooth-like and useful” that depth is.

Use a tooth band and a lower-suppression band in row space:

* `B_tooth`: rows covering crowns + roots
* `B_low`: rows below likely root/apex region

Then compute:

`Hard(c,d) = mean over r in B_tooth of hardResponse(R[c,d,r])`

`Grad(c,d) = mean over r in B_tooth of gradientResponse(R[c,d,r])`

`Low(c,d) = mean over r in B_low of hardResponse(R[c,d,r])`

A good first scoring function is:

`Score(c,d) = w1 * Hard(c,d) + w2 * Grad(c,d) - w3 * Low(c,d)`

Where:

* `hardResponse(I)` should rise with dentin/enamel/bone-like signal
* `gradientResponse(I)` should rise where tooth/bone boundaries are crisp
* `Low(c,d)` penalizes depth planes that mostly explain inferior junk / mandible background instead of teeth

### C. Make the thresholds adaptive to the scan

Do not hardcode global HU-like constants unless your volume normalization is already stable.

Use curved volume statistics:

* `T_soft = percentile(R in B_tooth, 60)`
* `T_hard = percentile(R in B_tooth, 85)`

Then:

`hardResponse(I) = clamp((I - T_soft) / (T_hard - T_soft), 0, 1)`

For gradient:

`gradientResponse(I[c,d,r]) = clamp( abs(R[c,d+1,r] - R[c,d-1,r]) + 0.5 * abs(R[c,d,r+1] - R[c,d,r-1]), 0, G_cap ) / G_cap`

This keeps the method scan-adaptive.

### D. Select one smooth depth path across columns

For each column, keep the top `K` depth candidates by `Score(c,d)`.
Then choose a smooth path `D[c]` across columns with dynamic programming.

Use energy:

`E = sum_c ( -Score(c, D[c]) ) + λ1 * sum_c |D[c] - D[c-1]| + λ2 * sum_c |D[c] - 2D[c-1] + D[c-2]|`

This avoids:

* per-column jitter
* sudden jumps into background
* noisy local maxima

This is the correct place to use dynamic programming.

### E. Support surface, not only one depth per column

Minimal useful version:

`S[c,r] = D[c] + A[c] * (r - r_mid)`

Where `A[c]` is a small per-column vertical tilt term.

Estimate `A[c]` by splitting the tooth band into upper and lower halves:

* `D_top[c] = best depth using upper tooth rows`
* `D_bot[c] = best depth using lower tooth rows`

Then:

`A[c] = clamp( (D_bot[c] - D_top[c]) / (r_bot - r_top), -A_max, A_max )`

This already handles tooth/root inclination much better than a pure columnwise constant depth.

### F. Optional later upgrade: full 2D surface

If residual distortion remains after the minimal version, move from:

`S[c,r] = D[c] + linear tilt`

to a full coarse 2D support surface solved on anchor rows:

`S[c,k]` on 6–10 vertical anchors, with smoothness in both `c` and `k`, then bilinear interpolate to all rows.

But I would not start there.

---

## 2. Choose the correct depth per pano location before reduction

This is the most important change.

At the moment, the pipeline reduces a wide depth slab and tries to filter afterward.
That is backwards.

Instead:

1. compute `S[c,r]` first
2. only then reduce a narrow local neighborhood around that selected depth

For each output pixel `(c,r)`, define local flattened coordinates:

`δ = d - S[c,r]`

Then only samples with small `|δ|` are allowed to contribute.

That is what turns this into a virtual pano instead of a curved slab average.

---

## 3. Preserve tooth/root visibility while suppressing lower/background anatomy

### A. Flatten around the selected support surface

Create a locally aligned depth stack:

`F[c,r,δ] = R[c, S[c,r] + δ, r]`

Now the tooth-bearing structure is centered around `δ = 0`, while off-trough clutter is displaced away from center.

### B. Use a narrow row-dependent trough width

Use a half-width `τ(r)` in mm, not a fixed full slab.

Recommended behavior:

* crown/root band: wider
* upper soft tissue band: narrower
* lower/inferior band: much narrower

For example:

* `τ_tooth ≈ 3–4 mm`
* `τ_upper ≈ 1.5–2 mm`
* `τ_lower ≈ 0.75–1.25 mm`

So:

`eligible if |δ| <= τ(r)`

### C. Make the reducer root-preserving, not mean-like

Do not use a plain mean.
That is exactly what turns the background into static and blurs roots.

Use a weighted upper-tail reducer on eligible samples:

1. collect eligible `F[c,r,δ_j]`
2. weight by closeness to center:

`w_j = exp( -0.5 * (δ_j / σ(r))^2 )`

3. sort by intensity
4. keep only the top `q` weighted mass, e.g. top `30–40%`
5. return weighted mean of that retained set

This gives you:

* more root retention than mean
* much less sparkle than max
* less overprojection than MIP

If you want the simplest first implementation, use a weighted 75th–85th percentile instead.

### D. Add a lower-band aggressiveness increase

For rows below the inferred apex zone:

* shrink `τ(r)`
* raise the minimum hard-response requirement slightly

So the lower band becomes harder to enter even if some noisy bright voxels exist.

### E. Fallback behavior

If no sample is eligible:

* use `F[c,r,0]` directly

That is better than reverting to a wide-slab mean.

---

## 4. Use per-column depth selection, surface flattening, dynamic programming, or something else?

Use all three, in this order:

### Recommended architecture

1. **Per-column candidate scoring**
2. **Dynamic-programming path selection across columns**
3. **Support surface construction**

   * start with `D[c] + tilt`
4. **Surface flattening**
5. **Narrow local reduction around the surface**

### What not to do

* do not keep tuning the current wide slab mean
* do not rely on post-hoc eligibility alone
* do not start with full segmentation
* do not use GAN/image-generation cleanup

### Why this is the right minimal architecture

It fixes the actual failure mode:

The current method mixes anatomy from the wrong depth first, and only then tries to reject it.
The new method selects the right depth first, so the reducer has much less junk to fight.

---

## 5. Minimal implementation path in this codebase

Given your current `cprWorker.ts` setup, I would implement this in the smallest possible sequence.

### Phase 1 — new mode beside current CPR

Add a new reconstruction mode, something like:

`mode: 'virtualPanoCpu'`

Do not mutate the current CPR path first.
Keep old output for A/B testing.

### Phase 2 — reuse current arch and sampler

Reuse:

* current arch centerline
* current tangent/normal computation
* current trilinear volume sampling

Add a function that builds the curved reformat volume:

`buildCurvedVolume(volume, arch, panoWidth, panoHeight, depthSamples, depthRangeMm) -> Float32Array R`

### Phase 3 — depth scoring and DP path

Add:

* `computeDepthScores(R, rowBands, thresholds) -> scores[c,d]`
* `selectDepthPathDP(scores, smoothnessParams) -> D[c]`
* `estimateColumnTilt(R, D, rowBands) -> A[c]`

### Phase 4 — flatten and reduce

Add:

* `renderFlattenedVirtualPano(R, D, A, rowPolicy, reducerPolicy) -> pano`

Where `rowPolicy` contains:

* `τ(r)`
* `σ(r)`
* lower-band penalties

and `reducerPolicy` is:

* weighted upper-mean or weighted percentile

### Phase 5 — only if needed later

If distortion remains:

* replace `A[c]` linear tilt with a coarse 2D surface `S[c,k]`

But do not start with that.

---

## 6. What new logs should improve if it works

Your current logs should move in the right direction, but some new logs are more important.

### Existing logs that should improve

These should all go down materially:

* `lowerBandBrightFraction`
* `lowerBandP50`
* `twoPassEligibilityDiagnostics.lowerBandEligibleFraction`

If the support surface is working, the lower band should stop looking like a retained bright fog.

### New logs to add

#### A. Surface/path quality

* `supportSurface.depthMinMm`
* `supportSurface.depthMaxMm`
* `supportSurface.depthStdMm`
* `supportSurface.pathJumpP95Mm`
* `supportSurface.curvatureP95`
* `supportSurface.candidateMarginMean`

  * mean(bestScore - secondBestScore)

Good behavior:

* low path jumps
* nontrivial but smooth depth variation
* positive candidate margin

#### B. Flattening quality

* `flattened.centerMassFraction`

  * fraction of retained energy within `|δ| < 1.5 mm`
* `flattened.offTroughEnergyRatio`

  * energy outside trough / energy inside trough
* `flattened.emptyFallbackFraction`

Good behavior:

* center mass fraction rises
* off-trough ratio falls
* empty fallback fraction stays low

#### C. Anatomy/readability quality

* `toothBandEdgeEnergy`

  * mean local gradient magnitude in tooth band
* `rootBandContrast`

  * median(tooth/root rows) - median(lower rows)
* `columnSharpnessP50`

  * vertical edge energy by column
* `lowerSuppressionRatio`

  * median(lower band) / median(tooth band)

Good behavior:

* tooth/root edge energy rises
* lower suppression ratio falls
* roots stay visible without background grain exploding

#### D. Surface engagement replaces old curvature prior story

Right now you have:

* `curvatureFactorMax = 0`

That should become much less important.
The new method should log actual data-driven surface motion, e.g.:

* `supportSurface.depthRangeMm`
* `supportSurface.tiltRangeMmPerRow`

Those should be nonzero on real scans even when old curvature priors never engaged.

---

## Recommended first implementation target

If I had to choose the smallest version that is still worth building, it would be this:

1. build `R[c,d,r]`
2. compute `Score(c,d)` from tooth-band hard response + gradient - lower-band penalty
3. pick smooth `D[c]` with DP
4. estimate simple per-column tilt `A[c]`
5. flatten around `S[c,r] = D[c] + A[c](r-r_mid)`
6. render with narrow row-dependent trough width and weighted upper-tail reducer

That is the minimal jump from “curved slab reduction” to “CPU virtual panoramic reconstruction”.

It is still fully CPU-only, implementation-feasible in your worker, and much more aligned with how a readable panoramic projection should be formed.

The attached clean pano is exactly the kind of target behavior this should move toward: teeth and roots remain coherent, while inferior/background anatomy stops contaminating the whole strip.

If you want, I can turn this into a code-agent-ready implementation spec for `cprWorker.ts` with concrete buffer shapes, function signatures, and pass order.
