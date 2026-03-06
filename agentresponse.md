Yes — the next change should be a **true two-pass worker**, but specifically a **per-slab-offset two-pass reconstruction**, not just “render one pano, make one row mask, render again.”

That is the minimal deeper change that breaks the current math failure.

The core issue now is exactly what you diagnosed: the current gate is mostly a **row-level multiplier shared by nearly all slab samples at a pixel**, so the weighted mean largely normalizes it away. To really clean the pano, the second pass must decide **which individual slab samples are eligible** before reduction.

---

## 1) Minimal deeper architecture change

### Recommended change

Implement a **two-pass worker with a small slab-depth stack**:

* **Pass 1:** render and store the per-depth sample planes
  `I[k][col,row]`, where `k` is the slab sample index / offset
* derive a **sample-eligibility mask**
  `M[k][col,row] ∈ {0,1}` or soft `[0,1]`
* **Pass 2:** rebuild the pano using only eligible slab samples before reduction

Because your slab sample count is small (`~3–6`), this is still CPU-friendly and memory-friendly enough.

### Why this is the right minimal change

It directly fixes the current failure mode:

* current row-level gating applies almost the same multiplier to all slab samples at a pixel
* the reducer then renormalizes those weights
* so contamination survives

A per-depth stack lets you say:

* this depth sample belongs to anatomy
* this one is lower/background contamination
* reject one, keep the other

That is the first change that is truly **sample-specific**.

### What not to do

Do not spend another cycle mainly retuning:

* winsorization
* bright rescue thresholds
* post-hoc denoise weighting
* VOI/global contrast

Those are secondary now.

---

## 2) Exactly how the first pass should estimate tooth-band / root-band / lower-background

The first pass should produce two things:

1. a provisional composite pano `P[col,row]`
2. the raw per-depth planes `I[k][col,row]`

Then estimate three structures:

* **tooth-band seeds**
* **root-band connected support**
* **lower-background exclusion**

### Pass 1A: render raw per-depth planes

For each pano pixel `(c,r)` and each slab sample index `k`:

* sample the volume at that slab offset
* store raw intensity in `I[k][c,r]`

Also compute a provisional composite:

* `P[c,r] = current reducer over all in-bounds k`
* this provisional pano is only for structure estimation

### Pass 1B: compute center-relative coordinates

Using the already selected local center / vertical profile logic, compute for each column:

* `centerRow[c]`
* `halfHeight[c]`

Then for each row:

```text
yNorm(c,r) = (r - centerRow[c]) / max(halfHeight[c], 1)
```

Use that only for structure estimation in pass 1, not as the final gate itself.

### Pass 1C: estimate tooth-band seed map

Create a **seed mask** `S[c,r]` from the provisional pano `P`:

A pixel is a tooth-band seed if all are true:

* `-0.30 <= yNorm <= 0.45`
* `profileConf(c,r) >= 0.55`
* `P[c,r] >= Pfloor[c] + 0.30 * (Ppeak[c] - Pfloor[c])`

Where:

* `Ppeak[c] = P[c, centerRow[c]]` or local peak near center
* `Pfloor[c] = lower quartile / local floor of the column profile window`
* `profileConf(c,r) = clamp((P[c,r] - Pfloor[c]) / max(Ppeak[c]-Pfloor[c], eps), 0, 1)`

This seed map is the “trusted tooth/root origin.”

### Pass 1D: build per-depth foreground maps

For each depth plane `k`, define a foreground candidate map `F[k][c,r]`.

Use a **lower threshold for growth** than for seed:

* seed threshold:

  * `Tseed[c] = Pfloor[c] + 0.35 * (Ppeak[c] - Pfloor[c])`
* growth threshold:

  * `Tgrow[c] = Pfloor[c] + 0.18 * (Ppeak[c] - Pfloor[c])`

Then:

* `SeedK[k][c,r] = S[c,r] && I[k][c,r] >= Tseed[c]`
* `F[k][c,r] = I[k][c,r] >= Tgrow[c]`

This is a hysteresis-style design:

* seed from strong central anatomy
* allow weaker downward continuation for roots

### Pass 1E: grow connected root-band support per depth plane

For each `k`, run a small 2D connected growth / flood fill on that plane:

Start from `SeedK[k]`, and grow through `F[k]` with constraints:

* connectivity: 4-connected is enough
* allowed vertical range:

  * up to around `yNorm <= 1.15`
* allowed upward range:

  * maybe `yNorm >= -0.45`
* optional small horizontal continuity:

  * allow neighbor columns `c±1`

Result:

* `M[k][c,r] = 1` if that per-depth pixel is part of a connected component originating from the tooth-band seed
* else `0`

This is the critical step.

### Why this works

True teeth and roots will form components that:

* start in the tooth band
* continue downward coherently

TV-static lower contamination may be bright, but it usually does **not** form a component that is connected back to the seed band in the same depth plane.

### Pass 1F: define lower-background

You do not need a separate complex classifier.

Define lower background implicitly as:

* rows in the lower region (`yNorm > 0.50`)
* that are **not connected** to tooth-band seeds in `M[k]`

So lower/background is basically:

```text
B[k][c,r] = (yNorm > 0.50) && (M[k][c,r] == 0)
```

That is enough for the second pass.

---

## 3) Exactly how the second pass should use that information at slab-sample level before reduction

Second pass should be simple:

For each pano pixel `(c,r)`:

* look at each slab sample index `k`
* if `M[k][c,r] == 0`, reject that sample before reduction
* if `M[k][c,r] == 1`, keep it with normal slab-depth weight

### Final per-sample rule

For each sample `k`:

```text
baseWeight = GaussianDepthWeight(k)

if M[k][c,r] == 0:
    reject
else:
    keep with weight = baseWeight
```

Optional soft version:

```text
weight = baseWeight * (0.75 + 0.25 * supportScore[k][c,r])
```

But for the first real reconstruction fix, **hard eligibility is better** than another soft attenuator.

### Safety fallback

If no sample is eligible at `(c,r)`:

Fallback in this order:

1. keep the **central-most in-bounds slab sample**
2. else use the old ungated reducer result for that pixel

This avoids holes.

### Why this fixes the current failure

Now the reducer is no longer averaging “all slab samples with slightly different weights.”

It is averaging a **subset** of slab samples chosen by anatomy support.

That is the first mechanism that can actually make the background disappear.

---

## 4) How to preserve true root/cortical signal while excluding contaminated lower anatomy

This is where the design needs to be careful.

The right preservation rule is:

### Preserve roots by connectivity, not by raw brightness alone

The current bright rescue is too blunt because bright contamination also gets rescued.

Instead:

* preserve a lower sample only if it is **connected to a tooth-band seed**
* not merely because it is bright

That is why the connected-component design matters.

### Use hysteresis thresholds

This is essential.

* **High threshold** to create tooth-band seeds
* **Lower threshold** to grow roots downward

So roots can remain visible even when weaker than crowns.

### Cap downward growth

Do not allow root-band growth indefinitely downward.

Recommended limit:

* `yNorm <= 1.10–1.20`

That keeps true roots while excluding deep lower anatomy.

### Optional rescue for thin bright cortices

If you want one extra safeguard:

If a sample is just outside the component but:

* `0.45 < yNorm < 1.0`
* and it is a local vertical maximum in its depth plane
* and adjacent rows in the same depth plane show connected support

then allow it as a one-step border dilation of `M`.

But do not use brightness alone anymore as the main rescue rule.

### Summary of preservation principle

Keep lower anatomy when it is:

* connected upward to tooth-band seeds
* inside the root-band depth window
* supported by the same depth plane’s continuity

Reject lower bright anatomy when it is:

* isolated
* deep
* unconnected to tooth-band support

That is the correct anatomy-aware separation.

---

## 5) Should the reducer itself change again?

**No major reducer redesign should be the next step.**

The main fix is now **sample eligibility before reduction**.

Keep the reducer mostly as-is:

* `computeWinsorizedWeightedMean()` over the **eligible** samples only

Maybe add one small safety rule:

* if eligible sample count is 1, return that sample directly
* if eligible sample count is 2+, use the current robust mean
* if eligible sample count is 0, use fallback

That is enough.

### Why not retune the reducer again

Your latest logs already show the problem is upstream:

* the gate changed total weight a lot
* but contamination remained severe

That means the reducer is not the bottleneck anymore.
The bottleneck is that contaminated samples are still being allowed to participate.

So:

* **eligibility is the main fix**
* reducer tuning is now secondary

---

## 6) Should the curvature-aware center prior thresholds be revised?

**Yes, but this is secondary to the reconstruction fix.**

Your current logs:

* `turnAngleMaxDeg ≈ 1.306`
* onset was `2°`
* so curvature logic never engaged

That means the threshold is too high for this scan.

### Recommended revision

After the reconstruction change, lower the curvature onset to something like:

* `smoothstep(0.25°, 1.0°)`
  or slightly more conservative:
* `smoothstep(0.35°, 1.2°)`

But smooth the turn-angle over neighboring columns first, or you risk wobble.

### Why revise it

If your scan’s true curvature only reaches ~1.3°, a 2° onset guarantees no effect.

So yes, that threshold should be revised.

### But priority

Do it **after** or alongside the two-pass eligibility change, not instead of it.

Because your contamination numbers are too severe to be explained by center priors alone.

---

## 7) Which exact log values should move if the deeper change is working

These should move first and clearly.

### Main acceptance metrics

* `lowerBandBrightFraction`

  * should drop **a lot**
  * from `0.8636` toward something far lower
* `lowerBandP50`

  * should collapse sharply
  * from `136.13` toward something much closer to the true background floor
* `selectedQualityScore`

  * should improve materially
  * current `-32.81` should move upward

### Edge metrics

* `detailBandHorizontalEdgeMean`

  * should decrease
  * because a lot of its current strength is just polluted lower/static texture
* `detailBandVerticalEdgeMean`

  * should hold or improve
  * if roots and tooth walls are preserved properly

The most important pattern is:

* **horizontal clutter energy down**
* **vertical tooth structure preserved or up**

### New logs you should add

For the two-pass design, log these:

#### Pass 1 structure logs

* `pass1SeedPixelCount`
* `pass1RootSupportPixelCount`
* `pass1LowerRejectedPixelCount`
* `pass1ConnectedComponentCountMean`
* `pass1ConnectedRootDepthCoverageMean`

#### Pass 2 eligibility logs

* `eligibleSampleCount`
* `rejectedSampleCount`
* `eligibleSampleFraction`
* `fallbackNoEligiblePixelCount`
* `rootConnectedRescueCount`

#### Crucial sanity metric

* `lowerBandEligibleFraction`

If the fix is working, **lower-band eligible fraction should drop hard**.

### What failure would look like

If the change is too aggressive:

* `lowerBandBrightFraction` drops
* but `detailBandVerticalEdgeMean` also drops
* roots become thin or broken
* `fallbackNoEligiblePixelCount` becomes high

If that happens:

* growth threshold is too strict
* downward range is too short
* or connected growth is too brittle

---

# Bottom line

The next deeper CPU-only fix should be:

## A true two-pass reconstruction with per-slab-depth eligibility masks

Not:

* more row-level gating
* more winsor tweaks
* more post-hoc denoise weighting

### The right structure is:

1. **Pass 1:** render per-depth planes `I[k]` and provisional pano `P`
2. seed tooth band from the provisional pano
3. grow connected root support per depth plane
4. define lower background as lower, unconnected samples
5. **Pass 2:** reduce only eligible slab samples

That is the minimal architecture change that can actually turn the current noisy lower “TV static” into real background while keeping teeth and roots bright.

If you want, I can now turn this into a concrete patch plan for:

* `cprWorker.ts` pass-1 buffers
* mask construction loops
* pass-2 eligibility logic
* exact fallback behavior and counters
