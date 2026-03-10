**A. Executive Verdict**

The legacy CPR pipeline has hit an unbreakable architectural ceiling. The only viable pivot that satisfies every hard constraint (CPU-only web-worker integration, no synthesis/postproc-only, real anatomy preservation, diagnosable output) is a **patient-specific focal trough defined by dynamic-programming (DP) / shortest-path optimization on an unrolled cylindrical cost map**, followed by narrow local reduction around that surface. This is the clear #1 recommendation — decisively better than every approach you have already exhausted. It directly implements the model shift you suspected (“choose support surface first, then reconstruct locally”) and is grounded in exactly the techniques validated in real CBCT-to-virtual-pano literature (patient-specific troughs, multiple adaptive curves, dynamic focal-plane concepts) while staying classical and lightweight.

**B. Ranked Architecture Options**

1. **DP / shortest-path focal-trough surface in unrolled cylindrical volume** (recommended — implement this).  
   Works because it produces a smooth, data-driven 2-D surface d(s) that follows actual tooth positions column-by-column with global smoothness enforcement. Narrow local reducer around it excludes everything else. Matches 2022–2024 methods that improved root visibility and reduced blur by moving beyond fixed slabs.

2. **Multi-arch-curve CPR with isometric offsets and layer fusion** (viable fallback, e.g. 2022 fast-automatic method).  
   Works as a modest extension of your current code (generate several parallel curves, reconstruct thin MPRs, fuse). Better root capture than single slab, CPU ~3 s. Still fundamentally slab-based, so will retain some residual speckle/contamination in malformed arches.

3. **Simulated-projection tomosynthesis with elliptical dynamic centers + focal-trough ray aggregation** (2024 arXiv style).  
   Technically elegant (true virtual acquisition geometry). Would give the cleanest possible output. But requires per-pixel ray-tracing through the volume — borderline with your “no GPU renderer” rule, higher integration cost, risk of re-introducing the white-streak blowout you already saw.

4. **Spline-fitted patient-specific trough from arch contour only** (average-trough style).  
   Does not work well enough here — purely geometric, ignores voxel intensities, unstable in missing-teeth or metal cases. You already effectively tried variants of this.

**C. Why Current CPR Saturated**

Your current “retry-mean-sharp-narrow” path is a **fixed-geometry curved-slab projection**: one arch curve → fixed radial offset + fixed thickness → global mean/MIP reducer across that entire slab.  

Architecturally this is doomed once tooth positions deviate from the nominal trough (which they always do — anterior protrusion, root angulation, occlusion, malpositioning). Every voxel in the output is a mixture of correct tooth + wrong anatomy at the same (s,z) but wrong d. The lower-band bright fraction and tooth-band P10/P90 contamination you logged are the inevitable mathematical consequence. No amount of scoring, VOI, conditional masking, or one-/two-pass eligibility can un-mix voxels that were already averaged before selection. Anterior shearing is pure projection geometry — rays are not locally normal to the true tooth long axis. You have reached the practical quality ceiling of any fixed-slab CPR; further tuning is wasted cycles.

**D. Recommended Model in Detail**

**DP focal-trough surface + narrow local reducer**.  

- Unroll the volume into cylindrical coordinates around the existing arch curve (s = arc-length, d = radial depth, z = height).  
- Build a 2-D cost map C[s][d] (tiny: ~400 × 120) where cost = mean-over-z (thresholded intensity or gradient magnitude) — high cost where teeth/bone actually live.  
- Run 1-D DP (standard path-finding with smoothness penalty λ·|Δd|² or limited transition window) to obtain the globally optimal smooth surface d_opt[s].  
- Reconstruct each output pixel as local mean (or median) over a narrow ±1.5–3 mm window around d_opt[s] at that (s,z).  

This is exactly “depth selection before reduction”. The surface is patient-specific and data-driven (not fixed slab, not simple CoM). Lower-band contamination disappears because the optimizer never chooses d values outside the tooth band. Anterior shearing disappears because the local kernel follows the true tooth orientation. Anatomy is 100 % real voxels — no synthesis.

**E. Implementation Strategy for Existing Codebase**

Integrate into the existing CPR worker pipeline; no new viewer/display changes required.

- **Data representation** (cprMath.ts): reuse your current arch-curve sampler. For each dense s-point (~0.5–1 mm), sample radial lines at multiple depths (reuse your existing voxel interpolator). Build cost map as 2-D TypedArray (Float32) — <1 MB.  
- **Per-column / per-row / per-depth structure**: cost map is purely (s,d); z is collapsed into the cost function (mean or max over tooth-band z only — cheap). Final pano keeps your existing (s,z) output grid.  
- **Optimization/search algorithm**: classic DP table — O(S·D·step) with step=3–5 (<<0.1 s). Backtrack to recover d_opt[s]. Optional post-spline smoothing on d_opt.  
- **Preserve tooth/root while suppressing lower/background**: narrow local mean/median around d_opt[s] (2–4 mm total). Lower anatomy is at systematically wrong d → automatically excluded. Add gentle cost penalty for very negative d (lower border) if needed.  
- **Avoid previous blowout failure**: use mean or median reducer (never raw max); clip intensities inside the local kernel to your current VOI before averaging; initialise cost map with soft threshold so saturated metal does not dominate the path.  
- **Code location**: extend cprWorker.ts (add new label e.g. “dp-focal-trough-narrow”). Keep your existing quality-scoring + selection logic — it will now select much higher-quality candidates. The experimental virtual-pano path you gated off can be resurrected as this (it failed because it lacked the narrow-kernel step).

All changes stay inside the worker; output format identical to current so panoImageLoader.ts / Cornerstone viewport unchanged.

**F. Runtime Feasibility**

- Cost-map construction: similar voxel sampling load to one current CPR pass (you already do this for mean/MIP slabs).  
- DP: negligible.  
- Final narrow reconstruction: faster than current full-slab because kernel is 3–5× thinner.  
- Total: 2–5 s on typical dental CBCT (cf. 2022 MDPI method at 3.04 s average on Xeon). Precompute arch curve + ROI crop once (already done). If still tight, subsample z for cost only or reduce depth bins to 80 — still excellent quality. Well inside single-digit seconds; faster than your current “retry” loops.

**G. Validation Metrics**

Metrics that will improve first (in order):  
- lowerBandP50 and lowerBandBrightFraction drop sharply (target <0.2).  
- toothBandMean rises + toothBandP10/P90 tightens (less contamination).  
- qualityScore jumps (your current –19 will become strongly positive).  
- selectedVoi stabilizes (less need for extreme windows).  

Image characteristics that improve first:  
- Background goes from 90 s-TV-static to clean dark.  
- Anterior teeth lose shearing — normal morphology.  
- Roots become continuously visible without ghosting.  
- Cortical bone and lamina dura sharp, tooth/background separation diagnostic.  

Remaining failure modes if only partly right: occasional local “kink” in edentulous zones (fix: increase smoothness λ) or slight residual blur if cost function too weak (tune to intensity + gradient hybrid). These are easy to iterate on; nothing like the current ceiling.

**H. Dead Ends / Don’t Do This**

- Any further slab-thickness / mean-vs-MIP / conditional masking tweaks inside the current CPR pipeline.  
- “Just denoise / CLAHE / VOI” post-processing as primary strategy.  
- Re-trying the old virtual-pano stage without the narrow local kernel around a proper surface.  
- Full 3-D graph-cut surface or ray-tracing tomosynthesis unless you have months of engineering budget.  
- Any ML / GAN / diffusion path (explicitly forbidden).  
- Per-column independent CoM or max without global smoothness (you already proved it unstable).  

This DP focal-trough model is the decisive, practical, research-grade pivot. It is the minimal change that actually changes the architecture. Implement the cost-map + DP stage first (one sprint), then narrow-kernel recon — you will see diagnostic-quality output immediately. Your current codebase is already 80 % there; this is the missing 20 % that matters.