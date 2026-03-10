A. Executive Verdict  
The legacy curved-slab CPR model has essentially hit its ceiling; further slab/VOI/mask tuning will not make your pano diagnostically robust. The best pivot is a **patient‑specific focal‑trough virtual pano**: estimate a smooth 3D support surface (or 2–3 parallel surfaces) over an *unrolled ribbon volume* around the arch using a dynamic‑programming / seam‑carving style optimization on a tooth‑likelihood map, then resample a narrow band around that surface to form the pano. Methods of this type from CBCT already produce pano images with clarity comparable to conventional radiographs while remaining CPU‑feasible. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7294321/)

***
## B. Ranked Architecture Options
### 1. Focal‑trough surface + DP over unrolled ribbon (recommended)
Core idea:  
- Build a 3D “ribbon” volume around the arch (parameters: arch coordinate \(s\), bucco‑lingual offset \(d\), vertical position \(z\)).  
- Collapse \(z\) to construct a 2D tooth‑likelihood map \(L(s,d)\) (intensity + gradient/edge features).  
- Use dynamic programming (or equivalent shortest‑path) to find a smooth depth surface \(d^\*(s)\) maximizing \(L\) minus curvature/smoothness penalties.  
- Generate the pano by sampling narrowly around \(d^\*(s)\) with a Gaussian kernel in \(d\) and limited thickness along \(z\).  

Why it is strong:  
- Architecturally matches successful CBCT‑to‑pano methods that construct a curved 3D panoramic surface then “develop” it into 2D, yielding clear dentition with minimal superimposition. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7438751/)
- Explicitly does **selection before reduction**, avoiding the mixed‑anatomy averaging that is killing your current CPR.  
- DP/seam‑carving on a 2D cost map is trivially CPU‑viable and easy to implement in a TS worker using typed arrays.  
- Can be extended later to 2 or 3 coupled surfaces (crowns, roots, mandibular canal) with small incremental complexity. [journals.plos](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0156976)

I recommend this as the primary architecture.
### 2. Multi‑surface focal‑trough model (crowns + roots + canal)
Core idea:  
- Same ribbon volume and DP machinery as (1), but estimate multiple roughly parallel support surfaces: e.g., one through crowns, one through root apices, optionally one through mandibular canal.  
- Blend these surfaces into the final pano in a height‑dependent way (upper rows from crown surface, lower rows from root surface, very basal from canal surface).  

Pros:  
- Better control over root vs crown sharpness; can avoid cutting roots off when focal trough is tuned for crowns.  
- Consistent with reports that CBCT‑reformatted panos with custom focal troughs can be as accurate or better than conventional panos for localizing structures like the mental foramen. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7294321/)

Cons:  
- More engineering and tuning (surface interactions, blending logic).  
- Gains are incremental once a good single‑surface pano exists.  

Verdict: **Second phase** after single‑surface DP is working.
### 3. Full panoramic scan simulation / tomosynthesis
Core idea:  
- Explicitly simulate the panoramic device: define a sequence of rotation centers and detector positions, then integrate CBCT attenuation along each ray using Beer–Lambert. [arxiv](https://arxiv.org/html/2408.09358v1)

Pros:  
- Conceptually truest to the real modality; recent work shows high‑quality synthetic panos from CBCT this way. [ieeexplore.ieee](https://ieeexplore.ieee.org/iel8/6287639/10820123/11015780.pdf)

Cons:  
- Ray‑driven reprojection over the full jaw is substantially heavier than a ribbon reslice; CPU‑only JS will struggle to maintain single‑digit‑second performance on typical CBCT volumes unless heavily downsampled or offloaded to native code.  
- Requires careful geometric modeling of your scanner / virtual pano device and aggressive optimization.  

Verdict: **Scientifically attractive but overkill** for a TS worker right now; reserve as a long‑term R&D path or for a native backend.
### 4. Further CPR / curved‑slab variants
All of the following live in the same family: mean/MIP slab tuning, eligibility masks, bright‑tail heuristics, center‑of‑mass selection inside an otherwise collapsing curved slab, etc.  

Verdict: They fundamentally keep **reduction before true anatomical selection**, so even sophisticated heuristics will keep mixing unwanted anatomy and artifacts. They are essentially exhausted and should not be the main direction.

***
## C. Why Current CPR Saturated
In reconstruction‑model terms, your current pipeline is essentially:

1. Follow a curved arch line through the volume.  
2. For each column, sample a **bucco‑lingual slab** (and sometimes some vertical thickness).  
3. Reduce that slab with a simple operator (mean, MIP, sharpened mean, etc.) to a single intensity per pixel column/row.  

Architectural limits:

- **Depth mixing of incompatible anatomy**: a mandibular cortex, lingual plate, canal walls, tongue surface, and even cervical spine can all be inside the same slab. Any reduction operator (mean, MIP, “smart mean”) is forced to average or pick among structures that are mutually exclusive in a true pano. This produces speckle, loss of contrast, and alien‑looking teeth where two incompatible surfaces are blended.  
- **No explicit focal trough**: real panoramic systems restrict sharp depiction to a narrow focal layer whose thickness and position vary along the arch and vertically. Your slab has a fixed depth and treats off‑trough anatomy as equally eligible, causing double contours and blurred cortices. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6131388/)
- **Local, myopic decision rules**: per‑sample or per‑depth heuristics operate independently for each column, so they jump between buccal and lingual cortices, or tooth vs background, from one column to the next. This manifests as shearing or broken anterior teeth. A pano needs a **globally smooth support surface**, not columnwise greedy picks.  
- **Reduction before selection**: once a slab has been collapsed, all information needed to disentangle mixed anatomy is lost. No amount of post‑hoc scoring or denoising can reconstruct which depth should have been chosen.  

Given you’ve already tried two‑pass masking and slab tuning without structural change, you’re seeing the inherent “noise floor” of a curved slab approach, not a tunable bug.

***
## D. Recommended Model in Detail
### High‑level model
Adopt a **single‑surface, focal‑trough virtual pano**:

1. Use your existing arch curve to define coordinates \((s,d,z)\):  
   - \(s\): position along dental arch (left→right).  
   - \(d\): bucco‑lingual offset along the local normal of the arch.  
   - \(z\): vertical (inferior–superior).  

2. Resample a **ribbon volume** \(V(s,d,z)\) around the arch with tri‑linear interpolation. This generalizes your current CPR sampling but keeps the full 3D ribbon instead of immediately collapsing in \(d\).  

3. Construct a **tooth‑likelihood map** \(L(s,d)\) by aggregating features over a specified vertical band (e.g., from just above occlusal plane down to near root apices):  
   - robust intensity measure (e.g., percentile in HU window favoring dentin/enamel range);  
   - gradient magnitude along \(d\) (cortical / enamel edges);  
   - local variance or Laplacian energy to prefer structured tooth regions over homogeneous tongue/air.  

   Approaches that infer a panoramic surface from similar tooth‑likelihood features have produced clear, low‑superimposition panos from CBCT and CT volumes. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7438751/)

4. Solve for a **smooth depth surface** \(d^\*(s)\) maximizing cumulative likelihood plus smoothness terms:  
   \[
   \max_{d(s)} \sum_s L(s,d(s)) - \lambda |d(s) - d(s-1)| - \mu |d(s) - 2d(s-1) + d(s-2)|
   \]
   Implement via DP / seam‑carving on the discrete grid (s,d).  

5. Generate the pano image \(P(s,z)\) by sampling **only a narrow band around \(d^\*(s)\)**:  
   \[
   P(s,z) = \sum_{k=-K}^{K} w_k \, V(s, d^\*(s)+k, z)
   \]
   with Gaussian weights \(w_k\) and small \(K\) (e.g., 1–2 voxel offsets).  

This implements “selection before reduction”: you first commit to a coherent anatomical layer, then average only very local neighbors around that layer.
### Handling upper/lower arches and roots
- For mandibular pano, construct arch on lower teeth and restrict vertical band for \(L(s,d)\) roughly from crown mid‑height to slightly below canal; this biases \(d^\*(s)\) toward the root/cortex region you care about.  
- To better cover both crowns and roots:  
  - either tune the vertical band used for \(L(s,d)\) so that high‑information root regions dominate the score,  
  - or add a second surface \(d_{\text{root}}^\*(s)\) optimized on a deeper vertical band, and drive lower rows of \(P(s,z)\) from that surface (architecture option 2).
### Avoiding blown‑out, streaky failure
Your prior virtual‑pano experiment likely failed because:

- A MIP‑like or max‑intensity criterion favored metal restorations and noise streaks.  
- VOI and normalization were applied after an aggressive local reducer that destroyed dynamic range.  

Mitigations in this model:

- **Robust feature design** in \(L(s,d)\):  
  - clamp contributions from very high HU (metal, > ~2500 HU) so they don’t dominate;  
  - combine gradient and mid‑range intensity, not raw peak intensity.  
- **Intensity normalization per column** before final rendering: subtract per‑column median of off‑tooth regions or apply a smooth gain curve to equalize exposure along the arch, as done in synthetic pano methods. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7438751/)
- Keep the final kernel narrow (K small) so you never integrate over entire low‑density regions that would wash out teeth.  

***
## E. Implementation Strategy for Existing Codebase
I’ll map this to your files and data flow.
### 1. Data representation
In `cprWorker.ts` / `cprMath.ts`:

- Define an **unrolled ribbon grid**:  
  - `numS` (columns, along arch) – you already have something analogous.  
  - `numD` (depth samples) – e.g., 32–64 between inner lingual and outer buccal limits.  
  - `numZ` (vertical samples) – reuse your pano image height.  

- Represent `V(s,d,z)` as a flat `Float32Array` in worker memory: e.g., index `(s * numD + d) * numZ + z`.  

- Maintain a precomputed table of **world coordinates** for each `(s,d,z)` so resampling CBCT voxels is just trilinear interpolation with pre‑looked‑up positions.
### 2. Building the ribbon
Modify the existing CPR sampling:

- Instead of immediately summing/averaging along depth, have the worker:  
  - iterate over `s`, then over `d`, then over `z` and fill `V[s,d,z]` by resampling CBCT.  
- You can reuse your arch polyline and normal computation from the current CPR math; you just add a loop over `d`.  

This is the heaviest step but still just trilinear interpolation – similar to what you already do, only with an extra dimension stored instead of reduced.
### 3. Computing the tooth‑likelihood map \(L(s,d)\)
In `cprWorker.ts`:

- Choose a vertical band `[zMin, zMax]` (e.g., a fixed offset relative to occlusal plane or jaw bounding box).  
- For each `(s,d)`:
  - Accumulate:  
    - mean/median intensity over `[zMin,zMax]`;  
    - mean gradient magnitude along `d` direction (approx via finite differences over `d` within the band);  
    - local variance.  
  - Normalize these features (e.g., z‑score or min‑max across whole map) and combine linearly with tuned weights into a scalar `L[s,d]`.  

- Store `L` as `Float32Array` of size `numS * numD`.  

This is straightforward numeric work and entirely CPU‑friendly.
### 4. Dynamic programming / shortest‑path surface
Still in worker:

- Treat each `(s,d)` as a node; processing order increasing `s`.  
- At `s = 0`, set `cost[0,d] = -L[0,d]`.  
- For `s > 0`:
  - For each `d`, consider only a small band of predecessors `d'` in `[d-Δ, d+Δ]` (Δ ~ 2–3) to enforce slope limits.  
  - Compute  
    `cost[s,d] = -L[s,d] + min_{d'} (cost[s-1,d'] + λ * |d-d'| + μ * |d - 2*d' + d''| approx)`;  
  - Store backpointers.  
- After filling all `s`, backtrack from the best `d` at the final column to obtain `dStar[s]`.  

Complexity is \(O(\text{numS} \cdot \text{numD} \cdot \Delta)\), which is tiny at typical sizes.
### 5. Pano synthesis
In `cprWorker.ts` (or a new module):

- Allocate an output `Float32Array` for pano pixels of size `numS * numZ`.  
- For each column `s` and row `z`:
  - Get `d0 = dStar[s]`.  
  - For small integer offsets `k` in `[-K, K]`, read `V[s, d0+k, z]` (clamping bounds) and aggregate with precomputed Gaussian weights into `P[s,z]`.  

- Optionally apply a light 1D smoothing along `s` to suppress remaining column‑wise jitter.  

Then the worker posts this pano buffer exactly as your existing CPR does, so:

- `panoImageLoader.ts` and the Cornerstone viewport path should remain mostly unchanged, aside from perhaps different metadata (VOI presets tuned to HU range of enamel/dentin).
### 6. Preserving tooth/root visibility & suppressing background
Architecturally, this pivot helps automatically, but you can further bias:

- When computing `L(s,d)`, down‑weight depths where the vertical band includes large air/tongue fractions (e.g., negative HU or very low gradient).  
- Use different `z` bands to build `L` for upper vs lower arches if you later support both.  
- Regularize `dStar` so it prefers depths roughly corresponding to expected jaw thickness from population arch data (e.g., priors derived from focal‑trough thickness studies). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6131388/)
### 7. Where to put experimental vs stable paths
- Keep your current legacy CPR under existing labels (e.g., `retry-mean-sharp-narrow`).  
- Add a new mode, say `"pano-focaltrough-dp"`, wired through `useCPROrchestrator.ts` and your logging.  
- Initially, log both legacy and new surfaces and thumbnails; once confident, flip the hanging protocol / default viewport to the new path.

***
## F. Runtime Feasibility
Assume a typical CBCT cropped to jaw: ~\(400^3\) voxels, and choose, say:

- `numS ≈ 1000` pano columns  
- `numD ≈ 48` depth samples  
- `numZ ≈ 400` pano rows
### For the recommended model
1. **Ribbon resampling**:  
   - Work: \(1000 × 48 × 400 ≈ 19M\) tri‑linear samples.  
   - This is comparable to a handful of full‑volume traversals – high but reasonable in an optimized worker using typed arrays and loop ordering that is cache‑friendly.  

2. **Feature computation for L**:  
   - A few tens of millions of float ops (simple adds/mults) – negligible relative to resampling.  

3. **DP optimization**:  
   - `1000 × 48 × Δ` with Δ ~ 5 → < 250k operations; trivial.  

4. **Pano synthesis**:  
   - `numS × numZ × (2K+1)` ~ `1000 × 400 × 5 = 2M` reads and multiplies.  

Qualitatively, the runtime is dominated by the **initial ribbon resampling**, which is similar in complexity to your current CPR (you already resample across z and s; here you also store multiple d layers). With tight loops and maybe moderate downsampling for the DP step (e.g., compute L on a coarser z grid), this should comfortably fit in **single‑digit seconds** on a modern desktop CPU in a worker.  

Precomputation/reuse opportunities:

- Cache the ribbon volume `V` for a given CBCT + arch; you can:  
  - vary VOI/windowing without re‑sampling;  
  - recompute `L` with different weights or vertical bands “for free”;  
  - try single vs multi‑surface variants sharing the same `V`.
### For multi‑surface variant
- Most cost is additional DP passes over `L` built with different vertical bands; still minor compared to resampling.  
- Additional pano synthesis is linear in `numS * numZ`.  

Still realistic in the 1–2× time of the single‑surface model.
### For full tomosynthesis
- Complexity roughly scales with *number of rays × average voxels per ray*.  
- Simulating a realistic panoramic scan easily requires tens to hundreds of millions of ray‑voxel intersections and exponentials; CPU‑only TS will likely exceed your single‑digit second budget unless you downsample heavily, which then reduces quality. [arxiv](https://arxiv.org/html/2408.09358v1)

So for your constraints, **focal‑trough DP is the only model that significantly changes reconstruction quality while remaining clearly CPU‑feasible**.

***
## G. Validation Metrics
For the DP focal‑trough model, I’d track the following.
### Numerical / log metrics
- **Tooth‑band contrast improvement**:  
  - Re‑compute `toothBandP90 - toothBandP10` and compare vs legacy path; expect a clear increase as the pano stops mixing background. Methods based on panoramic surfaces report clear visibility of enamel/dentin and pulp with thin layers. [journals.plos](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0156976)
- **Lower‑band bright fraction drop**:  
  - Your current `lowerBandBrightFraction` ~0.73 indicates strong contamination; expect this to drop substantially as the focal trough excludes lower soft‑tissue/background.  
- **Support‑surface smoothness**:  
  - RMS of `d^\*(s+1) - d^\*(s)` and second‑difference; should stay within anatomically plausible limits and be free of spikes.  
- **Rejection rate of pathological runs**:  
  - If you keep a quality score, add terms for excessive surface curvature or low aggregate `L`, and monitor how often runs are auto‑flagged.
### Image‑space characteristics that should improve first
- **Anterior teeth**:  
  - Less shearing and double contours; smoother, anatomically plausible incisor shapes.  
- **Mandibular border and canal**:  
  - Crisper cortices with reduced superimposed noise; canal more continuous. CBCT reformatted panos with custom focal troughs have achieved accurate localization of mental foramen and canal compared with conventional panos, so this is realistic. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7294321/)
- **Background speckle**:  
  - Dramatic reduction in “90s TV static” as air/tongue and opposite banks fall outside the selected depth.
### Residual failure modes if the model is only partly right
- **Arch tracking issues**:  
  - If the initial arch curve is off (e.g., in grossly misaligned or edentulous regions), DP will faithfully pick the “best” depth along the wrong path, causing local stretching/compression of anatomy.  
- **Metal dominance**:  
  - Without careful feature design, DP can still latch onto metal crowns/implants; you’ll see streaky high‑HU bands localized around those teeth.  
- **Edentulous segments**:  
  - Where no tooth‑like structure exists, the surface may become unstable or hug cortical bone; pano will still be readable but may show odd local thinning or bending of the jaw.  

These are manageable and can be iteratively addressed through better `L` design and priors.

***
## H. Dead Ends / Don’t Do This
Given your audit and the above architecture, I would stop investing in:

- **More curved‑slab heuristics**: changing slab thickness, number of samples, or mean/MIP combinations, with or without ad‑hoc masks; they cannot fix structural depth‑mixing.  
- **Display‑only VOI / CLAHE / denoiser tweaks as “solutions”**: they may make screenshots prettier but cannot recover lost anatomical separation.  
- **Pure center‑of‑mass or peak‑intensity depth selection** without global smoothness: too unstable in anterior, metal, and low‑contrast regions, and already partially tried.  
- **Complex multi‑pass eligibility masking inside the same slab reduction model**: still reduction‑first; architectural limit remains.  
- **Full panoramic ray‑tracing / tomosynthesis in TS worker** as near‑term fix: great research project, but not compatible with your CPU‑only, web‑viewer runtime constraints today. [arxiv](https://arxiv.org/html/2408.09358v1)
- **Any GAN/diffusion or aggressive “style” synthesis**: explicitly conflicts with your requirement to preserve real anatomy and diagnostic validity.  

***

In summary: the **focal‑trough support‑surface + DP over an unrolled ribbon volume** is the decisive pivot that addresses the true architectural failure of your current CPR path while staying realistic in a TS+worker, CPU‑only environment and being backed by successful CBCT‑to‑pano research. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7294321/)