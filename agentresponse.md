# Dynamic Profile-Weighted Reconstruction for CBCT Dental Panoramas

This document outlines a CPU-optimized architecture to replace standard slab-mean Curved Planar Reformation (CPR). Moving from a naive thick-slab CPR to a diagnostic-quality virtual panoramic radiograph requires shifting from pure geometry to density-aware anatomy extraction. 

The primary reason a standard slab-mean reduction creates a "speckled static" background and geometric distortion is that the human dental arch is not a 2D curve extruded vertically. Teeth lean, and roots diverge. Casting a thick, uniform slab inevitably captures air, soft tissue, the cervical spine, and opposite-arch scatter.

## 1. Estimating the Patient-Specific Focal Trough

Instead of a single manually drawn or generically fitted 2D curve, a 3D support surface must be extracted.



**The Method:**
1. **Axial Condensation:** Take a MeanIP (Mean Intensity Projection) of the axial slices in the Z-range where teeth are expected (ignoring the extreme top of the maxilla and bottom of the mandible).
2. **Threshold & Skeletonize:** Apply an Otsu threshold to isolate dense bone/enamel. Use a morphological skeletonization to get a 1-pixel thick arch.
3. **Polynomial Fit:** Fit a 4th-order polynomial or a cubic spline to the skeleton. This is your *Base Curve* $x_c(s), y_c(s)$, where $s$ is the arc length.
4. **Volume Unrolling:** Resample the 3D volume along the normals to this base curve. This creates an "Unrolled Volume" matrix $V(s, d, z)$, where $s$ is the column (horizontal position on pano), $d$ is depth along the normal (the thickness dimension), and $z$ is the vertical axis.

---

## 2. Choosing the Correct Depth per Pano Location

Instead of averaging all $d$ values like a standard slab, we dynamically find where the anatomy actually lies within the unrolled volume. 

**The Method:**
For every pixel $(s, z)$ in the final panorama, evaluate the ray of pixels along the depth axis $d$. Find the **Center of Mass (CoM) of density** along that ray. Because teeth and bone are the densest objects, the density CoM naturally tracks the inclination of the teeth.

Let $\mu(d)$ be the voxel intensity at depth $d$. Calculate the optimal depth $d_{opt}$ for a given $(s, z)$ coordinate:

$$d_{opt}(s,z) = \frac{\sum_{d=d_{min}}^{d_{max}} d \cdot (\mu(d))^p}{\sum_{d=d_{min}}^{d_{max}} (\mu(d))^p}$$

> **Note:** We raise $\mu$ to a power $p$ (e.g., $p=2$ or $3$) to heavily bias the center of mass toward dense enamel and bone, ignoring soft tissue.

---

## 3. Preserving Teeth/Roots & Suppressing Background

To recreate the diagnostic look of an X-ray, perform a **Gaussian-Weighted Integration** centered on $d_{opt}$.

$$P(s, z) = \sum_{d} \mu(d) \cdot \exp\left(-\frac{(d - d_{opt}(s,z))^2}{2\sigma^2}\right)$$

* $P(s, z)$ is the final pixel value on the panorama.
* $\sigma$ represents the width of the "focal trough." 

**Crucial CPU Optimization:** $\sigma$ should be variable. Make it narrow in the anterior region ($s$ values near the center) to avoid the spine, and wider in the posterior regions ($s$ values near the edges) to capture diverging molar roots.

---

## 4. Method Choice: Dynamic Programming vs. Column Depth

**Recommendation:** Start with the 2.5D Per-Column Depth (The CoM approach above). It is highly parallelizable and very fast on CPUs.

**When to use Dynamic Programming (DP):**
If the CoM approach yields jagged roots (because the optimal depth fluctuates too rapidly pixel-by-pixel), apply DP. Treat the unrolled volume slice $V(s, d, z_{fixed})$ as a grid and use DP to find the continuous path of maximum density from $s=0$ to $s=s_{max}$. 
* **Cost function:** Maximize voxel intensity while penalizing large jumps in $d$ (to ensure the surface is smooth and doesn't jump from a tooth to the spine).
* *Verdict:* Keep DP as a fallback. The exponentiated Center of Mass formula is usually mathematically sufficient for dental CBCTs and runs an order of magnitude faster on a CPU.

---

## 5. Minimal Implementation Path

To integrate this into an existing pipeline without a total rewrite:

1. **Keep** the existing curve-drawing/detection logic to get the base 2D curve.
2. **Replace** the current `slab-mean` loop.
3. **Insert** the "Unroll" step: Allocate a buffer for the normal rays.
4. **Implement** the $d_{opt}$ calculation for each $(s, z)$ coordinate.
5. **Calculate** the final pixel using the Gaussian-weighted sum.
6. **Apply** a final Contrast Limited Adaptive Histogram Equalization (CLAHE) to the resulting 2D image to enhance trabecular bone contrast.

---

## 6. New Logs & Diagnostics

Transition away from flat slab thresholding logs and track the behavior of the dynamic surface:

* `focalTrough.anteriorDepthVariance`: Measures how much $d_{opt}$ fluctuates in the front teeth. High variance indicates the algorithm is confusing teeth with the spine.
* `focalTrough.meanThickness_Anterior` vs `focalTrough.meanThickness_Posterior`: Confirms the Gaussian width ($\sigma$) is correctly adapting to the arch region.
* `reconstruction.weightedSignalRatio`: The ratio of the intensity integrated *within* $1\sigma$ of $d_{opt}$ versus the total intensity of the ray. A higher ratio indicates successful background suppression.