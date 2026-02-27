/**
 * cprWorker.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Web Worker: Generates the 2D panoramic "curtain" image from a 3D CBCT volume
 * by sampling along the precomputed RMF frames.
 *
 * THREADING MODEL
 * ───────────────
 * This file runs entirely off the main thread. It receives volume scalar data
 * via SharedArrayBuffer (zero-copy) or a cloned ArrayBuffer (safe fallback),
 * plus the precomputed RMF frame array from cprMath.ts.
 *
 * OUTPUT
 * ──────
 * A Float32Array of shape [panoHeight × panoWidth] in row-major order,
 * containing scalar values in the volume's native HU range.
 * The main thread wraps this in a Cornerstone3D IImage via panoImageLoader.ts.
 *
 * PERFORMANCE NOTES
 * ─────────────────
 * A 512×512×300 CBCT with 800×400 pano output and 20-sample slab = ~6.4M
 * trilinear lookups. On a modern browser this completes in ~800ms–2s in the
 * worker. The main thread remains fully interactive during generation.
 */

// ─────────────────────────────────────────────────────────────────────────────
// MESSAGE TYPES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * One RMF frame received from the main thread.
 * Matches the CPRFrame interface in cprMath.ts exactly.
 */
interface WorkerFrame {
  position: [number, number, number]; // World-space point on the arch (P_i)
  N_slab:   [number, number, number]; // Z-zeroed cheek-to-cheek direction
}

/**
 * Message sent FROM main thread TO worker.
 */
interface CPRWorkerInput {
  /** Volume scalar data. SharedArrayBuffer or cloned ArrayBuffer — both handled. */
  scalarData: Float32Array | Int16Array;
  /** True if scalarData.buffer is SharedArrayBuffer — informs transfer strategy */
  isSharedArrayBuffer: boolean;
  /** Volume dimensions [cols, rows, slices] = [x, y, z] */
  dimensions: [number, number, number];
  /** Voxel spacing in mm [dx, dy, dz] */
  spacing: [number, number, number];
  /** Volume origin in world space [ox, oy, oz] */
  origin: [number, number, number];
  /**
   * 3×3 direction cosines matrix, row-major.
   * Index layout: [Xx,Xy,Xz, Yx,Yy,Yz, Zx,Zy,Zz]
   */
  direction: number[];
  /** Equidistant RMF frames along the arch — one per output column */
  frames: WorkerFrame[];
  /** Output panoramic image width in pixels (= frames.length) */
  panoWidth: number;
  /** Output panoramic image height in pixels (= vertical FOV in voxels) */
  panoHeight: number;
  /**
   * Slab half-thickness in mm.
   * Samples are taken at -slabHalfThicknessMm … +slabHalfThicknessMm along N_slab.
   */
  slabHalfThicknessMm: number;
  /** Number of samples across the slab (odd number recommended, e.g. 21) */
  slabSamples: number;
  /** Aggregation mode: 'MIP' = maximum, 'MEAN' = average */
  aggregation: 'MIP' | 'MEAN';
}

/**
 * Message sent FROM worker TO main thread on success.
 */
interface CPRWorkerSuccess {
  type: 'SUCCESS';
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  /** min/max scalar values in the output — used by panoImageLoader for VOI */
  minValue: number;
  maxValue: number;
}

/**
 * Message sent FROM worker TO main thread on failure.
 */
interface CPRWorkerError {
  type: 'ERROR';
  message: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// COORDINATE MATH HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Converts a world-space point [wx, wy, wz] to continuous voxel indices [vi, vj, vk].
 *
 * VTK formula: voxelIndex = inverse(direction) × (worldPoint - origin) / spacing
 *
 * For the common case where direction is identity (axis-aligned CBCT), this
 * simplifies to a divide-subtract. The full matrix path handles tilted gantries.
 */
function worldToVoxel(
  wx: number, wy: number, wz: number,
  origin: [number, number, number],
  spacing: [number, number, number],
  invDir: number[] // 3×3 inverse direction cosines, row-major
): [number, number, number] {
  // Translate: relative to origin
  const rx = wx - origin[0];
  const ry = wy - origin[1];
  const rz = wz - origin[2];

  // Rotate by inverse direction, then scale by 1/spacing
  const vi = (invDir[0] * rx + invDir[1] * ry + invDir[2] * rz) / spacing[0];
  const vj = (invDir[3] * rx + invDir[4] * ry + invDir[5] * rz) / spacing[1];
  const vk = (invDir[6] * rx + invDir[7] * ry + invDir[8] * rz) / spacing[2];

  return [vi, vj, vk];
}

/**
 * Inverts a 3×3 matrix (row-major flat array of 9 elements).
 * Used once at worker startup to cache the inverse direction cosines.
 *
 * For orthogonal direction matrices (all real DICOM volumes), the inverse
 * equals the transpose — so this is just a transposition.
 */
function invertMatrix3(m: number[]): number[] {
  // Transpose (valid because direction matrices are orthogonal/rotation-only)
  return [
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// TRILINEAR INTERPOLATION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Samples the 3D scalar volume at continuous voxel coordinates (vi, vj, vk)
 * using trilinear interpolation directly against the flat scalar array.
 *
 * WHY NOT vtkImageData.getScalarValueFromWorldCoord?
 * That method includes bounds checks and JavaScript object overhead on every
 * call. With 6M+ samples per panoramic generation, the overhead is ~3× slower
 * than this direct typed-array implementation.
 *
 * WHY NOT nearest-neighbor?
 * Nearest-neighbor produces visible aliasing on enamel margins in dental CBCT.
 * At typical CBCT voxel sizes (0.2–0.4 mm), this can mimic fracture lines —
 * a patient safety issue. Trilinear is mandatory for diagnostic-grade output.
 *
 * @param data    - Flat typed scalar array from volume.imageData
 * @param dims    - [nx, ny, nz] voxel dimensions
 * @param vi      - Continuous voxel x-index
 * @param vj      - Continuous voxel y-index
 * @param vk      - Continuous voxel z-index
 * @param oobValue - Value returned for out-of-bounds samples (typically 0 or -1000)
 * @returns Interpolated scalar value in the volume's native units (HU)
 */
function trilinear(
  data: Float32Array | Int16Array,
  dims: [number, number, number],
  vi: number,
  vj: number,
  vk: number,
  oobValue: number = -1000
): number {
  const [nx, ny, nz] = dims;

  // Clamp check — return out-of-bounds value for samples outside the volume
  if (vi < 0 || vj < 0 || vk < 0 || vi >= nx - 1 || vj >= ny - 1 || vk >= nz - 1) {
    // Soft clamp: allow sampling up to 0.5 voxel outside for edge interpolation
    const ci = Math.max(0, Math.min(nx - 1.001, vi));
    const cj = Math.max(0, Math.min(ny - 1.001, vj));
    const ck = Math.max(0, Math.min(nz - 1.001, vk));
    if (vi < -0.5 || vj < -0.5 || vk < -0.5 || vi > nx - 0.5 || vj > ny - 0.5 || vk > nz - 0.5) {
      return oobValue;
    }
    // Fall through with clamped values
    return trilinearCore(data, nx, ny, ci, cj, ck);
  }

  return trilinearCore(data, nx, ny, vi, vj, vk);
}

/**
 * Core trilinear computation — assumes coordinates are in-bounds.
 * Separated to allow the clamp path above to reuse it.
 *
 * VTK flat index formula: idx = k*(nx*ny) + j*nx + i
 */
function trilinearCore(
  data: Float32Array | Int16Array,
  nx: number,
  ny: number,
  vi: number,
  vj: number,
  vk: number
): number {
  // Integer lower-bound voxel corners
  const i0 = Math.floor(vi);
  const j0 = Math.floor(vj);
  const k0 = Math.floor(vk);
  const i1 = i0 + 1;
  const j1 = j0 + 1;
  const k1 = k0 + 1;

  // Fractional offsets (interpolation weights)
  const fi = vi - i0; // weight toward i1
  const fj = vj - j0; // weight toward j1
  const fk = vk - k0; // weight toward k1

  // Precomputed slice stride
  const sliceStride = nx * ny;

  // 8 corner flat indices
  const c000 = k0 * sliceStride + j0 * nx + i0;
  const c100 = k0 * sliceStride + j0 * nx + i1;
  const c010 = k0 * sliceStride + j1 * nx + i0;
  const c110 = k0 * sliceStride + j1 * nx + i1;
  const c001 = k1 * sliceStride + j0 * nx + i0;
  const c101 = k1 * sliceStride + j0 * nx + i1;
  const c011 = k1 * sliceStride + j1 * nx + i0;
  const c111 = k1 * sliceStride + j1 * nx + i1;

  // 8 corner values
  const v000 = data[c000];
  const v100 = data[c100];
  const v010 = data[c010];
  const v110 = data[c110];
  const v001 = data[c001];
  const v101 = data[c101];
  const v011 = data[c011];
  const v111 = data[c111];

  // Trilinear blend — 7 lerps total
  // Interpolate along i (x)
  const c00 = v000 + fi * (v100 - v000);
  const c10 = v010 + fi * (v110 - v010);
  const c01 = v001 + fi * (v101 - v001);
  const c11 = v011 + fi * (v111 - v011);

  // Interpolate along j (y)
  const c0 = c00 + fj * (c10 - c00);
  const c1 = c01 + fj * (c11 - c01);

  // Interpolate along k (z)
  return c0 + fk * (c1 - c0);
}

// ─────────────────────────────────────────────────────────────────────────────
// THICK-SLAB SAMPLING LOOP
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generates the full panoramic pixel array by sampling the 3D volume
 * along each RMF frame's arch point and slab direction.
 *
 * COORDINATE SYSTEM
 * ─────────────────
 * Panoramic image axes:
 *   X-axis (columns, u) = arch position (equidistant along arch)
 *   Y-axis (rows,    v) = vertical (superior-inferior, along z-axis of patient)
 *
 * For each output pixel (u=column, v=row):
 *   1. Get world position P_i from frame[u]
 *   2. Offset vertically: P_v = P_i + (v - centerRow) * spacing[2] * [0,0,1]
 *   3. For each slab sample s:
 *        P_sample = P_v + (s / (slabSamples-1) * 2 - 1) * slabHalfThicknessMm * N_slab
 *   4. Convert P_sample to voxel coords and trilinear-sample
 *   5. Aggregate samples across slab (MIP or MEAN)
 *
 * @returns { pixelData, minValue, maxValue }
 */
function generatePanorama(input: CPRWorkerInput): {
  pixelData: Float32Array;
  minValue: number;
  maxValue: number;
} {
  const {
    scalarData, dimensions, spacing, origin, direction,
    frames, panoWidth, panoHeight,
    slabHalfThicknessMm, slabSamples, aggregation
  } = input;

  // Precompute inverse direction matrix once — used for every worldToVoxel call
  const invDir = invertMatrix3(direction);

  // Output pixel buffer — row-major [row * panoWidth + col]
  const pixelData = new Float32Array(panoWidth * panoHeight);
  let minValue = Infinity;
  let maxValue = -Infinity;

  // Vertical extent: how many mm above/below the arch midpoint?
  // panoHeight rows map to [-vertHalfMm … +vertHalfMm] in patient Z.
  // 15mm above and below the arch midline covers the full dental anatomy.
  const vertHalfMm = 15.0;
  const vertStepMm = (vertHalfMm * 2) / (panoHeight - 1);

  // Slab step: evenly spaced samples from -slabHalfThicknessMm to +slabHalfThicknessMm
  const slabStepMm = slabSamples > 1
    ? (slabHalfThicknessMm * 2) / (slabSamples - 1)
    : 0;

  // ── Main nested loop ──────────────────────────────────────────────────────
  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [nx, ny, nz_slab] = frame.N_slab; // Z=0, cheek-to-cheek direction

    for (let row = 0; row < panoHeight; row++) {
      // Vertical offset from arch midpoint in world Z (superior-inferior)
      const vertOffsetMm = -vertHalfMm + row * vertStepMm;

      // Base world point: arch position offset vertically along patient Z
      const bx = px;
      const by = py;
      const bz = pz + vertOffsetMm; // pure Z shift — no X/Y component

      // ── Thick-slab accumulation ───────────────────────────────────────────
      let accumulator = aggregation === 'MIP' ? -Infinity : 0;

      for (let s = 0; s < slabSamples; s++) {
        // Distance along N_slab for this sample: -half … +half
        const slabOffset = slabSamples > 1
          ? -slabHalfThicknessMm + s * slabStepMm
          : 0;

        // Sample world position
        const sx = bx + slabOffset * nx;
        const sy = by + slabOffset * ny;
        const sz = bz + slabOffset * nz_slab; // nz_slab is always 0 by construction

        // World → continuous voxel indices
        const [vi, vj, vk] = worldToVoxel(sx, sy, sz, origin, spacing, invDir);

        // Sample the volume with trilinear interpolation
        const sample = trilinear(scalarData, dimensions, vi, vj, vk);

        // Aggregate
        if (aggregation === 'MIP') {
          if (sample > accumulator) accumulator = sample;
        } else {
          accumulator += sample;
        }
      }

      // Finalize pixel value
      const pixelValue = aggregation === 'MEAN'
        ? accumulator / slabSamples
        : accumulator;

      const pixelIndex = row * panoWidth + col;
      pixelData[pixelIndex] = pixelValue;

      // Track range for VOI in image loader
      if (pixelValue < minValue) minValue = pixelValue;
      if (pixelValue > maxValue) maxValue = pixelValue;
    }
  }

  return { pixelData, minValue, maxValue };
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER MESSAGE HANDLER
// ─────────────────────────────────────────────────────────────────────────────

self.onmessage = function (event: MessageEvent<CPRWorkerInput>) {
  try {
    const input = event.data;

    // Validate that we received usable scalar data
    if (!input.scalarData || input.scalarData.length === 0) {
      throw new Error('Received empty or null scalar data.');
    }
    if (!input.frames || input.frames.length === 0) {
      throw new Error('Received empty frames array.');
    }

    // Log transfer mode for debugging — remove in production
    const bufferType = input.isSharedArrayBuffer ? 'SharedArrayBuffer' : 'ArrayBuffer (cloned)';
    console.debug(`[cprWorker] Starting panorama generation. Buffer: ${bufferType}`);
    console.debug(`[cprWorker] Volume dims: ${input.dimensions}  Frames: ${input.frames.length}`);
    console.debug(`[cprWorker] Output: ${input.panoWidth}×${input.panoHeight}  Slab: ${input.slabSamples} samples`);

    const start = performance.now();
    const { pixelData, minValue, maxValue } = generatePanorama(input);
    const elapsed = (performance.now() - start).toFixed(0);

    console.debug(`[cprWorker] Done in ${elapsed}ms. Range: [${minValue.toFixed(0)}, ${maxValue.toFixed(0)}] HU`);

    // Post result back to main thread.
    // Transfer pixelData.buffer so the main thread receives it without copying.
    // This is safe because the worker no longer needs it after posting.
    const response: CPRWorkerSuccess = {
      type: 'SUCCESS',
      pixelData,
      panoWidth: input.panoWidth,
      panoHeight: input.panoHeight,
      minValue,
      maxValue,
    };

    (self as unknown as Worker).postMessage(response, [pixelData.buffer]);

  } catch (err) {
    const response: CPRWorkerError = {
      type: 'ERROR',
      message: err instanceof Error ? err.message : String(err),
    };
    (self as unknown as Worker).postMessage(response);
  }
};
