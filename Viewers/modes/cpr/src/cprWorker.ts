/**
 * cprWorker.ts
 * Web Worker: Generates the 2D panoramic image from a 3D CBCT volume.
 */

interface WorkerFrame {
  position: [number, number, number];
  T?: [number, number, number];
  N_slab: [number, number, number];
}

interface CPRWorkerInput {
  scalarData: Float32Array | Int16Array;
  isSharedArrayBuffer: boolean;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
  worldToIndex?: ArrayLike<number>;
  verticalDir?: [number, number, number];
  frames: WorkerFrame[];
  panoWidth: number;
  panoHeight: number;
  vertHalfMm?: number;
  verticalCenterOffsetMm?: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  aggregation: 'MIP' | 'MEAN';
  applyModalityLut?: boolean;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  bitsAllocated?: number;
  highBit?: number;
  pixelRepresentation?: number;
  isPreScaled?: boolean;
  debugRunId?: string;
}

function isMat4ArrayLike(value: unknown): value is ArrayLike<number> {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as ArrayLike<number>;
  if (typeof candidate.length !== 'number' || candidate.length < 16) {
    return false;
  }

  for (let i = 0; i < 16; i++) {
    if (!Number.isFinite(Number(candidate[i]))) {
      return false;
    }
  }

  return true;
}

interface CPRWorkerSuccess {
  type: 'SUCCESS';
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  minValue: number;
  maxValue: number;
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  debugPayload?: {
    diagnostic: Record<string, unknown>;
    outputSignature: {
      sampledCount: number;
      checksum: number;
      absChecksum: number;
      first16: number[];
    };
  };
}

interface CPRWorkerError {
  type: 'ERROR';
  message: string;
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-10) {
    return [0, 0, 1];
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function dot3(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function percentile(values: number[], q: number): number {
  if (!values.length) {
    return NaN;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const index = Math.max(0, Math.min(sorted.length - 1, q * (sorted.length - 1)));
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) {
    return sorted[low];
  }
  const t = index - low;
  return sorted[low] * (1 - t) + sorted[high] * t;
}

function worldToVoxel(
  wx: number,
  wy: number,
  wz: number,
  origin: [number, number, number],
  spacing: [number, number, number],
  invDir: number[],
  worldToIndex?: ArrayLike<number>
): [number, number, number] {
  if (isMat4ArrayLike(worldToIndex)) {
    // vtk.js mat4 layout is column-major.
    const vi = worldToIndex[0] * wx + worldToIndex[4] * wy + worldToIndex[8] * wz + worldToIndex[12];
    const vj = worldToIndex[1] * wx + worldToIndex[5] * wy + worldToIndex[9] * wz + worldToIndex[13];
    const vk = worldToIndex[2] * wx + worldToIndex[6] * wy + worldToIndex[10] * wz + worldToIndex[14];
    return [vi, vj, vk];
  }

  const rx = wx - origin[0];
  const ry = wy - origin[1];
  const rz = wz - origin[2];

  const vi = (invDir[0] * rx + invDir[1] * ry + invDir[2] * rz) / spacing[0];
  const vj = (invDir[3] * rx + invDir[4] * ry + invDir[5] * rz) / spacing[1];
  const vk = (invDir[6] * rx + invDir[7] * ry + invDir[8] * rz) / spacing[2];

  return [vi, vj, vk];
}

function invertMatrix3(m: number[]): number[] {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

function indexToWorld(
  i: number,
  j: number,
  k: number,
  origin: [number, number, number],
  spacing: [number, number, number],
  direction: number[]
): [number, number, number] {
  const sx = i * spacing[0];
  const sy = j * spacing[1];
  const sz = k * spacing[2];

  return [
    origin[0] + direction[0] * sx + direction[3] * sy + direction[6] * sz,
    origin[1] + direction[1] * sx + direction[4] * sy + direction[7] * sz,
    origin[2] + direction[2] * sx + direction[5] * sy + direction[8] * sz,
  ];
}

function trilinear(
  data: Float32Array | Int16Array,
  dims: [number, number, number],
  vi: number,
  vj: number,
  vk: number,
  oobValue: number = -1000,
  normalizeSample?: (value: number) => number
): number {
  const [nx, ny, nz] = dims;

  if (vi < 0 || vj < 0 || vk < 0 || vi >= nx - 1 || vj >= ny - 1 || vk >= nz - 1) {
    const ci = Math.max(0, Math.min(nx - 1.001, vi));
    const cj = Math.max(0, Math.min(ny - 1.001, vj));
    const ck = Math.max(0, Math.min(nz - 1.001, vk));

    if (vi < -0.5 || vj < -0.5 || vk < -0.5 || vi > nx - 0.5 || vj > ny - 0.5 || vk > nz - 0.5) {
      return oobValue;
    }

    return trilinearCore(data, nx, ny, ci, cj, ck, normalizeSample);
  }

  return trilinearCore(data, nx, ny, vi, vj, vk, normalizeSample);
}

function trilinearCore(
  data: Float32Array | Int16Array,
  nx: number,
  ny: number,
  vi: number,
  vj: number,
  vk: number,
  normalizeSample?: (value: number) => number
): number {
  const i0 = Math.floor(vi);
  const j0 = Math.floor(vj);
  const k0 = Math.floor(vk);
  const i1 = i0 + 1;
  const j1 = j0 + 1;
  const k1 = k0 + 1;

  const fi = vi - i0;
  const fj = vj - j0;
  const fk = vk - k0;

  const sliceStride = nx * ny;

  const c000 = k0 * sliceStride + j0 * nx + i0;
  const c100 = k0 * sliceStride + j0 * nx + i1;
  const c010 = k0 * sliceStride + j1 * nx + i0;
  const c110 = k0 * sliceStride + j1 * nx + i1;
  const c001 = k1 * sliceStride + j0 * nx + i0;
  const c101 = k1 * sliceStride + j0 * nx + i1;
  const c011 = k1 * sliceStride + j1 * nx + i0;
  const c111 = k1 * sliceStride + j1 * nx + i1;

  const read = normalizeSample || ((value: number) => value);
  const v000 = read(data[c000]);
  const v100 = read(data[c100]);
  const v010 = read(data[c010]);
  const v110 = read(data[c110]);
  const v001 = read(data[c001]);
  const v101 = read(data[c101]);
  const v011 = read(data[c011]);
  const v111 = read(data[c111]);

  const c00 = v000 + fi * (v100 - v000);
  const c10 = v010 + fi * (v110 - v010);
  const c01 = v001 + fi * (v101 - v001);
  const c11 = v011 + fi * (v111 - v011);

  const c0 = c00 + fj * (c10 - c00);
  const c1 = c01 + fj * (c11 - c01);

  return c0 + fk * (c1 - c0);
}

// applyLightBilateralDenoise has been removed — strict MIP output is clean enough
// from hardware trilinear interpolation and does not require post-processing.

function computeArrayMinMax(buffer: Float32Array): { minValue: number; maxValue: number } {
  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < buffer.length; i++) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < minValue) {
      minValue = value;
    }
    if (value > maxValue) {
      maxValue = value;
    }
  }

  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return { minValue: 0, maxValue: 0 };
  }

  return { minValue, maxValue };
}

function computeOutputSignature(buffer: Float32Array): {
  sampledCount: number;
  checksum: number;
  absChecksum: number;
  first16: number[];
} {
  const sampledCount = Math.min(buffer.length, 4096);
  let checksum = 0;
  let absChecksum = 0;

  for (let i = 0; i < sampledCount; i++) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }
    checksum += value;
    absChecksum += Math.abs(value);
  }

  return {
    sampledCount,
    checksum: Math.round(checksum * 1000) / 1000,
    absChecksum: Math.round(absChecksum * 1000) / 1000,
    first16: Array.from(buffer.subarray(0, Math.min(16, buffer.length))).map(
      value => Math.round(Number(value) * 1000) / 1000
    ),
  };
}

function generatePanorama(input: CPRWorkerInput): {
  pixelData: Float32Array;
  minValue: number;
  maxValue: number;
  lutSamplePreview: number[];
  outputPixelPreview: number[];
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  effectiveVerticalHalfMm: number;
  verticalCenterOffsetMm: number;
  adaptiveVerticalIntervalCount: number;
  effectiveSlabSampleCount: number;
  robustMipTopCount: number;
  diagnosticPayload: Record<string, unknown>;
  outputSignature: {
    sampledCount: number;
    checksum: number;
    absChecksum: number;
    first16: number[];
  };
} {
  const {
    scalarData,
    dimensions,
    spacing,
    origin,
    direction,
    worldToIndex,
    verticalDir,
    frames,
    panoWidth,
    panoHeight,
    vertHalfMm: requestedVertHalfMm,
    verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
    slabHalfThicknessMm,
    slabSamples,
    aggregation,
    allowStoredValueNormalization,
    disableStoredValueNormalization,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    isPreScaled,
  } = input;

  const hasWorldToIndex = isMat4ArrayLike(worldToIndex);
  const invDir = hasWorldToIndex ? [] : invertMatrix3(direction);
  const pixelData = new Float32Array(panoWidth * panoHeight);
  let minValue = Infinity;
  let maxValue = -Infinity;
  const aggregationMode = aggregation === 'MEAN' ? 'MEAN' : 'MIP';
  const isMeanAggregation = aggregationMode === 'MEAN';
  const robustMipTopCount = 1;
  const safeSlope =
    Number.isFinite(rescaleSlope) && Math.abs(Number(rescaleSlope)) > 1e-8 ? Number(rescaleSlope) : 1;
  const safeIntercept = Number.isFinite(rescaleIntercept) ? Number(rescaleIntercept) : 0;
  const safeBitsStored =
    Number.isFinite(bitsStored) && Number(bitsStored) >= 1 ? Math.floor(Number(bitsStored)) : null;
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && Number(bitsAllocated) >= 1 ? Math.floor(Number(bitsAllocated)) : 16;
  const safeHighBit = Number.isFinite(highBit) ? Math.floor(Number(highBit)) : null;
  const safePixelRepresentation =
    Number.isFinite(pixelRepresentation) && (Number(pixelRepresentation) === 0 || Number(pixelRepresentation) === 1)
      ? Number(pixelRepresentation)
      : null;
  const safeIsPreScaled = isPreScaled === true;
  const nominalStoredMax =
    safeBitsStored !== null && safeBitsStored > 0 && safeBitsStored < 31
      ? (1 << safeBitsStored) - 1
      : Number.MAX_SAFE_INTEGER;
  const sampledMinMax = (() => {
    let min = Infinity;
    let max = -Infinity;
    const step = Math.max(1, Math.floor(scalarData.length / 4096));
    for (let i = 0; i < scalarData.length; i += step) {
      const value = Number(scalarData[i]);
      if (!Number.isFinite(value)) {
        continue;
      }
      if (value < min) {
        min = value;
      }
      if (value > max) {
        max = value;
      }
    }
    return {
      min: Number.isFinite(min) ? min : 0,
      max: Number.isFinite(max) ? max : 0,
    };
  })();
  const unsignedPackedArtifactDetected =
    !safeIsPreScaled &&
    scalarData instanceof Int16Array &&
    safeBitsStored !== null &&
    safeBitsStored < 16 &&
    safePixelRepresentation === 0 &&
    (sampledMinMax.min < -1 || sampledMinMax.max > nominalStoredMax + 8);
  const normalizationEligible =
    safeBitsStored !== null &&
    safeBitsStored < 16 &&
    safePixelRepresentation !== null &&
    scalarData instanceof Int16Array;
  // Respect explicit caller policy whenever provided; only fall back to
  // heuristic detection when no explicit normalization preference is given.
  const shouldNormalizeStoredValues =
    normalizationEligible &&
    (disableStoredValueNormalization === true
      ? false
      : typeof allowStoredValueNormalization === 'boolean'
        ? allowStoredValueNormalization
        : unsignedPackedArtifactDetected);
  const normalizedBitsStored = shouldNormalizeStoredValues ? (safeBitsStored as number) : 0;
  const normalizedHighBit = shouldNormalizeStoredValues
    ? Math.max(0, Math.min(safeBitsAllocated - 1, safeHighBit ?? normalizedBitsStored - 1))
    : 0;
  const bitAlignmentShift = shouldNormalizeStoredValues
    ? Math.max(-15, Math.min(15, normalizedHighBit + 1 - normalizedBitsStored))
    : 0;
  const storedMask = shouldNormalizeStoredValues ? (1 << normalizedBitsStored) - 1 : 0;
  const storedSignBit = shouldNormalizeStoredValues ? 1 << (normalizedBitsStored - 1) : 0;
  const storedRange = shouldNormalizeStoredValues ? 1 << normalizedBitsStored : 0;

  const normalizeStoredSample = shouldNormalizeStoredValues
    ? (value: number): number => {
      if (!Number.isFinite(value)) {
        return 0;
      }
      const intValue = Math.round(value);
      const rawU16 = intValue & 0xffff;
      // Align stored bits to LSB using DICOM HighBit before masking/sign extension.
      let aligned = rawU16;
      if (bitAlignmentShift > 0) {
        aligned = rawU16 >>> bitAlignmentShift;
      } else if (bitAlignmentShift < 0) {
        aligned = (rawU16 << -bitAlignmentShift) & 0xffff;
      }
      let normalized = aligned & storedMask;
      if (safePixelRepresentation === 1 && (normalized & storedSignBit) !== 0) {
        normalized -= storedRange;
      }
      return normalized;
    }
    : undefined;
  // Respect caller policy when provided. Some studies already expose HU-like
  // scalarData in the volume cache, even if DICOM metadata has non-identity
  // rescale values. Forcing LUT again can double-shift intensities.
  const hasExplicitApplyModalityLut = typeof input.applyModalityLut === 'boolean';
  const requestedModalityLutApplied = hasExplicitApplyModalityLut
    ? input.applyModalityLut
    : safeSlope !== 1 || safeIntercept !== 0;
  // Respect explicit caller LUT policy. If caller does not provide one,
  // retain legacy heuristic for packed unsigned recovery.
  const shouldApplyModalityLut = hasExplicitApplyModalityLut
    ? requestedModalityLutApplied
    : unsignedPackedArtifactDetected && !disableStoredValueNormalization
      ? !safeIsPreScaled && (safeSlope !== 1 || safeIntercept !== 0)
      : requestedModalityLutApplied;
  const lutSamplePreview: number[] = [];
  const interpolationOobValue = shouldApplyModalityLut ? (-1000 - safeIntercept) / safeSlope : -1000;
  const safeInterpolationOobValue = Number.isFinite(interpolationOobValue) ? interpolationOobValue : -1000;

  // Prefer precomputed vertical direction from orchestrator to keep main thread and worker
  // in exact agreement. Fallback to direction matrix K-axis if missing.
  const effectiveVerticalDir =
    Array.isArray(verticalDir) && verticalDir.length >= 3
      ? normalize3([verticalDir[0], verticalDir[1], verticalDir[2]])
      : normalize3([direction[6] ?? 0, direction[7] ?? 0, direction[8] ?? 1]);
  const vertHalfMm =
    Number.isFinite(requestedVertHalfMm) && Number(requestedVertHalfMm) > 0
      ? Number(requestedVertHalfMm)
      : 15.0;
  const panoHeightDen = Math.max(1, panoHeight - 1);
  const baseSlabSampleCount = Math.max(1, Math.floor(slabSamples));
  const positiveSpacings = spacing.filter(value => Number.isFinite(value) && Number(value) > 0);
  const minVolumeSpacingMm = positiveSpacings.length ? Math.min(...positiveSpacings) : 1;
  const targetSlabStepMm = Math.max(0.2, minVolumeSpacingMm * 0.75);
  const adaptiveSampleCount =
    slabHalfThicknessMm > 0 ? Math.ceil((slabHalfThicknessMm * 2) / targetSlabStepMm) + 1 : 1;
  // For thick slabs (≥1mm), use the requested count directly for GPU/CPU parity.
  // Cap at 32 to match GPU MAX_SLAB.
  const slabSampleCount = slabHalfThicknessMm <= 0.5
    ? baseSlabSampleCount   // thin slab: use exact requested count for max sharpness
    : Math.max(1, Math.min(32, baseSlabSampleCount));
  if (baseSlabSampleCount > 32) {
    console.warn('[cprWorker] Slab sample count clamped from', baseSlabSampleCount, 'to 32');
  }
  const slabStepMm = slabSampleCount > 1 ? (slabHalfThicknessMm * 2) / (slabSampleCount - 1) : 0;
  const [nx, ny, nz] = dimensions;

  // Compute a global center offset along vertical direction to keep the
  // requested vertical sampling window inside the volume bounds as much as possible.
  let verticalCenterOffsetMm = 0;
  let volumeMinProjection = Number.NEGATIVE_INFINITY;
  let volumeMaxProjection = Number.POSITIVE_INFINITY;
  if (nx > 1 && ny > 1 && nz > 1 && Array.isArray(frames) && frames.length > 0) {
    const maxI = nx - 1;
    const maxJ = ny - 1;
    const maxK = nz - 1;
    const cornerIndices: Array<[number, number, number]> = [
      [0, 0, 0],
      [maxI, 0, 0],
      [0, maxJ, 0],
      [0, 0, maxK],
      [maxI, maxJ, 0],
      [maxI, 0, maxK],
      [0, maxJ, maxK],
      [maxI, maxJ, maxK],
    ];
    let minProjection = Infinity;
    let maxProjection = -Infinity;

    for (const [ci, cj, ck] of cornerIndices) {
      const world = indexToWorld(ci, cj, ck, origin, spacing, direction);
      const projection = dot3(world, effectiveVerticalDir);
      if (projection < minProjection) {
        minProjection = projection;
      }
      if (projection > maxProjection) {
        maxProjection = projection;
      }
    }

    if (Number.isFinite(minProjection) && Number.isFinite(maxProjection)) {
      volumeMinProjection = minProjection;
      volumeMaxProjection = maxProjection;
      let minRequiredOffset = -Infinity;
      let maxAllowedOffset = Infinity;

      for (let idx = 0; idx < frames.length; idx++) {
        const frameProjection = dot3(frames[idx].position, effectiveVerticalDir);
        const requiredOffsetForLowerSide = minProjection + vertHalfMm - frameProjection;
        const allowedOffsetForUpperSide = maxProjection - vertHalfMm - frameProjection;
        if (requiredOffsetForLowerSide > minRequiredOffset) {
          minRequiredOffset = requiredOffsetForLowerSide;
        }
        if (allowedOffsetForUpperSide < maxAllowedOffset) {
          maxAllowedOffset = allowedOffsetForUpperSide;
        }
      }

      if (Number.isFinite(minRequiredOffset) && Number.isFinite(maxAllowedOffset)) {
        if (minRequiredOffset <= maxAllowedOffset) {
          verticalCenterOffsetMm = Math.max(
            minRequiredOffset,
            Math.min(0, maxAllowedOffset)
          );
        } else {
          // No single offset can fit the entire window for all frames; choose least-violation midpoint.
          verticalCenterOffsetMm = (minRequiredOffset + maxAllowedOffset) / 2;
        }
      }
    }
  }

  // Clamp only the user-requested additive offset, not the volume-aware offset.
  const baseCenterOffsetLimitMm = Math.min(5, Math.max(2, vertHalfMm * 0.3));
  const requestedCenterOffsetMm =
    Number.isFinite(requestedVerticalCenterOffsetMm) ? Number(requestedVerticalCenterOffsetMm) : 0;
  const clampedRequestedCenterOffsetMm = Math.max(
    -baseCenterOffsetLimitMm,
    Math.min(baseCenterOffsetLimitMm, requestedCenterOffsetMm)
  );
  verticalCenterOffsetMm += clampedRequestedCenterOffsetMm;
  // NOTE: Do NOT re-clamp verticalCenterOffsetMm to ±5mm here.
  // The volume-aware offset computed above (lines 568-631) must be preserved
  // so the vertical window can shift fully to stay inside the volume.

  // Dynamically shrink vertHalfMm when the vertical window would exceed volume bounds.
  let effectiveVerticalHalfMm = vertHalfMm;
  if (Number.isFinite(volumeMinProjection) && Number.isFinite(volumeMaxProjection) &&
    volumeMinProjection < volumeMaxProjection && frames.length > 0) {
    let maxSafeHalf = vertHalfMm;
    for (let idx = 0; idx < frames.length; idx++) {
      const fp = dot3(frames[idx].position, effectiveVerticalDir) + verticalCenterOffsetMm;
      const headroomAbove = volumeMaxProjection - fp;
      const headroomBelow = fp - volumeMinProjection;
      const headroom = Math.min(headroomAbove, headroomBelow);
      if (Number.isFinite(headroom) && headroom < maxSafeHalf) {
        maxSafeHalf = headroom;
      }
    }
    // Allow a slim 2mm OOB margin to avoid clipping right at the edge,
    // but prevent the large gray strips.  Floor at 8mm to keep the image usable.
    effectiveVerticalHalfMm = Math.max(8, Math.min(vertHalfMm, maxSafeHalf + 2));
  }
  const vertStepMm = (effectiveVerticalHalfMm * 2) / panoHeightDen;

  const slabDirs: Array<[number, number, number]> = new Array(panoWidth);
  // Compute per-column curvature for variable-thickness slab (thin slabs only)
  const perColumnSlabHalf: Float32Array = new Float32Array(panoWidth);
  {
    let previousSlabDir: [number, number, number] | null = null;
    for (let col = 0; col < panoWidth; col++) {
      let slabDir = normalize3(frames[col].N_slab);
      if (previousSlabDir && dot3(previousSlabDir, slabDir) < 0) {
        slabDir = [-slabDir[0], -slabDir[1], -slabDir[2]];
      }
      slabDirs[col] = slabDir;
      previousSlabDir = slabDir;
    }
    if (slabHalfThicknessMm <= 0.5) {
      // Variable-thickness: only for thin slabs to avoid over-sampling
      const slabCurvatureScale = slabHalfThicknessMm * 4;
      for (let col = 0; col < panoWidth; col++) {
        let curvature = 0;
        if (col > 0 && col < panoWidth - 1) {
          const T_prev = frames[col - 1].T;
          const T_next = frames[col + 1].T;
          const tangentDot = Math.max(-1, Math.min(1,
            T_prev[0] * T_next[0] + T_prev[1] * T_next[1] + T_prev[2] * T_next[2]
          ));
          curvature = Math.acos(tangentDot);
        }
        const curvatureBoost = Math.min(slabHalfThicknessMm, curvature * slabCurvatureScale);
        perColumnSlabHalf[col] = slabHalfThicknessMm + curvatureBoost;
      }
    } else {
      // Thick slabs: use fixed thickness (variable would oversample outside the arch)
      for (let col = 0; col < panoWidth; col++) {
        perColumnSlabHalf[col] = slabHalfThicknessMm;
      }
    }
  }

  const _dbgFirstFive: Array<{ row: number; vi: number; vj: number; vk: number; sample: number }> = [];
  let _dbgChecked = 0;
  let _dbgOob = 0;
  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];

    for (let row = 0; row < panoHeight; row++) {
      const vertOffsetMm = verticalCenterOffsetMm + (effectiveVerticalHalfMm - row * vertStepMm);

      const bx = px + vertOffsetMm * effectiveVerticalDir[0];
      const by = py + vertOffsetMm * effectiveVerticalDir[1];
      const bz = pz + vertOffsetMm * effectiveVerticalDir[2];

      let accumulator = 0;
      let max1 = -Infinity;
      let max2 = -Infinity;
      let max3 = -Infinity;
      let max4 = -Infinity;
      // Gaussian-weighted mean: σ = 60% of slab half-thickness (matches GPU shader)
      const meanSamples = isMeanAggregation ? new Float32Array(slabSampleCount) : null;
      const meanWeights = isMeanAggregation ? new Float32Array(slabSampleCount) : null;
      const gaussSigma = Math.max(0.3, perColumnSlabHalf[col] * 0.6);
      const gaussSigmaSq2 = 2 * gaussSigma * gaussSigma;

      for (let s = 0; s < slabSampleCount; s++) {
        const colSlabHalf = perColumnSlabHalf[col];
        const colSlabStep = slabSampleCount > 1 ? (colSlabHalf * 2) / (slabSampleCount - 1) : 0;
        const slabOffset = slabSampleCount > 1 ? -colSlabHalf + s * colSlabStep : 0;

        const sx = bx + slabOffset * slabDirX;
        const sy = by + slabOffset * slabDirY;
        const sz = bz + slabOffset * slabDirZ;

        const [vi, vj, vk] = worldToVoxel(sx, sy, sz, origin, spacing, invDir, worldToIndex);
        if (col === 0 && row < 5 && s === 0) {
          _dbgFirstFive.push({
            row,
            vi: Math.round(vi * 100) / 100,
            vj: Math.round(vj * 100) / 100,
            vk: Math.round(vk * 100) / 100,
            sample: 0,
          });
        }
        if (_dbgChecked < 200) {
          const _isOob = vi < -0.5 || vj < -0.5 || vk < -0.5 ||
            vi > dimensions[0] - 0.5 || vj > dimensions[1] - 0.5 || vk > dimensions[2] - 0.5;
          _dbgChecked++;
          if (_isOob) _dbgOob++;
        }
        let sample = trilinear(scalarData, dimensions, vi, vj, vk, safeInterpolationOobValue, normalizeStoredSample);
        if (shouldApplyModalityLut) {
          sample = sample * safeSlope + safeIntercept;
          if (lutSamplePreview.length < 5 && Number.isFinite(sample)) {
            lutSamplePreview.push(sample);
          }
        }
        if (!Number.isFinite(sample)) sample = -1000;

        // Wide safety clamp — matches GPU shader [-3000, 10000]
        sample = Math.max(-3000, Math.min(10000, sample));

        if (isMeanAggregation) {
          const gw = Math.exp(-(slabOffset * slabOffset) / gaussSigmaSq2);
          meanSamples![s] = sample;
          meanWeights![s] = gw;
        } else if (sample >= max1) {
          max4 = max3; max3 = max2; max2 = max1; max1 = sample;
        } else if (sample >= max2) {
          max4 = max3; max3 = max2; max2 = sample;
        } else if (sample >= max3) {
          max4 = max3; max3 = sample;
        } else if (sample > max4) {
          max4 = sample;
        }
      }

      let pixelValueRaw: number;
      if (isMeanAggregation) {
        // Winsorize top ~8% to suppress metal/enamel spikes (matches GPU shader)
        const trimHi = Math.max(1, Math.floor(slabSampleCount / 12));
        const winsorIdx = slabSampleCount - 1 - trimHi;
        const idxBuf = new Uint8Array(slabSampleCount);
        for (let i = 0; i < slabSampleCount; i++) idxBuf[i] = i;
        for (let i = 1; i < slabSampleCount; i++) {
          const keyVal = meanSamples![i];
          const keyIdx = idxBuf[i];
          let j = i - 1;
          while (j >= 0 && meanSamples![idxBuf[j]] > keyVal) {
            idxBuf[j + 1] = idxBuf[j];
            j--;
          }
          idxBuf[j + 1] = keyIdx;
        }
        // Winsorize: cap high values to the 92nd percentile value
        if (winsorIdx >= 0 && winsorIdx < slabSampleCount) {
          const winsorCap = meanSamples![idxBuf[winsorIdx]];
          for (let i = winsorIdx + 1; i < slabSampleCount; i++) {
            meanSamples![idxBuf[i]] = winsorCap;
          }
        }
        // Gaussian-weighted mean of winsorized samples
        let wSum = 0;
        let wTotal = 0;
        for (let i = 0; i < slabSampleCount; i++) {
          const idx = idxBuf[i];
          wSum += meanSamples![idx] * meanWeights![idx];
          wTotal += meanWeights![idx];
        }
        pixelValueRaw = wTotal > 0 ? wSum / wTotal : meanSamples![Math.floor(slabSampleCount / 2)];
      } else {
        // Strict MIP: take the maximum value (matches GPU shader behavior)
        pixelValueRaw = max1;
      }
      const pixelValue = Number.isFinite(pixelValueRaw) ? pixelValueRaw : -1000;

      const pixelIndex = row * panoWidth + col;
      pixelData[pixelIndex] = pixelValue;

      if (pixelValue < minValue) minValue = pixelValue;
      if (pixelValue > maxValue) maxValue = pixelValue;
    }
  }

  // No post-processing: strict MIP output is clean enough from trilinear interpolation.

  const diagnosticPayload = {
    oobRate: {
      checked: _dbgChecked,
      oob: _dbgOob,
      oobPercent: _dbgChecked > 0 ? Math.round((_dbgOob / _dbgChecked) * 100) + '%' : '0%',
    },
    firstFiveVoxelIndices: _dbgFirstFive,
    volumeDimensions: dimensions,
    volumeOrigin: origin,
    volumeSpacing: spacing,
    effectiveVerticalDir,
    slabDir: slabDirs?.[0]
      ? {
        N_slab: slabDirs[0],
        tangent: frames[0].T ?? null,
      }
      : null,
    verticalWindowMode: 'global-fixed-window',
    verticalHalfMm: effectiveVerticalHalfMm,
    verticalCenterOffsetMm,
    requestedVerticalCenterOffsetMm: clampedRequestedCenterOffsetMm,
    slabSampling: {
      requestedSamples: baseSlabSampleCount,
      effectiveSamples: slabSampleCount,
      slabHalfThicknessMm,
      aggregation: aggregationMode,
    },
    postProcessing: 'none',
    firstFrameWorldPos: frames?.[0]?.position ?? null,
    lastFrameWorldPos: frames?.[frames.length - 1]?.position ?? null,
  };
  console.warn('[CPR-DIAGNOSTIC]', diagnosticPayload);
  console.warn('[CPR-DIAGNOSTIC-JSON]', JSON.stringify(diagnosticPayload));

  const outputPixelPreview = Array.from(pixelData.subarray(0, Math.min(5, pixelData.length)));
  const outputSignature = computeOutputSignature(pixelData);
  return {
    pixelData,
    minValue,
    maxValue,
    lutSamplePreview,
    outputPixelPreview,
    modalityLutApplied: shouldApplyModalityLut,
    requestedModalityLutApplied,
    storedValueNormalizationApplied: shouldNormalizeStoredValues,
    unsignedPackedArtifactDetected,
    effectiveVerticalHalfMm,
    verticalCenterOffsetMm,
    adaptiveVerticalIntervalCount: 1,
    effectiveSlabSampleCount: slabSampleCount,
    robustMipTopCount,
    diagnosticPayload,
    outputSignature,
  };
}

// eslint-disable-next-line no-restricted-globals
self.onmessage = function (event: MessageEvent<CPRWorkerInput>) {
  try {
    const input = event.data;
    const workerTag = input.debugRunId ? `[cprWorker:${input.debugRunId}]` : '[cprWorker]';
    const safeSlope =
      Number.isFinite(input.rescaleSlope) && Math.abs(Number(input.rescaleSlope)) > 1e-8
        ? Number(input.rescaleSlope)
        : 1;
    const safeIntercept = Number.isFinite(input.rescaleIntercept)
      ? Number(input.rescaleIntercept)
      : 0;
    const modalityLutFlagMismatch =
      input.applyModalityLut === false && (safeSlope !== 1 || safeIntercept !== 0);

    if (!input.scalarData || input.scalarData.length === 0) {
      throw new Error('Received empty or null scalar data.');
    }
    if (!input.frames || input.frames.length === 0) {
      throw new Error('Received empty frames array.');
    }

    const bufferType = input.isSharedArrayBuffer ? 'SharedArrayBuffer' : 'ArrayBuffer (cloned)';
    console.debug(`${workerTag} Starting panorama generation. Buffer: ${bufferType}`);
    console.debug(`${workerTag} Volume dims: ${input.dimensions}  Frames: ${input.frames.length}`);
    console.debug(
      `${workerTag} Output: ${input.panoWidth}x${input.panoHeight}  Slab: ${input.slabSamples} samples`
    );

    const start = performance.now();
    const {
      pixelData,
      minValue,
      maxValue,
      lutSamplePreview,
      outputPixelPreview,
      modalityLutApplied,
      requestedModalityLutApplied,
      storedValueNormalizationApplied,
      unsignedPackedArtifactDetected,
      effectiveVerticalHalfMm,
      verticalCenterOffsetMm,
      adaptiveVerticalIntervalCount,
      effectiveSlabSampleCount,
      robustMipTopCount,
      diagnosticPayload,
      outputSignature,
    } = generatePanorama(input);
    const elapsed = (performance.now() - start).toFixed(0);

    console.debug(
      `${workerTag} Done in ${elapsed}ms. Range: [${minValue.toFixed(0)}, ${maxValue.toFixed(0)}] HU`
    );
    if (modalityLutApplied) {
      console.log(
        `${workerTag} LUT preview (first ${lutSamplePreview.length} converted samples):`,
        lutSamplePreview
      );
    }
    if (storedValueNormalizationApplied) {
      console.log(`${workerTag} Applied bitsStored/pixelRepresentation normalization before interpolation.`, {
        bitsStored: input.bitsStored,
        bitsAllocated: input.bitsAllocated,
        highBit: input.highBit,
        pixelRepresentation: input.pixelRepresentation,
      });
    }
    if (requestedModalityLutApplied !== modalityLutApplied) {
      console.warn(`${workerTag} LUT policy adjusted in worker`, {
        requestedApplyModalityLut: requestedModalityLutApplied,
        appliedModalityLut: modalityLutApplied,
        unsignedPackedArtifactDetected,
      });
    }
    console.debug(`${workerTag} Vertical sampling window`, {
      mode: 'global-fixed-window',
      configuredVertHalfMm: input.vertHalfMm,
      effectiveVerticalHalfMm,
      verticalCenterOffsetMm,
      sampledIntervals: adaptiveVerticalIntervalCount,
      requestedSlabSamples: input.slabSamples,
      effectiveSlabSampleCount,
      robustMipTopCount,
    });
    console.log(
      `${workerTag} Output preview (first ${outputPixelPreview.length} pano pixels):`,
      outputPixelPreview
    );
    console.log('[CPR-WORKER-JSON]', JSON.stringify({
      runId: input.debugRunId ?? null,
      diagnostic: diagnosticPayload,
      outputSignature,
      minValue,
      maxValue,
      effectiveVerticalHalfMm,
      verticalCenterOffsetMm,
      effectiveSlabSampleCount,
      robustMipTopCount,
    }));
    if (modalityLutFlagMismatch) {
      console.warn(
        `${workerTag} applyModalityLut=false with non-identity rescale metadata. ` +
        'Worker respected caller policy to avoid double-rescale.'
      );
    }

    const response: CPRWorkerSuccess = {
      type: 'SUCCESS',
      pixelData,
      panoWidth: input.panoWidth,
      panoHeight: input.panoHeight,
      minValue,
      maxValue,
      modalityLutApplied,
      requestedModalityLutApplied,
      storedValueNormalizationApplied,
      unsignedPackedArtifactDetected,
      debugPayload: {
        diagnostic: diagnosticPayload,
        outputSignature,
      },
    };

    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response, [pixelData.buffer]);
  } catch (err) {
    const response: CPRWorkerError = {
      type: 'ERROR',
      message: err instanceof Error ? err.message : String(err),
    };
    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response);
  }
};
