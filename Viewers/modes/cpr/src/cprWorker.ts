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

function applyLightBilateralDenoise(
  pixelData: Float32Array,
  width: number,
  height: number,
  blendWeight: number
): boolean {
  if (width < 3 || height < 3 || blendWeight <= 0) {
    return false;
  }

  const source = new Float32Array(pixelData);
  const sigmaRange = 220;
  const sigmaRangeDen = 2 * sigmaRange * sigmaRange;
  const clampedBlend = Math.max(0, Math.min(0.6, blendWeight));
  const keepWeight = 1 - clampedBlend;

  const neighbors: Array<[number, number, number]> = [
    [-1, -1, 0.45],
    [0, -1, 0.7],
    [1, -1, 0.45],
    [-1, 0, 0.7],
    [1, 0, 0.7],
    [-1, 1, 0.45],
    [0, 1, 0.7],
    [1, 1, 0.45],
  ];

  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      const index = row * width + col;
      const center = source[index];
      if (!Number.isFinite(center)) {
        continue;
      }

      let weightedSum = center;
      let weightTotal = 1;

      for (let n = 0; n < neighbors.length; n++) {
        const [dx, dy, spatialWeight] = neighbors[n];
        const neighborValue = source[(row + dy) * width + (col + dx)];
        if (!Number.isFinite(neighborValue)) {
          continue;
        }
        const delta = neighborValue - center;
        const rangeWeight = Math.exp(-(delta * delta) / sigmaRangeDen);
        const weight = spatialWeight * rangeWeight;
        weightedSum += neighborValue * weight;
        weightTotal += weight;
      }

      const filtered = weightTotal > 0 ? weightedSum / weightTotal : center;
      pixelData[index] = keepWeight * center + clampedBlend * filtered;
    }
  }

  return true;
}

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

function sortSamplePairsAscending(
  values: Float32Array,
  weights: Float32Array,
  count: number
): void {
  for (let i = 1; i < count; i++) {
    const keyValue = values[i];
    const keyWeight = weights[i];
    let j = i - 1;
    while (j >= 0 && values[j] > keyValue) {
      values[j + 1] = values[j];
      weights[j + 1] = weights[j];
      j--;
    }
    values[j + 1] = keyValue;
    weights[j + 1] = keyWeight;
  }
}

function computeWinsorizedWeightedMean(
  values: Float32Array,
  weights: Float32Array,
  count: number
): number {
  if (count <= 0) {
    return -1000;
  }

  const trimHi = count >= 6 ? Math.max(1, Math.floor(count / 12)) : 0;
  const winsorIndex = count - 1 - trimHi;
  if (trimHi > 0 && winsorIndex >= 0 && winsorIndex < count) {
    const winsorCap = values[winsorIndex];
    for (let i = winsorIndex + 1; i < count; i++) {
      values[i] = winsorCap;
    }
  }

  let weightedSum = 0;
  let weightTotal = 0;
  for (let i = 0; i < count; i++) {
    const weight = Number.isFinite(weights[i]) && weights[i] > 0 ? weights[i] : 1;
    weightedSum += values[i] * weight;
    weightTotal += weight;
  }

  return weightTotal > 0 ? weightedSum / weightTotal : values[count - 1];
}

function computeWeightedHighBandMean(
  values: Float32Array,
  weights: Float32Array,
  count: number,
  topCount: number
): number {
  if (count <= 0) {
    return -1000;
  }

  if (count > 1 && Number.isFinite(values[count - 2]) && values[count - 1] - values[count - 2] > 850) {
    values[count - 1] = 0.5 * (values[count - 1] + values[count - 2]);
  }

  const effectiveTopCount = Math.max(1, Math.min(count, topCount));
  const startIndex = count - effectiveTopCount;
  let weightedSum = 0;
  let weightTotal = 0;

  for (let i = startIndex; i < count; i++) {
    const weight = Number.isFinite(weights[i]) && weights[i] > 0 ? weights[i] : 1;
    weightedSum += values[i] * weight;
    weightTotal += weight;
  }

  return weightTotal > 0 ? weightedSum / weightTotal : values[count - 1];
}

function clampNumber(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function smoothFloatSeries(
  values: Float32Array,
  count: number,
  scratch: Float32Array,
  passes: number = 1
): void {
  if (count <= 2 || passes <= 0) {
    return;
  }

  let source = values;
  let target = scratch;

  for (let pass = 0; pass < passes; pass++) {
    target[0] = source[0];
    for (let i = 1; i < count - 1; i++) {
      target[i] = source[i - 1] * 0.25 + source[i] * 0.5 + source[i + 1] * 0.25;
    }
    target[count - 1] = source[count - 1];

    const nextSource = target;
    target = source;
    source = nextSource;
  }

  if (source !== values) {
    values.set(source.subarray(0, count), 0);
  }
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
  denoiseApplied: boolean;
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
  let robustMipTopCount = 1;
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
  const slabSampleCount = Math.max(
    1,
    Math.min(
      13,
      Math.max(
        baseSlabSampleCount,
        Math.min(adaptiveSampleCount, baseSlabSampleCount + 2)
      )
    )
  );
  if (!isMeanAggregation) {
    robustMipTopCount =
      slabSampleCount <= 1
        ? 1
        : slabSampleCount >= 13
          ? 5
          : slabSampleCount >= 9
            ? 4
            : slabSampleCount >= 7
              ? 3
              : 2;
  }
  const slabStepMm = slabSampleCount > 1 ? (slabHalfThicknessMm * 2) / (slabSampleCount - 1) : 0;
  const focalTroughSigmaMm =
    slabHalfThicknessMm > 0
      ? Math.max(0.35, minVolumeSpacingMm * 0.6, slabHalfThicknessMm * (isMeanAggregation ? 0.52 : 0.48))
      : 0.35;
  const focalTroughSigmaSq2 = 2 * focalTroughSigmaMm * focalTroughSigmaMm;
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

  const baseCenterOffsetLimitMm = Math.min(5, Math.max(2, vertHalfMm * 0.3));
  const requestedCenterOffsetMm =
    Number.isFinite(requestedVerticalCenterOffsetMm) ? Number(requestedVerticalCenterOffsetMm) : 0;
  const clampedRequestedCenterOffsetMm = Math.max(
    -baseCenterOffsetLimitMm,
    Math.min(baseCenterOffsetLimitMm, requestedCenterOffsetMm)
  );
  verticalCenterOffsetMm += clampedRequestedCenterOffsetMm;
  verticalCenterOffsetMm = Math.max(
    -baseCenterOffsetLimitMm,
    Math.min(baseCenterOffsetLimitMm, verticalCenterOffsetMm)
  );
  const effectiveVerticalHalfMm = vertHalfMm;
  const vertStepMm = (effectiveVerticalHalfMm * 2) / panoHeightDen;

  const slabDirs: Array<[number, number, number]> = new Array(panoWidth);
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
  }

  const _dbgFirstFive: Array<{ row: number; vi: number; vj: number; vk: number; sample: number }> = [];
  let _dbgChecked = 0;
  let _dbgOob = 0;
  const slabValueBuffer = new Float32Array(slabSampleCount);
  const slabWeightBuffer = new Float32Array(slabSampleCount);
  const adaptiveCenterSearchHalfRangeMm = Math.min(
    isMeanAggregation ? 2.4 : 1.8,
    Math.max(isMeanAggregation ? 0.9 : 0.7, effectiveVerticalHalfMm * (isMeanAggregation ? 0.16 : 0.11))
  );
  const adaptiveCandidateCount = Math.max(
    5,
    Math.min(
      7,
      Math.round((adaptiveCenterSearchHalfRangeMm * 2) / Math.max(0.75, minVolumeSpacingMm * 4)) + 1
    )
  );
  const adaptiveProfileSampleCount = 13;
  const adaptiveProfileLowerBandCount = Math.max(3, Math.floor(adaptiveProfileSampleCount * 0.25));
  const adaptiveProfileUpperBandCount = Math.max(5, Math.ceil(adaptiveProfileSampleCount * 0.55));
  const adaptiveCenterSmoothingRadiusCols = Math.max(8, Math.min(18, Math.round(panoWidth / 40)));
  const adaptiveCenterGlobalBlend = isMeanAggregation ? 0.34 : 0.42;
  const adaptiveCenterMaxDeviationMm = Math.min(
    isMeanAggregation ? 1.35 : 1.1,
    Math.max(isMeanAggregation ? 0.75 : 0.6, effectiveVerticalHalfMm * (isMeanAggregation ? 0.1 : 0.08))
  );
  const adaptiveCenterMaxAdjacentDeltaMm = Math.max(
    minVolumeSpacingMm * 0.45,
    Math.min(isMeanAggregation ? 0.18 : 0.15, minVolumeSpacingMm * (isMeanAggregation ? 0.9 : 0.75))
  );
  const adaptiveContinuityPenaltyScale = isMeanAggregation ? 18 : 14;
  const profileSampleBuffer = new Float32Array(adaptiveProfileSampleCount);
  const profileSampleScratch = new Float32Array(adaptiveProfileSampleCount);
  const localCenterOffsetsMm = new Float32Array(panoWidth);
  const localCenterMinOffsetsMm = new Float32Array(panoWidth);
  const localCenterMaxOffsetsMm = new Float32Array(panoWidth);
  const localCenterScratch = new Float32Array(panoWidth);
  const sampleReducedPoint = (
    bx: number,
    by: number,
    bz: number,
    slabDirX: number,
    slabDirY: number,
    slabDirZ: number,
    recordOobStats: boolean,
    debugCaptureRow: number | null
  ): number => {
    let validSampleCount = 0;

    for (let s = 0; s < slabSampleCount; s++) {
      const slabOffset = slabSampleCount > 1 ? -slabHalfThicknessMm + s * slabStepMm : 0;

      const sx = bx + slabOffset * slabDirX;
      const sy = by + slabOffset * slabDirY;
      const sz = bz + slabOffset * slabDirZ;

      const [vi, vj, vk] = worldToVoxel(sx, sy, sz, origin, spacing, invDir, worldToIndex);

      if (debugCaptureRow !== null && s === 0) {
        _dbgFirstFive.push({
          row: debugCaptureRow,
          vi: Math.round(vi * 100) / 100,
          vj: Math.round(vj * 100) / 100,
          vk: Math.round(vk * 100) / 100,
          sample: 0,
        });
      }

      if (recordOobStats && _dbgChecked < 200) {
        const isOob =
          vi < -0.5 ||
          vj < -0.5 ||
          vk < -0.5 ||
          vi > dimensions[0] - 0.5 ||
          vj > dimensions[1] - 0.5 ||
          vk > dimensions[2] - 0.5;
        _dbgChecked++;
        if (isOob) {
          _dbgOob++;
        }
      }

      if (
        vi < -0.5 ||
        vj < -0.5 ||
        vk < -0.5 ||
        vi > dimensions[0] - 0.5 ||
        vj > dimensions[1] - 0.5 ||
        vk > dimensions[2] - 0.5
      ) {
        continue;
      }

      let sample = trilinear(
        scalarData,
        dimensions,
        vi,
        vj,
        vk,
        safeInterpolationOobValue,
        normalizeStoredSample
      );
      if (shouldApplyModalityLut) {
        sample = sample * safeSlope + safeIntercept;
        if (lutSamplePreview.length < 5 && Number.isFinite(sample)) {
          lutSamplePreview.push(sample);
        }
      }
      if (!Number.isFinite(sample)) {
        sample = -1000;
      }

      const focalWeight =
        slabSampleCount > 1 ? Math.exp(-(slabOffset * slabOffset) / focalTroughSigmaSq2) : 1;
      slabValueBuffer[validSampleCount] = sample;
      slabWeightBuffer[validSampleCount] = focalWeight;
      validSampleCount++;
    }

    if (validSampleCount <= 0) {
      return -1000;
    }

    sortSamplePairsAscending(slabValueBuffer, slabWeightBuffer, validSampleCount);
    if (isMeanAggregation) {
      return computeWinsorizedWeightedMean(slabValueBuffer, slabWeightBuffer, validSampleCount);
    }

    return computeWeightedHighBandMean(
      slabValueBuffer,
      slabWeightBuffer,
      validSampleCount,
      robustMipTopCount
    );
  };

  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
    const frameProjection = dot3(frame.position, effectiveVerticalDir);
    const perFrameMinCenterOffsetMm = Number.isFinite(volumeMinProjection)
      ? volumeMinProjection + effectiveVerticalHalfMm - frameProjection
      : verticalCenterOffsetMm - adaptiveCenterSearchHalfRangeMm;
    const perFrameMaxCenterOffsetMm = Number.isFinite(volumeMaxProjection)
      ? volumeMaxProjection - effectiveVerticalHalfMm - frameProjection
      : verticalCenterOffsetMm + adaptiveCenterSearchHalfRangeMm;
    const searchMinCenterOffsetMm = Math.max(
      perFrameMinCenterOffsetMm,
      verticalCenterOffsetMm - adaptiveCenterSearchHalfRangeMm
    );
    const searchMaxCenterOffsetMm = Math.min(
      perFrameMaxCenterOffsetMm,
      verticalCenterOffsetMm + adaptiveCenterSearchHalfRangeMm
    );

    const safeSearchMin = Number.isFinite(searchMinCenterOffsetMm)
      ? searchMinCenterOffsetMm
      : verticalCenterOffsetMm;
    const safeSearchMax = Number.isFinite(searchMaxCenterOffsetMm)
      ? searchMaxCenterOffsetMm
      : verticalCenterOffsetMm;
    const clampedFallbackCenterOffsetMm = clampNumber(
      verticalCenterOffsetMm,
      Math.min(safeSearchMin, safeSearchMax),
      Math.max(safeSearchMin, safeSearchMax)
    );

    localCenterMinOffsetsMm[col] = Math.min(safeSearchMin, safeSearchMax);
    localCenterMaxOffsetsMm[col] = Math.max(safeSearchMin, safeSearchMax);

    if (safeSearchMin >= safeSearchMax - 1e-4) {
      localCenterOffsetsMm[col] = clampedFallbackCenterOffsetMm;
      continue;
    }

    let bestCenterOffsetMm = clampedFallbackCenterOffsetMm;
    let bestCandidateScore = Number.NEGATIVE_INFINITY;
    const previousCenterOffsetMm =
      col > 0 ? Number(localCenterOffsetsMm[col - 1]) : clampedFallbackCenterOffsetMm;

    for (let candidateIndex = 0; candidateIndex < adaptiveCandidateCount; candidateIndex++) {
      const t =
        adaptiveCandidateCount <= 1 ? 0.5 : candidateIndex / Math.max(1, adaptiveCandidateCount - 1);
      const candidateCenterOffsetMm =
        safeSearchMin + (safeSearchMax - safeSearchMin) * t;

      for (let sampleIndex = 0; sampleIndex < adaptiveProfileSampleCount; sampleIndex++) {
        const fraction =
          adaptiveProfileSampleCount <= 1
            ? 0.5
            : sampleIndex / Math.max(1, adaptiveProfileSampleCount - 1);
        const relativeOffsetMm = effectiveVerticalHalfMm - fraction * (effectiveVerticalHalfMm * 2);
        const sampleOffsetMm = candidateCenterOffsetMm + relativeOffsetMm;
        const bx = px + sampleOffsetMm * effectiveVerticalDir[0];
        const by = py + sampleOffsetMm * effectiveVerticalDir[1];
        const bz = pz + sampleOffsetMm * effectiveVerticalDir[2];
        profileSampleBuffer[sampleIndex] = sampleReducedPoint(
          bx,
          by,
          bz,
          slabDirX,
          slabDirY,
          slabDirZ,
          false,
          null
        );
      }

      smoothFloatSeries(
        profileSampleBuffer,
        adaptiveProfileSampleCount,
        profileSampleScratch,
        isMeanAggregation ? 2 : 1
      );

      const profileValues = Array.from(profileSampleBuffer.subarray(0, adaptiveProfileSampleCount));
      const lowerBandValues = profileValues.slice(adaptiveProfileSampleCount - adaptiveProfileLowerBandCount);
      const upperBandValues = profileValues.slice(0, adaptiveProfileUpperBandCount);
      const profileP20 = percentile(profileValues, 0.2);
      const profileP50 = percentile(profileValues, 0.5);
      const profileP80 = percentile(profileValues, 0.8);
      const lowerBandP50 = percentile(lowerBandValues, 0.5);
      const upperBandMax = upperBandValues.length ? Math.max(...upperBandValues) : profileP80;

      let detailDeltaAccum = 0;
      let detailDeltaCount = 0;
      const detailStartIndex = Math.max(1, Math.floor(adaptiveProfileSampleCount * 0.08));
      const detailEndIndex = Math.min(
        adaptiveProfileSampleCount - 1,
        Math.ceil(adaptiveProfileSampleCount * 0.78)
      );
      for (let detailIndex = detailStartIndex; detailIndex < detailEndIndex; detailIndex++) {
        detailDeltaAccum += Math.abs(
          profileSampleBuffer[detailIndex] - profileSampleBuffer[detailIndex - 1]
        );
        detailDeltaCount++;
      }
      const meanDetailDelta = detailDeltaCount > 0 ? detailDeltaAccum / detailDeltaCount : 0;

      let lowerBandBrightCount = 0;
      for (let lowerIndex = 0; lowerIndex < lowerBandValues.length; lowerIndex++) {
        if (lowerBandValues[lowerIndex] > -200) {
          lowerBandBrightCount++;
        }
      }
      const lowerBandBrightFraction =
        lowerBandValues.length > 0 ? lowerBandBrightCount / lowerBandValues.length : 0;

      const candidateScore =
        Math.max(0, profileP80 - profileP20) * 0.36 +
        meanDetailDelta * 0.22 +
        Math.max(0, upperBandMax - profileP50) * 0.12 -
        Math.max(0, lowerBandP50 + 200) * 1.55 -
        lowerBandBrightFraction * 280 -
        Math.abs(candidateCenterOffsetMm - verticalCenterOffsetMm) * 26 -
        Math.abs(candidateCenterOffsetMm - previousCenterOffsetMm) * adaptiveContinuityPenaltyScale;

      if (candidateScore > bestCandidateScore) {
        bestCandidateScore = candidateScore;
        bestCenterOffsetMm = candidateCenterOffsetMm;
      }
    }

    localCenterOffsetsMm[col] = clampNumber(
      bestCenterOffsetMm,
      localCenterMinOffsetsMm[col],
      localCenterMaxOffsetsMm[col]
    );
  }

  if (panoWidth > 2) {
    for (let col = 0; col < panoWidth; col++) {
      let weightedSum = 0;
      let weightTotal = 0;
      const neighborStart = Math.max(0, col - adaptiveCenterSmoothingRadiusCols);
      const neighborEnd = Math.min(panoWidth - 1, col + adaptiveCenterSmoothingRadiusCols);
      for (let neighbor = neighborStart; neighbor <= neighborEnd; neighbor++) {
        const weight = adaptiveCenterSmoothingRadiusCols + 1 - Math.abs(neighbor - col);
        weightedSum += localCenterOffsetsMm[neighbor] * weight;
        weightTotal += weight;
      }
      const smoothedCenterOffsetMm =
        weightTotal > 0 ? weightedSum / weightTotal : localCenterOffsetsMm[col];
      localCenterScratch[col] = clampNumber(
        smoothedCenterOffsetMm,
        localCenterMinOffsetsMm[col],
        localCenterMaxOffsetsMm[col]
      );
    }
    smoothFloatSeries(localCenterScratch, panoWidth, localCenterOffsetsMm, isMeanAggregation ? 3 : 2);
    localCenterOffsetsMm.set(localCenterScratch);
  }

  for (let col = 0; col < panoWidth; col++) {
    const regularizedMinCenterOffsetMm = Math.max(
      localCenterMinOffsetsMm[col],
      verticalCenterOffsetMm - adaptiveCenterMaxDeviationMm
    );
    const regularizedMaxCenterOffsetMm = Math.min(
      localCenterMaxOffsetsMm[col],
      verticalCenterOffsetMm + adaptiveCenterMaxDeviationMm
    );
    const blendedCenterOffsetMm =
      verticalCenterOffsetMm +
      (Number(localCenterOffsetsMm[col]) - verticalCenterOffsetMm) * adaptiveCenterGlobalBlend;
    localCenterOffsetsMm[col] = clampNumber(
      blendedCenterOffsetMm,
      Math.min(regularizedMinCenterOffsetMm, regularizedMaxCenterOffsetMm),
      Math.max(regularizedMinCenterOffsetMm, regularizedMaxCenterOffsetMm)
    );
  }

  if (panoWidth > 1) {
    for (let pass = 0; pass < 2; pass++) {
      for (let col = 1; col < panoWidth; col++) {
        const regularizedMinCenterOffsetMm = Math.max(
          localCenterMinOffsetsMm[col],
          verticalCenterOffsetMm - adaptiveCenterMaxDeviationMm
        );
        const regularizedMaxCenterOffsetMm = Math.min(
          localCenterMaxOffsetsMm[col],
          verticalCenterOffsetMm + adaptiveCenterMaxDeviationMm
        );
        const minAllowedCenterOffsetMm = Math.max(
          regularizedMinCenterOffsetMm,
          Number(localCenterOffsetsMm[col - 1]) - adaptiveCenterMaxAdjacentDeltaMm
        );
        const maxAllowedCenterOffsetMm = Math.min(
          regularizedMaxCenterOffsetMm,
          Number(localCenterOffsetsMm[col - 1]) + adaptiveCenterMaxAdjacentDeltaMm
        );
        localCenterOffsetsMm[col] = clampNumber(
          Number(localCenterOffsetsMm[col]),
          Math.min(minAllowedCenterOffsetMm, maxAllowedCenterOffsetMm),
          Math.max(minAllowedCenterOffsetMm, maxAllowedCenterOffsetMm)
        );
      }

      for (let col = panoWidth - 2; col >= 0; col--) {
        const regularizedMinCenterOffsetMm = Math.max(
          localCenterMinOffsetsMm[col],
          verticalCenterOffsetMm - adaptiveCenterMaxDeviationMm
        );
        const regularizedMaxCenterOffsetMm = Math.min(
          localCenterMaxOffsetsMm[col],
          verticalCenterOffsetMm + adaptiveCenterMaxDeviationMm
        );
        const minAllowedCenterOffsetMm = Math.max(
          regularizedMinCenterOffsetMm,
          Number(localCenterOffsetsMm[col + 1]) - adaptiveCenterMaxAdjacentDeltaMm
        );
        const maxAllowedCenterOffsetMm = Math.min(
          regularizedMaxCenterOffsetMm,
          Number(localCenterOffsetsMm[col + 1]) + adaptiveCenterMaxAdjacentDeltaMm
        );
        localCenterOffsetsMm[col] = clampNumber(
          Number(localCenterOffsetsMm[col]),
          Math.min(minAllowedCenterOffsetMm, maxAllowedCenterOffsetMm),
          Math.max(minAllowedCenterOffsetMm, maxAllowedCenterOffsetMm)
        );
      }
    }
  }

  let minLocalCenterOffsetMm = Infinity;
  let maxLocalCenterOffsetMm = -Infinity;
  let localCenterOffsetSumMm = 0;
  let maxLocalCenterAdjacentDeltaMm = 0;
  for (let col = 0; col < panoWidth; col++) {
    const localOffset = Number(localCenterOffsetsMm[col]);
    if (localOffset < minLocalCenterOffsetMm) {
      minLocalCenterOffsetMm = localOffset;
    }
    if (localOffset > maxLocalCenterOffsetMm) {
      maxLocalCenterOffsetMm = localOffset;
    }
    localCenterOffsetSumMm += localOffset;
    if (col > 0) {
      maxLocalCenterAdjacentDeltaMm = Math.max(
        maxLocalCenterAdjacentDeltaMm,
        Math.abs(localOffset - Number(localCenterOffsetsMm[col - 1]))
      );
    }
  }
  const meanLocalCenterOffsetMm = panoWidth > 0 ? localCenterOffsetSumMm / panoWidth : verticalCenterOffsetMm;

  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
    const columnVerticalCenterOffsetMm = localCenterOffsetsMm[col];

    for (let row = 0; row < panoHeight; row++) {
      const vertOffsetMm = columnVerticalCenterOffsetMm + (effectiveVerticalHalfMm - row * vertStepMm);

      const bx = px + vertOffsetMm * effectiveVerticalDir[0];
      const by = py + vertOffsetMm * effectiveVerticalDir[1];
      const bz = pz + vertOffsetMm * effectiveVerticalDir[2];
      const pixelValueRaw = sampleReducedPoint(
        bx,
        by,
        bz,
        slabDirX,
        slabDirY,
        slabDirZ,
        true,
        col === 0 && row < 5 ? row : null
      );
      const pixelValue = Number.isFinite(pixelValueRaw) ? pixelValueRaw : -1000;

      const pixelIndex = row * panoWidth + col;
      pixelData[pixelIndex] = pixelValue;

      if (pixelValue < minValue) {
        minValue = pixelValue;
      }
      if (pixelValue > maxValue) {
        maxValue = pixelValue;
      }
    }
  }

  const denoiseBlend = isMeanAggregation
    ? slabHalfThicknessMm <= 0.35
      ? 0
      : slabHalfThicknessMm <= 0.9
        ? 0.04
        : slabHalfThicknessMm <= 1.2
          ? 0.07
          : 0.1
    : 0.12 * Math.min(1, slabHalfThicknessMm / 1.5);
  const denoiseApplied =
    denoiseBlend > 0 && applyLightBilateralDenoise(pixelData, panoWidth, panoHeight, denoiseBlend);
  if (denoiseApplied) {
    const denoisedRange = computeArrayMinMax(pixelData);
    minValue = denoisedRange.minValue;
    maxValue = denoisedRange.maxValue;
  }

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
    verticalWindowMode: 'local-column-adaptive-window',
    verticalHalfMm: effectiveVerticalHalfMm,
    globalVerticalCenterOffsetMm: verticalCenterOffsetMm,
    verticalCenterOffsetMm: meanLocalCenterOffsetMm,
    localCenterOffsetMmStats: {
      min: minLocalCenterOffsetMm,
      max: maxLocalCenterOffsetMm,
      mean: meanLocalCenterOffsetMm,
      maxDeviationFromGlobal: Math.max(
        Math.abs(minLocalCenterOffsetMm - verticalCenterOffsetMm),
        Math.abs(maxLocalCenterOffsetMm - verticalCenterOffsetMm)
      ),
      maxAdjacentDeltaMm: maxLocalCenterAdjacentDeltaMm,
      first8: Array.from(localCenterOffsetsMm.subarray(0, Math.min(8, localCenterOffsetsMm.length))).map(
        value => Math.round(Number(value) * 1000) / 1000
      ),
    },
    requestedVerticalCenterOffsetMm: clampedRequestedCenterOffsetMm,
    adaptiveVerticalSearch: {
      halfRangeMm: adaptiveCenterSearchHalfRangeMm,
      candidateCount: adaptiveCandidateCount,
      profileSamples: adaptiveProfileSampleCount,
      smoothingRadiusCols: adaptiveCenterSmoothingRadiusCols,
      globalBlend: adaptiveCenterGlobalBlend,
      maxDeviationMm: adaptiveCenterMaxDeviationMm,
      maxAdjacentDeltaMm: adaptiveCenterMaxAdjacentDeltaMm,
    },
    slabSampling: {
      requestedSamples: baseSlabSampleCount,
      effectiveSamples: slabSampleCount,
      slabHalfThicknessMm,
      aggregation: aggregationMode,
      focalTroughSigmaMm,
      reduction: isMeanAggregation ? 'winsorized-weighted-mean' : 'weighted-high-band-mean',
    },
    denoise: {
      blend: denoiseBlend,
      applied: denoiseApplied,
    },
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
    verticalCenterOffsetMm: meanLocalCenterOffsetMm,
    adaptiveVerticalIntervalCount: adaptiveCandidateCount,
    effectiveSlabSampleCount: slabSampleCount,
    robustMipTopCount,
    denoiseApplied,
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
      denoiseApplied,
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
      denoiseApplied,
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
      denoiseApplied,
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
