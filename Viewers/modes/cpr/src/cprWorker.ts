import { disposeGpuPanoRenderer, renderPanoGpu } from './cprGpuRenderer';
import type { GpuPanoDebugMaps, GpuPanoDiagnostics } from './cprGpuRenderer';
import {
  createHuScalarTransform,
  normalizeScalarDataToHuFloat32,
  resolveStoredValueNormalizationPolicy,
} from './cprScalarPolicy';

/**
 * cprWorker.ts
 * Web Worker: Generates the 2D panoramic image from a 3D CBCT volume.
 */

interface WorkerFrame {
  position: [number, number, number];
  T?: [number, number, number];
  N_slab: [number, number, number];
  S?: [number, number, number];
}

interface CPRWorkerInput {
  scalarData: Float32Array | Int16Array | Uint16Array;
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
  rigidVerticalSliceMode?: boolean;
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
  attemptLabel?: string;
  reconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
  renderBackend?: 'gpu' | 'cpu';
  allowLegacyFallback?: boolean;
  sourceVolumeId?: string;
}

interface CPRWorkerVolumeState {
  sessionKey: string;
  scalarData: Float32Array | Int16Array | Uint16Array;
  isSharedArrayBuffer: boolean;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
  worldToIndex?: ArrayLike<number>;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  bitsAllocated?: number;
  highBit?: number;
  pixelRepresentation?: number;
  isPreScaled?: boolean;
  storedValueNormalizationApplied?: boolean;
  unsignedPackedArtifactDetected?: boolean;
  normalizationSignature?: string | null;
  preScaledHeuristicOverride?: boolean;
}

interface CPRWorkerInitVolumeInput extends CPRWorkerVolumeState {
  type: 'INIT_VOLUME';
  requestId: string;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
}

interface CPRWorkerRenderInput {
  type: 'RENDER';
  requestId: string;
  sessionKey: string;
  verticalDir?: [number, number, number];
  frames: WorkerFrame[];
  panoWidth: number;
  panoHeight: number;
  vertHalfMm?: number;
  verticalCenterOffsetMm?: number;
  rigidVerticalSliceMode?: boolean;
  slabHalfThicknessMm: number;
  slabSamples: number;
  aggregation: 'MIP' | 'MEAN';
  applyModalityLut?: boolean;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
  debugRunId?: string;
  attemptLabel?: string;
  reconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
  renderBackend?: 'gpu' | 'cpu';
  allowLegacyFallback?: boolean;
  sourceVolumeId?: string;
}

interface CPRWorkerDisposeInput {
  type: 'DISPOSE';
  requestId: string;
}

type CPRWorkerMessage = CPRWorkerInitVolumeInput | CPRWorkerRenderInput | CPRWorkerDisposeInput;

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

function resolveReconstructionMode(value: unknown): 'legacy' | 'virtualPanoPhase1' | 'virtualPano' {
  return value === 'virtualPanoPhase1' || value === 'virtualPano' || value === 'legacy'
    ? value
    : 'legacy';
}

function resolveCpuFallbackReconstructionMode(
  requestedMode: 'legacy' | 'virtualPanoPhase1' | 'virtualPano'
): 'virtualPano' {
  return requestedMode === 'virtualPano' ? requestedMode : 'virtualPano';
}

interface CPRWorkerSuccess {
  type: 'SUCCESS';
  requestId: string;
  pixelData: Float32Array | Uint16Array;
  meanMap?: Float32Array;
  maxMap?: Float32Array;
  sampleCountMap?: Float32Array;
  debugMaps?: GpuPanoDebugMaps;
  panoWidth: number;
  panoHeight: number;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  slope: number;
  intercept: number;
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
  requestId: string;
  message: string;
}

interface CPRWorkerInitSuccess {
  type: 'INIT_SUCCESS';
  requestId: string;
  sessionKey: string;
}

interface CPRWorkerDisposeSuccess {
  type: 'DISPOSE_SUCCESS';
  requestId: string;
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

function cross3(
  a: [number, number, number],
  b: [number, number, number]
): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
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
    const vi =
      worldToIndex[0] * wx + worldToIndex[4] * wy + worldToIndex[8] * wz + worldToIndex[12];
    const vj =
      worldToIndex[1] * wx + worldToIndex[5] * wy + worldToIndex[9] * wz + worldToIndex[13];
    const vk =
      worldToIndex[2] * wx + worldToIndex[6] * wy + worldToIndex[10] * wz + worldToIndex[14];
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
  // vtk.js stores the direction basis in column-major order. The fallback
  // worldToVoxel() path expects the inverse rows packed contiguously, so for an
  // orthonormal direction matrix the row-major transpose of the VTK basis is
  // read out in the original flat-array order.
  return [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]];
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
  data: Float32Array | Int16Array | Uint16Array,
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
  data: Float32Array | Int16Array | Uint16Array,
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
  blendWeight: number,
  backgroundWeights?: Float32Array
): boolean {
  if (width < 3 || height < 3 || blendWeight <= 0) {
    return false;
  }

  const source = new Float32Array(pixelData);
  const sigmaRange = 220;
  const sigmaRangeDen = 2 * sigmaRange * sigmaRange;
  const clampedBlend = Math.max(0, Math.min(0.6, blendWeight));

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

      const backgroundWeight =
        backgroundWeights && backgroundWeights.length === pixelData.length
          ? clampNumber(Number(backgroundWeights[index]), 0, 1)
          : 1;
      const localBlend = clampedBlend * backgroundWeight;
      if (localBlend <= 0.001) {
        continue;
      }
      const keepWeight = 1 - localBlend;

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
      pixelData[index] = keepWeight * center + localBlend * filtered;
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

function computeAutoDisplayWindow(pixelData: Float32Array): {
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
} {
  const { minValue, maxValue } = computeArrayMinMax(pixelData);
  const samples: number[] = [];
  const sampleStep = Math.max(1, Math.floor(pixelData.length / 20000));

  for (let index = 0; index < pixelData.length; index += sampleStep) {
    const value = Number(pixelData[index]);
    if (Number.isFinite(value)) {
      samples.push(value);
    }
  }

  if (samples.length >= 32) {
    const lower = percentile(samples, 0.01);
    const upper = percentile(samples, 0.99);
    if (Number.isFinite(lower) && Number.isFinite(upper) && upper > lower) {
      const windowWidth = Math.max(1, upper - lower);
      return {
        minValue,
        maxValue,
        windowWidth,
        windowCenter: lower + windowWidth / 2,
      };
    }
  }

  const safeMin = Number.isFinite(minValue) ? minValue : 0;
  const safeMax = Number.isFinite(maxValue) ? maxValue : safeMin + 1;
  const windowWidth = Math.max(1, safeMax - safeMin);

  return {
    minValue: safeMin,
    maxValue: safeMax,
    windowWidth,
    windowCenter: safeMin + windowWidth / 2,
  };
}

function quantizePanoForStackDisplay(
  pixelData: Float32Array,
  minValue: number,
  maxValue: number
): {
  pixelData: Uint16Array;
  slope: number;
  intercept: number;
} {
  const safeMin = Number.isFinite(minValue) ? Number(minValue) : 0;
  const safeMax = Number.isFinite(maxValue) ? Number(maxValue) : safeMin;
  const intercept = safeMin;
  const range = Math.max(0, safeMax - safeMin);
  const slope = range > 1e-6 ? range / 65535 : 1;
  const storedPixelData = new Uint16Array(pixelData.length);

  for (let i = 0; i < pixelData.length; i++) {
    const hu = Number(pixelData[i]);
    if (!Number.isFinite(hu)) {
      storedPixelData[i] = 0;
      continue;
    }

    const roundedHu = Math.round(hu);
    const storedValue = Math.round((roundedHu - intercept) / slope);
    storedPixelData[i] = Math.max(0, Math.min(65535, storedValue));
  }

  return {
    pixelData: storedPixelData,
    slope,
    intercept,
  };
}

void quantizePanoForStackDisplay;

function validateGpuReadback(pixelData: Float32Array, panoWidth: number, panoHeight: number): void {
  const expectedLength = Math.max(0, panoWidth * panoHeight);
  if (!(pixelData instanceof Float32Array) || pixelData.length !== expectedLength) {
    throw new Error(
      `GPU pano readback length mismatch: expected ${expectedLength}, got ${pixelData?.length ?? 0}.`
    );
  }

  let finiteCount = 0;
  const sampleStep = Math.max(1, Math.floor(pixelData.length / 4096));
  for (let index = 0; index < pixelData.length; index += sampleStep) {
    if (Number.isFinite(Number(pixelData[index]))) {
      finiteCount++;
      if (finiteCount >= 8) {
        return;
      }
    }
  }

  throw new Error('GPU pano readback did not contain enough finite pixels.');
}

interface ScalarSamplingPolicy {
  safeSlope: number;
  safeIntercept: number;
  requestedModalityLutApplied: boolean;
  shouldApplyModalityLut: boolean;
  shouldNormalizeStoredValues: boolean;
  unsignedPackedArtifactDetected: boolean;
  normalizeStoredSample?: (value: number) => number;
  normalizationSignature: string | null;
  safeInterpolationOobValue: number;
}

function resolveScalarSamplingPolicy(input: CPRWorkerInput): ScalarSamplingPolicy {
  const {
    scalarData,
    allowStoredValueNormalization,
    disableStoredValueNormalization,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    isPreScaled,
    applyModalityLut,
  } = input;
  const transform = createHuScalarTransform({
    scalarData,
    allowStoredValueNormalization,
    disableStoredValueNormalization,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    isPreScaled,
  });
  const requestedModalityLutApplied =
    typeof applyModalityLut === 'boolean'
      ? applyModalityLut
      : Math.abs(transform.safeSlope - 1) > 1e-6 || Math.abs(transform.safeIntercept) > 1e-6;

  return {
    safeSlope: transform.safeSlope,
    safeIntercept: transform.safeIntercept,
    requestedModalityLutApplied,
    shouldApplyModalityLut: transform.shouldApplyModalityLut,
    shouldNormalizeStoredValues: transform.shouldNormalizeStoredValues,
    unsignedPackedArtifactDetected: transform.unsignedPackedArtifactDetected,
    normalizeStoredSample: transform.normalizeStoredSample,
    normalizationSignature: transform.normalizationSignature,
    safeInterpolationOobValue: transform.safeInterpolationOobValue,
  };
}

function generateGpuPanorama(
  input: CPRWorkerInput,
  volumeCacheKey: string
): {
  pixelData: Float32Array;
  meanMap: Float32Array;
  maxMap: Float32Array;
  sampleCountMap: Float32Array;
  debugMaps?: GpuPanoDebugMaps;
  pipelineMode: 'single-pass' | 'multi-pass';
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  diagnostics?: GpuPanoDiagnostics;
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
    reconstructionMode,
  } = input;
  const scalarPolicy = resolveScalarSamplingPolicy(input);
  if (!isMat4ArrayLike(worldToIndex)) {
    throw new Error(
      '[CPR-GPU] Missing authoritative worldToIndex mat4. Falling back to CPU preserves parity.'
    );
  }

  const effectiveVerticalDir =
    Array.isArray(verticalDir) && verticalDir.length >= 3
      ? normalize3([verticalDir[0], verticalDir[1], verticalDir[2]])
      : normalize3([direction[6] ?? 0, direction[7] ?? 0, direction[8] ?? 1]);
  const effectiveVerticalHalfMm =
    Number.isFinite(requestedVertHalfMm) && Number(requestedVertHalfMm) > 0
      ? Number(requestedVertHalfMm)
      : 15.0;
  const verticalCenterOffsetMm = Number.isFinite(requestedVerticalCenterOffsetMm)
    ? Number(requestedVerticalCenterOffsetMm)
    : 0;
  const requestedSlabSamples = Math.max(1, Math.min(64, Math.floor(slabSamples)));
  const startedAt = performance.now();
  const requestedRenderBackend = input.renderBackend === 'cpu' ? 'cpu' : 'gpu';
  const formatGpuReadableValue = (value: number | null | undefined, fractionDigits = 1): string =>
    Number.isFinite(value) ? Number(value).toFixed(fractionDigits) : 'na';
  console.log(
    `[CPR-GPU-WORKER] run=${input.debugRunId ?? 'na'} requestedAggregation=${aggregation} ` +
      `reconstructionMode=${reconstructionMode ?? 'legacy'} ` +
      `effectivePipeline=renderer-selected ` +
      `virtualPano=${reconstructionMode === 'legacy' ? 'off' : 'gpu-internal'} ` +
      `backgroundSuppression=renderer-internal denoise=renderer-internal ` +
      `pano=${panoWidth}x${panoHeight} ` +
      `verticalHalfMm=${formatGpuReadableValue(effectiveVerticalHalfMm)} ` +
      `centerOffsetMm=${formatGpuReadableValue(verticalCenterOffsetMm)} ` +
      `slabHalfMm=${formatGpuReadableValue(slabHalfThicknessMm)} ` +
      `slabSamples=${requestedSlabSamples}`
  );
  const gpuResult = renderPanoGpu(
    {
      scalarData,
      dimensions,
      spacing,
      origin,
      direction,
      worldToIndex: Array.from(worldToIndex).slice(0, 16),
      frames: frames.map(frame => ({
        position: [frame.position[0], frame.position[1], frame.position[2]],
        N_slab: [frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]],
        S: Array.isArray(frame.S) ? [frame.S[0], frame.S[1], frame.S[2]] : undefined,
      })),
      panoWidth,
      panoHeight,
      verticalDir: effectiveVerticalDir,
      vertHalfMm: effectiveVerticalHalfMm,
      verticalCenterOffsetMm,
      slabHalfThicknessMm,
      slabSamples: requestedSlabSamples,
      rescaleSlope: scalarPolicy.safeSlope,
      rescaleIntercept: scalarPolicy.safeIntercept,
      applyRescale: scalarPolicy.shouldApplyModalityLut,
      normalizationSignature: scalarPolicy.normalizationSignature,
      normalizeStoredSample: scalarPolicy.normalizeStoredSample,
      returnDebugSidecars: !!input.debugRunId,
    },
    volumeCacheKey
  );
  validateGpuReadback(gpuResult.pixelData, panoWidth, panoHeight);
  const elapsedMs = performance.now() - startedAt;
  const { minValue, maxValue, windowWidth, windowCenter } = computeAutoDisplayWindow(
    gpuResult.pixelData
  );
  const phase2GatePassed = gpuResult.diagnostics?.phase2GatePassed === true;
  const gpuLogContext = {
    runId: input.debugRunId ?? null,
    attemptLabel: input.attemptLabel ?? null,
    displayedPath: 'worker-recon',
    backend: 'gpu',
    reconstructionMode: reconstructionMode ?? 'legacy',
    sourceVolumeId: input.sourceVolumeId ?? null,
    pipelineMode: gpuResult.pipelineMode,
  };
  console.log(
    '[CPR-GPU-WORKER-RESULT-JSON]',
    JSON.stringify({
      runId: input.debugRunId ?? null,
      attemptLabel: input.attemptLabel ?? null,
      displayedPath: 'worker-recon',
      backend: 'gpu',
      requestedBackend: requestedRenderBackend,
      reconstructionMode: reconstructionMode ?? 'legacy',
      sourceVolumeId: input.sourceVolumeId ?? null,
      pipelineMode: gpuResult.pipelineMode,
      expectedPipelineMode: gpuResult.diagnostics?.expectedPipelineMode ?? 'multi-pass',
      phase2GatePassed,
      panoWidth,
      panoHeight,
      requestedSlabHalfThicknessMm: slabHalfThicknessMm,
      requestedSlabSamples,
      slabHalfThicknessMm,
      slabSamples: requestedSlabSamples,
      minValue,
      maxValue,
      windowWidth,
      windowCenter,
    })
  );
  console.log(
    `[CPR-GPU-WORKER-RESULT] run=${input.debugRunId ?? 'na'} ` +
      `pipeline=${gpuResult.pipelineMode} ` +
      `min=${formatGpuReadableValue(minValue)} max=${formatGpuReadableValue(maxValue)} ` +
      `windowWidth=${formatGpuReadableValue(windowWidth)} ` +
      `windowCenter=${formatGpuReadableValue(windowCenter)} ` +
      `elapsedMs=${formatGpuReadableValue(elapsedMs, 0)} ` +
      `phase2Gate=${phase2GatePassed ? 'pass' : 'fail'}`
  );
  console.log(
    '[CPR-SUPPORT-SURFACE-JSON]',
    JSON.stringify(
      gpuResult.diagnostics?.supportSurface
        ? {
            ...gpuLogContext,
            ...gpuResult.diagnostics.supportSurface,
          }
        : {
            ...gpuLogContext,
            skippedReason:
              gpuResult.pipelineMode === 'multi-pass'
                ? 'support-diagnostics-unavailable'
                : 'pipeline-not-multi-pass',
          }
    )
  );
  console.log(
    '[CPR-DRR-JSON]',
    JSON.stringify(
      gpuResult.diagnostics?.drr
        ? {
            ...gpuLogContext,
            ...gpuResult.diagnostics.drr,
          }
        : {
            ...gpuLogContext,
            skippedReason:
              gpuResult.pipelineMode === 'multi-pass'
                ? 'drr-diagnostics-unavailable'
                : 'pipeline-not-multi-pass',
          }
    )
  );
  console.log(
    '[CPR-TONE-MAP-JSON]',
    JSON.stringify(
      gpuResult.diagnostics?.toneMap
        ? {
            ...gpuLogContext,
            ...gpuResult.diagnostics.toneMap,
          }
        : {
            ...gpuLogContext,
            skippedReason:
              gpuResult.pipelineMode === 'multi-pass'
                ? 'tone-diagnostics-unavailable'
                : 'pipeline-not-multi-pass',
          }
    )
  );
  console.log(
    '[CPR-SIDECAR-MAPS-JSON]',
    JSON.stringify({
      ...gpuLogContext,
      ...(gpuResult.diagnostics?.sidecarMaps ?? {
        debugEnabled: !!input.debugRunId,
        attachedMaps: [],
        attachedByteLength: 0,
      }),
    })
  );
  const outputSignature = computeOutputSignature(gpuResult.pixelData);
  const usesPerColumnVerticalDirs = frames.some(
    frame => Array.isArray(frame.S) && frame.S.length >= 3
  );

  return {
    pixelData: gpuResult.pixelData,
    meanMap: gpuResult.meanMap,
    maxMap: gpuResult.maxMap,
    sampleCountMap: gpuResult.sampleCountMap,
    debugMaps: gpuResult.debugMaps,
    pipelineMode: gpuResult.pipelineMode,
    minValue,
    maxValue,
    windowWidth,
    windowCenter,
    modalityLutApplied: scalarPolicy.shouldApplyModalityLut,
    requestedModalityLutApplied: scalarPolicy.requestedModalityLutApplied,
    storedValueNormalizationApplied: scalarPolicy.shouldNormalizeStoredValues,
    unsignedPackedArtifactDetected: scalarPolicy.unsignedPackedArtifactDetected,
    diagnostics: gpuResult.diagnostics,
    diagnosticPayload: {
      renderBackend: 'gpu',
      requestedRenderBackend,
      pipelineMode: gpuResult.pipelineMode,
      fallbackReason: phase2GatePassed ? null : 'gpu-phase2-gate-failed',
      attemptLabel: input.attemptLabel ?? null,
      reconstructionMode: reconstructionMode ?? 'legacy',
      verticalHalfMm: effectiveVerticalHalfMm,
      globalVerticalCenterOffsetMm: 0,
      baseVerticalCenterOffsetMm: verticalCenterOffsetMm,
      verticalCenterOffsetMm,
      requestedVerticalCenterOffsetMm: verticalCenterOffsetMm,
      fittedVerticalCenterOffsetMm: verticalCenterOffsetMm,
      slabSampling: {
        requestedSamples: requestedSlabSamples,
        effectiveSamples: requestedSlabSamples,
        slabHalfThicknessMm,
        aggregation:
          gpuResult.pipelineMode === 'multi-pass'
            ? 'GPU_SUPPORT_SURFACE_DRR'
            : 'GPU_FOCAL_TROUGH_DRR',
        reduction:
          gpuResult.pipelineMode === 'multi-pass'
            ? 'support estimation + support smoothing + DRR attenuation + tone mapping'
            : 'gaussian focal-trough weighted accumulation (sigma 1.5 mm)',
      },
      gpuRender: {
        enabled: true,
        pipelineMode: gpuResult.pipelineMode,
        expectedPipelineMode: gpuResult.diagnostics?.expectedPipelineMode ?? 'multi-pass',
        phase2GatePassed,
        degradedModeReason: gpuResult.diagnostics?.degradedModeReason ?? null,
        volumeCacheKey,
        usedPerColumnVerticalDirs: usesPerColumnVerticalDirs,
        authoritativeWorldToIndex: true,
        modalityLutApplied: scalarPolicy.shouldApplyModalityLut,
        storedValueNormalizationApplied: scalarPolicy.shouldNormalizeStoredValues,
        supportSurface: gpuResult.diagnostics?.supportSurface ?? null,
        drr: gpuResult.diagnostics?.drr ?? null,
        toneMap: gpuResult.diagnostics?.toneMap ?? null,
        sidecarMaps: gpuResult.diagnostics?.sidecarMaps ?? null,
      },
      outputDisplayWindow: {
        lower: minValue,
        upper: maxValue,
        windowWidth,
        windowCenter,
      },
      timingMs: {
        adaptiveCenterSearch: 0,
        pass1And2TwoPassRender: 0,
        virtualPanoPhase12: 0,
        suppressionAndDenoise: 0,
        diagnosticAssembly: 0,
        gpuRender: Math.round(elapsedMs),
        total: Math.round(elapsedMs),
      },
    },
    outputSignature,
  };
}

function summarizeVirtualPanoOutput(
  pixelData: Float32Array,
  panoWidth: number,
  panoHeight: number,
  panoCenterRow: number,
  selectedDepthMm: Float32Array,
  depthHalfRangeMm: number
): {
  minValue: number;
  maxValue: number;
  range: number;
  toothBandMean: number;
  toothBandP10: number;
  toothBandP90: number;
  lowerBandMean: number;
  lowerBandBrightFraction: number;
  supportDepthClampFraction: number;
} {
  const rowFromNormalizedOffset = (yNorm: number): number => {
    const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
    return Math.max(0, Math.min(panoHeight - 1, row));
  };
  const toothBandStartRow = Math.min(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
  const toothBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
  const lowerBandStartRow = Math.min(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
  const lowerBandEndRow = Math.max(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));

  let minValue = Infinity;
  let maxValue = -Infinity;
  let toothBandSum = 0;
  let toothBandCount = 0;
  let lowerBandSum = 0;
  let lowerBandCount = 0;
  let lowerBandBrightCount = 0;
  const toothBandValues: number[] = [];

  for (let row = 0; row < panoHeight; row++) {
    for (let col = 0; col < panoWidth; col++) {
      const value = Number(pixelData[planeIndex(col, row, panoWidth)]);
      if (!Number.isFinite(value)) {
        continue;
      }
      minValue = Math.min(minValue, value);
      maxValue = Math.max(maxValue, value);

      if (row >= toothBandStartRow && row <= toothBandEndRow) {
        toothBandSum += value;
        toothBandCount++;
        toothBandValues.push(value);
      }
      if (row >= lowerBandStartRow && row <= lowerBandEndRow) {
        lowerBandSum += value;
        lowerBandCount++;
        if (value > -200) {
          lowerBandBrightCount++;
        }
      }
    }
  }

  let supportDepthClampCount = 0;
  for (let col = 0; col < panoWidth; col++) {
    if (Math.abs(Number(selectedDepthMm[col])) > depthHalfRangeMm - 0.5) {
      supportDepthClampCount++;
    }
  }

  const safeMin = Number.isFinite(minValue) ? minValue : 0;
  const safeMax = Number.isFinite(maxValue) ? maxValue : 0;
  return {
    minValue: safeMin,
    maxValue: safeMax,
    range: safeMax - safeMin,
    toothBandMean: toothBandCount > 0 ? toothBandSum / toothBandCount : 0,
    toothBandP10: toothBandValues.length > 0 ? percentile(toothBandValues, 0.1) : 0,
    toothBandP90: toothBandValues.length > 0 ? percentile(toothBandValues, 0.9) : 0,
    lowerBandMean: lowerBandCount > 0 ? lowerBandSum / lowerBandCount : 0,
    lowerBandBrightFraction: lowerBandCount > 0 ? lowerBandBrightCount / lowerBandCount : 0,
    supportDepthClampFraction: panoWidth > 0 ? supportDepthClampCount / panoWidth : 0,
  };
}

function suppressLowerBackgroundUniformly(
  pixelData: Float32Array,
  panoWidth: number,
  panoHeight: number,
  panoCenterRow: number
): {
  coverageFraction: number;
  meanAttenuation: number;
  maxAttenuation: number;
} {
  if (panoWidth <= 0 || panoHeight < 8 || panoCenterRow <= 0) {
    return {
      coverageFraction: 0,
      meanAttenuation: 0,
      maxAttenuation: 0,
    };
  }

  const source = new Float32Array(pixelData);
  const rowFromNormalizedOffset = (yNorm: number): number => {
    const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
    return Math.max(0, Math.min(panoHeight - 1, row));
  };

  const startRow = Math.min(rowFromNormalizedOffset(0.7), panoHeight - 1);
  const endRow = Math.max(startRow + 1, Math.min(rowFromNormalizedOffset(1.08), panoHeight - 1));
  let attenuatedPixelCount = 0;
  let attenuationSum = 0;
  let maxAttenuation = 0;

  for (let row = startRow; row < panoHeight; row++) {
    const baseAttenuation = 0.52 * smoothstep01((row - startRow) / Math.max(1, endRow - startRow));
    if (baseAttenuation <= 0.001) {
      continue;
    }

    for (let col = 0; col < panoWidth; col++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const value = Number(source[pixelIndex]);
      if (!Number.isFinite(value)) {
        continue;
      }

      const intensityProtect = clampNumber((value - 1200) / 1200, 0, 0.35);
      const attenuation = baseAttenuation * (1 - intensityProtect);
      if (attenuation <= 0.005) {
        continue;
      }

      const darkTarget = Math.min(-800, value * 0.05);
      pixelData[pixelIndex] = value + (darkTarget - value) * attenuation;
      attenuatedPixelCount++;
      attenuationSum += attenuation;
      if (attenuation > maxAttenuation) {
        maxAttenuation = attenuation;
      }
    }
  }

  return {
    coverageFraction:
      pixelData.length > 0
        ? Math.round((attenuatedPixelCount / pixelData.length) * 100000) / 100000
        : 0,
    meanAttenuation:
      attenuatedPixelCount > 0
        ? Math.round((attenuationSum / attenuatedPixelCount) * 100000) / 100000
        : 0,
    maxAttenuation: Math.round(maxAttenuation * 100000) / 100000,
  };
}

function applyLightGaussianBlur3D(
  stack: Float32Array,
  width: number,
  height: number,
  depthSamples: number
): void {
  if (width <= 1 || height <= 1 || depthSamples <= 1 || stack.length === 0) {
    return;
  }

  const planeSize = width * height;
  const scratch = new Float32Array(stack.length);
  const kernel = [1, 2, 1] as const;

  for (let depth = 0; depth < depthSamples; depth++) {
    for (let row = 0; row < height; row++) {
      const pixelBase = row * width;
      for (let col = 0; col < width; col++) {
        let weightedSum = 0;
        let weightTotal = 0;
        const pixelIndex = pixelBase + col;
        for (let offset = -1; offset <= 1; offset++) {
          const neighborDepth = depth + offset;
          if (neighborDepth < 0 || neighborDepth >= depthSamples) {
            continue;
          }
          const value = Number(stack[stackIndex(neighborDepth, pixelIndex, planeSize)]);
          if (!Number.isFinite(value)) {
            continue;
          }
          const weight = kernel[offset + 1];
          weightedSum += value * weight;
          weightTotal += weight;
        }
        scratch[stackIndex(depth, pixelIndex, planeSize)] =
          weightTotal > 0 ? weightedSum / weightTotal : Number.NaN;
      }
    }
  }

  for (let depth = 0; depth < depthSamples; depth++) {
    const depthBase = depth * planeSize;
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        let weightedSum = 0;
        let weightTotal = 0;
        for (let offset = -1; offset <= 1; offset++) {
          const neighborRow = row + offset;
          if (neighborRow < 0 || neighborRow >= height) {
            continue;
          }
          const value = Number(scratch[depthBase + neighborRow * width + col]);
          if (!Number.isFinite(value)) {
            continue;
          }
          const weight = kernel[offset + 1];
          weightedSum += value * weight;
          weightTotal += weight;
        }
        stack[depthBase + row * width + col] =
          weightTotal > 0 ? weightedSum / weightTotal : Number.NaN;
      }
    }
  }

  for (let depth = 0; depth < depthSamples; depth++) {
    const depthBase = depth * planeSize;
    for (let row = 0; row < height; row++) {
      const rowBase = depthBase + row * width;
      for (let col = 0; col < width; col++) {
        let weightedSum = 0;
        let weightTotal = 0;
        for (let offset = -1; offset <= 1; offset++) {
          const neighborCol = col + offset;
          if (neighborCol < 0 || neighborCol >= width) {
            continue;
          }
          const value = Number(stack[rowBase + neighborCol]);
          if (!Number.isFinite(value)) {
            continue;
          }
          const weight = kernel[offset + 1];
          weightedSum += value * weight;
          weightTotal += weight;
        }
        scratch[rowBase + col] = weightTotal > 0 ? weightedSum / weightTotal : Number.NaN;
      }
    }
  }

  stack.set(scratch);
}

function renderVirtualPanoFromSupportPath(params: {
  virtualPanoStack: Float32Array;
  panoWidth: number;
  panoHeight: number;
  planeSize: number;
  panoCenterRow: number;
  virtualPanoDepthOffsetsMm: Float32Array;
  selectedDepthMm: Float32Array;
  softThresholdByCol: Float32Array;
  hardThresholdByCol: Float32Array;
  supportTiltMmByCol?: Float32Array;
  virtualPanoDepthHalfRangeMm: number;
}): {
  pixelData: Float32Array;
  summary: ReturnType<typeof summarizeVirtualPanoOutput>;
  diagnostics: {
    enabled: boolean;
    usedAsOutput: boolean;
    acceptedByLowerBandTolerance: boolean;
    renderSupportMode: 'singlePath';
    rejectReasons: string[];
    eligibleSampleFraction: number;
    lowerBandEligibleFraction: number;
    emptyFallbackFraction: number;
    retainedWeightFractionMean: number;
    offTroughEnergyRatio: number;
    lowerSuppressionRatio: number;
    toothBandMean: number;
    lowerBandMean: number;
    postRenderLowerBackgroundSuppression: {
      coverageFraction: number;
      meanAttenuation: number;
      maxAttenuation: number;
    };
    supportTiltMode: 'disabled' | 'linear';
    supportTiltFirst8Mm: number[];
    supportTiltMeanAbsMm: number;
    supportTiltMaxAbsMm: number;
    supportDepthFirst8Mm: number[];
  };
} {
  const {
    virtualPanoStack,
    panoWidth,
    panoHeight,
    planeSize,
    panoCenterRow,
    virtualPanoDepthOffsetsMm,
    selectedDepthMm,
    softThresholdByCol,
    hardThresholdByCol,
    supportTiltMmByCol,
    virtualPanoDepthHalfRangeMm,
  } = params;

  const depthSamples = virtualPanoDepthOffsetsMm.length;
  const pixelData = new Float32Array(planeSize);
  let eligibleSampleCount = 0;
  let lowerBandEligibleSampleCount = 0;
  let inBoundsSampleCount = 0;
  let lowerBandInBoundsSampleCount = 0;
  let fallbackNoEligibleCount = 0;
  let retainedWeightFractionSum = 0;
  let retainedWeightFractionCount = 0;
  let onSupportEnergy = 0;
  let offSupportEnergy = 0;
  const hasSupportTilt = !!supportTiltMmByCol && supportTiltMmByCol.length === panoWidth;
  let lowerBandRenderCount = 0;
  let lowerBandSuppressedCount = 0;
  let lowerBandAttenuationSum = 0;
  let lowerBandAttenuationMax = 0;
  const SUPPORT_SIGMA_MM = 0.9;
  const SUPPORT_DENOM = 2.0 * SUPPORT_SIGMA_MM * SUPPORT_SIGMA_MM;
  const SUPPORT_ENERGY_WINDOW_MM = SUPPORT_SIGMA_MM * 2.0;
  const ALPHA_SCALE = 0.82;

  for (let row = 0; row < panoHeight; row++) {
    const yNorm = panoCenterRow > 0 ? (row - panoCenterRow) / panoCenterRow : 0;

    for (let col = 0; col < panoWidth; col++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const supportTiltMm = hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0;
      const supportDepthMm = clampNumber(
        Number(selectedDepthMm[col]) + supportTiltMm * yNorm,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const softThreshold = Number(softThresholdByCol[col]);
      const hardThreshold = Number(hardThresholdByCol[col]);
      const hardDen = Math.max(hardThreshold - softThreshold, 80);
      const emissionDen = Math.max(hardDen * 1.4, 140);
      let retainedSampleCount = 0;
      let validDepthCount = 0;
      let accumulatedSignal = 0;
      let transmittance = 1;

      for (let depth = 0; depth < depthSamples; depth++) {
        const value = Number(virtualPanoStack[stackIndex(depth, pixelIndex, planeSize)]);
        if (!Number.isFinite(value)) {
          continue;
        }

        validDepthCount++;
        inBoundsSampleCount++;
        if (yNorm >= 0.65) {
          lowerBandInBoundsSampleCount++;
        }

        const depthMm = Number(virtualPanoDepthOffsetsMm[depth]);
        const absDelta = Math.abs(depthMm - supportDepthMm);
        const troughWeight = Math.exp(-(absDelta * absDelta) / SUPPORT_DENOM);
        if (absDelta <= SUPPORT_ENERGY_WINDOW_MM) {
          onSupportEnergy += Math.abs(value) * troughWeight;
        } else {
          offSupportEnergy += Math.abs(value) * troughWeight;
        }

        const hardResponse = clampNumber((value - softThreshold) / hardDen, 0, 1);
        const emission = clampNumber((value - softThreshold) / emissionDen, 0, 1);
        if (hardResponse > 0.02) {
          retainedSampleCount++;
          eligibleSampleCount++;
          if (yNorm >= 0.65) {
            lowerBandEligibleSampleCount++;
          }
        }

        const alpha = clampNumber(hardResponse * troughWeight * ALPHA_SCALE, 0, 0.98);
        if (alpha <= 1e-4 || emission <= 1e-4) {
          continue;
        }

        accumulatedSignal += transmittance * alpha * emission;
        transmittance *= 1 - alpha;
        if (transmittance < 0.01) {
          break;
        }
      }

      if (validDepthCount > 0) {
        retainedWeightFractionSum += retainedSampleCount / validDepthCount;
        retainedWeightFractionCount++;
      }

      const attenuationSignal = clampNumber(accumulatedSignal, 0, 1);
      if (retainedSampleCount <= 0 || validDepthCount <= 0) {
        fallbackNoEligibleCount++;
        pixelData[pixelIndex] = -1000;
      } else {
        pixelData[pixelIndex] = -1000 + attenuationSignal * 2600;
      }

      if (yNorm >= 0.65) {
        lowerBandRenderCount++;
        lowerBandAttenuationSum += 1 - attenuationSignal;
        lowerBandAttenuationMax = Math.max(lowerBandAttenuationMax, 1 - attenuationSignal);
        if (attenuationSignal < 0.15) {
          lowerBandSuppressedCount++;
        }
      }
    }
  }

  const summary = summarizeVirtualPanoOutput(
    pixelData,
    panoWidth,
    panoHeight,
    panoCenterRow,
    selectedDepthMm,
    virtualPanoDepthHalfRangeMm
  );
  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const lowerSuppressionRatio =
    summary.toothBandMean > 1e-6 ? summary.lowerBandMean / summary.toothBandMean : 1;
  const emptyFallbackFraction = planeSize > 0 ? fallbackNoEligibleCount / planeSize : 0;
  const eligibleSampleFraction =
    inBoundsSampleCount > 0 ? eligibleSampleCount / inBoundsSampleCount : 0;
  const lowerBandEligibleFraction =
    lowerBandInBoundsSampleCount > 0
      ? lowerBandEligibleSampleCount / lowerBandInBoundsSampleCount
      : 0;
  const retainedWeightFractionMean =
    retainedWeightFractionCount > 0 ? retainedWeightFractionSum / retainedWeightFractionCount : 0;
  const offTroughEnergyRatio = onSupportEnergy > 1e-6 ? offSupportEnergy / onSupportEnergy : 0;
  const rejectReasons: string[] = [];
  if (eligibleSampleFraction < 0.015) {
    rejectReasons.push('eligible-sample-fraction-too-low');
  }
  if (retainedWeightFractionMean < 0.02) {
    rejectReasons.push('retained-weight-fraction-too-low');
  }
  if (emptyFallbackFraction > 0.18) {
    rejectReasons.push('empty-fallback-fraction-too-high');
  }
  if (offTroughEnergyRatio > 1.1) {
    rejectReasons.push('off-trough-energy-too-high');
  }
  if (summary.supportDepthClampFraction > 0.32) {
    rejectReasons.push('support-depth-clamp-fraction-too-high');
  }
  if (summary.range < 900) {
    rejectReasons.push('range-too-low');
  }
  if (toothBandContrastRange < 240) {
    rejectReasons.push('tooth-band-contrast-too-low');
  }
  if (summary.toothBandMean > 920 || summary.toothBandP10 > 180) {
    rejectReasons.push('tooth-band-saturation');
  }
  if (summary.lowerBandBrightFraction > 0.62) {
    rejectReasons.push('lower-band-bright-fraction-too-high');
  }
  if (summary.lowerBandMean > 40) {
    rejectReasons.push('lower-band-mean-too-high');
  }
  if (lowerSuppressionRatio > 0.82) {
    rejectReasons.push('lower-suppression-ratio-too-high');
  }
  const lowerBandOnlyRejectReasons = new Set([
    'lower-band-bright-fraction-too-high',
    'lower-band-mean-too-high',
    'lower-suppression-ratio-too-high',
  ]);
  const nonLowerBandRejectReasons = rejectReasons.filter(
    reason => !lowerBandOnlyRejectReasons.has(reason)
  );
  const acceptedByLowerBandTolerance =
    rejectReasons.length > 0 &&
    nonLowerBandRejectReasons.length === 0 &&
    summary.lowerBandBrightFraction <= 0.76 &&
    summary.lowerBandMean <= 120 &&
    toothBandContrastRange >= 260 &&
    emptyFallbackFraction <= 0.12 &&
    retainedWeightFractionMean >= 0.04 &&
    offTroughEnergyRatio <= 0.9;
  const usedAsOutput = rejectReasons.length === 0 || acceptedByLowerBandTolerance;
  let supportTiltMeanAbsMm = 0;
  let supportTiltMaxAbsMm = 0;
  for (let col = 0; col < panoWidth; col++) {
    const absTiltMm = Math.abs(hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0);
    supportTiltMeanAbsMm += absTiltMm;
    supportTiltMaxAbsMm = Math.max(supportTiltMaxAbsMm, absTiltMm);
  }
  supportTiltMeanAbsMm = panoWidth > 0 ? supportTiltMeanAbsMm / panoWidth : 0;

  return {
    pixelData,
    summary,
    diagnostics: {
      enabled: true,
      usedAsOutput,
      acceptedByLowerBandTolerance,
      renderSupportMode: 'singlePath',
      rejectReasons,
      eligibleSampleFraction,
      lowerBandEligibleFraction,
      emptyFallbackFraction,
      retainedWeightFractionMean,
      offTroughEnergyRatio,
      lowerSuppressionRatio,
      toothBandMean: summary.toothBandMean,
      lowerBandMean: summary.lowerBandMean,
      postRenderLowerBackgroundSuppression: {
        coverageFraction:
          lowerBandRenderCount > 0 ? lowerBandSuppressedCount / lowerBandRenderCount : 0,
        meanAttenuation:
          lowerBandRenderCount > 0 ? lowerBandAttenuationSum / lowerBandRenderCount : 0,
        maxAttenuation: lowerBandAttenuationMax,
      },
      supportTiltMode: hasSupportTilt ? 'linear' : 'disabled',
      supportTiltFirst8Mm: Array.from(
        (supportTiltMmByCol || new Float32Array(0)).subarray(
          0,
          Math.min(8, supportTiltMmByCol?.length ?? 0)
        )
      ).map(value => Math.round(Number(value) * 1000) / 1000),
      supportTiltMeanAbsMm,
      supportTiltMaxAbsMm,
      supportDepthFirst8Mm: Array.from(
        selectedDepthMm.subarray(0, Math.min(8, selectedDepthMm.length))
      ).map(value => Math.round(Number(value) * 1000) / 1000),
    },
  };
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
  count: number,
  diagnostics?: {
    brightTailEvaluatedCount: number;
    brightTailCappedCount: number;
    brightTailPreservedCount: number;
    brightClusterPreservedCount: number;
  }
): number {
  if (count <= 0) {
    return -1000;
  }

  const trimHi = count >= 6 ? Math.max(1, Math.floor(count / 12)) : 0;
  const winsorIndex = count - 1 - trimHi;
  if (trimHi > 0 && winsorIndex >= 0 && winsorIndex < count) {
    const winsorCap = values[winsorIndex];
    const highValue = values[count - 1];
    const secondHighestValue = count > 1 ? values[count - 2] : highValue;
    const robustSpan = Math.max(80, highValue - values[0]);
    const topGap = highValue - secondHighestValue;
    const capGap = highValue - winsorCap;
    const clusterTolerance = Math.max(110, robustSpan * 0.16);
    let clusterStart = count - 1;
    while (clusterStart > 0 && highValue - values[clusterStart - 1] <= clusterTolerance) {
      clusterStart--;
    }
    const brightClusterSize = count - clusterStart;
    const hasCoherentBrightCluster =
      brightClusterSize >= 2 && topGap <= Math.max(180, robustSpan * 0.22);
    const hasMaterialTail = capGap > Math.max(100, robustSpan * 0.12);
    const isIsolatedTop = topGap > Math.max(160, robustSpan * 0.2);
    const shouldCapTail = hasMaterialTail && isIsolatedTop && !hasCoherentBrightCluster;

    if (diagnostics) {
      diagnostics.brightTailEvaluatedCount++;
    }

    if (shouldCapTail) {
      for (let i = winsorIndex + 1; i < count; i++) {
        values[i] = winsorCap;
      }
      if (diagnostics) {
        diagnostics.brightTailCappedCount++;
      }
    } else if (diagnostics) {
      diagnostics.brightTailPreservedCount++;
      if (hasCoherentBrightCluster) {
        diagnostics.brightClusterPreservedCount++;
      }
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

function computeRawMean(values: Float32Array, count: number): number {
  if (count <= 0) {
    return -1000;
  }

  let sum = 0;

  for (let i = 0; i < count; i++) {
    sum += values[i];
  }

  return sum / count;
}

function computePureMax(values: Float32Array, count: number): number {
  if (count <= 0) {
    return -1000;
  }

  let maxValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i++) {
    const value = Number(values[i]);
    if (Number.isFinite(value) && value > maxValue) {
      maxValue = value;
    }
  }

  return Number.isFinite(maxValue) ? maxValue : -1000;
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

  if (
    count > 1 &&
    Number.isFinite(values[count - 2]) &&
    values[count - 1] - values[count - 2] > 850
  ) {
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

function meanFloatWindow(values: Float32Array, startIndex: number, endIndex: number): number {
  const clampedStart = Math.max(0, Math.min(values.length - 1, Math.floor(startIndex)));
  const clampedEnd = Math.max(clampedStart, Math.min(values.length - 1, Math.floor(endIndex)));
  let sum = 0;
  let count = 0;
  for (let index = clampedStart; index <= clampedEnd; index++) {
    const value = Number(values[index]);
    if (!Number.isFinite(value)) {
      continue;
    }
    sum += value;
    count++;
  }
  return count > 0 ? sum / count : 0;
}

function smoothstep01(value: number): number {
  const t = clampNumber(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function smoothstepRange(edge0: number, edge1: number, value: number): number {
  if (!Number.isFinite(edge0) || !Number.isFinite(edge1) || Math.abs(edge1 - edge0) < 1e-6) {
    return value >= edge1 ? 1 : 0;
  }
  return smoothstep01((value - edge0) / (edge1 - edge0));
}

function angleDegreesBetween3(a: [number, number, number], b: [number, number, number]): number {
  const dotValue = dot3(normalize3(a), normalize3(b));
  return (Math.acos(clampNumber(dotValue, -1, 1)) * 180) / Math.PI;
}

function planeIndex(col: number, row: number, cols: number): number {
  return row * cols + col;
}

function stackIndex(depth: number, pixelIndex: number, planeSize: number): number {
  return depth * planeSize + pixelIndex;
}

function suppressLowerBackground(
  pixelData: Float32Array,
  width: number,
  height: number,
  rowSpacingMm: number
): {
  suppressionWeights: Float32Array;
  toothBandBottomRows: Float32Array;
  toothBandBottomRowStats: {
    min: number;
    max: number;
    mean: number;
    first8: number[];
  };
  backgroundSuppression: {
    coverageFraction: number;
    meanAttenuation: number;
    maxAttenuation: number;
  };
} {
  const suppressionWeights = new Float32Array(pixelData.length);
  const toothBandBottomRows = new Float32Array(width);
  if (width <= 0 || height < 8) {
    return {
      suppressionWeights,
      toothBandBottomRows,
      toothBandBottomRowStats: {
        min: 0,
        max: 0,
        mean: 0,
        first8: [],
      },
      backgroundSuppression: {
        coverageFraction: 0,
        meanAttenuation: 0,
        maxAttenuation: 0,
      },
    };
  }

  const source = new Float32Array(pixelData);
  const columnValues = new Float32Array(height);
  const columnScratch = new Float32Array(height);
  const columnGradient = new Float32Array(height);
  const rowScratchA = new Float32Array(width);
  const rowScratchB = new Float32Array(width);
  const columnEdgeProtectThresholds = new Float32Array(width);
  const columnIntensityProtectThresholds = new Float32Array(width);
  const rowSearchStart = Math.max(2, Math.floor(height * 0.24));
  const rowSearchEnd = Math.max(rowSearchStart + 2, Math.min(height - 4, Math.floor(height * 0.9)));

  for (let col = 0; col < width; col++) {
    const columnArray = new Array<number>(height);
    for (let row = 0; row < height; row++) {
      const value = Number(source[row * width + col]);
      columnValues[row] = Number.isFinite(value) ? value : -1000;
      columnArray[row] = columnValues[row];
    }

    smoothFloatSeries(columnValues, height, columnScratch, 2);

    const sortedColumnValues = columnArray.slice().sort((a, b) => a - b);
    const columnP25 = percentile(sortedColumnValues, 0.25);
    const columnP45 = percentile(sortedColumnValues, 0.45);
    const columnP7 = percentile(sortedColumnValues, 0.7);
    const columnP82 = percentile(sortedColumnValues, 0.82);
    const intensityScale = Math.max(120, columnP82 - columnP25);

    const gradientValues = new Array<number>(height);
    for (let row = 0; row < height; row++) {
      const prevValue = row > 0 ? columnValues[row - 1] : columnValues[row];
      const nextValue = row < height - 1 ? columnValues[row + 1] : columnValues[row];
      const gradient = Math.max(
        Math.abs(columnValues[row] - prevValue),
        Math.abs(nextValue - columnValues[row])
      );
      columnGradient[row] = gradient;
      gradientValues[row] = gradient;
    }
    const sortedGradientValues = gradientValues.slice().sort((a, b) => a - b);
    const edgeP5 = percentile(sortedGradientValues, 0.5);
    const edgeP82 = percentile(sortedGradientValues, 0.82);
    const edgeScale = Math.max(24, edgeP82 - edgeP5);

    let bestRow = Math.floor(height * 0.6);
    let bestScore = -Infinity;
    for (let row = rowSearchStart; row <= rowSearchEnd; row++) {
      const upperMean = meanFloatWindow(columnValues, row - 2, row);
      const lowerMean = meanFloatWindow(columnValues, row + 1, row + 4);
      const upperEdge = meanFloatWindow(columnGradient, row - 2, row + 1);
      const lowerEdge = meanFloatWindow(columnGradient, row + 1, row + 4);
      const localIntensity = clampNumber(
        (upperMean - (columnP45 - intensityScale * 0.08)) / Math.max(90, intensityScale * 0.85),
        0,
        1.5
      );
      const transitionContrast = clampNumber(
        (upperMean - lowerMean + 70) / Math.max(90, intensityScale * 0.7),
        0,
        1.5
      );
      const edgeSupport = clampNumber(
        (upperEdge - lowerEdge * 0.35 - edgeP5) / Math.max(18, edgeScale * 1.15),
        0,
        1.5
      );
      const depthBias = (row - rowSearchStart) / Math.max(1, rowSearchEnd - rowSearchStart);
      const candidateScore =
        localIntensity * 0.34 + transitionContrast * 0.38 + edgeSupport * 0.38 + depthBias * 0.12;
      if (candidateScore > bestScore) {
        bestScore = candidateScore;
        bestRow = row;
      }
    }

    toothBandBottomRows[col] = bestRow;
    columnEdgeProtectThresholds[col] = edgeP5 + Math.max(12, edgeScale * 0.3);
    columnIntensityProtectThresholds[col] = Math.max(columnP7, columnP45 + intensityScale * 0.28);
  }

  if (width > 2) {
    smoothFloatSeries(toothBandBottomRows, width, rowScratchA, 3);
    smoothFloatSeries(toothBandBottomRows, width, rowScratchB, 2);
    for (let col = 0; col < width; col++) {
      toothBandBottomRows[col] = clampNumber(
        toothBandBottomRows[col],
        rowSearchStart,
        rowSearchEnd
      );
    }
  }

  const rootPreserveRows = Math.max(2, Math.round(1.4 / Math.max(0.2, rowSpacingMm)));
  const taperRows = Math.max(rootPreserveRows + 6, Math.round(5.8 / Math.max(0.2, rowSpacingMm)));
  let attenuatedPixelCount = 0;
  let attenuationSum = 0;
  let maxAttenuation = 0;

  for (let col = 0; col < width; col++) {
    const bandBottomRow = Number(toothBandBottomRows[col]);
    const edgeProtectThreshold = Number(columnEdgeProtectThresholds[col]);
    const intensityProtectThreshold = Number(columnIntensityProtectThresholds[col]);
    for (let row = 0; row < height; row++) {
      const depthRows = row - bandBottomRow - 0.5;
      if (depthRows <= 0) {
        continue;
      }

      const baseSuppression = smoothstep01(
        (depthRows - rootPreserveRows) / Math.max(1, taperRows - rootPreserveRows)
      );
      if (baseSuppression <= 0) {
        continue;
      }

      const pixelIndex = row * width + col;
      const value = Number(source[pixelIndex]);
      if (!Number.isFinite(value)) {
        continue;
      }

      const prevValue = row > 0 ? Number(source[pixelIndex - width]) : value;
      const nextValue = row < height - 1 ? Number(source[pixelIndex + width]) : value;
      const localEdge = Math.max(Math.abs(value - prevValue), Math.abs(nextValue - value));
      const edgeProtect = clampNumber(
        (localEdge - edgeProtectThreshold) / Math.max(20, edgeProtectThreshold * 0.75),
        0,
        1
      );
      const intensityProtect = clampNumber(
        (value - intensityProtectThreshold) /
          Math.max(80, Math.abs(intensityProtectThreshold) * 0.25 + 120),
        0,
        1
      );
      const structureProtect = Math.min(1, Math.max(edgeProtect, intensityProtect * 0.85));
      const attenuation = baseSuppression * (1 - 0.88 * structureProtect);
      if (attenuation <= 0.005) {
        continue;
      }
      suppressionWeights[pixelIndex] = attenuation;
      // Apply actual pixel darkening — blend toward dark target
      const darkTarget = Math.min(-800, value * 0.1);
      pixelData[pixelIndex] = value + (darkTarget - value) * attenuation;
      attenuatedPixelCount++;
      attenuationSum += attenuation;
      if (attenuation > maxAttenuation) {
        maxAttenuation = attenuation;
      }
    }
  }

  let minBandBottomRow = Infinity;
  let maxBandBottomRow = -Infinity;
  let bandBottomRowSum = 0;
  for (let col = 0; col < width; col++) {
    const value = Number(toothBandBottomRows[col]);
    if (value < minBandBottomRow) {
      minBandBottomRow = value;
    }
    if (value > maxBandBottomRow) {
      maxBandBottomRow = value;
    }
    bandBottomRowSum += value;
  }

  return {
    suppressionWeights,
    toothBandBottomRows,
    toothBandBottomRowStats: {
      min: Number.isFinite(minBandBottomRow) ? Math.round(minBandBottomRow * 1000) / 1000 : 0,
      max: Number.isFinite(maxBandBottomRow) ? Math.round(maxBandBottomRow * 1000) / 1000 : 0,
      mean: width > 0 ? Math.round((bandBottomRowSum / width) * 1000) / 1000 : 0,
      first8: Array.from(
        toothBandBottomRows.subarray(0, Math.min(8, toothBandBottomRows.length))
      ).map(value => Math.round(Number(value) * 1000) / 1000),
    },
    backgroundSuppression: {
      coverageFraction:
        pixelData.length > 0
          ? Math.round((attenuatedPixelCount / pixelData.length) * 100000) / 100000
          : 0,
      meanAttenuation:
        attenuatedPixelCount > 0
          ? Math.round((attenuationSum / attenuatedPixelCount) * 100000) / 100000
          : 0,
      maxAttenuation: Math.round(maxAttenuation * 100000) / 100000,
    },
  };
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

function runBandDPOptimization(
  scoreByColDepth: Float32Array,
  panoWidth: number,
  depthSamples: number,
  depthOffsetsMm: Float32Array,
  candidateCount: number,
  smoothScratch: Float32Array,
  smoothPasses: number
): Float32Array {
  const selectedDepthMm = new Float32Array(panoWidth);
  if (panoWidth <= 0 || depthSamples <= 0) {
    return selectedDepthMm;
  }

  // Phase 1: Select top-K candidates per column from band-specific scores
  const candidateDepthIndices = new Int16Array(panoWidth * candidateCount);
  candidateDepthIndices.fill(-1);
  const candidateScores = new Float32Array(panoWidth * candidateCount);
  for (let i = 0; i < candidateScores.length; i++) {
    candidateScores[i] = Number.NEGATIVE_INFINITY;
  }

  for (let col = 0; col < panoWidth; col++) {
    const candidateBase = col * candidateCount;
    for (let depth = 0; depth < depthSamples; depth++) {
      const score = Number(scoreByColDepth[col * depthSamples + depth]);
      if (!Number.isFinite(score)) continue;
      for (let slot = 0; slot < candidateCount; slot++) {
        if (score <= Number(candidateScores[candidateBase + slot])) continue;
        for (let shift = candidateCount - 1; shift > slot; shift--) {
          candidateScores[candidateBase + shift] = candidateScores[candidateBase + shift - 1];
          candidateDepthIndices[candidateBase + shift] =
            candidateDepthIndices[candidateBase + shift - 1];
        }
        candidateScores[candidateBase + slot] = score;
        candidateDepthIndices[candidateBase + slot] = depth;
        break;
      }
    }
  }

  // Phase 2: DP with smoothness penalty
  const prevDp = new Float32Array(candidateCount);
  const currDp = new Float32Array(candidateCount);
  const backPointers = new Int16Array(panoWidth * candidateCount);
  backPointers.fill(-1);
  for (let slot = 0; slot < candidateCount; slot++) {
    prevDp[slot] = Number.POSITIVE_INFINITY;
    currDp[slot] = Number.POSITIVE_INFINITY;
  }

  // Initialize first column
  for (let slot = 0; slot < candidateCount; slot++) {
    const depthIndex = Number(candidateDepthIndices[slot]);
    if (depthIndex < 0) continue;
    prevDp[slot] = -Number(candidateScores[slot]);
  }

  // Forward pass
  for (let col = 1; col < panoWidth; col++) {
    for (let slot = 0; slot < candidateCount; slot++) {
      currDp[slot] = Number.POSITIVE_INFINITY;
    }
    const candidateBase = col * candidateCount;
    const previousCandidateBase = (col - 1) * candidateCount;

    for (let slot = 0; slot < candidateCount; slot++) {
      const depthIndex = Number(candidateDepthIndices[candidateBase + slot]);
      if (depthIndex < 0) continue;
      const depthMm = Number(depthOffsetsMm[depthIndex]);
      let bestPrevSlot = -1;
      let bestCost = Number.POSITIVE_INFINITY;

      for (let prevSlot = 0; prevSlot < candidateCount; prevSlot++) {
        const prevDepthIndex = Number(candidateDepthIndices[previousCandidateBase + prevSlot]);
        if (prevDepthIndex < 0 || !Number.isFinite(prevDp[prevSlot])) continue;
        const prevDepthMm = Number(depthOffsetsMm[prevDepthIndex]);
        const jumpMm = Math.abs(depthMm - prevDepthMm);
        const transitionCost = 0.28 * jumpMm + 0.18 * Math.max(0, jumpMm - 0.5);
        const candidateCost =
          prevDp[prevSlot] + transitionCost - Number(candidateScores[candidateBase + slot]);
        if (candidateCost < bestCost) {
          bestCost = candidateCost;
          bestPrevSlot = prevSlot;
        }
      }

      currDp[slot] = bestCost;
      backPointers[candidateBase + slot] = bestPrevSlot;
    }

    prevDp.set(currDp);
  }

  // Backtrack
  let bestFinalSlot = 0;
  let bestFinalCost = Number.POSITIVE_INFINITY;
  const finalCandidateBase = (panoWidth - 1) * candidateCount;
  for (let slot = 0; slot < candidateCount; slot++) {
    if (prevDp[slot] < bestFinalCost) {
      bestFinalCost = prevDp[slot];
      bestFinalSlot = slot;
    }
  }

  let backtrackSlot = bestFinalSlot;
  for (let col = panoWidth - 1; col >= 0; col--) {
    const candidateBase = col * candidateCount;
    const depthIndex = Math.max(0, Number(candidateDepthIndices[candidateBase + backtrackSlot]));
    selectedDepthMm[col] = Number(depthOffsetsMm[depthIndex]);
    const prevSlot = Number(backPointers[candidateBase + backtrackSlot]);
    backtrackSlot = prevSlot >= 0 ? prevSlot : 0;
  }

  // Smooth
  smoothFloatSeries(selectedDepthMm, panoWidth, smoothScratch, smoothPasses);

  return selectedDepthMm;
}

function generatePanorama(
  input: CPRWorkerInput,
  options?: {
    requestedReconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
    gpuFallbackCause?: 'gpu-render-failed' | 'gpu-phase2-gate-failed' | null;
  }
): {
  pixelData: Float32Array;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
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
  } = input;

  const requestedReconstructionMode = resolveReconstructionMode(
    options?.requestedReconstructionMode ?? input.reconstructionMode
  );
  const candidateReconstructionMode = resolveReconstructionMode(input.reconstructionMode);
  const reconstructionMode = candidateReconstructionMode;
  const allowLegacyFallback = input.allowLegacyFallback === true;
  const rigidVerticalSliceMode = input.rigidVerticalSliceMode === true;
  const enableVirtualPanoPhase1 = reconstructionMode !== 'legacy';
  const enableVirtualPanoRender = reconstructionMode === 'virtualPano';
  const shouldComputeVirtualPano = enableVirtualPanoPhase1;
  const _t0_start = performance.now();
  const hasWorldToIndex = isMat4ArrayLike(worldToIndex);
  const invDir = hasWorldToIndex ? [] : invertMatrix3(direction);
  const pixelData = new Float32Array(panoWidth * panoHeight);
  let minValue = Infinity;
  let maxValue = -Infinity;
  const aggregationMode = aggregation === 'MEAN' ? 'MEAN' : 'MIP';
  const isMeanAggregation = aggregationMode === 'MEAN';
  const robustMipTopCount = 1;
  const scalarPolicy = resolveScalarSamplingPolicy(input);
  const safeSlope = scalarPolicy.safeSlope;
  const safeIntercept = scalarPolicy.safeIntercept;
  const unsignedPackedArtifactDetected = scalarPolicy.unsignedPackedArtifactDetected;
  const shouldNormalizeStoredValues = scalarPolicy.shouldNormalizeStoredValues;
  const normalizeStoredSample = scalarPolicy.normalizeStoredSample;
  const requestedModalityLutApplied = scalarPolicy.requestedModalityLutApplied;
  const shouldApplyModalityLut = scalarPolicy.shouldApplyModalityLut;
  const lutSamplePreview: number[] = [];
  const safeInterpolationOobValue = scalarPolicy.safeInterpolationOobValue;

  // Prefer precomputed vertical direction from orchestrator to keep main thread and worker
  // in exact agreement. Fallback to direction matrix K-axis if missing.
  const effectiveVerticalDir =
    Array.isArray(verticalDir) && verticalDir.length >= 3
      ? normalize3([verticalDir[0], verticalDir[1], verticalDir[2]])
      : normalize3([direction[6] ?? 0, direction[7] ?? 0, direction[8] ?? 1]);
  const verticalDirs: Array<[number, number, number]> = new Array(panoWidth);
  {
    let previousVerticalDir: [number, number, number] | null = null;
    for (let col = 0; col < panoWidth; col++) {
      const frameVerticalDirCandidate =
        Array.isArray(frames[col]?.S) && frames[col].S!.length >= 3
          ? normalize3([frames[col].S![0], frames[col].S![1], frames[col].S![2]])
          : effectiveVerticalDir;
      let frameVerticalDir = frameVerticalDirCandidate;
      if (previousVerticalDir && dot3(previousVerticalDir, frameVerticalDir) < 0) {
        frameVerticalDir = [-frameVerticalDir[0], -frameVerticalDir[1], -frameVerticalDir[2]];
      }
      verticalDirs[col] = frameVerticalDir;
      previousVerticalDir = frameVerticalDir;
    }
  }
  const vertHalfMm =
    Number.isFinite(requestedVertHalfMm) && Number(requestedVertHalfMm) > 0
      ? Number(requestedVertHalfMm)
      : 15.0;
  const panoHeightDen = Math.max(1, panoHeight - 1);
  const baseSlabSampleCount = Math.max(1, Math.floor(slabSamples));
  const positiveSpacings = spacing.filter(value => Number.isFinite(value) && Number(value) > 0);
  const minVolumeSpacingMm = positiveSpacings.length ? Math.min(...positiveSpacings) : 1;
  const slabSampleCount = baseSlabSampleCount;
  const slabStepMm = slabSampleCount > 1 ? (slabHalfThicknessMm * 2) / (slabSampleCount - 1) : 0;
  const focalTroughSigmaMm =
    slabHalfThicknessMm > 0
      ? Math.max(
          0.35,
          minVolumeSpacingMm * 0.6,
          slabHalfThicknessMm * (isMeanAggregation ? 0.52 : 0.48)
        )
      : 0.35;
  const focalTroughSigmaSq2 = 2 * focalTroughSigmaMm * focalTroughSigmaMm;
  const [nx, ny, nz] = dimensions;
  let selectedReconstructionMode: 'legacy' | 'virtualPanoPhase1' | 'virtualPano' = 'legacy';

  // Compute a gentle fit offset along the vertical direction to keep the
  // requested sampling window mostly inside the volume. This should only
  // nudge the requested jaw band, not override it.
  let fittedVerticalCenterOffsetMm = 0;
  const volumeMinProjectionByCol = new Float32Array(panoWidth);
  const volumeMaxProjectionByCol = new Float32Array(panoWidth);
  volumeMinProjectionByCol.fill(Number.NEGATIVE_INFINITY);
  volumeMaxProjectionByCol.fill(Number.POSITIVE_INFINITY);
  if (
    !rigidVerticalSliceMode &&
    nx > 1 &&
    ny > 1 &&
    nz > 1 &&
    Array.isArray(frames) &&
    frames.length > 0
  ) {
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
    const cornerWorlds = cornerIndices.map(([ci, cj, ck]) =>
      indexToWorld(ci, cj, ck, origin, spacing, direction)
    );
    let minRequiredOffset = -Infinity;
    let maxAllowedOffset = Infinity;

    for (let idx = 0; idx < frames.length; idx++) {
      const verticalDirForFrame = verticalDirs[idx] || effectiveVerticalDir;
      let minProjection = Infinity;
      let maxProjection = -Infinity;
      for (let cornerIdx = 0; cornerIdx < cornerWorlds.length; cornerIdx++) {
        const projection = dot3(cornerWorlds[cornerIdx], verticalDirForFrame);
        if (projection < minProjection) {
          minProjection = projection;
        }
        if (projection > maxProjection) {
          maxProjection = projection;
        }
      }

      if (Number.isFinite(minProjection) && Number.isFinite(maxProjection)) {
        volumeMinProjectionByCol[idx] = minProjection;
        volumeMaxProjectionByCol[idx] = maxProjection;
        const frameProjection = dot3(frames[idx].position, verticalDirForFrame);
        const requiredOffsetForLowerSide = minProjection + vertHalfMm - frameProjection;
        const allowedOffsetForUpperSide = maxProjection - vertHalfMm - frameProjection;
        if (requiredOffsetForLowerSide > minRequiredOffset) {
          minRequiredOffset = requiredOffsetForLowerSide;
        }
        if (allowedOffsetForUpperSide < maxAllowedOffset) {
          maxAllowedOffset = allowedOffsetForUpperSide;
        }
      }
    }
    if (Number.isFinite(minRequiredOffset) && Number.isFinite(maxAllowedOffset)) {
      if (minRequiredOffset <= maxAllowedOffset) {
        fittedVerticalCenterOffsetMm = Math.max(minRequiredOffset, Math.min(0, maxAllowedOffset));
      } else {
        // No single offset can fit the entire window for all frames; choose least-violation midpoint.
        fittedVerticalCenterOffsetMm = (minRequiredOffset + maxAllowedOffset) / 2;
      }
    }
  }

  const baseCenterOffsetLimitMm = Math.min(8, Math.max(3.5, vertHalfMm * 0.5));
  const fitOffsetLimitMm = Math.min(
    3.2,
    Math.max(1.2, Math.min(baseCenterOffsetLimitMm * 0.6, vertHalfMm * 0.18))
  );
  const requestedCenterOffsetMm = rigidVerticalSliceMode
    ? 0
    : Number.isFinite(requestedVerticalCenterOffsetMm)
      ? Number(requestedVerticalCenterOffsetMm)
      : 0;
  const clampedRequestedCenterOffsetMm = rigidVerticalSliceMode
    ? 0
    : Math.max(
        -baseCenterOffsetLimitMm,
        Math.min(baseCenterOffsetLimitMm, requestedCenterOffsetMm)
      );
  const clampedFittedVerticalCenterOffsetMm = rigidVerticalSliceMode
    ? 0
    : clampNumber(fittedVerticalCenterOffsetMm, -fitOffsetLimitMm, fitOffsetLimitMm);
  let verticalCenterOffsetMm = clampedRequestedCenterOffsetMm + clampedFittedVerticalCenterOffsetMm;
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

  const _dbgFirstFive: Array<{ row: number; vi: number; vj: number; vk: number; sample: number }> =
    [];
  let _dbgChecked = 0;
  let _dbgOob = 0;
  const reductionDiagnostics = {
    brightTailEvaluatedCount: 0,
    brightTailCappedCount: 0,
    brightTailPreservedCount: 0,
    brightClusterPreservedCount: 0,
  };
  const twoPassEligibilityDiagnostics = {
    pass1SeedPixelCount: 0,
    pass1ForegroundPixelCount: 0,
    pass1ConnectedRootSupportCount: 0,
    pass2EligibleSampleCount: 0,
    pass2InBoundsSampleCount: 0,
    eligibleSampleFraction: 0,
    pass2FallbackNoEligibleCount: 0,
    pass2LowerBandInBoundsSampleCount: 0,
    pass2LowerBandEligibleSampleCount: 0,
    lowerBandEligibleFraction: 0,
    pass2SingleEligiblePixelCount: 0,
    pass2MultiEligiblePixelCount: 0,
  };
  const planeSize = panoWidth * panoHeight;
  const stackSize = planeSize * slabSampleCount;
  const slabValueBuffer = new Float32Array(slabSampleCount);
  const slabWeightBuffer = new Float32Array(slabSampleCount);
  const pass1IntensityStack = new Float32Array(stackSize);
  const pass1ProvisionalPano = new Float32Array(planeSize);
  const pass1SeedMask = new Uint8Array(stackSize);
  const pass1ForegroundMask = new Uint8Array(stackSize);
  const pass2EligibilityMask = new Uint8Array(stackSize);
  const centerRowByCol = new Int16Array(panoWidth);
  const halfHeightByCol = new Float32Array(panoWidth);
  const slabBaseWeights = new Float32Array(slabSampleCount);
  const slabCenterIndex = (slabSampleCount - 1) * 0.5;
  for (let s = 0; s < slabSampleCount; s++) {
    const slabOffset = slabSampleCount > 1 ? -slabHalfThicknessMm + s * slabStepMm : 0;
    slabBaseWeights[s] =
      slabSampleCount > 1 ? Math.exp(-(slabOffset * slabOffset) / focalTroughSigmaSq2) : 1;
  }
  const adaptiveCenterSearchHalfRangeMm = rigidVerticalSliceMode
    ? 0
    : Math.min(
        isMeanAggregation ? 5.4 : 4.2,
        Math.max(
          isMeanAggregation ? 1.6 : 1.2,
          effectiveVerticalHalfMm * (isMeanAggregation ? 0.34 : 0.24)
        )
      );
  const adaptiveCandidateCount = rigidVerticalSliceMode
    ? 1
    : Math.max(
        7,
        Math.min(
          11,
          Math.round(
            (adaptiveCenterSearchHalfRangeMm * 2) / Math.max(0.5, minVolumeSpacingMm * 2.5)
          ) + 1
        )
      );
  const adaptiveProfileSampleCount = 15;
  const adaptiveProfileLowerBandCount = Math.max(4, Math.floor(adaptiveProfileSampleCount * 0.32));
  const adaptiveProfileUpperBandCount = Math.max(5, Math.ceil(adaptiveProfileSampleCount * 0.5));
  const adaptiveCenterSmoothingRadiusCols = rigidVerticalSliceMode
    ? 0
    : Math.max(8, Math.min(18, Math.round(panoWidth / 40)));
  const adaptiveCenterGlobalBlend = rigidVerticalSliceMode ? 0 : isMeanAggregation ? 0.72 : 0.64;
  const baseAdaptiveCenterMaxDeviationMm = rigidVerticalSliceMode
    ? 0
    : Math.min(
        isMeanAggregation ? 4.2 : 3.4,
        Math.max(
          isMeanAggregation ? 1.6 : 1.2,
          effectiveVerticalHalfMm * (isMeanAggregation ? 0.26 : 0.2)
        )
      );
  const adaptiveCenterMaxAdjacentDeltaMm = rigidVerticalSliceMode
    ? 0
    : Math.max(
        minVolumeSpacingMm * 0.9,
        Math.min(
          isMeanAggregation ? 0.55 : 0.42,
          minVolumeSpacingMm * (isMeanAggregation ? 2.4 : 1.8)
        )
      );
  const adaptiveContinuityPenaltyScale = isMeanAggregation ? 10 : 8;
  const profileSampleBuffer = new Float32Array(adaptiveProfileSampleCount);
  const profileSampleScratch = new Float32Array(adaptiveProfileSampleCount);
  const finalRenderProfilePeakValues = new Float32Array(panoWidth);
  const finalRenderProfileFloorValues = new Float32Array(panoWidth);
  const localCenterOffsetsMm = new Float32Array(panoWidth);
  const localCenterMinOffsetsMm = new Float32Array(panoWidth);
  const localCenterMaxOffsetsMm = new Float32Array(panoWidth);
  const localCenterScratch = new Float32Array(panoWidth);
  const curvatureFactorRaw = new Float32Array(panoWidth);
  const curvatureFactorByCol = new Float32Array(panoWidth);
  const curvatureScratch = new Float32Array(panoWidth);
  const adaptiveCenterSearchHalfRangeByCol = new Float32Array(panoWidth);
  const adaptiveCenterMaxDeviationByCol = new Float32Array(panoWidth);
  const turnAngleDegreesByCol = new Float32Array(panoWidth);
  for (let col = 0; col < panoWidth; col++) {
    let turnAngleAccumDeg = 0;
    let turnAngleCount = 0;
    if (col > 0) {
      turnAngleAccumDeg += angleDegreesBetween3(frames[col - 1].T, frames[col].T);
      turnAngleCount++;
    }
    if (col + 1 < panoWidth) {
      turnAngleAccumDeg += angleDegreesBetween3(frames[col].T, frames[col + 1].T);
      turnAngleCount++;
    }
    const turnAngleDeg = turnAngleCount > 0 ? turnAngleAccumDeg / turnAngleCount : 0;
    turnAngleDegreesByCol[col] = turnAngleDeg;
    curvatureFactorRaw[col] = smoothstepRange(2, 7, turnAngleDeg);
  }
  if (panoWidth > 1) {
    for (let col = 0; col < panoWidth; col++) {
      let weightedSum = Number(curvatureFactorRaw[col]) * 0.5;
      let weightTotal = 0.5;
      if (col > 0) {
        weightedSum += Number(curvatureFactorRaw[col - 1]) * 0.25;
        weightTotal += 0.25;
      }
      if (col + 1 < panoWidth) {
        weightedSum += Number(curvatureFactorRaw[col + 1]) * 0.25;
        weightTotal += 0.25;
      }
      curvatureScratch[col] =
        weightTotal > 0 ? weightedSum / weightTotal : Number(curvatureFactorRaw[col]);
    }
    curvatureFactorByCol.set(curvatureScratch);
  } else {
    curvatureFactorByCol.set(curvatureFactorRaw);
  }
  for (let col = 0; col < panoWidth; col++) {
    const curvatureFactor = Number(curvatureFactorByCol[col]);
    adaptiveCenterSearchHalfRangeByCol[col] = rigidVerticalSliceMode
      ? 0
      : adaptiveCenterSearchHalfRangeMm * (1 + 0.2 * curvatureFactor);
    adaptiveCenterMaxDeviationByCol[col] = rigidVerticalSliceMode
      ? 0
      : baseAdaptiveCenterMaxDeviationMm * (1 + 0.15 * curvatureFactor);
  }
  const sampleReducedPoint = (
    bx: number,
    by: number,
    bz: number,
    slabDirX: number,
    slabDirY: number,
    slabDirZ: number,
    recordOobStats: boolean,
    debugCaptureRow: number | null,
    captureIntensityStack?: Float32Array,
    capturePixelIndex: number = -1
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
        if (captureIntensityStack && capturePixelIndex >= 0) {
          captureIntensityStack[stackIndex(s, capturePixelIndex, planeSize)] = Number.NaN;
        }
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
      if (captureIntensityStack && capturePixelIndex >= 0) {
        captureIntensityStack[stackIndex(s, capturePixelIndex, planeSize)] = sample;
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

    if (isMeanAggregation) {
      return computeRawMean(slabValueBuffer, validSampleCount);
    }
    return computePureMax(slabValueBuffer, validSampleCount);
  };

  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
    const [vertDirX, vertDirY, vertDirZ] = verticalDirs[col] || effectiveVerticalDir;
    if (rigidVerticalSliceMode) {
      localCenterMinOffsetsMm[col] = verticalCenterOffsetMm;
      localCenterMaxOffsetsMm[col] = verticalCenterOffsetMm;
      localCenterOffsetsMm[col] = verticalCenterOffsetMm;
      continue;
    }
    const localSearchHalfRangeMm = Number(adaptiveCenterSearchHalfRangeByCol[col]);
    const curvatureFactor = Number(curvatureFactorByCol[col]);
    const globalCenterPenaltyWeight = 26 * (1 - 0.65 * curvatureFactor);
    const continuityPenaltyWeight = adaptiveContinuityPenaltyScale * (1 - 0.35 * curvatureFactor);
    const minProjectionForFrame = Number(volumeMinProjectionByCol[col]);
    const maxProjectionForFrame = Number(volumeMaxProjectionByCol[col]);
    const frameProjection = dot3(frame.position, [vertDirX, vertDirY, vertDirZ]);
    const perFrameMinCenterOffsetMm = Number.isFinite(minProjectionForFrame)
      ? minProjectionForFrame + effectiveVerticalHalfMm - frameProjection
      : verticalCenterOffsetMm - localSearchHalfRangeMm;
    const perFrameMaxCenterOffsetMm = Number.isFinite(maxProjectionForFrame)
      ? maxProjectionForFrame - effectiveVerticalHalfMm - frameProjection
      : verticalCenterOffsetMm + localSearchHalfRangeMm;
    const searchMinCenterOffsetMm = Math.max(
      perFrameMinCenterOffsetMm,
      verticalCenterOffsetMm - localSearchHalfRangeMm
    );
    const searchMaxCenterOffsetMm = Math.min(
      perFrameMaxCenterOffsetMm,
      verticalCenterOffsetMm + localSearchHalfRangeMm
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
        adaptiveCandidateCount <= 1
          ? 0.5
          : candidateIndex / Math.max(1, adaptiveCandidateCount - 1);
      const candidateCenterOffsetMm = safeSearchMin + (safeSearchMax - safeSearchMin) * t;

      for (let sampleIndex = 0; sampleIndex < adaptiveProfileSampleCount; sampleIndex++) {
        const fraction =
          adaptiveProfileSampleCount <= 1
            ? 0.5
            : sampleIndex / Math.max(1, adaptiveProfileSampleCount - 1);
        const relativeOffsetMm = effectiveVerticalHalfMm - fraction * (effectiveVerticalHalfMm * 2);
        const sampleOffsetMm = candidateCenterOffsetMm + relativeOffsetMm;
        const bx = px + sampleOffsetMm * vertDirX;
        const by = py + sampleOffsetMm * vertDirY;
        const bz = pz + sampleOffsetMm * vertDirZ;
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
      const lowerBandValues = profileValues.slice(
        adaptiveProfileSampleCount - adaptiveProfileLowerBandCount
      );
      const upperBandValues = profileValues.slice(0, adaptiveProfileUpperBandCount);
      const profileP20 = percentile(profileValues, 0.2);
      const profileP50 = percentile(profileValues, 0.5);
      const profileP80 = percentile(profileValues, 0.8);
      const profileMin = profileValues.length > 0 ? Math.min(...profileValues) : profileP20;
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
        Math.max(0, lowerBandP50 + 350) * 2.4 -
        lowerBandBrightFraction * 420 -
        Math.max(0, profileMin + 650) * 0.42 -
        Math.max(0, profileP20 + 450) * 0.18 -
        Math.abs(candidateCenterOffsetMm - verticalCenterOffsetMm) * globalCenterPenaltyWeight -
        Math.abs(candidateCenterOffsetMm - previousCenterOffsetMm) * continuityPenaltyWeight;

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

  if (!rigidVerticalSliceMode && panoWidth > 2) {
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
    smoothFloatSeries(
      localCenterScratch,
      panoWidth,
      localCenterOffsetsMm,
      isMeanAggregation ? 3 : 2
    );
    localCenterOffsetsMm.set(localCenterScratch);
  }

  if (!rigidVerticalSliceMode) {
    for (let col = 0; col < panoWidth; col++) {
      const regularizedMinCenterOffsetMm = Math.max(
        localCenterMinOffsetsMm[col],
        verticalCenterOffsetMm - adaptiveCenterMaxDeviationByCol[col]
      );
      const regularizedMaxCenterOffsetMm = Math.min(
        localCenterMaxOffsetsMm[col],
        verticalCenterOffsetMm + adaptiveCenterMaxDeviationByCol[col]
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
  }

  if (!rigidVerticalSliceMode && panoWidth > 1) {
    for (let pass = 0; pass < 2; pass++) {
      for (let col = 1; col < panoWidth; col++) {
        const regularizedMinCenterOffsetMm = Math.max(
          localCenterMinOffsetsMm[col],
          verticalCenterOffsetMm - adaptiveCenterMaxDeviationByCol[col]
        );
        const regularizedMaxCenterOffsetMm = Math.min(
          localCenterMaxOffsetsMm[col],
          verticalCenterOffsetMm + adaptiveCenterMaxDeviationByCol[col]
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
          verticalCenterOffsetMm - adaptiveCenterMaxDeviationByCol[col]
        );
        const regularizedMaxCenterOffsetMm = Math.min(
          localCenterMaxOffsetsMm[col],
          verticalCenterOffsetMm + adaptiveCenterMaxDeviationByCol[col]
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
  const _t1_afterAdaptiveCenter = performance.now();

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
  const meanLocalCenterOffsetMm =
    panoWidth > 0 ? localCenterOffsetSumMm / panoWidth : verticalCenterOffsetMm;
  const panoCenterRow = panoHeightDen / 2;
  if (rigidVerticalSliceMode) {
    for (let col = 0; col < panoWidth; col++) {
      const frame = frames[col];
      const [px, py, pz] = frame.position;
      const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
      const [vertDirX, vertDirY, vertDirZ] = verticalDirs[col] || effectiveVerticalDir;
      const columnVerticalCenterOffsetMm = Number(localCenterOffsetsMm[col]);
      centerRowByCol[col] = Math.round(panoCenterRow);
      halfHeightByCol[col] = Math.max(1, panoCenterRow);

      for (let row = 0; row < panoHeight; row++) {
        const vertOffsetMm =
          columnVerticalCenterOffsetMm + (effectiveVerticalHalfMm - row * vertStepMm);
        const bx = px + vertOffsetMm * vertDirX;
        const by = py + vertOffsetMm * vertDirY;
        const bz = pz + vertOffsetMm * vertDirZ;
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

    const _t2_afterSimpleRender = performance.now();
    const finalRange = computeArrayMinMax(pixelData);
    minValue = finalRange.minValue;
    maxValue = finalRange.maxValue;
    const windowWidth = maxValue - minValue;
    const windowCenter = minValue + windowWidth / 2;
    const requestedRenderBackend = input.renderBackend === 'cpu' ? 'cpu' : 'gpu';
    const rigidFallbackReason =
      options?.gpuFallbackCause === 'gpu-phase2-gate-failed'
        ? 'gpu-phase2-gate-failed-cpu-legacy-fallback'
        : options?.gpuFallbackCause === 'gpu-render-failed'
          ? 'gpu-render-failed-cpu-legacy-fallback'
          : null;
    const rigidOutputSelection = {
      requestedReconstructionMode,
      candidateReconstructionMode: reconstructionMode,
      selectedReconstructionMode: 'legacy',
      virtualPanoAttempted: false,
      virtualPanoAccepted: false,
      virtualPanoRejected: false,
      acceptedByLowerBandTolerance: false,
      legacyFallbackAllowed: allowLegacyFallback,
      rejectReasons: [] as string[],
      fallbackReason: rigidFallbackReason,
    };
    const diagnosticPayload = {
      renderBackend: 'cpu',
      requestedRenderBackend,
      pipelineMode: 'legacy',
      fallbackReason: rigidFallbackReason,
      reconstructionMode: 'legacy',
      requestedReconstructionMode,
      candidateReconstructionMode: reconstructionMode,
      phase2GatePassed: options?.gpuFallbackCause === 'gpu-phase2-gate-failed' ? false : null,
      outputSelection: rigidOutputSelection,
      legacyFallbackAllowed: allowLegacyFallback,
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
      verticalAxisMode: 'frameS',
      slabDir: slabDirs?.[0]
        ? {
            N_slab: slabDirs[0],
            tangent: frames[0].T ?? null,
          }
        : null,
      verticalWindowMode: 'rigid-spline-slice-window',
      verticalHalfMm: effectiveVerticalHalfMm,
      globalVerticalCenterOffsetMm: 0,
      baseVerticalCenterOffsetMm: verticalCenterOffsetMm,
      verticalCenterOffsetMm: meanLocalCenterOffsetMm,
      rigidSliceMode: {
        enabled: true,
        projectedSplinePlaneLocked: true,
        requestedCenterOffsetMm: verticalCenterOffsetMm,
        fittedCenterOffsetAppliedMm: 0,
        localOffsetsLocked: true,
      },
      localCenterOffsetMmStats: {
        min: minLocalCenterOffsetMm,
        max: maxLocalCenterOffsetMm,
        mean: meanLocalCenterOffsetMm,
        maxDeviationFromGlobal: Math.max(
          Math.abs(minLocalCenterOffsetMm - verticalCenterOffsetMm),
          Math.abs(maxLocalCenterOffsetMm - verticalCenterOffsetMm)
        ),
        maxAdjacentDeltaMm: maxLocalCenterAdjacentDeltaMm,
        first8: Array.from(
          localCenterOffsetsMm.subarray(0, Math.min(8, localCenterOffsetsMm.length))
        ).map(value => Math.round(Number(value) * 1000) / 1000),
      },
      localCenterOffsetsMm: Array.from(localCenterOffsetsMm).map(
        value => Math.round(Number(value) * 1000) / 1000
      ),
      requestedVerticalCenterOffsetMm: verticalCenterOffsetMm,
      fittedVerticalCenterOffsetMm: 0,
      adaptiveVerticalSearch: {
        enabled: false,
        mode: 'disabled-rigid-slice',
        halfRangeMm: 0,
        candidateCount: 1,
        profileSamples: 1,
        smoothingRadiusCols: 0,
        globalBlend: 0,
        maxDeviationMm: 0,
        maxAdjacentDeltaMm: 0,
      },
      slabSampling: {
        requestedSamples: baseSlabSampleCount,
        effectiveSamples: slabSampleCount,
        slabHalfThicknessMm,
        aggregation: aggregationMode,
        focalTroughSigmaMm,
        reduction: isMeanAggregation ? 'raw-mean' : 'pure-max',
      },
      denoise: {
        blend: 0,
        requestedBlend: 0,
        applied: false,
        virtualRenderUsedAsOutput: false,
      },
      timingMs: {
        adaptiveCenterSearch: Math.round(_t1_afterAdaptiveCenter - _t0_start),
        pass1And2TwoPassRender: Math.round(_t2_afterSimpleRender - _t1_afterAdaptiveCenter),
        virtualPanoPhase12: 0,
        suppressionAndDenoise: 0,
        diagnosticAssembly: 0,
        total: Math.round(_t2_afterSimpleRender - _t0_start),
      },
      outputDisplayWindow: {
        lower: minValue,
        upper: maxValue,
        windowWidth,
        windowCenter,
      },
      virtualPanoPhase12: {
        enabled: false,
        phase: 1,
        reconstructionMode,
        skippedReason: 'RECONSTRUCTION_MODE_DISABLED',
      },
      virtualPanoRender: {
        enabled: false,
        usedAsOutput: false,
        skippedReason: 'RECONSTRUCTION_MODE_DISABLED',
      },
      backgroundSuppressionMode: 'disabled-rigid-pass-through',
      toothBandBottomRowStats: {
        min: 0,
        max: 0,
        mean: 0,
        first8: [] as number[],
      },
      backgroundSuppression: {
        coverageFraction: 0,
        meanAttenuation: 0,
        maxAttenuation: 0,
      },
      twoPassEligibilityDiagnostics,
      reductionDiagnostics,
      firstFrameWorldPos: frames?.[0]?.position ?? null,
      lastFrameWorldPos: frames?.[frames.length - 1]?.position ?? null,
    };
    const outputPixelPreview = Array.from(pixelData.subarray(0, Math.min(5, pixelData.length)));
    const outputSignature = computeOutputSignature(pixelData);
    return {
      pixelData,
      minValue,
      maxValue,
      windowWidth,
      windowCenter,
      lutSamplePreview,
      outputPixelPreview,
      modalityLutApplied: shouldApplyModalityLut,
      requestedModalityLutApplied,
      storedValueNormalizationApplied: shouldNormalizeStoredValues,
      unsignedPackedArtifactDetected,
      effectiveVerticalHalfMm,
      verticalCenterOffsetMm: meanLocalCenterOffsetMm,
      adaptiveVerticalIntervalCount: 1,
      effectiveSlabSampleCount: slabSampleCount,
      robustMipTopCount,
      denoiseApplied: false,
      diagnosticPayload,
      outputSignature,
    };
  }
  const queue = new Int32Array(planeSize);
  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
    const [vertDirX, vertDirY, vertDirZ] = verticalDirs[col] || effectiveVerticalDir;
    const columnVerticalCenterOffsetMm = localCenterOffsetsMm[col];
    centerRowByCol[col] = Math.round(panoCenterRow);
    halfHeightByCol[col] = Math.max(1, panoCenterRow);
    for (let sampleIndex = 0; sampleIndex < adaptiveProfileSampleCount; sampleIndex++) {
      const fraction =
        adaptiveProfileSampleCount <= 1
          ? 0.5
          : sampleIndex / Math.max(1, adaptiveProfileSampleCount - 1);
      const relativeOffsetMm = effectiveVerticalHalfMm - fraction * (effectiveVerticalHalfMm * 2);
      const sampleOffsetMm = columnVerticalCenterOffsetMm + relativeOffsetMm;
      const bx = px + sampleOffsetMm * vertDirX;
      const by = py + sampleOffsetMm * vertDirY;
      const bz = pz + sampleOffsetMm * vertDirZ;
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
    finalRenderProfilePeakValues[col] =
      profileSampleBuffer[
        Math.max(
          0,
          Math.min(adaptiveProfileSampleCount - 1, Math.floor(adaptiveProfileSampleCount / 2))
        )
      ];
    const profileValues = Array.from(profileSampleBuffer.subarray(0, adaptiveProfileSampleCount));
    finalRenderProfileFloorValues[col] = percentile(profileValues, 0.25);
  }
  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
    const [vertDirX, vertDirY, vertDirZ] = verticalDirs[col] || effectiveVerticalDir;
    const columnVerticalCenterOffsetMm = localCenterOffsetsMm[col];

    for (let row = 0; row < panoHeight; row++) {
      const vertOffsetMm =
        columnVerticalCenterOffsetMm + (effectiveVerticalHalfMm - row * vertStepMm);
      const pixelIdx = planeIndex(col, row, panoWidth);
      const bx = px + vertOffsetMm * vertDirX;
      const by = py + vertOffsetMm * vertDirY;
      const bz = pz + vertOffsetMm * vertDirZ;
      pass1ProvisionalPano[pixelIdx] = sampleReducedPoint(
        bx,
        by,
        bz,
        slabDirX,
        slabDirY,
        slabDirZ,
        true,
        col === 0 && row < 5 ? row : null,
        pass1IntensityStack,
        pixelIdx
      );
    }
  }
  for (let col = 0; col < panoWidth; col++) {
    const centerRow = centerRowByCol[col];
    const halfHeight = Math.max(halfHeightByCol[col], 1e-6);
    const peak = Number(finalRenderProfilePeakValues[col]);
    const floor = Number(finalRenderProfileFloorValues[col]);
    const bandSpan = Math.max(peak - floor, 1e-6);
    const seedThreshold = floor + 0.35 * bandSpan;
    const growThreshold = floor + 0.18 * bandSpan;
    for (let row = 0; row < panoHeight; row++) {
      const pixelIdx = planeIndex(col, row, panoWidth);
      const provisionalValue = Number(pass1ProvisionalPano[pixelIdx]);
      const yNorm = (row - centerRow) / halfHeight;
      const canSeedRow = yNorm >= -0.35 && yNorm <= 0.45 && provisionalValue >= seedThreshold;
      const canGrowRow = yNorm >= -0.45 && yNorm <= 1.15;
      for (let depth = 0; depth < slabSampleCount; depth++) {
        const idx = stackIndex(depth, pixelIdx, planeSize);
        const value = Number(pass1IntensityStack[idx]);
        if (!Number.isFinite(value)) {
          continue;
        }
        if (canGrowRow && value >= growThreshold && pass1ForegroundMask[idx] === 0) {
          pass1ForegroundMask[idx] = 1;
          twoPassEligibilityDiagnostics.pass1ForegroundPixelCount++;
        }
        if (canSeedRow && value >= seedThreshold && pass1SeedMask[idx] === 0) {
          pass1SeedMask[idx] = 1;
          twoPassEligibilityDiagnostics.pass1SeedPixelCount++;
          if (pass1ForegroundMask[idx] === 0) {
            pass1ForegroundMask[idx] = 1;
            twoPassEligibilityDiagnostics.pass1ForegroundPixelCount++;
          }
        }
      }
    }
  }
  for (let depth = 0; depth < slabSampleCount; depth++) {
    const depthOffset = depth * planeSize;
    let queueHead = 0;
    let queueTail = 0;
    for (let pixelIdx = 0; pixelIdx < planeSize; pixelIdx++) {
      const idx = depthOffset + pixelIdx;
      if (pass1SeedMask[idx] !== 1 || pass2EligibilityMask[idx] === 1) {
        continue;
      }
      pass2EligibilityMask[idx] = 1;
      queue[queueTail++] = pixelIdx;
      twoPassEligibilityDiagnostics.pass1ConnectedRootSupportCount++;
    }
    while (queueHead < queueTail) {
      const pixelIdx = queue[queueHead++];
      const row = Math.floor(pixelIdx / panoWidth);
      const col = pixelIdx - row * panoWidth;

      if (col > 0) {
        const neighborPixelIdx = pixelIdx - 1;
        const neighborIdx = depthOffset + neighborPixelIdx;
        if (pass2EligibilityMask[neighborIdx] === 0 && pass1ForegroundMask[neighborIdx] === 1) {
          pass2EligibilityMask[neighborIdx] = 1;
          queue[queueTail++] = neighborPixelIdx;
          twoPassEligibilityDiagnostics.pass1ConnectedRootSupportCount++;
        }
      }
      if (col + 1 < panoWidth) {
        const neighborPixelIdx = pixelIdx + 1;
        const neighborIdx = depthOffset + neighborPixelIdx;
        if (pass2EligibilityMask[neighborIdx] === 0 && pass1ForegroundMask[neighborIdx] === 1) {
          pass2EligibilityMask[neighborIdx] = 1;
          queue[queueTail++] = neighborPixelIdx;
          twoPassEligibilityDiagnostics.pass1ConnectedRootSupportCount++;
        }
      }
      if (row > 0) {
        const neighborPixelIdx = pixelIdx - panoWidth;
        const neighborIdx = depthOffset + neighborPixelIdx;
        if (pass2EligibilityMask[neighborIdx] === 0 && pass1ForegroundMask[neighborIdx] === 1) {
          pass2EligibilityMask[neighborIdx] = 1;
          queue[queueTail++] = neighborPixelIdx;
          twoPassEligibilityDiagnostics.pass1ConnectedRootSupportCount++;
        }
      }
      if (row + 1 < panoHeight) {
        const neighborPixelIdx = pixelIdx + panoWidth;
        const neighborIdx = depthOffset + neighborPixelIdx;
        if (pass2EligibilityMask[neighborIdx] === 0 && pass1ForegroundMask[neighborIdx] === 1) {
          pass2EligibilityMask[neighborIdx] = 1;
          queue[queueTail++] = neighborPixelIdx;
          twoPassEligibilityDiagnostics.pass1ConnectedRootSupportCount++;
        }
      }
    }
  }

  minValue = Infinity;
  maxValue = -Infinity;

  for (let col = 0; col < panoWidth; col++) {
    for (let row = 0; row < panoHeight; row++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const yNorm = (row - centerRowByCol[col]) / Math.max(halfHeightByCol[col], 1e-6);
      let inBoundsSampleCount = 0;
      let eligibleSampleCount = 0;
      let bestFallbackDepth = -1;
      let bestFallbackDistance = Infinity;

      for (let depth = 0; depth < slabSampleCount; depth++) {
        const idx = stackIndex(depth, pixelIndex, planeSize);
        const value = Number(pass1IntensityStack[idx]);
        if (!Number.isFinite(value)) {
          continue;
        }
        inBoundsSampleCount++;
        if (yNorm > 0.5) {
          twoPassEligibilityDiagnostics.pass2LowerBandInBoundsSampleCount++;
        }
        const fallbackDistance = Math.abs(depth - slabCenterIndex);
        if (fallbackDistance < bestFallbackDistance) {
          bestFallbackDistance = fallbackDistance;
          bestFallbackDepth = depth;
        }
        if (pass2EligibilityMask[idx] !== 1) {
          continue;
        }
        slabValueBuffer[eligibleSampleCount] = value;
        slabWeightBuffer[eligibleSampleCount] = slabBaseWeights[depth];
        eligibleSampleCount++;
        if (yNorm > 0.5) {
          twoPassEligibilityDiagnostics.pass2LowerBandEligibleSampleCount++;
        }
      }

      twoPassEligibilityDiagnostics.pass2InBoundsSampleCount += inBoundsSampleCount;
      twoPassEligibilityDiagnostics.pass2EligibleSampleCount += eligibleSampleCount;

      let pixelValue = -1000;
      if (eligibleSampleCount <= 0) {
        twoPassEligibilityDiagnostics.pass2FallbackNoEligibleCount++;
        pixelValue =
          bestFallbackDepth >= 0
            ? Number(pass1IntensityStack[stackIndex(bestFallbackDepth, pixelIndex, planeSize)])
            : Number(pass1ProvisionalPano[pixelIndex]);
      } else if (eligibleSampleCount === 1) {
        twoPassEligibilityDiagnostics.pass2SingleEligiblePixelCount++;
        pixelValue = Number(slabValueBuffer[0]);
      } else {
        twoPassEligibilityDiagnostics.pass2MultiEligiblePixelCount++;
        if (isMeanAggregation) {
          sortSamplePairsAscending(slabValueBuffer, slabWeightBuffer, eligibleSampleCount);
          pixelValue = computeWinsorizedWeightedMean(
            slabValueBuffer,
            slabWeightBuffer,
            eligibleSampleCount,
            reductionDiagnostics
          );
        } else {
          sortSamplePairsAscending(slabValueBuffer, slabWeightBuffer, eligibleSampleCount);
          pixelValue = computeWeightedHighBandMean(
            slabValueBuffer,
            slabWeightBuffer,
            eligibleSampleCount,
            robustMipTopCount
          );
        }
      }

      pixelData[pixelIndex] = pixelValue;

      if (pixelValue < minValue) {
        minValue = pixelValue;
      }
      if (pixelValue > maxValue) {
        maxValue = pixelValue;
      }
    }
  }
  twoPassEligibilityDiagnostics.eligibleSampleFraction =
    twoPassEligibilityDiagnostics.pass2EligibleSampleCount /
    Math.max(1, twoPassEligibilityDiagnostics.pass2InBoundsSampleCount);
  twoPassEligibilityDiagnostics.lowerBandEligibleFraction =
    twoPassEligibilityDiagnostics.pass2LowerBandEligibleSampleCount /
    Math.max(1, twoPassEligibilityDiagnostics.pass2LowerBandInBoundsSampleCount);
  const _t2_afterTwoPassRender = performance.now();

  const sampleWorldIntensityForVirtualPano = (wx: number, wy: number, wz: number): number => {
    const [vi, vj, vk] = worldToVoxel(wx, wy, wz, origin, spacing, invDir, worldToIndex);
    if (
      vi < -0.5 ||
      vj < -0.5 ||
      vk < -0.5 ||
      vi > dimensions[0] - 0.5 ||
      vj > dimensions[1] - 0.5 ||
      vk > dimensions[2] - 0.5
    ) {
      return Number.NaN;
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
    }
    return Number.isFinite(sample) ? sample : Number.NaN;
  };

  let virtualPanoPhase12Diagnostics: Record<string, unknown>;
  let virtualPanoRenderDiagnostics: Record<string, unknown> = {
    enabled: enableVirtualPanoRender,
    usedAsOutput: false,
    skippedReason: enableVirtualPanoRender
      ? 'NOT_RENDERED'
      : enableVirtualPanoPhase1
        ? 'PHASE1_DIAGNOSTICS_ONLY'
        : 'RECONSTRUCTION_MODE_DISABLED',
  };
  let virtualPanoAcceptedByGate = false;
  let virtualPanoSelectedForOutput = false;
  if (shouldComputeVirtualPano) {
    const virtualPanoDepthHalfRangeMm = 6.0;
    const virtualPanoDepthStepMm = 0.25;
    const virtualPanoDepthSamples =
      Math.max(3, Math.round((virtualPanoDepthHalfRangeMm * 2) / virtualPanoDepthStepMm) + 1) | 0;
    const virtualPanoCandidateCount = 5;
    const virtualPanoPlaneSize = planeSize;
    const virtualPanoDepthOffsetsMm = new Float32Array(virtualPanoDepthSamples);
    for (let depth = 0; depth < virtualPanoDepthSamples; depth++) {
      virtualPanoDepthOffsetsMm[depth] =
        -virtualPanoDepthHalfRangeMm + depth * virtualPanoDepthStepMm;
    }
    const virtualPanoStack = new Float32Array(virtualPanoPlaneSize * virtualPanoDepthSamples);
    const virtualScoreByColDepth = new Float32Array(panoWidth * virtualPanoDepthSamples);
    const virtualSupportTiltByCol: Float32Array | undefined = undefined;
    const virtualSoftThresholdByCol = new Float32Array(panoWidth);
    const virtualHardThresholdByCol = new Float32Array(panoWidth);
    const virtualGradCapByCol = new Float32Array(panoWidth);
    const virtualThresholdScratch = new Float32Array(panoWidth);
    const virtualDpSmoothScratch = new Float32Array(panoWidth);

    const rowFromNormalizedOffset = (yNorm: number): number => {
      const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
      return Math.max(0, Math.min(panoHeight - 1, row));
    };

    const toothBandStartRow = Math.min(
      rowFromNormalizedOffset(-0.35),
      rowFromNormalizedOffset(0.55)
    );
    const toothBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const topBandStartRow = Math.min(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.05));
    const topBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.05));
    const bottomBandStartRow = Math.min(
      rowFromNormalizedOffset(0.15),
      rowFromNormalizedOffset(0.65)
    );
    const bottomBandEndRow = Math.max(rowFromNormalizedOffset(0.15), rowFromNormalizedOffset(0.65));
    const lowerBandStartRow = Math.min(
      rowFromNormalizedOffset(0.65),
      rowFromNormalizedOffset(1.15)
    );
    const lowerBandEndRow = Math.max(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const virtualThresholdRowStep = 2;
    const virtualThresholdDepthStep = 2;
    const thresholdDepthMargin = Math.max(4, Math.floor(virtualPanoDepthSamples * 0.25));
    const thresholdDepthStart = thresholdDepthMargin;
    const thresholdDepthEnd = virtualPanoDepthSamples - 1 - thresholdDepthMargin;

    for (let col = 0; col < panoWidth; col++) {
      const frame = frames[col];
      const [px, py, pz] = frame.position;
      const [slabDirX, slabDirY, slabDirZ] = slabDirs[col];
      const [vertDirX, vertDirY, vertDirZ] = verticalDirs[col] || effectiveVerticalDir;
      const columnVerticalCenterOffsetMm = Number(localCenterOffsetsMm[col]);

      for (let row = 0; row < panoHeight; row++) {
        const vertOffsetMm =
          columnVerticalCenterOffsetMm + (effectiveVerticalHalfMm - row * vertStepMm);
        const pixelIndex = planeIndex(col, row, panoWidth);
        const bx = px + vertOffsetMm * vertDirX;
        const by = py + vertOffsetMm * vertDirY;
        const bz = pz + vertOffsetMm * vertDirZ;

        for (let depth = 0; depth < virtualPanoDepthSamples; depth++) {
          const depthOffsetMm = Number(virtualPanoDepthOffsetsMm[depth]);
          const sample = sampleWorldIntensityForVirtualPano(
            bx + depthOffsetMm * slabDirX,
            by + depthOffsetMm * slabDirY,
            bz + depthOffsetMm * slabDirZ
          );
          virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)] = sample;
        }
      }
    }

    applyLightGaussianBlur3D(virtualPanoStack, panoWidth, panoHeight, virtualPanoDepthSamples);

    for (let col = 0; col < panoWidth; col++) {
      const toothThresholdSamples: number[] = [];
      const gradientSamples: number[] = [];

      for (
        let depth = thresholdDepthStart;
        depth <= thresholdDepthEnd;
        depth += virtualThresholdDepthStep
      ) {
        for (let row = toothBandStartRow; row <= toothBandEndRow; row += virtualThresholdRowStep) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const value = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (Number.isFinite(value)) {
            toothThresholdSamples.push(value);
          }
        }
      }

      for (
        let depth = Math.max(1, thresholdDepthStart);
        depth < Math.min(virtualPanoDepthSamples - 1, thresholdDepthEnd + 1);
        depth += virtualThresholdDepthStep
      ) {
        for (
          let row = toothBandStartRow + 1;
          row < toothBandEndRow;
          row += virtualThresholdRowStep
        ) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const centerValue = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (!Number.isFinite(centerValue)) {
            continue;
          }
          const plusDepth = Number(
            virtualPanoStack[stackIndex(depth + 1, pixelIndex, virtualPanoPlaneSize)]
          );
          const minusDepth = Number(
            virtualPanoStack[stackIndex(depth - 1, pixelIndex, virtualPanoPlaneSize)]
          );
          const plusRow = Number(
            virtualPanoStack[
              stackIndex(depth, planeIndex(col, row + 1, panoWidth), virtualPanoPlaneSize)
            ]
          );
          const minusRow = Number(
            virtualPanoStack[
              stackIndex(depth, planeIndex(col, row - 1, panoWidth), virtualPanoPlaneSize)
            ]
          );
          if (
            Number.isFinite(plusDepth) &&
            Number.isFinite(minusDepth) &&
            Number.isFinite(plusRow) &&
            Number.isFinite(minusRow)
          ) {
            gradientSamples.push(
              Math.abs(plusDepth - minusDepth) + 0.5 * Math.abs(plusRow - minusRow)
            );
          }
        }
      }

      virtualSoftThresholdByCol[col] =
        toothThresholdSamples.length > 0 ? percentile(toothThresholdSamples, 0.3) : -250;
      virtualHardThresholdByCol[col] =
        toothThresholdSamples.length > 0 ? percentile(toothThresholdSamples, 0.6) : 250;
      virtualGradCapByCol[col] =
        gradientSamples.length > 0 ? Math.max(1, percentile(gradientSamples, 0.9)) : 200;
    }

    smoothFloatSeries(virtualSoftThresholdByCol, panoWidth, virtualThresholdScratch, 2);
    smoothFloatSeries(virtualHardThresholdByCol, panoWidth, virtualThresholdScratch, 2);
    smoothFloatSeries(virtualGradCapByCol, panoWidth, virtualThresholdScratch, 1);

    for (let col = 0; col < panoWidth; col++) {
      const softThreshold = Number(virtualSoftThresholdByCol[col]);
      const hardThreshold = Number(virtualHardThresholdByCol[col]);
      const hardDen = Math.max(hardThreshold - softThreshold, 1e-6);
      const gradCap = Math.max(1, Number(virtualGradCapByCol[col]));

      for (let depth = 0; depth < virtualPanoDepthSamples; depth++) {
        let topHardAccum = 0;
        let topHardCount = 0;
        let bottomHardAccum = 0;
        let bottomHardCount = 0;
        let gradAccum = 0;
        let gradCount = 0;
        let lowAccum = 0;
        let lowCount = 0;

        for (let row = topBandStartRow; row <= topBandEndRow; row++) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const centerValue = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (!Number.isFinite(centerValue)) {
            continue;
          }

          const hardResponse = clampNumber((centerValue - softThreshold) / hardDen, 0, 1);
          topHardAccum += hardResponse;
          topHardCount++;
        }

        for (let row = bottomBandStartRow; row <= bottomBandEndRow; row++) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const centerValue = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (!Number.isFinite(centerValue)) {
            continue;
          }

          const hardResponse = clampNumber((centerValue - softThreshold) / hardDen, 0, 1);
          bottomHardAccum += hardResponse;
          bottomHardCount++;
        }

        for (let row = toothBandStartRow; row <= toothBandEndRow; row++) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const centerValue = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (!Number.isFinite(centerValue)) {
            continue;
          }

          if (depth > 0 && depth + 1 < virtualPanoDepthSamples && row > 0 && row + 1 < panoHeight) {
            const plusDepth = Number(
              virtualPanoStack[stackIndex(depth + 1, pixelIndex, virtualPanoPlaneSize)]
            );
            const minusDepth = Number(
              virtualPanoStack[stackIndex(depth - 1, pixelIndex, virtualPanoPlaneSize)]
            );
            const plusRow = Number(
              virtualPanoStack[
                stackIndex(depth, planeIndex(col, row + 1, panoWidth), virtualPanoPlaneSize)
              ]
            );
            const minusRow = Number(
              virtualPanoStack[
                stackIndex(depth, planeIndex(col, row - 1, panoWidth), virtualPanoPlaneSize)
              ]
            );
            if (
              Number.isFinite(plusDepth) &&
              Number.isFinite(minusDepth) &&
              Number.isFinite(plusRow) &&
              Number.isFinite(minusRow)
            ) {
              const gradientValue =
                Math.abs(plusDepth - minusDepth) + 0.5 * Math.abs(plusRow - minusRow);
              gradAccum += clampNumber(gradientValue / gradCap, 0, 1);
              gradCount++;
            }
          }
        }

        for (let row = lowerBandStartRow; row <= lowerBandEndRow; row++) {
          const pixelIndex = planeIndex(col, row, panoWidth);
          const value = Number(
            virtualPanoStack[stackIndex(depth, pixelIndex, virtualPanoPlaneSize)]
          );
          if (!Number.isFinite(value)) {
            continue;
          }
          lowAccum += clampNumber((value - softThreshold) / hardDen, 0, 1);
          lowCount++;
        }

        const topHardMean = topHardCount > 0 ? topHardAccum / topHardCount : 0;
        const bottomHardMean = bottomHardCount > 0 ? bottomHardAccum / bottomHardCount : 0;
        const supportMean = Math.min(topHardMean, bottomHardMean);
        const gradMean = gradCount > 0 ? gradAccum / gradCount : 0;
        const lowMean = lowCount > 0 ? lowAccum / lowCount : 0;
        const depthMm = Number(virtualPanoDepthOffsetsMm[depth]);
        const edgeDistanceMm = virtualPanoDepthHalfRangeMm - Math.abs(depthMm);
        const edgePenalty = edgeDistanceMm <= 2.5 ? 0.45 * (1 - edgeDistanceMm / 2.5) : 0;
        const depthCenterPenalty =
          0.12 * Math.pow(Math.abs(depthMm) / Math.max(virtualPanoDepthHalfRangeMm, 1e-6), 1.5);
        const crownBiasPenalty = Math.max(0, topHardMean - bottomHardMean) * 0.18;
        const score =
          0.62 * supportMean +
          0.24 * gradMean -
          0.55 * lowMean -
          edgePenalty -
          depthCenterPenalty -
          crownBiasPenalty;
        virtualScoreByColDepth[col * virtualPanoDepthSamples + depth] = score;
      }
    }

    // Collapse the panoramic support to a single smooth depth path. Keep tilt
    // disabled during debugging so the projection stays physically constrained.
    const virtualSelectedDepthMm = runBandDPOptimization(
      virtualScoreByColDepth,
      panoWidth,
      virtualPanoDepthSamples,
      virtualPanoDepthOffsetsMm,
      virtualPanoCandidateCount,
      virtualDpSmoothScratch,
      2
    );
    const bandLabels = ['support'] as const;
    const bandAnchorRows = [Math.round((toothBandStartRow + toothBandEndRow) * 0.5)];
    const bandDepthsMm: Float32Array[] = [virtualSelectedDepthMm];
    let nonCrossingViolations = 0;

    // Compatibility no-op: the legacy band constraints are inert with one path.
    for (let col = 0; col < panoWidth; col++) {
      for (let b = 1; b < bandDepthsMm.length; b++) {
        const prevBandDepth = Number(bandDepthsMm[b - 1][col]);
        let currentDepth = Number(bandDepthsMm[b][col]);

        // Non-crossing: band[b] should not be more than 0.5mm "past" band[b-1]
        if (currentDepth < prevBandDepth - 0.5) {
          nonCrossingViolations++;
          currentDepth = prevBandDepth - 0.5;
        }

        // Maximum inter-band delta: 4mm
        const delta = currentDepth - prevBandDepth;
        if (Math.abs(delta) > 4.0) {
          currentDepth = prevBandDepth + clampNumber(delta, -4.0, 4.0);
        }

        // Mandibular base clamp: band 3 within root band ± 3mm
        if (b === 3) {
          const rootDepth = Number(bandDepthsMm[2][col]);
          currentDepth = clampNumber(currentDepth, rootDepth - 3.0, rootDepth + 3.0);
        }

        bandDepthsMm[b][col] = currentDepth;
      }
    }

    // Re-smooth the selected support path.
    for (let b = 0; b < bandDepthsMm.length; b++) {
      smoothFloatSeries(bandDepthsMm[b], panoWidth, virtualDpSmoothScratch, 2);
    }

    // Single-path support diagnostics
    let virtualDepthMinMm = Infinity;
    let virtualDepthMaxMm = -Infinity;
    let virtualDepthSumMm = 0;
    const virtualPathJumps: number[] = [];
    for (let col = 0; col < panoWidth; col++) {
      const depthMm = Number(virtualSelectedDepthMm[col]);
      virtualDepthMinMm = Math.min(virtualDepthMinMm, depthMm);
      virtualDepthMaxMm = Math.max(virtualDepthMaxMm, depthMm);
      virtualDepthSumMm += depthMm;
      if (col > 0) {
        virtualPathJumps.push(Math.abs(depthMm - Number(virtualSelectedDepthMm[col - 1])));
      }
    }
    const virtualDepthMeanMm = panoWidth > 0 ? virtualDepthSumMm / panoWidth : 0;
    let virtualDepthVarianceMm = 0;
    for (let col = 0; col < panoWidth; col++) {
      const depthDelta = Number(virtualSelectedDepthMm[col]) - virtualDepthMeanMm;
      virtualDepthVarianceMm += depthDelta * depthDelta;
    }
    const virtualDepthStdMm =
      panoWidth > 0 ? Math.sqrt(virtualDepthVarianceMm / Math.max(1, panoWidth)) : 0;
    const virtualPathJumpP95Mm =
      virtualPathJumps.length > 0 ? percentile(virtualPathJumps, 0.95) : 0;
    let virtualSupportDepthClampCount = 0;
    let virtualSupportEdgeShelfCount = 0;
    for (let col = 0; col < panoWidth; col++) {
      const absDepthMm = Math.abs(Number(virtualSelectedDepthMm[col]));
      if (absDepthMm > virtualPanoDepthHalfRangeMm - 0.5) {
        virtualSupportDepthClampCount++;
      }
      if (absDepthMm >= virtualPanoDepthHalfRangeMm - 1.5) {
        virtualSupportEdgeShelfCount++;
      }
    }

    // Support-path diagnostics
    const bandDiagnostics = bandLabels.map((label, b) => {
      const depths = bandDepthsMm[b];
      let bMin = Infinity,
        bMax = -Infinity;
      for (let col = 0; col < panoWidth; col++) {
        const d = Number(depths[col]);
        bMin = Math.min(bMin, d);
        bMax = Math.max(bMax, d);
      }
      return {
        label,
        anchorRow: bandAnchorRows[b],
        depthMinMm: Number.isFinite(bMin) ? Math.round(bMin * 1000) / 1000 : 0,
        depthMaxMm: Number.isFinite(bMax) ? Math.round(bMax * 1000) / 1000 : 0,
        first8Mm: Array.from(depths.subarray(0, Math.min(8, depths.length))).map(
          v => Math.round(Number(v) * 1000) / 1000
        ),
      };
    });

    // Legacy inter-band diagnostics collapse to zero with a single path.
    let interBandDeltaSum = 0;
    let interBandDeltaMax = 0;
    let interBandDeltaCount = 0;
    for (let col = 0; col < panoWidth; col++) {
      for (let b = 1; b < bandDepthsMm.length; b++) {
        const delta = Math.abs(Number(bandDepthsMm[b][col]) - Number(bandDepthsMm[b - 1][col]));
        interBandDeltaSum += delta;
        interBandDeltaMax = Math.max(interBandDeltaMax, delta);
        interBandDeltaCount++;
      }
    }
    const interBandDeltaMeanMm =
      interBandDeltaCount > 0 ? interBandDeltaSum / interBandDeltaCount : 0;

    virtualPanoPhase12Diagnostics = {
      enabled: true,
      phase: 2,
      reconstructionMode,
      depthHalfRangeMm: virtualPanoDepthHalfRangeMm,
      depthStepMm: virtualPanoDepthStepMm,
      depthSamples: virtualPanoDepthSamples,
      candidateCount: virtualPanoCandidateCount,
      rowBands: {
        tooth: [toothBandStartRow, toothBandEndRow],
        top: [topBandStartRow, topBandEndRow],
        bottom: [bottomBandStartRow, bottomBandEndRow],
        low: [lowerBandStartRow, lowerBandEndRow],
      },
      thresholds: {
        softMedian: percentile(Array.from(virtualSoftThresholdByCol), 0.5),
        hardMedian: percentile(Array.from(virtualHardThresholdByCol), 0.5),
        gradCapMedian: percentile(Array.from(virtualGradCapByCol), 0.5),
      },
      supportSurface: {
        depthMinMm: Number.isFinite(virtualDepthMinMm) ? virtualDepthMinMm : 0,
        depthMaxMm: Number.isFinite(virtualDepthMaxMm) ? virtualDepthMaxMm : 0,
        depthStdMm: virtualDepthStdMm,
        pathJumpP95Mm: virtualPathJumpP95Mm,
        supportDepthClampFraction: panoWidth > 0 ? virtualSupportDepthClampCount / panoWidth : 0,
        supportEdgeShelfFraction: panoWidth > 0 ? virtualSupportEdgeShelfCount / panoWidth : 0,
        selectedDepthFirst8Mm: Array.from(
          virtualSelectedDepthMm.subarray(0, Math.min(8, virtualSelectedDepthMm.length))
        ).map(value => Math.round(Number(value) * 1000) / 1000),
      },
      supportModel: {
        bandCount: bandDepthsMm.length,
        bands: bandDiagnostics,
        interBandDeltaMeanMm: Math.round(interBandDeltaMeanMm * 1000) / 1000,
        interBandDeltaMaxMm: Math.round(interBandDeltaMax * 1000) / 1000,
        nonCrossingViolations,
      },
    };

    if (enableVirtualPanoRender) {
      const virtualRender = renderVirtualPanoFromSupportPath({
        virtualPanoStack,
        panoWidth,
        panoHeight,
        planeSize: virtualPanoPlaneSize,
        panoCenterRow,
        virtualPanoDepthOffsetsMm,
        selectedDepthMm: virtualSelectedDepthMm,
        softThresholdByCol: virtualSoftThresholdByCol,
        hardThresholdByCol: virtualHardThresholdByCol,
        supportTiltMmByCol: virtualSupportTiltByCol,
        virtualPanoDepthHalfRangeMm,
      });

      virtualPanoRenderDiagnostics = {
        enabled: true,
        usedAsOutput: virtualRender.diagnostics.usedAsOutput,
        supportDepthClampFraction: virtualRender.summary.supportDepthClampFraction,
        lowerBandBrightFraction: virtualRender.summary.lowerBandBrightFraction,
        lowerBandMean: virtualRender.summary.lowerBandMean,
        toothBandMean: virtualRender.summary.toothBandMean,
        toothBandContrastRange:
          virtualRender.summary.toothBandP90 - virtualRender.summary.toothBandP10,
        range: virtualRender.summary.range,
        minValue: virtualRender.summary.minValue,
        maxValue: virtualRender.summary.maxValue,
        ...virtualRender.diagnostics,
      };

      const keepRejectedVirtualPanoOutput =
        !virtualRender.diagnostics.usedAsOutput && !allowLegacyFallback;

      if (virtualRender.diagnostics.usedAsOutput || keepRejectedVirtualPanoOutput) {
        virtualPanoAcceptedByGate = virtualRender.diagnostics.usedAsOutput;
        virtualPanoSelectedForOutput = true;
        selectedReconstructionMode = 'virtualPano';
        pixelData.set(virtualRender.pixelData);
        const virtualRange = computeArrayMinMax(pixelData);
        minValue = virtualRange.minValue;
        maxValue = virtualRange.maxValue;
      }
    }
  } else {
    virtualPanoPhase12Diagnostics = {
      enabled: false,
      phase: 1,
      reconstructionMode,
      skippedReason: 'RECONSTRUCTION_MODE_DISABLED',
    };
  }
  const _t3_afterVirtualPano = performance.now();

  const shouldApplyAdaptiveBackgroundSuppression = true;
  const backgroundSuppressionResult = suppressLowerBackground(
    pixelData,
    panoWidth,
    panoHeight,
    vertStepMm
  );
  const { toothBandBottomRowStats, backgroundSuppression, suppressionWeights } =
    backgroundSuppressionResult;

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
    denoiseBlend > 0 &&
    applyLightBilateralDenoise(pixelData, panoWidth, panoHeight, denoiseBlend, suppressionWeights);
  if (denoiseApplied) {
    const denoisedRange = computeArrayMinMax(pixelData);
    minValue = denoisedRange.minValue;
    maxValue = denoisedRange.maxValue;
  }
  const effectiveDenoiseBlend = denoiseApplied ? denoiseBlend : 0;
  const _t4_afterSuppressDenoise = performance.now();

  const finalRange = computeArrayMinMax(pixelData);
  minValue = finalRange.minValue;
  maxValue = finalRange.maxValue;
  const windowWidth = maxValue - minValue;
  const windowCenter = minValue + windowWidth / 2;
  const virtualPanoRejectReasons = Array.isArray(
    (virtualPanoRenderDiagnostics as { rejectReasons?: unknown }).rejectReasons
  )
    ? ((virtualPanoRenderDiagnostics as { rejectReasons: unknown[] }).rejectReasons.filter(
        reason => typeof reason === 'string'
      ) as string[])
    : [];
  const acceptedByLowerBandTolerance =
    (virtualPanoRenderDiagnostics as { acceptedByLowerBandTolerance?: unknown })
      .acceptedByLowerBandTolerance === true;
  const virtualPanoAttempted = enableVirtualPanoRender;
  const virtualPanoRejected = virtualPanoAttempted && !virtualPanoAcceptedByGate;
  const gpuFallbackCause = options?.gpuFallbackCause ?? null;
  const fallbackReason =
    gpuFallbackCause === 'gpu-phase2-gate-failed'
      ? virtualPanoAcceptedByGate
        ? 'gpu-phase2-gate-failed-cpu-virtual-pano-fallback'
        : enableVirtualPanoRender
          ? allowLegacyFallback
            ? 'gpu-phase2-gate-failed-cpu-virtual-pano-rejected'
            : 'gpu-phase2-gate-failed-cpu-virtual-pano-rejected-no-legacy-fallback'
          : allowLegacyFallback
            ? 'gpu-phase2-gate-failed-cpu-legacy-fallback'
            : 'gpu-phase2-gate-failed-no-legacy-fallback'
      : gpuFallbackCause === 'gpu-render-failed'
        ? virtualPanoAcceptedByGate
          ? 'gpu-render-failed-cpu-virtual-pano-fallback'
          : enableVirtualPanoRender
            ? allowLegacyFallback
              ? 'gpu-render-failed-cpu-virtual-pano-rejected'
              : 'gpu-render-failed-cpu-virtual-pano-rejected-no-legacy-fallback'
            : allowLegacyFallback
              ? 'gpu-render-failed-cpu-legacy-fallback'
              : 'gpu-render-failed-no-legacy-fallback'
        : requestedReconstructionMode === 'virtualPano' && virtualPanoRejected
          ? allowLegacyFallback
            ? 'cpu-virtual-pano-rejected-legacy-fallback'
            : 'cpu-virtual-pano-rejected-no-legacy-fallback'
          : null;
  if (!virtualPanoSelectedForOutput && allowLegacyFallback) {
    selectedReconstructionMode = 'legacy';
  }
  const outputSelection = {
    requestedReconstructionMode,
    candidateReconstructionMode,
    selectedReconstructionMode,
    virtualPanoAttempted,
    virtualPanoAccepted: virtualPanoAcceptedByGate,
    virtualPanoRejected,
    virtualPanoSelectedForOutput,
    acceptedByLowerBandTolerance,
    legacyFallbackAllowed: allowLegacyFallback,
    rejectReasons: virtualPanoRejectReasons,
    fallbackReason,
  };

  let meanTurnAngleDeg = 0;
  let maxTurnAngleDeg = 0;
  let meanCurvatureFactor = 0;
  let maxCurvatureFactor = 0;
  let minSearchHalfRangeMm = Infinity;
  let maxSearchHalfRangeMm = -Infinity;
  let minMaxDeviationMm = Infinity;
  let maxMaxDeviationMm = -Infinity;
  for (let col = 0; col < panoWidth; col++) {
    const turnAngleDeg = Number(turnAngleDegreesByCol[col]);
    const curvatureFactor = Number(curvatureFactorByCol[col]);
    const searchHalfRangeMm = Number(adaptiveCenterSearchHalfRangeByCol[col]);
    const maxDeviationMm = Number(adaptiveCenterMaxDeviationByCol[col]);
    meanTurnAngleDeg += turnAngleDeg;
    maxTurnAngleDeg = Math.max(maxTurnAngleDeg, turnAngleDeg);
    meanCurvatureFactor += curvatureFactor;
    maxCurvatureFactor = Math.max(maxCurvatureFactor, curvatureFactor);
    minSearchHalfRangeMm = Math.min(minSearchHalfRangeMm, searchHalfRangeMm);
    maxSearchHalfRangeMm = Math.max(maxSearchHalfRangeMm, searchHalfRangeMm);
    minMaxDeviationMm = Math.min(minMaxDeviationMm, maxDeviationMm);
    maxMaxDeviationMm = Math.max(maxMaxDeviationMm, maxDeviationMm);
  }
  meanTurnAngleDeg = panoWidth > 0 ? meanTurnAngleDeg / panoWidth : 0;
  meanCurvatureFactor = panoWidth > 0 ? meanCurvatureFactor / panoWidth : 0;
  const _t5_beforePayload = performance.now();

  const requestedRenderBackend = input.renderBackend === 'cpu' ? 'cpu' : 'gpu';
  const diagnosticPayload = {
    renderBackend: 'cpu',
    requestedRenderBackend,
    pipelineMode: selectedReconstructionMode,
    fallbackReason,
    reconstructionMode: selectedReconstructionMode,
    requestedReconstructionMode,
    candidateReconstructionMode,
    phase2GatePassed: gpuFallbackCause === 'gpu-phase2-gate-failed' ? false : null,
    outputSelection,
    legacyFallbackAllowed: allowLegacyFallback,
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
    verticalAxisMode: 'frameS',
    slabDir: slabDirs?.[0]
      ? {
          N_slab: slabDirs[0],
          tangent: frames[0].T ?? null,
        }
      : null,
    verticalWindowMode: rigidVerticalSliceMode
      ? 'rigid-spline-slice-window'
      : 'local-column-adaptive-window',
    verticalHalfMm: effectiveVerticalHalfMm,
    globalVerticalCenterOffsetMm: clampedFittedVerticalCenterOffsetMm,
    baseVerticalCenterOffsetMm: verticalCenterOffsetMm,
    verticalCenterOffsetMm: meanLocalCenterOffsetMm,
    rigidSliceMode: {
      enabled: rigidVerticalSliceMode,
      projectedSplinePlaneLocked: rigidVerticalSliceMode,
      requestedCenterOffsetMm: clampedRequestedCenterOffsetMm,
      fittedCenterOffsetAppliedMm: clampedFittedVerticalCenterOffsetMm,
      localOffsetsLocked: rigidVerticalSliceMode,
    },
    localCenterOffsetMmStats: {
      min: minLocalCenterOffsetMm,
      max: maxLocalCenterOffsetMm,
      mean: meanLocalCenterOffsetMm,
      maxDeviationFromGlobal: Math.max(
        Math.abs(minLocalCenterOffsetMm - verticalCenterOffsetMm),
        Math.abs(maxLocalCenterOffsetMm - verticalCenterOffsetMm)
      ),
      maxAdjacentDeltaMm: maxLocalCenterAdjacentDeltaMm,
      first8: Array.from(
        localCenterOffsetsMm.subarray(0, Math.min(8, localCenterOffsetsMm.length))
      ).map(value => Math.round(Number(value) * 1000) / 1000),
    },
    localCenterOffsetsMm: Array.from(localCenterOffsetsMm).map(
      value => Math.round(Number(value) * 1000) / 1000
    ),
    requestedVerticalCenterOffsetMm: clampedRequestedCenterOffsetMm,
    fittedVerticalCenterOffsetMm: clampedFittedVerticalCenterOffsetMm,
    adaptiveVerticalSearch: {
      enabled: !rigidVerticalSliceMode,
      mode: rigidVerticalSliceMode ? 'disabled-rigid-slice' : 'local-adaptive-search',
      halfRangeMm: adaptiveCenterSearchHalfRangeMm,
      candidateCount: adaptiveCandidateCount,
      profileSamples: adaptiveProfileSampleCount,
      smoothingRadiusCols: adaptiveCenterSmoothingRadiusCols,
      globalBlend: adaptiveCenterGlobalBlend,
      maxDeviationMm: baseAdaptiveCenterMaxDeviationMm,
      maxAdjacentDeltaMm: adaptiveCenterMaxAdjacentDeltaMm,
    },
    curvatureAwarePriors: {
      turnAngleMeanDeg: meanTurnAngleDeg,
      turnAngleMaxDeg: maxTurnAngleDeg,
      curvatureFactorMean: meanCurvatureFactor,
      curvatureFactorMax: maxCurvatureFactor,
      halfRangeMmMin: Number.isFinite(minSearchHalfRangeMm)
        ? minSearchHalfRangeMm
        : adaptiveCenterSearchHalfRangeMm,
      halfRangeMmMax: Number.isFinite(maxSearchHalfRangeMm)
        ? maxSearchHalfRangeMm
        : adaptiveCenterSearchHalfRangeMm,
      maxDeviationMmMin: Number.isFinite(minMaxDeviationMm)
        ? minMaxDeviationMm
        : baseAdaptiveCenterMaxDeviationMm,
      maxDeviationMmMax: Number.isFinite(maxMaxDeviationMm)
        ? maxMaxDeviationMm
        : baseAdaptiveCenterMaxDeviationMm,
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
      blend: effectiveDenoiseBlend,
      requestedBlend: denoiseBlend,
      applied: denoiseApplied,
      virtualRenderUsedAsOutput: virtualPanoSelectedForOutput,
    },
    timingMs: {
      adaptiveCenterSearch: Math.round(_t1_afterAdaptiveCenter - _t0_start),
      pass1And2TwoPassRender: Math.round(_t2_afterTwoPassRender - _t1_afterAdaptiveCenter),
      virtualPanoPhase12: Math.round(_t3_afterVirtualPano - _t2_afterTwoPassRender),
      suppressionAndDenoise: Math.round(_t4_afterSuppressDenoise - _t3_afterVirtualPano),
      diagnosticAssembly: Math.round(_t5_beforePayload - _t4_afterSuppressDenoise),
      total: Math.round(_t5_beforePayload - _t0_start),
    },
    outputDisplayWindow: {
      lower: minValue,
      upper: maxValue,
      windowWidth,
      windowCenter,
    },
    virtualPanoPhase12: virtualPanoPhase12Diagnostics,
    virtualPanoRender: virtualPanoRenderDiagnostics,
    backgroundSuppressionMode: shouldApplyAdaptiveBackgroundSuppression
      ? 'adaptive-lower-band'
      : 'disabled-raw-pass-through',
    toothBandBottomRowStats,
    backgroundSuppression,
    twoPassEligibilityDiagnostics,
    reductionDiagnostics,
    firstFrameWorldPos: frames?.[0]?.position ?? null,
    lastFrameWorldPos: frames?.[frames.length - 1]?.position ?? null,
  };
  const cpuLogContext = {
    runId: input.debugRunId ?? null,
    attemptLabel: input.attemptLabel ?? null,
    displayedPath: 'worker-recon',
    backend: 'cpu',
    requestedBackend: requestedRenderBackend,
    requestedReconstructionMode,
    reconstructionMode: selectedReconstructionMode,
    candidateReconstructionMode,
    sourceVolumeId: input.sourceVolumeId ?? null,
    pipelineMode: selectedReconstructionMode,
    fallbackReason,
    legacyFallbackAllowed: allowLegacyFallback,
  };
  console.log(
    '[CPR-SUPPORT-SURFACE-JSON]',
    JSON.stringify(
      virtualPanoPhase12Diagnostics.enabled === true
        ? {
            ...cpuLogContext,
            ...virtualPanoPhase12Diagnostics,
          }
        : {
            ...cpuLogContext,
            skippedReason:
              (virtualPanoPhase12Diagnostics as { skippedReason?: unknown }).skippedReason ??
              'virtual-pano-disabled',
          }
    )
  );
  console.log(
    '[CPR-DRR-JSON]',
    JSON.stringify(
      virtualPanoRenderDiagnostics.enabled === true
        ? {
            ...cpuLogContext,
            ...virtualPanoRenderDiagnostics,
          }
        : {
            ...cpuLogContext,
            skippedReason:
              (virtualPanoRenderDiagnostics as { skippedReason?: unknown }).skippedReason ??
              'virtual-pano-render-disabled',
          }
    )
  );
  console.log(
    '[CPR-TONE-MAP-JSON]',
    JSON.stringify({
      ...cpuLogContext,
      enabled: virtualPanoRenderDiagnostics.enabled === true,
      usedAsOutput: virtualPanoSelectedForOutput,
      acceptedByGate: virtualPanoAcceptedByGate,
      outputDisplayWindow: {
        lower: minValue,
        upper: maxValue,
        windowWidth,
        windowCenter,
      },
      outputSelection,
      rejectReasons: virtualPanoRejectReasons,
    })
  );
  console.log(
    '[CPR-SIDECAR-MAPS-JSON]',
    JSON.stringify({
      ...cpuLogContext,
      debugEnabled: !!input.debugRunId,
      attachedMaps: [],
      attachedByteLength: 0,
    })
  );
  if (gpuFallbackCause) {
    console.log(
      '[CPR-CPU-FALLBACK-JSON]',
      JSON.stringify({
        ...cpuLogContext,
        gpuFallbackCause,
        outputSelection,
      })
    );
  }
  const outputPixelPreview = Array.from(pixelData.subarray(0, Math.min(5, pixelData.length)));
  const outputSignature = computeOutputSignature(pixelData);
  return {
    pixelData,
    minValue,
    maxValue,
    windowWidth,
    windowCenter,
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

let cachedVolumeState: CPRWorkerVolumeState | null = null;

function buildRenderInput(
  cached: CPRWorkerVolumeState,
  render: CPRWorkerRenderInput
): CPRWorkerInput {
  return {
    scalarData: cached.scalarData,
    isSharedArrayBuffer: cached.isSharedArrayBuffer,
    dimensions: cached.dimensions,
    spacing: cached.spacing,
    origin: cached.origin,
    direction: cached.direction,
    worldToIndex: cached.worldToIndex,
    rescaleSlope: cached.rescaleSlope,
    rescaleIntercept: cached.rescaleIntercept,
    bitsStored: cached.bitsStored,
    bitsAllocated: cached.bitsAllocated,
    highBit: cached.highBit,
    pixelRepresentation: cached.pixelRepresentation,
    isPreScaled: cached.isPreScaled,
    verticalDir: render.verticalDir,
    frames: render.frames,
    panoWidth: render.panoWidth,
    panoHeight: render.panoHeight,
    vertHalfMm: render.vertHalfMm,
    verticalCenterOffsetMm: render.verticalCenterOffsetMm,
    rigidVerticalSliceMode: render.rigidVerticalSliceMode,
    slabHalfThicknessMm: render.slabHalfThicknessMm,
    slabSamples: render.slabSamples,
    aggregation: render.aggregation,
    applyModalityLut: render.applyModalityLut,
    allowStoredValueNormalization: render.allowStoredValueNormalization,
    disableStoredValueNormalization: render.disableStoredValueNormalization,
    debugRunId: render.debugRunId,
    attemptLabel: render.attemptLabel,
    reconstructionMode: render.reconstructionMode,
    renderBackend: render.renderBackend,
    allowLegacyFallback: render.allowLegacyFallback,
    sourceVolumeId: render.sourceVolumeId,
  };
}

function resolveWorkerRequestId(input: unknown): string {
  return typeof (input as { requestId?: unknown })?.requestId === 'string'
    ? String((input as { requestId: string }).requestId)
    : 'unknown-request';
}

// eslint-disable-next-line no-restricted-globals
self.onmessage = function (event: MessageEvent<CPRWorkerMessage>) {
  try {
    const input = event.data;
    const requestId = resolveWorkerRequestId(input);
    if (input.type === 'DISPOSE') {
      disposeGpuPanoRenderer();
      cachedVolumeState = null;
      const disposeResponse: CPRWorkerDisposeSuccess = {
        type: 'DISPOSE_SUCCESS',
        requestId,
      };
      (self as unknown as Worker).postMessage(disposeResponse);
      return;
    }

    if (input.type === 'INIT_VOLUME') {
      if (!input.scalarData || input.scalarData.length === 0) {
        throw new Error('Received empty or null scalar data during INIT_VOLUME.');
      }
      const policy = resolveStoredValueNormalizationPolicy({
        scalarData: input.scalarData,
        isPreScaled: input.isPreScaled,
        rescaleSlope: input.rescaleSlope,
        rescaleIntercept: input.rescaleIntercept,
        bitsStored: input.bitsStored,
        bitsAllocated: input.bitsAllocated,
        highBit: input.highBit,
        pixelRepresentation: input.pixelRepresentation,
        allowStoredValueNormalization: input.allowStoredValueNormalization,
        disableStoredValueNormalization: input.disableStoredValueNormalization,
      });
      if (policy.heuristicOverride && input.isPreScaled !== true) {
        console.warn(
          '[CPR-worker] INIT_VOLUME detected already-scaled HU source data; bypassing secondary rescale.',
          {
            sampledMin: policy.sampledMin,
            sampledMax: policy.sampledMax,
            rescaleSlope: policy.safeSlope,
            rescaleIntercept: policy.safeIntercept,
            bitsStored: policy.safeBitsStored,
            pixelRepresentation: input.pixelRepresentation,
          }
        );
      }
      const normalized = normalizeScalarDataToHuFloat32({
        scalarData: input.scalarData,
        isPreScaled: input.isPreScaled,
        rescaleSlope: input.rescaleSlope,
        rescaleIntercept: input.rescaleIntercept,
        bitsStored: input.bitsStored,
        bitsAllocated: input.bitsAllocated,
        highBit: input.highBit,
        pixelRepresentation: input.pixelRepresentation,
        allowStoredValueNormalization: input.allowStoredValueNormalization,
        disableStoredValueNormalization: input.disableStoredValueNormalization,
      });
      const cleanedHuScalarData = normalized.pixelData;
      cachedVolumeState = {
        sessionKey: input.sessionKey,
        scalarData: cleanedHuScalarData,
        isSharedArrayBuffer: cleanedHuScalarData.buffer instanceof SharedArrayBuffer,
        dimensions: input.dimensions,
        spacing: input.spacing,
        origin: input.origin,
        direction: input.direction,
        worldToIndex: input.worldToIndex,
        rescaleSlope: 1,
        rescaleIntercept: 0,
        bitsStored: policy.safeBitsStored,
        bitsAllocated: input.bitsAllocated,
        highBit: input.highBit,
        pixelRepresentation: input.pixelRepresentation,
        isPreScaled: true,
        storedValueNormalizationApplied: policy.shouldNormalizeStoredValues,
        unsignedPackedArtifactDetected: policy.unsignedPackedArtifactDetected,
        normalizationSignature: policy.normalizationSignature,
        preScaledHeuristicOverride: policy.heuristicOverride,
      };
      const initResponse: CPRWorkerInitSuccess = {
        type: 'INIT_SUCCESS',
        requestId,
        sessionKey: input.sessionKey,
      };
      (self as unknown as Worker).postMessage(initResponse);
      return;
    }

    if (input.type !== 'RENDER') {
      throw new Error('Unsupported worker message type.');
    }
    if (!cachedVolumeState) {
      throw new Error('Received RENDER before INIT_VOLUME.');
    }
    if (cachedVolumeState.sessionKey !== input.sessionKey) {
      throw new Error('Worker session key mismatch for RENDER request.');
    }

    const renderInput = buildRenderInput(cachedVolumeState, input);
    if (!renderInput.scalarData || renderInput.scalarData.length === 0) {
      throw new Error('Received empty or null scalar data.');
    }
    if (!renderInput.frames || renderInput.frames.length === 0) {
      throw new Error('Received empty frames array.');
    }
    const renderBackend = renderInput.renderBackend === 'cpu' ? 'cpu' : 'gpu';
    type OptionalGpuDiagnosticMaps = {
      meanMap?: Float32Array;
      maxMap?: Float32Array;
      sampleCountMap?: Float32Array;
      debugMaps?: GpuPanoDebugMaps;
    };
    let renderResult:
      | ReturnType<typeof generateGpuPanorama>
      | (Pick<
          ReturnType<typeof generatePanorama>,
          | 'pixelData'
          | 'minValue'
          | 'maxValue'
          | 'windowWidth'
          | 'windowCenter'
          | 'modalityLutApplied'
          | 'requestedModalityLutApplied'
          | 'storedValueNormalizationApplied'
          | 'unsignedPackedArtifactDetected'
          | 'diagnosticPayload'
          | 'outputSignature'
        > &
          OptionalGpuDiagnosticMaps)
      | undefined;

    if (renderBackend === 'gpu') {
      let lastGpuError: unknown = null;
      let gpuFallbackCause: 'gpu-render-failed' | 'gpu-phase2-gate-failed' | null = null;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          renderResult = generateGpuPanorama(renderInput, input.sessionKey);
          lastGpuError = null;
          break;
        } catch (gpuError) {
          lastGpuError = gpuError;
          console.warn(
            attempt === 0
              ? '[CPR-GPU] GPU panorama render failed. Disposing context and retrying once.'
              : '[CPR-GPU] GPU panorama render failed again after context reset.',
            gpuError
          );
          try {
            disposeGpuPanoRenderer();
          } catch (disposeError) {
            console.warn(
              '[CPR-GPU] Failed to dispose GPU renderer after render failure.',
              disposeError
            );
          }
        }
      }

      const gpuRenderDiagnostics =
        renderResult && 'diagnostics' in renderResult ? renderResult.diagnostics : null;
      const gpuPipelineMode =
        renderResult && 'pipelineMode' in renderResult ? renderResult.pipelineMode : 'unknown';
      const gpuPhase2GateFailed = !!renderResult && gpuRenderDiagnostics?.phase2GatePassed === false;

      if (!renderResult) {
        gpuFallbackCause = 'gpu-render-failed';
        console.warn(
          '[CPR-GPU] Falling back to CPU virtual pano renderer after GPU retry failure.',
          lastGpuError
        );
      } else if (gpuPhase2GateFailed) {
        gpuFallbackCause = 'gpu-phase2-gate-failed';
        console.warn(
          '[CPR-GPU] GPU panorama missed the Phase 2 gate. Falling back to CPU virtual pano renderer.',
          {
            runId: renderInput.debugRunId ?? null,
            attemptLabel: renderInput.attemptLabel ?? null,
            pipelineMode: gpuPipelineMode,
          }
        );
      }

      if (gpuFallbackCause) {
        const requestedReconstructionMode = resolveReconstructionMode(
          renderInput.reconstructionMode
        );
        const cpuFallbackReconstructionMode = resolveCpuFallbackReconstructionMode(
          requestedReconstructionMode
        );
        console.log(
          '[CPR-CPU-FALLBACK-JSON]',
          JSON.stringify({
            runId: renderInput.debugRunId ?? null,
            attemptLabel: renderInput.attemptLabel ?? null,
            displayedPath: 'worker-recon',
            backend: 'cpu',
            requestedBackend: 'gpu',
            requestedReconstructionMode,
            cpuFallbackReconstructionMode,
            gpuFallbackCause,
            legacyFallbackAllowed: renderInput.allowLegacyFallback === true,
            sourceVolumeId: renderInput.sourceVolumeId ?? null,
          })
        );
        renderResult = generatePanorama(
          {
            ...renderInput,
            renderBackend: 'cpu',
            reconstructionMode: cpuFallbackReconstructionMode,
          },
          {
            requestedReconstructionMode,
            gpuFallbackCause,
          }
        );
      }
    } else {
      renderResult = generatePanorama(renderInput);
    }

    const {
      pixelData,
      meanMap,
      maxMap,
      sampleCountMap,
      debugMaps,
      minValue,
      maxValue,
      windowWidth,
      windowCenter,
      modalityLutApplied,
      requestedModalityLutApplied,
      storedValueNormalizationApplied,
      unsignedPackedArtifactDetected,
      diagnosticPayload,
      outputSignature,
    } = renderResult;
    const storedValueNormalizationAppliedForOutput =
      !!cachedVolumeState?.storedValueNormalizationApplied || storedValueNormalizationApplied;
    const unsignedPackedArtifactDetectedForOutput =
      !!cachedVolumeState?.unsignedPackedArtifactDetected || unsignedPackedArtifactDetected;
    const displayPixelData = {
      pixelData,
      slope: 1,
      intercept: 0,
    };
    console.log('[WORKER-FINAL-MINMAX]', minValue, maxValue);
    const response: CPRWorkerSuccess = {
      type: 'SUCCESS',
      requestId,
      pixelData: displayPixelData.pixelData,
      meanMap,
      maxMap,
      sampleCountMap,
      debugMaps,
      panoWidth: renderInput.panoWidth,
      panoHeight: renderInput.panoHeight,
      minValue,
      maxValue,
      windowWidth,
      windowCenter,
      slope: displayPixelData.slope,
      intercept: displayPixelData.intercept,
      modalityLutApplied,
      requestedModalityLutApplied,
      storedValueNormalizationApplied: storedValueNormalizationAppliedForOutput,
      unsignedPackedArtifactDetected: unsignedPackedArtifactDetectedForOutput,
      debugPayload: {
        diagnostic: diagnosticPayload,
        outputSignature,
      },
    };

    const transferList: Transferable[] = [];
    const transferBufferSet = new Set<ArrayBufferLike>();
    const pushTransferBuffer = (typedArray?: ArrayBufferView | null) => {
      if (!typedArray?.buffer || transferBufferSet.has(typedArray.buffer)) {
        return;
      }
      transferBufferSet.add(typedArray.buffer);
      transferList.push(typedArray.buffer as Transferable);
    };
    pushTransferBuffer(displayPixelData.pixelData);
    pushTransferBuffer(meanMap);
    pushTransferBuffer(maxMap);
    pushTransferBuffer(sampleCountMap);
    pushTransferBuffer(debugMaps?.supportDepthMap);
    pushTransferBuffer(debugMaps?.supportConfidenceMap);
    pushTransferBuffer(debugMaps?.supportSpreadMap);
    pushTransferBuffer(debugMaps?.supportDensityMap);
    pushTransferBuffer(debugMaps?.totalAttenuationMap);
    pushTransferBuffer(debugMaps?.lowerPenaltyMap);
    pushTransferBuffer(debugMaps?.participatingSampleCountMap);
    pushTransferBuffer(debugMaps?.toneResponseMap);
    pushTransferBuffer(debugMaps?.troughHalfWidthMap);

    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response, transferList);
  } catch (err) {
    const requestId = resolveWorkerRequestId(event.data);
    const response: CPRWorkerError = {
      type: 'ERROR',
      requestId,
      message: err instanceof Error ? err.message : String(err),
    };
    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response);
  }
};
