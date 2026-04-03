import type { GpuPanoDebugMaps, GpuPanoDiagnostics } from './cprGpuRenderer';
import {
  createHuScalarTransform,
  resolveStoredValueNormalizationPolicy,
} from './cprScalarPolicy';

type GpuRendererModule = typeof import('./cprGpuRenderer');

let gpuRendererModule: GpuRendererModule | null = null;
let gpuRendererModulePromise: Promise<GpuRendererModule> | null = null;

// Avoid blocking worker bootstrap on the large GPU renderer module.
async function loadGpuRendererModule(): Promise<GpuRendererModule> {
  if (gpuRendererModule) {
    return gpuRendererModule;
  }

  if (!gpuRendererModulePromise) {
    gpuRendererModulePromise = import('./cprGpuRenderer')
      .then(module => {
        gpuRendererModule = module;
        return module;
      })
      .catch(error => {
        gpuRendererModulePromise = null;
        throw error;
      });
  }

  return gpuRendererModulePromise;
}

async function disposeGpuPanoRendererIfLoaded(): Promise<void> {
  if (!gpuRendererModule) {
    return;
  }

  gpuRendererModule.disposeGpuPanoRenderer();
}

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
  debugScalarSamplingMode?:
    | 'current'
    | 'lut-only'
    | 'no-stored-value-normalization'
    | 'raw-stored-values-debug';
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
  debugScalarSamplingMode?:
    | 'current'
    | 'lut-only'
    | 'no-stored-value-normalization'
    | 'raw-stored-values-debug';
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

interface CPRWorkerBootstrapCheckInput {
  type: 'BOOTSTRAP_CHECK';
  requestId: string;
}

type CPRWorkerMessage =
  | CPRWorkerInitVolumeInput
  | CPRWorkerRenderInput
  | CPRWorkerDisposeInput
  | CPRWorkerBootstrapCheckInput;

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

function resolveRequestedRenderBackend(value: unknown): 'gpu' | 'cpu' {
  return value === 'cpu' ? 'cpu' : 'gpu';
}

function shouldRouteRequestedReconstructionToTrueVirtualPano(input: {
  renderBackend?: 'gpu' | 'cpu';
  reconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
}): boolean {
  return (
    resolveRequestedRenderBackend(input.renderBackend) === 'gpu' &&
    resolveReconstructionMode(input.reconstructionMode) !== 'legacy'
  );
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
  stage?: string;
}

interface CPRWorkerLifecycle {
  type: 'WORKER_LIFECYCLE';
  scope: 'implementation';
  stage: string;
  detail?: Record<string, unknown>;
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

interface CPRWorkerBootstrapReady {
  type: 'BOOTSTRAP_READY';
  requestId: string;
}

function logWorkerRouteDecision(params: {
  runId?: string | null;
  attemptLabel?: string | null;
  sourceVolumeId?: string | null;
  requestedBackend: 'gpu' | 'cpu';
  requestedReconstructionMode: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
  actualRendererBackend: 'gpu' | 'cpu';
  actualRendererFamily: string;
  routeReason: string;
  gpuFallbackCause?: 'gpu-render-failed' | 'gpu-phase2-gate-failed' | null;
}): void {
  const shouldEmitRouteLog =
    params.requestedReconstructionMode !== 'legacy' ||
    params.requestedBackend !== params.actualRendererBackend ||
    !!params.gpuFallbackCause;
  if (!shouldEmitRouteLog) {
    return;
  }

  console.log(
    '[CPR-WORKER-ROUTE-JSON]',
    JSON.stringify({
      runId: params.runId ?? null,
      attemptLabel: params.attemptLabel ?? null,
      sourceVolumeId: params.sourceVolumeId ?? null,
      requestedBackend: params.requestedBackend,
      requestedReconstructionMode: params.requestedReconstructionMode,
      actualRendererBackend: params.actualRendererBackend,
      actualRendererFamily: params.actualRendererFamily,
      routeReason: params.routeReason,
      gpuFallbackCause: params.gpuFallbackCause ?? null,
    })
  );
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
      const localBlend = clampedBlend * Math.sqrt(Math.max(0, backgroundWeight));
      if (localBlend <= 0.001) {
        continue;
      }
      const keepWeight = 1 - localBlend;
      const localSigmaRange = sigmaRange + backgroundWeight * 420;
      const localSigmaRangeDen = 2 * localSigmaRange * localSigmaRange;

      let weightedSum = center;
      let weightTotal = 1;

      for (let n = 0; n < neighbors.length; n++) {
        const [dx, dy, spatialWeight] = neighbors[n];
        const neighborValue = source[(row + dy) * width + (col + dx)];
        if (!Number.isFinite(neighborValue)) {
          continue;
        }
        const delta = neighborValue - center;
        const rangeWeight = Math.exp(-(delta * delta) / localSigmaRangeDen);
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

function clampSuppressedBackgroundSpeckles(
  pixelData: Float32Array,
  width: number,
  height: number,
  suppressionWeights?: Float32Array
): {
  applied: boolean;
  pixelCount: number;
  meanReductionHu: number;
  maxReductionHu: number;
} {
  if (
    width < 3 ||
    height < 3 ||
    !suppressionWeights ||
    suppressionWeights.length !== pixelData.length
  ) {
    return {
      applied: false,
      pixelCount: 0,
      meanReductionHu: 0,
      maxReductionHu: 0,
    };
  }

  const source = new Float32Array(pixelData);
  let pixelCount = 0;
  let reductionSum = 0;
  let maxReductionHu = 0;

  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      const index = row * width + col;
      const suppressionWeight = clampNumber(Number(suppressionWeights[index]) || 0, 0, 1);
      if (suppressionWeight < 0.14) {
        continue;
      }

      const center = source[index];
      if (!Number.isFinite(center) || center <= -220) {
        continue;
      }

      let neighborSum = 0;
      let neighborCount = 0;
      let darkNeighborCount = 0;
      let muchDarkerNeighborCount = 0;
      let suppressedNeighborCount = 0;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) {
            continue;
          }
          const neighborIndex = (row + dy) * width + (col + dx);
          const neighborValue = source[neighborIndex];
          if (!Number.isFinite(neighborValue)) {
            continue;
          }
          neighborSum += neighborValue;
          neighborCount++;
          if (neighborValue <= -260) {
            darkNeighborCount++;
          }
          if (neighborValue <= center - 150) {
            muchDarkerNeighborCount++;
          }
          if ((Number(suppressionWeights[neighborIndex]) || 0) >= 0.12) {
            suppressedNeighborCount++;
          }
        }
      }

      if (neighborCount < 5) {
        continue;
      }

      const neighborMean = neighborSum / neighborCount;
      const isolatedHotSpeckle =
        center > neighborMean + 170 &&
        (darkNeighborCount >= 4 ||
          (muchDarkerNeighborCount >= 5 && suppressedNeighborCount >= 4));
      if (!isolatedHotSpeckle) {
        continue;
      }

      const clampTarget = Math.max(-780, neighborMean + 45);
      if (clampTarget >= center) {
        continue;
      }

      const reductionHu = center - clampTarget;
      pixelData[index] = clampTarget;
      pixelCount++;
      reductionSum += reductionHu;
      maxReductionHu = Math.max(maxReductionHu, reductionHu);
    }
  }

  return {
    applied: pixelCount > 0,
    pixelCount,
    meanReductionHu: pixelCount > 0 ? reductionSum / pixelCount : 0,
    maxReductionHu,
  };
}

const GPU_RESIDUAL_DENOISE_BLEND = 0.38;

function buildGpuResidualDenoiseWeights(
  lowerPenaltyMap: Float32Array,
  toneResponseMap: Float32Array
): Float32Array {
  const length = Math.min(lowerPenaltyMap.length, toneResponseMap.length);
  const weights = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const lowerPenalty = clampNumber(Number(lowerPenaltyMap[i]) || 0, 0, 2.5);
    const toneResponse = clampNumber(Number(toneResponseMap[i]) || 0, 0, 1);
    const lowerPenaltyGate = clampNumber(lowerPenalty / 0.28, 0, 1);
    weights[i] = clampNumber(
      0.10 + lowerPenaltyGate * (1 - toneResponse * 0.72),
      0.06,
      1
    );
  }

  return weights;
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
      const windowWidth = Math.min(Math.max(1, upper - lower), 900);
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
  const windowWidth = Math.min(Math.max(1, safeMax - safeMin), 900);

  return {
    minValue: safeMin,
    maxValue: safeMax,
    windowWidth,
    windowCenter: safeMin + windowWidth / 2,
  };
}

function summarizeFiniteFloat32Buffer(buffer: Float32Array): {
  min: number;
  max: number;
  p10: number;
  p50: number;
  p90: number;
} | null {
  const samples: number[] = [];

  for (let i = 0; i < buffer.length; i++) {
    const value = Number(buffer[i]);
    if (Number.isFinite(value)) {
      samples.push(value);
    }
  }

  if (!samples.length) {
    return null;
  }

  return {
    min: percentile(samples, 0),
    max: percentile(samples, 1),
    p10: percentile(samples, 0.1),
    p50: percentile(samples, 0.5),
    p90: percentile(samples, 0.9),
  };
}

const VIRTUAL_PANO_DP_MODEL_PARAMS = {
  jumpCostLinear: 0.34,
  jumpCostExtra: 0.24,
  jumpSoftThresholdMm: 0.4,
};

const CPU_VIRTUAL_PANO_MODEL_PARAMS = {
  thresholdSoftPercentile: 0.4,
  thresholdHardPercentile: 0.74,
  gradCapPercentile: 0.85,
  supportMeanWeight: 0.68,
  gradientWeight: 0.3,
  balanceWeight: 0.16,
  lowBandPenaltyWeight: 0.9,
  edgePenaltyScale: 0.58,
  depthCenterPenaltyScale: 0.16,
  crownBiasPenaltyScale: 0.32,
  supportReliabilityWeight: 0.24,
  supportAmbiguityWeight: 0.16,
  supportLowReliabilityPenalty: 0.26,
  supportEdgeShelfWeight: 0.18,
  supportRepairLowReliabilityThreshold: 0.38,
  supportRepairJumpMm: 0.66,
  supportRepairBlend: 0.74,
  supportRepairEdgeShelfMarginMm: 0.35,
  supportRepairScoreGapThreshold: 0.11,
  supportRepairBestDepthDriftMm: 0.7,
  supportEnvelopeAnchorReliabilityThreshold: 0.56,
  supportEnvelopeAnchorScoreGapThreshold: 0.12,
  supportEnvelopeHalfWidthMinMm: 0.72,
  supportEnvelopeHalfWidthMaxMm: 1.58,
  supportEnvelopePenaltyScale: 0.42,
  supportEnvelopeRepairBlend: 0.78,
  pathSmoothPasses: 3,
  supportDualArchCollapseMinRetention: 0.34,
  supportSigmaLowReliabilityBoostMm: 0.24,
  supportSigmaMm: 0.82,
  supportEnergyWindowScale: 1.6,
  alphaScale: 0.68,
  emissionDenScale: 2.4,
  emissionDenMin: 260,
  contextSigmaScale: 1.95,
  contextThresholdOffset: 180,
  contextThresholdBlend: 0.42,
  contextWeightFloor: 0.08,
  contextBlendBase: 0.18,
  contextBlendLowReliabilityScale: 0.26,
  contextBlendOuterRowScale: 0.22,
  contextBlendToothPreserve: 0.08,
  contextHuFloor: -820,
  contextHuCeiling: 780,
  archSyntheticDetailHalfWidthMm: 0.72,
  archSyntheticDetailLowReliabilityBoostMm: 0.34,
  archSyntheticContextHalfWidthMm: 1.45,
  archSyntheticContextLowReliabilityBoostMm: 0.44,
  archSyntheticDetailWeightFloor: 0.12,
  archSyntheticDetailToothWeightScale: 0.92,
  archSyntheticContextWeightFloor: 0.12,
  archSyntheticContextToothWeightScale: 0.24,
  archSyntheticPeakBlend: 0.18,
  archSyntheticContextBlendBase: 0.14,
  archSyntheticContextBlendLowReliabilityScale: 0.12,
  archSyntheticContextBlendMiddleBandScale: 0.1,
  archSyntheticContextBlendOuterBandScale: 0.04,
  archSyntheticToothPreserveScale: 0.3,
  archSyntheticShadowLiftStrength: 0.14,
  archSyntheticShadowFloorOffsetHu: 240,
  archSyntheticSeparationMinMm: 0.9,
  archSyntheticSeparationIdealMm: 2.1,
  archSyntheticAmbiguousContextTightenMm: 0.42,
  archSyntheticAmbiguousContextSuppressionHu: 145,
  archSyntheticAmbiguousBlendReduction: 0.16,
  archSyntheticAmbiguousShadowLiftReduction: 0.08,
  archSyntheticAmbiguousDarkenStrength: 0.12,
  archSyntheticOutputHuFloor: -940,
  archSyntheticOutputHuCeiling: 1680,
  dualArchProjectionWindowHalfWidthMm: 4.4,
  dualArchProjectionWindowLowReliabilityBoostMm: 0.85,
  dualArchProjectionOuterWindowScale: 1.05,
  dualArchProjectionSupportTiltScale: 0.18,
  dualArchProjectionBackgroundHu: -1000,
  dualArchProjectionQuietBackgroundHu: -820,
  dualArchProjectionTopBackgroundOnlyRowEnd: 0.34,
  dualArchProjectionOutputHuFloor: -1000,
  dualArchProjectionOutputHuCeiling: 4000,
  dualArchProjectionGateFloorHu: 0,
  dualArchProjectionTopFraction: 0.3,
  dualArchProjectionDepthFollowHalfWidthMm: 1.75,
  dualArchProjectionMaxAdjacentDepthDeltaMm: 0.9,
  dualArchProjectionContinuityPenaltyPerMm: 0.16,
  dualArchProjectionBaseDepthPenaltyPerMm: 0.05,
  dualArchProjectionAmbiguityGapThreshold: 0.045,
  dualArchProjectionFocalHalfWidthMm: 2.25,
  dualArchProjectionOffBandRescueFloorHu: 980,
  dualArchProjectionLowerBandOffBandRescueBoostHu: 120,
  dualArchProjectionLowerPenaltyDarkenHu: 140,
  dualArchProjectionLowerPenaltyMaxBlend: 0.42,
  lowerPenaltyRowStart: 0.5,
  lowerPenaltyRowEnd: 0.86,
  lowerPenaltyBase: 0.16,
  lowerPenaltyLowConfidenceScale: 0.62,
  attenuationStrength: 1.82,
  gamma: 1.12,
  signalFloor: 0.04,
  signalCeiling: 0.975,
  outputHuMin: -930,
  outputHuMax: 1550,
};

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
  debugScalarSamplingMode:
    | 'current'
    | 'lut-only'
    | 'no-stored-value-normalization'
    | 'raw-stored-values-debug';
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
    debugScalarSamplingMode,
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
  let shouldApplyModalityLut = transform.shouldApplyModalityLut;
  let shouldNormalizeStoredValues = transform.shouldNormalizeStoredValues;
  let normalizeStoredSample = transform.normalizeStoredSample;
  let normalizationSignature = transform.normalizationSignature;
  const effectiveDebugScalarSamplingMode =
    debugScalarSamplingMode === 'lut-only' ||
    debugScalarSamplingMode === 'no-stored-value-normalization' ||
    debugScalarSamplingMode === 'raw-stored-values-debug'
      ? debugScalarSamplingMode
      : 'current';

  if (effectiveDebugScalarSamplingMode === 'lut-only') {
    shouldApplyModalityLut = true;
    shouldNormalizeStoredValues = false;
    normalizeStoredSample = undefined;
    normalizationSignature = 'debug:lut-only';
  } else if (effectiveDebugScalarSamplingMode === 'no-stored-value-normalization') {
    shouldNormalizeStoredValues = false;
    normalizeStoredSample = undefined;
    normalizationSignature = 'debug:no-stored-value-normalization';
  } else if (effectiveDebugScalarSamplingMode === 'raw-stored-values-debug') {
    shouldApplyModalityLut = false;
    shouldNormalizeStoredValues = false;
    normalizeStoredSample = undefined;
    normalizationSignature = 'debug:raw-stored-values';
  }

  return {
    safeSlope: transform.safeSlope,
    safeIntercept: transform.safeIntercept,
    debugScalarSamplingMode: effectiveDebugScalarSamplingMode,
    requestedModalityLutApplied,
    shouldApplyModalityLut,
    shouldNormalizeStoredValues,
    unsignedPackedArtifactDetected: transform.unsignedPackedArtifactDetected,
    normalizeStoredSample,
    normalizationSignature,
    safeInterpolationOobValue: transform.safeInterpolationOobValue,
  };
}

interface GeneratedGpuPanorama {
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
}

async function generateGpuPanorama(
  input: CPRWorkerInput,
  volumeCacheKey: string
): Promise<GeneratedGpuPanorama> {
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
  const effectiveGpuReconstructionMode = (reconstructionMode ?? 'legacy') as string;
  const shouldEmitVerboseGpuLogs = effectiveGpuReconstructionMode !== 'legacy';
  const formatGpuReadableValue = (value: number | null | undefined, fractionDigits = 1): string =>
    Number.isFinite(value) ? Number(value).toFixed(fractionDigits) : 'na';
  if (shouldEmitVerboseGpuLogs) {
    console.log(
      `[CPR-GPU-WORKER] run=${input.debugRunId ?? 'na'} requestedAggregation=${aggregation} ` +
        `reconstructionMode=${effectiveGpuReconstructionMode} ` +
        `effectivePipeline=legacy-gpu-renderer ` +
        `virtualPano=${effectiveGpuReconstructionMode === 'legacy' ? 'off' : 'misrouted-nonlegacy-request'} ` +
        `backgroundSuppression=renderer-internal denoise=renderer-internal+worker-residual ` +
        `pano=${panoWidth}x${panoHeight} ` +
        `verticalHalfMm=${formatGpuReadableValue(effectiveVerticalHalfMm)} ` +
        `centerOffsetMm=${formatGpuReadableValue(verticalCenterOffsetMm)} ` +
        `slabHalfMm=${formatGpuReadableValue(slabHalfThicknessMm)} ` +
        `slabSamples=${requestedSlabSamples}`
    );
  }
  const { renderPanoGpu } = await loadGpuRendererModule();
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
  const temporaryDebugDisplayMode = gpuResult.diagnostics?.toneMap?.temporaryDisplayMode ?? null;
  const gpuResidualDenoiseApplied =
    gpuResult.pipelineMode === 'multi-pass' &&
    !temporaryDebugDisplayMode &&
    applyLightBilateralDenoise(
      gpuResult.pixelData,
      panoWidth,
      panoHeight,
      GPU_RESIDUAL_DENOISE_BLEND,
      buildGpuResidualDenoiseWeights(gpuResult.maxMap, gpuResult.sampleCountMap)
    );
  const elapsedMs = performance.now() - startedAt;
  const {
    minValue,
    maxValue,
    windowWidth,
    windowCenter,
  } = computeAutoDisplayWindow(gpuResult.pixelData);
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
  if (shouldEmitVerboseGpuLogs) {
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
        residualDenoiseBlend:
          gpuResult.pipelineMode === 'multi-pass' ? GPU_RESIDUAL_DENOISE_BLEND : 0,
        residualDenoiseApplied: gpuResidualDenoiseApplied,
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
        `residualDenoise=${gpuResidualDenoiseApplied ? 'on' : 'off'} ` +
        `phase2Gate=${phase2GatePassed ? 'pass' : 'fail'}`
    );
    console.log(
      '[CPR-SUPPORT-SURFACE-JSON]',
      JSON.stringify(
        gpuResult.diagnostics?.supportSurface
          ? {
              ...gpuLogContext,
              ...gpuResult.diagnostics.supportSurface,
              rawSupportSurface: gpuResult.diagnostics.rawSupportSurface ?? null,
              rawSupportPeaks: gpuResult.diagnostics.rawSupportPeaks ?? null,
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
  }
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
      renderRoute: 'gpu-legacy-direct',
      rendererFamily: 'legacy-gpu-panorama',
      routeIntegrity:
        reconstructionMode === 'legacy'
          ? 'matched-request'
          : 'unexpected-nonlegacy-on-legacy-gpu-route',
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
      scalarSampling: {
        debugScalarSamplingMode: scalarPolicy.debugScalarSamplingMode,
        normalizationSignature: scalarPolicy.normalizationSignature,
        requestedModalityLutApplied: scalarPolicy.requestedModalityLutApplied,
        modalityLutApplied: scalarPolicy.shouldApplyModalityLut,
        storedValueNormalizationApplied: scalarPolicy.shouldNormalizeStoredValues,
      },
      gpuRender: {
        enabled: true,
        debugScalarSamplingMode: scalarPolicy.debugScalarSamplingMode,
        pipelineMode: gpuResult.pipelineMode,
        expectedPipelineMode: gpuResult.diagnostics?.expectedPipelineMode ?? 'multi-pass',
        phase2GatePassed,
        degradedModeReason: gpuResult.diagnostics?.degradedModeReason ?? null,
        volumeCacheKey,
        usedPerColumnVerticalDirs: usesPerColumnVerticalDirs,
        authoritativeWorldToIndex: true,
        modalityLutApplied: scalarPolicy.shouldApplyModalityLut,
        storedValueNormalizationApplied: scalarPolicy.shouldNormalizeStoredValues,
        residualDenoise: {
          blend: gpuResult.pipelineMode === 'multi-pass' ? GPU_RESIDUAL_DENOISE_BLEND : 0,
          applied: gpuResidualDenoiseApplied,
        },
        rawSupportPeaks: gpuResult.diagnostics?.rawSupportPeaks ?? null,
        supportSurface: gpuResult.diagnostics?.supportSurface ?? null,
        rawSupportFormation: gpuResult.diagnostics?.rawSupportFormation ?? null,
        supportFormation: gpuResult.diagnostics?.supportFormation ?? null,
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
  depthHalfRangeMm: number,
  analysisCenterRow?: number
): {
  minValue: number;
  maxValue: number;
  range: number;
  toothBandMean: number;
  toothBandP10: number;
  toothBandP90: number;
  toothBandBrightFraction: number;
  lowerBandMean: number;
  lowerBandBrightFraction: number;
  supportDepthClampFraction: number;
} {
  const resolvedAnalysisCenterRow =
    Number.isFinite(analysisCenterRow) && Number(analysisCenterRow) >= 0
      ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(analysisCenterRow))))
      : panoCenterRow;
  const resolvedAnalysisHalfHeight = Math.max(
    1,
    Math.max(resolvedAnalysisCenterRow, panoHeight - 1 - resolvedAnalysisCenterRow)
  );
  const rowFromNormalizedOffset = (yNorm: number): number => {
    const row = Math.round(resolvedAnalysisCenterRow + yNorm * resolvedAnalysisHalfHeight);
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
  let toothBandBrightCount = 0;
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
        if (value > 1200) {
          toothBandBrightCount++;
        }
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
    toothBandBrightFraction: toothBandCount > 0 ? toothBandBrightCount / toothBandCount : 0,
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
  upperSupportDepthMm?: Float32Array;
  lowerSupportDepthMm?: Float32Array;
  upperSupportReliabilityByCol?: Float32Array;
  lowerSupportReliabilityByCol?: Float32Array;
  softThresholdByCol: Float32Array;
  hardThresholdByCol: Float32Array;
  supportTiltMmByCol?: Float32Array;
  virtualPanoDepthHalfRangeMm: number;
}): {
  pixelData: Float32Array;
  summary: ReturnType<typeof summarizeVirtualPanoOutput>;
  debugMaps: GpuPanoDebugMaps;
  diagnostics: {
    enabled: boolean;
    usedAsOutput: boolean;
    acceptedByLowerBandTolerance: boolean;
    acceptedByToothBandTolerance: boolean;
    renderSupportMode: 'singlePath' | 'dualArchBlend';
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
    upperSupportDepthFirst8Mm: number[];
    lowerSupportDepthFirst8Mm: number[];
    upperSupportReliabilityP50: number;
    lowerSupportReliabilityP50: number;
    supportBlendRows: {
      upperHoldEnd: number;
      lowerHoldStart: number;
    };
    troughSigmaMm: number;
    approxTroughHalfWidthMm: number;
    lowerPenaltyP50: number;
    lowerPenaltyP90: number;
    toneResponseP50: number;
    toneResponseP90: number;
    detailHuP50: number;
    contextHuP50: number;
    contextBlendMean: number;
    contextWeightFractionMean: number;
    columnSupportReliabilityP50: number;
    attenuationStrength: number;
    gamma: number;
    outputHuMin: number;
    outputHuMax: number;
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
    upperSupportDepthMm,
    lowerSupportDepthMm,
    upperSupportReliabilityByCol,
    lowerSupportReliabilityByCol,
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
  const hasDualArchSupport =
    !!upperSupportDepthMm &&
    !!lowerSupportDepthMm &&
    upperSupportDepthMm.length === panoWidth &&
    lowerSupportDepthMm.length === panoWidth;
  const supportDepthMap = new Float32Array(planeSize);
  const supportConfidenceMap = new Float32Array(planeSize);
  const upperSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const upperSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const supportBlendMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const totalAttenuationMap = new Float32Array(planeSize);
  const lowerPenaltyMap = new Float32Array(planeSize);
  const participatingSampleCountMap = new Float32Array(planeSize);
  const toneResponseMap = new Float32Array(planeSize);
  const troughHalfWidthMap = new Float32Array(planeSize);
  const detailHuMap = new Float32Array(planeSize);
  const contextHuMap = new Float32Array(planeSize);
  const contextBlendFactorMap = new Float32Array(planeSize);
  const columnSupportReliabilityMap = new Float32Array(planeSize);
  const SUPPORT_SIGMA_MM = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportSigmaMm;
  const upperArchSigmaMm = SUPPORT_SIGMA_MM * 0.9;
  const lowerArchSigmaMm = SUPPORT_SIGMA_MM * 1.1;
  const ALPHA_SCALE = CPU_VIRTUAL_PANO_MODEL_PARAMS.alphaScale;
  const upperHoldEndRow = Math.max(0, Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.08)));
  const lowerHoldStartRow = Math.max(
    upperHoldEndRow + 1,
    Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.42))
  );
  let contextWeightFractionSum = 0;
  let contextWeightFractionCount = 0;
  let contextBlendFactorSum = 0;
  let contextBlendFactorCount = 0;

  for (let row = 0; row < panoHeight; row++) {
    const yNorm = panoCenterRow > 0 ? (row - panoCenterRow) / panoCenterRow : 0;
    const archBlend =
      hasDualArchSupport && lowerHoldStartRow > upperHoldEndRow
        ? smoothstep01((row - upperHoldEndRow) / Math.max(1, lowerHoldStartRow - upperHoldEndRow))
        : 0;

    for (let col = 0; col < panoWidth; col++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const supportTiltMm = hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0;
      const mergedSupportDepthBaseMm = Number(selectedDepthMm[col]);
      const upperSupportDepthRawMm = hasDualArchSupport
        ? Number(upperSupportDepthMm?.[col] ?? mergedSupportDepthBaseMm)
        : mergedSupportDepthBaseMm;
      const lowerSupportDepthRawMm = hasDualArchSupport
        ? Number(lowerSupportDepthMm?.[col] ?? mergedSupportDepthBaseMm)
        : mergedSupportDepthBaseMm;
      const upperSupportReliability = hasDualArchSupport
        ? clampUnitInterval(Number(upperSupportReliabilityByCol?.[col]) || 0)
        : 0;
      const lowerSupportReliability = hasDualArchSupport
        ? clampUnitInterval(Number(lowerSupportReliabilityByCol?.[col]) || 0)
        : 0;
      const columnSupportReliability = hasDualArchSupport
        ? clampUnitInterval(
            (upperSupportReliability * (1 - archBlend) + lowerSupportReliability * archBlend) * 0.72 +
              Math.min(upperSupportReliability, lowerSupportReliability) * 0.28
          )
        : 1;
      const archSeparationRetention = hasDualArchSupport
        ? clampNumber(
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention +
              (1 - CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention) *
                smoothstep01((columnSupportReliability - 0.14) / 0.74),
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention,
            1
          )
        : 1;
      const upperSupportDepthBaseMm = hasDualArchSupport
        ? mergedSupportDepthBaseMm +
          (upperSupportDepthRawMm - mergedSupportDepthBaseMm) * archSeparationRetention
        : mergedSupportDepthBaseMm;
      const lowerSupportDepthBaseMm = hasDualArchSupport
        ? mergedSupportDepthBaseMm +
          (lowerSupportDepthRawMm - mergedSupportDepthBaseMm) * archSeparationRetention
        : mergedSupportDepthBaseMm;
      const supportSigmaMm = hasDualArchSupport
        ? upperArchSigmaMm * (1 - archBlend) +
          lowerArchSigmaMm * archBlend +
          Math.abs(lowerSupportDepthBaseMm - upperSupportDepthBaseMm) * 0.04 +
          (1 - columnSupportReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportSigmaLowReliabilityBoostMm
        : SUPPORT_SIGMA_MM;
      const supportDenom = 2.0 * supportSigmaMm * supportSigmaMm;
      const supportEnergyWindowMm =
        supportSigmaMm * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnergyWindowScale;
      const supportHalfWidthMm =
        supportSigmaMm * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnergyWindowScale;
      const contextSigmaMm = supportSigmaMm * CPU_VIRTUAL_PANO_MODEL_PARAMS.contextSigmaScale;
      const contextDenom = 2.0 * contextSigmaMm * contextSigmaMm;
      const supportDepthMm = clampNumber(
        mergedSupportDepthBaseMm + supportTiltMm * yNorm,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const softThreshold = Number(softThresholdByCol[col]);
      const hardThreshold = Number(hardThresholdByCol[col]);
      const hardDen = Math.max(hardThreshold - softThreshold, 80);
      const contextThresholdOffset = Math.max(
        CPU_VIRTUAL_PANO_MODEL_PARAMS.contextThresholdOffset,
        hardDen * CPU_VIRTUAL_PANO_MODEL_PARAMS.contextThresholdBlend
      );
      const contextSoftThreshold = softThreshold - contextThresholdOffset;
      const contextHardThreshold = hardThreshold - contextThresholdOffset * 0.18;
      const contextDen = Math.max(contextHardThreshold - contextSoftThreshold, 120);
      const emissionDen = Math.max(
        hardDen * CPU_VIRTUAL_PANO_MODEL_PARAMS.emissionDenScale,
        CPU_VIRTUAL_PANO_MODEL_PARAMS.emissionDenMin
      );
      let retainedSampleCount = 0;
      let validDepthCount = 0;
      let accumulatedSignal = 0;
      let transmittance = 1;
      let contextWeightedSum = 0;
      let contextWeightTotal = 0;

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
        const troughWeight = Math.exp(-(absDelta * absDelta) / supportDenom);
        const contextTroughWeight = Math.exp(-(absDelta * absDelta) / contextDenom);
        if (absDelta <= supportEnergyWindowMm) {
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
        const contextResponse = clampNumber((value - contextSoftThreshold) / contextDen, 0, 1);
        const contextWeight =
          contextTroughWeight *
          Math.max(
            CPU_VIRTUAL_PANO_MODEL_PARAMS.contextWeightFloor,
            Math.sqrt(contextResponse)
          );
        if (contextWeight > 1e-4) {
          contextWeightedSum +=
            clampNumber(
              value,
              CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuFloor,
              CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuCeiling
            ) * contextWeight;
          contextWeightTotal += contextWeight;
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
      const supportConfidence =
        validDepthCount > 0 ? clampNumber(retainedSampleCount / validDepthCount, 0, 1) : 0;
      const pseudoAttenuation =
        attenuationSignal > 0 ? -Math.log(Math.max(1e-3, 1 - attenuationSignal)) : 0;
      const lowerPenaltyRow =
        yNorm <= CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyRowStart
          ? 0
          : smoothstep01(
            (yNorm - CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyRowStart) /
            Math.max(
              1e-3,
              CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyRowEnd -
              CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyRowStart
            )
          );
      const lowerPenalty =
        lowerPenaltyRow *
        (CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyBase +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.lowerPenaltyLowConfidenceScale * (1 - supportConfidence));
      const correctedAttenuation = Math.max(0, pseudoAttenuation - lowerPenalty);
      let toneResponse =
        1 -
        Math.exp(
          -CPU_VIRTUAL_PANO_MODEL_PARAMS.attenuationStrength * correctedAttenuation
        );
      toneResponse = Math.pow(clampNumber(toneResponse, 0, 1), CPU_VIRTUAL_PANO_MODEL_PARAMS.gamma);
      toneResponse = smoothstepRange(
        CPU_VIRTUAL_PANO_MODEL_PARAMS.signalFloor,
        CPU_VIRTUAL_PANO_MODEL_PARAMS.signalCeiling,
        toneResponse
      );
      const detailHu =
        retainedSampleCount <= 0 || validDepthCount <= 0
          ? -1000
          : CPU_VIRTUAL_PANO_MODEL_PARAMS.outputHuMin +
            toneResponse *
              (CPU_VIRTUAL_PANO_MODEL_PARAMS.outputHuMax -
                CPU_VIRTUAL_PANO_MODEL_PARAMS.outputHuMin);
      const contextHu =
        contextWeightTotal > 1e-4
          ? clampNumber(
              contextWeightedSum / contextWeightTotal,
              CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuFloor,
              CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuCeiling
            )
          : detailHu;
      const structuralReliability = clampUnitInterval(
        supportConfidence * 0.45 + columnSupportReliability * 0.55
      );
      const toothBandFocus = 1 - clampUnitInterval(Math.abs(yNorm - 0.08) / 0.92);
      const contextBlendFactor = clampNumber(
        CPU_VIRTUAL_PANO_MODEL_PARAMS.contextBlendBase +
          (1 - structuralReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.contextBlendLowReliabilityScale +
          (1 - toothBandFocus) * CPU_VIRTUAL_PANO_MODEL_PARAMS.contextBlendOuterRowScale -
          toneResponse * toothBandFocus * CPU_VIRTUAL_PANO_MODEL_PARAMS.contextBlendToothPreserve,
        0.12,
        0.58
      );
      if (retainedSampleCount <= 0 || validDepthCount <= 0) {
        fallbackNoEligibleCount++;
        pixelData[pixelIndex] = contextWeightTotal > 1e-4 ? contextHu : -1000;
      } else {
        pixelData[pixelIndex] = detailHu * (1 - contextBlendFactor) + contextHu * contextBlendFactor;
      }
      detailHuMap[pixelIndex] = detailHu;
      contextHuMap[pixelIndex] = contextHu;
      contextBlendFactorMap[pixelIndex] = contextBlendFactor;
      columnSupportReliabilityMap[pixelIndex] = columnSupportReliability;
      supportDepthMap[pixelIndex] = supportDepthMm;
      supportConfidenceMap[pixelIndex] = supportConfidence;
      if (upperSupportDepthMap && lowerSupportDepthMap && supportBlendMap) {
        upperSupportDepthMap[pixelIndex] = upperSupportDepthBaseMm;
        lowerSupportDepthMap[pixelIndex] = lowerSupportDepthBaseMm;
        upperSupportConfidenceMap![pixelIndex] = upperSupportReliability;
        lowerSupportConfidenceMap![pixelIndex] = lowerSupportReliability;
        supportBlendMap[pixelIndex] = archBlend;
      }
      totalAttenuationMap[pixelIndex] = correctedAttenuation;
      lowerPenaltyMap[pixelIndex] = lowerPenalty;
      participatingSampleCountMap[pixelIndex] = retainedSampleCount;
      toneResponseMap[pixelIndex] = retainedSampleCount > 0 && validDepthCount > 0 ? toneResponse : 0;
      troughHalfWidthMap[pixelIndex] = supportHalfWidthMm;
      if (validDepthCount > 0) {
        contextWeightFractionSum += contextWeightTotal / validDepthCount;
        contextWeightFractionCount++;
      }
      contextBlendFactorSum += contextBlendFactor;
      contextBlendFactorCount++;

      if (yNorm >= 0.65) {
        lowerBandRenderCount++;
        lowerBandAttenuationSum += lowerPenalty;
        lowerBandAttenuationMax = Math.max(lowerBandAttenuationMax, lowerPenalty);
        if (toneResponse < 0.12) {
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
  if (
    summary.toothBandMean > 980 ||
    summary.toothBandP10 > 220 ||
    summary.toothBandBrightFraction > 0.42
  ) {
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
  const acceptedByToothBandTolerance =
    rejectReasons.length > 0 &&
    rejectReasons.every(reason => reason === 'tooth-band-saturation') &&
    summary.toothBandMean <= 1040 &&
    summary.toothBandP10 <= 180 &&
    summary.toothBandBrightFraction <= 0.3 &&
    toothBandContrastRange >= 520 &&
    lowerSuppressionRatio <= 0.5 &&
    emptyFallbackFraction <= 0.08 &&
    retainedWeightFractionMean >= 0.05 &&
    offTroughEnergyRatio <= 0.35;
  const usedAsOutput =
    rejectReasons.length === 0 || acceptedByLowerBandTolerance || acceptedByToothBandTolerance;
  let supportTiltMeanAbsMm = 0;
  let supportTiltMaxAbsMm = 0;
  for (let col = 0; col < panoWidth; col++) {
    const absTiltMm = Math.abs(hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0);
    supportTiltMeanAbsMm += absTiltMm;
    supportTiltMaxAbsMm = Math.max(supportTiltMaxAbsMm, absTiltMm);
  }
  supportTiltMeanAbsMm = panoWidth > 0 ? supportTiltMeanAbsMm / panoWidth : 0;
  const lowerPenaltySummary = summarizeFiniteFloat32Buffer(lowerPenaltyMap);
  const toneResponseSummary = summarizeFiniteFloat32Buffer(toneResponseMap);
  const detailHuSummary = summarizeFiniteFloat32Buffer(detailHuMap);
  const contextHuSummary = summarizeFiniteFloat32Buffer(contextHuMap);
  const columnSupportReliabilitySummary = summarizeFiniteFloat32Buffer(
    columnSupportReliabilityMap
  );

  return {
    pixelData,
    summary,
    debugMaps: {
      supportDepthMap,
      supportConfidenceMap,
      upperSupportDepthMap,
      lowerSupportDepthMap,
      upperSupportConfidenceMap,
      lowerSupportConfidenceMap,
      supportBlendMap,
      totalAttenuationMap,
      lowerPenaltyMap,
      participatingSampleCountMap,
      toneResponseMap,
      troughHalfWidthMap,
    },
    diagnostics: {
      enabled: true,
      usedAsOutput,
      acceptedByLowerBandTolerance,
      acceptedByToothBandTolerance,
      renderSupportMode: hasDualArchSupport ? 'dualArchBlend' : 'singlePath',
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
      upperSupportDepthFirst8Mm: Array.from(
        (upperSupportDepthMm || selectedDepthMm).subarray(
          0,
          Math.min(8, (upperSupportDepthMm || selectedDepthMm).length)
        )
      ).map(value => Math.round(Number(value) * 1000) / 1000),
      lowerSupportDepthFirst8Mm: Array.from(
        (lowerSupportDepthMm || selectedDepthMm).subarray(
          0,
          Math.min(8, (lowerSupportDepthMm || selectedDepthMm).length)
        )
      ).map(value => Math.round(Number(value) * 1000) / 1000),
      upperSupportReliabilityP50:
        upperSupportReliabilityByCol && upperSupportReliabilityByCol.length > 0
          ? percentile(Array.from(upperSupportReliabilityByCol), 0.5)
          : 0,
      lowerSupportReliabilityP50:
        lowerSupportReliabilityByCol && lowerSupportReliabilityByCol.length > 0
          ? percentile(Array.from(lowerSupportReliabilityByCol), 0.5)
          : 0,
      supportBlendRows: {
        upperHoldEnd: upperHoldEndRow,
        lowerHoldStart: lowerHoldStartRow,
      },
      troughSigmaMm: hasDualArchSupport ? 0.5 * (upperArchSigmaMm + lowerArchSigmaMm) : SUPPORT_SIGMA_MM,
      approxTroughHalfWidthMm:
        hasDualArchSupport
          ? 0.5 *
            ((upperArchSigmaMm + lowerArchSigmaMm) *
              CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnergyWindowScale)
          : SUPPORT_SIGMA_MM * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnergyWindowScale,
      lowerPenaltyP50: lowerPenaltySummary?.p50 ?? 0,
      lowerPenaltyP90: lowerPenaltySummary?.p90 ?? 0,
      toneResponseP50: toneResponseSummary?.p50 ?? 0,
      toneResponseP90: toneResponseSummary?.p90 ?? 0,
      detailHuP50: detailHuSummary?.p50 ?? 0,
      contextHuP50: contextHuSummary?.p50 ?? 0,
      contextBlendMean:
        contextBlendFactorCount > 0 ? contextBlendFactorSum / contextBlendFactorCount : 0,
      contextWeightFractionMean:
        contextWeightFractionCount > 0 ? contextWeightFractionSum / contextWeightFractionCount : 0,
      columnSupportReliabilityP50: columnSupportReliabilitySummary?.p50 ?? 0,
      attenuationStrength: CPU_VIRTUAL_PANO_MODEL_PARAMS.attenuationStrength,
      gamma: CPU_VIRTUAL_PANO_MODEL_PARAMS.gamma,
      outputHuMin: CPU_VIRTUAL_PANO_MODEL_PARAMS.outputHuMin,
      outputHuMax: CPU_VIRTUAL_PANO_MODEL_PARAMS.outputHuMax,
    },
  };
}

function renderDualArchToothProjectionPano(params: {
  virtualPanoStack: Float32Array;
  panoWidth: number;
  panoHeight: number;
  planeSize: number;
  panoCenterRow: number;
  adaptiveToothCenterRow?: number;
  centerRowByCol?: Int16Array;
  halfHeightByCol?: Float32Array;
  virtualPanoDepthOffsetsMm: Float32Array;
  selectedDepthMm: Float32Array;
  upperSupportDepthMm?: Float32Array;
  lowerSupportDepthMm?: Float32Array;
  upperSupportAnchorRow?: number;
  lowerSupportAnchorRow?: number;
  upperSupportReliabilityByCol?: Float32Array;
  lowerSupportReliabilityByCol?: Float32Array;
  softThresholdByCol?: Float32Array;
  hardThresholdByCol?: Float32Array;
  supportTiltMmByCol?: Float32Array;
  virtualPanoDepthHalfRangeMm: number;
}): {
  pixelData: Float32Array;
  summary: ReturnType<typeof summarizeVirtualPanoOutput>;
  debugMaps: GpuPanoDebugMaps;
  diagnostics: {
    enabled: boolean;
    usedAsOutput: boolean;
    acceptedByLowerBandTolerance: boolean;
    acceptedByToothBandTolerance: boolean;
    renderSupportMode: 'dualArchProjection' | 'singlePathProjection';
    pipelineVariant: 'dualArchDirectProjection';
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
    upperSupportDepthFirst8Mm: number[];
    lowerSupportDepthFirst8Mm: number[];
    upperSupportReliabilityP50: number;
    lowerSupportReliabilityP50: number;
    supportBlendRows: {
      upperHoldEnd: number;
      lowerHoldStart: number;
    };
    troughSigmaMm: number;
    approxTroughHalfWidthMm: number;
    lowerPenaltyP50: number;
    lowerPenaltyP90: number;
    toneResponseP50: number;
    toneResponseP90: number;
    detailHuP50: number;
    contextHuP50: number;
    contextBlendMean: number;
    contextWeightFractionMean: number;
    columnSupportReliabilityP50: number;
    upperDetailHuP50: number;
    lowerDetailHuP50: number;
    detailSampleFractionMean: number;
    shadowLiftMean: number;
    attenuationStrength: number;
    gamma: number;
    outputHuMin: number;
    outputHuMax: number;
  };
} {
  const {
    virtualPanoStack,
    panoWidth,
    panoHeight,
    planeSize,
    panoCenterRow,
    adaptiveToothCenterRow,
    centerRowByCol,
    halfHeightByCol,
    virtualPanoDepthOffsetsMm,
    selectedDepthMm,
    upperSupportDepthMm,
    lowerSupportDepthMm,
    upperSupportAnchorRow,
    lowerSupportAnchorRow,
    upperSupportReliabilityByCol,
    lowerSupportReliabilityByCol,
    softThresholdByCol,
    hardThresholdByCol,
    supportTiltMmByCol,
    virtualPanoDepthHalfRangeMm,
  } = params;

  const depthSamples = virtualPanoDepthOffsetsMm.length;
  const depthStepMm =
    depthSamples > 1
      ? Math.abs(Number(virtualPanoDepthOffsetsMm[1]) - Number(virtualPanoDepthOffsetsMm[0]))
      : 0.25;
  const pixelData = new Float32Array(planeSize);
  const backgroundHu = CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionBackgroundHu;
  const quietBackgroundHu = Math.max(
    backgroundHu,
    CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionQuietBackgroundHu
  );
  const outputHuFloor = CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionOutputHuFloor;
  const outputHuCeiling = CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionOutputHuCeiling;
  pixelData.fill(backgroundHu);

  const hasSupportTilt = !!supportTiltMmByCol && supportTiltMmByCol.length === panoWidth;
  const hasDualArchSupport =
    !!upperSupportDepthMm &&
    !!lowerSupportDepthMm &&
    upperSupportDepthMm.length === panoWidth &&
    lowerSupportDepthMm.length === panoWidth;
  const hasAdaptiveCenterRows = !!centerRowByCol && centerRowByCol.length === panoWidth;
  const hasAdaptiveHalfHeights = !!halfHeightByCol && halfHeightByCol.length === panoWidth;
  const hasThresholdProfiles =
    !!softThresholdByCol &&
    !!hardThresholdByCol &&
    softThresholdByCol.length === panoWidth &&
    hardThresholdByCol.length === panoWidth;
  const hasDualArchAnchors =
    hasDualArchSupport &&
    Number.isFinite(upperSupportAnchorRow) &&
    Number.isFinite(lowerSupportAnchorRow) &&
    Number(lowerSupportAnchorRow) > Number(upperSupportAnchorRow);
  const anchorReferenceRow = Number.isFinite(adaptiveToothCenterRow)
    ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(adaptiveToothCenterRow))))
    : panoCenterRow;
  const resolvedUpperAnchorRow = hasDualArchAnchors
    ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(upperSupportAnchorRow))))
    : Math.max(0, Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.18)));
  const resolvedLowerAnchorRow = hasDualArchAnchors
    ? Math.max(1, Math.min(panoHeight - 1, Math.round(Number(lowerSupportAnchorRow))))
    : Math.max(resolvedUpperAnchorRow + 1, Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.6)));
  const upperAnchorOffsetFromCenter = resolvedUpperAnchorRow - anchorReferenceRow;
  const lowerAnchorOffsetFromCenter = resolvedLowerAnchorRow - anchorReferenceRow;

  const supportDepthMap = new Float32Array(planeSize);
  const supportConfidenceMap = new Float32Array(planeSize);
  const upperSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const upperSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const supportBlendMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const totalAttenuationMap = new Float32Array(planeSize);
  const lowerPenaltyMap = new Float32Array(planeSize);
  const participatingSampleCountMap = new Float32Array(planeSize);
  const toneResponseMap = new Float32Array(planeSize);
  const troughHalfWidthMap = new Float32Array(planeSize);
  const detailHuMap = new Float32Array(planeSize);
  const contextHuMap = new Float32Array(planeSize);
  const contextBlendFactorMap = new Float32Array(planeSize);
  const columnSupportReliabilityMap = new Float32Array(planeSize);
  const upperDetailHuMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerDetailHuMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const eligibleValues = new Float32Array(depthSamples);
  const supportBlendUpperSummaryRows: number[] = [];
  const supportBlendLowerSummaryRows: number[] = [];

  let activePixelCount = 0;
  let eligibleSampleCount = 0;
  let lowerBandEligibleSampleCount = 0;
  let inBoundsSampleCount = 0;
  let lowerBandInBoundsSampleCount = 0;
  let fallbackNoEligibleCount = 0;
  let retainedWeightFractionSum = 0;
  let retainedWeightFractionCount = 0;
  let onSupportEnergy = 0;
  let offSupportEnergy = 0;
  let detailSampleFractionSum = 0;
  let detailSampleFractionCount = 0;

  for (let col = 0; col < panoWidth; col++) {
    const rawCenterRow = hasAdaptiveCenterRows ? Number(centerRowByCol?.[col]) : panoCenterRow;
    const resolvedCenterRow = Number.isFinite(rawCenterRow)
      ? Math.max(0, Math.min(panoHeight - 1, Math.round(rawCenterRow)))
      : panoCenterRow;
    const rawHalfHeight = hasAdaptiveHalfHeights
      ? Number(halfHeightByCol?.[col])
      : Math.max(resolvedCenterRow, panoHeight - 1 - resolvedCenterRow);
    const resolvedHalfHeight = Math.max(1, Number.isFinite(rawHalfHeight) ? rawHalfHeight : panoCenterRow);
    const upperAnchorRowForCol = hasDualArchAnchors
      ? clampNumber(resolvedCenterRow + upperAnchorOffsetFromCenter, 0, panoHeight - 2)
      : clampNumber(resolvedCenterRow - resolvedHalfHeight * 0.18, 0, panoHeight - 2);
    const lowerAnchorRowForCol = hasDualArchAnchors
      ? clampNumber(
          Math.max(upperAnchorRowForCol + 1, resolvedCenterRow + lowerAnchorOffsetFromCenter),
          1,
          panoHeight - 1
        )
      : clampNumber(
          Math.max(upperAnchorRowForCol + 1, resolvedCenterRow + resolvedHalfHeight * 0.22),
          1,
          panoHeight - 1
        );

    supportBlendUpperSummaryRows.push(Math.round(upperAnchorRowForCol));
    supportBlendLowerSummaryRows.push(Math.round(lowerAnchorRowForCol));

    const upperSupportDepthBaseMm =
      hasDualArchSupport && Number.isFinite(Number(upperSupportDepthMm?.[col]))
        ? Number(upperSupportDepthMm?.[col])
        : Number(selectedDepthMm[col]);
    const lowerSupportDepthBaseMm =
      hasDualArchSupport && Number.isFinite(Number(lowerSupportDepthMm?.[col]))
        ? Number(lowerSupportDepthMm?.[col])
        : Number(selectedDepthMm[col]);
    const mergedSupportDepthBaseMm = hasDualArchSupport
      ? 0.5 * (upperSupportDepthBaseMm + lowerSupportDepthBaseMm)
      : Number(selectedDepthMm[col]);
    const upperReliability = clampNumber(
      hasDualArchSupport && upperSupportReliabilityByCol ? Number(upperSupportReliabilityByCol[col]) || 0 : 1,
      0,
      1
    );
    const lowerReliability = clampNumber(
      hasDualArchSupport && lowerSupportReliabilityByCol ? Number(lowerSupportReliabilityByCol[col]) || 0 : 1,
      0,
      1
    );
    const columnSoftThreshold = hasThresholdProfiles ? Number(softThresholdByCol?.[col]) : -150;
    const columnHardThreshold = hasThresholdProfiles ? Number(hardThresholdByCol?.[col]) : 280;
    const columnHardDen = Math.max(140, columnHardThreshold - columnSoftThreshold);

    for (let row = 0; row < panoHeight; row++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const yNorm = resolvedHalfHeight > 0 ? (row - resolvedCenterRow) / resolvedHalfHeight : 0;
      const supportBlend =
        hasDualArchSupport && lowerAnchorRowForCol > upperAnchorRowForCol
          ? smoothstep01((row - upperAnchorRowForCol) / Math.max(1, lowerAnchorRowForCol - upperAnchorRowForCol))
          : 0;
      const supportDepthBaseMm = hasDualArchSupport
        ? upperSupportDepthBaseMm * (1 - supportBlend) + lowerSupportDepthBaseMm * supportBlend
        : mergedSupportDepthBaseMm;
      const supportReliability = hasDualArchSupport
        ? clampNumber(upperReliability * (1 - supportBlend) + lowerReliability * supportBlend, 0, 1)
        : 1;
      const columnSupportReliability = hasDualArchSupport
        ? clampNumber(
            supportReliability * 0.82 + Math.min(upperReliability, lowerReliability) * 0.18,
            0,
            1
          )
        : 1;
      const supportTiltMm = hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0;
      const supportDepthMm = clampNumber(
        supportDepthBaseMm +
          supportTiltMm * yNorm * CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionSupportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const upperSupportDepthMmLocal = clampNumber(
        upperSupportDepthBaseMm +
          supportTiltMm * yNorm * CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionSupportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const lowerSupportDepthMmLocal = clampNumber(
        lowerSupportDepthBaseMm +
          supportTiltMm * yNorm * CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionSupportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const supportWindowHalfWidthMm =
        CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionWindowHalfWidthMm +
        (1 - columnSupportReliability) *
          CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionWindowLowReliabilityBoostMm;
      const supportWindowSigmaMm = Math.max(depthStepMm * 2.5, supportWindowHalfWidthMm * 0.58);
      const outerWindowHalfWidthMm = Math.min(
        virtualPanoDepthHalfRangeMm,
        supportWindowHalfWidthMm * CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionOuterWindowScale
      );
      const lowerBandSuppression = smoothstep01(clampNumber((yNorm - 0.18) / 0.72, 0, 1));
      const focalEligibilityHalfWidthMm = Math.min(
        outerWindowHalfWidthMm,
        CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionFocalHalfWidthMm
      );
      const rowFocalEligibilityHalfWidthMm = Math.max(
        depthStepMm * 1.75,
        focalEligibilityHalfWidthMm -
          lowerBandSuppression * 0.85 -
          Math.max(0, 0.74 - columnSupportReliability) * 0.25
      );
      const rowToothAdmissionThreshold = clampNumber(
        0.08 +
          lowerBandSuppression * 0.24 +
          Math.max(0, 0.78 - columnSupportReliability) * 0.1,
        0.05,
        0.62
      );
      const rowOffBandToothAdmissionThreshold = clampNumber(
        rowToothAdmissionThreshold + 0.16,
        0.12,
        0.82
      );
      let eligibleCount = 0;
      let rayMax = backgroundHu;
      let validDepthCount = 0;
      let retainedTopSampleCount = 0;
      let bestToothlikeSample = backgroundHu;
      let bestToothlikeResponse = 0;

      activePixelCount++;
      supportDepthMap[pixelIndex] = supportDepthMm;
      supportConfidenceMap[pixelIndex] = columnSupportReliability;
      if (upperSupportDepthMap && lowerSupportDepthMap) {
        upperSupportDepthMap[pixelIndex] = upperSupportDepthMmLocal;
        lowerSupportDepthMap[pixelIndex] = lowerSupportDepthMmLocal;
      }
      if (upperSupportConfidenceMap && lowerSupportConfidenceMap) {
        upperSupportConfidenceMap[pixelIndex] = upperReliability;
        lowerSupportConfidenceMap[pixelIndex] = lowerReliability;
      }
      if (supportBlendMap) {
        supportBlendMap[pixelIndex] = supportBlend;
      }
      troughHalfWidthMap[pixelIndex] = rowFocalEligibilityHalfWidthMm;
      columnSupportReliabilityMap[pixelIndex] = columnSupportReliability;

      for (let depth = 0; depth < depthSamples; depth++) {
        const sample = Number(virtualPanoStack[stackIndex(depth, pixelIndex, planeSize)]);
        if (!Number.isFinite(sample)) {
          continue;
        }

        const depthOffsetMm = Number(virtualPanoDepthOffsetsMm[depth]);
        const absDepthDeltaMm = Math.abs(depthOffsetMm - supportDepthMm);
        const isOutsideFocalEligibilityWindow = absDepthDeltaMm > rowFocalEligibilityHalfWidthMm;
        if (isOutsideFocalEligibilityWindow) {
          offSupportEnergy += Math.max(0, sample + 950);
        }

        validDepthCount++;
        inBoundsSampleCount++;
        if (yNorm >= 0.65) {
          lowerBandInBoundsSampleCount++;
        }

        const supportWeight = Math.exp(
          -(absDepthDeltaMm * absDepthDeltaMm) / Math.max(1e-6, 2 * supportWindowSigmaMm * supportWindowSigmaMm)
        );
        if (!isOutsideFocalEligibilityWindow) {
          onSupportEnergy += Math.max(0, sample + 950) * supportWeight;
        }

        const clampedSample = clampNumber(sample, outputHuFloor, outputHuCeiling);
        const hardResponse = clampNumber(
          (clampedSample - columnSoftThreshold) / Math.max(1e-6, columnHardDen),
          0,
          1
        );
        if (
          hardResponse > bestToothlikeResponse + 1e-4 ||
          (Math.abs(hardResponse - bestToothlikeResponse) <= 1e-4 &&
            clampedSample > bestToothlikeSample)
        ) {
          bestToothlikeResponse = hardResponse;
          bestToothlikeSample = clampedSample;
        }
        if (validDepthCount === 1 || clampedSample > rayMax) {
          rayMax = clampedSample;
        }
        const offBandRescueFloorHu = Math.min(
          outputHuCeiling,
          CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionOffBandRescueFloorHu +
            lowerBandSuppression *
              CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionLowerBandOffBandRescueBoostHu +
            Math.max(0, 0.82 - columnSupportReliability) * 140
        );
        const passesReducerGate =
          clampedSample >= CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionGateFloorHu &&
          hardResponse >=
            (isOutsideFocalEligibilityWindow
              ? rowOffBandToothAdmissionThreshold
              : rowToothAdmissionThreshold) &&
          (
            !isOutsideFocalEligibilityWindow ||
            clampedSample >= offBandRescueFloorHu
          );
        if (passesReducerGate) {
          eligibleValues[eligibleCount] = clampedSample;
          eligibleCount++;
        }
      }

      let pixelValue = backgroundHu;
      if (eligibleCount > 0) {
        sortValuesAscending(eligibleValues, eligibleCount);
        const contributingCount = Math.max(1, Math.ceil(eligibleCount * 0.30));
        
        // CRITICAL: Must start at the end of the ascending array to get the highest values
        const startIndex = eligibleCount - contributingCount;
        let topSum = 0;
        
        for (let i = startIndex; i < eligibleCount; i++) {
          topSum += eligibleValues[i];
        }
        
        pixelValue = topSum / contributingCount;
        retainedTopSampleCount = contributingCount;
        eligibleSampleCount += eligibleCount;
        if (yNorm >= 0.65) {
          lowerBandEligibleSampleCount += eligibleCount;
        }
        const retainedWeightFraction = retainedTopSampleCount / Math.max(1, validDepthCount);
        retainedWeightFractionSum += retainedWeightFraction;
        retainedWeightFractionCount++;
        detailSampleFractionSum += retainedWeightFraction;
        detailSampleFractionCount++;
      } else {
        const upperRowBackgroundOnlyFallback =
          yNorm <= CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionTopBackgroundOnlyRowEnd;
        const shouldPreferBackgroundFallback =
          lowerBandSuppression > 0.32 ||
          (columnSupportReliability < 0.72 && yNorm > 0.28);
        if (
          !upperRowBackgroundOnlyFallback &&
          bestToothlikeResponse >= rowToothAdmissionThreshold * 0.72 &&
          bestToothlikeSample > backgroundHu
        ) {
          pixelValue = bestToothlikeSample;
        } else {
          pixelValue =
            shouldPreferBackgroundFallback || upperRowBackgroundOnlyFallback
              ? quietBackgroundHu
              : rayMax;
        }
        retainedTopSampleCount = 0;
        fallbackNoEligibleCount++;
      }

      const lowerPenalty =
        lowerBandSuppression *
        (0.14 + (1 - columnSupportReliability) * 0.48) *
        (retainedTopSampleCount <= 1 ? 1.15 : 1);
      lowerPenaltyMap[pixelIndex] = lowerPenalty;
      if (lowerPenalty > 1e-4 && pixelValue > quietBackgroundHu) {
        const darkTarget = Math.max(
          quietBackgroundHu,
          Math.min(
            pixelValue -
              CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionLowerPenaltyDarkenHu *
                lowerBandSuppression,
            Math.max(quietBackgroundHu, columnSoftThreshold - 60)
          )
        );
        pixelValue =
          pixelValue +
          (darkTarget - pixelValue) *
            Math.min(
              CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionLowerPenaltyMaxBlend,
              lowerPenalty
            );
      }
      pixelValue = clampNumber(pixelValue, outputHuFloor, outputHuCeiling);
      pixelData[pixelIndex] = pixelValue;
      detailHuMap[pixelIndex] = pixelValue;
      contextHuMap[pixelIndex] = pixelValue;
      participatingSampleCountMap[pixelIndex] = retainedTopSampleCount;
      totalAttenuationMap[pixelIndex] = lowerPenalty;
      toneResponseMap[pixelIndex] =
        validDepthCount > 0
          ? clampNumber(retainedTopSampleCount / Math.max(1, validDepthCount), 0, 1)
          : 0;
      if (upperDetailHuMap) {
        upperDetailHuMap[pixelIndex] = backgroundHu + (pixelValue - backgroundHu) * (1 - supportBlend);
      }
      if (lowerDetailHuMap) {
        lowerDetailHuMap[pixelIndex] = backgroundHu + (pixelValue - backgroundHu) * supportBlend;
      }
    }
  }

  const summary = summarizeVirtualPanoOutput(
    pixelData,
    panoWidth,
    panoHeight,
    panoCenterRow,
    selectedDepthMm,
    virtualPanoDepthHalfRangeMm,
    adaptiveToothCenterRow
  );
  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const lowerSuppressionRatio =
    summary.toothBandMean > 1e-6 ? summary.lowerBandMean / summary.toothBandMean : 1;
  const emptyFallbackFraction = activePixelCount > 0 ? fallbackNoEligibleCount / Math.max(1, activePixelCount) : 1;
  const eligibleSampleFraction =
    inBoundsSampleCount > 0 ? eligibleSampleCount / Math.max(1, inBoundsSampleCount) : 0;
  const lowerBandEligibleFraction =
    lowerBandInBoundsSampleCount > 0
      ? lowerBandEligibleSampleCount / Math.max(1, lowerBandInBoundsSampleCount)
      : 0;
  const retainedWeightFractionMean =
    retainedWeightFractionCount > 0 ? retainedWeightFractionSum / retainedWeightFractionCount : 0;
  const offTroughEnergyRatio = onSupportEnergy > 1e-6 ? offSupportEnergy / onSupportEnergy : 0;
  const lowerPenaltySummary = summarizeFiniteFloat32Buffer(lowerPenaltyMap);
  const toneResponseSummary = summarizeFiniteFloat32Buffer(toneResponseMap);
  const detailHuSummary = summarizeFiniteFloat32Buffer(detailHuMap);
  const contextHuSummary = summarizeFiniteFloat32Buffer(contextHuMap);
  const upperDetailHuSummary = upperDetailHuMap ? summarizeFiniteFloat32Buffer(upperDetailHuMap) : null;
  const lowerDetailHuSummary = lowerDetailHuMap ? summarizeFiniteFloat32Buffer(lowerDetailHuMap) : null;
  const columnSupportReliabilitySummary = summarizeFiniteFloat32Buffer(columnSupportReliabilityMap);
  const upperSupportReliabilitySummary = upperSupportReliabilityByCol
    ? summarizeFiniteFloat32Buffer(upperSupportReliabilityByCol)
    : null;
  const lowerSupportReliabilitySummary = lowerSupportReliabilityByCol
    ? summarizeFiniteFloat32Buffer(lowerSupportReliabilityByCol)
    : null;

  const rejectReasons: string[] = [];
  if (activePixelCount <= 0) {
    rejectReasons.push('active-projection-band-empty');
  }
  if (eligibleSampleFraction < 0.01) {
    rejectReasons.push('eligible-sample-fraction-too-low');
  }
  if (retainedWeightFractionMean < 0.08) {
    rejectReasons.push('retained-weight-fraction-too-low');
  }
  if (emptyFallbackFraction > 0.28) {
    rejectReasons.push('empty-fallback-fraction-too-high');
  }
  if (summary.supportDepthClampFraction > 0.36) {
    rejectReasons.push('support-depth-clamp-fraction-too-high');
  }
  if (summary.range < 500) {
    rejectReasons.push('range-too-low');
  }
  if (toothBandContrastRange < 160) {
    rejectReasons.push('tooth-band-contrast-too-low');
  }
  if (summary.toothBandMean > 1420 || summary.toothBandP10 > 320) {
    rejectReasons.push('tooth-band-saturation');
  }
  if (offTroughEnergyRatio > 1.8) {
    rejectReasons.push('off-trough-energy-too-high');
  }
  if (summary.lowerBandBrightFraction > 0.46) {
    rejectReasons.push('lower-band-bright-fraction-too-high');
  }
  if (summary.lowerBandMean > 40) {
    rejectReasons.push('lower-band-mean-too-high');
  }
  if (lowerSuppressionRatio > 0.82) {
    rejectReasons.push('lower-suppression-ratio-too-high');
  }

  const acceptedByLowerBandTolerance = false;
  const acceptedByToothBandTolerance =
    rejectReasons.length > 0 &&
    rejectReasons.every(reason => reason === 'tooth-band-saturation') &&
    summary.toothBandMean <= 1520 &&
    summary.toothBandP10 <= 260 &&
    toothBandContrastRange >= 220;
  const usedAsOutput = rejectReasons.length === 0 || acceptedByToothBandTolerance;

  let supportTiltMeanAbsMm = 0;
  let supportTiltMaxAbsMm = 0;
  for (let col = 0; col < panoWidth; col++) {
    const absTiltMm = Math.abs(hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0);
    supportTiltMeanAbsMm += absTiltMm;
    supportTiltMaxAbsMm = Math.max(supportTiltMaxAbsMm, absTiltMm);
  }
  supportTiltMeanAbsMm = panoWidth > 0 ? supportTiltMeanAbsMm / panoWidth : 0;
  const supportBlendUpperRepresentativeRow =
    supportBlendUpperSummaryRows.length > 0 ? Math.round(percentile(supportBlendUpperSummaryRows, 0.5)) : 0;
  const supportBlendLowerRepresentativeRow =
    supportBlendLowerSummaryRows.length > 0 ? Math.round(percentile(supportBlendLowerSummaryRows, 0.5)) : 0;

  return {
    pixelData,
    summary,
    debugMaps: {
      upperSupportDepthMap,
      lowerSupportDepthMap,
      upperSupportConfidenceMap,
      lowerSupportConfidenceMap,
      supportBlendMap,
      supportDepthMap,
      supportConfidenceMap,
      totalAttenuationMap,
      lowerPenaltyMap,
      participatingSampleCountMap,
      toneResponseMap,
      troughHalfWidthMap,
      detailHuMap,
      contextHuMap,
      contextBlendFactorMap,
      columnSupportReliabilityMap,
      upperDetailHuMap,
      lowerDetailHuMap,
    },
    diagnostics: {
      enabled: true,
      usedAsOutput,
      acceptedByLowerBandTolerance,
      acceptedByToothBandTolerance,
      renderSupportMode: hasDualArchSupport ? 'dualArchProjection' : 'singlePathProjection',
      pipelineVariant: 'dualArchDirectProjection',
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
        coverageFraction: 0,
        meanAttenuation: 0,
        maxAttenuation: 0,
      },
      supportTiltMode: hasSupportTilt ? 'linear' : 'disabled',
      supportTiltFirst8Mm: hasSupportTilt
        ? Array.from(supportTiltMmByCol!.subarray(0, Math.min(8, panoWidth))).map(
            value => Math.round(Number(value) * 1000) / 1000
          )
        : [],
      supportTiltMeanAbsMm,
      supportTiltMaxAbsMm,
      supportDepthFirst8Mm: Array.from(selectedDepthMm.subarray(0, Math.min(8, panoWidth))).map(
        value => Math.round(Number(value) * 1000) / 1000
      ),
      upperSupportDepthFirst8Mm: upperSupportDepthMm
        ? Array.from(upperSupportDepthMm.subarray(0, Math.min(8, panoWidth))).map(
            value => Math.round(Number(value) * 1000) / 1000
          )
        : [],
      lowerSupportDepthFirst8Mm: lowerSupportDepthMm
        ? Array.from(lowerSupportDepthMm.subarray(0, Math.min(8, panoWidth))).map(
            value => Math.round(Number(value) * 1000) / 1000
          )
        : [],
      upperSupportReliabilityP50: upperSupportReliabilitySummary?.p50 ?? 0,
      lowerSupportReliabilityP50: lowerSupportReliabilitySummary?.p50 ?? 0,
      supportBlendRows: {
        upperHoldEnd: supportBlendUpperRepresentativeRow,
        lowerHoldStart: supportBlendLowerRepresentativeRow,
      },
      troughSigmaMm: CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionWindowHalfWidthMm * 0.58,
      approxTroughHalfWidthMm: CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionWindowHalfWidthMm,
      lowerPenaltyP50: lowerPenaltySummary?.p50 ?? 0,
      lowerPenaltyP90: lowerPenaltySummary?.p90 ?? 0,
      toneResponseP50: toneResponseSummary?.p50 ?? 0,
      toneResponseP90: toneResponseSummary?.p90 ?? 0,
      detailHuP50: detailHuSummary?.p50 ?? 0,
      contextHuP50: contextHuSummary?.p50 ?? backgroundHu,
      contextBlendMean: 0,
      contextWeightFractionMean: 0,
      columnSupportReliabilityP50: columnSupportReliabilitySummary?.p50 ?? 0,
      upperDetailHuP50: upperDetailHuSummary?.p50 ?? detailHuSummary?.p50 ?? 0,
      lowerDetailHuP50: lowerDetailHuSummary?.p50 ?? detailHuSummary?.p50 ?? 0,
      detailSampleFractionMean:
        detailSampleFractionCount > 0 ? detailSampleFractionSum / detailSampleFractionCount : 0,
      shadowLiftMean: 0,
      attenuationStrength: 0,
      gamma: 1,
      outputHuMin: outputHuFloor,
      outputHuMax: outputHuCeiling,
    },
  };
}

function renderArchGuidedSyntheticPano(params: {
  virtualPanoStack: Float32Array;
  panoWidth: number;
  panoHeight: number;
  planeSize: number;
  panoCenterRow: number;
  adaptiveToothCenterRow?: number;
  centerRowByCol?: Int16Array;
  halfHeightByCol?: Float32Array;
  virtualPanoDepthOffsetsMm: Float32Array;
  selectedDepthMm: Float32Array;
  upperSupportDepthMm?: Float32Array;
  lowerSupportDepthMm?: Float32Array;
  upperSupportAnchorRow?: number;
  lowerSupportAnchorRow?: number;
  upperSupportReliabilityByCol?: Float32Array;
  lowerSupportReliabilityByCol?: Float32Array;
  softThresholdByCol: Float32Array;
  hardThresholdByCol: Float32Array;
  supportTiltMmByCol?: Float32Array;
  virtualPanoDepthHalfRangeMm: number;
}): {
  pixelData: Float32Array;
  summary: ReturnType<typeof summarizeVirtualPanoOutput>;
  debugMaps: GpuPanoDebugMaps;
  diagnostics: {
    enabled: boolean;
    usedAsOutput: boolean;
    acceptedByLowerBandTolerance: boolean;
    acceptedByToothBandTolerance: boolean;
    acceptedByContextTolerance: boolean;
    renderSupportMode: 'archGuidedDualLayer';
    pipelineVariant: 'archGuidedSynthetic';
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
    upperSupportDepthFirst8Mm: number[];
    lowerSupportDepthFirst8Mm: number[];
    upperSupportReliabilityP50: number;
    lowerSupportReliabilityP50: number;
    supportBlendRows: {
      upperHoldEnd: number;
      lowerHoldStart: number;
    };
    troughSigmaMm: number;
    approxTroughHalfWidthMm: number;
    lowerPenaltyP50: number;
    lowerPenaltyP90: number;
    toneResponseP50: number;
    toneResponseP90: number;
    detailHuP50: number;
    contextHuP50: number;
    contextBlendMean: number;
    contextWeightFractionMean: number;
    columnSupportReliabilityP50: number;
    upperDetailHuP50: number;
    lowerDetailHuP50: number;
    detailSampleFractionMean: number;
    shadowLiftMean: number;
    attenuationStrength: number;
    gamma: number;
    outputHuMin: number;
    outputHuMax: number;
  };
} {
  const {
    virtualPanoStack,
    panoWidth,
    panoHeight,
    planeSize,
    panoCenterRow,
    adaptiveToothCenterRow,
    centerRowByCol,
    halfHeightByCol,
    virtualPanoDepthOffsetsMm,
    selectedDepthMm,
    upperSupportDepthMm,
    lowerSupportDepthMm,
    upperSupportAnchorRow,
    lowerSupportAnchorRow,
    upperSupportReliabilityByCol,
    lowerSupportReliabilityByCol,
    softThresholdByCol,
    hardThresholdByCol,
    supportTiltMmByCol,
    virtualPanoDepthHalfRangeMm,
  } = params;

  const depthSamples = virtualPanoDepthOffsetsMm.length;
  const pixelData = new Float32Array(planeSize);
  const hasSupportTilt = !!supportTiltMmByCol && supportTiltMmByCol.length === panoWidth;
  const hasDualArchSupport =
    !!upperSupportDepthMm &&
    !!lowerSupportDepthMm &&
    upperSupportDepthMm.length === panoWidth &&
    lowerSupportDepthMm.length === panoWidth;
  const supportDepthMap = new Float32Array(planeSize);
  const supportConfidenceMap = new Float32Array(planeSize);
  const upperSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportDepthMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const upperSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerSupportConfidenceMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const supportBlendMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const totalAttenuationMap = new Float32Array(planeSize);
  const lowerPenaltyMap = new Float32Array(planeSize);
  const participatingSampleCountMap = new Float32Array(planeSize);
  const toneResponseMap = new Float32Array(planeSize);
  const troughHalfWidthMap = new Float32Array(planeSize);
  const detailHuMap = new Float32Array(planeSize);
  const contextHuMap = new Float32Array(planeSize);
  const contextBlendFactorMap = new Float32Array(planeSize);
  const columnSupportReliabilityMap = new Float32Array(planeSize);
  const upperDetailHuMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;
  const lowerDetailHuMap = hasDualArchSupport ? new Float32Array(planeSize) : undefined;

  const hasDualArchAnchors =
    hasDualArchSupport &&
    Number.isFinite(upperSupportAnchorRow) &&
    Number.isFinite(lowerSupportAnchorRow) &&
    Number(lowerSupportAnchorRow) > Number(upperSupportAnchorRow);
  const hasAdaptiveCenterRows = !!centerRowByCol && centerRowByCol.length === panoWidth;
  const hasAdaptiveHalfHeights = !!halfHeightByCol && halfHeightByCol.length === panoWidth;
  const anchorReferenceRow = Number.isFinite(adaptiveToothCenterRow)
    ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(adaptiveToothCenterRow))))
    : panoCenterRow;
  const resolvedUpperAnchorRow = hasDualArchAnchors
    ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(upperSupportAnchorRow))))
    : Math.max(0, Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.12)));
  const resolvedLowerAnchorRow = hasDualArchAnchors
    ? Math.max(0, Math.min(panoHeight - 1, Math.round(Number(lowerSupportAnchorRow))))
    : Math.max(resolvedUpperAnchorRow + 1, Math.min(panoHeight - 1, Math.round(panoCenterRow * 0.48)));
  const upperAnchorOffsetFromCenter = resolvedUpperAnchorRow - anchorReferenceRow;
  const lowerAnchorOffsetFromCenter = resolvedLowerAnchorRow - anchorReferenceRow;
  const columnCenterRows = new Int16Array(panoWidth);
  const columnHalfHeights = new Float32Array(panoWidth);
  const columnUpperAnchorRows = new Int16Array(panoWidth);
  const columnLowerAnchorRows = new Int16Array(panoWidth);
  const columnBlendCenterRows = new Int16Array(panoWidth);
  const columnInterArchRowSpans = new Int16Array(panoWidth);
  const columnUpperHoldEndRows = new Int16Array(panoWidth);
  const columnLowerHoldStartRows = new Int16Array(panoWidth);
  const supportBlendUpperSummaryRows: number[] = [];
  const supportBlendLowerSummaryRows: number[] = [];

  for (let col = 0; col < panoWidth; col++) {
    const rawCenterRow = hasAdaptiveCenterRows ? Number(centerRowByCol?.[col]) : panoCenterRow;
    const resolvedCenterRow = Number.isFinite(rawCenterRow)
      ? Math.max(0, Math.min(panoHeight - 1, Math.round(rawCenterRow)))
      : panoCenterRow;
    const rawHalfHeight = hasAdaptiveHalfHeights
      ? Number(halfHeightByCol?.[col])
      : Math.max(resolvedCenterRow, panoHeight - 1 - resolvedCenterRow);
    const resolvedHalfHeight = Math.max(
      1,
      Number.isFinite(rawHalfHeight) ? rawHalfHeight : panoCenterRow
    );
    const upperAnchorRowForCol = hasDualArchAnchors
      ? clampNumber(resolvedCenterRow + upperAnchorOffsetFromCenter, 0, panoHeight - 2)
      : clampNumber(resolvedCenterRow - resolvedHalfHeight * 0.18, 0, panoHeight - 2);
    const lowerAnchorRowForCol = hasDualArchAnchors
      ? clampNumber(
          Math.max(upperAnchorRowForCol + 1, resolvedCenterRow + lowerAnchorOffsetFromCenter),
          1,
          panoHeight - 1
        )
      : clampNumber(
          Math.max(upperAnchorRowForCol + 1, resolvedCenterRow + resolvedHalfHeight * 0.24),
          1,
          panoHeight - 1
        );
    const interArchRowSpanForCol = Math.max(
      12,
      Math.round(lowerAnchorRowForCol) - Math.round(upperAnchorRowForCol)
    );
    const blendCenterRowForCol = hasDualArchSupport
      ? Math.round((upperAnchorRowForCol + lowerAnchorRowForCol) * 0.5)
      : resolvedCenterRow;
    const blendHalfSpanRowsForCol = Math.max(8, Math.round(interArchRowSpanForCol * 0.28));
    const upperHoldEndRowForCol = Math.max(
      0,
      Math.min(panoHeight - 1, blendCenterRowForCol - blendHalfSpanRowsForCol)
    );
    const lowerHoldStartRowForCol = Math.max(
      upperHoldEndRowForCol + 1,
      Math.min(panoHeight - 1, blendCenterRowForCol + blendHalfSpanRowsForCol)
    );

    columnCenterRows[col] = resolvedCenterRow;
    columnHalfHeights[col] = resolvedHalfHeight;
    columnUpperAnchorRows[col] = Math.round(upperAnchorRowForCol);
    columnLowerAnchorRows[col] = Math.round(lowerAnchorRowForCol);
    columnBlendCenterRows[col] = blendCenterRowForCol;
    columnInterArchRowSpans[col] = interArchRowSpanForCol;
    columnUpperHoldEndRows[col] = upperHoldEndRowForCol;
    columnLowerHoldStartRows[col] = lowerHoldStartRowForCol;
    supportBlendUpperSummaryRows.push(upperHoldEndRowForCol);
    supportBlendLowerSummaryRows.push(lowerHoldStartRowForCol);
  }

  const supportBlendUpperRepresentativeRow =
    supportBlendUpperSummaryRows.length > 0
      ? Math.round(percentile(supportBlendUpperSummaryRows, 0.5))
      : 0;
  const supportBlendLowerRepresentativeRow =
    supportBlendLowerSummaryRows.length > 0
      ? Math.round(percentile(supportBlendLowerSummaryRows, 0.5))
      : 0;
  const detailHalfWidthBaseMm = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticDetailHalfWidthMm;
  const detailHalfWidthLowReliabilityBoostMm =
    CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticDetailLowReliabilityBoostMm;
  const contextHalfWidthBaseMm = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextHalfWidthMm;
  const contextHalfWidthLowReliabilityBoostMm =
    CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextLowReliabilityBoostMm;
  const detailWeightFloor = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticDetailWeightFloor;
  const detailToothWeightScale =
    CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticDetailToothWeightScale;
  const contextWeightFloor = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextWeightFloor;
  const contextToothWeightScale =
    CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextToothWeightScale;
  const peakBlend = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticPeakBlend;
  const outputHuFloor = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticOutputHuFloor;
  const outputHuCeiling = CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticOutputHuCeiling;
  const contextFloorHu = CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuFloor;
  const contextCeilingHu = CPU_VIRTUAL_PANO_MODEL_PARAMS.contextHuCeiling;

  let eligibleSampleCount = 0;
  let lowerBandEligibleSampleCount = 0;
  let inBoundsSampleCount = 0;
  let lowerBandInBoundsSampleCount = 0;
  let fallbackNoEligibleCount = 0;
  let retainedWeightFractionSum = 0;
  let retainedWeightFractionCount = 0;
  let onSupportEnergy = 0;
  let offSupportEnergy = 0;
  let lowerBandRenderCount = 0;
  let lowerBandSuppressedCount = 0;
  let lowerBandAttenuationSum = 0;
  let lowerBandAttenuationMax = 0;
  let contextWeightFractionSum = 0;
  let contextWeightFractionCount = 0;
  let contextBlendFactorSum = 0;
  let contextBlendFactorCount = 0;
  let detailSampleFractionSum = 0;
  let detailSampleFractionCount = 0;
  let shadowLiftSum = 0;
  let shadowLiftCount = 0;

  for (let row = 0; row < panoHeight; row++) {
    for (let col = 0; col < panoWidth; col++) {
      const pixelIndex = planeIndex(col, row, panoWidth);
      const columnCenterRow = Number(columnCenterRows[col]);
      const columnHalfHeight = Math.max(1, Number(columnHalfHeights[col]));
      const columnUpperAnchorRow = Number(columnUpperAnchorRows[col]);
      const columnLowerAnchorRow = Number(columnLowerAnchorRows[col]);
      const columnBlendCenterRow = Number(columnBlendCenterRows[col]);
      const columnInterArchRowSpan = Math.max(12, Number(columnInterArchRowSpans[col]));
      const columnUpperHoldEndRow = Number(columnUpperHoldEndRows[col]);
      const columnLowerHoldStartRow = Number(columnLowerHoldStartRows[col]);
      const yNorm = columnHalfHeight > 0 ? (row - columnCenterRow) / columnHalfHeight : 0;
      const upperRowFocus = hasDualArchSupport
        ? 1 -
          smoothstepRange(
            columnUpperAnchorRow - Math.max(18, Math.round(columnInterArchRowSpan * 0.75)),
            columnUpperAnchorRow + Math.max(10, Math.round(columnInterArchRowSpan * 0.22)),
            row
          )
        : 1 - smoothstepRange(-0.18, 0.34, yNorm);
      const lowerRowFocus = hasDualArchSupport
        ? smoothstepRange(
            columnLowerAnchorRow - Math.max(10, Math.round(columnInterArchRowSpan * 0.22)),
            columnLowerAnchorRow + Math.max(18, Math.round(columnInterArchRowSpan * 0.75)),
            row
          )
        : smoothstepRange(-0.04, 0.72, yNorm);
      const middleBandFocus = hasDualArchSupport
        ? 1 -
          clampUnitInterval(
            Math.abs(row - columnBlendCenterRow) / Math.max(16, Math.round(columnInterArchRowSpan * 0.72))
          )
        : 1 - clampUnitInterval(Math.abs(yNorm - 0.14) / 0.46);
      const toothBandFocus = hasDualArchSupport
        ? 1 -
          clampUnitInterval(
            Math.abs(row - columnBlendCenterRow) / Math.max(22, Math.round(columnInterArchRowSpan * 0.95))
          )
        : 1 - clampUnitInterval(Math.abs(yNorm - 0.12) / 0.86);
      const outerBandFocus = hasDualArchSupport
        ? clampUnitInterval(
            (Math.abs(row - columnBlendCenterRow) - Math.max(8, Math.round(columnInterArchRowSpan * 0.22))) /
              Math.max(12, Math.round(columnInterArchRowSpan * 0.7))
          )
        : clampUnitInterval((Math.abs(yNorm - 0.12) - 0.16) / 0.52);
      const lowerBandFocus = hasDualArchSupport
        ? smoothstep01(
            (row -
              Math.min(
                panoHeight - 1,
                columnLowerAnchorRow + Math.max(10, Math.round(columnInterArchRowSpan * 0.18))
              )) /
              Math.max(
                12,
                panoHeight -
                  1 -
                  Math.min(
                    panoHeight - 1,
                    columnLowerAnchorRow + Math.max(10, Math.round(columnInterArchRowSpan * 0.18))
                  )
              )
          )
        : clampUnitInterval((yNorm - 0.34) / 0.42);
      const interArchGapFocus = hasDualArchSupport
        ? clampUnitInterval(
            1 - Math.abs(row - columnBlendCenterRow) / Math.max(14, Math.round(columnInterArchRowSpan * 0.42))
          ) * clampUnitInterval(1 - (upperRowFocus + lowerRowFocus) * 0.72)
        : 0;
      const rowArchBlend =
        hasDualArchSupport && columnLowerHoldStartRow > columnUpperHoldEndRow
          ? smoothstep01(
              (row - columnUpperHoldEndRow) /
                Math.max(1, columnLowerHoldStartRow - columnUpperHoldEndRow)
            )
          : 0;
      const supportTiltMm = hasSupportTilt ? Number(supportTiltMmByCol?.[col]) : 0;
      const mergedSupportDepthBaseMm = Number(selectedDepthMm[col]);
      const upperSupportDepthRawMm = hasDualArchSupport
        ? Number(upperSupportDepthMm?.[col] ?? mergedSupportDepthBaseMm)
        : mergedSupportDepthBaseMm;
      const lowerSupportDepthRawMm = hasDualArchSupport
        ? Number(lowerSupportDepthMm?.[col] ?? mergedSupportDepthBaseMm)
        : mergedSupportDepthBaseMm;
      const upperSupportReliability = hasDualArchSupport
        ? clampUnitInterval(Number(upperSupportReliabilityByCol?.[col]) || 0)
        : 1;
      const lowerSupportReliability = hasDualArchSupport
        ? clampUnitInterval(Number(lowerSupportReliabilityByCol?.[col]) || 0)
        : 1;
      const columnSupportReliability = hasDualArchSupport
        ? clampUnitInterval(
            upperSupportReliability * (1 - rowArchBlend) * 0.5 +
              lowerSupportReliability * rowArchBlend * 0.5 +
              Math.min(upperSupportReliability, lowerSupportReliability) * 0.32 +
              Math.max(upperSupportReliability, lowerSupportReliability) * 0.18
          )
        : 1;
      const structuralReliability = hasDualArchSupport
        ? clampUnitInterval(
            upperSupportReliability * upperRowFocus * 0.44 +
              lowerSupportReliability * lowerRowFocus * 0.44 +
              columnSupportReliability * 0.36
          )
        : 1;
      const interArchSeparationMm = hasDualArchSupport
        ? Math.abs(lowerSupportDepthRawMm - upperSupportDepthRawMm)
        : virtualPanoDepthHalfRangeMm;
      const interArchSeparationConfidence = hasDualArchSupport
        ? smoothstep01(
            (interArchSeparationMm - CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticSeparationMinMm) /
              Math.max(
                0.2,
                CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticSeparationIdealMm -
                  CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticSeparationMinMm
              )
          )
        : 1;
      const dualArchAmbiguity = hasDualArchSupport
        ? clampUnitInterval(
            (1 - structuralReliability) * 0.42 + (1 - interArchSeparationConfidence) * 0.58
          )
        : 0;
      const archSeparationRetention = hasDualArchSupport
        ? clampNumber(
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention +
              (1 - CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention) *
                smoothstep01((structuralReliability - 0.12) / 0.76) *
                (0.28 + interArchSeparationConfidence * 0.72),
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportDualArchCollapseMinRetention,
            1
          )
        : 1;
      const upperSupportDepthBaseMm = hasDualArchSupport
        ? mergedSupportDepthBaseMm +
          (upperSupportDepthRawMm - mergedSupportDepthBaseMm) * archSeparationRetention
        : mergedSupportDepthBaseMm;
      const lowerSupportDepthBaseMm = hasDualArchSupport
        ? mergedSupportDepthBaseMm +
          (lowerSupportDepthRawMm - mergedSupportDepthBaseMm) * archSeparationRetention
        : mergedSupportDepthBaseMm;
      const supportTiltScale = hasDualArchSupport ? 0.18 : 0.28;
      const mergedSupportDepthMm = clampNumber(
        mergedSupportDepthBaseMm + supportTiltMm * yNorm * supportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const upperSupportDepthMmLocal = clampNumber(
        upperSupportDepthBaseMm + supportTiltMm * yNorm * supportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const lowerSupportDepthMmLocal = clampNumber(
        lowerSupportDepthBaseMm + supportTiltMm * yNorm * supportTiltScale,
        -virtualPanoDepthHalfRangeMm,
        virtualPanoDepthHalfRangeMm
      );
      const softThreshold = Number(softThresholdByCol[col]);
      const hardThreshold = Number(hardThresholdByCol[col]);
      const hardDen = Math.max(hardThreshold - softThreshold, 120);
      const detailHalfWidthMm =
        detailHalfWidthBaseMm + (1 - structuralReliability) * detailHalfWidthLowReliabilityBoostMm;
      const detailSigmaMm = Math.max(0.35, detailHalfWidthMm * 0.72);
      const detailDenom = 2 * detailSigmaMm * detailSigmaMm;
      const contextHalfWidthMm = Math.max(
        0.92,
        contextHalfWidthBaseMm +
          (1 - structuralReliability) * contextHalfWidthLowReliabilityBoostMm * 0.32 +
          middleBandFocus * 0.08 -
          dualArchAmbiguity *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticAmbiguousContextTightenMm *
            (0.7 + toothBandFocus * 0.3)
      );
      const contextSigmaMm = Math.max(0.65, contextHalfWidthMm * 0.78);
      const contextDenom = 2 * contextSigmaMm * contextSigmaMm;
      // When upper/lower support paths are nearly collapsed, wide context sampling
      // paints a white veil behind teeth. Tighten and down-weight that context first.
      const contextLeakSuppression = hasDualArchSupport
        ? clampNumber(
            1 - dualArchAmbiguity * (0.46 + toothBandFocus * 0.2 + lowerBandFocus * 0.12),
            0.18,
            1
          )
        : 1;
      const mergedContextSuppression = hasDualArchSupport
        ? clampNumber(
            1 - dualArchAmbiguity * 0.72 - (1 - interArchSeparationConfidence) * 0.18,
            0.08,
            1
          )
        : 1;

      let validDepthCount = 0;
      let detailEligibleCount = 0;
      let upperDetailWeightedSum = 0;
      let upperDetailWeightTotal = 0;
      let lowerDetailWeightedSum = 0;
      let lowerDetailWeightTotal = 0;
      let upperContextWeightedSum = 0;
      let upperContextWeightTotal = 0;
      let lowerContextWeightedSum = 0;
      let lowerContextWeightTotal = 0;
      let mergedContextWeightedSum = 0;
      let mergedContextWeightTotal = 0;
      let upperDetailPeak = Number.NEGATIVE_INFINITY;
      let lowerDetailPeak = Number.NEGATIVE_INFINITY;

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
        const hardResponse = clampNumber((value - softThreshold) / hardDen, 0, 1);
        const toothResponse = Math.pow(hardResponse, 1.18);
        const contextResponse = Math.sqrt(hardResponse);
        const upperDelta = Math.abs(depthMm - upperSupportDepthMmLocal);
        const lowerDelta = Math.abs(depthMm - lowerSupportDepthMmLocal);
        const mergedDelta = Math.abs(depthMm - mergedSupportDepthMm);
        const nearestDeltaMm = hasDualArchSupport
          ? Math.min(upperDelta, lowerDelta, mergedDelta)
          : Math.min(upperDelta, mergedDelta);
        if (nearestDeltaMm <= detailHalfWidthMm * 1.6) {
          onSupportEnergy += Math.abs(value) * (0.15 + toothResponse * 0.85);
        } else {
          offSupportEnergy += Math.abs(value) * (0.12 + contextResponse * 0.35);
        }

        const upperDetailKernel = Math.exp(-(upperDelta * upperDelta) / detailDenom);
        const lowerDetailKernel = Math.exp(-(lowerDelta * lowerDelta) / detailDenom);
        const mergedContextKernel = Math.exp(-(mergedDelta * mergedDelta) / contextDenom);
        const upperContextKernel = Math.exp(-(upperDelta * upperDelta) / contextDenom);
        const lowerContextKernel = Math.exp(-(lowerDelta * lowerDelta) / contextDenom);

        const upperDetailWeight =
          upperDetailKernel * (detailWeightFloor + toothResponse * detailToothWeightScale);
        const lowerDetailWeight =
          lowerDetailKernel * (detailWeightFloor + toothResponse * detailToothWeightScale);
        const upperContextWeight =
          upperContextKernel *
          (contextWeightFloor + contextResponse * contextToothWeightScale) *
          contextLeakSuppression;
        const lowerContextWeight =
          lowerContextKernel *
          (contextWeightFloor + contextResponse * contextToothWeightScale) *
          contextLeakSuppression;
        const mergedContextWeight =
          mergedContextKernel *
          (contextWeightFloor + contextResponse * (contextToothWeightScale * 0.74)) *
          contextLeakSuppression *
          mergedContextSuppression;

        if (upperDetailWeight > 1e-4) {
          upperDetailWeightedSum += value * upperDetailWeight;
          upperDetailWeightTotal += upperDetailWeight;
          if (hardResponse > 0.08) {
            upperDetailPeak = Math.max(upperDetailPeak, value);
          }
        }
        if (lowerDetailWeight > 1e-4) {
          lowerDetailWeightedSum += value * lowerDetailWeight;
          lowerDetailWeightTotal += lowerDetailWeight;
          if (hardResponse > 0.08) {
            lowerDetailPeak = Math.max(lowerDetailPeak, value);
          }
        }
        if (upperContextWeight > 1e-4) {
          upperContextWeightedSum +=
            clampNumber(value, contextFloorHu, contextCeilingHu) * upperContextWeight;
          upperContextWeightTotal += upperContextWeight;
        }
        if (lowerContextWeight > 1e-4) {
          lowerContextWeightedSum +=
            clampNumber(value, contextFloorHu, contextCeilingHu) * lowerContextWeight;
          lowerContextWeightTotal += lowerContextWeight;
        }
        if (mergedContextWeight > 1e-4) {
          mergedContextWeightedSum +=
            clampNumber(value, contextFloorHu, contextCeilingHu) * mergedContextWeight;
          mergedContextWeightTotal += mergedContextWeight;
        }
        if (
          hardResponse > 0.1 &&
          (upperDelta <= detailHalfWidthMm * 1.4 || lowerDelta <= detailHalfWidthMm * 1.4)
        ) {
          detailEligibleCount++;
          eligibleSampleCount++;
          if (yNorm >= 0.65) {
            lowerBandEligibleSampleCount++;
          }
        }
      }

      if (validDepthCount > 0) {
        retainedWeightFractionSum += detailEligibleCount / validDepthCount;
        retainedWeightFractionCount++;
        detailSampleFractionSum += detailEligibleCount / validDepthCount;
        detailSampleFractionCount++;
      }

      const upperDetailBaseHu =
        upperDetailWeightTotal > 1e-4 ? upperDetailWeightedSum / upperDetailWeightTotal : Number.NaN;
      const lowerDetailBaseHu =
        lowerDetailWeightTotal > 1e-4 ? lowerDetailWeightedSum / lowerDetailWeightTotal : Number.NaN;
      const detailPeakBlend = peakBlend * clampNumber(
        0.48 + structuralReliability * 0.32 + toothBandFocus * 0.24 - lowerBandFocus * 0.08,
        0.34,
        1
      );
      const upperDetailHu = Number.isFinite(upperDetailBaseHu)
        ? clampNumber(
            upperDetailBaseHu * (1 - detailPeakBlend) +
              (Number.isFinite(upperDetailPeak) ? upperDetailPeak : upperDetailBaseHu) *
                detailPeakBlend,
            outputHuFloor,
            outputHuCeiling
          )
        : Number.NaN;
      const lowerDetailHu = Number.isFinite(lowerDetailBaseHu)
        ? clampNumber(
            lowerDetailBaseHu * (1 - detailPeakBlend) +
              (Number.isFinite(lowerDetailPeak) ? lowerDetailPeak : lowerDetailBaseHu) *
                detailPeakBlend,
            outputHuFloor,
            outputHuCeiling
          )
        : Number.NaN;
      const upperContextHu =
        upperContextWeightTotal > 1e-4
          ? clampNumber(upperContextWeightedSum / upperContextWeightTotal, contextFloorHu, contextCeilingHu)
          : upperDetailHu;
      const lowerContextHu =
        lowerContextWeightTotal > 1e-4
          ? clampNumber(lowerContextWeightedSum / lowerContextWeightTotal, contextFloorHu, contextCeilingHu)
          : lowerDetailHu;
      const mergedContextHu =
        mergedContextWeightTotal > 1e-4
          ? clampNumber(mergedContextWeightedSum / mergedContextWeightTotal, contextFloorHu, contextCeilingHu)
          : Number.isFinite(upperContextHu) && Number.isFinite(lowerContextHu)
            ? 0.5 * (upperContextHu + lowerContextHu)
            : Number.isFinite(upperContextHu)
              ? upperContextHu
              : Number.isFinite(lowerContextHu)
                ? lowerContextHu
                : -1000;

      const resolvedUpperDetailHu = Number.isFinite(upperDetailHu) ? upperDetailHu : mergedContextHu;
      const resolvedLowerDetailHu = Number.isFinite(lowerDetailHu) ? lowerDetailHu : mergedContextHu;
      const resolvedUpperContextHu = Number.isFinite(upperContextHu) ? upperContextHu : mergedContextHu;
      const resolvedLowerContextHu = Number.isFinite(lowerContextHu) ? lowerContextHu : mergedContextHu;
      const upperDetailConfidence =
        validDepthCount > 0 ? clampUnitInterval(upperDetailWeightTotal / validDepthCount) : 0;
      const lowerDetailConfidence =
        validDepthCount > 0 ? clampUnitInterval(lowerDetailWeightTotal / validDepthCount) : 0;
      const detailReliability = clampUnitInterval(
        upperDetailConfidence * upperRowFocus * 0.5 +
          lowerDetailConfidence * lowerRowFocus * 0.5 +
          structuralReliability * 0.45
      );
      const upperLayerWeight = hasDualArchSupport
        ? clampNumber(upperRowFocus * (0.75 + upperSupportReliability * 0.4), 0, 1.5)
        : 1;
      const lowerLayerWeight = hasDualArchSupport
        ? clampNumber(lowerRowFocus * (0.75 + lowerSupportReliability * 0.4), 0, 1.5)
        : 0;
      const detailLayerWeightTotal = upperLayerWeight + lowerLayerWeight;
      const detailLayerHu =
        detailLayerWeightTotal > 1e-4
          ? (resolvedUpperDetailHu * upperLayerWeight + resolvedLowerDetailHu * lowerLayerWeight) /
            detailLayerWeightTotal
          : mergedContextHu;
      const contextUpperWeight = hasDualArchSupport
        ? (upperLayerWeight * 0.68 + 0.1) * contextLeakSuppression
        : 1;
      const contextLowerWeight = hasDualArchSupport
        ? (lowerLayerWeight * 0.68 + 0.1) * contextLeakSuppression
        : 0;
      const mergedContextLayerWeight = hasDualArchSupport
        ? clampNumber(
            0.65 * (0.35 + interArchSeparationConfidence * 0.45) * mergedContextSuppression,
            0.06,
            0.65
          )
        : 0.65;
      const contextLayerWeightTotal =
        contextUpperWeight + contextLowerWeight + mergedContextLayerWeight;
      const contextLayerHu =
        contextLayerWeightTotal > 1e-4
          ? (resolvedUpperContextHu * contextUpperWeight +
              resolvedLowerContextHu * contextLowerWeight +
              mergedContextHu * mergedContextLayerWeight) /
            contextLayerWeightTotal
          : mergedContextHu;
      // Keep the synthetic context darker than the tooth/detail layer so the pano
      // does not grow a bright veil over the teeth or a bright lower fog band.
      const contextRowSuppressionHu =
        lowerBandFocus * 220 +
        outerBandFocus * 90 +
        Math.max(0, 0.4 - detailReliability) * 180 +
        dualArchAmbiguity *
          CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticAmbiguousContextSuppressionHu *
          (0.75 + toothBandFocus * 0.25);
      const contextBrightnessCapHu =
        detailLayerHu -
        (80 +
          toothBandFocus * 70 +
          lowerBandFocus * 150 +
          outerBandFocus * 40 +
          dualArchAmbiguity * 120 +
          (1 - interArchSeparationConfidence) * 70);
      const constrainedContextLayerHu = clampNumber(
        Math.min(contextLayerHu - contextRowSuppressionHu, contextBrightnessCapHu),
        outputHuFloor,
        outputHuCeiling
      );
      const detailContextGapHu = detailLayerHu - constrainedContextLayerHu;
      const contextBlendFactor = clampNumber(
        CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextBlendBase +
          (1 - detailReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextBlendLowReliabilityScale +
          middleBandFocus * CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextBlendMiddleBandScale +
          (1 - toothBandFocus) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticContextBlendOuterBandScale -
          interArchGapFocus * 0.18 -
          lowerBandFocus * 0.14 -
          clampUnitInterval((detailContextGapHu - 110) / 260) * 0.08 -
          Math.max(upperDetailConfidence, lowerDetailConfidence) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticToothPreserveScale -
          dualArchAmbiguity * CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticAmbiguousBlendReduction -
          (1 - interArchSeparationConfidence) * 0.06,
        hasDualArchSupport ? 0 : 0.03,
        0.34
      );
      const shadowFloorHu =
        constrainedContextLayerHu - CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticShadowFloorOffsetHu;
      const shadowLiftFactor = clampNumber(
        (1 - detailReliability) * CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticShadowLiftStrength +
          middleBandFocus * 0.05 -
          interArchGapFocus * 0.03 -
          dualArchAmbiguity *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticAmbiguousShadowLiftReduction -
          lowerBandFocus * 0.02,
        0,
        0.16
      );
      let lowerBandDarken = 0;
      let ambiguityDarken = 0;
      let interArchGapDarken = 0;
      let pixelValue =
        detailLayerHu * (1 - contextBlendFactor) +
        constrainedContextLayerHu * contextBlendFactor;
      if (Number.isFinite(shadowFloorHu) && pixelValue < shadowFloorHu) {
        pixelValue = pixelValue * (1 - shadowLiftFactor) + shadowFloorHu * shadowLiftFactor;
      }
      if (lowerBandFocus > 0) {
        const lowerBandDarkTarget = Math.min(constrainedContextLayerHu - 80, -780);
        lowerBandDarken = clampNumber(
          lowerBandFocus * (0.18 + (1 - detailReliability) * 0.16),
          0,
          0.42
        );
        pixelValue = pixelValue * (1 - lowerBandDarken) + lowerBandDarkTarget * lowerBandDarken;
      }
      if (dualArchAmbiguity > 0.02) {
        const ambiguityDarkTarget = Math.min(
          constrainedContextLayerHu - 120 - (1 - interArchSeparationConfidence) * 60,
          -820 + toothBandFocus * 80
        );
        ambiguityDarken = clampNumber(
          dualArchAmbiguity *
            (CPU_VIRTUAL_PANO_MODEL_PARAMS.archSyntheticAmbiguousDarkenStrength +
              lowerBandFocus * 0.14 +
              outerBandFocus * 0.08),
          0,
          0.34
        );
        pixelValue = pixelValue * (1 - ambiguityDarken) + ambiguityDarkTarget * ambiguityDarken;
      }
      if (interArchGapFocus > 0.02) {
        const interArchGapDarkTarget = Math.min(
          constrainedContextLayerHu - 150 - dualArchAmbiguity * 60,
          -860 + toothBandFocus * 70 + detailReliability * 40
        );
        interArchGapDarken = clampNumber(
          interArchGapFocus *
            (0.26 +
              dualArchAmbiguity * 0.34 +
              (1 - detailReliability) * 0.18 +
              (1 - interArchSeparationConfidence) * 0.12),
          0,
          0.58
        );
        pixelValue =
          pixelValue * (1 - interArchGapDarken) + interArchGapDarkTarget * interArchGapDarken;
      }
      const highlightStartHu = 380 + toothBandFocus * 20 - lowerBandFocus * 70;
      const highlightCompression =
        pixelValue > highlightStartHu
          ? clampNumber(
              ((pixelValue - highlightStartHu) / 320) *
                clampNumber(
                  0.28 + (1 - detailReliability) * 0.32 + toothBandFocus * 0.12,
                  0,
                  0.62
                ),
              0,
              0.62
            )
          : 0;
      if (highlightCompression > 0) {
        const compressedHighlightTarget =
          highlightStartHu + (pixelValue - highlightStartHu) * 0.18;
        pixelValue =
          pixelValue * (1 - highlightCompression) +
          compressedHighlightTarget * highlightCompression;
      }
      pixelValue = clampNumber(pixelValue, outputHuFloor, outputHuCeiling);

      if (!Number.isFinite(pixelValue) || validDepthCount <= 0) {
        fallbackNoEligibleCount++;
        pixelData[pixelIndex] = mergedContextHu;
      } else {
        pixelData[pixelIndex] = pixelValue;
      }

      detailHuMap[pixelIndex] = detailLayerHu;
      contextHuMap[pixelIndex] = constrainedContextLayerHu;
      contextBlendFactorMap[pixelIndex] = contextBlendFactor;
      columnSupportReliabilityMap[pixelIndex] = detailReliability;
      if (upperDetailHuMap && lowerDetailHuMap) {
        upperDetailHuMap[pixelIndex] = resolvedUpperDetailHu;
        lowerDetailHuMap[pixelIndex] = resolvedLowerDetailHu;
      }
      supportDepthMap[pixelIndex] = mergedSupportDepthMm;
      supportConfidenceMap[pixelIndex] = detailReliability;
      if (upperSupportDepthMap && lowerSupportDepthMap && supportBlendMap) {
        upperSupportDepthMap[pixelIndex] = upperSupportDepthMmLocal;
        lowerSupportDepthMap[pixelIndex] = lowerSupportDepthMmLocal;
        upperSupportConfidenceMap![pixelIndex] = upperSupportReliability;
        lowerSupportConfidenceMap![pixelIndex] = lowerSupportReliability;
        supportBlendMap[pixelIndex] = rowArchBlend;
      }
      totalAttenuationMap[pixelIndex] = constrainedContextLayerHu;
      lowerPenaltyMap[pixelIndex] = Math.max(
        shadowLiftFactor,
        lowerBandDarken,
        ambiguityDarken,
        interArchGapDarken
      );
      participatingSampleCountMap[pixelIndex] = detailEligibleCount;
      toneResponseMap[pixelIndex] = clampNumber(
        1 -
          contextBlendFactor -
          lowerBandDarken * 0.55 -
          ambiguityDarken * 0.45 -
          interArchGapDarken * 0.35 -
          highlightCompression * 0.25,
        0,
        1
      );
      troughHalfWidthMap[pixelIndex] = detailHalfWidthMm;

      if (validDepthCount > 0) {
        const contextWeightFraction =
          (upperContextWeightTotal + lowerContextWeightTotal + mergedContextWeightTotal) /
          (3 * validDepthCount);
        contextWeightFractionSum += contextWeightFraction;
        contextWeightFractionCount++;
      }
      contextBlendFactorSum += contextBlendFactor;
      contextBlendFactorCount++;
      shadowLiftSum += shadowLiftFactor;
      shadowLiftCount++;

      if (yNorm >= 0.65) {
        lowerBandRenderCount++;
        lowerBandAttenuationSum += shadowLiftFactor;
        lowerBandAttenuationMax = Math.max(lowerBandAttenuationMax, shadowLiftFactor);
        if (pixelValue < shadowFloorHu + 90) {
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

  const detailHuSummary = summarizeFiniteFloat32Buffer(detailHuMap);
  const contextHuSummary = summarizeFiniteFloat32Buffer(contextHuMap);
  const upperDetailHuSummary = upperDetailHuMap ? summarizeFiniteFloat32Buffer(upperDetailHuMap) : null;
  const lowerDetailHuSummary = lowerDetailHuMap ? summarizeFiniteFloat32Buffer(lowerDetailHuMap) : null;
  const columnSupportReliabilitySummary = summarizeFiniteFloat32Buffer(columnSupportReliabilityMap);
  const lowerPenaltySummary = summarizeFiniteFloat32Buffer(lowerPenaltyMap);
  const toneResponseSummary = summarizeFiniteFloat32Buffer(toneResponseMap);

  const rejectReasons: string[] = [];
  if (eligibleSampleFraction < 0.02) {
    rejectReasons.push('eligible-sample-fraction-too-low');
  }
  if (retainedWeightFractionMean < 0.03) {
    rejectReasons.push('retained-weight-fraction-too-low');
  }
  if (emptyFallbackFraction > 0.24) {
    rejectReasons.push('empty-fallback-fraction-too-high');
  }
  if (offTroughEnergyRatio > 1.35) {
    rejectReasons.push('off-trough-energy-too-high');
  }
  if (summary.range < 650) {
    rejectReasons.push('range-too-low');
  }
  if (toothBandContrastRange < 210) {
    rejectReasons.push('tooth-band-contrast-too-low');
  }
  if ((contextHuSummary?.p50 ?? -1000) < -700) {
    rejectReasons.push('context-layer-too-dark');
  }
  if (
    contextWeightFractionCount > 0 &&
    contextWeightFractionSum / Math.max(1, contextWeightFractionCount) < 0.05
  ) {
    rejectReasons.push('context-layer-too-thin');
  }
  if (
    summary.toothBandMean > 1180 ||
    summary.toothBandP10 > 260 ||
    summary.lowerBandBrightFraction > 0.74
  ) {
    rejectReasons.push('tooth-band-saturation');
  }
  if (summary.lowerBandMean > 120) {
    rejectReasons.push('lower-band-mean-too-high');
  }
  if (lowerSuppressionRatio > 0.84) {
    rejectReasons.push('lower-suppression-ratio-too-high');
  }

  const lowerBandOnlyRejectReasons = new Set([
    'lower-band-mean-too-high',
    'lower-suppression-ratio-too-high',
  ]);
  const nonLowerBandRejectReasons = rejectReasons.filter(
    reason => !lowerBandOnlyRejectReasons.has(reason)
  );
  const acceptedByLowerBandTolerance =
    rejectReasons.length > 0 &&
    nonLowerBandRejectReasons.length === 0 &&
    summary.lowerBandMean <= 165 &&
    summary.lowerBandBrightFraction <= 0.8 &&
    toothBandContrastRange >= 240 &&
    emptyFallbackFraction <= 0.16;
  const acceptedByToothBandTolerance =
    rejectReasons.length > 0 &&
    rejectReasons.every(
      reason => reason === 'tooth-band-saturation' || reason === 'context-layer-too-dark'
    ) &&
    summary.toothBandMean <= 1260 &&
    toothBandContrastRange >= 360 &&
    lowerSuppressionRatio <= 0.58 &&
    emptyFallbackFraction <= 0.12;
  const acceptedByContextTolerance =
    rejectReasons.length > 0 &&
    rejectReasons.every(
      reason => reason === 'context-layer-too-dark' || reason === 'context-layer-too-thin'
    ) &&
    toothBandContrastRange >= 320 &&
    (detailHuSummary?.p50 ?? -1000) >= 120 &&
    emptyFallbackFraction <= 0.12;
  const usedAsOutput =
    rejectReasons.length === 0 ||
    acceptedByLowerBandTolerance ||
    acceptedByToothBandTolerance ||
    acceptedByContextTolerance;

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
    debugMaps: {
      supportDepthMap,
      supportConfidenceMap,
      upperSupportDepthMap,
      lowerSupportDepthMap,
      upperSupportConfidenceMap,
      lowerSupportConfidenceMap,
      supportBlendMap,
      totalAttenuationMap,
      lowerPenaltyMap,
      participatingSampleCountMap,
      toneResponseMap,
      troughHalfWidthMap,
      detailHuMap,
      contextHuMap,
      contextBlendFactorMap,
      columnSupportReliabilityMap,
      upperDetailHuMap,
      lowerDetailHuMap,
    },
    diagnostics: {
      enabled: true,
      usedAsOutput,
      acceptedByLowerBandTolerance,
      acceptedByToothBandTolerance,
      acceptedByContextTolerance,
      renderSupportMode: 'archGuidedDualLayer',
      pipelineVariant: 'archGuidedSynthetic',
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
      upperSupportDepthFirst8Mm: Array.from(
        (upperSupportDepthMm || selectedDepthMm).subarray(
          0,
          Math.min(8, (upperSupportDepthMm || selectedDepthMm).length)
        )
      ).map(value => Math.round(Number(value) * 1000) / 1000),
      lowerSupportDepthFirst8Mm: Array.from(
        (lowerSupportDepthMm || selectedDepthMm).subarray(
          0,
          Math.min(8, (lowerSupportDepthMm || selectedDepthMm).length)
        )
      ).map(value => Math.round(Number(value) * 1000) / 1000),
      upperSupportReliabilityP50:
        upperSupportReliabilityByCol && upperSupportReliabilityByCol.length > 0
          ? percentile(Array.from(upperSupportReliabilityByCol), 0.5)
          : 0,
      lowerSupportReliabilityP50:
        lowerSupportReliabilityByCol && lowerSupportReliabilityByCol.length > 0
          ? percentile(Array.from(lowerSupportReliabilityByCol), 0.5)
          : 0,
      supportBlendRows: {
        upperHoldEnd: supportBlendUpperRepresentativeRow,
        lowerHoldStart: supportBlendLowerRepresentativeRow,
      },
      troughSigmaMm: detailHalfWidthBaseMm,
      approxTroughHalfWidthMm: contextHalfWidthBaseMm,
      lowerPenaltyP50: lowerPenaltySummary?.p50 ?? 0,
      lowerPenaltyP90: lowerPenaltySummary?.p90 ?? 0,
      toneResponseP50: toneResponseSummary?.p50 ?? 0,
      toneResponseP90: toneResponseSummary?.p90 ?? 0,
      detailHuP50: detailHuSummary?.p50 ?? 0,
      contextHuP50: contextHuSummary?.p50 ?? 0,
      contextBlendMean:
        contextBlendFactorCount > 0 ? contextBlendFactorSum / contextBlendFactorCount : 0,
      contextWeightFractionMean:
        contextWeightFractionCount > 0 ? contextWeightFractionSum / contextWeightFractionCount : 0,
      columnSupportReliabilityP50: columnSupportReliabilitySummary?.p50 ?? 0,
      upperDetailHuP50: upperDetailHuSummary?.p50 ?? detailHuSummary?.p50 ?? 0,
      lowerDetailHuP50: lowerDetailHuSummary?.p50 ?? detailHuSummary?.p50 ?? 0,
      detailSampleFractionMean:
        detailSampleFractionCount > 0 ? detailSampleFractionSum / detailSampleFractionCount : 0,
      shadowLiftMean: shadowLiftCount > 0 ? shadowLiftSum / shadowLiftCount : 0,
      attenuationStrength: 0,
      gamma: 1,
      outputHuMin: outputHuFloor,
      outputHuMax: outputHuCeiling,
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

function sortValuesAscending(values: Float32Array, count: number): void {
  for (let i = 1; i < count; i++) {
    const keyValue = values[i];
    let j = i - 1;
    while (j >= 0 && values[j] > keyValue) {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = keyValue;
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

function resolveTopPercentileSampleCount(
  count: number,
  topFraction: number,
  minSamples: number,
  maxSamples: number
): number {
  if (count <= 0) {
    return 0;
  }

  const safeFraction = clampNumber(topFraction, 0.01, 1);
  const boundedMinSamples = Math.max(1, Math.min(count, Math.round(minSamples)));
  const boundedMaxSamples = Math.max(boundedMinSamples, Math.min(count, Math.round(maxSamples)));
  const percentileCount = Math.round(count * safeFraction);
  return Math.max(boundedMinSamples, Math.min(boundedMaxSamples, percentileCount));
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

function clampUnitInterval(value: number): number {
  return clampNumber(value, 0, 1);
}

function medianOfThree(a: number, b: number, c: number): number {
  if (a > b) {
    const temp = a;
    a = b;
    b = temp;
  }
  if (b > c) {
    const temp = b;
    b = c;
    c = temp;
  }
  if (a > b) {
    const temp = a;
    a = b;
    b = temp;
  }
  return b;
}

function computeRunLengthSummary(mask: Uint8Array): {
  runCount: number;
  longestRun: number;
  p50: number;
  p90: number;
} {
  const runLengths: number[] = [];
  let currentRun = 0;

  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      currentRun++;
    } else if (currentRun > 0) {
      runLengths.push(currentRun);
      currentRun = 0;
    }
  }

  if (currentRun > 0) {
    runLengths.push(currentRun);
  }

  if (runLengths.length === 0) {
    return {
      runCount: 0,
      longestRun: 0,
      p50: 0,
      p90: 0,
    };
  }

  return {
    runCount: runLengths.length,
    longestRun: Math.max(...runLengths),
    p50: percentile(runLengths, 0.5),
    p90: percentile(runLengths, 0.9),
  };
}

function getNearestDepthIndex(depthMm: number, depthOffsetsMm: Float32Array): number {
  if (depthOffsetsMm.length <= 0) {
    return -1;
  }
  if (depthOffsetsMm.length === 1) {
    return 0;
  }

  const depthStepMm = Number(depthOffsetsMm[1]) - Number(depthOffsetsMm[0]);
  if (!Number.isFinite(depthStepMm) || Math.abs(depthStepMm) < 1e-6) {
    let bestIndex = 0;
    let bestDistance = Math.abs(depthMm - Number(depthOffsetsMm[0]));
    for (let i = 1; i < depthOffsetsMm.length; i++) {
      const distance = Math.abs(depthMm - Number(depthOffsetsMm[i]));
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  const rawIndex = Math.round((depthMm - Number(depthOffsetsMm[0])) / depthStepMm);
  return Math.max(0, Math.min(depthOffsetsMm.length - 1, rawIndex));
}

function computeSelectedSupportReliabilityByColumn(params: {
  selectedDepthMm: Float32Array;
  reliabilityByColDepth: Float32Array;
  depthOffsetsMm: Float32Array;
  depthSamples: number;
}): Float32Array {
  const { selectedDepthMm, reliabilityByColDepth, depthOffsetsMm, depthSamples } = params;
  const selectedReliabilityByCol = new Float32Array(selectedDepthMm.length);

  for (let col = 0; col < selectedDepthMm.length; col++) {
    const depthIndex = getNearestDepthIndex(Number(selectedDepthMm[col]), depthOffsetsMm);
    if (depthIndex < 0 || depthIndex >= depthSamples) {
      selectedReliabilityByCol[col] = 0;
      continue;
    }
    selectedReliabilityByCol[col] = clampUnitInterval(
      Number(reliabilityByColDepth[col * depthSamples + depthIndex]) || 0
    );
  }

  return selectedReliabilityByCol;
}

function buildSupportDepthEnvelope(params: {
  bestDepthByCol: Float32Array;
  bestReliabilityByCol: Float32Array;
  scoreGapByCol: Float32Array;
  depthHalfRangeMm: number;
  smoothScratch: Float32Array;
}): {
  centerMmByCol: Float32Array;
  halfWidthMmByCol: Float32Array;
  anchorMask: Uint8Array;
  anchorFraction: number;
  usedRelaxedAnchors: boolean;
  halfWidthSummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
} {
  const {
    bestDepthByCol,
    bestReliabilityByCol,
    scoreGapByCol,
    depthHalfRangeMm,
    smoothScratch,
  } = params;
  void smoothScratch;
  const width = bestDepthByCol.length;
  const centerMmByCol = new Float32Array(width);
  const halfWidthMmByCol = new Float32Array(width);
  const anchorMask = new Uint8Array(width);
  const minAnchorCount = Math.max(8, Math.min(width, Math.round(width * 0.08)));

  const applyAnchorThresholds = (reliabilityThreshold: number, scoreGapThreshold: number): number => {
    anchorMask.fill(0);
    let anchorCount = 0;
    for (let col = 0; col < width; col++) {
      const depthMm = Number(bestDepthByCol[col]);
      const reliability = clampUnitInterval(Number(bestReliabilityByCol[col]) || 0);
      const scoreGap = Math.max(0, Number(scoreGapByCol[col]) || 0);
      if (
        Number.isFinite(depthMm) &&
        reliability >= reliabilityThreshold &&
        scoreGap >= scoreGapThreshold
      ) {
        anchorMask[col] = 1;
        anchorCount++;
      }
    }
    return anchorCount;
  };

  let usedRelaxedAnchors = false;
  let anchorCount = applyAnchorThresholds(
    CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeAnchorReliabilityThreshold,
    CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeAnchorScoreGapThreshold
  );

  if (anchorCount < minAnchorCount) {
    usedRelaxedAnchors = true;
    anchorCount = applyAnchorThresholds(
      CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeAnchorReliabilityThreshold - 0.08,
      CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeAnchorScoreGapThreshold - 0.03
    );
  }

  if (anchorCount === 0) {
    for (let col = 0; col < width; col++) {
      if (Number.isFinite(Number(bestDepthByCol[col]))) {
        anchorMask[col] = 1;
        anchorCount++;
      }
    }
  }

  const previousAnchorIndexByCol = new Int32Array(width);
  const nextAnchorIndexByCol = new Int32Array(width);
  let previousAnchor = -1;
  for (let col = 0; col < width; col++) {
    if (anchorMask[col]) {
      previousAnchor = col;
    }
    previousAnchorIndexByCol[col] = previousAnchor;
  }
  let nextAnchor = -1;
  for (let col = width - 1; col >= 0; col--) {
    if (anchorMask[col]) {
      nextAnchor = col;
    }
    nextAnchorIndexByCol[col] = nextAnchor;
  }

  for (let col = 0; col < width; col++) {
    if (anchorMask[col]) {
      centerMmByCol[col] = Number(bestDepthByCol[col]);
    } else {
      const leftIndex = previousAnchorIndexByCol[col];
      const rightIndex = nextAnchorIndexByCol[col];
      if (leftIndex >= 0 && rightIndex >= 0 && leftIndex !== rightIndex) {
        const t = (col - leftIndex) / Math.max(1, rightIndex - leftIndex);
        centerMmByCol[col] =
          Number(bestDepthByCol[leftIndex]) * (1 - t) + Number(bestDepthByCol[rightIndex]) * t;
      } else if (leftIndex >= 0) {
        centerMmByCol[col] = Number(bestDepthByCol[leftIndex]);
      } else if (rightIndex >= 0) {
        centerMmByCol[col] = Number(bestDepthByCol[rightIndex]);
      } else {
        centerMmByCol[col] = Number(bestDepthByCol[col]) || 0;
      }
    }
  }

  // Preserve the raw support envelope so posterior curvature is not flattened.

  for (let col = 0; col < width; col++) {
    const centerMm = clampNumber(
      Number(centerMmByCol[col]) || 0,
      -depthHalfRangeMm,
      depthHalfRangeMm
    );
    centerMmByCol[col] = centerMm;
    const reliability = clampUnitInterval(Number(bestReliabilityByCol[col]) || 0);
    const scoreGap = Math.max(0, Number(scoreGapByCol[col]) || 0);
    const gapConfidence = clampUnitInterval(
      scoreGap /
        Math.max(0.08, CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeAnchorScoreGapThreshold * 1.8)
    );
    const confidence = clampUnitInterval(reliability * 0.7 + gapConfidence * 0.3);
    const relaxedWidthBoost = usedRelaxedAnchors ? 0.12 : 0;
    const anchorWidthBias = anchorMask[col] ? -0.08 : 0.1;
    halfWidthMmByCol[col] = clampNumber(
      CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeHalfWidthMaxMm -
        confidence * 0.68 +
        relaxedWidthBoost +
        anchorWidthBias,
      CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeHalfWidthMinMm,
      CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeHalfWidthMaxMm
    );
  }

  return {
    centerMmByCol,
    halfWidthMmByCol,
    anchorMask,
    anchorFraction: width > 0 ? anchorCount / width : 0,
    usedRelaxedAnchors,
    halfWidthSummary: summarizeFiniteFloat32Buffer(halfWidthMmByCol),
  };
}

function applySupportDepthEnvelopeScorePenalty(params: {
  scoreByColDepth: Float32Array;
  bestReliabilityByCol: Float32Array;
  centerMmByCol: Float32Array;
  halfWidthMmByCol: Float32Array;
  depthOffsetsMm: Float32Array;
  panoWidth: number;
  depthSamples: number;
}): {
  adjustedScoreByColDepth: Float32Array;
  penalizedFraction: number;
  penaltySummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
} {
  const {
    scoreByColDepth,
    bestReliabilityByCol,
    centerMmByCol,
    halfWidthMmByCol,
    depthOffsetsMm,
    panoWidth,
    depthSamples,
  } = params;
  const adjustedScoreByColDepth = new Float32Array(scoreByColDepth);
  const penaltyValues = new Float32Array(scoreByColDepth.length);
  let penalizedCount = 0;

  for (let col = 0; col < panoWidth; col++) {
    const centerMm = Number(centerMmByCol[col]) || 0;
    const halfWidthMm = Math.max(0.35, Number(halfWidthMmByCol[col]) || 0.35);
    const reliability = clampUnitInterval(Number(bestReliabilityByCol[col]) || 0);
    for (let depth = 0; depth < depthSamples; depth++) {
      const scoreIndex = col * depthSamples + depth;
      const score = Number(scoreByColDepth[scoreIndex]);
      if (!Number.isFinite(score)) {
        continue;
      }
      const depthMm = Number(depthOffsetsMm[depth]);
      const exceedMm = Math.abs(depthMm - centerMm) - halfWidthMm;
      if (exceedMm <= 0) {
        continue;
      }
      const exceedNorm = exceedMm / Math.max(0.35, halfWidthMm);
      let penalty =
        CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopePenaltyScale *
        (1 + (1 - reliability) * 0.55) *
        exceedNorm *
        exceedNorm;
      if (exceedMm > halfWidthMm * 1.4) {
        penalty += 0.18;
      }
      adjustedScoreByColDepth[scoreIndex] = score - penalty;
      penaltyValues[scoreIndex] = penalty;
      penalizedCount++;
    }
  }

  return {
    adjustedScoreByColDepth,
    penalizedFraction:
      adjustedScoreByColDepth.length > 0 ? penalizedCount / adjustedScoreByColDepth.length : 0,
    penaltySummary: summarizeFiniteFloat32Buffer(penaltyValues),
  };
}

function stabilizeSupportDepthPath(params: {
  selectedDepthMm: Float32Array;
  selectedReliabilityByCol: Float32Array;
  reliabilityByColDepth: Float32Array;
  depthOffsetsMm: Float32Array;
  depthSamples: number;
  smoothScratch: Float32Array;
  depthHalfRangeMm: number;
  bestDepthByCol?: Float32Array;
  scoreGapByCol?: Float32Array;
  lowReliabilityThreshold?: number;
  jumpMmThreshold?: number;
  repairBlend?: number;
  edgeShelfMarginMm?: number;
  scoreGapThreshold?: number;
  bestDepthDriftMmThreshold?: number;
  smoothPasses?: number;
  envelopeCenterMmByCol?: Float32Array;
  envelopeHalfWidthMmByCol?: Float32Array;
  envelopeRepairBlend?: number;
}): {
  selectedReliabilityByCol: Float32Array;
  unstableMask: Uint8Array;
  unreliableFraction: number;
  runCount: number;
  longestRun: number;
  runP50: number;
  runP90: number;
  pathJumpP95Mm: number;
  pathJumpMaxMm: number;
  edgeShelfFraction: number;
  repairedColumnCount: number;
  ambiguousFraction: number;
  forcedDriftFraction: number;
  bestDepthDriftP50Mm: number;
  bestDepthDriftP95Mm: number;
  outsideEnvelopeFraction: number;
  envelopeClampFraction: number;
  envelopeDriftP50Mm: number;
  envelopeDriftP95Mm: number;
} {
  const {
    selectedDepthMm,
    selectedReliabilityByCol,
    reliabilityByColDepth,
    depthOffsetsMm,
    depthSamples,
    smoothScratch,
    depthHalfRangeMm,
    bestDepthByCol = new Float32Array(selectedDepthMm.length),
    scoreGapByCol = new Float32Array(selectedDepthMm.length),
    lowReliabilityThreshold = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairLowReliabilityThreshold,
    jumpMmThreshold = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairJumpMm,
    repairBlend = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairBlend,
    edgeShelfMarginMm = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairEdgeShelfMarginMm,
    scoreGapThreshold = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairScoreGapThreshold,
    bestDepthDriftMmThreshold = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportRepairBestDepthDriftMm,
    smoothPasses = 1,
    envelopeCenterMmByCol,
    envelopeHalfWidthMmByCol,
    envelopeRepairBlend = CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEnvelopeRepairBlend,
  } = params;
  void smoothScratch;
  void smoothPasses;

  const width = selectedDepthMm.length;
  const unstableMask = new Uint8Array(width);
  let repairedColumnCount = 0;
  let envelopeClampCount = 0;

  const computeWindowMedian = (col: number, radius: number): number => {
    const windowValues: number[] = [];
    const start = Math.max(0, col - radius);
    const end = Math.min(width - 1, col + radius);
    for (let index = start; index <= end; index++) {
      const value = Number(selectedDepthMm[index]);
      if (Number.isFinite(value)) {
        windowValues.push(value);
      }
    }
    if (windowValues.length === 0) {
      return Number(selectedDepthMm[col]) || 0;
    }
    if (windowValues.length === 1) {
      return windowValues[0];
    }
    if (windowValues.length === 2) {
      return 0.5 * (windowValues[0] + windowValues[1]);
    }
    if (windowValues.length === 3) {
      return medianOfThree(windowValues[0], windowValues[1], windowValues[2]);
    }
    return percentile(windowValues, 0.5);
  };

  for (let pass = 0; pass < 2; pass++) {
    for (let col = 0; col < width; col++) {
      const currentDepthMm = Number(selectedDepthMm[col]);
      const reliability = clampUnitInterval(Number(selectedReliabilityByCol[col]) || 0);
      const prevDepthMm = col > 0 ? Number(selectedDepthMm[col - 1]) : currentDepthMm;
      const nextDepthMm = col + 1 < width ? Number(selectedDepthMm[col + 1]) : currentDepthMm;
      const localMedianMm = computeWindowMedian(col, 2);
      const prevJumpMm = Math.abs(currentDepthMm - prevDepthMm);
      const nextJumpMm = Math.abs(nextDepthMm - currentDepthMm);
      const bestDepthMm = Number(bestDepthByCol[col]);
      const scoreGap = Math.max(0, Number(scoreGapByCol[col]) || 0);
      const bestDepthDriftMm = Number.isFinite(bestDepthMm)
        ? Math.abs(currentDepthMm - bestDepthMm)
        : 0;
      const envelopeCenterMm =
        envelopeCenterMmByCol && col < envelopeCenterMmByCol.length
          ? Number(envelopeCenterMmByCol[col])
          : currentDepthMm;
      const envelopeHalfWidthMm =
        envelopeHalfWidthMmByCol && col < envelopeHalfWidthMmByCol.length
          ? Math.max(0.35, Number(envelopeHalfWidthMmByCol[col]) || 0.35)
          : Number.POSITIVE_INFINITY;
      const outsideEnvelope = Math.abs(currentDepthMm - envelopeCenterMm) > envelopeHalfWidthMm;
      const edgeShelf = Math.abs(currentDepthMm) >= depthHalfRangeMm - edgeShelfMarginMm;
      const ambiguousSupport =
        scoreGap < scoreGapThreshold && reliability < Math.min(0.9, lowReliabilityThreshold + 0.18);
      const forcedDrift =
        bestDepthDriftMm > bestDepthDriftMmThreshold &&
        scoreGap < scoreGapThreshold * 2.5 &&
        reliability < 0.72;
      const unstable =
        reliability < lowReliabilityThreshold ||
        prevJumpMm > jumpMmThreshold ||
        nextJumpMm > jumpMmThreshold ||
        ambiguousSupport ||
        forcedDrift ||
        outsideEnvelope ||
        edgeShelf;

      unstableMask[col] = unstable ? 1 : 0;

      if (!unstable) {
        continue;
      }

      const blend = clampNumber(
        repairBlend + Math.max(0, lowReliabilityThreshold - reliability) * 0.5,
        repairBlend,
        0.92
      );
      let repairTargetMm = localMedianMm;
      if (ambiguousSupport && Number.isFinite(bestDepthMm)) {
        repairTargetMm = localMedianMm * 0.7 + bestDepthMm * 0.3;
      }
      if (forcedDrift && Number.isFinite(bestDepthMm)) {
        repairTargetMm = repairTargetMm * 0.55 + bestDepthMm * 0.45;
      }
      if (outsideEnvelope && Number.isFinite(envelopeCenterMm)) {
        repairTargetMm = repairTargetMm * (1 - envelopeRepairBlend) + envelopeCenterMm * envelopeRepairBlend;
      }
      let repairedDepth = currentDepthMm * (1 - blend) + repairTargetMm * blend;
      if (edgeShelf) {
        repairedDepth = clampNumber(
          repairedDepth,
          -depthHalfRangeMm + edgeShelfMarginMm,
          depthHalfRangeMm - edgeShelfMarginMm
        );
      }
      if (Number.isFinite(envelopeCenterMm) && Number.isFinite(envelopeHalfWidthMm)) {
        const clampedByEnvelope = clampNumber(
          repairedDepth,
          envelopeCenterMm - envelopeHalfWidthMm,
          envelopeCenterMm + envelopeHalfWidthMm
        );
        if (Math.abs(clampedByEnvelope - repairedDepth) > 1e-3) {
          envelopeClampCount++;
        }
        repairedDepth = clampedByEnvelope;
      }
      if (Math.abs(repairedDepth - currentDepthMm) > 1e-3) {
        selectedDepthMm[col] = repairedDepth;
        repairedColumnCount++;
      }
    }

    // Preserve the raw selected support path; do not smooth away tooth/root curvature.
  }

  const finalReliabilityByCol = new Float32Array(width);
  let edgeShelfCount = 0;
  let ambiguousCount = 0;
  let forcedDriftCount = 0;
  let outsideEnvelopeCount = 0;
  const jumps: number[] = [];
  const bestDepthDriftsMm: number[] = [];
  const envelopeDriftsMm: number[] = [];

  for (let col = 0; col < width; col++) {
    const depthMm = Number(selectedDepthMm[col]);
    const depthIndex = getNearestDepthIndex(depthMm, depthOffsetsMm);
    const bestDepthMm = Number(bestDepthByCol[col]);
    const scoreGap = Math.max(0, Number(scoreGapByCol[col]) || 0);
    const reliability =
      depthIndex >= 0 && depthIndex < depthSamples
        ? clampUnitInterval(
            Number(reliabilityByColDepth[col * depthSamples + depthIndex]) || 0
          )
        : clampUnitInterval(Number(selectedReliabilityByCol[col]) || 0);
    const prevJumpMm = col > 0 ? Math.abs(depthMm - Number(selectedDepthMm[col - 1])) : 0;
    const nextJumpMm = col + 1 < width ? Math.abs(Number(selectedDepthMm[col + 1]) - depthMm) : 0;
    const bestDepthDriftMm = Number.isFinite(bestDepthMm) ? Math.abs(depthMm - bestDepthMm) : 0;
    const envelopeCenterMm =
      envelopeCenterMmByCol && col < envelopeCenterMmByCol.length
        ? Number(envelopeCenterMmByCol[col])
        : depthMm;
    const envelopeHalfWidthMm =
      envelopeHalfWidthMmByCol && col < envelopeHalfWidthMmByCol.length
        ? Math.max(0.35, Number(envelopeHalfWidthMmByCol[col]) || 0.35)
        : Number.POSITIVE_INFINITY;
    const envelopeDriftMm = Number.isFinite(envelopeCenterMm)
      ? Math.abs(depthMm - envelopeCenterMm)
      : 0;
    const outsideEnvelope = envelopeDriftMm > envelopeHalfWidthMm;
    const ambiguousSupport =
      scoreGap < scoreGapThreshold && reliability < Math.min(0.9, lowReliabilityThreshold + 0.18);
    const forcedDrift =
      bestDepthDriftMm > bestDepthDriftMmThreshold &&
      scoreGap < scoreGapThreshold * 2.5 &&
      reliability < 0.72;
    const edgeShelf = Math.abs(depthMm) >= depthHalfRangeMm - edgeShelfMarginMm;
    let reliabilityPenalty = 1;
    if (ambiguousSupport) {
      reliabilityPenalty *= 0.82;
      ambiguousCount++;
    }
    if (forcedDrift) {
      reliabilityPenalty *= 0.72;
      forcedDriftCount++;
    }
    if (outsideEnvelope) {
      reliabilityPenalty *= 0.7;
      outsideEnvelopeCount++;
    }
    finalReliabilityByCol[col] = reliability;
    finalReliabilityByCol[col] = clampUnitInterval(finalReliabilityByCol[col] * reliabilityPenalty);
    unstableMask[col] =
      finalReliabilityByCol[col] < lowReliabilityThreshold ||
      prevJumpMm > jumpMmThreshold ||
      nextJumpMm > jumpMmThreshold ||
      ambiguousSupport ||
      forcedDrift ||
      outsideEnvelope ||
      edgeShelf
        ? 1
        : 0;

    if (edgeShelf) {
      edgeShelfCount++;
    }
    if (Number.isFinite(bestDepthDriftMm)) {
      bestDepthDriftsMm.push(bestDepthDriftMm);
    }
    if (Number.isFinite(envelopeDriftMm)) {
      envelopeDriftsMm.push(envelopeDriftMm);
    }
    if (col > 0) {
      jumps.push(prevJumpMm);
    }
  }

  const runSummary = computeRunLengthSummary(unstableMask);
  const unreliableCount = unstableMask.reduce((acc, value) => acc + value, 0);

  return {
    selectedReliabilityByCol: finalReliabilityByCol,
    unstableMask,
    unreliableFraction: width > 0 ? unreliableCount / width : 0,
    runCount: runSummary.runCount,
    longestRun: runSummary.longestRun,
    runP50: runSummary.p50,
    runP90: runSummary.p90,
    pathJumpP95Mm: jumps.length > 0 ? percentile(jumps, 0.95) : 0,
    pathJumpMaxMm: jumps.length > 0 ? Math.max(...jumps) : 0,
    edgeShelfFraction: width > 0 ? edgeShelfCount / width : 0,
    repairedColumnCount,
    ambiguousFraction: width > 0 ? ambiguousCount / width : 0,
    forcedDriftFraction: width > 0 ? forcedDriftCount / width : 0,
    bestDepthDriftP50Mm: bestDepthDriftsMm.length > 0 ? percentile(bestDepthDriftsMm, 0.5) : 0,
    bestDepthDriftP95Mm: bestDepthDriftsMm.length > 0 ? percentile(bestDepthDriftsMm, 0.95) : 0,
    outsideEnvelopeFraction: width > 0 ? outsideEnvelopeCount / width : 0,
    envelopeClampFraction: width > 0 ? envelopeClampCount / width : 0,
    envelopeDriftP50Mm: envelopeDriftsMm.length > 0 ? percentile(envelopeDriftsMm, 0.5) : 0,
    envelopeDriftP95Mm: envelopeDriftsMm.length > 0 ? percentile(envelopeDriftsMm, 0.95) : 0,
  };
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
  rowSpacingMm: number,
  minBandBottomRow?: number
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
  const clampedMinBandBottomRow = Number.isFinite(minBandBottomRow)
    ? clampNumber(Number(minBandBottomRow), rowSearchStart, rowSearchEnd)
    : rowSearchStart;

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

    toothBandBottomRows[col] = Math.max(bestRow, clampedMinBandBottomRow);
    columnEdgeProtectThresholds[col] = edgeP5 + Math.max(12, edgeScale * 0.3);
    columnIntensityProtectThresholds[col] = Math.max(columnP7, columnP45 + intensityScale * 0.28);
  }

  if (width > 2) {
    smoothFloatSeries(toothBandBottomRows, width, rowScratchA, 3);
    smoothFloatSeries(toothBandBottomRows, width, rowScratchB, 2);
    for (let col = 0; col < width; col++) {
      toothBandBottomRows[col] = clampNumber(
        toothBandBottomRows[col],
        clampedMinBandBottomRow,
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

  let minDetectedBandBottomRow = Infinity;
  let maxBandBottomRow = -Infinity;
  let bandBottomRowSum = 0;
  for (let col = 0; col < width; col++) {
    const value = Number(toothBandBottomRows[col]);
    if (value < minDetectedBandBottomRow) {
      minDetectedBandBottomRow = value;
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
      min: Number.isFinite(minDetectedBandBottomRow)
        ? Math.round(minDetectedBandBottomRow * 1000) / 1000
        : 0,
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
  void smoothScratch;
  void smoothPasses;
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
        const transitionCost =
          VIRTUAL_PANO_DP_MODEL_PARAMS.jumpCostLinear * jumpMm +
          VIRTUAL_PANO_DP_MODEL_PARAMS.jumpCostExtra *
          Math.max(0, jumpMm - VIRTUAL_PANO_DP_MODEL_PARAMS.jumpSoftThresholdMm);
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

  // Preserve the raw DP-selected path; do not smooth away posterior curvature.

  return selectedDepthMm;
}

function computeVirtualSupportPathFromScores(params: {
  scoreByColDepth: Float32Array;
  reliabilityByColDepth: Float32Array;
  panoWidth: number;
  depthSamples: number;
  depthOffsetsMm: Float32Array;
  candidateCount: number;
  smoothScratch: Float32Array;
  depthHalfRangeMm: number;
  pathSmoothPasses: number;
  lockToBaseDepthMm?: number;
  constrainedFollowHalfWidthMm?: number;
  maxAdjacentDepthDeltaMm?: number;
}): {
  selectedDepthMm: Float32Array;
  bestCandidateDepthMm: Float32Array;
  selectedReliabilityByCol: Float32Array;
  scoreGapByCol: Float32Array;
  bestCandidateReliabilitySummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
  bestCandidateDepthSummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
  selectedReliabilitySummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
  scoreGapSummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
  ambiguousColumnCount: number;
  depthMinMm: number;
  depthMaxMm: number;
  depthStdMm: number;
  pathJumpP95Mm: number;
  pathJumpMaxMm: number;
  supportDepthClampCount: number;
  pathStability: ReturnType<typeof stabilizeSupportDepthPath>;
  supportEnvelope: {
    anchorFraction: number;
    usedRelaxedAnchors: boolean;
    halfWidthSummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
    scorePenaltyFraction: number;
    scorePenaltySummary: ReturnType<typeof summarizeFiniteFloat32Buffer>;
  };
} {
  const {
    scoreByColDepth,
    reliabilityByColDepth,
    panoWidth,
    depthSamples,
    depthOffsetsMm,
    candidateCount,
    smoothScratch,
    depthHalfRangeMm,
    pathSmoothPasses,
    lockToBaseDepthMm,
    constrainedFollowHalfWidthMm,
    maxAdjacentDepthDeltaMm,
  } = params;

  if (Number.isFinite(lockToBaseDepthMm)) {
    const baseDepthMm = Number(lockToBaseDepthMm);
    const followHalfWidthMm = Number.isFinite(constrainedFollowHalfWidthMm)
      ? Math.max(0, Math.min(depthHalfRangeMm, Number(constrainedFollowHalfWidthMm)))
      : 0;
    const adjacentDepthClampMm = Number.isFinite(maxAdjacentDepthDeltaMm)
      ? Math.max(0, Number(maxAdjacentDepthDeltaMm))
      : 0;

    if (followHalfWidthMm > 0) {
      const constrainedDepthsMm = new Float32Array(panoWidth);
      const constrainedBestDepthsMm = new Float32Array(panoWidth);
      const constrainedReliabilityByCol = new Float32Array(panoWidth);
      const constrainedScoreGapByCol = new Float32Array(panoWidth);
      const constrainedEnvelopeHalfWidthMmByCol = new Float32Array(panoWidth);
      const constrainedPenaltyByColDepth = new Float32Array(Math.max(1, panoWidth * depthSamples));
      const constrainedUnstableMask = new Uint8Array(panoWidth);
      const minAllowedDepthMm = Math.max(-depthHalfRangeMm, baseDepthMm - followHalfWidthMm);
      const maxAllowedDepthMm = Math.min(depthHalfRangeMm, baseDepthMm + followHalfWidthMm);
      const continuityPenaltyPerMm =
        CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionContinuityPenaltyPerMm;
      const baseDepthPenaltyPerMm =
        CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionBaseDepthPenaltyPerMm;
      const ambiguityGapThreshold =
        CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionAmbiguityGapThreshold;
      const nearestDepthIndexToBase = (() => {
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;
        for (let depth = 0; depth < depthSamples; depth++) {
          const depthMm = Number(depthOffsetsMm[depth]);
          const distance = Math.abs(depthMm - baseDepthMm);
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestIndex = depth;
          }
        }
        return nearestIndex;
      })();
      const findNearestAllowedDepthIndex = (targetDepthMm: number): number => {
        let nearestIndex = nearestDepthIndexToBase;
        let nearestDistance = Number.POSITIVE_INFINITY;
        for (let depth = 0; depth < depthSamples; depth++) {
          const depthMm = Number(depthOffsetsMm[depth]);
          if (depthMm < minAllowedDepthMm || depthMm > maxAllowedDepthMm) {
            continue;
          }
          const distance = Math.abs(depthMm - targetDepthMm);
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestIndex = depth;
          }
        }
        return nearestIndex;
      };
      const computeConstrainedEffectiveScore = (
        col: number,
        depth: number,
        referenceDepthMm: number
      ): number => {
        const depthMm = Number(depthOffsetsMm[depth]);
        if (depthMm < minAllowedDepthMm || depthMm > maxAllowedDepthMm) {
          return Number.NEGATIVE_INFINITY;
        }

        const scoreIndex = col * depthSamples + depth;
        const score = Number(scoreByColDepth[scoreIndex]);
        if (!Number.isFinite(score)) {
          return Number.NEGATIVE_INFINITY;
        }

        const jumpMm = Math.abs(depthMm - referenceDepthMm);
        let effectiveScore =
          score -
          Math.abs(depthMm - baseDepthMm) * baseDepthPenaltyPerMm -
          jumpMm * continuityPenaltyPerMm;
        if (adjacentDepthClampMm > 0 && jumpMm > adjacentDepthClampMm) {
          effectiveScore -=
            (jumpMm - adjacentDepthClampMm) * continuityPenaltyPerMm * 2.2;
        }
        return effectiveScore;
      };
      let ambiguousColumnCount = 0;
      let depthMinMm = Infinity;
      let depthMaxMm = -Infinity;
      let depthSumMm = 0;
      let supportDepthClampCount = 0;

      constrainedEnvelopeHalfWidthMmByCol.fill(followHalfWidthMm);

      let previousSelectedDepthMm = baseDepthMm;
      for (let col = 0; col < panoWidth; col++) {
        let bestScore = Number.NEGATIVE_INFINITY;
        let secondScore = Number.NEGATIVE_INFINITY;
        let bestReliability = 0;
        let bestDepthIndex = -1;

        for (let depth = 0; depth < depthSamples; depth++) {
          const effectiveScore = computeConstrainedEffectiveScore(col, depth, previousSelectedDepthMm);
          if (!Number.isFinite(effectiveScore)) {
            continue;
          }

          if (effectiveScore > bestScore) {
            secondScore = bestScore;
            bestScore = effectiveScore;
            const scoreIndex = col * depthSamples + depth;
            bestReliability = clampUnitInterval(Number(reliabilityByColDepth[scoreIndex]) || 0);
            bestDepthIndex = depth;
          } else if (effectiveScore > secondScore) {
            secondScore = effectiveScore;
          }
        }

        if (bestDepthIndex < 0) {
          bestDepthIndex = nearestDepthIndexToBase;
          bestScore = 0;
          secondScore = 0;
          bestReliability = 0;
        }

        const selectedDepthMm = Number(depthOffsetsMm[bestDepthIndex]);
        constrainedDepthsMm[col] = selectedDepthMm;
        constrainedBestDepthsMm[col] = selectedDepthMm;
        constrainedReliabilityByCol[col] = bestReliability;
        const gap =
          Number.isFinite(bestScore) && Number.isFinite(secondScore) ? bestScore - secondScore : 0;
        constrainedScoreGapByCol[col] = Number.isFinite(gap) ? gap : 0;
        if (!Number.isFinite(secondScore) || gap < ambiguityGapThreshold) {
          ambiguousColumnCount++;
        }
        previousSelectedDepthMm = selectedDepthMm;
      }

      if (adjacentDepthClampMm > 0 && panoWidth > 1) {
        for (let col = 1; col < panoWidth; col++) {
          const prevDepthMm = Number(constrainedDepthsMm[col - 1]);
          const clampedDepthMm = clampNumber(
            Number(constrainedDepthsMm[col]),
            prevDepthMm - adjacentDepthClampMm,
            prevDepthMm + adjacentDepthClampMm
          );
          constrainedDepthsMm[col] = clampedDepthMm;
          constrainedBestDepthsMm[col] = clampedDepthMm;
        }
        for (let col = panoWidth - 2; col >= 0; col--) {
          const nextDepthMm = Number(constrainedDepthsMm[col + 1]);
          const clampedDepthMm = clampNumber(
            Number(constrainedDepthsMm[col]),
            nextDepthMm - adjacentDepthClampMm,
            nextDepthMm + adjacentDepthClampMm
          );
          constrainedDepthsMm[col] = clampedDepthMm;
          constrainedBestDepthsMm[col] = clampedDepthMm;
        }
      }

      ambiguousColumnCount = 0;
      for (let col = 0; col < panoWidth; col++) {
        const quantizedDepthIndex = findNearestAllowedDepthIndex(Number(constrainedDepthsMm[col]));
        const quantizedDepthMm = Number(depthOffsetsMm[quantizedDepthIndex]);
        constrainedDepthsMm[col] = quantizedDepthMm;
        constrainedBestDepthsMm[col] = quantizedDepthMm;
        const scoreIndex = col * depthSamples + quantizedDepthIndex;
        constrainedReliabilityByCol[col] = clampUnitInterval(
          Number(reliabilityByColDepth[scoreIndex]) || 0
        );

        const referenceDepthMm = col > 0 ? Number(constrainedDepthsMm[col - 1]) : baseDepthMm;
        let bestScore = Number.NEGATIVE_INFINITY;
        let secondScore = Number.NEGATIVE_INFINITY;
        for (let depth = 0; depth < depthSamples; depth++) {
          const effectiveScore = computeConstrainedEffectiveScore(col, depth, referenceDepthMm);
          if (!Number.isFinite(effectiveScore)) {
            continue;
          }

          if (effectiveScore > bestScore) {
            secondScore = bestScore;
            bestScore = effectiveScore;
          } else if (effectiveScore > secondScore) {
            secondScore = effectiveScore;
          }
        }

        const gap =
          Number.isFinite(bestScore) && Number.isFinite(secondScore) ? bestScore - secondScore : 0;
        constrainedScoreGapByCol[col] = Number.isFinite(gap) ? gap : 0;
        if (!Number.isFinite(secondScore) || gap < ambiguityGapThreshold) {
          ambiguousColumnCount++;
        }
      }

      const pathJumps: number[] = [];
      depthMinMm = Infinity;
      depthMaxMm = -Infinity;
      depthSumMm = 0;
      supportDepthClampCount = 0;
      for (let col = 0; col < panoWidth; col++) {
        const selectedDepthMm = Number(constrainedDepthsMm[col]);
        depthMinMm = Math.min(depthMinMm, selectedDepthMm);
        depthMaxMm = Math.max(depthMaxMm, selectedDepthMm);
        depthSumMm += selectedDepthMm;
        if (Math.abs(selectedDepthMm) > depthHalfRangeMm - 0.5) {
          supportDepthClampCount++;
        }
        if (col > 0) {
          pathJumps.push(Math.abs(selectedDepthMm - Number(constrainedDepthsMm[col - 1])));
        }
      }

      const depthMeanMm = panoWidth > 0 ? depthSumMm / panoWidth : 0;
      let depthVarianceMm = 0;
      for (let col = 0; col < panoWidth; col++) {
        const depthDelta = Number(constrainedDepthsMm[col]) - depthMeanMm;
        depthVarianceMm += depthDelta * depthDelta;
      }
      const depthStdMm = panoWidth > 0 ? Math.sqrt(depthVarianceMm / Math.max(1, panoWidth)) : 0;
      const constrainedReliabilitySummary = summarizeFiniteFloat32Buffer(constrainedReliabilityByCol);
      const constrainedDepthSummary = summarizeFiniteFloat32Buffer(constrainedDepthsMm);

      return {
        selectedDepthMm: constrainedDepthsMm,
        bestCandidateDepthMm: constrainedBestDepthsMm,
        selectedReliabilityByCol: constrainedReliabilityByCol,
        scoreGapByCol: constrainedScoreGapByCol,
        bestCandidateReliabilitySummary: constrainedReliabilitySummary,
        bestCandidateDepthSummary: constrainedDepthSummary,
        selectedReliabilitySummary: constrainedReliabilitySummary,
        scoreGapSummary: summarizeFiniteFloat32Buffer(constrainedScoreGapByCol),
        ambiguousColumnCount,
        depthMinMm: Number.isFinite(depthMinMm) ? depthMinMm : baseDepthMm,
        depthMaxMm: Number.isFinite(depthMaxMm) ? depthMaxMm : baseDepthMm,
        depthStdMm,
        pathJumpP95Mm: pathJumps.length > 0 ? percentile(pathJumps, 0.95) : 0,
        pathJumpMaxMm: pathJumps.length > 0 ? Math.max(...pathJumps) : 0,
        supportDepthClampCount,
        pathStability: {
          selectedReliabilityByCol: constrainedReliabilityByCol,
          unstableMask: constrainedUnstableMask,
          unreliableFraction: 0,
          runCount: 0,
          longestRun: 0,
          runP50: 0,
          runP90: 0,
          pathJumpP95Mm: pathJumps.length > 0 ? percentile(pathJumps, 0.95) : 0,
          pathJumpMaxMm: pathJumps.length > 0 ? Math.max(...pathJumps) : 0,
          edgeShelfFraction: 0,
          repairedColumnCount: 0,
          ambiguousFraction: panoWidth > 0 ? ambiguousColumnCount / panoWidth : 0,
          forcedDriftFraction: 0,
          bestDepthDriftP50Mm: 0,
          bestDepthDriftP95Mm: 0,
          outsideEnvelopeFraction: 0,
          envelopeClampFraction: 0,
          envelopeDriftP50Mm: 0,
          envelopeDriftP95Mm: 0,
        },
        supportEnvelope: {
          anchorFraction: panoWidth > 0 ? 1 : 0,
          usedRelaxedAnchors: false,
          halfWidthSummary: summarizeFiniteFloat32Buffer(constrainedEnvelopeHalfWidthMmByCol),
          scorePenaltyFraction: 0,
          scorePenaltySummary: summarizeFiniteFloat32Buffer(constrainedPenaltyByColDepth),
        },
      };
    }

    const lockedDepthMm = Number(lockToBaseDepthMm);
    const lockedDepthsMm = new Float32Array(panoWidth);
    const lockedBestDepthsMm = new Float32Array(panoWidth);
    const lockedReliabilityByCol = new Float32Array(panoWidth);
    const lockedScoreGapByCol = new Float32Array(panoWidth);
    const lockedEnvelopeHalfWidthMmByCol = new Float32Array(panoWidth);
    const lockedPenaltyByColDepth = new Float32Array(Math.max(1, panoWidth * depthSamples));
    const lockedUnstableMask = new Uint8Array(panoWidth);
    lockedDepthsMm.fill(lockedDepthMm);
    lockedBestDepthsMm.fill(lockedDepthMm);
    lockedReliabilityByCol.fill(1);
    lockedScoreGapByCol.fill(1);
    lockedEnvelopeHalfWidthMmByCol.fill(depthHalfRangeMm);
    const lockedReliabilitySummary = summarizeFiniteFloat32Buffer(lockedReliabilityByCol);
    const lockedDepthSummary = summarizeFiniteFloat32Buffer(lockedDepthsMm);

    return {
      selectedDepthMm: lockedDepthsMm,
      bestCandidateDepthMm: lockedBestDepthsMm,
      selectedReliabilityByCol: lockedReliabilityByCol,
      scoreGapByCol: lockedScoreGapByCol,
      bestCandidateReliabilitySummary: lockedReliabilitySummary,
      bestCandidateDepthSummary: lockedDepthSummary,
      selectedReliabilitySummary: lockedReliabilitySummary,
      scoreGapSummary: summarizeFiniteFloat32Buffer(lockedScoreGapByCol),
      ambiguousColumnCount: 0,
      depthMinMm: lockedDepthMm,
      depthMaxMm: lockedDepthMm,
      depthStdMm: 0,
      pathJumpP95Mm: 0,
      pathJumpMaxMm: 0,
      supportDepthClampCount:
        Math.abs(lockedDepthMm) > depthHalfRangeMm - 0.5 ? panoWidth : 0,
      pathStability: {
        selectedReliabilityByCol: lockedReliabilityByCol,
        unstableMask: lockedUnstableMask,
        unreliableFraction: 0,
        runCount: 0,
        longestRun: 0,
        runP50: 0,
        runP90: 0,
        pathJumpP95Mm: 0,
        pathJumpMaxMm: 0,
        edgeShelfFraction: 0,
        repairedColumnCount: 0,
        ambiguousFraction: 0,
        forcedDriftFraction: 0,
        bestDepthDriftP50Mm: 0,
        bestDepthDriftP95Mm: 0,
        outsideEnvelopeFraction: 0,
        envelopeClampFraction: 0,
        envelopeDriftP50Mm: 0,
        envelopeDriftP95Mm: 0,
      },
      supportEnvelope: {
        anchorFraction: panoWidth > 0 ? 1 : 0,
        usedRelaxedAnchors: false,
        halfWidthSummary: summarizeFiniteFloat32Buffer(lockedEnvelopeHalfWidthMmByCol),
        scorePenaltyFraction: 0,
        scorePenaltySummary: summarizeFiniteFloat32Buffer(lockedPenaltyByColDepth),
      },
    };
  }

  const columnBestScore = new Float32Array(panoWidth);
  const columnSecondScore = new Float32Array(panoWidth);
  const columnBestReliability = new Float32Array(panoWidth);
  const columnBestDepthIndex = new Int16Array(panoWidth);
  columnBestScore.fill(Number.NEGATIVE_INFINITY);
  columnSecondScore.fill(Number.NEGATIVE_INFINITY);
  columnBestDepthIndex.fill(-1);

  for (let col = 0; col < panoWidth; col++) {
    for (let depth = 0; depth < depthSamples; depth++) {
      const scoreIndex = col * depthSamples + depth;
      const score = Number(scoreByColDepth[scoreIndex]);
      if (!Number.isFinite(score)) {
        continue;
      }
      if (score > Number(columnBestScore[col])) {
        columnSecondScore[col] = columnBestScore[col];
        columnBestScore[col] = score;
        columnBestReliability[col] = clampUnitInterval(
          Number(reliabilityByColDepth[scoreIndex]) || 0
        );
        columnBestDepthIndex[col] = depth;
      } else if (score > Number(columnSecondScore[col])) {
        columnSecondScore[col] = score;
      }
    }
  }

  const bestCandidateDepthMm = new Float32Array(panoWidth);
  const scoreGapByCol = new Float32Array(panoWidth);
  let ambiguousColumnCount = 0;
  for (let col = 0; col < panoWidth; col++) {
    const depthIndex = Number(columnBestDepthIndex[col]);
    bestCandidateDepthMm[col] =
      depthIndex >= 0 && depthIndex < depthSamples ? Number(depthOffsetsMm[depthIndex]) : 0;
    const bestScore = Number(columnBestScore[col]);
    const secondScore = Number(columnSecondScore[col]);
    const gap =
      Number.isFinite(bestScore) && Number.isFinite(secondScore) ? bestScore - secondScore : 0;
    scoreGapByCol[col] = Number.isFinite(gap) ? gap : 0;
    if (!Number.isFinite(secondScore) || gap < 0.08) {
      ambiguousColumnCount++;
    }
  }
  const supportDepthEnvelope = buildSupportDepthEnvelope({
    bestDepthByCol: bestCandidateDepthMm,
    bestReliabilityByCol: columnBestReliability,
    scoreGapByCol,
    depthHalfRangeMm,
    smoothScratch,
  });
  const envelopeAdjustedScores = applySupportDepthEnvelopeScorePenalty({
    scoreByColDepth,
    bestReliabilityByCol: columnBestReliability,
    centerMmByCol: supportDepthEnvelope.centerMmByCol,
    halfWidthMmByCol: supportDepthEnvelope.halfWidthMmByCol,
    depthOffsetsMm,
    panoWidth,
    depthSamples,
  });
  const selectedDepthMm = runBandDPOptimization(
    envelopeAdjustedScores.adjustedScoreByColDepth,
    panoWidth,
    depthSamples,
    depthOffsetsMm,
    candidateCount,
    smoothScratch,
    pathSmoothPasses
  );

  const selectedReliabilityByColInitial = computeSelectedSupportReliabilityByColumn({
    selectedDepthMm,
    reliabilityByColDepth,
    depthOffsetsMm,
    depthSamples,
  });
  const pathStability = stabilizeSupportDepthPath({
    selectedDepthMm,
    selectedReliabilityByCol: selectedReliabilityByColInitial,
    reliabilityByColDepth,
    depthOffsetsMm,
    depthSamples,
    smoothScratch,
    depthHalfRangeMm,
    bestDepthByCol: bestCandidateDepthMm,
    scoreGapByCol,
    smoothPasses: Math.max(1, pathSmoothPasses - 1),
    envelopeCenterMmByCol: supportDepthEnvelope.centerMmByCol,
    envelopeHalfWidthMmByCol: supportDepthEnvelope.halfWidthMmByCol,
  });
  const selectedReliabilityByCol = computeSelectedSupportReliabilityByColumn({
    selectedDepthMm,
    reliabilityByColDepth,
    depthOffsetsMm,
    depthSamples,
  });

  let depthMinMm = Infinity;
  let depthMaxMm = -Infinity;
  let depthSumMm = 0;
  const pathJumps: number[] = [];
  let supportDepthClampCount = 0;
  for (let col = 0; col < panoWidth; col++) {
    const depthMm = Number(selectedDepthMm[col]);
    depthMinMm = Math.min(depthMinMm, depthMm);
    depthMaxMm = Math.max(depthMaxMm, depthMm);
    depthSumMm += depthMm;
    if (Math.abs(depthMm) > depthHalfRangeMm - 0.5) {
      supportDepthClampCount++;
    }
    if (col > 0) {
      pathJumps.push(Math.abs(depthMm - Number(selectedDepthMm[col - 1])));
    }
  }
  const depthMeanMm = panoWidth > 0 ? depthSumMm / panoWidth : 0;
  let depthVarianceMm = 0;
  for (let col = 0; col < panoWidth; col++) {
    const depthDelta = Number(selectedDepthMm[col]) - depthMeanMm;
    depthVarianceMm += depthDelta * depthDelta;
  }
  const depthStdMm = panoWidth > 0 ? Math.sqrt(depthVarianceMm / Math.max(1, panoWidth)) : 0;

  return {
    selectedDepthMm,
    bestCandidateDepthMm,
    selectedReliabilityByCol,
    scoreGapByCol,
    bestCandidateReliabilitySummary: summarizeFiniteFloat32Buffer(columnBestReliability),
    bestCandidateDepthSummary: summarizeFiniteFloat32Buffer(bestCandidateDepthMm),
    selectedReliabilitySummary: summarizeFiniteFloat32Buffer(selectedReliabilityByCol),
    scoreGapSummary: summarizeFiniteFloat32Buffer(scoreGapByCol),
    ambiguousColumnCount,
    depthMinMm: Number.isFinite(depthMinMm) ? depthMinMm : 0,
    depthMaxMm: Number.isFinite(depthMaxMm) ? depthMaxMm : 0,
    depthStdMm,
    pathJumpP95Mm: pathJumps.length > 0 ? percentile(pathJumps, 0.95) : 0,
    pathJumpMaxMm: pathJumps.length > 0 ? Math.max(...pathJumps) : 0,
    supportDepthClampCount,
    pathStability,
    supportEnvelope: {
      anchorFraction: supportDepthEnvelope.anchorFraction,
      usedRelaxedAnchors: supportDepthEnvelope.usedRelaxedAnchors,
      halfWidthSummary: supportDepthEnvelope.halfWidthSummary,
      scorePenaltyFraction: envelopeAdjustedScores.penalizedFraction,
      scorePenaltySummary: envelopeAdjustedScores.penaltySummary,
    },
  };
}

function generatePanorama(
  input: CPRWorkerInput,
  options?: {
    requestedReconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
    gpuFallbackCause?: 'gpu-render-failed' | 'gpu-phase2-gate-failed' | null;
  }
): {
  pixelData: Float32Array;
  debugMaps?: GpuPanoDebugMaps;
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
  const lockVerticalCenterOffsets = rigidVerticalSliceMode || enableVirtualPanoRender;
  const slabBaseWeights = new Float32Array(slabSampleCount);
  const slabCenterIndex = (slabSampleCount - 1) * 0.5;
  for (let s = 0; s < slabSampleCount; s++) {
    const slabOffset = slabSampleCount > 1 ? -slabHalfThicknessMm + s * slabStepMm : 0;
    slabBaseWeights[s] =
      slabSampleCount > 1 ? Math.exp(-(slabOffset * slabOffset) / focalTroughSigmaSq2) : 1;
  }
  const adaptiveCenterSearchHalfRangeMm = lockVerticalCenterOffsets
    ? 0
    : Math.min(
      enableVirtualPanoRender ? 3.6 : isMeanAggregation ? 5.4 : 4.2,
      Math.max(
        enableVirtualPanoRender ? 1.2 : isMeanAggregation ? 1.6 : 1.2,
        effectiveVerticalHalfMm * (enableVirtualPanoRender ? 0.22 : isMeanAggregation ? 0.34 : 0.24)
      )
    );
  const adaptiveCandidateCount = lockVerticalCenterOffsets
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
  const adaptiveCenterSmoothingRadiusCols = lockVerticalCenterOffsets
    ? 0
    : Math.max(8, Math.min(18, Math.round(panoWidth / 40)));
  const adaptiveCenterGlobalBlend = lockVerticalCenterOffsets
    ? 0
    : enableVirtualPanoRender
      ? 0.84
      : isMeanAggregation
        ? 0.72
        : 0.64;
  const baseAdaptiveCenterMaxDeviationMm = lockVerticalCenterOffsets
    ? 0
    : Math.min(
      enableVirtualPanoRender ? 2.6 : isMeanAggregation ? 4.2 : 3.4,
      Math.max(
        enableVirtualPanoRender ? 1.0 : isMeanAggregation ? 1.6 : 1.2,
        effectiveVerticalHalfMm * (enableVirtualPanoRender ? 0.16 : isMeanAggregation ? 0.26 : 0.2)
      )
    );
  const adaptiveCenterMaxAdjacentDeltaMm = lockVerticalCenterOffsets
    ? 0
    : Math.max(
      minVolumeSpacingMm * (enableVirtualPanoRender ? 0.7 : 0.9),
      Math.min(
        enableVirtualPanoRender ? 0.38 : isMeanAggregation ? 0.55 : 0.42,
        minVolumeSpacingMm * (enableVirtualPanoRender ? 1.6 : isMeanAggregation ? 2.4 : 1.8)
      )
    );
  const adaptiveContinuityPenaltyScale = isMeanAggregation ? 10 : 8;
  const profileSampleBuffer = new Float32Array(adaptiveProfileSampleCount);
  const profileSampleScratch = new Float32Array(adaptiveProfileSampleCount);
  const finalRenderProfilePeakValues = new Float32Array(panoWidth);
  const finalRenderProfileFloorValues = new Float32Array(panoWidth);
  const profilePeakRowByCol = new Int16Array(panoWidth);
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
    if (lockVerticalCenterOffsets) {
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

  if (!lockVerticalCenterOffsets && panoWidth > 2) {
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

  if (!lockVerticalCenterOffsets) {
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

  if (!lockVerticalCenterOffsets && panoWidth > 1) {
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
  let adaptiveToothCenterRow = panoCenterRow;
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
    const fallbackPeakSampleIndex = Math.max(
      0,
      Math.min(adaptiveProfileSampleCount - 1, Math.floor(adaptiveProfileSampleCount / 2))
    );
    const peakSearchStart = Math.max(1, Math.floor(adaptiveProfileSampleCount * 0.14));
    const peakSearchEnd = Math.max(
      peakSearchStart,
      Math.min(adaptiveProfileSampleCount - 2, Math.ceil(adaptiveProfileSampleCount * 0.82))
    );
    let peakSampleIndex = fallbackPeakSampleIndex;
    let peakSampleValue = Number.NEGATIVE_INFINITY;
    for (let sampleIndex = peakSearchStart; sampleIndex <= peakSearchEnd; sampleIndex++) {
      const value = Number(profileSampleBuffer[sampleIndex]);
      if (!Number.isFinite(value)) {
        continue;
      }
      if (value > peakSampleValue) {
        peakSampleValue = value;
        peakSampleIndex = sampleIndex;
      }
    }
    if (!Number.isFinite(peakSampleValue)) {
      peakSampleValue = Number(profileSampleBuffer[fallbackPeakSampleIndex]);
      peakSampleIndex = fallbackPeakSampleIndex;
    }
    const peakFraction =
      adaptiveProfileSampleCount <= 1
        ? 0.5
        : peakSampleIndex / Math.max(1, adaptiveProfileSampleCount - 1);
    const peakRow = Math.max(0, Math.min(panoHeight - 1, Math.round(peakFraction * (panoHeight - 1))));
    profilePeakRowByCol[col] = peakRow;
    centerRowByCol[col] = peakRow;
    halfHeightByCol[col] = Math.max(1, Math.max(peakRow, panoHeight - 1 - peakRow));
    finalRenderProfilePeakValues[col] = Number.isFinite(peakSampleValue)
      ? peakSampleValue
      : Number(profileSampleBuffer[fallbackPeakSampleIndex]);
    const profileValues = Array.from(profileSampleBuffer.subarray(0, adaptiveProfileSampleCount));
    finalRenderProfileFloorValues[col] = percentile(profileValues, 0.25);
  }
  if (panoWidth > 0) {
    adaptiveToothCenterRow = clampNumber(
      percentile(Array.from(profilePeakRowByCol), 0.5),
      0,
      panoHeight - 1
    );
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
  let finalDebugMaps: GpuPanoDebugMaps | undefined;
  let lowerArchAnchorRowForBackgroundSuppression: number | undefined;
  if (shouldComputeVirtualPano) {
    const virtualPanoDepthHalfRangeMm = 5.5;
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
    const virtualUpperScoreByColDepth = new Float32Array(panoWidth * virtualPanoDepthSamples);
    const virtualLowerScoreByColDepth = new Float32Array(panoWidth * virtualPanoDepthSamples);
    const virtualUpperSupportReliabilityByColDepth = new Float32Array(
      panoWidth * virtualPanoDepthSamples
    );
    const virtualLowerSupportReliabilityByColDepth = new Float32Array(
      panoWidth * virtualPanoDepthSamples
    );
    const virtualSupportTiltByCol: Float32Array | undefined = undefined;
    const virtualSoftThresholdByCol = new Float32Array(panoWidth);
    const virtualHardThresholdByCol = new Float32Array(panoWidth);
    const virtualGradCapByCol = new Float32Array(panoWidth);
    const virtualThresholdScratch = new Float32Array(panoWidth);
    const virtualDpSmoothScratch = new Float32Array(panoWidth);

    const rowFromNormalizedOffset = (yNorm: number): number => {
      const row = Math.round(adaptiveToothCenterRow + yNorm * Math.max(1, adaptiveToothCenterRow));
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

    // Depth-only prefilter suppresses quantum mottle without washing out lateral tooth edges.
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
        toothThresholdSamples.length > 0
          ? percentile(toothThresholdSamples, CPU_VIRTUAL_PANO_MODEL_PARAMS.thresholdSoftPercentile)
          : -250;
      virtualHardThresholdByCol[col] =
        toothThresholdSamples.length > 0
          ? percentile(toothThresholdSamples, CPU_VIRTUAL_PANO_MODEL_PARAMS.thresholdHardPercentile)
          : 250;
      virtualGradCapByCol[col] =
        gradientSamples.length > 0
          ? Math.max(1, percentile(gradientSamples, CPU_VIRTUAL_PANO_MODEL_PARAMS.gradCapPercentile))
          : 200;
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
        const crownRootBalance =
          1 - clampNumber(Math.abs(topHardMean - bottomHardMean) * 1.35, 0, 1);
        const depthMm = Number(virtualPanoDepthOffsetsMm[depth]);
        const edgeDistanceMm = virtualPanoDepthHalfRangeMm - Math.abs(depthMm);
        const edgePenalty =
          edgeDistanceMm <= 2.5
            ? CPU_VIRTUAL_PANO_MODEL_PARAMS.edgePenaltyScale * (1 - edgeDistanceMm / 2.5)
            : 0;
        const depthCenterPenalty =
          CPU_VIRTUAL_PANO_MODEL_PARAMS.depthCenterPenaltyScale *
          Math.pow(Math.abs(depthMm) / Math.max(virtualPanoDepthHalfRangeMm, 1e-6), 1.5);
        const crownBiasPenalty =
          Math.max(0, topHardMean - bottomHardMean) *
          CPU_VIRTUAL_PANO_MODEL_PARAMS.crownBiasPenaltyScale;
        const supportReliability = clampUnitInterval(
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * supportMean +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * crownRootBalance +
            (1 - lowMean) * 0.22 +
            Math.min(topHardMean, bottomHardMean) * 0.12
        );
        const supportAmbiguity = clampUnitInterval(Math.abs(topHardMean - bottomHardMean) * 1.15);
        const edgeShelfPenalty = clampUnitInterval(
          edgeDistanceMm <= 1.5 ? 1 - edgeDistanceMm / 1.5 : 0
        );
        const baseScore =
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * supportMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * crownRootBalance -
          CPU_VIRTUAL_PANO_MODEL_PARAMS.lowBandPenaltyWeight * lowMean -
          edgePenalty -
          depthCenterPenalty -
          crownBiasPenalty;
        const stabilityBoost =
          supportReliability * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportReliabilityWeight -
          supportAmbiguity * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportAmbiguityWeight -
          edgeShelfPenalty * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEdgeShelfWeight -
          Math.max(0, 0.35 - supportReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportLowReliabilityPenalty;
        const upperSupportMean = clampUnitInterval(topHardMean * 0.74 + supportMean * 0.26);
        const lowerSupportMean = clampUnitInterval(bottomHardMean * 0.74 + supportMean * 0.26);
        const upperArchBalance = clampUnitInterval(
          1 - Math.max(0, bottomHardMean - topHardMean) * 1.15
        );
        const lowerArchBalance = clampUnitInterval(
          1 - Math.max(0, topHardMean - bottomHardMean) * 1.15
        );
        const upperSupportReliability = clampUnitInterval(
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * upperSupportMean +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean * 0.95 +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * upperArchBalance * 0.82 +
            (1 - lowMean) * 0.18 +
            topHardMean * 0.16
        );
        const lowerSupportReliability = clampUnitInterval(
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * lowerSupportMean +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean * 0.95 +
            CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * lowerArchBalance * 0.82 +
            (1 - lowMean) * 0.14 +
            bottomHardMean * 0.16
        );
        const upperScore =
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * upperSupportMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * upperArchBalance * 0.82 -
          CPU_VIRTUAL_PANO_MODEL_PARAMS.lowBandPenaltyWeight * lowMean * 0.9 -
          edgePenalty -
          depthCenterPenalty * 0.9 -
          Math.max(0, bottomHardMean - topHardMean) * 0.28 +
          upperSupportReliability * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportReliabilityWeight -
          supportAmbiguity * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportAmbiguityWeight * 0.9 -
          edgeShelfPenalty * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEdgeShelfWeight * 0.9 -
          Math.max(0, 0.35 - upperSupportReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportLowReliabilityPenalty;
        const lowerScore =
          CPU_VIRTUAL_PANO_MODEL_PARAMS.supportMeanWeight * lowerSupportMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.gradientWeight * gradMean +
          CPU_VIRTUAL_PANO_MODEL_PARAMS.balanceWeight * lowerArchBalance * 0.82 -
          CPU_VIRTUAL_PANO_MODEL_PARAMS.lowBandPenaltyWeight * lowMean * 1.08 -
          edgePenalty -
          depthCenterPenalty * 0.9 -
          Math.max(0, topHardMean - bottomHardMean) * 0.28 +
          lowerSupportReliability * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportReliabilityWeight -
          supportAmbiguity * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportAmbiguityWeight * 0.9 -
          edgeShelfPenalty * CPU_VIRTUAL_PANO_MODEL_PARAMS.supportEdgeShelfWeight * 0.9 -
          Math.max(0, 0.35 - lowerSupportReliability) *
            CPU_VIRTUAL_PANO_MODEL_PARAMS.supportLowReliabilityPenalty;
        const scoreIndex = col * virtualPanoDepthSamples + depth;
        virtualUpperScoreByColDepth[scoreIndex] = upperScore;
        virtualLowerScoreByColDepth[scoreIndex] = lowerScore;
        virtualUpperSupportReliabilityByColDepth[scoreIndex] = upperSupportReliability;
        virtualLowerSupportReliabilityByColDepth[scoreIndex] = lowerSupportReliability;
      }
    }

    const manualSupportBaseDepthMm = enableVirtualPanoRender ? 0 : undefined;
    const manualSupportFollowHalfWidthMm = enableVirtualPanoRender
      ? CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionDepthFollowHalfWidthMm
      : undefined;
    const manualSupportMaxAdjacentDepthDeltaMm = enableVirtualPanoRender
      ? CPU_VIRTUAL_PANO_MODEL_PARAMS.dualArchProjectionMaxAdjacentDepthDeltaMm
      : undefined;
    const virtualUpperSupportPath = computeVirtualSupportPathFromScores({
      scoreByColDepth: virtualUpperScoreByColDepth,
      reliabilityByColDepth: virtualUpperSupportReliabilityByColDepth,
      panoWidth,
      depthSamples: virtualPanoDepthSamples,
      depthOffsetsMm: virtualPanoDepthOffsetsMm,
      candidateCount: virtualPanoCandidateCount,
      smoothScratch: virtualDpSmoothScratch,
      depthHalfRangeMm: virtualPanoDepthHalfRangeMm,
      pathSmoothPasses: CPU_VIRTUAL_PANO_MODEL_PARAMS.pathSmoothPasses,
      lockToBaseDepthMm: manualSupportBaseDepthMm,
      constrainedFollowHalfWidthMm: manualSupportFollowHalfWidthMm,
      maxAdjacentDepthDeltaMm: manualSupportMaxAdjacentDepthDeltaMm,
    });
    const virtualLowerSupportPath = computeVirtualSupportPathFromScores({
      scoreByColDepth: virtualLowerScoreByColDepth,
      reliabilityByColDepth: virtualLowerSupportReliabilityByColDepth,
      panoWidth,
      depthSamples: virtualPanoDepthSamples,
      depthOffsetsMm: virtualPanoDepthOffsetsMm,
      candidateCount: virtualPanoCandidateCount,
      smoothScratch: virtualDpSmoothScratch,
      depthHalfRangeMm: virtualPanoDepthHalfRangeMm,
      pathSmoothPasses: CPU_VIRTUAL_PANO_MODEL_PARAMS.pathSmoothPasses,
      lockToBaseDepthMm: manualSupportBaseDepthMm,
      constrainedFollowHalfWidthMm: manualSupportFollowHalfWidthMm,
      maxAdjacentDepthDeltaMm: manualSupportMaxAdjacentDepthDeltaMm,
    });
    const bandLabels = ['upperArch', 'lowerArch'] as const;
    const bandAnchorRows = [
      Math.round((topBandStartRow + topBandEndRow) * 0.5),
      Math.round((bottomBandStartRow + bottomBandEndRow) * 0.5),
    ];
    lowerArchAnchorRowForBackgroundSuppression = bandAnchorRows[1];
    const bandDepthsMm: Float32Array[] = [
      virtualUpperSupportPath.selectedDepthMm,
      virtualLowerSupportPath.selectedDepthMm,
    ];
    let nonCrossingViolations = 0;

    for (let col = 0; col < panoWidth; col++) {
      for (let b = 1; b < bandDepthsMm.length; b++) {
        const prevBandDepth = Number(bandDepthsMm[b - 1][col]);
        let currentDepth = Number(bandDepthsMm[b][col]);

        // Non-crossing: band[b] should not be more than 0.5mm "past" band[b-1]
        if (currentDepth < prevBandDepth - 0.5) {
          nonCrossingViolations++;
          currentDepth = prevBandDepth - 0.5;
        }

        bandDepthsMm[b][col] = currentDepth;
      }
    }

    // Preserve raw inter-band geometry; do not smooth the band paths after solving.

    const virtualMergedDepthMm = new Float32Array(panoWidth);
    for (let col = 0; col < panoWidth; col++) {
      virtualMergedDepthMm[col] =
        0.5 *
        (Number(virtualUpperSupportPath.selectedDepthMm[col]) +
          Number(virtualLowerSupportPath.selectedDepthMm[col]));
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
    const mergedSelectedReliabilitySummary = summarizeFiniteFloat32Buffer(
      new Float32Array(
        Array.from({ length: panoWidth }, (_, col) =>
          0.5 *
          (Number(virtualUpperSupportPath.selectedReliabilityByCol[col]) +
            Number(virtualLowerSupportPath.selectedReliabilityByCol[col]))
        )
      )
    );
    const mergedBestCandidateReliabilitySummary = summarizeFiniteFloat32Buffer(
      new Float32Array(
        Array.from({ length: panoWidth }, (_, col) =>
          0.5 *
          ((virtualUpperSupportPath.bestCandidateReliabilitySummary?.p50 ?? 0) +
            (virtualLowerSupportPath.bestCandidateReliabilitySummary?.p50 ?? 0))
        )
      )
    );
    const mergedBestCandidateDepthSummary = summarizeFiniteFloat32Buffer(virtualMergedDepthMm);
    const mergedScoreGapSummary = summarizeFiniteFloat32Buffer(
      new Float32Array(
        Array.from({ length: panoWidth }, (_, col) =>
          Math.min(
            Number(virtualUpperSupportPath.scoreGapByCol[col]),
            Number(virtualLowerSupportPath.scoreGapByCol[col])
          )
        )
      )
    );

    virtualPanoPhase12Diagnostics = {
      enabled: true,
      phase: 2,
      reconstructionMode,
      analysisCenterRow: adaptiveToothCenterRow,
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
      renderModel: {
        preferred: 'dualArchDirectProjection',
        supportPathFallbackAvailable: allowLegacyFallback,
      },
      supportSurface: {
        depthMinMm: Math.min(virtualUpperSupportPath.depthMinMm, virtualLowerSupportPath.depthMinMm),
        depthMaxMm: Math.max(virtualUpperSupportPath.depthMaxMm, virtualLowerSupportPath.depthMaxMm),
        depthStdMm: Math.max(virtualUpperSupportPath.depthStdMm, virtualLowerSupportPath.depthStdMm),
        pathJumpP95Mm: Math.max(
          virtualUpperSupportPath.pathStability.pathJumpP95Mm || virtualUpperSupportPath.pathJumpP95Mm,
          virtualLowerSupportPath.pathStability.pathJumpP95Mm || virtualLowerSupportPath.pathJumpP95Mm
        ),
        pathJumpMaxMm: Math.max(
          virtualUpperSupportPath.pathStability.pathJumpMaxMm || virtualUpperSupportPath.pathJumpMaxMm,
          virtualLowerSupportPath.pathStability.pathJumpMaxMm || virtualLowerSupportPath.pathJumpMaxMm
        ),
        supportDepthClampFraction:
          panoWidth > 0
            ? Math.max(
                virtualUpperSupportPath.supportDepthClampCount,
                virtualLowerSupportPath.supportDepthClampCount
              ) / panoWidth
            : 0,
        supportEdgeShelfFraction: Math.max(
          virtualUpperSupportPath.pathStability.edgeShelfFraction,
          virtualLowerSupportPath.pathStability.edgeShelfFraction
        ),
        reliableColumnFraction:
          panoWidth > 0
            ? 1 -
              Math.max(
                virtualUpperSupportPath.pathStability.unreliableFraction,
                virtualLowerSupportPath.pathStability.unreliableFraction
              )
            : 0,
        unreliableColumnFraction: Math.max(
          virtualUpperSupportPath.pathStability.unreliableFraction,
          virtualLowerSupportPath.pathStability.unreliableFraction
        ),
        unreliableRunCount: Math.max(
          virtualUpperSupportPath.pathStability.runCount,
          virtualLowerSupportPath.pathStability.runCount
        ),
        unreliableRunLongest: Math.max(
          virtualUpperSupportPath.pathStability.longestRun,
          virtualLowerSupportPath.pathStability.longestRun
        ),
        unreliableRunP50: Math.max(
          virtualUpperSupportPath.pathStability.runP50,
          virtualLowerSupportPath.pathStability.runP50
        ),
        unreliableRunP90: Math.max(
          virtualUpperSupportPath.pathStability.runP90,
          virtualLowerSupportPath.pathStability.runP90
        ),
        edgeShelfFraction: Math.max(
          virtualUpperSupportPath.pathStability.edgeShelfFraction,
          virtualLowerSupportPath.pathStability.edgeShelfFraction
        ),
        repairedColumnCount:
          virtualUpperSupportPath.pathStability.repairedColumnCount +
          virtualLowerSupportPath.pathStability.repairedColumnCount,
        forcedDriftFraction: Math.max(
          virtualUpperSupportPath.pathStability.forcedDriftFraction,
          virtualLowerSupportPath.pathStability.forcedDriftFraction
        ),
        outsideEnvelopeFraction: Math.max(
          virtualUpperSupportPath.pathStability.outsideEnvelopeFraction,
          virtualLowerSupportPath.pathStability.outsideEnvelopeFraction
        ),
        envelopeClampFraction: Math.max(
          virtualUpperSupportPath.pathStability.envelopeClampFraction,
          virtualLowerSupportPath.pathStability.envelopeClampFraction
        ),
        bestDepthDriftP50Mm: Math.max(
          virtualUpperSupportPath.pathStability.bestDepthDriftP50Mm,
          virtualLowerSupportPath.pathStability.bestDepthDriftP50Mm
        ),
        bestDepthDriftP95Mm: Math.max(
          virtualUpperSupportPath.pathStability.bestDepthDriftP95Mm,
          virtualLowerSupportPath.pathStability.bestDepthDriftP95Mm
        ),
        envelopeDriftP50Mm: Math.max(
          virtualUpperSupportPath.pathStability.envelopeDriftP50Mm,
          virtualLowerSupportPath.pathStability.envelopeDriftP50Mm
        ),
        envelopeDriftP95Mm: Math.max(
          virtualUpperSupportPath.pathStability.envelopeDriftP95Mm,
          virtualLowerSupportPath.pathStability.envelopeDriftP95Mm
        ),
        selectedReliabilityP10: mergedSelectedReliabilitySummary?.p10 ?? 0,
        selectedReliabilityP50: mergedSelectedReliabilitySummary?.p50 ?? 0,
        selectedReliabilityP90: mergedSelectedReliabilitySummary?.p90 ?? 0,
        bestCandidateReliabilityP10: mergedBestCandidateReliabilitySummary?.p10 ?? 0,
        bestCandidateReliabilityP50: mergedBestCandidateReliabilitySummary?.p50 ?? 0,
        bestCandidateReliabilityP90: mergedBestCandidateReliabilitySummary?.p90 ?? 0,
        bestCandidateDepthP10Mm: mergedBestCandidateDepthSummary?.p10 ?? 0,
        bestCandidateDepthP50Mm: mergedBestCandidateDepthSummary?.p50 ?? 0,
        bestCandidateDepthP90Mm: mergedBestCandidateDepthSummary?.p90 ?? 0,
        scoreGapP10: mergedScoreGapSummary?.p10 ?? 0,
        scoreGapP50: mergedScoreGapSummary?.p50 ?? 0,
        scoreGapP90: mergedScoreGapSummary?.p90 ?? 0,
        ambiguousColumnFraction:
          panoWidth > 0
            ? Math.max(
                virtualUpperSupportPath.ambiguousColumnCount / panoWidth,
                virtualLowerSupportPath.ambiguousColumnCount / panoWidth,
                virtualUpperSupportPath.pathStability.ambiguousFraction,
                virtualLowerSupportPath.pathStability.ambiguousFraction
              )
            : Math.max(
                virtualUpperSupportPath.pathStability.ambiguousFraction,
                virtualLowerSupportPath.pathStability.ambiguousFraction
              ),
        selectedDepthFirst8Mm: Array.from(
          virtualMergedDepthMm.subarray(0, Math.min(8, virtualMergedDepthMm.length))
        ).map(value => Math.round(Number(value) * 1000) / 1000),
      },
      supportModel: {
        bandCount: bandDepthsMm.length,
        bands: bandDiagnostics,
        interBandDeltaMeanMm: Math.round(interBandDeltaMeanMm * 1000) / 1000,
        interBandDeltaMaxMm: Math.round(interBandDeltaMax * 1000) / 1000,
        nonCrossingViolations,
        upperArch: {
          selectedReliabilityP50: virtualUpperSupportPath.selectedReliabilitySummary?.p50 ?? 0,
          ambiguousColumnFraction:
            panoWidth > 0
              ? Math.max(
                  virtualUpperSupportPath.ambiguousColumnCount / panoWidth,
                  virtualUpperSupportPath.pathStability.ambiguousFraction
                )
              : virtualUpperSupportPath.pathStability.ambiguousFraction,
          forcedDriftFraction: virtualUpperSupportPath.pathStability.forcedDriftFraction,
          outsideEnvelopeFraction: virtualUpperSupportPath.pathStability.outsideEnvelopeFraction,
          envelopeClampFraction: virtualUpperSupportPath.pathStability.envelopeClampFraction,
          pathJumpP95Mm:
            virtualUpperSupportPath.pathStability.pathJumpP95Mm || virtualUpperSupportPath.pathJumpP95Mm,
          depthStdMm: virtualUpperSupportPath.depthStdMm,
          envelopeAnchorFraction: virtualUpperSupportPath.supportEnvelope.anchorFraction,
          envelopeUsedRelaxedAnchors: virtualUpperSupportPath.supportEnvelope.usedRelaxedAnchors,
          envelopeHalfWidthP50Mm:
            virtualUpperSupportPath.supportEnvelope.halfWidthSummary?.p50 ?? 0,
          envelopePenaltyP50:
            virtualUpperSupportPath.supportEnvelope.scorePenaltySummary?.p50 ?? 0,
        },
        lowerArch: {
          selectedReliabilityP50: virtualLowerSupportPath.selectedReliabilitySummary?.p50 ?? 0,
          ambiguousColumnFraction:
            panoWidth > 0
              ? Math.max(
                  virtualLowerSupportPath.ambiguousColumnCount / panoWidth,
                  virtualLowerSupportPath.pathStability.ambiguousFraction
                )
              : virtualLowerSupportPath.pathStability.ambiguousFraction,
          forcedDriftFraction: virtualLowerSupportPath.pathStability.forcedDriftFraction,
          outsideEnvelopeFraction: virtualLowerSupportPath.pathStability.outsideEnvelopeFraction,
          envelopeClampFraction: virtualLowerSupportPath.pathStability.envelopeClampFraction,
          pathJumpP95Mm:
            virtualLowerSupportPath.pathStability.pathJumpP95Mm || virtualLowerSupportPath.pathJumpP95Mm,
          depthStdMm: virtualLowerSupportPath.depthStdMm,
          envelopeAnchorFraction: virtualLowerSupportPath.supportEnvelope.anchorFraction,
          envelopeUsedRelaxedAnchors: virtualLowerSupportPath.supportEnvelope.usedRelaxedAnchors,
          envelopeHalfWidthP50Mm:
            virtualLowerSupportPath.supportEnvelope.halfWidthSummary?.p50 ?? 0,
          envelopePenaltyP50:
            virtualLowerSupportPath.supportEnvelope.scorePenaltySummary?.p50 ?? 0,
        },
      },
    };

    if (enableVirtualPanoRender) {
      const directVirtualRender = renderDualArchToothProjectionPano({
        virtualPanoStack,
        panoWidth,
        panoHeight,
        planeSize: virtualPanoPlaneSize,
        panoCenterRow,
        adaptiveToothCenterRow,
        centerRowByCol,
        halfHeightByCol,
        virtualPanoDepthOffsetsMm,
        selectedDepthMm: virtualMergedDepthMm,
        upperSupportDepthMm: virtualUpperSupportPath.selectedDepthMm,
        lowerSupportDepthMm: virtualLowerSupportPath.selectedDepthMm,
        upperSupportAnchorRow: bandAnchorRows[0],
        lowerSupportAnchorRow: bandAnchorRows[1],
        upperSupportReliabilityByCol: virtualUpperSupportPath.selectedReliabilityByCol,
        lowerSupportReliabilityByCol: virtualLowerSupportPath.selectedReliabilityByCol,
        softThresholdByCol: virtualSoftThresholdByCol,
        hardThresholdByCol: virtualHardThresholdByCol,
        supportTiltMmByCol: virtualSupportTiltByCol,
        virtualPanoDepthHalfRangeMm,
      });
      const virtualRender = directVirtualRender;

      virtualPanoRenderDiagnostics = {
        enabled: true,
        analysisCenterRow: adaptiveToothCenterRow,
        rendererVariant: 'dual-arch-tooth-projection',
        preferredRendererVariant: 'dual-arch-tooth-projection',
        fallbackRendererVariant: null,
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

      const keepRejectedVirtualPanoOutput = !virtualRender.diagnostics.usedAsOutput && !allowLegacyFallback;

      if (virtualRender.diagnostics.usedAsOutput || keepRejectedVirtualPanoOutput) {
        virtualPanoAcceptedByGate = virtualRender.diagnostics.usedAsOutput;
        virtualPanoSelectedForOutput = true;
        selectedReconstructionMode = 'virtualPano';
        pixelData.set(virtualRender.pixelData);
        finalDebugMaps = virtualRender.debugMaps;
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

  const activeDualArchRendererVariant = (virtualPanoRenderDiagnostics as {
    rendererVariant?: unknown;
  }).rendererVariant;
  const shouldSkipDualArchPostDenoise =
    activeDualArchRendererVariant === 'dual-arch-tooth-projection' ||
    activeDualArchRendererVariant === 'arch-guided-dual-layer';
  // The direct dual-arch path still needs lower-band cleanup; keep this suppression on.
  const shouldApplyAdaptiveBackgroundSuppression = true;
  const backgroundSuppressionResult = shouldApplyAdaptiveBackgroundSuppression
    ? suppressLowerBackground(
        pixelData,
        panoWidth,
        panoHeight,
        vertStepMm,
        lowerArchAnchorRowForBackgroundSuppression
      )
    : {
        suppressionWeights: new Float32Array(pixelData.length),
        toothBandBottomRows: new Float32Array(panoWidth),
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
      };
  const { toothBandBottomRowStats, backgroundSuppression, suppressionWeights } =
    backgroundSuppressionResult;
  const residualSpeckleClamp = shouldApplyAdaptiveBackgroundSuppression
    ? clampSuppressedBackgroundSpeckles(pixelData, panoWidth, panoHeight, suppressionWeights)
    : {
        applied: false,
        pixelCount: 0,
        meanReductionHu: 0,
        maxReductionHu: 0,
      };

  const denoiseBlend =
    shouldSkipDualArchPostDenoise
      ? 0
      : selectedReconstructionMode === 'virtualPano'
      ? slabHalfThicknessMm <= 1.2
        ? 0.16
        : slabHalfThicknessMm <= 1.7
          ? 0.22
          : 0.24
      : isMeanAggregation
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
  const renderRoute =
    gpuFallbackCause
      ? 'gpu-fallback-to-true-virtual-pano'
      : requestedRenderBackend === 'gpu' && requestedReconstructionMode !== 'legacy'
        ? 'gpu-request-routed-to-true-virtual-pano'
        : 'cpu-direct';
  const rendererFamily =
    selectedReconstructionMode === 'virtualPano'
      ? 'true-virtual-pano'
      : 'cpu-legacy-panorama';
  const diagnosticPayload = {
    renderBackend: 'cpu',
    requestedRenderBackend,
    renderRoute,
    rendererFamily,
    routeIntegrity:
      requestedRenderBackend === 'gpu' && requestedReconstructionMode !== 'legacy'
        ? 'nonlegacy-request-honored-via-cpu-renderer'
        : 'matched-request',
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
      reduction: shouldSkipDualArchPostDenoise
        ? activeDualArchRendererVariant === 'arch-guided-dual-layer'
          ? 'arch-guided-dual-layer'
          : 'dual-arch-direct-weighted-high-band'
        : isMeanAggregation
          ? 'winsorized-weighted-mean'
          : 'weighted-high-band-mean',
    },
    denoise: {
      blend: effectiveDenoiseBlend,
      requestedBlend: denoiseBlend,
      applied: denoiseApplied,
      virtualRenderUsedAsOutput: virtualPanoSelectedForOutput,
      residualSpeckleClamp,
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
      : shouldSkipDualArchPostDenoise
        ? 'disabled-dual-arch-post-denoise'
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
      ...(virtualPanoRenderDiagnostics.enabled === true ? virtualPanoRenderDiagnostics : {}),
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
      attachedMaps: finalDebugMaps
        ? Object.entries(finalDebugMaps)
          .filter(([, value]) => !!value)
          .map(([name]) => name)
        : [],
      attachedByteLength: finalDebugMaps
        ? (finalDebugMaps.rawSupportDepthMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportConfidenceMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportSpreadMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportDensityMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportDenseFractionMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakHuSupportGateMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportDominantPeakOffsetMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportSecondaryPeakOffsetMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakDominanceMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakValidityMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakConflictMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportSecondPeakRatioMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakSeparationMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportPeakAmbiguityMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportScoreGapMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportLocalJumpMap?.byteLength ?? 0) +
        (finalDebugMaps.rawSupportContinuityMap?.byteLength ?? 0) +
        (finalDebugMaps.toothBandPriorMap?.byteLength ?? 0) +
        (finalDebugMaps.dominantDensePeakGateMap?.byteLength ?? 0) +
        (finalDebugMaps.toothBandStructureGuardMap?.byteLength ?? 0) +
        (finalDebugMaps.ambiguousBroadSupportPenaltyGateMap?.byteLength ?? 0) +
        (finalDebugMaps.protectedAmbiguousBroadSupportPenaltyGateMap?.byteLength ?? 0) +
        (finalDebugMaps.structuralSupportGateMap?.byteLength ?? 0) +
        (finalDebugMaps.peakStructureValidityMap?.byteLength ?? 0) +
        (finalDebugMaps.supportValidityMap?.byteLength ?? 0) +
        (finalDebugMaps.rowConfidenceGateMap?.byteLength ?? 0) +
        (finalDebugMaps.falseSupportConfidenceGateMap?.byteLength ?? 0) +
        (finalDebugMaps.falseSupportDensityGateMap?.byteLength ?? 0) +
        (finalDebugMaps.falseSupportSpreadGateMap?.byteLength ?? 0) +
        (finalDebugMaps.falseSupportVetoMap?.byteLength ?? 0) +
        (finalDebugMaps.rowBackgroundDensityGateMap?.byteLength ?? 0) +
        (finalDebugMaps.rowBackgroundSpreadGateMap?.byteLength ?? 0) +
        (finalDebugMaps.rowBackgroundPeakHuGateMap?.byteLength ?? 0) +
        (finalDebugMaps.rowBackgroundEdgeGateMap?.byteLength ?? 0) +
        (finalDebugMaps.rowBackgroundVetoMap?.byteLength ?? 0) +
        (finalDebugMaps.supportVetoTriggeredMap?.byteLength ?? 0) +
        (finalDebugMaps.upperSupportDepthMap?.byteLength ?? 0) +
        (finalDebugMaps.lowerSupportDepthMap?.byteLength ?? 0) +
        (finalDebugMaps.upperSupportConfidenceMap?.byteLength ?? 0) +
        (finalDebugMaps.lowerSupportConfidenceMap?.byteLength ?? 0) +
        (finalDebugMaps.supportBlendMap?.byteLength ?? 0) +
        (finalDebugMaps.supportDepthMap?.byteLength ?? 0) +
        (finalDebugMaps.supportConfidenceMap?.byteLength ?? 0) +
        (finalDebugMaps.supportSpreadMap?.byteLength ?? 0) +
        (finalDebugMaps.supportDensityMap?.byteLength ?? 0) +
        (finalDebugMaps.totalAttenuationMap?.byteLength ?? 0) +
        (finalDebugMaps.fogAttenuationMap?.byteLength ?? 0) +
        (finalDebugMaps.lowerPenaltyMap?.byteLength ?? 0) +
        (finalDebugMaps.participatingSampleCountMap?.byteLength ?? 0) +
        (finalDebugMaps.toneResponseMap?.byteLength ?? 0) +
        (finalDebugMaps.troughHalfWidthMap?.byteLength ?? 0) +
        (finalDebugMaps.effectiveTroughHalfWidthMap?.byteLength ?? 0) +
        (finalDebugMaps.continuityExpandedTroughHalfWidthMap?.byteLength ?? 0) +
        (finalDebugMaps.backgroundTroughNarrowGateMap?.byteLength ?? 0)
        + (finalDebugMaps.dominantToothBandGateMap?.byteLength ?? 0)
        + (finalDebugMaps.broadWeakToothBandGateMap?.byteLength ?? 0)
        + (finalDebugMaps.toothContinuityAdmissionGateMap?.byteLength ?? 0)
        : 0,
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
    debugMaps: finalDebugMaps,
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
    debugScalarSamplingMode: render.debugScalarSamplingMode,
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

function postWorkerMessage(
  message:
    | CPRWorkerBootstrapReady
    | CPRWorkerInitSuccess
    | CPRWorkerDisposeSuccess
    | CPRWorkerSuccess
    | CPRWorkerError
    | CPRWorkerLifecycle
): void {
  // eslint-disable-next-line no-restricted-globals
  (self as unknown as Worker).postMessage(message);
}

let workerLifecycleSequence = 0;

function logWorkerLifecycle(stage: string, detail?: Record<string, unknown>): void {
  console.log(
    '[CPR-WORKER-LIFECYCLE-JSON]',
    JSON.stringify({
      stage,
      ...(detail ?? {}),
    })
  );
}

function emitWorkerLifecycle(stage: string, detail?: Record<string, unknown>): void {
  workerLifecycleSequence += 1;
  const lifecycleDetail = {
    sequence: workerLifecycleSequence,
    ...(detail ?? {}),
  };
  logWorkerLifecycle(stage, lifecycleDetail);
  console.log(
    '[CPR-WORKER-PREPOST-JSON]',
    JSON.stringify({
      stage,
      ...lifecycleDetail,
    })
  );
  postWorkerMessage({
    type: 'WORKER_LIFECYCLE',
    scope: 'implementation',
    stage,
    detail: lifecycleDetail,
  });
}

// eslint-disable-next-line no-restricted-globals
self.addEventListener('error', event => {
  emitWorkerLifecycle('worker-global-error-event', {
    message: event.message ?? null,
    filename: event.filename ?? null,
    lineno: Number.isFinite(event.lineno) ? event.lineno : null,
    colno: Number.isFinite(event.colno) ? event.colno : null,
    errorMessage: event.error instanceof Error ? event.error.message : null,
  });
});

// eslint-disable-next-line no-restricted-globals
self.addEventListener('messageerror', event => {
  emitWorkerLifecycle('worker-messageerror-event', {
    origin: event.origin ?? null,
    lastEventId: event.lastEventId ?? null,
    dataType: event.data == null ? null : typeof event.data,
  });
});

// eslint-disable-next-line no-restricted-globals
self.addEventListener('unhandledrejection', event => {
  const reason =
    event.reason instanceof Error
      ? {
          name: event.reason.name,
          message: event.reason.message,
        }
      : {
          message: String(event.reason),
        };
  emitWorkerLifecycle('worker-unhandled-rejection-event', reason);
});

emitWorkerLifecycle('worker-implementation-script-loaded');

// eslint-disable-next-line no-restricted-globals
self.onmessage = async function (event: MessageEvent<CPRWorkerMessage>) {
  const input = event.data;
  const requestId = resolveWorkerRequestId(input);
  const requestType = typeof input?.type === 'string' ? input.type : 'unknown';
  try {
    if (input.type === 'BOOTSTRAP_CHECK') {
      emitWorkerLifecycle('bootstrap-check-received', {
        requestId,
      });
      const readyResponse: CPRWorkerBootstrapReady = {
        type: 'BOOTSTRAP_READY',
        requestId,
      };
      postWorkerMessage(readyResponse);
      emitWorkerLifecycle('bootstrap-ready-sent', {
        requestId,
      });
      return;
    }
    if (input.type === 'DISPOSE') {
      emitWorkerLifecycle('dispose-received', {
        requestId,
      });
      await disposeGpuPanoRendererIfLoaded();
      cachedVolumeState = null;
      const disposeResponse: CPRWorkerDisposeSuccess = {
        type: 'DISPOSE_SUCCESS',
        requestId,
      };
      postWorkerMessage(disposeResponse);
      emitWorkerLifecycle('dispose-success-sent', {
        requestId,
      });
      return;
    }

    if (input.type === 'INIT_VOLUME') {
      emitWorkerLifecycle('init-volume-received', {
        requestId,
        sessionKey: input.sessionKey,
        scalarType:
          input.scalarData && input.scalarData.constructor
            ? input.scalarData.constructor.name
            : 'unknown',
        scalarLength: input.scalarData?.length ?? 0,
        isSharedArrayBuffer: input.isSharedArrayBuffer === true,
      });
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
      cachedVolumeState = {
        sessionKey: input.sessionKey,
        scalarData: input.scalarData,
        isSharedArrayBuffer: input.isSharedArrayBuffer,
        dimensions: input.dimensions,
        spacing: input.spacing,
        origin: input.origin,
        direction: input.direction,
        worldToIndex: input.worldToIndex,
        rescaleSlope: policy.safeSlope,
        rescaleIntercept: policy.safeIntercept,
        bitsStored: policy.safeBitsStored,
        bitsAllocated: policy.safeBitsAllocated,
        highBit: policy.safeHighBit,
        pixelRepresentation:
          policy.safePixelRepresentation !== null
            ? policy.safePixelRepresentation
            : input.pixelRepresentation,
        isPreScaled: policy.effectiveIsPreScaled,
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
      postWorkerMessage(initResponse);
      emitWorkerLifecycle('init-success-sent', {
        requestId,
        sessionKey: input.sessionKey,
      });
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
    const requestedReconstructionMode = resolveReconstructionMode(renderInput.reconstructionMode);
    type OptionalGpuDiagnosticMaps = {
      meanMap?: Float32Array;
      maxMap?: Float32Array;
      sampleCountMap?: Float32Array;
      debugMaps?: GpuPanoDebugMaps;
    };
    let renderResult:
      | GeneratedGpuPanorama
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

    if (shouldRouteRequestedReconstructionToTrueVirtualPano(renderInput)) {
      logWorkerRouteDecision({
        runId: renderInput.debugRunId ?? null,
        attemptLabel: renderInput.attemptLabel ?? null,
        sourceVolumeId: renderInput.sourceVolumeId ?? null,
        requestedBackend: renderBackend,
        requestedReconstructionMode,
        actualRendererBackend: 'cpu',
        actualRendererFamily:
          requestedReconstructionMode === 'virtualPano'
            ? 'true-virtual-pano'
            : 'virtual-pano-phase1-analysis',
        routeReason: 'gpu-request-routed-to-true-virtual-pano-renderer',
      });
      renderResult = generatePanorama(renderInput, {
        requestedReconstructionMode,
      });
    } else if (renderBackend === 'gpu') {
      logWorkerRouteDecision({
        runId: renderInput.debugRunId ?? null,
        attemptLabel: renderInput.attemptLabel ?? null,
        sourceVolumeId: renderInput.sourceVolumeId ?? null,
        requestedBackend: renderBackend,
        requestedReconstructionMode,
        actualRendererBackend: 'gpu',
        actualRendererFamily: 'legacy-gpu-panorama',
        routeReason: 'gpu-legacy-direct-render',
      });
      let lastGpuError: unknown = null;
      let gpuFallbackCause: 'gpu-render-failed' | 'gpu-phase2-gate-failed' | null = null;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          renderResult = await generateGpuPanorama(renderInput, input.sessionKey);
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
            await disposeGpuPanoRendererIfLoaded();
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
        console.warn(
          '[CPR-GPU] GPU panorama missed the Phase 2 gate. Preserving GPU output for orchestrator ranking/QC instead of downgrading to CPU virtual pano.',
          {
            runId: renderInput.debugRunId ?? null,
            attemptLabel: renderInput.attemptLabel ?? null,
            pipelineMode: gpuPipelineMode,
          }
        );
      }

      if (gpuFallbackCause) {
        const cpuFallbackReconstructionMode = resolveCpuFallbackReconstructionMode(
          requestedReconstructionMode
        );
        logWorkerRouteDecision({
          runId: renderInput.debugRunId ?? null,
          attemptLabel: renderInput.attemptLabel ?? null,
          sourceVolumeId: renderInput.sourceVolumeId ?? null,
          requestedBackend: renderBackend,
          requestedReconstructionMode,
          actualRendererBackend: 'cpu',
          actualRendererFamily: 'true-virtual-pano',
          routeReason: 'gpu-fallback-to-true-virtual-pano-renderer',
          gpuFallbackCause,
        });
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
      logWorkerRouteDecision({
        runId: renderInput.debugRunId ?? null,
        attemptLabel: renderInput.attemptLabel ?? null,
        sourceVolumeId: renderInput.sourceVolumeId ?? null,
        requestedBackend: renderBackend,
        requestedReconstructionMode,
        actualRendererBackend: 'cpu',
        actualRendererFamily:
          requestedReconstructionMode === 'virtualPano'
            ? 'true-virtual-pano'
            : requestedReconstructionMode === 'virtualPanoPhase1'
              ? 'virtual-pano-phase1-analysis'
              : 'cpu-legacy-panorama',
        routeReason:
          requestedReconstructionMode === 'legacy'
            ? 'cpu-direct-legacy-render'
            : 'cpu-direct-true-virtual-pano-render',
      });
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
    pushTransferBuffer(debugMaps?.rawSupportDepthMap);
    pushTransferBuffer(debugMaps?.rawSupportConfidenceMap);
    pushTransferBuffer(debugMaps?.rawSupportSpreadMap);
    pushTransferBuffer(debugMaps?.rawSupportDensityMap);
    pushTransferBuffer(debugMaps?.rawSupportDenseFractionMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakHuSupportGateMap);
    pushTransferBuffer(debugMaps?.rawSupportDominantPeakOffsetMap);
    pushTransferBuffer(debugMaps?.rawSupportSecondaryPeakOffsetMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakDominanceMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakValidityMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakConflictMap);
    pushTransferBuffer(debugMaps?.rawSupportSecondPeakRatioMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakSeparationMap);
    pushTransferBuffer(debugMaps?.rawSupportPeakAmbiguityMap);
    pushTransferBuffer(debugMaps?.rawSupportScoreGapMap);
    pushTransferBuffer(debugMaps?.rawSupportLocalJumpMap);
    pushTransferBuffer(debugMaps?.rawSupportContinuityMap);
    pushTransferBuffer(debugMaps?.toothBandPriorMap);
    pushTransferBuffer(debugMaps?.dominantDensePeakGateMap);
    pushTransferBuffer(debugMaps?.toothBandStructureGuardMap);
    pushTransferBuffer(debugMaps?.ambiguousBroadSupportPenaltyGateMap);
    pushTransferBuffer(debugMaps?.protectedAmbiguousBroadSupportPenaltyGateMap);
    pushTransferBuffer(debugMaps?.structuralSupportGateMap);
    pushTransferBuffer(debugMaps?.peakStructureValidityMap);
    pushTransferBuffer(debugMaps?.supportValidityMap);
    pushTransferBuffer(debugMaps?.rowConfidenceGateMap);
    pushTransferBuffer(debugMaps?.falseSupportConfidenceGateMap);
    pushTransferBuffer(debugMaps?.falseSupportDensityGateMap);
    pushTransferBuffer(debugMaps?.falseSupportSpreadGateMap);
    pushTransferBuffer(debugMaps?.falseSupportVetoMap);
    pushTransferBuffer(debugMaps?.rowBackgroundDensityGateMap);
    pushTransferBuffer(debugMaps?.rowBackgroundSpreadGateMap);
    pushTransferBuffer(debugMaps?.rowBackgroundPeakHuGateMap);
    pushTransferBuffer(debugMaps?.rowBackgroundEdgeGateMap);
    pushTransferBuffer(debugMaps?.rowBackgroundVetoMap);
    pushTransferBuffer(debugMaps?.supportVetoTriggeredMap);
    pushTransferBuffer(debugMaps?.supportFailureDisplayMap);
    pushTransferBuffer(debugMaps?.upperSupportDepthMap);
    pushTransferBuffer(debugMaps?.lowerSupportDepthMap);
    pushTransferBuffer(debugMaps?.upperSupportConfidenceMap);
    pushTransferBuffer(debugMaps?.lowerSupportConfidenceMap);
    pushTransferBuffer(debugMaps?.supportBlendMap);
    pushTransferBuffer(debugMaps?.supportDepthMap);
    pushTransferBuffer(debugMaps?.supportConfidenceMap);
    pushTransferBuffer(debugMaps?.supportSpreadMap);
    pushTransferBuffer(debugMaps?.supportDensityMap);
    pushTransferBuffer(debugMaps?.supportLocalJumpMap);
    pushTransferBuffer(debugMaps?.supportContinuityMap);
    pushTransferBuffer(debugMaps?.totalAttenuationMap);
    pushTransferBuffer(debugMaps?.admissionAccumulationMap);
    pushTransferBuffer(debugMaps?.toneSuppressedAccumulationMap);
    pushTransferBuffer(debugMaps?.fogAttenuationMap);
    pushTransferBuffer(debugMaps?.lowerPenaltyMap);
    pushTransferBuffer(debugMaps?.participatingSampleCountMap);
    pushTransferBuffer(debugMaps?.toneResponseMap);
    pushTransferBuffer(debugMaps?.preToneAccumulationMap);
    pushTransferBuffer(debugMaps?.retainedSampleMaskMap);
    pushTransferBuffer(debugMaps?.middleBandLeakMap);
    pushTransferBuffer(debugMaps?.admissionMiddleBandLeakMap);
    pushTransferBuffer(debugMaps?.invalidSupportBlackoutMap);
    pushTransferBuffer(debugMaps?.toneStageSuppressionMap);
    pushTransferBuffer(debugMaps?.blackClipMap);
    pushTransferBuffer(debugMaps?.backgroundLeakToneMap);
    pushTransferBuffer(debugMaps?.backgroundLeakOutlier05Map);
    pushTransferBuffer(debugMaps?.backgroundLeakOutlier10Map);
    pushTransferBuffer(debugMaps?.admissionOnlyHuMap);
    pushTransferBuffer(debugMaps?.toneBypassHuMap);
    pushTransferBuffer(debugMaps?.troughHalfWidthMap);
    pushTransferBuffer(debugMaps?.effectiveTroughHalfWidthMap);
    pushTransferBuffer(debugMaps?.continuityExpandedTroughHalfWidthMap);
    pushTransferBuffer(debugMaps?.backgroundTroughNarrowGateMap);
    pushTransferBuffer(debugMaps?.dominantToothBandGateMap);
    pushTransferBuffer(debugMaps?.broadWeakToothBandGateMap);
    pushTransferBuffer(debugMaps?.toothContinuityAdmissionGateMap);

    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response, transferList);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    emitWorkerLifecycle('worker-caught-request-error', {
      requestType,
      requestId,
      message,
    });
    console.error('[CPR-WORKER-ERROR-JSON]', {
      stage: requestType,
      requestId,
      message,
    });
    const response: CPRWorkerError = {
      type: 'ERROR',
      requestId,
      message,
      stage: requestType,
    };
    postWorkerMessage(response);
  }
};
