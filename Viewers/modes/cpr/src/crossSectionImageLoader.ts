import * as cornerstone from '@cornerstonejs/core';

import type { CPRFrame, Point3 } from './cprMath';
import { normalizeScalarDataToHuFloat32 } from './cprScalarPolicy';
import {
  SyntheticCprIntensityDomain,
  classifySyntheticCprIntensityDomain,
  isSyntheticCprHuDomain,
} from './cprSyntheticDisplay';

const CROSSSECTION_DEFAULT_WINDOW_WIDTH = 1800;
const CROSSSECTION_DEFAULT_WINDOW_CENTER = 1200;
export const CROSSSECTION_CANONICAL_GRID_SIZE = 512;
export const CROSSSECTION_CANONICAL_PIXEL_SPACING_MM = 0.08;

export interface CrossSectionSeriesPayload {
  sourceVolumeId: string;
  frames: CPRFrame[];
  width: number;
  height: number;
  rowPixelSpacing?: number;
  columnPixelSpacing?: number;
  horizontalHalfWidthMm: number;
  verticalHalfHeightMm: number;
  verticalCenterOffsetMm?: number;
  minValue?: number;
  maxValue?: number;
  huDomain?: boolean;
  intensityDomain?: SyntheticCprIntensityDomain;
  windowWidth?: number;
  windowCenter?: number;
}

type NumericArray =
  | Float32Array
  | Float64Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array
  | Int32Array
  | Uint32Array;

type CrossSectionImagePayload = {
  pixelData: Float32Array;
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  rowPixelSpacing: number;
  columnPixelSpacing: number;
  huDomain: boolean;
  intensityDomain: SyntheticCprIntensityDomain;
};

type ResolvedCrossSectionSeriesPayload = CrossSectionSeriesPayload & {
  scalarData: Float32Array;
  frames: CPRFrame[];
  width: number;
  height: number;
  rowPixelSpacing: number;
  columnPixelSpacing: number;
  horizontalHalfWidthMm: number;
  verticalHalfHeightMm: number;
  verticalCenterOffsetMm: number;
  minValue: number;
  maxValue: number;
  huDomain: boolean;
  intensityDomain: SyntheticCprIntensityDomain;
  windowWidth: number;
  windowCenter: number;
  storedValueNormalizationApplied: boolean;
  applyModalityLut: boolean;
  interpolationOobValue: number;
  slope: number;
  intercept: number;
  frameOfReferenceUID: string;
  studyInstanceUID: string;
  modality: string;
  normalizeStoredSample?: (value: number) => number;
};

const CROSSSECTION_SERIES_INSTANCE_UID_PREFIX = 'CPR_CROSSSECTION_SERIES';
const CROSSSECTION_STUDY_INSTANCE_UID = 'CPR_CROSSSECTION_STUDY_INSTANCE';
const CROSSSECTION_FRAME_OF_REFERENCE_UID = 'CPR_CROSSSECTION_FRAME_OF_REFERENCE';
const CROSSSECTION_IMAGE_CACHE_LIMIT = 48;
const CROSSSECTION_THROUGH_PLANE_HALF_THICKNESS_MM = 1.5;
const CROSSSECTION_THROUGH_PLANE_SAMPLES = 11;
const CROSSSECTION_THROUGH_PLANE_AGGREGATION: 'MEAN' | 'MIP' = 'MEAN';
const CROSSSECTION_ENABLE_DENOISE = false;
const EPS = 1e-8;

const crossSectionSeriesCache = new Map<string, ResolvedCrossSectionSeriesPayload>();
const crossSectionImageCache = new Map<string, CrossSectionImagePayload>();
const crossSectionPendingLoads = new Map<string, Promise<cornerstone.Types.IImage>>();
let crossSectionSeriesCounter = 0;
let crossSectionLoaderRegistered = false;
let crossSectionMetadataRegistered = false;

function dot(a: Point3, b: Point3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Point3, b: Point3): Point3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function add(a: Point3, b: Point3): Point3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: Point3, b: Point3): Point3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v: Point3, amount: number): Point3 {
  return [v[0] * amount, v[1] * amount, v[2] * amount];
}

function norm(v: Point3): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: Point3, fallback: Point3 = [1, 0, 0]): Point3 {
  const length = norm(v);
  if (!Number.isFinite(length) || length < EPS) {
    return [fallback[0], fallback[1], fallback[2]];
  }

  return [v[0] / length, v[1] / length, v[2] / length];
}

function negate(v: Point3): Point3 {
  return [-v[0], -v[1], -v[2]];
}

function projectPerpendicular(v: Point3, axis: Point3): Point3 {
  const unitAxis = normalize(axis);
  const projected = scale(unitAxis, dot(v, unitAxis));
  return subtract(v, projected);
}

function pickStablePerpendicular(axis: Point3): Point3 {
  const unitAxis = normalize(axis, [1, 0, 0]);
  const candidates: Point3[] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  let best = candidates[0];
  let smallestAlignment = Math.abs(dot(unitAxis, best));

  for (let i = 1; i < candidates.length; i++) {
    const alignment = Math.abs(dot(unitAxis, candidates[i]));
    if (alignment < smallestAlignment) {
      smallestAlignment = alignment;
      best = candidates[i];
    }
  }

  const perpendicular = projectPerpendicular(best, unitAxis);
  if (norm(perpendicular) >= EPS) {
    return normalize(perpendicular, [0, 1, 0]);
  }

  return normalize(cross(unitAxis, [0, 1, 0]), [1, 0, 0]);
}

function clampPositiveInteger(value: unknown, fallback: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 1) {
    return fallback;
  }

  return Math.max(1, Math.round(numeric));
}

function toPositiveFinite(value: unknown, fallback: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback;
  }

  return numeric;
}

export function getCrossSectionCanonicalGridConfig(): {
  width: number;
  height: number;
  rowPixelSpacing: number;
  columnPixelSpacing: number;
  horizontalHalfWidthMm: number;
  verticalHalfHeightMm: number;
} {
  const width = CROSSSECTION_CANONICAL_GRID_SIZE;
  const height = CROSSSECTION_CANONICAL_GRID_SIZE;
  const rowPixelSpacing = CROSSSECTION_CANONICAL_PIXEL_SPACING_MM;
  const columnPixelSpacing = CROSSSECTION_CANONICAL_PIXEL_SPACING_MM;
  const horizontalHalfWidthMm = columnPixelSpacing * Math.max(0, width - 1) * 0.5;
  const verticalHalfHeightMm = rowPixelSpacing * Math.max(0, height - 1) * 0.5;

  return {
    width,
    height,
    rowPixelSpacing,
    columnPixelSpacing,
    horizontalHalfWidthMm,
    verticalHalfHeightMm,
  };
}

function estimateScalarRange(data: NumericArray): { minValue: number; maxValue: number } {
  if (!data?.length) {
    return { minValue: 0, maxValue: 0 };
  }

  let minValue = Infinity;
  let maxValue = -Infinity;
  const step = Math.max(1, Math.floor(data.length / 20000));

  for (let i = 0; i < data.length; i += step) {
    const value = Number(data[i]);
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

function getDefaultCrossSectionVoi(): {
  lower: number;
  upper: number;
  windowWidth: number;
  windowCenter: number;
} {
  const windowWidth = CROSSSECTION_DEFAULT_WINDOW_WIDTH;
  const windowCenter = CROSSSECTION_DEFAULT_WINDOW_CENTER;

  return {
    lower: windowCenter - windowWidth / 2,
    upper: windowCenter + windowWidth / 2,
    windowWidth,
    windowCenter,
  };
}

function computeExactScalarRange(data: ArrayLike<number>): { minValue: number; maxValue: number } {
  if (!data?.length) {
    return { minValue: 0, maxValue: 0 };
  }

  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < data.length; i++) {
    const value = Number(data[i]);
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

function getSourceImageId(
  volume: cornerstone.Types.IImageVolume
): string | undefined {
  const imageIds = (volume as cornerstone.Types.IImageVolume & { imageIds?: string[] }).imageIds;
  return Array.isArray(imageIds) && imageIds.length > 0 ? imageIds[0] : undefined;
}

function getVolumeRescale(
  volume: cornerstone.Types.IImageVolume
): { slope: number; intercept: number } {
  const firstImageId = getSourceImageId(volume);
  if (!firstImageId) {
    return { slope: 1, intercept: 0 };
  }

  const modalityLut = cornerstone.metaData.get('modalityLutModule', firstImageId) as
    | {
        rescaleSlope?: number;
        rescaleIntercept?: number;
      }
    | undefined;
  const slope = Number(modalityLut?.rescaleSlope);
  const intercept = Number(modalityLut?.rescaleIntercept);

  return {
    slope: Number.isFinite(slope) && Math.abs(slope) > 1e-8 ? slope : 1,
    intercept: Number.isFinite(intercept) ? intercept : 0,
  };
}

function getVolumePixelStorage(
  volume: cornerstone.Types.IImageVolume
): {
  bitsStored: number;
  bitsAllocated: number;
  highBit: number;
  pixelRepresentation: number;
  isPreScaled: boolean;
} {
  const firstImageId = getSourceImageId(volume);
  const isPreScaled = !!(volume as cornerstone.Types.IImageVolume & { isPreScaled?: boolean })
    .isPreScaled;

  if (!firstImageId) {
    return {
      bitsStored: 16,
      bitsAllocated: 16,
      highBit: 15,
      pixelRepresentation: 1,
      isPreScaled,
    };
  }

  const imagePixelModule = cornerstone.metaData.get('imagePixelModule', firstImageId) as
    | {
        bitsStored?: number;
        bitsAllocated?: number;
        highBit?: number;
        pixelRepresentation?: number;
      }
    | undefined;

  const bitsStored = Number(imagePixelModule?.bitsStored);
  const bitsAllocated = Number(imagePixelModule?.bitsAllocated);
  const highBit = Number(imagePixelModule?.highBit);
  const pixelRepresentation = Number(imagePixelModule?.pixelRepresentation);
  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 ? Math.floor(bitsStored) : 16;
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && bitsAllocated >= safeBitsStored
      ? Math.floor(bitsAllocated)
      : 16;
  const safeHighBit =
    Number.isFinite(highBit) && highBit >= 0 ? Math.floor(highBit) : safeBitsStored - 1;

  return {
    bitsStored: safeBitsStored,
    bitsAllocated: safeBitsAllocated,
    highBit: Math.max(0, Math.min(safeBitsAllocated - 1, safeHighBit)),
    pixelRepresentation:
      Number.isFinite(pixelRepresentation) && (pixelRepresentation === 0 || pixelRepresentation === 1)
        ? pixelRepresentation
        : 1,
    isPreScaled,
  };
}

function getSourceSeriesMetadata(
  volume: cornerstone.Types.IImageVolume
): {
  frameOfReferenceUID: string;
  studyInstanceUID: string;
  modality: string;
} {
  const firstImageId = getSourceImageId(volume);
  if (!firstImageId) {
    return {
      frameOfReferenceUID: CROSSSECTION_FRAME_OF_REFERENCE_UID,
      studyInstanceUID: CROSSSECTION_STUDY_INSTANCE_UID,
      modality: 'CT',
    };
  }

  const imagePlaneModule = cornerstone.metaData.get('imagePlaneModule', firstImageId) as
    | {
        frameOfReferenceUID?: string;
      }
    | undefined;
  const generalSeriesModule = cornerstone.metaData.get('generalSeriesModule', firstImageId) as
    | {
        studyInstanceUID?: string;
        modality?: string;
      }
    | undefined;

  return {
    frameOfReferenceUID:
      imagePlaneModule?.frameOfReferenceUID || CROSSSECTION_FRAME_OF_REFERENCE_UID,
    studyInstanceUID:
      generalSeriesModule?.studyInstanceUID || CROSSSECTION_STUDY_INSTANCE_UID,
    modality: generalSeriesModule?.modality || 'CT',
  };
}

function resolveSeriesPayload(
  payload: CrossSectionSeriesPayload
): ResolvedCrossSectionSeriesPayload {
  const sourceVolume = cornerstone.cache.getVolume(payload.sourceVolumeId) as
    | cornerstone.Types.IImageVolume
    | undefined;

  if (!sourceVolume?.imageData) {
    throw new Error(
      `[crossSectionImageLoader] Source volume "${payload.sourceVolumeId}" is not available in cache.`
    );
  }

  const sourceScalarData = sourceVolume.imageData.getPointData().getScalars().getData() as NumericArray;
  const { slope, intercept } = getVolumeRescale(sourceVolume);
  const pixelStorage = getVolumePixelStorage(sourceVolume);
  const normalized = normalizeScalarDataToHuFloat32({
    scalarData: sourceScalarData,
    isPreScaled: pixelStorage.isPreScaled,
    rescaleSlope: slope,
    rescaleIntercept: intercept,
    bitsStored: pixelStorage.bitsStored,
    bitsAllocated: pixelStorage.bitsAllocated,
    highBit: pixelStorage.highBit,
    pixelRepresentation: pixelStorage.pixelRepresentation,
  });
  const cleanedHuScalarData = normalized.pixelData;
  const { minValue: displayMin, maxValue: displayMax } = estimateScalarRange(cleanedHuScalarData);
  const requestedIntensityDomain =
    payload.intensityDomain === 'hu' ||
    payload.intensityDomain === 'native' ||
    payload.intensityDomain === 'unknown'
      ? payload.intensityDomain
      : payload.huDomain === true
        ? 'hu'
        : undefined;
  const intensityDomain = requestedIntensityDomain || 'hu';
  const derivedVoi = getDefaultCrossSectionVoi();
  const huDomain = true;
  const windowWidth =
    Number.isFinite(payload.windowWidth)
      ? Number(payload.windowWidth)
      : derivedVoi.windowWidth;
  const windowCenter = Number.isFinite(payload.windowCenter)
    ? Number(payload.windowCenter)
    : derivedVoi.windowCenter;
  const { frameOfReferenceUID, studyInstanceUID, modality } = getSourceSeriesMetadata(sourceVolume);
  const canonicalGrid = getCrossSectionCanonicalGridConfig();
  const width = clampPositiveInteger(payload.width, canonicalGrid.width);
  const height = clampPositiveInteger(payload.height, canonicalGrid.height);
  const rowPixelSpacing = toPositiveFinite(payload.rowPixelSpacing, canonicalGrid.rowPixelSpacing);
  const columnPixelSpacing = toPositiveFinite(
    payload.columnPixelSpacing,
    canonicalGrid.columnPixelSpacing
  );
  const horizontalHalfWidthMm = columnPixelSpacing * Math.max(0, width - 1) * 0.5;
  const verticalHalfHeightMm = rowPixelSpacing * Math.max(0, height - 1) * 0.5;

  return {
    ...payload,
    scalarData: cleanedHuScalarData,
    frames: payload.frames.map(frame => ({
      ...frame,
      position: [frame.position[0], frame.position[1], frame.position[2]],
      T: [frame.T[0], frame.T[1], frame.T[2]],
      N_camera: [frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]],
      N_slab: [frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]],
      S: [frame.S[0], frame.S[1], frame.S[2]],
    })),
    width,
    height,
    rowPixelSpacing,
    columnPixelSpacing,
    horizontalHalfWidthMm,
    verticalHalfHeightMm,
    verticalCenterOffsetMm: Number.isFinite(payload.verticalCenterOffsetMm)
      ? Number(payload.verticalCenterOffsetMm)
      : 0,
    minValue:
      Number.isFinite(payload.minValue) ? Number(payload.minValue) : displayMin,
    maxValue:
      Number.isFinite(payload.maxValue) ? Number(payload.maxValue) : displayMax,
    huDomain,
    intensityDomain,
    windowWidth,
    windowCenter,
    storedValueNormalizationApplied: normalized.transform.shouldNormalizeStoredValues,
    applyModalityLut: false,
    interpolationOobValue: -1000,
    slope: 1,
    intercept: 0,
    frameOfReferenceUID,
    studyInstanceUID,
    modality,
  };
}

function buildCrossSectionBasis(frame: CPRFrame): {
  normal: Point3;
  right: Point3;
  up: Point3;
} {
  const normal = normalize([frame.T[0], frame.T[1], frame.T[2]], [1, 0, 0]);
  const frameRight: Point3 = [frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]];
  const frameUp: Point3 = [frame.S[0], frame.S[1], frame.S[2]];

  let right = projectPerpendicular(frameRight, normal);
  if (norm(right) < EPS) {
    right = pickStablePerpendicular(normal);
  } else {
    right = normalize(right, pickStablePerpendicular(normal));
  }

  if (dot(right, frameRight) < 0) {
    right = negate(right);
  }

  let up = projectPerpendicular(frameUp, normal);
  if (norm(up) < EPS) {
    up = normalize(cross(normal, right), [0, 0, 1]);
  } else {
    up = normalize(up, normalize(cross(normal, right), [0, 0, 1]));
  }
  if (dot(up, frameUp) < 0) {
    right = negate(right);
    up = negate(up);
  }
  right = normalize(cross(up, normal), right);
  if (dot(right, frameRight) < 0) {
    right = negate(right);
    up = negate(up);
  }

  return { normal, right, up };
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
  if (worldToIndex && worldToIndex.length >= 16) {
    const vi =
      worldToIndex[0] * wx +
      worldToIndex[4] * wy +
      worldToIndex[8] * wz +
      worldToIndex[12];
    const vj =
      worldToIndex[1] * wx +
      worldToIndex[5] * wy +
      worldToIndex[9] * wz +
      worldToIndex[13];
    const vk =
      worldToIndex[2] * wx +
      worldToIndex[6] * wy +
      worldToIndex[10] * wz +
      worldToIndex[14];
    return [vi, vj, vk];
  }

  const rx = wx - origin[0];
  const ry = wy - origin[1];
  const rz = wz - origin[2];

  return [
    (invDir[0] * rx + invDir[1] * ry + invDir[2] * rz) / spacing[0],
    (invDir[3] * rx + invDir[4] * ry + invDir[5] * rz) / spacing[1],
    (invDir[6] * rx + invDir[7] * ry + invDir[8] * rz) / spacing[2],
  ];
}

function invertMatrix3(m: number[]): number[] {
  // vtk.js stores the direction basis in column-major order. The fallback
  // worldToVoxel() path expects the inverse rows packed contiguously, so for an
  // orthonormal direction matrix the row-major transpose of the VTK basis is
  // read out in the original flat-array order.
  return [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]];
}

function trilinear(
  data: NumericArray,
  dims: [number, number, number],
  vi: number,
  vj: number,
  vk: number,
  oobValue: number,
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
  data: NumericArray,
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
  const v000 = read(Number(data[c000]));
  const v100 = read(Number(data[c100]));
  const v010 = read(Number(data[c010]));
  const v110 = read(Number(data[c110]));
  const v001 = read(Number(data[c001]));
  const v101 = read(Number(data[c101]));
  const v011 = read(Number(data[c011]));
  const v111 = read(Number(data[c111]));

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
  const sigmaRange = 120;
  const sigmaRangeDen = 2 * sigmaRange * sigmaRange;
  const clampedBlend = Math.min(1.0, blendWeight);
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
    for (let column = 1; column < width - 1; column++) {
      const index = row * width + column;
      const center = source[index];
      if (!Number.isFinite(center)) {
        continue;
      }

      const keepWeight = 1 - clampedBlend;
      let weightedSum = center;
      let weightTotal = 1;

      for (let n = 0; n < neighbors.length; n++) {
        const [dx, dy, spatialWeight] = neighbors[n];
        const neighborValue = source[(row + dy) * width + (column + dx)];
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

function parseCrossSectionImageId(
  imageId: string
): { seriesId: string; frameIndex: number } | null {
  if (typeof imageId !== 'string' || !imageId.startsWith('cross://')) {
    return null;
  }

  const remainder = imageId.slice('cross://'.length);
  const separatorIndex = remainder.lastIndexOf('/');
  if (separatorIndex <= 0) {
    return null;
  }

  const seriesId = remainder.slice(0, separatorIndex);
  const frameIndex = Number(remainder.slice(separatorIndex + 1));
  if (!seriesId || !Number.isFinite(frameIndex)) {
    return null;
  }

  return {
    seriesId,
    frameIndex: Math.max(0, Math.floor(frameIndex)),
  };
}

function rememberCrossSectionImage(imageId: string, payload: CrossSectionImagePayload): void {
  if (crossSectionImageCache.has(imageId)) {
    crossSectionImageCache.delete(imageId);
  }

  crossSectionImageCache.set(imageId, payload);

  while (crossSectionImageCache.size > CROSSSECTION_IMAGE_CACHE_LIMIT) {
    const oldestKey = crossSectionImageCache.keys().next().value;
    if (!oldestKey) {
      break;
    }
    crossSectionImageCache.delete(oldestKey);
  }
}

function computeCrossSectionImage(
  seriesPayload: ResolvedCrossSectionSeriesPayload,
  frameIndex: number
): CrossSectionImagePayload {
  const frame = seriesPayload.frames[frameIndex];
  if (!frame) {
    throw new Error(
      `[crossSectionImageLoader] Frame ${frameIndex} is not available for source volume "${seriesPayload.sourceVolumeId}".`
    );
  }

  const sourceVolume = cornerstone.cache.getVolume(seriesPayload.sourceVolumeId) as
    | cornerstone.Types.IImageVolume
    | undefined;
  if (!sourceVolume?.imageData) {
    throw new Error(
      `[crossSectionImageLoader] Source volume "${seriesPayload.sourceVolumeId}" is not available in cache.`
    );
  }

  const imageData = sourceVolume.imageData;
  const scalarData = seriesPayload.scalarData;
  const dimensions = imageData.getDimensions() as [number, number, number];
  const spacing = imageData.getSpacing() as [number, number, number];
  const origin = imageData.getOrigin() as [number, number, number];
  const direction = Array.from(imageData.getDirection()) as number[];
  const worldToIndex =
    (imageData as { getWorldToIndex?: () => ArrayLike<number> | null | undefined }).getWorldToIndex?.() ??
    undefined;
  const safeWorldToIndex =
    worldToIndex &&
    worldToIndex.length >= 16 &&
    Number.isFinite(worldToIndex[0]) &&
    Number.isFinite(worldToIndex[1]) &&
    Number.isFinite(worldToIndex[2]) &&
    Number.isFinite(worldToIndex[4]) &&
    Number.isFinite(worldToIndex[5]) &&
    Number.isFinite(worldToIndex[6]) &&
    Number.isFinite(worldToIndex[8]) &&
    Number.isFinite(worldToIndex[9]) &&
    Number.isFinite(worldToIndex[10]) &&
    Number.isFinite(worldToIndex[12]) &&
    Number.isFinite(worldToIndex[13]) &&
    Number.isFinite(worldToIndex[14])
      ? worldToIndex
      : undefined;
  const invDirection = invertMatrix3(direction);
  const { normal, right, up } = buildCrossSectionBasis(frame);
  const width = seriesPayload.width;
  const height = seriesPayload.height;
  const columnPixelSpacing = seriesPayload.columnPixelSpacing;
  const rowPixelSpacing = seriesPayload.rowPixelSpacing;
  const columnCenterIndex = (width - 1) * 0.5;
  const rowCenterIndex = (height - 1) * 0.5;
  const frameCenter =
    Math.abs(seriesPayload.verticalCenterOffsetMm) > EPS
      ? add(frame.position, scale(up, seriesPayload.verticalCenterOffsetMm))
      : frame.position;
  const huDomain = isSyntheticCprHuDomain(seriesPayload.intensityDomain);
  const oobValue = seriesPayload.applyModalityLut
    ? seriesPayload.interpolationOobValue
    : huDomain
      ? -1000
      : seriesPayload.minValue;
  const pixelData = new Float32Array(width * height);
  let minValue = Infinity;
  let maxValue = -Infinity;
  const throughPlaneStepMm =
    CROSSSECTION_THROUGH_PLANE_SAMPLES > 1
      ? (CROSSSECTION_THROUGH_PLANE_HALF_THICKNESS_MM * 2) /
        (CROSSSECTION_THROUGH_PLANE_SAMPLES - 1)
      : 0;

  for (let row = 0; row < height; row++) {
    const rowOffsetMm = (rowCenterIndex - row) * rowPixelSpacing;

    for (let column = 0; column < width; column++) {
      const columnOffsetMm = (column - columnCenterIndex) * columnPixelSpacing;
      const baseX =
        frameCenter[0] + up[0] * rowOffsetMm + right[0] * columnOffsetMm;
      const baseY =
        frameCenter[1] + up[1] * rowOffsetMm + right[1] * columnOffsetMm;
      const baseZ =
        frameCenter[2] + up[2] * rowOffsetMm + right[2] * columnOffsetMm;
      let accumulatedValue =
        CROSSSECTION_THROUGH_PLANE_AGGREGATION === 'MIP' ? Number.NEGATIVE_INFINITY : 0;
      let validSampleCount = 0;

      for (let depthSampleIndex = 0; depthSampleIndex < CROSSSECTION_THROUGH_PLANE_SAMPLES; depthSampleIndex++) {
        const depthOffsetMm =
          CROSSSECTION_THROUGH_PLANE_SAMPLES > 1
            ? -CROSSSECTION_THROUGH_PLANE_HALF_THICKNESS_MM + depthSampleIndex * throughPlaneStepMm
            : 0;
        const sampleX = baseX + normal[0] * depthOffsetMm;
        const sampleY = baseY + normal[1] * depthOffsetMm;
        const sampleZ = baseZ + normal[2] * depthOffsetMm;
        const [vi, vj, vk] = worldToVoxel(
          sampleX,
          sampleY,
          sampleZ,
          origin,
          spacing,
          invDirection,
          safeWorldToIndex
        );

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

        let sampleValue = trilinear(
          scalarData,
          dimensions,
          vi,
          vj,
          vk,
          oobValue,
          seriesPayload.normalizeStoredSample
        );
        if (seriesPayload.applyModalityLut) {
          sampleValue = sampleValue * seriesPayload.slope + seriesPayload.intercept;
        }
        if (!Number.isFinite(sampleValue)) {
          continue;
        }

        if (CROSSSECTION_THROUGH_PLANE_AGGREGATION === 'MIP') {
          accumulatedValue =
            validSampleCount > 0
              ? Math.max(accumulatedValue, sampleValue)
              : sampleValue;
        } else {
          accumulatedValue += sampleValue;
        }
        validSampleCount++;
      }

      const value =
        validSampleCount > 0
          ? CROSSSECTION_THROUGH_PLANE_AGGREGATION === 'MIP'
            ? accumulatedValue
            : accumulatedValue / validSampleCount
          : oobValue;

      pixelData[row * width + column] = value;

      if (Number.isFinite(value)) {
        if (value < minValue) {
          minValue = value;
        }
        if (value > maxValue) {
          maxValue = value;
        }
      }
    }
  }

  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    minValue = seriesPayload.minValue;
    maxValue = seriesPayload.maxValue;
  }

  if (CROSSSECTION_ENABLE_DENOISE) {
    const firstPassDenoised = applyLightBilateralDenoise(pixelData, width, height, 0.65);
    const secondPassDenoised = applyLightBilateralDenoise(pixelData, width, height, 0.65);
    const denoised = firstPassDenoised || secondPassDenoised;

    if (denoised) {
      const denoisedRange = computeExactScalarRange(pixelData);
      minValue = denoisedRange.minValue;
      maxValue = denoisedRange.maxValue;
    }
  }

  const exactRange = computeExactScalarRange(pixelData);
  minValue = exactRange.minValue;
  maxValue = exactRange.maxValue;

  return {
    pixelData,
    width,
    height,
    minValue,
    maxValue,
    windowWidth: seriesPayload.windowWidth,
    windowCenter: seriesPayload.windowCenter,
    rowPixelSpacing,
    columnPixelSpacing,
    huDomain,
    intensityDomain: seriesPayload.intensityDomain,
  };
}

function createCrossSectionImageObject(
  imageId: string,
  payload: CrossSectionImagePayload
): cornerstone.Types.IImage {
  if (payload.pixelData.length !== payload.width * payload.height) {
    throw new Error('Dimension mismatch');
  }

  const resolved = getCrossSectionSeriesPayloadForImage(imageId);
  if (!resolved) {
    throw new Error(
      `[crossSectionImageLoader] No series payload cached for imageId "${imageId}" when creating image object.`
    );
  }

  const { seriesPayload } = resolved;

  return {
    imageId,
    minPixelValue: payload.minValue,
    maxPixelValue: payload.maxValue,
    slope: 1,
    intercept: 0,
    windowCenter: seriesPayload.windowCenter,
    windowWidth: seriesPayload.windowWidth,
    voiLUTFunction: 'LINEAR_EXACT',
    getPixelData: () => payload.pixelData,
    getCanvas: () => null,
    rows: payload.height,
    columns: payload.width,
    height: payload.height,
    width: payload.width,
    color: false,
    rgba: false,
    numComps: 1,
    columnPixelSpacing: payload.columnPixelSpacing,
    rowPixelSpacing: payload.rowPixelSpacing,
    sizeInBytes: payload.pixelData.byteLength,
    invert: false,
  };
}

function getCrossSectionSeriesPayloadForImage(
  imageId: string
): { seriesPayload: ResolvedCrossSectionSeriesPayload; frameIndex: number } | null {
  const parsed = parseCrossSectionImageId(imageId);
  if (!parsed) {
    return null;
  }

  const seriesPayload = crossSectionSeriesCache.get(parsed.seriesId);
  if (!seriesPayload) {
    return null;
  }

  return {
    seriesPayload,
    frameIndex: Math.max(0, Math.min(parsed.frameIndex, Math.max(0, seriesPayload.frames.length - 1))),
  };
}

function crossSectionImageLoader(imageId: string): {
  promise: Promise<cornerstone.Types.IImage>;
  cancelFn?: () => void;
} {
  const cachedPayload = crossSectionImageCache.get(imageId);
  if (cachedPayload) {
    return {
      promise: Promise.resolve(createCrossSectionImageObject(imageId, cachedPayload)),
    };
  }

  const pending = crossSectionPendingLoads.get(imageId);
  if (pending) {
    return { promise: pending };
  }

  const promise = Promise.resolve().then(() => {
    const resolved = getCrossSectionSeriesPayloadForImage(imageId);
    if (!resolved) {
      throw new Error(
        `[crossSectionImageLoader] No series payload cached for imageId "${imageId}".`
      );
    }

    const payload = computeCrossSectionImage(resolved.seriesPayload, resolved.frameIndex);
    rememberCrossSectionImage(imageId, payload);
    return createCrossSectionImageObject(imageId, payload);
  });

  const trackedPromise = promise.finally(() => {
    crossSectionPendingLoads.delete(imageId);
  });

  crossSectionPendingLoads.set(imageId, trackedPromise);

  return { promise: trackedPromise };
}

function crossSectionMetadataProvider(type: string, imageId: string) {
  const resolved = getCrossSectionSeriesPayloadForImage(imageId);
  if (!resolved) {
    return;
  }

  const { seriesPayload, frameIndex } = resolved;
  const frame = seriesPayload.frames[frameIndex];
  const imagePayload = crossSectionImageCache.get(imageId) || null;
  const { right, up } = buildCrossSectionBasis(frame);
  const width = seriesPayload.width;
  const height = seriesPayload.height;
  const rowPixelSpacing =
    imagePayload?.rowPixelSpacing ??
    toPositiveFinite(seriesPayload.rowPixelSpacing, CROSSSECTION_CANONICAL_PIXEL_SPACING_MM);
  const columnPixelSpacing =
    imagePayload?.columnPixelSpacing ??
    toPositiveFinite(seriesPayload.columnPixelSpacing, CROSSSECTION_CANONICAL_PIXEL_SPACING_MM);
  const halfWidth = columnPixelSpacing * Math.max(0, width - 1) * 0.5;
  const halfHeight = rowPixelSpacing * Math.max(0, height - 1) * 0.5;
  const frameCenter =
    Math.abs(seriesPayload.verticalCenterOffsetMm) > EPS
      ? add(frame.position, scale(up, seriesPayload.verticalCenterOffsetMm))
      : frame.position;
  const topLeft = add(subtract(frameCenter, scale(right, halfWidth)), scale(up, halfHeight));

  if (type === 'imagePlaneModule') {
    return {
      frameOfReferenceUID: seriesPayload.frameOfReferenceUID,
      rows: height,
      columns: width,
      rowCosines: right,
      columnCosines: negate(up),
      rowPixelSpacing,
      columnPixelSpacing,
      imagePositionPatient: topLeft,
      sliceThickness: Math.max(
        CROSSSECTION_THROUGH_PLANE_HALF_THICKNESS_MM * 2,
        Math.min(rowPixelSpacing, columnPixelSpacing)
      ),
      imageOrientationPatient: [...right, ...negate(up)],
    };
  }

  if (type === 'imagePixelModule') {
    return {
      samplesPerPixel: 1,
      photometricInterpretation: 'MONOCHROME2',
      bitsAllocated: 32,
      bitsStored: 32,
      highBit: 31,
      pixelRepresentation: 1,
    };
  }

  if (type === 'multiFrameModule') {
    return {
      numberOfFrames: 1,
    };
  }

  if (type === 'voiLutModule') {
    return {
      windowCenter: [seriesPayload.windowCenter],
      windowWidth: [seriesPayload.windowWidth],
      voiLUTFunction: 'LINEAR_EXACT',
    };
  }

  if (type === 'modalityLutModule') {
    const module = {
      rescaleIntercept: 0,
      rescaleSlope: 1,
    };

    if (isSyntheticCprHuDomain(seriesPayload.intensityDomain)) {
      return {
        ...module,
        rescaleType: 'HU',
      };
    }

    return module;
  }

  if (type === 'generalSeriesModule') {
    return {
      modality: seriesPayload.modality,
      seriesInstanceUID: `${CROSSSECTION_SERIES_INSTANCE_UID_PREFIX}_${parsedSeriesId(imageId)}`,
      studyInstanceUID: seriesPayload.studyInstanceUID,
      seriesNumber: 2,
    };
  }

  if (type === 'generalImageModule') {
    return {
      instanceNumber: frameIndex + 1,
    };
  }

  if (type === 'sopCommonModule') {
    return {
      sopClassUID: '1.2.840.10008.5.1.4.1.1.2',
      sopInstanceUID: `CPR_CROSSSECTION_${parsedSeriesId(imageId)}_${frameIndex + 1}`,
    };
  }

  return;
}

function parsedSeriesId(imageId: string): string {
  const parsed = parseCrossSectionImageId(imageId);
  return parsed?.seriesId || 'unknown';
}

export function createCrossSectionSeriesId(): string {
  crossSectionSeriesCounter += 1;
  return `series-${Date.now()}-${crossSectionSeriesCounter}`;
}

export function createCrossSectionImageIds(seriesId: string, frameCount: number): string[] {
  const safeFrameCount = Math.max(0, Math.floor(Number(frameCount) || 0));
  return Array.from({ length: safeFrameCount }, (_, index) => `cross://${seriesId}/${index}`);
}

export function setCrossSectionSeriesPayload(
  seriesId: string,
  payload: CrossSectionSeriesPayload
): void {
  crossSectionSeriesCache.set(seriesId, resolveSeriesPayload(payload));
}

export function updateCrossSectionSeriesDisplayDefaults(
  seriesId: string,
  voiRange: { lower: number; upper: number }
): void {
  const seriesPayload = crossSectionSeriesCache.get(seriesId);
  if (!seriesPayload) {
    return;
  }

  const lower = Number(voiRange.lower);
  const upper = Number(voiRange.upper);
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
    return;
  }

  const windowWidth = Math.max(1, upper - lower);
  const windowCenter = lower + windowWidth * 0.5;

  crossSectionSeriesCache.set(seriesId, {
    ...seriesPayload,
    windowWidth,
    windowCenter,
  });
}

export function clearCrossSectionImageCache(): void {
  crossSectionSeriesCache.clear();
  crossSectionImageCache.clear();
  crossSectionPendingLoads.clear();
}

export function registerCrossSectionImageLoader(): void {
  if (!crossSectionLoaderRegistered) {
    cornerstone.imageLoader.registerImageLoader('cross', crossSectionImageLoader);
    crossSectionLoaderRegistered = true;
  }

  if (!crossSectionMetadataRegistered) {
    cornerstone.metaData.addProvider(crossSectionMetadataProvider, 11002);
    crossSectionMetadataRegistered = true;
  }
}

export function getDefaultCrossSectionVoiRange(): { lower: number; upper: number } {
  const debugVoi = getDefaultCrossSectionVoi();
  return {
    lower: debugVoi.lower,
    upper: debugVoi.upper,
  };
}

export { crossSectionImageLoader };
