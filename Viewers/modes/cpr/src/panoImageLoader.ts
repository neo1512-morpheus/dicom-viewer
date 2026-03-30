import * as cornerstone from '@cornerstonejs/core';
import type { GpuPanoDebugMaps } from './cprGpuRenderer';

import {
  SyntheticCprIntensityDomain,
  getFallbackSyntheticCprVoi,
} from './cprSyntheticDisplay';

export type PanoQualityStatus = 'accepted' | 'degraded';

export interface PanoImagePayload {
  pixelData: Float32Array | Uint16Array;
  meanMap?: Float32Array;
  maxMap?: Float32Array;
  sampleCountMap?: Float32Array;
  debugMaps?: GpuPanoDebugMaps;
  probeContext?: {
    runId?: string | null;
    displayedPath?: string | null;
    backend?: string | null;
    requestedBackend?: string | null;
    pipelineMode?: string | null;
    reconstructionMode?: string | null;
    qualityStatus?: PanoQualityStatus | null;
    qualityGateRejectReasons?: string[];
    qualityGateSelectionReason?: string | null;
    qualityGateMessage?: string | null;
  };
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  qualityGatePassed?: boolean;
  qualityStatus?: PanoQualityStatus;
  qualityGateRejectReasons?: string[];
  qualityGateSelectionReason?: string | null;
  qualityGateMessage?: string | null;
  huDomain?: boolean;
  intensityDomain?: SyntheticCprIntensityDomain;
  columnPixelSpacing?: number;
  rowPixelSpacing?: number;
  windowWidth?: number;
  windowCenter?: number;
  slope?: number;
  intercept?: number;
}

const panoImageCache = new Map<string, PanoImagePayload>();
let latestPanoImageId: string | null = null;
const PANO_FRAME_OF_REFERENCE_UID = 'CPR_PANO_FRAME_OF_REFERENCE';
const PANO_SERIES_INSTANCE_UID = 'CPR_PANO_SERIES_INSTANCE';
const PANO_STUDY_INSTANCE_UID = 'CPR_PANO_STUDY_INSTANCE';
export const PANO_IMAGE_ID = 'pano://current';
let panoImageSequence = 0;
interface PanoDisplayMetadata {
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  slope: number;
  intercept: number;
}

function getPanoDisplayMetadata(payload: PanoImagePayload | null): PanoDisplayMetadata {
  const safeMin = Number.isFinite(payload?.minValue) ? Number(payload?.minValue) : -1000;
  const safeMax = Number.isFinite(payload?.maxValue) ? Number(payload?.maxValue) : 3000;
  const payloadSlope = Number.isFinite(payload?.slope) && Math.abs(Number(payload?.slope)) > 1e-8
    ? Number(payload?.slope)
    : 1;
  const payloadIntercept = Number.isFinite(payload?.intercept) ? Number(payload?.intercept) : 0;

  const derivedVoi = getFallbackSyntheticCprVoi();

  const windowWidth =
    Number.isFinite(payload?.windowWidth) && Number(payload?.windowWidth) > 1
      ? Number(payload?.windowWidth)
      : derivedVoi.windowWidth;

  const windowCenter = Number.isFinite(payload?.windowCenter)
    ? Number(payload?.windowCenter)
    : derivedVoi.windowCenter;

  return {
    minValue: safeMin,
    maxValue: safeMax,
    windowWidth,
    windowCenter,
    slope: payloadSlope,
    intercept: payloadIntercept,
  };
}

function getPanoStoredPixelRange(
  pixelData: Float32Array | Uint16Array
): {
  minValue: number;
  maxValue: number;
} {
  if (!pixelData.length) {
    return { minValue: 0, maxValue: 0 };
  }

  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < pixelData.length; i++) {
    const value = Number(pixelData[i]);
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

function getPanoPixelSpacing(payload: PanoImagePayload | null): {
  rowPixelSpacing: number;
  columnPixelSpacing: number;
} {
  const rowPixelSpacing =
    Number.isFinite(payload?.rowPixelSpacing) && Number(payload?.rowPixelSpacing) > 0
      ? Number(payload?.rowPixelSpacing)
      : 1;
  const columnPixelSpacing =
    Number.isFinite(payload?.columnPixelSpacing) && Number(payload?.columnPixelSpacing) > 0
      ? Number(payload?.columnPixelSpacing)
      : 1;

  return {
    rowPixelSpacing,
    columnPixelSpacing,
  };
}

export function createPanoImageId(): string {
  panoImageSequence += 1;
  return `pano://render-${Date.now()}-${panoImageSequence}`;
}

function evictCachedPanoImage(imageId: string | null | undefined): void {
  if (!imageId) {
    return;
  }

  try {
    const imageLoadObject = cornerstone.cache.getImageLoadObject(imageId);
    if (imageLoadObject) {
      cornerstone.cache.removeImageLoadObject(imageId);
    }
  } catch (error) {
    console.warn('[panoImageLoader] Failed to evict cached pano image.', {
      imageId,
      error,
    });
  }
}

export function setPanoImagePayload(imageId: string, payload: PanoImagePayload): void {
  evictCachedPanoImage(imageId);
  panoImageCache.set(imageId, payload);
  latestPanoImageId = imageId;
}

export function clearPanoImageCache(): void {
  const imageIdsToEvict = new Set<string>();
  for (const imageId of panoImageCache.keys()) {
    imageIdsToEvict.add(imageId);
  }
  if (latestPanoImageId) {
    imageIdsToEvict.add(latestPanoImageId);
  }

  imageIdsToEvict.forEach(imageId => evictCachedPanoImage(imageId));
  panoImageCache.clear();
  latestPanoImageId = null;
}

function createImageObject(imageId: string, payload: PanoImagePayload): cornerstone.Types.IImage {
  const { pixelData, width, height } = payload;
  const display = getPanoDisplayMetadata(payload);
  const spacing = getPanoPixelSpacing(payload);
  const storedRange = getPanoStoredPixelRange(pixelData);

  const image: cornerstone.Types.IImage = {
    imageId,
    minPixelValue: storedRange.minValue,
    maxPixelValue: storedRange.maxValue,
    slope: display.slope,
    intercept: display.intercept,
    windowCenter: display.windowCenter,
    windowWidth: display.windowWidth,
    voiLUTFunction: 'LINEAR_EXACT',
    getPixelData: () => pixelData,
    getCanvas: () => {
      return null;
    },
    rows: height,
    columns: width,
    height,
    width,
    color: false,
    rgba: false,
    numComps: 1,
    columnPixelSpacing: spacing.columnPixelSpacing,
    rowPixelSpacing: spacing.rowPixelSpacing,
    sizeInBytes: pixelData.byteLength,
    invert: false,
    modalityLUT: undefined,
  };

  return image;
}

function panoImageLoader(imageId: string): {
  promise: Promise<cornerstone.Types.IImage>;
  cancelFn?: () => void;
} {
  const promise = new Promise<cornerstone.Types.IImage>((resolve, reject) => {
    const payload = panoImageCache.get(imageId);

    if (!payload) {
      reject(
        new Error(
          `[panoImageLoader] No panoramic image cached for imageId: "${imageId}". ` +
            'Call setPanoImagePayload() before setting the stack on the pano viewport.'
        )
      );
      return;
    }

    const image = createImageObject(imageId, payload);
    console.log('[PANO-LOADER-METADATA]', {
      minPixelValue: image.minPixelValue,
      maxPixelValue: image.maxPixelValue,
      windowWidth: image.windowWidth,
      windowCenter: image.windowCenter,
      slope: image.slope,
      intercept: image.intercept,
      invert: image.invert,
      qualityGatePassed: payload.qualityGatePassed ?? null,
      qualityStatus: payload.qualityStatus ?? null,
      qualityGateRejectReasons: payload.qualityGateRejectReasons ?? [],
      qualityGateSelectionReason: payload.qualityGateSelectionReason ?? null,
    });
    resolve(image);
  });

  return { promise };
}

function getPanoPayloadForMetadata(imageId: string): PanoImagePayload | null {
  if (panoImageCache.has(imageId)) {
    return panoImageCache.get(imageId)!;
  }

  // Fallback for callers requesting metadata before stack assignment settles.
  if (latestPanoImageId && panoImageCache.has(latestPanoImageId)) {
    return panoImageCache.get(latestPanoImageId)!;
  }

  // Backward compatibility for any legacy path still using a fixed imageId.
  if (panoImageCache.has(PANO_IMAGE_ID)) {
    return panoImageCache.get(PANO_IMAGE_ID)!;
  }

  return null;
}

export function getPanoImagePayload(imageId?: string | null): PanoImagePayload | null {
  if (typeof imageId === 'string' && imageId.length > 0) {
    return getPanoPayloadForMetadata(imageId);
  }

  if (latestPanoImageId) {
    return getPanoPayloadForMetadata(latestPanoImageId);
  }

  return null;
}

function panoMetadataProvider(type: string, imageId: string) {
  if (typeof imageId !== 'string' || !imageId.startsWith('pano://')) {
    return;
  }

  const payload = getPanoPayloadForMetadata(imageId);
  const isHuDomain = payload?.intensityDomain === 'hu' || payload?.huDomain === true;
  const width = payload?.width ?? 1;
  const height = payload?.height ?? 1;
  const display = getPanoDisplayMetadata(payload);
  const spacing = getPanoPixelSpacing(payload);

  if (type === 'imagePlaneModule') {
    return {
      frameOfReferenceUID: PANO_FRAME_OF_REFERENCE_UID,
      rows: height,
      columns: width,
      rowCosines: [1, 0, 0],
      columnCosines: [0, 1, 0],
      rowPixelSpacing: spacing.rowPixelSpacing,
      columnPixelSpacing: spacing.columnPixelSpacing,
      imagePositionPatient: [0, 0, 0],
      sliceThickness: 1,
      imageOrientationPatient: [1, 0, 0, 0, 1, 0],
    };
  }

  if (type === 'imagePixelModule') {
    if (payload?.pixelData instanceof Uint16Array) {
      return {
        samplesPerPixel: 1,
        photometricInterpretation: 'MONOCHROME2',
        bitsAllocated: 16,
        bitsStored: 16,
        highBit: 15,
        pixelRepresentation: 0,
      };
    }

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
      windowCenter: [display.windowCenter],
      windowWidth: [display.windowWidth],
      voiLUTFunction: 'LINEAR_EXACT',
    };
  }

  if (type === 'modalityLutModule') {
    const module = {
      rescaleIntercept: display.intercept,
      rescaleSlope: display.slope,
    };
    if (isHuDomain) {
      return {
        ...module,
        rescaleType: 'HU',
      };
    }

    return module;
  }

  if (type === 'generalSeriesModule') {
    return {
      modality: 'OT',
      seriesInstanceUID: PANO_SERIES_INSTANCE_UID,
      studyInstanceUID: PANO_STUDY_INSTANCE_UID,
      seriesNumber: 1,
    };
  }

  if (type === 'generalImageModule') {
    return {
      instanceNumber: 1,
    };
  }

  if (type === 'sopCommonModule') {
    return {
      sopClassUID: '1.2.840.10008.5.1.4.1.1.2',
      sopInstanceUID: `CPR_PANO_${imageId}`,
    };
  }

  return;
}

// Hardening #2: idempotent registration guard for pano scheme.
let panoLoaderRegistered = false;
let panoMetadataRegistered = false;

export function registerPanoImageLoader(): void {
  if (panoLoaderRegistered) {
    return;
  }

  cornerstone.imageLoader.registerImageLoader('pano', panoImageLoader);
  panoLoaderRegistered = true;

  if (!panoMetadataRegistered) {
    cornerstone.metaData.addProvider(panoMetadataProvider, 11001);
    panoMetadataRegistered = true;
  }
}

export { panoImageLoader };
