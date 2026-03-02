import * as cornerstone from '@cornerstonejs/core';

export interface PanoImagePayload {
  pixelData: Float32Array;
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  huDomain?: boolean;
  columnPixelSpacing?: number;
  rowPixelSpacing?: number;
  windowWidth?: number;
  windowCenter?: number;
  slope?: number;
  intercept?: number;
}

const panoImageCache = new Map<string, PanoImagePayload>();
let latestPanoImageId: string | null = null;
let panoImageCounter = 0;
const PANO_FRAME_OF_REFERENCE_UID = 'CPR_PANO_FRAME_OF_REFERENCE';
const PANO_SERIES_INSTANCE_UID = 'CPR_PANO_SERIES_INSTANCE';
const PANO_STUDY_INSTANCE_UID = 'CPR_PANO_STUDY_INSTANCE';
const DEFAULT_PANO_WINDOW_WIDTH = 3000;
const DEFAULT_PANO_WINDOW_CENTER = 600;

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
  const isHuDomain = payload?.huDomain === true;
  const payloadSlope =
    Number.isFinite(payload?.slope) && Math.abs(Number(payload?.slope)) > 1e-8
      ? Number(payload?.slope)
      : 1;
  const payloadIntercept = Number.isFinite(payload?.intercept) ? Number(payload?.intercept) : 0;
  const hasNonIdentityRescale =
    Math.abs(payloadSlope - 1) > 1e-6 || Math.abs(payloadIntercept) > 1e-6;

  // Worker output for pano:// is expected to already be in HU space.
  // Always expose identity rescale metadata to avoid accidental double-LUT.
  if (hasNonIdentityRescale) {
    console.warn(
      '[panoImageLoader] Ignoring non-identity pano payload rescale and exposing identity modality LUT.',
      {
        payloadSlope,
        payloadIntercept,
      }
    );
  }
  const safeSlope = 1;
  const safeIntercept = 0;

  const modalityMin = safeMin;
  const modalityMax = safeMax;
  const dynamicRange = Math.max(1, modalityMax - modalityMin);
  const looksLikeHU = isHuDomain && modalityMin >= -5000 && modalityMax <= 7000 && dynamicRange >= 250;

  const windowWidth =
    Number.isFinite(payload?.windowWidth) && Number(payload?.windowWidth) > 1
      ? Number(payload?.windowWidth)
      : looksLikeHU
        ? DEFAULT_PANO_WINDOW_WIDTH
        : dynamicRange;

  const windowCenter = Number.isFinite(payload?.windowCenter)
    ? Number(payload?.windowCenter)
    : looksLikeHU
      ? DEFAULT_PANO_WINDOW_CENTER
      : modalityMin + dynamicRange / 2;

  return {
    minValue: safeMin,
    maxValue: safeMax,
    windowWidth,
    windowCenter,
    slope: safeSlope,
    intercept: safeIntercept,
  };
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
  panoImageCounter += 1;
  return `pano://current/${Date.now()}-${panoImageCounter}`;
}

export function setPanoImagePayload(imageId: string, payload: PanoImagePayload): void {
  panoImageCache.set(imageId, payload);
  latestPanoImageId = imageId;
  console.log('[CPR-LOADER-PAYLOAD-JSON]', JSON.stringify({
    imageId,
    width: payload.width,
    height: payload.height,
    minValue: payload.minValue,
    maxValue: payload.maxValue,
    huDomain: payload.huDomain === true,
    windowWidth: payload.windowWidth ?? null,
    windowCenter: payload.windowCenter ?? null,
    slope: payload.slope ?? null,
    intercept: payload.intercept ?? null,
    rowPixelSpacing: payload.rowPixelSpacing ?? null,
    columnPixelSpacing: payload.columnPixelSpacing ?? null,
    first8: Array.from(payload.pixelData.subarray(0, Math.min(8, payload.pixelData.length))).map(
      value => Math.round(Number(value) * 1000) / 1000
    ),
  }));
}

export function clearPanoImageCache(): void {
  panoImageCache.clear();
  latestPanoImageId = null;
}

function createImageObject(imageId: string, payload: PanoImagePayload): cornerstone.Types.IImage {
  const { pixelData, width, height } = payload;
  const display = getPanoDisplayMetadata(payload);
  const spacing = getPanoPixelSpacing(payload);
  console.log('[CPR-LOADER-IMAGE-JSON]', JSON.stringify({
    imageId,
    width,
    height,
    display,
    spacing,
    huDomain: payload.huDomain === true,
  }));

  const image: cornerstone.Types.IImage = {
    imageId,
    minPixelValue: display.minValue,
    maxPixelValue: display.maxValue,
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
      console.error('[CPR-LOADER-MISS-JSON]', JSON.stringify({ imageId, latestPanoImageId }));
      reject(
        new Error(
          `[panoImageLoader] No panoramic image cached for imageId: "${imageId}". ` +
            'Call setPanoImagePayload() before setting the stack on the pano viewport.'
        )
      );
      return;
    }

    resolve(createImageObject(imageId, payload));
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

function panoMetadataProvider(type: string, imageId: string) {
  if (typeof imageId !== 'string' || !imageId.startsWith('pano://')) {
    return;
  }

  const payload = getPanoPayloadForMetadata(imageId);
  const isHuDomain = payload?.huDomain === true;
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

export const PANO_IMAGE_ID = 'pano://current';

export { panoImageLoader };
