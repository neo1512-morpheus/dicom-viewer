/**
 * panoImageLoader.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Custom Cornerstone3D image loader for the `pano:` URI scheme.
 *
 * ROLE IN THE PIPELINE
 * ────────────────────
 * After the Web Worker returns the Float32Array panoramic pixel data,
 * the main thread calls:
 *
 *   panoImageCache.set('pano://current', workerResult);
 *   viewport.setStack(['pano://current']);
 *   viewport.render();
 *
 * Cornerstone3D calls this loader with imageId='pano://current'.
 * The loader resolves instantly from the in-memory cache (no network request)
 * and returns a valid IImage object that Cornerstone3D can display.
 *
 * REGISTRATION
 * ────────────
 * This loader is registered in extensions/cornerstone/src/index.tsx
 * inside preRegistration — see the snippet at the bottom of this file.
 */

import type * as cornerstone from '@cornerstonejs/core';

// ─────────────────────────────────────────────────────────────────────────────
// IN-MEMORY CACHE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Payload stored in the cache after the Worker completes.
 * The main thread writes this; the loader reads it.
 */
export interface PanoImagePayload {
  pixelData: Float32Array;
  width: number;
  height: number;
  /** Min scalar value in the pixel array — used for windowCenter/windowWidth */
  minValue: number;
  /** Max scalar value in the pixel array */
  maxValue: number;
}

/**
 * Module-level cache keyed by imageId.
 * Typically only one entry ('pano://current') exists at a time.
 * A new entry replaces the old one on each "Gen Pano" invocation.
 */
const panoImageCache = new Map<string, PanoImagePayload>();

/**
 * Call this BEFORE setting the stack on the panoramic viewport.
 * Must be called on the main thread.
 */
export function setPanoImagePayload(imageId: string, payload: PanoImagePayload): void {
  panoImageCache.set(imageId, payload);
}

/**
 * Call in onModeExit or when clearing CPR state.
 */
export function clearPanoImageCache(): void {
  panoImageCache.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE LOADER
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Wraps a Float32Array panoramic pixel buffer into a Cornerstone3D IImage object.
 *
 * KEY DECISIONS
 * ─────────────
 * - pixelData is Float32Array: Cornerstone3D renders Float32 directly without
 *   clamping, which preserves the full HU range of the CBCT data.
 *
 * - modalityLUT / voiLUT are NOT set here. Instead, windowCenter/windowWidth
 *   are derived from minValue/maxValue with a sensible dental bone-window default.
 *   The radiologist can adjust via the Window/Level tool as normal.
 *
 * - rows/columns use Cornerstone3D's convention (rows = height, columns = width).
 *
 * - The image is marked as NOT a color image (color: false, rgba: false).
 *
 * - slope/intercept: 1/0 because the Float32Array already contains true HU values
 *   (we sampled directly from the volume, which Cornerstone stores in HU after
 *   applying the DICOM rescale slope/intercept during volume loading).
 */
function createImageObject(
  imageId: string,
  payload: PanoImagePayload
): cornerstone.Types.IImage {
  const { pixelData, width, height, minValue, maxValue } = payload;

  // Bone window defaults — correct for CBCT dental scans.
  // Matches the voi setting used in the CPR hanging protocol (mpr.ts reference).
  const windowWidth = 4000;
  const windowCenter = 1000;

  // Minimum and maximum pixel values for Cornerstone3D's internal range checks
  const minPixelValue = minValue;
  const maxPixelValue = maxValue;

  /**
   * getPixelData() is called by Cornerstone3D when it needs to upload the
   * texture to the GPU. Return the same Float32Array reference every time —
   * no copying. Cornerstone3D does not modify this array.
   */
  const getPixelData = () => pixelData;

  const image: cornerstone.Types.IImage = {
    imageId,
    minPixelValue,
    maxPixelValue,
    slope: 1,
    intercept: 0,
    windowCenter,
    windowWidth,
    getPixelData,
    rows: height,        // Cornerstone: rows = vertical = height
    columns: width,      // Cornerstone: columns = horizontal = width
    height,
    width,
    color: false,
    rgba: false,
    columnPixelSpacing: 1, // synthetic image — no real physical spacing
    rowPixelSpacing: 1,    // update to mm/pixel if arch arc-length is known
    sizeInBytes: pixelData.byteLength,
    numberOfComponents: 1,
    // Synthetic images have no real DICOM pixel format
    // Cornerstone3D infers Float32 from the getPixelData() return type
    invert: false,
  };

  return image;
}

/**
 * The image loader function passed to cornerstone.imageLoader.registerImageLoader.
 *
 * Cornerstone3D calls this with the full imageId string (e.g. 'pano://current').
 * It must return { promise: Promise<IImage>, cancelFn?, decache? }.
 */
function panoImageLoader(
  imageId: string
): { promise: Promise<cornerstone.Types.IImage>; cancelFn?: () => void } {
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

    resolve(createImageObject(imageId, payload));
  });

  return { promise };
}

// ─────────────────────────────────────────────────────────────────────────────
// CANONICAL IMAGEIDS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The imageId used for the current panoramic result.
 * Use this constant everywhere — in the image loader call and in viewport.setStack().
 */
export const PANO_IMAGE_ID = 'pano://current';

// ─────────────────────────────────────────────────────────────────────────────
// EXPORTS
// ─────────────────────────────────────────────────────────────────────────────

export { panoImageLoader };

// ─────────────────────────────────────────────────────────────────────────────
// REGISTRATION SNIPPET
// Copy the block below into:
//   extensions/cornerstone/src/index.tsx  →  preRegistration function
// ─────────────────────────────────────────────────────────────────────────────

/*

──── ADD THIS IMPORT at the top of extensions/cornerstone/src/index.tsx ────

import { panoImageLoader } from '../../../modes/cpr/src/panoImageLoader';


──── ADD THIS LINE inside the preRegistration function, BEFORE init.call() ─

preRegistration: function (props: Types.Extensions.ExtensionParams): Promise<void> {
  const { servicesManager, serviceProvidersManager } = props;

  // === EXISTING SERVICE REGISTRATIONS (do not change) ===
  servicesManager.registerService(CornerstoneViewportService.REGISTRATION);
  servicesManager.registerService(ToolGroupService.REGISTRATION);
  servicesManager.registerService(SyncGroupService.REGISTRATION);
  servicesManager.registerService(SegmentationService.REGISTRATION);
  servicesManager.registerService(CornerstoneCacheService.REGISTRATION);
  servicesManager.registerService(ViewportActionCornersService.REGISTRATION);
  servicesManager.registerService(ColorbarService.REGISTRATION);
  serviceProvidersManager.registerProvider(
    ViewportActionCornersService.REGISTRATION.name,
    ViewportActionCornersProvider
  );

  // === ADD THIS: Register the pano: image loader scheme ===
  // Must be done here, not in a component, to prevent a race condition where
  // Cornerstone3D tries to resolve a pano: URI before the component mounts.
  cornerstone.imageLoader.registerImageLoader('pano', panoImageLoader);
  // ========================================================

  return init.call(this, props);
},

*/
