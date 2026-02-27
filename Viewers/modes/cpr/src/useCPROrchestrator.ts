import { useCallback, useEffect, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { vec3 } from 'gl-matrix';

import { cprStateService } from './CPRStateService';
import { setPanoImagePayload, createPanoImageId, clearPanoImageCache } from './panoImageLoader';
import { buildRMFFrames } from './cprMath';
import { CPR_CROSSSECTION_SYNC_EVENT, CPRCrossSectionSyncDetail } from './cprEvents';
import type { CPRFrame } from './cprMath';

const CPR_PANO_DEFAULT_WINDOW_WIDTH = 4000;
const CPR_PANO_DEFAULT_WINDOW_CENTER = 1000;

interface FloatBufferDebugSummary {
  sampledCount: number;
  min: number;
  max: number;
  p01: number;
  p50: number;
  p99: number;
  fractionBelowMinus950: number;
  fractionAbove3000: number;
}

interface PanoVoiSettings {
  lower: number;
  upper: number;
  windowWidth: number;
  windowCenter: number;
}

function percentileFromSorted(values: number[], q: number): number {
  if (!values.length) {
    return NaN;
  }

  const clampedQ = Math.max(0, Math.min(1, q));
  const position = clampedQ * (values.length - 1);
  const lo = Math.floor(position);
  const hi = Math.ceil(position);

  if (lo === hi) {
    return values[lo];
  }

  const t = position - lo;
  return values[lo] + t * (values[hi] - values[lo]);
}

function summarizeFloatBufferForDebug(buffer: Float32Array): FloatBufferDebugSummary | null {
  if (!buffer || buffer.length === 0) {
    return null;
  }

  const targetSamples = 20000;
  const step = Math.max(1, Math.floor(buffer.length / targetSamples));
  const samples: number[] = [];

  let min = Infinity;
  let max = -Infinity;
  let belowCount = 0;
  let aboveCount = 0;

  for (let i = 0; i < buffer.length; i += step) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }

    samples.push(value);
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
    if (value <= -950) {
      belowCount += 1;
    }
    if (value >= 3000) {
      aboveCount += 1;
    }
  }

  if (!samples.length || !Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }

  samples.sort((a, b) => a - b);
  const sampledCount = samples.length;

  return {
    sampledCount,
    min,
    max,
    p01: percentileFromSorted(samples, 0.01),
    p50: percentileFromSorted(samples, 0.5),
    p99: percentileFromSorted(samples, 0.99),
    fractionBelowMinus950: belowCount / sampledCount,
    fractionAbove3000: aboveCount / sampledCount,
  };
}

function computeAdaptivePanoVoi(
  summary: FloatBufferDebugSummary | null,
  minValue: number,
  maxValue: number
): PanoVoiSettings {
  const fallbackLower = CPR_PANO_DEFAULT_WINDOW_CENTER - CPR_PANO_DEFAULT_WINDOW_WIDTH / 2;
  const fallbackUpper = CPR_PANO_DEFAULT_WINDOW_CENTER + CPR_PANO_DEFAULT_WINDOW_WIDTH / 2;

  const safeMin = Number.isFinite(minValue) ? Number(minValue) : fallbackLower;
  const safeMax = Number.isFinite(maxValue) ? Number(maxValue) : fallbackUpper;
  const rangeLow = Math.min(safeMin, safeMax);
  const rangeHigh = Math.max(safeMin, safeMax);
  const p01 = summary?.p01;
  const p99 = summary?.p99;

  let lower = Number.isFinite(p01) ? Number(p01) : rangeLow;
  let upper = Number.isFinite(p99) ? Number(p99) : rangeHigh;

  if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) {
    lower = rangeLow;
    upper = rangeHigh;
  }

  // Avoid hard clipping by padding both sides of the robust percentile range.
  const robustSpan = Math.max(1, upper - lower);
  const padding = Math.max(50, robustSpan * 0.1);
  lower -= padding;
  upper += padding;

  if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) {
    lower = fallbackLower;
    upper = fallbackUpper;
  }

  const windowWidth = Math.max(1, upper - lower);
  const windowCenter = lower + windowWidth / 2;

  // Guardrail: if the data appears HU-like, avoid very narrow windows that
  // can make CPR pano look "washed white with black speckles".
  const looksLikeHU =
    Number.isFinite(rangeLow) &&
    Number.isFinite(rangeHigh) &&
    rangeLow >= -5000 &&
    rangeHigh <= 7000;

  if (looksLikeHU) {
    const MIN_HU_WINDOW_WIDTH = 2500;
    if (windowWidth < MIN_HU_WINDOW_WIDTH) {
      const centeredLower = windowCenter - MIN_HU_WINDOW_WIDTH / 2;
      const centeredUpper = windowCenter + MIN_HU_WINDOW_WIDTH / 2;
      return {
        lower: centeredLower,
        upper: centeredUpper,
        windowWidth: MIN_HU_WINDOW_WIDTH,
        windowCenter,
      };
    }
  }

  return { lower, upper, windowWidth, windowCenter };
}

function catmullRomPoint(
  Pm1: [number, number, number],
  P0: [number, number, number],
  P1: [number, number, number],
  P2: [number, number, number],
  t: number
): [number, number, number] {
  const t2 = t * t;
  const t3 = t2 * t;

  const b0 = -0.5 * t3 + t2 - 0.5 * t;
  const b1 = 1.5 * t3 - 2.5 * t2 + 1.0;
  const b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const b3 = 0.5 * t3 - 0.5 * t2;

  return [
    b0 * Pm1[0] + b1 * P0[0] + b2 * P1[0] + b3 * P2[0],
    b0 * Pm1[1] + b1 * P0[1] + b2 * P1[1] + b3 * P2[1],
    b0 * Pm1[2] + b1 * P0[2] + b2 * P1[2] + b3 * P2[2],
  ];
}

function catmullRomTangent(
  Pm1: [number, number, number],
  P0: [number, number, number],
  P1: [number, number, number],
  P2: [number, number, number],
  t: number
): [number, number, number] {
  const t2 = t * t;

  const d0 = -1.5 * t2 + 2.0 * t - 0.5;
  const d1 = 4.5 * t2 - 5.0 * t;
  const d2 = -4.5 * t2 + 4.0 * t + 0.5;
  const d3 = 1.5 * t2 - 1.0 * t;

  return [
    d0 * Pm1[0] + d1 * P0[0] + d2 * P1[0] + d3 * P2[0],
    d0 * Pm1[1] + d1 * P0[1] + d2 * P1[1] + d3 * P2[1],
    d0 * Pm1[2] + d1 * P0[2] + d2 * P1[2] + d3 * P2[2],
  ];
}

function dist3(a: [number, number, number], b: [number, number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const out = vec3.create();
  vec3.normalize(out, vec3.fromValues(v[0], v[1], v[2]));
  if (!Number.isFinite(out[0]) || !Number.isFinite(out[1]) || !Number.isFinite(out[2])) {
    return [1, 0, 0];
  }
  return [out[0], out[1], out[2]];
}

function sanitizeCameraBasis(
  normalIn: [number, number, number],
  upIn: [number, number, number],
  previousCamera?: {
    viewPlaneNormal?: cornerstone.Types.Point3;
    viewUp?: cornerstone.Types.Point3;
  }
): {
  viewPlaneNormal: [number, number, number];
  viewUp: [number, number, number];
} {
  const EPS = 1e-6;
  const normal = vec3.fromValues(
    Number(normalIn[0] ?? 0),
    Number(normalIn[1] ?? 0),
    Number(normalIn[2] ?? 0)
  );

  if (!Number.isFinite(normal[0]) || !Number.isFinite(normal[1]) || !Number.isFinite(normal[2])) {
    vec3.set(normal, 1, 0, 0);
  }
  if (vec3.length(normal) < EPS) {
    vec3.set(normal, 1, 0, 0);
  }
  vec3.normalize(normal, normal);

  const up = vec3.fromValues(Number(upIn[0] ?? 0), Number(upIn[1] ?? 0), Number(upIn[2] ?? 1));
  if (!Number.isFinite(up[0]) || !Number.isFinite(up[1]) || !Number.isFinite(up[2])) {
    vec3.set(up, 0, 0, 1);
  }

  // Enforce strict orthogonality: viewUp = viewUp - dot(viewUp, N) * N.
  const upProjection = vec3.scale(vec3.create(), normal, vec3.dot(up, normal));
  vec3.subtract(up, up, upProjection);

  if (vec3.length(up) < EPS) {
    const fallbackAxis = Math.abs(normal[2]) < 0.9 ? vec3.fromValues(0, 0, 1) : vec3.fromValues(0, 1, 0);
    vec3.cross(up, fallbackAxis, normal);
  }
  if (vec3.length(up) < EPS) {
    vec3.cross(up, vec3.fromValues(1, 0, 0), normal);
  }
  vec3.normalize(up, up);

  if (previousCamera?.viewPlaneNormal && previousCamera?.viewUp) {
    const prevNormal = vec3.fromValues(
      Number(previousCamera.viewPlaneNormal[0] ?? 0),
      Number(previousCamera.viewPlaneNormal[1] ?? 0),
      Number(previousCamera.viewPlaneNormal[2] ?? 0)
    );
    const prevUp = vec3.fromValues(
      Number(previousCamera.viewUp[0] ?? 0),
      Number(previousCamera.viewUp[1] ?? 0),
      Number(previousCamera.viewUp[2] ?? 0)
    );

    if (
      Number.isFinite(prevNormal[0]) &&
      Number.isFinite(prevNormal[1]) &&
      Number.isFinite(prevNormal[2]) &&
      vec3.length(prevNormal) >= EPS &&
      Number.isFinite(prevUp[0]) &&
      Number.isFinite(prevUp[1]) &&
      Number.isFinite(prevUp[2]) &&
      vec3.length(prevUp) >= EPS
    ) {
      vec3.normalize(prevNormal, prevNormal);

      const prevUpProjection = vec3.scale(vec3.create(), prevNormal, vec3.dot(prevUp, prevNormal));
      vec3.subtract(prevUp, prevUp, prevUpProjection);
      if (vec3.length(prevUp) >= EPS) {
        vec3.normalize(prevUp, prevUp);
      }

      // Keep orientation sign continuous across frame updates to avoid mirror flips.
      const continuityScore = vec3.dot(normal, prevNormal) + vec3.dot(up, prevUp);
      if (continuityScore < 0) {
        vec3.scale(normal, normal, -1);
        vec3.scale(up, up, -1);
      }
    }
  }

  return {
    viewPlaneNormal: [normal[0], normal[1], normal[2]],
    viewUp: [up[0], up[1], up[2]],
  };
}

function addPhantomEndpoints(pts: [number, number, number][]): [number, number, number][] {
  const first = pts[0];
  const second = pts[1];
  const last = pts[pts.length - 1];
  const penultimate = pts[pts.length - 2];

  const phantomStart: [number, number, number] = [
    2 * first[0] - second[0],
    2 * first[1] - second[1],
    2 * first[2] - second[2],
  ];
  const phantomEnd: [number, number, number] = [
    2 * last[0] - penultimate[0],
    2 * last[1] - penultimate[1],
    2 * last[2] - penultimate[2],
  ];

  return [phantomStart, ...pts, phantomEnd];
}

function buildArcLengthSpline(
  rawPoints: [number, number, number][],
  nSamples: number
): {
  positions: [number, number, number][];
  tangents: [number, number, number][];
} {
  if (rawPoints.length < 2) {
    throw new Error('[buildArcLengthSpline] Need at least 2 control points.');
  }

  const sampleCount = Math.max(2, Math.floor(nSamples));
  const extended = addPhantomEndpoints(rawPoints);
  const nSegments = extended.length - 3;

  const FINE_STEPS = 2000;
  const lutT: number[] = [0];
  const lutArc: number[] = [0];

  let prevPt = catmullRomPoint(extended[0], extended[1], extended[2], extended[3], 0);

  const stepsPerSegment = Math.max(1, Math.floor(FINE_STEPS / nSegments));
  for (let seg = 0; seg < nSegments; seg++) {
    const Pm1 = extended[seg];
    const P0 = extended[seg + 1];
    const P1 = extended[seg + 2];
    const P2 = extended[seg + 3];

    for (let step = 1; step <= stepsPerSegment; step++) {
      const localT = step / stepsPerSegment;
      const globalT = seg + localT;
      const pt = catmullRomPoint(Pm1, P0, P1, P2, localT);

      const arcLen = lutArc[lutArc.length - 1] + dist3(prevPt, pt);
      lutT.push(globalT);
      lutArc.push(arcLen);
      prevPt = pt;
    }
  }

  const totalArcLength = lutArc[lutArc.length - 1];
  const positions: [number, number, number][] = [];
  const tangents: [number, number, number][] = [];

  // Hardening #5: guard against divide-by-zero in nSamples denominator.
  const sampleDen = Math.max(1, sampleCount - 1);

  for (let i = 0; i < sampleCount; i++) {
    const targetArc = (i / sampleDen) * totalArcLength;

    let lo = 0;
    let hi = lutArc.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (lutArc[mid] < targetArc) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const arcRange = lutArc[hi] - lutArc[lo];
    const frac = arcRange < 1e-10 ? 0 : (targetArc - lutArc[lo]) / arcRange;
    const globalT = lutT[lo] + frac * (lutT[hi] - lutT[lo]);

    const seg = Math.min(Math.floor(globalT), nSegments - 1);
    const localT = globalT - seg;

    const Pm1 = extended[seg];
    const P0 = extended[seg + 1];
    const P1 = extended[seg + 2];
    const P2 = extended[seg + 3];

    positions.push(catmullRomPoint(Pm1, P0, P1, P2, localT));
    tangents.push(normalize3(catmullRomTangent(Pm1, P0, P1, P2, localT)));
  }

  return { positions, tangents };
}

function launchCPRWorker(params: {
  volume: cornerstone.Types.IImageVolume;
  frames: CPRFrame[];
  panoWidth: number;
  panoHeight: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  aggregation: 'MIP' | 'MEAN';
  verticalDir?: [number, number, number];
  debugRunId?: string;
}): Promise<{
  pixelData: Float32Array;
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
}> {
  return new Promise((resolve, reject) => {
    // @ts-expect-error Vite/webpack handles import.meta.url worker URLs at build time.
    const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });

    const {
      volume,
      frames,
      panoWidth,
      panoHeight,
      slabHalfThicknessMm,
      slabSamples,
      aggregation,
      verticalDir,
      debugRunId,
    } = params;
    const logPrefix = debugRunId ? `[CPR][${debugRunId}]` : '[CPR]';

    const scalarData = volume.imageData.getPointData().getScalars().getData() as
      | Float32Array
      | Int16Array;
    const { minValue: scalarMin, maxValue: scalarMax } = estimateScalarRange(scalarData);
    const { slope: rescaleSlope, intercept: rescaleIntercept } = getVolumeRescale(volume);
    const { bitsStored, pixelRepresentation } = getVolumePixelStorage(volume);
    const applyModalityLut = shouldApplyModalityLutForCPR(rescaleSlope, rescaleIntercept);

    const scalarType =
      (scalarData as { constructor?: { name?: string } })?.constructor?.name || 'UnknownTypedArray';
    console.log(`${logPrefix} launchCPRWorker intensity normalization decision`, {
      scalarType,
      scalarMin,
      scalarMax,
      rescaleSlope,
      rescaleIntercept,
      bitsStored,
      pixelRepresentation,
      applyModalityLut,
      modalityLutPolicy: 'APPLY_WHEN_NON_IDENTITY_RESCALE',
      aggregation,
      voiWindowWidth: CPR_PANO_DEFAULT_WINDOW_WIDTH,
      voiWindowCenter: CPR_PANO_DEFAULT_WINDOW_CENTER,
    });
    if (
      Number.isFinite(bitsStored) &&
      Number(bitsStored) > 0 &&
      Number(bitsStored) < 16 &&
      scalarMax > (1 << Number(bitsStored)) - 1
    ) {
      console.warn(`${logPrefix} Source scalar range exceeds nominal bitsStored range.`, {
        scalarMax,
        bitsStored,
        pixelRepresentation,
      });
    }

    if (!applyModalityLut && (Math.abs(rescaleSlope - 1) > 1e-6 || Math.abs(rescaleIntercept) > 1e-6)) {
      console.warn(
        `${logPrefix} applyModalityLut=false while source metadata has non-identity rescale. ` +
          'If source pixels are stored values (not HU), fixed WW/WL may cause washed-out pano.'
      );
    }
    if (!applyModalityLut && Math.abs(rescaleIntercept) > 1e-6 && scalarMin >= 0 && scalarMax <= 5000) {
      console.error(`${logPrefix} RESCALE_BYPASS_DETECTED`, {
        scalarMin,
        scalarMax,
        rescaleSlope,
        rescaleIntercept,
      });
    }

    const isSharedArrayBuffer = scalarData.buffer instanceof SharedArrayBuffer;
    const dataToSend = isSharedArrayBuffer
      ? scalarData
      : (scalarData.slice(0) as Float32Array | Int16Array);

    const serializedFrames = frames.map(f => ({
      position: Array.from(f.position) as [number, number, number],
      N_slab: Array.from(f.N_slab) as [number, number, number],
    }));

    worker.onmessage = event => {
      worker.terminate();
      const data = event.data;

      if (data.type === 'ERROR') {
        reject(new Error(`[cprWorker] ${data.message}`));
        return;
      }

      resolve({
        pixelData: data.pixelData,
        width: data.panoWidth,
        height: data.panoHeight,
        minValue: data.minValue,
        maxValue: data.maxValue,
      });
    };

    worker.onerror = err => {
      worker.terminate();
      reject(new Error(`[cprWorker] Uncaught worker error: ${err.message}`));
    };

    worker.postMessage({
      scalarData: dataToSend,
      isSharedArrayBuffer,
      dimensions: volume.imageData.getDimensions(),
      spacing: volume.imageData.getSpacing(),
      origin: volume.imageData.getOrigin(),
      direction: volume.imageData.getDirection(),
      frames: serializedFrames,
      panoWidth,
      panoHeight,
      slabHalfThicknessMm,
      slabSamples,
      aggregation,
      verticalDir,
      applyModalityLut,
      rescaleSlope,
      rescaleIntercept,
      bitsStored,
      pixelRepresentation,
      debugRunId,
    });
  });
}

function estimateScalarRange(
  scalarData: Float32Array | Int16Array
): { minValue: number; maxValue: number } {
  if (!scalarData || scalarData.length === 0) {
    return { minValue: 0, maxValue: 0 };
  }

  let minValue = Infinity;
  let maxValue = -Infinity;
  const step = Math.max(1, Math.floor(scalarData.length / 20000));

  for (let i = 0; i < scalarData.length; i += step) {
    const value = Number(scalarData[i]);
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

function getVolumeRescale(
  volume: cornerstone.Types.IImageVolume
): { slope: number; intercept: number } {
  const imageIds = (volume as cornerstone.Types.IImageVolume & { imageIds?: string[] }).imageIds;
  const firstImageId = Array.isArray(imageIds) && imageIds.length > 0 ? imageIds[0] : undefined;

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
): { bitsStored: number; pixelRepresentation: number } {
  const imageIds = (volume as cornerstone.Types.IImageVolume & { imageIds?: string[] }).imageIds;
  const firstImageId = Array.isArray(imageIds) && imageIds.length > 0 ? imageIds[0] : undefined;

  if (!firstImageId) {
    return { bitsStored: 16, pixelRepresentation: 1 };
  }

  const imagePixelModule = cornerstone.metaData.get('imagePixelModule', firstImageId) as
    | {
        bitsStored?: number;
        pixelRepresentation?: number;
      }
    | undefined;

  const bitsStored = Number(imagePixelModule?.bitsStored);
  const pixelRepresentation = Number(imagePixelModule?.pixelRepresentation);

  return {
    bitsStored: Number.isFinite(bitsStored) && bitsStored >= 1 ? bitsStored : 16,
    pixelRepresentation:
      Number.isFinite(pixelRepresentation) && (pixelRepresentation === 0 || pixelRepresentation === 1)
        ? pixelRepresentation
        : 1,
  };
}

function shouldApplyModalityLutForCPR(
  slope: number,
  intercept: number
): boolean {
  // Policy: whenever modality rescale is non-identity, apply it in CPR generation.
  // This avoids producing pano output in stored-value space.
  return Math.abs(slope - 1) > 1e-6 || Math.abs(intercept) > 1e-6;
}

function findViewportByLogicalId(
  servicesManager: any,
  logicalViewportId: string
) {
  const { cornerstoneViewportService, viewportGridService } = servicesManager.services;

  type GridViewportLike = {
    viewportOptions?: {
      viewportId?: string;
    };
  };

  const directViewport = cornerstoneViewportService.getCornerstoneViewport(logicalViewportId);
  if (directViewport) {
    return directViewport;
  }

  const gridViewports = viewportGridService.getState().viewports as
    | Map<string, GridViewportLike>
    | Record<string, GridViewportLike>
    | undefined;
  const gridEntries: Array<[string, GridViewportLike]> =
    gridViewports && typeof (gridViewports as Map<string, GridViewportLike>).entries === 'function'
      ? Array.from((gridViewports as Map<string, GridViewportLike>).entries())
      : Object.entries((gridViewports || {}) as Record<string, GridViewportLike>);

  for (const [gridViewportId, gridViewport] of gridEntries) {
    if (gridViewport?.viewportOptions?.viewportId !== logicalViewportId) {
      continue;
    }

    const mappedViewport = cornerstoneViewportService.getCornerstoneViewport(gridViewportId);
    if (mappedViewport) {
      return mappedViewport;
    }
  }

  return null;
}

function isStackViewportLike(
  viewport: cornerstone.Types.IViewport | null
): viewport is cornerstone.Types.IStackViewport {
  if (!viewport) {
    return false;
  }

  const candidate = viewport as cornerstone.Types.IStackViewport;

  return (
    typeof candidate.setStack === 'function' &&
    typeof candidate.getCurrentImageIdIndex === 'function'
  );
}

async function waitForViewportByLogicalId(
  servicesManager: any,
  logicalViewportId: string,
  timeoutMs = 12000
) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const viewport = findViewportByLogicalId(servicesManager, logicalViewportId);
    if (viewport) {
      return viewport;
    }

    await new Promise(resolve => window.setTimeout(resolve, 50));
  }

  return null;
}

async function waitForPanoStackViewport(
  servicesManager: any,
  timeoutMs = 12000
): Promise<cornerstone.Types.IStackViewport> {
  const startedAt = Date.now();
  let lastViewportType = 'none';

  while (Date.now() - startedAt < timeoutMs) {
    const viewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
    if (viewport) {
      lastViewportType = viewport.constructor?.name || 'unknown';

      if (isStackViewportLike(viewport)) {
        return viewport as cornerstone.Types.IStackViewport;
      }
    }

    await new Promise(resolve => window.setTimeout(resolve, 50));
  }

  throw new Error(
    `[CPR] cpr-pano viewport not ready within timeout. Last resolved type: ${lastViewportType}`
  );
}

function getCurrentStackImageId(viewport: cornerstone.Types.IViewport | null): string | null {
  if (!viewport || !isStackViewportLike(viewport)) {
    return null;
  }

  const imageIds = viewport.getImageIds?.();
  const rawIndex = viewport.getCurrentImageIdIndex?.();
  const index = Number(rawIndex);

  if (!Array.isArray(imageIds) || imageIds.length === 0 || !Number.isFinite(index)) {
    return null;
  }

  const safeIndex = Math.max(0, Math.min(imageIds.length - 1, Math.floor(index)));
  return imageIds[safeIndex] ?? null;
}

function toFiniteNumber(value: unknown): number | undefined {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function summarizeColormap(
  colormap: unknown
): { name?: string; opacity?: number; type?: string } | null {
  if (!colormap || typeof colormap !== 'object') {
    return null;
  }

  const colormapObj = colormap as Record<string, unknown>;
  const name =
    (typeof colormapObj.name === 'string' && colormapObj.name) ||
    (typeof colormapObj.id === 'string' && colormapObj.id) ||
    undefined;
  const opacity = toFiniteNumber(colormapObj.opacity);
  const type =
    (typeof colormapObj.type === 'string' && colormapObj.type) ||
    (typeof colormapObj.Name === 'string' && colormapObj.Name) ||
    undefined;

  return { name, opacity, type };
}

function logPanoViewportSnapshot(
  runId: string,
  stage: string,
  viewport: cornerstone.Types.IViewport | null
): void {
  if (!viewport) {
    console.log(`[CPR][${runId}] PANO_SNAPSHOT ${stage}`, {
      missingViewport: true,
    });
    return;
  }

  const currentImageId = getCurrentStackImageId(viewport);
  const properties = (viewport.getProperties?.() || {}) as {
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
    invert?: unknown;
    colormap?: unknown;
  };
  const stackViewport = viewport as cornerstone.Types.IStackViewport & {
    getCornerstoneImage?: () => {
      minPixelValue?: number;
      maxPixelValue?: number;
      slope?: number;
      intercept?: number;
      windowWidth?: number;
      windowCenter?: number;
      imageId?: string;
    } | null;
  };
  const cornerstoneImage = stackViewport.getCornerstoneImage?.() || null;
  const modalityLut = currentImageId
    ? (cornerstone.metaData.get('modalityLutModule', currentImageId) as
        | {
            rescaleSlope?: number;
            rescaleIntercept?: number;
          }
        | undefined)
    : undefined;
  const voiLut = currentImageId
    ? (cornerstone.metaData.get('voiLutModule', currentImageId) as
        | {
            windowWidth?: number[] | number;
            windowCenter?: number[] | number;
            voiLUTFunction?: string;
          }
        | undefined)
    : undefined;
  const metadataWindowWidth = Array.isArray(voiLut?.windowWidth)
    ? toFiniteNumber(voiLut?.windowWidth?.[0])
    : toFiniteNumber(voiLut?.windowWidth);
  const metadataWindowCenter = Array.isArray(voiLut?.windowCenter)
    ? toFiniteNumber(voiLut?.windowCenter?.[0])
    : toFiniteNumber(voiLut?.windowCenter);

  console.log(`[CPR][${runId}] PANO_SNAPSHOT ${stage}`, {
    viewportId: viewport.id,
    viewportType: viewport.constructor?.name || 'unknown',
    currentImageId,
    isPanoScheme: typeof currentImageId === 'string' && currentImageId.startsWith('pano://'),
    imageCount: isStackViewportLike(viewport) ? viewport.getImageIds?.().length : undefined,
    currentImageIndex: isStackViewportLike(viewport)
      ? toFiniteNumber(viewport.getCurrentImageIdIndex?.())
      : undefined,
    viewportProperties: {
      voiRange: properties.voiRange,
      VOILUTFunction: properties.VOILUTFunction,
      invert: properties.invert,
      colormap: summarizeColormap(properties.colormap),
    },
    cornerstoneImage: cornerstoneImage
      ? {
          imageId: cornerstoneImage.imageId,
          minPixelValue: cornerstoneImage.minPixelValue,
          maxPixelValue: cornerstoneImage.maxPixelValue,
          slope: cornerstoneImage.slope,
          intercept: cornerstoneImage.intercept,
          windowWidth: cornerstoneImage.windowWidth,
          windowCenter: cornerstoneImage.windowCenter,
        }
      : null,
    metadata: {
      modalityLutSlope: toFiniteNumber(modalityLut?.rescaleSlope),
      modalityLutIntercept: toFiniteNumber(modalityLut?.rescaleIntercept),
      voiWindowWidth: metadataWindowWidth,
      voiWindowCenter: metadataWindowCenter,
      voiLUTFunction: voiLut?.voiLUTFunction,
    },
  });
}

function applyPanoDisplaySettings(
  runId: string,
  stage: string,
  viewport: cornerstone.Types.IViewport | null,
  adaptiveVoi: PanoVoiSettings
): void {
  if (!viewport || !isStackViewportLike(viewport)) {
    console.warn(`[CPR][${runId}] applyPanoDisplaySettings skipped at ${stage}: stack viewport missing`);
    return;
  }

  viewport.setProperties({
    // Prevent Cornerstone from recomputing VOI and overriding our panoramic range.
    isComputedVOI: false,
    voiRange: {
      lower: adaptiveVoi.lower,
      upper: adaptiveVoi.upper,
    },
    invert: false,
    colormap: undefined,
    VOILUTFunction: 'LINEAR_EXACT',
  });
}

function cloneCameraState(
  viewport: cornerstone.Types.IViewport
): Record<string, cornerstone.Types.Point3 | number | boolean | undefined> | null {
  const camera = viewport?.getCamera?.();
  if (!camera) {
    return null;
  }

  return {
    focalPoint: Array.isArray(camera.focalPoint) ? [...camera.focalPoint] : undefined,
    position: Array.isArray(camera.position) ? [...camera.position] : undefined,
    viewPlaneNormal: Array.isArray(camera.viewPlaneNormal)
      ? [...camera.viewPlaneNormal]
      : undefined,
    viewUp: Array.isArray(camera.viewUp) ? [...camera.viewUp] : undefined,
    parallelScale: camera.parallelScale,
    parallelProjection: camera.parallelProjection,
  };
}

type AnnotationLike = {
  annotationUID?: string;
  modifiedTimestamp?: number;
  metadata?: {
    toolName?: string;
  };
  data?: {
    contour?: {
      closed?: boolean;
    };
    handles?: {
      activeHandleIndex?: number | null;
      points?: PointLike[];
    };
  };
  highlighted?: boolean;
  invalidated?: boolean;
};

function getAllSplineAnnotations(): AnnotationLike[] {
  const allAnnotations = cornerstoneTools.annotation.state.getAllAnnotations?.() || [];
  return (allAnnotations as AnnotationLike[]).filter(
    annotation => annotation?.metadata?.toolName === 'SplineROI'
  );
}

function removeSplineAnnotationsExcept(keepAnnotationUID?: string | null): void {
  const splines = getAllSplineAnnotations();

  splines.forEach(annotation => {
    const annotationUID = annotation?.annotationUID;
    if (!annotationUID || annotationUID === keepAnnotationUID) {
      return;
    }

    cornerstoneTools.annotation.state.removeAnnotation(annotationUID);
  });
}

function removeAllSplineAnnotations(): void {
  removeSplineAnnotationsExcept(null);
}

function getLatestAnnotation(annotations: AnnotationLike[]): AnnotationLike | null {
  if (!Array.isArray(annotations) || annotations.length === 0) {
    return null;
  }

  let latest = annotations[0];
  let latestTimestamp = Number(latest?.modifiedTimestamp ?? 0);

  for (let i = 1; i < annotations.length; i++) {
    const candidate = annotations[i];
    const candidateTimestamp = Number(candidate?.modifiedTimestamp ?? 0);
    if (candidateTimestamp >= latestTimestamp) {
      latest = candidate;
      latestTimestamp = candidateTimestamp;
    }
  }

  return latest;
}

function getLatestSplineAnnotation(axialElement?: HTMLDivElement | null): AnnotationLike | null {
  if (!axialElement) {
    return null;
  }

  const localAnnotations =
    cornerstoneTools.annotation.state.getAnnotations('SplineROI', axialElement) || [];
  return getLatestAnnotation(localAnnotations as AnnotationLike[]);
}

function clearActiveManipulationPreservingSpline(
  axialElement: HTMLDivElement,
  annotation: AnnotationLike
): void {
  const annotationUID = annotation?.annotationUID;
  const canceledAnnotationUID = cornerstoneTools.cancelActiveManipulations?.(axialElement);

  if (!annotationUID || canceledAnnotationUID !== annotationUID) {
    return;
  }

  const existingAnnotation = cornerstoneTools.annotation.state.getAnnotation?.(annotationUID);
  if (existingAnnotation) {
    return;
  }

  try {
    if (annotation?.data?.contour) {
      annotation.data.contour.closed = false;
    }
    annotation.highlighted = false;
    annotation.invalidated = true;
    if (annotation?.data?.handles) {
      annotation.data.handles.activeHandleIndex = null;
    }

    const annotationToRestore = annotation as Parameters<
      typeof cornerstoneTools.annotation.state.addAnnotation
    >[0];
    cornerstoneTools.annotation.state.addAnnotation(annotationToRestore, axialElement);
    cornerstoneTools.utilities.triggerAnnotationRender(axialElement);
  } catch (error) {
    console.warn('[CPR] Failed to restore spline annotation after interaction cancel.', error);
  }
}

function setCrossSectionForFrame(frame: CPRFrame, servicesManager: any): void {
  const crossViewport = findViewportByLogicalId(servicesManager, 'cpr-crosssection');
  if (!crossViewport) {
    return;
  }

  const previousCamera = crossViewport.getCamera?.();
  const basis = sanitizeCameraBasis(
    Array.from(frame.T) as [number, number, number],
    Array.from(frame.S) as [number, number, number],
    previousCamera
  );
  const previousParallelScale = Number(previousCamera?.parallelScale);
  const parallelScale =
    Number.isFinite(previousParallelScale) && previousParallelScale > 0
      ? previousParallelScale
      : 20;

  crossViewport.setCamera({
    focalPoint: Array.from(frame.position) as [number, number, number],
    viewPlaneNormal: basis.viewPlaneNormal,
    viewUp: basis.viewUp,
    parallelScale,
    parallelProjection: true,
  });

  crossViewport.render();
}

function initializeCrossSection(
  frames: CPRFrame[],
  servicesManager: any
): void {
  const { syncGroupService } = servicesManager.services;

  const crossViewport = findViewportByLogicalId(servicesManager, 'cpr-crosssection');
  if (!crossViewport) {
    console.error('[CPR] cpr-crosssection viewport not found after HP switch.');
    return;
  }

  const initialFrameIndex = Math.max(
    0,
    Math.min(cprStateService.getCurrentFrameIndex(), Math.max(0, frames.length - 1))
  );
  setCrossSectionForFrame(frames[initialFrameIndex], servicesManager);

  // Hardening #2: idempotency guard for sync group insertion.
  const syncId = 'cpr-crosssection-sync';
  const existingSync = syncGroupService.getSynchronizer(syncId);
  const renderingEngineId = crossViewport.getRenderingEngine().id;
  const crossViewportId = crossViewport.id;
  const alreadyInSync =
    !!existingSync &&
    (existingSync.hasTargetViewport(renderingEngineId, crossViewportId) ||
      existingSync.hasSourceViewport(renderingEngineId, crossViewportId));

  if (!alreadyInSync) {
    syncGroupService.addViewportToSyncGroup(crossViewportId, renderingEngineId, {
      type: 'imageslice',
      id: syncId,
      source: false,
      target: true,
    });
  }
}

interface UseCPROrchestratorProps {
  servicesManager: any;
  commandsManager: any;
  sourceViewportId?: string;
  panoWidth?: number;
  panoHeight?: number;
  slabHalfThicknessMm?: number;
  slabSamples?: number;
  aggregation?: 'MIP' | 'MEAN';
}

interface UseCPROrchestratorReturn {
  onDone: () => Promise<void>;
  onRedraw: () => Promise<void>;
  onSliderChange: (frameIndex: number) => void;
  isGenerating: boolean;
  error: string | null;
}

type PointObject = { x: number; y: number; z: number };
type PointLike = [number, number, number] | PointObject;

function getVolumeVerticalDirection(
  direction?: ArrayLike<number> | null
): [number, number, number] {
  if (!direction || direction.length < 9) {
    return [0, 0, 1];
  }

  const zAxis: [number, number, number] = [
    Number(direction[6] ?? 0),
    Number(direction[7] ?? 0),
    Number(direction[8] ?? 1),
  ];

  return normalize3(zAxis);
}

function toXYZTuple(p: PointLike): [number, number, number] {
  if (Array.isArray(p) && p.length >= 3) {
    return [p[0], p[1], p[2]];
  }

  const pointObj = p as PointObject;
  return [pointObj.x, pointObj.y, pointObj.z];
}

export function useCPROrchestrator({
  servicesManager,
  commandsManager,
  sourceViewportId,
  panoWidth = 800,
  panoHeight = 400,
  slabHalfThicknessMm = 7,
  slabSamples = 21,
  aggregation = 'MEAN',
}: UseCPROrchestratorProps): UseCPROrchestratorReturn {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hpSubscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const hpTimeoutRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);
  const lastSetupCleanupRef = useRef<string | null>(null);

  const clearProtocolListener = useCallback(() => {
    if (hpSubscriptionRef.current) {
      hpSubscriptionRef.current.unsubscribe();
      hpSubscriptionRef.current = null;
    }

    if (hpTimeoutRef.current != null) {
      window.clearTimeout(hpTimeoutRef.current);
      hpTimeoutRef.current = null;
    }
  }, []);

  const ensureSetupViewportInteraction = useCallback(async () => {
    const axialViewport = await waitForViewportByLogicalId(servicesManager, 'cpr-axial', 4000);
    if (!axialViewport || !isMountedRef.current) {
      return;
    }

    commandsManager.runCommand('setViewportActive', { viewportId: axialViewport.id });
    commandsManager.runCommand('setToolActive', {
      toolName: 'SplineROI',
      toolGroupId: 'mpr',
    });
  }, [servicesManager, commandsManager]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      clearProtocolListener();
    };
  }, [clearProtocolListener]);

  useEffect(() => {
    const { hangingProtocolService } = servicesManager.services;

    const maybeCleanupForSetupStage = () => {
      const { protocolId, stageIndex } = hangingProtocolService.getState();
      const stageKey = `${protocolId}:${stageIndex}`;

      if (protocolId === 'cpr' && stageIndex === 0) {
        if (lastSetupCleanupRef.current === stageKey) {
          return;
        }

        lastSetupCleanupRef.current = stageKey;
        const setupAxialViewport = findViewportByLogicalId(servicesManager, 'cpr-axial');
        if (setupAxialViewport?.element) {
          cornerstoneTools.cancelActiveManipulations?.(setupAxialViewport.element);
        }
        cprStateService.clear();
        clearPanoImageCache();
        removeAllSplineAnnotations();

        if (isMountedRef.current) {
          setError(null);
          setIsGenerating(false);
        }

        void ensureSetupViewportInteraction();
        return;
      }

      lastSetupCleanupRef.current = null;
    };

    maybeCleanupForSetupStage();

    const protocolAppliedEvent =
      hangingProtocolService.EVENTS.PROTOCOL_APPLIED ||
      hangingProtocolService.EVENTS.PROTOCOL_CHANGED;

    if (!protocolAppliedEvent) {
      return;
    }

    const subscription = hangingProtocolService.subscribe(
      protocolAppliedEvent,
      maybeCleanupForSetupStage
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [servicesManager, ensureSetupViewportInteraction]);

  const onDone = useCallback(async () => {
    setIsGenerating(true);
    setError(null);
    const debugRunId = `cpr-${Date.now().toString(36)}`;
    console.log(`[CPR][${debugRunId}] onDone started`);

    try {
      const { cornerstoneViewportService, hangingProtocolService, viewportGridService } =
        servicesManager.services;

      const fallbackViewportId =
        sourceViewportId || viewportGridService.getState().activeViewportId;
      const axialViewport =
        (sourceViewportId &&
          (findViewportByLogicalId(servicesManager, sourceViewportId) ||
            cornerstoneViewportService.getCornerstoneViewport(sourceViewportId))) ||
        findViewportByLogicalId(servicesManager, 'cpr-axial') ||
        cornerstoneViewportService.getCornerstoneViewport('cpr-axial') ||
        (fallbackViewportId &&
          cornerstoneViewportService.getCornerstoneViewport(fallbackViewportId));

      if (!axialViewport) {
        throw new Error('No valid source viewport found for CPR generation.');
      }

      const preservedAxialCamera = cloneCameraState(axialViewport);
      const axialElement = axialViewport.element;
      if (!axialElement) {
        throw new Error('Source viewport element is not ready.');
      }

      let latestAnnotation = getLatestSplineAnnotation(axialElement);
      if (!latestAnnotation) {
        throw new Error('No SplineROI annotation found. Please draw the jaw arch first.');
      }

      clearActiveManipulationPreservingSpline(axialElement, latestAnnotation);

      commandsManager.runCommand('setToolActive', { toolName: 'Pan', toolGroupId: 'mpr' });

      latestAnnotation =
        (latestAnnotation?.annotationUID &&
          cornerstoneTools.annotation.state.getAnnotation?.(latestAnnotation.annotationUID)) ||
        getLatestSplineAnnotation(axialElement) ||
        latestAnnotation;
      const latestAnnotationUID = latestAnnotation?.annotationUID ?? null;

      // Hardening #4: tolerate both [x,y,z] arrays and {x,y,z} point objects.
      const rawPoints = (latestAnnotation?.data?.handles?.points ?? []) as PointLike[];
      const rawControlPoints: [number, number, number][] = rawPoints
        .map(toXYZTuple)
        .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]));

      if (rawControlPoints.length < 2) {
        throw new Error('Arch annotation needs at least 2 valid points.');
      }

      const volumeActors = (axialViewport as cornerstone.Types.IVolumeViewport).getActors();
      if (!volumeActors || volumeActors.length === 0) {
        throw new Error('No volume loaded in the selected source viewport.');
      }

      const sourceVolumeId = volumeActors[0].uid;
      const volume = cornerstone.cache.getVolume(sourceVolumeId);
      if (!volume) {
        throw new Error(`Volume ${sourceVolumeId} not found in cache.`);
      }
      console.log(`[CPR][${debugRunId}] source volume resolved`, {
        sourceVolumeId,
      });

      const verticalDir = getVolumeVerticalDirection(volume.imageData?.getDirection?.());
      const { positions, tangents } = buildArcLengthSpline(rawControlPoints, panoWidth);
      const frames = buildRMFFrames(positions, tangents);
      const panoImageId = createPanoImageId();
      console.log(`[CPR][${debugRunId}] generated panoImageId`, { panoImageId });

      // Precompute pano before stage switch so cpr-pano does not visibly flash
      // the source stack before the pano:// image arrives.
      const panoWorkerResult = await launchCPRWorker({
        volume,
        frames,
        panoWidth,
        panoHeight,
        slabHalfThicknessMm,
        slabSamples,
        aggregation,
        verticalDir,
        debugRunId,
      });

      const panoDebugSummary = summarizeFloatBufferForDebug(panoWorkerResult.pixelData);
      const adaptiveVoi = computeAdaptivePanoVoi(
        panoDebugSummary,
        panoWorkerResult.minValue,
        panoWorkerResult.maxValue
      );
      if (panoDebugSummary) {
        console.log(`[CPR][${debugRunId}] Worker pano output stats (sampled)`, {
          min: panoDebugSummary.min,
          p01: panoDebugSummary.p01,
          p50: panoDebugSummary.p50,
          p99: panoDebugSummary.p99,
          max: panoDebugSummary.max,
          fractionBelowMinus950: panoDebugSummary.fractionBelowMinus950,
          fractionAbove3000: panoDebugSummary.fractionAbove3000,
          sampledCount: panoDebugSummary.sampledCount,
          configuredVoiLower: adaptiveVoi.lower,
          configuredVoiUpper: adaptiveVoi.upper,
        });

        if (panoDebugSummary.fractionAbove3000 > 0.6 || panoDebugSummary.fractionBelowMinus950 > 0.6) {
          console.warn(
            `[CPR][${debugRunId}] Majority of pano samples are outside reference band [-950, 3000]. ` +
              `Adaptive VOI is [${adaptiveVoi.lower.toFixed(0)}, ${adaptiveVoi.upper.toFixed(0)}].`
          );
        }
      }

      // Keep exactly one arch annotation to avoid stale/ghost spline overlays.
      removeSplineAnnotationsExcept(latestAnnotationUID);
      cprStateService.setArchData(rawControlPoints, frames, sourceVolumeId, latestAnnotationUID);

      clearProtocolListener();

      let protocolAppliedHandled = false;
      const onProtocolApplied = async () => {
        if (protocolAppliedHandled) {
          return;
        }
        protocolAppliedHandled = true;

        clearProtocolListener();

        try {
          // Restore axial camera asynchronously so pano swap is not delayed.
          void (async () => {
            try {
              const axialViewportAfterSwitch = await waitForViewportByLogicalId(
                servicesManager,
                'cpr-axial',
                4000
              );
              if (axialViewportAfterSwitch && preservedAxialCamera) {
                axialViewportAfterSwitch.setCamera(
                  preservedAxialCamera as Partial<ReturnType<typeof axialViewportAfterSwitch.getCamera>>
                );
                axialViewportAfterSwitch.render();
                cornerstoneTools.utilities.triggerAnnotationRender(axialViewportAfterSwitch.element);
              }
            } catch (axialRestoreError) {
              console.warn(
                `[CPR][${debugRunId}] Failed to restore axial camera after stage switch`,
                axialRestoreError
              );
            }
          })();

          clearPanoImageCache();
          setPanoImagePayload(panoImageId, {
            pixelData: panoWorkerResult.pixelData,
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            slope: 1,
            intercept: 0,
          });

          console.log(`[CPR][${debugRunId}] Pano payload metadata pushed to loader`, {
            panoImageId,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            voiLower: adaptiveVoi.lower,
            voiUpper: adaptiveVoi.upper,
            slope: 1,
            intercept: 0,
          });

          const panoViewport = await waitForPanoStackViewport(servicesManager);
          logPanoViewportSnapshot(debugRunId, 'before-setStack', panoViewport);
          await panoViewport.setStack([panoImageId], 0);
          logPanoViewportSnapshot(debugRunId, 'after-setStack', panoViewport);
          applyPanoDisplaySettings(debugRunId, 'after-setStack', panoViewport, adaptiveVoi);
          logPanoViewportSnapshot(debugRunId, 'after-setProperties', panoViewport);
          panoViewport.render();
          logPanoViewportSnapshot(debugRunId, 'after-render', panoViewport);
          // Re-assert VOI after first rendered frame in case stack initialization re-applies defaults.
          if (panoViewport.element) {
            let reappliedOnRender = false;
            const onFirstImageRendered = () => {
              if (reappliedOnRender) {
                return;
              }
              reappliedOnRender = true;
              panoViewport.element.removeEventListener(
                cornerstone.Enums.Events.IMAGE_RENDERED,
                onFirstImageRendered
              );

              const liveViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
              applyPanoDisplaySettings(
                debugRunId,
                'after-first-image-rendered',
                liveViewport,
                adaptiveVoi
              );
              liveViewport?.render?.();
              logPanoViewportSnapshot(
                debugRunId,
                'after-first-image-rendered-reapply',
                liveViewport
              );
            };

            panoViewport.element.addEventListener(
              cornerstone.Enums.Events.IMAGE_RENDERED,
              onFirstImageRendered
            );
          }
          [120, 500, 1500].forEach(delayMs => {
            window.setTimeout(() => {
              const livePanoViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
              applyPanoDisplaySettings(
                debugRunId,
                `post-render+${delayMs}ms-reapply`,
                livePanoViewport,
                adaptiveVoi
              );
              livePanoViewport?.render?.();
              logPanoViewportSnapshot(debugRunId, `post-render+${delayMs}ms`, livePanoViewport);
            }, delayMs);
          });

          commandsManager.runCommand('setToolActive', {
            toolName: 'CPRCursor',
            toolGroupId: 'cprPano',
          });
          cornerstoneTools.utilities.triggerAnnotationRender(panoViewport.element);

          initializeCrossSection(frames, servicesManager);
        } catch (innerErr) {
          const msg = innerErr instanceof Error ? innerErr.message : String(innerErr);
          console.error(`[CPR][${debugRunId}] Pipeline failed after HP switch:`, msg);
          if (isMountedRef.current) {
            setError(msg);
          }
        } finally {
          if (isMountedRef.current) {
            setIsGenerating(false);
          }
        }
      };

      const protocolAppliedEvent =
        hangingProtocolService.EVENTS.PROTOCOL_APPLIED ||
        hangingProtocolService.EVENTS.PROTOCOL_CHANGED;

      if (!protocolAppliedEvent) {
        throw new Error('[CPR] No supported hanging protocol event found.');
      }

      hpSubscriptionRef.current = hangingProtocolService.subscribe(
        protocolAppliedEvent,
        onProtocolApplied
      );

      // Hardening #3: safety timeout in case protocol change event never arrives.
      hpTimeoutRef.current = window.setTimeout(() => {
        clearProtocolListener();
        if (isMountedRef.current) {
          const timeoutMsg =
            '[CPR] Timed out waiting for hanging protocol change after stage switch.';
          console.error(`[CPR][${debugRunId}] ${timeoutMsg}`);
          setError(timeoutMsg);
          setIsGenerating(false);
        }
      }, 12000);

      console.log(`[CPR][${debugRunId}] switching to CPR stage 1`);
      commandsManager.runCommand('setHangingProtocol', {
        protocolId: 'cpr',
        stageIndex: 1,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[CPR][${debugRunId}] onDone failed:`, msg);
      if (isMountedRef.current) {
        setError(msg);
        setIsGenerating(false);
      }
    }
  }, [
    servicesManager,
    commandsManager,
    sourceViewportId,
    panoWidth,
    panoHeight,
    slabHalfThicknessMm,
    slabSamples,
    aggregation,
    clearProtocolListener,
  ]);

  const onRedraw = useCallback(async () => {
    setError(null);
    setIsGenerating(false);
    clearProtocolListener();

    const currentAxialViewport = findViewportByLogicalId(servicesManager, 'cpr-axial');
    if (currentAxialViewport?.element) {
      cornerstoneTools.cancelActiveManipulations?.(currentAxialViewport.element);
    }

    cprStateService.clear();
    clearPanoImageCache();
    removeAllSplineAnnotations();

    try {
      const { hangingProtocolService } = servicesManager.services;
      const protocolAppliedEvent =
        hangingProtocolService.EVENTS.PROTOCOL_APPLIED ||
        hangingProtocolService.EVENTS.PROTOCOL_CHANGED;

      if (!protocolAppliedEvent) {
        throw new Error('[CPR] No supported hanging protocol event found.');
      }

      let protocolAppliedHandled = false;
      hpSubscriptionRef.current = hangingProtocolService.subscribe(
        protocolAppliedEvent,
        async () => {
          if (protocolAppliedHandled) {
            return;
          }
          protocolAppliedHandled = true;

          clearProtocolListener();

          try {
            const axialViewport = await waitForViewportByLogicalId(
              servicesManager,
              'cpr-axial',
              4000
            );
            if (axialViewport) {
              commandsManager.runCommand('setViewportActive', { viewportId: axialViewport.id });
              axialViewport.render();
            }

            commandsManager.runCommand('setToolActive', {
              toolName: 'SplineROI',
              toolGroupId: 'mpr',
            });
          } catch (innerErr) {
            const msg = innerErr instanceof Error ? innerErr.message : String(innerErr);
            if (isMountedRef.current) {
              setError(msg);
            }
          }
        }
      );

      hpTimeoutRef.current = window.setTimeout(() => {
        clearProtocolListener();
        if (isMountedRef.current) {
          const timeoutMsg = '[CPR] Timed out while re-entering CPR setup stage.';
          setError(timeoutMsg);
        }
      }, 6000);

      commandsManager.runCommand('setHangingProtocol', {
        protocolId: 'cpr',
        stageIndex: 0,
        reset: true,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (isMountedRef.current) {
        setError(msg);
      }
    }
  }, [servicesManager, commandsManager, clearProtocolListener]);

  const onSliderChange = useCallback(
    (frameIndex: number) => {
      if (!cprStateService.hasData()) {
        return;
      }

      const frames = cprStateService.getFrames();
      const clampedIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
      cprStateService.setCurrentFrameIndex(clampedIndex);
      setCrossSectionForFrame(frames[clampedIndex], servicesManager);
    },
    [servicesManager]
  );

  useEffect(() => {
    const onCrossSectionSync = (evt: Event) => {
      const detail = (evt as CustomEvent<CPRCrossSectionSyncDetail>).detail;
      if (!detail || !Number.isFinite(detail.frameIndex)) {
        return;
      }

      onSliderChange(detail.frameIndex);
    };

    cornerstone.eventTarget.addEventListener(CPR_CROSSSECTION_SYNC_EVENT, onCrossSectionSync);

    return () => {
      cornerstone.eventTarget.removeEventListener(CPR_CROSSSECTION_SYNC_EVENT, onCrossSectionSync);
    };
  }, [onSliderChange]);

  return { onDone, onRedraw, onSliderChange, isGenerating, error };
}
