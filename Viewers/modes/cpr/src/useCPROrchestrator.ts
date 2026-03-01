import { useCallback, useEffect, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { vec3 } from 'gl-matrix';

import { cprStateService } from './CPRStateService';
import { setPanoImagePayload, createPanoImageId, clearPanoImageCache } from './panoImageLoader';
import { buildRMFFrames } from './cprMath';
import { CPR_CROSSSECTION_SYNC_EVENT, CPRCrossSectionSyncDetail } from './cprEvents';
import type { CPRFrame } from './cprMath';

const CPR_PANO_DEFAULT_WINDOW_WIDTH = 3000;
const CPR_PANO_DEFAULT_WINDOW_CENTER = 600;
const CPR_PANO_MAX_DIMENSION = 4096;
const CPR_PANO_DEFAULT_VERTICAL_HALF_MM = 20;
const CPR_PANO_MAX_VERTICAL_HALF_MM = 26;
const CPR_PANO_TARGET_ASPECT = 1.35;

interface FloatBufferDebugSummary {
  sampledCount: number;
  min: number;
  max: number;
  p01: number;
  p50: number;
  p99: number;
  fractionBelowMinus950: number;
  fractionAbove3000: number;
  meanAbsDelta: number;
}

interface PanoVoiSettings {
  lower: number;
  upper: number;
  windowWidth: number;
  windowCenter: number;
}

function isHuLikeRange(minValue: number, maxValue: number): boolean {
  return (
    Number.isFinite(minValue) &&
    Number.isFinite(maxValue) &&
    minValue >= -5000 &&
    maxValue <= 7000
  );
}

function isLikelyStoredValueRange(
  minValue: number,
  maxValue: number,
  bitsStored: number,
  pixelRepresentation: number
): boolean {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return false;
  }

  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 && bitsStored <= 31 ? Math.floor(bitsStored) : 16;
  const isSigned = Number(pixelRepresentation) === 1;

  if (isSigned) {
    const signedMin = safeBitsStored < 31 ? -(1 << (safeBitsStored - 1)) : Number.MIN_SAFE_INTEGER;
    const signedMax =
      safeBitsStored < 31 ? (1 << (safeBitsStored - 1)) - 1 : Number.MAX_SAFE_INTEGER;
    return minValue >= signedMin - 8 && maxValue <= signedMax + 8;
  }

  const unsignedMax = safeBitsStored < 31 ? (1 << safeBitsStored) - 1 : Number.MAX_SAFE_INTEGER;
  return minValue >= -1 && maxValue <= unsignedMax + 8;
}

function isSeverelyCorruptedPanoOutput(summary: FloatBufferDebugSummary | null): boolean {
  if (!summary || summary.sampledCount < 100) {
    return false;
  }

  const dominantLowBand =
    summary.fractionBelowMinus950 > 0.85 && summary.p50 < -2500 && summary.max > 10000;
  const dominantHighBand =
    summary.fractionAbove3000 > 0.85 && summary.p50 > 5000 && summary.min < -2000;

  return dominantLowBand || dominantHighBand;
}

function isLikelyPoorPanoQuality(summary: FloatBufferDebugSummary | null): boolean {
  if (!summary || summary.sampledCount < 100) {
    return true;
  }

  if (isSeverelyCorruptedPanoOutput(summary)) {
    return true;
  }

  const robustSpan = summary.p99 - summary.p01;
  const hasExtremeOutliers = summary.min < -9000 || summary.max > 14000;
  const looksMostlySaturatedHigh = summary.fractionAbove3000 > 0.6;
  const isLowContrast = Number.isFinite(robustSpan) && robustSpan < 700;
  const hasSplitTailDistribution =
    summary.fractionBelowMinus950 > 0.5 && summary.fractionAbove3000 > 0.2;
  const hasMedianOutOfTypicalRange = summary.p50 < -1800 || summary.p50 > 2600;
  const hasStrongSpeckleNoise = summary.meanAbsDelta > 780;

  return (
    hasExtremeOutliers ||
    looksMostlySaturatedHigh ||
    isLowContrast ||
    hasSplitTailDistribution ||
    hasMedianOutOfTypicalRange ||
    hasStrongSpeckleNoise
  );
}

function scorePanoQuality(summary: FloatBufferDebugSummary | null): number {
  if (!summary || summary.sampledCount < 100) {
    return -Infinity;
  }

  const robustSpan = summary.p99 - summary.p01;
  let score = 0;

  if (Number.isFinite(robustSpan)) {
    if (robustSpan >= 1500 && robustSpan <= 6500) {
      score += 4;
    } else if (robustSpan >= 900 && robustSpan <= 9000) {
      score += 2;
    } else {
      score -= 2;
    }
  }

  score += Math.max(-4, 3 - summary.fractionAbove3000 * 8);
  score += summary.max > 14000 ? -3 : 2;
  score += summary.min < -9000 ? -3 : 2;
  if (summary.p50 >= -700 && summary.p50 <= 1700) {
    score += 4;
  } else {
    score -= Math.min(6, Math.abs(summary.p50 - 500) / 600);
  }

  if (summary.fractionBelowMinus950 > 0.5 && summary.fractionAbove3000 > 0.2) {
    score -= 5;
  }

  if (Number.isFinite(robustSpan) && robustSpan > 10000) {
    score -= Math.min(6, (robustSpan - 10000) / 1200);
  }

  if (summary.meanAbsDelta <= 380) {
    score += 3;
  } else if (summary.meanAbsDelta <= 620) {
    score += 1;
  } else {
    score -= Math.min(8, (summary.meanAbsDelta - 620) / 120);
  }

  return score;
}

function applyLinearRescaleToPixelData(
  source: Float32Array,
  slope: number,
  intercept: number
): { pixelData: Float32Array; minValue: number; maxValue: number } | null {
  const safeSlope = Number.isFinite(slope) && Math.abs(Number(slope)) > 1e-8 ? Number(slope) : 1;
  const safeIntercept = Number.isFinite(intercept) ? Number(intercept) : 0;
  const hasNonIdentityRescale = Math.abs(safeSlope - 1) > 1e-6 || Math.abs(safeIntercept) > 1e-6;

  if (!hasNonIdentityRescale || !source || source.length === 0) {
    return null;
  }

  const converted = new Float32Array(source.length);
  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < source.length; i++) {
    const value = Number(source[i]);
    const transformed = Number.isFinite(value) ? value * safeSlope + safeIntercept : -1000;
    const safeValue = Number.isFinite(transformed) ? transformed : -1000;
    converted[i] = safeValue;
    if (safeValue < minValue) {
      minValue = safeValue;
    }
    if (safeValue > maxValue) {
      maxValue = safeValue;
    }
  }

  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return null;
  }

  return {
    pixelData: converted,
    minValue,
    maxValue,
  };
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
  let lastSampleValue: number | null = null;
  let absDeltaAccum = 0;
  let absDeltaCount = 0;

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
    if (lastSampleValue !== null) {
      absDeltaAccum += Math.abs(value - lastSampleValue);
      absDeltaCount += 1;
    }
    lastSampleValue = value;
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
    meanAbsDelta: absDeltaCount > 0 ? absDeltaAccum / absDeltaCount : 0,
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
  const padding = Math.max(30, robustSpan * 0.05);
  lower -= padding;
  upper += padding;

  if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) {
    lower = fallbackLower;
    upper = fallbackUpper;
  }

  const windowWidth = Math.max(1, upper - lower);
  const windowCenter = lower + windowWidth / 2;

  // If robust VOI lands outside plausible HU-like bounds with split tails,
  // avoid hard fallback to fixed dental VOI (which can produce white/black
  // banding). Instead use a bounded robust window around the median.
  const hasSplitOutliers =
    !!summary &&
    summary.fractionBelowMinus950 > 0.4 &&
    summary.fractionAbove3000 > 0.15;
  const isExtremeWindow = lower < -5000 || upper > 7000;
  if (hasSplitOutliers && isExtremeWindow) {
    const robustCenter = Number.isFinite(summary?.p50) ? Number(summary?.p50) : windowCenter;
    const boundedWidth = Math.max(2500, Math.min(12000, robustSpan * 1.2));
    const splitCappedWidth = Math.min(boundedWidth, 3200);
    const splitCappedLower = robustCenter - splitCappedWidth / 2;
    const splitCappedUpper = robustCenter + splitCappedWidth / 2;
    return {
      lower: splitCappedLower,
      upper: splitCappedUpper,
      windowWidth: splitCappedWidth,
      windowCenter: robustCenter,
    };
  }

  // Guardrail: if the data appears HU-like, avoid very narrow windows that
  // can make CPR pano look "washed white with black speckles".
  const looksLikeHU = isHuLikeRange(rangeLow, rangeHigh);

  if (looksLikeHU) {
    const MIN_HU_WINDOW_WIDTH = 2300;
    const MAX_DENTAL_WINDOW_WIDTH = 3200;
    const widthWithMin = Math.max(windowWidth, MIN_HU_WINDOW_WIDTH);
    const cappedWidth = Math.min(widthWithMin, MAX_DENTAL_WINDOW_WIDTH);
    const dentalCenterMin = 300;
    const dentalCenterMax = 800;
    const biasedCenter =
      windowCenter < dentalCenterMin
        ? windowCenter + (dentalCenterMin - windowCenter) * 0.4
        : windowCenter > dentalCenterMax
          ? windowCenter - (windowCenter - dentalCenterMax) * 0.4
          : windowCenter;
    const cappedLower = biasedCenter - cappedWidth / 2;
    const cappedUpper = biasedCenter + cappedWidth / 2;
    return {
      lower: cappedLower,
      upper: cappedUpper,
      windowWidth: cappedWidth,
      windowCenter: biasedCenter,
    };
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

function toPositiveFinite(value: unknown, fallback: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback;
  }

  return numeric;
}

function clampPanoDimension(value: unknown): number {
  const numeric = Number(value);
  const safeValue = Number.isFinite(numeric) ? Math.round(numeric) : 2;
  return Math.max(2, Math.min(CPR_PANO_MAX_DIMENSION, safeValue));
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const out = vec3.create();
  const inVec = vec3.fromValues(v[0], v[1], v[2]);
  if (!Number.isFinite(inVec[0]) || !Number.isFinite(inVec[1]) || !Number.isFinite(inVec[2])) {
    return [1, 0, 0];
  }
  if (vec3.length(inVec) < 1e-8) {
    return [1, 0, 0];
  }
  vec3.normalize(out, inVec);
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

function computeSplineTotalArcLength(rawPoints: [number, number, number][]): number {
  if (rawPoints.length < 2) {
    return 0;
  }

  const extended = addPhantomEndpoints(rawPoints);
  const nSegments = extended.length - 3;
  if (nSegments <= 0) {
    return 0;
  }

  const FINE_STEPS = 2000;
  const stepsPerSegment = Math.max(1, Math.floor(FINE_STEPS / nSegments));
  let prevPt = catmullRomPoint(extended[0], extended[1], extended[2], extended[3], 0);
  let totalArcLength = 0;

  for (let seg = 0; seg < nSegments; seg++) {
    const Pm1 = extended[seg];
    const P0 = extended[seg + 1];
    const P1 = extended[seg + 2];
    const P2 = extended[seg + 3];

    for (let step = 1; step <= stepsPerSegment; step++) {
      const localT = step / stepsPerSegment;
      const pt = catmullRomPoint(Pm1, P0, P1, P2, localT);
      totalArcLength += dist3(prevPt, pt);
      prevPt = pt;
    }
  }

  return Number.isFinite(totalArcLength) && totalArcLength > 0 ? totalArcLength : 0;
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

interface CPRWorkerLaunchResult {
  pixelData: Float32Array;
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  effectiveIsPreScaled: boolean;
  rescaleSlope: number;
  rescaleIntercept: number;
  bitsStored: number;
  pixelRepresentation: number;
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
  forceApplyModalityLut?: boolean;
  modalityLutOverride?: boolean;
  forceDisableStoredValueNormalization?: boolean;
  verticalHalfMm?: number;
  debugRunId?: string;
}): Promise<CPRWorkerLaunchResult> {
  return new Promise((resolve, reject) => {
    // @ts-expect-error Vite/webpack handles import.meta.url worker URLs at build time.
    const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });

    const {
      volume,
      frames,
      panoWidth,
      panoHeight: requestedPanoHeight,
      slabHalfThicknessMm,
      slabSamples,
      aggregation,
      verticalDir,
      forceApplyModalityLut,
      modalityLutOverride,
      forceDisableStoredValueNormalization,
      verticalHalfMm,
      debugRunId,
    } = params;
    const logPrefix = debugRunId ? `[CPR][${debugRunId}]` : '[CPR]';

    const scalarData = volume.imageData.getPointData().getScalars().getData() as
      | Float32Array
      | Int16Array;
    const dimensions = volume.imageData.getDimensions() as [number, number, number];
    const spacing = volume.imageData.getSpacing() as [number, number, number];
    const worldToIndexMatrix =
      (volume.imageData as { getWorldToIndex?: () => ArrayLike<number> | null | undefined })
        .getWorldToIndex?.() ?? null;
    const worldToIndexCandidate = worldToIndexMatrix
      ? Array.from(worldToIndexMatrix).slice(0, 16)
      : null;
    const hasWorldToIndex =
      !!worldToIndexCandidate &&
      worldToIndexCandidate.length >= 16 &&
      worldToIndexCandidate.every(value => Number.isFinite(value));
    const worldToIndex = hasWorldToIndex ? worldToIndexCandidate : undefined;
    const panoHeight = Math.max(1, Math.floor(Number(requestedPanoHeight) || 1));
    const dynamicVertHalfMm =
      Number.isFinite(verticalHalfMm) && Number(verticalHalfMm) > 0 ? Number(verticalHalfMm) : 15;
    console.log('[CPR] vertical sampling config', {
      requestedPanoHeight,
      panoHeight,
      dynamicVertHalfMm,
      verticalHalfMmOverride: verticalHalfMm,
    });
    const { minValue: scalarMin, maxValue: scalarMax } = estimateScalarRange(scalarData);
    const { slope: rescaleSlope, intercept: rescaleIntercept } = getVolumeRescale(volume);
    const { bitsStored, bitsAllocated, highBit, pixelRepresentation, isPreScaled: rawIsPreScaled } =
      getVolumePixelStorage(volume);
    const effectiveIsPreScaled = resolveEffectivePreScaledFlag({
      isPreScaled: rawIsPreScaled,
      scalarMin,
      scalarMax,
      slope: rescaleSlope,
      intercept: rescaleIntercept,
      bitsStored,
      pixelRepresentation,
    });

    const scalarType =
      (scalarData as { constructor?: { name?: string } })?.constructor?.name || 'UnknownTypedArray';

    if (rawIsPreScaled && !effectiveIsPreScaled) {
      console.warn(
        `${logPrefix} volume.isPreScaled=true but scalar range is not HU-like with non-identity rescale; ` +
          'overriding preScaled flag for CPR worker.'
      );
    }

    if (
      !effectiveIsPreScaled &&
      Number.isFinite(bitsStored) &&
      Number(bitsStored) > 0 &&
      Number(bitsStored) < 16 &&
      scalarMax > (1 << Number(bitsStored)) - 1
    ) {
      console.warn(`${logPrefix} Source scalar range exceeds nominal bitsStored range.`, {
        scalarMax,
        bitsStored,
        bitsAllocated,
        highBit,
        pixelRepresentation,
      });
    }

    const nominalBitsStored =
      Number.isFinite(bitsStored) && Number(bitsStored) > 0 ? Math.floor(Number(bitsStored)) : 16;
    const nominalStoredMax =
      nominalBitsStored > 0 && nominalBitsStored < 31 ? (1 << nominalBitsStored) - 1 : Number.MAX_SAFE_INTEGER;
    const nominalSignedMin =
      nominalBitsStored > 0 && nominalBitsStored < 31 ? -(1 << (nominalBitsStored - 1)) : Number.MIN_SAFE_INTEGER;
    const hasUnsignedPackedArtifact =
      !effectiveIsPreScaled &&
      scalarData instanceof Int16Array &&
      nominalBitsStored < 16 &&
      Number(pixelRepresentation) === 0 &&
      (scalarMin < -1 || scalarMax > nominalStoredMax + 8);
    const hasBitDepthRangeMismatch =
      !effectiveIsPreScaled &&
      scalarData instanceof Int16Array &&
      nominalBitsStored < 16 &&
      (scalarMin < nominalSignedMin - 8 || scalarMax > nominalStoredMax + 8);
    const allowStoredValueNormalization =
      !forceDisableStoredValueNormalization &&
      !effectiveIsPreScaled &&
      scalarData instanceof Int16Array &&
      nominalBitsStored < 16 &&
      (hasUnsignedPackedArtifact ||
        hasBitDepthRangeMismatch ||
        (Number(pixelRepresentation) === 0 && scalarMin >= 0 && scalarMax <= nominalStoredMax));
    const heuristicApplyModalityLut = shouldApplyModalityLutForCPR({
      slope: rescaleSlope,
      intercept: rescaleIntercept,
      scalarMin,
      scalarMax,
      bitsStored,
      pixelRepresentation,
      allowStoredValueNormalization,
      isPreScaled: effectiveIsPreScaled,
    });
    const applyModalityLut =
      typeof modalityLutOverride === 'boolean'
        ? modalityLutOverride
        : forceApplyModalityLut
          ? true
          : heuristicApplyModalityLut;

    console.log(`${logPrefix} launchCPRWorker intensity normalization decision`, {
      scalarType,
      scalarMin,
      scalarMax,
      rescaleSlope,
      rescaleIntercept,
      bitsStored,
      bitsAllocated,
      highBit,
      pixelRepresentation,
      rawIsPreScaled,
      effectiveIsPreScaled,
      forceApplyModalityLut: !!forceApplyModalityLut,
      modalityLutOverride,
      forceDisableStoredValueNormalization: !!forceDisableStoredValueNormalization,
      heuristicApplyModalityLut,
      applyModalityLut,
      allowStoredValueNormalization,
      hasUnsignedPackedArtifact,
      hasBitDepthRangeMismatch,
      modalityLutPolicy: 'HEURISTIC_APPLY_FOR_STORED_VALUES',
      aggregation,
      hasWorldToIndex,
      requestedPanoHeight,
      finalPanoHeight: panoHeight,
      dynamicVertHalfMm,
      voiWindowWidth: CPR_PANO_DEFAULT_WINDOW_WIDTH,
      voiWindowCenter: CPR_PANO_DEFAULT_WINDOW_CENTER,
    });

    if (
      !effectiveIsPreScaled &&
      !applyModalityLut &&
      (Math.abs(rescaleSlope - 1) > 1e-6 || Math.abs(rescaleIntercept) > 1e-6)
    ) {
      console.warn(
        `${logPrefix} applyModalityLut=false while source metadata has non-identity rescale. ` +
        'If source pixels are stored values (not HU), fixed WW/WL may cause washed-out pano.'
      );
    }
    if (
      !effectiveIsPreScaled &&
      !applyModalityLut &&
      Math.abs(rescaleIntercept) > 1e-6 &&
      scalarMin >= 0 &&
      scalarMax <= 5000
    ) {
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
      T: Array.from(f.T) as [number, number, number],
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
        modalityLutApplied: data.modalityLutApplied === true,
        requestedModalityLutApplied: data.requestedModalityLutApplied === true,
        storedValueNormalizationApplied: data.storedValueNormalizationApplied === true,
        unsignedPackedArtifactDetected: data.unsignedPackedArtifactDetected === true,
        effectiveIsPreScaled,
        rescaleSlope,
        rescaleIntercept,
        bitsStored,
        pixelRepresentation,
      });
    };

    worker.onerror = err => {
      worker.terminate();
      reject(new Error(`[cprWorker] Uncaught worker error: ${err.message}`));
    };

    worker.postMessage({
      scalarData: dataToSend,
      isSharedArrayBuffer,
      dimensions,
      spacing,
      origin: volume.imageData.getOrigin(),
      direction: volume.imageData.getDirection(),
      worldToIndex,
      frames: serializedFrames,
      panoWidth,
      panoHeight,
      vertHalfMm: dynamicVertHalfMm,
      slabHalfThicknessMm,
      slabSamples,
      aggregation,
      verticalDir,
      applyModalityLut,
      allowStoredValueNormalization,
      disableStoredValueNormalization:
        forceDisableStoredValueNormalization === true ? true : undefined,
      rescaleSlope,
      rescaleIntercept,
      bitsStored,
      bitsAllocated,
      highBit,
      pixelRepresentation,
      isPreScaled: effectiveIsPreScaled,
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
): {
  bitsStored: number;
  bitsAllocated: number;
  highBit: number;
  pixelRepresentation: number;
  isPreScaled: boolean;
} {
  const imageIds = (volume as cornerstone.Types.IImageVolume & { imageIds?: string[] }).imageIds;
  const firstImageId = Array.isArray(imageIds) && imageIds.length > 0 ? imageIds[0] : undefined;
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
  const safeBitsStored = Number.isFinite(bitsStored) && bitsStored >= 1 ? Math.floor(bitsStored) : 16;
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && bitsAllocated >= safeBitsStored
      ? Math.floor(bitsAllocated)
      : 16;
  const safeHighBit = Number.isFinite(highBit) && highBit >= 0 ? Math.floor(highBit) : safeBitsStored - 1;

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

function shouldApplyModalityLutForCPR(params: {
  slope: number;
  intercept: number;
  scalarMin: number;
  scalarMax: number;
  bitsStored: number;
  pixelRepresentation: number;
  allowStoredValueNormalization: boolean;
  isPreScaled: boolean;
}): boolean {
  const {
    slope,
    intercept,
    scalarMin,
    scalarMax,
    bitsStored,
    pixelRepresentation,
    allowStoredValueNormalization,
    isPreScaled,
  } = params;
  const hasNonIdentityRescale = Math.abs(slope - 1) > 1e-6 || Math.abs(intercept) > 1e-6;

  // Identity metadata means no LUT conversion is needed.
  if (!hasNonIdentityRescale) {
    return false;
  }

  // Streaming volume loaders commonly pre-apply modality scaling into scalarData.
  // Re-applying slope/intercept in CPR would double-rescale and corrupt pano intensities.
  if (isPreScaled) {
    return false;
  }

  // If we are explicitly normalizing packed stored values, apply LUT afterwards
  // to land in HU-like display space.
  if (allowStoredValueNormalization) {
    return true;
  }

  const isUnsigned = pixelRepresentation === 0;
  const looksLikeHU = isHuLikeRange(scalarMin, scalarMax);

  // Unsigned stored-value data cannot contain negatives. If negatives exist,
  // cached scalar data is already transformed; applying intercept again would
  // double-shift intensities. Restrict this shortcut to HU-like ranges only.
  if (isUnsigned && scalarMin < -1 && looksLikeHU) {
    return false;
  }

  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 && bitsStored <= 31
      ? Math.floor(bitsStored)
      : 16;

  // For sub-16-bit data, only apply modality LUT when scalar range still looks
  // like native stored-value range for that bit depth.
  if (safeBitsStored < 16) {
    const storedMin = isUnsigned ? 0 : -(1 << (safeBitsStored - 1));
    const storedMax = isUnsigned ? (1 << safeBitsStored) - 1 : (1 << (safeBitsStored - 1)) - 1;
    const margin = Math.max(8, Math.round((storedMax - storedMin) * 0.02));
    const looksStored =
      Number.isFinite(scalarMin) &&
      Number.isFinite(scalarMax) &&
      scalarMin >= storedMin - margin &&
      scalarMax <= storedMax + margin;

    if (!looksLikeHU) {
      return true;
    }

    return looksStored;
  }

  return true;
}

function resolveEffectivePreScaledFlag(params: {
  isPreScaled: boolean;
  scalarMin: number;
  scalarMax: number;
  slope: number;
  intercept: number;
  bitsStored: number;
  pixelRepresentation: number;
}): boolean {
  const { isPreScaled, scalarMin, scalarMax, slope, intercept, bitsStored, pixelRepresentation } = params;

  if (!isPreScaled) {
    return false;
  }

  const hasNonIdentityRescale = Math.abs(slope - 1) > 1e-6 || Math.abs(intercept) > 1e-6;
  if (!hasNonIdentityRescale) {
    return true;
  }

  // Extremely wide ranges are unlikely to be stable HU-space scalars and are
  // safer to treat as stored values for downstream recovery retries.
  const looksImplausiblyWideRange =
    Number.isFinite(scalarMin) &&
    Number.isFinite(scalarMax) &&
    (scalarMin < -9000 || scalarMax > 14000);
  if (looksImplausiblyWideRange) {
    return false;
  }

  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 && bitsStored <= 31 ? Math.floor(bitsStored) : 16;
  const isUnsigned = pixelRepresentation === 0;
  const unsignedStoredMax =
    safeBitsStored > 0 && safeBitsStored < 31 ? (1 << safeBitsStored) - 1 : Number.MAX_SAFE_INTEGER;
  const looksLikeUnsignedStoredRange =
    isUnsigned &&
    Number.isFinite(scalarMin) &&
    Number.isFinite(scalarMax) &&
    scalarMin >= -1 &&
    scalarMax <= unsignedStoredMax + 8;

  // If loader marks volume as pre-scaled and scalar range does not look like
  // native unsigned stored values, prefer preserving current scalar domain.
  if (!looksLikeUnsignedStoredRange) {
    return true;
  }

  return isHuLikeRange(scalarMin, scalarMax);
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
  const properties = ((viewport as { getProperties?: () => unknown }).getProperties?.() || {}) as {
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
  } as any);
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

function summarizeFrameGeometry(
  frames: CPRFrame[],
  verticalDir: [number, number, number]
): {
  sampled: number;
  meanAbsDotTSlab: number;
  p95AbsDotTSlab: number;
  meanAbsDotTVertical: number;
} {
  if (!Array.isArray(frames) || frames.length === 0) {
    return {
      sampled: 0,
      meanAbsDotTSlab: 0,
      p95AbsDotTSlab: 0,
      meanAbsDotTVertical: 0,
    };
  }

  const dotsTSlab: number[] = [];
  const dotsTVertical: number[] = [];
  const stride = Math.max(1, Math.floor(frames.length / 128));
  const v = normalize3(verticalDir);

  for (let i = 0; i < frames.length; i += stride) {
    const frame = frames[i];
    const t = normalize3([frame.T[0], frame.T[1], frame.T[2]]);
    const nSlab = normalize3([frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]]);
    dotsTSlab.push(Math.abs(t[0] * nSlab[0] + t[1] * nSlab[1] + t[2] * nSlab[2]));
    dotsTVertical.push(Math.abs(t[0] * v[0] + t[1] * v[1] + t[2] * v[2]));
  }

  const mean = (values: number[]) =>
    values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const p95 = (() => {
    if (dotsTSlab.length === 0) {
      return 0;
    }
    const sorted = dotsTSlab.slice().sort((a, b) => a - b);
    const index = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
    return sorted[index];
  })();

  return {
    sampled: dotsTSlab.length,
    meanAbsDotTSlab: mean(dotsTSlab),
    p95AbsDotTSlab: p95,
    meanAbsDotTVertical: mean(dotsTVertical),
  };
}

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

function getSourceViewportVerticalDirection(
  sourceViewport: any,
  volumeDirection?: ArrayLike<number> | null
): [number, number, number] {
  if (volumeDirection && volumeDirection.length >= 9) {
    return getVolumeVerticalDirection(volumeDirection);
  }

  const camera = sourceViewport?.getCamera?.();
  const cameraUp = camera?.viewUp;

  if (Array.isArray(cameraUp) && cameraUp.length >= 3) {
    const candidate: [number, number, number] = [
      Number(cameraUp[0] ?? 0),
      Number(cameraUp[1] ?? 0),
      Number(cameraUp[2] ?? 0),
    ];
    if (Number.isFinite(candidate[0]) && Number.isFinite(candidate[1]) && Number.isFinite(candidate[2])) {
      return normalize3(candidate);
    }
  }

  return [0, 0, 1];
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
  panoWidth: requestedPanoWidth = 800,
  panoHeight: requestedPanoHeight = 400,
  slabHalfThicknessMm = 4,
  slabSamples = 11,
  aggregation = 'MIP',
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

      const requestedPanoWidthPx = clampPanoDimension(requestedPanoWidth);
      const requestedPanoHeightPx = clampPanoDimension(requestedPanoHeight);
      console.log(`[CPR][${debugRunId}] deriving pano dimensions`, {
        requestedPanoWidth: requestedPanoWidthPx,
        requestedPanoHeight: requestedPanoHeightPx,
      });

      const rawVolumeSpacing = volume.imageData?.getSpacing?.() as
        | [number, number, number]
        | undefined;
      const spacingX = toPositiveFinite(rawVolumeSpacing?.[0], 1);
      const spacingY = toPositiveFinite(rawVolumeSpacing?.[1], 1);
      const spacingZ = toPositiveFinite(rawVolumeSpacing?.[2], 1);
      const minSpacing = Math.min(spacingX, spacingY, spacingZ);
      const totalArcLength = computeSplineTotalArcLength(rawControlPoints);
      const safeTotalArcLength = Math.max(minSpacing, toPositiveFinite(totalArcLength, minSpacing));
      const idealPanoWidth = Math.round(safeTotalArcLength / minSpacing);
      const minimumPanoWidthPx = Math.max(320, Math.round(requestedPanoWidthPx * 0.6));
      const finalPanoWidth = clampPanoDimension(Math.max(idealPanoWidth, minimumPanoWidthPx));
      const columnPixelSpacing = toPositiveFinite(
        safeTotalArcLength / Math.max(1, finalPanoWidth - 1),
        minSpacing
      );
      const autoVerticalHalfMm = toPositiveFinite(
        safeTotalArcLength / (2 * CPR_PANO_TARGET_ASPECT),
        CPR_PANO_DEFAULT_VERTICAL_HALF_MM
      );
      const baseVerticalHalfMm = Math.max(
        CPR_PANO_DEFAULT_VERTICAL_HALF_MM,
        Math.min(CPR_PANO_MAX_VERTICAL_HALF_MM, autoVerticalHalfMm)
      );
      const thinnerVerticalHalfMm = Math.max(14, Math.min(24, baseVerticalHalfMm * 0.82));
      const minimumPanoHeightPx = Math.max(160, Math.round(requestedPanoHeightPx * 0.55));

      const verticalDir = getSourceViewportVerticalDirection(
        axialViewport,
        volume.imageData?.getDirection?.()
      );
      const { positions, tangents } = buildArcLengthSpline(rawControlPoints, finalPanoWidth);
      const frames = buildRMFFrames(positions, tangents, verticalDir);
      console.log(`[CPR][${debugRunId}] frame geometry summary`, {
        totalArcLength: safeTotalArcLength,
        minSpacing,
        idealPanoWidth,
        finalPanoWidth,
        columnPixelSpacing,
        baseVerticalHalfMm,
        thinnerVerticalHalfMm,
        minimumPanoHeightPx,
        verticalDir,
        ...summarizeFrameGeometry(frames, verticalDir),
      });
      const panoImageId = createPanoImageId();
      console.log(`[CPR][${debugRunId}] generated panoImageId`, { panoImageId });

      // Precompute pano before stage switch so cpr-pano does not visibly flash
      // the source stack before the pano:// image arrives.
      const workerInput = {
        volume,
        frames,
        slabHalfThicknessMm,
        slabSamples,
        aggregation,
        verticalDir,
        verticalHalfMm: baseVerticalHalfMm,
        debugRunId,
      };
      const runWorkerAttempt = async (
        label: string,
        overrides: Partial<Parameters<typeof launchCPRWorker>[0]>
      ): Promise<{
        label: string;
        result: CPRWorkerLaunchResult;
        summary: FloatBufferDebugSummary | null;
        voi: PanoVoiSettings;
        qualityBase: number;
        qualityScore: number;
        huDomain: boolean;
        convertedToHu: boolean;
        rescaleSkippedAsUnsafe: boolean;
        panoWidth: number;
        panoHeight: number;
        actualVertHalfMm: number;
        columnPixelSpacing: number;
        rowPixelSpacing: number;
      }> => {
        const overrideVerticalHalfMm = Number(overrides.verticalHalfMm);
        const actualVertHalfMm =
          Number.isFinite(overrideVerticalHalfMm) && overrideVerticalHalfMm > 0
            ? overrideVerticalHalfMm
            : toPositiveFinite(workerInput.verticalHalfMm, baseVerticalHalfMm);
        const idealPanoHeight = Math.round((actualVertHalfMm * 2) / minSpacing);
        const aspectDrivenHeight = Math.round(finalPanoWidth / CPR_PANO_TARGET_ASPECT);
        const finalPanoHeight = clampPanoDimension(
          Math.max(idealPanoHeight, aspectDrivenHeight, minimumPanoHeightPx)
        );
        const rowPixelSpacing = toPositiveFinite(
          (actualVertHalfMm * 2) / Math.max(1, finalPanoHeight - 1),
          minSpacing
        );
        const rawResult = await launchCPRWorker({
          ...workerInput,
          ...overrides,
          panoWidth: finalPanoWidth,
          panoHeight: finalPanoHeight,
          verticalHalfMm: actualVertHalfMm,
        });
        let result = rawResult;
        const hasIdentityRescaleMetadata =
          Math.abs(result.rescaleSlope - 1) <= 1e-6 && Math.abs(result.rescaleIntercept) <= 1e-6;
        let huDomain =
          result.modalityLutApplied || result.effectiveIsPreScaled || hasIdentityRescaleMetadata;
        let convertedToHu = false;
        let rescaleSkippedAsUnsafe = false;

        if (!huDomain) {
          const canSafelyApplyRescale = isLikelyStoredValueRange(
            result.minValue,
            result.maxValue,
            result.bitsStored,
            result.pixelRepresentation
          );
          if (canSafelyApplyRescale) {
            const converted = applyLinearRescaleToPixelData(
              result.pixelData,
              result.rescaleSlope,
              result.rescaleIntercept
            );
            if (converted) {
              result = {
                ...result,
                pixelData: converted.pixelData,
                minValue: converted.minValue,
                maxValue: converted.maxValue,
                modalityLutApplied: true,
              };
              huDomain = true;
              convertedToHu = true;
            }
          } else {
            // Metadata appears inconsistent with sampled intensity domain;
            // avoid applying potentially destructive linear rescale.
            huDomain = false;
            rescaleSkippedAsUnsafe = true;
          }
        }

        const summary = summarizeFloatBufferForDebug(result.pixelData);
        const voi = computeAdaptivePanoVoi(summary, result.minValue, result.maxValue);
        const qualityBase = scorePanoQuality(summary);
        const splitPenalty = summary
          ? Math.max(0, summary.fractionBelowMinus950 - 0.3) * 8 +
            Math.max(0, summary.fractionAbove3000 - 0.2) * 10 +
            Math.max(0, summary.p50 - 2200) / 300 +
            Math.max(0, -1700 - summary.p50) / 300 +
            Math.max(0, summary.max - 12000) / 1500 +
            Math.max(0, -9000 - summary.min) / 1500
          : 8;
        const qualityScore = qualityBase + (huDomain ? 0 : -100) - splitPenalty;

        console.log(`[CPR][${debugRunId}] pano attempt ${label}`, {
          qualityBase,
          qualityScore,
          minValue: result.minValue,
          maxValue: result.maxValue,
          p01: summary?.p01,
          p50: summary?.p50,
          p99: summary?.p99,
          meanAbsDelta: summary?.meanAbsDelta,
          fractionBelowMinus950: summary?.fractionBelowMinus950,
          fractionAbove3000: summary?.fractionAbove3000,
          huDomain,
          convertedToHu,
          rescaleSkippedAsUnsafe,
          hasIdentityRescaleMetadata,
          modalityLutApplied: result.modalityLutApplied,
          requestedModalityLutApplied: result.requestedModalityLutApplied,
          effectiveIsPreScaled: result.effectiveIsPreScaled,
          rescaleSlope: result.rescaleSlope,
          rescaleIntercept: result.rescaleIntercept,
          actualVertHalfMm,
          idealPanoHeight,
          finalPanoHeight,
          columnPixelSpacing,
          rowPixelSpacing,
          overrides,
        });

        return {
          label,
          result,
          summary,
          voi,
          qualityBase,
          qualityScore,
          huDomain,
          convertedToHu,
          rescaleSkippedAsUnsafe,
          panoWidth: finalPanoWidth,
          panoHeight: finalPanoHeight,
          actualVertHalfMm,
          columnPixelSpacing,
          rowPixelSpacing,
        };
      };

      let bestAttempt = await runWorkerAttempt('primary', {
        modalityLutOverride: true,
        verticalHalfMm: thinnerVerticalHalfMm,
        slabHalfThicknessMm: 3,
        slabSamples: 9,
        aggregation: 'MEAN',
      });

      if (isLikelyPoorPanoQuality(bestAttempt.summary)) {
        const retryConfigs: Array<{
          label: string;
          overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
        }> = [];
        if (bestAttempt.result.effectiveIsPreScaled) {
          // Avoid forcing LUT on data that is already effectively pre-scaled.
          retryConfigs.push({
            label: 'retry-no-lut-mip-thinner',
            overrides: {
              modalityLutOverride: false,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MIP',
            },
          });
          retryConfigs.push({
            label: 'retry-no-lut-mean-thinner',
            overrides: {
              modalityLutOverride: false,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MEAN',
            },
          });
        } else {
          retryConfigs.push({
            label: 'retry-no-lut-mip-thinner',
            overrides: {
              modalityLutOverride: false,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MIP',
            },
          });
          retryConfigs.push({
            label: 'retry-force-lut-mip-with-normalization',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MIP',
            },
          });
          retryConfigs.push({
            label: 'retry-force-lut-mean-with-normalization',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-force-lut-mip-no-normalization',
            overrides: {
              modalityLutOverride: true,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MIP',
            },
          });
          retryConfigs.push({
            label: 'retry-force-lut-mean-no-normalization',
            overrides: {
              modalityLutOverride: true,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: thinnerVerticalHalfMm,
              slabHalfThicknessMm: 3,
              slabSamples: 9,
              aggregation: 'MEAN',
            },
          });
        }

        for (const retryConfig of retryConfigs) {
          const attempt = await runWorkerAttempt(retryConfig.label, retryConfig.overrides);
          if (
            attempt.qualityScore > bestAttempt.qualityScore ||
            (Math.abs(attempt.qualityScore - bestAttempt.qualityScore) < 1e-6 &&
              attempt.qualityBase > bestAttempt.qualityBase)
          ) {
            bestAttempt = attempt;
          }
        }
      }

      if (!bestAttempt.huDomain) {
        console.warn(
          `[CPR][${debugRunId}] Selected panoramic attempt is not in HU domain and could not be safely normalized.`,
          {
            label: bestAttempt.label,
            qualityBase: bestAttempt.qualityBase,
            qualityScore: bestAttempt.qualityScore,
          }
        );
      }

      if (isLikelyPoorPanoQuality(bestAttempt.summary)) {
        console.warn(
          `[CPR][${debugRunId}] proceeding with best available pano despite quality warning`,
          {
            label: bestAttempt.label,
            qualityBase: bestAttempt.qualityBase,
            qualityScore: bestAttempt.qualityScore,
            summary: bestAttempt.summary,
          }
        );
      }

      let panoWorkerResult = bestAttempt.result;
      let panoDebugSummary = bestAttempt.summary;
      let adaptiveVoi = bestAttempt.voi;
      const selectedPanoWidth = bestAttempt.panoWidth;
      const selectedPanoHeight = bestAttempt.panoHeight;
      const selectedActualVertHalfMm = bestAttempt.actualVertHalfMm;
      const selectedColumnPixelSpacing = bestAttempt.columnPixelSpacing;
      const selectedRowPixelSpacing = bestAttempt.rowPixelSpacing;
      console.log(`[CPR][${debugRunId}] selected pano attempt`, {
        label: bestAttempt.label,
        qualityBase: bestAttempt.qualityBase,
        qualityScore: bestAttempt.qualityScore,
        huDomain: bestAttempt.huDomain,
        convertedToHu: bestAttempt.convertedToHu,
        rescaleSkippedAsUnsafe: bestAttempt.rescaleSkippedAsUnsafe,
        panoWidth: selectedPanoWidth,
        panoHeight: selectedPanoHeight,
        actualVertHalfMm: selectedActualVertHalfMm,
        columnPixelSpacing: selectedColumnPixelSpacing,
        rowPixelSpacing: selectedRowPixelSpacing,
      });

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
            huDomain: bestAttempt.huDomain,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            slope: 1,
            intercept: 0,
            columnPixelSpacing: selectedColumnPixelSpacing,
            rowPixelSpacing: selectedRowPixelSpacing,
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
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            columnPixelSpacing: selectedColumnPixelSpacing,
            rowPixelSpacing: selectedRowPixelSpacing,
          });

          const panoViewport = await waitForPanoStackViewport(servicesManager);
          logPanoViewportSnapshot(debugRunId, 'before-setStack', panoViewport);
          await panoViewport.setStack([panoImageId], 0);
          if (typeof (panoViewport as { resetCamera?: () => void }).resetCamera === 'function') {
            panoViewport.resetCamera();
          }
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
              if (
                liveViewport &&
                typeof (liveViewport as { resetCamera?: () => void }).resetCamera === 'function'
              ) {
                liveViewport.resetCamera();
              }
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
    requestedPanoWidth,
    requestedPanoHeight,
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
