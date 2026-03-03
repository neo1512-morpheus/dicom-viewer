import { useCallback, useEffect, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { vec3 } from 'gl-matrix';

import { cprStateService } from './CPRStateService';
import { setPanoImagePayload, createPanoImageId, clearPanoImageCache } from './panoImageLoader';
import { buildRMFFrames } from './cprMath';
import { CPR_CROSSSECTION_SYNC_EVENT, CPRCrossSectionSyncDetail } from './cprEvents';
import { renderPanoGpu, isGpuPanoAvailable } from './cprGpuRenderer';
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

  // Minimal sanity checks for MIP output.
  // MIP naturally produces higher fractionAbove3000 (bone/enamel dominates), so the old
  // looksMostlySaturatedHigh and speckle checks are removed — they were tuned for trimmed-mean.
  const hasExtremeOutliers = summary.min < -9000 || summary.max > 14000;
  const noUsableDynamicRange = (summary.max - summary.min) < 50;
  const allAir = summary.fractionBelowMinus950 > 0.95;
  const allHighSaturation = summary.fractionAbove3000 > 0.95;
  const isAllNonFinite = !Number.isFinite(summary.min) || !Number.isFinite(summary.max);

  return (
    hasExtremeOutliers ||
    noUsableDynamicRange ||
    allAir ||
    allHighSaturation ||
    isAllNonFinite
  );
}

function getHardRejectReason(summary: FloatBufferDebugSummary | null): string | null {
  if (!summary || summary.sampledCount < 100) {
    return 'insufficient-samples';
  }

  if (isSeverelyCorruptedPanoOutput(summary)) {
    return 'severely-corrupted';
  }

  // Minimal sanity checks for MIP output.
  // Old speckle/saturation heuristics removed — they were tuned for trimmed-mean aggregation.
  if (!Number.isFinite(summary.min) || !Number.isFinite(summary.max)) {
    return 'non-finite-range';
  }

  if ((summary.max - summary.min) < 50) {
    return 'flat-image';
  }

  if (summary.fractionBelowMinus950 > 0.95) {
    return 'all-air';
  }

  if (summary.fractionAbove3000 > 0.95) {
    return 'all-high-saturation';
  }

  return null;
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
  if (summary.p50 >= -500 && summary.p50 <= 1200) {
    score += 4;
  } else {
    score -= Math.min(8, Math.abs(summary.p50 - 450) / 450);
  }

  if (summary.fractionBelowMinus950 > 0.5 && summary.fractionAbove3000 > 0.2) {
    score -= 5;
  }

  if (Number.isFinite(robustSpan) && robustSpan > 10000) {
    score -= Math.min(6, (robustSpan - 10000) / 1200);
  }

  if (summary.meanAbsDelta <= 320) {
    score += 3;
  } else if (summary.meanAbsDelta <= 520) {
    score += 1;
  } else {
    score -= Math.min(12, (summary.meanAbsDelta - 520) / 80);
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
    const denseHighNoAir =
      !!summary && summary.p50 > 900 && summary.fractionBelowMinus950 < 0.01 && summary.p99 > 2000;
    if (denseHighNoAir) {
      const lower = -400;
      const upper = 2200;
      const windowWidth = upper - lower;
      return {
        lower,
        upper,
        windowWidth,
        windowCenter: lower + windowWidth / 2,
      };
    }

    const MIN_HU_WINDOW_WIDTH = 2800;
    const MAX_DENTAL_WINDOW_WIDTH = 4500;
    const widthWithMin = Math.max(windowWidth, MIN_HU_WINDOW_WIDTH);
    const cappedWidth = Math.min(widthWithMin, MAX_DENTAL_WINDOW_WIDTH);
    const dentalCenterMin = 200;
    const dentalCenterMax = 600;
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
  workerDebugPayload?: {
    diagnostic?: Record<string, unknown>;
    outputSignature?: {
      sampledCount?: number;
      checksum?: number;
      absChecksum?: number;
      first16?: number[];
    };
  };
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
  verticalCenterOffsetMm?: number;
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
      verticalCenterOffsetMm,
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
      verticalCenterOffsetMmOverride: verticalCenterOffsetMm,
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

      console.log(`${logPrefix} [CPR-WORKER-MESSAGE-JSON]`, JSON.stringify({
        width: data.panoWidth,
        height: data.panoHeight,
        minValue: data.minValue,
        maxValue: data.maxValue,
        modalityLutApplied: data.modalityLutApplied === true,
        requestedModalityLutApplied: data.requestedModalityLutApplied === true,
        storedValueNormalizationApplied: data.storedValueNormalizationApplied === true,
        unsignedPackedArtifactDetected: data.unsignedPackedArtifactDetected === true,
        workerDebugPayload: data.debugPayload ?? null,
      }));

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
        workerDebugPayload:
          data.debugPayload && typeof data.debugPayload === 'object' ? data.debugPayload : undefined,
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
      verticalCenterOffsetMm,
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

function createStoredValueNormalizer(params: {
  scalarData: Float32Array | Int16Array;
  scalarMin: number;
  scalarMax: number;
  bitsStored: number;
  bitsAllocated: number;
  highBit: number;
  pixelRepresentation: number;
  isPreScaled: boolean;
}): {
  allowStoredValueNormalization: boolean;
  hasUnsignedPackedArtifact: boolean;
  hasBitDepthRangeMismatch: boolean;
  normalizationSignature: string | null;
  normalizeStoredSample?: (value: number) => number;
} {
  const {
    scalarData,
    scalarMin,
    scalarMax,
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    isPreScaled,
  } = params;

  const nominalBitsStored =
    Number.isFinite(bitsStored) && Number(bitsStored) > 0 ? Math.floor(Number(bitsStored)) : 16;
  const nominalStoredMax =
    nominalBitsStored > 0 && nominalBitsStored < 31 ? (1 << nominalBitsStored) - 1 : Number.MAX_SAFE_INTEGER;
  const nominalSignedMin =
    nominalBitsStored > 0 && nominalBitsStored < 31 ? -(1 << (nominalBitsStored - 1)) : Number.MIN_SAFE_INTEGER;

  const hasUnsignedPackedArtifact =
    !isPreScaled &&
    scalarData instanceof Int16Array &&
    nominalBitsStored < 16 &&
    Number(pixelRepresentation) === 0 &&
    (scalarMin < -1 || scalarMax > nominalStoredMax + 8);

  const hasBitDepthRangeMismatch =
    !isPreScaled &&
    scalarData instanceof Int16Array &&
    nominalBitsStored < 16 &&
    (scalarMin < nominalSignedMin - 8 || scalarMax > nominalStoredMax + 8);

  const allowStoredValueNormalization =
    !isPreScaled &&
    scalarData instanceof Int16Array &&
    nominalBitsStored > 0 &&
    nominalBitsStored < 16 &&
    (hasUnsignedPackedArtifact ||
      hasBitDepthRangeMismatch ||
      (Number(pixelRepresentation) === 0 && scalarMin >= 0 && scalarMax <= nominalStoredMax));

  if (!allowStoredValueNormalization) {
    return {
      allowStoredValueNormalization,
      hasUnsignedPackedArtifact,
      hasBitDepthRangeMismatch,
      normalizationSignature: null,
    };
  }

  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && Number(bitsAllocated) >= nominalBitsStored
      ? Math.floor(Number(bitsAllocated))
      : 16;
  const safeHighBit =
    Number.isFinite(highBit) && Number(highBit) >= 0 ? Math.floor(Number(highBit)) : nominalBitsStored - 1;
  const normalizedHighBit = Math.max(0, Math.min(safeBitsAllocated - 1, safeHighBit));
  const bitAlignmentShift = Math.max(-15, Math.min(15, normalizedHighBit + 1 - nominalBitsStored));
  const storedMask = (1 << nominalBitsStored) - 1;
  const storedSignBit = 1 << (nominalBitsStored - 1);
  const storedRange = 1 << nominalBitsStored;
  const normalizationSignature = `${nominalBitsStored}:${safeBitsAllocated}:${normalizedHighBit}:${Number(pixelRepresentation)}`;

  const normalizeStoredSample = (value: number): number => {
    if (!Number.isFinite(value)) {
      return 0;
    }
    const intValue = Math.round(value);
    const rawU16 = intValue & 0xffff;
    let aligned = rawU16;
    if (bitAlignmentShift > 0) {
      aligned = rawU16 >>> bitAlignmentShift;
    } else if (bitAlignmentShift < 0) {
      aligned = (rawU16 << -bitAlignmentShift) & 0xffff;
    }
    let normalized = aligned & storedMask;
    if (Number(pixelRepresentation) === 1 && (normalized & storedSignBit) !== 0) {
      normalized -= storedRange;
    }
    return normalized;
  };

  return {
    allowStoredValueNormalization,
    hasUnsignedPackedArtifact,
    hasBitDepthRangeMismatch,
    normalizationSignature,
    normalizeStoredSample,
  };
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
    console.log('[CPR-PANO-SNAPSHOT-JSON]', JSON.stringify({
      runId,
      stage,
      missingViewport: true,
    }));
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
  console.log('[CPR-PANO-SNAPSHOT-JSON]', JSON.stringify({
    runId,
    stage,
    viewportId: viewport.id,
    currentImageId,
    isPanoScheme: typeof currentImageId === 'string' && currentImageId.startsWith('pano://'),
    viewportVoiRange: properties.voiRange ?? null,
    viewportVoiLutFunction: properties.VOILUTFunction ?? null,
    imageWindowWidth: cornerstoneImage?.windowWidth ?? null,
    imageWindowCenter: cornerstoneImage?.windowCenter ?? null,
    metadataWindowWidth,
    metadataWindowCenter,
    metadataVoiLutFunction: voiLut?.voiLUTFunction ?? null,
  }));
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
  const appliedProperties = ((viewport as { getProperties?: () => unknown }).getProperties?.() || {}) as {
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
  };
  console.log('[CPR-VOI-APPLY-JSON]', JSON.stringify({
    runId,
    stage,
    requestedVoi: adaptiveVoi,
    appliedVoiRange: appliedProperties.voiRange ?? null,
    appliedVoiLutFunction: appliedProperties.VOILUTFunction ?? null,
  }));
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
  slabHalfThicknessMm = 2.5,
  slabSamples = 13,
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
      const narrowVerticalHalfMm = Math.max(12, Math.min(16, thinnerVerticalHalfMm * 0.68));
      const mediumVerticalHalfMm = Math.max(
        narrowVerticalHalfMm + 1.2,
        Math.min(18, thinnerVerticalHalfMm * 0.82)
      );
      const broadVerticalHalfMm = thinnerVerticalHalfMm;
      const neutralVerticalCenterOffsetMm = 0;
      const subtleMandibularCenterOffsetMm = -2;
      const mandibularCenterOffsetMm = -3.5;
      const strongMandibularCenterOffsetMm = -5;
      const balancedSlabHalfThicknessMm = 2.5;
      const balancedSlabSamples = 13;
      const fastSlabHalfThicknessMm = 1.5;
      const fastSlabSamples = 9;
      const meanFallbackSlabHalfThicknessMm = 1.0;
      const meanFallbackSlabSamples = 7;
      const sharpMeanSlabHalfThicknessMm = 3.5;   // 7mm total slab — balanced dental panoramic
      const sharpMeanSlabSamples = (() => {
        const targetStepMm = Math.max(0.2, minSpacing * 0.75);
        const adaptive = Math.ceil((3.5 * 2) / targetStepMm) + 1;
        const clamped = Math.max(9, Math.min(32, adaptive)); // Match GPU MAX_SLAB cap
        if (adaptive > 32) {
          console.warn(`[CPR][${debugRunId}] Slab sample count clamped from ${adaptive} to 32`);
        }
        return clamped;
      })();
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
        verticalCenterOffsetMm: neutralVerticalCenterOffsetMm,
        debugRunId,
      };
      const runWorkerAttempt = async (
        label: string,
        overrides: Partial<Parameters<typeof launchCPRWorker>[0]>
      ): Promise<{
        label: string;
        result: CPRWorkerLaunchResult;
        workerDebugPayload?: CPRWorkerLaunchResult['workerDebugPayload'];
        summary: FloatBufferDebugSummary | null;
        voi: PanoVoiSettings;
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        huDomain: boolean;
        convertedToHu: boolean;
        rescaleSkippedAsUnsafe: boolean;
        panoWidth: number;
        panoHeight: number;
        actualVertHalfMm: number;
        verticalCenterOffsetMm: number;
        columnPixelSpacing: number;
        rowPixelSpacing: number;
        aggregation: 'MIP' | 'MEAN';
        slabHalfThicknessMm: number;
        slabSamples: number;
      }> => {
        const requestedAggregation = (overrides.aggregation ?? workerInput.aggregation) === 'MEAN' ? 'MEAN' : 'MIP';
        const requestedSlabHalfThicknessMm = toPositiveFinite(
          overrides.slabHalfThicknessMm,
          toPositiveFinite(workerInput.slabHalfThicknessMm, balancedSlabHalfThicknessMm)
        );
        const requestedSlabSamples = Math.max(
          1,
          Math.floor(
            toPositiveFinite(overrides.slabSamples, toPositiveFinite(workerInput.slabSamples, balancedSlabSamples))
          )
        );
        const requestedVerticalCenterOffsetMm = toFiniteNumber(
          overrides.verticalCenterOffsetMm ?? workerInput.verticalCenterOffsetMm
        ) ?? 0;
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
          verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
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
        const denseFillPenalty = summary
          ? Math.max(0, summary.p50 - 850) / 70 + Math.max(0, summary.p50 - 1150) / 45
          : 0;
        const noAirPenalty = summary
          ? summary.fractionBelowMinus950 < 0.005
            ? 6
            : summary.fractionBelowMinus950 < 0.015
              ? 3
              : 0
          : 0;
        const elevatedP01Penalty = summary ? Math.max(0, summary.p01 + 780) / 80 : 0;
        const aggregationPenalty = 0;
        const hardRejectReason = getHardRejectReason(summary);
        const hardRejectPenalty = hardRejectReason ? 30 : 0;
        const qualityScore =
          qualityBase +
          (huDomain ? 0 : -100) -
          splitPenalty -
          denseFillPenalty -
          noAirPenalty -
          elevatedP01Penalty -
          aggregationPenalty -
          hardRejectPenalty;

        console.log(`[CPR][${debugRunId}] pano attempt ${label}`, {
          qualityBase,
          qualityScore,
          aggregation: requestedAggregation,
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
          verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
          idealPanoHeight,
          finalPanoHeight,
          slabHalfThicknessMm: requestedSlabHalfThicknessMm,
          slabSamples: requestedSlabSamples,
          splitPenalty,
          denseFillPenalty,
          noAirPenalty,
          elevatedP01Penalty,
          hardRejectReason,
          hardRejectPenalty,
          columnPixelSpacing,
          rowPixelSpacing,
          overrides,
        });
        console.log(
          '[CPR-ATTEMPT-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            qualityBase,
            qualityScore,
            aggregation: requestedAggregation,
            slabHalfThicknessMm: requestedSlabHalfThicknessMm,
            slabSamples: requestedSlabSamples,
            verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
            huDomain,
            convertedToHu,
            rescaleSkippedAsUnsafe,
            minValue: result.minValue,
            maxValue: result.maxValue,
            p01: summary?.p01 ?? null,
            p50: summary?.p50 ?? null,
            p99: summary?.p99 ?? null,
            meanAbsDelta: summary?.meanAbsDelta ?? null,
            fractionBelowMinus950: summary?.fractionBelowMinus950 ?? null,
            fractionAbove3000: summary?.fractionAbove3000 ?? null,
            splitPenalty,
            denseFillPenalty,
            noAirPenalty,
            elevatedP01Penalty,
            hardRejectReason,
            hardRejectPenalty,
            voi,
            workerDebugPayload: result.workerDebugPayload ?? null,
          })
        );

        return {
          label,
          result,
          workerDebugPayload: result.workerDebugPayload,
          summary,
          voi,
          qualityBase,
          qualityScore,
          hardRejectReason,
          huDomain,
          convertedToHu,
          rescaleSkippedAsUnsafe,
          panoWidth: finalPanoWidth,
          panoHeight: finalPanoHeight,
          actualVertHalfMm,
          verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
          columnPixelSpacing,
          rowPixelSpacing,
          aggregation: requestedAggregation,
          slabHalfThicknessMm: requestedSlabHalfThicknessMm,
          slabSamples: requestedSlabSamples,
        };
      };

      const attemptAudit: Array<{
        label: string;
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        aggregation: 'MIP' | 'MEAN';
        slabHalfThicknessMm: number;
        slabSamples: number;
        verticalCenterOffsetMm: number;
        huDomain: boolean;
        p01: number | null;
        p50: number | null;
        p99: number | null;
        meanAbsDelta: number | null;
        fractionBelowMinus950: number | null;
        fractionAbove3000: number | null;
      }> = [];
      const recordAttemptAudit = (attempt: {
        label: string;
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        aggregation: 'MIP' | 'MEAN';
        slabHalfThicknessMm: number;
        slabSamples: number;
        verticalCenterOffsetMm: number;
        huDomain: boolean;
        summary: FloatBufferDebugSummary | null;
      }): void => {
        attemptAudit.push({
          label: attempt.label,
          qualityBase: attempt.qualityBase,
          qualityScore: attempt.qualityScore,
          hardRejectReason: attempt.hardRejectReason,
          aggregation: attempt.aggregation,
          slabHalfThicknessMm: attempt.slabHalfThicknessMm,
          slabSamples: attempt.slabSamples,
          verticalCenterOffsetMm: attempt.verticalCenterOffsetMm,
          huDomain: attempt.huDomain,
          p01: attempt.summary?.p01 ?? null,
          p50: attempt.summary?.p50 ?? null,
          p99: attempt.summary?.p99 ?? null,
          meanAbsDelta: attempt.summary?.meanAbsDelta ?? null,
          fractionBelowMinus950: attempt.summary?.fractionBelowMinus950 ?? null,
          fractionAbove3000: attempt.summary?.fractionAbove3000 ?? null,
        });
      };

      // ── GPU RENDER PATH ──────────────────────────────────────────
      // Try GPU-accelerated rendering first. Falls back to CPU workers if unavailable.
      let gpuRenderSucceeded = false;
      let gpuPanoResult: { pixelData: Float32Array; width: number; height: number; minValue: number; maxValue: number } | null = null;

      if (isGpuPanoAvailable()) {
        try {
          const gpuVertHalfMm = mediumVerticalHalfMm;
          const gpuVertCenterOffsetMm = subtleMandibularCenterOffsetMm;
          const gpuActualVertHalfMm = gpuVertHalfMm;

          // Extract volume data directly (these are scoped inside launchCPRWorker)
          const gpuScalarData = volume.imageData.getPointData().getScalars().getData() as
            | Float32Array
            | Int16Array;
          const gpuDimensions = volume.imageData.getDimensions() as [number, number, number];
          const gpuSpacing = volume.imageData.getSpacing() as [number, number, number];
          const gpuOrigin = volume.imageData.getOrigin() as [number, number, number];
          const gpuDirection = Array.from(volume.imageData.getDirection());
          const gpuWorldToIndexMat =
            (volume.imageData as { getWorldToIndex?: () => ArrayLike<number> | null | undefined })
              .getWorldToIndex?.() ?? null;
          const gpuWorldToIndex = gpuWorldToIndexMat
            ? Array.from(gpuWorldToIndexMat).slice(0, 16)
            : null;

          // Determine rescale parameters — use same heuristic as CPU worker
          const { slope: gpuRescaleSlope, intercept: gpuRescaleIntercept } = getVolumeRescale(volume);
          const gpuPixStorage = getVolumePixelStorage(volume);
          const { minValue: gpuScalarMin, maxValue: gpuScalarMax } = estimateScalarRange(gpuScalarData);
          const gpuIsPreScaled = resolveEffectivePreScaledFlag({
            isPreScaled: gpuPixStorage.isPreScaled,
            scalarMin: gpuScalarMin,
            scalarMax: gpuScalarMax,
            slope: gpuRescaleSlope,
            intercept: gpuRescaleIntercept,
            bitsStored: gpuPixStorage.bitsStored,
            pixelRepresentation: gpuPixStorage.pixelRepresentation,
          });
          const gpuStoredValueNormalization = createStoredValueNormalizer({
            scalarData: gpuScalarData,
            scalarMin: gpuScalarMin,
            scalarMax: gpuScalarMax,
            bitsStored: gpuPixStorage.bitsStored,
            bitsAllocated: gpuPixStorage.bitsAllocated,
            highBit: gpuPixStorage.highBit,
            pixelRepresentation: gpuPixStorage.pixelRepresentation,
            isPreScaled: gpuIsPreScaled,
          });
          // Use the same heuristic as the CPU worker for applying modality LUT.
          // Use effective preScaled status (not raw metadata) so implausible
          // ranges can recover via normalization + LUT on GPU too.
          const gpuHeuristicApplyLut = shouldApplyModalityLutForCPR({
            slope: gpuRescaleSlope,
            intercept: gpuRescaleIntercept,
            scalarMin: gpuScalarMin,
            scalarMax: gpuScalarMax,
            bitsStored: gpuPixStorage.bitsStored,
            pixelRepresentation: gpuPixStorage.pixelRepresentation,
            allowStoredValueNormalization: gpuStoredValueNormalization.allowStoredValueNormalization,
            isPreScaled: gpuIsPreScaled,
          });
          const gpuApplyRescale = gpuIsPreScaled ? false : gpuHeuristicApplyLut;
          const gpuHasNonIdentityRescale =
            Math.abs(gpuRescaleSlope - 1) > 1e-6 || Math.abs(gpuRescaleIntercept) > 1e-6;

          console.log(`[CPR][${debugRunId}] GPU rescale diagnosis`, {
            rescaleSlope: gpuRescaleSlope,
            rescaleIntercept: gpuRescaleIntercept,
            rawIsPreScaled: gpuPixStorage.isPreScaled,
            isPreScaled: gpuIsPreScaled,
            heuristicApplyLut: gpuHeuristicApplyLut,
            applyRescale: gpuApplyRescale,
            scalarMin: gpuScalarMin,
            scalarMax: gpuScalarMax,
            bitsStored: gpuPixStorage.bitsStored,
            pixelRepresentation: gpuPixStorage.pixelRepresentation,
            allowStoredValueNormalization: gpuStoredValueNormalization.allowStoredValueNormalization,
            hasUnsignedPackedArtifact: gpuStoredValueNormalization.hasUnsignedPackedArtifact,
            hasBitDepthRangeMismatch: gpuStoredValueNormalization.hasBitDepthRangeMismatch,
            normalizationSignature: gpuStoredValueNormalization.normalizationSignature,
          });

          console.log(`[CPR][${debugRunId}] Attempting GPU panoramic render...`);
          const gpuFrames = frames.map(f => ({
            position: f.position as [number, number, number],
            N_slab: f.N_slab as [number, number, number],
          }));
          const gpuPanoHeight = clampPanoDimension(
            Math.max(
              Math.round((gpuActualVertHalfMm * 2) / minSpacing),
              Math.round(finalPanoWidth / CPR_PANO_TARGET_ASPECT),
              minimumPanoHeightPx
            )
          );

          const runGpuAttempt = (
            label: string,
            applyRescaleFlag: boolean
          ): {
            label: string;
            applyRescale: boolean;
            result: { pixelData: Float32Array; width: number; height: number; minValue: number; maxValue: number };
            summary: FloatBufferDebugSummary | null;
            qualityScore: number;
            hardRejectReason: string | null;
            likelyPoor: boolean;
          } => {
            const result = renderPanoGpu(
              {
                scalarData: gpuScalarData,
                dimensions: gpuDimensions,
                spacing: gpuSpacing,
                origin: gpuOrigin,
                direction: gpuDirection,
                worldToIndex: gpuWorldToIndex,
                frames: gpuFrames,
                panoWidth: finalPanoWidth,
                panoHeight: gpuPanoHeight,
                verticalDir: verticalDir as [number, number, number],
                vertHalfMm: gpuActualVertHalfMm,
                verticalCenterOffsetMm: gpuVertCenterOffsetMm,
                slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
                slabSamples: sharpMeanSlabSamples,
                rescaleSlope: gpuRescaleSlope,
                rescaleIntercept: gpuRescaleIntercept,
                applyRescale: applyRescaleFlag,
                normalizationSignature: gpuStoredValueNormalization.normalizationSignature,
                normalizeStoredSample: gpuStoredValueNormalization.normalizeStoredSample,
              },
              sourceVolumeId
            );

            const summary = summarizeFloatBufferForDebug(result.pixelData);
            const qualityBase = scorePanoQuality(summary);
            const hardRejectReason = getHardRejectReason(summary);
            const hardRejectPenalty = hardRejectReason ? 30 : 0;
            const qualityScore = qualityBase - hardRejectPenalty;
            const likelyPoor = isLikelyPoorPanoQuality(summary);

            console.log(`[CPR][${debugRunId}] GPU attempt ${label}`, {
              applyRescale: applyRescaleFlag,
              minValue: result.minValue,
              maxValue: result.maxValue,
              p01: summary?.p01,
              p50: summary?.p50,
              p99: summary?.p99,
              meanAbsDelta: summary?.meanAbsDelta,
              fractionBelowMinus950: summary?.fractionBelowMinus950,
              fractionAbove3000: summary?.fractionAbove3000,
              qualityScore,
              hardRejectReason,
              likelyPoor,
            });

            return {
              label,
              applyRescale: applyRescaleFlag,
              result,
              summary,
              qualityScore,
              hardRejectReason,
              likelyPoor,
            };
          };

          const primaryGpuAttempt = runGpuAttempt('primary', gpuApplyRescale);
          let selectedGpuAttempt = primaryGpuAttempt;

          const toggledApplyRescale = !gpuApplyRescale;
          if (
            gpuHasNonIdentityRescale &&
            (primaryGpuAttempt.likelyPoor || !!primaryGpuAttempt.hardRejectReason)
          ) {
            const retryGpuAttempt = runGpuAttempt('rescale-toggled-retry', toggledApplyRescale);
            const primaryScore = Number.isFinite(primaryGpuAttempt.qualityScore)
              ? primaryGpuAttempt.qualityScore
              : -Infinity;
            const retryScore = Number.isFinite(retryGpuAttempt.qualityScore)
              ? retryGpuAttempt.qualityScore
              : -Infinity;
            const retryLooksBetter =
              (!retryGpuAttempt.likelyPoor && primaryGpuAttempt.likelyPoor) ||
              retryScore > primaryScore + 0.25;
            if (retryLooksBetter) {
              selectedGpuAttempt = retryGpuAttempt;
            }
          }

          if (selectedGpuAttempt.likelyPoor) {
            console.warn(
              `[CPR][${debugRunId}] GPU output still looks corrupted after recovery attempts. Falling back to CPU workers.`,
              {
                selectedAttempt: selectedGpuAttempt.label,
                applyRescale: selectedGpuAttempt.applyRescale,
                minValue: selectedGpuAttempt.result.minValue,
                maxValue: selectedGpuAttempt.result.maxValue,
                hardRejectReason: selectedGpuAttempt.hardRejectReason,
              }
            );
            gpuRenderSucceeded = false;
          } else {
            gpuPanoResult = selectedGpuAttempt.result;
            gpuRenderSucceeded = true;
            console.log(
              `[CPR][${debugRunId}] GPU render succeeded (${selectedGpuAttempt.label}): ${selectedGpuAttempt.result.width}x${selectedGpuAttempt.result.height}`,
              {
                applyRescale: selectedGpuAttempt.applyRescale,
                minValue: selectedGpuAttempt.result.minValue,
                maxValue: selectedGpuAttempt.result.maxValue,
                qualityScore: selectedGpuAttempt.qualityScore,
              }
            );
          }
        } catch (gpuError) {
          console.warn(`[CPR][${debugRunId}] GPU render failed, falling back to CPU workers.`, gpuError);
          gpuRenderSucceeded = false;
        }
      } else {
        console.log(`[CPR][${debugRunId}] GPU not available, using CPU workers.`);
      }

      // If GPU succeeded, skip all CPU worker attempts
      if (gpuRenderSucceeded && gpuPanoResult) {
        const gpuSummary = summarizeFloatBufferForDebug(gpuPanoResult.pixelData);

        // ── VOI STRATEGY: inherit from axial viewport, or use sensible dental defaults ──
        // The adaptive algorithm was systematically biased high, causing dense structures
        // (bone/enamel >2500 HU) to clip white. Axial viewport already has correct W/L.
        let gpuVoi: PanoVoiSettings = (() => {
          try {
            const axialVp = findViewportByLogicalId(servicesManager, 'cpr-axial');
            const axialProps = (axialVp as any)?.getProperties?.();
            const axialVoiRange = axialProps?.voiRange;
            if (
              axialVoiRange &&
              Number.isFinite(axialVoiRange.lower) &&
              Number.isFinite(axialVoiRange.upper) &&
              axialVoiRange.upper > axialVoiRange.lower
            ) {
              const lower = axialVoiRange.lower;
              const upper = axialVoiRange.upper;
              const windowWidth = upper - lower;
              const windowCenter = lower + windowWidth / 2;
              console.log(`[CPR][${debugRunId}] GPU VOI: inherited from axial viewport`, { lower, upper, windowWidth, windowCenter });
              return { lower, upper, windowWidth, windowCenter };
            }
          } catch (voiErr) {
            console.warn(`[CPR][${debugRunId}] Could not read axial viewport VOI, using dental default`, voiErr);
          }
          // Dental panoramic default: W:4000, L:500 → range [-1500, 2500]
          // Covers air (-1000), soft tissue (0-100), bone (200-1900), dense enamel (2000-4000)
          const defaultWW = 4000;
          const defaultWL = 500;
          const lower = defaultWL - defaultWW / 2;
          const upper = defaultWL + defaultWW / 2;
          console.log(`[CPR][${debugRunId}] GPU VOI: using dental panoramic default`, { lower, upper, windowWidth: defaultWW, windowCenter: defaultWL });
          return { lower, upper, windowWidth: defaultWW, windowCenter: defaultWL };
        })();
        const gpuPanoHeight = gpuPanoResult.height;
        const gpuActualVertHalfMm = mediumVerticalHalfMm;
        const gpuRowPixelSpacing = toPositiveFinite(
          (gpuActualVertHalfMm * 2) / Math.max(1, gpuPanoHeight - 1),
          minSpacing
        );

        // Use GPU result directly — bypass all CPU worker attempts
        const panoWorkerResult = gpuPanoResult;
        const panoDebugSummary = gpuSummary;
        const adaptiveVoi = gpuVoi;
        const selectedPanoWidth = gpuPanoResult.width;
        const selectedPanoHeight = gpuPanoHeight;
        const selectedActualVertHalfMm = gpuActualVertHalfMm;
        const selectedColumnPixelSpacing = columnPixelSpacing;
        const selectedRowPixelSpacing = gpuRowPixelSpacing;

        console.log(`[CPR][${debugRunId}] Using GPU render result`, {
          panoWidth: selectedPanoWidth,
          panoHeight: selectedPanoHeight,
          minValue: panoWorkerResult.minValue,
          maxValue: panoWorkerResult.maxValue,
          voiLower: adaptiveVoi.lower,
          voiUpper: adaptiveVoi.upper,
        });

        // Skip to the stage switch / image loading code below
        // (same flow as CPU path starting from line "Keep exactly one arch annotation")
        removeSplineAnnotationsExcept(latestAnnotationUID);
        cprStateService.setArchData(rawControlPoints, frames, sourceVolumeId, latestAnnotationUID);
        clearProtocolListener();

        let protocolAppliedHandled = false;
        const onProtocolApplied = async () => {
          if (protocolAppliedHandled) return;
          protocolAppliedHandled = true;
          clearProtocolListener();
          try {
            void (async () => {
              try {
                const axialViewportAfterSwitch = await waitForViewportByLogicalId(
                  servicesManager, 'cpr-axial', 4000
                );
                if (axialViewportAfterSwitch && preservedAxialCamera) {
                  axialViewportAfterSwitch.setCamera(
                    preservedAxialCamera as Partial<ReturnType<typeof axialViewportAfterSwitch.getCamera>>
                  );
                  axialViewportAfterSwitch.render();
                  cornerstoneTools.utilities.triggerAnnotationRender(axialViewportAfterSwitch.element);
                }
              } catch (e) {
                console.warn(`[CPR][${debugRunId}] Failed to restore axial camera`, e);
              }
            })();

            clearPanoImageCache();
            setPanoImagePayload(panoImageId, {
              pixelData: panoWorkerResult.pixelData,
              width: panoWorkerResult.width,
              height: panoWorkerResult.height,
              minValue: panoWorkerResult.minValue,
              maxValue: panoWorkerResult.maxValue,
              huDomain: true,
              windowWidth: adaptiveVoi.windowWidth,
              windowCenter: adaptiveVoi.windowCenter,
              slope: 1,
              intercept: 0,
              columnPixelSpacing: selectedColumnPixelSpacing,
              rowPixelSpacing: selectedRowPixelSpacing,
            });

            const panoViewport = await waitForPanoStackViewport(servicesManager);
            logPanoViewportSnapshot(debugRunId, 'gpu-before-setStack', panoViewport);
            // Apply camera reset BEFORE setStack so it cannot stomp our post-load VOI
            if (typeof (panoViewport as { resetCamera?: () => void }).resetCamera === 'function') {
              panoViewport.resetCamera();
            }
            await panoViewport.setStack([panoImageId], 0);
            logPanoViewportSnapshot(debugRunId, 'gpu-after-setStack', panoViewport);
            applyPanoDisplaySettings(debugRunId, 'gpu-after-setStack', panoViewport, adaptiveVoi);
            panoViewport.render();

            // Re-assert VOI after first render
            if (panoViewport.element) {
              let reappliedOnRender = false;
              const onFirstImageRendered = () => {
                if (reappliedOnRender) return;
                reappliedOnRender = true;
                panoViewport.element.removeEventListener(
                  cornerstone.Enums.Events.IMAGE_RENDERED,
                  onFirstImageRendered
                );
                const liveViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
                if (liveViewport && typeof (liveViewport as { resetCamera?: () => void }).resetCamera === 'function') {
                  liveViewport.resetCamera();
                }
                applyPanoDisplaySettings(debugRunId, 'gpu-after-first-image-rendered', liveViewport, adaptiveVoi);
                liveViewport?.render?.();
              };
              panoViewport.element.addEventListener(
                cornerstone.Enums.Events.IMAGE_RENDERED,
                onFirstImageRendered
              );
            }
            [120, 500, 1500].forEach(delayMs => {
              window.setTimeout(() => {
                const livePanoViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
                applyPanoDisplaySettings(debugRunId, `gpu-post+${delayMs}ms`, livePanoViewport, adaptiveVoi);
                livePanoViewport?.render?.();
              }, delayMs);
            });

            commandsManager.runCommand('setToolActive', {
              toolName: 'CPRCursor',
              toolGroupId: 'cprPano',
            });
            cornerstoneTools.utilities.triggerAnnotationRender(panoViewport.element);
            initializeCrossSection(frames, servicesManager);
          } catch (err) {
            console.error(`[CPR][${debugRunId}] GPU post-render error`, err);
            if (isMountedRef.current) {
              setError(err instanceof Error ? err.message : String(err));
            }
          } finally {
            if (isMountedRef.current) {
              setIsGenerating(false);
            }
          }
        };

        const gpuProtocolAppliedEvent =
          hangingProtocolService.EVENTS.PROTOCOL_APPLIED ||
          hangingProtocolService.EVENTS.PROTOCOL_CHANGED;

        if (!gpuProtocolAppliedEvent) {
          throw new Error('[CPR] No supported hanging protocol event found for GPU path.');
        }

        hpSubscriptionRef.current = hangingProtocolService.subscribe(
          gpuProtocolAppliedEvent,
          onProtocolApplied
        );

        // Safety timeout in case protocol change event never arrives
        hpTimeoutRef.current = window.setTimeout(() => {
          clearProtocolListener();
          if (isMountedRef.current) {
            console.error(`[CPR][${debugRunId}] GPU path: timed out waiting for protocol change.`);
            setError('[CPR] Timed out waiting for hanging protocol change after GPU render.');
            setIsGenerating(false);
          }
        }, 12000);

        console.log(`[CPR][${debugRunId}] GPU path: switching to CPR stage 1`);
        commandsManager.runCommand('setHangingProtocol', {
          protocolId: 'cpr',
          stageIndex: 1,
        });
        return; // GPU path complete — skip all CPU worker code below
      }

      // ── CPU WORKER FALLBACK PATH ─────────────────────────────────
      let bestAttempt = await runWorkerAttempt('primary-mean-sharp-medium', {
        modalityLutOverride: true,
        verticalHalfMm: mediumVerticalHalfMm,
        verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
        slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
        slabSamples: sharpMeanSlabSamples,
        aggregation: 'MEAN',
      });
      recordAttemptAudit(bestAttempt);

      const retryConfigs: Array<{
        label: string;
        overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
      }> = [
          // MEAN with narrow vertical coverage (focused on teeth)
          {
            label: 'retry-mean-sharp-narrow',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: mandibularCenterOffsetMm,
              slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
              slabSamples: sharpMeanSlabSamples,
              aggregation: 'MEAN',
            },
          },
          // MEAN with broader vertical coverage
          {
            label: 'retry-mean-sharp-broad',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: broadVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
              slabSamples: sharpMeanSlabSamples,
              aggregation: 'MEAN',
            },
          },
          // MEAN with no normalization fallback
          {
            label: 'retry-mean-sharp-no-normalization',
            overrides: {
              modalityLutOverride: true,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: mandibularCenterOffsetMm,
              slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
              slabSamples: sharpMeanSlabSamples,
              aggregation: 'MIP',
            },
          },
        ];

      for (const retryConfig of retryConfigs) {
        const attempt = await runWorkerAttempt(retryConfig.label, retryConfig.overrides);
        recordAttemptAudit(attempt);
        const bestHardRejected = !!bestAttempt.hardRejectReason;
        const attemptHardRejected = !!attempt.hardRejectReason;
        if (
          (bestHardRejected && !attemptHardRejected) ||
          (bestHardRejected === attemptHardRejected &&
            (attempt.qualityScore > bestAttempt.qualityScore ||
              (Math.abs(attempt.qualityScore - bestAttempt.qualityScore) < 1e-6 &&
                attempt.qualityBase > bestAttempt.qualityBase)))
        ) {
          bestAttempt = attempt;
        }
      }

      const rankedAttempts = attemptAudit
        .slice()
        .sort((a, b) => {
          const aHardRejected = !!a.hardRejectReason;
          const bHardRejected = !!b.hardRejectReason;
          if (aHardRejected !== bHardRejected) {
            return aHardRejected ? 1 : -1;
          }
          return (b.qualityScore - a.qualityScore) || (b.qualityBase - a.qualityBase);
        });
      const selectedAttempt = rankedAttempts[0];
      const runnerUpAttempt = rankedAttempts.length > 1 ? rankedAttempts[1] : null;
      console.log(
        '[CPR-ATTEMPT-LIST-JSON]',
        JSON.stringify({
          runId: debugRunId,
          attempts: attemptAudit,
          selectedLabel: bestAttempt.label,
        })
      );
      console.log(
        '[CPR-SELECTED-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: bestAttempt.label,
          selectedAggregation: bestAttempt.aggregation,
          selectedSlabHalfThicknessMm: bestAttempt.slabHalfThicknessMm,
          selectedSlabSamples: bestAttempt.slabSamples,
          selectedVerticalCenterOffsetMm: bestAttempt.verticalCenterOffsetMm,
          selectedHardRejectReason: bestAttempt.hardRejectReason,
          selectedQualityScore: bestAttempt.qualityScore,
          selectedQualityBase: bestAttempt.qualityBase,
          selectedSummary: bestAttempt.summary,
          selectedVoi: bestAttempt.voi,
          selectedWorkerDebugPayload: bestAttempt.workerDebugPayload ?? null,
          runnerUp: runnerUpAttempt,
          scoreDeltaToRunnerUp:
            runnerUpAttempt ? bestAttempt.qualityScore - runnerUpAttempt.qualityScore : null,
          baseDeltaToRunnerUp:
            runnerUpAttempt ? bestAttempt.qualityBase - runnerUpAttempt.qualityBase : null,
          selectedMatchesRankedTop: selectedAttempt?.label === bestAttempt.label,
        })
      );

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
      // Override the data-adaptive VOI with axial viewport W/L — avoids the high-bias
      // problem where enamel/bone clips white because adaptive center is too high.
      let adaptiveVoi: PanoVoiSettings = (() => {
        try {
          const axialVpForVoi = findViewportByLogicalId(servicesManager, 'cpr-axial');
          const axialP = (axialVpForVoi as any)?.getProperties?.();
          const axialRange = axialP?.voiRange;
          if (
            axialRange &&
            Number.isFinite(axialRange.lower) &&
            Number.isFinite(axialRange.upper) &&
            axialRange.upper > axialRange.lower
          ) {
            const lower = axialRange.lower;
            const upper = axialRange.upper;
            const windowWidth = upper - lower;
            const windowCenter = lower + windowWidth / 2;
            console.log(`[CPR][${debugRunId}] CPU VOI: inherited from axial viewport`, { lower, upper, windowWidth, windowCenter });
            return { lower, upper, windowWidth, windowCenter };
          }
        } catch (_e) { /* ignore */ }
        // Dental panoramic default: W:4000, L:500 → range [-1500, 2500]
        const ww = 4000, wl = 500;
        console.log(`[CPR][${debugRunId}] CPU VOI: using dental panoramic default W:${ww} L:${wl}`);
        return { lower: wl - ww / 2, upper: wl + ww / 2, windowWidth: ww, windowCenter: wl };
      })();
      const selectedPanoWidth = bestAttempt.panoWidth;
      const selectedPanoHeight = bestAttempt.panoHeight;
      const selectedActualVertHalfMm = bestAttempt.actualVertHalfMm;
      const selectedColumnPixelSpacing = bestAttempt.columnPixelSpacing;
      const selectedRowPixelSpacing = bestAttempt.rowPixelSpacing;
      console.log(`[CPR][${debugRunId}] selected pano attempt`, {
        label: bestAttempt.label,
        qualityBase: bestAttempt.qualityBase,
        qualityScore: bestAttempt.qualityScore,
        hardRejectReason: bestAttempt.hardRejectReason,
        aggregation: bestAttempt.aggregation,
        huDomain: bestAttempt.huDomain,
        convertedToHu: bestAttempt.convertedToHu,
        rescaleSkippedAsUnsafe: bestAttempt.rescaleSkippedAsUnsafe,
        panoWidth: selectedPanoWidth,
        panoHeight: selectedPanoHeight,
        actualVertHalfMm: selectedActualVertHalfMm,
        verticalCenterOffsetMm: bestAttempt.verticalCenterOffsetMm,
        slabHalfThicknessMm: bestAttempt.slabHalfThicknessMm,
        slabSamples: bestAttempt.slabSamples,
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
          console.log('[CPR-LOADER-METADATA-JSON]', JSON.stringify({
            runId: debugRunId,
            panoImageId,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            voiLower: adaptiveVoi.lower,
            voiUpper: adaptiveVoi.upper,
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            columnPixelSpacing: selectedColumnPixelSpacing,
            rowPixelSpacing: selectedRowPixelSpacing,
            selectedAttempt: {
              label: bestAttempt.label,
              aggregation: bestAttempt.aggregation,
              verticalCenterOffsetMm: bestAttempt.verticalCenterOffsetMm,
              slabHalfThicknessMm: bestAttempt.slabHalfThicknessMm,
              slabSamples: bestAttempt.slabSamples,
              qualityScore: bestAttempt.qualityScore,
            },
          }));

          const panoViewport = await waitForPanoStackViewport(servicesManager);
          logPanoViewportSnapshot(debugRunId, 'before-setStack', panoViewport);
          // Apply camera reset BEFORE setStack so it cannot stomp our post-load VOI
          if (typeof (panoViewport as { resetCamera?: () => void }).resetCamera === 'function') {
            panoViewport.resetCamera();
          }
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
