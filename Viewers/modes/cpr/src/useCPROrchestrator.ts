import { useCallback, useEffect, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { vec3 } from 'gl-matrix';

import { cprStateService } from './CPRStateService';
import { setPanoImagePayload, createPanoImageId, clearPanoImageCache } from './panoImageLoader';
import {
  SyntheticCprIntensityDomain,
  classifySyntheticCprIntensityDomain,
  isSyntheticCprHuDomain,
} from './cprSyntheticDisplay';
import {
  clearCrossSectionImageCache,
  createCrossSectionImageIds,
  createCrossSectionSeriesId,
  setCrossSectionSeriesPayload,
  updateCrossSectionSeriesDisplayDefaults,
} from './crossSectionImageLoader';
import { buildRMFFrames } from './cprMath';
import { CPR_CROSSSECTION_SYNC_EVENT, CPRCrossSectionSyncDetail } from './cprEvents';
import { buildCrossSectionCameraForFrame } from './cprCrossSectionCamera';
import type { CPRFrame } from './cprMath';

const CPR_PANO_DEFAULT_WINDOW_WIDTH = 3000;
const CPR_PANO_DEFAULT_WINDOW_CENTER = 400;
const CPR_PANO_MAX_DIMENSION = 4096;
const CPR_PANO_DEFAULT_VERTICAL_HALF_MM = 18;
const CPR_PANO_MAX_VERTICAL_HALF_MM = 28;
const CPR_PANO_TARGET_ASPECT = 3.2;

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
  toothBandMean: number;
  toothBandP10: number;
  toothBandP90: number;
  toothBandBrightFraction: number;
  lowerBandP50: number;
  lowerBandBrightFraction: number;
  detailBandHorizontalEdgeMean: number;
  detailBandVerticalEdgeMean: number;
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
  const hasModerateSpeckleNoise = summary.meanAbsDelta > 640;
  const hasLowerBandFill =
    summary.lowerBandP50 > 140 || summary.lowerBandBrightFraction > 0.7;
  const detailAnisotropyRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
  const balancedDetailEdgeMean = Math.min(
    summary.detailBandHorizontalEdgeMean,
    summary.detailBandVerticalEdgeMean * 2.4
  );
  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const hasToothBandSaturation = summary.toothBandP10 > 220 || summary.toothBandMean > 880;
  const hasWhiteBlobArtifact =
    summary.toothBandBrightFraction > 0.36 &&
    summary.toothBandP10 > 120 &&
    summary.toothBandP90 > 1450;
  const hasWeakToothBandContrast = toothBandContrastRange < 260;
  const hasWeakDentalSeparation = balancedDetailEdgeMean < 38;
  const hasShearWarping =
    summary.detailBandHorizontalEdgeMean > 220 && detailAnisotropyRatio > 3;

  return (
    hasExtremeOutliers ||
    looksMostlySaturatedHigh ||
    isLowContrast ||
    hasSplitTailDistribution ||
    hasMedianOutOfTypicalRange ||
    hasStrongSpeckleNoise ||
    hasModerateSpeckleNoise ||
    hasLowerBandFill ||
    hasToothBandSaturation ||
    hasWhiteBlobArtifact ||
    hasWeakToothBandContrast ||
    hasWeakDentalSeparation ||
    hasShearWarping
  );
}

function getHardRejectReason(summary: FloatBufferDebugSummary | null): string | null {
  if (!summary || summary.sampledCount < 100) {
    return 'insufficient-samples';
  }

  if (isSeverelyCorruptedPanoOutput(summary)) {
    return 'severely-corrupted';
  }

  if (summary.meanAbsDelta > 760) {
    return 'speckle-noise';
  }

  if (summary.toothBandP10 > 280 && summary.toothBandMean > 950) {
    return 'tooth-band-saturation';
  }

  if (
    summary.toothBandBrightFraction > 0.42 &&
    summary.toothBandP10 > 120 &&
    summary.toothBandP90 > 1550
  ) {
    return 'white-blob-artifact';
  }

  if (summary.lowerBandP50 > 260 && summary.lowerBandBrightFraction > 0.78) {
    return 'lower-band-fill';
  }

  if (summary.fractionAbove3000 > 0.35 && summary.p50 > 950) {
    return 'high-saturation';
  }

  if (summary.fractionBelowMinus950 < 0.003 && summary.p50 > 1200) {
    return 'dense-fill';
  }

  if (summary.p01 > -450 && summary.p50 > 900) {
    return 'no-air-high-median';
  }

  const detailAnisotropyRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
  if (summary.detailBandHorizontalEdgeMean > 320 && detailAnisotropyRatio > 4.8) {
    return 'shear-warping';
  }

  return null;
}

function scorePanoQuality(summary: FloatBufferDebugSummary | null): number {
  if (!summary || summary.sampledCount < 100) {
    return -Infinity;
  }

  const robustSpan = summary.p99 - summary.p01;
  const detailAnisotropyRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
  const balancedDetailEdgeMean = Math.min(
    summary.detailBandHorizontalEdgeMean,
    summary.detailBandVerticalEdgeMean * 2.4
  );
  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  let score = 0;

  if (Number.isFinite(robustSpan)) {
    if (robustSpan >= 1300 && robustSpan <= 4200) {
      score += 4;
    } else if (robustSpan >= 900 && robustSpan <= 5600) {
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

  if (summary.lowerBandP50 <= -220) {
    score += 4;
  } else if (summary.lowerBandP50 <= -120) {
    score += 2;
  } else if (summary.lowerBandP50 <= -45) {
    score += 0;
  } else {
    score -= Math.min(12, (summary.lowerBandP50 + 45) / 45);
  }

  if (summary.lowerBandBrightFraction <= 0.18) {
    score += 4;
  } else if (summary.lowerBandBrightFraction <= 0.32) {
    score += 2;
  } else if (summary.lowerBandBrightFraction <= 0.5) {
    score += 0;
  } else {
    score -= Math.min(14, (summary.lowerBandBrightFraction - 0.5) * 28);
  }

  if (balancedDetailEdgeMean >= 70 && balancedDetailEdgeMean <= 220) {
    score += 5;
  } else if (balancedDetailEdgeMean >= 45) {
    score += 2;
  } else {
    score -= 3;
  }

  if (summary.toothBandMean >= 160 && summary.toothBandMean <= 760) {
    score += 3;
  } else {
    score -= Math.min(10, Math.abs(summary.toothBandMean - 460) / 110);
  }

  if (summary.toothBandP10 <= -120) {
    score += 2;
  } else if (summary.toothBandP10 <= 80) {
    score += 1;
  } else {
    score -= Math.min(12, (summary.toothBandP10 - 80) / 32);
  }

  if (summary.toothBandBrightFraction <= 0.14) {
    score += 2;
  } else if (summary.toothBandBrightFraction <= 0.24) {
    score += 1;
  } else {
    score -= Math.min(14, (summary.toothBandBrightFraction - 0.24) * 24);
  }

  if (toothBandContrastRange >= 360 && toothBandContrastRange <= 1700) {
    score += 3;
  } else if (toothBandContrastRange >= 240) {
    score += 1;
  } else {
    score -= Math.min(8, (240 - toothBandContrastRange) / 36);
  }

  if (summary.detailBandHorizontalEdgeMean > 260) {
    score -= Math.min(6, (summary.detailBandHorizontalEdgeMean - 260) / 24);
  }

  if (detailAnisotropyRatio > 2.8) {
    score -= Math.min(8, (detailAnisotropyRatio - 2.8) * 4.5);
  }

  if (summary.fractionBelowMinus950 > 0.5 && summary.fractionAbove3000 > 0.2) {
    score -= 5;
  }

  if (Number.isFinite(robustSpan) && robustSpan > 10000) {
    score -= Math.min(6, (robustSpan - 10000) / 1200);
  }

  if (summary.meanAbsDelta <= 300) {
    score += 3;
  } else if (summary.meanAbsDelta <= 430) {
    score += 2;
  } else if (summary.meanAbsDelta <= 560) {
    score += 0;
  } else {
    score -= Math.min(12, (summary.meanAbsDelta - 520) / 80);
  }

  return score;
}

function scoreHardRejectedPanoFallback(summary: FloatBufferDebugSummary | null): number {
  if (!summary || summary.sampledCount < 100) {
    return -Infinity;
  }

  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const detailRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);

  let score = 0;
  score -= Math.max(0, summary.lowerBandP50 + 160) / 45;
  score -= Math.max(0, summary.lowerBandBrightFraction - 0.28) * 30;
  score -= Math.max(0, summary.toothBandMean - 760) / 80;
  score -= Math.max(0, summary.toothBandP10 - 80) / 30;
  score -= Math.max(0, summary.toothBandBrightFraction - 0.24) * 26;
  score += Math.max(-6, Math.min(6, (toothBandContrastRange - 320) / 90));
  score -= Math.max(0, detailRatio - 2.8) * 3.2;
  score -= Math.max(0, summary.meanAbsDelta - 520) / 90;
  score -= Math.max(0, summary.fractionAbove3000 - 0.005) * 120;
  return score;
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

function summarizeFloatBufferForDebug(
  buffer: Float32Array,
  width?: number,
  height?: number
): FloatBufferDebugSummary | null {
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
  const hasGrid = !!width && !!height && width > 1 && height > 1 && width * height <= buffer.length;
  const toothBandSamples: number[] = [];
  let toothBandBrightCount = 0;
  let toothBandCount = 0;
  let toothBandSum = 0;
  const lowerBandSamples: number[] = [];
  let lowerBandBrightCount = 0;
  let lowerBandCount = 0;
  let detailBandHorizontalEdgeAccum = 0;
  let detailBandHorizontalEdgeCount = 0;
  let detailBandVerticalEdgeAccum = 0;
  let detailBandVerticalEdgeCount = 0;

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

  if (hasGrid) {
    const safeWidth = Number(width);
    const safeHeight = Number(height);
    const rowStep = Math.max(1, Math.floor(safeHeight / 140));
    const colStep = Math.max(1, Math.floor(safeWidth / 180));
    const panoCenterRow = (safeHeight - 1) / 2;
    const rowFromNormalizedOffset = (yNorm: number): number => {
      const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
      return Math.max(0, Math.min(safeHeight - 1, row));
    };
    const toothBandStartRow = Math.min(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const toothBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const lowerBandStartRow = Math.min(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const lowerBandEndRow = Math.max(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const detailBandStartRow = Math.max(0, Math.floor(safeHeight * 0.12));
    const detailBandEndRow = Math.min(safeHeight - 1, Math.ceil(safeHeight * 0.72));

    for (let row = 0; row < safeHeight; row += rowStep) {
      for (let col = 0; col < safeWidth; col += colStep) {
        const index = row * safeWidth + col;
        const value = Number(buffer[index]);
        if (!Number.isFinite(value)) {
          continue;
        }

        if (row >= lowerBandStartRow && row <= lowerBandEndRow) {
          lowerBandSamples.push(value);
          lowerBandCount++;
          if (value > -180) {
            lowerBandBrightCount++;
          }
        }

        if (row >= toothBandStartRow && row <= toothBandEndRow) {
          toothBandSamples.push(value);
          toothBandCount++;
          toothBandSum += value;
          if (value > 1200) {
            toothBandBrightCount++;
          }
        }

        if (row >= detailBandStartRow && row <= detailBandEndRow) {
          const nextCol = col + colStep;
          if (nextCol < safeWidth) {
            const neighborValue = Number(buffer[row * safeWidth + nextCol]);
            if (Number.isFinite(neighborValue)) {
              detailBandHorizontalEdgeAccum += Math.abs(neighborValue - value);
              detailBandHorizontalEdgeCount++;
            }
          }
          const nextRow = row + rowStep;
          if (nextRow <= detailBandEndRow) {
            const neighborValue = Number(buffer[nextRow * safeWidth + col]);
            if (Number.isFinite(neighborValue)) {
              detailBandVerticalEdgeAccum += Math.abs(neighborValue - value);
              detailBandVerticalEdgeCount++;
            }
          }
        }
      }
    }
  }

  if (!samples.length || !Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }

  samples.sort((a, b) => a - b);
  if (toothBandSamples.length) {
    toothBandSamples.sort((a, b) => a - b);
  }
  if (lowerBandSamples.length) {
    lowerBandSamples.sort((a, b) => a - b);
  }
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
    toothBandMean: toothBandCount > 0 ? toothBandSum / toothBandCount : 0,
    toothBandP10: toothBandSamples.length ? percentileFromSorted(toothBandSamples, 0.1) : 0,
    toothBandP90: toothBandSamples.length ? percentileFromSorted(toothBandSamples, 0.9) : 0,
    toothBandBrightFraction: toothBandCount > 0 ? toothBandBrightCount / toothBandCount : 0,
    lowerBandP50: lowerBandSamples.length ? percentileFromSorted(lowerBandSamples, 0.5) : 0,
    lowerBandBrightFraction: lowerBandCount > 0 ? lowerBandBrightCount / lowerBandCount : 0,
    detailBandHorizontalEdgeMean:
      detailBandHorizontalEdgeCount > 0 ? detailBandHorizontalEdgeAccum / detailBandHorizontalEdgeCount : 0,
    detailBandVerticalEdgeMean:
      detailBandVerticalEdgeCount > 0 ? detailBandVerticalEdgeAccum / detailBandVerticalEdgeCount : 0,
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
  const lower = Math.min(safeMin, safeMax);
  const upper = Math.max(safeMin, safeMax);
  const p01 = summary?.p01;
  const p99 = summary?.p99;

  let adaptiveLower = Number.isFinite(p01) ? Number(p01) : lower;
  let adaptiveUpper = Number.isFinite(p99) ? Number(p99) : upper;

  if (!Number.isFinite(adaptiveLower) || !Number.isFinite(adaptiveUpper) || adaptiveUpper <= adaptiveLower) {
    adaptiveLower = lower;
    adaptiveUpper = upper;
  }

  const robustSpan = Math.max(1, adaptiveUpper - adaptiveLower);
  const padding = Math.max(20, robustSpan * 0.03);
  adaptiveLower -= padding;
  adaptiveUpper += padding;

  if (!Number.isFinite(adaptiveLower) || !Number.isFinite(adaptiveUpper) || adaptiveUpper <= adaptiveLower) {
    adaptiveLower = fallbackLower;
    adaptiveUpper = fallbackUpper;
  }

  const windowWidth = Math.max(1, adaptiveUpper - adaptiveLower);
  const windowCenter = adaptiveLower + windowWidth / 2;

  const hasSplitOutliers =
    !!summary &&
    summary.fractionBelowMinus950 > 0.4 &&
    summary.fractionAbove3000 > 0.15;
  const isExtremeWindow = adaptiveLower < -5000 || adaptiveUpper > 7000;
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

  const looksLikeHU = isHuLikeRange(lower, upper);

  if (looksLikeHU) {
    const denseHighNoAir =
      !!summary && summary.p50 > 900 && summary.fractionBelowMinus950 < 0.01 && summary.p99 > 2000;
    if (denseHighNoAir) {
      const lower = -1000;
      const upper = 2200;
      const windowWidth = upper - lower;
      return {
        lower,
        upper,
        windowWidth,
        windowCenter: lower + windowWidth / 2,
      };
    }

    if (summary && summary.lowerBandBrightFraction > 0.62) {
      const targetWidth =
        summary.detailBandHorizontalEdgeMean > 210 ? 1300 : 1450;
      const targetCenter = Math.max(
        120,
        Math.min(260, 170 + (summary.lowerBandBrightFraction - 0.62) * 260 + Math.max(0, summary.p50) * 0.15)
      );
      return {
        lower: targetCenter - targetWidth / 2,
        upper: targetCenter + targetWidth / 2,
        windowWidth: targetWidth,
        windowCenter: targetCenter,
      };
    }

    const minHuWindowWidth =
      summary && summary.detailBandHorizontalEdgeMean >= 85 ? 1600 : 1750;
    const maxDentalWindowWidth =
      summary && summary.lowerBandBrightFraction > 0.45 ? 2200 : 2500;
    const widthWithMin = Math.max(windowWidth, minHuWindowWidth);
    const cappedWidth = Math.min(widthWithMin, maxDentalWindowWidth);
    const dentalCenterMin = -120;
    const dentalCenterMax = 380;
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

  return { lower: adaptiveLower, upper: adaptiveUpper, windowWidth, windowCenter };
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

function dot3(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function projectPointOntoPlane(
  point: [number, number, number],
  planePoint: [number, number, number],
  planeNormal: [number, number, number]
): [number, number, number] {
  const normalizedPlaneNormal = normalize3(planeNormal);
  const delta: [number, number, number] = [
    point[0] - planePoint[0],
    point[1] - planePoint[1],
    point[2] - planePoint[2],
  ];
  const distanceToPlane = dot3(delta, normalizedPlaneNormal);

  return [
    point[0] - normalizedPlaneNormal[0] * distanceToPlane,
    point[1] - normalizedPlaneNormal[1] * distanceToPlane,
    point[2] - normalizedPlaneNormal[2] * distanceToPlane,
  ];
}

function projectControlPointsOntoSlicePlane(
  points: [number, number, number][],
  planePointLike: unknown,
  planeNormalLike: unknown
): {
  points: [number, number, number][];
  diagnostics: {
    projected: boolean;
    maxDistanceMm: number;
    meanDistanceMm: number;
  };
} {
  if (
    !Array.isArray(planePointLike) ||
    !Array.isArray(planeNormalLike) ||
    planePointLike.length < 3 ||
    planeNormalLike.length < 3
  ) {
    return {
      points,
      diagnostics: {
        projected: false,
        maxDistanceMm: 0,
        meanDistanceMm: 0,
      },
    };
  }

  const planePoint: [number, number, number] = [
    Number(planePointLike[0] ?? 0),
    Number(planePointLike[1] ?? 0),
    Number(planePointLike[2] ?? 0),
  ];
  const planeNormal: [number, number, number] = [
    Number(planeNormalLike[0] ?? 0),
    Number(planeNormalLike[1] ?? 0),
    Number(planeNormalLike[2] ?? 0),
  ];
  if (
    !Number.isFinite(planePoint[0]) ||
    !Number.isFinite(planePoint[1]) ||
    !Number.isFinite(planePoint[2]) ||
    !Number.isFinite(planeNormal[0]) ||
    !Number.isFinite(planeNormal[1]) ||
    !Number.isFinite(planeNormal[2])
  ) {
    return {
      points,
      diagnostics: {
        projected: false,
        maxDistanceMm: 0,
        meanDistanceMm: 0,
      },
    };
  }

  let maxDistanceMm = 0;
  let totalDistanceMm = 0;
  const normalizedPlaneNormal = normalize3(planeNormal);
  const projectedPoints = points.map(point => {
    const delta: [number, number, number] = [
      point[0] - planePoint[0],
      point[1] - planePoint[1],
      point[2] - planePoint[2],
    ];
    const distanceToPlane = dot3(delta, normalizedPlaneNormal);
    const absDistance = Math.abs(distanceToPlane);
    maxDistanceMm = Math.max(maxDistanceMm, absDistance);
    totalDistanceMm += absDistance;
    return projectPointOntoPlane(point, planePoint, normalizedPlaneNormal);
  });

  return {
    points: projectedPoints,
    diagnostics: {
      projected: true,
      maxDistanceMm,
      meanDistanceMm: projectedPoints.length ? totalDistanceMm / projectedPoints.length : 0,
    },
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

function median(values: number[]): number {
  if (!values.length) {
    return 0;
  }

  const sorted = values.slice().sort((a, b) => a - b);
  const centerIndex = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[centerIndex];
  }

  return (sorted[centerIndex - 1] + sorted[centerIndex]) * 0.5;
}

function computeSplineEdgeLengths(points: [number, number, number][]): number[] {
  const edgeLengths: number[] = [];

  for (let i = 1; i < points.length; i++) {
    edgeLengths.push(dist3(points[i - 1], points[i]));
  }

  return edgeLengths;
}

function sanitizeSplineControlPoints(
  rawPoints: [number, number, number][],
  minSpacingMm: number
): {
  points: [number, number, number][];
  diagnostics: {
    inputCount: number;
    outputCount: number;
    consecutiveDuplicateRemoved: number;
    loopbackToStartRemoved: number;
    terminalClosureRemoved: boolean;
    dedupeThresholdMm: number;
    closureThresholdMm: number;
    suspiciousJumpThresholdMm: number;
    medianEdgeLengthMm: number;
    maxEdgeLengthMm: number;
    suspiciousJumpCount: number;
    severeJumpDetected: boolean;
  };
} {
  const safeMinSpacingMm = Math.max(0.2, toPositiveFinite(minSpacingMm, 0.3));
  const rawEdgeLengths = computeSplineEdgeLengths(rawPoints).filter(length => Number.isFinite(length) && length > 0);
  const medianRawEdgeLengthMm = median(rawEdgeLengths);
  const dedupeThresholdMm = Math.max(
    safeMinSpacingMm * 1.25,
    Math.min(1.25, medianRawEdgeLengthMm > 0 ? medianRawEdgeLengthMm * 0.22 : safeMinSpacingMm * 1.6)
  );
  const closureThresholdMm = Math.max(
    dedupeThresholdMm * 1.5,
    Math.min(2.4, medianRawEdgeLengthMm > 0 ? medianRawEdgeLengthMm * 0.55 : dedupeThresholdMm * 2.2)
  );
  const suspiciousJumpThresholdMm = Math.max(
    12,
    medianRawEdgeLengthMm > 0 ? medianRawEdgeLengthMm * 5.5 : safeMinSpacingMm * 14
  );

  const sanitized: [number, number, number][] = [];
  let consecutiveDuplicateRemoved = 0;
  let loopbackToStartRemoved = 0;

  for (let index = 0; index < rawPoints.length; index++) {
    const candidate = rawPoints[index];
    if (!sanitized.length) {
      sanitized.push(candidate);
      continue;
    }

    const previous = sanitized[sanitized.length - 1];
    if (dist3(previous, candidate) <= dedupeThresholdMm) {
      consecutiveDuplicateRemoved++;
      continue;
    }

    const isInteriorPoint = index < rawPoints.length - 1;
    if (
      isInteriorPoint &&
      sanitized.length >= 3 &&
      dist3(candidate, sanitized[0]) <= closureThresholdMm
    ) {
      loopbackToStartRemoved++;
      continue;
    }

    sanitized.push(candidate);
  }

  let terminalClosureRemoved = false;
  if (
    sanitized.length >= 3 &&
    dist3(sanitized[0], sanitized[sanitized.length - 1]) <= closureThresholdMm
  ) {
    sanitized.pop();
    terminalClosureRemoved = true;
  }

  const sanitizedEdgeLengths = computeSplineEdgeLengths(sanitized).filter(
    length => Number.isFinite(length) && length > 0
  );
  const medianEdgeLengthMm = median(sanitizedEdgeLengths);
  const maxEdgeLengthMm = sanitizedEdgeLengths.length ? Math.max(...sanitizedEdgeLengths) : 0;
  const suspiciousJumpCount = sanitizedEdgeLengths.filter(
    length => length >= suspiciousJumpThresholdMm
  ).length;
  const severeJumpDetected =
    suspiciousJumpCount > 0 &&
    maxEdgeLengthMm >= Math.max(24, medianEdgeLengthMm > 0 ? medianEdgeLengthMm * 8 : 24);

  return {
    points: sanitized,
    diagnostics: {
      inputCount: rawPoints.length,
      outputCount: sanitized.length,
      consecutiveDuplicateRemoved,
      loopbackToStartRemoved,
      terminalClosureRemoved,
      dedupeThresholdMm,
      closureThresholdMm,
      suspiciousJumpThresholdMm,
      medianEdgeLengthMm,
      maxEdgeLengthMm,
      suspiciousJumpCount,
      severeJumpDetected,
    },
  };
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
  windowWidth: number;
  windowCenter: number;
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

interface CPRWorkerInitSuccessMessage {
  type: 'INIT_SUCCESS';
  sessionKey: string;
}

interface CPRWorkerSuccessMessage {
  type: 'SUCCESS';
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  debugPayload?: CPRWorkerLaunchResult['workerDebugPayload'];
}

interface CPRWorkerErrorMessage {
  type: 'ERROR';
  message: string;
}

type CPRWorkerResponseMessage =
  | CPRWorkerInitSuccessMessage
  | CPRWorkerSuccessMessage
  | CPRWorkerErrorMessage;

interface CPRWorkerSession {
  worker: Worker;
  volumeKey: string;
  sessionKey: string;
}

let activeCPRWorkerSession: CPRWorkerSession | null = null;

function buildCPRWorkerVolumeKey(params: {
  volume: cornerstone.Types.IImageVolume;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  scalarLength: number;
  scalarType: string;
  rescaleSlope: number;
  rescaleIntercept: number;
}): string {
  const volumeId = (params.volume as { volumeId?: string | null }).volumeId ?? 'unknown-volume';
  return [
    volumeId,
    params.dimensions.join('x'),
    params.spacing.map(v => Number(v).toFixed(4)).join(','),
    params.origin.map(v => Number(v).toFixed(4)).join(','),
    params.scalarLength,
    params.scalarType,
    Number(params.rescaleSlope).toFixed(6),
    Number(params.rescaleIntercept).toFixed(6),
  ].join('|');
}

function terminateCPRWorkerSession(): void {
  if (activeCPRWorkerSession) {
    activeCPRWorkerSession.worker.terminate();
    activeCPRWorkerSession = null;
  }
}

function postMessageToCPRWorker<T extends CPRWorkerResponseMessage>(
  worker: Worker,
  payload: unknown,
  transferList?: Transferable[]
): Promise<T> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      worker.removeEventListener('message', handleMessage as EventListener);
      worker.removeEventListener('error', handleError as EventListener);
    };
    const handleMessage = (event: MessageEvent<CPRWorkerResponseMessage>) => {
      cleanup();
      const data = event.data;
      if (!data || typeof data !== 'object') {
        reject(new Error('[cprWorker] Invalid worker response.'));
        return;
      }
      if (data.type === 'ERROR') {
        reject(new Error(`[cprWorker] ${data.message}`));
        return;
      }
      resolve(data as T);
    };
    const handleError = (event: ErrorEvent) => {
      cleanup();
      reject(new Error(`[cprWorker] Uncaught worker error: ${event.message}`));
    };

    worker.addEventListener('message', handleMessage as EventListener);
    worker.addEventListener('error', handleError as EventListener);
    if (transferList && transferList.length > 0) {
      worker.postMessage(payload, transferList);
    } else {
      worker.postMessage(payload);
    }
  });
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
  rigidVerticalSliceMode?: boolean;
  debugRunId?: string;
  reconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
}): Promise<CPRWorkerLaunchResult> {
  return new Promise(async (resolve, reject) => {
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
      rigidVerticalSliceMode = false,
      debugRunId,
      reconstructionMode = 'legacy',
    } = params;
    const logPrefix = debugRunId ? `[CPR][${debugRunId}]` : '[CPR]';

    try {
      const scalarData = volume.imageData.getPointData().getScalars().getData() as
        | Float32Array
        | Int16Array;
      const dimensions = volume.imageData.getDimensions() as [number, number, number];
      const spacing = volume.imageData.getSpacing() as [number, number, number];
      const origin = volume.imageData.getOrigin() as [number, number, number];
      const direction = Array.from(volume.imageData.getDirection());
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
        rigidVerticalSliceMode,
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
      const volumeKey = buildCPRWorkerVolumeKey({
        volume,
        dimensions,
        spacing,
        origin,
        scalarLength: scalarData.length,
        scalarType,
        rescaleSlope,
        rescaleIntercept,
      });

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

      if (!activeCPRWorkerSession || activeCPRWorkerSession.volumeKey !== volumeKey) {
        terminateCPRWorkerSession();
        const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });
        const sessionKey = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        const isSharedArrayBuffer = scalarData.buffer instanceof SharedArrayBuffer;
        const initScalarData = isSharedArrayBuffer
          ? scalarData
          : (scalarData.slice(0) as Float32Array | Int16Array);
        const initPayload = {
          type: 'INIT_VOLUME' as const,
          sessionKey,
          scalarData: initScalarData,
          isSharedArrayBuffer,
          dimensions,
          spacing,
          origin,
          direction,
          worldToIndex,
          rescaleSlope,
          rescaleIntercept,
          bitsStored,
          bitsAllocated,
          highBit,
          pixelRepresentation,
          isPreScaled: effectiveIsPreScaled,
        };
        const transferList =
          isSharedArrayBuffer || !initScalarData.buffer ? undefined : [initScalarData.buffer];
        await postMessageToCPRWorker<CPRWorkerInitSuccessMessage>(worker, initPayload, transferList);
        activeCPRWorkerSession = {
          worker,
          volumeKey,
          sessionKey,
        };
      }

      const serializedFrames = frames.map(f => ({
        position: Array.from(f.position) as [number, number, number],
        T: Array.from(f.T) as [number, number, number],
        N_slab: Array.from(f.N_slab) as [number, number, number],
        S: Array.from(f.S) as [number, number, number],
      }));

      const data = await postMessageToCPRWorker<CPRWorkerSuccessMessage>(
        activeCPRWorkerSession.worker,
        {
          type: 'RENDER' as const,
          sessionKey: activeCPRWorkerSession.sessionKey,
          frames: serializedFrames,
          panoWidth,
          panoHeight,
          vertHalfMm: dynamicVertHalfMm,
          verticalCenterOffsetMm,
          rigidVerticalSliceMode,
          slabHalfThicknessMm,
          slabSamples,
          aggregation,
          verticalDir,
          applyModalityLut,
          allowStoredValueNormalization,
          disableStoredValueNormalization:
            forceDisableStoredValueNormalization === true ? true : undefined,
          debugRunId,
          reconstructionMode,
        }
      );

      console.log(`${logPrefix} [CPR-WORKER-MESSAGE-JSON]`, JSON.stringify({
        width: data.panoWidth,
        height: data.panoHeight,
        minValue: data.minValue,
        maxValue: data.maxValue,
        windowWidth: data.windowWidth,
        windowCenter: data.windowCenter,
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
        windowWidth: data.windowWidth,
        windowCenter: data.windowCenter,
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
    } catch (error) {
      if (activeCPRWorkerSession) {
        terminateCPRWorkerSession();
      }
      reject(error instanceof Error ? error : new Error(String(error)));
    }
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

async function waitForVolumeToFullyLoad(
  volume: LoadableImageVolume,
  debugRunId: string,
  timeoutMs = 45000
): Promise<void> {
  const volumeId = volume.volumeId || 'unknown-volume';
  const initialStatus = {
    volumeId,
    loaded: volume.loadStatus?.loaded === true,
    loading: volume.loadStatus?.loading === true,
    timeoutMs,
  };

  console.log(`[CPR][${debugRunId}] source volume readiness check`, initialStatus);
  console.log('[CPR-SOURCE-VOLUME-LOAD-JSON]', JSON.stringify({
    runId: debugRunId,
    status: initialStatus.loaded ? 'already-loaded' : 'waiting',
    ...initialStatus,
  }));

  if (!volume.loadStatus || volume.loadStatus.loaded === true || typeof volume.load !== 'function') {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    let timeoutHandle = 0;

    const cleanup = () => {
      cornerstone.eventTarget.removeEventListener(
        cornerstone.Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED,
        onVolumeLoadCompleted as EventListener
      );
      window.clearTimeout(timeoutHandle);
    };

    const finish = (status: string) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      console.log('[CPR-SOURCE-VOLUME-LOAD-JSON]', JSON.stringify({
        runId: debugRunId,
        status,
        volumeId,
        loaded: volume.loadStatus?.loaded === true,
        loading: volume.loadStatus?.loading === true,
      }));
      resolve();
    };

    const onVolumeLoadCompleted = (event: Event) => {
      const detail = (event as CustomEvent<{ volumeId?: string }>).detail;
      if (detail?.volumeId && detail.volumeId !== volumeId) {
        return;
      }
      finish('loaded');
    };

    cornerstone.eventTarget.addEventListener(
      cornerstone.Enums.Events.IMAGE_VOLUME_LOADING_COMPLETED,
      onVolumeLoadCompleted as EventListener
    );

    timeoutHandle = window.setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(new Error(`[CPR] Timed out waiting for source volume "${volumeId}" to fully load.`));
    }, timeoutMs);

    if (volume.loadStatus?.loaded === true) {
      finish('loaded-before-wait');
      return;
    }

    if (!volume.loadStatus?.loading) {
      volume.load();
      console.log('[CPR-SOURCE-VOLUME-LOAD-JSON]', JSON.stringify({
        runId: debugRunId,
        status: 'triggered-load',
        volumeId,
      }));
    }
  });
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

async function waitForStackViewportByLogicalId(
  servicesManager: any,
  logicalViewportId: string,
  timeoutMs = 12000
): Promise<cornerstone.Types.IStackViewport> {
  const startedAt = Date.now();
  let lastViewportType = 'none';

  while (Date.now() - startedAt < timeoutMs) {
    const viewport = findViewportByLogicalId(servicesManager, logicalViewportId);
    if (viewport) {
      lastViewportType = viewport.constructor?.name || 'unknown';

      if (isStackViewportLike(viewport)) {
        return viewport as cornerstone.Types.IStackViewport;
      }
    }

    await new Promise(resolve => window.setTimeout(resolve, 50));
  }

  throw new Error(
    `[CPR] ${logicalViewportId} stack viewport not ready within timeout. Last resolved type: ${lastViewportType}`
  );
}

async function waitForPanoStackViewport(
  servicesManager: any,
  timeoutMs = 12000
): Promise<cornerstone.Types.IStackViewport> {
  return waitForStackViewportByLogicalId(servicesManager, 'cpr-pano', timeoutMs);
}

async function waitForCrossSectionStackViewport(
  servicesManager: any,
  timeoutMs = 12000
): Promise<cornerstone.Types.IStackViewport> {
  return waitForStackViewportByLogicalId(servicesManager, 'cpr-crosssection', timeoutMs);
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

function getCurrentCrossSectionSeriesId(
  viewport: cornerstone.Types.IViewport | null
): string | null {
  if (!viewport || !isStackViewportLike(viewport)) {
    return null;
  }

  const imageId = getCurrentStackImageId(viewport) ?? viewport.getImageIds?.()?.[0] ?? null;
  if (typeof imageId !== 'string' || !imageId.startsWith('cross://')) {
    return null;
  }

  const remainder = imageId.slice('cross://'.length);
  const separatorIndex = remainder.lastIndexOf('/');
  if (separatorIndex <= 0) {
    return null;
  }

  return remainder.slice(0, separatorIndex) || null;
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

type PreservedCrossSectionDisplaySettings = {
  voiRange: {
    lower: number;
    upper: number;
  };
  VOILUTFunction: string;
};

const pendingCrossSectionVoiReapplyListeners = new WeakMap<
  EventTarget,
  EventListenerOrEventListenerObject
>();

function captureCrossSectionDisplaySettings(
  viewport: cornerstone.Types.IStackViewport | null
): PreservedCrossSectionDisplaySettings | null {
  if (!viewport) {
    return null;
  }

  const properties = ((viewport as { getProperties?: () => unknown }).getProperties?.() || {}) as {
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
  };
  const lower = toFiniteNumber(properties.voiRange?.lower);
  const upper = toFiniteNumber(properties.voiRange?.upper);
  if (lower === undefined || upper === undefined) {
    return null;
  }

  return {
    voiRange: { lower, upper },
    VOILUTFunction:
      typeof properties.VOILUTFunction === 'string' && properties.VOILUTFunction
        ? properties.VOILUTFunction
        : 'LINEAR_EXACT',
  };
}

function applyCrossSectionDisplaySettings(
  viewport: cornerstone.Types.IViewport | null,
  settings: PreservedCrossSectionDisplaySettings | null
): void {
  if (!settings || !viewport || !isStackViewportLike(viewport)) {
    return;
  }

  viewport.setProperties({
    isComputedVOI: false,
    voiRange: {
      lower: settings.voiRange.lower,
      upper: settings.voiRange.upper,
    },
    VOILUTFunction: settings.VOILUTFunction,
  } as any);
}

function applyCrossSectionCameraAlignment(
  viewport: cornerstone.Types.IViewport | null,
  frame: CPRFrame,
  verticalCenterOffsetMm?: number,
  previousCameraOverride?: ReturnType<typeof cloneCameraState> | null
): void {
  if (!viewport) {
    return;
  }

  const previousCamera = previousCameraOverride ?? viewport.getCamera?.();
  const nextCamera = buildCrossSectionCameraForFrame(
    frame,
    previousCamera,
    20,
    toFiniteNumber(verticalCenterOffsetMm) ?? 0
  );
  viewport.setCamera?.(nextCamera);

  const appliedCamera = viewport.getCamera?.();
  const requestedRight = normalize3([
    nextCamera.viewUp[1] * nextCamera.viewPlaneNormal[2] -
      nextCamera.viewUp[2] * nextCamera.viewPlaneNormal[1],
    nextCamera.viewUp[2] * nextCamera.viewPlaneNormal[0] -
      nextCamera.viewUp[0] * nextCamera.viewPlaneNormal[2],
    nextCamera.viewUp[0] * nextCamera.viewPlaneNormal[1] -
      nextCamera.viewUp[1] * nextCamera.viewPlaneNormal[0],
  ]);
  const cameraAxis = normalize3([frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]]);
  const appliedRight =
    appliedCamera?.viewUp && appliedCamera?.viewPlaneNormal
      ? normalize3([
        appliedCamera.viewUp[1] * appliedCamera.viewPlaneNormal[2] -
          appliedCamera.viewUp[2] * appliedCamera.viewPlaneNormal[1],
        appliedCamera.viewUp[2] * appliedCamera.viewPlaneNormal[0] -
          appliedCamera.viewUp[0] * appliedCamera.viewPlaneNormal[2],
        appliedCamera.viewUp[0] * appliedCamera.viewPlaneNormal[1] -
          appliedCamera.viewUp[1] * appliedCamera.viewPlaneNormal[0],
      ])
      : null;

  console.log('[CPR-CROSSSECTION-CAMERA]', {
    frameIndex: frame.index,
    viewportId: viewport.id,
    verticalCenterOffsetMm: toFiniteNumber(verticalCenterOffsetMm) ?? 0,
    requested: {
      focalPoint: nextCamera.focalPoint,
      position: nextCamera.position,
      viewPlaneNormal: nextCamera.viewPlaneNormal,
      viewUp: nextCamera.viewUp,
      viewRight: requestedRight,
      parallelScale: nextCamera.parallelScale,
      flipHorizontal: nextCamera.flipHorizontal,
      flipVertical: nextCamera.flipVertical,
      dotRightCamera:
        requestedRight[0] * cameraAxis[0] +
        requestedRight[1] * cameraAxis[1] +
        requestedRight[2] * cameraAxis[2],
    },
    applied: appliedCamera
      ? {
        focalPoint: appliedCamera.focalPoint,
        position: appliedCamera.position,
        viewPlaneNormal: appliedCamera.viewPlaneNormal,
        viewUp: appliedCamera.viewUp,
        viewRight: appliedRight,
        parallelScale: appliedCamera.parallelScale,
        flipHorizontal: appliedCamera.flipHorizontal,
        flipVertical: appliedCamera.flipVertical,
        dotRightCamera:
          appliedRight
            ? (
              appliedRight[0] * cameraAxis[0] +
              appliedRight[1] * cameraAxis[1] +
              appliedRight[2] * cameraAxis[2]
            )
            : null,
      }
      : null,
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

function setCrossSectionForFrame(
  frame: CPRFrame,
  servicesManager: any,
  verticalCenterOffsetMm?: number
): void {
  const crossViewport = findViewportByLogicalId(servicesManager, 'cpr-crosssection');
  if (!crossViewport) {
    return;
  }

  if (isStackViewportLike(crossViewport)) {
    if (crossViewport.element) {
      const pendingListener = pendingCrossSectionVoiReapplyListeners.get(crossViewport.element);
      if (pendingListener) {
        crossViewport.element.removeEventListener(
          cornerstone.Enums.Events.IMAGE_RENDERED,
          pendingListener
        );
        pendingCrossSectionVoiReapplyListeners.delete(crossViewport.element);
      }
    }

    const preservedDisplaySettings = captureCrossSectionDisplaySettings(crossViewport);
    const preservedCamera = cloneCameraState(crossViewport);
    const currentImageIndex = Number(crossViewport.getCurrentImageIdIndex?.());
    const safeCurrentIndex = Number.isFinite(currentImageIndex) ? Math.floor(currentImageIndex) : -1;

    if (crossViewport.element && safeCurrentIndex !== frame.index) {
      const targetImageIndex = frame.index;
      const onTargetImageRendered: EventListener = () => {
        const liveViewport = findViewportByLogicalId(servicesManager, 'cpr-crosssection');
        if (!liveViewport || !isStackViewportLike(liveViewport)) {
          return;
        }

        const renderedImageIndex = Number(liveViewport.getCurrentImageIdIndex?.());
        if (!Number.isFinite(renderedImageIndex) || Math.floor(renderedImageIndex) !== targetImageIndex) {
          return;
        }

        crossViewport.element.removeEventListener(
          cornerstone.Enums.Events.IMAGE_RENDERED,
          onTargetImageRendered
        );
        pendingCrossSectionVoiReapplyListeners.delete(crossViewport.element);
        applyCrossSectionCameraAlignment(
          liveViewport,
          frame,
          verticalCenterOffsetMm,
          preservedCamera
        );
        applyCrossSectionDisplaySettings(liveViewport, preservedDisplaySettings);
        liveViewport.render?.();
      };

      crossViewport.element.addEventListener(
        cornerstone.Enums.Events.IMAGE_RENDERED,
        onTargetImageRendered
      );
      pendingCrossSectionVoiReapplyListeners.set(crossViewport.element, onTargetImageRendered);
      applyCrossSectionDisplaySettings(crossViewport, preservedDisplaySettings);
      const crossSectionSeriesId = getCurrentCrossSectionSeriesId(crossViewport);
      const savedVoi = preservedDisplaySettings?.voiRange;
      if (crossSectionSeriesId && savedVoi) {
        updateCrossSectionSeriesDisplayDefaults(crossSectionSeriesId, savedVoi);
      }

      cornerstoneTools.utilities.jumpToSlice(crossViewport.element, {
        imageIndex: frame.index,
        debounceLoading: true,
      });
    } else {
      applyCrossSectionDisplaySettings(crossViewport, preservedDisplaySettings);
      applyCrossSectionCameraAlignment(
        crossViewport,
        frame,
        verticalCenterOffsetMm,
        preservedCamera
      );
    }
    crossViewport.render?.();
    return;
  }

  applyCrossSectionCameraAlignment(crossViewport, frame, verticalCenterOffsetMm);
  crossViewport.render();
}

function buildCrossSectionStackConfig(
  _sourceVolumeId: string,
  _verticalHalfHeightMm?: number
): {
  width: number;
  height: number;
  rowPixelSpacing: number;
  columnPixelSpacing: number;
  horizontalHalfWidthMm: number;
  verticalHalfHeightMm: number;
} {
  const width = 256;
  const height = 256;
  const rowPixelSpacing = 0.16;
  const columnPixelSpacing = 0.16;
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

type VolumeLoadStatusLike = {
  loaded?: boolean;
  loading?: boolean;
};

type LoadableImageVolume = cornerstone.Types.IImageVolume & {
  loadStatus?: VolumeLoadStatusLike;
  load?: (callback?: (...args: unknown[]) => void) => void;
};

async function initializeCrossSection(
  frames: CPRFrame[],
  sourceVolumeId: string,
  verticalHalfHeightMm: number | undefined,
  verticalCenterOffsetMm: number | undefined,
  samplingOffsetsActive: boolean,
  servicesManager: any
): Promise<void> {
  const { syncGroupService } = servicesManager.services;
  const crossViewport = await waitForCrossSectionStackViewport(servicesManager);

  const initialFrameIndex = Math.max(
    0,
    Math.min(cprStateService.getCurrentFrameIndex(), Math.max(0, frames.length - 1))
  );
  const stackConfig = buildCrossSectionStackConfig(sourceVolumeId, verticalHalfHeightMm);
  const crossSeriesId = createCrossSectionSeriesId();
  const crossImageIds = createCrossSectionImageIds(crossSeriesId, frames.length);

  clearCrossSectionImageCache();
  setCrossSectionSeriesPayload(crossSeriesId, {
    sourceVolumeId,
    frames,
    width: stackConfig.width,
    height: stackConfig.height,
    rowPixelSpacing: stackConfig.rowPixelSpacing,
    columnPixelSpacing: stackConfig.columnPixelSpacing,
    horizontalHalfWidthMm: stackConfig.horizontalHalfWidthMm,
    verticalHalfHeightMm: stackConfig.verticalHalfHeightMm,
    verticalCenterOffsetMm: toFiniteNumber(verticalCenterOffsetMm) ?? 0,
  });
  console.log('[CPR-CROSSSECTION-CONFIG-JSON]', JSON.stringify({
    sourceVolumeId,
    frameCount: frames.length,
    initialFrameIndex,
    width: stackConfig.width,
    height: stackConfig.height,
    rowPixelSpacing: stackConfig.rowPixelSpacing,
    columnPixelSpacing: stackConfig.columnPixelSpacing,
    horizontalHalfWidthMm: stackConfig.horizontalHalfWidthMm,
    verticalHalfHeightMm: stackConfig.verticalHalfHeightMm,
    verticalCenterOffsetMm: toFiniteNumber(verticalCenterOffsetMm) ?? 0,
    geometryMode: 'fixed-grid-rigid-world',
    samplingOffsetsActive,
  }));

  await crossViewport.setStack(crossImageIds, initialFrameIndex);
  crossViewport.resetProperties?.();
  crossViewport.setProperties?.({
    invert: false,
    VOILUTFunction: 'LINEAR_EXACT',
  } as any);
  crossViewport.resetCamera?.();
  applyCrossSectionCameraAlignment(
    crossViewport,
    frames[initialFrameIndex],
    verticalCenterOffsetMm
  );
  crossViewport.render?.();

  const syncId = 'cpr-crosssection-sync';
  const renderingEngineId = crossViewport.getRenderingEngine().id;
  const crossViewportId = crossViewport.id;
  syncGroupService.removeViewportFromSyncGroup(crossViewportId, renderingEngineId, syncId);
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
  const sliderAnimationFrameRef = useRef<number | null>(null);
  const pendingSliderFrameIndexRef = useRef<number | null>(null);
  const crossSectionVerticalCenterOffsetMmRef = useRef(0);

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
      if (sliderAnimationFrameRef.current != null) {
        window.cancelAnimationFrame(sliderAnimationFrameRef.current);
        sliderAnimationFrameRef.current = null;
      }
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
      const rawControlPointsWorld: [number, number, number][] = rawPoints
        .map(toXYZTuple)
        .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]));

      if (rawControlPointsWorld.length < 2) {
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
      await waitForVolumeToFullyLoad(volume as LoadableImageVolume, debugRunId);
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
      const projectedSpline = projectControlPointsOntoSlicePlane(
        rawControlPointsWorld,
        preservedAxialCamera?.focalPoint,
        preservedAxialCamera?.viewPlaneNormal
      );
      const rawControlPoints = projectedSpline.points;
      console.log('[CPR-SPLINE-PLANE-JSON]', JSON.stringify({
        runId: debugRunId,
        rawPointCount: rawControlPointsWorld.length,
        projectedPointCount: rawControlPoints.length,
        projected: projectedSpline.diagnostics.projected,
        maxDistanceMm: projectedSpline.diagnostics.maxDistanceMm,
        meanDistanceMm: projectedSpline.diagnostics.meanDistanceMm,
      }));
      console.log('[CPR-RIGID-SLICE-JSON]', JSON.stringify({
        runId: debugRunId,
        enabled: true,
        slicePlanePoint: preservedAxialCamera?.focalPoint ?? null,
        slicePlaneNormal: preservedAxialCamera?.viewPlaneNormal ?? null,
        usesProjectedSplinePlane: projectedSpline.diagnostics.projected,
        projectedMaxDistanceMm: projectedSpline.diagnostics.maxDistanceMm,
        projectedMeanDistanceMm: projectedSpline.diagnostics.meanDistanceMm,
        rigidVerticalCenterOffsetMm: 0,
      }));
      const sanitizedSpline = sanitizeSplineControlPoints(rawControlPoints, minSpacing);
      const controlPoints = sanitizedSpline.points;
      console.log('[CPR-SPLINE-SANITIZED-JSON]', JSON.stringify({
        runId: debugRunId,
        rawPointCount: rawControlPointsWorld.length,
        sanitizedPointCount: controlPoints.length,
        diagnostics: sanitizedSpline.diagnostics,
      }));
      if (sanitizedSpline.diagnostics.severeJumpDetected) {
        console.warn(
          `[CPR][${debugRunId}] Spline contains one or more suspicious long edges; proceeding with sanitized control points.`,
          sanitizedSpline.diagnostics
        );
      }
      if (controlPoints.length < 2) {
        throw new Error('Arch annotation collapsed after sanitization. Please redraw the arch.');
      }

      const totalArcLength = computeSplineTotalArcLength(controlPoints);
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
        Math.min(CPR_PANO_MAX_VERTICAL_HALF_MM, autoVerticalHalfMm * 0.84)
      );
      const thinnerVerticalHalfMm = Math.max(14, Math.min(19, baseVerticalHalfMm * 0.82));
      const narrowVerticalHalfMm = Math.max(11.5, Math.min(15.5, thinnerVerticalHalfMm * 0.86));
      const toothBandVerticalHalfMm = Math.max(10, Math.min(13.5, narrowVerticalHalfMm * 0.9));
      const mediumVerticalHalfMm = Math.max(
        narrowVerticalHalfMm + 1.25,
        Math.min(18, baseVerticalHalfMm * 0.92)
      );
      const broadVerticalHalfMm = Math.max(
        mediumVerticalHalfMm + 1.5,
        Math.min(22, mediumVerticalHalfMm * 1.14)
      );
      const neutralVerticalCenterOffsetMm = 0;
      const rigidSliceVerticalCenterOffsetMm = 0;
      const rigidVerticalSliceModeEnabled = false;
      const mildSuperiorCenterOffsetMm = 1.4;
      const strongSuperiorCenterOffsetMm = 2.6;
      const superiorMandibularCenterOffsetMm = -2.4;
      const subtleMandibularCenterOffsetMm = -3.8;
      const mandibularCenterOffsetMm = -5.2;
      const strongMandibularCenterOffsetMm = -6.6;
      const balancedSlabHalfThicknessMm = 1.6;
      const balancedSlabSamples = 9;
      const fastSlabHalfThicknessMm = 1.2;
      const fastSlabSamples = 7;
      const balancedMeanSlabHalfThicknessMm = 1.5;
      const balancedMeanSlabSamples = 11;
      const focusedMeanSlabHalfThicknessMm = 0.35;
      const focusedMeanSlabSamples = 3;
      const broadMeanSlabHalfThicknessMm = 2.0;
      const broadMeanSlabSamples = 15;
      const meanFallbackSlabHalfThicknessMm = 1.0;
      const meanFallbackSlabSamples = 9;
      const sharpMeanSlabHalfThicknessMm = 0.2;   // ~2 voxels total slab
      const sharpMeanSlabSamples = 3;
      const minimumPanoHeightPx = Math.max(160, Math.round(requestedPanoHeightPx * 0.55));

      const verticalDir = getSourceViewportVerticalDirection(
        axialViewport,
        volume.imageData?.getDirection?.()
      );
      const { positions, tangents } = buildArcLengthSpline(controlPoints, finalPanoWidth);
      const frames = buildRMFFrames(positions, tangents, verticalDir);
      console.log(`[CPR][${debugRunId}] frame geometry summary`, {
        rawControlPointCount: rawControlPoints.length,
        controlPointCount: controlPoints.length,
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
      const panoAttemptSequenceStartMs = performance.now();
      let launchedAttemptCount = 0;
      let launchedMipFallbackCount = 0;
      let earlyExitReason: string | null = null;
      const workerInput = {
        volume,
        frames,
        slabHalfThicknessMm,
        slabSamples,
        aggregation,
        verticalDir,
        verticalHalfMm: baseVerticalHalfMm,
        verticalCenterOffsetMm: rigidSliceVerticalCenterOffsetMm,
        rigidVerticalSliceMode: rigidVerticalSliceModeEnabled,
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
        intensityDomain: SyntheticCprIntensityDomain;
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
        durationMs: number;
        requestOverrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
        workerTimingMs?: {
          adaptiveCenterSearch?: number;
          pass1And2TwoPassRender?: number;
          virtualPanoPhase12?: number;
          suppressionAndDenoise?: number;
          diagnosticAssembly?: number;
          total?: number;
        } | null;
      }> => {
        launchedAttemptCount++;
        const attemptStartMs = performance.now();
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
          rigidVerticalSliceModeEnabled
            ? rigidSliceVerticalCenterOffsetMm
            : overrides.verticalCenterOffsetMm ?? workerInput.verticalCenterOffsetMm
        ) ?? rigidSliceVerticalCenterOffsetMm;
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
          rigidVerticalSliceMode: rigidVerticalSliceModeEnabled,
        });
        const result = rawResult;
        const hasIdentityRescaleMetadata =
          Math.abs(result.rescaleSlope - 1) <= 1e-6 && Math.abs(result.rescaleIntercept) <= 1e-6;
        const intensityDomain = classifySyntheticCprIntensityDomain({
          modalityLutApplied: result.modalityLutApplied,
          effectiveIsPreScaled: result.effectiveIsPreScaled,
          minValue: result.minValue,
          maxValue: result.maxValue,
        });
        const huDomain = isSyntheticCprHuDomain(intensityDomain);
        const convertedToHu = false;
        const rescaleSkippedAsUnsafe = intensityDomain === 'unknown';
        const summary = summarizeFloatBufferForDebug(result.pixelData, finalPanoWidth, finalPanoHeight);
        const voi = computeAdaptivePanoVoi(summary, result.minValue, result.maxValue);
        const qualityBase = scorePanoQuality(summary);
        const workerDiagnostic =
          result.workerDebugPayload &&
            typeof result.workerDebugPayload === 'object' &&
            result.workerDebugPayload.diagnostic &&
            typeof result.workerDebugPayload.diagnostic === 'object'
            ? (result.workerDebugPayload.diagnostic as Record<string, unknown>)
            : null;
        const actualVerticalCenterOffsetMm = toFiniteNumber(workerDiagnostic?.verticalCenterOffsetMm);
        const baseVerticalCenterOffsetMm = toFiniteNumber(workerDiagnostic?.baseVerticalCenterOffsetMm);
        const fittedVerticalCenterOffsetMm = toFiniteNumber(
          workerDiagnostic?.fittedVerticalCenterOffsetMm ?? workerDiagnostic?.globalVerticalCenterOffsetMm
        );
        const actualCenterDriftMm =
          actualVerticalCenterOffsetMm !== undefined
            ? Math.abs(actualVerticalCenterOffsetMm - requestedVerticalCenterOffsetMm)
            : 0;
        const baseCenterDriftMm =
          baseVerticalCenterOffsetMm !== undefined
            ? Math.abs(baseVerticalCenterOffsetMm - requestedVerticalCenterOffsetMm)
            : actualCenterDriftMm;
        const splitPenalty = summary
          ? Math.max(0, summary.fractionBelowMinus950 - 0.3) * 8 +
          Math.max(0, summary.fractionAbove3000 - 0.2) * 10 +
          Math.max(0, summary.p50 - 2200) / 300 +
          Math.max(0, -1700 - summary.p50) / 300 +
          Math.max(0, summary.max - 12000) / 1500 +
          Math.max(0, -9000 - summary.min) / 1500
          : 8;
        const denseFillPenalty = summary
          ? Math.max(0, summary.p50 - 900) / 90 + Math.max(0, summary.p50 - 1200) / 55
          : 0;
        const specklePenalty = summary ? Math.max(0, summary.meanAbsDelta - 460) / 60 : 0;
        const focalTroughPenalty =
          Math.max(0, actualVertHalfMm - 45) / 2.0 +
          Math.max(0, requestedSlabHalfThicknessMm - 0.8) *
          (requestedAggregation === 'MIP' ? 5 : 3.5);
        const lowerBandFillPenalty = summary
          ? Math.max(0, summary.lowerBandP50 + 140) / 38 +
          Math.max(0, summary.lowerBandBrightFraction - 0.24) * 24 +
          Math.max(0, summary.lowerBandBrightFraction - 0.55) * 42
          : 0;
        const toothBandSaturationPenalty = summary
          ? Math.max(0, summary.toothBandMean - 760) / 65 +
          Math.max(0, summary.toothBandP10 - 80) / 28 +
          Math.max(0, summary.toothBandBrightFraction - 0.24) * 22
          : 0;
        const detailBalanceRatio = summary
          ? summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean)
          : 1;
        const balancedDetailEdgeMean = summary
          ? Math.min(summary.detailBandHorizontalEdgeMean, summary.detailBandVerticalEdgeMean * 2.4)
          : 0;
        const toothBandContrastRange = summary ? summary.toothBandP90 - summary.toothBandP10 : 0;
        const detailReward = summary
          ? Math.max(-2.5, Math.min(4, (balancedDetailEdgeMean - 55) / 28)) -
          Math.max(0, detailBalanceRatio - 2.6) * 2.2
          : 0;
        const deformationPenalty = summary
          ? Math.max(0, detailBalanceRatio - 2.8) * 3.6 +
          Math.max(0, summary.detailBandHorizontalEdgeMean - 240) / 28
          : 0;
        const tallFillPenalty = summary
          ? Math.max(0, actualVertHalfMm - 42) * Math.max(0, summary.lowerBandBrightFraction - 0.45) * 28
          : 0;
        const noAirPenalty = summary
          ? summary.fractionBelowMinus950 < 0.005
            ? 2
            : summary.fractionBelowMinus950 < 0.015
              ? 1
              : 0
          : 0;
        const elevatedP01Penalty = summary ? Math.max(0, summary.p01 + 780) / 80 : 0;
        const centerDriftPenalty =
          Math.max(0, actualCenterDriftMm - 1.5) * 6 +
          Math.max(0, baseCenterDriftMm - 1.0) * 8;
        const aggregationPenalty =
          requestedAggregation === 'MIP'
            ? 4.5 + (summary ? Math.max(0, summary.meanAbsDelta - 300) / 45 : 0)
            : 0;
        const baseHardRejectReason = getHardRejectReason(summary);
        const excessiveCenterDrift =
          actualCenterDriftMm > Math.max(4, actualVertHalfMm * 0.2) ||
          baseCenterDriftMm > Math.max(3.5, actualVertHalfMm * 0.16);
        const hardRejectReason =
          !baseHardRejectReason && excessiveCenterDrift
            ? 'vertical-center-drift'
            : !baseHardRejectReason &&
                !!summary &&
                requestedAggregation === 'MEAN' &&
                actualVertHalfMm > 45 &&
                summary.lowerBandBrightFraction > 0.64
              ? 'tall-lower-band-fill'
              : baseHardRejectReason;
        const hardRejectPenalty = hardRejectReason ? 30 : 0;
        const intensityDomainPenalty = intensityDomain === 'unknown' ? 12 : 0;
        const qualityScore =
          qualityBase +
          detailReward +
          (huDomain ? 2 : 0) -
          splitPenalty -
          denseFillPenalty -
          specklePenalty -
          focalTroughPenalty -
          lowerBandFillPenalty -
          toothBandSaturationPenalty -
          noAirPenalty -
          elevatedP01Penalty -
          centerDriftPenalty -
          deformationPenalty -
          tallFillPenalty -
          intensityDomainPenalty -
          aggregationPenalty -
          hardRejectPenalty;
        const attemptDurationMs = performance.now() - attemptStartMs;
        const workerTimingMs =
          result.workerDebugPayload &&
            typeof result.workerDebugPayload === 'object' &&
            result.workerDebugPayload.diagnostic &&
            typeof result.workerDebugPayload.diagnostic === 'object' &&
            (result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs &&
            typeof (result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs === 'object'
            ? ((result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs as {
              adaptiveCenterSearch?: number;
              pass1And2TwoPassRender?: number;
              virtualPanoPhase12?: number;
              suppressionAndDenoise?: number;
              diagnosticAssembly?: number;
              total?: number;
            })
            : null;

        console.log(`[CPR][${debugRunId}] pano attempt ${label}`, {
          durationMs: Math.round(attemptDurationMs),
          workerTimingMs,
          qualityBase,
          qualityScore,
          aggregation: requestedAggregation,
          minValue: result.minValue,
          maxValue: result.maxValue,
          p01: summary?.p01,
          p50: summary?.p50,
          p99: summary?.p99,
          meanAbsDelta: summary?.meanAbsDelta,
          lowerBandP50: summary?.lowerBandP50,
          lowerBandBrightFraction: summary?.lowerBandBrightFraction,
          detailBandHorizontalEdgeMean: summary?.detailBandHorizontalEdgeMean,
          detailBandVerticalEdgeMean: summary?.detailBandVerticalEdgeMean,
          fractionBelowMinus950: summary?.fractionBelowMinus950,
          fractionAbove3000: summary?.fractionAbove3000,
          actualVerticalCenterOffsetMm,
          baseVerticalCenterOffsetMm,
          fittedVerticalCenterOffsetMm,
          actualCenterDriftMm,
          baseCenterDriftMm,
          intensityDomain,
          huDomain,
          convertedToHu,
          rescaleSkippedAsUnsafe,
          hasIdentityRescaleMetadata,
          modalityLutApplied: result.modalityLutApplied,
          requestedModalityLutApplied: result.requestedModalityLutApplied,
          effectiveIsPreScaled: result.effectiveIsPreScaled,
          rescaleSlope: result.rescaleSlope,
          rescaleIntercept: result.rescaleIntercept,
          intensityDomainPenalty,
          actualVertHalfMm,
          verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
          idealPanoHeight,
          finalPanoHeight,
          slabHalfThicknessMm: requestedSlabHalfThicknessMm,
          slabSamples: requestedSlabSamples,
          splitPenalty,
          denseFillPenalty,
          specklePenalty,
          focalTroughPenalty,
          lowerBandFillPenalty,
          toothBandSaturationPenalty,
          detailReward,
          deformationPenalty,
          tallFillPenalty,
          noAirPenalty,
          elevatedP01Penalty,
          centerDriftPenalty,
          toothBandMean: summary?.toothBandMean,
          toothBandP10: summary?.toothBandP10,
          toothBandP90: summary?.toothBandP90,
          toothBandBrightFraction: summary?.toothBandBrightFraction,
          toothBandContrastRange,
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
            durationMs: Math.round(attemptDurationMs),
            workerTimingMs,
            intensityDomain,
            huDomain,
            convertedToHu,
            rescaleSkippedAsUnsafe,
            minValue: result.minValue,
            maxValue: result.maxValue,
            p01: summary?.p01 ?? null,
            p50: summary?.p50 ?? null,
            p99: summary?.p99 ?? null,
            meanAbsDelta: summary?.meanAbsDelta ?? null,
            toothBandMean: summary?.toothBandMean ?? null,
            toothBandP10: summary?.toothBandP10 ?? null,
            toothBandP90: summary?.toothBandP90 ?? null,
            toothBandBrightFraction: summary?.toothBandBrightFraction ?? null,
            lowerBandP50: summary?.lowerBandP50 ?? null,
            lowerBandBrightFraction: summary?.lowerBandBrightFraction ?? null,
            detailBandHorizontalEdgeMean: summary?.detailBandHorizontalEdgeMean ?? null,
            detailBandVerticalEdgeMean: summary?.detailBandVerticalEdgeMean ?? null,
            toothBandSaturationPenalty,
            fractionBelowMinus950: summary?.fractionBelowMinus950 ?? null,
            fractionAbove3000: summary?.fractionAbove3000 ?? null,
            splitPenalty,
            denseFillPenalty,
            specklePenalty,
            focalTroughPenalty,
            lowerBandFillPenalty,
            detailReward,
            deformationPenalty,
            tallFillPenalty,
            noAirPenalty,
            elevatedP01Penalty,
            actualVerticalCenterOffsetMm,
            baseVerticalCenterOffsetMm,
            fittedVerticalCenterOffsetMm,
            actualCenterDriftMm,
            baseCenterDriftMm,
            centerDriftPenalty,
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
          intensityDomain,
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
          durationMs: attemptDurationMs,
          requestOverrides: { ...overrides },
          workerTimingMs,
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
        intensityDomain: SyntheticCprIntensityDomain;
        huDomain: boolean;
        sampledCount: number | null;
        min: number | null;
        max: number | null;
        p01: number | null;
        p50: number | null;
        p99: number | null;
        meanAbsDelta: number | null;
        toothBandMean: number | null;
        toothBandP10: number | null;
        toothBandP90: number | null;
        toothBandBrightFraction: number | null;
        lowerBandP50: number | null;
        lowerBandBrightFraction: number | null;
        detailBandHorizontalEdgeMean: number | null;
        detailBandVerticalEdgeMean: number | null;
        fractionBelowMinus950: number | null;
        fractionAbove3000: number | null;
        durationMs: number;
        workerTimingMs?: {
          adaptiveCenterSearch?: number;
          pass1And2TwoPassRender?: number;
          virtualPanoPhase12?: number;
          suppressionAndDenoise?: number;
          diagnosticAssembly?: number;
          total?: number;
        } | null;
      }> = [];
      const attemptResults: Array<Awaited<ReturnType<typeof runWorkerAttempt>>> = [];
      const recordAttemptAudit = (attempt: {
        label: string;
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        aggregation: 'MIP' | 'MEAN';
        slabHalfThicknessMm: number;
        slabSamples: number;
        verticalCenterOffsetMm: number;
        intensityDomain: SyntheticCprIntensityDomain;
        huDomain: boolean;
        summary: FloatBufferDebugSummary | null;
        durationMs: number;
        workerTimingMs?: {
          adaptiveCenterSearch?: number;
          pass1And2TwoPassRender?: number;
          virtualPanoPhase12?: number;
          suppressionAndDenoise?: number;
          diagnosticAssembly?: number;
          total?: number;
        } | null;
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
          intensityDomain: attempt.intensityDomain,
          huDomain: attempt.huDomain,
          sampledCount: attempt.summary?.sampledCount ?? null,
          min: attempt.summary?.min ?? null,
          max: attempt.summary?.max ?? null,
          p01: attempt.summary?.p01 ?? null,
          p50: attempt.summary?.p50 ?? null,
          p99: attempt.summary?.p99 ?? null,
          meanAbsDelta: attempt.summary?.meanAbsDelta ?? null,
          toothBandMean: attempt.summary?.toothBandMean ?? null,
          toothBandP10: attempt.summary?.toothBandP10 ?? null,
          toothBandP90: attempt.summary?.toothBandP90 ?? null,
          toothBandBrightFraction: attempt.summary?.toothBandBrightFraction ?? null,
          lowerBandP50: attempt.summary?.lowerBandP50 ?? null,
          lowerBandBrightFraction: attempt.summary?.lowerBandBrightFraction ?? null,
          detailBandHorizontalEdgeMean: attempt.summary?.detailBandHorizontalEdgeMean ?? null,
          detailBandVerticalEdgeMean: attempt.summary?.detailBandVerticalEdgeMean ?? null,
          fractionBelowMinus950: attempt.summary?.fractionBelowMinus950 ?? null,
          fractionAbove3000: attempt.summary?.fractionAbove3000 ?? null,
          durationMs: Math.round(attempt.durationMs),
          workerTimingMs: attempt.workerTimingMs ?? null,
        });
      };
      const recordAttempt = (attempt: Awaited<ReturnType<typeof runWorkerAttempt>>): void => {
        attemptResults.push(attempt);
        recordAttemptAudit(attempt);
      };
      const isGoodEnoughPanoAttempt = (attempt: {
        qualityScore: number;
        hardRejectReason: string | null;
        summary: FloatBufferDebugSummary | null;
      }): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return false;
        }
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        const hasCleanLowerBand =
          summary.lowerBandBrightFraction <= 0.32 && summary.lowerBandP50 <= -120;
        const hasCleanToothBand =
          summary.toothBandBrightFraction <= 0.26 &&
          summary.toothBandP10 <= 140 &&
          summary.toothBandMean <= 820;
        const hasVeryStrongOverallScore =
          attempt.qualityScore >= 24 &&
          summary.lowerBandBrightFraction <= 0.4 &&
          summary.lowerBandP50 <= -60 &&
          hasCleanToothBand &&
          detailRatio <= 3.4;
        return (
          hasVeryStrongOverallScore ||
          (
            attempt.qualityScore >= 16 &&
            hasCleanLowerBand &&
            hasCleanToothBand &&
            summary.meanAbsDelta <= 520 &&
            summary.fractionAbove3000 <= 0.005 &&
            detailRatio <= 3.4
          )
        );
      };
      const needsFallbackAttempt = (attempt: {
        qualityScore: number;
        hardRejectReason: string | null;
        summary: FloatBufferDebugSummary | null;
      }): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return true;
        }
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        return (
          attempt.qualityScore < 2 ||
          summary.lowerBandBrightFraction > 0.62 ||
          summary.lowerBandP50 > 120 ||
          summary.meanAbsDelta > 680 ||
          detailRatio > 4.2
        );
      };
      const needsExtendedMeanRetrySearch = (attempt: {
        qualityScore: number;
        hardRejectReason: string | null;
        summary: FloatBufferDebugSummary | null;
      }): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return true;
        }
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        return (
          attempt.qualityScore < 10 ||
          summary.toothBandBrightFraction > 0.3 ||
          summary.toothBandP10 > 180 ||
          summary.toothBandMean > 860 ||
          summary.lowerBandBrightFraction > 0.42 ||
          summary.lowerBandP50 > -40 ||
          summary.meanAbsDelta > 560 ||
          detailRatio > 3.6
        );
      };

      let bestAttempt = await runWorkerAttempt('primary-mean-toothband-narrow', {
        modalityLutOverride: true,
        verticalHalfMm: toothBandVerticalHalfMm,
        verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
        slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
        slabSamples: focusedMeanSlabSamples,
        aggregation: 'MEAN',
      });
      recordAttempt(bestAttempt);

      if (isGoodEnoughPanoAttempt(bestAttempt)) {
        earlyExitReason = 'primary-good-enough';
      } else {
        const retryConfigs: Array<{
          label: string;
          overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
        }> = [];

        if (bestAttempt.result.effectiveIsPreScaled && needsExtendedMeanRetrySearch(bestAttempt)) {
          // Evaluate a no-LUT variant when source appears pre-scaled.
          retryConfigs.push({
            label: 'retry-no-lut-mean-narrow',
            overrides: {
              modalityLutOverride: false,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: toothBandVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
              slabSamples: focusedMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
        } else if (needsExtendedMeanRetrySearch(bestAttempt)) {
          retryConfigs.push({
            label: 'retry-force-lut-mean-narrow-no-normalization',
            overrides: {
              modalityLutOverride: true,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: toothBandVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
              slabSamples: focusedMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
        }

        // Balanced MEAN attempts for clinically readable tooth and background separation
        retryConfigs.push({
          label: 'retry-mean-toothband-neutral',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: toothBandVerticalHalfMm,
            verticalCenterOffsetMm: neutralVerticalCenterOffsetMm,
            slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
            slabSamples: focusedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-toothband-superior',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: toothBandVerticalHalfMm,
            verticalCenterOffsetMm: superiorMandibularCenterOffsetMm,
            slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
            slabSamples: focusedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-toothband-balanced',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: toothBandVerticalHalfMm,
            verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
            slabHalfThicknessMm: meanFallbackSlabHalfThicknessMm,
            slabSamples: meanFallbackSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-toothband-rooted',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: narrowVerticalHalfMm,
            verticalCenterOffsetMm: mandibularCenterOffsetMm,
            slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
            slabSamples: focusedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-balanced-medium',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: mediumVerticalHalfMm,
            verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
            slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
            slabSamples: balancedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-balanced-medium-strong-bias',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: mediumVerticalHalfMm,
            verticalCenterOffsetMm: strongMandibularCenterOffsetMm,
            slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
            slabSamples: balancedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        // Sharp MEAN attempts - thin slab for maximum tooth separation
        retryConfigs.push({
          label: 'retry-mean-sharp-narrow',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: toothBandVerticalHalfMm,
            verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
            slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
            slabSamples: sharpMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        // Softer MEAN fallback attempt only when the primary/core search is still poor.
        if (needsFallbackAttempt(bestAttempt)) {
          retryConfigs.push({
            label: 'retry-mean-fallback-narrow',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: mandibularCenterOffsetMm,
              slabHalfThicknessMm: meanFallbackSlabHalfThicknessMm,
              slabSamples: meanFallbackSlabSamples,
              aggregation: 'MEAN',
            },
          });
        }
        if (needsExtendedMeanRetrySearch(bestAttempt)) {
          retryConfigs.push({
            label: 'retry-mean-balanced-narrow',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
              slabSamples: balancedMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-balanced-neutral',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: neutralVerticalCenterOffsetMm,
              slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
              slabSamples: balancedMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-balanced-mild-superior',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: mildSuperiorCenterOffsetMm,
              slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
              slabSamples: balancedMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-broad-medium',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: mediumVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: broadMeanSlabHalfThicknessMm,
              slabSamples: broadMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-broad-neutral',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: mediumVerticalHalfMm,
              verticalCenterOffsetMm: neutralVerticalCenterOffsetMm,
              slabHalfThicknessMm: broadMeanSlabHalfThicknessMm,
              slabSamples: broadMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-broad-superior',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: mediumVerticalHalfMm,
              verticalCenterOffsetMm: strongSuperiorCenterOffsetMm,
              slabHalfThicknessMm: broadMeanSlabHalfThicknessMm,
              slabSamples: broadMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-sharp-medium',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: mediumVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: sharpMeanSlabHalfThicknessMm,
              slabSamples: sharpMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-broad-biased',
            overrides: {
              modalityLutOverride: true,
              verticalHalfMm: broadVerticalHalfMm,
              verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
              slabHalfThicknessMm: broadMeanSlabHalfThicknessMm,
              slabSamples: broadMeanSlabSamples,
              aggregation: 'MEAN',
            },
          });
          retryConfigs.push({
            label: 'retry-mean-fallback-narrow-no-normalization',
            overrides: {
              modalityLutOverride: true,
              forceDisableStoredValueNormalization: true,
              verticalHalfMm: narrowVerticalHalfMm,
              verticalCenterOffsetMm: mandibularCenterOffsetMm,
              slabHalfThicknessMm: meanFallbackSlabHalfThicknessMm,
              slabSamples: meanFallbackSlabSamples,
              aggregation: 'MEAN',
            },
          });
        }

        for (const retryConfig of retryConfigs) {
          const attempt = await runWorkerAttempt(retryConfig.label, retryConfig.overrides);
          recordAttempt(attempt);
          const bestHardRejected = !!bestAttempt.hardRejectReason;
          const attemptHardRejected = !!attempt.hardRejectReason;
          const bestFallbackScore = scoreHardRejectedPanoFallback(bestAttempt.summary);
          const attemptFallbackScore = scoreHardRejectedPanoFallback(attempt.summary);
          if (
            (bestHardRejected && !attemptHardRejected) ||
            (bestHardRejected === attemptHardRejected &&
              (bestHardRejected
                ? (
                  attemptFallbackScore > bestFallbackScore ||
                  (Math.abs(attemptFallbackScore - bestFallbackScore) < 1e-6 &&
                    (attempt.qualityScore > bestAttempt.qualityScore ||
                      (Math.abs(attempt.qualityScore - bestAttempt.qualityScore) < 1e-6 &&
                        attempt.qualityBase > bestAttempt.qualityBase)))
                )
                : (
                  attempt.qualityScore > bestAttempt.qualityScore ||
                  (Math.abs(attempt.qualityScore - bestAttempt.qualityScore) < 1e-6 &&
                    attempt.qualityBase > bestAttempt.qualityBase)
                )))
          ) {
            bestAttempt = attempt;
          }
          if (isGoodEnoughPanoAttempt(bestAttempt)) {
            earlyExitReason = `retry-good-enough:${bestAttempt.label}`;
            break;
          }
        }
      }

      const shouldRunMipFallbacks =
        !earlyExitReason &&
        (
          !!bestAttempt.hardRejectReason ||
          (bestAttempt.qualityScore < 0 && isLikelyPoorPanoQuality(bestAttempt.summary)) ||
          !!(
            bestAttempt.summary &&
            (
              bestAttempt.summary.lowerBandBrightFraction > 0.78 ||
              bestAttempt.summary.lowerBandP50 > 260
            )
          )
        );

      if (shouldRunMipFallbacks) {
        const mipFallbackConfigs: Array<{
          label: string;
          overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
        }> = [
            {
              label: 'retry-balanced-mip-narrow-fallback',
              overrides: {
                modalityLutOverride: true,
                verticalHalfMm: narrowVerticalHalfMm,
                verticalCenterOffsetMm: mandibularCenterOffsetMm,
                slabHalfThicknessMm: balancedSlabHalfThicknessMm,
                slabSamples: balancedSlabSamples,
                aggregation: 'MIP',
              },
            },
          ];

        for (const retryConfig of mipFallbackConfigs) {
          launchedMipFallbackCount++;
          const attempt = await runWorkerAttempt(retryConfig.label, retryConfig.overrides);
          recordAttempt(attempt);
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
          if (isGoodEnoughPanoAttempt(bestAttempt)) {
            earlyExitReason = `mip-good-enough:${bestAttempt.label}`;
            break;
          }
        }
      }

      const totalAttemptDurationMs = performance.now() - panoAttemptSequenceStartMs;
      const phase2BaseAttempt = attemptResults
        .filter(attempt => attempt.aggregation === 'MEAN')
        .sort((a, b) => {
          const aHardRejected = !!a.hardRejectReason;
          const bHardRejected = !!b.hardRejectReason;
          if (aHardRejected !== bHardRejected) {
            return aHardRejected ? 1 : -1;
          }
          return (b.qualityScore - a.qualityScore) || (b.qualityBase - a.qualityBase);
        })[0] || null;
      const phase1VirtualPano: {
        executed: boolean;
        skippedReason: string | null;
        error: string | null;
        timingMs: Record<string, unknown> | null;
        diagnostics: Record<string, unknown> | null;
      } = {
        executed: false,
        skippedReason: null,
        error: null,
        timingMs: null,
        diagnostics: null,
      };
      if (!phase2BaseAttempt) {
        phase1VirtualPano.skippedReason = 'NO_MEAN_ATTEMPT';
      } else {
        try {
          const phase1Result = await launchCPRWorker({
            ...workerInput,
            ...phase2BaseAttempt.requestOverrides,
            panoWidth: phase2BaseAttempt.panoWidth,
            panoHeight: phase2BaseAttempt.panoHeight,
            verticalHalfMm: phase2BaseAttempt.actualVertHalfMm,
            verticalCenterOffsetMm: phase2BaseAttempt.verticalCenterOffsetMm,
            slabHalfThicknessMm: phase2BaseAttempt.slabHalfThicknessMm,
            slabSamples: phase2BaseAttempt.slabSamples,
            aggregation: phase2BaseAttempt.aggregation,
            debugRunId: `${debugRunId}-phase1`,
            reconstructionMode: 'virtualPanoPhase1',
          });
          const phase1DiagnosticPayload =
            phase1Result.workerDebugPayload &&
              typeof phase1Result.workerDebugPayload === 'object' &&
              phase1Result.workerDebugPayload.diagnostic &&
              typeof phase1Result.workerDebugPayload.diagnostic === 'object'
              ? (phase1Result.workerDebugPayload.diagnostic as Record<string, unknown>)
              : null;
          phase1VirtualPano.executed = true;
          phase1VirtualPano.timingMs =
            phase1DiagnosticPayload &&
              phase1DiagnosticPayload.timingMs &&
              typeof phase1DiagnosticPayload.timingMs === 'object'
              ? (phase1DiagnosticPayload.timingMs as Record<string, unknown>)
              : null;
          phase1VirtualPano.diagnostics =
            phase1DiagnosticPayload &&
              phase1DiagnosticPayload.virtualPanoPhase12 &&
              typeof phase1DiagnosticPayload.virtualPanoPhase12 === 'object'
              ? (phase1DiagnosticPayload.virtualPanoPhase12 as Record<string, unknown>)
              : null;
        } catch (phase1Error) {
          phase1VirtualPano.error =
            phase1Error instanceof Error ? phase1Error.message : String(phase1Error);
        }
      }
      console.log(
        '[CPR-PHASE1-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: phase2BaseAttempt?.label ?? null,
          selectedAggregation: phase2BaseAttempt?.aggregation ?? null,
          phase1VirtualPano,
        })
      );
      const phase2VirtualPano: {
        executed: boolean;
        skippedReason: string | null;
        error: string | null;
        timingMs: Record<string, unknown> | null;
        diagnostics: Record<string, unknown> | null;
        summary: FloatBufferDebugSummary | null;
        voi: PanoVoiSettings | null;
        usedAsDisplayedOutput: boolean;
        borderlineAcceptedReason: string | null;
      } = {
        executed: false,
        skippedReason: null,
        error: null,
        timingMs: null,
        diagnostics: null,
        summary: null,
        voi: null,
        usedAsDisplayedOutput: false,
        borderlineAcceptedReason: null,
      };
      let phase2WorkerResult: CPRWorkerLaunchResult | null = null;
      if (!phase2BaseAttempt) {
        phase2VirtualPano.skippedReason = 'NO_MEAN_ATTEMPT';
      } else {
        try {
          const phase2Result = await launchCPRWorker({
            ...workerInput,
            ...phase2BaseAttempt.requestOverrides,
            panoWidth: phase2BaseAttempt.panoWidth,
            panoHeight: phase2BaseAttempt.panoHeight,
            verticalHalfMm: phase2BaseAttempt.actualVertHalfMm,
            verticalCenterOffsetMm: phase2BaseAttempt.verticalCenterOffsetMm,
            slabHalfThicknessMm: phase2BaseAttempt.slabHalfThicknessMm,
            slabSamples: phase2BaseAttempt.slabSamples,
            aggregation: phase2BaseAttempt.aggregation,
            debugRunId: `${debugRunId}-phase2`,
            reconstructionMode: 'virtualPano',
          });
          const phase2DiagnosticPayload =
            phase2Result.workerDebugPayload &&
              typeof phase2Result.workerDebugPayload === 'object' &&
              phase2Result.workerDebugPayload.diagnostic &&
              typeof phase2Result.workerDebugPayload.diagnostic === 'object'
              ? (phase2Result.workerDebugPayload.diagnostic as Record<string, unknown>)
              : null;
          const phase2Summary = summarizeFloatBufferForDebug(
            phase2Result.pixelData,
            phase2BaseAttempt.panoWidth,
            phase2BaseAttempt.panoHeight
          );
          const phase2Voi = computeAdaptivePanoVoi(
            phase2Summary,
            phase2Result.minValue,
            phase2Result.maxValue
          );
          const phase2RenderDiagnostics =
            phase2DiagnosticPayload &&
              phase2DiagnosticPayload.virtualPanoRender &&
              typeof phase2DiagnosticPayload.virtualPanoRender === 'object'
              ? (phase2DiagnosticPayload.virtualPanoRender as Record<string, unknown>)
              : null;

          phase2VirtualPano.executed = true;
          phase2VirtualPano.timingMs =
            phase2DiagnosticPayload &&
              phase2DiagnosticPayload.timingMs &&
              typeof phase2DiagnosticPayload.timingMs === 'object'
              ? (phase2DiagnosticPayload.timingMs as Record<string, unknown>)
              : null;
          phase2VirtualPano.diagnostics = phase2RenderDiagnostics;
          phase2VirtualPano.summary = phase2Summary;
          phase2VirtualPano.voi = phase2Voi;
          const phase2RejectReasons =
            phase2RenderDiagnostics && Array.isArray(phase2RenderDiagnostics.rejectReasons)
              ? phase2RenderDiagnostics.rejectReasons.filter(
                reason => typeof reason === 'string'
              )
              : [];
          phase2VirtualPano.usedAsDisplayedOutput =
            !!phase2RenderDiagnostics &&
            phase2RenderDiagnostics.usedAsOutput === true;
          const acceptedByLowerBandTolerance =
            !!phase2RenderDiagnostics &&
            phase2RenderDiagnostics.acceptedByLowerBandTolerance === true;
          if (phase2VirtualPano.usedAsDisplayedOutput && acceptedByLowerBandTolerance) {
            phase2VirtualPano.borderlineAcceptedReason = 'lower-band-tolerated-in-worker';
          } else if (
            phase2VirtualPano.usedAsDisplayedOutput &&
            phase2RejectReasons.length === 1 &&
            phase2RejectReasons[0] === 'lower-band-bright-fraction-too-high'
          ) {
            phase2VirtualPano.borderlineAcceptedReason = 'lower-band-bright-fraction-only-rejected';
          } else if (
            phase2RenderDiagnostics &&
            phase2RenderDiagnostics.usedAsOutput !== true
          ) {
            phase2VirtualPano.borderlineAcceptedReason = 'phase2-rejected-by-worker';
          }
          phase2WorkerResult = phase2Result;
        } catch (phase2Error) {
          phase2VirtualPano.error =
            phase2Error instanceof Error ? phase2Error.message : String(phase2Error);
        }
      }
      console.log(
        '[CPR-PHASE2-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: phase2BaseAttempt?.label ?? null,
          selectedAggregation: phase2BaseAttempt?.aggregation ?? null,
          phase2VirtualPano,
        })
      );
      const rankedAttempts = attemptAudit
        .slice()
        .sort((a, b) => {
          const aHardRejected = !!a.hardRejectReason;
          const bHardRejected = !!b.hardRejectReason;
          if (aHardRejected !== bHardRejected) {
            return aHardRejected ? 1 : -1;
          }
          if (aHardRejected && bHardRejected) {
            const fallbackDelta =
              scoreHardRejectedPanoFallback({
                sampledCount: a.sampledCount ?? 0,
                min: a.min ?? 0,
                max: a.max ?? 0,
                p01: a.p01 ?? 0,
                p50: a.p50 ?? 0,
                p99: a.p99 ?? 0,
                fractionBelowMinus950: a.fractionBelowMinus950 ?? 0,
                fractionAbove3000: a.fractionAbove3000 ?? 0,
                meanAbsDelta: a.meanAbsDelta ?? 0,
                toothBandMean: a.toothBandMean ?? 0,
                toothBandP10: a.toothBandP10 ?? 0,
                toothBandP90: a.toothBandP90 ?? 0,
                toothBandBrightFraction: a.toothBandBrightFraction ?? 0,
                lowerBandP50: a.lowerBandP50 ?? 0,
                lowerBandBrightFraction: a.lowerBandBrightFraction ?? 0,
                detailBandHorizontalEdgeMean: a.detailBandHorizontalEdgeMean ?? 0,
                detailBandVerticalEdgeMean: a.detailBandVerticalEdgeMean ?? 0,
              }) -
              scoreHardRejectedPanoFallback({
                sampledCount: b.sampledCount ?? 0,
                min: b.min ?? 0,
                max: b.max ?? 0,
                p01: b.p01 ?? 0,
                p50: b.p50 ?? 0,
                p99: b.p99 ?? 0,
                fractionBelowMinus950: b.fractionBelowMinus950 ?? 0,
                fractionAbove3000: b.fractionAbove3000 ?? 0,
                meanAbsDelta: b.meanAbsDelta ?? 0,
                toothBandMean: b.toothBandMean ?? 0,
                toothBandP10: b.toothBandP10 ?? 0,
                toothBandP90: b.toothBandP90 ?? 0,
                toothBandBrightFraction: b.toothBandBrightFraction ?? 0,
                lowerBandP50: b.lowerBandP50 ?? 0,
                lowerBandBrightFraction: b.lowerBandBrightFraction ?? 0,
                detailBandHorizontalEdgeMean: b.detailBandHorizontalEdgeMean ?? 0,
                detailBandVerticalEdgeMean: b.detailBandVerticalEdgeMean ?? 0,
              });
            if (Math.abs(fallbackDelta) > 1e-6) {
              return fallbackDelta > 0 ? -1 : 1;
            }
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
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
        })
      );
      console.log(
        '[CPR-SELECTED-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: bestAttempt.label,
          displayedOutputMode: phase2VirtualPano.usedAsDisplayedOutput ? 'virtualPanoPhase2' : 'legacy',
          displayedSourceLabel: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2BaseAttempt?.label ?? bestAttempt.label
            : bestAttempt.label,
          displayedSourceAggregation: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2BaseAttempt?.aggregation ?? bestAttempt.aggregation
            : bestAttempt.aggregation,
          selectedAggregation: bestAttempt.aggregation,
          selectedActualVertHalfMm: bestAttempt.actualVertHalfMm,
          selectedSlabHalfThicknessMm: bestAttempt.slabHalfThicknessMm,
          selectedSlabSamples: bestAttempt.slabSamples,
          selectedVerticalCenterOffsetMm: bestAttempt.verticalCenterOffsetMm,
          selectedColumnPixelSpacing: bestAttempt.columnPixelSpacing,
          selectedRowPixelSpacing: bestAttempt.rowPixelSpacing,
          selectedHardRejectReason: bestAttempt.hardRejectReason,
          selectedQualityScore: bestAttempt.qualityScore,
          selectedQualityBase: bestAttempt.qualityBase,
          selectedSummary: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2VirtualPano.summary
            : bestAttempt.summary,
          selectedVoi: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2VirtualPano.voi
            : bestAttempt.voi,
          selectedWorkerDebugPayload: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2WorkerResult?.workerDebugPayload ?? null
            : bestAttempt.workerDebugPayload ?? null,
          phase1VirtualPano,
          phase2VirtualPano,
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
          runnerUp: runnerUpAttempt,
          scoreDeltaToRunnerUp:
            runnerUpAttempt ? bestAttempt.qualityScore - runnerUpAttempt.qualityScore : null,
          baseDeltaToRunnerUp:
            runnerUpAttempt ? bestAttempt.qualityBase - runnerUpAttempt.qualityBase : null,
          selectedMatchesRankedTop: selectedAttempt?.label === bestAttempt.label,
        })
      );

      if (bestAttempt.intensityDomain === 'unknown') {
        console.warn(
          `[CPR][${debugRunId}] Selected panoramic attempt remains in an unknown intensity domain.`,
          {
            label: bestAttempt.label,
            qualityBase: bestAttempt.qualityBase,
            qualityScore: bestAttempt.qualityScore,
            intensityDomain: bestAttempt.intensityDomain,
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
      let displayedOutputMode: 'legacy' | 'virtualPanoPhase2' = 'legacy';
      let displayedSourceLabel = bestAttempt.label;
      let displayedSourceAggregation = bestAttempt.aggregation;
      let selectedPanoWidth = bestAttempt.panoWidth;
      let selectedPanoHeight = bestAttempt.panoHeight;
      let selectedActualVertHalfMm = bestAttempt.actualVertHalfMm;
      let selectedColumnPixelSpacing = bestAttempt.columnPixelSpacing;
      let selectedRowPixelSpacing = bestAttempt.rowPixelSpacing;
      if (
        phase2VirtualPano.usedAsDisplayedOutput &&
        phase2WorkerResult &&
        phase2VirtualPano.summary &&
        phase2VirtualPano.voi &&
        phase2BaseAttempt
      ) {
        panoWorkerResult = phase2WorkerResult;
        panoDebugSummary = phase2VirtualPano.summary;
        adaptiveVoi = phase2VirtualPano.voi;
        displayedOutputMode = 'virtualPanoPhase2';
        displayedSourceLabel = phase2BaseAttempt.label;
        displayedSourceAggregation = phase2BaseAttempt.aggregation;
        selectedPanoWidth = phase2BaseAttempt.panoWidth;
        selectedPanoHeight = phase2BaseAttempt.panoHeight;
        selectedActualVertHalfMm = phase2BaseAttempt.actualVertHalfMm;
        selectedColumnPixelSpacing = phase2BaseAttempt.columnPixelSpacing;
        selectedRowPixelSpacing = phase2BaseAttempt.rowPixelSpacing;
      }
      adaptiveVoi = {
        lower: CPR_PANO_DEFAULT_WINDOW_CENTER - CPR_PANO_DEFAULT_WINDOW_WIDTH / 2,
        upper: CPR_PANO_DEFAULT_WINDOW_CENTER + CPR_PANO_DEFAULT_WINDOW_WIDTH / 2,
        windowWidth: CPR_PANO_DEFAULT_WINDOW_WIDTH,
        windowCenter: CPR_PANO_DEFAULT_WINDOW_CENTER,
      };
      const displayedWorkerDebugPayload =
        phase2VirtualPano.usedAsDisplayedOutput
          ? phase2WorkerResult?.workerDebugPayload ?? null
          : bestAttempt.workerDebugPayload ?? null;
      const displayedWorkerDiagnostic =
        displayedWorkerDebugPayload &&
          typeof displayedWorkerDebugPayload === 'object' &&
          displayedWorkerDebugPayload.diagnostic &&
          typeof displayedWorkerDebugPayload.diagnostic === 'object'
          ? (displayedWorkerDebugPayload.diagnostic as Record<string, unknown>)
          : null;
      const selectedLocalCenterOffsetsMm = Array.isArray(displayedWorkerDiagnostic?.localCenterOffsetsMm)
        ? displayedWorkerDiagnostic.localCenterOffsetsMm
          .map(value => Number(value))
          .filter(value => Number.isFinite(value))
        : [];
      const crossSectionVerticalCenterOffsetMm =
        (phase2VirtualPano.usedAsDisplayedOutput
          ? phase2BaseAttempt?.verticalCenterOffsetMm
          : bestAttempt.verticalCenterOffsetMm) ?? 0;
      crossSectionVerticalCenterOffsetMmRef.current = crossSectionVerticalCenterOffsetMm;
      console.log('[CPR-CROSSSECTION-GEOMETRY-JSON]', JSON.stringify({
        runId: debugRunId,
        displayedOutputMode,
        correctedFrameCount: frames.length,
        sourceFrameCount: frames.length,
        usingWorkerLocalCenterOffsets: selectedLocalCenterOffsetsMm.length === frames.length,
        crossSectionVerticalCenterOffsetMm,
        localCenterOffsetsFirst8: selectedLocalCenterOffsetsMm.slice(0, 8),
        geometryPolicy: 'worker-offsets-render-only',
      }));
      console.log(`[CPR][${debugRunId}] selected pano attempt`, {
        label: bestAttempt.label,
        displayedOutputMode,
        displayedSourceLabel,
        displayedSourceAggregation,
        qualityBase: bestAttempt.qualityBase,
        qualityScore: bestAttempt.qualityScore,
        hardRejectReason: bestAttempt.hardRejectReason,
        aggregation: bestAttempt.aggregation,
        intensityDomain: bestAttempt.intensityDomain,
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
          toothBandMean: panoDebugSummary.toothBandMean,
          toothBandP10: panoDebugSummary.toothBandP10,
          toothBandP90: panoDebugSummary.toothBandP90,
          toothBandBrightFraction: panoDebugSummary.toothBandBrightFraction,
          lowerBandP50: panoDebugSummary.lowerBandP50,
          lowerBandBrightFraction: panoDebugSummary.lowerBandBrightFraction,
          detailBandHorizontalEdgeMean: panoDebugSummary.detailBandHorizontalEdgeMean,
          detailBandVerticalEdgeMean: panoDebugSummary.detailBandVerticalEdgeMean,
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
      cprStateService.setArchData(controlPoints, frames, sourceVolumeId, latestAnnotationUID);

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
                const { parallelScale: _ignoredParallelScale, ...cameraWithoutParallelScale } =
                  preservedAxialCamera;
                axialViewportAfterSwitch.setCamera(
                  cameraWithoutParallelScale as Partial<ReturnType<typeof axialViewportAfterSwitch.getCamera>>
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

          if (panoWorkerResult.pixelData.length !== panoWorkerResult.width * panoWorkerResult.height) {
            throw new Error('Dimension mismatch');
          }

          clearPanoImageCache();
          setPanoImagePayload(panoImageId, {
            pixelData: panoWorkerResult.pixelData,
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            huDomain: bestAttempt.huDomain,
            intensityDomain: bestAttempt.intensityDomain,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            slope: 1,
            intercept: 0,
            columnPixelSpacing: selectedColumnPixelSpacing,
            rowPixelSpacing: selectedRowPixelSpacing,
          });

          console.log(`[CPR][${debugRunId}] Pano payload metadata pushed to loader`, {
            displayedOutputMode,
            panoImageId,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            intensityDomain: bestAttempt.intensityDomain,
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
            displayedOutputMode,
            panoImageId,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            intensityDomain: bestAttempt.intensityDomain,
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

          cornerstoneTools.utilities.triggerAnnotationRender(panoViewport.element);

          await new Promise<void>((resolve, reject) => {
            window.setTimeout(() => {
              if (!isMountedRef.current) {
                resolve();
                return;
              }

              void initializeCrossSection(
                frames,
                sourceVolumeId,
                selectedActualVertHalfMm,
                crossSectionVerticalCenterOffsetMm,
                selectedLocalCenterOffsetsMm.length === frames.length,
                servicesManager
              )
                .then(resolve)
                .catch(reject);
            }, 50);
          });
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

  const flushSliderFrameChange = useCallback(() => {
    sliderAnimationFrameRef.current = null;

    if (!cprStateService.hasData()) {
      pendingSliderFrameIndexRef.current = null;
      return;
    }

    const frames = cprStateService.getFrames();
    if (!frames.length) {
      pendingSliderFrameIndexRef.current = null;
      return;
    }

    const requestedIndex = pendingSliderFrameIndexRef.current;
    if (requestedIndex == null) {
      return;
    }
    pendingSliderFrameIndexRef.current = null;

    const clampedIndex = Math.max(0, Math.min(requestedIndex, frames.length - 1));
    cprStateService.setCurrentFrameIndex(clampedIndex);
    const frame = frames[clampedIndex];

    const centroid = frames.reduce<[number, number, number]>(
      (acc, currentFrame) => [
        acc[0] + currentFrame.position[0],
        acc[1] + currentFrame.position[1],
        acc[2] + currentFrame.position[2],
      ],
      [0, 0, 0]
    );
    centroid[0] /= frames.length;
    centroid[1] /= frames.length;
    centroid[2] /= frames.length;

    const radial = normalize3([
      frame.position[0] - centroid[0],
      frame.position[1] - centroid[1],
      frame.position[2] - centroid[2],
    ]);
    const tangent = normalize3([frame.T[0], frame.T[1], frame.T[2]]);
    const nCamera = normalize3([frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]]);
    const nSlab = normalize3([frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]]);
    const s = normalize3([frame.S[0], frame.S[1], frame.S[2]]);
    console.log('[CPR-CROSSSECTION-FRAME]', {
      frameIndex: clampedIndex,
      position: frame.position,
      tangent,
      nCamera,
      nSlab,
      s,
      radial,
      dotCameraRadial:
        nCamera[0] * radial[0] + nCamera[1] * radial[1] + nCamera[2] * radial[2],
      dotSlabRadial: nSlab[0] * radial[0] + nSlab[1] * radial[1] + nSlab[2] * radial[2],
      dotTCamera:
        tangent[0] * nCamera[0] + tangent[1] * nCamera[1] + tangent[2] * nCamera[2],
      dotTSlab: tangent[0] * nSlab[0] + tangent[1] * nSlab[1] + tangent[2] * nSlab[2],
      dotTS: tangent[0] * s[0] + tangent[1] * s[1] + tangent[2] * s[2],
      dotCameraS: nCamera[0] * s[0] + nCamera[1] * s[1] + nCamera[2] * s[2],
      dotSlabS: nSlab[0] * s[0] + nSlab[1] * s[1] + nSlab[2] * s[2],
    });

    setCrossSectionForFrame(
      frame,
      servicesManager,
      crossSectionVerticalCenterOffsetMmRef.current
    );
    const panoViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
    if (panoViewport?.element) {
      cornerstoneTools.utilities.triggerAnnotationRender(panoViewport.element);
    }
  }, [servicesManager]);

  const onSliderChange = useCallback(
    (frameIndex: number) => {
      if (!cprStateService.hasData()) {
        return;
      }

      const frames = cprStateService.getFrames();
      if (!frames.length) {
        return;
      }

      pendingSliderFrameIndexRef.current = Math.max(0, Math.min(frameIndex, frames.length - 1));
      if (sliderAnimationFrameRef.current != null) {
        return;
      }

      sliderAnimationFrameRef.current = window.requestAnimationFrame(() => {
        flushSliderFrameChange();
      });
    },
    [flushSliderFrameChange]
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
