import type { PanoV2NumericSummary, PanoV2TrackLabel } from './panoV2Geometry';
import type {
  PanoV2BandLayerRenderResult,
  PanoV2FullPlaneLayerImages,
  PanoV2LayerRenderResult,
  PanoV2RenderedLayer,
} from './panoV2LayerRenderer';

const COLUMN_SCORE_ROW_RADIUS = 18;
const COLUMN_SWITCH_PENALTY = 92;
const COLUMN_RESELECT_MARGIN = 52;
const OUTER_EDGE_FEATHER_ROWS = 10;
const AMBIGUITY_MARGIN_RATIO = 0.055;
const AMBIGUITY_MARGIN_ABS = 12;
const SCORE_SMOOTH_WEIGHTS = [0.12, 0.2, 0.36, 0.2, 0.12] as const;
const REGION_PREFERENCE_BONUS = 22;
const REGION_PREFERENCE_MAX_DEFICIT = 34;
const SHARPNESS_FALLBACK_BONUS = 10;
const SHARPNESS_FALLBACK_MAX_DEFICIT = 18;
const SHORT_RUN_MAX_LENGTH = 5;
const SHORT_RUN_RESELECT_MARGIN = 46;
const SOFT_SWITCH_BLEND_MARGIN = 30;
const SOFT_SWITCH_BLEND_MAX_WEIGHT = 0.34;
const SOFT_SWITCH_BLEND_SWITCH_TOLERANCE = 0.12;
const COVERAGE_WEIGHT_SOFT_START = 0.025;
const COVERAGE_WEIGHT_FULL = 0.2;
const ANATOMY_WEIGHT_LOW_HU = -520;
const ANATOMY_WEIGHT_HIGH_HU = 660;

interface PanoV2FusionLayerScore {
  layerId: string;
  score: number;
}

interface PanoV2FusionProbeColumn {
  col: number;
  dominantLayerId: string;
  dominantLayerIndex: number;
  ambiguous: boolean;
  layerScores: PanoV2FusionLayerScore[];
}

interface PanoV2FusionBandDiagnostics {
  label: PanoV2TrackLabel;
  switchCount: number;
  switchFraction: number;
  ambiguousColumnFraction: number;
  dominantLayerUsage: Array<{
    layerId: string;
    columnFraction: number;
  }>;
  sampledColumns: PanoV2FusionProbeColumn[];
}

export interface PanoV2FusionRowCoverage {
  row: number;
  coverageType: 'upper-fusion' | 'lower-fusion' | 'gap-fill' | 'empty' | 'direct-bypass';
  upperFusionCovered: boolean;
  lowerFusionCovered: boolean;
  gapFillCovered: boolean;
  leftEmpty: boolean;
  upperPixelFraction: number;
  lowerPixelFraction: number;
  overlapPixelFraction: number;
  coveredPixelFraction: number;
}

export interface PanoV2FusionDiagnostics {
  enabled: true;
  phase: 4;
  model: 'fusion-and-roi-composition';
  implementationVersion: string;
  renderBypass: boolean;
  outputCoverageFraction: number;
  overlapFraction: number;
  gapCenterRow: number;
  ghostingRisk: 'low' | 'moderate' | 'high';
  normalizationHuWindow: {
    lower: number;
    upper: number;
  };
  outputRange: PanoV2NumericSummary;
  rowCoverageByRow: PanoV2FusionRowCoverage[];
  upperArch: PanoV2FusionBandDiagnostics;
  lowerArch: PanoV2FusionBandDiagnostics;
}

export interface PanoV2FusionResult {
  pixelData: Float32Array;
  diagnostics: PanoV2FusionDiagnostics;
}

interface PanoV2BandFusionResult {
  diagnostic: PanoV2FusionBandDiagnostics;
  bandImageHu: Float32Array;
  selectedLayerByCol: Int16Array;
}

interface PanoV2BandBlendPlan {
  secondaryLayerByCol: Int16Array;
  secondaryWeightByCol: Float32Array;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function roundTo(value: number, decimals = 3): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const scale = 10 ** decimals;
  return Math.round(value * scale) / scale;
}

function smoothstepUnit(value: number): number {
  const clamped = clampNumber(value, 0, 1);
  return clamped * clamped * (3 - 2 * clamped);
}

function smoothstepRange(value: number, start: number, end: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return value >= end ? 1 : 0;
  }
  return smoothstepUnit((value - start) / (end - start));
}

function percentile(values: number[], ratio: number): number {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const clampedRatio = clampNumber(ratio, 0, 1);
  const position = clampedRatio * (sorted.length - 1);
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }
  const blend = position - lowerIndex;
  return sorted[lowerIndex] * (1 - blend) + sorted[upperIndex] * blend;
}

function summarize(values: number[]): PanoV2NumericSummary {
  if (!values.length) {
    return { min: 0, p50: 0, p90: 0, max: 0, mean: 0 };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (const value of values) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  }
  return {
    min: roundTo(min),
    p50: roundTo(percentile(values, 0.5)),
    p90: roundTo(percentile(values, 0.9)),
    max: roundTo(max),
    mean: roundTo(sum / values.length),
  };
}

function buildEvenlySpacedIndices(length: number, targetCount: number): number[] {
  if (length <= 0) {
    return [];
  }
  if (length <= targetCount) {
    return Array.from({ length }, (_, index) => index);
  }
  const indices = new Set<number>([0, length - 1]);
  const denominator = Math.max(1, targetCount - 1);
  for (let sampleIndex = 0; sampleIndex < targetCount; sampleIndex++) {
    indices.add(Math.round((sampleIndex / denominator) * (length - 1)));
  }
  return Array.from(indices).sort((left, right) => left - right);
}

function layerIdToIndex(
  band: PanoV2BandLayerRenderResult,
  layerId: string | null | undefined
): number {
  if (!layerId) {
    return -1;
  }
  return band.layers.findIndex(layer => layer.diagnostic.layerId === layerId);
}

function buildPreferredLayerIndexByCol(band: PanoV2BandLayerRenderResult): Int16Array {
  const width = band.layers[0]?.image.length
    ? Math.max(1, Math.round(band.layers[0].image.length / Math.max(1, band.rowCount)))
    : 0;
  const preferredLayerByCol = new Int16Array(Math.max(1, width));
  preferredLayerByCol.fill(-1);
  const leftMolarLayerIndex = layerIdToIndex(band, band.bestByRegion.leftMolar);
  const incisorLayerIndex = layerIdToIndex(band, band.bestByRegion.incisors);
  const rightMolarLayerIndex = layerIdToIndex(band, band.bestByRegion.rightMolar);
  const sharpnessLayerIndex = layerIdToIndex(band, band.bestByRegion.sharpnessPeak);

  for (let col = 0; col < width; col++) {
    const columnRatio = width > 1 ? col / (width - 1) : 0.5;
    let preferredLayerIndex = sharpnessLayerIndex;
    if (columnRatio < 0.22) {
      preferredLayerIndex = leftMolarLayerIndex >= 0 ? leftMolarLayerIndex : sharpnessLayerIndex;
    } else if (columnRatio >= 0.35 && columnRatio <= 0.65) {
      preferredLayerIndex = incisorLayerIndex >= 0 ? incisorLayerIndex : sharpnessLayerIndex;
    } else if (columnRatio > 0.78) {
      preferredLayerIndex = rightMolarLayerIndex >= 0 ? rightMolarLayerIndex : sharpnessLayerIndex;
    }
    preferredLayerByCol[col] = preferredLayerIndex >= 0 ? preferredLayerIndex : 0;
  }

  return preferredLayerByCol;
}

function smoothColumnScores(scores: Float32Array): Float32Array {
  const smoothedScores = new Float32Array(scores.length);
  for (let col = 0; col < scores.length; col++) {
    let weightedSum = 0;
    let weightTotal = 0;
    for (let offset = -2; offset <= 2; offset++) {
      const sampleCol = col + offset;
      if (sampleCol < 0 || sampleCol >= scores.length) {
        continue;
      }
      const sampleValue = Number(scores[sampleCol]);
      if (!Number.isFinite(sampleValue)) {
        continue;
      }
      const weight = SCORE_SMOOTH_WEIGHTS[offset + 2];
      weightedSum += sampleValue * weight;
      weightTotal += weight;
    }
    smoothedScores[col] =
      weightTotal > 1e-6 ? weightedSum / weightTotal : Number(scores[col] ?? Number.NEGATIVE_INFINITY);
  }
  return smoothedScores;
}

function applyLayerPreferences(
  band: PanoV2BandLayerRenderResult,
  scoresByLayer: Float32Array[]
): void {
  const preferredLayerByCol = buildPreferredLayerIndexByCol(band);
  const sharpnessLayerIndex = layerIdToIndex(band, band.bestByRegion.sharpnessPeak);

  for (let col = 0; col < preferredLayerByCol.length; col++) {
    let bestScore = Number.NEGATIVE_INFINITY;
    for (let layerIndex = 0; layerIndex < scoresByLayer.length; layerIndex++) {
      const score = Number(scoresByLayer[layerIndex][col]);
      if (Number.isFinite(score) && score > bestScore) {
        bestScore = score;
      }
    }
    if (!Number.isFinite(bestScore)) {
      continue;
    }
    const preferredLayerIndex = preferredLayerByCol[col];
    if (preferredLayerIndex >= 0 && preferredLayerIndex < scoresByLayer.length) {
      const preferredScore = Number(scoresByLayer[preferredLayerIndex][col]);
      if (
        Number.isFinite(preferredScore) &&
        bestScore - preferredScore <= REGION_PREFERENCE_MAX_DEFICIT
      ) {
        scoresByLayer[preferredLayerIndex][col] = preferredScore + REGION_PREFERENCE_BONUS;
        continue;
      }
    }
    if (sharpnessLayerIndex >= 0 && sharpnessLayerIndex < scoresByLayer.length) {
      const sharpnessScore = Number(scoresByLayer[sharpnessLayerIndex][col]);
      if (
        Number.isFinite(sharpnessScore) &&
        bestScore - sharpnessScore <= SHARPNESS_FALLBACK_MAX_DEFICIT
      ) {
        scoresByLayer[sharpnessLayerIndex][col] = sharpnessScore + SHARPNESS_FALLBACK_BONUS;
      }
    }
  }
}

function computeColumnLayerScore(
  layer: PanoV2RenderedLayer,
  band: PanoV2BandLayerRenderResult,
  col: number
): number {
  const width = Math.max(1, Math.round(layer.image.length / Math.max(1, band.rowCount)));
  const rowStart = Math.max(0, band.localAnchorRow - COLUMN_SCORE_ROW_RADIUS);
  const rowEnd = Math.min(band.rowCount - 1, band.localAnchorRow + COLUMN_SCORE_ROW_RADIUS);
  let detailSum = 0;
  let detailCount = 0;
  let anatomyCount = 0;
  let denseCount = 0;
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;

  for (let row = rowStart; row <= rowEnd; row++) {
    const index = row * width + col;
    const value = Number(layer.image[index]);
    if (!Number.isFinite(value)) {
      continue;
    }
    minValue = Math.min(minValue, value);
    maxValue = Math.max(maxValue, value);
    if (value > -650) {
      anatomyCount++;
    }
    if (value > 180) {
      denseCount++;
    }
    let detail = 0;
    if (row > rowStart) {
      const up = Number(layer.image[(row - 1) * width + col]);
      if (Number.isFinite(up)) {
        detail += Math.abs(value - up);
      }
    }
    if (col > 0) {
      const left = Number(layer.image[row * width + (col - 1)]);
      if (Number.isFinite(left)) {
        detail += Math.abs(value - left) * 0.55;
      }
    }
    detailSum += detail;
    detailCount++;
  }

  if (detailCount === 0 || !Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return Number.NEGATIVE_INFINITY;
  }

  const anatomyFraction = anatomyCount / detailCount;
  const denseFraction = denseCount / detailCount;
  const localContrast = clampNumber(maxValue - minValue, 0, 1600);
  const detailMean = detailSum / detailCount;
  return detailMean * 0.58 + localContrast * 0.14 + anatomyFraction * 150 + denseFraction * 75;
}

function solveDominantLayerPath(scoresByLayer: Float32Array[]): Int16Array {
  const layerCount = scoresByLayer.length;
  const width = scoresByLayer[0]?.length ?? 0;
  const previous = new Float32Array(layerCount);
  const current = new Float32Array(layerCount);
  const backPointers = new Int16Array(Math.max(1, width * layerCount));

  previous.fill(Number.NEGATIVE_INFINITY);
  current.fill(Number.NEGATIVE_INFINITY);

  for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
    previous[layerIndex] = Number(scoresByLayer[layerIndex][0] ?? Number.NEGATIVE_INFINITY);
  }

  for (let col = 1; col < width; col++) {
    for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
      const localScore = Number(scoresByLayer[layerIndex][col]);
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestPreviousLayer = layerIndex;
      for (let previousLayerIndex = 0; previousLayerIndex < layerCount; previousLayerIndex++) {
        const transitionPenalty =
          Math.abs(layerIndex - previousLayerIndex) * COLUMN_SWITCH_PENALTY;
        const candidateScore = previous[previousLayerIndex] - transitionPenalty + localScore;
        if (candidateScore > bestScore) {
          bestScore = candidateScore;
          bestPreviousLayer = previousLayerIndex;
        }
      }
      current[layerIndex] = bestScore;
      backPointers[col * layerCount + layerIndex] = bestPreviousLayer;
    }
    previous.set(current);
  }

  const path = new Int16Array(Math.max(1, width));
  let bestFinalLayer = 0;
  let bestFinalScore = Number.NEGATIVE_INFINITY;
  for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
    if (previous[layerIndex] > bestFinalScore) {
      bestFinalScore = previous[layerIndex];
      bestFinalLayer = layerIndex;
    }
  }
  path[width - 1] = bestFinalLayer;
  for (let col = width - 1; col > 0; col--) {
    path[col - 1] = backPointers[col * layerCount + path[col]];
  }

  // Collapse isolated one-column switches unless the local score gain is meaningful.
  for (let pass = 0; pass < 2; pass++) {
    for (let col = 1; col + 1 < width; col++) {
      const previousLayer = path[col - 1];
      const nextLayer = path[col + 1];
      const currentLayer = path[col];
      if (previousLayer !== nextLayer || currentLayer === previousLayer) {
        continue;
      }
      const currentScore = Number(scoresByLayer[currentLayer][col]);
      const neighborScore = Number(scoresByLayer[previousLayer][col]);
      if (currentScore - neighborScore < COLUMN_RESELECT_MARGIN) {
        path[col] = previousLayer;
      }
    }
  }

  return path;
}

function collapseShortRuns(path: Int16Array, scoresByLayer: Float32Array[]): void {
  let col = 0;
  while (col < path.length) {
    const runStart = col;
    const runLayer = path[col];
    while (col + 1 < path.length && path[col + 1] === runLayer) {
      col++;
    }
    const runEnd = col;
    const runLength = runEnd - runStart + 1;
    const leftLayer = runStart > 0 ? path[runStart - 1] : -1;
    const rightLayer = runEnd + 1 < path.length ? path[runEnd + 1] : -1;
    if (
      runLength <= SHORT_RUN_MAX_LENGTH &&
      leftLayer >= 0 &&
      leftLayer === rightLayer &&
      leftLayer !== runLayer
    ) {
      let runLayerScore = 0;
      let neighborLayerScore = 0;
      let comparedCount = 0;
      for (let runCol = runStart; runCol <= runEnd; runCol++) {
        const currentScore = Number(scoresByLayer[runLayer][runCol]);
        const neighborScore = Number(scoresByLayer[leftLayer][runCol]);
        if (!Number.isFinite(currentScore) || !Number.isFinite(neighborScore)) {
          continue;
        }
        runLayerScore += currentScore;
        neighborLayerScore += neighborScore;
        comparedCount++;
      }
      if (
        comparedCount > 0 &&
        runLayerScore / comparedCount - neighborLayerScore / comparedCount <
          SHORT_RUN_RESELECT_MARGIN
      ) {
        for (let runCol = runStart; runCol <= runEnd; runCol++) {
          path[runCol] = leftLayer;
        }
      }
    }
    col++;
  }
}

function buildLayerBlendPlan(
  scoresByLayer: Float32Array[],
  selectedLayerByCol: Int16Array
): PanoV2BandBlendPlan {
  const width = selectedLayerByCol.length;
  const secondaryLayerByCol = new Int16Array(Math.max(1, width));
  const secondaryWeightByCol = new Float32Array(Math.max(1, width));
  secondaryLayerByCol.fill(-1);
  secondaryWeightByCol.fill(0);

  for (let col = 0; col < width; col++) {
    const dominantLayerIndex = selectedLayerByCol[col];
    const dominantScore = Number(scoresByLayer[dominantLayerIndex]?.[col] ?? Number.NEGATIVE_INFINITY);
    if (!Number.isFinite(dominantScore)) {
      continue;
    }
    let secondaryLayerIndex = -1;
    let secondaryScore = Number.NEGATIVE_INFINITY;
    for (let layerIndex = 0; layerIndex < scoresByLayer.length; layerIndex++) {
      if (layerIndex === dominantLayerIndex) {
        continue;
      }
      const score = Number(scoresByLayer[layerIndex]?.[col] ?? Number.NEGATIVE_INFINITY);
      if (Number.isFinite(score) && score > secondaryScore) {
        secondaryScore = score;
        secondaryLayerIndex = layerIndex;
      }
    }
    if (secondaryLayerIndex < 0 || !Number.isFinite(secondaryScore)) {
      continue;
    }
    const deficit = dominantScore - secondaryScore;
    const blendWeight =
      deficit < SOFT_SWITCH_BLEND_MARGIN
        ? smoothstepUnit(1 - deficit / SOFT_SWITCH_BLEND_MARGIN) * SOFT_SWITCH_BLEND_MAX_WEIGHT
        : 0;
    if (blendWeight <= 0.02) {
      continue;
    }
    secondaryLayerByCol[col] = secondaryLayerIndex;
    secondaryWeightByCol[col] = blendWeight;
  }

  return {
    secondaryLayerByCol,
    secondaryWeightByCol,
  };
}

function isSoftSwitchBoundary(
  col: number,
  selectedLayerByCol: Int16Array,
  blendPlan: PanoV2BandBlendPlan
): boolean {
  if (col <= 0 || col >= selectedLayerByCol.length) {
    return false;
  }
  const previousLayerIndex = selectedLayerByCol[col - 1];
  const currentLayerIndex = selectedLayerByCol[col];
  if (previousLayerIndex === currentLayerIndex) {
    return false;
  }
  const currentCanBlendPrevious =
    blendPlan.secondaryLayerByCol[col] === previousLayerIndex &&
    blendPlan.secondaryWeightByCol[col] >= SOFT_SWITCH_BLEND_SWITCH_TOLERANCE;
  const previousCanBlendCurrent =
    blendPlan.secondaryLayerByCol[col - 1] === currentLayerIndex &&
    blendPlan.secondaryWeightByCol[col - 1] >= SOFT_SWITCH_BLEND_SWITCH_TOLERANCE;
  return currentCanBlendPrevious || previousCanBlendCurrent;
}

function isContinuityStableColumn(
  col: number,
  selectedLayerByCol: Int16Array
): boolean {
  const currentLayerIndex = selectedLayerByCol[col];
  return (
    (col > 0 && selectedLayerByCol[col - 1] === currentLayerIndex) ||
    (col + 1 < selectedLayerByCol.length && selectedLayerByCol[col + 1] === currentLayerIndex)
  );
}

function computeTaperWeight(distance: number, plateauRows: number, reachRows: number): number {
  if (distance <= plateauRows) {
    return 1;
  }
  if (reachRows <= plateauRows) {
    return 0;
  }
  return smoothstepUnit(1 - (distance - plateauRows) / Math.max(1, reachRows - plateauRows));
}

function computeBandEdgeWeight(
  band: PanoV2BandLayerRenderResult,
  localRow: number
): number {
  const outerEdgeDistance =
    band.label === 'upperArch' ? localRow + 1 : band.rowCount - localRow;
  return smoothstepUnit(outerEdgeDistance / OUTER_EDGE_FEATHER_ROWS);
}

function computeBandFocusWeight(
  band: PanoV2BandLayerRenderResult,
  localRow: number
): number {
  const distanceAbove = Math.max(0, band.localAnchorRow - localRow);
  const distanceBelow = Math.max(0, localRow - band.localAnchorRow);
  const plateauRows = Math.max(8, Math.round(band.rowCount * 0.08));
  if (band.label === 'upperArch') {
    const topReach = Math.max(18, Math.round(band.rowCount * 0.34));
    const gapReach = Math.max(42, Math.round(band.rowCount * 0.82));
    const topWeight = computeTaperWeight(distanceAbove, plateauRows, topReach);
    const gapWeight = computeTaperWeight(distanceBelow, plateauRows, gapReach);
    const gapLift =
      localRow >= band.localAnchorRow
        ? computeTaperWeight(distanceBelow, 0, gapReach) * 0.22
        : 0;
    return clampNumber(Math.max(gapLift, Math.min(topWeight, gapWeight)), 0, 1);
  }

  const gapReach = Math.max(42, Math.round(band.rowCount * 0.82));
  const lowerReach = Math.max(18, Math.round(band.rowCount * 0.3));
  const gapWeight = computeTaperWeight(distanceAbove, plateauRows, gapReach);
  const lowerWeight = computeTaperWeight(distanceBelow, plateauRows, lowerReach);
  const gapLift =
    localRow <= band.localAnchorRow
      ? computeTaperWeight(distanceAbove, 0, gapReach) * 0.22
      : 0;
  return clampNumber(Math.max(gapLift, Math.min(gapWeight, lowerWeight)), 0, 1);
}

function suppressOutsideArchRoiHu(params: {
  row: number;
  fusedHu: number;
  upperArch: PanoV2BandLayerRenderResult;
  lowerArch: PanoV2BandLayerRenderResult;
}): number {
  const { row, fusedHu, upperArch, lowerArch } = params;
  let suppressedHu = fusedHu;

  const upperSoftTopRow =
    upperArch.anchorRow - Math.max(10, Math.round(upperArch.rowCount * 0.18));
  if (row < upperSoftTopRow) {
    const topFalloffRows = Math.max(18, Math.round(upperArch.rowCount * 0.26));
    const topT = smoothstepUnit((upperSoftTopRow - row) / Math.max(1, topFalloffRows));
    const topSuppression = topT * 0.94;
    suppressedHu = suppressedHu * (1 - topSuppression) + -940 * topSuppression;
  }

  const lowerSoftBottomRow =
    lowerArch.anchorRow + Math.max(12, Math.round(lowerArch.rowCount * 0.12));
  if (row > lowerSoftBottomRow) {
    const bottomFalloffRows = Math.max(20, Math.round(lowerArch.rowCount * 0.22));
    const bottomT = smoothstepUnit((row - lowerSoftBottomRow) / Math.max(1, bottomFalloffRows));
    const bottomSuppression = bottomT * 0.985;
    suppressedHu = suppressedHu * (1 - bottomSuppression) + -965 * bottomSuppression;
  }

  return suppressedHu;
}

function computeRowAirBaselineHu(params: {
  row: number;
  upperArch: PanoV2BandLayerRenderResult;
  lowerArch: PanoV2BandLayerRenderResult;
}): number {
  const { row, upperArch, lowerArch } = params;
  if (row <= upperArch.anchorRow) {
    const topSpan = Math.max(1, upperArch.anchorRow - upperArch.rowStart);
    return -780 - smoothstepRange(upperArch.anchorRow - row, 0, topSpan) * 160;
  }
  if (row >= lowerArch.anchorRow) {
    const bottomSpan = Math.max(1, lowerArch.rowEnd - lowerArch.anchorRow);
    return -760 - smoothstepRange(row - lowerArch.anchorRow, 0, bottomSpan) * 200;
  }
  const gapHalfSpan = Math.max(1, (lowerArch.anchorRow - upperArch.anchorRow) * 0.5);
  const gapCenterRow = (upperArch.anchorRow + lowerArch.anchorRow) * 0.5;
  const normalizedDistance = Math.abs(row - gapCenterRow) / gapHalfSpan;
  return -720 + smoothstepUnit(1 - clampNumber(normalizedDistance, 0, 1)) * 80;
}

function buildBandFusionResult(band: PanoV2BandLayerRenderResult): PanoV2BandFusionResult {
  const width = band.layers[0]?.image.length
    ? Math.max(1, Math.round(band.layers[0].image.length / Math.max(1, band.rowCount)))
    : 0;
  const rawScoresByLayer = band.layers.map(layer => {
    const scores = new Float32Array(Math.max(1, width));
    for (let col = 0; col < width; col++) {
      scores[col] = computeColumnLayerScore(layer, band, col);
    }
    return scores;
  });
  const scoresByLayer = rawScoresByLayer.map(scores => smoothColumnScores(scores));
  applyLayerPreferences(band, scoresByLayer);
  const selectedLayerByCol = solveDominantLayerPath(scoresByLayer);
  collapseShortRuns(selectedLayerByCol, scoresByLayer);
  const blendPlan = buildLayerBlendPlan(scoresByLayer, selectedLayerByCol);
  const bandImageHu = new Float32Array(Math.max(1, width * band.rowCount));
  bandImageHu.fill(Number.NaN);
  const sampledColumns = buildEvenlySpacedIndices(width, 5);
  let switchCount = 0;
  let ambiguousCount = 0;
  const usageCounts = new Int16Array(band.layers.length);

  for (let col = 0; col < width; col++) {
    const selectedLayerIndex = selectedLayerByCol[col];
    usageCounts[selectedLayerIndex]++;
    if (
      col > 0 &&
      selectedLayerByCol[col - 1] !== selectedLayerIndex &&
      !isSoftSwitchBoundary(col, selectedLayerByCol, blendPlan)
    ) {
      switchCount++;
    }
    const layerScores = band.layers.map((layer, layerIndex) => ({
      layerId: layer.diagnostic.layerId,
      score: roundTo(Number(scoresByLayer[layerIndex][col] ?? Number.NEGATIVE_INFINITY)),
    }));
    const sortedScores = [...layerScores]
      .map(entry => entry.score)
      .filter(value => Number.isFinite(value))
      .sort((left, right) => right - left);
    const ambiguous =
      sortedScores.length > 1 &&
      sortedScores[0] - sortedScores[1] <=
        Math.max(AMBIGUITY_MARGIN_ABS, sortedScores[0] * AMBIGUITY_MARGIN_RATIO);
    if (ambiguous && !isContinuityStableColumn(col, selectedLayerByCol)) {
      ambiguousCount++;
    }

    const primaryLayerImage = band.layers[selectedLayerIndex].image;
    const secondaryLayerIndex = blendPlan.secondaryLayerByCol[col];
    const secondaryWeight = Number(blendPlan.secondaryWeightByCol[col] ?? 0);
    const secondaryLayerImage =
      secondaryLayerIndex >= 0 ? band.layers[secondaryLayerIndex].image : null;
    for (let row = 0; row < band.rowCount; row++) {
      const primaryValue = Number(primaryLayerImage[row * width + col]);
      if (!Number.isFinite(primaryValue)) {
        continue;
      }
      let blendedValue = primaryValue;
      if (secondaryLayerImage && secondaryWeight > 0.02) {
        const secondaryValue = Number(secondaryLayerImage[row * width + col]);
        if (Number.isFinite(secondaryValue)) {
          blendedValue =
            primaryValue * (1 - secondaryWeight) + secondaryValue * secondaryWeight;
        }
      }
      bandImageHu[row * width + col] = blendedValue;
    }
  }

  return {
    diagnostic: {
      label: band.label,
      switchCount,
      switchFraction: roundTo(width > 1 ? switchCount / (width - 1) : 0, 4),
      ambiguousColumnFraction: roundTo(width > 0 ? ambiguousCount / width : 0, 4),
      dominantLayerUsage: band.layers.map((layer, layerIndex) => ({
        layerId: layer.diagnostic.layerId,
        columnFraction: roundTo(width > 0 ? usageCounts[layerIndex] / width : 0, 4),
      })),
      sampledColumns: sampledColumns.map(col => ({
        col,
        dominantLayerId: band.layers[selectedLayerByCol[col]].diagnostic.layerId,
        dominantLayerIndex: selectedLayerByCol[col],
        ambiguous:
          (() => {
            const values = band.layers
              .map((_, layerIndex) => Number(scoresByLayer[layerIndex][col]))
              .filter(value => Number.isFinite(value))
              .sort((left, right) => right - left);
            return (
              values.length > 1 &&
              values[0] - values[1] <=
                Math.max(AMBIGUITY_MARGIN_ABS, values[0] * AMBIGUITY_MARGIN_RATIO)
            );
          })(),
        layerScores: band.layers.map((layer, layerIndex) => ({
          layerId: layer.diagnostic.layerId,
          score: roundTo(Number(scoresByLayer[layerIndex][col])),
        })),
      })),
    },
    bandImageHu,
    selectedLayerByCol,
  };
}

function buildGhostingRisk(params: {
  upperArch: PanoV2FusionBandDiagnostics;
  lowerArch: PanoV2FusionBandDiagnostics;
}): 'low' | 'moderate' | 'high' {
  const switchFraction = Math.max(params.upperArch.switchFraction, params.lowerArch.switchFraction);
  const ambiguousFraction = Math.max(
    params.upperArch.ambiguousColumnFraction,
    params.lowerArch.ambiguousColumnFraction
  );
  if (switchFraction > 0.24 || ambiguousFraction > 0.5) {
    return 'high';
  }
  if (switchFraction > 0.12 || ambiguousFraction > 0.32) {
    return 'moderate';
  }
  return 'low';
}

function buildRowCoverageByRow(params: {
  panoWidth: number;
  panoHeight: number;
  upperWeightSum: Float32Array;
  lowerWeightSum: Float32Array;
}): PanoV2FusionRowCoverage[] {
  const { panoWidth, panoHeight, upperWeightSum, lowerWeightSum } = params;
  const rowCoverageByRow: PanoV2FusionRowCoverage[] = [];
  const safeWidth = Math.max(1, panoWidth);

  for (let row = 0; row < panoHeight; row++) {
    let upperPixelCount = 0;
    let lowerPixelCount = 0;
    let overlapPixelCount = 0;
    let coveredPixelCount = 0;
    const rowOffset = row * panoWidth;
    for (let col = 0; col < panoWidth; col++) {
      const index = rowOffset + col;
      const upperCovered = Number(upperWeightSum[index]) > 1e-4;
      const lowerCovered = Number(lowerWeightSum[index]) > 1e-4;
      if (upperCovered) {
        upperPixelCount++;
      }
      if (lowerCovered) {
        lowerPixelCount++;
      }
      if (upperCovered && lowerCovered) {
        overlapPixelCount++;
      }
      if (upperCovered || lowerCovered) {
        coveredPixelCount++;
      }
    }
    const upperFusionCovered = upperPixelCount > 0;
    const lowerFusionCovered = lowerPixelCount > 0;
    const gapFillCovered = upperFusionCovered && lowerFusionCovered;
    const leftEmpty = !upperFusionCovered && !lowerFusionCovered;
    rowCoverageByRow.push({
      row,
      coverageType: leftEmpty
        ? 'empty'
        : gapFillCovered
          ? 'gap-fill'
          : upperFusionCovered
            ? 'upper-fusion'
            : 'lower-fusion',
      upperFusionCovered,
      lowerFusionCovered,
      gapFillCovered,
      leftEmpty,
      upperPixelFraction: roundTo(upperPixelCount / safeWidth, 4),
      lowerPixelFraction: roundTo(lowerPixelCount / safeWidth, 4),
      overlapPixelFraction: roundTo(overlapPixelCount / safeWidth, 4),
      coveredPixelFraction: roundTo(coveredPixelCount / safeWidth, 4),
    });
  }

  return rowCoverageByRow;
}

export function buildPanoV2FusionResult(params: {
  panoWidth: number;
  panoHeight: number;
  layerRender: PanoV2LayerRenderResult;
  fullPlaneLayerImages?: PanoV2FullPlaneLayerImages | null;
}): PanoV2FusionResult {
  const { panoWidth, panoHeight, layerRender, fullPlaneLayerImages } = params;
  if (layerRender.model === 'thick-slab-mip-render' && fullPlaneLayerImages) {
    const planeSize = Math.max(1, panoWidth * panoHeight);
    const pixelData = new Float32Array(planeSize);
    const finalValues: number[] = [];
    const sourceImages = [
      ...fullPlaneLayerImages.upperArch,
      ...fullPlaneLayerImages.lowerArch,
    ];
    for (let index = 0; index < planeSize; index++) {
      let selectedValue = Number.NEGATIVE_INFINITY;
      let hasFiniteValue = false;
      for (let imageIndex = 0; imageIndex < sourceImages.length; imageIndex++) {
        const value = Number(sourceImages[imageIndex]?.[index]);
        if (!Number.isFinite(value)) {
          continue;
        }
        if (!hasFiniteValue || value > selectedValue) {
          selectedValue = value;
          hasFiniteValue = true;
        }
      }
      const clampedValue = clampNumber(hasFiniteValue ? selectedValue : -1000, -1000, 3500);
      pixelData[index] = clampedValue;
      finalValues.push(clampedValue);
    }

    const rowCoverageByRow: PanoV2FusionRowCoverage[] = Array.from(
      { length: panoHeight },
      (_, row) => ({
        row,
        coverageType: 'direct-bypass',
        upperFusionCovered: false,
        lowerFusionCovered: false,
        gapFillCovered: false,
        leftEmpty: false,
        upperPixelFraction: 0,
        lowerPixelFraction: 0,
        overlapPixelFraction: 0,
        coveredPixelFraction: 1,
      })
    );

    return {
      pixelData,
      diagnostics: {
        enabled: true,
        phase: 4,
        model: 'fusion-and-roi-composition',
        implementationVersion: '2026-04-06-fusion-hu-cache-bust-1',
        renderBypass: true,
        outputCoverageFraction: 1,
        overlapFraction: 0,
        gapCenterRow: Math.round((layerRender.upperArch.anchorRow + layerRender.lowerArch.anchorRow) * 0.5),
        ghostingRisk: 'low',
        normalizationHuWindow: {
          lower: -1000,
          upper: 3500,
        },
        outputRange: summarize(finalValues),
        rowCoverageByRow,
        upperArch: {
          label: layerRender.upperArch.label,
          switchCount: 0,
          switchFraction: 0,
          ambiguousColumnFraction: 0,
          dominantLayerUsage: layerRender.upperArch.layers.map((layer, layerIndex) => ({
            layerId: layer.diagnostic.layerId,
            columnFraction: layerIndex === 0 ? 1 : 0,
          })),
          sampledColumns: [],
        },
        lowerArch: {
          label: layerRender.lowerArch.label,
          switchCount: 0,
          switchFraction: 0,
          ambiguousColumnFraction: 0,
          dominantLayerUsage: layerRender.lowerArch.layers.map((layer, layerIndex) => ({
            layerId: layer.diagnostic.layerId,
            columnFraction: layerIndex === 0 ? 1 : 0,
          })),
          sampledColumns: [],
        },
      },
    };
  }
  const upperBand = buildBandFusionResult(layerRender.upperArch);
  const lowerBand = buildBandFusionResult(layerRender.lowerArch);
  const weightedHuSum = new Float32Array(Math.max(1, panoWidth * panoHeight));
  const weightSum = new Float32Array(Math.max(1, panoWidth * panoHeight));
  const upperWeightSum = new Float32Array(Math.max(1, panoWidth * panoHeight));
  const lowerWeightSum = new Float32Array(Math.max(1, panoWidth * panoHeight));
  let overlapCount = 0;

  const applyBand = (
    band: PanoV2BandLayerRenderResult,
    bandResult: PanoV2BandFusionResult,
    bandKey: 'upper' | 'lower'
  ): void => {
    for (let col = 0; col < panoWidth; col++) {
      for (let localRow = 0; localRow < band.rowCount; localRow++) {
        const globalRow = band.rowStart + localRow;
        if (globalRow < 0 || globalRow >= panoHeight) {
          continue;
        }
        const pixelIndex = globalRow * panoWidth + col;
        const value = Number(bandResult.bandImageHu[localRow * panoWidth + col]);
        if (!Number.isFinite(value)) {
          continue;
        }
        const edgeWeight = computeBandEdgeWeight(band, localRow);
        const focusWeight = computeBandFocusWeight(band, localRow);
        const anatomyWeight = smoothstepRange(value, ANATOMY_WEIGHT_LOW_HU, ANATOMY_WEIGHT_HIGH_HU);
        const contributionWeight = edgeWeight * focusWeight * (0.08 + anatomyWeight * 0.92);
        if (contributionWeight <= 1e-4) {
          continue;
        }
        if (weightSum[pixelIndex] > 1e-4) {
          overlapCount++;
        }
        weightedHuSum[pixelIndex] += value * contributionWeight;
        weightSum[pixelIndex] += contributionWeight;
        if (bandKey === 'upper') {
          upperWeightSum[pixelIndex] += contributionWeight;
        } else {
          lowerWeightSum[pixelIndex] += contributionWeight;
        }
      }
    }
  };

  applyBand(layerRender.upperArch, upperBand, 'upper');
  applyBand(layerRender.lowerArch, lowerBand, 'lower');

  const coveredHuValues: number[] = [];
  const fusedHuBuffer = new Float32Array(Math.max(1, panoWidth * panoHeight));
  fusedHuBuffer.fill(Number.NaN);
  const pixelData = new Float32Array(Math.max(1, panoWidth * panoHeight));
  let coveredPixelCount = 0;
  for (let index = 0; index < pixelData.length; index++) {
    const weight = Number(weightSum[index]);
    if (weight <= 1e-4) {
      continue;
    }
    const rawFusedHu = Number(weightedHuSum[index]) / weight;
    const row = Math.floor(index / Math.max(1, panoWidth));
    const fusedHu = suppressOutsideArchRoiHu({
      row,
      fusedHu: rawFusedHu,
      upperArch: layerRender.upperArch,
      lowerArch: layerRender.lowerArch,
    });
    if (!Number.isFinite(fusedHu)) {
      continue;
    }
    coveredPixelCount++;
    fusedHuBuffer[index] = fusedHu;
    coveredHuValues.push(fusedHu);
  }

  let lowerHu = percentile(coveredHuValues, 0.08);
  let upperHu = percentile(coveredHuValues, 0.995);
  if (!Number.isFinite(lowerHu) || !Number.isFinite(upperHu) || upperHu - lowerHu < 320) {
    lowerHu = percentile(coveredHuValues, 0.02);
    upperHu = percentile(coveredHuValues, 0.98);
  }
  if (!Number.isFinite(lowerHu) || !Number.isFinite(upperHu) || upperHu <= lowerHu) {
    lowerHu = -850;
    upperHu = 950;
  }

  const finalValues: number[] = [];
  for (let index = 0; index < pixelData.length; index++) {
    const fusedHu = Number(fusedHuBuffer[index]);
    const row = Math.floor(index / Math.max(1, panoWidth));
    const airBaselineHu = computeRowAirBaselineHu({
      row,
      upperArch: layerRender.upperArch,
      lowerArch: layerRender.lowerArch,
    });
    if (!Number.isFinite(fusedHu)) {
      pixelData[index] = airBaselineHu;
      finalValues.push(airBaselineHu);
      continue;
    }
    const weight = Number(weightSum[index]);
    const coverage = smoothstepRange(weight, COVERAGE_WEIGHT_SOFT_START, COVERAGE_WEIGHT_FULL);
    const finalHu = airBaselineHu * (1 - coverage) + fusedHu * coverage;
    pixelData[index] = finalHu;
    finalValues.push(finalHu);
  }

  const rowCoverageByRow = buildRowCoverageByRow({
    panoWidth,
    panoHeight,
    upperWeightSum,
    lowerWeightSum,
  });

  return {
    pixelData,
    diagnostics: {
      enabled: true,
      phase: 4,
      model: 'fusion-and-roi-composition',
      implementationVersion: '2026-04-06-fusion-hu-cache-bust-1',
      renderBypass: false,
      outputCoverageFraction: roundTo(pixelData.length > 0 ? coveredPixelCount / pixelData.length : 0, 4),
      overlapFraction: roundTo(coveredPixelCount > 0 ? overlapCount / coveredPixelCount : 0, 4),
      gapCenterRow: Math.round((layerRender.upperArch.anchorRow + layerRender.lowerArch.anchorRow) * 0.5),
      ghostingRisk: buildGhostingRisk({
        upperArch: upperBand.diagnostic,
        lowerArch: lowerBand.diagnostic,
      }),
      normalizationHuWindow: {
        lower: roundTo(lowerHu),
        upper: roundTo(upperHu),
      },
      outputRange: summarize(finalValues),
      rowCoverageByRow,
      upperArch: upperBand.diagnostic,
      lowerArch: lowerBand.diagnostic,
    },
  };
}
