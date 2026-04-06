export type PanoV2TrackLabel = 'upperArch' | 'lowerArch';

export interface PanoV2TrackInput {
  label: PanoV2TrackLabel;
  anchorRow: number;
  selectedDepthMm: Float32Array;
  selectedReliabilityByCol: Float32Array;
  scoreGapByCol: Float32Array;
  unstableMask: Uint8Array;
  envelopeHalfWidthMmByCol?: Float32Array | null;
  ambiguousGapThreshold?: number;
  lowConfidenceThreshold?: number;
}

export interface PanoV2RunDiagnostic {
  startCol: number;
  endCol: number;
  length: number;
  maxJumpMm: number;
}

export interface PanoV2NumericSummary {
  min: number;
  p50: number;
  p90: number;
  max: number;
  mean: number;
}

export interface PanoV2SampledColumn {
  col: number;
  depthMm: number;
  confidence: number;
  scoreGap: number;
  zoneHalfWidthMm: number;
  ambiguous: boolean;
  continuityBreak: boolean;
  unstable: boolean;
}

export interface PanoV2TrackDiagnostics {
  label: PanoV2TrackLabel;
  anchorRow: number;
  columnCount: number;
  ambiguousGapThreshold: number;
  lowConfidenceThreshold: number;
  continuityJumpThresholdMm: number;
  supportCenterDepthFirst8Mm: number[];
  supportCenterDepthLast8Mm: number[];
  localConfidenceFirst8: number[];
  localConfidenceLast8: number[];
  reconstructionZoneHalfWidthFirst8Mm: number[];
  reconstructionZoneHalfWidthLast8Mm: number[];
  sampledColumns: PanoV2SampledColumn[];
  depthRangeMm: PanoV2NumericSummary;
  localConfidence: PanoV2NumericSummary;
  scoreGap: PanoV2NumericSummary;
  reconstructionZoneHalfWidthMm: PanoV2NumericSummary;
  pathJumpMm: PanoV2NumericSummary;
  ambiguousColumnFraction: number;
  lowConfidenceFraction: number;
  continuityBreakFraction: number;
  unstableColumnFraction: number;
  ambiguousRuns: PanoV2RunDiagnostic[];
  continuityBreakRuns: PanoV2RunDiagnostic[];
  unstableRuns: PanoV2RunDiagnostic[];
}

export interface PanoV2GeometryDiagnostics {
  enabled: true;
  phase: 1;
  model: 'dual-arch-support-geometry';
  panoWidth: number;
  sampledColumnCount: number;
  upperArch: PanoV2TrackDiagnostics;
  lowerArch: PanoV2TrackDiagnostics;
  interArchSeparationMm: PanoV2NumericSummary;
  nonCrossingColumnFraction: number;
  combinedAmbiguousColumnFraction: number;
  combinedContinuityBreakFraction: number;
}

export interface PanoV2TrackRefinementInput {
  label: PanoV2TrackLabel;
  anchorRow: number;
  panoWidth: number;
  depthSamples: number;
  depthOffsetsMm: Float32Array;
  rawScoreByColDepth: Float32Array;
  rawReliabilityByColDepth: Float32Array;
  baseSelectedDepthMm: Float32Array;
  baseSelectedReliabilityByCol: Float32Array;
  baseScoreGapByCol: Float32Array;
  baseUnstableMask: Uint8Array;
  baseEnvelopeHalfWidthMmByCol?: Float32Array | null;
}

export interface PanoV2TrackRefinementResult extends PanoV2TrackInput {
  ambiguousColumnCount: number;
  depthMinMm: number;
  depthMaxMm: number;
  depthStdMm: number;
  pathJumpP95Mm: number;
  pathJumpMaxMm: number;
  refinementDiagnostics: {
    smoothedBaseDepthFirst8Mm: number[];
    smoothedBaseDepthLast8Mm: number[];
    narrowedSearchHalfWidthFirst8Mm: number[];
    narrowedSearchHalfWidthLast8Mm: number[];
    continuityPenaltyPerMm: number;
    priorPenaltyPerMm: number;
  };
}

const PANO_V2_REFINEMENT_MIN_SEARCH_HALF_WIDTH_MM = 0.68;
const PANO_V2_REFINEMENT_MAX_SEARCH_HALF_WIDTH_MM = 1.32;
const PANO_V2_REFINEMENT_PRIOR_PENALTY_PER_MM = 0.26;
const PANO_V2_REFINEMENT_TRANSITION_PENALTY_PER_MM = 0.34;
const PANO_V2_REFINEMENT_EDGE_MARGIN_MM = 0.3;
const PANO_V2_REFINEMENT_EDGE_PENALTY = 0.24;
const PANO_V2_REFINEMENT_RELIABILITY_BONUS = 0.32;
const PANO_V2_REFINEMENT_AMBIGUITY_GAP_THRESHOLD = 0.05;
const PANO_V2_REFINEMENT_LOW_CONFIDENCE_THRESHOLD = 0.58;
const PANO_V2_REFINEMENT_PATH_JUMP_THRESHOLD_MM = 0.72;
const PANO_V2_REFINEMENT_SHORT_RUN_MAX_COLUMNS = 2;
const PANO_V2_REFINEMENT_SHORT_RUN_MARGIN = 0.065;
const PANO_V2_REFINEMENT_REPAIR_MAX_RUN_COLUMNS = 12;
const PANO_V2_REFINEMENT_REPAIR_SCORE_MARGIN = 0.075;
const PANO_V2_REFINEMENT_REPAIR_STABLE_RELIABILITY = 0.62;
const PANO_V2_REFINEMENT_REPAIR_STABLE_SCORE_GAP = 0.04;
const PANO_V2_REFINEMENT_NEIGHBOR_STABILITY_TOLERANCE_MM = 0.36;
const PANO_V2_REFINEMENT_REPAIRED_AMBIGUITY_GAP_THRESHOLD = 0.038;
const PANO_V2_REFINEMENT_REPAIRED_LOW_CONFIDENCE_THRESHOLD = 0.48;

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

function toRoundedArray(values: ArrayLike<number>, start: number, count: number): number[] {
  const result: number[] = [];
  const safeStart = Math.max(0, start);
  const safeEnd = Math.min(values.length, safeStart + Math.max(0, count));
  for (let index = safeStart; index < safeEnd; index++) {
    result.push(roundTo(Number(values[index])));
  }
  return result;
}

function percentile(values: number[], ratio: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const clampedRatio = Math.min(1, Math.max(0, ratio));
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
  if (values.length === 0) {
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

function buildRunDiagnostics(mask: boolean[], jumpsMm: number[]): PanoV2RunDiagnostic[] {
  const runs: PanoV2RunDiagnostic[] = [];
  let start = -1;
  for (let col = 0; col <= mask.length; col++) {
    const active = col < mask.length ? mask[col] : false;
    if (active && start < 0) {
      start = col;
      continue;
    }
    if (active || start < 0) {
      continue;
    }
    let maxJumpMm = 0;
    for (let jumpCol = start; jumpCol <= col - 1; jumpCol++) {
      maxJumpMm = Math.max(maxJumpMm, Math.abs(jumpsMm[jumpCol] ?? 0));
    }
    runs.push({
      startCol: start,
      endCol: col - 1,
      length: col - start,
      maxJumpMm: roundTo(maxJumpMm),
    });
    start = -1;
  }
  return runs;
}

function buildSampleColumnIndices(columnCount: number, targetSampleCount = 17): number[] {
  if (columnCount <= 0) {
    return [];
  }
  if (columnCount <= targetSampleCount) {
    return Array.from({ length: columnCount }, (_, col) => col);
  }
  const indices = new Set<number>([0, columnCount - 1]);
  const denominator = Math.max(1, targetSampleCount - 1);
  for (let sample = 0; sample < targetSampleCount; sample++) {
    const ratio = sample / denominator;
    indices.add(Math.min(columnCount - 1, Math.max(0, Math.round(ratio * (columnCount - 1)))));
  }
  return Array.from(indices).sort((left, right) => left - right);
}

function meanWindow(values: Float32Array, center: number, radius: number): number {
  let sum = 0;
  let count = 0;
  const start = Math.max(0, center - radius);
  const end = Math.min(values.length - 1, center + radius);
  for (let index = start; index <= end; index++) {
    const value = Number(values[index]);
    if (!Number.isFinite(value)) {
      continue;
    }
    sum += value;
    count++;
  }
  return count > 0 ? sum / count : Number(values[center] ?? 0);
}

function getNearestDepthIndex(targetDepthMm: number, depthOffsetsMm: Float32Array): number {
  let nearestIndex = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < depthOffsetsMm.length; index++) {
    const distance = Math.abs(Number(depthOffsetsMm[index]) - targetDepthMm);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  }
  return nearestIndex;
}

function localRefinementScore(params: {
  col: number;
  depthIndex: number;
  depthMm: number;
  rawScoreByColDepth: Float32Array;
  rawReliabilityByColDepth: Float32Array;
  depthSamples: number;
  targetDepthMm: number;
  allowedHalfWidthMm: number;
  depthHalfRangeMm: number;
}): number {
  const {
    col,
    depthIndex,
    depthMm,
    rawScoreByColDepth,
    rawReliabilityByColDepth,
    depthSamples,
    targetDepthMm,
    allowedHalfWidthMm,
    depthHalfRangeMm,
  } = params;
  if (Math.abs(depthMm - targetDepthMm) > allowedHalfWidthMm) {
    return Number.NEGATIVE_INFINITY;
  }
  const scoreIndex = col * depthSamples + depthIndex;
  const rawScore = Number(rawScoreByColDepth[scoreIndex]);
  if (!Number.isFinite(rawScore)) {
    return Number.NEGATIVE_INFINITY;
  }
  const reliability = clampNumber(Number(rawReliabilityByColDepth[scoreIndex]) || 0, 0, 1);
  const targetPenalty =
    Math.abs(depthMm - targetDepthMm) * PANO_V2_REFINEMENT_PRIOR_PENALTY_PER_MM;
  const edgeDistanceMm = Math.abs(depthMm) - Math.max(0.3, depthHalfRangeMm - PANO_V2_REFINEMENT_EDGE_MARGIN_MM);
  const edgePenalty =
    edgeDistanceMm > 0 ? edgeDistanceMm * PANO_V2_REFINEMENT_EDGE_PENALTY : 0;
  return (
    rawScore +
    reliability * PANO_V2_REFINEMENT_RELIABILITY_BONUS -
    targetPenalty -
    edgePenalty
  );
}

function buildNarrowedSearchHalfWidthMm(input: PanoV2TrackRefinementInput): Float32Array {
  const widths = new Float32Array(input.panoWidth);
  for (let col = 0; col < input.panoWidth; col++) {
    const baseWidth = Number(input.baseEnvelopeHalfWidthMmByCol?.[col] ?? 1.35);
    const baseReliability = clampNumber(Number(input.baseSelectedReliabilityByCol[col]) || 0, 0, 1);
    const baseScoreGap = Math.max(0, Number(input.baseScoreGapByCol[col]) || 0);
    const unstable = Number(input.baseUnstableMask[col]) > 0;
    const weakBase = unstable || baseReliability < 0.82 || baseScoreGap < 0.085;
    const narrowedWidth = weakBase
      ? baseWidth * 0.82 + 0.08
      : baseWidth * 0.68 + 0.05;
    widths[col] = clampNumber(
      narrowedWidth,
      PANO_V2_REFINEMENT_MIN_SEARCH_HALF_WIDTH_MM,
      PANO_V2_REFINEMENT_MAX_SEARCH_HALF_WIDTH_MM
    );
  }
  return widths;
}

function collapseShortRuns(
  selectedDepthIndices: Int16Array,
  localScoresByColDepth: Float32Array,
  depthSamples: number
): void {
  const width = selectedDepthIndices.length;
  let runStart = 0;
  while (runStart < width) {
    const layerIndex = selectedDepthIndices[runStart];
    let runEnd = runStart;
    while (runEnd + 1 < width && selectedDepthIndices[runEnd + 1] === layerIndex) {
      runEnd++;
    }
    const runLength = runEnd - runStart + 1;
    const leftIndex = runStart > 0 ? selectedDepthIndices[runStart - 1] : -1;
    const rightIndex = runEnd + 1 < width ? selectedDepthIndices[runEnd + 1] : -1;
    if (
      runLength <= PANO_V2_REFINEMENT_SHORT_RUN_MAX_COLUMNS &&
      leftIndex >= 0 &&
      leftIndex === rightIndex &&
      leftIndex !== layerIndex
    ) {
      let shouldCollapse = true;
      for (let col = runStart; col <= runEnd; col++) {
        const currentScore = Number(localScoresByColDepth[col * depthSamples + layerIndex]);
        const neighborScore = Number(localScoresByColDepth[col * depthSamples + leftIndex]);
        if (
          !Number.isFinite(currentScore) ||
          !Number.isFinite(neighborScore) ||
          currentScore - neighborScore > PANO_V2_REFINEMENT_SHORT_RUN_MARGIN
        ) {
          shouldCollapse = false;
          break;
        }
      }
      if (shouldCollapse) {
        for (let col = runStart; col <= runEnd; col++) {
          selectedDepthIndices[col] = leftIndex;
        }
      }
    }
    runStart = runEnd + 1;
  }
}

function computeColumnScoreGap(
  localScoresByColDepth: Float32Array,
  depthSamples: number,
  col: number,
  selectedDepthIndex: number
): number {
  const safeSelectedDepthIndex = clampNumber(selectedDepthIndex, 0, Math.max(0, depthSamples - 1));
  let bestLocalScore = Number(
    localScoresByColDepth[col * depthSamples + safeSelectedDepthIndex] ?? Number.NEGATIVE_INFINITY
  );
  let secondBestLocalScore = Number.NEGATIVE_INFINITY;
  for (let depthIndex = 0; depthIndex < depthSamples; depthIndex++) {
    if (depthIndex === safeSelectedDepthIndex) {
      continue;
    }
    const candidateScore = Number(localScoresByColDepth[col * depthSamples + depthIndex]);
    if (!Number.isFinite(candidateScore)) {
      continue;
    }
    if (candidateScore > bestLocalScore) {
      secondBestLocalScore = bestLocalScore;
      bestLocalScore = candidateScore;
    } else if (candidateScore > secondBestLocalScore) {
      secondBestLocalScore = candidateScore;
    }
  }
  return Number.isFinite(bestLocalScore) && Number.isFinite(secondBestLocalScore)
    ? bestLocalScore - secondBestLocalScore
    : Number.isFinite(bestLocalScore)
      ? bestLocalScore
      : 0;
}

function stabilizeSelectedDepthIndices(params: {
  selectedDepthIndices: Int16Array;
  localScoresByColDepth: Float32Array;
  depthSamples: number;
  depthOffsetsMm: Float32Array;
  smoothedBaseDepthMm: Float32Array;
  rawReliabilityByColDepth: Float32Array;
  baseUnstableMask: Uint8Array;
}): Uint8Array {
  const {
    selectedDepthIndices,
    localScoresByColDepth,
    depthSamples,
    depthOffsetsMm,
    smoothedBaseDepthMm,
    rawReliabilityByColDepth,
    baseUnstableMask,
  } = params;
  const width = selectedDepthIndices.length;
  const repairedMask = new Uint8Array(width);
  if (width <= 0 || depthSamples <= 0) {
    return repairedMask;
  }

  const stableSeedMask = new Uint8Array(width);
  for (let col = 0; col < width; col++) {
    const selectedDepthIndex = clampNumber(
      Number(selectedDepthIndices[col] ?? 0),
      0,
      Math.max(0, depthSamples - 1)
    );
    const reliability = clampNumber(
      Number(rawReliabilityByColDepth[col * depthSamples + selectedDepthIndex]) || 0,
      0,
      1
    );
    const scoreGap = computeColumnScoreGap(
      localScoresByColDepth,
      depthSamples,
      col,
      selectedDepthIndex
    );
    stableSeedMask[col] =
      Number(baseUnstableMask[col]) <= 0 &&
      reliability >= PANO_V2_REFINEMENT_REPAIR_STABLE_RELIABILITY &&
      scoreGap >= PANO_V2_REFINEMENT_REPAIR_STABLE_SCORE_GAP
        ? 1
        : 0;
  }

  let runStart = 0;
  while (runStart < width) {
    if (stableSeedMask[runStart] > 0) {
      runStart++;
      continue;
    }
    let runEnd = runStart;
    while (runEnd + 1 < width && stableSeedMask[runEnd + 1] <= 0) {
      runEnd++;
    }
    const runLength = runEnd - runStart + 1;
    const leftStableCol = runStart > 0 && stableSeedMask[runStart - 1] > 0 ? runStart - 1 : -1;
    const rightStableCol =
      runEnd + 1 < width && stableSeedMask[runEnd + 1] > 0 ? runEnd + 1 : -1;
    const canRepair =
      runLength <= PANO_V2_REFINEMENT_REPAIR_MAX_RUN_COLUMNS &&
      (leftStableCol >= 0 || rightStableCol >= 0);
    if (canRepair) {
      const leftDepthMm =
        leftStableCol >= 0
          ? Number(depthOffsetsMm[selectedDepthIndices[leftStableCol]] ?? 0)
          : Number.NaN;
      const rightDepthMm =
        rightStableCol >= 0
          ? Number(depthOffsetsMm[selectedDepthIndices[rightStableCol]] ?? 0)
          : Number.NaN;
      for (let col = runStart; col <= runEnd; col++) {
        let targetDepthMm = Number(smoothedBaseDepthMm[col] ?? 0);
        if (leftStableCol >= 0 && rightStableCol >= 0 && rightStableCol > leftStableCol) {
          const t = (col - leftStableCol) / Math.max(1, rightStableCol - leftStableCol);
          const bridgedDepthMm = leftDepthMm * (1 - t) + rightDepthMm * t;
          targetDepthMm = bridgedDepthMm * 0.72 + targetDepthMm * 0.28;
        } else if (leftStableCol >= 0) {
          targetDepthMm = leftDepthMm * 0.7 + targetDepthMm * 0.3;
        } else if (rightStableCol >= 0) {
          targetDepthMm = rightDepthMm * 0.7 + targetDepthMm * 0.3;
        }

        const targetDepthIndex = getNearestDepthIndex(targetDepthMm, depthOffsetsMm);
        const repairedScore = Number(
          localScoresByColDepth[col * depthSamples + targetDepthIndex] ?? Number.NEGATIVE_INFINITY
        );
        let bestScore = Number.NEGATIVE_INFINITY;
        for (let depthIndex = 0; depthIndex < depthSamples; depthIndex++) {
          const candidateScore = Number(localScoresByColDepth[col * depthSamples + depthIndex]);
          if (Number.isFinite(candidateScore) && candidateScore > bestScore) {
            bestScore = candidateScore;
          }
        }
        if (
          Number.isFinite(repairedScore) &&
          Number.isFinite(bestScore) &&
          bestScore - repairedScore <= PANO_V2_REFINEMENT_REPAIR_SCORE_MARGIN
        ) {
          selectedDepthIndices[col] = targetDepthIndex;
          repairedMask[col] = 1;
        }
      }
    }
    runStart = runEnd + 1;
  }

  return repairedMask;
}

export function refinePanoV2TrackFromScores(
  input: PanoV2TrackRefinementInput
): PanoV2TrackRefinementResult {
  const width = Math.min(
    input.panoWidth,
    input.baseSelectedDepthMm.length,
    input.baseSelectedReliabilityByCol.length,
    input.baseScoreGapByCol.length
  );
  if (width <= 0 || input.depthSamples <= 0 || input.depthOffsetsMm.length === 0) {
    return {
      label: input.label,
      anchorRow: input.anchorRow,
      selectedDepthMm: new Float32Array(0),
      selectedReliabilityByCol: new Float32Array(0),
      scoreGapByCol: new Float32Array(0),
      unstableMask: new Uint8Array(0),
      envelopeHalfWidthMmByCol: new Float32Array(0),
      ambiguousColumnCount: 0,
      depthMinMm: 0,
      depthMaxMm: 0,
      depthStdMm: 0,
      pathJumpP95Mm: 0,
      pathJumpMaxMm: 0,
      refinementDiagnostics: {
        smoothedBaseDepthFirst8Mm: [],
        smoothedBaseDepthLast8Mm: [],
        narrowedSearchHalfWidthFirst8Mm: [],
        narrowedSearchHalfWidthLast8Mm: [],
        continuityPenaltyPerMm: roundTo(PANO_V2_REFINEMENT_TRANSITION_PENALTY_PER_MM),
        priorPenaltyPerMm: roundTo(PANO_V2_REFINEMENT_PRIOR_PENALTY_PER_MM),
      },
    };
  }

  const depthHalfRangeMm = Math.max(
    0.5,
    Math.max(
      Math.abs(Number(input.depthOffsetsMm[0] ?? 0)),
      Math.abs(Number(input.depthOffsetsMm[input.depthOffsetsMm.length - 1] ?? 0))
    )
  );
  const smoothedBaseDepthMm = new Float32Array(width);
  for (let col = 0; col < width; col++) {
    smoothedBaseDepthMm[col] = meanWindow(input.baseSelectedDepthMm, col, 3);
  }
  const narrowedSearchHalfWidthMm = buildNarrowedSearchHalfWidthMm(input);
  const localScoresByColDepth = new Float32Array(Math.max(1, width * input.depthSamples));
  localScoresByColDepth.fill(Number.NEGATIVE_INFINITY);
  for (let col = 0; col < width; col++) {
    const targetDepthMm = Number(smoothedBaseDepthMm[col] ?? 0);
    const allowedHalfWidthMm = Number(narrowedSearchHalfWidthMm[col] ?? 1);
    for (let depthIndex = 0; depthIndex < input.depthSamples; depthIndex++) {
      const depthMm = Number(input.depthOffsetsMm[depthIndex] ?? 0);
      localScoresByColDepth[col * input.depthSamples + depthIndex] = localRefinementScore({
        col,
        depthIndex,
        depthMm,
        rawScoreByColDepth: input.rawScoreByColDepth,
        rawReliabilityByColDepth: input.rawReliabilityByColDepth,
        depthSamples: input.depthSamples,
        targetDepthMm,
        allowedHalfWidthMm,
        depthHalfRangeMm,
      });
    }
  }

  const backPointers = new Int16Array(Math.max(1, width * input.depthSamples));
  const previousScores = new Float32Array(input.depthSamples);
  const currentScores = new Float32Array(input.depthSamples);
  previousScores.fill(Number.NEGATIVE_INFINITY);
  currentScores.fill(Number.NEGATIVE_INFINITY);

  for (let depthIndex = 0; depthIndex < input.depthSamples; depthIndex++) {
    previousScores[depthIndex] = Number(localScoresByColDepth[depthIndex] ?? Number.NEGATIVE_INFINITY);
  }

  for (let col = 1; col < width; col++) {
    for (let depthIndex = 0; depthIndex < input.depthSamples; depthIndex++) {
      const localScore = Number(localScoresByColDepth[col * input.depthSamples + depthIndex]);
      if (!Number.isFinite(localScore)) {
        currentScores[depthIndex] = Number.NEGATIVE_INFINITY;
        backPointers[col * input.depthSamples + depthIndex] = depthIndex;
        continue;
      }
      const depthMm = Number(input.depthOffsetsMm[depthIndex] ?? 0);
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestPreviousDepthIndex = depthIndex;
      for (let previousDepthIndex = 0; previousDepthIndex < input.depthSamples; previousDepthIndex++) {
        const previousScore = Number(previousScores[previousDepthIndex]);
        if (!Number.isFinite(previousScore)) {
          continue;
        }
        const previousDepthMm = Number(input.depthOffsetsMm[previousDepthIndex] ?? 0);
        const transitionPenalty =
          Math.abs(depthMm - previousDepthMm) * PANO_V2_REFINEMENT_TRANSITION_PENALTY_PER_MM;
        const candidateScore = previousScore + localScore - transitionPenalty;
        if (candidateScore > bestScore) {
          bestScore = candidateScore;
          bestPreviousDepthIndex = previousDepthIndex;
        }
      }
      currentScores[depthIndex] = bestScore;
      backPointers[col * input.depthSamples + depthIndex] = bestPreviousDepthIndex;
    }
    previousScores.set(currentScores);
  }

  const selectedDepthIndices = new Int16Array(width);
  let bestFinalDepthIndex = 0;
  let bestFinalScore = Number.NEGATIVE_INFINITY;
  for (let depthIndex = 0; depthIndex < input.depthSamples; depthIndex++) {
    if (previousScores[depthIndex] > bestFinalScore) {
      bestFinalScore = previousScores[depthIndex];
      bestFinalDepthIndex = depthIndex;
    }
  }
  selectedDepthIndices[width - 1] = bestFinalDepthIndex;
  for (let col = width - 1; col > 0; col--) {
    selectedDepthIndices[col - 1] =
      backPointers[col * input.depthSamples + selectedDepthIndices[col]];
  }
  collapseShortRuns(selectedDepthIndices, localScoresByColDepth, input.depthSamples);
  const repairedMask = stabilizeSelectedDepthIndices({
    selectedDepthIndices,
    localScoresByColDepth,
    depthSamples: input.depthSamples,
    depthOffsetsMm: input.depthOffsetsMm,
    smoothedBaseDepthMm,
    rawReliabilityByColDepth: input.rawReliabilityByColDepth,
    baseUnstableMask: input.baseUnstableMask,
  });

  const selectedDepthMm = new Float32Array(width);
  const selectedReliabilityByCol = new Float32Array(width);
  const scoreGapByCol = new Float32Array(width);
  const unstableMask = new Uint8Array(width);
  const envelopeHalfWidthMmByCol = new Float32Array(width);
  const jumpsMm: number[] = [];
  let ambiguousColumnCount = 0;
  let depthMinMm = Number.POSITIVE_INFINITY;
  let depthMaxMm = Number.NEGATIVE_INFINITY;
  let depthSumMm = 0;

  for (let col = 0; col < width; col++) {
    const repaired = repairedMask[col] > 0;
    const baseUnstable = Number(input.baseUnstableMask[col]) > 0;
    let selectedDepthIndex = selectedDepthIndices[col];
    if (selectedDepthIndex < 0 || selectedDepthIndex >= input.depthSamples) {
      selectedDepthIndex = getNearestDepthIndex(Number(smoothedBaseDepthMm[col] ?? 0), input.depthOffsetsMm);
    }
    let selectedDepth = Number(input.depthOffsetsMm[selectedDepthIndex] ?? 0);
    const localMedianDepth =
      col > 0 && col + 1 < width
        ? (Number(input.depthOffsetsMm[selectedDepthIndices[col - 1]] ?? selectedDepth) +
            selectedDepth +
            Number(input.depthOffsetsMm[selectedDepthIndices[col + 1]] ?? selectedDepth)) /
          3
        : selectedDepth;
    const jumpMm =
      col > 0 ? Math.abs(selectedDepth - Number(selectedDepthMm[col - 1] ?? selectedDepth)) : 0;
    const reliabilityIndex = col * input.depthSamples + selectedDepthIndex;
    let reliability = clampNumber(Number(input.rawReliabilityByColDepth[reliabilityIndex]) || 0, 0, 1);
    if (jumpMm > PANO_V2_REFINEMENT_PATH_JUMP_THRESHOLD_MM) {
      const repairedDepth = 0.55 * localMedianDepth + 0.45 * Number(smoothedBaseDepthMm[col] ?? selectedDepth);
      selectedDepthIndex = getNearestDepthIndex(repairedDepth, input.depthOffsetsMm);
      selectedDepth = Number(input.depthOffsetsMm[selectedDepthIndex] ?? 0);
      reliability = clampNumber(Number(input.rawReliabilityByColDepth[col * input.depthSamples + selectedDepthIndex]) || 0, 0, 1);
    }

    const scoreGap = computeColumnScoreGap(
      localScoresByColDepth,
      input.depthSamples,
      col,
      selectedDepthIndex
    );
    const localNeighborDepthMm =
      col > 0 && col + 1 < width
        ? 0.5 *
          (Number(input.depthOffsetsMm[selectedDepthIndices[col - 1]] ?? selectedDepth) +
            Number(input.depthOffsetsMm[selectedDepthIndices[col + 1]] ?? selectedDepth))
        : selectedDepth;
    const neighborhoodStable =
      repaired ||
      (col > 0 &&
        col + 1 < width &&
        Math.abs(selectedDepth - localNeighborDepthMm) <=
          PANO_V2_REFINEMENT_NEIGHBOR_STABILITY_TOLERANCE_MM &&
        jumpMm <= PANO_V2_REFINEMENT_PATH_JUMP_THRESHOLD_MM * 0.95);
    const ambiguityGapThreshold = repaired
      ? PANO_V2_REFINEMENT_REPAIRED_AMBIGUITY_GAP_THRESHOLD
      : PANO_V2_REFINEMENT_AMBIGUITY_GAP_THRESHOLD;
    const lowConfidenceThreshold = repaired
      ? PANO_V2_REFINEMENT_REPAIRED_LOW_CONFIDENCE_THRESHOLD
      : PANO_V2_REFINEMENT_LOW_CONFIDENCE_THRESHOLD;
    const ambiguous =
      scoreGap < ambiguityGapThreshold || reliability < lowConfidenceThreshold;
    const baseInstabilityPersists =
      baseUnstable && !repaired && reliability < 0.72 && scoreGap < 0.09;
    const unstable =
      jumpMm > PANO_V2_REFINEMENT_PATH_JUMP_THRESHOLD_MM ||
      baseInstabilityPersists ||
      (ambiguous && !neighborhoodStable);

    selectedDepthMm[col] = selectedDepth;
    selectedReliabilityByCol[col] = clampNumber(
      reliability * (unstable ? 0.92 : repaired ? 1.04 : 1),
      0,
      1
    );
    scoreGapByCol[col] = Math.max(0, scoreGap);
    envelopeHalfWidthMmByCol[col] = clampNumber(
      Number(narrowedSearchHalfWidthMm[col] ?? 1) * (repaired ? 0.9 : unstable ? 0.94 : 0.84),
      0.62,
      1.18
    );
    unstableMask[col] = unstable ? 1 : 0;
    if (ambiguous && !neighborhoodStable) {
      ambiguousColumnCount++;
    }
    depthMinMm = Math.min(depthMinMm, selectedDepth);
    depthMaxMm = Math.max(depthMaxMm, selectedDepth);
    depthSumMm += selectedDepth;
    if (col > 0) {
      jumpsMm.push(Math.abs(selectedDepth - Number(selectedDepthMm[col - 1])));
    }
  }

  const depthMeanMm = width > 0 ? depthSumMm / width : 0;
  let depthVarianceMm = 0;
  for (let col = 0; col < width; col++) {
    const delta = Number(selectedDepthMm[col]) - depthMeanMm;
    depthVarianceMm += delta * delta;
  }

  return {
    label: input.label,
    anchorRow: input.anchorRow,
    selectedDepthMm,
    selectedReliabilityByCol,
    scoreGapByCol,
    unstableMask,
    envelopeHalfWidthMmByCol,
    ambiguousColumnCount,
    depthMinMm: Number.isFinite(depthMinMm) ? roundTo(depthMinMm) : 0,
    depthMaxMm: Number.isFinite(depthMaxMm) ? roundTo(depthMaxMm) : 0,
    depthStdMm: roundTo(width > 0 ? Math.sqrt(depthVarianceMm / Math.max(1, width)) : 0),
    pathJumpP95Mm: roundTo(jumpsMm.length > 0 ? percentile(jumpsMm, 0.95) : 0),
    pathJumpMaxMm: roundTo(jumpsMm.length > 0 ? Math.max(...jumpsMm) : 0),
    refinementDiagnostics: {
      smoothedBaseDepthFirst8Mm: toRoundedArray(smoothedBaseDepthMm, 0, 8),
      smoothedBaseDepthLast8Mm: toRoundedArray(smoothedBaseDepthMm, Math.max(0, width - 8), 8),
      narrowedSearchHalfWidthFirst8Mm: toRoundedArray(narrowedSearchHalfWidthMm, 0, 8),
      narrowedSearchHalfWidthLast8Mm: toRoundedArray(
        narrowedSearchHalfWidthMm,
        Math.max(0, width - 8),
        8
      ),
      continuityPenaltyPerMm: roundTo(PANO_V2_REFINEMENT_TRANSITION_PENALTY_PER_MM),
      priorPenaltyPerMm: roundTo(PANO_V2_REFINEMENT_PRIOR_PENALTY_PER_MM),
    },
  };
}

function buildTrackDiagnostics(input: PanoV2TrackInput): PanoV2TrackDiagnostics {
  const columnCount = input.selectedDepthMm.length;
  const ambiguousGapThreshold = input.ambiguousGapThreshold ?? 0.08;
  const lowConfidenceThreshold = input.lowConfidenceThreshold ?? 0.55;
  const depths = Array.from(input.selectedDepthMm, value => Number(value));
  const confidences = Array.from(input.selectedReliabilityByCol, value => Number(value));
  const scoreGaps = Array.from(input.scoreGapByCol, value => Number(value));
  const zoneHalfWidths = input.envelopeHalfWidthMmByCol
    ? Array.from(input.envelopeHalfWidthMmByCol, value => Number(value))
    : Array.from({ length: columnCount }, () => 0);
  const jumpsMm = Array.from({ length: columnCount }, (_, col) =>
    col > 0 ? Math.abs(depths[col] - depths[col - 1]) : 0
  );
  const continuityJumpThresholdMm = roundTo(
    Math.max(0.9, percentile(jumpsMm.slice(1), 0.9) * 1.35)
  );
  const ambiguousMask = Array.from({ length: columnCount }, (_, col) => {
    const scoreGap = scoreGaps[col] ?? 0;
    const confidence = confidences[col] ?? 0;
    return scoreGap < ambiguousGapThreshold || confidence < lowConfidenceThreshold;
  });
  const unstableMask = Array.from({ length: columnCount }, (_, col) => Number(input.unstableMask[col]) > 0);
  const continuityBreakMask = Array.from({ length: columnCount }, (_, col) => {
    if (unstableMask[col]) {
      return true;
    }
    if (col === 0) {
      return false;
    }
    return jumpsMm[col] > continuityJumpThresholdMm;
  });
  const sampledColumns = buildSampleColumnIndices(columnCount).map(col => ({
    col,
    depthMm: roundTo(depths[col] ?? 0),
    confidence: roundTo(confidences[col] ?? 0),
    scoreGap: roundTo(scoreGaps[col] ?? 0),
    zoneHalfWidthMm: roundTo(zoneHalfWidths[col] ?? 0),
    ambiguous: ambiguousMask[col],
    continuityBreak: continuityBreakMask[col],
    unstable: unstableMask[col],
  }));

  const ambiguousCount = ambiguousMask.filter(Boolean).length;
  const unstableCount = unstableMask.filter(Boolean).length;
  const continuityBreakCount = continuityBreakMask.filter(Boolean).length;
  const lowConfidenceCount = confidences.filter(value => value < lowConfidenceThreshold).length;

  return {
    label: input.label,
    anchorRow: input.anchorRow,
    columnCount,
    ambiguousGapThreshold: roundTo(ambiguousGapThreshold),
    lowConfidenceThreshold: roundTo(lowConfidenceThreshold),
    continuityJumpThresholdMm,
    supportCenterDepthFirst8Mm: toRoundedArray(depths, 0, 8),
    supportCenterDepthLast8Mm: toRoundedArray(depths, Math.max(0, columnCount - 8), 8),
    localConfidenceFirst8: toRoundedArray(confidences, 0, 8),
    localConfidenceLast8: toRoundedArray(confidences, Math.max(0, columnCount - 8), 8),
    reconstructionZoneHalfWidthFirst8Mm: toRoundedArray(zoneHalfWidths, 0, 8),
    reconstructionZoneHalfWidthLast8Mm: toRoundedArray(zoneHalfWidths, Math.max(0, columnCount - 8), 8),
    sampledColumns,
    depthRangeMm: summarize(depths),
    localConfidence: summarize(confidences),
    scoreGap: summarize(scoreGaps),
    reconstructionZoneHalfWidthMm: summarize(zoneHalfWidths),
    pathJumpMm: summarize(jumpsMm),
    ambiguousColumnFraction: roundTo(columnCount > 0 ? ambiguousCount / columnCount : 0, 4),
    lowConfidenceFraction: roundTo(columnCount > 0 ? lowConfidenceCount / columnCount : 0, 4),
    continuityBreakFraction: roundTo(columnCount > 0 ? continuityBreakCount / columnCount : 0, 4),
    unstableColumnFraction: roundTo(columnCount > 0 ? unstableCount / columnCount : 0, 4),
    ambiguousRuns: buildRunDiagnostics(ambiguousMask, jumpsMm),
    continuityBreakRuns: buildRunDiagnostics(continuityBreakMask, jumpsMm),
    unstableRuns: buildRunDiagnostics(unstableMask, jumpsMm),
  };
}

export function buildPanoV2GeometryDiagnostics(params: {
  panoWidth: number;
  upperArch: PanoV2TrackInput;
  lowerArch: PanoV2TrackInput;
}): PanoV2GeometryDiagnostics {
  const upperArch = buildTrackDiagnostics(params.upperArch);
  const lowerArch = buildTrackDiagnostics(params.lowerArch);
  const interArchSeparationMm: number[] = [];
  let nonCrossingCount = 0;
  const columnCount = Math.min(
    params.panoWidth,
    params.upperArch.selectedDepthMm.length,
    params.lowerArch.selectedDepthMm.length
  );

  for (let col = 0; col < columnCount; col++) {
    const upperDepthMm = Number(params.upperArch.selectedDepthMm[col]);
    const lowerDepthMm = Number(params.lowerArch.selectedDepthMm[col]);
    const separationMm = Math.abs(lowerDepthMm - upperDepthMm);
    interArchSeparationMm.push(separationMm);
    if (lowerDepthMm >= upperDepthMm) {
      nonCrossingCount++;
    }
  }

  return {
    enabled: true,
    phase: 1,
    model: 'dual-arch-support-geometry',
    panoWidth: params.panoWidth,
    sampledColumnCount: upperArch.sampledColumns.length,
    upperArch,
    lowerArch,
    interArchSeparationMm: summarize(interArchSeparationMm),
    nonCrossingColumnFraction: roundTo(columnCount > 0 ? nonCrossingCount / columnCount : 0, 4),
    combinedAmbiguousColumnFraction: roundTo(
      Math.max(upperArch.ambiguousColumnFraction, lowerArch.ambiguousColumnFraction),
      4
    ),
    combinedContinuityBreakFraction: roundTo(
      Math.max(upperArch.continuityBreakFraction, lowerArch.continuityBreakFraction),
      4
    ),
  };
}
