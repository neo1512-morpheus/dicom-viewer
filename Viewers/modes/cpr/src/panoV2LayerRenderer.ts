import type { PanoV2NumericSummary, PanoV2TrackLabel } from './panoV2Geometry';
import type { PanoV2StraightenedVolume, PanoV2StraightenedVolumeBand } from './panoV2Volume';

export const PANO_V2_LAYER_COUNT_PER_ARCH = 3;

const LAYER_OFFSET_FACTORS = [-0.82, 0, 0.82] as const;
const REGION_POSTERIOR_CUTOFF = 0.22;
const REGION_ANTERIOR_MIN = 0.35;
const REGION_ANTERIOR_MAX = 0.65;
const REGION_RIGHT_POSTERIOR_MIN = 0.78;
const TOOTH_ROW_BAND_ROW_RADIUS = 18;
export const PANO_V2_MIP_SLAB_HALF_WIDTH_MM = 10;
const PANO_V2_MIP_DEPTH_STEP_MM = 0.3;
const PANO_V2_BYPASS_AIR_AS_MISSING_HU = -950;
const PANO_V2_GAP_ANALYSIS_DENSE_HU_THRESHOLD = 500;
const PANO_V2_GAP_ANALYSIS_SAMPLE_STRIDE_COLUMNS = 10;
const PANO_V2_CROSS_ARCH_RESCUE_LOW_SIGNAL_HU = 650;
const PANO_V2_CROSS_ARCH_RESCUE_DENSE_NEIGHBOR_HU = 1400;
const PANO_V2_CROSS_ARCH_RESCUE_NEIGHBOR_COUNT = 4;
const PANO_V2_CROSS_ARCH_RESCUE_MIN_HU = 1000;
const PANO_V2_CROSS_ARCH_RESCUE_MARGIN_HU = 500;
const PANO_V2_CROSS_ARCH_RESCUE_RATIO = 2.2;
const PANO_V2_BYPASS_COLUMN_DENSE_HU = 900;
const PANO_V2_BYPASS_COLUMN_STRONG_HU = 1400;
const PANO_V2_BYPASS_COLUMN_SUPPORT_BASE_HU = 250;
const PANO_V2_BYPASS_COLUMN_MIN_DENSE_ROWS = 2;
const PANO_V2_BYPASS_TARGET_SMOOTHING_RADIUS = 4;

interface PanoV2LayerProbeRow {
  localRow: number;
  panoRow: number;
  valueHu: number | null;
}

interface PanoV2LayerProbeColumn {
  col: number;
  representativeRows: PanoV2LayerProbeRow[];
}

export interface PanoV2LayerMetrics {
  finiteFraction: number;
  anatomySignalFraction: number;
  valueRangeHu: PanoV2NumericSummary;
  detailMean: number;
  sharpnessPeak: number;
  toothContrastHu: number;
  incisorDetail: number;
  leftMolarDetail: number;
  rightMolarDetail: number;
}

export interface PanoV2LayerCandidateDiagnostics {
  layerId: string;
  band: PanoV2TrackLabel;
  targetOffsetMm: number;
  nearestDepthOffsetMm: number;
  nearestDepthIndex: number;
  tradeoffLabel: 'incisor-biased' | 'posterior-biased' | 'balanced';
  metrics: PanoV2LayerMetrics;
  toothRowBand: {
    localRowStart: number;
    localRowEnd: number;
    panoRowStart: number;
    panoRowEnd: number;
    sampledVoxelCount: number;
    medianHu: number;
    fractionAbove700Hu: number;
    centerDepthOffsetMm: number;
    requestedCenterDepthOffsetMm: number;
    exceeds1500Hu: boolean;
  };
  sampledColumns: PanoV2LayerProbeColumn[];
}

export interface PanoV2RenderedLayer {
  diagnostic: PanoV2LayerCandidateDiagnostics;
  image: Float32Array;
}

export interface PanoV2BandLayerDiagnostics {
  label: PanoV2TrackLabel;
  layerCount: number;
  layers: PanoV2LayerCandidateDiagnostics[];
  bestByRegion: {
    incisors: string | null;
    leftMolar: string | null;
    rightMolar: string | null;
    sharpnessPeak: string | null;
  };
}

export interface PanoV2BandLayerRenderResult {
  label: PanoV2TrackLabel;
  anchorRow: number;
  localAnchorRow: number;
  rowStart: number;
  rowEnd: number;
  rowCount: number;
  layers: PanoV2RenderedLayer[];
  bestByRegion: {
    incisors: string | null;
    leftMolar: string | null;
    rightMolar: string | null;
    sharpnessPeak: string | null;
  };
}

export interface PanoV2LayerRenderResult {
  enabled: true;
  phase: 3;
  model: 'thick-slab-mip-render';
  layerCountPerArch: number;
  mip: {
    slabHalfWidthMm: number;
    depthStepMm: number;
    sampledColumnCount: number;
    mipVoiOverride: true;
    layerRowConstraintsApplied: true;
    verticalHalfMmUsed: number;
    verticalCenterOffsetMm: number;
    slabDirectionColumn: number;
    slabDirectionVector: [number, number, number];
    toothBandMedianHu: number;
    toothBandMaxHu: number;
    totalRenderMs: number;
  };
  upperArch: PanoV2BandLayerRenderResult;
  lowerArch: PanoV2BandLayerRenderResult;
}

export interface PanoV2LayerRenderDiagnostics {
  enabled: true;
  phase: 3;
  model: 'thick-slab-mip-render';
  layerCountPerArch: number;
  mip: {
    slabHalfWidthMm: number;
    depthStepMm: number;
    sampledColumnCount: number;
    mipVoiOverride: true;
    layerRowConstraintsApplied: true;
    verticalHalfMmUsed: number;
    verticalCenterOffsetMm: number;
    slabDirectionColumn: number;
    slabDirectionVector: [number, number, number];
    toothBandMedianHu: number;
    toothBandMaxHu: number;
    totalRenderMs: number;
  };
  upperArch: PanoV2BandLayerDiagnostics;
  lowerArch: PanoV2BandLayerDiagnostics;
}

export interface PanoV2FullPlaneLayerImages {
  upperArch: Float32Array[];
  lowerArch: Float32Array[];
}

export interface PanoV2BypassCompositeImages {
  upperArch: Float32Array;
  lowerArch: Float32Array;
  upperTargetOffsetMmByCol: Float32Array;
  lowerTargetOffsetMmByCol: Float32Array;
  upperTargetOffsetMmMap: Float32Array;
  lowerTargetOffsetMmMap: Float32Array;
}

export interface PanoV2GapBoundarySample {
  col: number;
  upperLastDenseRow: number | null;
  lowerFirstDenseRow: number | null;
  gapHeightPx: number;
}

export interface PanoV2GapBoundaryAnalysis {
  denseHuThreshold: number;
  sampleStrideColumns: number;
  upperLastDenseRowByCol: Int16Array;
  lowerFirstDenseRowByCol: Int16Array;
  sampledColumns: PanoV2GapBoundarySample[];
}

export interface PanoV2MipRaySample {
  localDepthOffsetMm: number;
  sampleHu: number | null;
}

export interface PanoV2DarkPocketPoint {
  col: number;
  row: number;
  finalHu: number;
  upperBypassHu: number | null;
  lowerBypassHu: number | null;
  selectedSource: 'upper-arch-bypass' | 'lower-arch-bypass';
  surroundingMedianHu: number;
  surroundingMaxHu: number;
  darknessScore: number;
  upperRayWinnerHu: number | null;
  upperRayWinnerDepthOffsetMm: number | null;
  lowerRayWinnerHu: number | null;
  lowerRayWinnerDepthOffsetMm: number | null;
  upperRaySamples: PanoV2MipRaySample[];
  lowerRaySamples: PanoV2MipRaySample[];
}

export interface PanoV2DarkPocketDiagnostics {
  searchRegion: {
    colStart: number;
    colEnd: number;
    rowStart: number;
    rowEnd: number;
    gapCenterRow: number;
    upperAnchorRow: number;
    topBorderExclusionRows: number;
    denseBandCenterRow: number | null;
    denseRowThresholdHu: number;
  };
  candidateCountBeforeSuppression: number;
  darkestPockets: PanoV2DarkPocketPoint[];
}

export interface PanoV2GeometryRayDebugPoint {
  col: number;
  row: number;
  arch: PanoV2TrackLabel;
  splineCenterWorld: [number, number, number];
  rowBaseWorld: [number, number, number];
  rayStartWorld: [number, number, number];
  rayMidWorld: [number, number, number];
  rayEndWorld: [number, number, number];
  slabDir: [number, number, number];
  verticalDir: [number, number, number];
  depthHalfRangeMm: number;
  mipSlabHalfWidthMm: number;
  centerDepthMm: number;
  targetOffsetMm: number;
  rayMidDepthMm: number;
  rayStartDepthMm: number;
  rayEndDepthMm: number;
  columnVerticalCenterOffsetMm: number;
  rowOffsetMm: number;
  upperBypassHu: number | null;
  lowerBypassHu: number | null;
  rayWinnerHu: number | null;
  rayWinnerDepthOffsetMm: number | null;
}

export interface PanoV2GeometryRayComparisonDiagnostics {
  row: number;
  gapCenterRow: number;
  arch: PanoV2TrackLabel;
  failing: PanoV2GeometryRayDebugPoint;
  reference: PanoV2GeometryRayDebugPoint;
  comparison: {
    slabDirDot: number;
    slabDirAngleDeg: number;
    verticalDirDot: number;
    verticalDirAngleDeg: number;
    splineCenterDistanceMm: number;
    rowBaseDistanceMm: number;
    rayStartDistanceMm: number;
    rayMidDistanceMm: number;
    rayEndDistanceMm: number;
    centerDepthDeltaMm: number;
    targetOffsetDeltaMm: number;
    rayMidDepthDeltaMm: number;
    columnVerticalCenterOffsetDeltaMm: number;
    rowOffsetDeltaMm: number;
    failingWinnerNearSlabEdge: boolean;
    referenceWinnerNearSlabEdge: boolean;
    anomalyKind:
      | 'vector-flip-or-rotation'
      | 'vertical-shift-discontinuity'
      | 'depth-center-jump'
      | 'insufficient-depth-window'
      | 'smooth-geometry-wrong-depth'
      | 'no-clear-geometry-anomaly';
    anomalyReason: string;
  };
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

function roundVector3(value: [number, number, number]): [number, number, number] {
  return [roundTo(value[0]), roundTo(value[1]), roundTo(value[2])];
}

function dot3(
  left: [number, number, number],
  right: [number, number, number]
): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function length3(value: [number, number, number]): number {
  return Math.sqrt(dot3(value, value));
}

function distance3(
  left: [number, number, number],
  right: [number, number, number]
): number {
  const dx = left[0] - right[0];
  const dy = left[1] - right[1];
  const dz = left[2] - right[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function angleBetweenUnitLikeVectorsDeg(
  left: [number, number, number],
  right: [number, number, number]
): number {
  const leftLength = length3(left);
  const rightLength = length3(right);
  if (!(leftLength > 1e-6 && rightLength > 1e-6)) {
    return 0;
  }
  const cosine = clampNumber(dot3(left, right) / (leftLength * rightLength), -1, 1);
  return roundTo((Math.acos(cosine) * 180) / Math.PI);
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
    const fraction = sampleIndex / denominator;
    indices.add(Math.round(fraction * (length - 1)));
  }
  return Array.from(indices).sort((left, right) => left - right);
}

function bandValueIndex(
  band: PanoV2StraightenedVolumeBand,
  col: number,
  rowIndex: number,
  depthIndex: number
): number {
  return ((col * band.rowCount + rowIndex) * band.depthCount + depthIndex) | 0;
}

function sampleBandValue(
  band: PanoV2StraightenedVolumeBand,
  col: number,
  rowIndex: number,
  depthIndex: number
): number {
  return Number(band.valuesHu[bandValueIndex(band, col, rowIndex, depthIndex)]);
}

function resolveNearestDepthIndex(depthOffsetsMm: Float32Array, targetDepthMm: number): number {
  if (depthOffsetsMm.length <= 0) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < depthOffsetsMm.length; index++) {
    const distance = Math.abs(Number(depthOffsetsMm[index] ?? 0) - targetDepthMm);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function sampleBandDepthSignal(
  band: PanoV2StraightenedVolumeBand,
  col: number,
  rowIndex: number,
  depthIndex: number
): number {
  let weightedSum = 0;
  let totalWeight = 0;
  for (let sampleRow = rowIndex - 1; sampleRow <= rowIndex + 1; sampleRow++) {
    if (sampleRow < 0 || sampleRow >= band.rowCount) {
      continue;
    }
    const weight = sampleRow === rowIndex ? 1 : 0.6;
    const value = sampleBandValue(band, col, sampleRow, depthIndex);
    if (!Number.isFinite(value)) {
      continue;
    }
    weightedSum += value * weight;
    totalWeight += weight;
  }
  if (!(totalWeight > 0)) {
    return Number.NEGATIVE_INFINITY;
  }
  return weightedSum / totalWeight;
}

function buildRowAdaptiveTargetOffsetPlan(params: {
  band: PanoV2StraightenedVolumeBand;
  panoWidth: number;
  panoHeight: number;
  assignedRowStartInclusive: number;
  assignedRowEndExclusive: number;
  columnTargetOffsetMmByCol: Float32Array;
}): {
  targetOffsetMmByCol: Float32Array;
  targetOffsetMmMap: Float32Array;
} {
  const {
    band,
    panoWidth,
    panoHeight,
    assignedRowStartInclusive,
    assignedRowEndExclusive,
    columnTargetOffsetMmByCol,
  } = params;
  const width = Math.max(1, panoWidth);
  const localSize = Math.max(1, width * band.rowCount);
  const localOffsetMm = new Float32Array(localSize);
  const localSignalHu = new Float32Array(localSize);
  localOffsetMm.fill(0);
  localSignalHu.fill(Number.NEGATIVE_INFINITY);

  const anchorRowIndex = clampNumber(band.localAnchorRow, 0, Math.max(0, band.rowCount - 1));
  const continuityPenaltyPerMm = Math.max(
    40,
    (band.valueRangeHu.p90 - band.valueRangeHu.p50) * 0.08
  );
  const anchorPenaltyPerMm = Math.max(28, continuityPenaltyPerMm * 0.5);

  for (let col = 0; col < width; col++) {
    const regionalTargetOffsetMm = Number(columnTargetOffsetMmByCol[col] ?? 0);
    const regionalDepthIndex = resolveNearestDepthIndex(band.depthOffsetsMm, regionalTargetOffsetMm);
    let bestAnchorDepthIndex = regionalDepthIndex;
    let bestAnchorScore = Number.NEGATIVE_INFINITY;

    for (let depthIndex = 0; depthIndex < band.depthCount; depthIndex++) {
      const signalHu = sampleBandDepthSignal(band, col, anchorRowIndex, depthIndex);
      if (!Number.isFinite(signalHu)) {
        continue;
      }
      const depthOffsetMm = Number(band.depthOffsetsMm[depthIndex] ?? 0);
      const candidateScore =
        signalHu - Math.abs(depthOffsetMm - regionalTargetOffsetMm) * anchorPenaltyPerMm;
      if (candidateScore > bestAnchorScore) {
        bestAnchorScore = candidateScore;
        bestAnchorDepthIndex = depthIndex;
      }
    }

    const writeLocalSelection = (rowIndex: number, depthIndex: number): void => {
      const localIndex = rowIndex * width + col;
      localOffsetMm[localIndex] = Number(band.depthOffsetsMm[depthIndex] ?? 0);
      localSignalHu[localIndex] = sampleBandDepthSignal(band, col, rowIndex, depthIndex);
    };

    writeLocalSelection(anchorRowIndex, bestAnchorDepthIndex);

    for (let rowIndex = anchorRowIndex - 1; rowIndex >= 0; rowIndex--) {
      const previousDepthOffsetMm = Number(localOffsetMm[(rowIndex + 1) * width + col] ?? 0);
      let bestDepthIndex = bestAnchorDepthIndex;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (let depthIndex = 0; depthIndex < band.depthCount; depthIndex++) {
        const signalHu = sampleBandDepthSignal(band, col, rowIndex, depthIndex);
        if (!Number.isFinite(signalHu)) {
          continue;
        }
        const depthOffsetMm = Number(band.depthOffsetsMm[depthIndex] ?? 0);
        const continuityPenalty =
          Math.abs(depthOffsetMm - previousDepthOffsetMm) * continuityPenaltyPerMm;
        const regionalPenalty =
          Math.abs(depthOffsetMm - regionalTargetOffsetMm) * (anchorPenaltyPerMm * 0.25);
        const candidateScore = signalHu - continuityPenalty - regionalPenalty;
        if (candidateScore > bestScore) {
          bestScore = candidateScore;
          bestDepthIndex = depthIndex;
        }
      }
      writeLocalSelection(rowIndex, bestDepthIndex);
    }

    for (let rowIndex = anchorRowIndex + 1; rowIndex < band.rowCount; rowIndex++) {
      const previousDepthOffsetMm = Number(localOffsetMm[(rowIndex - 1) * width + col] ?? 0);
      let bestDepthIndex = bestAnchorDepthIndex;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (let depthIndex = 0; depthIndex < band.depthCount; depthIndex++) {
        const signalHu = sampleBandDepthSignal(band, col, rowIndex, depthIndex);
        if (!Number.isFinite(signalHu)) {
          continue;
        }
        const depthOffsetMm = Number(band.depthOffsetsMm[depthIndex] ?? 0);
        const continuityPenalty =
          Math.abs(depthOffsetMm - previousDepthOffsetMm) * continuityPenaltyPerMm;
        const regionalPenalty =
          Math.abs(depthOffsetMm - regionalTargetOffsetMm) * (anchorPenaltyPerMm * 0.25);
        const candidateScore = signalHu - continuityPenalty - regionalPenalty;
        if (candidateScore > bestScore) {
          bestScore = candidateScore;
          bestDepthIndex = depthIndex;
        }
      }
      writeLocalSelection(rowIndex, bestDepthIndex);
    }
  }

  const horizontallySmoothedLocalOffsetMm = new Float32Array(localSize);
  for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
    const rowSignals: number[] = [];
    for (let col = 0; col < width; col++) {
      const signalHu = Number(localSignalHu[rowIndex * width + col]);
      if (Number.isFinite(signalHu)) {
        rowSignals.push(signalHu);
      }
    }
    const rowSignalMedianHu = rowSignals.length > 0 ? percentile(rowSignals, 0.5) : 0;
    const rowSignalP90Hu = rowSignals.length > 0 ? percentile(rowSignals, 0.9) : rowSignalMedianHu;
    const rowSignalScaleHu = Math.max(1, rowSignalP90Hu - rowSignalMedianHu);

    for (let col = 0; col < width; col++) {
      let weightedOffsetSum = 0;
      let totalWeight = 0;
      for (let sampleCol = Math.max(0, col - 3); sampleCol <= Math.min(width - 1, col + 3); sampleCol++) {
        const distance = Math.abs(sampleCol - col);
        const baseWeight = distance === 0 ? 4 : distance === 1 ? 3 : distance === 2 ? 2 : 1;
        const signalHu = Number(localSignalHu[rowIndex * width + sampleCol]);
        const signalWeight =
          Number.isFinite(signalHu) && signalHu > rowSignalMedianHu
            ? 1 + (signalHu - rowSignalMedianHu) / rowSignalScaleHu
            : 0.6;
        const centerWeightBoost = sampleCol === col ? 1.75 : 1;
        const weight = baseWeight * signalWeight * centerWeightBoost;
        weightedOffsetSum += Number(localOffsetMm[rowIndex * width + sampleCol]) * weight;
        totalWeight += weight;
      }
      horizontallySmoothedLocalOffsetMm[rowIndex * width + col] =
        totalWeight > 0
          ? weightedOffsetSum / totalWeight
          : Number(localOffsetMm[rowIndex * width + col]);
    }
  }

  const regularizedLocalOffsetMm = new Float32Array(localSize);
  const depthStepMm =
    band.depthCount > 1
      ? Math.abs(Number(band.depthOffsetsMm[1] ?? 0) - Number(band.depthOffsetsMm[0] ?? 0))
      : Math.max(0.25, band.depthHalfRangeMm / 12);
  const baseMaxColumnDeltaMm = Math.max(
    depthStepMm * 3.5,
    band.depthHalfRangeMm / Math.max(14, width * 0.06)
  );

  for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
    const rowSignals: number[] = [];
    for (let col = 0; col < width; col++) {
      const signalHu = Number(localSignalHu[rowIndex * width + col]);
      if (Number.isFinite(signalHu)) {
        rowSignals.push(signalHu);
      }
    }
    const rowSignalMedianHu = rowSignals.length > 0 ? percentile(rowSignals, 0.5) : 0;
    const rowSignalP90Hu = rowSignals.length > 0 ? percentile(rowSignals, 0.9) : rowSignalMedianHu;
    const rowSignalScaleHu = Math.max(1, rowSignalP90Hu - rowSignalMedianHu);
    const forward = new Float32Array(width);
    const backward = new Float32Array(width);

    forward[0] = Number(horizontallySmoothedLocalOffsetMm[rowIndex * width]);
    for (let col = 1; col < width; col++) {
      const index = rowIndex * width + col;
      const desiredOffsetMm = Number(horizontallySmoothedLocalOffsetMm[index]);
      const signalHu = Number(localSignalHu[index]);
      const signalBoost =
        Number.isFinite(signalHu) && signalHu > rowSignalMedianHu
          ? Math.min(1.25, (signalHu - rowSignalMedianHu) / rowSignalScaleHu)
          : 0;
      const maxColumnDeltaMm = baseMaxColumnDeltaMm * (1 + signalBoost * 0.7);
      const previousOffsetMm = Number(forward[col - 1]);
      forward[col] = clampNumber(
        desiredOffsetMm,
        previousOffsetMm - maxColumnDeltaMm,
        previousOffsetMm + maxColumnDeltaMm
      );
    }

    backward[width - 1] = Number(horizontallySmoothedLocalOffsetMm[rowIndex * width + width - 1]);
    for (let col = width - 2; col >= 0; col--) {
      const index = rowIndex * width + col;
      const desiredOffsetMm = Number(horizontallySmoothedLocalOffsetMm[index]);
      const signalHu = Number(localSignalHu[index]);
      const signalBoost =
        Number.isFinite(signalHu) && signalHu > rowSignalMedianHu
          ? Math.min(1.25, (signalHu - rowSignalMedianHu) / rowSignalScaleHu)
          : 0;
      const maxColumnDeltaMm = baseMaxColumnDeltaMm * (1 + signalBoost * 0.7);
      const nextOffsetMm = Number(backward[col + 1]);
      backward[col] = clampNumber(
        desiredOffsetMm,
        nextOffsetMm - maxColumnDeltaMm,
        nextOffsetMm + maxColumnDeltaMm
      );
    }

    for (let col = 0; col < width; col++) {
      regularizedLocalOffsetMm[rowIndex * width + col] =
        (Number(forward[col]) + Number(backward[col])) * 0.5;
    }
  }

  const finalLocalOffsetMm = new Float32Array(localSize);
  for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
    for (let col = 0; col < width; col++) {
      let weightedOffsetSum = Number(regularizedLocalOffsetMm[rowIndex * width + col]) * 1.7;
      let totalWeight = 1.7;
      if (rowIndex > 0) {
        weightedOffsetSum += Number(regularizedLocalOffsetMm[(rowIndex - 1) * width + col]) * 0.45;
        totalWeight += 0.45;
      }
      if (rowIndex + 1 < band.rowCount) {
        weightedOffsetSum += Number(regularizedLocalOffsetMm[(rowIndex + 1) * width + col]) * 0.45;
        totalWeight += 0.45;
      }
      finalLocalOffsetMm[rowIndex * width + col] =
        totalWeight > 0
          ? weightedOffsetSum / totalWeight
          : Number(regularizedLocalOffsetMm[rowIndex * width + col]);
    }
  }

  const planeSize = Math.max(1, panoWidth * panoHeight);
  const targetOffsetMmMap = new Float32Array(planeSize);
  targetOffsetMmMap.fill(Number.NaN);
  const representativeTargetOffsetMmByCol = new Float32Array(width);
  const representativeLocalRow = clampNumber(anchorRowIndex, 0, Math.max(0, band.rowCount - 1));

  for (let col = 0; col < width; col++) {
    representativeTargetOffsetMmByCol[col] = Number(
      finalLocalOffsetMm[representativeLocalRow * width + col] ??
        columnTargetOffsetMmByCol[col] ??
        0
    );
    for (
      let row = Math.max(0, assignedRowStartInclusive);
      row < Math.min(panoHeight, assignedRowEndExclusive);
      row++
    ) {
      // Extend the nearest band-edge offset beyond the sampled support rows so the
      // bypass renderer does not switch depth-selection modes at the band boundary.
      const localRow = clampNumber(row - band.rowStart, 0, Math.max(0, band.rowCount - 1));
      const localIndex = localRow * width + col;
      const planeIndex = row * width + col;
      targetOffsetMmMap[planeIndex] = Number(finalLocalOffsetMm[localIndex]);
    }
  }

  return {
    targetOffsetMmByCol: representativeTargetOffsetMmByCol,
    targetOffsetMmMap,
  };
}

function buildMipBandImage(
  volume: PanoV2StraightenedVolume,
  band: PanoV2StraightenedVolumeBand,
  targetOffsetMm: number
): {
  image: Float32Array;
  toothBandValuesHu: number[];
  renderMs: number;
} {
  const width = band.centerDepthMmByCol.length;
  const image = new Float32Array(Math.max(1, width * band.rowCount));
  image.fill(Number.NaN);
  const toothBandValuesHu: number[] = [];
  const renderStartMs = performance.now();

  for (let col = 0; col < width; col++) {
    const frame = volume.frames[col];
    if (!frame) {
      continue;
    }
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = frame.mipSlabDir ?? frame.slabDir;
    const [vertDirX, vertDirY, vertDirZ] = frame.verticalDir;
    const centerDepthMm = Number(band.centerDepthMmByCol[col] ?? 0);
    const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);

    for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
      const verticalOffsetMm = columnVerticalCenterOffsetMm + Number(band.rowOffsetsMm[rowIndex] ?? 0);
      const bx = px + verticalOffsetMm * vertDirX;
      const by = py + verticalOffsetMm * vertDirY;
      const bz = pz + verticalOffsetMm * vertDirZ;
      let mipHu = Number.NEGATIVE_INFINITY;
      for (
        let localDepthOffsetMm = -PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
        localDepthOffsetMm <= PANO_V2_MIP_SLAB_HALF_WIDTH_MM + 1e-6;
        localDepthOffsetMm += PANO_V2_MIP_DEPTH_STEP_MM
      ) {
        const localDepthMm = centerDepthMm + targetOffsetMm + localDepthOffsetMm;
        const sample = volume.sampleWorldIntensity(
          bx + localDepthMm * slabDirX,
          by + localDepthMm * slabDirY,
          bz + localDepthMm * slabDirZ
        );
        if (Number.isFinite(sample) && sample > mipHu) {
          mipHu = sample;
        }
      }
      if (Number.isFinite(mipHu)) {
        image[rowIndex * width + col] = mipHu;
        if (
          rowIndex >= Math.max(0, band.localAnchorRow - TOOTH_ROW_BAND_ROW_RADIUS) &&
          rowIndex <= Math.min(band.rowCount - 1, band.localAnchorRow + TOOTH_ROW_BAND_ROW_RADIUS)
        ) {
          toothBandValuesHu.push(mipHu);
        }
      }
    }
  }

  return {
    image,
    toothBandValuesHu,
    renderMs: roundTo(performance.now() - renderStartMs),
  };
}

function buildFullPlaneMipImage(params: {
  volume: PanoV2StraightenedVolume;
  centerDepthMmByCol: Float32Array;
  targetOffsetMmByCol: Float32Array;
  targetOffsetMmMap?: Float32Array | null;
  rowStartInclusive: number;
  rowEndExclusive: number;
  outsideAssignedRangeHu?: number;
}): Float32Array {
  const {
    volume,
    centerDepthMmByCol,
    targetOffsetMmByCol,
    targetOffsetMmMap,
    rowStartInclusive,
    rowEndExclusive,
    outsideAssignedRangeHu = 150,
  } = params;
  const width = volume.panoWidth;
  const height = volume.panoHeight;
  const image = new Float32Array(Math.max(1, width * height));
  image.fill(Number.NaN);
  const safeRowStartInclusive = clampNumber(Math.floor(rowStartInclusive), 0, Math.max(0, height));
  const safeRowEndExclusive = clampNumber(Math.ceil(rowEndExclusive), 0, Math.max(0, height));

  for (let col = 0; col < width; col++) {
    const frame = volume.frames[col];
    if (!frame) {
      continue;
    }
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = frame.mipSlabDir ?? frame.slabDir;
    const [vertDirX, vertDirY, vertDirZ] = frame.verticalDir;
    const centerDepthMm = Number(centerDepthMmByCol[col] ?? 0);
    const targetOffsetMm = Number(targetOffsetMmByCol[col] ?? 0);
    const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);

    for (let row = 0; row < height; row++) {
      if (row < safeRowStartInclusive || row >= safeRowEndExclusive) {
        image[row * width + col] = outsideAssignedRangeHu;
        continue;
      }
      const planeIndex = row * width + col;
      const rowOffsetMm = volume.effectiveVerticalHalfMm - row * volume.vertStepMm;
      const verticalOffsetMm = columnVerticalCenterOffsetMm + rowOffsetMm;
      const bx = px + verticalOffsetMm * vertDirX;
      const by = py + verticalOffsetMm * vertDirY;
      const bz = pz + verticalOffsetMm * vertDirZ;
      const rowTargetOffsetMm = Number(targetOffsetMmMap?.[planeIndex]);
      const targetOffsetMmForPixel = Number.isFinite(rowTargetOffsetMm)
        ? rowTargetOffsetMm
        : targetOffsetMm;
      let mipHu = Number.NEGATIVE_INFINITY;
      for (
        let localDepthOffsetMm = -PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
        localDepthOffsetMm <= PANO_V2_MIP_SLAB_HALF_WIDTH_MM + 1e-6;
        localDepthOffsetMm += PANO_V2_MIP_DEPTH_STEP_MM
      ) {
        const localDepthMm = centerDepthMm + targetOffsetMmForPixel + localDepthOffsetMm;
        const sample = volume.sampleWorldIntensity(
          bx + localDepthMm * slabDirX,
          by + localDepthMm * slabDirY,
          bz + localDepthMm * slabDirZ
        );
        if (Number.isFinite(sample) && sample > mipHu) {
          mipHu = sample;
        }
      }
      if (Number.isFinite(mipHu) && mipHu > PANO_V2_BYPASS_AIR_AS_MISSING_HU) {
        image[row * width + col] = mipHu;
      }
    }
  }

  return image;
}

function samplePanoV2MipWinner(params: {
  volume: PanoV2StraightenedVolume;
  centerDepthMmByCol: Float32Array;
  targetOffsetMmByCol: Float32Array;
  targetOffsetMmMap?: Float32Array | null;
  col: number;
  row: number;
}): {
  winnerHu: number | null;
  winnerDepthOffsetMm: number | null;
} {
  const { volume, centerDepthMmByCol, targetOffsetMmByCol, targetOffsetMmMap, col, row } = params;
  const frame = volume.frames[col];
  if (!frame || row < 0 || row >= volume.panoHeight) {
    return {
      winnerHu: null,
      winnerDepthOffsetMm: null,
    };
  }
  const [px, py, pz] = frame.position;
  const [slabDirX, slabDirY, slabDirZ] = frame.mipSlabDir ?? frame.slabDir;
  const [vertDirX, vertDirY, vertDirZ] = frame.verticalDir;
  const centerDepthMm = Number(centerDepthMmByCol[col] ?? 0);
  const rowTargetOffsetMm = Number(targetOffsetMmMap?.[row * volume.panoWidth + col]);
  const targetOffsetMm = Number.isFinite(rowTargetOffsetMm)
    ? rowTargetOffsetMm
    : Number(targetOffsetMmByCol[col] ?? 0);
  const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);
  const rowOffsetMm = volume.effectiveVerticalHalfMm - row * volume.vertStepMm;
  const verticalOffsetMm = columnVerticalCenterOffsetMm + rowOffsetMm;
  const bx = px + verticalOffsetMm * vertDirX;
  const by = py + verticalOffsetMm * vertDirY;
  const bz = pz + verticalOffsetMm * vertDirZ;
  let winnerHu = Number.NEGATIVE_INFINITY;
  let winnerDepthOffsetMm = Number.NaN;

  for (
    let localDepthOffsetMm = -PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
    localDepthOffsetMm <= PANO_V2_MIP_SLAB_HALF_WIDTH_MM + 1e-6;
    localDepthOffsetMm += PANO_V2_MIP_DEPTH_STEP_MM
  ) {
    const localDepthMm = centerDepthMm + targetOffsetMm + localDepthOffsetMm;
    const sample = volume.sampleWorldIntensity(
      bx + localDepthMm * slabDirX,
      by + localDepthMm * slabDirY,
      bz + localDepthMm * slabDirZ
    );
    if (Number.isFinite(sample) && sample > winnerHu) {
      winnerHu = sample;
      winnerDepthOffsetMm = localDepthOffsetMm;
    }
  }

  return {
    winnerHu: Number.isFinite(winnerHu) ? roundTo(winnerHu) : null,
    winnerDepthOffsetMm: Number.isFinite(winnerDepthOffsetMm) ? roundTo(winnerDepthOffsetMm) : null,
  };
}

function samplePanoV2MipRay(params: {
  volume: PanoV2StraightenedVolume;
  centerDepthMmByCol: Float32Array;
  targetOffsetMmByCol: Float32Array;
  targetOffsetMmMap?: Float32Array | null;
  col: number;
  row: number;
}): {
  winnerHu: number | null;
  winnerDepthOffsetMm: number | null;
  samples: PanoV2MipRaySample[];
} {
  const { volume, centerDepthMmByCol, targetOffsetMmByCol, targetOffsetMmMap, col, row } = params;
  const frame = volume.frames[col];
  if (!frame || row < 0 || row >= volume.panoHeight) {
    return {
      winnerHu: null,
      winnerDepthOffsetMm: null,
      samples: [],
    };
  }
  const [px, py, pz] = frame.position;
  const [slabDirX, slabDirY, slabDirZ] = frame.mipSlabDir ?? frame.slabDir;
  const [vertDirX, vertDirY, vertDirZ] = frame.verticalDir;
  const centerDepthMm = Number(centerDepthMmByCol[col] ?? 0);
  const rowTargetOffsetMm = Number(targetOffsetMmMap?.[row * volume.panoWidth + col]);
  const targetOffsetMm = Number.isFinite(rowTargetOffsetMm)
    ? rowTargetOffsetMm
    : Number(targetOffsetMmByCol[col] ?? 0);
  const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);
  const rowOffsetMm = volume.effectiveVerticalHalfMm - row * volume.vertStepMm;
  const verticalOffsetMm = columnVerticalCenterOffsetMm + rowOffsetMm;
  const bx = px + verticalOffsetMm * vertDirX;
  const by = py + verticalOffsetMm * vertDirY;
  const bz = pz + verticalOffsetMm * vertDirZ;
  const samples: PanoV2MipRaySample[] = [];
  let winnerHu = Number.NEGATIVE_INFINITY;
  let winnerDepthOffsetMm = Number.NaN;

  for (
    let localDepthOffsetMm = -PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
    localDepthOffsetMm <= PANO_V2_MIP_SLAB_HALF_WIDTH_MM + 1e-6;
    localDepthOffsetMm += PANO_V2_MIP_DEPTH_STEP_MM
  ) {
    const localDepthMm = centerDepthMm + targetOffsetMm + localDepthOffsetMm;
    const sample = volume.sampleWorldIntensity(
      bx + localDepthMm * slabDirX,
      by + localDepthMm * slabDirY,
      bz + localDepthMm * slabDirZ
    );
    const finiteSample = Number.isFinite(sample) ? Number(sample) : Number.NaN;
    samples.push({
      localDepthOffsetMm: roundTo(localDepthOffsetMm),
      sampleHu: Number.isFinite(finiteSample) ? roundTo(finiteSample) : null,
    });
    if (Number.isFinite(finiteSample) && finiteSample > winnerHu) {
      winnerHu = finiteSample;
      winnerDepthOffsetMm = localDepthOffsetMm;
    }
  }

  return {
    winnerHu: Number.isFinite(winnerHu) ? roundTo(winnerHu) : null,
    winnerDepthOffsetMm: Number.isFinite(winnerDepthOffsetMm) ? roundTo(winnerDepthOffsetMm) : null,
    samples,
  };
}

function hasDenseAssignedNeighborhood(params: {
  image: Float32Array;
  width: number;
  height: number;
  row: number;
  col: number;
  rowStartInclusive: number;
  rowEndExclusive: number;
}): boolean {
  const { image, width, height, row, col, rowStartInclusive, rowEndExclusive } = params;
  let denseNeighborCount = 0;
  for (let sampleRow = row - 2; sampleRow <= row + 2; sampleRow++) {
    if (sampleRow < rowStartInclusive || sampleRow >= rowEndExclusive || sampleRow < 0 || sampleRow >= height) {
      continue;
    }
    for (let sampleCol = col - 2; sampleCol <= col + 2; sampleCol++) {
      if (sampleCol < 0 || sampleCol >= width || (sampleCol === col && sampleRow === row)) {
        continue;
      }
      const sampleValue = Number(image[sampleRow * width + sampleCol]);
      if (Number.isFinite(sampleValue) && sampleValue >= PANO_V2_CROSS_ARCH_RESCUE_DENSE_NEIGHBOR_HU) {
        denseNeighborCount++;
        if (denseNeighborCount >= PANO_V2_CROSS_ARCH_RESCUE_NEIGHBOR_COUNT) {
          return true;
        }
      }
    }
  }
  return false;
}

function applyCrossArchBypassRescue(params: {
  volume: PanoV2StraightenedVolume;
  targetImage: Float32Array;
  oppositeCenterDepthMmByCol: Float32Array;
  oppositeTargetOffsetMmByCol: Float32Array;
  oppositeTargetOffsetMmMap?: Float32Array | null;
  rowStartInclusive: number;
  rowEndExclusive: number;
}): void {
  const {
    volume,
    targetImage,
    oppositeCenterDepthMmByCol,
    oppositeTargetOffsetMmByCol,
    oppositeTargetOffsetMmMap,
    rowStartInclusive,
    rowEndExclusive,
  } = params;
  const width = volume.panoWidth;
  const height = volume.panoHeight;
  const safeRowStartInclusive = clampNumber(Math.floor(rowStartInclusive), 0, Math.max(0, height));
  const safeRowEndExclusive = clampNumber(Math.ceil(rowEndExclusive), safeRowStartInclusive, Math.max(0, height));

  for (let row = safeRowStartInclusive; row < safeRowEndExclusive; row++) {
    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      const currentValue = Number(targetImage[index]);
      if (Number.isFinite(currentValue) && currentValue >= PANO_V2_CROSS_ARCH_RESCUE_LOW_SIGNAL_HU) {
        continue;
      }
      if (
        !hasDenseAssignedNeighborhood({
          image: targetImage,
          width,
          height,
          row,
          col,
          rowStartInclusive: safeRowStartInclusive,
          rowEndExclusive: safeRowEndExclusive,
        })
      ) {
        continue;
      }
      const oppositeWinner = samplePanoV2MipWinner({
        volume,
        centerDepthMmByCol: oppositeCenterDepthMmByCol,
        targetOffsetMmByCol: oppositeTargetOffsetMmByCol,
        targetOffsetMmMap: oppositeTargetOffsetMmMap,
        col,
        row,
      }).winnerHu;
      if (
        !Number.isFinite(oppositeWinner) ||
        oppositeWinner < PANO_V2_CROSS_ARCH_RESCUE_MIN_HU ||
        (Number.isFinite(currentValue) &&
          (oppositeWinner < currentValue + PANO_V2_CROSS_ARCH_RESCUE_MARGIN_HU ||
            oppositeWinner < currentValue * PANO_V2_CROSS_ARCH_RESCUE_RATIO))
      ) {
        continue;
      }
      targetImage[index] = oppositeWinner;
    }
  }
}

function buildPanoV2GeometryRayDebugPoint(params: {
  volume: PanoV2StraightenedVolume;
  band: PanoV2StraightenedVolumeBand;
  targetOffsetMmByCol: Float32Array;
  targetOffsetMmMap?: Float32Array | null;
  upperBypassImage: Float32Array;
  lowerBypassImage: Float32Array;
  col: number;
  row: number;
}): PanoV2GeometryRayDebugPoint | null {
  const {
    volume,
    band,
    targetOffsetMmByCol,
    targetOffsetMmMap,
    upperBypassImage,
    lowerBypassImage,
    col,
    row,
  } = params;
  const safeCol = clampNumber(Math.round(col), 0, Math.max(0, volume.panoWidth - 1));
  const safeRow = clampNumber(Math.round(row), 0, Math.max(0, volume.panoHeight - 1));
  const frame = volume.frames[safeCol];
  if (!frame) {
    return null;
  }

  const slabDir = (frame.mipSlabDir ?? frame.slabDir) as [number, number, number];
  const verticalDir = frame.verticalDir as [number, number, number];
  const splineCenterWorld = frame.position as [number, number, number];
  const centerDepthMm = Number(band.centerDepthMmByCol[safeCol] ?? 0);
  const rowTargetOffsetMm = Number(targetOffsetMmMap?.[safeRow * volume.panoWidth + safeCol]);
  const targetOffsetMm = Number.isFinite(rowTargetOffsetMm)
    ? rowTargetOffsetMm
    : Number(targetOffsetMmByCol[safeCol] ?? 0);
  const rayMidDepthMm = centerDepthMm + targetOffsetMm;
  const rayStartDepthMm = rayMidDepthMm - PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
  const rayEndDepthMm = rayMidDepthMm + PANO_V2_MIP_SLAB_HALF_WIDTH_MM;
  const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);
  const rowOffsetMm = volume.effectiveVerticalHalfMm - safeRow * volume.vertStepMm;
  const verticalOffsetMm = columnVerticalCenterOffsetMm + rowOffsetMm;
  const rowBaseWorld: [number, number, number] = [
    splineCenterWorld[0] + verticalOffsetMm * verticalDir[0],
    splineCenterWorld[1] + verticalOffsetMm * verticalDir[1],
    splineCenterWorld[2] + verticalOffsetMm * verticalDir[2],
  ];
  const rayStartWorld: [number, number, number] = [
    rowBaseWorld[0] + rayStartDepthMm * slabDir[0],
    rowBaseWorld[1] + rayStartDepthMm * slabDir[1],
    rowBaseWorld[2] + rayStartDepthMm * slabDir[2],
  ];
  const rayMidWorld: [number, number, number] = [
    rowBaseWorld[0] + rayMidDepthMm * slabDir[0],
    rowBaseWorld[1] + rayMidDepthMm * slabDir[1],
    rowBaseWorld[2] + rayMidDepthMm * slabDir[2],
  ];
  const rayEndWorld: [number, number, number] = [
    rowBaseWorld[0] + rayEndDepthMm * slabDir[0],
    rowBaseWorld[1] + rayEndDepthMm * slabDir[1],
    rowBaseWorld[2] + rayEndDepthMm * slabDir[2],
  ];
  const rayDebug = samplePanoV2MipRay({
    volume,
    centerDepthMmByCol: band.centerDepthMmByCol,
    targetOffsetMmByCol,
    targetOffsetMmMap,
    col: safeCol,
    row: safeRow,
  });

  return {
    col: safeCol,
    row: safeRow,
    arch: band.label,
    splineCenterWorld: roundVector3(splineCenterWorld),
    rowBaseWorld: roundVector3(rowBaseWorld),
    rayStartWorld: roundVector3(rayStartWorld),
    rayMidWorld: roundVector3(rayMidWorld),
    rayEndWorld: roundVector3(rayEndWorld),
    slabDir: roundVector3(slabDir),
    verticalDir: roundVector3(verticalDir),
    depthHalfRangeMm: roundTo(band.depthHalfRangeMm),
    mipSlabHalfWidthMm: roundTo(PANO_V2_MIP_SLAB_HALF_WIDTH_MM),
    centerDepthMm: roundTo(centerDepthMm),
    targetOffsetMm: roundTo(targetOffsetMm),
    rayMidDepthMm: roundTo(rayMidDepthMm),
    rayStartDepthMm: roundTo(rayStartDepthMm),
    rayEndDepthMm: roundTo(rayEndDepthMm),
    columnVerticalCenterOffsetMm: roundTo(columnVerticalCenterOffsetMm),
    rowOffsetMm: roundTo(rowOffsetMm),
    upperBypassHu: Number.isFinite(Number(upperBypassImage[safeRow * volume.panoWidth + safeCol]))
      ? roundTo(Number(upperBypassImage[safeRow * volume.panoWidth + safeCol]))
      : null,
    lowerBypassHu: Number.isFinite(Number(lowerBypassImage[safeRow * volume.panoWidth + safeCol]))
      ? roundTo(Number(lowerBypassImage[safeRow * volume.panoWidth + safeCol]))
      : null,
    rayWinnerHu: rayDebug.winnerHu,
    rayWinnerDepthOffsetMm: rayDebug.winnerDepthOffsetMm,
  };
}

export function buildPanoV2GeometryRayComparisonDiagnostics(params: {
  volume: PanoV2StraightenedVolume;
  layerRender: PanoV2LayerRenderResult;
  bypassCompositeImages: PanoV2BypassCompositeImages;
  row: number;
  failingCol: number;
  referenceCol: number;
}): PanoV2GeometryRayComparisonDiagnostics | null {
  const { volume, layerRender, bypassCompositeImages, row, failingCol, referenceCol } = params;
  const gapCenterRow = computePanoV2GapCenterRow(layerRender);
  const archLabel: PanoV2TrackLabel = row < gapCenterRow ? 'upperArch' : 'lowerArch';
  const band = archLabel === 'upperArch' ? volume.upperArch : volume.lowerArch;
  const targetOffsetMmByCol =
    archLabel === 'upperArch'
      ? bypassCompositeImages.upperTargetOffsetMmByCol
      : bypassCompositeImages.lowerTargetOffsetMmByCol;
  const targetOffsetMmMap =
    archLabel === 'upperArch'
      ? bypassCompositeImages.upperTargetOffsetMmMap
      : bypassCompositeImages.lowerTargetOffsetMmMap;

  const failing = buildPanoV2GeometryRayDebugPoint({
    volume,
    band,
    targetOffsetMmByCol,
    targetOffsetMmMap,
    upperBypassImage: bypassCompositeImages.upperArch,
    lowerBypassImage: bypassCompositeImages.lowerArch,
    col: failingCol,
    row,
  });
  const reference = buildPanoV2GeometryRayDebugPoint({
    volume,
    band,
    targetOffsetMmByCol,
    targetOffsetMmMap,
    upperBypassImage: bypassCompositeImages.upperArch,
    lowerBypassImage: bypassCompositeImages.lowerArch,
    col: referenceCol,
    row,
  });
  if (!failing || !reference) {
    return null;
  }

  const slabDirDot = roundTo(dot3(failing.slabDir, reference.slabDir), 4);
  const slabDirAngleDeg = angleBetweenUnitLikeVectorsDeg(failing.slabDir, reference.slabDir);
  const verticalDirDot = roundTo(dot3(failing.verticalDir, reference.verticalDir), 4);
  const verticalDirAngleDeg = angleBetweenUnitLikeVectorsDeg(
    failing.verticalDir,
    reference.verticalDir
  );
  const splineCenterDistanceMm = roundTo(
    distance3(failing.splineCenterWorld, reference.splineCenterWorld)
  );
  const rowBaseDistanceMm = roundTo(distance3(failing.rowBaseWorld, reference.rowBaseWorld));
  const rayStartDistanceMm = roundTo(distance3(failing.rayStartWorld, reference.rayStartWorld));
  const rayMidDistanceMm = roundTo(distance3(failing.rayMidWorld, reference.rayMidWorld));
  const rayEndDistanceMm = roundTo(distance3(failing.rayEndWorld, reference.rayEndWorld));
  const centerDepthDeltaMm = roundTo(failing.centerDepthMm - reference.centerDepthMm);
  const targetOffsetDeltaMm = roundTo(failing.targetOffsetMm - reference.targetOffsetMm);
  const rayMidDepthDeltaMm = roundTo(failing.rayMidDepthMm - reference.rayMidDepthMm);
  const columnVerticalCenterOffsetDeltaMm = roundTo(
    failing.columnVerticalCenterOffsetMm - reference.columnVerticalCenterOffsetMm
  );
  const rowOffsetDeltaMm = roundTo(failing.rowOffsetMm - reference.rowOffsetMm);
  const failingWinnerNearSlabEdge =
    Number.isFinite(failing.rayWinnerDepthOffsetMm) &&
    Math.abs(Number(failing.rayWinnerDepthOffsetMm)) >= PANO_V2_MIP_SLAB_HALF_WIDTH_MM - 1.5;
  const referenceWinnerNearSlabEdge =
    Number.isFinite(reference.rayWinnerDepthOffsetMm) &&
    Math.abs(Number(reference.rayWinnerDepthOffsetMm)) >= PANO_V2_MIP_SLAB_HALF_WIDTH_MM - 1.5;

  let anomalyKind: PanoV2GeometryRayComparisonDiagnostics['comparison']['anomalyKind'] =
    'no-clear-geometry-anomaly';
  let anomalyReason =
    'No sharp vector flip, vertical shift break, or depth-center jump was detected between the two columns.';
  const badLowSignal = !(Number.isFinite(failing.rayWinnerHu) && Number(failing.rayWinnerHu) >= 300);
  const referenceStrongSignal =
    Number.isFinite(reference.rayWinnerHu) && Number(reference.rayWinnerHu) >= 1000;

  if (slabDirAngleDeg >= 35 || verticalDirAngleDeg >= 35 || slabDirDot < 0.82 || verticalDirDot < 0.82) {
    anomalyKind = 'vector-flip-or-rotation';
    anomalyReason =
      'Neighboring columns diverge in slab/vertical direction, indicating a geometric vector rotation or flip rather than a pure depth miss.';
  } else if (
    Math.abs(columnVerticalCenterOffsetDeltaMm) >= 1.5 ||
    rowBaseDistanceMm >= splineCenterDistanceMm + 1.5
  ) {
    anomalyKind = 'vertical-shift-discontinuity';
    anomalyReason =
      'The row-adjusted ray base jumps more than the spline centerline itself, indicating a vertical center-offset discontinuity.';
  } else if (
    Math.abs(centerDepthDeltaMm) >= 1.5 ||
    Math.abs(targetOffsetDeltaMm) >= 1.25 ||
    Math.abs(rayMidDepthDeltaMm) >= 1.75 ||
    rayMidDistanceMm >= splineCenterDistanceMm + 1.75
  ) {
    anomalyKind = 'depth-center-jump';
    anomalyReason =
      'The ray midpoint shifts materially between neighboring columns, indicating a depth-center or target-offset discontinuity.';
  } else if (badLowSignal && referenceStrongSignal && referenceWinnerNearSlabEdge) {
    anomalyKind = 'insufficient-depth-window';
    anomalyReason =
      'The reference column finds dense anatomy only near the slab boundary while the failing column stays low-signal, indicating the sampling window is too shallow or depth-centered incorrectly.';
  } else if (badLowSignal) {
    anomalyKind = 'smooth-geometry-wrong-depth';
    anomalyReason =
      'The neighboring columns remain directionally smooth, but the failing ray is centered on low-HU tissue instead of dense anatomy. That points to a smooth but wrong depth placement rather than a vector flip.';
  }

  return {
    row,
    gapCenterRow,
    arch: archLabel,
    failing,
    reference,
    comparison: {
      slabDirDot,
      slabDirAngleDeg,
      verticalDirDot,
      verticalDirAngleDeg,
      splineCenterDistanceMm,
      rowBaseDistanceMm,
      rayStartDistanceMm,
      rayMidDistanceMm,
      rayEndDistanceMm,
      centerDepthDeltaMm,
      targetOffsetDeltaMm,
      rayMidDepthDeltaMm,
      columnVerticalCenterOffsetDeltaMm,
      rowOffsetDeltaMm,
      failingWinnerNearSlabEdge,
      referenceWinnerNearSlabEdge,
      anomalyKind,
      anomalyReason,
    },
  };
}

export function computePanoV2GapCenterRow(result: {
  upperArch: { anchorRow: number };
  lowerArch: { anchorRow: number };
}): number {
  return Math.round((result.upperArch.anchorRow + result.lowerArch.anchorRow) * 0.5);
}

function resolveTradeoffLabel(
  metrics: PanoV2LayerMetrics
): 'incisor-biased' | 'posterior-biased' | 'balanced' {
  const posteriorDetail = Math.max(metrics.leftMolarDetail, metrics.rightMolarDetail);
  if (metrics.incisorDetail > posteriorDetail * 1.08) {
    return 'incisor-biased';
  }
  if (posteriorDetail > metrics.incisorDetail * 1.08) {
    return 'posterior-biased';
  }
  return 'balanced';
}

function buildLayerMetrics(
  band: PanoV2StraightenedVolumeBand,
  image: Float32Array
): PanoV2LayerMetrics {
  const width = band.centerDepthMmByCol.length;
  const values: number[] = [];
  const detailSamples: number[] = [];
  let finiteCount = 0;
  let anatomySignalCount = 0;
  let detailSum = 0;
  let incisorDetailSum = 0;
  let incisorDetailCount = 0;
  let leftMolarDetailSum = 0;
  let leftMolarDetailCount = 0;
  let rightMolarDetailSum = 0;
  let rightMolarDetailCount = 0;

  for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
    for (let col = 0; col < width; col++) {
      const value = Number(image[rowIndex * width + col]);
      if (!Number.isFinite(value)) {
        continue;
      }
      finiteCount++;
      values.push(value);
      if (value > -650) {
        anatomySignalCount++;
      }

      const left = col > 0 ? Number(image[rowIndex * width + (col - 1)]) : Number.NaN;
      const up = rowIndex > 0 ? Number(image[(rowIndex - 1) * width + col]) : Number.NaN;
      let detail = 0;
      if (Number.isFinite(left)) {
        detail += Math.abs(value - left);
      }
      if (Number.isFinite(up)) {
        detail += Math.abs(value - up);
      }
      detailSamples.push(detail);
      detailSum += detail;

      const columnRatio = width > 1 ? col / (width - 1) : 0.5;
      if (columnRatio < REGION_POSTERIOR_CUTOFF) {
        leftMolarDetailSum += detail;
        leftMolarDetailCount++;
      } else if (columnRatio >= REGION_ANTERIOR_MIN && columnRatio <= REGION_ANTERIOR_MAX) {
        incisorDetailSum += detail;
        incisorDetailCount++;
      } else if (columnRatio > REGION_RIGHT_POSTERIOR_MIN) {
        rightMolarDetailSum += detail;
        rightMolarDetailCount++;
      }
    }
  }

  const valueSummary = summarize(values);

  return {
    finiteFraction: roundTo(image.length > 0 ? finiteCount / image.length : 0, 4),
    anatomySignalFraction: roundTo(image.length > 0 ? anatomySignalCount / image.length : 0, 4),
    valueRangeHu: valueSummary,
    detailMean: roundTo(detailSamples.length > 0 ? detailSum / detailSamples.length : 0),
    sharpnessPeak: roundTo(percentile(detailSamples, 0.9)),
    toothContrastHu: roundTo(
      valueSummary.p90 - valueSummary.min > 0 ? percentile(values, 0.9) - percentile(values, 0.1) : 0
    ),
    incisorDetail: roundTo(incisorDetailCount > 0 ? incisorDetailSum / incisorDetailCount : 0),
    leftMolarDetail: roundTo(leftMolarDetailCount > 0 ? leftMolarDetailSum / leftMolarDetailCount : 0),
    rightMolarDetail: roundTo(
      rightMolarDetailCount > 0 ? rightMolarDetailSum / rightMolarDetailCount : 0
    ),
  };
}

function buildLayerToothRowBandDiagnostics(params: {
  band: PanoV2StraightenedVolumeBand;
  image: Float32Array;
  nearestDepthOffsetMm: number;
  targetOffsetMm: number;
}): PanoV2LayerCandidateDiagnostics['toothRowBand'] {
  const { band, image, nearestDepthOffsetMm, targetOffsetMm } = params;
  const width = band.centerDepthMmByCol.length;
  const localRowStart = clampNumber(
    band.localAnchorRow - TOOTH_ROW_BAND_ROW_RADIUS,
    0,
    band.rowCount - 1
  );
  const localRowEnd = clampNumber(
    band.localAnchorRow + TOOTH_ROW_BAND_ROW_RADIUS,
    0,
    band.rowCount - 1
  );
  const sampledValues: number[] = [];
  let voxelsAbove700Hu = 0;
  let exceeds1500Hu = false;

  for (let rowIndex = localRowStart; rowIndex <= localRowEnd; rowIndex++) {
    for (let col = 0; col < width; col++) {
      const value = Number(image[rowIndex * width + col]);
      if (!Number.isFinite(value)) {
        continue;
      }
      sampledValues.push(value);
      if (value > 700) {
        voxelsAbove700Hu++;
      }
      if (value > 1500) {
        exceeds1500Hu = true;
      }
    }
  }

  return {
    localRowStart,
    localRowEnd,
    panoRowStart: band.rowStart + localRowStart,
    panoRowEnd: band.rowStart + localRowEnd,
    sampledVoxelCount: sampledValues.length,
    medianHu: roundTo(percentile(sampledValues, 0.5)),
    fractionAbove700Hu: roundTo(
      sampledValues.length > 0 ? voxelsAbove700Hu / sampledValues.length : 0,
      4
    ),
    centerDepthOffsetMm: roundTo(nearestDepthOffsetMm),
    requestedCenterDepthOffsetMm: roundTo(targetOffsetMm),
    exceeds1500Hu,
  };
}

function buildLayerProbes(
  band: PanoV2StraightenedVolumeBand,
  image: Float32Array
): PanoV2LayerProbeColumn[] {
  const width = band.centerDepthMmByCol.length;
  const representativeRowIndices = Array.from(
    new Set<number>([
      clampNumber(band.localAnchorRow - 12, 0, band.rowCount - 1),
      clampNumber(band.localAnchorRow, 0, band.rowCount - 1),
      clampNumber(band.localAnchorRow + 12, 0, band.rowCount - 1),
    ])
  ).sort((left, right) => left - right);

  return buildEvenlySpacedIndices(width, 3).map(col => ({
    col,
    representativeRows: representativeRowIndices.map(localRow => {
      const value = Number(image[localRow * width + col]);
      return {
        localRow,
        panoRow: band.rowStart + localRow,
        valueHu: Number.isFinite(value) ? roundTo(value) : null,
      };
    }),
  }));
}

function buildBandLayerRenderResult(
  volume: PanoV2StraightenedVolume,
  band: PanoV2StraightenedVolumeBand
): {
  band: PanoV2BandLayerRenderResult;
  toothBandValuesHu: number[];
  renderMs: number;
} {
  const allToothBandValuesHu: number[] = [];
  let totalRenderMs = 0;
  const layers = LAYER_OFFSET_FACTORS.map((offsetFactor, layerIndex) => {
    const targetOffsetMm = roundTo(band.depthHalfRangeMm * offsetFactor);
    const mipLayer = buildMipBandImage(volume, band, targetOffsetMm);
    for (let index = 0; index < mipLayer.toothBandValuesHu.length; index++) {
      allToothBandValuesHu.push(mipLayer.toothBandValuesHu[index]);
    }
    totalRenderMs += mipLayer.renderMs;
    const metrics = buildLayerMetrics(band, mipLayer.image);
    return {
      diagnostic: {
        layerId: `${band.label}-layer-${layerIndex + 1}`,
        band: band.label,
        targetOffsetMm,
        nearestDepthOffsetMm: targetOffsetMm,
        nearestDepthIndex: Math.round(
          PANO_V2_MIP_SLAB_HALF_WIDTH_MM / PANO_V2_MIP_DEPTH_STEP_MM
        ),
        tradeoffLabel: resolveTradeoffLabel(metrics),
        metrics,
        toothRowBand: buildLayerToothRowBandDiagnostics({
          band,
          image: mipLayer.image,
          nearestDepthOffsetMm: targetOffsetMm,
          targetOffsetMm,
        }),
        sampledColumns: buildLayerProbes(band, mipLayer.image),
      },
      image: mipLayer.image,
    };
  });

  const pickBest = (selector: (layer: PanoV2RenderedLayer) => number): string | null => {
    let bestLayer: PanoV2RenderedLayer | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const layer of layers) {
      const score = selector(layer);
      if (score > bestScore) {
        bestScore = score;
        bestLayer = layer;
      }
    }
    return bestLayer?.diagnostic.layerId ?? null;
  };

  return {
    band: {
      label: band.label,
      anchorRow: band.anchorRow,
      localAnchorRow: band.localAnchorRow,
      rowStart: band.rowStart,
      rowEnd: band.rowEnd,
      rowCount: band.rowCount,
      layers,
      bestByRegion: {
        incisors: pickBest(layer => layer.diagnostic.metrics.incisorDetail),
        leftMolar: pickBest(layer => layer.diagnostic.metrics.leftMolarDetail),
        rightMolar: pickBest(layer => layer.diagnostic.metrics.rightMolarDetail),
        sharpnessPeak: pickBest(layer => layer.diagnostic.metrics.sharpnessPeak),
      },
    },
    toothBandValuesHu: allToothBandValuesHu,
    renderMs: totalRenderMs,
  };
}

function toBandLayerDiagnostics(
  band: PanoV2BandLayerRenderResult
): PanoV2BandLayerDiagnostics {
  return {
    label: band.label,
    layerCount: band.layers.length,
    layers: band.layers.map(layer => layer.diagnostic),
    bestByRegion: band.bestByRegion,
  };
}

export function buildPanoV2LayerRenderResult(
  volume: PanoV2StraightenedVolume
): PanoV2LayerRenderResult {
  const upperArchResult = buildBandLayerRenderResult(volume, volume.upperArch);
  const lowerArchResult = buildBandLayerRenderResult(volume, volume.lowerArch);
  const combinedToothBandValuesHu: number[] = [];
  for (let index = 0; index < upperArchResult.toothBandValuesHu.length; index++) {
    combinedToothBandValuesHu.push(upperArchResult.toothBandValuesHu[index]);
  }
  for (let index = 0; index < lowerArchResult.toothBandValuesHu.length; index++) {
    combinedToothBandValuesHu.push(lowerArchResult.toothBandValuesHu[index]);
  }
  let toothBandMaxHu = 0;
  for (let index = 0; index < combinedToothBandValuesHu.length; index++) {
    const value = combinedToothBandValuesHu[index];
    if (index === 0 || value > toothBandMaxHu) {
      toothBandMaxHu = value;
    }
  }
  const slabDirectionColumn = Math.max(
    0,
    Math.min(volume.frames.length - 1, Math.floor(volume.frames.length * 0.5))
  );
  const representativeFrame = volume.frames[slabDirectionColumn];
  const representativeSlabDirection = representativeFrame?.mipSlabDir ?? representativeFrame?.slabDir ?? [0, 0, 0];
  const verticalCenterOffsetMmUsed =
    volume.frames.length > 0
      ? volume.frames.reduce(
          (sum, frame) => sum + Number(frame.columnVerticalCenterOffsetMm || 0),
          0
        ) / volume.frames.length
      : 0;
  return {
    enabled: true,
    phase: 3,
    model: 'thick-slab-mip-render',
    layerCountPerArch: PANO_V2_LAYER_COUNT_PER_ARCH,
    mip: {
      slabHalfWidthMm: PANO_V2_MIP_SLAB_HALF_WIDTH_MM,
      depthStepMm: PANO_V2_MIP_DEPTH_STEP_MM,
      sampledColumnCount: volume.panoWidth,
      mipVoiOverride: true,
      layerRowConstraintsApplied: true,
      verticalHalfMmUsed: roundTo(volume.effectiveVerticalHalfMm),
      verticalCenterOffsetMm: roundTo(verticalCenterOffsetMmUsed),
      slabDirectionColumn,
      slabDirectionVector: [
        roundTo(representativeSlabDirection[0]),
        roundTo(representativeSlabDirection[1]),
        roundTo(representativeSlabDirection[2]),
      ],
      toothBandMedianHu: roundTo(percentile(combinedToothBandValuesHu, 0.5)),
      toothBandMaxHu: roundTo(combinedToothBandValuesHu.length > 0 ? toothBandMaxHu : 0),
      totalRenderMs: roundTo(upperArchResult.renderMs + lowerArchResult.renderMs),
    },
    upperArch: upperArchResult.band,
    lowerArch: lowerArchResult.band,
  };
}

export function toPanoV2LayerRenderDiagnostics(
  result: PanoV2LayerRenderResult
): PanoV2LayerRenderDiagnostics {
  return {
    enabled: true,
    phase: 3,
    model: 'thick-slab-mip-render',
    layerCountPerArch: result.layerCountPerArch,
    mip: result.mip,
    upperArch: toBandLayerDiagnostics(result.upperArch),
    lowerArch: toBandLayerDiagnostics(result.lowerArch),
  };
}

export function buildPanoV2LayerRenderDiagnostics(
  volume: PanoV2StraightenedVolume
): PanoV2LayerRenderDiagnostics {
  return toPanoV2LayerRenderDiagnostics(buildPanoV2LayerRenderResult(volume));
}

function buildFullPlaneBandLayerImages(params: {
  band: PanoV2BandLayerRenderResult;
  panoWidth: number;
  panoHeight: number;
}): Float32Array[] {
  const { band, panoWidth, panoHeight } = params;
  const planeSize = Math.max(1, panoWidth * panoHeight);
  return band.layers.map(layer => {
    const fullPlaneImage = new Float32Array(planeSize);
    fullPlaneImage.fill(-1000);
    for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
      const panoRow = band.rowStart + rowIndex;
      if (panoRow < 0 || panoRow >= panoHeight) {
        continue;
      }
      const targetOffset = panoRow * panoWidth;
      const sourceOffset = rowIndex * panoWidth;
      fullPlaneImage.set(layer.image.subarray(sourceOffset, sourceOffset + panoWidth), targetOffset);
    }
    return fullPlaneImage;
  });
}

function resolvePreferredBypassLayerIndex(band: PanoV2BandLayerRenderResult): number {
  const sharpnessLayerId = band.bestByRegion.sharpnessPeak;
  const sharpnessLayerIndex =
    typeof sharpnessLayerId === 'string'
      ? band.layers.findIndex(layer => layer.diagnostic.layerId === sharpnessLayerId)
      : -1;
  if (sharpnessLayerIndex >= 0) {
    return sharpnessLayerIndex;
  }
  return clampNumber(Math.floor(band.layers.length * 0.5), 0, Math.max(0, band.layers.length - 1));
}

function resolveBypassLayerIndexFromId(
  band: PanoV2BandLayerRenderResult,
  layerId: string | null | undefined
): number {
  if (typeof layerId === 'string') {
    const index = band.layers.findIndex(layer => layer.diagnostic.layerId === layerId);
    if (index >= 0) {
      return index;
    }
  }
  return resolvePreferredBypassLayerIndex(band);
}

function buildRegionalBypassTargetOffsetMmByCol(
  band: PanoV2BandLayerRenderResult,
  width: number
): Float32Array {
  const fallbackIndex = resolvePreferredBypassLayerIndex(band);
  const fallbackOffsetMm = Number(band.layers[fallbackIndex]?.diagnostic.targetOffsetMm ?? 0);
  const leftOffsetMm = Number(
    band.layers[resolveBypassLayerIndexFromId(band, band.bestByRegion.leftMolar)]?.diagnostic
      .targetOffsetMm ?? fallbackOffsetMm
  );
  const centerOffsetMm = Number(
    band.layers[resolveBypassLayerIndexFromId(band, band.bestByRegion.incisors)]?.diagnostic
      .targetOffsetMm ?? fallbackOffsetMm
  );
  const rightOffsetMm = Number(
    band.layers[resolveBypassLayerIndexFromId(band, band.bestByRegion.rightMolar)]?.diagnostic
      .targetOffsetMm ?? fallbackOffsetMm
  );
  const targetOffsetMmByCol = new Float32Array(Math.max(1, width));

  for (let col = 0; col < width; col++) {
    const normalizedCol = width <= 1 ? 0.5 : col / Math.max(1, width - 1);
    let targetOffsetMm = centerOffsetMm;
    if (normalizedCol <= 0.25) {
      targetOffsetMm = leftOffsetMm;
    } else if (normalizedCol < 0.4) {
      const blend = (normalizedCol - 0.25) / 0.15;
      targetOffsetMm = leftOffsetMm * (1 - blend) + centerOffsetMm * blend;
    } else if (normalizedCol <= 0.6) {
      targetOffsetMm = centerOffsetMm;
    } else if (normalizedCol < 0.75) {
      const blend = (normalizedCol - 0.6) / 0.15;
      targetOffsetMm = centerOffsetMm * (1 - blend) + rightOffsetMm * blend;
    } else {
      targetOffsetMm = rightOffsetMm;
    }
    targetOffsetMmByCol[col] = targetOffsetMm;
  }

  return targetOffsetMmByCol;
}

function buildBypassTargetOffsetMmByCol(
  band: PanoV2BandLayerRenderResult,
  width: number
): Float32Array {
  const safeWidth = Math.max(1, width);
  const regionalTargetOffsetMmByCol = buildRegionalBypassTargetOffsetMmByCol(band, safeWidth);
  const layerTargetOffsetsMm = band.layers.map(layer =>
    Number(layer.diagnostic.targetOffsetMm ?? 0)
  );
  const selectedTargetOffsetMmByCol = new Float32Array(safeWidth);
  const trustByCol = new Uint8Array(safeWidth);

  for (let col = 0; col < safeWidth; col++) {
    let bestLayerIndex = -1;
    let bestStrongRowCount = -1;
    let bestDenseRowCount = -1;
    let bestPeakHu = Number.NEGATIVE_INFINITY;
    let bestSupportScore = Number.NEGATIVE_INFINITY;

    for (let layerIndex = 0; layerIndex < band.layers.length; layerIndex++) {
      const image = band.layers[layerIndex]?.image;
      if (!image) {
        continue;
      }
      let peakHu = Number.NEGATIVE_INFINITY;
      let denseRowCount = 0;
      let strongRowCount = 0;
      let supportScore = 0;
      for (let rowIndex = 0; rowIndex < band.rowCount; rowIndex++) {
        const value = Number(image[rowIndex * safeWidth + col]);
        if (!Number.isFinite(value)) {
          continue;
        }
        if (value > peakHu) {
          peakHu = value;
        }
        if (value >= PANO_V2_BYPASS_COLUMN_DENSE_HU) {
          denseRowCount++;
        }
        if (value >= PANO_V2_BYPASS_COLUMN_STRONG_HU) {
          strongRowCount++;
        }
        if (value > PANO_V2_BYPASS_COLUMN_SUPPORT_BASE_HU) {
          supportScore += value - PANO_V2_BYPASS_COLUMN_SUPPORT_BASE_HU;
        }
      }

      const isBetterCandidate =
        strongRowCount > bestStrongRowCount ||
        (strongRowCount === bestStrongRowCount && denseRowCount > bestDenseRowCount) ||
        (strongRowCount === bestStrongRowCount &&
          denseRowCount === bestDenseRowCount &&
          peakHu > bestPeakHu) ||
        (strongRowCount === bestStrongRowCount &&
          denseRowCount === bestDenseRowCount &&
          peakHu === bestPeakHu &&
          supportScore > bestSupportScore);
      if (isBetterCandidate) {
        bestLayerIndex = layerIndex;
        bestStrongRowCount = strongRowCount;
        bestDenseRowCount = denseRowCount;
        bestPeakHu = peakHu;
        bestSupportScore = supportScore;
      }
    }

    const hasTrustedColumnSelection =
      bestLayerIndex >= 0 &&
      Number.isFinite(bestPeakHu) &&
      bestPeakHu >= PANO_V2_BYPASS_COLUMN_DENSE_HU &&
      (bestDenseRowCount >= PANO_V2_BYPASS_COLUMN_MIN_DENSE_ROWS || bestStrongRowCount > 0);
    selectedTargetOffsetMmByCol[col] = hasTrustedColumnSelection
      ? Number(layerTargetOffsetsMm[bestLayerIndex] ?? regionalTargetOffsetMmByCol[col] ?? 0)
      : regionalTargetOffsetMmByCol[col];
    trustByCol[col] = hasTrustedColumnSelection ? 1 : 0;
  }

  const smoothedTargetOffsetMmByCol = new Float32Array(safeWidth);
  for (let col = 0; col < safeWidth; col++) {
    let weightedSum = 0;
    let totalWeight = 0;
    let trustedNeighborCount = 0;
    const windowStart = Math.max(0, col - PANO_V2_BYPASS_TARGET_SMOOTHING_RADIUS);
    const windowEnd = Math.min(safeWidth - 1, col + PANO_V2_BYPASS_TARGET_SMOOTHING_RADIUS);
    for (let sampleCol = windowStart; sampleCol <= windowEnd; sampleCol++) {
      const distance = Math.abs(sampleCol - col);
      const baseWeight = PANO_V2_BYPASS_TARGET_SMOOTHING_RADIUS + 1 - distance;
      const isTrusted = trustByCol[sampleCol] === 1;
      if (isTrusted) {
        trustedNeighborCount++;
      }
      const trustWeight = isTrusted ? 2.5 : 0.35;
      const weight = baseWeight * trustWeight;
      weightedSum += Number(selectedTargetOffsetMmByCol[sampleCol] ?? regionalTargetOffsetMmByCol[col]) * weight;
      totalWeight += weight;
    }
    smoothedTargetOffsetMmByCol[col] =
      trustedNeighborCount > 0 && totalWeight > 0
        ? weightedSum / totalWeight
        : regionalTargetOffsetMmByCol[col];
  }

  return smoothedTargetOffsetMmByCol;
}

export function buildPanoV2FullPlaneLayerImages(params: {
  result: PanoV2LayerRenderResult;
  panoWidth: number;
  panoHeight: number;
}): PanoV2FullPlaneLayerImages {
  return {
    upperArch: buildFullPlaneBandLayerImages({
      band: params.result.upperArch,
      panoWidth: params.panoWidth,
      panoHeight: params.panoHeight,
    }),
    lowerArch: buildFullPlaneBandLayerImages({
      band: params.result.lowerArch,
      panoWidth: params.panoWidth,
      panoHeight: params.panoHeight,
    }),
  };
}

export function buildPanoV2BypassCompositeImages(params: {
  result: PanoV2LayerRenderResult;
  volume: PanoV2StraightenedVolume;
}): PanoV2BypassCompositeImages {
  const gapCenterRow = computePanoV2GapCenterRow(params.result);
  const upperColumnTargetOffsetMmByCol = buildBypassTargetOffsetMmByCol(
    params.result.upperArch,
    params.volume.panoWidth
  );
  const lowerColumnTargetOffsetMmByCol = buildBypassTargetOffsetMmByCol(
    params.result.lowerArch,
    params.volume.panoWidth
  );
  const upperTargetPlan = buildRowAdaptiveTargetOffsetPlan({
    band: params.volume.upperArch,
    panoWidth: params.volume.panoWidth,
    panoHeight: params.volume.panoHeight,
    assignedRowStartInclusive: 0,
    assignedRowEndExclusive: gapCenterRow,
    columnTargetOffsetMmByCol: upperColumnTargetOffsetMmByCol,
  });
  const lowerTargetPlan = buildRowAdaptiveTargetOffsetPlan({
    band: params.volume.lowerArch,
    panoWidth: params.volume.panoWidth,
    panoHeight: params.volume.panoHeight,
    assignedRowStartInclusive: gapCenterRow,
    assignedRowEndExclusive: params.volume.panoHeight,
    columnTargetOffsetMmByCol: lowerColumnTargetOffsetMmByCol,
  });
  const upperArch = buildFullPlaneMipImage({
    volume: params.volume,
    centerDepthMmByCol: params.volume.upperArch.centerDepthMmByCol,
    targetOffsetMmByCol: upperTargetPlan.targetOffsetMmByCol,
    targetOffsetMmMap: upperTargetPlan.targetOffsetMmMap,
    rowStartInclusive: 0,
    rowEndExclusive: gapCenterRow,
  });
  const lowerArch = buildFullPlaneMipImage({
    volume: params.volume,
    centerDepthMmByCol: params.volume.lowerArch.centerDepthMmByCol,
    targetOffsetMmByCol: lowerTargetPlan.targetOffsetMmByCol,
    targetOffsetMmMap: lowerTargetPlan.targetOffsetMmMap,
    rowStartInclusive: gapCenterRow,
    rowEndExclusive: params.volume.panoHeight,
  });

  applyCrossArchBypassRescue({
    volume: params.volume,
    targetImage: upperArch,
    oppositeCenterDepthMmByCol: params.volume.lowerArch.centerDepthMmByCol,
    oppositeTargetOffsetMmByCol: lowerTargetPlan.targetOffsetMmByCol,
    oppositeTargetOffsetMmMap: lowerTargetPlan.targetOffsetMmMap,
    rowStartInclusive: 0,
    rowEndExclusive: gapCenterRow,
  });
  applyCrossArchBypassRescue({
    volume: params.volume,
    targetImage: lowerArch,
    oppositeCenterDepthMmByCol: params.volume.upperArch.centerDepthMmByCol,
    oppositeTargetOffsetMmByCol: upperTargetPlan.targetOffsetMmByCol,
    oppositeTargetOffsetMmMap: upperTargetPlan.targetOffsetMmMap,
    rowStartInclusive: gapCenterRow,
    rowEndExclusive: params.volume.panoHeight,
  });

  return {
    upperArch,
    lowerArch,
    upperTargetOffsetMmByCol: upperTargetPlan.targetOffsetMmByCol,
    lowerTargetOffsetMmByCol: lowerTargetPlan.targetOffsetMmByCol,
    upperTargetOffsetMmMap: upperTargetPlan.targetOffsetMmMap,
    lowerTargetOffsetMmMap: lowerTargetPlan.targetOffsetMmMap,
  };
}

export function buildPanoV2GapBoundaryAnalysis(params: {
  upperArchImage: Float32Array;
  lowerArchImage: Float32Array;
  panoWidth: number;
  panoHeight: number;
  denseHuThreshold?: number;
  sampleStrideColumns?: number;
}): PanoV2GapBoundaryAnalysis {
  const {
    upperArchImage,
    lowerArchImage,
    panoWidth,
    panoHeight,
    denseHuThreshold = PANO_V2_GAP_ANALYSIS_DENSE_HU_THRESHOLD,
    sampleStrideColumns = PANO_V2_GAP_ANALYSIS_SAMPLE_STRIDE_COLUMNS,
  } = params;
  const upperLastDenseRowByCol = new Int16Array(Math.max(1, panoWidth));
  const lowerFirstDenseRowByCol = new Int16Array(Math.max(1, panoWidth));
  upperLastDenseRowByCol.fill(-1);
  lowerFirstDenseRowByCol.fill(-1);

  for (let col = 0; col < panoWidth; col++) {
    let upperLastDenseRow = -1;
    for (let row = 0; row < panoHeight; row++) {
      const value = Number(upperArchImage[row * panoWidth + col]);
      if (Number.isFinite(value) && value > denseHuThreshold) {
        upperLastDenseRow = row;
      }
    }
    upperLastDenseRowByCol[col] = upperLastDenseRow;

    let lowerFirstDenseRow = -1;
    for (let row = 0; row < panoHeight; row++) {
      const value = Number(lowerArchImage[row * panoWidth + col]);
      if (Number.isFinite(value) && value > denseHuThreshold) {
        lowerFirstDenseRow = row;
        break;
      }
    }
    lowerFirstDenseRowByCol[col] = lowerFirstDenseRow;
  }

  const sampledColumns: PanoV2GapBoundarySample[] = [];
  for (let col = 0; col < panoWidth; col++) {
    const shouldSample = col % Math.max(1, sampleStrideColumns) === 0 || col === panoWidth - 1;
    if (!shouldSample) {
      continue;
    }
    const upperLastDenseRow = Number(upperLastDenseRowByCol[col]);
    const lowerFirstDenseRow = Number(lowerFirstDenseRowByCol[col]);
    const hasGap =
      upperLastDenseRow >= 0 &&
      lowerFirstDenseRow >= 0 &&
      lowerFirstDenseRow > upperLastDenseRow + 1;
    sampledColumns.push({
      col,
      upperLastDenseRow: upperLastDenseRow >= 0 ? upperLastDenseRow : null,
      lowerFirstDenseRow: lowerFirstDenseRow >= 0 ? lowerFirstDenseRow : null,
      gapHeightPx: hasGap ? lowerFirstDenseRow - upperLastDenseRow - 1 : 0,
    });
  }

  return {
    denseHuThreshold,
    sampleStrideColumns: Math.max(1, sampleStrideColumns),
    upperLastDenseRowByCol,
    lowerFirstDenseRowByCol,
    sampledColumns,
  };
}

export function buildPanoV2DarkPocketDiagnostics(params: {
  volume: PanoV2StraightenedVolume;
  layerRender: PanoV2LayerRenderResult;
  bypassCompositeImages: PanoV2BypassCompositeImages;
  finalPixelData: Float32Array;
}): PanoV2DarkPocketDiagnostics {
  const { volume, layerRender, bypassCompositeImages, finalPixelData } = params;
  const gapCenterRow = computePanoV2GapCenterRow(layerRender);
  const upperAnchorRow = Math.round(layerRender.upperArch.anchorRow);
  const panoWidth = volume.panoWidth;
  const panoHeight = volume.panoHeight;
  const colStart = Math.min(Math.max(6, 0), Math.max(0, panoWidth - 1));
  const colEnd = Math.max(colStart, panoWidth - 7);
  const denseRowThresholdHu = 1800;
  const topBorderExclusionRows = clampNumber(
    Math.round(gapCenterRow * 0.18),
    24,
    Math.min(80, Math.max(24, gapCenterRow - 1))
  );
  let denseBandCenterRow: number | null = null;
  let densestRowBrightCount = -1;
  const denseRowSearchStart = Math.min(topBorderExclusionRows, Math.max(0, gapCenterRow - 1));
  const denseRowSearchEnd = Math.max(denseRowSearchStart, Math.max(0, gapCenterRow - 1));
  for (let row = denseRowSearchStart; row <= denseRowSearchEnd; row++) {
    let brightCount = 0;
    for (let col = colStart; col <= colEnd; col++) {
      const value = Number(finalPixelData[row * panoWidth + col]);
      if (Number.isFinite(value) && value >= denseRowThresholdHu) {
        brightCount++;
      }
    }
    if (brightCount > densestRowBrightCount) {
      densestRowBrightCount = brightCount;
      denseBandCenterRow = row;
    }
  }
  const searchBandHalfHeight = 54;
  const rowStart = denseBandCenterRow === null
    ? denseRowSearchStart
    : clampNumber(denseBandCenterRow - searchBandHalfHeight, denseRowSearchStart, denseRowSearchEnd);
  const rowEnd = denseBandCenterRow === null
    ? denseRowSearchEnd
    : clampNumber(denseBandCenterRow + searchBandHalfHeight, rowStart, denseRowSearchEnd);
  const neighborhoodRadius = 6;
  const rawCandidates: Array<{
    col: number;
    row: number;
    finalHu: number;
    surroundingMedianHu: number;
    surroundingMaxHu: number;
    darknessScore: number;
  }> = [];

  for (let row = rowStart; row <= rowEnd; row++) {
    for (let col = colStart; col <= colEnd; col++) {
      const index = row * panoWidth + col;
      const finalHu = Number(finalPixelData[index]);
      if (!Number.isFinite(finalHu) || finalHu > 1200) {
        continue;
      }
      const neighborhoodValues: number[] = [];
      let neighborhoodMaxHu = Number.NEGATIVE_INFINITY;
      for (let sampleRow = row - neighborhoodRadius; sampleRow <= row + neighborhoodRadius; sampleRow++) {
        if (sampleRow < rowStart || sampleRow > rowEnd) {
          continue;
        }
        for (let sampleCol = col - neighborhoodRadius; sampleCol <= col + neighborhoodRadius; sampleCol++) {
          if (sampleCol < colStart || sampleCol > colEnd) {
            continue;
          }
          const isPerimeter =
            sampleRow === row - neighborhoodRadius ||
            sampleRow === row + neighborhoodRadius ||
            sampleCol === col - neighborhoodRadius ||
            sampleCol === col + neighborhoodRadius;
          if (!isPerimeter) {
            continue;
          }
          const sampleValue = Number(finalPixelData[sampleRow * panoWidth + sampleCol]);
          if (!Number.isFinite(sampleValue)) {
            continue;
          }
          neighborhoodValues.push(sampleValue);
          if (sampleValue > neighborhoodMaxHu) {
            neighborhoodMaxHu = sampleValue;
          }
        }
      }
      if (neighborhoodValues.length < 12 || !Number.isFinite(neighborhoodMaxHu)) {
        continue;
      }
      const surroundingMedianHu = percentile(neighborhoodValues, 0.5);
      if (surroundingMedianHu <= finalHu + 250 || neighborhoodMaxHu <= 1600) {
        continue;
      }
      rawCandidates.push({
        col,
        row,
        finalHu: roundTo(finalHu),
        surroundingMedianHu: roundTo(surroundingMedianHu),
        surroundingMaxHu: roundTo(neighborhoodMaxHu),
        darknessScore: roundTo((surroundingMedianHu - finalHu) + (neighborhoodMaxHu - finalHu) * 0.2),
      });
    }
  }

  rawCandidates.sort((left, right) => {
    if (right.darknessScore !== left.darknessScore) {
      return right.darknessScore - left.darknessScore;
    }
    return left.finalHu - right.finalHu;
  });

  const darkestPockets: PanoV2DarkPocketPoint[] = [];
  for (let candidateIndex = 0; candidateIndex < rawCandidates.length; candidateIndex++) {
    const candidate = rawCandidates[candidateIndex];
    const tooCloseToExisting = darkestPockets.some(
      pocket =>
        Math.abs(pocket.col - candidate.col) <= 12 &&
        Math.abs(pocket.row - candidate.row) <= 12
    );
    if (tooCloseToExisting) {
      continue;
    }
    const index = candidate.row * panoWidth + candidate.col;
    const upperBypassHu = Number(bypassCompositeImages.upperArch[index]);
    const lowerBypassHu = Number(bypassCompositeImages.lowerArch[index]);
    const upperRay = samplePanoV2MipRay({
      volume,
      centerDepthMmByCol: volume.upperArch.centerDepthMmByCol,
      targetOffsetMmByCol: bypassCompositeImages.upperTargetOffsetMmByCol,
      targetOffsetMmMap: bypassCompositeImages.upperTargetOffsetMmMap,
      col: candidate.col,
      row: candidate.row,
    });
    const lowerRay = samplePanoV2MipRay({
      volume,
      centerDepthMmByCol: volume.lowerArch.centerDepthMmByCol,
      targetOffsetMmByCol: bypassCompositeImages.lowerTargetOffsetMmByCol,
      targetOffsetMmMap: bypassCompositeImages.lowerTargetOffsetMmMap,
      col: candidate.col,
      row: candidate.row,
    });
    darkestPockets.push({
      col: candidate.col,
      row: candidate.row,
      finalHu: candidate.finalHu,
      upperBypassHu: Number.isFinite(upperBypassHu) ? roundTo(upperBypassHu) : null,
      lowerBypassHu: Number.isFinite(lowerBypassHu) ? roundTo(lowerBypassHu) : null,
      selectedSource: candidate.row < gapCenterRow ? 'upper-arch-bypass' : 'lower-arch-bypass',
      surroundingMedianHu: candidate.surroundingMedianHu,
      surroundingMaxHu: candidate.surroundingMaxHu,
      darknessScore: candidate.darknessScore,
      upperRayWinnerHu: upperRay.winnerHu,
      upperRayWinnerDepthOffsetMm: upperRay.winnerDepthOffsetMm,
      lowerRayWinnerHu: lowerRay.winnerHu,
      lowerRayWinnerDepthOffsetMm: lowerRay.winnerDepthOffsetMm,
      upperRaySamples: upperRay.samples,
      lowerRaySamples: lowerRay.samples,
    });
    if (darkestPockets.length >= 6) {
      break;
    }
  }

  return {
    searchRegion: {
      colStart,
      colEnd,
      rowStart,
      rowEnd,
      gapCenterRow,
      upperAnchorRow,
      topBorderExclusionRows,
      denseBandCenterRow,
      denseRowThresholdHu,
    },
    candidateCountBeforeSuppression: rawCandidates.length,
    darkestPockets,
  };
}
