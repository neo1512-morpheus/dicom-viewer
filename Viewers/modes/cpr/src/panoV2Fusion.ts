import type { PanoV2NumericSummary, PanoV2TrackLabel } from './panoV2Geometry';
import type {
  PanoV2BandLayerRenderResult,
  PanoV2GapBoundaryAnalysis,
  PanoV2BypassCompositeImages,
  PanoV2FullPlaneLayerImages,
  PanoV2LayerRenderResult,
  PanoV2RenderedLayer,
} from './panoV2LayerRenderer';
import { buildPanoV2GapBoundaryAnalysis, computePanoV2GapCenterRow } from './panoV2LayerRenderer';

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
const BYPASS_LOW_SIGNAL_HU = 300;
const BYPASS_SAME_ARCH_RESCUE_MIN_HU = 700;
const BYPASS_SAME_ARCH_RESCUE_MARGIN_HU = 250;
const BYPASS_MISS_NEIGHBOR_RADIUS = 2;
const BYPASS_MISS_MIN_NEIGHBOR_COUNT = 8;
const BYPASS_MISS_MIN_DENSE_NEIGHBOR_COUNT = 5;
const BYPASS_MISS_DENSE_NEIGHBOR_HU = 1800;
const BYPASS_MISS_REPAIR_BLEND = 0.88;
const BYPASS_MISS_EDGE_BLEND = 0.32;
const BYPASS_DARK_POCKET_MAX_HU = 1200;
const BYPASS_DARK_POCKET_NEIGHBOR_RADIUS = 6;
const BYPASS_DARK_POCKET_MIN_DENSE_NEIGHBOR_COUNT = 12;
const BYPASS_DARK_POCKET_MIN_MEDIAN_HU = 2400;
const BYPASS_DARK_POCKET_MIN_REPAIR_GAIN_HU = 120;
const BYPASS_DARK_CLUSTER_MAX_HU = 2200;
const BYPASS_DARK_CLUSTER_MAX_PIXELS = 900;
const BYPASS_DARK_CLUSTER_MIN_PERIMETER_SAMPLES = 20;
const BYPASS_DARK_CLUSTER_MIN_MEDIAN_HU = 2200;
const BYPASS_DARK_CLUSTER_MIN_MAX_HU = 3000;
const BYPASS_DARK_CLUSTER_MIN_GAIN_HU = 120;
const BYPASS_DARK_CLUSTER_BLEND_WEIGHT = 0.5;
const BYPASS_PERITOOTH_MAX_HU = 2200;
const BYPASS_PERITOOTH_TOOTH_NEIGHBOR_HU = 3000;
const BYPASS_PERITOOTH_MIN_TOOTH_NEIGHBOR_COUNT = 6;
const BYPASS_PERITOOTH_MIN_SUPPORT_SAMPLE_COUNT = 8;
const BYPASS_PERITOOTH_MIN_GAIN_HU = 90;
const BYPASS_PERITOOTH_REPAIR_BLEND = 0.78;
const BYPASS_UNSUPPORTED_CONTEXT_MIN_HU = 1400;
const BYPASS_UNSUPPORTED_DENSE_HU = 2800;
const BYPASS_UNSUPPORTED_MIN_CONTEXT_SAMPLE_COUNT = 8;
const BYPASS_UNSUPPORTED_MIN_DENSE_SAMPLE_COUNT = 10;
const BYPASS_UNSUPPORTED_MAX_SEARCH_RADIUS = 24;
const BYPASS_UNSUPPORTED_MIN_OUTPUT_HU = 2100;
const BYPASS_UNSUPPORTED_MAX_OUTPUT_HU = 2850;
const BYPASS_SEAM_TONE_CORE_HALF_SPAN_ROWS = 12;
const BYPASS_SEAM_TONE_FEATHER_HALF_SPAN_ROWS = 28;
const BYPASS_SEAM_TONE_INPUT_LOW_HU = 700;
const BYPASS_SEAM_TONE_INPUT_HIGH_HU = 1600;
const BYPASS_SEAM_TONE_OUTPUT_LOW_HU = 2200;
const BYPASS_SEAM_TONE_OUTPUT_HIGH_HU = 2800;
const LOWER_BACKGROUND_TRIM_MIN_SIGNAL_HU = 280;
const LOWER_BACKGROUND_TRIM_MIN_COLUMN_PEAK_HU = 650;
const LOWER_BACKGROUND_TRIM_RELATIVE_SIGNAL = 0.16;
const LOWER_BACKGROUND_TRIM_MIN_LOW_RUN_ROWS = 6;
const LOWER_BACKGROUND_TRIM_BOUNDARY_MARGIN_ROWS = 6;
const LOWER_BACKGROUND_TRIM_FEATHER_ROWS = 18;
const LOWER_BACKGROUND_TRIM_SMOOTH_RADIUS_COLS = 4;
const LOWER_BACKGROUND_TRIM_MAX_ADJACENT_DELTA_ROWS = 6;

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
  gapAnalysis: PanoV2GapAnalysisDiagnostics | null;
  gapFillAnalysis: PanoV2GapFillDiagnostics | null;
  toneMappingAnalysis: PanoV2ToneMappingDiagnostics | null;
  bypassMissRepairAnalysis: PanoV2BypassMissRepairAnalysis | null;
  upperArch: PanoV2FusionBandDiagnostics;
  lowerArch: PanoV2FusionBandDiagnostics;
}

export interface PanoV2FusionResult {
  pixelData: Float32Array;
  diagnostics: PanoV2FusionDiagnostics;
  bypassMissMask: Float32Array | null;
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

export interface PanoV2GapRowRangeByColumn {
  col: number;
  upperLastDenseRow: number | null;
  lowerFirstDenseRow: number | null;
  gapHeightPx: number;
  gapMedianHu: number | null;
}

export interface PanoV2GapPatchPoint {
  col: number;
  row: number;
  finalHu: number;
  upperHu: number | null;
  lowerHu: number | null;
  source: 'upper-arch-layer' | 'lower-arch-layer' | 'gap-zone';
}

export interface PanoV2GapAnalysisDiagnostics {
  denseHuThreshold: number;
  sampleStrideColumns: number;
  gapPixelCount: number;
  gapRowRangeByColumn: PanoV2GapRowRangeByColumn[];
  fractionOfGapBelowMinus500: number;
  fractionOfGapBetweenMinus500AndZero: number;
  fractionOfGapAboveZero: number;
  patchSearchRegion: {
    colStart: number;
    colEnd: number;
  };
  patchPoints: PanoV2GapPatchPoint[];
}

export interface PanoV2GapFillDiagnostics {
  totalPixelsFilled: number;
  largestHolePx: number;
  largestHoleCol: number | null;
  upperZonePixelsFilled: number;
  lowerZonePixelsFilled: number;
}

export interface PanoV2ToneMappingDiagnostics {
  pixelsRemapped: number;
  preRemapMedianInRange: number | null;
  postRemapMedianInRange: number | null;
  seamBandPixelsLifted: number;
  seamBandPreLiftMedian: number | null;
  seamBandPostLiftMedian: number | null;
  seamBandRowStart: number;
  seamBandRowEnd: number;
}

export interface PanoV2BypassMissRepairAnalysis {
  unresolvedPixelCount: number;
  unresolvedToothEdgeFillCount: number;
  unresolvedUnsupportedBandFillCount: number;
  lowOutlierPixelCount: number;
  repairedPixelCount: number;
  darkPocketRepairCount: number;
  darkClusterRepairPixelCount: number;
  darkClusterRepairComponentCount: number;
  darkClusterBlendPixelCount: number;
  periToothRepairPixelCount: number;
  blendedPixelCount: number;
  excludedPixelCount: number;
}

function fillColumnHolesInZone(params: {
  pixelData: Float32Array;
  panoWidth: number;
  col: number;
  rowStartInclusive: number;
  rowEndExclusive: number;
  holeThresholdHu?: number;
  denseThresholdHu?: number;
  maxFillHu?: number;
}): {
  pixelsFilled: number;
  largestHolePx: number;
} {
  const {
    pixelData,
    panoWidth,
    col,
    rowStartInclusive,
    rowEndExclusive,
    holeThresholdHu = -200,
    denseThresholdHu = 200,
    maxFillHu = 300,
  } = params;
  let row = rowStartInclusive;
  let pixelsFilled = 0;
  let largestHolePx = 0;

  while (row < rowEndExclusive) {
    const value = Number(pixelData[row * panoWidth + col]);
    if (!Number.isFinite(value) || value >= holeThresholdHu) {
      row++;
      continue;
    }
    const holeStart = row;
    while (row < rowEndExclusive) {
      const holeValue = Number(pixelData[row * panoWidth + col]);
      if (!Number.isFinite(holeValue) || holeValue < holeThresholdHu) {
        row++;
        continue;
      }
      break;
    }
    const holeEndExclusive = row;

    let denseAboveRow = holeStart - 1;
    while (denseAboveRow >= rowStartInclusive) {
      const denseAboveValue = Number(pixelData[denseAboveRow * panoWidth + col]);
      if (Number.isFinite(denseAboveValue) && denseAboveValue > denseThresholdHu) {
        break;
      }
      denseAboveRow--;
    }

    let denseBelowRow = holeEndExclusive;
    while (denseBelowRow < rowEndExclusive) {
      const denseBelowValue = Number(pixelData[denseBelowRow * panoWidth + col]);
      if (Number.isFinite(denseBelowValue) && denseBelowValue > denseThresholdHu) {
        break;
      }
      denseBelowRow++;
    }

    if (denseAboveRow < rowStartInclusive || denseBelowRow >= rowEndExclusive) {
      continue;
    }

    const denseAboveValue = Number(pixelData[denseAboveRow * panoWidth + col]);
    const denseBelowValue = Number(pixelData[denseBelowRow * panoWidth + col]);
    if (!(denseAboveValue > denseThresholdHu && denseBelowValue > denseThresholdHu)) {
      continue;
    }

    const holeLength = holeEndExclusive - holeStart;
    largestHolePx = Math.max(largestHolePx, holeLength);
    for (let fillRow = holeStart; fillRow < holeEndExclusive; fillRow++) {
      const blend =
        (fillRow - denseAboveRow) / Math.max(1, denseBelowRow - denseAboveRow);
      const interpolatedValue = denseAboveValue * (1 - blend) + denseBelowValue * blend;
      pixelData[fillRow * panoWidth + col] = Math.min(maxFillHu, interpolatedValue);
      pixelsFilled++;
    }
  }

  return {
    pixelsFilled,
    largestHolePx,
  };
}

function applyBypassGapFill(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  gapCenterRow: number;
}): PanoV2GapFillDiagnostics {
  const { pixelData, panoWidth, panoHeight, gapCenterRow } = params;
  let upperZonePixelsFilled = 0;
  let lowerZonePixelsFilled = 0;
  let largestHolePx = 0;
  let largestHoleCol: number | null = null;

  for (let col = 0; col < panoWidth; col++) {
    const upperResult = fillColumnHolesInZone({
      pixelData,
      panoWidth,
      col,
      rowStartInclusive: 0,
      rowEndExclusive: clampNumber(gapCenterRow, 0, panoHeight),
    });
    upperZonePixelsFilled += upperResult.pixelsFilled;
    if (upperResult.largestHolePx > largestHolePx) {
      largestHolePx = upperResult.largestHolePx;
      largestHoleCol = col;
    }

    const lowerResult = fillColumnHolesInZone({
      pixelData,
      panoWidth,
      col,
      rowStartInclusive: clampNumber(gapCenterRow, 0, panoHeight),
      rowEndExclusive: panoHeight,
    });
    lowerZonePixelsFilled += lowerResult.pixelsFilled;
    if (lowerResult.largestHolePx > largestHolePx) {
      largestHolePx = lowerResult.largestHolePx;
      largestHoleCol = col;
    }
  }

  return {
    totalPixelsFilled: upperZonePixelsFilled + lowerZonePixelsFilled,
    largestHolePx,
    largestHoleCol,
    upperZonePixelsFilled,
    lowerZonePixelsFilled,
  };
}

function applyBypassToneMapping(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  gapCenterRow: number;
}): PanoV2ToneMappingDiagnostics {
  const { pixelData, panoWidth, panoHeight, gapCenterRow } = params;
  const preRemapValues: number[] = [];
  const postRemapValues: number[] = [];
  const seamBandPreLiftValues: number[] = [];
  const seamBandPostLiftValues: number[] = [];
  const seamBandRowStart = clampNumber(
    gapCenterRow - BYPASS_SEAM_TONE_FEATHER_HALF_SPAN_ROWS,
    0,
    Math.max(0, panoHeight - 1)
  );
  const seamBandRowEnd = clampNumber(
    gapCenterRow + BYPASS_SEAM_TONE_FEATHER_HALF_SPAN_ROWS,
    0,
    Math.max(0, panoHeight - 1)
  );

  for (let row = 0; row < panoHeight; row++) {
    const rowDistance = Math.abs(row - gapCenterRow);
    let seamLiftWeight = 0;
    if (rowDistance <= BYPASS_SEAM_TONE_CORE_HALF_SPAN_ROWS) {
      seamLiftWeight = 1;
    } else if (rowDistance <= BYPASS_SEAM_TONE_FEATHER_HALF_SPAN_ROWS) {
      seamLiftWeight =
        1 -
        (rowDistance - BYPASS_SEAM_TONE_CORE_HALF_SPAN_ROWS) /
          Math.max(
            1,
            BYPASS_SEAM_TONE_FEATHER_HALF_SPAN_ROWS - BYPASS_SEAM_TONE_CORE_HALF_SPAN_ROWS
          );
    }

    for (let col = 0; col < panoWidth; col++) {
      const index = row * panoWidth + col;
      const value = Number(pixelData[index]);
      if (!Number.isFinite(value)) {
        continue;
      }

      let remappedValue = value;
      if (value >= -500 && value < 600) {
        preRemapValues.push(value);
        remappedValue = 800 + ((value + 500) / 1100) * 400;
        postRemapValues.push(remappedValue);
      }

      if (
        seamLiftWeight > 0 &&
        remappedValue >= BYPASS_SEAM_TONE_INPUT_LOW_HU &&
        remappedValue < BYPASS_SEAM_TONE_INPUT_HIGH_HU
      ) {
        const normalizedValue =
          (remappedValue - BYPASS_SEAM_TONE_INPUT_LOW_HU) /
          Math.max(1, BYPASS_SEAM_TONE_INPUT_HIGH_HU - BYPASS_SEAM_TONE_INPUT_LOW_HU);
        const liftedTarget =
          BYPASS_SEAM_TONE_OUTPUT_LOW_HU +
          clampNumber(normalizedValue, 0, 1) *
            (BYPASS_SEAM_TONE_OUTPUT_HIGH_HU - BYPASS_SEAM_TONE_OUTPUT_LOW_HU);
        const liftedValue =
          remappedValue * (1 - seamLiftWeight) + liftedTarget * seamLiftWeight;
        if (liftedValue > remappedValue + 1) {
          seamBandPreLiftValues.push(remappedValue);
          seamBandPostLiftValues.push(liftedValue);
          remappedValue = liftedValue;
        }
      }

      if (remappedValue !== value) {
        pixelData[index] = remappedValue;
      }
    }
  }

  return {
    pixelsRemapped: preRemapValues.length,
    preRemapMedianInRange:
      preRemapValues.length > 0 ? roundTo(percentile(preRemapValues, 0.5)) : null,
    postRemapMedianInRange:
      postRemapValues.length > 0 ? roundTo(percentile(postRemapValues, 0.5)) : null,
    seamBandPixelsLifted: seamBandPreLiftValues.length,
    seamBandPreLiftMedian:
      seamBandPreLiftValues.length > 0 ? roundTo(percentile(seamBandPreLiftValues, 0.5)) : null,
    seamBandPostLiftMedian:
      seamBandPostLiftValues.length > 0 ? roundTo(percentile(seamBandPostLiftValues, 0.5)) : null,
    seamBandRowStart,
    seamBandRowEnd,
  };
}

function estimateBypassInteriorFillHu(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  row: number;
  col: number;
  rowStartInclusive: number;
  rowEndExclusive: number;
}): number | null {
  const { pixelData, panoWidth, panoHeight, row, col, rowStartInclusive, rowEndExclusive } = params;
  const preferredValues: number[] = [];
  const fallbackValues: number[] = [];

  for (let radius = 1; radius <= 18; radius++) {
    const rowMin = Math.max(rowStartInclusive, row - radius);
    const rowMax = Math.min(rowEndExclusive - 1, row + radius);
    const colMin = Math.max(0, col - radius);
    const colMax = Math.min(panoWidth - 1, col + radius);

    for (let sampleRow = rowMin; sampleRow <= rowMax; sampleRow++) {
      for (let sampleCol = colMin; sampleCol <= colMax; sampleCol++) {
        const isPerimeter =
          sampleRow === rowMin ||
          sampleRow === rowMax ||
          sampleCol === colMin ||
          sampleCol === colMax;
        if (!isPerimeter) {
          continue;
        }
        const sampleValue = Number(pixelData[sampleRow * panoWidth + sampleCol]);
        if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
          continue;
        }
        fallbackValues.push(sampleValue);
        if (sampleValue > 600 && sampleValue < 3200) {
          preferredValues.push(sampleValue);
        }
      }
    }

    if (preferredValues.length >= 8 || fallbackValues.length >= 20) {
      break;
    }
  }

  const sourceValues =
    preferredValues.length >= 4 ? preferredValues : fallbackValues.length > 0 ? fallbackValues : null;
  if (!sourceValues || sourceValues.length === 0) {
    return null;
  }

  const estimatedValue =
    sourceValues.length >= 5
      ? percentile(sourceValues, preferredValues.length >= 4 ? 0.35 : 0.5)
      : sourceValues.reduce((sum, value) => sum + value, 0) / sourceValues.length;

  return clampNumber(estimatedValue, 700, 2200);
}

function estimateBypassToothEdgeFillHu(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  row: number;
  col: number;
  rowStartInclusive: number;
  rowEndExclusive: number;
}): number | null {
  const { pixelData, panoWidth, panoHeight, row, col, rowStartInclusive, rowEndExclusive } = params;
  const contextValues: number[] = [];
  const denseValues: number[] = [];

  for (let radius = 2; radius <= 14; radius++) {
    const rowMin = Math.max(rowStartInclusive, row - radius);
    const rowMax = Math.min(rowEndExclusive - 1, row + radius);
    const colMin = Math.max(0, col - radius);
    const colMax = Math.min(panoWidth - 1, col + radius);

    for (let sampleRow = rowMin; sampleRow <= rowMax; sampleRow++) {
      if (sampleRow < 0 || sampleRow >= panoHeight) {
        continue;
      }
      for (let sampleCol = colMin; sampleCol <= colMax; sampleCol++) {
        const isPerimeter =
          sampleRow === rowMin ||
          sampleRow === rowMax ||
          sampleCol === colMin ||
          sampleCol === colMax;
        if (!isPerimeter) {
          continue;
        }
        const sampleValue = Number(pixelData[sampleRow * panoWidth + sampleCol]);
        if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
          continue;
        }
        if (sampleValue >= 2800) {
          denseValues.push(sampleValue);
        } else if (sampleValue >= 900 && sampleValue < 2800) {
          contextValues.push(sampleValue);
        }
      }
    }

    if (contextValues.length >= 6 || denseValues.length >= 12) {
      break;
    }
  }

  if (contextValues.length >= 6) {
    return clampNumber(percentile(contextValues, 0.45), 1600, 2600);
  }

  if (denseValues.length >= 12) {
    const denseReference =
      denseValues.length >= 5 ? percentile(denseValues, 0.35) : percentile(denseValues, 0.5);
    return clampNumber(denseReference * 0.68, 1800, 2600);
  }

  return null;
}

function estimateBypassUnsupportedFillHu(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  row: number;
  col: number;
  rowStartInclusive: number;
  rowEndExclusive: number;
  supportRowStart: number;
  supportRowEnd: number;
}): number | null {
  const {
    pixelData,
    panoWidth,
    panoHeight,
    row,
    col,
    rowStartInclusive,
    rowEndExclusive,
    supportRowStart,
    supportRowEnd,
  } = params;
  const outsideSupportDistance =
    row < supportRowStart ? supportRowStart - row : row > supportRowEnd ? row - supportRowEnd : 0;
  if (outsideSupportDistance <= 0) {
    return null;
  }

  const contextValues: number[] = [];
  const denseValues: number[] = [];
  const maxRadius = Math.max(
    10,
    Math.min(
      BYPASS_UNSUPPORTED_MAX_SEARCH_RADIUS,
      Math.max(1, outsideSupportDistance) + 10
    )
  );

  for (let radius = 4; radius <= maxRadius; radius++) {
    const rowMin = Math.max(rowStartInclusive, row - radius);
    const rowMax = Math.min(rowEndExclusive - 1, row + radius);
    const colMin = Math.max(0, col - radius);
    const colMax = Math.min(panoWidth - 1, col + radius);

    for (let sampleRow = rowMin; sampleRow <= rowMax; sampleRow++) {
      if (sampleRow < 0 || sampleRow >= panoHeight) {
        continue;
      }
      for (let sampleCol = colMin; sampleCol <= colMax; sampleCol++) {
        const isPerimeter =
          sampleRow === rowMin ||
          sampleRow === rowMax ||
          sampleCol === colMin ||
          sampleCol === colMax;
        if (!isPerimeter) {
          continue;
        }
        const sampleValue = Number(pixelData[sampleRow * panoWidth + sampleCol]);
        if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
          continue;
        }
        if (sampleValue >= BYPASS_UNSUPPORTED_DENSE_HU) {
          denseValues.push(sampleValue);
        } else if (sampleValue >= BYPASS_UNSUPPORTED_CONTEXT_MIN_HU) {
          contextValues.push(sampleValue);
        }
      }
    }

    if (
      contextValues.length >= BYPASS_UNSUPPORTED_MIN_CONTEXT_SAMPLE_COUNT ||
      denseValues.length >= BYPASS_UNSUPPORTED_MIN_DENSE_SAMPLE_COUNT
    ) {
      break;
    }
  }

  if (contextValues.length >= BYPASS_UNSUPPORTED_MIN_CONTEXT_SAMPLE_COUNT) {
    return clampNumber(
      percentile(contextValues, 0.35),
      1900,
      BYPASS_UNSUPPORTED_MAX_OUTPUT_HU
    );
  }

  if (denseValues.length >= BYPASS_UNSUPPORTED_MIN_DENSE_SAMPLE_COUNT) {
    const denseReference =
      denseValues.length >= 5 ? percentile(denseValues, 0.25) : percentile(denseValues, 0.5);
    return clampNumber(
      denseReference * 0.72,
      BYPASS_UNSUPPORTED_MIN_OUTPUT_HU,
      BYPASS_UNSUPPORTED_MAX_OUTPUT_HU
    );
  }

  return null;
}

function repairBypassUnresolvedPixels(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  gapCenterRow: number;
  upperBandRowStart?: number;
  upperBandRowEnd?: number;
  lowerBandRowStart?: number;
  lowerBandRowEnd?: number;
  missMask?: Float32Array | null;
}): {
  repairedPixelCount: number;
  toothEdgeFillCount: number;
  unsupportedBandFillCount: number;
} {
  const {
    pixelData,
    panoWidth,
    panoHeight,
    gapCenterRow,
    upperBandRowStart = 0,
    upperBandRowEnd = Math.max(0, gapCenterRow - 1),
    lowerBandRowStart = clampNumber(gapCenterRow, 0, Math.max(0, panoHeight - 1)),
    lowerBandRowEnd = Math.max(0, panoHeight - 1),
    missMask = null,
  } = params;
  const zoneRanges: Array<[number, number]> = [
    [0, clampNumber(gapCenterRow, 0, panoHeight)],
    [clampNumber(gapCenterRow, 0, panoHeight), panoHeight],
  ];
  let repairedPixelCount = 0;
  let toothEdgeFillCount = 0;
  let unsupportedBandFillCount = 0;

  for (let zoneIndex = 0; zoneIndex < zoneRanges.length; zoneIndex++) {
    const [rowStartInclusive, rowEndExclusive] = zoneRanges[zoneIndex];
    if (rowEndExclusive <= rowStartInclusive) {
      continue;
    }

    for (let row = rowStartInclusive; row < rowEndExclusive; row++) {
      let col = 0;
      while (col < panoWidth) {
        const index = row * panoWidth + col;
        if (Number.isFinite(pixelData[index])) {
          col++;
          continue;
        }
        const holeStart = col;
        while (col < panoWidth && !Number.isFinite(pixelData[row * panoWidth + col])) {
          col++;
        }
        const holeEndExclusive = col;
        const leftCol = holeStart - 1;
        const rightCol = holeEndExclusive;
        if (leftCol < 0 || rightCol >= panoWidth) {
          continue;
        }
        const leftValue = Number(pixelData[row * panoWidth + leftCol]);
        const rightValue = Number(pixelData[row * panoWidth + rightCol]);
        if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) {
          continue;
        }
        for (let fillCol = holeStart; fillCol < holeEndExclusive; fillCol++) {
          const blend = (fillCol - leftCol) / Math.max(1, rightCol - leftCol);
          const fillIndex = row * panoWidth + fillCol;
          pixelData[fillIndex] = leftValue * (1 - blend) + rightValue * blend;
          if (missMask) {
            missMask[fillIndex] = 1;
          }
          repairedPixelCount++;
        }
      }
    }

    for (let col = 0; col < panoWidth; col++) {
      let row = rowStartInclusive;
      while (row < rowEndExclusive) {
        const index = row * panoWidth + col;
        if (Number.isFinite(pixelData[index])) {
          row++;
          continue;
        }
        const holeStart = row;
        while (row < rowEndExclusive && !Number.isFinite(pixelData[row * panoWidth + col])) {
          row++;
        }
        const holeEndExclusive = row;
        const upperRow = holeStart - 1;
        const lowerRow = holeEndExclusive;
        if (upperRow < rowStartInclusive || lowerRow >= rowEndExclusive) {
          continue;
        }
        const upperValue = Number(pixelData[upperRow * panoWidth + col]);
        const lowerValue = Number(pixelData[lowerRow * panoWidth + col]);
        if (!Number.isFinite(upperValue) || !Number.isFinite(lowerValue)) {
          continue;
        }
        for (let fillRow = holeStart; fillRow < holeEndExclusive; fillRow++) {
          const blend = (fillRow - upperRow) / Math.max(1, lowerRow - upperRow);
          const fillIndex = fillRow * panoWidth + col;
          pixelData[fillIndex] = upperValue * (1 - blend) + lowerValue * blend;
          if (missMask) {
            missMask[fillIndex] = 1;
          }
          repairedPixelCount++;
        }
      }
    }

    const repairedSnapshot = new Float32Array(pixelData);
    for (let row = rowStartInclusive + 1; row < rowEndExclusive - 1; row++) {
      for (let col = 1; col < panoWidth - 1; col++) {
        const index = row * panoWidth + col;
        if (Number.isFinite(repairedSnapshot[index])) {
          continue;
        }
        let sum = 0;
        let count = 0;
        for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
          for (let colOffset = -1; colOffset <= 1; colOffset++) {
            if (rowOffset === 0 && colOffset === 0) {
              continue;
            }
            const sampleValue = Number(
              repairedSnapshot[(row + rowOffset) * panoWidth + (col + colOffset)]
            );
            if (!Number.isFinite(sampleValue)) {
              continue;
            }
            sum += sampleValue;
            count++;
          }
        }
        if (count >= 3) {
          pixelData[index] = sum / count;
          if (missMask) {
            missMask[index] = 1;
          }
          repairedPixelCount++;
        }
      }
    }
  }

  for (let zoneIndex = 0; zoneIndex < zoneRanges.length; zoneIndex++) {
    const [rowStartInclusive, rowEndExclusive] = zoneRanges[zoneIndex];
    for (let row = rowStartInclusive; row < rowEndExclusive; row++) {
      for (let col = 0; col < panoWidth; col++) {
        const index = row * panoWidth + col;
        if (Number.isFinite(pixelData[index])) {
          continue;
        }
        const isOuterImageBorder =
          row <= 0 || row >= panoHeight - 1 || col <= 0 || col >= panoWidth - 1;
        if (isOuterImageBorder) {
          pixelData[index] = -1000;
          if (missMask) {
            missMask[index] = 1;
          }
          repairedPixelCount++;
          continue;
        }
        const estimatedFillHu = estimateBypassInteriorFillHu({
          pixelData,
          panoWidth,
          panoHeight,
          row,
          col,
          rowStartInclusive,
          rowEndExclusive,
        });
        const toothEdgeFillHu =
          estimatedFillHu == null
            ? estimateBypassToothEdgeFillHu({
                pixelData,
                panoWidth,
                panoHeight,
                row,
                col,
                rowStartInclusive,
                rowEndExclusive,
              })
            : null;
        const unsupportedBandFillHu =
          estimatedFillHu == null && toothEdgeFillHu == null
            ? estimateBypassUnsupportedFillHu({
                pixelData,
                panoWidth,
                panoHeight,
                row,
                col,
                rowStartInclusive,
                rowEndExclusive,
                supportRowStart: zoneIndex === 0 ? upperBandRowStart : lowerBandRowStart,
                supportRowEnd: zoneIndex === 0 ? upperBandRowEnd : lowerBandRowEnd,
              })
            : null;
        pixelData[index] = estimatedFillHu ?? toothEdgeFillHu ?? unsupportedBandFillHu ?? 900;
        if (toothEdgeFillHu != null) {
          toothEdgeFillCount++;
        }
        if (unsupportedBandFillHu != null) {
          unsupportedBandFillCount++;
        }
        if (missMask) {
          missMask[index] = 1;
        }
        repairedPixelCount++;
      }
    }
  }

  return {
    repairedPixelCount,
    toothEdgeFillCount,
    unsupportedBandFillCount,
  };
}

interface PanoV2BypassNeighborhoodStats {
  count: number;
  denseCount: number;
  median: number;
  p10: number;
  p90: number;
}

interface PanoV2PeriToothNeighborhoodStats {
  toothNeighborCount: number;
  supportCount: number;
  supportMedian: number;
  supportP10: number;
  supportP90: number;
}

function summarizeBypassNeighborhood(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  row: number;
  col: number;
  radius: number;
  missMask?: Float32Array | null;
}): PanoV2BypassNeighborhoodStats | null {
  const { pixelData, panoWidth, panoHeight, row, col, radius, missMask = null } = params;
  const values: number[] = [];
  let denseCount = 0;

  for (let rowOffset = -radius; rowOffset <= radius; rowOffset++) {
    const sampleRow = row + rowOffset;
    if (sampleRow < 0 || sampleRow >= panoHeight) {
      continue;
    }
    for (let colOffset = -radius; colOffset <= radius; colOffset++) {
      const sampleCol = col + colOffset;
      if (
        sampleCol < 0 ||
        sampleCol >= panoWidth ||
        (rowOffset === 0 && colOffset === 0)
      ) {
        continue;
      }
      const sampleIndex = sampleRow * panoWidth + sampleCol;
      if (missMask && Number(missMask[sampleIndex]) > 0.5) {
        continue;
      }
      const sampleValue = Number(pixelData[sampleIndex]);
      if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
        continue;
      }
      values.push(sampleValue);
      if (sampleValue >= BYPASS_MISS_DENSE_NEIGHBOR_HU) {
        denseCount++;
      }
    }
  }

  if (values.length < BYPASS_MISS_MIN_NEIGHBOR_COUNT) {
    return null;
  }

  return {
    count: values.length,
    denseCount,
    median: percentile(values, 0.5),
    p10: percentile(values, 0.1),
    p90: percentile(values, 0.9),
  };
}

function summarizePeriToothNeighborhood(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  row: number;
  col: number;
  radius: number;
}): PanoV2PeriToothNeighborhoodStats | null {
  const { pixelData, panoWidth, panoHeight, row, col, radius } = params;
  const supportValues: number[] = [];
  let toothNeighborCount = 0;

  for (let rowOffset = -radius; rowOffset <= radius; rowOffset++) {
    const sampleRow = row + rowOffset;
    if (sampleRow < 0 || sampleRow >= panoHeight) {
      continue;
    }
    for (let colOffset = -radius; colOffset <= radius; colOffset++) {
      const sampleCol = col + colOffset;
      if (
        sampleCol < 0 ||
        sampleCol >= panoWidth ||
        (rowOffset === 0 && colOffset === 0)
      ) {
        continue;
      }
      const sampleValue = Number(pixelData[sampleRow * panoWidth + sampleCol]);
      if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
        continue;
      }
      if (sampleValue >= BYPASS_PERITOOTH_TOOTH_NEIGHBOR_HU) {
        toothNeighborCount++;
      } else if (sampleValue >= 900) {
        supportValues.push(sampleValue);
      }
    }
  }

  if (
    toothNeighborCount < BYPASS_PERITOOTH_MIN_TOOTH_NEIGHBOR_COUNT ||
    supportValues.length < BYPASS_PERITOOTH_MIN_SUPPORT_SAMPLE_COUNT
  ) {
    return null;
  }

  return {
    toothNeighborCount,
    supportCount: supportValues.length,
    supportMedian: percentile(supportValues, 0.5),
    supportP10: percentile(supportValues, 0.1),
    supportP90: percentile(supportValues, 0.9),
  };
}

function estimateSameArchBypassBaseline(params: {
  fullPlaneLayerImages: Float32Array[] | null | undefined;
  index: number;
}): number | null {
  const { fullPlaneLayerImages, index } = params;
  if (!fullPlaneLayerImages || fullPlaneLayerImages.length === 0) {
    return null;
  }

  const layerValues: number[] = [];
  for (let imageIndex = 0; imageIndex < fullPlaneLayerImages.length; imageIndex++) {
    const value = Number(fullPlaneLayerImages[imageIndex]?.[index]);
    if (!Number.isFinite(value) || value <= -950) {
      continue;
    }
    layerValues.push(value);
  }

  if (!layerValues.length) {
    return null;
  }

  return percentile(layerValues, layerValues.length >= 3 ? 0.7 : 0.5);
}

function blendRepairedBypassMissPixels(params: {
  pixelData: Float32Array;
  missMask: Float32Array;
  panoWidth: number;
  panoHeight: number;
}): number {
  const { pixelData, missMask, panoWidth, panoHeight } = params;
  const snapshot = new Float32Array(pixelData);
  let blendedPixelCount = 0;

  for (let row = 1; row < panoHeight - 1; row++) {
    for (let col = 1; col < panoWidth - 1; col++) {
      const index = row * panoWidth + col;
      if (Number(missMask[index]) <= 0.5) {
        continue;
      }
      const neighborhood = summarizeBypassNeighborhood({
        pixelData: snapshot,
        panoWidth,
        panoHeight,
        row,
        col,
        radius: 1,
        missMask: null,
      });
      if (!neighborhood) {
        continue;
      }
      pixelData[index] =
        snapshot[index] * (1 - BYPASS_MISS_EDGE_BLEND) +
        neighborhood.median * BYPASS_MISS_EDGE_BLEND;
      blendedPixelCount++;
    }
  }

  return blendedPixelCount;
}

function repairBypassMissPixels(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  gapCenterRow: number;
  fullPlaneLayerImages?: PanoV2FullPlaneLayerImages | null;
  missMask: Float32Array;
}): PanoV2BypassMissRepairAnalysis {
  const {
    pixelData,
    panoWidth,
    panoHeight,
    gapCenterRow,
    fullPlaneLayerImages = null,
    missMask,
  } = params;
  const snapshot = new Float32Array(pixelData);
  let lowOutlierPixelCount = 0;
  let repairedPixelCount = 0;

  for (let row = 1; row < panoHeight - 1; row++) {
    const sameArchLayerImages =
      row < gapCenterRow ? fullPlaneLayerImages?.upperArch : fullPlaneLayerImages?.lowerArch;
    for (let col = 1; col < panoWidth - 1; col++) {
      const index = row * panoWidth + col;
      if (Number(missMask[index]) > 0.5) {
        continue;
      }
      const currentValue = Number(snapshot[index]);
      if (!Number.isFinite(currentValue)) {
        continue;
      }
      const neighborhood = summarizeBypassNeighborhood({
        pixelData: snapshot,
        panoWidth,
        panoHeight,
        row,
        col,
        radius: BYPASS_MISS_NEIGHBOR_RADIUS,
        missMask,
      });
      if (
        !neighborhood ||
        neighborhood.denseCount < BYPASS_MISS_MIN_DENSE_NEIGHBOR_COUNT
      ) {
        continue;
      }

      const dropMargin = neighborhood.median - currentValue;
      const relativeDrop =
        neighborhood.median > 1 ? currentValue / neighborhood.median : Number.POSITIVE_INFINITY;
      const isStructuralSink =
        dropMargin >= Math.max(420, (neighborhood.median - neighborhood.p10) * 0.72) &&
        relativeDrop <= 0.74;
      if (!isStructuralSink) {
        continue;
      }

      lowOutlierPixelCount++;
      const sameArchBaseline = estimateSameArchBypassBaseline({
        fullPlaneLayerImages: sameArchLayerImages,
        index,
      });
      let repairTarget = neighborhood.median;
      if (sameArchBaseline != null) {
        const baselineWeight =
          sameArchBaseline >= neighborhood.median * 0.92
            ? 0.52
            : sameArchBaseline >= neighborhood.p10
              ? 0.34
              : 0.18;
        repairTarget =
          neighborhood.median * (1 - baselineWeight) + sameArchBaseline * baselineWeight;
      }
      repairTarget = clampNumber(
        repairTarget,
        Math.max(900, neighborhood.p10),
        Math.max(neighborhood.p90, neighborhood.median)
      );
      const repairedValue =
        currentValue * (1 - BYPASS_MISS_REPAIR_BLEND) + repairTarget * BYPASS_MISS_REPAIR_BLEND;
      if (repairedValue <= currentValue + 60) {
        continue;
      }
      pixelData[index] = repairedValue;
      missMask[index] = 1;
      repairedPixelCount++;
    }
  }

  const darkPocketSnapshot = new Float32Array(pixelData);
  let darkPocketRepairCount = 0;

  for (let row = 1; row < panoHeight - 1; row++) {
    const sameArchLayerImages =
      row < gapCenterRow ? fullPlaneLayerImages?.upperArch : fullPlaneLayerImages?.lowerArch;
    for (let col = 1; col < panoWidth - 1; col++) {
      const index = row * panoWidth + col;
      const currentValue = Number(darkPocketSnapshot[index]);
      if (!Number.isFinite(currentValue) || currentValue > BYPASS_DARK_POCKET_MAX_HU) {
        continue;
      }

      const neighborhood = summarizeBypassNeighborhood({
        pixelData: darkPocketSnapshot,
        panoWidth,
        panoHeight,
        row,
        col,
        radius: BYPASS_DARK_POCKET_NEIGHBOR_RADIUS,
        missMask: null,
      });
      if (
        !neighborhood ||
        neighborhood.denseCount < BYPASS_DARK_POCKET_MIN_DENSE_NEIGHBOR_COUNT ||
        neighborhood.median < BYPASS_DARK_POCKET_MIN_MEDIAN_HU
      ) {
        continue;
      }

      const dropMargin = neighborhood.median - currentValue;
      if (
        dropMargin < Math.max(250, (neighborhood.median - neighborhood.p10) * 0.24) ||
        neighborhood.p90 <= neighborhood.median * 0.72
      ) {
        continue;
      }

      const sameArchBaseline = estimateSameArchBypassBaseline({
        fullPlaneLayerImages: sameArchLayerImages,
        index,
      });
      let repairTarget = neighborhood.median;
      if (sameArchBaseline != null) {
        const baselineWeight =
          sameArchBaseline >= neighborhood.median * 0.92
            ? 0.5
            : sameArchBaseline >= neighborhood.p10
              ? 0.32
              : 0.16;
        repairTarget =
          neighborhood.median * (1 - baselineWeight) + sameArchBaseline * baselineWeight;
      }
      repairTarget = clampNumber(
        repairTarget,
        Math.max(1800, neighborhood.p10),
        Math.max(neighborhood.median, neighborhood.p90)
      );
      const repairedValue =
        currentValue * (1 - BYPASS_MISS_REPAIR_BLEND) + repairTarget * BYPASS_MISS_REPAIR_BLEND;
      if (repairedValue <= currentValue + BYPASS_DARK_POCKET_MIN_REPAIR_GAIN_HU) {
        continue;
      }

      pixelData[index] = repairedValue;
      missMask[index] = 1;
      darkPocketRepairCount++;
    }
  }

  const darkClusterSnapshot = new Float32Array(pixelData);
  const visited = new Uint8Array(pixelData.length);
  const darkClusterMask = new Float32Array(pixelData.length);
  let darkClusterRepairPixelCount = 0;
  let darkClusterRepairComponentCount = 0;
  let darkClusterBlendPixelCount = 0;
  let periToothRepairPixelCount = 0;
  const neighborOffsets = [
    -1, 0, 1,
  ];

  for (let row = 1; row < panoHeight - 1; row++) {
    const sameArchLayerImages =
      row < gapCenterRow ? fullPlaneLayerImages?.upperArch : fullPlaneLayerImages?.lowerArch;
    for (let col = 1; col < panoWidth - 1; col++) {
      const startIndex = row * panoWidth + col;
      if (visited[startIndex]) {
        continue;
      }
      visited[startIndex] = 1;
      const startValue = Number(darkClusterSnapshot[startIndex]);
      if (!Number.isFinite(startValue) || startValue > BYPASS_DARK_CLUSTER_MAX_HU) {
        continue;
      }

      const componentRows: number[] = [];
      const componentCols: number[] = [];
      const componentIndices: number[] = [];
      const queueRows = [row];
      const queueCols = [col];
      let queueIndex = 0;
      let componentSum = 0;

      while (queueIndex < queueRows.length && componentIndices.length <= BYPASS_DARK_CLUSTER_MAX_PIXELS) {
        const currentRow = queueRows[queueIndex];
        const currentCol = queueCols[queueIndex];
        queueIndex++;
        const currentIndex = currentRow * panoWidth + currentCol;
        const currentValue = Number(darkClusterSnapshot[currentIndex]);
        if (!Number.isFinite(currentValue) || currentValue > BYPASS_DARK_CLUSTER_MAX_HU) {
          continue;
        }
        componentRows.push(currentRow);
        componentCols.push(currentCol);
        componentIndices.push(currentIndex);
        componentSum += currentValue;

        for (let rowOffsetIndex = 0; rowOffsetIndex < neighborOffsets.length; rowOffsetIndex++) {
          const rowOffset = neighborOffsets[rowOffsetIndex];
          for (let colOffsetIndex = 0; colOffsetIndex < neighborOffsets.length; colOffsetIndex++) {
            const colOffset = neighborOffsets[colOffsetIndex];
            if (rowOffset === 0 && colOffset === 0) {
              continue;
            }
            const nextRow = currentRow + rowOffset;
            const nextCol = currentCol + colOffset;
            if (nextRow <= 0 || nextRow >= panoHeight - 1 || nextCol <= 0 || nextCol >= panoWidth - 1) {
              continue;
            }
            const nextIndex = nextRow * panoWidth + nextCol;
            if (visited[nextIndex]) {
              continue;
            }
            visited[nextIndex] = 1;
            const nextValue = Number(darkClusterSnapshot[nextIndex]);
            if (!Number.isFinite(nextValue) || nextValue > BYPASS_DARK_CLUSTER_MAX_HU) {
              continue;
            }
            queueRows.push(nextRow);
            queueCols.push(nextCol);
          }
        }
      }

      if (
        componentIndices.length === 0 ||
        componentIndices.length > BYPASS_DARK_CLUSTER_MAX_PIXELS
      ) {
        continue;
      }

      const componentIndexSet = new Set(componentIndices);
      const perimeterValues: number[] = [];
      let perimeterMax = Number.NEGATIVE_INFINITY;

      for (let componentPointIndex = 0; componentPointIndex < componentIndices.length; componentPointIndex++) {
        const currentRow = componentRows[componentPointIndex];
        const currentCol = componentCols[componentPointIndex];
        for (let rowOffsetIndex = 0; rowOffsetIndex < neighborOffsets.length; rowOffsetIndex++) {
          const rowOffset = neighborOffsets[rowOffsetIndex];
          for (let colOffsetIndex = 0; colOffsetIndex < neighborOffsets.length; colOffsetIndex++) {
            const colOffset = neighborOffsets[colOffsetIndex];
            if (rowOffset === 0 && colOffset === 0) {
              continue;
            }
            const sampleRow = currentRow + rowOffset;
            const sampleCol = currentCol + colOffset;
            if (sampleRow < 0 || sampleRow >= panoHeight || sampleCol < 0 || sampleCol >= panoWidth) {
              continue;
            }
            const sampleIndex = sampleRow * panoWidth + sampleCol;
            if (componentIndexSet.has(sampleIndex)) {
              continue;
            }
            const sampleValue = Number(darkClusterSnapshot[sampleIndex]);
            if (!Number.isFinite(sampleValue) || sampleValue <= -950) {
              continue;
            }
            perimeterValues.push(sampleValue);
            if (sampleValue > perimeterMax) {
              perimeterMax = sampleValue;
            }
          }
        }
      }

      if (
        perimeterValues.length < BYPASS_DARK_CLUSTER_MIN_PERIMETER_SAMPLES ||
        !Number.isFinite(perimeterMax)
      ) {
        continue;
      }

      const perimeterMedian = percentile(perimeterValues, 0.5);
      if (
        perimeterMedian < BYPASS_DARK_CLUSTER_MIN_MEDIAN_HU ||
        perimeterMax < BYPASS_DARK_CLUSTER_MIN_MAX_HU
      ) {
        continue;
      }

      let sameArchBaselineSum = 0;
      let sameArchBaselineCount = 0;
      for (let componentPointIndex = 0; componentPointIndex < componentIndices.length; componentPointIndex++) {
        const baseline = estimateSameArchBypassBaseline({
          fullPlaneLayerImages: sameArchLayerImages,
          index: componentIndices[componentPointIndex],
        });
        if (baseline == null) {
          continue;
        }
        sameArchBaselineSum += baseline;
        sameArchBaselineCount++;
      }
      const sameArchBaseline =
        sameArchBaselineCount > 0 ? sameArchBaselineSum / sameArchBaselineCount : null;
      let repairTarget = perimeterMedian;
      if (sameArchBaseline != null) {
        const baselineWeight =
          sameArchBaseline >= perimeterMedian * 0.92
            ? 0.48
            : sameArchBaseline >= perimeterMedian * 0.75
              ? 0.3
              : 0.15;
        repairTarget = perimeterMedian * (1 - baselineWeight) + sameArchBaseline * baselineWeight;
      }
      repairTarget = clampNumber(
        repairTarget,
        Math.max(1800, percentile(perimeterValues, 0.1)),
        Math.max(perimeterMedian, percentile(perimeterValues, 0.9))
      );

      const componentMean = componentSum / Math.max(1, componentIndices.length);
      if (repairTarget <= componentMean + BYPASS_DARK_CLUSTER_MIN_GAIN_HU) {
        continue;
      }

      let repairedPixelsThisComponent = 0;
      for (let componentPointIndex = 0; componentPointIndex < componentIndices.length; componentPointIndex++) {
        const currentIndex = componentIndices[componentPointIndex];
        const currentValue = Number(pixelData[currentIndex]);
        const repairedValue =
          currentValue * (1 - BYPASS_MISS_REPAIR_BLEND) + repairTarget * BYPASS_MISS_REPAIR_BLEND;
        if (repairedValue <= currentValue + BYPASS_DARK_CLUSTER_MIN_GAIN_HU * 0.5) {
          continue;
        }
        pixelData[currentIndex] = repairedValue;
        missMask[currentIndex] = 1;
        darkClusterMask[currentIndex] = 1;
        darkClusterRepairPixelCount++;
        repairedPixelsThisComponent++;
      }

      if (repairedPixelsThisComponent > 0) {
        darkClusterRepairComponentCount++;
      }
    }
  }

  if (darkClusterRepairPixelCount > 0) {
    const clusterBlendSnapshot = new Float32Array(pixelData);
    for (let row = 2; row < panoHeight - 2; row++) {
      for (let col = 2; col < panoWidth - 2; col++) {
        const index = row * panoWidth + col;
        if (Number(darkClusterMask[index]) <= 0.5) {
          continue;
        }
        const neighborhood = summarizeBypassNeighborhood({
          pixelData: clusterBlendSnapshot,
          panoWidth,
          panoHeight,
          row,
          col,
          radius: 2,
          missMask: null,
        });
        if (!neighborhood || neighborhood.median < BYPASS_DARK_CLUSTER_MIN_MEDIAN_HU) {
          continue;
        }
        pixelData[index] =
          clusterBlendSnapshot[index] * (1 - BYPASS_DARK_CLUSTER_BLEND_WEIGHT) +
          neighborhood.median * BYPASS_DARK_CLUSTER_BLEND_WEIGHT;
        darkClusterBlendPixelCount++;
      }
    }
  }

  const periToothSnapshot = new Float32Array(pixelData);
  for (let row = 1; row < panoHeight - 1; row++) {
    for (let col = 1; col < panoWidth - 1; col++) {
      const index = row * panoWidth + col;
      if (Number(missMask[index]) > 0.5) {
        continue;
      }
      const currentValue = Number(periToothSnapshot[index]);
      if (!Number.isFinite(currentValue) || currentValue > BYPASS_PERITOOTH_MAX_HU) {
        continue;
      }
      const neighborhood = summarizePeriToothNeighborhood({
        pixelData: periToothSnapshot,
        panoWidth,
        panoHeight,
        row,
        col,
        radius: 2,
      });
      if (!neighborhood) {
        continue;
      }
      const repairTarget = clampNumber(
        Math.max(2200, neighborhood.supportMedian),
        Math.max(1800, neighborhood.supportP10),
        Math.max(neighborhood.supportMedian, neighborhood.supportP90)
      );
      const repairedValue =
        currentValue * (1 - BYPASS_PERITOOTH_REPAIR_BLEND) +
        repairTarget * BYPASS_PERITOOTH_REPAIR_BLEND;
      if (repairedValue <= currentValue + BYPASS_PERITOOTH_MIN_GAIN_HU) {
        continue;
      }
      pixelData[index] = repairedValue;
      missMask[index] = 1;
      periToothRepairPixelCount++;
    }
  }

  const blendedPixelCount =
    repairedPixelCount > 0 ||
    darkPocketRepairCount > 0 ||
    darkClusterRepairPixelCount > 0 ||
    periToothRepairPixelCount > 0
      ? blendRepairedBypassMissPixels({
          pixelData,
          missMask,
          panoWidth,
          panoHeight,
        })
      : 0;
  let excludedPixelCount = 0;
  for (let index = 0; index < missMask.length; index++) {
    if (Number(missMask[index]) > 0.5) {
      excludedPixelCount++;
    }
  }

  return {
    unresolvedPixelCount: 0,
    unresolvedToothEdgeFillCount: 0,
    unresolvedUnsupportedBandFillCount: 0,
    lowOutlierPixelCount,
    repairedPixelCount,
    darkPocketRepairCount,
    darkClusterRepairPixelCount,
    darkClusterRepairComponentCount,
    darkClusterBlendPixelCount,
    periToothRepairPixelCount,
    blendedPixelCount,
    excludedPixelCount,
  };
}

function suppressLowerBackgroundTail(params: {
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  gapCenterRow: number;
  layerRender: PanoV2LayerRenderResult;
  fullPlaneLayerImages?: PanoV2FullPlaneLayerImages | null;
  exclusionMask?: Float32Array | null;
}): void {
  const {
    pixelData,
    panoWidth,
    panoHeight,
    gapCenterRow,
    layerRender,
    fullPlaneLayerImages = null,
    exclusionMask = null,
  } = params;
  const lowerLayers = fullPlaneLayerImages?.lowerArch;
  if (!lowerLayers || lowerLayers.length === 0 || panoWidth <= 0 || panoHeight <= 0) {
    return;
  }

  const lowerAnchorRow = clampNumber(
    Math.round(layerRender.lowerArch.anchorRow),
    clampNumber(gapCenterRow, 0, Math.max(0, panoHeight - 1)),
    Math.max(0, panoHeight - 1)
  );
  const lowerRowEnd = clampNumber(
    Math.round(layerRender.lowerArch.rowEnd),
    lowerAnchorRow,
    Math.max(0, panoHeight - 1)
  );
  const rawBoundaryByCol = new Int16Array(Math.max(1, panoWidth));
  const smoothedBoundaryByCol = new Int16Array(Math.max(1, panoWidth));
  const baselineFloor = Math.min(
    Math.max(0, panoHeight - 1),
    lowerRowEnd + LOWER_BACKGROUND_TRIM_BOUNDARY_MARGIN_ROWS
  );

  for (let col = 0; col < panoWidth; col++) {
    const baselineValues: number[] = [];
    let peakBaseline = Number.NEGATIVE_INFINITY;

    for (let row = lowerAnchorRow; row <= lowerRowEnd; row++) {
      const baseline = estimateSameArchBypassBaseline({
        fullPlaneLayerImages: lowerLayers,
        index: row * panoWidth + col,
      });
      if (!Number.isFinite(baseline ?? Number.NaN)) {
        continue;
      }
      const finiteBaseline = Number(baseline);
      baselineValues.push(finiteBaseline);
      if (finiteBaseline > peakBaseline) {
        peakBaseline = finiteBaseline;
      }
    }

    if (peakBaseline < LOWER_BACKGROUND_TRIM_MIN_COLUMN_PEAK_HU || baselineValues.length === 0) {
      rawBoundaryByCol[col] = baselineFloor;
      continue;
    }

    const anatomyThreshold = clampNumber(
      Math.max(
        LOWER_BACKGROUND_TRIM_MIN_SIGNAL_HU,
        peakBaseline * LOWER_BACKGROUND_TRIM_RELATIVE_SIGNAL
      ),
      LOWER_BACKGROUND_TRIM_MIN_SIGNAL_HU,
      950
    );
    let lastSupportedRow = lowerAnchorRow;
    let lowRunLength = 0;

    for (let row = lowerAnchorRow; row <= lowerRowEnd; row++) {
      const baseline = estimateSameArchBypassBaseline({
        fullPlaneLayerImages: lowerLayers,
        index: row * panoWidth + col,
      });
      if (baseline != null && baseline >= anatomyThreshold) {
        lastSupportedRow = row;
        lowRunLength = 0;
        continue;
      }

      lowRunLength++;
      if (lowRunLength >= LOWER_BACKGROUND_TRIM_MIN_LOW_RUN_ROWS && row > lastSupportedRow + 2) {
        break;
      }
    }

    rawBoundaryByCol[col] = clampNumber(
      lastSupportedRow + LOWER_BACKGROUND_TRIM_BOUNDARY_MARGIN_ROWS,
      lowerAnchorRow,
      Math.max(lowerAnchorRow, panoHeight - 1)
    );
  }

  for (let col = 0; col < panoWidth; col++) {
    const neighborhood: number[] = [];
    for (
      let sampleCol = Math.max(0, col - LOWER_BACKGROUND_TRIM_SMOOTH_RADIUS_COLS);
      sampleCol <= Math.min(panoWidth - 1, col + LOWER_BACKGROUND_TRIM_SMOOTH_RADIUS_COLS);
      sampleCol++
    ) {
      neighborhood.push(Number(rawBoundaryByCol[sampleCol]));
    }
    smoothedBoundaryByCol[col] = clampNumber(
      Math.round(percentile(neighborhood, 0.5)),
      lowerAnchorRow,
      Math.max(lowerAnchorRow, panoHeight - 1)
    );
  }

  for (let col = 1; col < panoWidth; col++) {
    const previous = Number(smoothedBoundaryByCol[col - 1]);
    smoothedBoundaryByCol[col] = clampNumber(
      Number(smoothedBoundaryByCol[col]),
      previous - LOWER_BACKGROUND_TRIM_MAX_ADJACENT_DELTA_ROWS,
      previous + LOWER_BACKGROUND_TRIM_MAX_ADJACENT_DELTA_ROWS
    );
  }
  for (let col = panoWidth - 2; col >= 0; col--) {
    const next = Number(smoothedBoundaryByCol[col + 1]);
    smoothedBoundaryByCol[col] = clampNumber(
      Number(smoothedBoundaryByCol[col]),
      next - LOWER_BACKGROUND_TRIM_MAX_ADJACENT_DELTA_ROWS,
      next + LOWER_BACKGROUND_TRIM_MAX_ADJACENT_DELTA_ROWS
    );
  }

  for (let col = 0; col < panoWidth; col++) {
    const boundaryRow = Number(smoothedBoundaryByCol[col]);
    for (let row = boundaryRow + 1; row < panoHeight; row++) {
      const index = row * panoWidth + col;
      const currentValue = Number(pixelData[index]);
      if (!Number.isFinite(currentValue)) {
        continue;
      }

      const distance = row - boundaryRow;
      if (distance <= LOWER_BACKGROUND_TRIM_FEATHER_ROWS) {
        const blendWeight = smoothstepUnit(
          distance / Math.max(1, LOWER_BACKGROUND_TRIM_FEATHER_ROWS)
        );
        pixelData[index] = currentValue * (1 - blendWeight) + -1000 * blendWeight;
      } else {
        pixelData[index] = -1000;
      }

      if (exclusionMask) {
        exclusionMask[index] = 1;
      }
    }
  }
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

function roundNullable(value: number | null): number | null {
  return Number.isFinite(value ?? Number.NaN) ? roundTo(Number(value)) : null;
}

function inferGapPointSource(params: {
  upperValue: number;
  lowerValue: number;
}): 'upper-arch-layer' | 'lower-arch-layer' | 'gap-zone' {
  const { upperValue, lowerValue } = params;
  const upperFinite = Number.isFinite(upperValue);
  const lowerFinite = Number.isFinite(lowerValue);
  if (!upperFinite && !lowerFinite) {
    return 'gap-zone';
  }
  if (upperFinite && !lowerFinite) {
    return 'upper-arch-layer';
  }
  if (!upperFinite && lowerFinite) {
    return 'lower-arch-layer';
  }
  if (upperValue > lowerValue + 1e-3) {
    return 'upper-arch-layer';
  }
  if (lowerValue > upperValue + 1e-3) {
    return 'lower-arch-layer';
  }
  return 'gap-zone';
}

function buildPanoV2GapAnalysis(params: {
  panoWidth: number;
  panoHeight: number;
  finalPixelData: Float32Array;
  upperArchImage: Float32Array;
  lowerArchImage: Float32Array;
}): PanoV2GapAnalysisDiagnostics {
  const { panoWidth, panoHeight, finalPixelData, upperArchImage, lowerArchImage } = params;
  const boundaryAnalysis: PanoV2GapBoundaryAnalysis = buildPanoV2GapBoundaryAnalysis({
    upperArchImage,
    lowerArchImage,
    panoWidth,
    panoHeight,
  });
  let gapPixelCount = 0;
  let gapBelowMinus500Count = 0;
  let gapBetweenMinus500AndZeroCount = 0;
  let gapAboveZeroCount = 0;

  for (let col = 0; col < panoWidth; col++) {
    const upperLastDenseRow = Number(boundaryAnalysis.upperLastDenseRowByCol[col]);
    const lowerFirstDenseRow = Number(boundaryAnalysis.lowerFirstDenseRowByCol[col]);
    if (upperLastDenseRow < 0 || lowerFirstDenseRow < 0 || lowerFirstDenseRow <= upperLastDenseRow + 1) {
      continue;
    }
    for (let row = upperLastDenseRow + 1; row < lowerFirstDenseRow; row++) {
      const value = Number(finalPixelData[row * panoWidth + col]);
      if (!Number.isFinite(value)) {
        continue;
      }
      gapPixelCount++;
      if (value <= -500) {
        gapBelowMinus500Count++;
      } else if (value <= 0) {
        gapBetweenMinus500AndZeroCount++;
      } else {
        gapAboveZeroCount++;
      }
    }
  }

  const gapRowRangeByColumn: PanoV2GapRowRangeByColumn[] = boundaryAnalysis.sampledColumns.map(sample => {
    const gapValues: number[] = [];
    if (
      sample.upperLastDenseRow !== null &&
      sample.lowerFirstDenseRow !== null &&
      sample.lowerFirstDenseRow > sample.upperLastDenseRow + 1
    ) {
      for (let row = sample.upperLastDenseRow + 1; row < sample.lowerFirstDenseRow; row++) {
        const value = Number(finalPixelData[row * panoWidth + sample.col]);
        if (Number.isFinite(value)) {
          gapValues.push(value);
        }
      }
    }
    return {
      col: sample.col,
      upperLastDenseRow: sample.upperLastDenseRow,
      lowerFirstDenseRow: sample.lowerFirstDenseRow,
      gapHeightPx: sample.gapHeightPx,
      gapMedianHu: gapValues.length > 0 ? roundTo(percentile(gapValues, 0.5)) : null,
    };
  });

  const patchSearchRegion = {
    colStart: clampNumber(Math.round((panoWidth - 1) * 0.18), 0, Math.max(0, panoWidth - 1)),
    colEnd: clampNumber(Math.round((panoWidth - 1) * 0.58), 0, Math.max(0, panoWidth - 1)),
  };
  const patchCandidates: PanoV2GapPatchPoint[] = [];
  for (let col = patchSearchRegion.colStart; col <= patchSearchRegion.colEnd; col++) {
    const upperLastDenseRow = Number(boundaryAnalysis.upperLastDenseRowByCol[col]);
    const lowerFirstDenseRow = Number(boundaryAnalysis.lowerFirstDenseRowByCol[col]);
    if (upperLastDenseRow < 0 || lowerFirstDenseRow < 0 || lowerFirstDenseRow <= upperLastDenseRow + 1) {
      continue;
    }
    let lowestGapHu = Number.POSITIVE_INFINITY;
    let lowestGapRow = -1;
    for (let row = upperLastDenseRow + 1; row < lowerFirstDenseRow; row++) {
      const value = Number(finalPixelData[row * panoWidth + col]);
      if (!Number.isFinite(value) || value >= lowestGapHu) {
        continue;
      }
      lowestGapHu = value;
      lowestGapRow = row;
    }
    if (!Number.isFinite(lowestGapHu) || lowestGapRow < 0 || lowestGapHu > -500) {
      continue;
    }
    const index = lowestGapRow * panoWidth + col;
    const upperValue = Number(upperArchImage[index]);
    const lowerValue = Number(lowerArchImage[index]);
    patchCandidates.push({
      col,
      row: lowestGapRow,
      finalHu: roundTo(lowestGapHu),
      upperHu: roundNullable(upperValue),
      lowerHu: roundNullable(lowerValue),
      source: inferGapPointSource({
        upperValue,
        lowerValue,
      }),
    });
  }
  patchCandidates.sort((left, right) => {
    if (left.finalHu !== right.finalHu) {
      return left.finalHu - right.finalHu;
    }
    if (left.col !== right.col) {
      return left.col - right.col;
    }
    return left.row - right.row;
  });

  return {
    denseHuThreshold: boundaryAnalysis.denseHuThreshold,
    sampleStrideColumns: boundaryAnalysis.sampleStrideColumns,
    gapPixelCount,
    gapRowRangeByColumn,
    fractionOfGapBelowMinus500:
      gapPixelCount > 0 ? roundTo(gapBelowMinus500Count / gapPixelCount, 4) : 0,
    fractionOfGapBetweenMinus500AndZero:
      gapPixelCount > 0 ? roundTo(gapBetweenMinus500AndZeroCount / gapPixelCount, 4) : 0,
    fractionOfGapAboveZero: gapPixelCount > 0 ? roundTo(gapAboveZeroCount / gapPixelCount, 4) : 0,
    patchSearchRegion,
    patchPoints: patchCandidates.slice(0, 16),
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
  bypassCompositeImages?: PanoV2BypassCompositeImages | null;
}): PanoV2FusionResult {
  const { panoWidth, panoHeight, layerRender, fullPlaneLayerImages, bypassCompositeImages } = params;
  if (layerRender.model === 'thick-slab-mip-render' && bypassCompositeImages) {
    const planeSize = Math.max(1, panoWidth * panoHeight);
    const pixelData = new Float32Array(planeSize);
    const bypassMissMask = new Float32Array(planeSize);
    const gapCenterRow = computePanoV2GapCenterRow(layerRender);
    for (let index = 0; index < planeSize; index++) {
      const row = Math.floor(index / Math.max(1, panoWidth));
      const useUpperArch = row < gapCenterRow;
      let selectedValue = Number.NEGATIVE_INFINITY;
      let hasFiniteValue = false;
      const sourceImages = useUpperArch
        ? [bypassCompositeImages.upperArch]
        : [bypassCompositeImages.lowerArch];
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
      const fallbackImages = fullPlaneLayerImages
        ? useUpperArch
          ? fullPlaneLayerImages.upperArch
          : fullPlaneLayerImages.lowerArch
        : [];
      let strongestFallbackValue = Number.NEGATIVE_INFINITY;
      let hasStrongFallbackValue = false;
      if (fallbackImages.length > 0) {
        for (let imageIndex = 0; imageIndex < fallbackImages.length; imageIndex++) {
          const value = Number(fallbackImages[imageIndex]?.[index]);
          if (!Number.isFinite(value) || value <= -950) {
            continue;
          }
          if (value > strongestFallbackValue) {
            strongestFallbackValue = value;
            hasStrongFallbackValue = true;
          }
          if (!hasFiniteValue && value > selectedValue) {
            selectedValue = value;
            hasFiniteValue = true;
          }
        }
      }
      if (
        hasFiniteValue &&
        selectedValue < BYPASS_LOW_SIGNAL_HU &&
        hasStrongFallbackValue &&
        strongestFallbackValue >= BYPASS_SAME_ARCH_RESCUE_MIN_HU &&
        strongestFallbackValue >= selectedValue + BYPASS_SAME_ARCH_RESCUE_MARGIN_HU
      ) {
        selectedValue = strongestFallbackValue;
      }
      pixelData[index] = hasFiniteValue ? clampNumber(selectedValue, -1000, 3500) : Number.NaN;
    }
    const unresolvedRepairAnalysis = repairBypassUnresolvedPixels({
      pixelData,
      panoWidth,
      panoHeight,
      gapCenterRow,
      upperBandRowStart: layerRender.upperArch.rowStart,
      upperBandRowEnd: layerRender.upperArch.rowEnd,
      lowerBandRowStart: layerRender.lowerArch.rowStart,
      lowerBandRowEnd: layerRender.lowerArch.rowEnd,
      missMask: bypassMissMask,
    });
    const gapFillAnalysis = applyBypassGapFill({
      pixelData,
      panoWidth,
      panoHeight,
      gapCenterRow,
    });
    const bypassMissRepairAnalysis = repairBypassMissPixels({
      pixelData,
      panoWidth,
      panoHeight,
      gapCenterRow,
      fullPlaneLayerImages,
      missMask: bypassMissMask,
    });
    bypassMissRepairAnalysis.unresolvedPixelCount = unresolvedRepairAnalysis.repairedPixelCount;
    bypassMissRepairAnalysis.unresolvedToothEdgeFillCount =
      unresolvedRepairAnalysis.toothEdgeFillCount;
    bypassMissRepairAnalysis.unresolvedUnsupportedBandFillCount =
      unresolvedRepairAnalysis.unsupportedBandFillCount;
    const toneMappingAnalysis = applyBypassToneMapping({
      pixelData,
      panoWidth,
      panoHeight,
      gapCenterRow,
    });
    suppressLowerBackgroundTail({
      pixelData,
      panoWidth,
      panoHeight,
      gapCenterRow,
      layerRender,
      fullPlaneLayerImages,
      exclusionMask: bypassMissMask,
    });
    const finalValues = Array.from(pixelData);

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
    const gapAnalysis = buildPanoV2GapAnalysis({
      panoWidth,
      panoHeight,
      finalPixelData: pixelData,
      upperArchImage: bypassCompositeImages.upperArch,
      lowerArchImage: bypassCompositeImages.lowerArch,
    });

    return {
      pixelData,
      diagnostics: {
        enabled: true,
        phase: 4,
        model: 'fusion-and-roi-composition',
        implementationVersion: '2026-04-07-fusion-band-support-fallback-6',
        renderBypass: true,
        outputCoverageFraction: 1,
        overlapFraction: 0,
        gapCenterRow,
        ghostingRisk: 'low',
        normalizationHuWindow: {
          lower: -1000,
          upper: 3500,
        },
        outputRange: summarize(finalValues),
        rowCoverageByRow,
        gapAnalysis,
        gapFillAnalysis,
        toneMappingAnalysis,
        bypassMissRepairAnalysis,
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
      bypassMissMask,
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
      gapAnalysis: null,
      gapFillAnalysis: null,
      toneMappingAnalysis: null,
      bypassMissRepairAnalysis: null,
      upperArch: upperBand.diagnostic,
      lowerArch: lowerBand.diagnostic,
    },
    bypassMissMask: null,
  };
}
