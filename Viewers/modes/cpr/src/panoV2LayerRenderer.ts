import type { PanoV2NumericSummary, PanoV2TrackLabel } from './panoV2Geometry';
import type { PanoV2StraightenedVolume, PanoV2StraightenedVolumeBand } from './panoV2Volume';

export const PANO_V2_LAYER_COUNT_PER_ARCH = 3;

const LAYER_OFFSET_FACTORS = [-0.45, 0, 0.45] as const;
const REGION_POSTERIOR_CUTOFF = 0.22;
const REGION_ANTERIOR_MIN = 0.35;
const REGION_ANTERIOR_MAX = 0.65;
const REGION_RIGHT_POSTERIOR_MIN = 0.78;
const TOOTH_ROW_BAND_ROW_RADIUS = 18;
const PANO_V2_MIP_SLAB_HALF_WIDTH_MM = 10;
const PANO_V2_MIP_DEPTH_STEP_MM = 0.3;
const PANO_V2_GAP_ANALYSIS_DENSE_HU_THRESHOLD = 500;
const PANO_V2_GAP_ANALYSIS_SAMPLE_STRIDE_COLUMNS = 10;

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
}): Float32Array {
  const { volume, centerDepthMmByCol, targetOffsetMmByCol } = params;
  const width = volume.panoWidth;
  const height = volume.panoHeight;
  const image = new Float32Array(Math.max(1, width * height));
  image.fill(Number.NaN);

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
      const rowOffsetMm = volume.effectiveVerticalHalfMm - row * volume.vertStepMm;
      const verticalOffsetMm = columnVerticalCenterOffsetMm + rowOffsetMm;
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
        image[row * width + col] = mipHu;
      }
    }
  }

  return image;
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

function buildBypassTargetOffsetMmByCol(
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
  const upperTargetOffsetMmByCol = buildBypassTargetOffsetMmByCol(
    params.result.upperArch,
    params.volume.panoWidth
  );
  const lowerTargetOffsetMmByCol = buildBypassTargetOffsetMmByCol(
    params.result.lowerArch,
    params.volume.panoWidth
  );

  return {
    upperArch: buildFullPlaneMipImage({
      volume: params.volume,
      centerDepthMmByCol: params.volume.upperArch.centerDepthMmByCol,
      targetOffsetMmByCol: upperTargetOffsetMmByCol,
    }),
    lowerArch: buildFullPlaneMipImage({
      volume: params.volume,
      centerDepthMmByCol: params.volume.lowerArch.centerDepthMmByCol,
      targetOffsetMmByCol: lowerTargetOffsetMmByCol,
    }),
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
