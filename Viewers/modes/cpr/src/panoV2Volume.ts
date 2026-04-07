import type { PanoV2NumericSummary, PanoV2TrackLabel } from './panoV2Geometry';

export interface PanoV2StraightenedFrame {
  position: [number, number, number];
  slabDir: [number, number, number];
  mipSlabDir?: [number, number, number];
  verticalDir: [number, number, number];
  columnVerticalCenterOffsetMm: number;
}

export interface PanoV2StraightenedTrack {
  label: PanoV2TrackLabel;
  anchorRow: number;
  centerDepthMmByCol: Float32Array;
  envelopeHalfWidthMmByCol?: Float32Array | null;
  searchDepthHalfRangeMm?: number;
}

export interface PanoV2StraightenedVolumeInput {
  panoWidth: number;
  panoHeight: number;
  effectiveVerticalHalfMm: number;
  vertStepMm: number;
  rowHalfSpanMm: number;
  depthSampleCount: number;
  frames: PanoV2StraightenedFrame[];
  sampleWorldIntensity: (wx: number, wy: number, wz: number) => number;
  upperArch: PanoV2StraightenedTrack;
  lowerArch: PanoV2StraightenedTrack;
}

export interface PanoV2StraightenedVolumeBand {
  label: PanoV2TrackLabel;
  anchorRow: number;
  localAnchorRow: number;
  rowStart: number;
  rowEnd: number;
  rowCount: number;
  depthCount: number;
  rowHalfSpanMm: number;
  depthHalfRangeMm: number;
  rowOffsetsMm: Float32Array;
  depthOffsetsMm: Float32Array;
  centerDepthMmByCol: Float32Array;
  envelopeHalfWidthMmByCol: Float32Array | null;
  valuesHu: Float32Array;
  finiteSampleFraction: number;
  valueRangeHu: PanoV2NumericSummary;
  centerDepthSummaryMm: PanoV2NumericSummary;
  envelopeHalfWidthSummaryMm: PanoV2NumericSummary;
}

export interface PanoV2StraightenedVolume {
  enabled: true;
  phase: 2;
  model: 'straightened-local-volume';
  panoWidth: number;
  panoHeight: number;
  sampler: 'world-hu-trilinear';
  effectiveVerticalHalfMm: number;
  vertStepMm: number;
  frames: PanoV2StraightenedFrame[];
  sampleWorldIntensity: (wx: number, wy: number, wz: number) => number;
  upperArch: PanoV2StraightenedVolumeBand;
  lowerArch: PanoV2StraightenedVolumeBand;
}

interface PanoV2StraightenedVolumeRowProbe {
  localRow: number;
  panoRow: number;
  rowOffsetMm: number;
  depthValuesHu: Array<number | null>;
}

interface PanoV2StraightenedVolumeColumnProbe {
  col: number;
  centerDepthMm: number;
  representativeRows: PanoV2StraightenedVolumeRowProbe[];
}

interface PanoV2StraightenedVolumeBandDiagnostics {
  label: PanoV2TrackLabel;
  anchorRow: number;
  localAnchorRow: number;
  rowStart: number;
  rowEnd: number;
  rowCount: number;
  depthCount: number;
  rowHalfSpanMm: number;
  depthHalfRangeMm: number;
  rowOffsetsFirst8Mm: number[];
  rowOffsetsLast8Mm: number[];
  depthOffsetsMm: number[];
  finiteSampleFraction: number;
  centerDepthSummaryMm: PanoV2NumericSummary;
  envelopeHalfWidthSummaryMm: PanoV2NumericSummary;
  valueRangeHu: PanoV2NumericSummary;
  sampledColumns: PanoV2StraightenedVolumeColumnProbe[];
}

export interface PanoV2StraightenedVolumeDiagnostics {
  enabled: true;
  phase: 2;
  model: 'straightened-local-volume';
  panoWidth: number;
  panoHeight: number;
  sampler: 'world-hu-trilinear';
  upperArch: PanoV2StraightenedVolumeBandDiagnostics;
  lowerArch: PanoV2StraightenedVolumeBandDiagnostics;
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
  if (values.length === 0) {
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

function summarizeFinite(values: ArrayLike<number>): PanoV2NumericSummary {
  const finiteValues: number[] = [];
  for (let index = 0; index < values.length; index++) {
    const value = Number(values[index]);
    if (Number.isFinite(value)) {
      finiteValues.push(value);
    }
  }
  if (!finiteValues.length) {
    return { min: 0, p50: 0, p90: 0, max: 0, mean: 0 };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  for (const value of finiteValues) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  }
  return {
    min: roundTo(min),
    p50: roundTo(percentile(finiteValues, 0.5)),
    p90: roundTo(percentile(finiteValues, 0.9)),
    max: roundTo(max),
    mean: roundTo(sum / finiteValues.length),
  };
}

function buildDepthOffsetsMm(
  envelopeHalfWidthMmByCol: Float32Array | null | undefined,
  depthSampleCount: number,
  searchDepthHalfRangeMm?: number
): { depthHalfRangeMm: number; depthOffsetsMm: Float32Array } {
  const envelopeSummary = summarizeFinite(envelopeHalfWidthMmByCol ?? []);
  const envelopeP90 = envelopeSummary.p90 > 0 ? envelopeSummary.p90 : 1.35;
  const envelopeP50 = envelopeSummary.p50 > 0 ? envelopeSummary.p50 : envelopeP90;
  const envelopeMax = envelopeSummary.max > 0 ? envelopeSummary.max : envelopeP90;
  const preferredSearchHalfRangeMm = Number.isFinite(searchDepthHalfRangeMm)
    ? Math.max(1.5, Number(searchDepthHalfRangeMm))
    : Math.max(envelopeMax * 1.2, envelopeP90 * 1.35, 2.45);
  const envelopeDrivenHalfRangeMm = Math.max(
    envelopeP90 * 1.45 + envelopeP50 * 0.25,
    envelopeMax * 1.12 + 0.35,
    preferredSearchHalfRangeMm * 0.55
  );
  const depthHalfRangeMm = clampNumber(
    envelopeDrivenHalfRangeMm,
    Math.min(preferredSearchHalfRangeMm, Math.max(1.35, preferredSearchHalfRangeMm * 0.24)),
    preferredSearchHalfRangeMm
  );
  const safeDepthSampleCount = Math.max(3, depthSampleCount | 0);
  const depthOffsetsMm = new Float32Array(safeDepthSampleCount);
  const denominator = Math.max(1, safeDepthSampleCount - 1);
  for (let depthIndex = 0; depthIndex < safeDepthSampleCount; depthIndex++) {
    const fraction = depthIndex / denominator;
    depthOffsetsMm[depthIndex] = -depthHalfRangeMm + fraction * (depthHalfRangeMm * 2);
  }
  return {
    depthHalfRangeMm,
    depthOffsetsMm,
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

function buildRowProbeIndices(rowCount: number, localAnchorRow: number): number[] {
  const indices = new Set<number>();
  if (rowCount > 0) {
    indices.add(0);
    indices.add(rowCount - 1);
    indices.add(clampNumber(localAnchorRow, 0, rowCount - 1));
    indices.add(clampNumber(Math.round((rowCount - 1) * 0.25), 0, rowCount - 1));
    indices.add(clampNumber(Math.round((rowCount - 1) * 0.75), 0, rowCount - 1));
  }
  return Array.from(indices).sort((left, right) => left - right);
}

function straightenedBandIndex(
  col: number,
  rowIndex: number,
  depthIndex: number,
  rowCount: number,
  depthCount: number
): number {
  return ((col * rowCount + rowIndex) * depthCount + depthIndex) | 0;
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

function resolveBandRowSpanMm(label: PanoV2TrackLabel, rowHalfSpanMm: number): {
  aboveMm: number;
  belowMm: number;
} {
  const safeHalfSpanMm = Math.max(6, rowHalfSpanMm);
  if (label === 'upperArch') {
    return {
      // Expand the tooth/root capture zone without reopening the far superior air field.
      aboveMm: clampNumber(safeHalfSpanMm * 0.74, 6.5, 8.6),
      belowMm: clampNumber(safeHalfSpanMm * 1.14, 10.5, 14.5),
    };
  }
  return {
    // Recover more of the lower tooth-bearing ridge above the anchor while keeping the lower tail shorter.
    aboveMm: clampNumber(safeHalfSpanMm * 1.16, 10.5, 14.8),
    belowMm: clampNumber(safeHalfSpanMm * 0.74, 6.5, 9),
  };
}

function buildStraightenedBand(params: {
  panoWidth: number;
  panoHeight: number;
  effectiveVerticalHalfMm: number;
  vertStepMm: number;
  rowHalfSpanMm: number;
  depthSampleCount: number;
  frames: PanoV2StraightenedFrame[];
  sampleWorldIntensity: (wx: number, wy: number, wz: number) => number;
  track: PanoV2StraightenedTrack;
}): PanoV2StraightenedVolumeBand {
  const {
    panoWidth,
    panoHeight,
    effectiveVerticalHalfMm,
    vertStepMm,
    rowHalfSpanMm,
    depthSampleCount,
    frames,
    sampleWorldIntensity,
    track,
  } = params;
  const rowSpanMm = resolveBandRowSpanMm(track.label, rowHalfSpanMm);
  const aboveRows = Math.max(14, Math.round(rowSpanMm.aboveMm / Math.max(1e-3, vertStepMm)));
  const belowRows = Math.max(14, Math.round(rowSpanMm.belowMm / Math.max(1e-3, vertStepMm)));
  const rowStart = Math.max(0, track.anchorRow - aboveRows);
  const rowEnd = Math.min(panoHeight - 1, track.anchorRow + belowRows);
  const rowCount = Math.max(1, rowEnd - rowStart + 1);
  const localAnchorRow = clampNumber(track.anchorRow - rowStart, 0, rowCount - 1);
  const rowOffsetsMm = new Float32Array(rowCount);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    const panoRow = rowStart + rowIndex;
    rowOffsetsMm[rowIndex] = effectiveVerticalHalfMm - panoRow * vertStepMm;
  }

  const depthDefinition = buildDepthOffsetsMm(
    track.envelopeHalfWidthMmByCol,
    depthSampleCount,
    track.searchDepthHalfRangeMm
  );
  const valuesHu = new Float32Array(Math.max(1, panoWidth * rowCount * depthDefinition.depthOffsetsMm.length));
  valuesHu.fill(Number.NaN);
  let finiteSampleCount = 0;

  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    if (!frame) {
      continue;
    }
    const [px, py, pz] = frame.position;
    const [slabDirX, slabDirY, slabDirZ] = frame.slabDir;
    const [vertDirX, vertDirY, vertDirZ] = frame.verticalDir;
    const centerDepthMm = Number(track.centerDepthMmByCol[col] ?? 0);
    const columnVerticalCenterOffsetMm = Number(frame.columnVerticalCenterOffsetMm ?? 0);

    for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
      const verticalOffsetMm = columnVerticalCenterOffsetMm + Number(rowOffsetsMm[rowIndex]);
      const bx = px + verticalOffsetMm * vertDirX;
      const by = py + verticalOffsetMm * vertDirY;
      const bz = pz + verticalOffsetMm * vertDirZ;

      for (let depthIndex = 0; depthIndex < depthDefinition.depthOffsetsMm.length; depthIndex++) {
        const localDepthMm = centerDepthMm + Number(depthDefinition.depthOffsetsMm[depthIndex]);
        const sample = sampleWorldIntensity(
          bx + localDepthMm * slabDirX,
          by + localDepthMm * slabDirY,
          bz + localDepthMm * slabDirZ
        );
        valuesHu[straightenedBandIndex(col, rowIndex, depthIndex, rowCount, depthDefinition.depthOffsetsMm.length)] =
          Number.isFinite(sample) ? sample : Number.NaN;
        if (Number.isFinite(sample)) {
          finiteSampleCount++;
        }
      }
    }
  }

  return {
    label: track.label,
    anchorRow: track.anchorRow,
    localAnchorRow,
    rowStart,
    rowEnd,
    rowCount,
    depthCount: depthDefinition.depthOffsetsMm.length,
    rowHalfSpanMm: roundTo(Math.max(rowSpanMm.aboveMm, rowSpanMm.belowMm)),
    depthHalfRangeMm: roundTo(depthDefinition.depthHalfRangeMm),
    rowOffsetsMm,
    depthOffsetsMm: depthDefinition.depthOffsetsMm,
    centerDepthMmByCol: track.centerDepthMmByCol,
    envelopeHalfWidthMmByCol: track.envelopeHalfWidthMmByCol ?? null,
    valuesHu,
    finiteSampleFraction: roundTo(
      valuesHu.length > 0 ? finiteSampleCount / valuesHu.length : 0,
      4
    ),
    valueRangeHu: summarizeFinite(valuesHu),
    centerDepthSummaryMm: summarizeFinite(track.centerDepthMmByCol),
    envelopeHalfWidthSummaryMm: summarizeFinite(track.envelopeHalfWidthMmByCol ?? []),
  };
}

export function buildPanoV2StraightenedVolume(
  params: PanoV2StraightenedVolumeInput
): PanoV2StraightenedVolume {
  return {
    enabled: true,
    phase: 2,
    model: 'straightened-local-volume',
    panoWidth: params.panoWidth,
    panoHeight: params.panoHeight,
    sampler: 'world-hu-trilinear',
    effectiveVerticalHalfMm: params.effectiveVerticalHalfMm,
    vertStepMm: params.vertStepMm,
    frames: params.frames,
    sampleWorldIntensity: params.sampleWorldIntensity,
    upperArch: buildStraightenedBand({
      ...params,
      track: params.upperArch,
    }),
    lowerArch: buildStraightenedBand({
      ...params,
      track: params.lowerArch,
    }),
  };
}

export function samplePanoV2StraightenedBandValue(params: {
  band: PanoV2StraightenedVolumeBand;
  col: number;
  rowIndex: number;
  depthIndex: number;
}): number {
  const col = Math.round(
    clampNumber(params.col, 0, Math.max(0, params.band.centerDepthMmByCol.length - 1))
  );
  const rowIndex = Math.round(
    clampNumber(params.rowIndex, 0, Math.max(0, params.band.rowCount - 1))
  );
  const depthIndex = Math.round(
    clampNumber(params.depthIndex, 0, Math.max(0, params.band.depthCount - 1))
  );
  return Number(
    params.band.valuesHu[
      straightenedBandIndex(col, rowIndex, depthIndex, params.band.rowCount, params.band.depthCount)
    ]
  );
}

function buildBandDiagnostics(band: PanoV2StraightenedVolumeBand): PanoV2StraightenedVolumeBandDiagnostics {
  const sampledColumns = buildEvenlySpacedIndices(band.centerDepthMmByCol.length, 5).map(col => {
    const representativeRows = buildRowProbeIndices(band.rowCount, band.localAnchorRow).map(rowIndex => ({
      localRow: rowIndex,
      panoRow: band.rowStart + rowIndex,
      rowOffsetMm: roundTo(Number(band.rowOffsetsMm[rowIndex] ?? 0)),
      depthValuesHu: Array.from({ length: band.depthCount }, (_, depthIndex) => {
        const value = samplePanoV2StraightenedBandValue({
          band,
          col,
          rowIndex,
          depthIndex,
        });
        return Number.isFinite(value) ? roundTo(value) : null;
      }),
    }));

    return {
      col,
      centerDepthMm: roundTo(Number(band.centerDepthMmByCol[col] ?? 0)),
      representativeRows,
    };
  });

  return {
    label: band.label,
    anchorRow: band.anchorRow,
    localAnchorRow: band.localAnchorRow,
    rowStart: band.rowStart,
    rowEnd: band.rowEnd,
    rowCount: band.rowCount,
    depthCount: band.depthCount,
    rowHalfSpanMm: roundTo(band.rowHalfSpanMm),
    depthHalfRangeMm: roundTo(band.depthHalfRangeMm),
    rowOffsetsFirst8Mm: toRoundedArray(band.rowOffsetsMm, 0, 8),
    rowOffsetsLast8Mm: toRoundedArray(band.rowOffsetsMm, Math.max(0, band.rowCount - 8), 8),
    depthOffsetsMm: Array.from(band.depthOffsetsMm, value => roundTo(Number(value))),
    finiteSampleFraction: roundTo(band.finiteSampleFraction, 4),
    centerDepthSummaryMm: band.centerDepthSummaryMm,
    envelopeHalfWidthSummaryMm: band.envelopeHalfWidthSummaryMm,
    valueRangeHu: band.valueRangeHu,
    sampledColumns,
  };
}

export function buildPanoV2StraightenedVolumeDiagnostics(
  volume: PanoV2StraightenedVolume
): PanoV2StraightenedVolumeDiagnostics {
  return {
    enabled: true,
    phase: 2,
    model: 'straightened-local-volume',
    panoWidth: volume.panoWidth,
    panoHeight: volume.panoHeight,
    sampler: volume.sampler,
    upperArch: buildBandDiagnostics(volume.upperArch),
    lowerArch: buildBandDiagnostics(volume.lowerArch),
  };
}
