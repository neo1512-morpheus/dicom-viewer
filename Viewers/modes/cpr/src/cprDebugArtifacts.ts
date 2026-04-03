export type CprDebugImageMode = 'hu' | 'unit' | 'signed' | 'positive';

export interface CprDebugReportRow {
  label: string;
  checksum: number | null;
  supportP50: number | null;
  bgOutlier05: number | null;
  bgOutlier10: number | null;
  blackClip: number | null;
  lowerBandBrightFraction: number | null;
  rejectReasons: string[];
  displayable: boolean;
  legacyQualityScore?: number | null;
  legacyQualityBase?: number | null;
  displaySelectionScore?: number | null;
  preferredTightRootFamily?: boolean;
  effectiveTroughHalfWidthP50Mm?: number | null;
  participatingSamplesP50?: number | null;
  duplicateOfLabel?: string | null;
}

export interface CprDebugReportImage {
  title: string;
  caption?: string | null;
  filenameStem: string;
  width: number;
  height: number;
  pixels: Float32Array | Uint16Array;
  mode: CprDebugImageMode;
  lower?: number | null;
  upper?: number | null;
}

export interface CprDebugReportSummarySection {
  title: string;
  lines: string[];
}

export interface CprDebugProbeRow {
  clickIndex: number;
  imageId: string | null;
  col: number;
  row: number;
  mappingMode?: string | null;
  displayedPath?: string | null;
  backend?: string | null;
  pipelineMode?: string | null;
  reconstructionMode?: string | null;
  rawSupportDepthMm?: number | null;
  rawSupportPeakDominance?: number | null;
  rawSupportPeakValidity?: number | null;
  rawSupportSecondPeakRatio?: number | null;
  rawSupportPeakAmbiguity?: number | null;
  rawSupportScoreGap?: number | null;
  rawSupportDenseFraction?: number | null;
  rawSupportPeakHuSupportGate?: number | null;
  supportCenterMm?: number | null;
  supportSpreadMm?: number | null;
  supportValidity?: number | null;
  supportDensity?: number | null;
  supportConfidence?: number | null;
  dominantDensePeakGate?: number | null;
  toothBandStructureGuard?: number | null;
  protectedAmbiguousBroadSupportPenaltyGate?: number | null;
  falseSupportVeto?: number | null;
  rowBackgroundVeto?: number | null;
  supportVetoTriggered?: number | null;
  effectiveTroughHalfWidthMm?: number | null;
  continuityExpandedTroughHalfWidthMm?: number | null;
  dominantToothBandGate?: number | null;
  broadWeakToothBandGate?: number | null;
  toothContinuityAdmissionGate?: number | null;
  admissionAccumulation?: number | null;
  preToneAccumulation?: number | null;
  blackClip?: number | null;
  retainedSampleMask?: number | null;
  middleBandLeak?: number | null;
  holeMetricWouldCount?: boolean;
  holeMetricReasons?: string[];
}

const downloadedCprDebugArtifactRunIds = new Set<string>();

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function sanitizeFilenamePart(value: string): string {
  return value.replace(/[^a-z0-9._-]+/gi, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function collectFiniteSamples(
  buffer: Float32Array | Uint16Array,
  maxSamples = 4096
): number[] {
  if (!buffer.length) {
    return [];
  }

  const step = Math.max(1, Math.floor(buffer.length / maxSamples));
  const samples: number[] = [];
  for (let i = 0; i < buffer.length; i += step) {
    const value = Number(buffer[i]);
    if (Number.isFinite(value)) {
      samples.push(value);
    }
  }
  if (!samples.length) {
    return [];
  }
  samples.sort((a, b) => a - b);
  return samples;
}

function percentileFromSorted(values: number[], fraction: number): number | null {
  if (!values.length) {
    return null;
  }

  const safeFraction = clamp01(fraction);
  const index = Math.max(0, Math.min(values.length - 1, Math.round((values.length - 1) * safeFraction)));
  return values[index];
}

function resolveDisplayWindow(image: CprDebugReportImage): { lower: number; upper: number } {
  if (Number.isFinite(image.lower) && Number.isFinite(image.upper) && Number(image.upper) > Number(image.lower)) {
    return { lower: Number(image.lower), upper: Number(image.upper) };
  }

  const samples = collectFiniteSamples(image.pixels);
  if (!samples.length) {
    return { lower: 0, upper: 1 };
  }

  if (image.mode === 'unit') {
    return { lower: 0, upper: 1 };
  }

  if (image.mode === 'signed') {
    const p01 = percentileFromSorted(samples, 0.01) ?? -1;
    const p99 = percentileFromSorted(samples, 0.99) ?? 1;
    const magnitude = Math.max(Math.abs(p01), Math.abs(p99), 1e-6);
    return { lower: -magnitude, upper: magnitude };
  }

  if (image.mode === 'positive') {
    const p99 = percentileFromSorted(samples, 0.99) ?? 1;
    return { lower: 0, upper: Math.max(p99, 1e-6) };
  }

  const p01 = percentileFromSorted(samples, 0.01) ?? 0;
  const p99 = percentileFromSorted(samples, 0.99) ?? 1;
  if (p99 <= p01) {
    return { lower: p01 - 1, upper: p99 + 1 };
  }
  return { lower: p01, upper: p99 };
}

function encodeImageDataUrl(image: CprDebugReportImage): string | null {
  if (typeof document === 'undefined' || !image.width || !image.height || !image.pixels.length) {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }

  const { lower, upper } = resolveDisplayWindow(image);
  const span = Math.max(upper - lower, 1e-6);
  const imageData = context.createImageData(image.width, image.height);

  for (let i = 0; i < image.width * image.height; i++) {
    const pixelValue = Number(image.pixels[i]);
    let normalized = clamp01((pixelValue - lower) / span);
    let red = 0;
    let green = 0;
    let blue = 0;

    if (image.mode === 'signed') {
      const centered = clamp01((pixelValue - lower) / span);
      red = Math.round(centered * 255);
      blue = Math.round((1 - centered) * 255);
      green = Math.round((1 - Math.abs(centered - 0.5) * 2) * 180);
    } else {
      const gray = Math.round(normalized * 255);
      red = gray;
      green = gray;
      blue = gray;
    }

    const outIndex = i * 4;
    imageData.data[outIndex] = red;
    imageData.data[outIndex + 1] = green;
    imageData.data[outIndex + 2] = blue;
    imageData.data[outIndex + 3] = 255;
  }

  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

function triggerDownload(filename: string, text: string, mimeType: string): void {
  if (typeof document === 'undefined' || typeof URL === 'undefined') {
    return;
  }

  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function buildAttemptTableCsv(rows: CprDebugReportRow[]): string {
  const header = [
    'label',
    'checksum',
    'supportP50',
    'bgOutlier05',
    'bgOutlier10',
    'blackClip',
    'lowerBandBrightFraction',
    'legacyQualityScore',
    'legacyQualityBase',
    'displaySelectionScore',
    'preferredTightRootFamily',
    'effectiveTroughHalfWidthP50Mm',
    'participatingSamplesP50',
    'rejectReasons',
    'displayable',
    'duplicateOfLabel',
  ];

  const body = rows.map(row =>
    [
      row.label,
      row.checksum ?? '',
      row.supportP50 ?? '',
      row.bgOutlier05 ?? '',
      row.bgOutlier10 ?? '',
      row.blackClip ?? '',
      row.lowerBandBrightFraction ?? '',
      row.legacyQualityScore ?? '',
      row.legacyQualityBase ?? '',
      row.displaySelectionScore ?? '',
      row.preferredTightRootFamily ? 'yes' : 'no',
      row.effectiveTroughHalfWidthP50Mm ?? '',
      row.participatingSamplesP50 ?? '',
      row.rejectReasons.join('|'),
      row.displayable ? 'yes' : 'no',
      row.duplicateOfLabel ?? '',
    ]
      .map(value => `"${String(value).replace(/"/g, '""')}"`)
      .join(',')
  );

  return [header.join(','), ...body].join('\n');
}

function buildProbeTableCsv(rows: CprDebugProbeRow[]): string {
  const header = [
    'clickIndex',
    'imageId',
    'col',
    'row',
    'mappingMode',
    'displayedPath',
    'backend',
    'pipelineMode',
    'reconstructionMode',
    'rawSupportDepthMm',
    'rawSupportPeakDominance',
    'rawSupportPeakValidity',
    'rawSupportSecondPeakRatio',
    'rawSupportPeakAmbiguity',
    'rawSupportScoreGap',
    'rawSupportDenseFraction',
    'rawSupportPeakHuSupportGate',
    'supportCenterMm',
    'supportSpreadMm',
    'supportValidity',
    'supportDensity',
    'supportConfidence',
    'dominantDensePeakGate',
    'toothBandStructureGuard',
    'protectedAmbiguousBroadSupportPenaltyGate',
    'falseSupportVeto',
    'rowBackgroundVeto',
    'supportVetoTriggered',
    'effectiveTroughHalfWidthMm',
    'continuityExpandedTroughHalfWidthMm',
    'dominantToothBandGate',
    'broadWeakToothBandGate',
    'toothContinuityAdmissionGate',
    'admissionAccumulation',
    'preToneAccumulation',
    'blackClip',
    'retainedSampleMask',
    'middleBandLeak',
    'holeMetricWouldCount',
    'holeMetricReasons',
  ];

  const body = rows.map(row =>
    [
      row.clickIndex,
      row.imageId ?? '',
      row.col,
      row.row,
      row.mappingMode ?? '',
      row.displayedPath ?? '',
      row.backend ?? '',
      row.pipelineMode ?? '',
      row.reconstructionMode ?? '',
      row.rawSupportDepthMm ?? '',
      row.rawSupportPeakDominance ?? '',
      row.rawSupportPeakValidity ?? '',
      row.rawSupportSecondPeakRatio ?? '',
      row.rawSupportPeakAmbiguity ?? '',
      row.rawSupportScoreGap ?? '',
      row.rawSupportDenseFraction ?? '',
      row.rawSupportPeakHuSupportGate ?? '',
      row.supportCenterMm ?? '',
      row.supportSpreadMm ?? '',
      row.supportValidity ?? '',
      row.supportDensity ?? '',
      row.supportConfidence ?? '',
      row.dominantDensePeakGate ?? '',
      row.toothBandStructureGuard ?? '',
      row.protectedAmbiguousBroadSupportPenaltyGate ?? '',
      row.falseSupportVeto ?? '',
      row.rowBackgroundVeto ?? '',
      row.supportVetoTriggered ?? '',
      row.effectiveTroughHalfWidthMm ?? '',
      row.continuityExpandedTroughHalfWidthMm ?? '',
      row.dominantToothBandGate ?? '',
      row.broadWeakToothBandGate ?? '',
      row.toothContinuityAdmissionGate ?? '',
      row.admissionAccumulation ?? '',
      row.preToneAccumulation ?? '',
      row.blackClip ?? '',
      row.retainedSampleMask ?? '',
      row.middleBandLeak ?? '',
      row.holeMetricWouldCount === undefined ? '' : row.holeMetricWouldCount ? 'yes' : 'no',
      row.holeMetricReasons?.join('|') ?? '',
    ]
      .map(value => `"${String(value).replace(/"/g, '""')}"`)
      .join(',')
  );

  return [header.join(','), ...body].join('\n');
}

function buildReportHtml(params: {
  title: string;
  runId: string;
  rows: CprDebugReportRow[];
  probeRows?: CprDebugProbeRow[];
  images: Array<CprDebugReportImage & { dataUrl: string | null }>;
  summarySections?: CprDebugReportSummarySection[];
}): string {
  const rowsHtml = params.rows
    .map(
      row => `<tr>
  <td>${escapeHtml(row.label)}</td>
  <td>${row.checksum ?? 'na'}</td>
  <td>${row.supportP50 ?? 'na'}</td>
  <td>${row.bgOutlier05 ?? 'na'}</td>
  <td>${row.bgOutlier10 ?? 'na'}</td>
  <td>${row.blackClip ?? 'na'}</td>
  <td>${row.lowerBandBrightFraction ?? 'na'}</td>
  <td>${row.legacyQualityScore ?? 'na'}</td>
  <td>${row.legacyQualityBase ?? 'na'}</td>
  <td>${row.displaySelectionScore ?? 'na'}</td>
  <td>${row.preferredTightRootFamily ? 'yes' : 'no'}</td>
  <td>${row.effectiveTroughHalfWidthP50Mm ?? 'na'}</td>
  <td>${row.participatingSamplesP50 ?? 'na'}</td>
  <td>${escapeHtml(row.rejectReasons.join(', ') || 'none')}</td>
  <td>${row.displayable ? 'yes' : 'no'}</td>
  <td>${escapeHtml(row.duplicateOfLabel ?? '')}</td>
</tr>`
    )
    .join('\n');
  const probeRowsHtml = (params.probeRows ?? [])
    .map(
      row => `<tr>
  <td>${row.clickIndex}</td>
  <td>${row.col}</td>
  <td>${row.row}</td>
  <td>${escapeHtml(row.mappingMode ?? '')}</td>
  <td>${escapeHtml(row.displayedPath ?? '')}</td>
  <td>${escapeHtml(row.backend ?? '')}</td>
  <td>${escapeHtml(row.pipelineMode ?? '')}</td>
  <td>${escapeHtml(row.reconstructionMode ?? '')}</td>
  <td>${row.rawSupportPeakDominance ?? 'na'}</td>
  <td>${row.rawSupportPeakValidity ?? 'na'}</td>
  <td>${row.rawSupportSecondPeakRatio ?? 'na'}</td>
  <td>${row.rawSupportPeakAmbiguity ?? 'na'}</td>
  <td>${row.rawSupportScoreGap ?? 'na'}</td>
  <td>${row.rawSupportPeakHuSupportGate ?? 'na'}</td>
  <td>${row.supportCenterMm ?? 'na'}</td>
  <td>${row.supportSpreadMm ?? 'na'}</td>
  <td>${row.supportValidity ?? 'na'}</td>
  <td>${row.supportDensity ?? 'na'}</td>
  <td>${row.supportConfidence ?? 'na'}</td>
  <td>${row.dominantDensePeakGate ?? 'na'}</td>
  <td>${row.toothBandStructureGuard ?? 'na'}</td>
  <td>${row.protectedAmbiguousBroadSupportPenaltyGate ?? 'na'}</td>
  <td>${row.falseSupportVeto ?? 'na'}</td>
  <td>${row.rowBackgroundVeto ?? 'na'}</td>
  <td>${row.supportVetoTriggered ?? 'na'}</td>
  <td>${row.effectiveTroughHalfWidthMm ?? 'na'}</td>
  <td>${row.admissionAccumulation ?? 'na'}</td>
  <td>${row.preToneAccumulation ?? 'na'}</td>
  <td>${row.blackClip ?? 'na'}</td>
  <td>${row.retainedSampleMask ?? 'na'}</td>
  <td>${row.middleBandLeak ?? 'na'}</td>
  <td>${row.holeMetricWouldCount === undefined ? 'na' : row.holeMetricWouldCount ? 'yes' : 'no'}</td>
  <td>${escapeHtml(row.holeMetricReasons?.join(', ') ?? '')}</td>
</tr>`
    )
    .join('\n');

  const imagesHtml = params.images
    .map(image => {
      const caption = image.caption ? `<div class="caption">${escapeHtml(image.caption)}</div>` : '';
      const img = image.dataUrl
        ? `<img alt="${escapeHtml(image.title)}" src="${image.dataUrl}" />`
        : `<div class="missing">image unavailable</div>`;
      return `<section class="card">
  <h3>${escapeHtml(image.title)}</h3>
  ${img}
  ${caption}
</section>`;
    })
    .join('\n');
  const summaryHtml = (params.summarySections ?? [])
    .map(
      section => `<section class="card">
  <h2>${escapeHtml(section.title)}</h2>
  <div class="meta">${escapeHtml(section.lines.join('\n'))}</div>
</section>`
    )
    .join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(params.title)}</title>
  <style>
    body { font-family: Georgia, serif; background: #f5f0e8; color: #1f1914; margin: 24px; }
    h1, h2, h3 { margin: 0 0 12px; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0 28px; font-size: 12px; }
    th, td { border: 1px solid #b7a996; padding: 6px 8px; vertical-align: top; }
    th { background: #d8c8b2; text-align: left; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .card { background: #fffaf3; border: 1px solid #c9b8a5; padding: 12px; }
    .card img { width: 100%; image-rendering: pixelated; border: 1px solid #c9b8a5; background: #000; }
    .caption, .meta { font-size: 12px; line-height: 1.4; margin-top: 8px; white-space: pre-wrap; }
    .missing { display: flex; align-items: center; justify-content: center; height: 160px; background: #ece0d1; color: #6d5747; border: 1px dashed #b7a996; }
  </style>
</head>
<body>
  <h1>${escapeHtml(params.title)}</h1>
  <div class="meta">Run: ${escapeHtml(params.runId)}</div>
  ${
    summaryHtml
      ? `<h2>Summary</h2>
  <div class="grid">
    ${summaryHtml}
  </div>`
      : ''
  }
  <h2>Attempt Table</h2>
  <table>
    <thead>
      <tr>
        <th>Label</th>
        <th>Checksum</th>
        <th>supportP50</th>
        <th>bg05</th>
        <th>bg10</th>
        <th>blackClip</th>
        <th>lowerBandBright</th>
        <th>legacyScore</th>
        <th>legacyBase</th>
        <th>displayScore</th>
        <th>preferredTightRoot</th>
        <th>troughP50</th>
        <th>samplesP50</th>
        <th>rejectReasons</th>
        <th>displayable</th>
        <th>duplicateOf</th>
      </tr>
    </thead>
    <tbody>
      ${rowsHtml}
    </tbody>
  </table>
  ${
    probeRowsHtml
      ? `<h2>Probe Clicks</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>col</th>
        <th>row</th>
        <th>mappingMode</th>
        <th>displayedPath</th>
        <th>backend</th>
        <th>pipelineMode</th>
        <th>reconstructionMode</th>
        <th>peakDom</th>
        <th>peakValid</th>
        <th>secondPeakRatio</th>
        <th>peakAmbiguity</th>
        <th>scoreGap</th>
        <th>peakHuGate</th>
        <th>supportCenter</th>
        <th>supportSpread</th>
        <th>supportValidity</th>
        <th>supportDensity</th>
        <th>supportConfidence</th>
        <th>dominantGate</th>
        <th>toothGuard</th>
        <th>protectedAmbig</th>
        <th>falseVeto</th>
        <th>rowBgVeto</th>
        <th>supportVeto</th>
        <th>troughWidthUsed</th>
        <th>admission</th>
        <th>preTone</th>
        <th>blackClip</th>
        <th>retained</th>
        <th>middleLeak</th>
        <th>holeMetric</th>
        <th>holeReasons</th>
      </tr>
    </thead>
    <tbody>
      ${probeRowsHtml}
    </tbody>
  </table>`
      : ''
  }
  <h2>Images</h2>
  <div class="grid">
    ${imagesHtml}
  </div>
</body>
</html>`;
}

export function downloadCprDebugArtifacts(params: {
  title: string;
  runId: string;
  rows: CprDebugReportRow[];
  probeRows?: CprDebugProbeRow[];
  images: CprDebugReportImage[];
  summarySections?: CprDebugReportSummarySection[];
}): void {
  if (typeof document === 'undefined') {
    return;
  }

  const safeRunId = sanitizeFilenamePart(params.runId || 'cpr-debug');
  if (downloadedCprDebugArtifactRunIds.has(safeRunId)) {
    return;
  }

  const encodedImages = params.images.map(image => ({
    ...image,
    dataUrl: encodeImageDataUrl(image),
  }));
  const html = buildReportHtml({
    title: params.title,
    runId: params.runId,
    rows: params.rows,
    probeRows: params.probeRows,
    images: encodedImages,
    summarySections: params.summarySections,
  });
  const csvSections = [buildAttemptTableCsv(params.rows)];
  if (params.probeRows?.length) {
    csvSections.push('', buildProbeTableCsv(params.probeRows));
  }
  const csv = csvSections.join('\n');

  downloadedCprDebugArtifactRunIds.add(safeRunId);
  triggerDownload(`cpr-debug-report-${safeRunId}.html`, html, 'text/html;charset=utf-8');
  triggerDownload(`cpr-attempt-table-${safeRunId}.csv`, csv, 'text/csv;charset=utf-8');
}
