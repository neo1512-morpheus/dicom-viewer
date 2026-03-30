import { useCallback, useEffect, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { vec3 } from 'gl-matrix';

import { cprStateService } from './CPRStateService';
import {
  setPanoImagePayload,
  createPanoImageId,
  clearPanoImageCache,
  getPanoImagePayload,
} from './panoImageLoader';
import type { PanoImagePayload, PanoQualityStatus } from './panoImageLoader';
import {
  SyntheticCprIntensityDomain,
  classifySyntheticCprIntensityDomain,
  isSyntheticCprHuDomain,
} from './cprSyntheticDisplay';
import { buildRMFFrames } from './cprMath';
import {
  CPR_CROSSSECTION_SYNC_EVENT,
  CPR_PANO_HOST_ATTACHED_EVENT,
  CPRCrossSectionSyncDetail,
} from './cprEvents';
import { buildCrossSectionCameraForFrame } from './cprCrossSectionCamera';
import { downloadCprDebugArtifacts } from './cprDebugArtifacts';
import type { CPRFrame } from './cprMath';
import { attachVtkPanoCpr } from './vtkPanoCprRenderer';
import type { AttachedVtkPanoCpr, HostedVtkPanoReattachState } from './vtkPanoCprRenderer';

type CprPanoDisplayPath = 'vtk-reference' | 'worker-recon';
type CprPanoReconBackend = 'gpu' | 'cpu';
type CprPanoDisplayBackend = CprPanoReconBackend | 'vtk';
type CprPanoDisplaySource = 'panoImageLoader' | 'vtk-hosted-pano-actor';

const CPR_PANO_DEFAULT_WINDOW_WIDTH = 2000;
const CPR_PANO_DEFAULT_WINDOW_CENTER = 1000;
const CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU = -350;
const CPR_DUAL_ARCH_PROJECTION_SOFT_SCAN_P99_THRESHOLD_HU = 1100;
const CPR_DUAL_ARCH_PROJECTION_SOFT_MIN_WINDOW_WIDTH = 1350;
const CPR_DUAL_ARCH_PROJECTION_STANDARD_MIN_WINDOW_WIDTH = 1450;
const CPR_PANO_DISPLAY_PATH_DEFAULT: CprPanoDisplayPath = 'worker-recon';
const CPR_PANO_RECON_BACKEND_DEFAULT: CprPanoReconBackend = 'gpu';
const CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK = false;
const CPR_DEBUG_EXPORT_ATTEMPT_REPORT_DEFAULT = false;
const CPR_DEBUG_EXPORT_TOP_ATTEMPT_COUNT = 5;
const CPR_DEBUG_DISPLAY_SELECTION_FOCUS_LABELS = [
  'retry-mean-broad-neutral',
  'retry-mean-balanced-medium-stronger-bias-tight-slab',
  'retry-mean-balanced-medium-strong-bias-tight-slab',
  'retry-mean-balanced-medium-root-biased-tight-slab',
  'retry-mean-toothband-rooted',
  'retry-mean-sharp-narrow',
] as const;
const CPR_GPU_PANO_PRIORITY_RETRY_LABELS = [
  'retry-mean-balanced-medium-stronger-bias-tight-slab',
  'retry-mean-balanced-medium-root-biased-tight-slab',
  'retry-mean-toothband-rooted',
  'retry-mean-balanced-medium',
  'retry-mean-balanced-neutral',
  'retry-mean-broad-neutral',
  'retry-no-lut-mean-narrow',
] as const;
const CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS = [
  'retry-mean-balanced-medium-stronger-bias-tight-slab',
  'retry-mean-balanced-medium-root-biased-tight-slab',
  'retry-mean-toothband-rooted',
  'retry-mean-balanced-medium-strong-bias-tight-slab',
  'retry-mean-balanced-medium',
  'retry-mean-balanced-neutral',
  'retry-mean-broad-neutral',
  'primary-mean-toothband-narrow',
] as const;
const CPR_GPU_PANO_DEFAULT_SLAB_HALF_THICKNESS_MM = 2;
const CPR_GPU_PANO_DEFAULT_SLAB_SAMPLES = 15;
const CPR_VTK_PANO_PRESET_WINDOW_WIDTH = 3200;
const CPR_VTK_PANO_PRESET_WINDOW_CENTER = 600;
const CPR_PANO_GENERATION_DEBOUNCE_MS = 300;
const CPR_PANO_MAX_DIMENSION = 4096;
const CPR_PANO_DEFAULT_VERTICAL_HALF_MM = 18;
const CPR_PANO_MAX_VERTICAL_HALF_MM = 28;
const CPR_PANO_TARGET_ASPECT = 3.2;
const CPR_CROSSSECTION_DEFAULT_SLAB_THICKNESS_MM = 1.5;
const CPR_CROSSSECTION_DEFAULT_BLEND_MODE = cornerstone.Enums.BlendModes.AVERAGE_INTENSITY_BLEND;
const CPR_CROSSSECTION_RENDER_WAIT_TIMEOUT_MS = 1500;
const CPR_TEMP_DEBUG_PIN_DISPLAYED_ATTEMPT_LABELS: readonly string[] = [];
let activePanoDebugProbeCleanup: (() => void) | null = null;

function parseCprDebugBooleanFlag(value: string | null | undefined): boolean | null {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  if (
    normalized === '1' ||
    normalized === 'true' ||
    normalized === 'yes' ||
    normalized === 'on'
  ) {
    return true;
  }

  if (
    normalized === '0' ||
    normalized === 'false' ||
    normalized === 'no' ||
    normalized === 'off'
  ) {
    return false;
  }

  return null;
}

function shouldExportCprAttemptReport(): boolean {
  if (typeof window === 'undefined') {
    return CPR_DEBUG_EXPORT_ATTEMPT_REPORT_DEFAULT;
  }

  const queryValue = parseCprDebugBooleanFlag(
    new URLSearchParams(window.location.search).get('cprDebugExportAttemptReport')
  );
  if (queryValue !== null) {
    return queryValue;
  }

  try {
    const storedValue = parseCprDebugBooleanFlag(
      window.localStorage?.getItem('cpr.debug.exportAttemptReport')
    );
    if (storedValue !== null) {
      return storedValue;
    }
  } catch (error) {
    console.warn('[CPR] Failed to read debug export flag from localStorage.', error);
  }

  return CPR_DEBUG_EXPORT_ATTEMPT_REPORT_DEFAULT;
}

type PanoDebugProbeSample = {
  imageId: string | null;
  col: number;
  row: number;
  index: number;
  finalHu: number;
  meanHu?: number;
  rayMax?: number;
  maxLift?: number;
  validSampleCount?: number;
  rawSupportDepthMm?: number;
  rawSupportConfidence?: number;
  rawSupportSpreadMm?: number;
  rawSupportDensity?: number;
  rawSupportPeakDominance?: number;
  rawSupportPeakValidity?: number;
  rawSupportPeakConflict?: number;
  rawSupportSecondPeakRatio?: number;
  rawSupportPeakSeparationMm?: number;
  rawSupportPeakAmbiguity?: number;
  rawSupportLocalJumpMm?: number;
  rawSupportContinuity?: number;
  supportFailureDisplay?: number;
  supportDepthMm?: number;
  supportConfidence?: number;
  supportSpreadMm?: number;
  supportDensity?: number;
  supportLocalJumpMm?: number;
  supportContinuity?: number;
  invalidSupportBlackout?: number;
  supportDepthDeltaMm?: number;
  supportConfidenceDelta?: number;
  supportSpreadDeltaMm?: number;
  supportDensityDelta?: number;
  nominalTroughHalfWidthMm?: number;
  effectiveTroughHalfWidthMm?: number;
  backgroundTroughNarrowGate?: number;
  troughLowerMm?: number;
  troughUpperMm?: number;
  lowerPenalty?: number;
  localTransmittance?: number;
  toneMappedValue?: number;
  participatingSampleCount?: number;
  displayedPath?: string | null;
  backend?: string | null;
  pipelineMode?: string | null;
  reconstructionMode?: string | null;
  mappingMode?: string;
  canvasX?: number;
  canvasY?: number;
};

function clearPanoDebugProbe(): void {
  if (activePanoDebugProbeCleanup) {
    activePanoDebugProbeCleanup();
    activePanoDebugProbeCleanup = null;
  }
}

function readOptionalProbeMapValue(
  map: Float32Array | undefined,
  index: number
): number | undefined {
  if (!map || index < 0 || index >= map.length) {
    return undefined;
  }

  const value = Number(map[index]);
  return Number.isFinite(value) ? value : undefined;
}

function readPanoDebugProbeSample(
  payload: PanoImagePayload | null,
  imageId: string | null,
  col: number,
  row: number
): PanoDebugProbeSample | null {
  const hasLegacyMaps = !!payload?.meanMap && !!payload?.maxMap;
  const hasDebugMaps = !!payload?.debugMaps;
  if (!hasLegacyMaps && !hasDebugMaps) {
    console.warn('[CPR] Pano debug probe unavailable: diagnostic sidecars are missing.', {
      imageId,
      hasPayload: !!payload,
      hasLegacyMaps,
      hasDebugMaps,
    });
    return null;
  }

  const safeCol = Math.max(0, Math.min(payload.width - 1, Math.round(col)));
  const safeRow = Math.max(0, Math.min(payload.height - 1, Math.round(row)));
  const index = safeRow * payload.width + safeCol;
  const hasPhase2ProbeMaps =
    !!payload.debugMaps?.supportDepthMap || !!payload.debugMaps?.toneResponseMap;
  const finalHu = Number(payload.pixelData[index]);
  const meanHu =
    !hasPhase2ProbeMaps && payload.meanMap ? Number(payload.meanMap[index]) : undefined;
  const rayMax = !hasPhase2ProbeMaps && payload.maxMap ? Number(payload.maxMap[index]) : undefined;
  const validSampleCount =
    payload.sampleCountMap && !payload.debugMaps?.toneResponseMap
      ? Number(payload.sampleCountMap[index])
      : undefined;
  const supportDepthMm = readOptionalProbeMapValue(payload.debugMaps?.supportDepthMap, index);
  const rawSupportDepthMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportDepthMap,
    index
  );
  const supportConfidence = readOptionalProbeMapValue(
    payload.debugMaps?.supportConfidenceMap,
    index
  );
  const rawSupportConfidence = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportConfidenceMap,
    index
  );
  const supportSpreadMm = readOptionalProbeMapValue(payload.debugMaps?.supportSpreadMap, index);
  const rawSupportSpreadMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportSpreadMap,
    index
  );
  const supportDensity = readOptionalProbeMapValue(payload.debugMaps?.supportDensityMap, index);
  const rawSupportDensity = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportDensityMap,
    index
  );
  const rawSupportPeakDominance = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakDominanceMap,
    index
  );
  const rawSupportPeakValidity = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakValidityMap,
    index
  );
  const rawSupportPeakConflict = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakConflictMap,
    index
  );
  const rawSupportSecondPeakRatio = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportSecondPeakRatioMap,
    index
  );
  const rawSupportPeakSeparationMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakSeparationMap,
    index
  );
  const rawSupportPeakAmbiguity = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakAmbiguityMap,
    index
  );
  const rawSupportLocalJumpMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportLocalJumpMap,
    index
  );
  const rawSupportContinuity = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportContinuityMap,
    index
  );
  const supportFailureDisplay = readOptionalProbeMapValue(
    payload.debugMaps?.supportFailureDisplayMap,
    index
  );
  const supportLocalJumpMm = readOptionalProbeMapValue(
    payload.debugMaps?.supportLocalJumpMap,
    index
  );
  const supportContinuity = readOptionalProbeMapValue(
    payload.debugMaps?.supportContinuityMap,
    index
  );
  const nominalTroughHalfWidthMm = readOptionalProbeMapValue(
    payload.debugMaps?.troughHalfWidthMap,
    index
  );
  const effectiveTroughHalfWidthMm = readOptionalProbeMapValue(
    payload.debugMaps?.effectiveTroughHalfWidthMap,
    index
  );
  const backgroundTroughNarrowGate = readOptionalProbeMapValue(
    payload.debugMaps?.backgroundTroughNarrowGateMap,
    index
  );
  const invalidSupportBlackout = readOptionalProbeMapValue(
    payload.debugMaps?.invalidSupportBlackoutMap,
    index
  );
  const troughHalfWidthMm =
    typeof effectiveTroughHalfWidthMm === 'number'
      ? effectiveTroughHalfWidthMm
      : nominalTroughHalfWidthMm;
  const lowerPenalty = readOptionalProbeMapValue(payload.debugMaps?.lowerPenaltyMap, index);
  const totalAttenuation = readOptionalProbeMapValue(payload.debugMaps?.totalAttenuationMap, index);
  const toneMappedValue = readOptionalProbeMapValue(payload.debugMaps?.toneResponseMap, index);
  const participatingSampleCount = readOptionalProbeMapValue(
    payload.debugMaps?.participatingSampleCountMap,
    index
  );
  const localTransmittance =
    typeof totalAttenuation === 'number' ? Math.exp(-Math.max(totalAttenuation, 0)) : undefined;

  return {
    imageId,
    col: safeCol,
    row: safeRow,
    index,
    finalHu,
    meanHu: Number.isFinite(meanHu) ? meanHu : undefined,
    rayMax: Number.isFinite(rayMax) ? rayMax : undefined,
    maxLift:
      Number.isFinite(meanHu) && Number.isFinite(rayMax)
        ? Number(rayMax) - Number(meanHu)
        : undefined,
    validSampleCount: Number.isFinite(validSampleCount) ? validSampleCount : undefined,
    rawSupportDepthMm,
    rawSupportConfidence,
    rawSupportSpreadMm,
    rawSupportDensity,
    rawSupportPeakDominance,
    rawSupportPeakValidity,
    rawSupportPeakConflict,
    rawSupportSecondPeakRatio,
    rawSupportPeakSeparationMm,
    rawSupportPeakAmbiguity,
    rawSupportLocalJumpMm,
    rawSupportContinuity,
    supportFailureDisplay,
    supportDepthMm,
    supportConfidence,
    supportSpreadMm,
    supportDensity,
    supportLocalJumpMm,
    supportContinuity,
    supportDepthDeltaMm:
      typeof supportDepthMm === 'number' && typeof rawSupportDepthMm === 'number'
        ? supportDepthMm - rawSupportDepthMm
        : undefined,
    supportConfidenceDelta:
      typeof supportConfidence === 'number' && typeof rawSupportConfidence === 'number'
        ? supportConfidence - rawSupportConfidence
        : undefined,
    supportSpreadDeltaMm:
      typeof supportSpreadMm === 'number' && typeof rawSupportSpreadMm === 'number'
        ? supportSpreadMm - rawSupportSpreadMm
        : undefined,
    supportDensityDelta:
      typeof supportDensity === 'number' && typeof rawSupportDensity === 'number'
        ? supportDensity - rawSupportDensity
        : undefined,
    nominalTroughHalfWidthMm,
    effectiveTroughHalfWidthMm,
    backgroundTroughNarrowGate,
    invalidSupportBlackout,
    troughLowerMm:
      typeof supportDepthMm === 'number' && typeof troughHalfWidthMm === 'number'
        ? supportDepthMm - troughHalfWidthMm
        : undefined,
    troughUpperMm:
      typeof supportDepthMm === 'number' && typeof troughHalfWidthMm === 'number'
        ? supportDepthMm + troughHalfWidthMm
        : undefined,
    lowerPenalty,
    localTransmittance,
    toneMappedValue,
    participatingSampleCount,
    displayedPath: payload.probeContext?.displayedPath ?? null,
    backend: payload.probeContext?.backend ?? null,
    pipelineMode: payload.probeContext?.pipelineMode ?? null,
    reconstructionMode: payload.probeContext?.reconstructionMode ?? null,
  };
}

function resolvePanoProbePixelFromMouse(
  viewport: cornerstone.Types.IStackViewport,
  payload: PanoImagePayload,
  event: MouseEvent
): {
  col: number;
  row: number;
  canvasX: number;
  canvasY: number;
  mappingMode: 'canvasToWorld' | 'elementLinear';
} | null {
  const element = viewport.element as HTMLDivElement | null | undefined;
  if (!element) {
    return null;
  }

  const rect = element.getBoundingClientRect();
  const canvasX = event.clientX - rect.left;
  const canvasY = event.clientY - rect.top;
  let col = Number.NaN;
  let row = Number.NaN;
  let mappingMode: 'canvasToWorld' | 'elementLinear' = 'elementLinear';
  const columnPixelSpacing =
    Number.isFinite(payload.columnPixelSpacing) && Number(payload.columnPixelSpacing) > 0
      ? Number(payload.columnPixelSpacing)
      : 1;
  const rowPixelSpacing =
    Number.isFinite(payload.rowPixelSpacing) && Number(payload.rowPixelSpacing) > 0
      ? Number(payload.rowPixelSpacing)
      : 1;
  const viewportWithCanvasToWorld = viewport as cornerstone.Types.IStackViewport & {
    canvasToWorld?: (point: [number, number]) => [number, number, number] | number[];
  };

  if (typeof viewportWithCanvasToWorld.canvasToWorld === 'function') {
    try {
      const world = viewportWithCanvasToWorld.canvasToWorld([canvasX, canvasY]);
      const worldX = Number(world?.[0]);
      const worldY = Number(world?.[1]);
      if (Number.isFinite(worldX) && Number.isFinite(worldY)) {
        col = Math.round(worldX / columnPixelSpacing);
        row = Math.round(worldY / rowPixelSpacing);
        mappingMode = 'canvasToWorld';
      }
    } catch (error) {
      console.warn(
        '[CPR] Pano debug probe canvasToWorld mapping failed; falling back to element mapping.',
        error
      );
    }
  }

  if (!Number.isFinite(col) || !Number.isFinite(row)) {
    const safeWidth = Math.max(rect.width, 1);
    const safeHeight = Math.max(rect.height, 1);
    col = Math.round((canvasX / safeWidth) * Math.max(payload.width - 1, 0));
    row = Math.round((canvasY / safeHeight) * Math.max(payload.height - 1, 0));
    mappingMode = 'elementLinear';
  }

  return {
    col,
    row,
    canvasX,
    canvasY,
    mappingMode,
  };
}

function installPanoDebugProbe(
  runId: string,
  viewport: cornerstone.Types.IStackViewport,
  panoImageId: string
): void {
  clearPanoDebugProbe();

  if (typeof window === 'undefined') {
    return;
  }

  const element = viewport.element as HTMLDivElement | null | undefined;
  if (!element) {
    return;
  }

  const debugHost = window as Window & {
    cprDebug?: {
      latestPanoImageId?: string | null;
      probePanoAt?: (
        col: number,
        row: number,
        imageId?: string | null
      ) => PanoDebugProbeSample | null;
      detachPanoClickProbe?: () => void;
    };
  };

  const probeAt = (
    col: number,
    row: number,
    imageId: string | null = getCurrentStackImageId(viewport) ?? panoImageId
  ): PanoDebugProbeSample | null => {
    const payload = getPanoImagePayload(imageId);
    const sample = readPanoDebugProbeSample(payload, imageId, col, row);
    if (!sample) {
      return null;
    }

    console.log(`[CPR][${runId}] PANO_DEBUG_PROBE`, sample);
    if (
      typeof sample.supportDepthMm === 'number' ||
      typeof sample.localTransmittance === 'number' ||
      typeof sample.toneMappedValue === 'number'
    ) {
      console.log(
        '[CPR-PROBE-EXT-JSON]',
        JSON.stringify({
          runId,
          imageId: sample.imageId,
          col: sample.col,
          row: sample.row,
          displayedPath: sample.displayedPath ?? null,
          backend: sample.backend ?? null,
          pipelineMode: sample.pipelineMode ?? null,
          reconstructionMode: sample.reconstructionMode ?? null,
          rawSupportDepthMm: sample.rawSupportDepthMm ?? null,
          rawSupportConfidence: sample.rawSupportConfidence ?? null,
          rawSupportSpreadMm: sample.rawSupportSpreadMm ?? null,
          rawSupportDensity: sample.rawSupportDensity ?? null,
          rawSupportPeakDominance: sample.rawSupportPeakDominance ?? null,
          rawSupportPeakValidity: sample.rawSupportPeakValidity ?? null,
          rawSupportPeakConflict: sample.rawSupportPeakConflict ?? null,
          rawSupportSecondPeakRatio: sample.rawSupportSecondPeakRatio ?? null,
          rawSupportPeakSeparationMm: sample.rawSupportPeakSeparationMm ?? null,
          rawSupportPeakAmbiguity: sample.rawSupportPeakAmbiguity ?? null,
          rawSupportLocalJumpMm: sample.rawSupportLocalJumpMm ?? null,
          rawSupportContinuity: sample.rawSupportContinuity ?? null,
          supportFailureDisplay: sample.supportFailureDisplay ?? null,
          supportDepthMm: sample.supportDepthMm ?? null,
          supportConfidence: sample.supportConfidence ?? null,
          supportSpreadMm: sample.supportSpreadMm ?? null,
          supportDensity: sample.supportDensity ?? null,
          supportLocalJumpMm: sample.supportLocalJumpMm ?? null,
          supportContinuity: sample.supportContinuity ?? null,
          supportDepthDeltaMm: sample.supportDepthDeltaMm ?? null,
          supportConfidenceDelta: sample.supportConfidenceDelta ?? null,
          supportSpreadDeltaMm: sample.supportSpreadDeltaMm ?? null,
          supportDensityDelta: sample.supportDensityDelta ?? null,
          nominalTroughHalfWidthMm: sample.nominalTroughHalfWidthMm ?? null,
          effectiveTroughHalfWidthMm: sample.effectiveTroughHalfWidthMm ?? null,
          backgroundTroughNarrowGate: sample.backgroundTroughNarrowGate ?? null,
          invalidSupportBlackout: sample.invalidSupportBlackout ?? null,
          troughLowerMm: sample.troughLowerMm ?? null,
          troughUpperMm: sample.troughUpperMm ?? null,
          localTransmittance: sample.localTransmittance ?? null,
          toneMappedValue: sample.toneMappedValue ?? null,
          lowerPenalty: sample.lowerPenalty ?? null,
          participatingSampleCount: sample.participatingSampleCount ?? null,
        })
      );
    }
    return sample;
  };

  const onClick = (event: MouseEvent) => {
    const imageId = getCurrentStackImageId(viewport) ?? panoImageId;
    const payload = getPanoImagePayload(imageId);
    if (!payload) {
      console.warn('[CPR] Pano debug probe click ignored: no pano payload found.', { imageId });
      return;
    }

    const location = resolvePanoProbePixelFromMouse(viewport, payload, event);
    if (!location) {
      console.warn('[CPR] Pano debug probe click ignored: viewport element unavailable.');
      return;
    }

    const sample = probeAt(location.col, location.row, imageId);
    if (!sample) {
      return;
    }

    console.log(`[CPR][${runId}] PANO_DEBUG_PROBE_CLICK`, {
      ...sample,
      canvasX: Math.round(location.canvasX * 100) / 100,
      canvasY: Math.round(location.canvasY * 100) / 100,
      mappingMode: location.mappingMode,
    });
  };

  element.addEventListener('click', onClick);

  debugHost.cprDebug = {
    ...debugHost.cprDebug,
    latestPanoImageId: panoImageId,
    probePanoAt: probeAt,
    detachPanoClickProbe: clearPanoDebugProbe,
  };

  console.log(`[CPR][${runId}] Installed pano debug probe`, {
    panoImageId,
    usage: 'Click the pano viewport or call window.cprDebug?.probePanoAt(col, row).',
  });

  activePanoDebugProbeCleanup = () => {
    element.removeEventListener('click', onClick);
    const currentDebug = debugHost.cprDebug;
    if (currentDebug?.detachPanoClickProbe === clearPanoDebugProbe) {
      delete currentDebug.detachPanoClickProbe;
    }
  };
}

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
  backgroundToneSampleCount?: number;
  backgroundToneP95?: number;
  backgroundToneP99?: number;
  backgroundToneMax?: number;
  backgroundOutlierFraction05?: number;
  backgroundOutlierFraction10?: number;
}

interface PanoVoiSettings {
  lower: number;
  upper: number;
  windowWidth: number;
  windowCenter: number;
}

interface HostedPanoVoiAuthority {
  runId: string | null;
  sourceVolumeId: string | null;
  authoritativeVoi: PanoVoiSettings | null;
  suppressPanoViewportEventsUntil: number;
}

const CPR_VTK_PANO_STARTUP_VOI_GUARD_MS = 2000;

function formatReadablePanoValue(value: number | null | undefined, fractionDigits = 1): string {
  if (!Number.isFinite(value)) {
    return 'na';
  }

  return Number(value).toFixed(fractionDigits);
}

function toNonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

function extractCprOutputSignature(
  workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined
): CprOutputSignature {
  const outputSignature =
    workerDebugPayload &&
    typeof workerDebugPayload === 'object' &&
    workerDebugPayload.outputSignature &&
    typeof workerDebugPayload.outputSignature === 'object'
      ? workerDebugPayload.outputSignature
      : null;

  return {
    sampledCount: Number.isFinite(outputSignature?.sampledCount)
      ? Number(outputSignature?.sampledCount)
      : null,
    checksum: Number.isFinite(outputSignature?.checksum) ? Number(outputSignature?.checksum) : null,
    absChecksum: Number.isFinite(outputSignature?.absChecksum)
      ? Number(outputSignature?.absChecksum)
      : null,
    first16: Array.isArray(outputSignature?.first16)
      ? outputSignature.first16
          .map(value => (Number.isFinite(value) ? Number(value) : null))
          .filter((value): value is number => value !== null)
      : [],
  };
}

function buildCprOutputSignatureKey(signature: CprOutputSignature): string | null {
  if (!Number.isFinite(signature.checksum) || !signature.first16.length) {
    return null;
  }

  return JSON.stringify([signature.checksum, signature.first16]);
}

function parsePanoReconBackend(value: unknown): CprPanoReconBackend | null {
  return value === 'gpu' || value === 'cpu' ? value : null;
}

function resolveWorkerDisplayRouteDiagnostic(
  workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined
): {
  backend: CprPanoReconBackend;
  requestedBackend: CprPanoReconBackend;
  pipelineMode: string;
  fallbackReason: string | null;
  phase2GatePassed: boolean | null;
} {
  const diagnostic =
    workerDebugPayload &&
    typeof workerDebugPayload === 'object' &&
    workerDebugPayload.diagnostic &&
    typeof workerDebugPayload.diagnostic === 'object'
      ? (workerDebugPayload.diagnostic as Record<string, unknown>)
      : null;
  const gpuRender =
    diagnostic?.gpuRender && typeof diagnostic.gpuRender === 'object'
      ? (diagnostic.gpuRender as Record<string, unknown>)
      : null;
  const requestedBackend =
    parsePanoReconBackend(diagnostic?.requestedRenderBackend) ?? CPR_PANO_RECON_BACKEND_DEFAULT;
  const backend = parsePanoReconBackend(diagnostic?.renderBackend) ?? requestedBackend;
  const phase2GatePassed =
    typeof gpuRender?.phase2GatePassed === 'boolean'
      ? Boolean(gpuRender?.phase2GatePassed)
      : typeof diagnostic?.phase2GatePassed === 'boolean'
        ? Boolean(diagnostic?.phase2GatePassed)
        : null;
  const pipelineMode =
    toNonEmptyString(diagnostic?.pipelineMode) ||
    toNonEmptyString(gpuRender?.pipelineMode) ||
    toNonEmptyString(diagnostic?.reconstructionMode) ||
    'unknown';
  const fallbackReason =
    toNonEmptyString(diagnostic?.fallbackReason) ||
    (requestedBackend === 'gpu' && backend === 'gpu' && phase2GatePassed === false
      ? 'gpu-phase2-gate-failed'
      : null) ||
    (requestedBackend === 'gpu' && backend === 'cpu' ? 'gpu-request-resolved-on-cpu' : null);

  return {
    backend,
    requestedBackend,
    pipelineMode,
    fallbackReason,
    phase2GatePassed,
  };
}

function resolveDisplayedOutputModeFromWorkerResult(params: {
  routeDiagnostic: ReturnType<typeof resolveWorkerDisplayRouteDiagnostic>;
  workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined;
}): 'legacy' | 'workerGpuPhase2' | 'virtualPanoPhase2' {
  const diagnostic =
    params.workerDebugPayload &&
    typeof params.workerDebugPayload === 'object' &&
    params.workerDebugPayload.diagnostic &&
    typeof params.workerDebugPayload.diagnostic === 'object'
      ? (params.workerDebugPayload.diagnostic as Record<string, unknown>)
      : null;
  const resolvedReconstructionMode =
    toNonEmptyString(diagnostic?.reconstructionMode) ||
    toNonEmptyString(diagnostic?.pipelineMode) ||
    'legacy';

  if (params.routeDiagnostic.backend === 'cpu' && resolvedReconstructionMode === 'virtualPano') {
    return 'virtualPanoPhase2';
  }

  if (
    params.routeDiagnostic.backend === 'gpu' &&
    params.routeDiagnostic.phase2GatePassed !== false
  ) {
    return 'workerGpuPhase2';
  }

  return 'legacy';
}

function logPanoDisplayRoute(params: {
  runId: string;
  displayedPath: CprPanoDisplayPath;
  backend: CprPanoDisplayBackend;
  pipelineMode: string;
  fallbackReason?: string | null;
  requestedBackend?: CprPanoReconBackend | null;
}): void {
  const requestedBackendSuffix =
    params.requestedBackend && params.requestedBackend !== params.backend
      ? ` requestedBackend=${params.requestedBackend}`
      : '';
  console.log(
    `[CPR-PANO-DISPLAY] run=${params.runId} displayedPath=${params.displayedPath} ` +
      `backend=${params.backend} pipelineMode=${params.pipelineMode} ` +
      `fallbackReason=${params.fallbackReason ?? 'none'}${requestedBackendSuffix}`
  );
}

function logReconModeJson(params: {
  runId: string;
  displayedPath: CprPanoDisplayPath;
  backend: CprPanoDisplayBackend;
  pipelineMode: string;
  reconstructionMode: string;
  displaySource: CprPanoDisplaySource;
  referencePathAvailable: boolean;
  sourceVolumeId?: string | null;
  fallbackReason?: string | null;
  requestedBackend?: CprPanoReconBackend | null;
  phase2GatePassed?: boolean | null;
}): void {
  console.log(
    '[CPR-RECON-MODE-JSON]',
    JSON.stringify({
      runId: params.runId,
      displayedPath: params.displayedPath,
      backend: params.backend,
      requestedBackend: params.requestedBackend ?? null,
      pipelineMode: params.pipelineMode,
      reconstructionMode: params.reconstructionMode,
      displaySource: params.displaySource,
      referencePathAvailable: params.referencePathAvailable,
      referenceOrLegacyFallbackAllowed: CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK,
      sourceVolumeId: params.sourceVolumeId ?? null,
      fallbackReason: params.fallbackReason ?? null,
      phase2GatePassed:
        typeof params.phase2GatePassed === 'boolean' ? params.phase2GatePassed : null,
    })
  );
}

function isHuLikeRange(minValue: number, maxValue: number): boolean {
  return (
    Number.isFinite(minValue) && Number.isFinite(maxValue) && minValue >= -5000 && maxValue <= 7000
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

function isDualArchProjectionRenderMode(renderSupportMode: string | null | undefined): boolean {
  return renderSupportMode === 'dualArchProjection';
}

function isLikelyPoorPanoQuality(
  summary: FloatBufferDebugSummary | null,
  renderSupportMode?: string | null
): boolean {
  if (!summary || summary.sampledCount < 100) {
    return true;
  }

  const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);

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
  const hasLowerBandFill = isDualArchProjection
    ? summary.lowerBandP50 > 420 || summary.lowerBandBrightFraction > 0.92
    : summary.lowerBandP50 > 140 || summary.lowerBandBrightFraction > 0.7;
  const detailAnisotropyRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
  const balancedDetailEdgeMean = Math.min(
    summary.detailBandHorizontalEdgeMean,
    summary.detailBandVerticalEdgeMean * 2.4
  );
  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const hasToothBandSaturation = isDualArchProjection
    ? summary.toothBandP10 > 320 || summary.toothBandMean > 1280
    : summary.toothBandP10 > 220 || summary.toothBandMean > 880;
  const hasWhiteBlobArtifact =
    summary.toothBandBrightFraction > 0.36 &&
    summary.toothBandP10 > (isDualArchProjection ? 180 : 120) &&
    summary.toothBandP90 > (isDualArchProjection ? 1700 : 1450);
  const hasWeakToothBandContrast = toothBandContrastRange < (isDualArchProjection ? 100 : 120);
  const hasWeakDentalSeparation = balancedDetailEdgeMean < 38;
  const hasShearWarping = summary.detailBandHorizontalEdgeMean > 220 && detailAnisotropyRatio > 3;

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

function getHardRejectReason(
  summary: FloatBufferDebugSummary | null,
  renderSupportMode?: string | null
): string | null {
  if (!summary || summary.sampledCount < 100) {
    return 'insufficient-samples';
  }

  const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);

  if (isSeverelyCorruptedPanoOutput(summary)) {
    return 'severely-corrupted';
  }

  if (summary.meanAbsDelta > 760) {
    return 'speckle-noise';
  }

  if (
    summary.toothBandP10 > (isDualArchProjection ? 360 : 280) &&
    summary.toothBandMean > (isDualArchProjection ? 1280 : 950)
  ) {
    return 'tooth-band-saturation';
  }

  if (
    summary.toothBandBrightFraction > 0.42 &&
    summary.toothBandP10 > (isDualArchProjection ? 180 : 120) &&
    summary.toothBandP90 > (isDualArchProjection ? 1750 : 1550)
  ) {
    return 'white-blob-artifact';
  }

  if (
    summary.lowerBandP50 > (isDualArchProjection ? 520 : 260) &&
    summary.lowerBandBrightFraction > (isDualArchProjection ? 0.96 : 0.78)
  ) {
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

function scorePanoQuality(
  summary: FloatBufferDebugSummary | null,
  renderSupportMode?: string | null
): number {
  if (!summary || summary.sampledCount < 100) {
    return -Infinity;
  }

  const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);

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

  if (isDualArchProjection) {
    if (summary.lowerBandP50 <= 180) {
      score += 4;
    } else if (summary.lowerBandP50 <= 280) {
      score += 2;
    } else if (summary.lowerBandP50 <= 420) {
      score += 0;
    } else {
      score -= Math.min(10, (summary.lowerBandP50 - 420) / 55);
    }

    if (summary.lowerBandBrightFraction <= 0.72) {
      score += 4;
    } else if (summary.lowerBandBrightFraction <= 0.82) {
      score += 2;
    } else if (summary.lowerBandBrightFraction <= 0.9) {
      score += 0;
    } else {
      score -= Math.min(10, (summary.lowerBandBrightFraction - 0.9) * 55);
    }
  } else {
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
  }

  if (balancedDetailEdgeMean >= 70 && balancedDetailEdgeMean <= 220) {
    score += 5;
  } else if (balancedDetailEdgeMean >= 45) {
    score += 2;
  } else {
    score -= 3;
  }

  if (
    summary.toothBandMean >= 160 &&
    summary.toothBandMean <= (isDualArchProjection ? 980 : 760)
  ) {
    score += 3;
  } else {
    const targetToothMean = isDualArchProjection ? 600 : 460;
    score -= Math.min(10, Math.abs(summary.toothBandMean - targetToothMean) / 110);
  }

  if (summary.toothBandP10 <= -120) {
    score += 2;
  } else if (summary.toothBandP10 <= (isDualArchProjection ? 180 : 80)) {
    score += 1;
  } else {
    score -= Math.min(12, (summary.toothBandP10 - (isDualArchProjection ? 180 : 80)) / 32);
  }

  if (summary.toothBandBrightFraction <= 0.14) {
    score += 2;
  } else if (summary.toothBandBrightFraction <= 0.24) {
    score += 1;
  } else {
    score -= Math.min(14, (summary.toothBandBrightFraction - 0.24) * 24);
  }

  if (toothBandContrastRange >= (isDualArchProjection ? 280 : 360) && toothBandContrastRange <= 1700) {
    score += 3;
  } else if (toothBandContrastRange >= (isDualArchProjection ? 180 : 240)) {
    score += 1;
  } else {
    score -= Math.min(8, ((isDualArchProjection ? 180 : 240) - toothBandContrastRange) / 36);
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

function scoreHardRejectedPanoFallback(
  summary: FloatBufferDebugSummary | null,
  renderSupportMode?: string | null
): number {
  if (!summary || summary.sampledCount < 100) {
    return -Infinity;
  }

  const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);

  const toothBandContrastRange = summary.toothBandP90 - summary.toothBandP10;
  const detailRatio =
    summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);

  let score = 0;
  score -= Math.max(0, summary.lowerBandP50 - (isDualArchProjection ? 240 : -160)) / 45;
  score -= Math.max(0, summary.lowerBandBrightFraction - (isDualArchProjection ? 0.78 : 0.28)) * 30;
  score -= Math.max(0, summary.toothBandMean - 760) / 80;
  score -= Math.max(0, summary.toothBandP10 - (isDualArchProjection ? 180 : 80)) / 30;
  score -= Math.max(0, summary.toothBandBrightFraction - 0.24) * 26;
  score += Math.max(-6, Math.min(6, (toothBandContrastRange - 320) / 90));
  score -= Math.max(0, detailRatio - 2.8) * 3.2;
  score -= Math.max(0, summary.meanAbsDelta - 520) / 90;
  score -= Math.max(0, summary.fractionAbove3000 - 0.005) * 120;
  return score;
}

type Phase4QualityGateCandidateSource =
  | 'worker-gpu-support-surface'
  | 'worker-cpu-virtual-pano'
  | 'worker-cpu-arch-guided-synthetic'
  | 'worker-legacy';

interface Phase4QualityGateMetrics {
  sampledCount: number | null;
  lowerBandBrightFraction: number | null;
  lowerBandP50: number | null;
  toothBandMean: number | null;
  toothBandP10: number | null;
  toothBandP90: number | null;
  toothBandContrastRange: number | null;
  detailBandHorizontalEdgeMean: number | null;
  detailBandVerticalEdgeMean: number | null;
  supportDepthClampFraction: number | null;
  supportDepthStdMm: number | null;
  pathJumpP95Mm: number | null;
  supportConfidenceP10: number | null;
  supportConfidenceP50: number | null;
  supportConfidenceP90: number | null;
  supportPathConfidenceP10: number | null;
  supportPathConfidenceP50: number | null;
  supportPathConfidenceP90: number | null;
  supportUnstableColumnFraction: number | null;
  supportLongestUnstableRunColumns: number | null;
  supportAmbiguousColumnFraction: number | null;
  supportForcedDriftFraction: number | null;
  supportBestDepthDriftP95Mm: number | null;
  supportScoreGapP50: number | null;
  troughSigmaP50Mm: number | null;
  approxTroughHalfWidthP50Mm: number | null;
  effectiveTroughHalfWidthP50Mm: number | null;
  effectiveTroughHalfWidthP90Mm: number | null;
  participatingSamplesP50: number | null;
  participatingSamplesP90: number | null;
  backgroundTroughNarrowGateP50: number | null;
  backgroundTroughNarrowGateP90: number | null;
  blackClipFraction: number | null;
  fractionBelowMinus950: number | null;
  fractionAbove3000: number | null;
  renderSupportMode: string | null;
  contextHuP50: number | null;
  contextBlendMean: number | null;
  contextWeightFractionMean: number | null;
  columnSupportReliabilityP50: number | null;
  upperDetailHuP50: number | null;
  lowerDetailHuP50: number | null;
  detailSampleFractionMean: number | null;
  shadowLiftMean: number | null;
}

interface Phase4SupportSurfaceRiskSummary {
  riskLevel: 'low' | 'moderate' | 'elevated' | 'high';
  riskScore: number;
  riskFlags: string[];
  baselineFingerprint: string;
  stability: {
    supportConfidenceP10: number | null;
    supportConfidenceP50: number | null;
    supportConfidenceP90: number | null;
    supportPathConfidenceP10: number | null;
    supportPathConfidenceP50: number | null;
    supportPathConfidenceP90: number | null;
    supportDepthClampFraction: number | null;
    supportDepthStdMm: number | null;
    pathJumpP95Mm: number | null;
    supportUnstableColumnFraction: number | null;
    supportLongestUnstableRunColumns: number | null;
    supportAmbiguousColumnFraction: number | null;
    supportForcedDriftFraction: number | null;
    supportBestDepthDriftP95Mm: number | null;
    supportScoreGapP50: number | null;
  };
  background: {
    blackClipFraction: number | null;
    backgroundToneP95: number | null;
    backgroundToneP99: number | null;
    backgroundToneMax: number | null;
    backgroundOutlierFraction05: number | null;
    backgroundOutlierFraction10: number | null;
  };
  anatomy: {
    lowerBandBrightFraction: number | null;
    lowerBandP50: number | null;
    toothBandMean: number | null;
    toothBandP10: number | null;
    toothBandP90: number | null;
    toothBandContrastRange: number | null;
    fractionBelowMinus950: number | null;
    fractionAbove3000: number | null;
  };
}

interface Phase4QualityGateCandidate {
  candidateSource: Phase4QualityGateCandidateSource;
  displayedPath: CprPanoDisplayPath;
  backend: CprPanoDisplayBackend;
  requestedBackend: CprPanoReconBackend | null;
  attemptLabel: string | null;
  requestedReconstructionMode: string | null;
  reconstructionMode: string;
  pipelineMode: string;
  sourceVolumeId: string | null;
  fallbackReason: string | null;
  phase2GatePassed: boolean | null;
  qualityBase: number | null;
  qualityScore: number | null;
  hardRejectReason: string | null;
  workerRejectReasons: string[];
  workerUsedAsOutput: boolean | null;
  borderlineAcceptedReason: string | null;
  pass: boolean;
  rejectReasons: string[];
  metrics: Phase4QualityGateMetrics;
  supportSurfaceRiskSummary: Phase4SupportSurfaceRiskSummary;
}

interface Phase4DegradedPreviewAssessment {
  catastrophic: boolean;
  catastrophicReasons: string[];
}

interface Phase4DisplaySelectionBreakdown {
  legacyQualityScore: number;
  legacyQualityBase: number;
  legacyDetailReward: number;
  qualityGatePassedBonus: number;
  legacyQualityScoreComponent: number;
  supportConfidenceReward: number;
  supportPathConfidenceReward: number;
  backgroundLeakagePenalty: number;
  troughAdmissionPenalty: number;
  broadWeakSupportPenalty: number;
  preferredTightRootFamilyBonus: number;
  hardRejectPenalty: number;
  duplicatePenalty: number;
  displaySelectionScore: number;
  supportConfidenceP50: number | null;
  supportPathConfidenceP50: number | null;
  backgroundOutlierFraction05: number | null;
  backgroundOutlierFraction10: number | null;
  effectiveTroughHalfWidthP50Mm: number | null;
  effectiveTroughHalfWidthP90Mm: number | null;
  participatingSamplesP50: number | null;
  participatingSamplesP90: number | null;
  backgroundTroughNarrowGateP90: number | null;
  actualVertHalfMm: number;
  verticalCenterOffsetMm: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  duplicateOfLabel: string | null;
  preferredTightRootFamily: boolean;
  renderSupportMode: string | null;
}

function clampPhase4DisplaySelectionUnit(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function buildPhase4DisplaySelectionBreakdown(params: {
  qualityGatePassed?: boolean | null;
  qualityBase: number;
  qualityScore: number;
  detailReward: number;
  hardRejectReason: string | null;
  summary: FloatBufferDebugSummary | null;
  metrics: Phase4QualityGateMetrics;
  requestedBackend: CprPanoReconBackend;
  resolvedBackend: CprPanoReconBackend;
  aggregation: 'MIP' | 'MEAN';
  actualVertHalfMm: number;
  verticalCenterOffsetMm: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  duplicateOfLabel?: string | null;
}): Phase4DisplaySelectionBreakdown {
  const supportConfidenceP50 = params.metrics.supportConfidenceP50;
  const supportPathConfidenceP50 = params.metrics.supportPathConfidenceP50;
  const backgroundOutlierFraction05 = params.summary?.backgroundOutlierFraction05 ?? null;
  const backgroundOutlierFraction10 = params.summary?.backgroundOutlierFraction10 ?? null;
  const effectiveTroughHalfWidthP50Mm = params.metrics.effectiveTroughHalfWidthP50Mm;
  const effectiveTroughHalfWidthP90Mm = params.metrics.effectiveTroughHalfWidthP90Mm;
  const participatingSamplesP50 = params.metrics.participatingSamplesP50;
  const participatingSamplesP90 = params.metrics.participatingSamplesP90;
  const backgroundTroughNarrowGateP90 = params.metrics.backgroundTroughNarrowGateP90;
  const isGpuMeanAttempt =
    params.requestedBackend === 'gpu' &&
    params.resolvedBackend === 'gpu' &&
    params.aggregation === 'MEAN';
  const preferredTightRootFamily =
    isGpuMeanAttempt &&
    supportConfidenceP50 !== null &&
    supportConfidenceP50 >= 0.09 &&
    (backgroundOutlierFraction05 ?? Number.POSITIVE_INFINITY) <= 0.32 &&
    (backgroundOutlierFraction10 ?? Number.POSITIVE_INFINITY) <= 0.12 &&
    params.slabHalfThicknessMm <= 1.6 &&
    params.actualVertHalfMm <= 15.2 &&
    params.verticalCenterOffsetMm <= -5.0;
  const qualityGatePassedBonus = params.qualityGatePassed === true ? 200 : 0;
  const legacyQualityScoreComponent = params.qualityScore * 0.3;
  const supportConfidenceReward =
    supportConfidenceP50 !== null
      ? clampPhase4DisplaySelectionUnit((supportConfidenceP50 - 0.06) / 0.06) * 5.8
      : 0;
  const supportPathConfidenceReward =
    supportPathConfidenceP50 !== null
      ? clampPhase4DisplaySelectionUnit((supportPathConfidenceP50 - 0.66) / 0.08) * 1.1
      : 0;
  const backgroundLeakagePenalty =
    Math.max(0, (backgroundOutlierFraction05 ?? 0) - 0.24) * 30 +
    Math.max(0, (backgroundOutlierFraction10 ?? 0) - 0.09) * 38;
  const troughAdmissionPenalty =
    Math.max(0, (effectiveTroughHalfWidthP50Mm ?? 0) - 0.39) * 16 +
    Math.max(0, (participatingSamplesP50 ?? 0) - 3.35) * 7 +
    Math.max(0, (backgroundTroughNarrowGateP90 ?? 0) - 0.97) * 12;
  const broadWeakSupportPenalty =
    supportConfidenceP50 !== null && supportConfidenceP50 < 0.09
      ? Math.max(0, params.slabHalfThicknessMm - 1.55) * 7 +
        Math.max(0, params.actualVertHalfMm - 15.2) * 1.4 +
        Math.max(0, params.verticalCenterOffsetMm + 4.5) * 0.65
      : 0;
  const preferredTightRootFamilyBonus = preferredTightRootFamily
    ? clampPhase4DisplaySelectionUnit((-params.verticalCenterOffsetMm - 4.8) / 2.8) * 2.5 +
      clampPhase4DisplaySelectionUnit((15.2 - params.actualVertHalfMm) / 2.8) * 1.2 +
      clampPhase4DisplaySelectionUnit((1.6 - params.slabHalfThicknessMm) / 0.35) * 1.4
    : 0;
  const hardRejectPenalty = params.hardRejectReason ? 18 : 0;
  const duplicatePenalty = params.duplicateOfLabel ? 12 : 0;
  const displaySelectionScore =
    qualityGatePassedBonus +
    legacyQualityScoreComponent +
    supportConfidenceReward +
    supportPathConfidenceReward +
    preferredTightRootFamilyBonus -
    backgroundLeakagePenalty -
    troughAdmissionPenalty -
    broadWeakSupportPenalty -
    hardRejectPenalty -
    duplicatePenalty;

  return {
    legacyQualityScore: params.qualityScore,
    legacyQualityBase: params.qualityBase,
    legacyDetailReward: params.detailReward,
    qualityGatePassedBonus,
    legacyQualityScoreComponent,
    supportConfidenceReward,
    supportPathConfidenceReward,
    backgroundLeakagePenalty,
    troughAdmissionPenalty,
    broadWeakSupportPenalty,
    preferredTightRootFamilyBonus,
    hardRejectPenalty,
    duplicatePenalty,
    displaySelectionScore,
    supportConfidenceP50,
    supportPathConfidenceP50,
    backgroundOutlierFraction05,
    backgroundOutlierFraction10,
    effectiveTroughHalfWidthP50Mm,
    effectiveTroughHalfWidthP90Mm,
    participatingSamplesP50,
    participatingSamplesP90,
    backgroundTroughNarrowGateP90,
    actualVertHalfMm: params.actualVertHalfMm,
    verticalCenterOffsetMm: params.verticalCenterOffsetMm,
    slabHalfThicknessMm: params.slabHalfThicknessMm,
    slabSamples: params.slabSamples,
    duplicateOfLabel: params.duplicateOfLabel ?? null,
    preferredTightRootFamily,
    renderSupportMode: params.metrics.renderSupportMode,
  };
}

function isMateriallyCleanerPhase4DisplayCandidate(
  a: Phase4DisplaySelectionBreakdown,
  b: Phase4DisplaySelectionBreakdown
): boolean {
  if (
    a.supportConfidenceP50 === null ||
    b.supportConfidenceP50 === null ||
    a.backgroundOutlierFraction05 === null ||
    b.backgroundOutlierFraction05 === null ||
    a.backgroundOutlierFraction10 === null ||
    b.backgroundOutlierFraction10 === null
  ) {
    return false;
  }

  const supportImproved = a.supportConfidenceP50 >= b.supportConfidenceP50 + 0.012;
  const background05Improved =
    a.backgroundOutlierFraction05 <= b.backgroundOutlierFraction05 - 0.02;
  const background10Improved =
    a.backgroundOutlierFraction10 <= b.backgroundOutlierFraction10 - 0.01;
  const troughImproved =
    a.effectiveTroughHalfWidthP50Mm !== null &&
    b.effectiveTroughHalfWidthP50Mm !== null &&
    a.effectiveTroughHalfWidthP50Mm <= b.effectiveTroughHalfWidthP50Mm - 0.02;
  const sampleAdmissionImproved =
    a.participatingSamplesP50 !== null &&
    b.participatingSamplesP50 !== null &&
    a.participatingSamplesP50 <= b.participatingSamplesP50 - 0.25;

  return (
    supportImproved &&
    background05Improved &&
    background10Improved &&
    (troughImproved || sampleAdmissionImproved)
  );
}

function comparePhase4DisplaySelectionBreakdowns(
  a: {
    qualityGatePassed?: boolean | null;
    qualityBase: number;
    qualityScore: number;
    hardRejectReason: string | null;
    summary: FloatBufferDebugSummary | null;
    breakdown: Phase4DisplaySelectionBreakdown;
  },
  b: {
    qualityGatePassed?: boolean | null;
    qualityBase: number;
    qualityScore: number;
    hardRejectReason: string | null;
    summary: FloatBufferDebugSummary | null;
    breakdown: Phase4DisplaySelectionBreakdown;
  }
): number {
  const aGatePassed = a.qualityGatePassed ?? true;
  const bGatePassed = b.qualityGatePassed ?? true;
  if (aGatePassed !== bGatePassed) {
    return aGatePassed ? -1 : 1;
  }
  const aHardRejected = !!a.hardRejectReason;
  const bHardRejected = !!b.hardRejectReason;
  if (aHardRejected !== bHardRejected) {
    return aHardRejected ? 1 : -1;
  }
  if (aHardRejected && bHardRejected) {
    const fallbackDelta =
      scoreHardRejectedPanoFallback(a.summary, a.breakdown.renderSupportMode) -
      scoreHardRejectedPanoFallback(b.summary, b.breakdown.renderSupportMode);
    if (Math.abs(fallbackDelta) > 1e-6) {
      return fallbackDelta > 0 ? -1 : 1;
    }
  }
  const aCleanerThanB = isMateriallyCleanerPhase4DisplayCandidate(a.breakdown, b.breakdown);
  const bCleanerThanA = isMateriallyCleanerPhase4DisplayCandidate(b.breakdown, a.breakdown);
  if (aCleanerThanB !== bCleanerThanA) {
    return aCleanerThanB ? -1 : 1;
  }
  if (
    a.breakdown.preferredTightRootFamily !== b.breakdown.preferredTightRootFamily &&
    Math.abs(a.breakdown.displaySelectionScore - b.breakdown.displaySelectionScore) <= 4
  ) {
    return a.breakdown.preferredTightRootFamily ? -1 : 1;
  }

  return (
    b.breakdown.displaySelectionScore - a.breakdown.displaySelectionScore ||
    b.qualityScore - a.qualityScore ||
    b.qualityBase - a.qualityBase
  );
}

function compareRankedPanoOutputs(
  a: {
    qualityGatePassed?: boolean | null;
    qualityBase: number;
    qualityScore: number;
    hardRejectReason: string | null;
    summary: FloatBufferDebugSummary | null;
    qualityGateMetrics?: Phase4QualityGateMetrics | null;
    supportSurfaceRiskSummary?: Phase4SupportSurfaceRiskSummary | null;
  },
  b: {
    qualityGatePassed?: boolean | null;
    qualityBase: number;
    qualityScore: number;
    hardRejectReason: string | null;
    summary: FloatBufferDebugSummary | null;
    qualityGateMetrics?: Phase4QualityGateMetrics | null;
    supportSurfaceRiskSummary?: Phase4SupportSurfaceRiskSummary | null;
  }
): number {
  const aGatePassed = a.qualityGatePassed ?? true;
  const bGatePassed = b.qualityGatePassed ?? true;
  if (aGatePassed !== bGatePassed) {
    return aGatePassed ? -1 : 1;
  }
  const aHardRejected = !!a.hardRejectReason;
  const bHardRejected = !!b.hardRejectReason;
  if (aHardRejected !== bHardRejected) {
    return aHardRejected ? 1 : -1;
  }
  if (aHardRejected && bHardRejected) {
    const fallbackDelta =
      scoreHardRejectedPanoFallback(a.summary, a.qualityGateMetrics?.renderSupportMode) -
      scoreHardRejectedPanoFallback(b.summary, b.qualityGateMetrics?.renderSupportMode);
    if (Math.abs(fallbackDelta) > 1e-6) {
      return fallbackDelta > 0 ? -1 : 1;
    }
  }
  const getRiskRank = (riskSummary?: Phase4SupportSurfaceRiskSummary | null): number => {
    switch (riskSummary?.riskLevel) {
      case 'high':
        return 3;
      case 'elevated':
        return 2;
      case 'moderate':
        return 1;
      default:
        return 0;
    }
  };
  const aRiskRank = getRiskRank(a.supportSurfaceRiskSummary);
  const bRiskRank = getRiskRank(b.supportSurfaceRiskSummary);
  if (aRiskRank !== bRiskRank) {
    return aRiskRank - bRiskRank;
  }
  return b.qualityScore - a.qualityScore || b.qualityBase - a.qualityBase;
}

function readPhase4DiagnosticRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

function readPhase4DiagnosticNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function readPhase4DiagnosticBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function readPhase4DiagnosticStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function roundPhase4DiagnosticNumber(value: number | null | undefined, digits = 3): number | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }

  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function formatPhase4FingerprintValue(value: number | null | undefined, digits = 3): string {
  const rounded = roundPhase4DiagnosticNumber(value, digits);
  return rounded === null ? 'null' : String(rounded);
}

function isPhase4LowerBandRejectReason(reason: string): boolean {
  return (
    reason === 'lower-band-bright-fraction-too-high' ||
    reason === 'lower-band-p50-too-high' ||
    reason === 'lower-band-mean-too-high' ||
    reason === 'lower-suppression-ratio-too-high'
  );
}

function extractPhase4QualityGateMetrics(
  summary: FloatBufferDebugSummary | null,
  workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined
): Phase4QualityGateMetrics {
  const diagnostic = readPhase4DiagnosticRecord(workerDebugPayload?.diagnostic);
  const gpuRender = readPhase4DiagnosticRecord(diagnostic?.gpuRender);
  const gpuSupportSurface = readPhase4DiagnosticRecord(gpuRender?.supportSurface);
  const gpuToneMap = readPhase4DiagnosticRecord(gpuRender?.toneMap);
  const gpuDrr = readPhase4DiagnosticRecord(gpuRender?.drr);
  const virtualPanoPhase12 = readPhase4DiagnosticRecord(diagnostic?.virtualPanoPhase12);
  const cpuSupportSurface = readPhase4DiagnosticRecord(virtualPanoPhase12?.supportSurface);
  const virtualPanoRender = readPhase4DiagnosticRecord(diagnostic?.virtualPanoRender);
  const supportSurface = gpuSupportSurface ?? cpuSupportSurface;

  return {
    sampledCount: summary?.sampledCount ?? null,
    lowerBandBrightFraction: summary?.lowerBandBrightFraction ?? null,
    lowerBandP50: summary?.lowerBandP50 ?? null,
    toothBandMean: summary?.toothBandMean ?? null,
    toothBandP10: summary?.toothBandP10 ?? null,
    toothBandP90: summary?.toothBandP90 ?? null,
    toothBandContrastRange:
      summary ? summary.toothBandP90 - summary.toothBandP10 : readPhase4DiagnosticNumber(virtualPanoRender?.toothBandContrastRange),
    detailBandHorizontalEdgeMean: summary?.detailBandHorizontalEdgeMean ?? null,
    detailBandVerticalEdgeMean: summary?.detailBandVerticalEdgeMean ?? null,
    supportDepthClampFraction:
      readPhase4DiagnosticNumber(supportSurface?.clampFraction) ??
      readPhase4DiagnosticNumber(supportSurface?.supportDepthClampFraction) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportDepthClampFraction),
    supportDepthStdMm: readPhase4DiagnosticNumber(supportSurface?.depthStdMm),
    pathJumpP95Mm: readPhase4DiagnosticNumber(supportSurface?.pathJumpP95Mm),
    supportConfidenceP10: readPhase4DiagnosticNumber(supportSurface?.confidenceP10),
    supportConfidenceP50: readPhase4DiagnosticNumber(supportSurface?.confidenceP50),
    supportConfidenceP90: readPhase4DiagnosticNumber(supportSurface?.confidenceP90),
    supportPathConfidenceP10:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP10) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP10),
    supportPathConfidenceP50:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP50) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP50),
    supportPathConfidenceP90:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP90) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP90),
    supportUnstableColumnFraction:
      readPhase4DiagnosticNumber(supportSurface?.unstableColumnFraction) ??
      readPhase4DiagnosticNumber(supportSurface?.unreliableColumnFraction),
    supportLongestUnstableRunColumns:
      readPhase4DiagnosticNumber(supportSurface?.longestUnstableRunColumns) ??
      readPhase4DiagnosticNumber(supportSurface?.unreliableRunLongest),
    supportAmbiguousColumnFraction: readPhase4DiagnosticNumber(supportSurface?.ambiguousColumnFraction),
    supportForcedDriftFraction: readPhase4DiagnosticNumber(supportSurface?.forcedDriftFraction),
    supportBestDepthDriftP95Mm: readPhase4DiagnosticNumber(supportSurface?.bestDepthDriftP95Mm),
    supportScoreGapP50: readPhase4DiagnosticNumber(supportSurface?.scoreGapP50),
    troughSigmaP50Mm: readPhase4DiagnosticNumber(gpuDrr?.troughSigmaP50Mm),
    approxTroughHalfWidthP50Mm: readPhase4DiagnosticNumber(gpuDrr?.approxTroughHalfWidthP50Mm),
    effectiveTroughHalfWidthP50Mm: readPhase4DiagnosticNumber(gpuDrr?.effectiveTroughHalfWidthP50Mm),
    effectiveTroughHalfWidthP90Mm: readPhase4DiagnosticNumber(gpuDrr?.effectiveTroughHalfWidthP90Mm),
    participatingSamplesP50: readPhase4DiagnosticNumber(gpuDrr?.participatingSamplesP50),
    participatingSamplesP90: readPhase4DiagnosticNumber(gpuDrr?.participatingSamplesP90),
    backgroundTroughNarrowGateP50: readPhase4DiagnosticNumber(gpuDrr?.backgroundTroughNarrowGateP50),
    backgroundTroughNarrowGateP90: readPhase4DiagnosticNumber(gpuDrr?.backgroundTroughNarrowGateP90),
    blackClipFraction:
      readPhase4DiagnosticNumber(gpuToneMap?.blackClipFraction) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.blackClipFraction),
    fractionBelowMinus950: summary?.fractionBelowMinus950 ?? null,
    fractionAbove3000: summary?.fractionAbove3000 ?? null,
    renderSupportMode: toNonEmptyString(virtualPanoRender?.renderSupportMode),
    contextHuP50: readPhase4DiagnosticNumber(virtualPanoRender?.contextHuP50),
    contextBlendMean: readPhase4DiagnosticNumber(virtualPanoRender?.contextBlendMean),
    contextWeightFractionMean: readPhase4DiagnosticNumber(virtualPanoRender?.contextWeightFractionMean),
    columnSupportReliabilityP50: readPhase4DiagnosticNumber(
      virtualPanoRender?.columnSupportReliabilityP50
    ),
    upperDetailHuP50: readPhase4DiagnosticNumber(virtualPanoRender?.upperDetailHuP50),
    lowerDetailHuP50: readPhase4DiagnosticNumber(virtualPanoRender?.lowerDetailHuP50),
    detailSampleFractionMean: readPhase4DiagnosticNumber(virtualPanoRender?.detailSampleFractionMean),
    shadowLiftMean: readPhase4DiagnosticNumber(virtualPanoRender?.shadowLiftMean),
  };
}

function buildPhase4QualityGateCandidate(params: {
  attemptLabel?: string | null;
  displayedPath: CprPanoDisplayPath;
  sourceVolumeId?: string | null;
  summary: FloatBufferDebugSummary | null;
  qualityBase?: number | null;
  qualityScore?: number | null;
  hardRejectReason?: string | null;
  workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined;
  backendOverride?: CprPanoDisplayBackend | null;
  requestedBackendOverride?: CprPanoReconBackend | null;
  fallbackReasonOverride?: string | null;
  pipelineModeOverride?: string | null;
  reconstructionModeOverride?: string | null;
}): Phase4QualityGateCandidate {
  const diagnostic = readPhase4DiagnosticRecord(params.workerDebugPayload?.diagnostic);
  const routeDiagnostic = resolveWorkerDisplayRouteDiagnostic(params.workerDebugPayload);
  const outputSelection = readPhase4DiagnosticRecord(diagnostic?.outputSelection);
  const virtualPanoRender = readPhase4DiagnosticRecord(diagnostic?.virtualPanoRender);
  const requestedReconstructionMode =
    toNonEmptyString(diagnostic?.requestedReconstructionMode) ??
    toNonEmptyString(outputSelection?.requestedReconstructionMode) ??
    null;
  const reconstructionMode =
    params.reconstructionModeOverride ??
    toNonEmptyString(diagnostic?.reconstructionMode) ??
    toNonEmptyString(outputSelection?.selectedReconstructionMode) ??
    routeDiagnostic.pipelineMode;
  const backend = params.backendOverride ?? routeDiagnostic.backend;
  const requestedBackend = params.requestedBackendOverride ?? routeDiagnostic.requestedBackend;
  const fallbackReason = params.fallbackReasonOverride ?? routeDiagnostic.fallbackReason;
  const pipelineMode = params.pipelineModeOverride ?? routeDiagnostic.pipelineMode;
  const outputSelectionRejectReasons = readPhase4DiagnosticStringArray(outputSelection?.rejectReasons);
  const virtualPanoRenderRejectReasons = readPhase4DiagnosticStringArray(
    virtualPanoRender?.rejectReasons
  );
  const workerRejectReasons =
    outputSelectionRejectReasons.length > 0
      ? outputSelectionRejectReasons
      : virtualPanoRenderRejectReasons;
  const workerUsedAsOutput =
    readPhase4DiagnosticBoolean(outputSelection?.virtualPanoAccepted) ??
    readPhase4DiagnosticBoolean(virtualPanoRender?.usedAsOutput);
  const workerAcceptedByLowerBandTolerance =
    readPhase4DiagnosticBoolean(outputSelection?.acceptedByLowerBandTolerance) === true ||
    readPhase4DiagnosticBoolean(virtualPanoRender?.acceptedByLowerBandTolerance) === true;
  const metrics = extractPhase4QualityGateMetrics(params.summary, params.workerDebugPayload);
  const candidateSource: Phase4QualityGateCandidateSource =
    backend === 'gpu'
      ? 'worker-gpu-support-surface'
      : reconstructionMode === 'virtualPano' &&
          metrics.renderSupportMode === 'archGuidedDualLayer'
        ? 'worker-cpu-arch-guided-synthetic'
        : reconstructionMode === 'virtualPano'
        ? 'worker-cpu-virtual-pano'
        : 'worker-legacy';
  const isDualArchProjection = metrics.renderSupportMode === 'dualArchProjection';

  const rejectReasons: string[] = [];
  if (!params.summary || params.summary.sampledCount < 100) {
    rejectReasons.push('summary-unavailable');
  }
  if (candidateSource === 'worker-gpu-support-surface' && routeDiagnostic.phase2GatePassed === false) {
    rejectReasons.push('gpu-phase2-gate-failed');
  }
  if (params.hardRejectReason) {
    rejectReasons.push(`hard-reject:${params.hardRejectReason}`);
  }
  if (
    candidateSource === 'worker-legacy' &&
    requestedReconstructionMode !== 'legacy' &&
    !CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK
  ) {
    rejectReasons.push('legacy-fallback-not-allowed');
  }
  if (
    metrics.lowerBandBrightFraction !== null &&
    metrics.lowerBandBrightFraction > (isDualArchProjection ? 0.94 : 0.25)
  ) {
    rejectReasons.push('lower-band-bright-fraction-too-high');
  }
  if (metrics.lowerBandP50 !== null && metrics.lowerBandP50 > (isDualArchProjection ? 420 : -100)) {
    rejectReasons.push('lower-band-p50-too-high');
  }
  if (
    metrics.toothBandContrastRange !== null &&
    metrics.toothBandContrastRange < (isDualArchProjection ? 120 : 150)
  ) {
    rejectReasons.push('tooth-band-contrast-too-low');
  }
  if (
    metrics.supportDepthClampFraction !== null &&
    metrics.supportDepthClampFraction > 0.15
  ) {
    rejectReasons.push('support-depth-clamp-fraction-too-high');
  }
  if (metrics.pathJumpP95Mm !== null && metrics.pathJumpP95Mm > 1.2) {
    rejectReasons.push('path-jump-p95-too-high');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    ((metrics.supportConfidenceP50 !== null && metrics.supportConfidenceP50 < 0.09) ||
      (metrics.supportPathConfidenceP50 !== null && metrics.supportPathConfidenceP50 < 0.15))
  ) {
    rejectReasons.push('support-confidence-too-low');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    (((params.summary?.backgroundOutlierFraction05 ?? 0) > 0.24) ||
      ((params.summary?.backgroundOutlierFraction05 ?? 0) > 0.20 &&
        (params.summary?.backgroundOutlierFraction10 ?? 0) > 0.09))
  ) {
    rejectReasons.push('background-outlier-fraction-too-high');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    (((metrics.supportUnstableColumnFraction ?? 0) > 0.24) ||
      ((metrics.supportLongestUnstableRunColumns ?? 0) > 30) ||
      (((metrics.supportAmbiguousColumnFraction ?? 0) > 0.2) &&
        ((metrics.supportScoreGapP50 ?? 1) < 0.11)))
  ) {
    rejectReasons.push('support-columns-unstable');
  }
  if (
    candidateSource === 'worker-cpu-virtual-pano' &&
    (((metrics.supportUnstableColumnFraction ?? 0) > (isDualArchProjection ? 0.36 : 0.24)) ||
      ((metrics.supportLongestUnstableRunColumns ?? 0) > (isDualArchProjection ? 28 : 14)) ||
      ((metrics.supportDepthStdMm ?? 0) > (isDualArchProjection ? 1.8 : 1.3)))
  ) {
    rejectReasons.push('virtual-support-columns-unstable');
  }
  if (
    candidateSource === 'worker-cpu-virtual-pano' &&
    (((metrics.supportAmbiguousColumnFraction ?? 0) > (isDualArchProjection ? 0.62 : 0.4) &&
      (metrics.supportScoreGapP50 ?? 1) < (isDualArchProjection ? 0.05 : 0.08)) ||
      ((metrics.supportForcedDriftFraction ?? 0) > (isDualArchProjection ? 0.2 : 0.12)) ||
      ((metrics.supportBestDepthDriftP95Mm ?? 0) > (isDualArchProjection ? 2.6 : 1.8)))
  ) {
    rejectReasons.push('virtual-support-ambiguity-too-high');
  }
  if (
    candidateSource === 'worker-cpu-arch-guided-synthetic' &&
    (((metrics.contextWeightFractionMean ?? 0) < 0.045) ||
      ((metrics.detailSampleFractionMean ?? 0) < 0.028) ||
      ((metrics.columnSupportReliabilityP50 ?? 0) < 0.2))
  ) {
    rejectReasons.push('arch-guided-context-coverage-too-low');
  }
  if (
    candidateSource === 'worker-cpu-arch-guided-synthetic' &&
    (((metrics.toothBandContrastRange ?? 0) < 180) ||
      ((metrics.contextHuP50 ?? -1000) < -720) ||
      ((metrics.upperDetailHuP50 ?? -1000) < 40 && (metrics.lowerDetailHuP50 ?? -1000) < 40))
  ) {
    rejectReasons.push('arch-guided-anatomy-contrast-too-low');
  }
  if (!workerAcceptedByLowerBandTolerance && workerUsedAsOutput === false) {
    rejectReasons.push(...workerRejectReasons);
  }

  const uniqueRejectReasons = Array.from(new Set(rejectReasons));
  const lowerBandOnlyRejects =
    uniqueRejectReasons.length > 0 &&
    uniqueRejectReasons.every(reason => isPhase4LowerBandRejectReason(reason));
  const borderlineAcceptedReason =
    workerAcceptedByLowerBandTolerance
      ? 'lower-band-tolerated-in-worker'
      : lowerBandOnlyRejects &&
          (params.qualityScore ?? Number.NEGATIVE_INFINITY) >= 12 &&
          (metrics.toothBandContrastRange ?? 0) >= 150
        ? 'lower-band-only-borderline'
        : null;
  const pass =
    uniqueRejectReasons.length === 0 ||
    (borderlineAcceptedReason !== null && lowerBandOnlyRejects);
  const supportSurfaceRiskSummary = buildPhase4SupportSurfaceRiskSummary({
    candidateSource,
    backend,
    requestedBackend,
    pipelineMode,
    reconstructionMode,
    qualityScore:
      typeof params.qualityScore === 'number' && Number.isFinite(params.qualityScore)
        ? params.qualityScore
        : null,
    qualityBase:
      typeof params.qualityBase === 'number' && Number.isFinite(params.qualityBase)
        ? params.qualityBase
        : null,
    pass,
    hardRejectReason: params.hardRejectReason ?? null,
    metrics,
    summary: params.summary,
    borderlineAcceptedReason,
  });

  return {
    candidateSource,
    displayedPath: params.displayedPath,
    backend,
    requestedBackend,
    attemptLabel: params.attemptLabel ?? null,
    requestedReconstructionMode,
    reconstructionMode,
    pipelineMode,
    sourceVolumeId: params.sourceVolumeId ?? null,
    fallbackReason,
    phase2GatePassed: routeDiagnostic.phase2GatePassed,
    qualityBase:
      typeof params.qualityBase === 'number' && Number.isFinite(params.qualityBase)
        ? params.qualityBase
        : null,
    qualityScore:
      typeof params.qualityScore === 'number' && Number.isFinite(params.qualityScore)
        ? params.qualityScore
        : null,
    hardRejectReason: params.hardRejectReason ?? null,
    workerRejectReasons,
    workerUsedAsOutput,
    borderlineAcceptedReason,
    pass,
    rejectReasons: uniqueRejectReasons,
    metrics,
    supportSurfaceRiskSummary,
  };
}

function buildPhase4SupportSurfaceRiskSummary(params: {
  candidateSource: Phase4QualityGateCandidateSource;
  backend: CprPanoDisplayBackend;
  requestedBackend: CprPanoReconBackend | null;
  pipelineMode: string;
  reconstructionMode: string;
  qualityScore: number | null;
  qualityBase: number | null;
  pass: boolean;
  hardRejectReason: string | null;
  metrics: Phase4QualityGateMetrics;
  summary: FloatBufferDebugSummary | null;
  borderlineAcceptedReason: string | null;
}): Phase4SupportSurfaceRiskSummary {
  const riskFlags: string[] = [];
  let riskScore = 0;
  const usesLegacySupportPathRiskModel =
    params.candidateSource !== 'worker-cpu-arch-guided-synthetic';
  const isDualArchProjection = isDualArchProjectionRenderMode(params.metrics.renderSupportMode);
  const backgroundOutlierFraction05 = params.summary?.backgroundOutlierFraction05 ?? null;
  const backgroundOutlierFraction10 = params.summary?.backgroundOutlierFraction10 ?? null;
  const detailBandHorizontalEdgeMean = params.summary?.detailBandHorizontalEdgeMean ?? null;
  const detailBandVerticalEdgeMean = params.summary?.detailBandVerticalEdgeMean ?? null;
  const addRiskFlag = (flag: string, condition: boolean, weight = 1): void => {
    if (!condition) {
      return;
    }
    riskFlags.push(flag);
    riskScore += weight;
  };

  addRiskFlag(
    'support-confidence-low',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportConfidenceP50 !== null && params.metrics.supportConfidenceP50 < 0.12) ||
        (params.metrics.supportConfidenceP10 !== null && params.metrics.supportConfidenceP10 < 0.04)),
    3
  );
  addRiskFlag(
    'support-depth-unstable',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportDepthStdMm !== null && params.metrics.supportDepthStdMm > 0.65) ||
        (params.metrics.pathJumpP95Mm !== null && params.metrics.pathJumpP95Mm > 1.2)),
    3
  );
  addRiskFlag(
    'support-path-confidence-low',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportPathConfidenceP50 !== null &&
        params.metrics.supportPathConfidenceP50 < 0.18) ||
        (params.metrics.supportPathConfidenceP10 !== null &&
          params.metrics.supportPathConfidenceP10 < 0.06)),
    2
  );
  addRiskFlag(
    'support-columns-unstable',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportUnstableColumnFraction !== null &&
        params.metrics.supportUnstableColumnFraction > 0.18) ||
        (params.metrics.supportLongestUnstableRunColumns !== null &&
          params.metrics.supportLongestUnstableRunColumns > 28)),
    3
  );
  addRiskFlag(
    'support-ambiguity-high',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportAmbiguousColumnFraction !== null &&
        params.metrics.supportAmbiguousColumnFraction > 0.22) ||
        (params.metrics.supportScoreGapP50 !== null && params.metrics.supportScoreGapP50 < 0.12)),
    2
  );
  addRiskFlag(
    'support-forced-drift',
    usesLegacySupportPathRiskModel &&
      ((params.metrics.supportForcedDriftFraction !== null &&
        params.metrics.supportForcedDriftFraction > 0.08) ||
        (params.metrics.supportBestDepthDriftP95Mm !== null &&
          params.metrics.supportBestDepthDriftP95Mm > 0.9)),
    2
  );
  addRiskFlag(
    'support-depth-clamped',
    usesLegacySupportPathRiskModel &&
      params.metrics.supportDepthClampFraction !== null &&
      params.metrics.supportDepthClampFraction > 0.15,
    2
  );
  addRiskFlag(
    'background-clipped',
    params.metrics.blackClipFraction !== null && params.metrics.blackClipFraction > 0.42,
    2
  );
  addRiskFlag(
    'background-leakage',
    (backgroundOutlierFraction05 !== null && backgroundOutlierFraction05 > 0.18) ||
      (backgroundOutlierFraction10 !== null && backgroundOutlierFraction10 > 0.08),
    2
  );
  addRiskFlag(
    'lower-band-filled',
    (params.metrics.lowerBandBrightFraction !== null &&
      params.metrics.lowerBandBrightFraction > (isDualArchProjection ? 0.9 : 0.25)) ||
      (params.metrics.lowerBandP50 !== null &&
        params.metrics.lowerBandP50 > (isDualArchProjection ? 360 : -100)),
    2
  );
  addRiskFlag(
    'tooth-band-contrast-low',
    params.metrics.toothBandContrastRange !== null &&
      params.metrics.toothBandContrastRange > 0 &&
      params.metrics.toothBandContrastRange < (isDualArchProjection ? 120 : 150),
    2
  );
  addRiskFlag(
    'tooth-band-saturated',
    params.metrics.toothBandMean !== null && params.metrics.toothBandMean > 760,
    1
  );
  addRiskFlag(
    'air-suppression-weak',
    params.metrics.fractionBelowMinus950 !== null &&
      params.metrics.fractionBelowMinus950 < (isDualArchProjection ? 0.005 : 0.015),
    1
  );
  addRiskFlag(
    'upper-lower-balance-poor',
    detailBandVerticalEdgeMean !== null &&
      detailBandVerticalEdgeMean > 0 &&
      detailBandHorizontalEdgeMean !== null &&
      detailBandHorizontalEdgeMean / Math.max(1, detailBandVerticalEdgeMean) > 2.8,
    1
  );
  addRiskFlag('candidate-hard-reject', !!params.hardRejectReason, 4);
  addRiskFlag('quality-gate-failed', !params.pass, 2);
  addRiskFlag(
    'borderline-accepted',
    params.borderlineAcceptedReason !== null && !params.pass,
    1
  );

  const riskLevel =
    riskScore >= 9 ? 'high' : riskScore >= 6 ? 'elevated' : riskScore >= 3 ? 'moderate' : 'low';
  const baselineFingerprint = [
    `backend=${params.backend}`,
    `requestedBackend=${params.requestedBackend ?? 'null'}`,
    `reconstructionMode=${params.reconstructionMode}`,
    `pipelineMode=${params.pipelineMode}`,
    `candidateSource=${params.candidateSource}`,
    `qualityBase=${formatPhase4FingerprintValue(params.qualityBase, 2)}`,
    `qualityScore=${formatPhase4FingerprintValue(params.qualityScore, 2)}`,
    `pass=${params.pass ? 1 : 0}`,
    `hardReject=${params.hardRejectReason ?? 'none'}`,
    `borderline=${params.borderlineAcceptedReason ?? 'none'}`,
    `supportP10=${formatPhase4FingerprintValue(params.metrics.supportConfidenceP10)}`,
    `supportP50=${formatPhase4FingerprintValue(params.metrics.supportConfidenceP50)}`,
    `supportP90=${formatPhase4FingerprintValue(params.metrics.supportConfidenceP90)}`,
    `supportClamp=${formatPhase4FingerprintValue(params.metrics.supportDepthClampFraction)}`,
    `supportStd=${formatPhase4FingerprintValue(params.metrics.supportDepthStdMm)}`,
    `pathJumpP95=${formatPhase4FingerprintValue(params.metrics.pathJumpP95Mm)}`,
    `pathConfP50=${formatPhase4FingerprintValue(params.metrics.supportPathConfidenceP50, 4)}`,
    `unstableCols=${formatPhase4FingerprintValue(params.metrics.supportUnstableColumnFraction, 4)}`,
    `unstableRun=${formatPhase4FingerprintValue(params.metrics.supportLongestUnstableRunColumns, 0)}`,
    `ambiguousCols=${formatPhase4FingerprintValue(params.metrics.supportAmbiguousColumnFraction, 4)}`,
    `forcedDrift=${formatPhase4FingerprintValue(params.metrics.supportForcedDriftFraction, 4)}`,
    `driftP95=${formatPhase4FingerprintValue(params.metrics.supportBestDepthDriftP95Mm)}`,
    `scoreGapP50=${formatPhase4FingerprintValue(params.metrics.supportScoreGapP50, 4)}`,
    `blackClip=${formatPhase4FingerprintValue(params.metrics.blackClipFraction, 4)}`,
    `bgOutlier05=${formatPhase4FingerprintValue(params.summary?.backgroundOutlierFraction05, 4)}`,
    `bgOutlier10=${formatPhase4FingerprintValue(params.summary?.backgroundOutlierFraction10, 4)}`,
    `lowerBandBright=${formatPhase4FingerprintValue(params.metrics.lowerBandBrightFraction, 4)}`,
    `lowerBandP50=${formatPhase4FingerprintValue(params.metrics.lowerBandP50)}`,
    `toothContrast=${formatPhase4FingerprintValue(params.metrics.toothBandContrastRange)}`,
    `flags=${riskFlags.length ? riskFlags.join(',') : 'none'}`,
  ].join('|');

  return {
    riskLevel,
    riskScore,
    riskFlags,
    baselineFingerprint,
    stability: {
      supportConfidenceP10: params.metrics.supportConfidenceP10,
      supportConfidenceP50: params.metrics.supportConfidenceP50,
      supportConfidenceP90: params.metrics.supportConfidenceP90,
      supportPathConfidenceP10: params.metrics.supportPathConfidenceP10,
      supportPathConfidenceP50: params.metrics.supportPathConfidenceP50,
      supportPathConfidenceP90: params.metrics.supportPathConfidenceP90,
      supportDepthClampFraction: params.metrics.supportDepthClampFraction,
      supportDepthStdMm: params.metrics.supportDepthStdMm,
      pathJumpP95Mm: params.metrics.pathJumpP95Mm,
      supportUnstableColumnFraction: params.metrics.supportUnstableColumnFraction,
      supportLongestUnstableRunColumns: params.metrics.supportLongestUnstableRunColumns,
      supportAmbiguousColumnFraction: params.metrics.supportAmbiguousColumnFraction,
      supportForcedDriftFraction: params.metrics.supportForcedDriftFraction,
      supportBestDepthDriftP95Mm: params.metrics.supportBestDepthDriftP95Mm,
      supportScoreGapP50: params.metrics.supportScoreGapP50,
    },
    background: {
      blackClipFraction: params.metrics.blackClipFraction,
      backgroundToneP95: params.summary?.backgroundToneP95 ?? null,
      backgroundToneP99: params.summary?.backgroundToneP99 ?? null,
      backgroundToneMax: params.summary?.backgroundToneMax ?? null,
      backgroundOutlierFraction05: params.summary?.backgroundOutlierFraction05 ?? null,
      backgroundOutlierFraction10: params.summary?.backgroundOutlierFraction10 ?? null,
    },
    anatomy: {
      lowerBandBrightFraction: params.metrics.lowerBandBrightFraction,
      lowerBandP50: params.metrics.lowerBandP50,
      toothBandMean: params.metrics.toothBandMean,
      toothBandP10: params.metrics.toothBandP10,
      toothBandP90: params.metrics.toothBandP90,
      toothBandContrastRange: params.metrics.toothBandContrastRange,
      fractionBelowMinus950: params.metrics.fractionBelowMinus950,
      fractionAbove3000: params.metrics.fractionAbove3000,
    },
  };
}

function summarizePhase4QualityGateCandidate(candidate: Phase4QualityGateCandidate | null) {
  if (!candidate) {
    return null;
  }

  return {
    candidateSource: candidate.candidateSource,
    attemptLabel: candidate.attemptLabel,
    displayedPath: candidate.displayedPath,
    backend: candidate.backend,
    requestedBackend: candidate.requestedBackend,
    requestedReconstructionMode: candidate.requestedReconstructionMode,
    reconstructionMode: candidate.reconstructionMode,
    pipelineMode: candidate.pipelineMode,
    sourceVolumeId: candidate.sourceVolumeId,
    fallbackReason: candidate.fallbackReason,
    phase2GatePassed: candidate.phase2GatePassed,
    qualityBase: candidate.qualityBase,
    qualityScore: candidate.qualityScore,
    hardRejectReason: candidate.hardRejectReason,
    workerRejectReasons: candidate.workerRejectReasons,
    workerUsedAsOutput: candidate.workerUsedAsOutput,
    borderlineAcceptedReason: candidate.borderlineAcceptedReason,
    pass: candidate.pass,
    rejectReasons: candidate.rejectReasons,
    metrics: candidate.metrics,
    supportSurfaceRiskSummary: candidate.supportSurfaceRiskSummary,
  };
}

function assessPhase4DegradedPreviewCandidate(
  candidate: Phase4QualityGateCandidate | null
): Phase4DegradedPreviewAssessment {
  if (!candidate) {
    return {
      catastrophic: true,
      catastrophicReasons: ['candidate-missing'],
    };
  }

  const catastrophicReasons: string[] = [];
  const nonLowerBandWorkerRejectReasons = candidate.workerRejectReasons.filter(
    reason => !isPhase4LowerBandRejectReason(reason)
  );
  const isDualArchProjection = isDualArchProjectionRenderMode(candidate.metrics.renderSupportMode);

  if ((candidate.metrics.sampledCount ?? 0) < 100) {
    catastrophicReasons.push('summary-unavailable');
  }
  if (candidate.hardRejectReason) {
    catastrophicReasons.push(`hard-reject:${candidate.hardRejectReason}`);
  }
  if (candidate.workerUsedAsOutput === false && nonLowerBandWorkerRejectReasons.length > 0) {
    catastrophicReasons.push(
      ...nonLowerBandWorkerRejectReasons.map(reason => `worker-reject:${reason}`)
    );
  }
  if ((candidate.qualityScore ?? Number.NEGATIVE_INFINITY) < -20) {
    catastrophicReasons.push('quality-score-extremely-low');
  }
  if ((candidate.metrics.blackClipFraction ?? 0) > 0.72) {
    catastrophicReasons.push('black-clip-extreme');
  }
  if ((candidate.metrics.fractionBelowMinus950 ?? 0) > 0.75) {
    catastrophicReasons.push('lower-clipping-extreme');
  }
  if ((candidate.metrics.fractionAbove3000 ?? 0) > 0.18) {
    catastrophicReasons.push('upper-clipping-extreme');
  }
  if ((candidate.metrics.lowerBandBrightFraction ?? 0) > (isDualArchProjection ? 0.94 : 0.42)) {
    catastrophicReasons.push('lower-band-filled-extreme');
  }
  if ((candidate.metrics.toothBandMean ?? Number.NEGATIVE_INFINITY) > 1100) {
    catastrophicReasons.push('tooth-band-saturation-extreme');
  }
  if (
    candidate.metrics.toothBandContrastRange !== null &&
    candidate.metrics.toothBandContrastRange > 0 &&
    candidate.metrics.toothBandContrastRange < (isDualArchProjection ? 90 : 110)
  ) {
    catastrophicReasons.push('tooth-band-contrast-extremely-low');
  }
  if (
    (candidate.metrics.supportDepthStdMm ?? 0) > 2.4 &&
    (candidate.metrics.supportUnstableColumnFraction ?? 0) > 0.35 &&
    (candidate.metrics.supportAmbiguousColumnFraction ?? 0) > 0.65
  ) {
    catastrophicReasons.push('support-surface-catastrophic-instability');
  }

  return {
    catastrophic: catastrophicReasons.length > 0,
    catastrophicReasons: Array.from(new Set(catastrophicReasons)),
  };
}

function formatPhase4RejectReasonsForMessage(reasons: string[], maxReasons = 4): string {
  if (!reasons.length) {
    return 'none';
  }

  const visibleReasons = reasons.slice(0, maxReasons);
  const suffix =
    reasons.length > visibleReasons.length ? ` (+${reasons.length - visibleReasons.length} more)` : '';

  return `${visibleReasons.join(', ')}${suffix}`;
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
  buffer: Float32Array | Uint16Array,
  width?: number,
  height?: number,
  debugMaps?: PanoImagePayload['debugMaps'],
  analysisCenterRow?: number
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
  const toneResponseMap = debugMaps?.toneResponseMap;
  const supportConfidenceMap = debugMaps?.supportConfidenceMap;
  const lowerPenaltyMap = debugMaps?.lowerPenaltyMap;
  const participatingSampleCountMap = debugMaps?.participatingSampleCountMap;
  const backgroundToneSamples: number[] = [];
  let backgroundToneOutlierCount05 = 0;
  let backgroundToneOutlierCount10 = 0;
  let backgroundToneMax = 0;

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
    const panoCenterRow =
      Number.isFinite(analysisCenterRow) && Number(analysisCenterRow) >= 0
        ? Math.max(0, Math.min(safeHeight - 1, Number(analysisCenterRow)))
        : (safeHeight - 1) / 2;
    const rowFromNormalizedOffset = (yNorm: number): number => {
      const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
      return Math.max(0, Math.min(safeHeight - 1, row));
    };
    const toothBandStartRow = Math.min(
      rowFromNormalizedOffset(-0.35),
      rowFromNormalizedOffset(0.55)
    );
    const toothBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const lowerBandStartRow = Math.min(
      rowFromNormalizedOffset(0.65),
      rowFromNormalizedOffset(1.15)
    );
    const lowerBandEndRow = Math.max(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const detailBandStartRow = Math.max(0, Math.floor(safeHeight * 0.12));
    const detailBandEndRow = Math.min(safeHeight - 1, Math.ceil(safeHeight * 0.72));
    const canSampleBackgroundTone = !!toneResponseMap && !!supportConfidenceMap;
    const backgroundSupportConfidenceMax = 0.03;
    const backgroundLowerPenaltyMax = 0.05;
    const backgroundParticipatingSampleCountMin = 0.5;
    const backgroundHuMax = -300;
    const backgroundToneOutlierThreshold05 = 0.05;
    const backgroundToneOutlierThreshold10 = 0.1;

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

        if (canSampleBackgroundTone && value <= backgroundHuMax) {
          const toneResponse = readOptionalProbeMapValue(toneResponseMap, index);
          const supportConfidence = readOptionalProbeMapValue(supportConfidenceMap, index);
          if (toneResponse !== undefined && supportConfidence !== undefined) {
            const lowerPenalty = readOptionalProbeMapValue(lowerPenaltyMap, index) ?? 0;
            const participatingSampleCount =
              readOptionalProbeMapValue(participatingSampleCountMap, index) ?? 1;
            if (
              supportConfidence <= backgroundSupportConfidenceMax &&
              lowerPenalty <= backgroundLowerPenaltyMax &&
              participatingSampleCount >= backgroundParticipatingSampleCountMin
            ) {
              const clampedTone = Math.max(0, Math.min(1, toneResponse));
              backgroundToneSamples.push(clampedTone);
              if (clampedTone > backgroundToneOutlierThreshold05) {
                backgroundToneOutlierCount05++;
              }
              if (clampedTone > backgroundToneOutlierThreshold10) {
                backgroundToneOutlierCount10++;
              }
              if (clampedTone > backgroundToneMax) {
                backgroundToneMax = clampedTone;
              }
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
  if (backgroundToneSamples.length) {
    backgroundToneSamples.sort((a, b) => a - b);
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
      detailBandHorizontalEdgeCount > 0
        ? detailBandHorizontalEdgeAccum / detailBandHorizontalEdgeCount
        : 0,
    detailBandVerticalEdgeMean:
      detailBandVerticalEdgeCount > 0
        ? detailBandVerticalEdgeAccum / detailBandVerticalEdgeCount
        : 0,
    backgroundToneSampleCount: backgroundToneSamples.length,
    backgroundToneP95: backgroundToneSamples.length
      ? percentileFromSorted(backgroundToneSamples, 0.95)
      : 0,
    backgroundToneP99: backgroundToneSamples.length
      ? percentileFromSorted(backgroundToneSamples, 0.99)
      : 0,
    backgroundToneMax,
    backgroundOutlierFraction05: backgroundToneSamples.length
      ? backgroundToneOutlierCount05 / backgroundToneSamples.length
      : 0,
    backgroundOutlierFraction10: backgroundToneSamples.length
      ? backgroundToneOutlierCount10 / backgroundToneSamples.length
      : 0,
  };
}

function reconstructPanoFloatBuffer(
  pixelData: Float32Array | Uint16Array,
  slope: number,
  intercept: number
): Float32Array {
  const safeSlope = Number.isFinite(slope) && Math.abs(Number(slope)) > 1e-8 ? Number(slope) : 1;
  const safeIntercept = Number.isFinite(intercept) ? Number(intercept) : 0;

  if (
    pixelData instanceof Float32Array &&
    Math.abs(safeSlope - 1) <= 1e-6 &&
    Math.abs(safeIntercept) <= 1e-6
  ) {
    return pixelData;
  }

  const reconstructed = new Float32Array(pixelData.length);
  for (let i = 0; i < pixelData.length; i++) {
    reconstructed[i] = Number(pixelData[i]) * safeSlope + safeIntercept;
  }

  return reconstructed;
}

function computeAdaptivePanoVoi(
  summary: FloatBufferDebugSummary | null,
  minValue: number,
  maxValue: number,
  renderSupportMode?: string | null
): PanoVoiSettings {
  const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);
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

  if (
    !Number.isFinite(adaptiveLower) ||
    !Number.isFinite(adaptiveUpper) ||
    adaptiveUpper <= adaptiveLower
  ) {
    adaptiveLower = lower;
    adaptiveUpper = upper;
  }

  const robustSpan = Math.max(1, adaptiveUpper - adaptiveLower);
  const padding = Math.max(20, robustSpan * 0.03);
  adaptiveLower -= padding;
  adaptiveUpper += padding;

  if (
    !Number.isFinite(adaptiveLower) ||
    !Number.isFinite(adaptiveUpper) ||
    adaptiveUpper <= adaptiveLower
  ) {
    adaptiveLower = fallbackLower;
    adaptiveUpper = fallbackUpper;
  }

  const windowWidth = Math.max(1, adaptiveUpper - adaptiveLower);
  const windowCenter = adaptiveLower + windowWidth / 2;
  const looksLikeHU = isHuLikeRange(lower, upper);

  if (isDualArchProjection) {
    const dataRangeWidth =
      summary && Number.isFinite(summary.p99) && Number.isFinite(summary.p01)
        ? Math.max(1, Number(summary.p99) - Number(summary.p01))
        : windowWidth;
    const dualArchP99 =
      summary && Number.isFinite(summary.p99) ? Number(summary.p99) : adaptiveUpper;
    const minDualArchWindowWidth =
      dualArchP99 <= CPR_DUAL_ARCH_PROJECTION_SOFT_SCAN_P99_THRESHOLD_HU
        ? CPR_DUAL_ARCH_PROJECTION_SOFT_MIN_WINDOW_WIDTH
        : CPR_DUAL_ARCH_PROJECTION_STANDARD_MIN_WINDOW_WIDTH;
    const maxDualArchWindowWidth = Math.max(
      minDualArchWindowWidth,
      Math.min(dataRangeWidth * 3.0, 2800)
    );
    const widthFromClampedLower = Math.max(
      1,
      adaptiveUpper - CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU
    );
    const clampedWindowWidth = Math.min(
      Math.max(widthFromClampedLower, minDualArchWindowWidth),
      maxDualArchWindowWidth
    );
    const clampedLower = CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU;
    const clampedUpper = clampedLower + clampedWindowWidth;

    return {
      lower: clampedLower,
      upper: clampedUpper,
      windowWidth: clampedWindowWidth,
      windowCenter: clampedLower + clampedWindowWidth / 2,
    };
  }

  const hasSplitOutliers =
    !!summary && summary.fractionBelowMinus950 > 0.4 && summary.fractionAbove3000 > 0.15;
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

    if (summary && !isDualArchProjection && summary.lowerBandBrightFraction > 0.62) {
      const targetWidth = summary.detailBandHorizontalEdgeMean > 210 ? 1300 : 1450;
      const targetCenter = Math.max(
        120,
        Math.min(
          260,
          170 + (summary.lowerBandBrightFraction - 0.62) * 260 + Math.max(0, summary.p50) * 0.15
        )
      );
      return {
        lower: targetCenter - targetWidth / 2,
        upper: targetCenter + targetWidth / 2,
        windowWidth: targetWidth,
        windowCenter: targetCenter,
      };
    }

    if (
      summary &&
      !isDualArchProjection &&
      summary.lowerBandBrightFraction <= 0.08 &&
      summary.lowerBandP50 <= -700 &&
      summary.p99 >= 540 &&
      summary.detailBandHorizontalEdgeMean >= 170
    ) {
      const targetWidth = summary.p99 >= 680 ? 1300 : 1375;
      const targetCenter = Math.max(-20, Math.min(120, summary.p50 * 0.12 + 20));
      return {
        lower: targetCenter - targetWidth / 2,
        upper: targetCenter + targetWidth / 2,
        windowWidth: targetWidth,
        windowCenter: targetCenter,
      };
    }

    const dataRangeWidth =
      summary && Number.isFinite(summary.p99) && Number.isFinite(summary.p01)
        ? Math.max(1, Number(summary.p99) - Number(summary.p01))
        : windowWidth;
    const minHuWindowWidth = Math.max(
      isDualArchProjection ? 1800 : 0,
      Math.min(dataRangeWidth * 1.15, summary && summary.detailBandHorizontalEdgeMean >= 85 ? 1600 : 1750)
    );
    const maxDentalWindowWidth = Math.min(
      dataRangeWidth * 3.0,
      summary && summary.lowerBandBrightFraction > 0.45 && !isDualArchProjection ? 2200 : isDualArchProjection ? 2800 : 2500
    );
    const widthWithMin = Math.max(windowWidth, minHuWindowWidth);
    const cappedWidth = Math.min(widthWithMin, maxDentalWindowWidth);
    const dentalCenterMin = isDualArchProjection ? -40 : -120;
    const dentalCenterMax = isDualArchProjection ? 460 : 380;
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

const CENTRIPETAL_CATMULL_ROM_ALPHA = 0.5;
const CENTRIPETAL_CATMULL_ROM_EPS = 1e-4;
const SPLINE_LUT_FINE_STEPS = 2000;

function lerpPoint3(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): [number, number, number] {
  const clampedT = Math.max(0, Math.min(1, t));
  const invT = 1 - clampedT;
  return [
    a[0] * invT + b[0] * clampedT,
    a[1] * invT + b[1] * clampedT,
    a[2] * invT + b[2] * clampedT,
  ];
}

function computeCentripetalKnot(
  prevT: number,
  a: [number, number, number],
  b: [number, number, number]
): number {
  const distance = Math.max(dist3(a, b), CENTRIPETAL_CATMULL_ROM_EPS);
  return prevT + Math.pow(distance, CENTRIPETAL_CATMULL_ROM_ALPHA);
}

function interpolateByKnot(
  a: [number, number, number],
  b: [number, number, number],
  ta: number,
  tb: number,
  t: number
): [number, number, number] {
  const knotDelta = tb - ta;
  if (!Number.isFinite(knotDelta) || Math.abs(knotDelta) < 1e-8) {
    return lerpPoint3(a, b, 0.5);
  }

  const blend = (t - ta) / knotDelta;
  return lerpPoint3(a, b, blend);
}

function centripetalCatmullRomPoint(
  Pm1: [number, number, number],
  P0: [number, number, number],
  P1: [number, number, number],
  P2: [number, number, number],
  t: number
): [number, number, number] {
  const t0 = 0;
  const t1 = computeCentripetalKnot(t0, Pm1, P0);
  const t2 = computeCentripetalKnot(t1, P0, P1);
  const t3 = computeCentripetalKnot(t2, P1, P2);
  const clampedT = Math.max(0, Math.min(1, t));
  const knotT = t1 + (t2 - t1) * clampedT;

  const A1 = interpolateByKnot(Pm1, P0, t0, t1, knotT);
  const A2 = interpolateByKnot(P0, P1, t1, t2, knotT);
  const A3 = interpolateByKnot(P1, P2, t2, t3, knotT);
  const B1 = interpolateByKnot(A1, A2, t0, t2, knotT);
  const B2 = interpolateByKnot(A2, A3, t1, t3, knotT);
  return interpolateByKnot(B1, B2, t1, t2, knotT);
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

function toFiniteVector3Tuple(
  value: ArrayLike<number> | null | undefined
): [number, number, number] | null {
  if (!value || value.length < 3) {
    return null;
  }

  const x = Number(value[0]);
  const y = Number(value[1]);
  const z = Number(value[2]);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return null;
  }

  return [x, y, z];
}

function toFiniteVector2Tuple(
  value: ArrayLike<number> | null | undefined
): [number, number] | null {
  if (!value || value.length < 2) {
    return null;
  }

  const x = Number(value[0]);
  const y = Number(value[1]);

  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  return [x, y];
}

function areAlignedDirectionVectors(
  a: [number, number, number] | null,
  b: [number, number, number] | null,
  epsilon = 1e-4
): boolean {
  if (!a || !b) {
    return false;
  }

  const normalizedA = normalize3(a);
  const normalizedB = normalize3(b);
  return dot3(normalizedA, normalizedB) >= 1 - epsilon;
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

type SplineArcLengthLut = {
  extended: [number, number, number][];
  nSegments: number;
  lutT: number[];
  lutArc: number[];
  totalArcLength: number;
};

function buildSplineArcLengthLut(rawPoints: [number, number, number][]): SplineArcLengthLut {
  const extended = addPhantomEndpoints(rawPoints);
  const nSegments = extended.length - 3;
  const lutT: number[] = [0];
  const lutArc: number[] = [0];

  if (nSegments <= 0) {
    return {
      extended,
      nSegments,
      lutT,
      lutArc,
      totalArcLength: 0,
    };
  }

  const stepsPerSegment = Math.max(8, Math.ceil(SPLINE_LUT_FINE_STEPS / nSegments));
  let prevPt = centripetalCatmullRomPoint(extended[0], extended[1], extended[2], extended[3], 0);

  for (let seg = 0; seg < nSegments; seg++) {
    const Pm1 = extended[seg];
    const P0 = extended[seg + 1];
    const P1 = extended[seg + 2];
    const P2 = extended[seg + 3];

    for (let step = 1; step <= stepsPerSegment; step++) {
      const localT = step / stepsPerSegment;
      const globalT = seg + localT;
      const pt = centripetalCatmullRomPoint(Pm1, P0, P1, P2, localT);
      const arcLen = lutArc[lutArc.length - 1] + dist3(prevPt, pt);
      lutT.push(globalT);
      lutArc.push(arcLen);
      prevPt = pt;
    }
  }

  return {
    extended,
    nSegments,
    lutT,
    lutArc,
    totalArcLength: lutArc[lutArc.length - 1],
  };
}

function computeTangentsFromPositions(
  positions: [number, number, number][]
): [number, number, number][] {
  if (positions.length === 0) {
    return [];
  }

  if (positions.length === 1) {
    return [[1, 0, 0]];
  }

  return positions.map((position, index) => {
    const prev = positions[Math.max(0, index - 1)];
    const next = positions[Math.min(positions.length - 1, index + 1)];
    const tangent: [number, number, number] = [
      next[0] - prev[0],
      next[1] - prev[1],
      next[2] - prev[2],
    ];

    if (index === 0) {
      tangent[0] = positions[1][0] - position[0];
      tangent[1] = positions[1][1] - position[1];
      tangent[2] = positions[1][2] - position[2];
    } else if (index === positions.length - 1) {
      tangent[0] = position[0] - positions[index - 1][0];
      tangent[1] = position[1] - positions[index - 1][1];
      tangent[2] = position[2] - positions[index - 1][2];
    }

    return normalize3(tangent);
  });
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
  const rawEdgeLengths = computeSplineEdgeLengths(rawPoints).filter(
    length => Number.isFinite(length) && length > 0
  );
  const medianRawEdgeLengthMm = median(rawEdgeLengths);
  const dedupeThresholdMm = Math.max(
    safeMinSpacingMm * 1.25,
    Math.min(
      1.25,
      medianRawEdgeLengthMm > 0 ? medianRawEdgeLengthMm * 0.22 : safeMinSpacingMm * 1.6
    )
  );
  const closureThresholdMm = Math.max(
    dedupeThresholdMm * 1.5,
    Math.min(
      2.4,
      medianRawEdgeLengthMm > 0 ? medianRawEdgeLengthMm * 0.55 : dedupeThresholdMm * 2.2
    )
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

  const { totalArcLength } = buildSplineArcLengthLut(rawPoints);
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
  const { extended, nSegments, lutT, lutArc, totalArcLength } = buildSplineArcLengthLut(rawPoints);
  const positions: [number, number, number][] = [];
  if (nSegments <= 0 || totalArcLength <= 0) {
    const startPoint = rawPoints[0];
    const endPoint = rawPoints[rawPoints.length - 1];
    const fallbackPositions = Array.from({ length: sampleCount }, (_, index) => {
      const blend = sampleCount > 1 ? index / (sampleCount - 1) : 0;
      return lerpPoint3(startPoint, endPoint, blend);
    });
    return {
      positions: fallbackPositions,
      tangents: computeTangentsFromPositions(fallbackPositions),
    };
  }

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

    positions.push(centripetalCatmullRomPoint(Pm1, P0, P1, P2, localT));
  }

  return { positions, tangents: computeTangentsFromPositions(positions) };
}

interface CPRWorkerLaunchResult {
  pixelData: Float32Array | Uint16Array;
  meanMap?: Float32Array;
  maxMap?: Float32Array;
  sampleCountMap?: Float32Array;
  debugMaps?: PanoImagePayload['debugMaps'];
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  slope: number;
  intercept: number;
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

interface CprOutputSignature {
  sampledCount: number | null;
  checksum: number | null;
  absChecksum: number | null;
  first16: number[];
}

interface CPRWorkerInitSuccessMessage {
  type: 'INIT_SUCCESS';
  requestId: string;
  sessionKey: string;
}

interface CPRWorkerSuccessMessage {
  type: 'SUCCESS';
  requestId: string;
  pixelData: Float32Array | Uint16Array;
  meanMap?: Float32Array;
  maxMap?: Float32Array;
  sampleCountMap?: Float32Array;
  debugMaps?: PanoImagePayload['debugMaps'];
  panoWidth: number;
  panoHeight: number;
  minValue: number;
  maxValue: number;
  windowWidth: number;
  windowCenter: number;
  slope: number;
  intercept: number;
  modalityLutApplied: boolean;
  requestedModalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
  unsignedPackedArtifactDetected: boolean;
  debugPayload?: CPRWorkerLaunchResult['workerDebugPayload'];
}

interface CPRWorkerDisposeSuccessMessage {
  type: 'DISPOSE_SUCCESS';
  requestId: string;
}

interface CPRWorkerErrorMessage {
  type: 'ERROR';
  requestId: string;
  message: string;
}

type CPRWorkerResponseMessage =
  | CPRWorkerInitSuccessMessage
  | CPRWorkerSuccessMessage
  | CPRWorkerDisposeSuccessMessage
  | CPRWorkerErrorMessage;

type CPRWorkerExpectedResponseType = Exclude<CPRWorkerResponseMessage['type'], 'ERROR'>;

type PendingCPRWorkerRequest = {
  expectedTypes: Set<CPRWorkerExpectedResponseType>;
  requestType: string;
  resolve: (value: CPRWorkerResponseMessage) => void;
  reject: (error: Error) => void;
  timeoutId?: number;
};

interface CPRWorkerSession {
  worker: Worker;
  volumeKey: string;
  sessionKey: string;
  pendingRequests: Map<string, PendingCPRWorkerRequest>;
  isTerminating: boolean;
  terminatePromise?: Promise<void>;
  cleanupListeners: () => void;
  rejectPendingRequests: (
    error: Error,
    predicate?: (requestId: string, request: PendingCPRWorkerRequest) => boolean
  ) => void;
}

let activeCPRWorkerSession: CPRWorkerSession | null = null;
let cprWorkerRequestCounter = 0;

function buildCPRWorkerVolumeKey(params: {
  volume: cornerstone.Types.IImageVolume;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  scalarLength: number;
  scalarType: string;
  rescaleSlope: number;
  rescaleIntercept: number;
  intensityPolicyKey: string;
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
    params.intensityPolicyKey,
  ].join('|');
}

function createCPRWorkerRequestId(): string {
  cprWorkerRequestCounter += 1;
  return `cpr-worker-${Date.now().toString(36)}-${cprWorkerRequestCounter.toString(36)}`;
}

function createCPRWorkerSession(
  worker: Worker,
  volumeKey: string,
  sessionKey: string
): CPRWorkerSession {
  const session: CPRWorkerSession = {
    worker,
    volumeKey,
    sessionKey,
    pendingRequests: new Map(),
    isTerminating: false,
    cleanupListeners: () => {},
    rejectPendingRequests: () => {},
  };

  const rejectPendingRequests = (
    error: Error,
    predicate: (requestId: string, request: PendingCPRWorkerRequest) => boolean = () => true
  ) => {
    for (const [requestId, request] of Array.from(session.pendingRequests.entries())) {
      if (!predicate(requestId, request)) {
        continue;
      }

      session.pendingRequests.delete(requestId);
      if (request.timeoutId != null) {
        window.clearTimeout(request.timeoutId);
      }
      request.reject(error);
    }
  };

  const handleMessage = (event: MessageEvent<CPRWorkerResponseMessage>) => {
    const data = event.data;
    if (!data || typeof data !== 'object') {
      return;
    }

    const requestId =
      typeof (data as { requestId?: unknown }).requestId === 'string'
        ? (data as { requestId: string }).requestId
        : null;
    if (!requestId) {
      console.warn('[cprWorker] Ignoring worker response without requestId.', data);
      return;
    }

    const pendingRequest = session.pendingRequests.get(requestId);
    if (!pendingRequest) {
      return;
    }

    session.pendingRequests.delete(requestId);
    if (pendingRequest.timeoutId != null) {
      window.clearTimeout(pendingRequest.timeoutId);
    }

    if (data.type === 'ERROR') {
      pendingRequest.reject(new Error(`[cprWorker] ${data.message}`));
      return;
    }

    if (!pendingRequest.expectedTypes.has(data.type)) {
      pendingRequest.reject(
        new Error(
          `[cprWorker] Unexpected worker response "${data.type}" for ${pendingRequest.requestType}.`
        )
      );
      return;
    }

    pendingRequest.resolve(data);
  };

  const handleError = (event: ErrorEvent) => {
    rejectPendingRequests(new Error(`[cprWorker] Uncaught worker error: ${event.message}`));
  };

  worker.addEventListener('message', handleMessage as EventListener);
  worker.addEventListener('error', handleError as EventListener);

  session.cleanupListeners = () => {
    worker.removeEventListener('message', handleMessage as EventListener);
    worker.removeEventListener('error', handleError as EventListener);
  };
  session.rejectPendingRequests = rejectPendingRequests;

  return session;
}

async function terminateCPRWorkerSession(sessionArg?: CPRWorkerSession | null): Promise<void> {
  const session = sessionArg ?? activeCPRWorkerSession;
  if (!session) {
    return;
  }

  if (activeCPRWorkerSession === session) {
    activeCPRWorkerSession = null;
  }

  if (session.isTerminating) {
    await session.terminatePromise;
    return;
  }

  session.isTerminating = true;
  session.terminatePromise = (async () => {
    session.rejectPendingRequests(
      new Error('[cprWorker] Worker session terminated during teardown.')
    );

    try {
      await postMessageToCPRWorker<CPRWorkerDisposeSuccessMessage>(
        session,
        {
          type: 'DISPOSE' as const,
        },
        {
          expectedTypes: ['DISPOSE_SUCCESS'],
          timeoutMs: 500,
        }
      );
    } catch (disposeError) {
      console.warn('[CPR] Failed to dispose CPR worker before termination.', disposeError);
    } finally {
      session.rejectPendingRequests(
        new Error('[cprWorker] Worker session terminated before pending request completion.')
      );
      session.cleanupListeners();
      session.worker.terminate();
    }
  })();

  await session.terminatePromise;
}

async function ensureCPRWorkerVolumeSession(params: {
  volume: cornerstone.Types.IImageVolume;
  requestedPanoHeight?: number;
  forceApplyModalityLut?: boolean;
  modalityLutOverride?: boolean;
  forceDisableStoredValueNormalization?: boolean;
  debugRunId?: string;
}): Promise<void> {
  const {
    volume,
    requestedPanoHeight,
    forceApplyModalityLut,
    modalityLutOverride,
    forceDisableStoredValueNormalization,
    debugRunId,
  } = params;
  const logPrefix = debugRunId ? `[CPR][${debugRunId}]` : '[CPR]';

  const scalarData = volume.imageData.getPointData().getScalars().getData() as
    | Float32Array
    | Int16Array;
  const dimensions = volume.imageData.getDimensions() as [number, number, number];
  const spacing = volume.imageData.getSpacing() as [number, number, number];
  const origin = volume.imageData.getOrigin() as [number, number, number];
  const direction = Array.from(volume.imageData.getDirection());
  const worldToIndexMatrix =
    (
      volume.imageData as { getWorldToIndex?: () => ArrayLike<number> | null | undefined }
    ).getWorldToIndex?.() ?? null;
  const worldToIndexCandidate = worldToIndexMatrix ? Array.from(worldToIndexMatrix).slice(0, 16) : null;
  const hasWorldToIndex =
    !!worldToIndexCandidate &&
    worldToIndexCandidate.length >= 16 &&
    worldToIndexCandidate.every(value => Number.isFinite(value));
  const worldToIndex = hasWorldToIndex ? worldToIndexCandidate : undefined;
  const { minValue: scalarMin, maxValue: scalarMax } = estimateScalarRange(scalarData);
  const { slope: rescaleSlope, intercept: rescaleIntercept } = getVolumeRescale(volume);
  const {
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    isPreScaled: rawIsPreScaled,
  } = getVolumePixelStorage(volume);
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
  const nominalBitsStored =
    Number.isFinite(bitsStored) && Number(bitsStored) > 0 ? Math.floor(Number(bitsStored)) : 16;
  const nominalStoredMax =
    nominalBitsStored > 0 && nominalBitsStored < 31
      ? (1 << nominalBitsStored) - 1
      : Number.MAX_SAFE_INTEGER;
  const nominalSignedMin =
    nominalBitsStored > 0 && nominalBitsStored < 31
      ? -(1 << (nominalBitsStored - 1))
      : Number.MIN_SAFE_INTEGER;
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
  const intensityPolicyKey = [
    `preScaled=${effectiveIsPreScaled ? 1 : 0}`,
    `applyLut=${applyModalityLut ? 1 : 0}`,
    `allowNorm=${allowStoredValueNormalization ? 1 : 0}`,
    `disableNorm=${forceDisableStoredValueNormalization === true ? 1 : 0}`,
    `override=${typeof modalityLutOverride === 'boolean' ? Number(modalityLutOverride) : 'na'}`,
    `slope=${Number(rescaleSlope).toFixed(6)}`,
    `intercept=${Number(rescaleIntercept).toFixed(6)}`,
  ].join('|');
  const volumeKey = buildCPRWorkerVolumeKey({
    volume,
    dimensions,
    spacing,
    origin,
    scalarLength: scalarData.length,
    scalarType,
    rescaleSlope,
    rescaleIntercept,
    intensityPolicyKey,
  });

  if (activeCPRWorkerSession?.volumeKey === volumeKey) {
    console.log(
      '[CPR-WORKER-PREWARM-JSON]',
      JSON.stringify({
        runId: debugRunId ?? null,
        status: 'reused-existing-session',
        volumeId: volume.volumeId ?? null,
        requestedPanoHeight: requestedPanoHeight ?? null,
        applyModalityLut,
        allowStoredValueNormalization,
      })
    );
    return;
  }

  await terminateCPRWorkerSession();
  const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });
  const sessionKey = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const workerSession = createCPRWorkerSession(worker, volumeKey, sessionKey);
  activeCPRWorkerSession = workerSession;
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
    allowStoredValueNormalization,
    disableStoredValueNormalization: forceDisableStoredValueNormalization === true ? true : undefined,
  };
  const transferList =
    isSharedArrayBuffer || !initScalarData.buffer ? undefined : [initScalarData.buffer];

  try {
    await postMessageToCPRWorker<CPRWorkerInitSuccessMessage>(workerSession, initPayload, {
      expectedTypes: ['INIT_SUCCESS'],
      transferList,
    });
    console.log(
      '[CPR-WORKER-PREWARM-JSON]',
      JSON.stringify({
        runId: debugRunId ?? null,
        status: 'initialized',
        volumeId: volume.volumeId ?? null,
        requestedPanoHeight: requestedPanoHeight ?? null,
        scalarType,
        scalarLength: scalarData.length,
        applyModalityLut,
        allowStoredValueNormalization,
        hasUnsignedPackedArtifact,
        hasBitDepthRangeMismatch,
      })
    );
  } catch (initError) {
    console.warn(`${logPrefix} Failed to prewarm CPR worker session.`, initError);
    await terminateCPRWorkerSession(workerSession);
    throw initError;
  }
}

function postMessageToCPRWorker<T extends CPRWorkerResponseMessage>(
  session: CPRWorkerSession,
  payload: unknown,
  options: {
    expectedTypes: T['type'][];
    transferList?: Transferable[];
    timeoutMs?: number;
  }
): Promise<T> {
  const requestType =
    payload &&
    typeof payload === 'object' &&
    typeof (payload as { type?: unknown }).type === 'string'
      ? String((payload as { type: string }).type)
      : 'UNKNOWN';

  if (session.isTerminating && requestType !== 'DISPOSE') {
    return Promise.reject(new Error('[cprWorker] Worker session is terminating.'));
  }

  const requestId = createCPRWorkerRequestId();
  const expectedTypes = new Set(options.expectedTypes as CPRWorkerExpectedResponseType[]);
  const message = {
    ...(payload as Record<string, unknown>),
    requestId,
  };

  return new Promise((resolve, reject) => {
    const pendingRequest: PendingCPRWorkerRequest = {
      expectedTypes,
      requestType,
      resolve: response => {
        resolve(response as T);
      },
      reject,
    };

    if (options.timeoutMs && options.timeoutMs > 0) {
      pendingRequest.timeoutId = window.setTimeout(() => {
        const liveRequest = session.pendingRequests.get(requestId);
        if (!liveRequest) {
          return;
        }

        session.pendingRequests.delete(requestId);
        liveRequest.reject(
          new Error(
            `[cprWorker] Timed out waiting for ${Array.from(expectedTypes).join('/')} from ${requestType}.`
          )
        );
      }, options.timeoutMs);
    }

    session.pendingRequests.set(requestId, pendingRequest);

    try {
      if (options.transferList && options.transferList.length > 0) {
        session.worker.postMessage(message, options.transferList);
      } else {
        session.worker.postMessage(message);
      }
    } catch (postError) {
      session.pendingRequests.delete(requestId);
      if (pendingRequest.timeoutId != null) {
        window.clearTimeout(pendingRequest.timeoutId);
      }
      reject(postError instanceof Error ? postError : new Error(String(postError)));
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
  renderBackend?: 'gpu' | 'cpu';
  verticalDir?: [number, number, number];
  forceApplyModalityLut?: boolean;
  modalityLutOverride?: boolean;
  forceDisableStoredValueNormalization?: boolean;
  verticalHalfMm?: number;
  verticalCenterOffsetMm?: number;
  rigidVerticalSliceMode?: boolean;
  debugRunId?: string;
  attemptLabel?: string;
  reconstructionMode?: 'legacy' | 'virtualPanoPhase1' | 'virtualPano';
  allowLegacyFallback?: boolean;
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
      renderBackend = CPR_PANO_RECON_BACKEND_DEFAULT,
      verticalDir,
      forceApplyModalityLut,
      modalityLutOverride,
      forceDisableStoredValueNormalization,
      verticalHalfMm,
      verticalCenterOffsetMm,
      rigidVerticalSliceMode = false,
      debugRunId,
      attemptLabel,
      reconstructionMode = 'legacy',
      allowLegacyFallback = CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK,
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
        (
          volume.imageData as { getWorldToIndex?: () => ArrayLike<number> | null | undefined }
        ).getWorldToIndex?.() ?? null;
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
      const {
        bitsStored,
        bitsAllocated,
        highBit,
        pixelRepresentation,
        isPreScaled: rawIsPreScaled,
      } = getVolumePixelStorage(volume);
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
        (scalarData as { constructor?: { name?: string } })?.constructor?.name ||
        'UnknownTypedArray';
      const sourceVolumeId = volume.volumeId || undefined;

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
        nominalBitsStored > 0 && nominalBitsStored < 31
          ? (1 << nominalBitsStored) - 1
          : Number.MAX_SAFE_INTEGER;
      const nominalSignedMin =
        nominalBitsStored > 0 && nominalBitsStored < 31
          ? -(1 << (nominalBitsStored - 1))
          : Number.MIN_SAFE_INTEGER;
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
      const intensityPolicyKey = [
        `preScaled=${effectiveIsPreScaled ? 1 : 0}`,
        `applyLut=${applyModalityLut ? 1 : 0}`,
        `allowNorm=${allowStoredValueNormalization ? 1 : 0}`,
        `disableNorm=${forceDisableStoredValueNormalization === true ? 1 : 0}`,
        `override=${typeof modalityLutOverride === 'boolean' ? Number(modalityLutOverride) : 'na'}`,
        `slope=${Number(rescaleSlope).toFixed(6)}`,
        `intercept=${Number(rescaleIntercept).toFixed(6)}`,
      ].join('|');
      const volumeKey = buildCPRWorkerVolumeKey({
        volume,
        dimensions,
        spacing,
        origin,
        scalarLength: scalarData.length,
        scalarType,
        rescaleSlope,
        rescaleIntercept,
        intensityPolicyKey,
      });

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
        intensityPolicyKey,
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
        await terminateCPRWorkerSession();
        const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });
        const sessionKey = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        const workerSession = createCPRWorkerSession(worker, volumeKey, sessionKey);
        activeCPRWorkerSession = workerSession;
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
          allowStoredValueNormalization,
          disableStoredValueNormalization:
            forceDisableStoredValueNormalization === true ? true : undefined,
        };
        const transferList =
          isSharedArrayBuffer || !initScalarData.buffer ? undefined : [initScalarData.buffer];
        try {
          await postMessageToCPRWorker<CPRWorkerInitSuccessMessage>(workerSession, initPayload, {
            expectedTypes: ['INIT_SUCCESS'],
            transferList,
          });
        } catch (initError) {
          await terminateCPRWorkerSession(workerSession);
          throw initError;
        }
      }

      if (!activeCPRWorkerSession) {
        throw new Error('[cprWorker] Worker session was not initialized.');
      }

      const serializedFrames = frames.map(f => ({
        position: Array.from(f.position) as [number, number, number],
        T: Array.from(f.T) as [number, number, number],
        N_slab: Array.from(f.N_slab) as [number, number, number],
        S: Array.from(f.S) as [number, number, number],
      }));

      const data = await postMessageToCPRWorker<CPRWorkerSuccessMessage>(
        activeCPRWorkerSession,
        {
          type: 'RENDER' as const,
          sessionKey: activeCPRWorkerSession.sessionKey,
          sourceVolumeId,
          frames: serializedFrames,
          panoWidth,
          panoHeight,
          vertHalfMm: dynamicVertHalfMm,
          verticalCenterOffsetMm,
          rigidVerticalSliceMode,
          slabHalfThicknessMm,
          slabSamples,
          aggregation,
          renderBackend,
          verticalDir,
          applyModalityLut,
          allowStoredValueNormalization,
          disableStoredValueNormalization:
            forceDisableStoredValueNormalization === true ? true : undefined,
          debugRunId,
          attemptLabel,
          reconstructionMode,
          allowLegacyFallback,
        },
        {
          expectedTypes: ['SUCCESS'],
        }
      );

      const workerDiagnostic =
        data.debugPayload &&
        typeof data.debugPayload === 'object' &&
        data.debugPayload.diagnostic &&
        typeof data.debugPayload.diagnostic === 'object'
          ? (data.debugPayload.diagnostic as Record<string, unknown>)
          : null;
      const workerMessageLog = JSON.stringify({
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
        diagnosticSummary: workerDiagnostic
          ? {
              renderRoute: workerDiagnostic.renderRoute ?? null,
              rendererFamily: workerDiagnostic.rendererFamily ?? null,
              reconstructionMode: workerDiagnostic.reconstructionMode ?? null,
              requestedReconstructionMode: workerDiagnostic.requestedReconstructionMode ?? null,
              fallbackReason: workerDiagnostic.fallbackReason ?? null,
              outputSelection: workerDiagnostic.outputSelection ?? null,
            }
          : null,
        outputSignature:
          data.debugPayload && typeof data.debugPayload === 'object'
            ? data.debugPayload.outputSignature ?? null
            : null,
      });
      if (reconstructionMode === 'legacy') {
        console.debug(`${logPrefix} [CPR-WORKER-MESSAGE-JSON]`, workerMessageLog);
      } else {
        console.log(`${logPrefix} [CPR-WORKER-MESSAGE-JSON]`, workerMessageLog);
      }

      resolve({
        pixelData: data.pixelData,
        meanMap: data.meanMap,
        maxMap: data.maxMap,
        sampleCountMap: data.sampleCountMap,
        debugMaps: data.debugMaps,
        width: data.panoWidth,
        height: data.panoHeight,
        minValue: data.minValue,
        maxValue: data.maxValue,
        windowWidth: data.windowWidth,
        windowCenter: data.windowCenter,
        slope: data.slope,
        intercept: data.intercept,
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
          data.debugPayload && typeof data.debugPayload === 'object'
            ? data.debugPayload
            : undefined,
      });
    } catch (error) {
      if (activeCPRWorkerSession) {
        await terminateCPRWorkerSession();
      }
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

function estimateScalarRange(scalarData: Float32Array | Int16Array): {
  minValue: number;
  maxValue: number;
} {
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

function getVolumeRescale(volume: cornerstone.Types.IImageVolume): {
  slope: number;
  intercept: number;
} {
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

function getVolumePixelStorage(volume: cornerstone.Types.IImageVolume): {
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
  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 ? Math.floor(bitsStored) : 16;
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && bitsAllocated >= safeBitsStored
      ? Math.floor(bitsAllocated)
      : 16;
  const safeHighBit =
    Number.isFinite(highBit) && highBit >= 0 ? Math.floor(highBit) : safeBitsStored - 1;

  return {
    bitsStored: safeBitsStored,
    bitsAllocated: safeBitsAllocated,
    highBit: Math.max(0, Math.min(safeBitsAllocated - 1, safeHighBit)),
    pixelRepresentation:
      Number.isFinite(pixelRepresentation) &&
      (pixelRepresentation === 0 || pixelRepresentation === 1)
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
  const { isPreScaled, scalarMin, scalarMax, slope, intercept, bitsStored, pixelRepresentation } =
    params;

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
    Number.isFinite(bitsStored) && bitsStored >= 1 && bitsStored <= 31
      ? Math.floor(bitsStored)
      : 16;
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

function findViewportByLogicalId(servicesManager: any, logicalViewportId: string) {
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

type GridViewportStateLike = {
  viewportId?: string;
  displaySetInstanceUIDs?: string[];
  displaySetOptions?: Array<Record<string, unknown>>;
  viewportOptions?: Record<string, unknown> & {
    viewportId?: string;
    viewportType?: string;
  };
};

function findViewportGridStateByLogicalId(
  servicesManager: any,
  logicalViewportId: string
): { stateViewportId: string; gridViewport: GridViewportStateLike } | null {
  const { viewportGridService } = servicesManager.services;
  const gridViewports = viewportGridService.getState().viewports as
    | Map<string, GridViewportStateLike>
    | Record<string, GridViewportStateLike>
    | undefined;
  const gridEntries: Array<[string, GridViewportStateLike]> =
    gridViewports &&
    typeof (gridViewports as Map<string, GridViewportStateLike>).entries === 'function'
      ? Array.from((gridViewports as Map<string, GridViewportStateLike>).entries())
      : Object.entries((gridViewports || {}) as Record<string, GridViewportStateLike>);

  for (const [gridViewportId, gridViewport] of gridEntries) {
    if (gridViewport?.viewportOptions?.viewportId !== logicalViewportId) {
      continue;
    }

    const stateViewportId =
      typeof gridViewport?.viewportId === 'string' && gridViewport.viewportId.length > 0
        ? gridViewport.viewportId
        : gridViewportId;

    return {
      stateViewportId,
      gridViewport,
    };
  }

  return null;
}

async function ensureViewportTypeByLogicalId(
  servicesManager: any,
  logicalViewportId: string,
  targetViewportType: 'stack' | 'volume',
  runId: string
): Promise<void> {
  const { viewportGridService } = servicesManager.services;
  const gridState = findViewportGridStateByLogicalId(servicesManager, logicalViewportId);
  if (!gridState) {
    console.warn(
      `[CPR][${runId}] Unable to enforce viewport type; logical viewport "${logicalViewportId}" was not found in the grid state.`
    );
    return;
  }

  const currentViewport = findViewportByLogicalId(servicesManager, logicalViewportId);
  const actualTypeMatches =
    targetViewportType === 'stack'
      ? isStackViewportLike(currentViewport)
      : isVolumeViewportLike(currentViewport);
  const currentViewportType = toNonEmptyString(
    gridState.gridViewport.viewportOptions?.viewportType
  );

  if (actualTypeMatches && currentViewportType === targetViewportType) {
    return;
  }

  const displaySetInstanceUIDs = Array.isArray(gridState.gridViewport.displaySetInstanceUIDs)
    ? gridState.gridViewport.displaySetInstanceUIDs.filter(
        (displaySetInstanceUID): displaySetInstanceUID is string =>
          typeof displaySetInstanceUID === 'string' && displaySetInstanceUID.length > 0
      )
    : [];
  if (!displaySetInstanceUIDs.length) {
    console.warn(
      `[CPR][${runId}] Unable to enforce viewport type "${targetViewportType}" for "${logicalViewportId}" because the viewport has no bound display set.`
    );
    return;
  }

  const nextViewportOptions = {
    ...(gridState.gridViewport.viewportOptions || {}),
    viewportId: logicalViewportId,
    viewportType: targetViewportType,
  };
  const nextDisplaySetOptions = Array.isArray(gridState.gridViewport.displaySetOptions)
    ? gridState.gridViewport.displaySetOptions.map(displaySetOptions =>
        displaySetOptions && typeof displaySetOptions === 'object'
          ? { ...displaySetOptions }
          : displaySetOptions
      )
    : [];

  console.log(
    `[CPR-VIEWPORT-TYPE-SWITCH] run=${runId} logicalViewportId=${logicalViewportId} ` +
      `from=${currentViewportType ?? 'unknown'} to=${targetViewportType}`
  );

  viewportGridService.setDisplaySetsForViewport({
    viewportId: gridState.stateViewportId,
    displaySetInstanceUIDs,
    viewportOptions: nextViewportOptions,
    displaySetOptions: nextDisplaySetOptions,
  });

  if (targetViewportType === 'stack') {
    await waitForStackViewportByLogicalId(servicesManager, logicalViewportId, 4000);
    return;
  }

  await waitForVolumeViewportByLogicalId(servicesManager, logicalViewportId, 4000);
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
  console.log(
    '[CPR-SOURCE-VOLUME-LOAD-JSON]',
    JSON.stringify({
      runId: debugRunId,
      status: initialStatus.loaded ? 'already-loaded' : 'waiting',
      ...initialStatus,
    })
  );

  if (
    !volume.loadStatus ||
    volume.loadStatus.loaded === true ||
    typeof volume.load !== 'function'
  ) {
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
      console.log(
        '[CPR-SOURCE-VOLUME-LOAD-JSON]',
        JSON.stringify({
          runId: debugRunId,
          status,
          volumeId,
          loaded: volume.loadStatus?.loaded === true,
          loading: volume.loadStatus?.loading === true,
        })
      );
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
      console.log(
        '[CPR-SOURCE-VOLUME-LOAD-JSON]',
        JSON.stringify({
          runId: debugRunId,
          status: 'triggered-load',
          volumeId,
        })
      );
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

function clonePanoVoiSettings(
  voi: Partial<PanoVoiSettings> | null | undefined
): PanoVoiSettings | null {
  const lower = toFiniteNumber(voi?.lower);
  const upper = toFiniteNumber(voi?.upper);
  const windowWidth = toFiniteNumber(voi?.windowWidth);
  const windowCenter = toFiniteNumber(voi?.windowCenter);

  if (
    lower == null ||
    upper == null ||
    windowWidth == null ||
    windowCenter == null ||
    upper <= lower ||
    windowWidth <= 0
  ) {
    return null;
  }

  return {
    lower: Number(lower),
    upper: Number(upper),
    windowWidth: Number(windowWidth),
    windowCenter: Number(windowCenter),
  };
}

function createPanoVoiFromWindowLevel(
  windowWidth: number,
  windowCenter: number
): PanoVoiSettings | null {
  const safeWindowWidth = toFiniteNumber(windowWidth);
  const safeWindowCenter = toFiniteNumber(windowCenter);

  if (safeWindowWidth == null || safeWindowCenter == null || safeWindowWidth <= 0) {
    return null;
  }

  const lower = safeWindowCenter - safeWindowWidth / 2;
  const upper = safeWindowCenter + safeWindowWidth / 2;
  return clonePanoVoiSettings({
    lower,
    upper,
    windowWidth: safeWindowWidth,
    windowCenter: safeWindowCenter,
  });
}

function createPanoVoiFromRange(
  range:
    | {
        lower?: number | null;
        upper?: number | null;
      }
    | null
    | undefined
): PanoVoiSettings | null {
  const lower = toFiniteNumber(range?.lower);
  const upper = toFiniteNumber(range?.upper);

  if (lower == null || upper == null || upper <= lower) {
    return null;
  }

  const windowWidth = upper - lower;
  return clonePanoVoiSettings({
    lower,
    upper,
    windowWidth,
    windowCenter: lower + windowWidth / 2,
  });
}

function arePanoVoiSettingsClose(
  left: PanoVoiSettings | null,
  right: PanoVoiSettings | null,
  tolerance = 2
): boolean {
  if (!left || !right) {
    return false;
  }

  return (
    Math.abs(left.lower - right.lower) <= tolerance &&
    Math.abs(left.upper - right.upper) <= tolerance &&
    Math.abs(left.windowWidth - right.windowWidth) <= tolerance &&
    Math.abs(left.windowCenter - right.windowCenter) <= tolerance
  );
}

function evaluateHostedPanoViewportVoiUpdate(params: {
  nextVoi: PanoVoiSettings;
  authoritativeVoi: PanoVoiSettings | null;
  suppressPanoViewportEventsUntil: number;
  now: number;
}): { accept: boolean; shouldRepair: boolean; reason: string } {
  const { nextVoi, authoritativeVoi, suppressPanoViewportEventsUntil, now } = params;

  if (!authoritativeVoi) {
    return { accept: true, shouldRepair: false, reason: 'no-authority' };
  }

  if (now < suppressPanoViewportEventsUntil) {
    if (arePanoVoiSettingsClose(nextVoi, authoritativeVoi)) {
      return { accept: false, shouldRepair: false, reason: 'startup-match' };
    }

    return { accept: false, shouldRepair: true, reason: 'startup-divergence' };
  }

  const authorityWidth = Math.max(authoritativeVoi.windowWidth, 1);
  const authorityIsHuLike = isHuLikeRange(authoritativeVoi.lower, authoritativeVoi.upper);
  const centerDelta = Math.abs(nextVoi.windowCenter - authoritativeVoi.windowCenter);
  const lowerDelta = Math.abs(nextVoi.lower - authoritativeVoi.lower);
  const upperDelta = Math.abs(nextVoi.upper - authoritativeVoi.upper);
  const widthRatio = nextVoi.windowWidth / authorityWidth;
  const implausiblyNarrow = widthRatio < 0.01 && centerDelta > Math.max(150, authorityWidth * 0.15);
  const implausiblyWide = widthRatio > 12 && centerDelta > Math.max(1500, authorityWidth * 0.75);
  const implausiblyDisplaced =
    centerDelta > Math.max(2500, authorityWidth * 1.5) &&
    lowerDelta > Math.max(1250, authorityWidth * 0.75) &&
    upperDelta > Math.max(1250, authorityWidth * 0.75);
  const brokeHuDomain = authorityIsHuLike && !isHuLikeRange(nextVoi.lower, nextVoi.upper);

  if (brokeHuDomain) {
    return { accept: false, shouldRepair: true, reason: 'left-hu-domain' };
  }
  if (implausiblyNarrow) {
    return { accept: false, shouldRepair: true, reason: 'implausibly-narrow' };
  }
  if (implausiblyWide) {
    return { accept: false, shouldRepair: true, reason: 'implausibly-wide' };
  }
  if (implausiblyDisplaced) {
    return { accept: false, shouldRepair: true, reason: 'implausibly-displaced' };
  }

  return { accept: true, shouldRepair: false, reason: 'accepted' };
}

function isVolumeViewportLike(
  viewport: cornerstone.Types.IViewport | null
): viewport is cornerstone.Types.IVolumeViewport {
  if (!viewport) {
    return false;
  }

  const candidate = viewport as cornerstone.Types.IVolumeViewport;

  return typeof candidate.setVolumes === 'function' && typeof candidate.setBlendMode === 'function';
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

async function waitForPanoVolumeViewport(
  servicesManager: any,
  timeoutMs = 12000
): Promise<cornerstone.Types.IVolumeViewport> {
  return waitForVolumeViewportByLogicalId(servicesManager, 'cpr-pano', timeoutMs);
}

async function waitForVolumeViewportByLogicalId(
  servicesManager: any,
  logicalViewportId: string,
  timeoutMs = 12000
): Promise<cornerstone.Types.IVolumeViewport> {
  const startedAt = Date.now();
  let lastViewportType = 'none';

  while (Date.now() - startedAt < timeoutMs) {
    const viewport = findViewportByLogicalId(servicesManager, logicalViewportId);
    if (viewport) {
      lastViewportType = viewport.constructor?.name || 'unknown';

      if (isVolumeViewportLike(viewport)) {
        return viewport as cornerstone.Types.IVolumeViewport;
      }
    }

    await new Promise(resolve => window.setTimeout(resolve, 50));
  }

  throw new Error(
    `[CPR] ${logicalViewportId} volume viewport not ready within timeout. Last resolved type: ${lastViewportType}`
  );
}

function viewportHasVolumeId(
  viewport: cornerstone.Types.IViewport | null,
  volumeId: string
): boolean {
  if (!viewport || !isVolumeViewportLike(viewport) || !volumeId) {
    return false;
  }

  try {
    if (typeof (viewport as cornerstone.Types.IVolumeViewport).hasVolumeId === 'function') {
      if ((viewport as cornerstone.Types.IVolumeViewport).hasVolumeId(volumeId)) {
        return true;
      }
    }
  } catch (error) {
    console.warn('[CPR] Failed to query viewport volume presence via hasVolumeId.', error);
  }

  try {
    const actors = (viewport as cornerstone.Types.IVolumeViewport).getActors?.() ?? [];
    return Array.isArray(actors) && actors.some(actor => actor?.uid === volumeId);
  } catch (error) {
    console.warn('[CPR] Failed to query viewport actors while checking volume presence.', error);
  }

  return false;
}

function formatCprDoneLogValue(value: unknown): string {
  if (value === null || value === undefined) {
    return 'na';
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toString() : 'na';
  }

  if (typeof value === 'string' || typeof value === 'boolean') {
    return String(value);
  }

  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function logCprDonePhase(
  runId: string,
  phase: string,
  servicesManager: any,
  extra: Record<string, unknown> = {}
): void {
  const { cornerstoneViewportService } = servicesManager.services;
  const cacheSizeBytes = cornerstone.cache.getCacheSize?.();
  const cacheFreeBytes = cornerstone.cache.getBytesAvailable?.();
  const activeViewportCount =
    typeof cornerstoneViewportService?.getViewportIds === 'function'
      ? cornerstoneViewportService.getViewportIds().length
      : 'na';

  const extraText = Object.entries(extra)
    .map(([key, value]) => `${key}=${formatCprDoneLogValue(value)}`)
    .join(' ');

  console.log(
    `CPR-DONE-PHASE run=${runId} phase=${phase} cacheSizeBytes=${formatCprDoneLogValue(
      cacheSizeBytes
    )} cacheFreeBytes=${formatCprDoneLogValue(
      cacheFreeBytes
    )} activeViewportCount=${formatCprDoneLogValue(activeViewportCount)}${
      extraText ? ` ${extraText}` : ''
    }`
  );
}

async function waitForCrossSectionVolumeViewport(
  servicesManager: any,
  timeoutMs = 12000
): Promise<cornerstone.Types.IVolumeViewport> {
  return waitForVolumeViewportByLogicalId(servicesManager, 'cpr-crosssection', timeoutMs);
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
    console.log(
      '[CPR-PANO-SNAPSHOT-JSON]',
      JSON.stringify({
        runId,
        stage,
        missingViewport: true,
      })
    );
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
  console.log(
    '[CPR-PANO-SNAPSHOT-JSON]',
    JSON.stringify({
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
    })
  );
}

function applyPanoDisplaySettings(
  runId: string,
  stage: string,
  viewport: cornerstone.Types.IViewport | null,
  adaptiveVoi: PanoVoiSettings
): void {
  if (!viewport || !isStackViewportLike(viewport)) {
    console.warn(
      `[CPR][${runId}] applyPanoDisplaySettings skipped at ${stage}: stack viewport missing`
    );
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
  const appliedProperties = ((viewport as { getProperties?: () => unknown }).getProperties?.() ||
    {}) as {
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
  };
  console.log(
    '[CPR-VOI-APPLY-JSON]',
    JSON.stringify({
      runId,
      stage,
      requestedVoi: adaptiveVoi,
      appliedVoiRange: appliedProperties.voiRange ?? null,
      appliedVoiLutFunction: appliedProperties.VOILUTFunction ?? null,
    })
  );
}

function getInitialPanoWindowLevelFromSourceViewport(servicesManager: any): {
  windowWidth: number;
  windowCenter: number;
  source: string;
} {
  const panoPreset = {
    windowWidth: CPR_VTK_PANO_PRESET_WINDOW_WIDTH,
    windowCenter: CPR_VTK_PANO_PRESET_WINDOW_CENTER,
  };
  const axialViewport = findViewportByLogicalId(servicesManager, 'cpr-axial');
  if (axialViewport && isVolumeViewportLike(axialViewport)) {
    const axialVolumeViewport = axialViewport as cornerstone.Types.IVolumeViewport;
    const actorUID = axialVolumeViewport.getActors?.()?.[0]?.uid;
    const properties = ((actorUID
      ? axialVolumeViewport.getProperties?.(actorUID) || axialVolumeViewport.getProperties?.()
      : axialVolumeViewport.getProperties?.()) || {}) as {
      voiRange?: { lower?: number; upper?: number };
    };
    const lower = toFiniteNumber(properties.voiRange?.lower);
    const upper = toFiniteNumber(properties.voiRange?.upper);
    if (Number.isFinite(lower) && Number.isFinite(upper) && Number(upper) > Number(lower)) {
      const windowWidth = Number(upper) - Number(lower);
      const axialCenter = Number(lower) + windowWidth / 2;
      const blendedWindowWidth = Math.max(
        1800,
        Math.min(4200, windowWidth * 0.35 + panoPreset.windowWidth * 0.65)
      );
      const blendedWindowCenter = Math.max(
        -200,
        Math.min(1400, axialCenter * 0.35 + panoPreset.windowCenter * 0.65)
      );
      return {
        windowWidth: blendedWindowWidth,
        windowCenter: blendedWindowCenter,
        source: 'pano-hu-preset+axial-fallback',
      };
    }
  }

  return {
    windowWidth: panoPreset.windowWidth,
    windowCenter: panoPreset.windowCenter,
    source: 'pano-hu-preset',
  };
}

type VtkPanoPresetCandidate = {
  label: string;
  verticalHalfMm: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  score: number;
};

type VtkPanoProjectionMode = 'AVERAGE' | 'MAX' | 'MIN';

function roundToStep(value: number, step: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(step) || step <= 0) {
    return value;
  }

  return Math.round(value / step) * step;
}

function toOddSampleCount(value: number, minValue: number, maxValue: number): number {
  const clamped = Math.max(minValue, Math.min(maxValue, Math.round(value)));
  return clamped % 2 === 0 ? Math.max(minValue, Math.min(maxValue, clamped + 1)) : clamped;
}

function selectReadableVtkPanoPreset(params: {
  baseVerticalHalfMm: number;
  requestedSlabHalfThicknessMm: number;
  requestedSlabSamples: number;
  minSpacing: number;
  aggregation: 'MIP' | 'MEAN';
}): {
  selected: VtkPanoPresetCandidate;
  candidates: VtkPanoPresetCandidate[];
} {
  const desiredVerticalHalfMm = Math.max(15, Math.min(22, params.baseVerticalHalfMm));
  const averageProjection = params.aggregation === 'MEAN';
  const baseDesiredSlabHalfMm =
    params.minSpacing <= 0.2
      ? 0.7
      : params.minSpacing <= 0.3
        ? 0.9
        : params.minSpacing <= 0.45
          ? 1.1
          : 1.3;
  const desiredSlabHalfMm = averageProjection
    ? Math.max(0.45, Math.min(0.85, roundToStep(baseDesiredSlabHalfMm * 0.65, 0.05)))
    : baseDesiredSlabHalfMm;
  const desiredSamples = toOddSampleCount(
    (desiredSlabHalfMm * 2) / Math.max(params.minSpacing, 0.12),
    averageProjection ? 3 : 3,
    averageProjection ? 11 : 17
  );
  const baseSlabHalfMm = Math.max(0.3, params.requestedSlabHalfThicknessMm);
  const baseSamples = Math.max(3, params.requestedSlabSamples);
  const rawCandidates = [
    {
      label: 'focused',
      verticalHalfMm: Math.max(14, Math.min(20, roundToStep(desiredVerticalHalfMm - 1.5, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.5 : 0.35,
        Math.min(
          averageProjection ? 1.7 : 0.9,
          roundToStep(
            averageProjection
              ? Math.max(Math.min(baseSlabHalfMm, 1.7), Math.min(desiredSlabHalfMm, 1.7), 1.5)
              : Math.min(baseSlabHalfMm, desiredSlabHalfMm),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 9), Math.min(desiredSamples, 9), 9)
          : Math.min(baseSamples, desiredSamples),
        averageProjection ? 9 : 3,
        averageProjection ? 9 : 11
      ),
    },
    {
      label: 'balanced',
      verticalHalfMm: Math.max(15, Math.min(22, roundToStep(desiredVerticalHalfMm, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.75 : 0.5,
        Math.min(
          averageProjection ? 2.0 : 1.4,
          roundToStep(
            averageProjection
              ? Math.max(Math.min(baseSlabHalfMm, 2.0), Math.min(desiredSlabHalfMm, 2.0), 1.75)
              : Math.max(desiredSlabHalfMm, Math.min(baseSlabHalfMm, 1.4)),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 11), Math.min(desiredSamples, 11), 11)
          : Math.max(baseSamples, desiredSamples),
        averageProjection ? 11 : 5,
        averageProjection ? 11 : 15
      ),
    },
    {
      label: 'broad',
      verticalHalfMm: Math.max(16, Math.min(24, roundToStep(desiredVerticalHalfMm + 2, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.9 : 0.9,
        Math.min(
          averageProjection ? 2.0 : 2.2,
          roundToStep(
            Math.max(
              averageProjection
                ? Math.min(baseSlabHalfMm, 2.0)
                : baseSlabHalfMm,
              averageProjection
                ? Math.min(desiredSlabHalfMm + 0.15, 2.0)
                : desiredSlabHalfMm + 0.35
            ),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 11), Math.min(desiredSamples + 2, 11), 11)
          : Math.max(baseSamples, desiredSamples + 2),
        averageProjection ? 11 : 7,
        averageProjection ? 11 : 19
      ),
    },
  ];
  const candidates = rawCandidates.map(candidate => {
    const slabPenalty =
      Math.abs(candidate.slabHalfThicknessMm - desiredSlabHalfMm) * (averageProjection ? 0.9 : 4.2);
    const verticalPenalty = Math.abs(candidate.verticalHalfMm - desiredVerticalHalfMm) * 0.8;
    const samplePenalty =
      Math.abs(candidate.slabSamples - desiredSamples) * (averageProjection ? 0.12 : 0.35);
    const broadPenalty =
      candidate.slabHalfThicknessMm > (averageProjection ? 2.0 : 1.8)
        ? averageProjection
          ? 0.15
          : 1.5
        : 0;
    const tallPenalty = candidate.verticalHalfMm > 23 ? 0.8 : 0;
    const labelBias =
      candidate.label === 'balanced' ? 1.2 : candidate.label === 'focused' ? 1.0 : 0.9;
    return {
      ...candidate,
      score:
        10 + labelBias - slabPenalty - verticalPenalty - samplePenalty - broadPenalty - tallPenalty,
    };
  });
  const selected =
    candidates
      .slice()
      .sort((a, b) => b.score - a.score || a.slabHalfThicknessMm - b.slabHalfThicknessMm)[0] ||
    candidates[0];

  return {
    selected,
    candidates,
  };
}

function selectVtkPanoProjectionMode(
  aggregation: 'MIP' | 'MEAN' | null | undefined
): VtkPanoProjectionMode {
  return aggregation === 'MEAN' ? 'AVERAGE' : 'MAX';
}

async function hydratePanoViewportForCornerstoneUi(
  viewport: cornerstone.Types.IVolumeViewport,
  sourceVolumeId: string,
  windowWidth: number,
  windowCenter: number,
  displayState: VolumeViewportDisplayState | null,
  runId: string
): Promise<void> {
  await viewport.setVolumes([
    {
      volumeId: sourceVolumeId,
      blendMode: CPR_CROSSSECTION_DEFAULT_BLEND_MODE,
      slabThickness: CPR_CROSSSECTION_DEFAULT_SLAB_THICKNESS_MM,
    },
  ]);

  if (displayState) {
    restoreVolumeViewportDisplayState(viewport, displayState);
  }

  const { lower, upper } = cornerstone.utilities.windowLevel.toLowHighRange(
    windowWidth,
    windowCenter
  );

  viewport.setProperties(
    {
      isComputedVOI: false,
      voiRange: { lower, upper },
      VOILUTFunction: 'LINEAR_EXACT',
    } as any,
    sourceVolumeId
  );
  viewport.render();

  console.log(
    `CPR-VTK-PANO-UI-HYDRATE run=${runId} viewportId=${viewport.id} volumeId=${sourceVolumeId} windowWidth=${windowWidth.toFixed(
      2
    )} windowCenter=${windowCenter.toFixed(2)} hasDisplayState=${displayState ? 'yes' : 'no'}`
  );
}

function logVtkPanoViewportSnapshot(
  runId: string,
  stage: string,
  viewport: cornerstone.Types.IViewport | null
): void {
  if (!viewport) {
    console.log(`CPR-VTK-PANO-SNAPSHOT run=${runId} stage=${stage} missingViewport=true`);
    return;
  }

  const actorUIDs = viewport.getActors?.().map(actorEntry => actorEntry.uid) || [];
  console.log(
    `CPR-VTK-PANO-SNAPSHOT run=${runId} stage=${stage} viewportId=${viewport.id} viewportType=${
      viewport.constructor?.name || 'unknown'
    } actorCount=${actorUIDs.length} actorUIDs=${actorUIDs.join('|') || 'none'}`
  );
}

function getViewportElementDimensions(viewport: cornerstone.Types.IViewport | null): {
  width: number;
  height: number;
} {
  const element = viewport?.element as HTMLDivElement | null | undefined;

  if (!element) {
    return { width: 0, height: 0 };
  }

  const rect = element.getBoundingClientRect?.();
  const width = Number(rect?.width ?? element.clientWidth ?? 0);
  const height = Number(rect?.height ?? element.clientHeight ?? 0);

  return {
    width: Number.isFinite(width) ? width : 0,
    height: Number.isFinite(height) ? height : 0,
  };
}

async function waitForAnimationFrames(frameCount = 1): Promise<void> {
  const safeFrameCount = Math.max(1, Math.floor(frameCount));

  for (let index = 0; index < safeFrameCount; index++) {
    await new Promise<void>(resolve => {
      window.requestAnimationFrame(() => resolve());
    });
  }
}

async function waitForViewportElementToHaveSize(
  viewport: cornerstone.Types.IViewport | null,
  timeoutMs = CPR_CROSSSECTION_RENDER_WAIT_TIMEOUT_MS
): Promise<void> {
  if (!viewport || typeof window === 'undefined') {
    return;
  }

  const hasRenderableSize = () => {
    const { width, height } = getViewportElementDimensions(viewport);

    return width > 0 && height > 0;
  };

  if (!hasRenderableSize()) {
    await new Promise<void>((resolve, reject) => {
      const startedAt = Date.now();

      const tick = () => {
        if (hasRenderableSize()) {
          resolve();
          return;
        }

        if (Date.now() - startedAt >= timeoutMs) {
          reject(
            new Error('[CPR] Timed out waiting for the cross-section viewport to report a size.')
          );
          return;
        }

        window.requestAnimationFrame(tick);
      };

      window.requestAnimationFrame(tick);
    });
  }

  await new Promise<void>(resolve => {
    window.requestAnimationFrame(() => resolve());
  });
}

function setViewportVisibility(
  viewport: { element?: HTMLDivElement | null } | null | undefined,
  isVisible: boolean
): void {
  const element = viewport?.element;
  if (!element) {
    return;
  }

  element.style.opacity = isVisible ? '1' : '0';
  element.style.pointerEvents = isVisible ? 'auto' : 'none';
}

function restoreCprViewportVisibility(servicesManager: any, logicalViewportId: string): void {
  setViewportVisibility(findViewportByLogicalId(servicesManager, logicalViewportId), true);
}

function restoreAllCprViewportVisibility(servicesManager: any): void {
  restoreCprViewportVisibility(servicesManager, 'cpr-axial');
  restoreCprViewportVisibility(servicesManager, 'cpr-pano');
  restoreCprViewportVisibility(servicesManager, 'cpr-crosssection');
}

function resolveCrossSectionVerticalCenterOffsetMm(
  frameIndex: number,
  fallbackVerticalCenterOffsetMm?: number,
  verticalCenterOffsetsMm?: number[] | null
): number {
  const safeFallbackVerticalCenterOffsetMm = toFiniteNumber(fallbackVerticalCenterOffsetMm) ?? 0;
  if (!Array.isArray(verticalCenterOffsetsMm) || verticalCenterOffsetsMm.length === 0) {
    return safeFallbackVerticalCenterOffsetMm;
  }

  const maxIndex = Math.max(0, verticalCenterOffsetsMm.length - 1);
  const safeIndex = Math.max(0, Math.min(Math.round(frameIndex), maxIndex));
  const localVerticalCenterOffsetMm = Number(verticalCenterOffsetsMm[safeIndex]);

  return Number.isFinite(localVerticalCenterOffsetMm)
    ? localVerticalCenterOffsetMm
    : safeFallbackVerticalCenterOffsetMm;
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

function buildAxialRestoreCamera(
  preservedCamera: Record<string, cornerstone.Types.Point3 | number | boolean | undefined> | null
): Partial<cornerstone.Types.ICamera> | null {
  if (!preservedCamera) {
    return null;
  }

  // Keep the slice plane/orientation, but let the new viewport keep its own fit-scale.
  return {
    focalPoint: preservedCamera.focalPoint as cornerstone.Types.Point3 | undefined,
    position: preservedCamera.position as cornerstone.Types.Point3 | undefined,
    viewPlaneNormal: preservedCamera.viewPlaneNormal as cornerstone.Types.Point3 | undefined,
    viewUp: preservedCamera.viewUp as cornerstone.Types.Point3 | undefined,
    parallelProjection:
      typeof preservedCamera.parallelProjection === 'boolean'
        ? preservedCamera.parallelProjection
        : undefined,
  };
}

type VolumeViewportDisplayState = {
  volumeId?: string;
  properties: {
    isComputedVOI?: boolean;
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
    invert?: unknown;
    colormap?: unknown;
  };
};

function cloneVoiRange(
  voiRange: { lower?: number; upper?: number } | null | undefined
): { lower?: number; upper?: number } | undefined {
  if (!voiRange || typeof voiRange !== 'object') {
    return undefined;
  }

  return {
    lower: toFiniteNumber(voiRange.lower),
    upper: toFiniteNumber(voiRange.upper),
  };
}

function getVolumeViewportDisplayVolumeId(
  viewport: cornerstone.Types.IVolumeViewport
): string | undefined {
  void viewport;
  const sourceVolumeId = cprStateService.getSourceVolumeId();
  return typeof sourceVolumeId === 'string' && sourceVolumeId.length > 0
    ? sourceVolumeId
    : undefined;
}

function captureVolumeViewportDisplayState(
  viewport: cornerstone.Types.IVolumeViewport
): VolumeViewportDisplayState | null {
  const properties = ((viewport as { getProperties?: () => unknown }).getProperties?.() || {}) as {
    isComputedVOI?: boolean;
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
    invert?: unknown;
    colormap?: unknown;
  };

  return {
    volumeId: getVolumeViewportDisplayVolumeId(viewport),
    properties: {
      isComputedVOI:
        typeof properties.isComputedVOI === 'boolean' ? properties.isComputedVOI : undefined,
      voiRange: cloneVoiRange(properties.voiRange),
      VOILUTFunction: properties.VOILUTFunction,
      invert: properties.invert,
      colormap: properties.colormap,
    },
  };
}

function restoreVolumeViewportDisplayState(
  viewport: cornerstone.Types.IVolumeViewport,
  displayState: VolumeViewportDisplayState | null
): void {
  if (!displayState) {
    return;
  }

  const nextProperties: {
    isComputedVOI?: boolean;
    voiRange?: { lower?: number; upper?: number };
    VOILUTFunction?: unknown;
    invert?: unknown;
    colormap?: unknown;
  } = {};

  if (typeof displayState.properties.isComputedVOI === 'boolean') {
    nextProperties.isComputedVOI = displayState.properties.isComputedVOI;
  }
  if (displayState.properties.voiRange) {
    nextProperties.voiRange = {
      lower: displayState.properties.voiRange.lower,
      upper: displayState.properties.voiRange.upper,
    };
  }
  if (displayState.properties.VOILUTFunction !== undefined) {
    nextProperties.VOILUTFunction = displayState.properties.VOILUTFunction;
  }
  if (displayState.properties.invert !== undefined) {
    nextProperties.invert = displayState.properties.invert;
  }
  if (displayState.properties.colormap !== undefined) {
    nextProperties.colormap = displayState.properties.colormap;
  }

  if (Object.keys(nextProperties).length === 0) {
    return;
  }

  (
    viewport as {
      setProperties: (properties: typeof nextProperties, volumeId?: string) => void;
    }
  ).setProperties(nextProperties, displayState.volumeId);
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

function cloneAnnotationLike<T>(annotation: T | null | undefined): T | null {
  if (!annotation) {
    return null;
  }

  try {
    const structuredCloneImpl = (globalThis as { structuredClone?: <V>(value: V) => V })
      .structuredClone;
    if (typeof structuredCloneImpl === 'function') {
      return structuredCloneImpl(annotation);
    }
  } catch (error) {
    console.warn(
      '[CPR] structuredClone failed for spline annotation snapshot; falling back to JSON clone.',
      error
    );
  }

  try {
    return JSON.parse(JSON.stringify(annotation)) as T;
  } catch (error) {
    console.warn('[CPR] JSON clone failed for spline annotation snapshot.', error);
    return null;
  }
}

function ensureSplineAnnotationVisible(
  axialViewport: cornerstone.Types.IViewport,
  annotationUID: string | null,
  annotationSnapshot: AnnotationLike | null,
  runId: string,
  stage: string
): void {
  const axialElement = axialViewport.element as HTMLDivElement | null | undefined;
  if (!axialElement) {
    return;
  }

  const repaintSplineAnnotation = () => {
    cornerstoneTools.utilities.triggerAnnotationRender(axialElement);
    const renderingEngine = axialViewport.getRenderingEngine?.();
    const enabledViewport = renderingEngine?.getViewport?.(axialViewport.id);
    if (enabledViewport?.element) {
      cornerstoneTools.utilities.triggerAnnotationRender(enabledViewport.element);
    }
    axialViewport.render?.();
    console.log('DIAG-TRIPWIRE: spline-redraw-triggered for cpr-axial');
  };

  const localAnnotations =
    (cornerstoneTools.annotation.state.getAnnotations(
      'SplineROI',
      axialElement
    ) as AnnotationLike[]) || [];
  const localCount = localAnnotations.length;
  const hasTargetLocally = annotationUID
    ? localAnnotations.some(annotation => annotation?.annotationUID === annotationUID)
    : localCount > 0;

  console.log(
    `[CPR-SPLINE-STATE] run=${runId} stage=${stage} localCount=${localCount} globalCount=${
      getAllSplineAnnotations().length
    } targetUID=${annotationUID || 'none'} hasTargetLocally=${hasTargetLocally}`
  );

  if (hasTargetLocally) {
    repaintSplineAnnotation();
    return;
  }

  const snapshotToRestore = cloneAnnotationLike(annotationSnapshot);
  if (!snapshotToRestore) {
    console.warn(
      `[CPR][${runId}] Missing spline snapshot during ${stage}; axial viewport will not be able to re-render the arch.`
    );
    return;
  }

  if (annotationUID) {
    const existingGlobal = cornerstoneTools.annotation.state.getAnnotation?.(annotationUID);
    if (existingGlobal) {
      cornerstoneTools.annotation.state.removeAnnotation(annotationUID);
    }
    snapshotToRestore.annotationUID = annotationUID;
  }

  if (snapshotToRestore?.data?.handles) {
    snapshotToRestore.data.handles.activeHandleIndex = null;
  }
  snapshotToRestore.highlighted = false;
  snapshotToRestore.invalidated = true;

  cornerstoneTools.annotation.state.addAnnotation(
    snapshotToRestore as Parameters<typeof cornerstoneTools.annotation.state.addAnnotation>[0],
    axialElement
  );
  repaintSplineAnnotation();

  const restoredLocalCount =
    (
      cornerstoneTools.annotation.state.getAnnotations(
        'SplineROI',
        axialElement
      ) as AnnotationLike[]
    )?.length || 0;
  console.log(
    `[CPR-SPLINE-RESTORE] run=${runId} stage=${stage} restoredUID=${
      snapshotToRestore.annotationUID || 'generated'
    } localCount=${restoredLocalCount}`
  );
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
  if (!crossViewport || !isVolumeViewportLike(crossViewport)) {
    return;
  }

  const previousCamera = cloneCameraState(crossViewport);
  const previousDisplayState = captureVolumeViewportDisplayState(crossViewport);
  crossViewport.setCamera(
    buildCrossSectionCameraForFrame(frame, previousCamera, verticalCenterOffsetMm)
  );
  const reapplyDisplayState = () => {
    restoreVolumeViewportDisplayState(crossViewport, previousDisplayState);
    crossViewport.render?.();
  };
  reapplyDisplayState();
  const crossViewportElement = crossViewport.element;
  if (crossViewportElement) {
    let isVolumeListenerActive = true;
    const cleanupVolumeListener = () => {
      if (!isVolumeListenerActive) {
        return;
      }
      isVolumeListenerActive = false;
      crossViewportElement.removeEventListener(
        cornerstone.Enums.Events.VOLUME_NEW_IMAGE,
        onVolumeNewImage as EventListener
      );
    };
    const onVolumeNewImage = () => {
      if (!isVolumeListenerActive) {
        return;
      }
      cleanupVolumeListener();
      reapplyDisplayState();
    };

    crossViewportElement.addEventListener(
      cornerstone.Enums.Events.VOLUME_NEW_IMAGE,
      onVolumeNewImage as EventListener
    );

    window.requestAnimationFrame(() => {
      reapplyDisplayState();
      window.setTimeout(() => {
        cleanupVolumeListener();
      }, 250);
    });
    return;
  }
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
  verticalCenterOffsetsMm: number[] | undefined,
  samplingOffsetsActive: boolean,
  servicesManager: any
): Promise<void> {
  const { syncGroupService } = servicesManager.services;
  const crossViewport = await waitForCrossSectionVolumeViewport(servicesManager);

  const initialFrameIndex = Math.max(
    0,
    Math.min(cprStateService.getCurrentFrameIndex(), Math.max(0, frames.length - 1))
  );
  const initialFrameVerticalCenterOffsetMm = resolveCrossSectionVerticalCenterOffsetMm(
    initialFrameIndex,
    verticalCenterOffsetMm,
    verticalCenterOffsetsMm
  );
  console.log(
    '[CPR-CROSSSECTION-CONFIG-JSON]',
    JSON.stringify({
      sourceVolumeId,
      frameCount: frames.length,
      initialFrameIndex,
      verticalHalfHeightMm: toFiniteNumber(verticalHalfHeightMm) ?? null,
      verticalCenterOffsetMm: toFiniteNumber(verticalCenterOffsetMm) ?? 0,
      initialFrameVerticalCenterOffsetMm,
      geometryMode: 'hybrid-volume-voi-custom-camera',
      samplingOffsetsActive,
      blendMode: 'avg',
      slabThicknessMm: CPR_CROSSSECTION_DEFAULT_SLAB_THICKNESS_MM,
      voiMode: 'native-volume-default',
    })
  );

  setViewportVisibility(crossViewport, false);
  try {
    const crossSectionVolumeAlreadyBound = viewportHasVolumeId(crossViewport, sourceVolumeId);
    console.log(
      `CPR-CROSSSECTION-SETVOLUMES sourceVolumeId=${sourceVolumeId} viewportId=${crossViewport.id} ` +
        `action=${crossSectionVolumeAlreadyBound ? 'reuse-existing' : 'set-volumes'}`
    );
    if (!crossSectionVolumeAlreadyBound) {
      await crossViewport.setVolumes([
        {
          volumeId: sourceVolumeId,
          blendMode: CPR_CROSSSECTION_DEFAULT_BLEND_MODE,
          slabThickness: CPR_CROSSSECTION_DEFAULT_SLAB_THICKNESS_MM,
        },
      ]);
    }
    await waitForViewportElementToHaveSize(crossViewport);
    const initialCrossSectionCamera = buildCrossSectionCameraForFrame(
      frames[initialFrameIndex],
      undefined,
      initialFrameVerticalCenterOffsetMm
    );
    crossViewport.setCamera(initialCrossSectionCamera);
    crossViewport.render?.();
    await new Promise<void>(resolve => {
      window.requestAnimationFrame(() => {
        resolve();
      });
    });
  } finally {
    setViewportVisibility(crossViewport, true);
  }

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
  warning: string | null;
}

type PointObject = { x: number; y: number; z: number };
type PointLike = [number, number, number] | PointObject;

type PanoCameraBridgeState = {
  actorUID: string;
  viewportId: string;
  basePan: [number, number];
  baseZoom: number;
  baseViewUp: [number, number, number];
  baseViewPlaneNormal: [number, number, number];
};

type CPRHostedPanoHandle = {
  actorUID: string;
  updateWindowLevel(windowWidth: number, windowCenter: number): void;
  resetCamera(): void;
  render(): void;
  resize(): void;
  getReattachState(): HostedVtkPanoReattachState | null;
  captureCameraSyncBaseline(): void;
  syncCamera(
    syncState:
      | {
          panDeltaPx?: ArrayLike<number> | null;
          zoomRatio?: number | null;
          viewportHeightPx?: number | null;
        }
      | null
      | undefined
  ): void;
  clearCameraSyncBaseline(): void;
  dispose(): void;
};

type CPRHostedPanoViewportElement = HTMLElement & {
  __cprVtkPanoHost?: CPRHostedPanoHandle;
};

function getHostedPanoHandle(
  viewport: cornerstone.Types.IViewport | null | undefined
): CPRHostedPanoHandle | null {
  const element = viewport?.element as CPRHostedPanoViewportElement | null | undefined;
  return element?.__cprVtkPanoHost || null;
}

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
    if (
      Number.isFinite(candidate[0]) &&
      Number.isFinite(candidate[1]) &&
      Number.isFinite(candidate[2])
    ) {
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
  slabHalfThicknessMm = CPR_GPU_PANO_DEFAULT_SLAB_HALF_THICKNESS_MM,
  slabSamples = CPR_GPU_PANO_DEFAULT_SLAB_SAMPLES,
  aggregation = 'MIP',
}: UseCPROrchestratorProps): UseCPROrchestratorReturn {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const hpSubscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const hpTimeoutRef = useRef<number | null>(null);
  const lastSetupCleanupRef = useRef(false);
  const lastProtocolIdRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);
  const sliderAnimationFrameRef = useRef<number | null>(null);
  const pendingSliderFrameIndexRef = useRef<number | null>(null);
  const generationInFlightRef = useRef(false);
  const lastGenerationStartedAtRef = useRef(0);
  const attachedVtkPanoRef = useRef<AttachedVtkPanoCpr | CPRHostedPanoHandle | null>(null);
  const persistedSplineAnnotationRef = useRef<AnnotationLike | null>(null);
  const panoCameraBridgeRef = useRef<PanoCameraBridgeState | null>(null);
  const hostedPanoVoiAuthorityRef = useRef<HostedPanoVoiAuthority>({
    runId: null,
    sourceVolumeId: null,
    authoritativeVoi: null,
    suppressPanoViewportEventsUntil: 0,
  });
  const hostedPanoVoiRepairTimeoutRef = useRef<number | null>(null);
  const pendingAxialRestoreCleanupRef = useRef<(() => void) | null>(null);
  const axialRestoreTokenRef = useRef(0);

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

  const markGenerationIdle = useCallback(() => {
    generationInFlightRef.current = false;
    if (isMountedRef.current) {
      setIsGenerating(false);
    }
  }, []);

  const clearCprOneUpStateAndCache = useCallback(
    (reason: string) => {
      const { hangingProtocolService, stateSyncService } = servicesManager.services;
      const syncState = stateSyncService.getState?.() || {};
      const viewportGridStore = { ...(syncState.viewportGridStore || {}) };
      const activeStudyUID = hangingProtocolService.getState?.()?.activeStudyUID;
      const cacheKey = activeStudyUID ? `${activeStudyUID}:cpr:1` : null;
      if (cacheKey && viewportGridStore[cacheKey]) {
        delete viewportGridStore[cacheKey];
      }
      console.log(`[CPR-ONE-UP] cleared cached stage-1 layout reason=${reason}`);
      stateSyncService.store({
        viewportGridStore,
        toggleCprOneUpViewportGridStore: {},
        cprOneUpPanoStore: null,
      });
    },
    [servicesManager]
  );

  const getLiveHostedPano = useCallback(() => {
    const viewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
    const hostedPano = getHostedPanoHandle(viewport);
    return { viewport, hostedPano };
  }, [servicesManager]);

  const setAxialTransitioning = useCallback((isTransitioning: boolean, reason: string) => {
    const previousValue = cprStateService.isAxialTransitioning();
    cprStateService.setAxialTransitioning(isTransitioning);
    if (previousValue !== isTransitioning) {
      console.log(
        `[CPR-AXIAL-CURTAIN] state=${isTransitioning ? 'hidden' : 'visible'} reason=${reason}`
      );
    }
  }, []);

  const clearPendingAxialViewportRestore = useCallback((reason: string) => {
    axialRestoreTokenRef.current += 1;
    const cleanup = pendingAxialRestoreCleanupRef.current;
    pendingAxialRestoreCleanupRef.current = null;
    if (cleanup) {
      console.log(`[CPR-AXIAL-RESTORE] clearing pending restore reason=${reason}`);
      cleanup();
    }
  }, []);

  const clearHostedPanoVoiRepair = useCallback(() => {
    if (hostedPanoVoiRepairTimeoutRef.current != null) {
      window.clearTimeout(hostedPanoVoiRepairTimeoutRef.current);
      hostedPanoVoiRepairTimeoutRef.current = null;
    }
  }, []);

  const clearHostedPanoVoiAuthority = useCallback(
    (reason: string) => {
      clearHostedPanoVoiRepair();
      const previousAuthority = hostedPanoVoiAuthorityRef.current;
      if (previousAuthority.authoritativeVoi || previousAuthority.sourceVolumeId) {
        console.log(
          `[CPR-VTK-PANO-VOI-AUTHORITY] clear reason=${reason} run=${
            previousAuthority.runId || 'na'
          } sourceVolumeId=${previousAuthority.sourceVolumeId || 'none'}`
        );
      }
      hostedPanoVoiAuthorityRef.current = {
        runId: null,
        sourceVolumeId: null,
        authoritativeVoi: null,
        suppressPanoViewportEventsUntil: 0,
      };
    },
    [clearHostedPanoVoiRepair]
  );

  const setHostedPanoVoiAuthority = useCallback(
    (params: {
      runId: string;
      sourceVolumeId: string | null;
      voi: PanoVoiSettings | null;
      source: string;
      suppressPanoViewportEventsForMs?: number;
    }) => {
      const nextAuthority = clonePanoVoiSettings(params.voi);
      if (!nextAuthority) {
        return;
      }

      hostedPanoVoiAuthorityRef.current = {
        runId: params.runId,
        sourceVolumeId: params.sourceVolumeId,
        authoritativeVoi: nextAuthority,
        suppressPanoViewportEventsUntil:
          Date.now() + Math.max(0, params.suppressPanoViewportEventsForMs ?? 0),
      };
      console.log(
        `[CPR-VTK-PANO-VOI-AUTHORITY] run=${params.runId} source=${params.source} sourceVolumeId=${
          params.sourceVolumeId || 'none'
        } windowWidth=${nextAuthority.windowWidth.toFixed(2)} windowCenter=${nextAuthority.windowCenter.toFixed(
          2
        )} lower=${nextAuthority.lower.toFixed(2)} upper=${nextAuthority.upper.toFixed(2)} suppressMs=${Math.max(
          0,
          params.suppressPanoViewportEventsForMs ?? 0
        )}`
      );
    },
    []
  );

  const scheduleHostedPanoVoiRepair = useCallback(
    (reason: string) => {
      clearHostedPanoVoiRepair();
      const authority = hostedPanoVoiAuthorityRef.current;
      const authoritativeVoi = clonePanoVoiSettings(authority.authoritativeVoi);
      if (!authoritativeVoi) {
        return;
      }

      hostedPanoVoiRepairTimeoutRef.current = window.setTimeout(() => {
        hostedPanoVoiRepairTimeoutRef.current = null;
        if (!isMountedRef.current) {
          return;
        }

        const livePanoViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
        if (!livePanoViewport) {
          return;
        }

        console.warn(
          `[CPR-VTK-PANO-VOI-GUARD] run=${authority.runId || 'na'} repairingHiddenViewport reason=${reason} ` +
            `windowWidth=${authoritativeVoi.windowWidth.toFixed(2)} windowCenter=${authoritativeVoi.windowCenter.toFixed(
              2
            )}`
        );
        applyPanoDisplaySettings(
          authority.runId || 'na',
          `vtk-guard-repair:${reason}`,
          livePanoViewport,
          authoritativeVoi
        );
        livePanoViewport.render?.();
      }, 0);
    },
    [clearHostedPanoVoiRepair, servicesManager]
  );

  const clearPanoCameraBridge = useCallback(
    (reason: string) => {
      const bridge = panoCameraBridgeRef.current;
      if (bridge) {
        console.log(
          `CPR-VTK-PANO-CAMERA-BRIDGE clear reason=${reason} actorUID=${bridge.actorUID} viewportId=${bridge.viewportId}`
        );
      }
      panoCameraBridgeRef.current = null;
      attachedVtkPanoRef.current?.clearCameraSyncBaseline?.();
      getLiveHostedPano().hostedPano?.clearCameraSyncBaseline?.();
    },
    [getLiveHostedPano]
  );

  const disposeAttachedVtkPano = useCallback(
    (reason: string) => {
      clearPanoCameraBridge(`${reason}-dispose`);
      const attached = getLiveHostedPano().hostedPano || attachedVtkPanoRef.current;
      if (!attached) {
        clearHostedPanoVoiAuthority(`${reason}-no-attached`);
        return;
      }

      console.log(`CPR-VTK-PANO-DETACH reason=${reason} actorUID=${attached.actorUID}`);
      try {
        attached.dispose();
      } finally {
        attachedVtkPanoRef.current = null;
        clearHostedPanoVoiAuthority(reason);
      }
    },
    [clearHostedPanoVoiAuthority, clearPanoCameraBridge, getLiveHostedPano]
  );

  const initializePanoCameraBridge = useCallback(
    (
      viewport: cornerstone.Types.IVolumeViewport,
      attached: AttachedVtkPanoCpr | CPRHostedPanoHandle,
      runId: string
    ): boolean => {
      const viewportWithPanZoom = viewport as cornerstone.Types.IVolumeViewport & {
        getPan?: () => ArrayLike<number> | null | undefined;
        getZoom?: () => number;
      };
      const camera = viewport.getCamera?.();
      const basePan = toFiniteVector2Tuple(viewportWithPanZoom.getPan?.());
      const baseZoom = Number(viewportWithPanZoom.getZoom?.());
      const baseViewUp = toFiniteVector3Tuple(camera?.viewUp);
      const baseViewPlaneNormal = toFiniteVector3Tuple(camera?.viewPlaneNormal);

      if (
        !basePan ||
        !Number.isFinite(baseZoom) ||
        baseZoom <= 0 ||
        !baseViewUp ||
        !baseViewPlaneNormal
      ) {
        clearPanoCameraBridge(`${runId}-initialize-invalid`);
        return false;
      }

      attached.captureCameraSyncBaseline();
      panoCameraBridgeRef.current = {
        actorUID: attached.actorUID,
        viewportId: viewport.id,
        basePan,
        baseZoom,
        baseViewUp,
        baseViewPlaneNormal,
      };

      console.log(
        `CPR-VTK-PANO-CAMERA-BRIDGE init run=${runId} actorUID=${attached.actorUID} viewportId=${viewport.id} basePan=${basePan[0].toFixed(
          2
        )},${basePan[1].toFixed(2)} baseZoom=${baseZoom.toFixed(4)}`
      );
      console.log(
        `DIAG-TRIPWIRE: camera-bridge-armed run=${runId} actorUID=${attached.actorUID} viewportId=${viewport.id}`
      );

      return true;
    },
    [clearPanoCameraBridge]
  );

  const rearmPanoCameraBridgeFromLiveViewport = useCallback(
    (reason: string): boolean => {
      const { viewport, hostedPano } = getLiveHostedPano();
      if (!hostedPano || !viewport || !isVolumeViewportLike(viewport)) {
        clearPanoCameraBridge(`${reason}-missing-live-host`);
        return false;
      }

      attachedVtkPanoRef.current = hostedPano;
      return initializePanoCameraBridge(
        viewport as cornerstone.Types.IVolumeViewport,
        hostedPano,
        reason
      );
    },
    [clearPanoCameraBridge, getLiveHostedPano, initializePanoCameraBridge]
  );

  const scheduleAxialViewportRestore = useCallback(
    (
      runId: string,
      annotationUID: string | null,
      annotationSnapshot: AnnotationLike | null,
      preservedAxialCamera: Record<
        string,
        cornerstone.Types.Point3 | number | boolean | undefined
      > | null,
      stage: string
    ) => {
      clearPendingAxialViewportRestore(`${stage}-reschedule`);
      const restoreToken = axialRestoreTokenRef.current;

      void (async () => {
        try {
          const axialViewportAfterSwitch = await waitForViewportByLogicalId(
            servicesManager,
            'cpr-axial',
            4000
          );
          if (
            !axialViewportAfterSwitch ||
            !isMountedRef.current ||
            restoreToken !== axialRestoreTokenRef.current
          ) {
            return;
          }

          setAxialTransitioning(true, `${stage}-mounted`);
          setViewportVisibility(axialViewportAfterSwitch, false);
          let axialElement: HTMLDivElement | null = null;
          let restoreTimeoutId: number | null = null;

          const onImageRendered = (event: Event) => {
            if (restoreToken !== axialRestoreTokenRef.current || !isMountedRef.current) {
              cleanup();
              return;
            }

            const detail = (
              event as CustomEvent<{
                viewportId?: string;
                viewportStatus?: unknown;
              }>
            ).detail;

            console.log(
              `DIAG-TRIPWIRE: axial-image-rendered run=${runId} stage=${stage} phase=${phase} targetViewportId=${
                axialViewportAfterSwitch.id
              } eventViewportId=${detail?.viewportId || 'unknown'} viewportStatus=${String(
                detail?.viewportStatus ?? 'unknown'
              )}`
            );

            if (detail?.viewportId !== axialViewportAfterSwitch.id) {
              return;
            }

            if (detail?.viewportStatus !== cornerstone.Enums.ViewportStatus.RENDERED) {
              return;
            }

            if (phase === 'await-initial-render' && preservedAxialCamera) {
              phase = 'await-restored-render';
              const axialRestoreCamera = buildAxialRestoreCamera(preservedAxialCamera);
              if (axialRestoreCamera) {
                axialViewportAfterSwitch.setCamera(axialRestoreCamera);
              }
              axialViewportAfterSwitch.render();
              return;
            }

            cleanup();
            const rehydratePointCount = Array.isArray(annotationSnapshot?.data?.handles?.points)
              ? annotationSnapshot.data.handles.points.length
              : 0;
            console.log(
              `DIAG-TRIPWIRE: spline-rehydrate run=${runId} stage=${stage} annotationUID=${
                annotationUID || 'none'
              } pointCount=${rehydratePointCount}`
            );
            ensureSplineAnnotationVisible(
              axialViewportAfterSwitch,
              annotationUID,
              annotationSnapshot,
              runId,
              stage
            );
          };

          const cleanup = () => {
            if (restoreTimeoutId != null) {
              window.clearTimeout(restoreTimeoutId);
              restoreTimeoutId = null;
            }
            axialElement?.removeEventListener(
              cornerstone.Enums.Events.IMAGE_RENDERED,
              onImageRendered as EventListener
            );
            setViewportVisibility(axialViewportAfterSwitch, true);
            setAxialTransitioning(false, `${stage}-cleanup`);
            if (pendingAxialRestoreCleanupRef.current === cleanup) {
              pendingAxialRestoreCleanupRef.current = null;
            }
          };

          const waitForAxialElement = async (): Promise<HTMLDivElement | null> => {
            const startedAt = performance.now();

            while (restoreToken === axialRestoreTokenRef.current && isMountedRef.current) {
              const nextElement = axialViewportAfterSwitch.element as
                | HTMLDivElement
                | null
                | undefined;
              if (nextElement) {
                return nextElement;
              }

              if (restoreTimeoutId == null || performance.now() - startedAt >= 1450) {
                return null;
              }

              await waitForAnimationFrames(1);
            }

            return null;
          };

          let phase: 'await-initial-render' | 'await-restored-render' = preservedAxialCamera
            ? 'await-initial-render'
            : 'await-restored-render';

          pendingAxialRestoreCleanupRef.current = cleanup;
          restoreTimeoutId = window.setTimeout(() => {
            cleanup();
          }, 1500);

          axialElement = await waitForAxialElement();
          if (
            !axialElement ||
            restoreToken !== axialRestoreTokenRef.current ||
            !isMountedRef.current ||
            pendingAxialRestoreCleanupRef.current !== cleanup
          ) {
            cleanup();
            return;
          }

          axialElement.addEventListener(
            cornerstone.Enums.Events.IMAGE_RENDERED,
            onImageRendered as EventListener
          );
          servicesManager.services.cornerstoneViewportService.resize(true);
          axialViewportAfterSwitch.render();
        } catch (axialRestoreError) {
          if (restoreToken !== axialRestoreTokenRef.current) {
            return;
          }
          clearPendingAxialViewportRestore(`${stage}-failed`);
          setAxialTransitioning(false, `${stage}-failed`);
          restoreCprViewportVisibility(servicesManager, 'cpr-axial');
          console.warn(
            `[CPR][${runId}] Failed to restore axial camera after stage switch`,
            axialRestoreError
          );
        }
      })();
    },
    [clearPendingAxialViewportRestore, servicesManager, setAxialTransitioning]
  );

  const ensureSetupViewportInteraction = useCallback(async () => {
    try {
      const axialViewport = await waitForViewportByLogicalId(servicesManager, 'cpr-axial', 4000);
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
  }, [commandsManager, servicesManager]);

  const clearCprRuntimeState = useCallback(
    (reason: string, reenableSetupInteraction = false) => {
      const currentAxialViewport = findViewportByLogicalId(servicesManager, 'cpr-axial');
      if (currentAxialViewport?.element) {
        cornerstoneTools.cancelActiveManipulations?.(currentAxialViewport.element);
      }

      clearPendingAxialViewportRestore(reason);
      setAxialTransitioning(false, reason);
      restoreAllCprViewportVisibility(servicesManager);
      clearProtocolListener();
      disposeAttachedVtkPano(reason);
      clearPanoDebugProbe();
      cprStateService.clear();
      persistedSplineAnnotationRef.current = null;
      clearPanoImageCache();
      removeAllSplineAnnotations();
      clearCprOneUpStateAndCache(reason);
      if (isMountedRef.current) {
        setError(null);
        setWarning(null);
      }
      markGenerationIdle();
      if (reenableSetupInteraction) {
        void ensureSetupViewportInteraction();
      }
    },
    [
      clearCprOneUpStateAndCache,
      clearPendingAxialViewportRestore,
      clearProtocolListener,
      disposeAttachedVtkPano,
      ensureSetupViewportInteraction,
      markGenerationIdle,
      servicesManager,
      setAxialTransitioning,
    ]
  );

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      generationInFlightRef.current = false;
      if (sliderAnimationFrameRef.current != null) {
        window.cancelAnimationFrame(sliderAnimationFrameRef.current);
        sliderAnimationFrameRef.current = null;
      }
      clearCprRuntimeState('hook-cleanup');
      void terminateCPRWorkerSession();
    };
  }, [clearCprRuntimeState]);

  useEffect(() => {
    const { hangingProtocolService } = servicesManager.services;
    const protocolEvent =
      hangingProtocolService.EVENTS.PROTOCOL_APPLIED ||
      hangingProtocolService.EVENTS.PROTOCOL_CHANGED;

    if (!protocolEvent) {
      return;
    }

    const handleProtocolState = () => {
      const hpState = hangingProtocolService.getState?.() || {};
      const currentProtocolId =
        typeof hpState.protocolId === 'string' && hpState.protocolId.length > 0
          ? hpState.protocolId
          : null;
      const currentStageIndex = Number.isFinite(hpState.stageIndex)
        ? Number(hpState.stageIndex)
        : null;
      const previousProtocolId = lastProtocolIdRef.current;

      if (previousProtocolId === 'cpr' && currentProtocolId !== 'cpr') {
        lastSetupCleanupRef.current = false;
        clearCprRuntimeState('protocol-exit');
      } else if (currentProtocolId === 'cpr' && currentStageIndex === 0) {
        if (!lastSetupCleanupRef.current) {
          lastSetupCleanupRef.current = true;
          clearCprRuntimeState('setup-stage', true);
        }
      } else {
        lastSetupCleanupRef.current = false;
      }

      lastProtocolIdRef.current = currentProtocolId;
    };

    handleProtocolState();
    const subscription = hangingProtocolService.subscribe(protocolEvent, handleProtocolState);

    return () => {
      subscription?.unsubscribe?.();
    };
  }, [clearCprRuntimeState, servicesManager]);

  const onDone = useCallback(async () => {
    const startedAt = performance.now();
    if (generationInFlightRef.current) {
      console.warn('[CPR] Ignoring duplicate onDone while a generation is already in flight.');
      return;
    }
    if (startedAt - lastGenerationStartedAtRef.current < CPR_PANO_GENERATION_DEBOUNCE_MS) {
      console.warn('[CPR] Ignoring debounced onDone request.');
      return;
    }

    generationInFlightRef.current = true;
    lastGenerationStartedAtRef.current = startedAt;
    setIsGenerating(true);
    setError(null);
    setWarning(null);
    const debugRunId = `cpr-${Date.now().toString(36)}`;
    console.log(`[CPR][${debugRunId}] onDone started`);

    try {
      clearPendingAxialViewportRestore('onDone-start');
      setAxialTransitioning(true, 'onDone-start');
      clearPanoCameraBridge('onDone-start');
      clearProtocolListener();
      disposeAttachedVtkPano('regenerate');
      clearPanoDebugProbe();

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
      const preservedAxialDisplayState = isVolumeViewportLike(axialViewport)
        ? captureVolumeViewportDisplayState(axialViewport as cornerstone.Types.IVolumeViewport)
        : null;
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
      const latestAnnotationSnapshot = cloneAnnotationLike(latestAnnotation);
      persistedSplineAnnotationRef.current = latestAnnotationSnapshot;

      console.log(
        `[CPR-SPLINE-PERSIST] run=${debugRunId} annotationUID=${latestAnnotationUID || 'none'} globalCount=${
          getAllSplineAnnotations().length
        }`
      );

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
      const workerWarmupPromise = ensureCPRWorkerVolumeSession({
        volume,
        requestedPanoHeight: requestedPanoHeightPx,
        modalityLutOverride: true,
        debugRunId: `${debugRunId}-warmup`,
      }).catch(warmupError => {
        console.warn(`[CPR][${debugRunId}] CPR worker prewarm failed; falling back to lazy init.`, {
          error: warmupError instanceof Error ? warmupError.message : String(warmupError),
        });
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
      console.log(
        '[CPR-SPLINE-PLANE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          rawPointCount: rawControlPointsWorld.length,
          projectedPointCount: rawControlPoints.length,
          projected: projectedSpline.diagnostics.projected,
          maxDistanceMm: projectedSpline.diagnostics.maxDistanceMm,
          meanDistanceMm: projectedSpline.diagnostics.meanDistanceMm,
        })
      );
      console.log(
        '[CPR-RIGID-SLICE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          enabled: true,
          slicePlanePoint: preservedAxialCamera?.focalPoint ?? null,
          slicePlaneNormal: preservedAxialCamera?.viewPlaneNormal ?? null,
          usesProjectedSplinePlane: projectedSpline.diagnostics.projected,
          projectedMaxDistanceMm: projectedSpline.diagnostics.maxDistanceMm,
          projectedMeanDistanceMm: projectedSpline.diagnostics.meanDistanceMm,
          rigidVerticalCenterOffsetMm: 0,
        })
      );
      const sanitizedSpline = sanitizeSplineControlPoints(rawControlPoints, minSpacing);
      const controlPoints = sanitizedSpline.points;
      console.log(
        '[CPR-SPLINE-SANITIZED-JSON]',
        JSON.stringify({
          runId: debugRunId,
          rawPointCount: rawControlPointsWorld.length,
          sanitizedPointCount: controlPoints.length,
          diagnostics: sanitizedSpline.diagnostics,
        })
      );
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
      const strongerMandibularCenterOffsetMm = -7.2;
      const rootedMandibularCenterOffsetMm = -5.8;
      const balancedSlabHalfThicknessMm = 1.6;
      const balancedSlabSamples = 9;
      const fastSlabHalfThicknessMm = 1.2;
      const fastSlabSamples = 7;
      const balancedMeanSlabHalfThicknessMm = 1.75;
      const balancedMeanSlabSamples = 11;
      const leakageFocusedMeanSlabHalfThicknessMm = 1.45;
      const leakageFocusedMeanSlabSamples = 13;
      const leakageSharpMeanSlabHalfThicknessMm = 1.35;
      const leakageSharpMeanSlabSamples = 15;
      const focusedMeanSlabHalfThicknessMm = 1.5;
      const focusedMeanSlabSamples = 9;
      const broadMeanSlabHalfThicknessMm = 2.0;
      const broadMeanSlabSamples = 11;
      const meanFallbackSlabHalfThicknessMm = 2.0;
      const meanFallbackSlabSamples = 11;
      const sharpMeanSlabHalfThicknessMm = 1.25;
      const sharpMeanSlabSamples = 7;
      const rootedMediumVerticalHalfMm = Math.max(
        toothBandVerticalHalfMm + 1.8,
        Math.min(mediumVerticalHalfMm, toothBandVerticalHalfMm + 3.2)
      );
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

      const vtkPresetSelection = selectReadableVtkPanoPreset({
        baseVerticalHalfMm,
        requestedSlabHalfThicknessMm: CPR_GPU_PANO_DEFAULT_SLAB_HALF_THICKNESS_MM,
        requestedSlabSamples: CPR_GPU_PANO_DEFAULT_SLAB_SAMPLES,
        minSpacing,
        aggregation,
      });
      const vtkVerticalHalfMm = vtkPresetSelection.selected.verticalHalfMm;
      const vtkSlabHalfThicknessMm = vtkPresetSelection.selected.slabHalfThicknessMm;
      const vtkSlabSamples = vtkPresetSelection.selected.slabSamples;
      const vtkProjectionMode = selectVtkPanoProjectionMode(aggregation);
      const vtkActualVerticalHalfMm = vtkVerticalHalfMm;
      const vtkCrossSectionVerticalCenterOffsetMm = 0;
      const vtkLocalCenterOffsetsMm: number[] = [];

      if (CPR_PANO_DISPLAY_PATH_DEFAULT === 'vtk-reference') {
        console.log(
          `[CPR-VTK-PANO-PRESET] run=${debugRunId} selected=${vtkPresetSelection.selected.label} ` +
            `score=${formatReadablePanoValue(vtkPresetSelection.selected.score, 2)} ` +
            `verticalHalfMm=${formatReadablePanoValue(vtkVerticalHalfMm, 2)} ` +
            `slabHalfMm=${formatReadablePanoValue(vtkSlabHalfThicknessMm, 2)} ` +
            `slabSamples=${vtkSlabSamples}`
        );
        console.log(
          '[CPR-VTK-PANO-PRESET-JSON]',
          JSON.stringify({
            runId: debugRunId,
            minSpacing,
            requested: {
              baseVerticalHalfMm,
              slabHalfThicknessMm: toPositiveFinite(
                slabHalfThicknessMm,
                CPR_GPU_PANO_DEFAULT_SLAB_HALF_THICKNESS_MM
              ),
              slabSamples: Math.max(1, Math.round(slabSamples)),
            },
            candidates: vtkPresetSelection.candidates,
            selected: vtkPresetSelection.selected,
          })
        );
        console.log(
          `CPR-VTK-PANO-PATH run=${debugRunId} displayedPath=${CPR_PANO_DISPLAY_PATH_DEFAULT} targetViewportType=volume sourceVolumeId=${sourceVolumeId} frameCount=${frames.length} projectionMode=${vtkProjectionMode} aggregation=${aggregation} slabThicknessMm=${(
            vtkSlabHalfThicknessMm * 2
          ).toFixed(2)} slabSamples=${vtkSlabSamples} widthMm=${(vtkVerticalHalfMm * 2).toFixed(2)}`
        );
        logPanoDisplayRoute({
          runId: debugRunId,
          displayedPath: 'vtk-reference',
          backend: 'vtk',
          pipelineMode: `vtk-${String(vtkProjectionMode).toLowerCase()}`,
          fallbackReason: null,
        });
        logReconModeJson({
          runId: debugRunId,
          displayedPath: 'vtk-reference',
          backend: 'vtk',
          pipelineMode: `vtk-${String(vtkProjectionMode).toLowerCase()}`,
          reconstructionMode: 'vtk-reference',
          displaySource: 'vtk-hosted-pano-actor',
          referencePathAvailable: true,
          sourceVolumeId,
          fallbackReason: null,
        });

        console.log(
          `[CPR-SPLINE-PRESERVE] run=${debugRunId} mode=vtk-reference annotationUID=${
            latestAnnotationUID || 'none'
          } skipping pre-transition spline cleanup`
        );
        logCprDonePhase(debugRunId, 'vtk-stage0-ready', servicesManager, {
          sourceVolumeId,
          frameCount: frames.length,
        });
        cprStateService.setArchData(
          controlPoints,
          frames,
          sourceVolumeId,
          latestAnnotationUID,
          vtkCrossSectionVerticalCenterOffsetMm,
          vtkLocalCenterOffsetsMm
        );
        clearPanoImageCache();
        clearProtocolListener();
        setViewportVisibility(findViewportByLogicalId(servicesManager, 'cpr-pano'), false);

        logCprDonePhase(debugRunId, 'before-stage1-switch', servicesManager, {
          sourceVolumeId,
        });

        let protocolAppliedHandled = false;
        const onProtocolApplied = async () => {
          if (protocolAppliedHandled) {
            return;
          }
          protocolAppliedHandled = true;

          clearProtocolListener();

          try {
            scheduleAxialViewportRestore(
              debugRunId,
              latestAnnotationUID,
              latestAnnotationSnapshot,
              preservedAxialCamera,
              'vtk-after-stage-switch'
            );

            logCprDonePhase(debugRunId, 'stage1-applied', servicesManager, {
              sourceVolumeId,
            });
            await waitForAnimationFrames(1);
            await ensureViewportTypeByLogicalId(servicesManager, 'cpr-pano', 'volume', debugRunId);
            const panoViewport = await waitForPanoVolumeViewport(servicesManager);
            setViewportVisibility(panoViewport, false);
            await waitForViewportElementToHaveSize(panoViewport);
            logCprDonePhase(debugRunId, 'pano-viewport-ready', servicesManager, {
              viewportId: panoViewport.id,
            });
            logVtkPanoViewportSnapshot(debugRunId, 'before-attach', panoViewport);

            const wl = getInitialPanoWindowLevelFromSourceViewport(servicesManager);
            const initialPanoVoi = createPanoVoiFromWindowLevel(wl.windowWidth, wl.windowCenter);
            setHostedPanoVoiAuthority({
              runId: debugRunId,
              sourceVolumeId,
              voi: initialPanoVoi,
              source: `vtk-initial:${wl.source}`,
              suppressPanoViewportEventsForMs: CPR_VTK_PANO_STARTUP_VOI_GUARD_MS,
            });
            console.log(
              `CPR-VTK-PANO-WL run=${debugRunId} source=${wl.source} windowWidth=${wl.windowWidth.toFixed(
                2
              )} windowCenter=${wl.windowCenter.toFixed(2)}`
            );
            console.log('Centerline frames count:', frames.length);
            logCprDonePhase(debugRunId, 'before-pano-ui-hydrate', servicesManager, {
              viewportId: panoViewport.id,
              sourceVolumeId,
            });
            await hydratePanoViewportForCornerstoneUi(
              panoViewport,
              sourceVolumeId,
              wl.windowWidth,
              wl.windowCenter,
              preservedAxialDisplayState,
              debugRunId
            );
            console.log(
              `DIAG-TRIPWIRE: pano-ui-wakeup completed. Viewport has volume: ${panoViewport.hasVolumeId(
                sourceVolumeId
              )}`
            );
            logCprDonePhase(debugRunId, 'after-pano-ui-hydrate', servicesManager, {
              viewportHasSourceVolume: panoViewport.hasVolumeId(sourceVolumeId),
            });
            logVtkPanoViewportSnapshot(debugRunId, 'after-ui-hydrate', panoViewport);

            logCprDonePhase(debugRunId, 'before-hosted-pano-attach', servicesManager, {
              sourceVolumeId,
            });
            const attached = await attachVtkPanoCpr({
              viewport: panoViewport,
              sourceVolumeId,
              frames,
              verticalHalfMm: vtkVerticalHalfMm,
              slabHalfThicknessMm: vtkSlabHalfThicknessMm,
              slabSamples: vtkSlabSamples,
              projectionMode: vtkProjectionMode,
              initialWindowWidth: wl.windowWidth,
              initialWindowCenter: wl.windowCenter,
              runId: debugRunId,
            });
            attachedVtkPanoRef.current = attached;
            setViewportVisibility(panoViewport, true);
            logCprDonePhase(debugRunId, 'after-hosted-pano-attach', servicesManager, {
              actorUID: attached.actorUID,
            });
            setHostedPanoVoiAuthority({
              runId: debugRunId,
              sourceVolumeId,
              voi: initialPanoVoi,
              source: 'vtk-attached',
              suppressPanoViewportEventsForMs: CPR_VTK_PANO_STARTUP_VOI_GUARD_MS,
            });
            initializePanoCameraBridge(panoViewport, attached, debugRunId);

            logVtkPanoViewportSnapshot(debugRunId, 'after-attach', panoViewport);
            if (panoViewport.element) {
              cornerstoneTools.utilities.triggerAnnotationRender(panoViewport.element);
            }

            logCprDonePhase(debugRunId, 'before-crosssection-init', servicesManager, {
              sourceVolumeId,
            });
            await new Promise<void>((resolve, reject) => {
              window.setTimeout(() => {
                if (!isMountedRef.current) {
                  resolve();
                  return;
                }

                void initializeCrossSection(
                  frames,
                  sourceVolumeId,
                  vtkActualVerticalHalfMm,
                  vtkCrossSectionVerticalCenterOffsetMm,
                  vtkLocalCenterOffsetsMm,
                  false,
                  servicesManager
                )
                  .then(resolve)
                  .catch(reject);
              }, 50);
            });
            logCprDonePhase(debugRunId, 'after-crosssection-init', servicesManager, {
              sourceVolumeId,
            });
          } catch (innerErr) {
            const msg = innerErr instanceof Error ? innerErr.message : String(innerErr);
            console.error(`[CPR][${debugRunId}] VTK pano pipeline failed after HP switch:`, msg);
            clearPendingAxialViewportRestore('vtk-stage1-failed');
            setAxialTransitioning(false, 'vtk-stage1-failed');
            restoreAllCprViewportVisibility(servicesManager);
            logCprDonePhase(debugRunId, 'stage1-failed', servicesManager, {
              error: msg,
            });
            if (isMountedRef.current) {
              setError(msg);
            }
          } finally {
            markGenerationIdle();
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

        hpTimeoutRef.current = window.setTimeout(() => {
          clearProtocolListener();
          setAxialTransitioning(false, 'vtk-stage1-timeout');
          if (isMountedRef.current) {
            const timeoutMsg =
              '[CPR] Timed out waiting for hanging protocol change after stage switch.';
            console.error(`[CPR][${debugRunId}] ${timeoutMsg}`);
            setError(timeoutMsg);
          }
          markGenerationIdle();
        }, 12000);

        console.log(`[CPR][${debugRunId}] switching to CPR stage 1`);
        logCprDonePhase(debugRunId, 'request-stage1-switch', servicesManager, {
          sourceVolumeId,
        });
        commandsManager.runCommand('setHangingProtocol', {
          protocolId: 'cpr',
          stageIndex: 1,
        });
        return;
      }

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
        renderBackend: CPR_PANO_RECON_BACKEND_DEFAULT as CprPanoReconBackend,
        verticalDir,
        verticalHalfMm: baseVerticalHalfMm,
        verticalCenterOffsetMm: rigidSliceVerticalCenterOffsetMm,
        rigidVerticalSliceMode: rigidVerticalSliceModeEnabled,
        debugRunId,
      };
      const gpuBackendEnabled: boolean = workerInput.renderBackend !== 'cpu';
      console.log(
        `[CPR-PANO-PATH] run=${debugRunId} displayedPath=worker-recon defaultBackend=${workerInput.renderBackend} ` +
          `gpuBackend=${gpuBackendEnabled ? 'yes' : 'no'} ` +
          `defaultAggregation=${workerInput.aggregation} ` +
          `phase1VirtualPano=${gpuBackendEnabled ? 'skip-diagnostic-on-gpu-backend' : 'candidate'} ` +
          `phase2VirtualPano=candidate`
      );
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
        detailReward: number;
        qualityGatePassed: boolean;
        qualityGateRejectReasons: string[];
        qualityGateMetrics: Phase4QualityGateMetrics;
        supportSurfaceRiskSummary: Phase4SupportSurfaceRiskSummary;
        supportSurfaceBaselineFingerprint: string;
        supportSurfaceRiskLevel: Phase4SupportSurfaceRiskSummary['riskLevel'];
        supportSurfaceRiskFlags: string[];
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
        requestedBackend: CprPanoReconBackend;
        resolvedBackend: CprPanoReconBackend;
        pipelineMode: string;
        fallbackReason: string | null;
        reconstructionMode: string;
        workerTimingMs?: {
          adaptiveCenterSearch?: number;
          pass1And2TwoPassRender?: number;
          virtualPanoPhase12?: number;
          suppressionAndDenoise?: number;
          diagnosticAssembly?: number;
          total?: number;
        } | null;
        outputSignature: CprOutputSignature;
      }> => {
        launchedAttemptCount++;
        const attemptStartMs = performance.now();
        const requestedAggregation =
          (overrides.aggregation ?? workerInput.aggregation) === 'MEAN' ? 'MEAN' : 'MIP';
        const requestedRenderBackend =
          (overrides.renderBackend ?? workerInput.renderBackend) === 'cpu' ? 'cpu' : 'gpu';
        const requestedRigidVerticalSliceMode =
          overrides.rigidVerticalSliceMode === true ||
          (overrides.rigidVerticalSliceMode !== false && workerInput.rigidVerticalSliceMode === true);
        const requestedSlabHalfThicknessMm = toPositiveFinite(
          overrides.slabHalfThicknessMm,
          toPositiveFinite(workerInput.slabHalfThicknessMm, balancedSlabHalfThicknessMm)
        );
        const requestedSlabSamples = Math.max(
          1,
          Math.floor(
            toPositiveFinite(
              overrides.slabSamples,
              toPositiveFinite(workerInput.slabSamples, balancedSlabSamples)
            )
          )
        );
        const requestedVerticalCenterOffsetMm =
          toFiniteNumber(
            requestedRigidVerticalSliceMode
              ? overrides.verticalCenterOffsetMm ?? rigidSliceVerticalCenterOffsetMm
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
        console.debug(
          '[CPR-GPU-REQUEST-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            renderBackend: requestedRenderBackend,
            aggregation: requestedAggregation,
            panoWidth: finalPanoWidth,
            panoHeight: finalPanoHeight,
            slabHalfThicknessMm: requestedSlabHalfThicknessMm,
            slabSamples: requestedSlabSamples,
            verticalHalfMm: actualVertHalfMm,
            verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
            rigidVerticalSliceMode: requestedRigidVerticalSliceMode,
          })
        );
        const rawResult = await launchCPRWorker({
          ...workerInput,
          ...overrides,
          attemptLabel: label,
          panoWidth: finalPanoWidth,
          panoHeight: finalPanoHeight,
          renderBackend: requestedRenderBackend,
          slabHalfThicknessMm: requestedSlabHalfThicknessMm,
          slabSamples: requestedSlabSamples,
          verticalHalfMm: actualVertHalfMm,
          verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
          rigidVerticalSliceMode: requestedRigidVerticalSliceMode,
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
        const summaryPixelData = reconstructPanoFloatBuffer(
          result.pixelData,
          result.slope,
          result.intercept
        );
        const workerDiagnosticForSummary =
          result.workerDebugPayload &&
          typeof result.workerDebugPayload === 'object' &&
          result.workerDebugPayload.diagnostic &&
          typeof result.workerDebugPayload.diagnostic === 'object'
            ? (result.workerDebugPayload.diagnostic as Record<string, unknown>)
            : null;
        const summaryAnalysisCenterRow =
          readPhase4DiagnosticNumber(
            readPhase4DiagnosticRecord(workerDiagnosticForSummary?.virtualPanoRender)?.analysisCenterRow
          ) ??
          readPhase4DiagnosticNumber(
            readPhase4DiagnosticRecord(workerDiagnosticForSummary?.virtualPanoPhase12)?.analysisCenterRow
          );
        const summary = summarizeFloatBufferForDebug(
          summaryPixelData,
          finalPanoWidth,
          finalPanoHeight,
          result.debugMaps,
          summaryAnalysisCenterRow ?? undefined
        );
        const supportSurfaceMetrics = extractPhase4QualityGateMetrics(summary, result.workerDebugPayload);
        const renderSupportMode = supportSurfaceMetrics.renderSupportMode;
        const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);
        const workerVoi = isDualArchProjection
          ? null
          : createPanoVoiFromWindowLevel(result.windowWidth, result.windowCenter);
        const adaptiveVoi = computeAdaptivePanoVoi(
          summary,
          result.minValue,
          result.maxValue,
          renderSupportMode
        );
        const voi = workerVoi ?? adaptiveVoi;
        const voiSource = isDualArchProjection
          ? 'dual-arch-black-clamp'
          : workerVoi
            ? 'worker-window-passthrough'
            : 'adaptive-fallback';
        console.debug(
          '[CPR-GPU-VOI-APPLY-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            renderBackend: requestedRenderBackend,
            slabHalfThicknessMm: requestedSlabHalfThicknessMm,
            slabSamples: requestedSlabSamples,
            workerWindowWidth: result.windowWidth,
            workerWindowCenter: result.windowCenter,
            appliedWindowWidth: voi.windowWidth,
            appliedWindowCenter: voi.windowCenter,
            appliedLower: voi.lower,
            appliedUpper: voi.upper,
            minValue: result.minValue,
            maxValue: result.maxValue,
            lowerClampHu: isDualArchProjection ? CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU : null,
            voiSource,
          })
        );
        console.debug(
          `[CPR-PANO-VOI] run=${debugRunId} label=${label} backend=${requestedRenderBackend} ` +
            `source=${voiSource} ` +
            `workerWW=${formatReadablePanoValue(result.windowWidth)} ` +
            `workerWC=${formatReadablePanoValue(result.windowCenter)} ` +
            `appliedWW=${formatReadablePanoValue(voi.windowWidth)} ` +
            `appliedWC=${formatReadablePanoValue(voi.windowCenter)} ` +
            `appliedLower=${formatReadablePanoValue(voi.lower)} ` +
            `appliedUpper=${formatReadablePanoValue(voi.upper)} ` +
            `lowerClamp=${isDualArchProjection ? CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU : 'none'} ` +
            `min=${formatReadablePanoValue(result.minValue)} ` +
            `max=${formatReadablePanoValue(result.maxValue)}`
        );
        const qualityBase = summary ? scorePanoQuality(summary, renderSupportMode) : 0;
        const workerDiagnostic =
          result.workerDebugPayload &&
          typeof result.workerDebugPayload === 'object' &&
          result.workerDebugPayload.diagnostic &&
          typeof result.workerDebugPayload.diagnostic === 'object'
            ? (result.workerDebugPayload.diagnostic as Record<string, unknown>)
            : null;
        const workerGpuRenderDiagnostic =
          workerDiagnostic?.gpuRender && typeof workerDiagnostic.gpuRender === 'object'
            ? (workerDiagnostic.gpuRender as Record<string, unknown>)
            : null;
        const routeDiagnostic = resolveWorkerDisplayRouteDiagnostic(result.workerDebugPayload);
        const resolvedReconstructionMode =
          toNonEmptyString(workerDiagnostic?.reconstructionMode) ||
          toNonEmptyString(workerDiagnostic?.pipelineMode) ||
          routeDiagnostic.pipelineMode;
        const phase2GatePassed =
          typeof workerGpuRenderDiagnostic?.phase2GatePassed === 'boolean'
            ? Boolean(workerGpuRenderDiagnostic.phase2GatePassed)
            : typeof workerDiagnostic?.phase2GatePassed === 'boolean'
              ? Boolean(workerDiagnostic.phase2GatePassed)
              : null;
        const actualVerticalCenterOffsetMm = toFiniteNumber(
          workerDiagnostic?.verticalCenterOffsetMm
        );
        const baseVerticalCenterOffsetMm = toFiniteNumber(
          workerDiagnostic?.baseVerticalCenterOffsetMm
        );
        const fittedVerticalCenterOffsetMm = toFiniteNumber(
          workerDiagnostic?.fittedVerticalCenterOffsetMm ??
            workerDiagnostic?.globalVerticalCenterOffsetMm
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
          Math.max(
            0,
            requestedSlabHalfThicknessMm - (requestedAggregation === 'MIP' ? 0.8 : 1.8)
          ) * (requestedAggregation === 'MIP' ? 5 : 1.4);
        const lowerBandFillPenalty = summary
          ? isDualArchProjection
            ? Math.max(0, summary.lowerBandP50 - 220) / 55 +
              Math.max(0, summary.lowerBandBrightFraction - 0.72) * 10 +
              Math.max(0, summary.lowerBandBrightFraction - 0.88) * 40
            : Math.max(0, summary.lowerBandP50 + 140) / 38 +
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
          ? Math.max(0, actualVertHalfMm - 42) *
            Math.max(0, summary.lowerBandBrightFraction - (isDualArchProjection ? 0.82 : 0.45)) *
            (isDualArchProjection ? 10 : 28)
          : 0;
        const noAirPenalty = summary
          ? summary.fractionBelowMinus950 < (isDualArchProjection ? 0.001 : 0.005)
            ? 2
            : summary.fractionBelowMinus950 < (isDualArchProjection ? 0.003 : 0.015)
              ? 1
              : 0
          : 0;
        const elevatedP01Penalty = summary ? Math.max(0, summary.p01 + 780) / 80 : 0;
        const centerDriftPenalty =
          Math.max(0, actualCenterDriftMm - 1.5) * 6 + Math.max(0, baseCenterDriftMm - 1.0) * 8;
        const blackClipPenalty =
          supportSurfaceMetrics.blackClipFraction !== null
            ? Math.max(0, supportSurfaceMetrics.blackClipFraction - 0.42) * 18 +
              Math.max(0, supportSurfaceMetrics.blackClipFraction - 0.52) * 26
            : 0;
        const supportConfidencePenalty =
          requestedRenderBackend === 'gpu' && routeDiagnostic.backend === 'gpu'
            ? (supportSurfaceMetrics.supportConfidenceP50 !== null
                ? Math.max(0, 0.16 - supportSurfaceMetrics.supportConfidenceP50) * 18
                : 0) +
              (supportSurfaceMetrics.supportPathConfidenceP50 !== null
                ? Math.max(0, 0.2 - supportSurfaceMetrics.supportPathConfidenceP50) * 10
                : 0)
            : 0;
        const supportContinuityPenalty =
          requestedRenderBackend === 'gpu' && routeDiagnostic.backend === 'gpu'
            ? (supportSurfaceMetrics.supportUnstableColumnFraction !== null
                ? Math.max(0, supportSurfaceMetrics.supportUnstableColumnFraction - 0.18) * 14
                : 0) +
              (supportSurfaceMetrics.supportAmbiguousColumnFraction !== null
                ? Math.max(0, supportSurfaceMetrics.supportAmbiguousColumnFraction - 0.18) * 10
                : 0) +
              (supportSurfaceMetrics.supportBestDepthDriftP95Mm !== null
                ? Math.max(0, supportSurfaceMetrics.supportBestDepthDriftP95Mm - 0.9) * 4
                : 0)
            : 0;
        const backgroundLeakagePenalty = summary
          ? Math.max(0, (summary.backgroundOutlierFraction05 ?? 0) - 0.16) * 20 +
            Math.max(0, (summary.backgroundOutlierFraction10 ?? 0) - 0.07) * 26
          : 0;
        const aggregationPenalty =
          requestedAggregation === 'MIP'
            ? 4.5 + (summary ? Math.max(0, summary.meanAbsDelta - 300) / 45 : 0)
            : 0;
        const phase2GatePenalty =
          requestedRenderBackend === 'gpu' &&
          routeDiagnostic.backend === 'gpu' &&
          phase2GatePassed === false
            ? 30
            : 0;
        const baseHardRejectReason = getHardRejectReason(summary, renderSupportMode);
        const excessiveCenterDrift =
          actualCenterDriftMm > Math.max(4, actualVertHalfMm * 0.2) ||
          baseCenterDriftMm > Math.max(3.5, actualVertHalfMm * 0.16);
        const unstableSupportSurface =
          requestedRenderBackend === 'gpu' &&
          routeDiagnostic.backend === 'gpu' &&
          requestedAggregation === 'MEAN' &&
          ((supportSurfaceMetrics.supportDepthStdMm !== null &&
            supportSurfaceMetrics.supportDepthStdMm > 0.65) ||
            (supportSurfaceMetrics.pathJumpP95Mm !== null &&
              supportSurfaceMetrics.pathJumpP95Mm > 1.4) ||
            (supportSurfaceMetrics.supportUnstableColumnFraction !== null &&
              supportSurfaceMetrics.supportUnstableColumnFraction > 0.22) ||
            (supportSurfaceMetrics.supportLongestUnstableRunColumns !== null &&
              supportSurfaceMetrics.supportLongestUnstableRunColumns > 42) ||
            ((supportSurfaceMetrics.supportBestDepthDriftP95Mm !== null &&
              supportSurfaceMetrics.supportBestDepthDriftP95Mm > 1.1) &&
              (supportSurfaceMetrics.supportAmbiguousColumnFraction !== null &&
                supportSurfaceMetrics.supportAmbiguousColumnFraction > 0.18)));
        const unstableVirtualPanoSupportSurface =
          resolvedReconstructionMode === 'virtualPano' &&
          routeDiagnostic.backend === 'cpu' &&
          ((supportSurfaceMetrics.supportDepthStdMm !== null &&
            supportSurfaceMetrics.supportDepthStdMm > 1.3) ||
            (supportSurfaceMetrics.pathJumpP95Mm !== null &&
              supportSurfaceMetrics.pathJumpP95Mm > 0.82) ||
            (supportSurfaceMetrics.supportUnstableColumnFraction !== null &&
              supportSurfaceMetrics.supportUnstableColumnFraction > 0.24) ||
            (supportSurfaceMetrics.supportLongestUnstableRunColumns !== null &&
              supportSurfaceMetrics.supportLongestUnstableRunColumns > 14) ||
            ((supportSurfaceMetrics.supportAmbiguousColumnFraction !== null &&
              supportSurfaceMetrics.supportAmbiguousColumnFraction > 0.4) &&
              (supportSurfaceMetrics.supportScoreGapP50 !== null &&
                supportSurfaceMetrics.supportScoreGapP50 < 0.08)) ||
            (supportSurfaceMetrics.supportForcedDriftFraction !== null &&
              supportSurfaceMetrics.supportForcedDriftFraction > 0.12) ||
            (supportSurfaceMetrics.supportBestDepthDriftP95Mm !== null &&
              supportSurfaceMetrics.supportBestDepthDriftP95Mm > 1.8));
        const hardRejectReason =
          requestedRenderBackend === 'gpu' &&
          routeDiagnostic.backend === 'gpu' &&
          phase2GatePassed === false
            ? 'gpu-phase2-gate-failed'
            : !baseHardRejectReason && unstableSupportSurface
              ? 'support-surface-instability'
            : !baseHardRejectReason && unstableVirtualPanoSupportSurface
              ? 'virtual-pano-support-instability'
            : !baseHardRejectReason && excessiveCenterDrift
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
          blackClipPenalty -
          supportConfidencePenalty -
          supportContinuityPenalty -
          backgroundLeakagePenalty -
          deformationPenalty -
          tallFillPenalty -
          intensityDomainPenalty -
          aggregationPenalty -
          phase2GatePenalty -
          hardRejectPenalty;
        const qualityGateCandidate = buildPhase4QualityGateCandidate({
          attemptLabel: label,
          displayedPath: 'worker-recon',
          sourceVolumeId,
          summary,
          qualityBase,
          qualityScore,
          hardRejectReason,
          workerDebugPayload: result.workerDebugPayload ?? null,
        });
        const qualityGatePassed = qualityGateCandidate.pass;
        const qualityGateRejectReasons = qualityGateCandidate.rejectReasons;
        const attemptDurationMs = performance.now() - attemptStartMs;
        const workerTimingMs =
          result.workerDebugPayload &&
          typeof result.workerDebugPayload === 'object' &&
          result.workerDebugPayload.diagnostic &&
          typeof result.workerDebugPayload.diagnostic === 'object' &&
          (result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs &&
          typeof (result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs ===
            'object'
            ? ((result.workerDebugPayload.diagnostic as Record<string, unknown>).timingMs as {
                adaptiveCenterSearch?: number;
                pass1And2TwoPassRender?: number;
                virtualPanoPhase12?: number;
                suppressionAndDenoise?: number;
                diagnosticAssembly?: number;
                total?: number;
              })
            : null;
        const outputSignature = extractCprOutputSignature(result.workerDebugPayload);

        console.debug(`[CPR][${debugRunId}] pano attempt ${label}`, {
          durationMs: Math.round(attemptDurationMs),
          workerTimingMs,
          outputSignature,
          qualityBase,
          qualityScore,
          qualityGatePassed,
          qualityGateRejectReasons,
          supportSurfaceRiskSummary: qualityGateCandidate.supportSurfaceRiskSummary,
          supportSurfaceBaselineFingerprint:
            qualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
          aggregation: requestedAggregation,
          requestedBackend: requestedRenderBackend,
          resolvedBackend: routeDiagnostic.backend,
          pipelineMode: routeDiagnostic.pipelineMode,
          reconstructionMode: resolvedReconstructionMode,
          fallbackReason: routeDiagnostic.fallbackReason,
          minValue: result.minValue,
          maxValue: result.maxValue,
          p01: summary?.p01,
          p50: summary?.p50,
          p99: summary?.p99,
          meanAbsDelta: summary?.meanAbsDelta,
          lowerBandP50: summary?.lowerBandP50,
          lowerBandBrightFraction: summary?.lowerBandBrightFraction,
          backgroundToneSampleCount: summary?.backgroundToneSampleCount,
          backgroundToneP95: summary?.backgroundToneP95,
          backgroundToneP99: summary?.backgroundToneP99,
          backgroundToneMax: summary?.backgroundToneMax,
          backgroundOutlierFraction05: summary?.backgroundOutlierFraction05,
          backgroundOutlierFraction10: summary?.backgroundOutlierFraction10,
          supportConfidenceP50: supportSurfaceMetrics.supportConfidenceP50,
          supportDepthStdMm: supportSurfaceMetrics.supportDepthStdMm,
          pathJumpP95Mm: supportSurfaceMetrics.pathJumpP95Mm,
          supportPathConfidenceP50: supportSurfaceMetrics.supportPathConfidenceP50,
          supportUnstableColumnFraction: supportSurfaceMetrics.supportUnstableColumnFraction,
          supportLongestUnstableRunColumns: supportSurfaceMetrics.supportLongestUnstableRunColumns,
          supportAmbiguousColumnFraction: supportSurfaceMetrics.supportAmbiguousColumnFraction,
          supportForcedDriftFraction: supportSurfaceMetrics.supportForcedDriftFraction,
          supportBestDepthDriftP95Mm: supportSurfaceMetrics.supportBestDepthDriftP95Mm,
          supportScoreGapP50: supportSurfaceMetrics.supportScoreGapP50,
          blackClipFraction: supportSurfaceMetrics.blackClipFraction,
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
          phase2GatePassed,
          phase2GatePenalty,
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
          blackClipPenalty,
          supportConfidencePenalty,
          supportContinuityPenalty,
          backgroundLeakagePenalty,
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
        console.debug(
          '[CPR-ATTEMPT-SIGNATURE-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            requestedBackend: requestedRenderBackend,
            resolvedBackend: routeDiagnostic.backend,
            pipelineMode: routeDiagnostic.pipelineMode,
            reconstructionMode: resolvedReconstructionMode,
            aggregation: requestedAggregation,
            verticalHalfMm: actualVertHalfMm,
            verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
            actualVerticalCenterOffsetMm,
            slabHalfThicknessMm: requestedSlabHalfThicknessMm,
            slabSamples: requestedSlabSamples,
            modalityLutApplied: result.modalityLutApplied,
            requestedModalityLutApplied: result.requestedModalityLutApplied,
            storedValueNormalizationApplied: result.storedValueNormalizationApplied,
            effectiveIsPreScaled: result.effectiveIsPreScaled,
            outputSignature,
          })
        );
        console.debug(
          `[CPR-PANO-ATTEMPT] run=${debugRunId} label=${label} requestedBackend=${requestedRenderBackend} ` +
            `resolvedBackend=${routeDiagnostic.backend} pipelineMode=${routeDiagnostic.pipelineMode} ` +
            `reconstructionMode=${resolvedReconstructionMode} ` +
            `fallbackReason=${routeDiagnostic.fallbackReason ?? 'none'} ` +
            `aggregation=${requestedAggregation} score=${formatReadablePanoValue(qualityScore, 2)} ` +
            `base=${formatReadablePanoValue(qualityBase, 2)} hardReject=${hardRejectReason ?? 'none'} ` +
            `risk=${qualityGateCandidate.supportSurfaceRiskSummary.riskLevel} ` +
            `riskFlags=${qualityGateCandidate.supportSurfaceRiskSummary.riskFlags.join(',') || 'none'} ` +
            `durationMs=${formatReadablePanoValue(attemptDurationMs, 0)} ` +
            `p50=${formatReadablePanoValue(summary?.p50)} ` +
            `meanAbsDelta=${formatReadablePanoValue(summary?.meanAbsDelta)} ` +
            `lowerBandP50=${formatReadablePanoValue(summary?.lowerBandP50)} ` +
            `lowerBandBrightPct=${formatReadablePanoValue(
              summary ? summary.lowerBandBrightFraction * 100 : undefined
            )} ` +
            `bgToneP99=${formatReadablePanoValue(summary?.backgroundToneP99, 3)} ` +
            `bgBrightPct05=${formatReadablePanoValue(
              summary ? summary.backgroundOutlierFraction05 * 100 : undefined,
              1
            )} ` +
            `bgBrightPct10=${formatReadablePanoValue(
              summary ? summary.backgroundOutlierFraction10 * 100 : undefined,
              1
            )} ` +
            `blackClip=${formatReadablePanoValue(supportSurfaceMetrics.blackClipFraction, 3)} ` +
            `supportP50=${formatReadablePanoValue(
              supportSurfaceMetrics.supportConfidenceP50,
              3
            )} ` +
            `supportPathConf=${formatReadablePanoValue(
              supportSurfaceMetrics.supportPathConfidenceP50,
              3
            )} ` +
            `unstableColsPct=${formatReadablePanoValue(
              supportSurfaceMetrics.supportUnstableColumnFraction !== null
                ? supportSurfaceMetrics.supportUnstableColumnFraction * 100
                : undefined,
              1
            )} ` +
            `ambiguousColsPct=${formatReadablePanoValue(
              supportSurfaceMetrics.supportAmbiguousColumnFraction !== null
                ? supportSurfaceMetrics.supportAmbiguousColumnFraction * 100
                : undefined,
              1
            )} ` +
            `driftP95Mm=${formatReadablePanoValue(
              supportSurfaceMetrics.supportBestDepthDriftP95Mm,
              2
            )} ` +
            `gpuRiskPenalty=${formatReadablePanoValue(
              supportConfidencePenalty + supportContinuityPenalty + backgroundLeakagePenalty,
              2
            )} ` +
            `toothMean=${formatReadablePanoValue(summary?.toothBandMean)} ` +
            `detailRatio=${formatReadablePanoValue(detailBalanceRatio, 2)} ` +
            `centerDriftMm=${formatReadablePanoValue(actualCenterDriftMm, 2)} ` +
            `voi=adaptive`
        );
        console.debug(
          '[CPR-ATTEMPT-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            qualityBase,
            qualityScore,
            qualityGatePassed,
            qualityGateRejectReasons,
            supportSurfaceRiskSummary: qualityGateCandidate.supportSurfaceRiskSummary,
            supportSurfaceBaselineFingerprint:
              qualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
            requestedBackend: requestedRenderBackend,
            resolvedBackend: routeDiagnostic.backend,
            pipelineMode: routeDiagnostic.pipelineMode,
            reconstructionMode: resolvedReconstructionMode,
            fallbackReason: routeDiagnostic.fallbackReason,
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
            phase2GatePassed,
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
            backgroundToneSampleCount: summary?.backgroundToneSampleCount ?? null,
            backgroundToneP95: summary?.backgroundToneP95 ?? null,
            backgroundToneP99: summary?.backgroundToneP99 ?? null,
            backgroundToneMax: summary?.backgroundToneMax ?? null,
            backgroundOutlierFraction05: summary?.backgroundOutlierFraction05 ?? null,
            backgroundOutlierFraction10: summary?.backgroundOutlierFraction10 ?? null,
            detailBandHorizontalEdgeMean: summary?.detailBandHorizontalEdgeMean ?? null,
            detailBandVerticalEdgeMean: summary?.detailBandVerticalEdgeMean ?? null,
            supportConfidenceP50: supportSurfaceMetrics.supportConfidenceP50,
            supportPathConfidenceP10: supportSurfaceMetrics.supportPathConfidenceP10,
            supportPathConfidenceP50: supportSurfaceMetrics.supportPathConfidenceP50,
            supportPathConfidenceP90: supportSurfaceMetrics.supportPathConfidenceP90,
            supportUnstableColumnFraction: supportSurfaceMetrics.supportUnstableColumnFraction,
            supportLongestUnstableRunColumns: supportSurfaceMetrics.supportLongestUnstableRunColumns,
            supportAmbiguousColumnFraction: supportSurfaceMetrics.supportAmbiguousColumnFraction,
            supportForcedDriftFraction: supportSurfaceMetrics.supportForcedDriftFraction,
            supportBestDepthDriftP95Mm: supportSurfaceMetrics.supportBestDepthDriftP95Mm,
            supportScoreGapP50: supportSurfaceMetrics.supportScoreGapP50,
            blackClipFraction: supportSurfaceMetrics.blackClipFraction,
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
            blackClipPenalty,
            supportConfidencePenalty,
            supportContinuityPenalty,
            backgroundLeakagePenalty,
            phase2GatePenalty,
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
        console.log(
          '[CPR-PHASE0-BASELINE-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label,
            riskLevel: qualityGateCandidate.supportSurfaceRiskSummary.riskLevel,
            riskScore: qualityGateCandidate.supportSurfaceRiskSummary.riskScore,
            riskFlags: qualityGateCandidate.supportSurfaceRiskSummary.riskFlags,
            baselineFingerprint:
              qualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
            metrics: qualityGateCandidate.supportSurfaceRiskSummary.stability,
            background: qualityGateCandidate.supportSurfaceRiskSummary.background,
            anatomy: qualityGateCandidate.supportSurfaceRiskSummary.anatomy,
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
          detailReward,
          qualityGatePassed,
          qualityGateRejectReasons,
          qualityGateMetrics: supportSurfaceMetrics,
          supportSurfaceRiskSummary: qualityGateCandidate.supportSurfaceRiskSummary,
          supportSurfaceBaselineFingerprint:
            qualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
          supportSurfaceRiskLevel: qualityGateCandidate.supportSurfaceRiskSummary.riskLevel,
          supportSurfaceRiskFlags: qualityGateCandidate.supportSurfaceRiskSummary.riskFlags,
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
          requestedBackend: requestedRenderBackend,
          resolvedBackend: routeDiagnostic.backend,
          pipelineMode: routeDiagnostic.pipelineMode,
          fallbackReason: routeDiagnostic.fallbackReason,
          reconstructionMode: resolvedReconstructionMode,
          workerTimingMs,
          outputSignature,
        };
      };

      const attemptAudit: Array<{
        label: string;
        qualityGatePassed: boolean;
        qualityGateRejectReasons: string[];
        supportSurfaceRiskSummary: Phase4SupportSurfaceRiskSummary;
        supportSurfaceBaselineFingerprint: string;
        supportSurfaceRiskLevel: Phase4SupportSurfaceRiskSummary['riskLevel'];
        supportSurfaceRiskFlags: string[];
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        requestedBackend: CprPanoReconBackend;
        resolvedBackend: CprPanoReconBackend;
        pipelineMode: string;
        fallbackReason: string | null;
        reconstructionMode: string;
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
        backgroundToneSampleCount: number | null;
        backgroundToneP95: number | null;
        backgroundToneP99: number | null;
        backgroundToneMax: number | null;
        backgroundOutlierFraction05: number | null;
        backgroundOutlierFraction10: number | null;
        detailBandHorizontalEdgeMean: number | null;
        detailBandVerticalEdgeMean: number | null;
        fractionBelowMinus950: number | null;
        fractionAbove3000: number | null;
        durationMs: number;
        outputSignature: CprOutputSignature;
        duplicateOfLabel?: string | null;
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
      const outputSignatureSeenByKey = new Map<string, string>();
      const recordAttemptAudit = (attempt: {
        label: string;
        qualityGatePassed: boolean;
        qualityGateRejectReasons: string[];
        supportSurfaceRiskSummary: Phase4SupportSurfaceRiskSummary;
        supportSurfaceBaselineFingerprint: string;
        supportSurfaceRiskLevel: Phase4SupportSurfaceRiskSummary['riskLevel'];
        supportSurfaceRiskFlags: string[];
        qualityBase: number;
        qualityScore: number;
        hardRejectReason: string | null;
        requestedBackend: CprPanoReconBackend;
        resolvedBackend: CprPanoReconBackend;
        pipelineMode: string;
        fallbackReason: string | null;
        reconstructionMode: string;
        aggregation: 'MIP' | 'MEAN';
        slabHalfThicknessMm: number;
        slabSamples: number;
        verticalCenterOffsetMm: number;
        intensityDomain: SyntheticCprIntensityDomain;
        huDomain: boolean;
        summary: FloatBufferDebugSummary | null;
        durationMs: number;
        outputSignature: CprOutputSignature;
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
          qualityGatePassed: attempt.qualityGatePassed,
          qualityGateRejectReasons: attempt.qualityGateRejectReasons,
          supportSurfaceRiskSummary: attempt.supportSurfaceRiskSummary,
          supportSurfaceBaselineFingerprint: attempt.supportSurfaceRiskSummary.baselineFingerprint,
          supportSurfaceRiskLevel: attempt.supportSurfaceRiskSummary.riskLevel,
          supportSurfaceRiskFlags: attempt.supportSurfaceRiskSummary.riskFlags,
          qualityBase: attempt.qualityBase,
          qualityScore: attempt.qualityScore,
          hardRejectReason: attempt.hardRejectReason,
          requestedBackend: attempt.requestedBackend,
          resolvedBackend: attempt.resolvedBackend,
          pipelineMode: attempt.pipelineMode,
          fallbackReason: attempt.fallbackReason,
          reconstructionMode: attempt.reconstructionMode,
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
          backgroundToneSampleCount: attempt.summary?.backgroundToneSampleCount ?? null,
          backgroundToneP95: attempt.summary?.backgroundToneP95 ?? null,
          backgroundToneP99: attempt.summary?.backgroundToneP99 ?? null,
          backgroundToneMax: attempt.summary?.backgroundToneMax ?? null,
          backgroundOutlierFraction05: attempt.summary?.backgroundOutlierFraction05 ?? null,
          backgroundOutlierFraction10: attempt.summary?.backgroundOutlierFraction10 ?? null,
          detailBandHorizontalEdgeMean: attempt.summary?.detailBandHorizontalEdgeMean ?? null,
          detailBandVerticalEdgeMean: attempt.summary?.detailBandVerticalEdgeMean ?? null,
          fractionBelowMinus950: attempt.summary?.fractionBelowMinus950 ?? null,
          fractionAbove3000: attempt.summary?.fractionAbove3000 ?? null,
          durationMs: Math.round(attempt.durationMs),
          outputSignature: attempt.outputSignature,
          duplicateOfLabel: null,
          workerTimingMs: attempt.workerTimingMs ?? null,
        });
      };
      const recordAttempt = (attempt: Awaited<ReturnType<typeof runWorkerAttempt>>): void => {
        attemptResults.push(attempt);
        recordAttemptAudit(attempt);
        const latestAuditEntry = attemptAudit[attemptAudit.length - 1];
        const signatureKey = buildCprOutputSignatureKey(attempt.outputSignature);
        if (!latestAuditEntry || !signatureKey) {
          return;
        }
        const duplicateOfLabel = outputSignatureSeenByKey.get(signatureKey) ?? null;
        latestAuditEntry.duplicateOfLabel = duplicateOfLabel;
        if (!duplicateOfLabel) {
          outputSignatureSeenByKey.set(signatureKey, attempt.label);
          return;
        }

        console.warn(
          `[CPR][${debugRunId}] Attempt "${attempt.label}" produced the same output signature as "${duplicateOfLabel}".`,
          {
            label: attempt.label,
            duplicateOfLabel,
            outputSignature: attempt.outputSignature,
          }
        );
        console.debug(
          '[CPR-ATTEMPT-DUPLICATE-SIGNATURE-JSON]',
          JSON.stringify({
            runId: debugRunId,
            label: attempt.label,
            duplicateOfLabel,
            outputSignature: attempt.outputSignature,
          })
        );
      };
      const shouldPromoteAttempt = (
        candidate: Awaited<ReturnType<typeof runWorkerAttempt>>,
        currentBest: Awaited<ReturnType<typeof runWorkerAttempt>>
      ): boolean => {
        return compareRankedPanoOutputs(candidate, currentBest) < 0;
      };
      const selectPreferredMeanAttempt = (): Awaited<ReturnType<typeof runWorkerAttempt>> | null => {
        const meanAttempts = attemptResults.filter(attempt => attempt.aggregation === 'MEAN');
        if (!meanAttempts.length) {
          return null;
        }

        let selected = meanAttempts[0];
        for (let index = 1; index < meanAttempts.length; index++) {
          if (shouldPromoteAttempt(meanAttempts[index], selected)) {
            selected = meanAttempts[index];
          }
        }
        return selected;
      };
      const selectPreferredMeanAttemptForPhase2 = (): Awaited<
        ReturnType<typeof runWorkerAttempt>
      > | null => {
        const meanAttempts = attemptResults.filter(attempt => attempt.aggregation === 'MEAN');
        if (!meanAttempts.length) {
          return null;
        }

        const byLabel = new Map(meanAttempts.map(attempt => [attempt.label, attempt] as const));
        for (const label of CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS) {
          const matched = byLabel.get(label);
          if (matched) {
            return matched;
          }
        }

        return selectPreferredMeanAttempt();
      };
      type EvaluatedPanoAttempt = Awaited<ReturnType<typeof runWorkerAttempt>>;
      const collectAttemptEscalationReasons = (attempt: EvaluatedPanoAttempt): string[] => {
        const reasons: string[] = [];
        if (attempt.hardRejectReason) {
          reasons.push(`hard-reject:${attempt.hardRejectReason}`);
        }
        if (!attempt.qualityGatePassed) {
          if (attempt.qualityGateRejectReasons.length > 0) {
            reasons.push(
              ...attempt.qualityGateRejectReasons.map(reason => `quality-gate:${reason}`)
            );
          } else {
            reasons.push('quality-gate:failed');
          }
        }
        if (attempt.requestedBackend === 'gpu' && attempt.resolvedBackend === 'gpu') {
          if (
            attempt.supportSurfaceRiskLevel === 'elevated' ||
            attempt.supportSurfaceRiskLevel === 'high'
          ) {
            reasons.push(`support-risk:${attempt.supportSurfaceRiskLevel}`);
          }
          const blockingRiskFlags = new Set([
            'support-confidence-low',
            'support-path-confidence-low',
            'support-columns-unstable',
            'support-ambiguity-high',
            'support-forced-drift',
            'background-leakage',
          ]);
          for (const riskFlag of attempt.supportSurfaceRiskFlags) {
            if (blockingRiskFlags.has(riskFlag)) {
              reasons.push(`risk-flag:${riskFlag}`);
            }
          }
          if ((attempt.summary?.backgroundOutlierFraction05 ?? 0) > 0.2) {
            reasons.push('background-outlier-fraction05-high');
          }
          if ((attempt.summary?.backgroundOutlierFraction10 ?? 0) > 0.08) {
            reasons.push('background-outlier-fraction10-high');
          }
        }
        return Array.from(new Set(reasons));
      };
      const logAttemptEscalation = (
        decisionStage: string,
        attempt: EvaluatedPanoAttempt,
        reasons: string[]
      ): void => {
        if (reasons.length === 0) {
          return;
        }
        console.debug(
          '[CPR-RETRY-ESCALATION-JSON]',
          JSON.stringify({
            runId: debugRunId,
            decisionStage,
            attemptLabel: attempt.label,
            requestedBackend: attempt.requestedBackend,
            resolvedBackend: attempt.resolvedBackend,
            reconstructionMode: attempt.reconstructionMode,
            aggregation: attempt.aggregation,
            qualityScore: attempt.qualityScore,
            qualityGatePassed: attempt.qualityGatePassed,
            qualityGateRejectReasons: attempt.qualityGateRejectReasons,
            supportRiskLevel: attempt.supportSurfaceRiskLevel,
            supportRiskFlags: attempt.supportSurfaceRiskFlags,
            hardRejectReason: attempt.hardRejectReason,
            reasons,
          })
        );
      };
      const isGoodEnoughPanoAttempt = (attempt: EvaluatedPanoAttempt): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return false;
        }
        if (collectAttemptEscalationReasons(attempt).length > 0) {
          return false;
        }
        const isDualArchProjection = isDualArchProjectionRenderMode(
          attempt.qualityGateMetrics.renderSupportMode
        );
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        const hasCleanLowerBand = isDualArchProjection
          ? summary.lowerBandBrightFraction <= 0.8 && summary.lowerBandP50 <= 260
          : summary.lowerBandBrightFraction <= 0.32 && summary.lowerBandP50 <= -120;
        const hasCleanToothBand =
          summary.toothBandBrightFraction <= (isDualArchProjection ? 0.32 : 0.26) &&
          summary.toothBandP10 <= (isDualArchProjection ? 220 : 140) &&
          summary.toothBandMean <= (isDualArchProjection ? 980 : 820);
        const hasVeryStrongOverallScore =
          attempt.qualityScore >= 24 &&
          summary.lowerBandBrightFraction <= (isDualArchProjection ? 0.84 : 0.4) &&
          summary.lowerBandP50 <= (isDualArchProjection ? 300 : -60) &&
          hasCleanToothBand &&
          detailRatio <= 3.4;
        return (
          hasVeryStrongOverallScore ||
          (attempt.qualityScore >= 16 &&
            hasCleanLowerBand &&
            hasCleanToothBand &&
            summary.meanAbsDelta <= 520 &&
            summary.fractionAbove3000 <= 0.005 &&
            detailRatio <= 3.4)
        );
      };
      const needsFallbackAttempt = (attempt: EvaluatedPanoAttempt): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return true;
        }
        if (collectAttemptEscalationReasons(attempt).length > 0) {
          return true;
        }
        const isDualArchProjection = isDualArchProjectionRenderMode(
          attempt.qualityGateMetrics.renderSupportMode
        );
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        return (
          attempt.qualityScore < 2 ||
          summary.lowerBandBrightFraction > (isDualArchProjection ? 0.9 : 0.62) ||
          summary.lowerBandP50 > (isDualArchProjection ? 420 : 120) ||
          summary.meanAbsDelta > 680 ||
          detailRatio > 4.2
        );
      };
      const needsExtendedMeanRetrySearch = (attempt: EvaluatedPanoAttempt): boolean => {
        const summary = attempt.summary;
        if (!summary || !!attempt.hardRejectReason) {
          return true;
        }
        if (collectAttemptEscalationReasons(attempt).length > 0) {
          return true;
        }
        const isDualArchProjection = isDualArchProjectionRenderMode(
          attempt.qualityGateMetrics.renderSupportMode
        );
        const detailRatio =
          summary.detailBandHorizontalEdgeMean / Math.max(1, summary.detailBandVerticalEdgeMean);
        return (
          attempt.qualityScore < 10 ||
          summary.toothBandBrightFraction > (isDualArchProjection ? 0.38 : 0.3) ||
          summary.toothBandP10 > (isDualArchProjection ? 240 : 180) ||
          summary.toothBandMean > (isDualArchProjection ? 1040 : 860) ||
          summary.lowerBandBrightFraction > (isDualArchProjection ? 0.84 : 0.42) ||
          summary.lowerBandP50 > (isDualArchProjection ? 320 : -40) ||
          summary.meanAbsDelta > 560 ||
          detailRatio > 3.6
        );
      };
      const prioritizeGpuRetryConfigs = (
        retryConfigs: Array<{
          label: string;
          overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
        }>
      ) => {
        if (!gpuBackendEnabled) {
          return retryConfigs;
        }

        const byLabel = new Map(retryConfigs.map(config => [config.label, config] as const));
        const prioritized = CPR_GPU_PANO_PRIORITY_RETRY_LABELS.map(label => byLabel.get(label)).filter(
          (
            config
          ): config is {
            label: string;
            overrides: Partial<Parameters<typeof launchCPRWorker>[0]>;
          } => !!config
        );

        if (prioritized.length === 0) {
          return retryConfigs;
        }

        console.debug(
          '[CPR-GPU-RETRY-PLAN-JSON]',
          JSON.stringify({
            runId: debugRunId,
            originalRetryCount: retryConfigs.length,
            prioritizedRetryCount: prioritized.length,
            prioritizedLabels: prioritized.map(config => config.label),
            skippedLabels: retryConfigs
              .filter(config => !prioritized.some(selected => selected.label === config.label))
              .map(config => config.label),
          })
        );

        return prioritized;
      };
      const shouldRunCpuVirtualPanoFallback = (attempt: EvaluatedPanoAttempt | null): boolean => {
        if (!gpuBackendEnabled || earlyExitReason || !attempt) {
          return false;
        }

        const severeRiskFlags = new Set([
          'support-columns-unstable',
          'support-ambiguity-high',
          'support-forced-drift',
        ]);
        const hasSevereRiskFlag = attempt.supportSurfaceRiskFlags.some(flag => severeRiskFlags.has(flag));
        const unstableColumnFraction = attempt.qualityGateMetrics.supportUnstableColumnFraction ?? 0;
        const depthDriftP95Mm = attempt.qualityGateMetrics.supportBestDepthDriftP95Mm ?? 0;

        return (
          attempt.hardRejectReason === 'support-surface-instability' ||
          attempt.hardRejectReason === 'vertical-center-drift' ||
          hasSevereRiskFlag ||
          unstableColumnFraction >= 0.2 ||
          depthDriftP95Mm >= 1.25
        );
      };

      await workerWarmupPromise;
      let bestAttempt = await runWorkerAttempt('primary-mean-toothband-narrow', {
        modalityLutOverride: true,
        verticalHalfMm: toothBandVerticalHalfMm,
        verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
        slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
        slabSamples: focusedMeanSlabSamples,
        aggregation: 'MEAN',
      });
      recordAttempt(bestAttempt);
      const primaryAttemptEscalationReasons = collectAttemptEscalationReasons(bestAttempt);
      logAttemptEscalation('primary-attempt', bestAttempt, primaryAttemptEscalationReasons);

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
        retryConfigs.push({
          label: 'retry-mean-balanced-medium-root-biased-tight-slab',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: rootedMediumVerticalHalfMm,
            verticalCenterOffsetMm: rootedMandibularCenterOffsetMm,
            slabHalfThicknessMm: leakageFocusedMeanSlabHalfThicknessMm,
            slabSamples: leakageFocusedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-balanced-medium-strong-bias-tight-slab',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: rootedMediumVerticalHalfMm,
            verticalCenterOffsetMm: strongMandibularCenterOffsetMm,
            slabHalfThicknessMm: leakageFocusedMeanSlabHalfThicknessMm,
            slabSamples: leakageFocusedMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        retryConfigs.push({
          label: 'retry-mean-balanced-medium-stronger-bias-tight-slab',
          overrides: {
            modalityLutOverride: true,
            verticalHalfMm: rootedMediumVerticalHalfMm,
            verticalCenterOffsetMm: strongerMandibularCenterOffsetMm,
            slabHalfThicknessMm: leakageSharpMeanSlabHalfThicknessMm,
            slabSamples: leakageSharpMeanSlabSamples,
            aggregation: 'MEAN',
          },
        });
        // Sharp MEAN attempts - still volumetric, but slightly tighter than the main DRR slabs
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

        const prioritizedRetryConfigs = prioritizeGpuRetryConfigs(retryConfigs);
        for (const retryConfig of prioritizedRetryConfigs) {
          const attempt = await runWorkerAttempt(retryConfig.label, retryConfig.overrides);
          recordAttempt(attempt);
          if (shouldPromoteAttempt(attempt, bestAttempt)) {
            bestAttempt = attempt;
          }
          if (isGoodEnoughPanoAttempt(bestAttempt)) {
            earlyExitReason = `retry-good-enough:${bestAttempt.label}`;
            break;
          }
        }

        if (
          gpuBackendEnabled &&
          bestAttempt.requestedBackend === 'gpu' &&
          bestAttempt.resolvedBackend === 'gpu' &&
          (bestAttempt.hardRejectReason === 'support-surface-instability' ||
            bestAttempt.hardRejectReason === 'vertical-center-drift')
        ) {
          const targetedGpuMeanCenterOffsetMm =
            bestAttempt.hardRejectReason === 'vertical-center-drift'
              ? mildSuperiorCenterOffsetMm
              : neutralVerticalCenterOffsetMm;
          const targetedGpuMeanAttempt = await runWorkerAttempt(
            'retry-mean-toothband-balanced-gpu-targeted',
            {
              modalityLutOverride: true,
              verticalHalfMm: toothBandVerticalHalfMm,
              verticalCenterOffsetMm: targetedGpuMeanCenterOffsetMm,
              slabHalfThicknessMm: meanFallbackSlabHalfThicknessMm,
              slabSamples: meanFallbackSlabSamples,
              aggregation: 'MEAN',
            }
          );
          recordAttempt(targetedGpuMeanAttempt);
          if (shouldPromoteAttempt(targetedGpuMeanAttempt, bestAttempt)) {
            bestAttempt = targetedGpuMeanAttempt;
          }
          if (isGoodEnoughPanoAttempt(bestAttempt)) {
            earlyExitReason = `retry-good-enough:${bestAttempt.label}`;
          }
        }
      }

      const cpuVirtualPanoBaseAttempt = gpuBackendEnabled ? selectPreferredMeanAttempt() : null;
      const cpuVirtualPanoFallbackReasons = cpuVirtualPanoBaseAttempt
        ? collectAttemptEscalationReasons(cpuVirtualPanoBaseAttempt)
        : [];
      const shouldRunExplicitCpuVirtualPanoFallback = false;

      if (cpuVirtualPanoBaseAttempt) {
        console.debug(
          '[CPR-CPU-VIRTUAL-PANO-GATE-JSON]',
          JSON.stringify({
            runId: debugRunId,
            baseAttemptLabel: cpuVirtualPanoBaseAttempt.label,
            baseAttemptRequestedBackend: cpuVirtualPanoBaseAttempt.requestedBackend,
            baseAttemptResolvedBackend: cpuVirtualPanoBaseAttempt.resolvedBackend,
            baseAttemptReconstructionMode: cpuVirtualPanoBaseAttempt.reconstructionMode,
            baseAttemptQualityGatePassed: cpuVirtualPanoBaseAttempt.qualityGatePassed,
            baseAttemptHardRejectReason: cpuVirtualPanoBaseAttempt.hardRejectReason,
            baseAttemptQualityScore: cpuVirtualPanoBaseAttempt.qualityScore,
            baseAttemptSupportRiskLevel: cpuVirtualPanoBaseAttempt.supportSurfaceRiskLevel,
            baseAttemptSupportRiskFlags: cpuVirtualPanoBaseAttempt.supportSurfaceRiskFlags,
            reasons: cpuVirtualPanoFallbackReasons,
            shouldRun: shouldRunExplicitCpuVirtualPanoFallback,
            blockedByEarlyExit: !!earlyExitReason,
          })
        );
      }

      if (shouldRunExplicitCpuVirtualPanoFallback && cpuVirtualPanoBaseAttempt) {
        console.debug(
          '[CPR-CPU-VIRTUAL-PANO-FALLBACK-ATTEMPT-JSON]',
          JSON.stringify({
            runId: debugRunId,
            baseAttemptLabel: cpuVirtualPanoBaseAttempt.label,
            baseAttemptRequestedBackend: cpuVirtualPanoBaseAttempt.requestedBackend,
            baseAttemptResolvedBackend: cpuVirtualPanoBaseAttempt.resolvedBackend,
            baseAttemptReconstructionMode: cpuVirtualPanoBaseAttempt.reconstructionMode,
            baseAttemptQualityGatePassed: cpuVirtualPanoBaseAttempt.qualityGatePassed,
            baseAttemptHardRejectReason: cpuVirtualPanoBaseAttempt.hardRejectReason,
            baseAttemptQualityScore: cpuVirtualPanoBaseAttempt.qualityScore,
            fallbackReasons: cpuVirtualPanoFallbackReasons,
          })
        );

        const cpuVirtualPanoAttempt = await runWorkerAttempt('retry-cpu-virtual-pano-fallback', {
          ...cpuVirtualPanoBaseAttempt.requestOverrides,
          renderBackend: 'cpu',
          reconstructionMode: 'virtualPano',
          aggregation: cpuVirtualPanoBaseAttempt.aggregation,
          verticalHalfMm: cpuVirtualPanoBaseAttempt.actualVertHalfMm,
          verticalCenterOffsetMm: cpuVirtualPanoBaseAttempt.verticalCenterOffsetMm,
          slabHalfThicknessMm: cpuVirtualPanoBaseAttempt.slabHalfThicknessMm,
          slabSamples: cpuVirtualPanoBaseAttempt.slabSamples,
        });
        recordAttempt(cpuVirtualPanoAttempt);
        if (shouldPromoteAttempt(cpuVirtualPanoAttempt, bestAttempt)) {
          bestAttempt = cpuVirtualPanoAttempt;
        }
        if (isGoodEnoughPanoAttempt(bestAttempt)) {
          earlyExitReason = `cpu-virtual-pano-good-enough:${bestAttempt.label}`;
        }
      }

      const shouldRunMipFallbacks =
        !gpuBackendEnabled &&
        !earlyExitReason &&
        (!!bestAttempt.hardRejectReason ||
          (bestAttempt.qualityScore < 0 &&
            isLikelyPoorPanoQuality(
              bestAttempt.summary,
              bestAttempt.qualityGateMetrics.renderSupportMode
            )) ||
          !!(
            bestAttempt.summary &&
            (bestAttempt.summary.lowerBandBrightFraction >
              (isDualArchProjectionRenderMode(bestAttempt.qualityGateMetrics.renderSupportMode)
                ? 0.94
                : 0.78) ||
              bestAttempt.summary.lowerBandP50 >
                (isDualArchProjectionRenderMode(bestAttempt.qualityGateMetrics.renderSupportMode)
                  ? 420
                  : 260))
          ));

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
          if (shouldPromoteAttempt(attempt, bestAttempt)) {
            bestAttempt = attempt;
          }
          if (isGoodEnoughPanoAttempt(bestAttempt)) {
            earlyExitReason = `mip-good-enough:${bestAttempt.label}`;
            break;
          }
        }
      }

      const totalAttemptDurationMs = performance.now() - panoAttemptSequenceStartMs;
      const phase2BaseAttempt = selectPreferredMeanAttemptForPhase2();
      console.log(
        '[CPR-PHASE2-BASE-SELECTION-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: phase2BaseAttempt?.label ?? null,
          prioritizedLabels: CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS,
          fallbackLabel:
            phase2BaseAttempt && !CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS.includes(
              phase2BaseAttempt.label as (typeof CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS)[number]
            )
              ? phase2BaseAttempt.label
              : null,
        })
      );
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
      if (gpuBackendEnabled) {
        phase1VirtualPano.skippedReason = 'GPU_BACKEND_ACTIVE_USE_ATTEMPT_PIPELINE';
      } else if (!phase2BaseAttempt) {
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
      console.log(
        `[CPR-PHASE1] run=${debugRunId} base=${phase2BaseAttempt?.label ?? 'none'} ` +
          `aggregation=${phase2BaseAttempt?.aggregation ?? 'none'} ` +
          `executed=${phase1VirtualPano.executed ? 'yes' : 'no'} ` +
          `skipped=${phase1VirtualPano.skippedReason ?? 'none'} ` +
          `error=${phase1VirtualPano.error ?? 'none'}`
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
        workerAcceptedForOutput: boolean;
        orchestratorGatePassed: boolean | null;
        orchestratorRejectReasons: string[];
        displayDecisionReason: string | null;
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
        workerAcceptedForOutput: false,
        orchestratorGatePassed: null,
        orchestratorRejectReasons: [],
        displayDecisionReason: null,
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
            phase2BaseAttempt.panoHeight,
            phase2Result.debugMaps,
            readPhase4DiagnosticNumber(
              readPhase4DiagnosticRecord(phase2DiagnosticPayload?.virtualPanoRender)?.analysisCenterRow
            ) ??
              readPhase4DiagnosticNumber(
                readPhase4DiagnosticRecord(phase2DiagnosticPayload?.virtualPanoPhase12)
                  ?.analysisCenterRow
              ) ??
              undefined
          );
          const phase2RenderDiagnostics =
            phase2DiagnosticPayload &&
            phase2DiagnosticPayload.virtualPanoRender &&
            typeof phase2DiagnosticPayload.virtualPanoRender === 'object'
              ? (phase2DiagnosticPayload.virtualPanoRender as Record<string, unknown>)
              : null;
          const phase2RenderSupportMode = toNonEmptyString(phase2RenderDiagnostics?.renderSupportMode);
          const phase2Voi = computeAdaptivePanoVoi(
            phase2Summary,
            phase2Result.minValue,
            phase2Result.maxValue,
            phase2RenderSupportMode
          );

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
              ? phase2RenderDiagnostics.rejectReasons.filter(reason => typeof reason === 'string')
              : [];
          phase2VirtualPano.workerAcceptedForOutput =
            !!phase2RenderDiagnostics && phase2RenderDiagnostics.usedAsOutput === true;
          phase2VirtualPano.usedAsDisplayedOutput = phase2VirtualPano.workerAcceptedForOutput;
          const acceptedByLowerBandTolerance =
            !!phase2RenderDiagnostics &&
            phase2RenderDiagnostics.acceptedByLowerBandTolerance === true;
          const acceptedByToothBandTolerance =
            !!phase2RenderDiagnostics &&
            phase2RenderDiagnostics.acceptedByToothBandTolerance === true;
          if (phase2VirtualPano.usedAsDisplayedOutput && acceptedByLowerBandTolerance) {
            phase2VirtualPano.borderlineAcceptedReason = 'lower-band-tolerated-in-worker';
          } else if (phase2VirtualPano.usedAsDisplayedOutput && acceptedByToothBandTolerance) {
            phase2VirtualPano.borderlineAcceptedReason = 'tooth-band-tolerated-in-worker';
          } else if (
            phase2VirtualPano.usedAsDisplayedOutput &&
            phase2RejectReasons.length === 1 &&
            phase2RejectReasons[0] === 'lower-band-bright-fraction-too-high'
          ) {
            phase2VirtualPano.borderlineAcceptedReason = 'lower-band-bright-fraction-only-rejected';
          } else if (phase2RenderDiagnostics && phase2RenderDiagnostics.usedAsOutput !== true) {
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
      console.log(
        `[CPR-PHASE2] run=${debugRunId} base=${phase2BaseAttempt?.label ?? 'none'} ` +
          `aggregation=${phase2BaseAttempt?.aggregation ?? 'none'} ` +
          `executed=${phase2VirtualPano.executed ? 'yes' : 'no'} ` +
          `workerAccepted=${phase2VirtualPano.workerAcceptedForOutput ? 'yes' : 'no'} ` +
          `usedAsOutput=${phase2VirtualPano.usedAsDisplayedOutput ? 'yes' : 'no'} ` +
          `skipped=${phase2VirtualPano.skippedReason ?? 'none'} ` +
          `borderline=${phase2VirtualPano.borderlineAcceptedReason ?? 'none'} ` +
          `error=${phase2VirtualPano.error ?? 'none'}`
      );
      let phase2RigidSlicePreviewAttempt: Awaited<ReturnType<typeof runWorkerAttempt>> | null = null;
      const explicitPhase2QualityGateCandidate =
        phase2WorkerResult && phase2VirtualPano.summary && phase2BaseAttempt
          ? buildPhase4QualityGateCandidate({
              attemptLabel: phase2BaseAttempt.label,
              displayedPath: 'worker-recon',
              sourceVolumeId,
              summary: phase2VirtualPano.summary,
              qualityBase: scorePanoQuality(
                phase2VirtualPano.summary,
                toNonEmptyString(phase2VirtualPano.diagnostics?.renderSupportMode)
              ),
              qualityScore: scorePanoQuality(
                phase2VirtualPano.summary,
                toNonEmptyString(phase2VirtualPano.diagnostics?.renderSupportMode)
              ),
              hardRejectReason: getHardRejectReason(
                phase2VirtualPano.summary,
                toNonEmptyString(phase2VirtualPano.diagnostics?.renderSupportMode)
              ),
              workerDebugPayload: phase2WorkerResult.workerDebugPayload ?? null,
            })
          : null;
      if (explicitPhase2QualityGateCandidate) {
        phase2VirtualPano.orchestratorGatePassed = explicitPhase2QualityGateCandidate.pass;
        phase2VirtualPano.orchestratorRejectReasons =
          explicitPhase2QualityGateCandidate.rejectReasons.slice();
      }
      if (phase2VirtualPano.workerAcceptedForOutput && explicitPhase2QualityGateCandidate) {
        phase2VirtualPano.usedAsDisplayedOutput = explicitPhase2QualityGateCandidate.pass;
        phase2VirtualPano.displayDecisionReason = explicitPhase2QualityGateCandidate.pass
          ? 'accepted-by-worker-and-orchestrator'
          : 'rejected-by-orchestrator-gate';
      } else if (phase2VirtualPano.workerAcceptedForOutput) {
        phase2VirtualPano.displayDecisionReason = 'accepted-by-worker';
      } else if (phase2VirtualPano.executed) {
        phase2VirtualPano.displayDecisionReason = 'rejected-by-worker';
      }
      const shouldRunPhase2RigidSlicePreview =
        phase2VirtualPano.displayDecisionReason === 'rejected-by-orchestrator-gate' &&
        !!phase2BaseAttempt;
      if (shouldRunPhase2RigidSlicePreview && phase2BaseAttempt) {
        const rigidSliceVerticalHalfMm = Math.max(
          phase2BaseAttempt.actualVertHalfMm,
          mediumVerticalHalfMm
        );
        const rigidSliceSlabHalfThicknessMm = Math.max(
          0.85,
          Math.min(1.1, phase2BaseAttempt.slabHalfThicknessMm * 0.72)
        );
        const rigidSliceSlabSamples = 5;
        phase2RigidSlicePreviewAttempt = await runWorkerAttempt(
          'phase2-rigid-spline-slice-preview',
          {
            ...phase2BaseAttempt.requestOverrides,
            renderBackend: 'cpu',
            reconstructionMode: 'legacy',
            rigidVerticalSliceMode: true,
            aggregation: 'MEAN',
            modalityLutOverride: true,
            verticalHalfMm: rigidSliceVerticalHalfMm,
            verticalCenterOffsetMm: rigidSliceVerticalCenterOffsetMm,
            slabHalfThicknessMm: rigidSliceSlabHalfThicknessMm,
            slabSamples: rigidSliceSlabSamples,
          }
        );
        recordAttempt(phase2RigidSlicePreviewAttempt);
        console.log(
          '[CPR-RIGID-SLICE-FALLBACK-JSON]',
          JSON.stringify({
            runId: debugRunId,
            reason: 'phase2-quality-gate-rejected-prepare-rigid-spline-slice-preview',
            phase2BaseLabel: phase2BaseAttempt.label,
            rigidSliceLabel: phase2RigidSlicePreviewAttempt.label,
            rigidSliceQualityGatePassed: phase2RigidSlicePreviewAttempt.qualityGatePassed,
            rigidSliceQualityGateRejectReasons:
              phase2RigidSlicePreviewAttempt.qualityGateRejectReasons,
            rigidSliceRequestedBackend: phase2RigidSlicePreviewAttempt.requestedBackend,
            rigidSliceResolvedBackend: phase2RigidSlicePreviewAttempt.resolvedBackend,
            rigidSlicePipelineMode: phase2RigidSlicePreviewAttempt.pipelineMode,
            rigidSliceReconstructionMode: phase2RigidSlicePreviewAttempt.reconstructionMode,
            rigidSliceVerticalHalfMm: phase2RigidSlicePreviewAttempt.actualVertHalfMm,
            rigidSliceVerticalCenterOffsetMm:
              phase2RigidSlicePreviewAttempt.verticalCenterOffsetMm,
            rigidSliceSlabHalfThicknessMm:
              phase2RigidSlicePreviewAttempt.slabHalfThicknessMm,
            rigidSliceSlabSamples: phase2RigidSlicePreviewAttempt.slabSamples,
            rigidSliceOutputSignature: phase2RigidSlicePreviewAttempt.outputSignature,
          })
        );
      }
      const rankedAttemptAudit = attemptAudit.slice().sort((a, b) => {
        if (a.qualityGatePassed !== b.qualityGatePassed) {
          return a.qualityGatePassed ? -1 : 1;
        }
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
        return b.qualityScore - a.qualityScore || b.qualityBase - a.qualityBase;
      });
      const rankedAttempts: Array<Awaited<ReturnType<typeof runWorkerAttempt>>> =
        attemptResults.slice().sort(compareRankedPanoOutputs);
      const rankedTopAttempt: Awaited<ReturnType<typeof runWorkerAttempt>> =
        rankedAttempts[0] ?? bestAttempt;
      let selectedAttempt: Awaited<ReturnType<typeof runWorkerAttempt>> = rankedTopAttempt;
      const runnerUpAttempt: Awaited<ReturnType<typeof runWorkerAttempt>> | null =
        rankedAttempts.length > 1 ? rankedAttempts[1] : null;
      let selectedAttemptOverrideReason: string | null = null;
      let temporarilyPinnedDisplayedAttempt:
        | Awaited<ReturnType<typeof runWorkerAttempt>>
        | null = null;
      let selectedAttemptRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
        selectedAttempt.workerDebugPayload
      );
      let selectedDisplayedOutputMode: 'legacy' | 'workerGpuPhase2' | 'virtualPanoPhase2' =
        resolveDisplayedOutputModeFromWorkerResult({
          routeDiagnostic: selectedAttemptRouteDiagnostic,
          workerDebugPayload: selectedAttempt.workerDebugPayload,
        });
      let selectedDisplayedSourceLabel = selectedAttempt.label;
      let selectedDisplayedSourceAggregation = selectedAttempt.aggregation;
      const topGpuAttempt =
        attemptResults
          .filter(
            attempt =>
              attempt.requestedBackend === 'gpu' && attempt.resolvedBackend === 'gpu'
          )
          .slice()
          .sort(compareRankedPanoOutputs)[0] ?? null;
      const topCpuVirtualPanoAttempt =
        attemptResults
          .filter(
            attempt =>
              attempt.resolvedBackend === 'cpu' && attempt.reconstructionMode === 'virtualPano'
          )
          .slice()
          .sort(compareRankedPanoOutputs)[0] ?? null;
      const topLegacyAttempt =
        attemptResults
          .filter(
            attempt => attempt.resolvedBackend === 'cpu' && attempt.reconstructionMode === 'legacy'
          )
          .slice()
          .sort(compareRankedPanoOutputs)[0] ?? null;
      const shouldPreferPhase2BasePreviewFallback =
        !phase2VirtualPano.usedAsDisplayedOutput &&
        phase2VirtualPano.displayDecisionReason === 'rejected-by-orchestrator-gate' &&
        !!phase2BaseAttempt?.result &&
        phase2BaseAttempt.result.pixelData.length ===
          phase2BaseAttempt.result.width * phase2BaseAttempt.result.height &&
        !!phase2BaseAttempt.voi;
      const shouldPreferPhase2RigidSlicePreviewFallback =
        shouldPreferPhase2BasePreviewFallback &&
        !!phase2RigidSlicePreviewAttempt?.result &&
        phase2RigidSlicePreviewAttempt.result.pixelData.length ===
          phase2RigidSlicePreviewAttempt.result.width * phase2RigidSlicePreviewAttempt.result.height &&
        !!phase2RigidSlicePreviewAttempt.voi;
      if (shouldPreferPhase2RigidSlicePreviewFallback && phase2RigidSlicePreviewAttempt) {
        selectedAttempt = phase2RigidSlicePreviewAttempt;
        selectedAttemptOverrideReason =
          'phase2-quality-gate-rejected-use-rigid-spline-slice-preview';
      } else if (shouldPreferPhase2BasePreviewFallback && phase2BaseAttempt) {
        selectedAttempt = phase2BaseAttempt;
        selectedAttemptOverrideReason =
          'phase2-quality-gate-rejected-hold-phase2-base-preview';
      }
      temporarilyPinnedDisplayedAttempt = phase2VirtualPano.usedAsDisplayedOutput
        ? null
        : CPR_TEMP_DEBUG_PIN_DISPLAYED_ATTEMPT_LABELS.map(label =>
            attemptResults.find(
              attempt =>
                attempt.label === label &&
                !!attempt.result &&
                attempt.result.pixelData.length === attempt.result.width * attempt.result.height &&
                !!attempt.voi
            )
          ).find(
            (
              attempt
            ): attempt is NonNullable<(typeof attemptResults)[number]> => !!attempt
          ) ?? null;
      selectedAttemptRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
        selectedAttempt.workerDebugPayload
      );
      selectedDisplayedOutputMode = phase2VirtualPano.usedAsDisplayedOutput
        ? 'virtualPanoPhase2'
        : resolveDisplayedOutputModeFromWorkerResult({
            routeDiagnostic: selectedAttemptRouteDiagnostic,
            workerDebugPayload: selectedAttempt.workerDebugPayload,
          });
      selectedDisplayedSourceLabel = phase2VirtualPano.usedAsDisplayedOutput
        ? phase2BaseAttempt?.label ?? selectedAttempt.label
        : selectedAttempt.label;
      selectedDisplayedSourceAggregation = phase2VirtualPano.usedAsDisplayedOutput
        ? phase2BaseAttempt?.aggregation ?? selectedAttempt.aggregation
        : selectedAttempt.aggregation;
      console.debug(
        '[CPR-ATTEMPT-LIST-JSON]',
        JSON.stringify({
          runId: debugRunId,
          attempts: rankedAttemptAudit,
          rankedTopLabel: rankedTopAttempt.label,
          selectedLabel: selectedAttempt.label,
          selectedAttemptOverrideReason,
          temporaryDisplayedAttemptLabel: temporarilyPinnedDisplayedAttempt?.label ?? null,
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
        })
      );
      if (selectedAttemptOverrideReason) {
        console.debug(
          '[CPR-PHASE2-FALLBACK-PREVIEW-JSON]',
          JSON.stringify({
            runId: debugRunId,
            reason: selectedAttemptOverrideReason,
            rankedTopLabel: rankedTopAttempt.label,
            selectedLabel: selectedAttempt.label,
            phase2BaseLabel: phase2BaseAttempt?.label ?? null,
            rigidSlicePreviewLabel: phase2RigidSlicePreviewAttempt?.label ?? null,
            phase2DisplayDecisionReason: phase2VirtualPano.displayDecisionReason,
            phase2OrchestratorRejectReasons: phase2VirtualPano.orchestratorRejectReasons,
          })
        );
      }
      console.log(
        '[CPR-PHASE2-QUALITY-GATE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: phase2BaseAttempt?.label ?? null,
          workerAcceptedForOutput: phase2VirtualPano.workerAcceptedForOutput,
          orchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
          orchestratorRejectReasons: phase2VirtualPano.orchestratorRejectReasons,
          displayDecisionReason: phase2VirtualPano.displayDecisionReason,
          qualityGateCandidate: explicitPhase2QualityGateCandidate,
        })
      );
      console.log(
        `[CPR-PHASE2-DISPLAY-DECISION] run=${debugRunId} base=${phase2BaseAttempt?.label ?? 'none'} ` +
          `workerAccepted=${phase2VirtualPano.workerAcceptedForOutput ? 'yes' : 'no'} ` +
          `orchestratorGate=${phase2VirtualPano.orchestratorGatePassed === null ? 'na' : phase2VirtualPano.orchestratorGatePassed ? 'pass' : 'fail'} ` +
          `usedAsOutput=${phase2VirtualPano.usedAsDisplayedOutput ? 'yes' : 'no'} ` +
          `decision=${phase2VirtualPano.displayDecisionReason ?? 'none'} ` +
          `rejectReasons=${phase2VirtualPano.orchestratorRejectReasons.join(',') || 'none'}`
      );
      const gpuQualityGateCandidate = topGpuAttempt
        ? buildPhase4QualityGateCandidate({
            attemptLabel: topGpuAttempt.label,
            displayedPath: 'worker-recon',
            sourceVolumeId,
            summary: topGpuAttempt.summary,
            qualityBase: topGpuAttempt.qualityBase,
            qualityScore: topGpuAttempt.qualityScore,
            hardRejectReason: topGpuAttempt.hardRejectReason,
            workerDebugPayload: topGpuAttempt.workerDebugPayload ?? null,
          })
        : null;
      const cpuQualityGateCandidate =
        explicitPhase2QualityGateCandidate ??
        (topCpuVirtualPanoAttempt
          ? buildPhase4QualityGateCandidate({
              attemptLabel: topCpuVirtualPanoAttempt.label,
              displayedPath: 'worker-recon',
              sourceVolumeId,
              summary: topCpuVirtualPanoAttempt.summary,
              qualityBase: topCpuVirtualPanoAttempt.qualityBase,
              qualityScore: topCpuVirtualPanoAttempt.qualityScore,
              hardRejectReason: topCpuVirtualPanoAttempt.hardRejectReason,
              workerDebugPayload: topCpuVirtualPanoAttempt.workerDebugPayload ?? null,
            })
          : null);
      const legacyQualityGateCandidate = topLegacyAttempt
        ? buildPhase4QualityGateCandidate({
            attemptLabel: topLegacyAttempt.label,
            displayedPath: 'worker-recon',
            sourceVolumeId,
            summary: topLegacyAttempt.summary,
            qualityBase: topLegacyAttempt.qualityBase,
            qualityScore: topLegacyAttempt.qualityScore,
            hardRejectReason: topLegacyAttempt.hardRejectReason,
            workerDebugPayload: topLegacyAttempt.workerDebugPayload ?? null,
          })
        : null;
      let selectedQualityGateCandidate =
        phase2VirtualPano.usedAsDisplayedOutput && explicitPhase2QualityGateCandidate
          ? explicitPhase2QualityGateCandidate
          : buildPhase4QualityGateCandidate({
              attemptLabel: selectedAttempt.label,
              displayedPath: 'worker-recon',
              sourceVolumeId,
              summary: selectedAttempt.summary,
              qualityBase: selectedAttempt.qualityBase,
              qualityScore: selectedAttempt.qualityScore,
              hardRejectReason: selectedAttempt.hardRejectReason,
              workerDebugPayload: selectedAttempt.workerDebugPayload ?? null,
            });
      let qualityGateSelectionReason = selectedQualityGateCandidate.pass
        ? selectedQualityGateCandidate.candidateSource === 'worker-gpu-support-surface'
          ? 'gpu-candidate-passed'
          : selectedQualityGateCandidate.candidateSource === 'worker-cpu-virtual-pano' ||
              selectedQualityGateCandidate.candidateSource ===
                'worker-cpu-arch-guided-synthetic'
            ? gpuQualityGateCandidate && !gpuQualityGateCandidate.pass
              ? 'gpu-rejected-cpu-selected'
              : 'cpu-selected'
            : selectedQualityGateCandidate.requestedReconstructionMode === 'legacy'
              ? 'explicit-legacy-selection'
              : 'legacy-fallback-allowed'
        : CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK
          ? 'selected-after-gate-failure-reference-or-legacy-allowed'
          : 'selected-after-gate-failure-reference-and-legacy-blocked';
      const shouldPreservePreviousPanoDisplay =
        !selectedQualityGateCandidate.pass &&
        !CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK;
      const currentPanoViewportBeforeRefresh = shouldPreservePreviousPanoDisplay
        ? findViewportByLogicalId(servicesManager, 'cpr-pano')
        : null;
      const preservedPreviousPanoImageId = shouldPreservePreviousPanoDisplay
        ? getCurrentStackImageId(currentPanoViewportBeforeRefresh)
        : null;
      const previousDisplayedPanoPayload = preservedPreviousPanoImageId
        ? getPanoImagePayload(preservedPreviousPanoImageId)
        : null;
      const preservedPreviousPanoPayload =
        previousDisplayedPanoPayload?.qualityGatePassed === true
          ? previousDisplayedPanoPayload
          : null;
      const preservedPreviousPanoViewportProperties =
        ((currentPanoViewportBeforeRefresh as { getProperties?: () => unknown } | null)?.getProperties?.() ||
          {}) as {
          voiRange?: { lower?: number; upper?: number };
        };
      const preservedPreviousPanoVoi =
        (preservedPreviousPanoPayload
          ? createPanoVoiFromWindowLevel(
              Number(preservedPreviousPanoPayload.windowWidth),
              Number(preservedPreviousPanoPayload.windowCenter)
            )
          : null) ?? createPanoVoiFromRange(preservedPreviousPanoViewportProperties.voiRange);
      const canPreservePreviousPanoDisplay =
        shouldPreservePreviousPanoDisplay &&
        !!preservedPreviousPanoImageId &&
        !!preservedPreviousPanoPayload;
      const phase2WinsDegradedFallbackRanking =
        !!explicitPhase2QualityGateCandidate &&
        compareRankedPanoOutputs(
          {
            qualityGatePassed: explicitPhase2QualityGateCandidate.pass,
            qualityBase: explicitPhase2QualityGateCandidate.qualityBase,
            qualityScore: explicitPhase2QualityGateCandidate.qualityScore,
            hardRejectReason: explicitPhase2QualityGateCandidate.hardRejectReason,
            summary: phase2VirtualPano.summary,
            supportSurfaceRiskSummary: explicitPhase2QualityGateCandidate.supportSurfaceRiskSummary,
          },
          {
            qualityGatePassed: selectedQualityGateCandidate.pass,
            qualityBase: selectedQualityGateCandidate.qualityBase,
            qualityScore: selectedQualityGateCandidate.qualityScore,
            hardRejectReason: selectedQualityGateCandidate.hardRejectReason,
            summary: selectedAttempt.summary,
            supportSurfaceRiskSummary: selectedQualityGateCandidate.supportSurfaceRiskSummary,
          }
        ) < 0;
      const shouldDisplayPhase2DiagnosticDraft =
        shouldPreservePreviousPanoDisplay &&
        !canPreservePreviousPanoDisplay &&
        !phase2VirtualPano.usedAsDisplayedOutput &&
        !!explicitPhase2QualityGateCandidate &&
        !!phase2BaseAttempt &&
        !!phase2WorkerResult &&
        !!phase2VirtualPano.summary &&
        !!phase2VirtualPano.voi &&
        ((phase2VirtualPano.workerAcceptedForOutput &&
          phase2VirtualPano.displayDecisionReason === 'rejected-by-orchestrator-gate') ||
          (phase2VirtualPano.displayDecisionReason === 'rejected-by-worker' &&
            phase2WinsDegradedFallbackRanking));
      if (
        shouldDisplayPhase2DiagnosticDraft &&
        explicitPhase2QualityGateCandidate &&
        phase2BaseAttempt
      ) {
        const phase2DraftDisplayReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'draft-displayed-after-orchestrator-gate-reject'
          : 'draft-displayed-after-worker-reject';
        selectedAttempt = phase2BaseAttempt;
        selectedAttemptOverrideReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'phase2-quality-gate-rejected-display-primary-diagnostic-draft'
          : 'phase2-worker-rejected-display-primary-diagnostic-draft';
        selectedAttemptRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
          selectedAttempt.workerDebugPayload
        );
        phase2VirtualPano.usedAsDisplayedOutput = true;
        phase2VirtualPano.displayDecisionReason = phase2DraftDisplayReason;
        selectedDisplayedOutputMode = 'virtualPanoPhase2';
        selectedDisplayedSourceLabel = phase2BaseAttempt.label;
        selectedDisplayedSourceAggregation = phase2BaseAttempt.aggregation;
        selectedQualityGateCandidate = explicitPhase2QualityGateCandidate;
        qualityGateSelectionReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'phase2-diagnostic-draft-displayed-after-gate-reject'
          : 'phase2-diagnostic-draft-displayed-after-worker-reject';
        console.debug(
          '[CPR-PHASE2-DRAFT-DISPLAY-JSON]',
          JSON.stringify({
            runId: debugRunId,
            reason: phase2DraftDisplayReason,
            phase2BaseLabel: phase2BaseAttempt.label,
            phase2WorkerAcceptedForOutput: phase2VirtualPano.workerAcceptedForOutput,
            phase2OrchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
            phase2OrchestratorRejectReasons: phase2VirtualPano.orchestratorRejectReasons,
            phase2WinsDegradedFallbackRanking,
            selectedAttemptOverrideReason,
          })
        );
      }
      const duplicateOfLabelByAttemptLabel = new Map(
        attemptAudit.map(entry => [entry.label, entry.duplicateOfLabel ?? null] as const)
      );
      const displaySelectionBreakdownByAttemptLabel = new Map(
        attemptResults.map(attempt => [
          attempt.label,
          buildPhase4DisplaySelectionBreakdown({
            qualityGatePassed: attempt.qualityGatePassed,
            qualityBase: attempt.qualityBase,
            qualityScore: attempt.qualityScore,
            detailReward: attempt.detailReward,
            hardRejectReason: attempt.hardRejectReason,
            summary: attempt.summary,
            metrics: attempt.qualityGateMetrics,
            requestedBackend: attempt.requestedBackend,
            resolvedBackend: attempt.resolvedBackend,
            aggregation: attempt.aggregation,
            actualVertHalfMm: attempt.actualVertHalfMm,
            verticalCenterOffsetMm: attempt.verticalCenterOffsetMm,
            slabHalfThicknessMm: attempt.slabHalfThicknessMm,
            slabSamples: attempt.slabSamples,
            duplicateOfLabel: duplicateOfLabelByAttemptLabel.get(attempt.label) ?? null,
          }),
        ] as const)
      );
      const compareDisplayRankedAttempts = (
        a: Awaited<ReturnType<typeof runWorkerAttempt>>,
        b: Awaited<ReturnType<typeof runWorkerAttempt>>
      ): number => {
        const aBreakdown = displaySelectionBreakdownByAttemptLabel.get(a.label);
        const bBreakdown = displaySelectionBreakdownByAttemptLabel.get(b.label);
        if (!aBreakdown || !bBreakdown) {
          return compareRankedPanoOutputs(a, b);
        }

        return comparePhase4DisplaySelectionBreakdowns(
          {
            qualityGatePassed: a.qualityGatePassed,
            qualityBase: a.qualityBase,
            qualityScore: a.qualityScore,
            hardRejectReason: a.hardRejectReason,
            summary: a.summary,
            breakdown: aBreakdown,
          },
          {
            qualityGatePassed: b.qualityGatePassed,
            qualityBase: b.qualityBase,
            qualityScore: b.qualityScore,
            hardRejectReason: b.hardRejectReason,
            summary: b.summary,
            breakdown: bBreakdown,
          }
        );
      };
      const degradedPreviewCandidateOptions = attemptResults
        .filter(
          attempt =>
            !!attempt.result &&
            attempt.result.pixelData.length === attempt.result.width * attempt.result.height &&
            !!attempt.voi
        )
        .map(attempt => {
          const candidate = buildPhase4QualityGateCandidate({
            attemptLabel: attempt.label,
            displayedPath: 'worker-recon',
            sourceVolumeId,
            summary: attempt.summary,
            qualityBase: attempt.qualityBase,
            qualityScore: attempt.qualityScore,
            hardRejectReason: attempt.hardRejectReason,
            workerDebugPayload: attempt.workerDebugPayload ?? null,
          });
          const degradedPreviewAssessment = assessPhase4DegradedPreviewCandidate(candidate);

          return {
            attempt,
            candidate,
            degradedPreviewAssessment,
          };
        });
      const degradedPreviewDisplayableOptions = degradedPreviewCandidateOptions.filter(
        option => !option.candidate.pass && !option.degradedPreviewAssessment.catastrophic
      );
      const degradedPreviewDisplayableOptionByCanonicalLabel = new Map<
        string,
        (typeof degradedPreviewDisplayableOptions)[number]
      >();
      for (const option of degradedPreviewDisplayableOptions) {
        const canonicalLabel =
          duplicateOfLabelByAttemptLabel.get(option.attempt.label) ?? option.attempt.label;
        const existing = degradedPreviewDisplayableOptionByCanonicalLabel.get(canonicalLabel);
        if (!existing || compareDisplayRankedAttempts(option.attempt, existing.attempt) < 0) {
          degradedPreviewDisplayableOptionByCanonicalLabel.set(canonicalLabel, option);
        }
      }
      const degradedPreviewUniqueDisplayableOptions = Array.from(
        degradedPreviewDisplayableOptionByCanonicalLabel.values()
      );
      const degradedPreviewRankedDisplayableOptions =
        degradedPreviewUniqueDisplayableOptions
          .slice()
          .sort((a, b) => compareDisplayRankedAttempts(a.attempt, b.attempt));
      const degradedPreviewFallbackChoice =
        !selectedQualityGateCandidate.pass
          ? degradedPreviewRankedDisplayableOptions[0] ?? null
          : null;
      const degradedPreviewCatastrophicOptions = degradedPreviewCandidateOptions.filter(
        option => option.degradedPreviewAssessment.catastrophic
      );
      const degradedPreviewCandidateByAttemptLabel = new Map(
        degradedPreviewCandidateOptions.map(option => [
          option.attempt.label,
          {
            candidate: option.candidate,
            displayable: option.candidate.pass || !option.degradedPreviewAssessment.catastrophic,
            catastrophicReasons: option.degradedPreviewAssessment.catastrophicReasons,
          },
        ])
      );
      const buildDisplayRankingLogRow = (
        attempt: Awaited<ReturnType<typeof runWorkerAttempt>>
      ): Record<string, unknown> => {
        const breakdown = displaySelectionBreakdownByAttemptLabel.get(attempt.label);
        const displayable =
          degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.displayable ??
          attempt.qualityGatePassed;
        const winnerBreakdown = degradedPreviewFallbackChoice
          ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewFallbackChoice.attempt.label) ??
            null
          : null;

        return {
          label: attempt.label,
          displayable,
          rejectReasons: attempt.qualityGateRejectReasons,
          duplicateOfLabel: duplicateOfLabelByAttemptLabel.get(attempt.label) ?? null,
          legacyQualityScore: attempt.qualityScore,
          legacyQualityBase: attempt.qualityBase,
          legacyDetailReward: attempt.detailReward,
          displaySelectionScore: breakdown?.displaySelectionScore ?? null,
          preferredTightRootFamily: breakdown?.preferredTightRootFamily ?? false,
          supportP50: breakdown?.supportConfidenceP50 ?? null,
          supportPathP50: breakdown?.supportPathConfidenceP50 ?? null,
          bgOutlier05: breakdown?.backgroundOutlierFraction05 ?? null,
          bgOutlier10: breakdown?.backgroundOutlierFraction10 ?? null,
          effectiveTroughHalfWidthP50Mm: breakdown?.effectiveTroughHalfWidthP50Mm ?? null,
          effectiveTroughHalfWidthP90Mm: breakdown?.effectiveTroughHalfWidthP90Mm ?? null,
          participatingSamplesP50: breakdown?.participatingSamplesP50 ?? null,
          participatingSamplesP90: breakdown?.participatingSamplesP90 ?? null,
          actualVertHalfMm: breakdown?.actualVertHalfMm ?? attempt.actualVertHalfMm,
          verticalCenterOffsetMm:
            breakdown?.verticalCenterOffsetMm ?? attempt.verticalCenterOffsetMm,
          slabHalfThicknessMm:
            breakdown?.slabHalfThicknessMm ?? attempt.slabHalfThicknessMm,
          slabSamples: breakdown?.slabSamples ?? attempt.slabSamples,
          breakdown: breakdown
            ? {
                qualityGatePassedBonus: breakdown.qualityGatePassedBonus,
                legacyQualityScoreComponent: breakdown.legacyQualityScoreComponent,
                supportConfidenceReward: breakdown.supportConfidenceReward,
                supportPathConfidenceReward: breakdown.supportPathConfidenceReward,
                backgroundLeakagePenalty: breakdown.backgroundLeakagePenalty,
                troughAdmissionPenalty: breakdown.troughAdmissionPenalty,
                broadWeakSupportPenalty: breakdown.broadWeakSupportPenalty,
                preferredTightRootFamilyBonus: breakdown.preferredTightRootFamilyBonus,
                hardRejectPenalty: breakdown.hardRejectPenalty,
                duplicatePenalty: breakdown.duplicatePenalty,
              }
            : null,
          comparisonToWinner:
            breakdown && winnerBreakdown
              ? {
                  deltaDisplaySelectionScore:
                    breakdown.displaySelectionScore - winnerBreakdown.displaySelectionScore,
                  deltaLegacyQualityScore: attempt.qualityScore - degradedPreviewFallbackChoice!.attempt.qualityScore,
                  deltaSupportP50:
                    (breakdown.supportConfidenceP50 ?? 0) -
                    (winnerBreakdown.supportConfidenceP50 ?? 0),
                  deltaBgOutlier05:
                    (breakdown.backgroundOutlierFraction05 ?? 0) -
                    (winnerBreakdown.backgroundOutlierFraction05 ?? 0),
                  deltaBgOutlier10:
                    (breakdown.backgroundOutlierFraction10 ?? 0) -
                    (winnerBreakdown.backgroundOutlierFraction10 ?? 0),
                  deltaEffectiveTroughHalfWidthP50Mm:
                    (breakdown.effectiveTroughHalfWidthP50Mm ?? 0) -
                    (winnerBreakdown.effectiveTroughHalfWidthP50Mm ?? 0),
                  deltaParticipatingSamplesP50:
                    (breakdown.participatingSamplesP50 ?? 0) -
                    (winnerBreakdown.participatingSamplesP50 ?? 0),
                  materiallyCleanerThanWinner: isMateriallyCleanerPhase4DisplayCandidate(
                    breakdown,
                    winnerBreakdown
                  ),
                  winnerMateriallyCleaner: isMateriallyCleanerPhase4DisplayCandidate(
                    winnerBreakdown,
                    breakdown
                  ),
                }
              : null,
        };
      };
      const displayRankingTopRows = degradedPreviewRankedDisplayableOptions
        .slice(0, 8)
        .map(option => buildDisplayRankingLogRow(option.attempt));
      const displayRankingFocusedRows = Array.from(CPR_DEBUG_DISPLAY_SELECTION_FOCUS_LABELS)
        .map(label => attemptResults.find(attempt => attempt.label === label))
        .filter(
          (
            attempt
          ): attempt is NonNullable<(typeof attemptResults)[number]> => !!attempt
        )
        .map(attempt => buildDisplayRankingLogRow(attempt));
      const qualityGateFallbackSummary = {
        selectionReason: qualityGateSelectionReason,
        referenceOrLegacyFallbackAllowed: CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK,
        finalSelectedOutput: summarizePhase4QualityGateCandidate(selectedQualityGateCandidate),
        candidates: {
          gpu: summarizePhase4QualityGateCandidate(gpuQualityGateCandidate),
          cpu: summarizePhase4QualityGateCandidate(cpuQualityGateCandidate),
          legacy: summarizePhase4QualityGateCandidate(legacyQualityGateCandidate),
        },
        degradedPreview: {
          available: !!degradedPreviewFallbackChoice,
          selectedAttemptLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
          selectedCandidate: summarizePhase4QualityGateCandidate(
            degradedPreviewFallbackChoice?.candidate ?? null
          ),
          selectedRejectReasons: degradedPreviewFallbackChoice?.candidate.rejectReasons ?? [],
          displayableCandidateCount: degradedPreviewUniqueDisplayableOptions.length,
          duplicateSignatureCollapsedCount:
            degradedPreviewDisplayableOptions.length - degradedPreviewUniqueDisplayableOptions.length,
          selectedDisplaySelectionScore:
            degradedPreviewFallbackChoice &&
            displaySelectionBreakdownByAttemptLabel.has(degradedPreviewFallbackChoice.attempt.label)
              ? displaySelectionBreakdownByAttemptLabel.get(
                  degradedPreviewFallbackChoice.attempt.label
                )?.displaySelectionScore ?? null
              : null,
          catastrophicCandidateCount: degradedPreviewCatastrophicOptions.length,
          catastrophicCandidates: degradedPreviewCatastrophicOptions.slice(0, 8).map(option => ({
            attemptLabel: option.attempt.label,
            catastrophicReasons: option.degradedPreviewAssessment.catastrophicReasons,
            rejectReasons: option.candidate.rejectReasons,
          })),
        },
      };
      console.log(
        '[CPR-QUALITY-GATE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          displayedPath: 'worker-recon',
          backend: selectedQualityGateCandidate.backend,
          reconstructionMode: selectedQualityGateCandidate.reconstructionMode,
          sourceVolumeId,
          selectionRule: {
            primary: 'worker-gpu-support-surface',
            secondary: 'worker-cpu-virtual-pano-or-arch-guided-synthetic',
            tertiary: 'reference-or-legacy-only-if-flag-enabled',
          },
          referencePathAvailable: true,
          referenceOrLegacyFallbackAllowed: CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK,
          displayedOutputMode: selectedDisplayedOutputMode,
          displayedSourceLabel: selectedDisplayedSourceLabel,
          displayedSourceAggregation: selectedDisplayedSourceAggregation,
          selectedSupportSurfaceRiskSummary:
            selectedQualityGateCandidate.supportSurfaceRiskSummary,
          selectedSupportSurfaceBaselineFingerprint:
            selectedQualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
          finalSelectedOutput: qualityGateFallbackSummary.finalSelectedOutput,
          candidates: qualityGateFallbackSummary.candidates,
          degradedPreview: qualityGateFallbackSummary.degradedPreview,
          temporaryDisplayPin: {
            active: !!temporarilyPinnedDisplayedAttempt,
            requestedLabels: CPR_TEMP_DEBUG_PIN_DISPLAYED_ATTEMPT_LABELS,
            displayedAttemptLabel: temporarilyPinnedDisplayedAttempt?.label ?? null,
            rankedTopLabel: rankedTopAttempt.label,
          },
          fallbackPath: {
            selectionReason: qualityGateSelectionReason,
            gpuRejected: gpuQualityGateCandidate ? !gpuQualityGateCandidate.pass : null,
            gpuRejectReasons: gpuQualityGateCandidate?.rejectReasons ?? [],
            cpuRejected: cpuQualityGateCandidate ? !cpuQualityGateCandidate.pass : null,
            cpuRejectReasons: cpuQualityGateCandidate?.rejectReasons ?? [],
            legacyRejected: legacyQualityGateCandidate ? !legacyQualityGateCandidate.pass : null,
            legacyRejectReasons: legacyQualityGateCandidate?.rejectReasons ?? [],
            selectedDespiteGateFailure: !selectedQualityGateCandidate.pass,
            preservePreviousPanoRequested: shouldPreservePreviousPanoDisplay,
            preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
            preservedPreviousPanoImageId,
            degradedPreviewAvailable: !!degradedPreviewFallbackChoice,
            degradedPreviewAttemptLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
            degradedPreviewRejectReasons: degradedPreviewFallbackChoice?.candidate.rejectReasons ?? [],
            degradedPreviewCatastrophicCandidateCount: degradedPreviewCatastrophicOptions.length,
          },
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
        })
      );
      console.log(
        '[CPR-DISPLAY-RANKING-TABLE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectionMode: 'degraded-display',
          winnerLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
          winnerDisplaySelectionScore: degradedPreviewFallbackChoice
            ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewFallbackChoice.attempt.label)
                ?.displaySelectionScore ?? null
            : null,
          duplicateSignatureDisplayableCount:
            degradedPreviewDisplayableOptions.length - degradedPreviewUniqueDisplayableOptions.length,
          topDisplayableRows: displayRankingTopRows,
          focusedRows: displayRankingFocusedRows,
        })
      );
      console.log(
        '[CPR-SELECTED-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: selectedAttempt.label,
          rankedTopLabel: rankedTopAttempt.label,
          displayedOutputMode: selectedDisplayedOutputMode,
          temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
          temporaryDisplayedAttemptLabel: temporarilyPinnedDisplayedAttempt?.label ?? null,
          selectedRequestedBackend: selectedAttempt.requestedBackend,
          selectedResolvedBackend: selectedAttempt.resolvedBackend,
          selectedPipelineMode: selectedAttempt.pipelineMode,
          selectedFallbackReason: selectedAttempt.fallbackReason,
          selectedReconstructionMode: selectedAttempt.reconstructionMode,
          displayedSourceLabel: selectedDisplayedSourceLabel,
          displayedSourceAggregation: selectedDisplayedSourceAggregation,
          selectedAggregation: selectedAttempt.aggregation,
          selectedActualVertHalfMm: selectedAttempt.actualVertHalfMm,
          selectedSlabHalfThicknessMm: selectedAttempt.slabHalfThicknessMm,
          selectedSlabSamples: selectedAttempt.slabSamples,
          selectedVerticalCenterOffsetMm: selectedAttempt.verticalCenterOffsetMm,
          selectedColumnPixelSpacing: selectedAttempt.columnPixelSpacing,
          selectedRowPixelSpacing: selectedAttempt.rowPixelSpacing,
          selectedHardRejectReason: selectedAttempt.hardRejectReason,
          selectedQualityScore: selectedAttempt.qualityScore,
          selectedQualityBase: selectedAttempt.qualityBase,
          selectedAttemptOverrideReason,
          selectedPhase2GatePassed: selectedAttemptRouteDiagnostic.phase2GatePassed,
          selectedSummary: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2VirtualPano.summary
            : selectedAttempt.summary,
          selectedVoi: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2VirtualPano.voi
            : selectedAttempt.voi,
          selectedWorkerDebugPayload: phase2VirtualPano.usedAsDisplayedOutput
            ? phase2WorkerResult?.workerDebugPayload ?? null
            : selectedAttempt.workerDebugPayload ?? null,
          phase1VirtualPano,
          phase2VirtualPano,
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
          runnerUp: runnerUpAttempt,
          scoreDeltaToRunnerUp: runnerUpAttempt
            ? selectedAttempt.qualityScore - runnerUpAttempt.qualityScore
            : null,
          baseDeltaToRunnerUp: runnerUpAttempt
            ? selectedAttempt.qualityBase - runnerUpAttempt.qualityBase
            : null,
          selectedMatchesRankedTop: selectedAttempt.label === rankedTopAttempt.label,
          qualityGateSelectionReason,
          qualityGateSelectedOutput: qualityGateFallbackSummary.finalSelectedOutput,
          qualityGateCandidates: qualityGateFallbackSummary.candidates,
          qualityGateDegradedPreview: qualityGateFallbackSummary.degradedPreview,
          degradedDisplayRankingTopLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
          degradedDisplayRankingTopScore: degradedPreviewFallbackChoice
            ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewFallbackChoice.attempt.label)
                ?.displaySelectionScore ?? null
            : null,
          selectedSupportSurfaceRiskSummary:
            selectedQualityGateCandidate.supportSurfaceRiskSummary,
          selectedSupportSurfaceBaselineFingerprint:
            selectedQualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
          preservePreviousPanoRequested: shouldPreservePreviousPanoDisplay,
          preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
          preservedPreviousPanoImageId,
        })
      );
      console.debug(
        '[CPR-ATTEMPT-TABLE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          rankedAttempts: rankedAttemptAudit.map(attempt => ({
            label: attempt.label,
            checksum: attempt.outputSignature.checksum,
            supportP50: attempt.supportSurfaceRiskSummary.stability.supportConfidenceP50,
            bgOutlier05: attempt.backgroundOutlierFraction05,
            bgOutlier10: attempt.backgroundOutlierFraction10,
            blackClip: attempt.supportSurfaceRiskSummary.background.blackClipFraction,
            lowerBandBrightFraction: attempt.lowerBandBrightFraction,
            rejectReasons: attempt.qualityGateRejectReasons,
            displayable:
              degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.displayable ??
              attempt.qualityGatePassed,
            legacyQualityScore: attempt.qualityScore,
            legacyQualityBase: attempt.qualityBase,
            displaySelectionScore:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.displaySelectionScore ??
              null,
            preferredTightRootFamily:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.preferredTightRootFamily ??
              false,
            effectiveTroughHalfWidthP50Mm:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)
                ?.effectiveTroughHalfWidthP50Mm ?? null,
            participatingSamplesP50:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.participatingSamplesP50 ??
              null,
            duplicateOfLabel: attempt.duplicateOfLabel ?? null,
            catastrophicReasons:
              degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.catastrophicReasons ?? [],
          })),
        })
      );

      if (shouldExportCprAttemptReport()) {
        try {
          const attemptLookupByLabel = new Map(attemptResults.map(attempt => [attempt.label, attempt]));
          const displayRankedAttemptLabels = degradedPreviewRankedDisplayableOptions
            .slice(0, CPR_DEBUG_EXPORT_TOP_ATTEMPT_COUNT)
            .map(option => option.attempt.label);
          const reportAttemptLabels = Array.from(
            new Set(
              [
                ...rankedAttempts.slice(0, CPR_DEBUG_EXPORT_TOP_ATTEMPT_COUNT).map(
                  attempt => attempt.label
                ),
                ...displayRankedAttemptLabels,
                'primary-mean-toothband-narrow',
                'retry-force-lut-mean-narrow-no-normalization',
                'retry-mean-balanced-medium-strong-bias',
                ...CPR_DEBUG_DISPLAY_SELECTION_FOCUS_LABELS,
                'retry-cpu-virtual-pano-fallback',
              ].filter(label => attemptLookupByLabel.has(label))
            )
          );
          const reportRows = rankedAttemptAudit.map(attempt => ({
            label: attempt.label,
            checksum: attempt.outputSignature.checksum,
            supportP50: attempt.supportSurfaceRiskSummary.stability.supportConfidenceP50,
            bgOutlier05: attempt.backgroundOutlierFraction05,
            bgOutlier10: attempt.backgroundOutlierFraction10,
            blackClip: attempt.supportSurfaceRiskSummary.background.blackClipFraction,
            lowerBandBrightFraction: attempt.lowerBandBrightFraction,
            rejectReasons: attempt.qualityGateRejectReasons,
            displayable:
              degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.displayable ??
              attempt.qualityGatePassed,
            legacyQualityScore: attempt.qualityScore,
            legacyQualityBase: attempt.qualityBase,
            displaySelectionScore:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.displaySelectionScore ??
              null,
            preferredTightRootFamily:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.preferredTightRootFamily ??
              false,
            effectiveTroughHalfWidthP50Mm:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)
                ?.effectiveTroughHalfWidthP50Mm ?? null,
            participatingSamplesP50:
              displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.participatingSamplesP50 ??
              null,
            duplicateOfLabel: attempt.duplicateOfLabel ?? null,
          }));
          const reportImages: Parameters<typeof downloadCprDebugArtifacts>[0]['images'] = [];
          const appendReportImage = (
            title: string,
            filenameStem: string,
            pixels: Float32Array | Uint16Array,
            width: number,
            height: number,
            mode: 'hu' | 'unit' | 'signed' | 'positive',
            lower: number | null | undefined,
            upper: number | null | undefined,
            caption?: string | null
          ) => {
            if (!pixels?.length || width <= 0 || height <= 0) {
              return;
            }
            reportImages.push({
              title,
              caption: caption ?? null,
              filenameStem,
              width,
              height,
              pixels,
              mode,
              lower,
              upper,
            });
          };
          for (const label of reportAttemptLabels) {
            const attempt = attemptLookupByLabel.get(label);
            if (!attempt) {
              continue;
            }
            appendReportImage(
              `Attempt Thumbnail: ${label}`,
              `${label}-thumbnail`,
              attempt.result.pixelData,
              attempt.result.width,
              attempt.result.height,
              'hu',
              attempt.voi?.lower ?? attempt.result.minValue,
              attempt.voi?.upper ?? attempt.result.maxValue,
              [
                `checksum=${attempt.outputSignature.checksum ?? 'na'}`,
                `score=${formatReadablePanoValue(attempt.qualityScore, 2)}`,
                `displayScore=${formatReadablePanoValue(
                  displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.displaySelectionScore,
                  2
                )}`,
                `supportP50=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.stability.supportConfidenceP50,
                  3
                )}`,
                `bg05=${formatReadablePanoValue(attempt.summary?.backgroundOutlierFraction05, 3)}`,
                `bg10=${formatReadablePanoValue(attempt.summary?.backgroundOutlierFraction10, 3)}`,
                `blackClip=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.background.blackClipFraction,
                  3
                )}`,
                `displayable=${
                  degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.displayable
                    ? 'yes'
                    : 'no'
                }`,
                `preferredTightRootFamily=${
                  displaySelectionBreakdownByAttemptLabel.get(attempt.label)?.preferredTightRootFamily
                    ? 'yes'
                    : 'no'
                }`,
                `rejects=${attempt.qualityGateRejectReasons.join(', ') || 'none'}`,
                attemptLookupByLabel.get(label)?.outputSignature.checksum !== null &&
                reportRows.find(row => row.label === label)?.duplicateOfLabel
                  ? `duplicateOf=${reportRows.find(row => row.label === label)?.duplicateOfLabel}`
                  : null,
              ]
                .filter(Boolean)
                .join('\n')
            );
          }
          const sidecarAttemptSpecs = [
            {
              prefix: 'selected-gpu',
              title: `Selected GPU Winner: ${topGpuAttempt?.label ?? 'none'}`,
              attempt: topGpuAttempt,
            },
            {
              prefix: 'cpu-fallback',
              title: `CPU Virtual Pano Fallback: ${topCpuVirtualPanoAttempt?.label ?? 'none'}`,
              attempt: topCpuVirtualPanoAttempt,
            },
          ];
          const sidecarMapSpecs: Array<{
            key: keyof NonNullable<PanoImagePayload['debugMaps']>;
            mode: 'hu' | 'unit' | 'signed' | 'positive';
            title: string;
          }> = [
            { key: 'supportConfidenceMap', mode: 'unit', title: 'supportConfidenceMap' },
            { key: 'supportDepthMap', mode: 'signed', title: 'supportDepthMap' },
            {
              key: 'backgroundTroughNarrowGateMap',
              mode: 'unit',
              title: 'backgroundTroughNarrowGateMap',
            },
            { key: 'totalAttenuationMap', mode: 'positive', title: 'totalAttenuationMap' },
            { key: 'toneResponseMap', mode: 'unit', title: 'toneResponseMap' },
          ];
          for (const spec of sidecarAttemptSpecs) {
            const attempt = spec.attempt;
            if (!attempt?.result.debugMaps) {
              continue;
            }
            for (const mapSpec of sidecarMapSpecs) {
              const pixels = attempt.result.debugMaps?.[mapSpec.key];
              if (!pixels) {
                continue;
              }
              appendReportImage(
                `${spec.title} ${mapSpec.title}`,
                `${spec.prefix}-${attempt.label}-${String(mapSpec.key)}`,
                pixels,
                attempt.result.width,
                attempt.result.height,
                mapSpec.mode,
                mapSpec.mode === 'unit'
                  ? 0
                  : mapSpec.mode === 'positive'
                    ? 0
                    : null,
                mapSpec.mode === 'unit'
                  ? 1
                  : mapSpec.mode === 'positive'
                    ? null
                    : null,
                [
                  `label=${attempt.label}`,
                  `checksum=${attempt.outputSignature.checksum ?? 'na'}`,
                  `backend=${attempt.resolvedBackend}`,
                  `reconstructionMode=${attempt.reconstructionMode}`,
                ].join('\n')
              );
            }
          }
          downloadCprDebugArtifacts({
            title: `CPR Attempt Debug Report ${debugRunId}`,
            runId: debugRunId,
            rows: reportRows,
            images: reportImages,
          });
        } catch (artifactError) {
          console.warn(
            `[CPR][${debugRunId}] Failed to export CPR attempt debug artifacts.`,
            artifactError
          );
        }
      }

      if (selectedAttempt.intensityDomain === 'unknown') {
        console.warn(
          `[CPR][${debugRunId}] Selected panoramic attempt remains in an unknown intensity domain.`,
          {
            label: selectedAttempt.label,
            qualityBase: selectedAttempt.qualityBase,
            qualityScore: selectedAttempt.qualityScore,
            intensityDomain: selectedAttempt.intensityDomain,
          }
        );
      }

      if (!selectedQualityGateCandidate.pass && !CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK) {
        const selectedAttemptDisplayDiagnostic =
          degradedPreviewCandidateByAttemptLabel.get(selectedAttempt.label) ?? null;
        const selectedAttemptDisplayable =
          selectedAttemptDisplayDiagnostic?.displayable ?? selectedAttempt.qualityGatePassed;
        const selectedAttemptCatastrophicReasons =
          selectedAttemptDisplayDiagnostic?.catastrophicReasons ?? [];
        const degradedPreviewRejectReasons =
          degradedPreviewFallbackChoice?.candidate.rejectReasons ?? selectedQualityGateCandidate.rejectReasons;
        const qualityGateFallbackReason = canPreservePreviousPanoDisplay
          ? 'quality-gate-blocked-keeping-current-accepted-pano'
          : shouldDisplayPhase2DiagnosticDraft
            ? 'quality-gate-failed-displaying-primary-diagnostic-draft'
          : selectedAttemptDisplayable
            ? 'quality-gate-failed-displaying-ranked-top'
            : 'quality-gate-blocked-ranked-top-not-displayable';
        const qualityGateFallbackMessage = canPreservePreviousPanoDisplay
          ? `[CPR] Quality gate rejected the new panoramic output. Keeping the current accepted pano. Reject reasons: ${formatPhase4RejectReasonsForMessage(
              selectedQualityGateCandidate.rejectReasons
            )}.`
          : shouldDisplayPhase2DiagnosticDraft
            ? `[CPR] Quality gate rejected the panoramic output. Displaying the primary virtual pano draft from "${phase2BaseAttempt?.label ?? selectedAttempt.label}" because no previously accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                selectedQualityGateCandidate.rejectReasons
              )}.`
          : selectedAttemptDisplayable
            ? `[CPR] Quality gate rejected the panoramic output. Displaying best-ranked pano from "${selectedAttempt.label}" because no previously accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                selectedQualityGateCandidate.rejectReasons
              )}.`
            : `[CPR] Quality gate rejected the panoramic output and the best-ranked pano "${selectedAttempt.label}" is not displayable. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                selectedQualityGateCandidate.rejectReasons
              )}. Catastrophic reasons: ${
                selectedAttemptCatastrophicReasons.join(', ') || 'none'
              }.`;
        const qualityGateCanContinue =
          canPreservePreviousPanoDisplay ||
          selectedAttemptDisplayable ||
          shouldDisplayPhase2DiagnosticDraft;

        if (!qualityGateCanContinue) {
          console.warn(`[CPR][${debugRunId}] ${qualityGateFallbackMessage}`, {
            selectedAttemptLabel: selectedAttempt.label,
            selectedHardRejectReason: selectedAttempt.hardRejectReason,
            selectionReason: qualityGateSelectionReason,
            selectedAttemptDisplayable,
            selectedAttemptCatastrophicReasons,
            preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
            preservedPreviousPanoImageId,
            qualityGateRejectReasons: selectedQualityGateCandidate.rejectReasons,
            catastrophicCandidateCount: degradedPreviewCatastrophicOptions.length,
          });
          console.log(
            '[CPR-QUALITY-GATE-BLOCKED-JSON]',
            JSON.stringify({
              runId: debugRunId,
              reason: qualityGateFallbackReason,
              message: qualityGateFallbackMessage,
              selectedAttemptLabel: selectedAttempt.label,
              selectedHardRejectReason: selectedAttempt.hardRejectReason,
              selectionReason: qualityGateSelectionReason,
              selectedAttemptDisplayable,
              selectedAttemptCatastrophicReasons,
              displayingPhase2DiagnosticDraft: shouldDisplayPhase2DiagnosticDraft,
              preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
              preservedPreviousPanoImageId,
              qualityGate: qualityGateFallbackSummary,
            })
          );
          console.log(
            '[CPR-RECON-FALLBACK-JSON]',
            JSON.stringify({
              runId: debugRunId,
              displayedPath: 'worker-recon',
              backend: selectedAttempt.resolvedBackend,
              requestedBackend: selectedAttempt.requestedBackend,
              pipelineMode: selectedAttempt.pipelineMode,
              reconstructionMode: selectedAttempt.reconstructionMode,
              sourceVolumeId,
              referencePathAvailable: true,
              displaySource: 'quality-gate-blocked-before-stage-switch',
              fallbackReason: qualityGateFallbackReason,
              phase2GatePassed: selectedAttemptRouteDiagnostic.phase2GatePassed,
              displayedOutputMode: selectedDisplayedOutputMode,
              displayedSourceLabel: selectedDisplayedSourceLabel,
              displayedSourceAggregation: selectedDisplayedSourceAggregation,
              selectedAttemptLabel: selectedAttempt.label,
              temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
              selectedAttemptHardRejectReason: selectedAttempt.hardRejectReason,
              qualityGate: qualityGateFallbackSummary,
              phase1VirtualPano,
              phase2VirtualPano,
            })
          );
          setAxialTransitioning(false, qualityGateFallbackReason);
          restoreAllCprViewportVisibility(servicesManager);
          if (isMountedRef.current) {
            setWarning(null);
            setError(qualityGateFallbackMessage);
          }
          markGenerationIdle();
          return;
        }

        console.warn(`[CPR][${debugRunId}] ${qualityGateFallbackMessage}`, {
          selectedAttemptLabel: selectedAttempt.label,
          selectedHardRejectReason: selectedAttempt.hardRejectReason,
          selectionReason: qualityGateSelectionReason,
          selectedAttemptDisplayable,
          selectedAttemptCatastrophicReasons,
          displayingPhase2DiagnosticDraft: shouldDisplayPhase2DiagnosticDraft,
          preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
          preservedPreviousPanoImageId,
          degradedPreviewAttemptLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
          qualityGateRejectReasons: selectedQualityGateCandidate.rejectReasons,
        });
        console.log(
          '[CPR-QUALITY-GATE-RANKED-TOP-JSON]',
          JSON.stringify({
            runId: debugRunId,
            reason: qualityGateFallbackReason,
            message: qualityGateFallbackMessage,
            selectedAttemptLabel: selectedAttempt.label,
            selectedHardRejectReason: selectedAttempt.hardRejectReason,
            selectedAttemptDisplayable,
            selectedAttemptCatastrophicReasons,
            degradedPreviewAttemptLabel: degradedPreviewFallbackChoice?.attempt.label ?? null,
            degradedPreviewRejectReasons,
            selectionReason: qualityGateSelectionReason,
            preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
            preservedPreviousPanoImageId,
            qualityGate: qualityGateFallbackSummary,
          })
        );
        if (isMountedRef.current) {
          setError(null);
          setWarning(qualityGateFallbackMessage);
        }
      }

      if (
        isLikelyPoorPanoQuality(
          selectedAttempt.summary,
          selectedAttempt.qualityGateMetrics.renderSupportMode
        )
      ) {
        console.warn(
          `[CPR][${debugRunId}] proceeding with best available pano despite quality warning`,
          {
            label: selectedAttempt.label,
            qualityBase: selectedAttempt.qualityBase,
            qualityScore: selectedAttempt.qualityScore,
            summary: selectedAttempt.summary,
          }
        );
      }

      const displayedAttempt = temporarilyPinnedDisplayedAttempt ?? selectedAttempt;
      const displayedQualityGateCandidate = selectedQualityGateCandidate;
      const usingDegradedPreviewFallback = false;
      const displayingQcFailedRankedTop =
        !temporarilyPinnedDisplayedAttempt &&
        shouldPreservePreviousPanoDisplay &&
        !canPreservePreviousPanoDisplay &&
        !displayedQualityGateCandidate.pass;
      let panoWorkerResult = displayedAttempt.result;
      let panoDebugSummary = displayedAttempt.summary;
      let adaptiveVoi = displayedAttempt.voi;
      let displayedOutputMode: 'legacy' | 'workerGpuPhase2' | 'virtualPanoPhase2' =
        selectedDisplayedOutputMode;
      let displayedSourceLabel = displayedAttempt.label;
      let displayedSourceAggregation = displayedAttempt.aggregation;
      let selectedPanoWidth = displayedAttempt.panoWidth;
      let selectedPanoHeight = displayedAttempt.panoHeight;
      let selectedActualVertHalfMm = displayedAttempt.actualVertHalfMm;
      let selectedColumnPixelSpacing = displayedAttempt.columnPixelSpacing;
      let selectedRowPixelSpacing = displayedAttempt.rowPixelSpacing;
      if (
        !usingDegradedPreviewFallback &&
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
      const displayedWorkerDebugPayload = phase2VirtualPano.usedAsDisplayedOutput
        ? phase2WorkerResult?.workerDebugPayload ?? null
        : displayedAttempt.workerDebugPayload ?? null;
      const displayedRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
        displayedWorkerDebugPayload
      );
      const displayedWorkerDiagnostic =
        displayedWorkerDebugPayload &&
        typeof displayedWorkerDebugPayload === 'object' &&
        displayedWorkerDebugPayload.diagnostic &&
        typeof displayedWorkerDebugPayload.diagnostic === 'object'
          ? (displayedWorkerDebugPayload.diagnostic as Record<string, unknown>)
          : null;
      const selectedLocalCenterOffsetsMm = Array.isArray(
        displayedWorkerDiagnostic?.localCenterOffsetsMm
      )
        ? displayedWorkerDiagnostic.localCenterOffsetsMm
            .map(value => Number(value))
            .filter(value => Number.isFinite(value))
        : [];
      const crossSectionVerticalCenterOffsetMm =
        (phase2VirtualPano.usedAsDisplayedOutput
          ? phase2BaseAttempt?.verticalCenterOffsetMm
          : displayedAttempt.verticalCenterOffsetMm) ?? 0;
      console.log(
        '[CPR-CROSSSECTION-GEOMETRY-JSON]',
        JSON.stringify({
          runId: debugRunId,
          displayedOutputMode,
          correctedFrameCount: frames.length,
          sourceFrameCount: frames.length,
          usingWorkerLocalCenterOffsets: selectedLocalCenterOffsetsMm.length === frames.length,
          crossSectionVerticalCenterOffsetMm,
          localCenterOffsetsFirst8: selectedLocalCenterOffsetsMm.slice(0, 8),
          geometryPolicy: 'worker-offsets-render-only',
        })
      );
      const displayedWorkerTimingMs =
        displayedWorkerDiagnostic?.timingMs &&
        typeof displayedWorkerDiagnostic.timingMs === 'object'
          ? (displayedWorkerDiagnostic.timingMs as Record<string, unknown>)
          : phase2VirtualPano.usedAsDisplayedOutput
            ? phase2VirtualPano.timingMs
            : displayedAttempt.workerTimingMs ?? null;
      const displayedOutputSignature = extractCprOutputSignature(displayedWorkerDebugPayload);
      const rankedTopOutputSignature = rankedTopAttempt.outputSignature;
      const phase2OutputSignature =
        phase2WorkerResult?.workerDebugPayload
          ? extractCprOutputSignature(phase2WorkerResult.workerDebugPayload)
          : null;
      const rankedTopOutputSignatureKey = buildCprOutputSignatureKey(rankedTopOutputSignature);
      const selectedOutputSignatureKey = buildCprOutputSignatureKey(selectedAttempt.outputSignature);
      const displayedOutputSignatureKey = buildCprOutputSignatureKey(displayedOutputSignature);
      const phase2OutputSignatureKey = phase2OutputSignature
        ? buildCprOutputSignatureKey(phase2OutputSignature)
        : null;
      console.log(
        '[CPR-DISPLAY-SIGNATURE-COMPARISON-JSON]',
        JSON.stringify({
          runId: debugRunId,
          rankedTop: {
            label: rankedTopAttempt.label,
            checksum: rankedTopOutputSignature.checksum,
            signatureKey: rankedTopOutputSignatureKey,
          },
          selected: {
            label: selectedAttempt.label,
            checksum: selectedAttempt.outputSignature.checksum,
            signatureKey: selectedOutputSignatureKey,
          },
          displayed: {
            label: phase2VirtualPano.usedAsDisplayedOutput
              ? displayedSourceLabel
              : displayedAttempt.label,
            checksum: displayedOutputSignature.checksum,
            signatureKey: displayedOutputSignatureKey,
            outputMode: displayedOutputMode,
          },
          phase2VirtualPano: phase2OutputSignature
            ? {
                label: phase2BaseAttempt?.label ?? null,
                checksum: phase2OutputSignature.checksum,
                signatureKey: phase2OutputSignatureKey,
                workerAcceptedForOutput: phase2VirtualPano.workerAcceptedForOutput,
                orchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
                displayDecisionReason: phase2VirtualPano.displayDecisionReason,
              }
            : null,
          comparisons: {
            selectedMatchesRankedTop:
              !!selectedOutputSignatureKey &&
              !!rankedTopOutputSignatureKey &&
              selectedOutputSignatureKey === rankedTopOutputSignatureKey,
            displayedMatchesSelected:
              !!displayedOutputSignatureKey &&
              !!selectedOutputSignatureKey &&
              displayedOutputSignatureKey === selectedOutputSignatureKey,
            phase2MatchesDisplayed:
              !!displayedOutputSignatureKey &&
              !!phase2OutputSignatureKey &&
              displayedOutputSignatureKey === phase2OutputSignatureKey,
            selectedVsRankedTopChecksumDelta:
              selectedAttempt.outputSignature.checksum !== null &&
              rankedTopOutputSignature.checksum !== null
                ? Math.round(
                    (selectedAttempt.outputSignature.checksum -
                      rankedTopOutputSignature.checksum) *
                      1000
                  ) / 1000
                : null,
            displayedVsSelectedChecksumDelta:
              displayedOutputSignature.checksum !== null &&
              selectedAttempt.outputSignature.checksum !== null
                ? Math.round(
                    (displayedOutputSignature.checksum -
                      selectedAttempt.outputSignature.checksum) *
                      1000
                  ) / 1000
                : null,
          },
        })
      );
      console.log(
        '[CPR-PHASE0-RUN-SUMMARY-JSON]',
        JSON.stringify({
          runId: debugRunId,
          sourceVolumeId,
          attemptExecution: {
            attemptCount: launchedAttemptCount,
            mipFallbackCount: launchedMipFallbackCount,
            totalDurationMs: Math.round(totalAttemptDurationMs),
            earlyExitReason,
          },
          phase1VirtualPano: {
            executed: phase1VirtualPano.executed,
            skippedReason: phase1VirtualPano.skippedReason,
            error: phase1VirtualPano.error,
          },
          phase2VirtualPano: {
            executed: phase2VirtualPano.executed,
            usedAsDisplayedOutput: phase2VirtualPano.usedAsDisplayedOutput,
            skippedReason: phase2VirtualPano.skippedReason,
            error: phase2VirtualPano.error,
          },
          rankedTop: {
            label: selectedAttempt.label,
            requestedBackend: selectedAttempt.requestedBackend,
            resolvedBackend: selectedAttempt.resolvedBackend,
            pipelineMode: selectedAttempt.pipelineMode,
            reconstructionMode: selectedAttempt.reconstructionMode,
            qualityGatePassed: selectedAttempt.qualityGatePassed,
            qualityScore: selectedAttempt.qualityScore,
            qualityBase: selectedAttempt.qualityBase,
            outputSignature: selectedAttempt.outputSignature,
          },
          displayed: {
            label: phase2VirtualPano.usedAsDisplayedOutput
              ? displayedSourceLabel
              : displayedAttempt.label,
            sourceLabel: displayedSourceLabel,
            displayedOutputMode,
            requestedBackend: displayedRouteDiagnostic.requestedBackend,
            resolvedBackend: displayedRouteDiagnostic.backend,
            pipelineMode: displayedRouteDiagnostic.pipelineMode,
            fallbackReason: displayedRouteDiagnostic.fallbackReason,
            phase2GatePassed: displayedRouteDiagnostic.phase2GatePassed,
            reconstructionMode:
              toNonEmptyString(displayedWorkerDiagnostic?.reconstructionMode) ??
              displayedOutputMode,
            qualityStatus: displayingQcFailedRankedTop ? 'degraded' : 'accepted',
            qualityGatePassed: displayedQualityGateCandidate.pass,
            qualityGateSelectionReason,
            qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
            workerTimingMs: displayedWorkerTimingMs,
            outputSignature: displayedOutputSignature,
          },
        })
      );
      console.log(`[CPR][${debugRunId}] selected pano candidate`, {
        label: phase2VirtualPano.usedAsDisplayedOutput ? displayedSourceLabel : displayedAttempt.label,
        displayedAttemptLabel: displayedAttempt.label,
        temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
        usingDegradedPreviewFallback,
        displayingQcFailedRankedTop,
        displayedOutputMode,
        displayedSourceLabel,
        displayedSourceAggregation,
        displayedPath: 'worker-recon',
        displayedBackend: displayedRouteDiagnostic.backend,
        displayedPipelineMode: displayedRouteDiagnostic.pipelineMode,
        fallbackReason: displayedRouteDiagnostic.fallbackReason,
        phase2GatePassed: displayedRouteDiagnostic.phase2GatePassed,
        qualityGateSelectionReason,
        qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
        qualityGatePassed: displayedQualityGateCandidate.pass,
        qualityStatus: displayingQcFailedRankedTop ? 'degraded' : 'accepted',
        preservePreviousPanoRequested: shouldPreservePreviousPanoDisplay,
        preservePreviousPanoAvailable: canPreservePreviousPanoDisplay,
        preservedPreviousPanoImageId,
        selectedSupportRiskLevel: displayedQualityGateCandidate.supportSurfaceRiskSummary.riskLevel,
        selectedSupportRiskFlags: displayedQualityGateCandidate.supportSurfaceRiskSummary.riskFlags,
        selectedSupportBaselineFingerprint:
          displayedQualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
        qualityBase: displayedAttempt.qualityBase,
        qualityScore: displayedAttempt.qualityScore,
        hardRejectReason: displayedAttempt.hardRejectReason,
        aggregation: displayedAttempt.aggregation,
        intensityDomain: displayedAttempt.intensityDomain,
        huDomain: displayedAttempt.huDomain,
        convertedToHu: displayedAttempt.convertedToHu,
        rescaleSkippedAsUnsafe: displayedAttempt.rescaleSkippedAsUnsafe,
        panoWidth: selectedPanoWidth,
        panoHeight: selectedPanoHeight,
        actualVertHalfMm: selectedActualVertHalfMm,
        verticalCenterOffsetMm: displayedAttempt.verticalCenterOffsetMm,
        slabHalfThicknessMm: displayedAttempt.slabHalfThicknessMm,
        slabSamples: displayedAttempt.slabSamples,
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
          backgroundToneSampleCount: panoDebugSummary.backgroundToneSampleCount,
          backgroundToneP95: panoDebugSummary.backgroundToneP95,
          backgroundToneP99: panoDebugSummary.backgroundToneP99,
          backgroundToneMax: panoDebugSummary.backgroundToneMax,
          backgroundOutlierFraction05: panoDebugSummary.backgroundOutlierFraction05,
          backgroundOutlierFraction10: panoDebugSummary.backgroundOutlierFraction10,
          detailBandHorizontalEdgeMean: panoDebugSummary.detailBandHorizontalEdgeMean,
          detailBandVerticalEdgeMean: panoDebugSummary.detailBandVerticalEdgeMean,
          fractionBelowMinus950: panoDebugSummary.fractionBelowMinus950,
          fractionAbove3000: panoDebugSummary.fractionAbove3000,
          sampledCount: panoDebugSummary.sampledCount,
          configuredVoiLower: adaptiveVoi.lower,
          configuredVoiUpper: adaptiveVoi.upper,
        });

        if (
          panoDebugSummary.fractionAbove3000 > 0.6 ||
          panoDebugSummary.fractionBelowMinus950 > 0.6
        ) {
          console.warn(
            `[CPR][${debugRunId}] Majority of pano samples are outside reference band [-950, 3000]. ` +
              `Adaptive VOI is [${adaptiveVoi.lower.toFixed(0)}, ${adaptiveVoi.upper.toFixed(0)}].`
          );
        }
      }

      console.log(
        `[CPR-SPLINE-PRESERVE] run=${debugRunId} mode=worker-recon annotationUID=${
          latestAnnotationUID || 'none'
        } skipping pre-transition spline cleanup`
      );
      cprStateService.setArchData(
        controlPoints,
        frames,
        sourceVolumeId,
        latestAnnotationUID,
        crossSectionVerticalCenterOffsetMm,
        selectedLocalCenterOffsetsMm
      );

      clearProtocolListener();

      let protocolAppliedHandled = false;
      const onProtocolApplied = async () => {
        if (protocolAppliedHandled) {
          return;
        }
        protocolAppliedHandled = true;

        clearProtocolListener();

        try {
          scheduleAxialViewportRestore(
            debugRunId,
            latestAnnotationUID,
            latestAnnotationSnapshot,
            preservedAxialCamera,
            'legacy-after-stage-switch'
          );

          if (
            panoWorkerResult.pixelData.length !==
            panoWorkerResult.width * panoWorkerResult.height
          ) {
            throw new Error('Dimension mismatch');
          }

          const preservedPayloadForDisplay = canPreservePreviousPanoDisplay
            ? preservedPreviousPanoPayload
            : null;
          const displayingDegradedPanoPreview = false;
          const qualityStatusForDisplay: PanoQualityStatus = displayingQcFailedRankedTop
            ? 'degraded'
            : 'accepted';
          const qualityMessageForDisplay = displayingQcFailedRankedTop
            ? `[CPR] Displaying best-ranked pano from "${displayedAttempt.label}" because all candidates failed QC and no previously accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                displayedQualityGateCandidate.rejectReasons
              )}.`
            : null;
          const payloadForDisplay = preservedPayloadForDisplay ?? {
            pixelData: panoWorkerResult.pixelData,
            meanMap: panoWorkerResult.meanMap,
            maxMap: panoWorkerResult.maxMap,
            sampleCountMap: panoWorkerResult.sampleCountMap,
            debugMaps: panoWorkerResult.debugMaps,
            probeContext: {
              runId: debugRunId,
              displayedPath: 'worker-recon',
              backend: displayedRouteDiagnostic.backend,
              requestedBackend: displayedRouteDiagnostic.requestedBackend,
              pipelineMode: displayedRouteDiagnostic.pipelineMode,
              reconstructionMode:
                toNonEmptyString(displayedWorkerDiagnostic?.reconstructionMode) ??
                displayedOutputMode,
              qualityStatus: qualityStatusForDisplay,
              qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
              qualityGateSelectionReason: qualityGateSelectionReason,
              qualityGateMessage: qualityMessageForDisplay,
            },
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            qualityGatePassed: displayedQualityGateCandidate.pass,
            qualityStatus: qualityStatusForDisplay,
            qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
            qualityGateSelectionReason: qualityGateSelectionReason,
            qualityGateMessage: qualityMessageForDisplay,
            huDomain: displayedAttempt.huDomain,
            intensityDomain: displayedAttempt.intensityDomain,
            windowWidth: adaptiveVoi.windowWidth,
            windowCenter: adaptiveVoi.windowCenter,
            slope: panoWorkerResult.slope,
            intercept: panoWorkerResult.intercept,
            columnPixelSpacing: selectedColumnPixelSpacing,
            rowPixelSpacing: selectedRowPixelSpacing,
          };
          const adaptiveVoiForDisplay =
            preservedPayloadForDisplay
              ? preservedPreviousPanoVoi ??
                createPanoVoiFromWindowLevel(
                  Number(payloadForDisplay.windowWidth),
                  Number(payloadForDisplay.windowCenter)
                ) ??
                adaptiveVoi
              : adaptiveVoi;
          if (preservedPayloadForDisplay) {
            console.warn(
              `[CPR][${debugRunId}] preserving previously displayed pano because the new worker output failed the quality gate`,
              {
                selectionReason: qualityGateSelectionReason,
                selectedAttemptLabel: selectedAttempt.label,
                selectedHardRejectReason: selectedAttempt.hardRejectReason,
                preservedPreviousPanoImageId,
              }
            );
            console.log(
              '[CPR-PRESERVE-PREVIOUS-PANO-JSON]',
              JSON.stringify({
                runId: debugRunId,
                selectionReason: qualityGateSelectionReason,
                selectedAttemptLabel: selectedAttempt.label,
                selectedHardRejectReason: selectedAttempt.hardRejectReason,
                preservedPreviousPanoImageId,
                selectedDespiteGateFailure: !selectedQualityGateCandidate.pass,
              })
            );
          }
          if (displayingQcFailedRankedTop) {
            console.warn(
              `[CPR][${debugRunId}] displaying best-ranked pano despite QC failure because no previously accepted pano is available`,
              {
                selectionReason: qualityGateSelectionReason,
                selectedAttemptLabel: displayedAttempt.label,
                selectedHardRejectReason: displayedAttempt.hardRejectReason,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
              }
            );
            console.log(
              '[CPR-DISPLAY-QC-FAILED-RANKED-TOP-JSON]',
              JSON.stringify({
                runId: debugRunId,
                selectionReason: qualityGateSelectionReason,
                selectedAttemptLabel: displayedAttempt.label,
                selectedHardRejectReason: displayedAttempt.hardRejectReason,
                selectedDespiteGateFailure: !displayedQualityGateCandidate.pass,
                qualityStatus: qualityStatusForDisplay,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
                rankedTopLabel: rankedTopAttempt.label,
              })
            );
          }

          clearPanoImageCache();
          setPanoImagePayload(panoImageId, payloadForDisplay);

          console.log(`[CPR][${debugRunId}] Pano payload metadata pushed to loader`, {
            displayedOutputMode,
            panoImageId,
            preservedPreviousPanoDisplay: !!preservedPayloadForDisplay,
            displayingDegradedPanoPreview,
            displayingQcFailedRankedTop,
            qualityGatePassed: payloadForDisplay.qualityGatePassed ?? null,
            qualityStatus: payloadForDisplay.qualityStatus ?? null,
            qualityGateRejectReasons: payloadForDisplay.qualityGateRejectReasons ?? [],
            minValue: payloadForDisplay.minValue,
            maxValue: payloadForDisplay.maxValue,
            intensityDomain: payloadForDisplay.intensityDomain,
            windowWidth: adaptiveVoiForDisplay.windowWidth,
            windowCenter: adaptiveVoiForDisplay.windowCenter,
            voiLower: adaptiveVoiForDisplay.lower,
            voiUpper: adaptiveVoiForDisplay.upper,
            slope: payloadForDisplay.slope,
            intercept: payloadForDisplay.intercept,
            width: payloadForDisplay.width,
            height: payloadForDisplay.height,
            columnPixelSpacing: payloadForDisplay.columnPixelSpacing,
            rowPixelSpacing: payloadForDisplay.rowPixelSpacing,
          });
          console.log(
            '[CPR-LOADER-METADATA-JSON]',
            JSON.stringify({
              runId: debugRunId,
              displayedOutputMode,
              panoImageId,
              preservedPreviousPanoDisplay: !!preservedPayloadForDisplay,
              displayingDegradedPanoPreview,
              displayingQcFailedRankedTop,
              preservedPreviousPanoImageId,
              qualityGatePassed: payloadForDisplay.qualityGatePassed ?? null,
              qualityStatus: payloadForDisplay.qualityStatus ?? null,
              qualityGateRejectReasons: payloadForDisplay.qualityGateRejectReasons ?? [],
              minValue: payloadForDisplay.minValue,
              maxValue: payloadForDisplay.maxValue,
              intensityDomain: payloadForDisplay.intensityDomain,
              windowWidth: adaptiveVoiForDisplay.windowWidth,
              windowCenter: adaptiveVoiForDisplay.windowCenter,
              voiLower: adaptiveVoiForDisplay.lower,
              voiUpper: adaptiveVoiForDisplay.upper,
              slope: payloadForDisplay.slope,
              intercept: payloadForDisplay.intercept,
              width: payloadForDisplay.width,
              height: payloadForDisplay.height,
              columnPixelSpacing: payloadForDisplay.columnPixelSpacing,
              rowPixelSpacing: payloadForDisplay.rowPixelSpacing,
              outputSignature: displayedOutputSignature,
              selectedAttempt: {
                label: displayedAttempt.label,
                aggregation: displayedAttempt.aggregation,
                verticalCenterOffsetMm: displayedAttempt.verticalCenterOffsetMm,
                slabHalfThicknessMm: displayedAttempt.slabHalfThicknessMm,
                slabSamples: displayedAttempt.slabSamples,
                qualityScore: displayedAttempt.qualityScore,
                rankedTopLabel: rankedTopAttempt.label,
                temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
              },
            })
          );

          await ensureViewportTypeByLogicalId(servicesManager, 'cpr-pano', 'stack', debugRunId);
          const panoViewport = await waitForPanoStackViewport(servicesManager);
          setViewportVisibility(panoViewport, true);
          logPanoViewportSnapshot(debugRunId, 'before-setStack', panoViewport);
          await panoViewport.setStack([panoImageId], 0);
          installPanoDebugProbe(debugRunId, panoViewport, panoImageId);
          if (typeof (panoViewport as { resetCamera?: () => void }).resetCamera === 'function') {
            panoViewport.resetCamera();
          }
          logPanoViewportSnapshot(debugRunId, 'after-setStack', panoViewport);
          applyPanoDisplaySettings(debugRunId, 'after-setStack', panoViewport, adaptiveVoiForDisplay);
          logPanoViewportSnapshot(debugRunId, 'after-setProperties', panoViewport);
          panoViewport.render();
          logPanoViewportSnapshot(debugRunId, 'after-render', panoViewport);
          logPanoDisplayRoute({
            runId: debugRunId,
            displayedPath: 'worker-recon',
            backend: displayedRouteDiagnostic.backend,
            requestedBackend: displayedRouteDiagnostic.requestedBackend,
            pipelineMode: displayedRouteDiagnostic.pipelineMode,
            fallbackReason: displayedRouteDiagnostic.fallbackReason,
          });
          logReconModeJson({
            runId: debugRunId,
            displayedPath: 'worker-recon',
            backend: displayedRouteDiagnostic.backend,
            requestedBackend: displayedRouteDiagnostic.requestedBackend,
            pipelineMode: displayedRouteDiagnostic.pipelineMode,
            reconstructionMode: displayedOutputMode,
            displaySource: 'panoImageLoader',
            referencePathAvailable: true,
            sourceVolumeId,
            fallbackReason: displayedRouteDiagnostic.fallbackReason,
            phase2GatePassed: displayedRouteDiagnostic.phase2GatePassed,
          });
          const shouldLogReconFallback =
            displayedRouteDiagnostic.backend !== displayedRouteDiagnostic.requestedBackend ||
            !!displayedRouteDiagnostic.fallbackReason ||
            phase2VirtualPano.usedAsDisplayedOutput ||
            !!phase1VirtualPano.error ||
            !!phase2VirtualPano.error ||
            !displayedQualityGateCandidate.pass;
          if (shouldLogReconFallback) {
            console.log(
              '[CPR-RECON-FALLBACK-JSON]',
              JSON.stringify({
                runId: debugRunId,
                displayedPath: 'worker-recon',
                backend: displayedRouteDiagnostic.backend,
                requestedBackend: displayedRouteDiagnostic.requestedBackend,
                pipelineMode: displayedRouteDiagnostic.pipelineMode,
                reconstructionMode: displayedOutputMode,
                sourceVolumeId,
                referencePathAvailable: true,
                displaySource: 'panoImageLoader',
                fallbackReason: displayedRouteDiagnostic.fallbackReason,
                phase2GatePassed: displayedRouteDiagnostic.phase2GatePassed,
                displayedOutputMode,
                displayedSourceLabel,
                displayedSourceAggregation,
                selectedAttemptLabel: selectedAttempt.label,
                displayedAttemptLabel: displayedAttempt.label,
                temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
                selectedAttemptHardRejectReason: selectedAttempt.hardRejectReason,
                displayingDegradedPanoPreview,
                displayingQcFailedRankedTop,
                preservedPreviousPanoDisplay: !!preservedPayloadForDisplay,
                qualityStatus: qualityStatusForDisplay,
                qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
                selectedSupportSurfaceRiskSummary:
                  displayedQualityGateCandidate.supportSurfaceRiskSummary,
                selectedSupportSurfaceBaselineFingerprint:
                  displayedQualityGateCandidate.supportSurfaceRiskSummary.baselineFingerprint,
                qualityGate: qualityGateFallbackSummary,
                phase1VirtualPano,
                phase2VirtualPano,
              })
            );
          }
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
                adaptiveVoiForDisplay
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
                adaptiveVoiForDisplay
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
                selectedLocalCenterOffsetsMm,
                selectedLocalCenterOffsetsMm.length === frames.length,
                servicesManager
              )
                .then(resolve)
                .catch(reject);
            }, 50);
          });
        } catch (innerErr) {
          const msg = innerErr instanceof Error ? innerErr.message : String(innerErr);
          console.log(
            '[CPR-RECON-FALLBACK-JSON]',
            JSON.stringify({
              runId: debugRunId,
              displayedPath: 'worker-recon',
              backend: displayedRouteDiagnostic.backend,
              requestedBackend: displayedRouteDiagnostic.requestedBackend,
              pipelineMode: displayedRouteDiagnostic.pipelineMode,
              reconstructionMode: displayedOutputMode,
              sourceVolumeId,
              referencePathAvailable: true,
              displaySource: 'panoImageLoader',
              fallbackReason: `display-attach-failed:${msg}`,
              phase2GatePassed: displayedRouteDiagnostic.phase2GatePassed,
              displayedOutputMode,
              displayedSourceLabel,
              displayedSourceAggregation,
              selectedAttemptLabel: selectedAttempt.label,
              displayedAttemptLabel: displayedAttempt.label,
              temporaryDisplayPinned: !!temporarilyPinnedDisplayedAttempt,
              selectedAttemptHardRejectReason: selectedAttempt.hardRejectReason,
              qualityGate: qualityGateFallbackSummary,
              phase1VirtualPano,
              phase2VirtualPano,
            })
          );
          console.error(`[CPR][${debugRunId}] Pipeline failed after HP switch:`, msg);
          clearPendingAxialViewportRestore('legacy-stage1-failed');
          setAxialTransitioning(false, 'legacy-stage1-failed');
          restoreAllCprViewportVisibility(servicesManager);
          if (isMountedRef.current) {
            setError(msg);
          }
        } finally {
          markGenerationIdle();
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
        setAxialTransitioning(false, 'legacy-stage1-timeout');
        if (isMountedRef.current) {
          const timeoutMsg =
            '[CPR] Timed out waiting for hanging protocol change after stage switch.';
          console.error(`[CPR][${debugRunId}] ${timeoutMsg}`);
          setError(timeoutMsg);
        }
        markGenerationIdle();
      }, 12000);

      console.log(`[CPR][${debugRunId}] switching to CPR stage 1`);
      commandsManager.runCommand('setHangingProtocol', {
        protocolId: 'cpr',
        stageIndex: 1,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[CPR][${debugRunId}] onDone failed:`, msg);
      setAxialTransitioning(false, 'onDone-failed');
      if (isMountedRef.current) {
        setError(msg);
      }
      markGenerationIdle();
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
    clearPendingAxialViewportRestore,
    clearPanoCameraBridge,
    clearProtocolListener,
    initializePanoCameraBridge,
    markGenerationIdle,
    disposeAttachedVtkPano,
    scheduleAxialViewportRestore,
    setHostedPanoVoiAuthority,
    setAxialTransitioning,
  ]);

  const onRedraw = useCallback(async () => {
    clearCprRuntimeState('redraw');
    await terminateCPRWorkerSession();

    try {
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
  }, [commandsManager, clearCprRuntimeState]);

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

    const verticalCenterOffsetMm =
      cprStateService.getCrossSectionVerticalCenterOffsetMm(clampedIndex);
    setCrossSectionForFrame(frame, servicesManager, verticalCenterOffsetMm);
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

  useEffect(() => {
    if (CPR_PANO_DISPLAY_PATH_DEFAULT !== 'vtk-reference') {
      return;
    }

    const onPanoHostAttached = () => {
      if (!cprStateService.hasData()) {
        return;
      }

      window.setTimeout(() => {
        if (!isMountedRef.current) {
          return;
        }

        const liveHostedPano = getLiveHostedPano().hostedPano;
        if (liveHostedPano) {
          attachedVtkPanoRef.current = liveHostedPano;
        }
        rearmPanoCameraBridgeFromLiveViewport('pano-host-attached');
      }, 0);
    };

    cornerstone.eventTarget.addEventListener(CPR_PANO_HOST_ATTACHED_EVENT, onPanoHostAttached);

    return () => {
      cornerstone.eventTarget.removeEventListener(CPR_PANO_HOST_ATTACHED_EVENT, onPanoHostAttached);
    };
  }, [getLiveHostedPano, rearmPanoCameraBridgeFromLiveViewport]);

  useEffect(() => {
    if (CPR_PANO_DISPLAY_PATH_DEFAULT !== 'vtk-reference' || typeof document === 'undefined') {
      return;
    }

    const { cornerstoneViewportService } = servicesManager.services;

    const onVoiModified = (evt: Event) => {
      const detail = (
        evt as CustomEvent<{
          viewportId?: string;
          volumeId?: string;
          range?: { lower?: number; upper?: number };
        }>
      ).detail;
      const viewportId = detail?.viewportId;
      if (!viewportId) {
        return;
      }

      const logicalViewportId =
        cornerstoneViewportService.getViewportInfo(viewportId)?.viewportOptions?.viewportId ||
        viewportId;
      if (logicalViewportId !== 'cpr-pano' && logicalViewportId !== 'cpr-crosssection') {
        return;
      }

      const attachedPano = getLiveHostedPano().hostedPano || attachedVtkPanoRef.current;
      if (logicalViewportId === 'cpr-pano') {
        console.log(
          `DIAG-TRIPWIRE: voi-modified logicalViewportId=${logicalViewportId} viewportId=${viewportId} attachedPano=${
            attachedPano ? 'yes' : 'no'
          } range=${JSON.stringify(detail?.range ?? null)}`
        );
      }
      if (!attachedPano) {
        return;
      }

      const nextVoi = createPanoVoiFromRange(detail?.range);
      if (!nextVoi) {
        return;
      }

      const authority = hostedPanoVoiAuthorityRef.current;
      const authoritativeVoi = clonePanoVoiSettings(authority.authoritativeVoi);
      if (
        logicalViewportId === 'cpr-pano' &&
        authority.sourceVolumeId &&
        detail?.volumeId &&
        detail.volumeId !== authority.sourceVolumeId
      ) {
        console.warn(
          `[CPR-VTK-PANO-VOI-GUARD] run=${authority.runId || 'na'} rejectingPanoEvent reason=volume-mismatch ` +
            `eventVolumeId=${detail.volumeId} sourceVolumeId=${authority.sourceVolumeId}`
        );
        return;
      }

      if (logicalViewportId === 'cpr-pano') {
        const evaluation = evaluateHostedPanoViewportVoiUpdate({
          nextVoi,
          authoritativeVoi,
          suppressPanoViewportEventsUntil: authority.suppressPanoViewportEventsUntil,
          now: Date.now(),
        });
        if (!evaluation.accept) {
          const authoritySummary = authoritativeVoi
            ? ` authorityWW=${authoritativeVoi.windowWidth.toFixed(2)} authorityWC=${authoritativeVoi.windowCenter.toFixed(
                2
              )}`
            : '';
          console.warn(
            `[CPR-VTK-PANO-VOI-GUARD] run=${authority.runId || 'na'} rejectingPanoEvent reason=${evaluation.reason} ` +
              `eventWW=${nextVoi.windowWidth.toFixed(2)} eventWC=${nextVoi.windowCenter.toFixed(2)}${authoritySummary}`
          );
          if (evaluation.shouldRepair) {
            scheduleHostedPanoVoiRepair(evaluation.reason);
          }
          return;
        }
      }

      attachedPano.updateWindowLevel(nextVoi.windowWidth, nextVoi.windowCenter);
      setHostedPanoVoiAuthority({
        runId: authority.runId || 'na',
        sourceVolumeId: authority.sourceVolumeId || detail?.volumeId || null,
        voi: nextVoi,
        source: logicalViewportId,
      });
    };

    document.addEventListener(cornerstone.Enums.Events.VOI_MODIFIED, onVoiModified, true);

    return () => {
      document.removeEventListener(cornerstone.Enums.Events.VOI_MODIFIED, onVoiModified, true);
    };
  }, [getLiveHostedPano, scheduleHostedPanoVoiRepair, servicesManager, setHostedPanoVoiAuthority]);

  useEffect(() => {
    if (CPR_PANO_DISPLAY_PATH_DEFAULT !== 'vtk-reference' || typeof document === 'undefined') {
      return;
    }

    const { cornerstoneViewportService } = servicesManager.services;

    const onCameraModified = (evt: Event) => {
      const detail = (
        evt as CustomEvent<{
          viewportId?: string;
          camera?: cornerstone.Types.ICamera;
        }>
      ).detail;
      const viewportId = detail?.viewportId;
      if (!viewportId) {
        return;
      }

      const logicalViewportId =
        cornerstoneViewportService.getViewportInfo(viewportId)?.viewportOptions?.viewportId ||
        viewportId;
      if (logicalViewportId !== 'cpr-pano') {
        return;
      }

      const livePanoViewport = findViewportByLogicalId(servicesManager, 'cpr-pano');
      const attachedPano = getHostedPanoHandle(livePanoViewport) || attachedVtkPanoRef.current;
      const bridgeState = panoCameraBridgeRef.current;
      if (!attachedPano || !livePanoViewport || !isVolumeViewportLike(livePanoViewport)) {
        return;
      }

      if (!bridgeState || bridgeState.actorUID !== attachedPano.actorUID) {
        rearmPanoCameraBridgeFromLiveViewport('camera-modified-rearm');
        return;
      }

      if (viewportId !== livePanoViewport.id) {
        return;
      }

      const camera = livePanoViewport.getCamera?.() || detail?.camera;
      const viewportWithPanZoom = livePanoViewport as
        | (cornerstone.Types.IViewport & {
            getPan?: () => ArrayLike<number> | null | undefined;
            getZoom?: () => number;
          })
        | null;
      const currentPan = toFiniteVector2Tuple(viewportWithPanZoom?.getPan?.());
      const currentZoom = Number(viewportWithPanZoom?.getZoom?.());
      const currentViewUp = toFiniteVector3Tuple(camera?.viewUp);
      const currentViewPlaneNormal = toFiniteVector3Tuple(camera?.viewPlaneNormal);
      const viewportHeightPx = Number(livePanoViewport.element?.clientHeight || 0);

      if (
        !camera ||
        !currentPan ||
        !Number.isFinite(currentZoom) ||
        currentZoom <= 0 ||
        !Number.isFinite(viewportHeightPx) ||
        viewportHeightPx <= 0
      ) {
        return;
      }

      if (
        !areAlignedDirectionVectors(currentViewUp, bridgeState.baseViewUp) ||
        !areAlignedDirectionVectors(currentViewPlaneNormal, bridgeState.baseViewPlaneNormal)
      ) {
        console.log(
          `CPR-VTK-PANO-CAMERA-BRIDGE ignore-orientation-change viewportId=${livePanoViewport.id} actorUID=${bridgeState.actorUID}`
        );
        return;
      }

      attachedPano.syncCamera({
        panDeltaPx: [
          currentPan[0] - bridgeState.basePan[0],
          currentPan[1] - bridgeState.basePan[1],
        ],
        zoomRatio: currentZoom / bridgeState.baseZoom,
        viewportHeightPx,
      });
    };

    document.addEventListener(cornerstone.Enums.Events.CAMERA_MODIFIED, onCameraModified, true);

    return () => {
      document.removeEventListener(
        cornerstone.Enums.Events.CAMERA_MODIFIED,
        onCameraModified,
        true
      );
    };
  }, [rearmPanoCameraBridgeFromLiveViewport, servicesManager]);

  useEffect(() => {
    return () => {
      clearPendingAxialViewportRestore('unmount');
      setAxialTransitioning(false, 'unmount');
      clearPanoCameraBridge('unmount');
      persistedSplineAnnotationRef.current = null;
      disposeAttachedVtkPano('unmount');
      clearPanoDebugProbe();
      void terminateCPRWorkerSession();
    };
  }, [
    clearPendingAxialViewportRestore,
    clearPanoCameraBridge,
    disposeAttachedVtkPano,
    setAxialTransitioning,
  ]);

  return { onDone, onRedraw, onSliderChange, isGenerating, error, warning };
}
