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
import {
  getPanoV2Phase0ReconstructionMode,
  isVirtualPanoLikeCandidateMode,
  resolvePanoDisplayedOutputMode,
} from './panoV2OrchestratorBridge';
import type { PanoDisplayedOutputMode, PanoReconstructionMode } from './panoV2Types';

type CprPanoDisplayPath = 'vtk-reference' | 'worker-recon';
type CprPanoReconBackend = 'gpu' | 'cpu';
type CprPanoDisplayBackend = CprPanoReconBackend | 'vtk';
type CprPanoDisplaySource = 'panoImageLoader' | 'vtk-hosted-pano-actor';

const CPR_PANO_DEFAULT_WINDOW_WIDTH = 2000;
const CPR_PANO_DEFAULT_WINDOW_CENTER = 1000;
const CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU = -150;
const CPR_DUAL_ARCH_PROJECTION_SOFT_SCAN_P99_THRESHOLD_HU = 1100;
const CPR_DUAL_ARCH_PROJECTION_SOFT_MIN_WINDOW_WIDTH = 1350;
const CPR_DUAL_ARCH_PROJECTION_STANDARD_MIN_WINDOW_WIDTH = 1450;
const CPR_PANO_DISPLAY_PATH_DEFAULT: CprPanoDisplayPath = 'worker-recon';
const CPR_PANO_RECON_BACKEND_DEFAULT: CprPanoReconBackend = 'gpu';
const CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK = false;
// For practical worker-recon display, prefer a fast stable pano over a long
// retry ladder that often ends in the same emergency virtual-pano fallback.
const CPR_PANO_QUALITY_FIRST_MODE = false;
const CPR_PANO_ALLOW_DEGRADED_FIRST_RUN_DISPLAY = true;
const CPR_DEBUG_EXPORT_ATTEMPT_REPORT_DEFAULT = false;
const CPR_DEBUG_EXPORT_TOP_ATTEMPT_COUNT = 5;
const CPR_DEBUG_FORCE_REJECTED_PANOV2_DISPLAY_DEFAULT = false;
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
  'retry-mean-balanced-neutral',
  'retry-mean-broad-neutral',
  'retry-mean-toothband-neutral',
  'retry-mean-balanced-medium',
  'retry-mean-toothband-balanced',
  'retry-mean-toothband-rooted',
  'retry-mean-balanced-medium-root-biased-tight-slab',
  'retry-mean-balanced-medium-strong-bias-tight-slab',
  'retry-mean-balanced-medium-stronger-bias-tight-slab',
  'primary-mean-toothband-narrow',
] as const;
const CPR_GPU_PANO_DEFAULT_SLAB_HALF_THICKNESS_MM = 2;
const CPR_GPU_PANO_DEFAULT_SLAB_SAMPLES = 15;
const CPR_VTK_PANO_PRESET_WINDOW_WIDTH = 3200;
const CPR_VTK_PANO_PRESET_WINDOW_CENTER = 600;
const CPR_PANO_GENERATION_DEBOUNCE_MS = 300;
const CPR_PANO_MAX_DIMENSION = 4096;
const CPR_PANO_FIXED_VERTICAL_HALF_MM = 35;
const CPR_WORKER_INIT_TIMEOUT_MS = 8000;
const CPR_WORKER_RENDER_TIMEOUT_MS = 15000;
const CPR_WORKER_DISPOSE_TIMEOUT_MS = 3000;
const CPR_CROSSSECTION_DEFAULT_SLAB_THICKNESS_MM = 1.5;
const CPR_CROSSSECTION_DEFAULT_BLEND_MODE = cornerstone.Enums.BlendModes.AVERAGE_INTENSITY_BLEND;
const CPR_CROSSSECTION_RENDER_WAIT_TIMEOUT_MS = 1500;
const CPR_TEMP_DEBUG_PIN_DISPLAYED_ATTEMPT_LABELS: readonly string[] = [];
let activePanoDebugProbeCleanup: (() => void) | null = null;
let activePanoDebugProbeRunId: string | null = null;
const panoDebugProbeClicksByRun = new Map<string, PanoDebugProbeSample[]>();
const panoDebugArtifactExportersByRun = new Map<string, () => void>();

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

function readCprDebugBooleanFlag(
  queryKeys: readonly string[],
  storageKeys: readonly string[]
): boolean | null {
  if (typeof window === 'undefined') {
    return null;
  }

  const searchParams = new URLSearchParams(window.location.search);
  for (const key of queryKeys) {
    const queryValue = parseCprDebugBooleanFlag(searchParams.get(key));
    if (queryValue !== null) {
      return queryValue;
    }
  }

  try {
    for (const key of storageKeys) {
      const storedValue = parseCprDebugBooleanFlag(window.localStorage?.getItem(key));
      if (storedValue !== null) {
        return storedValue;
      }
    }
  } catch (error) {
    console.warn('[CPR] Failed to read debug flag from localStorage.', error);
  }

  return null;
}

function shouldExportCprAttemptReport(): boolean {
  return (
    readCprDebugBooleanFlag(
      ['cprDebugExportAttemptReport'],
      ['cpr.debug.exportAttemptReport']
    ) ?? CPR_DEBUG_EXPORT_ATTEMPT_REPORT_DEFAULT
  );
}

function shouldForceDisplayGpuCandidateEvenIfRejected(): boolean {
  return (
    readCprDebugBooleanFlag(
      ['cprForceDisplayGpuCandidateEvenIfRejected', 'forceDisplayGpuCandidateEvenIfRejected'],
      [
        'cpr.debug.forceDisplayGpuCandidateEvenIfRejected',
        'forceDisplayGpuCandidateEvenIfRejected',
      ]
    ) ?? false
  );
}

function shouldForceDisplayRejectedPanoV2FusionForDebug(): boolean {
  return (
    readCprDebugBooleanFlag(
      [
        'cprForceDisplayRejectedPanoV2FusionForDebug',
        'forceDisplayRejectedPanoV2FusionForDebug',
      ],
      [
        'cpr.debug.forceDisplayRejectedPanoV2FusionForDebug',
        'forceDisplayRejectedPanoV2FusionForDebug',
      ]
    ) ?? CPR_DEBUG_FORCE_REJECTED_PANOV2_DISPLAY_DEFAULT
  );
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
  rawSupportScoreGap?: number;
  rawSupportDenseFraction?: number;
  rawSupportPeakHuSupportGate?: number;
  rawSupportDominantPeakOffsetMm?: number;
  rawSupportSecondaryPeakOffsetMm?: number;
  rawSupportLocalJumpMm?: number;
  rawSupportContinuity?: number;
  supportFailureDisplay?: number;
  supportDepthMm?: number;
  supportConfidence?: number;
  supportSpreadMm?: number;
  supportDensity?: number;
  toothBandPrior?: number;
  dominantDensePeakGate?: number;
  toothBandStructureGuard?: number;
  ambiguousBroadSupportPenaltyGate?: number;
  protectedAmbiguousBroadSupportPenaltyGate?: number;
  structuralSupportGate?: number;
  peakStructureValidity?: number;
  supportValidity?: number;
  rowConfidenceGate?: number;
  falseSupportConfidenceGate?: number;
  falseSupportDensityGate?: number;
  falseSupportSpreadGate?: number;
  falseSupportVeto?: number;
  rowBackgroundDensityGate?: number;
  rowBackgroundSpreadGate?: number;
  rowBackgroundPeakHuGate?: number;
  rowBackgroundEdgeGate?: number;
  rowBackgroundVeto?: number;
  supportVetoTriggered?: number;
  supportLocalJumpMm?: number;
  supportContinuity?: number;
  invalidSupportBlackout?: number;
  supportDepthDeltaMm?: number;
  supportConfidenceDelta?: number;
  supportSpreadDeltaMm?: number;
  supportDensityDelta?: number;
  nominalTroughHalfWidthMm?: number;
  effectiveTroughHalfWidthMm?: number;
  continuityExpandedTroughHalfWidthMm?: number;
  backgroundTroughNarrowGate?: number;
  dominantToothBandGate?: number;
  broadWeakToothBandGate?: number;
  toothContinuityAdmissionGate?: number;
  troughLowerMm?: number;
  troughUpperMm?: number;
  lowerPenalty?: number;
  localTransmittance?: number;
  toneMappedValue?: number;
  admissionAccumulation?: number;
  preToneAccumulation?: number;
  toneSuppressedAccumulation?: number;
  toneStageSuppression?: number;
  blackClip?: number;
  admissionOnlyHu?: number;
  toneBypassHu?: number;
  retainedSampleMask?: number;
  middleBandLeak?: number;
  participatingSampleCount?: number;
  renderBranchCode?: number;
  selectedSupportHypothesis?: number;
  focalTroughSharpness?: number;
  outOfTroughSuppression?: number;
  rawProjectedAttenuation?: number;
  finalDisplayImage?: number;
  displayedPath?: string | null;
  backend?: string | null;
  pipelineMode?: string | null;
  reconstructionMode?: string | null;
  holeMetricBlackClipped?: boolean;
  holeMetricLowRetained?: boolean;
  holeMetricLowPreTone?: boolean;
  holeMetricLeakMarked?: boolean;
  holeMetricWouldCount?: boolean;
  holeMetricReasons?: string[];
  mappingMode?: string;
  canvasX?: number;
  canvasY?: number;
};

function clearPanoDebugProbe(): void {
  if (activePanoDebugProbeRunId) {
    panoDebugProbeClicksByRun.delete(activePanoDebugProbeRunId);
    panoDebugArtifactExportersByRun.delete(activePanoDebugProbeRunId);
    activePanoDebugProbeRunId = null;
  }

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
  const toothBandPrior = readOptionalProbeMapValue(payload.debugMaps?.toothBandPriorMap, index);
  const dominantDensePeakGate = readOptionalProbeMapValue(
    payload.debugMaps?.dominantDensePeakGateMap,
    index
  );
  const toothBandStructureGuard = readOptionalProbeMapValue(
    payload.debugMaps?.toothBandStructureGuardMap,
    index
  );
  const ambiguousBroadSupportPenaltyGate = readOptionalProbeMapValue(
    payload.debugMaps?.ambiguousBroadSupportPenaltyGateMap,
    index
  );
  const protectedAmbiguousBroadSupportPenaltyGate = readOptionalProbeMapValue(
    payload.debugMaps?.protectedAmbiguousBroadSupportPenaltyGateMap,
    index
  );
  const structuralSupportGate = readOptionalProbeMapValue(
    payload.debugMaps?.structuralSupportGateMap,
    index
  );
  const peakStructureValidity = readOptionalProbeMapValue(
    payload.debugMaps?.peakStructureValidityMap,
    index
  );
  const supportValidity = readOptionalProbeMapValue(
    payload.debugMaps?.supportValidityMap,
    index
  );
  const rowConfidenceGate = readOptionalProbeMapValue(
    payload.debugMaps?.rowConfidenceGateMap,
    index
  );
  const falseSupportConfidenceGate = readOptionalProbeMapValue(
    payload.debugMaps?.falseSupportConfidenceGateMap,
    index
  );
  const falseSupportDensityGate = readOptionalProbeMapValue(
    payload.debugMaps?.falseSupportDensityGateMap,
    index
  );
  const falseSupportSpreadGate = readOptionalProbeMapValue(
    payload.debugMaps?.falseSupportSpreadGateMap,
    index
  );
  const falseSupportVeto = readOptionalProbeMapValue(
    payload.debugMaps?.falseSupportVetoMap,
    index
  );
  const rowBackgroundDensityGate = readOptionalProbeMapValue(
    payload.debugMaps?.rowBackgroundDensityGateMap,
    index
  );
  const rowBackgroundSpreadGate = readOptionalProbeMapValue(
    payload.debugMaps?.rowBackgroundSpreadGateMap,
    index
  );
  const rowBackgroundPeakHuGate = readOptionalProbeMapValue(
    payload.debugMaps?.rowBackgroundPeakHuGateMap,
    index
  );
  const rowBackgroundEdgeGate = readOptionalProbeMapValue(
    payload.debugMaps?.rowBackgroundEdgeGateMap,
    index
  );
  const rowBackgroundVeto = readOptionalProbeMapValue(
    payload.debugMaps?.rowBackgroundVetoMap,
    index
  );
  const supportVetoTriggered = readOptionalProbeMapValue(
    payload.debugMaps?.supportVetoTriggeredMap,
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
  const rawSupportScoreGap = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportScoreGapMap,
    index
  );
  const rawSupportDenseFraction = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportDenseFractionMap,
    index
  );
  const rawSupportPeakHuSupportGate = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportPeakHuSupportGateMap,
    index
  );
  const rawSupportDominantPeakOffsetMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportDominantPeakOffsetMap,
    index
  );
  const rawSupportSecondaryPeakOffsetMm = readOptionalProbeMapValue(
    payload.debugMaps?.rawSupportSecondaryPeakOffsetMap,
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
  const continuityExpandedTroughHalfWidthMm = readOptionalProbeMapValue(
    payload.debugMaps?.continuityExpandedTroughHalfWidthMap,
    index
  );
  const backgroundTroughNarrowGate = readOptionalProbeMapValue(
    payload.debugMaps?.backgroundTroughNarrowGateMap,
    index
  );
  const dominantToothBandGate = readOptionalProbeMapValue(
    payload.debugMaps?.dominantToothBandGateMap,
    index
  );
  const broadWeakToothBandGate = readOptionalProbeMapValue(
    payload.debugMaps?.broadWeakToothBandGateMap,
    index
  );
  const toothContinuityAdmissionGate = readOptionalProbeMapValue(
    payload.debugMaps?.toothContinuityAdmissionGateMap,
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
  const admissionAccumulation = readOptionalProbeMapValue(
    payload.debugMaps?.admissionAccumulationMap,
    index
  );
  const preToneAccumulation = readOptionalProbeMapValue(
    payload.debugMaps?.preToneAccumulationMap,
    index
  );
  const toneSuppressedAccumulation = readOptionalProbeMapValue(
    payload.debugMaps?.toneSuppressedAccumulationMap,
    index
  );
  const toneStageSuppression = readOptionalProbeMapValue(
    payload.debugMaps?.toneStageSuppressionMap,
    index
  );
  const blackClip = readOptionalProbeMapValue(payload.debugMaps?.blackClipMap, index);
  const admissionOnlyHu = readOptionalProbeMapValue(payload.debugMaps?.admissionOnlyHuMap, index);
  const toneBypassHu = readOptionalProbeMapValue(payload.debugMaps?.toneBypassHuMap, index);
  const retainedSampleMask = readOptionalProbeMapValue(
    payload.debugMaps?.retainedSampleMaskMap,
    index
  );
  const middleBandLeak = readOptionalProbeMapValue(payload.debugMaps?.middleBandLeakMap, index);
  const participatingSampleCount = readOptionalProbeMapValue(
    payload.debugMaps?.participatingSampleCountMap,
    index
  );
  const renderBranchCode = readOptionalProbeMapValue(payload.debugMaps?.renderBranchMap, index);
  const selectedSupportHypothesis = readOptionalProbeMapValue(
    payload.debugMaps?.selectedSupportHypothesisMap,
    index
  );
  const focalTroughSharpness = readOptionalProbeMapValue(
    payload.debugMaps?.focalTroughSharpnessMap,
    index
  );
  const outOfTroughSuppression = readOptionalProbeMapValue(
    payload.debugMaps?.outOfTroughSuppressionMap,
    index
  );
  const rawProjectedAttenuation = readOptionalProbeMapValue(
    payload.debugMaps?.rawProjectedAttenuationMap,
    index
  );
  const finalDisplayImage = readOptionalProbeMapValue(payload.debugMaps?.finalDisplayImageMap, index);
  const localTransmittance =
    typeof totalAttenuation === 'number' ? Math.exp(-Math.max(totalAttenuation, 0)) : undefined;
  const holeMetricBlackClipThreshold = Math.max(
    0,
    Math.min(
      1,
      Number(payload.probeContext?.holeMetricBlackClipThreshold) ||
        CPR_TOOTH_BAND_BLACK_CLIP_THRESHOLD
    )
  );
  const holeMetricRetainedWeightMax = Math.max(
    0,
    Math.min(
      1,
      Number(payload.probeContext?.holeMetricRetainedWeightMax) ||
        CPR_TOOTH_BAND_HOLE_RETAINED_WEIGHT_MAX
    )
  );
  const holeMetricPreToneThreshold = Math.max(
    0,
    Number(payload.probeContext?.holeMetricPreToneThreshold) || 0
  );
  const holeMetricLeakMin = Math.max(
    0,
    Math.min(1, Number(payload.probeContext?.holeMetricLeakMin) || CPR_TOOTH_BAND_HOLE_LEAK_MIN)
  );
  const holeMetricBlackClipped =
    typeof toneMappedValue === 'number' ? toneMappedValue <= holeMetricBlackClipThreshold : false;
  const holeMetricLowRetained =
    typeof retainedSampleMask === 'number' ? retainedSampleMask <= holeMetricRetainedWeightMax : false;
  const holeMetricLowPreTone =
    typeof preToneAccumulation === 'number' ? preToneAccumulation <= holeMetricPreToneThreshold : false;
  const holeMetricLeakMarked =
    typeof middleBandLeak === 'number' ? middleBandLeak >= holeMetricLeakMin : false;
  const holeMetricWouldCount =
    holeMetricBlackClipped && holeMetricLowRetained && (holeMetricLowPreTone || holeMetricLeakMarked);
  const holeMetricReasons: string[] = [];
  if (holeMetricBlackClipped) {
    holeMetricReasons.push('black-clipped');
  }
  if (holeMetricLowRetained) {
    holeMetricReasons.push('low-retained');
  }
  if (holeMetricLowPreTone) {
    holeMetricReasons.push('low-pre-tone');
  }
  if (holeMetricLeakMarked) {
    holeMetricReasons.push('leak-marked');
  }

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
    rawSupportScoreGap,
    rawSupportDenseFraction,
    rawSupportPeakHuSupportGate,
    rawSupportDominantPeakOffsetMm,
    rawSupportSecondaryPeakOffsetMm,
    rawSupportLocalJumpMm,
    rawSupportContinuity,
    supportFailureDisplay,
    supportDepthMm,
    supportConfidence,
    supportSpreadMm,
    supportDensity,
    toothBandPrior,
    dominantDensePeakGate,
    toothBandStructureGuard,
    ambiguousBroadSupportPenaltyGate,
    protectedAmbiguousBroadSupportPenaltyGate,
    structuralSupportGate,
    peakStructureValidity,
    supportValidity,
    rowConfidenceGate,
    falseSupportConfidenceGate,
    falseSupportDensityGate,
    falseSupportSpreadGate,
    falseSupportVeto,
    rowBackgroundDensityGate,
    rowBackgroundSpreadGate,
    rowBackgroundPeakHuGate,
    rowBackgroundEdgeGate,
    rowBackgroundVeto,
    supportVetoTriggered,
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
    continuityExpandedTroughHalfWidthMm,
    backgroundTroughNarrowGate,
    dominantToothBandGate,
    broadWeakToothBandGate,
    toothContinuityAdmissionGate,
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
    admissionAccumulation,
    preToneAccumulation,
    toneSuppressedAccumulation,
    toneStageSuppression,
    blackClip,
    admissionOnlyHu,
    toneBypassHu,
    retainedSampleMask,
    middleBandLeak,
    participatingSampleCount,
    renderBranchCode,
    selectedSupportHypothesis,
    focalTroughSharpness,
    outOfTroughSuppression,
    rawProjectedAttenuation,
    finalDisplayImage,
    displayedPath: payload.probeContext?.displayedPath ?? null,
    backend: payload.probeContext?.backend ?? null,
    pipelineMode: payload.probeContext?.pipelineMode ?? null,
    reconstructionMode: payload.probeContext?.reconstructionMode ?? null,
    holeMetricBlackClipped,
    holeMetricLowRetained,
    holeMetricLowPreTone,
    holeMetricLeakMarked,
    holeMetricWouldCount,
    holeMetricReasons,
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
  activePanoDebugProbeRunId = runId;
  panoDebugProbeClicksByRun.set(runId, []);

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
          rawSupportScoreGap: sample.rawSupportScoreGap ?? null,
          rawSupportDenseFraction: sample.rawSupportDenseFraction ?? null,
          rawSupportPeakHuSupportGate: sample.rawSupportPeakHuSupportGate ?? null,
          rawSupportDominantPeakOffsetMm: sample.rawSupportDominantPeakOffsetMm ?? null,
          rawSupportSecondaryPeakOffsetMm: sample.rawSupportSecondaryPeakOffsetMm ?? null,
          rawSupportLocalJumpMm: sample.rawSupportLocalJumpMm ?? null,
          rawSupportContinuity: sample.rawSupportContinuity ?? null,
          supportFailureDisplay: sample.supportFailureDisplay ?? null,
          supportDepthMm: sample.supportDepthMm ?? null,
          supportConfidence: sample.supportConfidence ?? null,
          supportSpreadMm: sample.supportSpreadMm ?? null,
          supportDensity: sample.supportDensity ?? null,
          toothBandPrior: sample.toothBandPrior ?? null,
          dominantDensePeakGate: sample.dominantDensePeakGate ?? null,
          toothBandStructureGuard: sample.toothBandStructureGuard ?? null,
          ambiguousBroadSupportPenaltyGate: sample.ambiguousBroadSupportPenaltyGate ?? null,
          protectedAmbiguousBroadSupportPenaltyGate:
            sample.protectedAmbiguousBroadSupportPenaltyGate ?? null,
          structuralSupportGate: sample.structuralSupportGate ?? null,
          peakStructureValidity: sample.peakStructureValidity ?? null,
          supportValidity: sample.supportValidity ?? null,
          rowConfidenceGate: sample.rowConfidenceGate ?? null,
          falseSupportConfidenceGate: sample.falseSupportConfidenceGate ?? null,
          falseSupportDensityGate: sample.falseSupportDensityGate ?? null,
          falseSupportSpreadGate: sample.falseSupportSpreadGate ?? null,
          falseSupportVeto: sample.falseSupportVeto ?? null,
          rowBackgroundDensityGate: sample.rowBackgroundDensityGate ?? null,
          rowBackgroundSpreadGate: sample.rowBackgroundSpreadGate ?? null,
          rowBackgroundPeakHuGate: sample.rowBackgroundPeakHuGate ?? null,
          rowBackgroundEdgeGate: sample.rowBackgroundEdgeGate ?? null,
          rowBackgroundVeto: sample.rowBackgroundVeto ?? null,
          supportVetoTriggered: sample.supportVetoTriggered ?? null,
          supportLocalJumpMm: sample.supportLocalJumpMm ?? null,
          supportContinuity: sample.supportContinuity ?? null,
          supportDepthDeltaMm: sample.supportDepthDeltaMm ?? null,
          supportConfidenceDelta: sample.supportConfidenceDelta ?? null,
          supportSpreadDeltaMm: sample.supportSpreadDeltaMm ?? null,
          supportDensityDelta: sample.supportDensityDelta ?? null,
          nominalTroughHalfWidthMm: sample.nominalTroughHalfWidthMm ?? null,
          effectiveTroughHalfWidthMm: sample.effectiveTroughHalfWidthMm ?? null,
          continuityExpandedTroughHalfWidthMm:
            sample.continuityExpandedTroughHalfWidthMm ?? null,
          backgroundTroughNarrowGate: sample.backgroundTroughNarrowGate ?? null,
          dominantToothBandGate: sample.dominantToothBandGate ?? null,
          broadWeakToothBandGate: sample.broadWeakToothBandGate ?? null,
          toothContinuityAdmissionGate: sample.toothContinuityAdmissionGate ?? null,
          invalidSupportBlackout: sample.invalidSupportBlackout ?? null,
          troughLowerMm: sample.troughLowerMm ?? null,
          troughUpperMm: sample.troughUpperMm ?? null,
          localTransmittance: sample.localTransmittance ?? null,
          toneMappedValue: sample.toneMappedValue ?? null,
          preToneAccumulation: sample.preToneAccumulation ?? null,
          admissionAccumulation: sample.admissionAccumulation ?? null,
          retainedSampleMask: sample.retainedSampleMask ?? null,
          middleBandLeak: sample.middleBandLeak ?? null,
          blackClip: sample.blackClip ?? null,
          renderBranchCode: sample.renderBranchCode ?? null,
          selectedSupportHypothesis: sample.selectedSupportHypothesis ?? null,
          focalTroughSharpness: sample.focalTroughSharpness ?? null,
          outOfTroughSuppression: sample.outOfTroughSuppression ?? null,
          rawProjectedAttenuation: sample.rawProjectedAttenuation ?? null,
          finalDisplayImage: sample.finalDisplayImage ?? null,
          holeMetricBlackClipped: sample.holeMetricBlackClipped ?? null,
          holeMetricLowRetained: sample.holeMetricLowRetained ?? null,
          holeMetricLowPreTone: sample.holeMetricLowPreTone ?? null,
          holeMetricLeakMarked: sample.holeMetricLeakMarked ?? null,
          holeMetricWouldCount: sample.holeMetricWouldCount ?? null,
          holeMetricReasons: sample.holeMetricReasons ?? [],
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
    const clickSamples = panoDebugProbeClicksByRun.get(runId) ?? [];
    clickSamples.push({
      ...sample,
      canvasX: Math.round(location.canvasX * 100) / 100,
      canvasY: Math.round(location.canvasY * 100) / 100,
      mappingMode: location.mappingMode,
    });
    panoDebugProbeClicksByRun.set(runId, clickSamples);
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
  toothBandHoleFraction?: number;
  toothBandBlackClipFraction?: number;
  toothBandHolePreToneThreshold?: number;
  toothBandRetainedWeightP10?: number;
  toothBandRetainedWeightP50?: number;
  toothBandRetainedWeightRowP10?: number;
  toothBandRetainedWeightRowP50?: number;
  toothBandRetainedWeightRowP90?: number;
  toothBandRetainedWeightColumnP10?: number;
  toothBandRetainedWeightColumnP50?: number;
  toothBandRetainedWeightColumnP90?: number;
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
  backgroundBands?: {
    top: FloatBufferBackgroundBandSummary;
    middle: FloatBufferBackgroundBandSummary;
    bottom: FloatBufferBackgroundBandSummary;
    dominantOutlierBand05: FloatBufferBackgroundBandDominantLabel;
    dominantOutlierBand10: FloatBufferBackgroundBandDominantLabel;
  };
}

interface FloatBufferBackgroundBandSummary {
  sampleCount: number;
  backgroundToneP95: number;
  backgroundToneP99: number;
  backgroundToneMax: number;
  backgroundOutlierFraction05: number;
  backgroundOutlierFraction10: number;
  backgroundOutlierContribution05: number;
  backgroundOutlierContribution10: number;
}

type FloatBufferBackgroundBandName = 'top' | 'middle' | 'bottom';
type FloatBufferBackgroundBandDominantLabel =
  | FloatBufferBackgroundBandName
  | 'mixed'
  | 'none';

interface ToothBandStageDiagnostics {
  sampleCount: number;
  retainedWeightP10: number | null;
  retainedWeightP50: number | null;
  retainedWeightRowP10: number | null;
  retainedWeightRowP50: number | null;
  retainedWeightRowP90: number | null;
  retainedWeightColumnP10: number | null;
  retainedWeightColumnP50: number | null;
  retainedWeightColumnP90: number | null;
  admissionAccumulationP10: number | null;
  admissionAccumulationP50: number | null;
  admissionAccumulationP90: number | null;
  toneSuppressedAccumulationP10: number | null;
  toneSuppressedAccumulationP50: number | null;
  toneSuppressedAccumulationP90: number | null;
  toneStageSuppressionP50: number | null;
  toneStageSuppressionP90: number | null;
  invalidSupportBlackoutP50: number | null;
  invalidSupportBlackoutP90: number | null;
  blackClipFraction: number | null;
  middleBandLeakP50: number | null;
  middleBandLeakP90: number | null;
  admissionOnlyHuP10: number | null;
  admissionOnlyHuP50: number | null;
  admissionOnlyHuP90: number | null;
  toneBypassHuP10: number | null;
  toneBypassHuP50: number | null;
  toneBypassHuP90: number | null;
  finalPostToneHuP10: number | null;
  finalPostToneHuP50: number | null;
  finalPostToneHuP90: number | null;
  stageHint: 'support-admission-collapse' | 'tone-blackout-destruction' | 'normalization-or-mixed';
  stageEvidence: string[];
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
const CPR_TOOTH_BAND_BLACK_CLIP_THRESHOLD = 0.02;
const CPR_TOOTH_BAND_HOLE_RETAINED_WEIGHT_MAX = 0.18;
const CPR_TOOTH_BAND_HOLE_LEAK_MIN = 0.08;
const CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION = 0.05;
const CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION = 0.16;
const CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX = 0.02;
const CPR_MIDDLE_BAND_DOMINANT_LEAK_CONTRIBUTION05_MIN = 0.68;
const CPR_MIDDLE_BAND_DOMINANT_LEAK_CONTRIBUTION10_MIN = 0.72;
const CPR_TARGET_TOOTH_DEBUG_LABEL = 'retry-mean-balanced-medium-root-biased-tight-slab';

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
}): PanoDisplayedOutputMode {
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

  return resolvePanoDisplayedOutputMode({
    backend: params.routeDiagnostic.backend,
    phase2GatePassed: params.routeDiagnostic.phase2GatePassed,
    reconstructionMode: resolvedReconstructionMode,
    pipelineMode: toNonEmptyString(diagnostic?.pipelineMode),
  });
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

function resolveSyntheticCprIntensityDomainForRenderMode(
  intensityDomain: SyntheticCprIntensityDomain,
  renderSupportMode: string | null | undefined
): SyntheticCprIntensityDomain {
  return renderSupportMode === 'panoV2Fusion' ? 'native' : intensityDomain;
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
  return renderSupportMode === 'dualArchProjection' || renderSupportMode === 'archGuidedDualLayer';
}

function isNativeDisplayPanoRenderMode(renderSupportMode: string | null | undefined): boolean {
  return renderSupportMode === 'panoV2Fusion';
}

function isRadiographVirtualPanoRenderMode(renderSupportMode: string | null | undefined): boolean {
  return renderSupportMode === 'radiographDualHypothesis';
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

  if (isRadiographVirtualPanoRenderMode(renderSupportMode)) {
    return null;
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
  score -= Math.max(0, summary.lowerBandP50 - (isDualArchProjection ? -340 : -160)) / 40;
  score -= Math.max(0, summary.lowerBandBrightFraction - (isDualArchProjection ? 0.18 : 0.28)) * 42;
  score -= Math.max(0, summary.toothBandMean - 760) / 80;
  score -= Math.max(0, summary.toothBandP10 - (isDualArchProjection ? 140 : 80)) / 24;
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
  | 'worker-cpu-virtual-panoramic-radiograph'
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
  toothBandHoleFraction: number | null;
  toothBandBlackClipFraction: number | null;
  toothBandRetainedWeightP10: number | null;
  toothBandRetainedWeightP50: number | null;
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
  rendererVariant: string | null;
  pipelineVariant: string | null;
  renderBypass: boolean | null;
  workerBranchSelected: boolean | null;
  workerQcAccepted: boolean | null;
  workerQcStage: string | null;
  metricStage: string | null;
  rejectReasonsStage: string | null;
  contextHuP50: number | null;
  contextBlendMean: number | null;
  contextWeightFractionMean: number | null;
  columnSupportReliabilityP50: number | null;
  upperDetailHuP50: number | null;
  lowerDetailHuP50: number | null;
  detailSampleFractionMean: number | null;
  shadowLiftMean: number | null;
  focalSharpnessCenterThirdP50: number | null;
  interToothValleyContrast: number | null;
  intraToothGradationScore: number | null;
  crownHighlightSaturationFraction: number | null;
  occlusalDarkCapFraction: number | null;
  offTroughEnergyTopRatio: number | null;
  offTroughEnergyMiddleRatio: number | null;
  offTroughEnergyBottomRatio: number | null;
  lowerFieldSpeckleFraction: number | null;
  lowerFieldVerticalStreakScore: number | null;
  underRootVerticalSmearScore: number | null;
  hypothesisSwitchFraction: number | null;
  finalDisplayLowClipFraction: number | null;
  finalDisplayHighClipFraction: number | null;
  finalDisplayHistogramOccupancy: number | null;
  rawZeroButFinalBrightFraction: number | null;
  outsideRowsBrightnessP95: number | null;
  gapRowsBrightnessP95: number | null;
  bandBoundaryJumpMean: number | null;
}

type Phase4RadiographMetricStageName = 'rawProjection' | 'postSuppression' | 'finalDisplay';

interface Phase4RadiographLocalizedReadout {
  centerTeeth: {
    focalSharpnessP50: number | null;
    rawProjectedAttenuationP50: number | null;
    postSuppressionP50: number | null;
    preNormalizeCompositeP50: number | null;
    postNormalizeDisplayP50: number | null;
    finalDisplayP50: number | null;
    backgroundPresentationP50: number | null;
    hasSignalFraction: number | null;
    contextContributionP50: number | null;
    backgroundFillContributionP50: number | null;
    hypothesisSwitchFraction: number | null;
  };
  upperCloudBand: {
    outOfTroughSuppressionP50: number | null;
    rawProjectedAttenuationP50: number | null;
    postSuppressionP50: number | null;
    preNormalizeCompositeP50: number | null;
    postNormalizeDisplayP50: number | null;
    finalDisplayP50: number | null;
    backgroundPresentationP50: number | null;
    hasSignalFraction: number | null;
    contextContributionP50: number | null;
    backgroundFillContributionP50: number | null;
  };
  lowerGranularField: {
    outOfTroughSuppressionP50: number | null;
    rawProjectedAttenuationP50: number | null;
    postSuppressionP50: number | null;
    preNormalizeCompositeP50: number | null;
    postNormalizeDisplayP50: number | null;
    finalDisplayP50: number | null;
    backgroundPresentationP50: number | null;
    hasSignalFraction: number | null;
    contextContributionP50: number | null;
    backgroundFillContributionP50: number | null;
    speckleFraction: number | null;
    verticalStreakScore: number | null;
  };
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
    backgroundBands: {
      top: FloatBufferBackgroundBandSummary;
      middle: FloatBufferBackgroundBandSummary;
      bottom: FloatBufferBackgroundBandSummary;
      dominantOutlierBand05: FloatBufferBackgroundBandDominantLabel;
      dominantOutlierBand10: FloatBufferBackgroundBandDominantLabel;
    } | null;
  };
  anatomy: {
    lowerBandBrightFraction: number | null;
    lowerBandP50: number | null;
    toothBandMean: number | null;
    toothBandP10: number | null;
    toothBandP90: number | null;
    toothBandContrastRange: number | null;
    toothBandHoleFraction: number | null;
    toothBandBlackClipFraction: number | null;
    toothBandRetainedWeightP10: number | null;
    toothBandRetainedWeightP50: number | null;
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
  workerBranchSelected: boolean | null;
  workerQcAccepted: boolean | null;
  workerQcStage: string | null;
  metricStage: string | null;
  rejectReasonsStage: string | null;
  workerUsedAsOutput: boolean | null;
  orchestratorAccepted: boolean;
  borderlineAcceptedReason: string | null;
  pass: boolean;
  rejectReasons: string[];
  metrics: Phase4QualityGateMetrics;
  localizedReadout: Phase4RadiographLocalizedReadout | null;
  supportSurfaceRiskSummary: Phase4SupportSurfaceRiskSummary;
}

interface Phase4DegradedPreviewAssessment {
  catastrophic: boolean;
  catastrophicReasons: string[];
  blockedAsDegradedPreview: boolean;
  blockedAsDegradedPreviewReasons: string[];
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
  toothBandDamagePenalty: number;
  middleBandLeakPenalty: number;
  balancedMeanRecoveryBonus: number;
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
  toothBandHoleFraction: number | null;
  toothBandBlackClipFraction: number | null;
  toothBandRetainedWeightP10: number | null;
  toothBandRetainedWeightP50: number | null;
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

function smoothstepNumber(edge0: number, edge1: number, value: number): number {
  if (!Number.isFinite(edge0) || !Number.isFinite(edge1) || !Number.isFinite(value)) {
    return 0;
  }

  if (edge0 === edge1) {
    return value >= edge1 ? 1 : 0;
  }

  const t = clampPhase4DisplaySelectionUnit((value - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function hasPhase4DominantMiddleBandLeakage(
  summary: FloatBufferDebugSummary | null | undefined
): boolean {
  const backgroundBands = summary?.backgroundBands;
  if (!backgroundBands) {
    return false;
  }

  return (
    (backgroundBands.dominantOutlierBand05 === 'middle' &&
      (summary?.backgroundOutlierFraction05 ?? 0) > 0.17 &&
      backgroundBands.middle.backgroundOutlierContribution05 >=
        CPR_MIDDLE_BAND_DOMINANT_LEAK_CONTRIBUTION05_MIN) ||
    (backgroundBands.dominantOutlierBand10 === 'middle' &&
      (summary?.backgroundOutlierFraction10 ?? 0) > 0.07 &&
      backgroundBands.middle.backgroundOutlierContribution10 >=
        CPR_MIDDLE_BAND_DOMINANT_LEAK_CONTRIBUTION10_MIN)
  );
}

function computePhase4MiddleBandLeakPenalty(
  summary: FloatBufferDebugSummary | null | undefined
): number {
  const backgroundBands = summary?.backgroundBands;
  if (!backgroundBands) {
    return 0;
  }

  const middleContribution05 = backgroundBands.middle.backgroundOutlierContribution05;
  const middleContribution10 = backgroundBands.middle.backgroundOutlierContribution10;

  return (
    (backgroundBands.dominantOutlierBand05 === 'middle'
      ? Math.max(0, middleContribution05 - 0.52) * 34
      : Math.max(0, middleContribution05 - 0.62) * 10) +
    (backgroundBands.dominantOutlierBand10 === 'middle'
      ? Math.max(0, middleContribution10 - 0.45) * 40
      : Math.max(0, middleContribution10 - 0.58) * 12)
  );
}

function computePhase4ToothBandDamagePenalty(params: {
  toothBandHoleFraction?: number | null;
  toothBandBlackClipFraction?: number | null;
  toothBandRetainedWeightP10?: number | null;
  toothBandRetainedWeightP50?: number | null;
}): number {
  const toothBandHoleFraction = params.toothBandHoleFraction;
  const toothBandBlackClipFraction = params.toothBandBlackClipFraction;
  const toothBandRetainedWeightP10 = params.toothBandRetainedWeightP10;
  const toothBandRetainedWeightP50 = params.toothBandRetainedWeightP50;

  return (
    Math.max(0, (toothBandHoleFraction ?? 0) - 0.018) * 260 +
    Math.max(0, (toothBandBlackClipFraction ?? 0) - 0.12) * 120 +
    (toothBandRetainedWeightP10 !== null && toothBandRetainedWeightP10 !== undefined
      ? Math.max(0, 0.84 - toothBandRetainedWeightP10) * 55
      : 0) +
    (toothBandRetainedWeightP50 !== null && toothBandRetainedWeightP50 !== undefined
      ? Math.max(0, 0.9 - toothBandRetainedWeightP50) * 32
      : 0)
  );
}

function computePhase4BalancedMeanRecoveryBonus(params: {
  isGpuMeanAttempt: boolean;
  summary: FloatBufferDebugSummary | null;
  actualVertHalfMm: number;
  verticalCenterOffsetMm: number;
  slabHalfThicknessMm: number;
}): number {
  const { isGpuMeanAttempt, summary, actualVertHalfMm, verticalCenterOffsetMm, slabHalfThicknessMm } =
    params;
  if (!isGpuMeanAttempt || !summary || hasPhase4DominantMiddleBandLeakage(summary)) {
    return 0;
  }

  const verticalGate = 1 - smoothstepNumber(50.2, 52.2, actualVertHalfMm);
  const slabGate = 1 - smoothstepNumber(1.62, 1.88, slabHalfThicknessMm);
  const centerGate = 1 - smoothstepNumber(0, 0.75, Math.abs(verticalCenterOffsetMm + 4.4));
  const retainedGate = smoothstepNumber(0.72, 0.84, summary.toothBandRetainedWeightP10 ?? 0);
  const holeGate = 1 - smoothstepNumber(0.05, 0.075, summary.toothBandHoleFraction ?? 0);
  const blackClipGate =
    1 - smoothstepNumber(0.16, 0.2, summary.toothBandBlackClipFraction ?? 0);
  const background05Gate =
    1 - smoothstepNumber(0.19, 0.225, summary.backgroundOutlierFraction05 ?? 0);
  const background10Gate =
    1 - smoothstepNumber(0.08, 0.105, summary.backgroundOutlierFraction10 ?? 0);

  return (
    verticalGate *
    slabGate *
    centerGate *
    retainedGate *
    holeGate *
    blackClipGate *
    background05Gate *
    background10Gate *
    10
  );
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
  const toothBandHoleFraction = params.metrics.toothBandHoleFraction;
  const toothBandBlackClipFraction = params.metrics.toothBandBlackClipFraction;
  const toothBandRetainedWeightP10 = params.metrics.toothBandRetainedWeightP10;
  const toothBandRetainedWeightP50 = params.metrics.toothBandRetainedWeightP50;
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
  const toothBandDamagePenalty = computePhase4ToothBandDamagePenalty({
    toothBandHoleFraction,
    toothBandBlackClipFraction,
    toothBandRetainedWeightP10,
    toothBandRetainedWeightP50,
  });
  const middleBandLeakPenalty = computePhase4MiddleBandLeakPenalty(params.summary);
  const balancedMeanRecoveryBonus = computePhase4BalancedMeanRecoveryBonus({
    isGpuMeanAttempt,
    summary: params.summary,
    actualVertHalfMm: params.actualVertHalfMm,
    verticalCenterOffsetMm: params.verticalCenterOffsetMm,
    slabHalfThicknessMm: params.slabHalfThicknessMm,
  });
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
    balancedMeanRecoveryBonus +
    preferredTightRootFamilyBonus -
    backgroundLeakagePenalty -
    troughAdmissionPenalty -
    broadWeakSupportPenalty -
    toothBandDamagePenalty -
    middleBandLeakPenalty -
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
    toothBandDamagePenalty,
    middleBandLeakPenalty,
    balancedMeanRecoveryBonus,
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
    toothBandHoleFraction,
    toothBandBlackClipFraction,
    toothBandRetainedWeightP10,
    toothBandRetainedWeightP50,
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
  const toothHoleImproved =
    a.toothBandHoleFraction !== null &&
    b.toothBandHoleFraction !== null &&
    a.toothBandHoleFraction <= b.toothBandHoleFraction - 0.015;
  const toothBlackClipImproved =
    a.toothBandBlackClipFraction !== null &&
    b.toothBandBlackClipFraction !== null &&
    a.toothBandBlackClipFraction <= b.toothBandBlackClipFraction - 0.06;
  const toothRetainedP50Improved =
    a.toothBandRetainedWeightP50 !== null &&
    b.toothBandRetainedWeightP50 !== null &&
    a.toothBandRetainedWeightP50 >= b.toothBandRetainedWeightP50 + 0.08;
  const toothRetainedP10Improved =
    a.toothBandRetainedWeightP10 !== null &&
    b.toothBandRetainedWeightP10 !== null &&
    a.toothBandRetainedWeightP10 >= b.toothBandRetainedWeightP10 + 0.05;

  return (
    (supportImproved &&
      background05Improved &&
      background10Improved &&
      (troughImproved || sampleAdmissionImproved)) ||
    (toothHoleImproved &&
      (toothBlackClipImproved || toothRetainedP50Improved || toothRetainedP10Improved))
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
  const aRiskScore = a.supportSurfaceRiskSummary?.riskScore ?? 0;
  const bRiskScore = b.supportSurfaceRiskSummary?.riskScore ?? 0;
  if (aRiskScore !== bRiskScore) {
    return aRiskScore - bRiskScore;
  }
  const aToothBandDamagePenalty = computePhase4ToothBandDamagePenalty({
    toothBandHoleFraction: a.qualityGateMetrics?.toothBandHoleFraction,
    toothBandBlackClipFraction: a.qualityGateMetrics?.toothBandBlackClipFraction,
    toothBandRetainedWeightP10: a.qualityGateMetrics?.toothBandRetainedWeightP10,
    toothBandRetainedWeightP50: a.qualityGateMetrics?.toothBandRetainedWeightP50,
  });
  const bToothBandDamagePenalty = computePhase4ToothBandDamagePenalty({
    toothBandHoleFraction: b.qualityGateMetrics?.toothBandHoleFraction,
    toothBandBlackClipFraction: b.qualityGateMetrics?.toothBandBlackClipFraction,
    toothBandRetainedWeightP10: b.qualityGateMetrics?.toothBandRetainedWeightP10,
    toothBandRetainedWeightP50: b.qualityGateMetrics?.toothBandRetainedWeightP50,
  });
  if (Math.abs(aToothBandDamagePenalty - bToothBandDamagePenalty) > 0.35) {
    return aToothBandDamagePenalty - bToothBandDamagePenalty;
  }
  const aMiddleBandLeakPenalty = computePhase4MiddleBandLeakPenalty(a.summary);
  const bMiddleBandLeakPenalty = computePhase4MiddleBandLeakPenalty(b.summary);
  if (Math.abs(aMiddleBandLeakPenalty - bMiddleBandLeakPenalty) > 0.5) {
    return aMiddleBandLeakPenalty - bMiddleBandLeakPenalty;
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

function normalizePhase4RadiographMetricStageName(
  value: unknown
): Phase4RadiographMetricStageName | null {
  return value === 'rawProjection' || value === 'postSuppression' || value === 'finalDisplay'
    ? value
    : null;
}

function stripPhase4StagePrefix(reason: string): string {
  const match = /^(rawProjection|postSuppression|finalDisplay):(.*)$/.exec(reason);
  return match ? match[2] : reason;
}

function stampPhase4ReasonStage(
  reason: string,
  stage: Phase4RadiographMetricStageName | null
): string {
  const baseReason = stripPhase4StagePrefix(reason);
  return stage ? `${stage}:${baseReason}` : baseReason;
}

function readPhase4RadiographLocalizedReadout(
  value: unknown
): Phase4RadiographLocalizedReadout | null {
  const record = readPhase4DiagnosticRecord(value);
  const centerTeeth = readPhase4DiagnosticRecord(record?.centerTeeth);
  const upperCloudBand = readPhase4DiagnosticRecord(record?.upperCloudBand);
  const lowerGranularField = readPhase4DiagnosticRecord(record?.lowerGranularField);
  if (!centerTeeth && !upperCloudBand && !lowerGranularField) {
    return null;
  }

  return {
    centerTeeth: {
      focalSharpnessP50: readPhase4DiagnosticNumber(centerTeeth?.focalSharpnessP50),
      rawProjectedAttenuationP50: readPhase4DiagnosticNumber(
        centerTeeth?.rawProjectedAttenuationP50
      ),
      postSuppressionP50: readPhase4DiagnosticNumber(centerTeeth?.postSuppressionP50),
      preNormalizeCompositeP50: readPhase4DiagnosticNumber(
        centerTeeth?.preNormalizeCompositeP50
      ),
      postNormalizeDisplayP50: readPhase4DiagnosticNumber(
        centerTeeth?.postNormalizeDisplayP50
      ),
      finalDisplayP50: readPhase4DiagnosticNumber(centerTeeth?.finalDisplayP50),
      backgroundPresentationP50: readPhase4DiagnosticNumber(
        centerTeeth?.backgroundPresentationP50
      ),
      hasSignalFraction: readPhase4DiagnosticNumber(centerTeeth?.hasSignalFraction),
      contextContributionP50: readPhase4DiagnosticNumber(centerTeeth?.contextContributionP50),
      backgroundFillContributionP50: readPhase4DiagnosticNumber(
        centerTeeth?.backgroundFillContributionP50
      ),
      hypothesisSwitchFraction: readPhase4DiagnosticNumber(centerTeeth?.hypothesisSwitchFraction),
    },
    upperCloudBand: {
      outOfTroughSuppressionP50: readPhase4DiagnosticNumber(
        upperCloudBand?.outOfTroughSuppressionP50
      ),
      rawProjectedAttenuationP50: readPhase4DiagnosticNumber(
        upperCloudBand?.rawProjectedAttenuationP50
      ),
      postSuppressionP50: readPhase4DiagnosticNumber(upperCloudBand?.postSuppressionP50),
      preNormalizeCompositeP50: readPhase4DiagnosticNumber(
        upperCloudBand?.preNormalizeCompositeP50
      ),
      postNormalizeDisplayP50: readPhase4DiagnosticNumber(
        upperCloudBand?.postNormalizeDisplayP50
      ),
      finalDisplayP50: readPhase4DiagnosticNumber(upperCloudBand?.finalDisplayP50),
      backgroundPresentationP50: readPhase4DiagnosticNumber(
        upperCloudBand?.backgroundPresentationP50
      ),
      hasSignalFraction: readPhase4DiagnosticNumber(upperCloudBand?.hasSignalFraction),
      contextContributionP50: readPhase4DiagnosticNumber(upperCloudBand?.contextContributionP50),
      backgroundFillContributionP50: readPhase4DiagnosticNumber(
        upperCloudBand?.backgroundFillContributionP50
      ),
    },
    lowerGranularField: {
      outOfTroughSuppressionP50: readPhase4DiagnosticNumber(
        lowerGranularField?.outOfTroughSuppressionP50
      ),
      rawProjectedAttenuationP50: readPhase4DiagnosticNumber(
        lowerGranularField?.rawProjectedAttenuationP50
      ),
      postSuppressionP50: readPhase4DiagnosticNumber(lowerGranularField?.postSuppressionP50),
      preNormalizeCompositeP50: readPhase4DiagnosticNumber(
        lowerGranularField?.preNormalizeCompositeP50
      ),
      postNormalizeDisplayP50: readPhase4DiagnosticNumber(
        lowerGranularField?.postNormalizeDisplayP50
      ),
      finalDisplayP50: readPhase4DiagnosticNumber(lowerGranularField?.finalDisplayP50),
      backgroundPresentationP50: readPhase4DiagnosticNumber(
        lowerGranularField?.backgroundPresentationP50
      ),
      hasSignalFraction: readPhase4DiagnosticNumber(lowerGranularField?.hasSignalFraction),
      contextContributionP50: readPhase4DiagnosticNumber(
        lowerGranularField?.contextContributionP50
      ),
      backgroundFillContributionP50: readPhase4DiagnosticNumber(
        lowerGranularField?.backgroundFillContributionP50
      ),
      speckleFraction: readPhase4DiagnosticNumber(lowerGranularField?.speckleFraction),
      verticalStreakScore: readPhase4DiagnosticNumber(lowerGranularField?.verticalStreakScore),
    },
  };
}

function resolvePhase4RadiographMetricStage(params: {
  workerQcStage: string | null;
  rawStage: Record<string, unknown> | null;
  postStage: Record<string, unknown> | null;
  finalStage: Record<string, unknown> | null;
}): {
  name: Phase4RadiographMetricStageName;
  record: Record<string, unknown> | null;
} {
  const requestedStage = normalizePhase4RadiographMetricStageName(params.workerQcStage);
  const stageByName: Record<Phase4RadiographMetricStageName, Record<string, unknown> | null> = {
    rawProjection: params.rawStage,
    postSuppression: params.postStage,
    finalDisplay: params.finalStage,
  };
  if (requestedStage && stageByName[requestedStage]) {
    return { name: requestedStage, record: stageByName[requestedStage] };
  }
  if (params.finalStage) {
    return { name: 'finalDisplay', record: params.finalStage };
  }
  if (params.postStage) {
    return { name: 'postSuppression', record: params.postStage };
  }
  return {
    name: requestedStage ?? 'rawProjection',
    record: params.rawStage,
  };
}

function isPhase4LowerBandRejectReason(reason: string): boolean {
  const baseReason = stripPhase4StagePrefix(reason);
  return (
    baseReason === 'lower-band-bright-fraction-too-high' ||
    baseReason === 'lower-band-p50-too-high' ||
    baseReason === 'lower-band-mean-too-high' ||
    baseReason === 'lower-suppression-ratio-too-high'
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
  const outputSelection = readPhase4DiagnosticRecord(diagnostic?.outputSelection);
  const virtualPanoRenderRawStage = readPhase4DiagnosticRecord(virtualPanoRender?.rawProjectionStage);
  const virtualPanoRenderPostStage = readPhase4DiagnosticRecord(
    virtualPanoRender?.postSuppressionStage
  );
  const virtualPanoRenderFinalStage = readPhase4DiagnosticRecord(virtualPanoRender?.finalDisplayStage);
  const supportSurface = gpuSupportSurface ?? cpuSupportSurface;
  const renderSupportMode = toNonEmptyString(virtualPanoRender?.renderSupportMode);
  const rendererVariant =
    toNonEmptyString(virtualPanoRender?.rendererVariant) ??
    toNonEmptyString(virtualPanoRender?.rendererFamilyName);
  const pipelineVariant = toNonEmptyString(virtualPanoRender?.pipelineVariant);
  const renderBypass = readPhase4DiagnosticBoolean(virtualPanoRender?.renderBypass);
  const workerQcStage =
    toNonEmptyString(outputSelection?.workerQcStage) ??
    toNonEmptyString(virtualPanoRender?.workerQcStage);
  const isRadiographRenderer =
    rendererVariant === 'virtual-panoramic-radiograph' ||
    pipelineVariant === 'virtualPanoramicRadiograph' ||
    isRadiographVirtualPanoRenderMode(renderSupportMode);
  const selectedRadiographStage = isRadiographRenderer
    ? resolvePhase4RadiographMetricStage({
        workerQcStage,
        rawStage: virtualPanoRenderRawStage,
        postStage: virtualPanoRenderPostStage,
        finalStage: virtualPanoRenderFinalStage,
      })
    : null;
  const selectedRadiographStageRecord = selectedRadiographStage?.record ?? null;
  const metricStage = isRadiographRenderer ? selectedRadiographStage?.name ?? workerQcStage : null;
  const readSelectedRadiographMetric = (key: string): number | null =>
    readPhase4DiagnosticNumber(selectedRadiographStageRecord?.[key]);
  const lowerBandBrightFraction = isRadiographRenderer
    ? readSelectedRadiographMetric('lowerBandBrightFraction') ?? summary?.lowerBandBrightFraction ?? null
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.lowerBandBrightFraction) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.lowerBandBrightFraction) ??
      summary?.lowerBandBrightFraction ??
      null;
  const lowerBandP50 = isRadiographRenderer
    ? readSelectedRadiographMetric('lowerBandP50') ?? summary?.lowerBandP50 ?? null
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.lowerBandP50) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.lowerBandP50) ??
      summary?.lowerBandP50 ??
      null;
  const toothBandMean = isRadiographRenderer
    ? readSelectedRadiographMetric('toothBandMean') ?? summary?.toothBandMean ?? null
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.toothBandMean) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.toothBandMean) ??
      summary?.toothBandMean ??
      null;
  const toothBandP10 = isRadiographRenderer
    ? readSelectedRadiographMetric('toothBandP10') ?? summary?.toothBandP10 ?? null
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.toothBandP10) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.toothBandP10) ??
      summary?.toothBandP10 ??
      null;
  const toothBandP90 = isRadiographRenderer
    ? readSelectedRadiographMetric('toothBandP90') ?? summary?.toothBandP90 ?? null
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.toothBandP90) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.toothBandP90) ??
      summary?.toothBandP90 ??
      null;
  const toothBandContrastRange = isRadiographRenderer
    ? readSelectedRadiographMetric('toothBandContrastRange') ??
      (summary ? summary.toothBandP90 - summary.toothBandP10 : null)
    : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.toothBandContrastRange) ??
      readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.toothBandContrastRange) ??
      (summary ? summary.toothBandP90 - summary.toothBandP10 : null);
  const blackClipFraction = isRadiographRenderer
    ? readSelectedRadiographMetric('finalDisplayLowClipFraction') ??
      readPhase4DiagnosticNumber(virtualPanoRender?.blackClipFraction)
    : readPhase4DiagnosticNumber(gpuToneMap?.blackClipFraction) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.blackClipFraction);

  return {
    sampledCount: summary?.sampledCount ?? null,
    lowerBandBrightFraction,
    lowerBandP50,
    toothBandMean,
    toothBandP10,
    toothBandP90,
    toothBandContrastRange,
    toothBandHoleFraction: summary?.toothBandHoleFraction ?? null,
    toothBandBlackClipFraction: summary?.toothBandBlackClipFraction ?? null,
    toothBandRetainedWeightP10: summary?.toothBandRetainedWeightP10 ?? null,
    toothBandRetainedWeightP50: summary?.toothBandRetainedWeightP50 ?? null,
    detailBandHorizontalEdgeMean: summary?.detailBandHorizontalEdgeMean ?? null,
    detailBandVerticalEdgeMean: summary?.detailBandVerticalEdgeMean ?? null,
    supportDepthClampFraction:
      readPhase4DiagnosticNumber(supportSurface?.clampFraction) ??
      readPhase4DiagnosticNumber(supportSurface?.supportDepthClampFraction) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportDepthClampFraction),
    supportDepthStdMm: readPhase4DiagnosticNumber(supportSurface?.depthStdMm),
    pathJumpP95Mm: readPhase4DiagnosticNumber(supportSurface?.pathJumpP95Mm),
    supportConfidenceP10:
      readPhase4DiagnosticNumber(supportSurface?.confidenceP10) ??
      readPhase4DiagnosticNumber(supportSurface?.supportConfidenceP10) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportConfidenceP10),
    supportConfidenceP50:
      readPhase4DiagnosticNumber(supportSurface?.confidenceP50) ??
      readPhase4DiagnosticNumber(supportSurface?.supportConfidenceP50) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportConfidenceP50),
    supportConfidenceP90:
      readPhase4DiagnosticNumber(supportSurface?.confidenceP90) ??
      readPhase4DiagnosticNumber(supportSurface?.supportConfidenceP90) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportConfidenceP90),
    supportPathConfidenceP10:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP10) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP10) ??
      readPhase4DiagnosticNumber(supportSurface?.supportPathConfidenceP10) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportPathConfidenceP10),
    supportPathConfidenceP50:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP50) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP50) ??
      readPhase4DiagnosticNumber(supportSurface?.supportPathConfidenceP50) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportPathConfidenceP50),
    supportPathConfidenceP90:
      readPhase4DiagnosticNumber(supportSurface?.pathConfidenceP90) ??
      readPhase4DiagnosticNumber(supportSurface?.selectedReliabilityP90) ??
      readPhase4DiagnosticNumber(supportSurface?.supportPathConfidenceP90) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.supportPathConfidenceP90),
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
    troughSigmaP50Mm:
      readPhase4DiagnosticNumber(gpuDrr?.troughSigmaP50Mm) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.troughSigmaMm),
    approxTroughHalfWidthP50Mm:
      readPhase4DiagnosticNumber(gpuDrr?.approxTroughHalfWidthP50Mm) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.approxTroughHalfWidthMm),
    effectiveTroughHalfWidthP50Mm: readPhase4DiagnosticNumber(gpuDrr?.effectiveTroughHalfWidthP50Mm),
    effectiveTroughHalfWidthP90Mm: readPhase4DiagnosticNumber(gpuDrr?.effectiveTroughHalfWidthP90Mm),
    participatingSamplesP50:
      readPhase4DiagnosticNumber(gpuDrr?.participatingSamplesP50) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.participatingSamplesP50),
    participatingSamplesP90:
      readPhase4DiagnosticNumber(gpuDrr?.participatingSamplesP90) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.participatingSamplesP90),
    backgroundTroughNarrowGateP50:
      readPhase4DiagnosticNumber(gpuDrr?.backgroundTroughNarrowGateP50) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.backgroundTroughNarrowGateP50),
    backgroundTroughNarrowGateP90:
      readPhase4DiagnosticNumber(gpuDrr?.backgroundTroughNarrowGateP90) ??
      readPhase4DiagnosticNumber(virtualPanoRender?.backgroundTroughNarrowGateP90),
    blackClipFraction,
    fractionBelowMinus950: summary?.fractionBelowMinus950 ?? null,
    fractionAbove3000: summary?.fractionAbove3000 ?? null,
    renderSupportMode,
    rendererVariant,
    pipelineVariant,
    renderBypass,
    workerBranchSelected:
      readPhase4DiagnosticBoolean(outputSelection?.workerBranchSelected) ??
      readPhase4DiagnosticBoolean(outputSelection?.virtualPanoSelectedForOutput),
    workerQcAccepted:
      readPhase4DiagnosticBoolean(outputSelection?.workerQcAccepted) ??
      readPhase4DiagnosticBoolean(outputSelection?.virtualPanoAccepted) ??
      readPhase4DiagnosticBoolean(virtualPanoRender?.workerQcAccepted),
    workerQcStage,
    metricStage,
    rejectReasonsStage:
      toNonEmptyString(virtualPanoRender?.rejectReasonsStage) ?? metricStage,
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
    focalSharpnessCenterThirdP50: isRadiographRenderer
      ? readSelectedRadiographMetric('focalSharpnessCenterThirdP50')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.focalSharpnessCenterThirdP50) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.focalSharpnessCenterThirdP50) ??
        readPhase4DiagnosticNumber(virtualPanoRenderRawStage?.focalSharpnessCenterThirdP50),
    interToothValleyContrast: isRadiographRenderer
      ? readSelectedRadiographMetric('interToothValleyContrast')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.interToothValleyContrast) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.interToothValleyContrast),
    intraToothGradationScore: isRadiographRenderer
      ? readSelectedRadiographMetric('intraToothGradationScore')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.intraToothGradationScore) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.intraToothGradationScore),
    crownHighlightSaturationFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('crownHighlightSaturationFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.crownHighlightSaturationFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.crownHighlightSaturationFraction),
    occlusalDarkCapFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('occlusalDarkCapFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.occlusalDarkCapFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.occlusalDarkCapFraction),
    offTroughEnergyTopRatio: isRadiographRenderer
      ? readSelectedRadiographMetric('offTroughEnergyTopRatio')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.offTroughEnergyTopRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.offTroughEnergyTopRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderRawStage?.offTroughEnergyTopRatio),
    offTroughEnergyMiddleRatio: isRadiographRenderer
      ? readSelectedRadiographMetric('offTroughEnergyMiddleRatio')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.offTroughEnergyMiddleRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.offTroughEnergyMiddleRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderRawStage?.offTroughEnergyMiddleRatio),
    offTroughEnergyBottomRatio: isRadiographRenderer
      ? readSelectedRadiographMetric('offTroughEnergyBottomRatio')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.offTroughEnergyBottomRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.offTroughEnergyBottomRatio) ??
        readPhase4DiagnosticNumber(virtualPanoRenderRawStage?.offTroughEnergyBottomRatio),
    lowerFieldSpeckleFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('lowerFieldSpeckleFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.lowerFieldSpeckleFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.lowerFieldSpeckleFraction),
    lowerFieldVerticalStreakScore: isRadiographRenderer
      ? readSelectedRadiographMetric('lowerFieldVerticalStreakScore')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.lowerFieldVerticalStreakScore) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.lowerFieldVerticalStreakScore),
    underRootVerticalSmearScore: isRadiographRenderer
      ? readSelectedRadiographMetric('underRootVerticalSmearScore')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.underRootVerticalSmearScore) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.underRootVerticalSmearScore),
    hypothesisSwitchFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('hypothesisSwitchFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.hypothesisSwitchFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.hypothesisSwitchFraction),
    finalDisplayLowClipFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('finalDisplayLowClipFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.finalDisplayLowClipFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.finalDisplayLowClipFraction),
    finalDisplayHighClipFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('finalDisplayHighClipFraction')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.finalDisplayHighClipFraction) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.finalDisplayHighClipFraction),
    finalDisplayHistogramOccupancy: isRadiographRenderer
      ? readSelectedRadiographMetric('finalDisplayHistogramOccupancy')
      : readPhase4DiagnosticNumber(virtualPanoRenderFinalStage?.finalDisplayHistogramOccupancy) ??
        readPhase4DiagnosticNumber(virtualPanoRenderPostStage?.finalDisplayHistogramOccupancy),
    rawZeroButFinalBrightFraction: isRadiographRenderer
      ? readSelectedRadiographMetric('rawZeroButFinalBrightFraction')
      : null,
    outsideRowsBrightnessP95: isRadiographRenderer
      ? readSelectedRadiographMetric('outsideRowsBrightnessP95')
      : null,
    gapRowsBrightnessP95: isRadiographRenderer
      ? readSelectedRadiographMetric('gapRowsBrightnessP95')
      : null,
    bandBoundaryJumpMean: isRadiographRenderer
      ? readSelectedRadiographMetric('bandBoundaryJumpMean')
      : null,
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
  const virtualPanoRenderRawStage = readPhase4DiagnosticRecord(virtualPanoRender?.rawProjectionStage);
  const virtualPanoRenderPostStage = readPhase4DiagnosticRecord(
    virtualPanoRender?.postSuppressionStage
  );
  const virtualPanoRenderFinalStage = readPhase4DiagnosticRecord(virtualPanoRender?.finalDisplayStage);
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
  const workerBranchSelected =
    readPhase4DiagnosticBoolean(outputSelection?.workerBranchSelected) ??
    readPhase4DiagnosticBoolean(outputSelection?.virtualPanoSelectedForOutput);
  const workerQcAccepted =
    readPhase4DiagnosticBoolean(outputSelection?.workerQcAccepted) ??
    readPhase4DiagnosticBoolean(outputSelection?.virtualPanoAccepted) ??
    readPhase4DiagnosticBoolean(virtualPanoRender?.workerQcAccepted) ??
    readPhase4DiagnosticBoolean(virtualPanoRender?.usedAsOutput);
  const workerQcStage =
    toNonEmptyString(outputSelection?.workerQcStage) ??
    toNonEmptyString(virtualPanoRender?.workerQcStage) ??
    null;
  const workerUsedAsOutput = workerQcAccepted;
  const workerAcceptedByLowerBandTolerance =
    readPhase4DiagnosticBoolean(outputSelection?.acceptedByLowerBandTolerance) === true ||
    readPhase4DiagnosticBoolean(virtualPanoRender?.acceptedByLowerBandTolerance) === true;
  const workerAcceptedByToothBandTolerance =
    readPhase4DiagnosticBoolean(outputSelection?.acceptedByToothBandTolerance) === true ||
    readPhase4DiagnosticBoolean(virtualPanoRender?.acceptedByToothBandTolerance) === true;
  const metrics = extractPhase4QualityGateMetrics(params.summary, params.workerDebugPayload);
  const candidateSource: Phase4QualityGateCandidateSource =
    backend === 'gpu'
      ? 'worker-gpu-support-surface'
      : isVirtualPanoLikeCandidateMode(reconstructionMode) &&
          (metrics.rendererVariant === 'virtual-panoramic-radiograph' ||
            metrics.pipelineVariant === 'virtualPanoramicRadiograph' ||
            isRadiographVirtualPanoRenderMode(metrics.renderSupportMode))
        ? 'worker-cpu-virtual-panoramic-radiograph'
      : isVirtualPanoLikeCandidateMode(reconstructionMode) &&
          metrics.renderSupportMode === 'archGuidedDualLayer'
        ? 'worker-cpu-arch-guided-synthetic'
        : isVirtualPanoLikeCandidateMode(reconstructionMode)
        ? 'worker-cpu-virtual-pano'
        : 'worker-legacy';
  const radiographMetricStage = normalizePhase4RadiographMetricStageName(
    metrics.rejectReasonsStage ?? metrics.metricStage ?? workerQcStage
  );
  const selectedRadiographStageRecord =
    candidateSource === 'worker-cpu-virtual-panoramic-radiograph'
      ? resolvePhase4RadiographMetricStage({
          workerQcStage: radiographMetricStage,
          rawStage: virtualPanoRenderRawStage,
          postStage: virtualPanoRenderPostStage,
          finalStage: virtualPanoRenderFinalStage,
        }).record
      : null;
  const localizedReadout =
    candidateSource === 'worker-cpu-virtual-panoramic-radiograph'
      ? readPhase4RadiographLocalizedReadout(
          selectedRadiographStageRecord?.localizedReadout ?? virtualPanoRender?.localizedReadout
        )
      : null;
  const stagedWorkerRejectReasons =
    candidateSource === 'worker-cpu-virtual-panoramic-radiograph'
      ? workerRejectReasons.map(reason => stampPhase4ReasonStage(reason, radiographMetricStage))
      : workerRejectReasons;
  const isDualArchProjection = isDualArchProjectionRenderMode(metrics.renderSupportMode);
  const gpuConfidenceStructurallyStable =
    candidateSource === 'worker-gpu-support-surface' &&
    (metrics.supportPathConfidenceP50 ?? 0) >= 0.68 &&
    (metrics.supportUnstableColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.06 &&
    (metrics.supportLongestUnstableRunColumns ?? Number.POSITIVE_INFINITY) <= 4;
  const severeGpuToothBandHoles =
    candidateSource === 'worker-gpu-support-surface' &&
    metrics.toothBandHoleFraction !== null &&
    metrics.toothBandHoleFraction > CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION;
  const severeGpuToothBandBlackClip =
    candidateSource === 'worker-gpu-support-surface' &&
    metrics.toothBandBlackClipFraction !== null &&
    metrics.toothBandBlackClipFraction > CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION;
  const severeGpuToothBandRetentionCollapse =
    candidateSource === 'worker-gpu-support-surface' &&
    metrics.toothBandRetainedWeightP10 !== null &&
    metrics.toothBandRetainedWeightP10 <= CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX;
  const dominantGpuMiddleBandLeakage =
    candidateSource === 'worker-gpu-support-surface' &&
    hasPhase4DominantMiddleBandLeakage(params.summary);
  const workerAcceptedVirtualPanoStructurallyUsable =
    candidateSource === 'worker-cpu-virtual-pano' &&
    workerQcAccepted === true &&
    (workerAcceptedByLowerBandTolerance ||
      workerAcceptedByToothBandTolerance ||
      (((metrics.lowerBandBrightFraction ?? Number.POSITIVE_INFINITY) <= 0.14) &&
        ((metrics.lowerBandP50 ?? Number.POSITIVE_INFINITY) <= -300) &&
        ((metrics.toothBandContrastRange ?? 0) >= 500) &&
        ((metrics.toothBandBlackClipFraction ?? Number.POSITIVE_INFINITY) <= 0.12) &&
        ((metrics.supportUnstableColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.28) &&
        ((metrics.supportAmbiguousColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.68) &&
        ((metrics.supportScoreGapP50 ?? 0) >= 0.03) &&
        ((metrics.supportBestDepthDriftP95Mm ?? Number.POSITIVE_INFINITY) <= 1.3)));
  const isPanoV2FusionBypassCandidate =
    candidateSource === 'worker-cpu-virtual-pano' &&
    metrics.rendererVariant === 'pano-v2-fusion' &&
    metrics.renderBypass === true;
  const effectiveHardRejectReason =
    isPanoV2FusionBypassCandidate && params.hardRejectReason === 'tooth-band-saturation'
      ? null
      : params.hardRejectReason ?? null;

  const rejectReasons: string[] = [];
  const addRejectReason = (reason: string): void => {
    rejectReasons.push(
      candidateSource === 'worker-cpu-virtual-panoramic-radiograph'
        ? stampPhase4ReasonStage(reason, radiographMetricStage)
        : reason
    );
  };
  if (!params.summary || params.summary.sampledCount < 100) {
    addRejectReason('summary-unavailable');
  }
  if (candidateSource === 'worker-gpu-support-surface' && routeDiagnostic.phase2GatePassed === false) {
    addRejectReason('gpu-phase2-gate-failed');
  }
  if (effectiveHardRejectReason) {
    addRejectReason(`hard-reject:${effectiveHardRejectReason}`);
  }
  if (
    candidateSource === 'worker-legacy' &&
    requestedReconstructionMode !== 'legacy' &&
    !CPR_PANO_ALLOW_REFERENCE_OR_LEGACY_FALLBACK
  ) {
    addRejectReason('legacy-fallback-not-allowed');
  }
  if (
    candidateSource !== 'worker-cpu-virtual-panoramic-radiograph' &&
    metrics.lowerBandBrightFraction !== null &&
    metrics.lowerBandBrightFraction > (isDualArchProjection ? 0.94 : 0.25)
  ) {
    addRejectReason('lower-band-bright-fraction-too-high');
  }
  if (
    candidateSource !== 'worker-cpu-virtual-panoramic-radiograph' &&
    metrics.lowerBandP50 !== null &&
    metrics.lowerBandP50 > (isDualArchProjection ? 420 : -100)
  ) {
    addRejectReason('lower-band-p50-too-high');
  }
  if (
    candidateSource !== 'worker-cpu-virtual-panoramic-radiograph' &&
    !isPanoV2FusionBypassCandidate &&
    metrics.toothBandContrastRange !== null &&
    metrics.toothBandContrastRange < (isDualArchProjection ? 120 : 150)
  ) {
    addRejectReason('tooth-band-contrast-too-low');
  }
  if (
    metrics.supportDepthClampFraction !== null &&
    metrics.supportDepthClampFraction > 0.15
  ) {
    addRejectReason('support-depth-clamp-fraction-too-high');
  }
  if (metrics.pathJumpP95Mm !== null && metrics.pathJumpP95Mm > 1.2) {
    addRejectReason('path-jump-p95-too-high');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    (((metrics.supportConfidenceP50 !== null && metrics.supportConfidenceP50 < 0.09) &&
      !gpuConfidenceStructurallyStable) ||
      (metrics.supportPathConfidenceP50 !== null && metrics.supportPathConfidenceP50 < 0.15))
  ) {
    addRejectReason('support-confidence-too-low');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    (((params.summary?.backgroundOutlierFraction05 ?? 0) > 0.24) ||
      ((params.summary?.backgroundOutlierFraction05 ?? 0) > 0.20 &&
        (params.summary?.backgroundOutlierFraction10 ?? 0) > 0.09))
  ) {
    addRejectReason('background-outlier-fraction-too-high');
  }
  if (severeGpuToothBandHoles) {
    addRejectReason('tooth-band-hole-fraction-too-high');
  }
  if (severeGpuToothBandBlackClip) {
    addRejectReason('tooth-band-black-clip-too-high');
  }
  if (severeGpuToothBandRetentionCollapse) {
    addRejectReason('tooth-band-retained-weight-collapsed');
  }
  if (dominantGpuMiddleBandLeakage) {
    addRejectReason('middle-band-leakage-dominant');
  }
  if (
    candidateSource === 'worker-gpu-support-surface' &&
    (((metrics.supportUnstableColumnFraction ?? 0) > 0.24) ||
      ((metrics.supportLongestUnstableRunColumns ?? 0) > 30) ||
      (((metrics.supportAmbiguousColumnFraction ?? 0) > 0.2) &&
        ((metrics.supportScoreGapP50 ?? 1) < 0.11)))
  ) {
    addRejectReason('support-columns-unstable');
  }
  if (
    candidateSource === 'worker-cpu-virtual-pano' &&
    !isPanoV2FusionBypassCandidate &&
    !workerAcceptedVirtualPanoStructurallyUsable &&
    (((metrics.supportUnstableColumnFraction ?? 0) > (isDualArchProjection ? 0.36 : 0.24)) ||
      ((metrics.supportLongestUnstableRunColumns ?? 0) > (isDualArchProjection ? 28 : 14)) ||
      ((metrics.supportDepthStdMm ?? 0) > (isDualArchProjection ? 1.8 : 1.3)))
  ) {
    addRejectReason('virtual-support-columns-unstable');
  }
  if (
    candidateSource === 'worker-cpu-virtual-pano' &&
    !workerAcceptedVirtualPanoStructurallyUsable &&
    (((metrics.supportAmbiguousColumnFraction ?? 0) > (isDualArchProjection ? 0.62 : 0.4) &&
      (metrics.supportScoreGapP50 ?? 1) < (isDualArchProjection ? 0.05 : 0.08)) ||
      ((metrics.supportForcedDriftFraction ?? 0) > (isDualArchProjection ? 0.2 : 0.12)) ||
      ((metrics.supportBestDepthDriftP95Mm ?? 0) > (isDualArchProjection ? 2.6 : 1.8)))
  ) {
    addRejectReason('virtual-support-ambiguity-too-high');
  }
  if (candidateSource === 'worker-cpu-virtual-panoramic-radiograph') {
    if ((metrics.focalSharpnessCenterThirdP50 ?? Number.POSITIVE_INFINITY) < 0.56) {
      addRejectReason('radiograph-focal-trough-sharpness-too-low');
    }
    if ((metrics.interToothValleyContrast ?? Number.POSITIVE_INFINITY) < 70) {
      addRejectReason('radiograph-tooth-separation-too-low');
    }
    if ((metrics.intraToothGradationScore ?? Number.POSITIVE_INFINITY) < 52) {
      addRejectReason('radiograph-internal-gradation-too-low');
    }
    if ((metrics.crownHighlightSaturationFraction ?? 0) > 0.18) {
      addRejectReason('radiograph-crown-highlight-saturation-too-high');
    }
    if ((metrics.occlusalDarkCapFraction ?? 0) > 0.19) {
      addRejectReason('radiograph-occlusal-dark-cap-too-high');
    }
    if ((metrics.offTroughEnergyTopRatio ?? 0) > 0.52) {
      addRejectReason('radiograph-off-trough-top-energy-too-high');
    }
    if ((metrics.offTroughEnergyMiddleRatio ?? 0) > 0.38) {
      addRejectReason('radiograph-off-trough-middle-energy-too-high');
    }
    if ((metrics.offTroughEnergyBottomRatio ?? 0) > 0.34) {
      addRejectReason('radiograph-off-trough-bottom-energy-too-high');
    }
    if (
      (metrics.lowerBandBrightFraction ?? 0) > 0.18 ||
      (metrics.rawZeroButFinalBrightFraction ?? 0) > 0.04 ||
      (metrics.outsideRowsBrightnessP95 ?? 0) > 0.15 ||
      (metrics.gapRowsBrightnessP95 ?? 0) > 0.15
    ) {
      addRejectReason('radiograph-lower-field-bright-fill-too-high');
    }
    if ((metrics.bandBoundaryJumpMean ?? 0) > 0.14) {
      addRejectReason('radiograph-band-boundary-jump-too-high');
    }
    if ((metrics.lowerFieldSpeckleFraction ?? 0) > 0.11) {
      addRejectReason('radiograph-lower-field-speckle-too-high');
    }
    if ((metrics.lowerFieldVerticalStreakScore ?? 0) > 2.15) {
      addRejectReason('radiograph-lower-field-vertical-streak-too-high');
    }
    if ((metrics.underRootVerticalSmearScore ?? 0) > 2.5) {
      addRejectReason('radiograph-under-root-vertical-smear-too-high');
    }
    if ((metrics.hypothesisSwitchFraction ?? 0) > 0.18) {
      addRejectReason('radiograph-hypothesis-switch-too-high');
    }
    if ((metrics.finalDisplayLowClipFraction ?? 0) > 0.11) {
      addRejectReason('radiograph-final-display-low-clipping-too-high');
    }
    if ((metrics.finalDisplayHighClipFraction ?? 0) > 0.095) {
      addRejectReason('radiograph-final-display-high-clipping-too-high');
    }
    if ((metrics.finalDisplayHistogramOccupancy ?? Number.POSITIVE_INFINITY) < 0.16) {
      addRejectReason('radiograph-final-display-histogram-occupancy-too-low');
    }
  }
  if (
    candidateSource === 'worker-cpu-arch-guided-synthetic' &&
    (((metrics.contextWeightFractionMean ?? 0) < 0.045) ||
      ((metrics.detailSampleFractionMean ?? 0) < 0.028) ||
      ((metrics.columnSupportReliabilityP50 ?? 0) < 0.2))
  ) {
    addRejectReason('arch-guided-context-coverage-too-low');
  }
  if (
    candidateSource === 'worker-cpu-arch-guided-synthetic' &&
    (((metrics.toothBandContrastRange ?? 0) < 180) ||
      ((metrics.contextHuP50 ?? -1000) < -720) ||
      ((metrics.upperDetailHuP50 ?? -1000) < 40 && (metrics.lowerDetailHuP50 ?? -1000) < 40))
  ) {
    addRejectReason('arch-guided-anatomy-contrast-too-low');
  }
  if (!workerAcceptedByLowerBandTolerance && workerQcAccepted === false) {
    rejectReasons.push(...stagedWorkerRejectReasons);
  }

  const uniqueRejectReasons = Array.from(new Set(rejectReasons));
  const lowerBandOnlyRejects =
    uniqueRejectReasons.length > 0 &&
    uniqueRejectReasons.every(reason => isPhase4LowerBandRejectReason(reason));
  const isPanoV2FusionCandidate =
    candidateSource === 'worker-cpu-virtual-pano' && metrics.renderSupportMode === 'panoV2Fusion';
  const severeCpuLowerBandContamination =
    candidateSource === 'worker-cpu-virtual-pano' &&
    ((metrics.lowerBandBrightFraction ?? 0) > 0.38 ||
      (metrics.lowerBandP50 ?? Number.POSITIVE_INFINITY) > -80);
  const ambiguityOnlyRejects =
    uniqueRejectReasons.length > 0 &&
    uniqueRejectReasons.every(reason => reason === 'virtual-support-ambiguity-too-high');
  const ambiguityOnlyBorderlineAccepted =
    candidateSource === 'worker-cpu-virtual-pano' &&
    isDualArchProjection &&
    !isPanoV2FusionCandidate &&
    ambiguityOnlyRejects &&
    workerUsedAsOutput !== false &&
    (params.qualityScore ?? Number.NEGATIVE_INFINITY) >= 24 &&
    (metrics.supportAmbiguousColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.72 &&
    (metrics.supportScoreGapP50 ?? 0) >= 0.018 &&
    (metrics.supportPathConfidenceP50 ?? 0) >= 0.68 &&
    (metrics.toothBandContrastRange ?? 0) >= 260 &&
    (metrics.lowerBandBrightFraction ?? Number.POSITIVE_INFINITY) <= 0.7 &&
    (metrics.lowerBandP50 ?? Number.POSITIVE_INFINITY) <= 140;
  const borderlineAcceptedReason =
    workerAcceptedByLowerBandTolerance && !isPanoV2FusionCandidate
      ? 'lower-band-tolerated-in-worker'
      : lowerBandOnlyRejects &&
          !isPanoV2FusionCandidate &&
          !severeCpuLowerBandContamination &&
          (params.qualityScore ?? Number.NEGATIVE_INFINITY) >= 12 &&
          (metrics.toothBandContrastRange ?? 0) >= 150
        ? 'lower-band-only-borderline'
        : ambiguityOnlyBorderlineAccepted
          ? 'ambiguity-only-borderline'
        : null;
  const pass =
    uniqueRejectReasons.length === 0 ||
    (borderlineAcceptedReason !== null &&
      (lowerBandOnlyRejects || ambiguityOnlyBorderlineAccepted));
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
    hardRejectReason: effectiveHardRejectReason,
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
    hardRejectReason: effectiveHardRejectReason,
    workerRejectReasons: stagedWorkerRejectReasons,
    workerBranchSelected,
    workerQcAccepted,
    workerQcStage,
    metricStage: metrics.metricStage,
    rejectReasonsStage: metrics.rejectReasonsStage,
    workerUsedAsOutput,
    orchestratorAccepted: pass,
    borderlineAcceptedReason,
    pass,
    rejectReasons: uniqueRejectReasons,
    metrics,
    localizedReadout,
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
  const isRadiographCandidate =
    params.candidateSource === 'worker-cpu-virtual-panoramic-radiograph';
  const radiographStage = normalizePhase4RadiographMetricStageName(
    params.metrics.rejectReasonsStage ?? params.metrics.metricStage ?? params.metrics.workerQcStage
  );
  const usesLegacySupportPathRiskModel =
    !isRadiographCandidate && params.candidateSource !== 'worker-cpu-arch-guided-synthetic';
  const isDualArchProjection = isDualArchProjectionRenderMode(params.metrics.renderSupportMode);
  const gpuConfidenceStructurallyStable =
    params.candidateSource === 'worker-gpu-support-surface' &&
    (params.metrics.supportPathConfidenceP50 ?? 0) >= 0.68 &&
    (params.metrics.supportUnstableColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.06 &&
    (params.metrics.supportLongestUnstableRunColumns ?? Number.POSITIVE_INFINITY) <= 4;
  const backgroundOutlierFraction05 = params.summary?.backgroundOutlierFraction05 ?? null;
  const backgroundOutlierFraction10 = params.summary?.backgroundOutlierFraction10 ?? null;
  const detailBandHorizontalEdgeMean = params.summary?.detailBandHorizontalEdgeMean ?? null;
  const detailBandVerticalEdgeMean = params.summary?.detailBandVerticalEdgeMean ?? null;
  const toothBandHoleFraction = params.metrics.toothBandHoleFraction;
  const toothBandBlackClipFraction = params.metrics.toothBandBlackClipFraction;
  const toothBandRetainedWeightP10 = params.metrics.toothBandRetainedWeightP10;
  const toothBandRetainedWeightP50 = params.metrics.toothBandRetainedWeightP50;
  const addRiskFlag = (flag: string, condition: boolean, weight = 1): void => {
    if (!condition) {
      return;
    }
    riskFlags.push(flag);
    riskScore += weight;
  };
  const stageRiskFlag = (flag: string): string =>
    isRadiographCandidate ? stampPhase4ReasonStage(flag, radiographStage) : flag;

  addRiskFlag(
    'support-confidence-low',
    usesLegacySupportPathRiskModel &&
      !gpuConfidenceStructurallyStable &&
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
    stageRiskFlag('background-clipped'),
    !isRadiographCandidate &&
      params.metrics.blackClipFraction !== null &&
      params.metrics.blackClipFraction > 0.42,
    2
  );
  addRiskFlag(
    stageRiskFlag('background-leakage'),
    !isRadiographCandidate &&
      ((backgroundOutlierFraction05 !== null && backgroundOutlierFraction05 > 0.18) ||
        (backgroundOutlierFraction10 !== null && backgroundOutlierFraction10 > 0.08)),
    2
  );
  addRiskFlag(
    stageRiskFlag('lower-band-filled'),
    !isRadiographCandidate &&
      (((params.metrics.lowerBandBrightFraction !== null &&
        params.metrics.lowerBandBrightFraction > (isDualArchProjection ? 0.9 : 0.25)) ||
        (params.metrics.lowerBandP50 !== null &&
          params.metrics.lowerBandP50 > (isDualArchProjection ? 360 : -100)))),
    2
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-contrast-low'),
    !isRadiographCandidate &&
      params.metrics.toothBandContrastRange !== null &&
      params.metrics.toothBandContrastRange > 0 &&
      params.metrics.toothBandContrastRange < (isDualArchProjection ? 120 : 150),
    2
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-saturated'),
    !isRadiographCandidate &&
      params.metrics.toothBandMean !== null &&
      params.metrics.toothBandMean > 760,
    1
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-holes'),
    !isRadiographCandidate && toothBandHoleFraction !== null && toothBandHoleFraction > 0.028,
    4
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-black-clipped'),
    !isRadiographCandidate &&
      toothBandBlackClipFraction !== null &&
      toothBandBlackClipFraction > CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION,
    3
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-retention-low'),
    !isRadiographCandidate &&
      ((toothBandRetainedWeightP10 !== null &&
        toothBandRetainedWeightP10 < CPR_TOOTH_BAND_HOLE_RETAINED_WEIGHT_MAX) ||
        (toothBandRetainedWeightP50 !== null && toothBandRetainedWeightP50 < 0.42)),
    2
  );
  addRiskFlag(
    stageRiskFlag('tooth-band-retention-collapsed'),
    !isRadiographCandidate &&
      toothBandRetainedWeightP10 !== null &&
      toothBandRetainedWeightP10 <= CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX,
    5
  );
  addRiskFlag(
    stageRiskFlag('middle-band-leakage-dominant'),
    !isRadiographCandidate && hasPhase4DominantMiddleBandLeakage(params.summary),
    3
  );
  addRiskFlag(
    stageRiskFlag('air-suppression-weak'),
    !isRadiographCandidate &&
      params.metrics.fractionBelowMinus950 !== null &&
      params.metrics.fractionBelowMinus950 < (isDualArchProjection ? 0.005 : 0.015),
    1
  );
  addRiskFlag(
    stageRiskFlag('upper-lower-balance-poor'),
    !isRadiographCandidate &&
      detailBandVerticalEdgeMean !== null &&
      detailBandVerticalEdgeMean > 0 &&
      detailBandHorizontalEdgeMean !== null &&
      detailBandHorizontalEdgeMean / Math.max(1, detailBandVerticalEdgeMean) > 2.8,
    1
  );
  addRiskFlag(
    stageRiskFlag('radiograph-focal-trough-sharpness-soft'),
    isRadiographCandidate &&
      (params.metrics.focalSharpnessCenterThirdP50 ?? Number.POSITIVE_INFINITY) < 0.6,
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-tooth-separation-soft'),
    isRadiographCandidate &&
      (params.metrics.interToothValleyContrast ?? Number.POSITIVE_INFINITY) < 78,
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-internal-gradation-soft'),
    isRadiographCandidate &&
      (params.metrics.intraToothGradationScore ?? Number.POSITIVE_INFINITY) < 60,
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-crown-highlights-hot'),
    isRadiographCandidate &&
      (params.metrics.crownHighlightSaturationFraction ?? 0) > 0.14,
    2
  );
  addRiskFlag(
    stageRiskFlag('radiograph-dark-roof-over-crowns'),
    isRadiographCandidate &&
      (params.metrics.occlusalDarkCapFraction ?? 0) > 0.14,
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-off-trough-top-energy-high'),
    isRadiographCandidate &&
      (params.metrics.offTroughEnergyTopRatio ?? 0) > 0.44,
    2
  );
  addRiskFlag(
    stageRiskFlag('radiograph-off-trough-middle-energy-high'),
    isRadiographCandidate &&
      (params.metrics.offTroughEnergyMiddleRatio ?? 0) > 0.32,
    2
  );
  addRiskFlag(
    stageRiskFlag('radiograph-off-trough-bottom-energy-high'),
    isRadiographCandidate &&
      (params.metrics.offTroughEnergyBottomRatio ?? 0) > 0.28,
    2
  );
  addRiskFlag(
    stageRiskFlag('radiograph-lower-field-grainy'),
    isRadiographCandidate &&
      (((params.metrics.lowerFieldSpeckleFraction ?? 0) > 0.09) ||
        ((params.metrics.lowerFieldVerticalStreakScore ?? 0) > 1.8)),
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-under-root-smear'),
    isRadiographCandidate &&
      (params.metrics.underRootVerticalSmearScore ?? 0) > 2.0,
    3
  );
  addRiskFlag(
    stageRiskFlag('radiograph-hypothesis-switching'),
    isRadiographCandidate &&
      (params.metrics.hypothesisSwitchFraction ?? 0) > 0.12,
    2
  );
  addRiskFlag(
    stageRiskFlag('radiograph-final-display-clipped'),
    isRadiographCandidate &&
      (((params.metrics.finalDisplayLowClipFraction ?? 0) > 0.09) ||
        ((params.metrics.finalDisplayHighClipFraction ?? 0) > 0.08) ||
        ((params.metrics.finalDisplayHistogramOccupancy ?? Number.POSITIVE_INFINITY) < 0.2)),
    3
  );
  addRiskFlag(stageRiskFlag('candidate-hard-reject'), !!params.hardRejectReason, 4);
  addRiskFlag(stageRiskFlag('quality-gate-failed'), !params.pass, 2);
  addRiskFlag(
    stageRiskFlag('borderline-accepted'),
    params.borderlineAcceptedReason !== null && !params.pass,
    1
  );

  const riskLevel =
    riskScore >= 9 ? 'high' : riskScore >= 6 ? 'elevated' : riskScore >= 3 ? 'moderate' : 'low';
  const baselineFingerprint = isRadiographCandidate
    ? [
        `backend=${params.backend}`,
        `requestedBackend=${params.requestedBackend ?? 'null'}`,
        `reconstructionMode=${params.reconstructionMode}`,
        `pipelineMode=${params.pipelineMode}`,
        `candidateSource=${params.candidateSource}`,
        `workerBranchSelected=${params.metrics.workerBranchSelected === null ? 'null' : params.metrics.workerBranchSelected ? 1 : 0}`,
        `workerQcAccepted=${params.metrics.workerQcAccepted === null ? 'null' : params.metrics.workerQcAccepted ? 1 : 0}`,
        `workerQcStage=${params.metrics.workerQcStage ?? 'null'}`,
        `metricStage=${params.metrics.metricStage ?? 'null'}`,
        `rejectStage=${params.metrics.rejectReasonsStage ?? 'null'}`,
        `qualityBase=${formatPhase4FingerprintValue(params.qualityBase, 2)}`,
        `qualityScore=${formatPhase4FingerprintValue(params.qualityScore, 2)}`,
        `pass=${params.pass ? 1 : 0}`,
        `hardReject=${params.hardRejectReason ?? 'none'}`,
        `borderline=${params.borderlineAcceptedReason ?? 'none'}`,
        `focalSharpness=${formatPhase4FingerprintValue(
          params.metrics.focalSharpnessCenterThirdP50,
          4
        )}`,
        `toothSeparation=${formatPhase4FingerprintValue(
          params.metrics.interToothValleyContrast,
          2
        )}`,
        `internalGradation=${formatPhase4FingerprintValue(
          params.metrics.intraToothGradationScore,
          2
        )}`,
        `crownHot=${formatPhase4FingerprintValue(
          params.metrics.crownHighlightSaturationFraction,
          4
        )}`,
        `darkRoof=${formatPhase4FingerprintValue(params.metrics.occlusalDarkCapFraction, 4)}`,
        `offTroughTop=${formatPhase4FingerprintValue(params.metrics.offTroughEnergyTopRatio, 4)}`,
        `offTroughMid=${formatPhase4FingerprintValue(
          params.metrics.offTroughEnergyMiddleRatio,
          4
        )}`,
        `offTroughBottom=${formatPhase4FingerprintValue(
          params.metrics.offTroughEnergyBottomRatio,
          4
        )}`,
        `lowerSpeckle=${formatPhase4FingerprintValue(
          params.metrics.lowerFieldSpeckleFraction,
          4
        )}`,
        `lowerStreak=${formatPhase4FingerprintValue(
          params.metrics.lowerFieldVerticalStreakScore,
          3
        )}`,
        `underRootSmear=${formatPhase4FingerprintValue(
          params.metrics.underRootVerticalSmearScore,
          3
        )}`,
        `hypothesisSwitch=${formatPhase4FingerprintValue(
          params.metrics.hypothesisSwitchFraction,
          4
        )}`,
        `lowClip=${formatPhase4FingerprintValue(
          params.metrics.finalDisplayLowClipFraction,
          4
        )}`,
        `highClip=${formatPhase4FingerprintValue(
          params.metrics.finalDisplayHighClipFraction,
          4
        )}`,
        `histOcc=${formatPhase4FingerprintValue(
          params.metrics.finalDisplayHistogramOccupancy,
          4
        )}`,
        `lowerBandBright=${formatPhase4FingerprintValue(params.metrics.lowerBandBrightFraction, 4)}`,
        `lowerBandP50=${formatPhase4FingerprintValue(params.metrics.lowerBandP50)}`,
        `toothContrast=${formatPhase4FingerprintValue(params.metrics.toothBandContrastRange)}`,
        `detailP50Upper=${formatPhase4FingerprintValue(params.metrics.upperDetailHuP50)}`,
        `detailP50Lower=${formatPhase4FingerprintValue(params.metrics.lowerDetailHuP50)}`,
        `detailFraction=${formatPhase4FingerprintValue(params.metrics.detailSampleFractionMean, 4)}`,
        `shadowLift=${formatPhase4FingerprintValue(params.metrics.shadowLiftMean, 4)}`,
        `flags=${riskFlags.length ? riskFlags.join(',') : 'none'}`,
      ].join('|')
    : [
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
        `bgBand05=${params.summary?.backgroundBands?.dominantOutlierBand05 ?? 'none'}`,
        `bgBand10=${params.summary?.backgroundBands?.dominantOutlierBand10 ?? 'none'}`,
        `bgMid05=${formatPhase4FingerprintValue(
          params.summary?.backgroundBands?.middle.backgroundOutlierContribution05,
          4
        )}`,
        `bgMid10=${formatPhase4FingerprintValue(
          params.summary?.backgroundBands?.middle.backgroundOutlierContribution10,
          4
        )}`,
        `lowerBandBright=${formatPhase4FingerprintValue(params.metrics.lowerBandBrightFraction, 4)}`,
        `lowerBandP50=${formatPhase4FingerprintValue(params.metrics.lowerBandP50)}`,
        `toothContrast=${formatPhase4FingerprintValue(params.metrics.toothBandContrastRange)}`,
        `toothHole=${formatPhase4FingerprintValue(toothBandHoleFraction, 4)}`,
        `toothBlackClip=${formatPhase4FingerprintValue(toothBandBlackClipFraction, 4)}`,
        `toothRetainedP10=${formatPhase4FingerprintValue(toothBandRetainedWeightP10, 4)}`,
        `toothRetainedP50=${formatPhase4FingerprintValue(toothBandRetainedWeightP50, 4)}`,
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
      backgroundBands: params.summary?.backgroundBands ?? null,
    },
    anatomy: {
      lowerBandBrightFraction: params.metrics.lowerBandBrightFraction,
      lowerBandP50: params.metrics.lowerBandP50,
      toothBandMean: params.metrics.toothBandMean,
      toothBandP10: params.metrics.toothBandP10,
      toothBandP90: params.metrics.toothBandP90,
      toothBandContrastRange: params.metrics.toothBandContrastRange,
      toothBandHoleFraction,
      toothBandBlackClipFraction,
      toothBandRetainedWeightP10,
      toothBandRetainedWeightP50,
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
    workerBranchSelected: candidate.workerBranchSelected,
    workerQcAccepted: candidate.workerQcAccepted,
    workerQcStage: candidate.workerQcStage,
    metricStage: candidate.metricStage,
    rejectReasonsStage: candidate.rejectReasonsStage,
    workerUsedAsOutput: candidate.workerUsedAsOutput,
    orchestratorAccepted: candidate.orchestratorAccepted,
    borderlineAcceptedReason: candidate.borderlineAcceptedReason,
    pass: candidate.pass,
    rejectReasons: candidate.rejectReasons,
    metrics: candidate.metrics,
    localizedReadout: candidate.localizedReadout,
    supportSurfaceRiskSummary: candidate.supportSurfaceRiskSummary,
  };
}

function collectPhase4DegradedPreviewBlockedReasons(
  candidate: Phase4QualityGateCandidate | null
): string[] {
  if (!candidate) {
    return ['candidate-missing'];
  }

  const blockedReasons: string[] = [];
  for (const rejectReason of candidate.rejectReasons) {
    if (
      rejectReason === 'tooth-band-hole-fraction-too-high' ||
      rejectReason === 'tooth-band-black-clip-too-high' ||
      rejectReason === 'tooth-band-retained-weight-collapsed' ||
      rejectReason === 'middle-band-leakage-dominant'
    ) {
      blockedReasons.push(rejectReason);
    }
  }

  if (
    candidate.metrics.toothBandHoleFraction !== null &&
    candidate.metrics.toothBandHoleFraction > CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION
  ) {
    blockedReasons.push('tooth-band-hole-fraction-too-high');
  }
  if (
    candidate.metrics.toothBandBlackClipFraction !== null &&
    candidate.metrics.toothBandBlackClipFraction > CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION
  ) {
    blockedReasons.push('tooth-band-black-clip-too-high');
  }
  if (
    candidate.metrics.toothBandRetainedWeightP10 !== null &&
    candidate.metrics.toothBandRetainedWeightP10 <= CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX
  ) {
    blockedReasons.push('tooth-band-retained-weight-collapsed');
  }
  if (
    candidate.supportSurfaceRiskSummary.riskFlags.includes('middle-band-leakage-dominant') ||
    candidate.rejectReasons.includes('middle-band-leakage-dominant')
  ) {
    blockedReasons.push('middle-band-leakage-dominant');
  }

  return Array.from(new Set(blockedReasons));
}

function assessPhase4DegradedPreviewCandidate(
  candidate: Phase4QualityGateCandidate | null
): Phase4DegradedPreviewAssessment {
  if (!candidate) {
    return {
      catastrophic: true,
      catastrophicReasons: ['candidate-missing'],
      blockedAsDegradedPreview: true,
      blockedAsDegradedPreviewReasons: ['candidate-missing'],
    };
  }

  const catastrophicReasons: string[] = [];
  const blockedAsDegradedPreviewReasons = collectPhase4DegradedPreviewBlockedReasons(candidate);
  const nonLowerBandWorkerRejectReasons = candidate.workerRejectReasons.filter(
    reason => !isPhase4LowerBandRejectReason(reason)
  );
  const isDualArchProjection = isDualArchProjectionRenderMode(candidate.metrics.renderSupportMode);
  const allowVirtualPanoSupportInstabilityPreview =
    candidate.hardRejectReason === 'virtual-pano-support-instability' &&
    isVirtualPanoLikeCandidateMode(candidate.reconstructionMode) &&
    candidate.backend === 'cpu' &&
    (candidate.metrics.lowerBandBrightFraction ?? Number.POSITIVE_INFINITY) <= 0.18 &&
    (candidate.metrics.lowerBandP50 ?? Number.POSITIVE_INFINITY) <= -320 &&
    (candidate.metrics.toothBandContrastRange ?? 0) >= 500 &&
    (candidate.metrics.toothBandP10 ?? Number.POSITIVE_INFINITY) <= 150 &&
    (candidate.metrics.toothBandBlackClipFraction ?? Number.POSITIVE_INFINITY) <= 0.065;

  if ((candidate.metrics.sampledCount ?? 0) < 100) {
    catastrophicReasons.push('summary-unavailable');
  }
  if (candidate.hardRejectReason && !allowVirtualPanoSupportInstabilityPreview) {
    catastrophicReasons.push(`hard-reject:${candidate.hardRejectReason}`);
  }
  if (candidate.workerQcAccepted === false && nonLowerBandWorkerRejectReasons.length > 0) {
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
    blockedAsDegradedPreview: blockedAsDegradedPreviewReasons.length > 0,
    blockedAsDegradedPreviewReasons,
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
  const preToneAccumulationMap = debugMaps?.preToneAccumulationMap;
  const retainedSampleMaskMap = debugMaps?.retainedSampleMaskMap;
  const middleBandLeakMap = debugMaps?.middleBandLeakMap;
  const selectedSupportHypothesisMap = debugMaps?.selectedSupportHypothesisMap;
  const rawProjectedAttenuationMap = debugMaps?.rawProjectedAttenuationMap;
  const backgroundToneSamples: number[] = [];
  let backgroundToneOutlierCount05 = 0;
  let backgroundToneOutlierCount10 = 0;
  let backgroundToneMax = 0;
  const toothBandPreToneAccumulationSamples: number[] = [];
  const toothBandRetainedWeightSamples: number[] = [];
  const toothBandToneResponseSamples: number[] = [];
  const toothBandLeakSamples: number[] = [];
  let toothBandRetainedWeightRowValues: number[] = [];
  let toothBandRetainedWeightColumnValues: number[] = [];
  const toothBandHoleSignals: Array<{
    preToneAccumulation: number;
    retainedWeight: number;
    toneResponse: number;
    middleBandLeak: number;
  }> = [];
  const backgroundBandBuckets: Record<
    FloatBufferBackgroundBandName,
    {
      samples: number[];
      outlierCount05: number;
      outlierCount10: number;
      toneMax: number;
    }
  > = {
    top: { samples: [], outlierCount05: 0, outlierCount10: 0, toneMax: 0 },
    middle: { samples: [], outlierCount05: 0, outlierCount10: 0, toneMax: 0 },
    bottom: { samples: [], outlierCount05: 0, outlierCount10: 0, toneMax: 0 },
  };

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
    const isRadiographDisplayDomain =
      (!!rawProjectedAttenuationMap || !!selectedSupportHypothesisMap) && max >= 512;
    const rowFromNormalizedOffset = (yNorm: number): number => {
      const row = Math.round(panoCenterRow + yNorm * panoCenterRow);
      return Math.max(0, Math.min(safeHeight - 1, row));
    };
    const deriveRadiographBandRows = ():
      | {
          upperRows: [number, number];
          gapRows: [number, number];
          lowerRows: [number, number];
          bgTopRows: [number, number];
          bgGapRows: [number, number];
          bgBottomRows: [number, number];
        }
      | null => {
      if (
        !selectedSupportHypothesisMap ||
        selectedSupportHypothesisMap.length < safeWidth * safeHeight
      ) {
        return null;
      }
      const upperCounts = new Float32Array(safeHeight);
      const lowerCounts = new Float32Array(safeHeight);
      for (let row = 0; row < safeHeight; row += rowStep) {
        for (let col = 0; col < safeWidth; col += colStep) {
          const code = Math.round(Number(selectedSupportHypothesisMap[row * safeWidth + col]) || 0);
          if (code === 1) {
            upperCounts[row] += 1;
          } else if (code === 2) {
            lowerCounts[row] += 1;
          }
        }
      }
      let upperAnchorRow = -1;
      let lowerAnchorRow = -1;
      let upperAnchorCount = 0;
      let lowerAnchorCount = 0;
      for (let row = 0; row < safeHeight; row++) {
        if (upperCounts[row] > upperAnchorCount) {
          upperAnchorCount = upperCounts[row];
          upperAnchorRow = row;
        }
        if (lowerCounts[row] > lowerAnchorCount) {
          lowerAnchorCount = lowerCounts[row];
          lowerAnchorRow = row;
        }
      }
      if (
        upperAnchorRow < 0 ||
        lowerAnchorRow < 0 ||
        upperAnchorCount <= 0 ||
        lowerAnchorCount <= 0 ||
        lowerAnchorRow <= upperAnchorRow
      ) {
        return null;
      }
      const midRow = Math.round(0.5 * (upperAnchorRow + lowerAnchorRow));
      const toothHalf = Math.max(
        16,
        Math.min(24, Math.round(0.35 * (lowerAnchorRow - upperAnchorRow)))
      );
      const gapHalf = 5;
      return {
        upperRows: [
          Math.max(0, upperAnchorRow - toothHalf),
          Math.max(0, Math.min(safeHeight - 1, midRow - gapHalf - 1)),
        ],
        gapRows: [
          Math.max(0, Math.min(safeHeight - 1, midRow - gapHalf)),
          Math.max(0, Math.min(safeHeight - 1, midRow + gapHalf)),
        ],
        lowerRows: [
          Math.max(0, Math.min(safeHeight - 1, midRow + gapHalf + 1)),
          Math.max(0, Math.min(safeHeight - 1, lowerAnchorRow + toothHalf)),
        ],
        bgTopRows: [0, Math.max(0, upperAnchorRow - toothHalf - 8)],
        bgGapRows: [
          Math.max(0, Math.min(safeHeight - 1, midRow - gapHalf + 2)),
          Math.max(0, Math.min(safeHeight - 1, midRow + gapHalf - 2)),
        ],
        bgBottomRows: [
          Math.max(0, Math.min(safeHeight - 1, lowerAnchorRow + toothHalf + 8)),
          Math.max(0, safeHeight - 1),
        ],
      };
    };
    const radiographBandRows = isRadiographDisplayDomain ? deriveRadiographBandRows() : null;
    const toothBandStartRow = radiographBandRows
      ? Math.min(radiographBandRows.upperRows[0], radiographBandRows.lowerRows[0])
      : Math.min(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const toothBandEndRow = radiographBandRows
      ? Math.max(radiographBandRows.upperRows[1], radiographBandRows.lowerRows[1])
      : Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
    const lowerBandStartRow = radiographBandRows
      ? radiographBandRows.bgBottomRows[0]
      : Math.min(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const lowerBandEndRow = radiographBandRows
      ? radiographBandRows.bgBottomRows[1]
      : Math.max(rowFromNormalizedOffset(0.65), rowFromNormalizedOffset(1.15));
    const detailBandStartRow = Math.max(0, Math.floor(safeHeight * 0.12));
    const detailBandEndRow = Math.min(safeHeight - 1, Math.ceil(safeHeight * 0.72));
    const toothBandBrightThreshold = isRadiographDisplayDomain ? min + (max - min) * 0.82 : 1200;
    const lowerBandBrightThreshold = isRadiographDisplayDomain ? min + (max - min) * 0.18 : -180;
    const toothBandRowRetainedWeightSums = retainedSampleMaskMap ? new Float32Array(safeHeight) : null;
    const toothBandRowRetainedWeightCounts = retainedSampleMaskMap
      ? new Float32Array(safeHeight)
      : null;
    const toothBandColumnRetainedWeightSums = retainedSampleMaskMap
      ? new Float32Array(safeWidth)
      : null;
    const toothBandColumnRetainedWeightCounts = retainedSampleMaskMap
      ? new Float32Array(safeWidth)
      : null;
    const canSampleBackgroundTone = !!toneResponseMap && !!supportConfidenceMap;
    const backgroundSupportConfidenceMax = 0.03;
    const backgroundLowerPenaltyMax = 0.05;
    const backgroundParticipatingSampleCountMin = 0.5;
    const backgroundHuMax = isRadiographDisplayDomain ? Number.POSITIVE_INFINITY : -300;
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
          if (value > lowerBandBrightThreshold) {
            lowerBandBrightCount++;
          }
        }

        if (row >= toothBandStartRow && row <= toothBandEndRow) {
          toothBandSamples.push(value);
          toothBandCount++;
          toothBandSum += value;
          if (value > toothBandBrightThreshold) {
            toothBandBrightCount++;
          }
          const toneResponse = readOptionalProbeMapValue(toneResponseMap, index);
          if (toneResponse !== undefined) {
            const clampedToneResponse = Math.max(0, Math.min(1, toneResponse));
            toothBandToneResponseSamples.push(clampedToneResponse);
            const preToneAccumulation =
              readOptionalProbeMapValue(preToneAccumulationMap, index) ?? undefined;
            const retainedWeight = readOptionalProbeMapValue(retainedSampleMaskMap, index) ?? undefined;
            const middleBandLeak = readOptionalProbeMapValue(middleBandLeakMap, index) ?? undefined;
            if (preToneAccumulation !== undefined) {
              toothBandPreToneAccumulationSamples.push(Math.max(0, preToneAccumulation));
            }
            if (retainedWeight !== undefined) {
              const clampedRetainedWeight = Math.max(0, Math.min(1, retainedWeight));
              toothBandRetainedWeightSamples.push(clampedRetainedWeight);
              if (
                toothBandRowRetainedWeightSums &&
                toothBandRowRetainedWeightCounts &&
                toothBandColumnRetainedWeightSums &&
                toothBandColumnRetainedWeightCounts
              ) {
                toothBandRowRetainedWeightSums[row] += clampedRetainedWeight;
                toothBandRowRetainedWeightCounts[row] += 1;
                toothBandColumnRetainedWeightSums[col] += clampedRetainedWeight;
                toothBandColumnRetainedWeightCounts[col] += 1;
              }
            }
            if (middleBandLeak !== undefined) {
              toothBandLeakSamples.push(Math.max(0, Math.min(1, middleBandLeak)));
            }
            if (
              preToneAccumulation !== undefined &&
              retainedWeight !== undefined &&
              middleBandLeak !== undefined
            ) {
              toothBandHoleSignals.push({
                preToneAccumulation: Math.max(0, preToneAccumulation),
                retainedWeight: Math.max(0, Math.min(1, retainedWeight)),
                toneResponse: clampedToneResponse,
                middleBandLeak: Math.max(0, Math.min(1, middleBandLeak)),
              });
            }
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

        const radiographBackgroundBandName: FloatBufferBackgroundBandName | null = radiographBandRows
          ? row >= radiographBandRows.bgTopRows[0] && row <= radiographBandRows.bgTopRows[1]
            ? 'top'
            : row >= radiographBandRows.bgBottomRows[0] && row <= radiographBandRows.bgBottomRows[1]
            ? 'bottom'
            : row >= radiographBandRows.bgGapRows[0] && row <= radiographBandRows.bgGapRows[1]
            ? 'middle'
            : null
          : null;
        if (isRadiographDisplayDomain && toneResponseMap && radiographBackgroundBandName) {
          const toneResponse = readOptionalProbeMapValue(toneResponseMap, index);
          if (toneResponse !== undefined) {
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
            const backgroundBandBucket = backgroundBandBuckets[radiographBackgroundBandName];
            backgroundBandBucket.samples.push(clampedTone);
            if (clampedTone > backgroundToneOutlierThreshold05) {
              backgroundBandBucket.outlierCount05++;
            }
            if (clampedTone > backgroundToneOutlierThreshold10) {
              backgroundBandBucket.outlierCount10++;
            }
            if (clampedTone > backgroundBandBucket.toneMax) {
              backgroundBandBucket.toneMax = clampedTone;
            }
          }
        } else if (canSampleBackgroundTone && value <= backgroundHuMax) {
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
              const backgroundBandName: FloatBufferBackgroundBandName =
                row < toothBandStartRow ? 'top' : row > toothBandEndRow ? 'bottom' : 'middle';
              const backgroundBandBucket = backgroundBandBuckets[backgroundBandName];
              backgroundBandBucket.samples.push(clampedTone);
              if (clampedTone > backgroundToneOutlierThreshold05) {
                backgroundBandBucket.outlierCount05++;
              }
              if (clampedTone > backgroundToneOutlierThreshold10) {
                backgroundBandBucket.outlierCount10++;
              }
              if (clampedTone > backgroundBandBucket.toneMax) {
                backgroundBandBucket.toneMax = clampedTone;
              }
            }
          }
        }
      }
    }

    if (
      toothBandRowRetainedWeightSums &&
      toothBandRowRetainedWeightCounts &&
      toothBandColumnRetainedWeightSums &&
      toothBandColumnRetainedWeightCounts
    ) {
      toothBandRetainedWeightRowValues = Array.from(
        toothBandRowRetainedWeightSums,
        (sum, row) =>
          toothBandRowRetainedWeightCounts[row] > 0
            ? sum / Math.max(toothBandRowRetainedWeightCounts[row], 1)
            : Number.NaN
      ).filter(value => Number.isFinite(value));
      toothBandRetainedWeightColumnValues = Array.from(
        toothBandColumnRetainedWeightSums,
        (sum, col) =>
          toothBandColumnRetainedWeightCounts[col] > 0
            ? sum / Math.max(toothBandColumnRetainedWeightCounts[col], 1)
            : Number.NaN
      ).filter(value => Number.isFinite(value));
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
  if (toothBandPreToneAccumulationSamples.length) {
    toothBandPreToneAccumulationSamples.sort((a, b) => a - b);
  }
  if (toothBandRetainedWeightSamples.length) {
    toothBandRetainedWeightSamples.sort((a, b) => a - b);
  }
  if (toothBandToneResponseSamples.length) {
    toothBandToneResponseSamples.sort((a, b) => a - b);
  }
  if (toothBandLeakSamples.length) {
    toothBandLeakSamples.sort((a, b) => a - b);
  }
  if (toothBandRetainedWeightRowValues.length) {
    toothBandRetainedWeightRowValues.sort((a, b) => a - b);
  }
  if (toothBandRetainedWeightColumnValues.length) {
    toothBandRetainedWeightColumnValues.sort((a, b) => a - b);
  }
  const backgroundBandOrder: FloatBufferBackgroundBandName[] = ['top', 'middle', 'bottom'];
  for (const bandName of backgroundBandOrder) {
    backgroundBandBuckets[bandName].samples.sort((a, b) => a - b);
  }
  const sampledCount = samples.length;
  const buildBackgroundBandSummary = (
    bandName: FloatBufferBackgroundBandName
  ): FloatBufferBackgroundBandSummary => {
    const bandBucket = backgroundBandBuckets[bandName];
    return {
      sampleCount: bandBucket.samples.length,
      backgroundToneP95: bandBucket.samples.length
        ? percentileFromSorted(bandBucket.samples, 0.95)
        : 0,
      backgroundToneP99: bandBucket.samples.length
        ? percentileFromSorted(bandBucket.samples, 0.99)
        : 0,
      backgroundToneMax: bandBucket.toneMax,
      backgroundOutlierFraction05: bandBucket.samples.length
        ? bandBucket.outlierCount05 / bandBucket.samples.length
        : 0,
      backgroundOutlierFraction10: bandBucket.samples.length
        ? bandBucket.outlierCount10 / bandBucket.samples.length
        : 0,
      backgroundOutlierContribution05:
        backgroundToneOutlierCount05 > 0
          ? bandBucket.outlierCount05 / backgroundToneOutlierCount05
          : 0,
      backgroundOutlierContribution10:
        backgroundToneOutlierCount10 > 0
          ? bandBucket.outlierCount10 / backgroundToneOutlierCount10
          : 0,
    };
  };
  const determineDominantBackgroundBand = (
    outlierKey: 'outlierCount05' | 'outlierCount10'
  ): FloatBufferBackgroundBandDominantLabel => {
    const rankedBands = backgroundBandOrder
      .map(bandName => ({
        bandName,
        count: backgroundBandBuckets[bandName][outlierKey],
      }))
      .sort((a, b) => b.count - a.count);
    if (!rankedBands.length || rankedBands[0].count <= 0) {
      return 'none';
    }
    if (rankedBands[1]?.count && rankedBands[0].count <= rankedBands[1].count * 1.15) {
      return 'mixed';
    }
    return rankedBands[0].bandName;
  };
  const backgroundBands = {
    top: buildBackgroundBandSummary('top'),
    middle: buildBackgroundBandSummary('middle'),
    bottom: buildBackgroundBandSummary('bottom'),
    dominantOutlierBand05: determineDominantBackgroundBand('outlierCount05'),
    dominantOutlierBand10: determineDominantBackgroundBand('outlierCount10'),
  };
  const toothBandBlackClipFraction = toothBandToneResponseSamples.length
    ? toothBandToneResponseSamples.filter(
        toneResponse => toneResponse <= CPR_TOOTH_BAND_BLACK_CLIP_THRESHOLD
      ).length / toothBandToneResponseSamples.length
    : undefined;
  let toothBandHoleFraction: number | undefined;
  let toothBandHolePreToneThreshold: number | undefined;
  if (toothBandHoleSignals.length && toothBandPreToneAccumulationSamples.length) {
    toothBandHolePreToneThreshold = Math.max(
      0.012,
      percentileFromSorted(toothBandPreToneAccumulationSamples, 0.15) * 1.2
    );
    let toothBandHoleCount = 0;
    for (const signal of toothBandHoleSignals) {
      const blackClipped = signal.toneResponse <= CPR_TOOTH_BAND_BLACK_CLIP_THRESHOLD;
      const lowRetained =
        signal.retainedWeight <= CPR_TOOTH_BAND_HOLE_RETAINED_WEIGHT_MAX;
      const lowPreTone = signal.preToneAccumulation <= toothBandHolePreToneThreshold;
      const leakMarked = signal.middleBandLeak >= CPR_TOOTH_BAND_HOLE_LEAK_MIN;
      if (blackClipped && lowRetained && (lowPreTone || leakMarked)) {
        toothBandHoleCount++;
      }
    }
    toothBandHoleFraction = toothBandHoleCount / toothBandHoleSignals.length;
  }

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
    toothBandHoleFraction,
    toothBandBlackClipFraction,
    toothBandHolePreToneThreshold,
    toothBandRetainedWeightP10: toothBandRetainedWeightSamples.length
      ? percentileFromSorted(toothBandRetainedWeightSamples, 0.1)
      : undefined,
    toothBandRetainedWeightP50: toothBandRetainedWeightSamples.length
      ? percentileFromSorted(toothBandRetainedWeightSamples, 0.5)
      : undefined,
    toothBandRetainedWeightRowP10: toothBandRetainedWeightRowValues.length
      ? percentileFromSorted(toothBandRetainedWeightRowValues, 0.1)
      : undefined,
    toothBandRetainedWeightRowP50: toothBandRetainedWeightRowValues.length
      ? percentileFromSorted(toothBandRetainedWeightRowValues, 0.5)
      : undefined,
    toothBandRetainedWeightRowP90: toothBandRetainedWeightRowValues.length
      ? percentileFromSorted(toothBandRetainedWeightRowValues, 0.9)
      : undefined,
    toothBandRetainedWeightColumnP10: toothBandRetainedWeightColumnValues.length
      ? percentileFromSorted(toothBandRetainedWeightColumnValues, 0.1)
      : undefined,
    toothBandRetainedWeightColumnP50: toothBandRetainedWeightColumnValues.length
      ? percentileFromSorted(toothBandRetainedWeightColumnValues, 0.5)
      : undefined,
    toothBandRetainedWeightColumnP90: toothBandRetainedWeightColumnValues.length
      ? percentileFromSorted(toothBandRetainedWeightColumnValues, 0.9)
      : undefined,
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
    backgroundBands,
  };
}

function buildToothBandStageDiagnostics(params: {
  buffer: Float32Array | Uint16Array;
  width?: number;
  height?: number;
  debugMaps?: PanoImagePayload['debugMaps'];
  analysisCenterRow?: number;
  summary?: FloatBufferDebugSummary | null;
}): ToothBandStageDiagnostics | null {
  const { buffer, width, height, debugMaps, analysisCenterRow, summary } = params;
  if (
    !buffer?.length ||
    !width ||
    !height ||
    width <= 1 ||
    height <= 1 ||
    width * height > buffer.length ||
    !debugMaps
  ) {
    return null;
  }

  const admissionAccumulationMap = debugMaps.admissionAccumulationMap;
  const toneSuppressedAccumulationMap =
    debugMaps.toneSuppressedAccumulationMap ?? debugMaps.preToneAccumulationMap;
  const retainedSampleMaskMap = debugMaps.retainedSampleMaskMap;
  const toneStageSuppressionMap = debugMaps.toneStageSuppressionMap;
  const invalidSupportBlackoutMap = debugMaps.invalidSupportBlackoutMap;
  const blackClipMap = debugMaps.blackClipMap;
  const middleBandLeakMap = debugMaps.middleBandLeakMap;
  const admissionOnlyHuMap = debugMaps.admissionOnlyHuMap;
  const toneBypassHuMap = debugMaps.toneBypassHuMap;
  if (
    !admissionAccumulationMap ||
    !toneSuppressedAccumulationMap ||
    !retainedSampleMaskMap ||
    !toneStageSuppressionMap ||
    !invalidSupportBlackoutMap ||
    !blackClipMap ||
    !middleBandLeakMap ||
    !admissionOnlyHuMap ||
    !toneBypassHuMap
  ) {
    return null;
  }

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
  const toothBandStartRow = Math.min(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
  const toothBandEndRow = Math.max(rowFromNormalizedOffset(-0.35), rowFromNormalizedOffset(0.55));
  const admissionAccumulationSamples: number[] = [];
  const toneSuppressedAccumulationSamples: number[] = [];
  const toneStageSuppressionSamples: number[] = [];
  const invalidSupportBlackoutSamples: number[] = [];
  const blackClipSamples: number[] = [];
  const middleBandLeakSamples: number[] = [];
  const admissionOnlyHuSamples: number[] = [];
  const toneBypassHuSamples: number[] = [];
  const finalPostToneHuSamples: number[] = [];

  for (let row = toothBandStartRow; row <= toothBandEndRow; row += rowStep) {
    for (let col = 0; col < safeWidth; col += colStep) {
      const index = row * safeWidth + col;
      const finalValue = Number(buffer[index]);
      if (Number.isFinite(finalValue)) {
        finalPostToneHuSamples.push(finalValue);
      }
      const admissionAccumulation = readOptionalProbeMapValue(admissionAccumulationMap, index);
      if (admissionAccumulation !== undefined) {
        admissionAccumulationSamples.push(Math.max(0, admissionAccumulation));
      }
      const toneSuppressedAccumulation = readOptionalProbeMapValue(
        toneSuppressedAccumulationMap,
        index
      );
      if (toneSuppressedAccumulation !== undefined) {
        toneSuppressedAccumulationSamples.push(Math.max(0, toneSuppressedAccumulation));
      }
      const toneStageSuppression = readOptionalProbeMapValue(toneStageSuppressionMap, index);
      if (toneStageSuppression !== undefined) {
        toneStageSuppressionSamples.push(Math.max(0, Math.min(1, toneStageSuppression)));
      }
      const invalidSupportBlackout = readOptionalProbeMapValue(invalidSupportBlackoutMap, index);
      if (invalidSupportBlackout !== undefined) {
        invalidSupportBlackoutSamples.push(Math.max(0, Math.min(1, invalidSupportBlackout)));
      }
      const blackClip = readOptionalProbeMapValue(blackClipMap, index);
      if (blackClip !== undefined) {
        blackClipSamples.push(Math.max(0, Math.min(1, blackClip)));
      }
      const middleBandLeak = readOptionalProbeMapValue(middleBandLeakMap, index);
      if (middleBandLeak !== undefined) {
        middleBandLeakSamples.push(Math.max(0, Math.min(1, middleBandLeak)));
      }
      const admissionOnlyHu = readOptionalProbeMapValue(admissionOnlyHuMap, index);
      if (admissionOnlyHu !== undefined) {
        admissionOnlyHuSamples.push(admissionOnlyHu);
      }
      const toneBypassHu = readOptionalProbeMapValue(toneBypassHuMap, index);
      if (toneBypassHu !== undefined) {
        toneBypassHuSamples.push(toneBypassHu);
      }
    }
  }

  const sortNumeric = (values: number[]): number[] => values.sort((a, b) => a - b);
  const percentileOrNull = (values: number[], percentile: number): number | null =>
    values.length ? percentileFromSorted(sortNumeric(values.slice()), percentile) : null;
  const stageEvidence: string[] = [];
  const retainedWeightP10 = summary?.toothBandRetainedWeightP10 ?? null;
  const retainedWeightP50 = summary?.toothBandRetainedWeightP50 ?? null;
  const retainedWeightRowP10 = summary?.toothBandRetainedWeightRowP10 ?? null;
  const retainedWeightRowP50 = summary?.toothBandRetainedWeightRowP50 ?? null;
  const retainedWeightRowP90 = summary?.toothBandRetainedWeightRowP90 ?? null;
  const retainedWeightColumnP10 = summary?.toothBandRetainedWeightColumnP10 ?? null;
  const retainedWeightColumnP50 = summary?.toothBandRetainedWeightColumnP50 ?? null;
  const retainedWeightColumnP90 = summary?.toothBandRetainedWeightColumnP90 ?? null;
  const toneStageSuppressionP50 = percentileOrNull(toneStageSuppressionSamples, 0.5);
  const toneStageSuppressionP90 = percentileOrNull(toneStageSuppressionSamples, 0.9);
  const invalidSupportBlackoutP50 = percentileOrNull(invalidSupportBlackoutSamples, 0.5);
  const invalidSupportBlackoutP90 = percentileOrNull(invalidSupportBlackoutSamples, 0.9);
  const blackClipFraction = blackClipSamples.length
    ? blackClipSamples.filter(value => value > 0.5).length / blackClipSamples.length
    : null;
  const admissionAccumulationP50 = percentileOrNull(admissionAccumulationSamples, 0.5);
  const toneSuppressedAccumulationP50 = percentileOrNull(toneSuppressedAccumulationSamples, 0.5);
  const admissionOnlyHuP50 = percentileOrNull(admissionOnlyHuSamples, 0.5);
  const toneBypassHuP50 = percentileOrNull(toneBypassHuSamples, 0.5);
  const finalPostToneHuP50 = percentileOrNull(finalPostToneHuSamples, 0.5);
  let stageHint: ToothBandStageDiagnostics['stageHint'] = 'normalization-or-mixed';

  if ((retainedWeightP10 ?? 1) <= 0.02 || (retainedWeightRowP10 ?? 1) <= 0.08) {
    stageHint = 'support-admission-collapse';
    stageEvidence.push('tooth-band retained weight collapses before tone');
  }
  if (
    (toneStageSuppressionP90 ?? 0) >= 0.4 ||
    (invalidSupportBlackoutP90 ?? 0) >= 0.4 ||
    (blackClipFraction ?? 0) >= 0.25
  ) {
    stageHint =
      stageHint === 'support-admission-collapse'
        ? 'normalization-or-mixed'
        : 'tone-blackout-destruction';
    stageEvidence.push('tone/blackout stage still removes a large fraction of admitted tooth-band signal');
  }
  if (
    (admissionOnlyHuP50 ?? Number.NEGATIVE_INFINITY) <=
      (toneBypassHuP50 ?? Number.POSITIVE_INFINITY) + 80 &&
    (toneBypassHuP50 ?? Number.NEGATIVE_INFINITY) <=
      (finalPostToneHuP50 ?? Number.POSITIVE_INFINITY) + 80
  ) {
    stageEvidence.push('normalization may still dominate because admission-only and tone-bypass remain similarly collapsed');
  }
  if (!stageEvidence.length) {
    stageEvidence.push('no single stage dominates from static thresholds');
  }

  return {
    sampleCount: finalPostToneHuSamples.length,
    retainedWeightP10,
    retainedWeightP50,
    retainedWeightRowP10,
    retainedWeightRowP50,
    retainedWeightRowP90,
    retainedWeightColumnP10,
    retainedWeightColumnP50,
    retainedWeightColumnP90,
    admissionAccumulationP10: percentileOrNull(admissionAccumulationSamples, 0.1),
    admissionAccumulationP50,
    admissionAccumulationP90: percentileOrNull(admissionAccumulationSamples, 0.9),
    toneSuppressedAccumulationP10: percentileOrNull(toneSuppressedAccumulationSamples, 0.1),
    toneSuppressedAccumulationP50,
    toneSuppressedAccumulationP90: percentileOrNull(toneSuppressedAccumulationSamples, 0.9),
    toneStageSuppressionP50,
    toneStageSuppressionP90,
    invalidSupportBlackoutP50,
    invalidSupportBlackoutP90,
    blackClipFraction,
    middleBandLeakP50: percentileOrNull(middleBandLeakSamples, 0.5),
    middleBandLeakP90: percentileOrNull(middleBandLeakSamples, 0.9),
    admissionOnlyHuP10: percentileOrNull(admissionOnlyHuSamples, 0.1),
    admissionOnlyHuP50,
    admissionOnlyHuP90: percentileOrNull(admissionOnlyHuSamples, 0.9),
    toneBypassHuP10: percentileOrNull(toneBypassHuSamples, 0.1),
    toneBypassHuP50,
    toneBypassHuP90: percentileOrNull(toneBypassHuSamples, 0.9),
    finalPostToneHuP10: percentileOrNull(finalPostToneHuSamples, 0.1),
    finalPostToneHuP50,
    finalPostToneHuP90: percentileOrNull(finalPostToneHuSamples, 0.9),
    stageHint,
    stageEvidence,
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
  const looksLikeHU =
    !isNativeDisplayPanoRenderMode(renderSupportMode) && isHuLikeRange(lower, upper);

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
      Math.min(dataRangeWidth * 3.0, 8000)
    );
    const clampedLower = Math.min(CPR_DUAL_ARCH_PROJECTION_BLACK_POINT_HU, adaptiveLower);
    const widthFromClampedLower = Math.max(1, adaptiveUpper - clampedLower);
    const clampedWindowWidth = Math.min(
      Math.max(widthFromClampedLower, minDualArchWindowWidth),
      maxDualArchWindowWidth
    );
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

interface CPRWorkerBootstrapReadyMessage {
  type: 'BOOTSTRAP_READY';
  requestId: string;
}

interface CPRWorkerErrorMessage {
  type: 'ERROR';
  requestId: string;
  message: string;
  stage?: string;
}

interface CPRWorkerLifecycleMessage {
  type: 'WORKER_LIFECYCLE';
  scope: 'entry-wrapper' | 'bootstrap' | 'implementation';
  stage: string;
  detail?: Record<string, unknown>;
}

type CPRWorkerResponseMessage =
  | CPRWorkerBootstrapReadyMessage
  | CPRWorkerInitSuccessMessage
  | CPRWorkerSuccessMessage
  | CPRWorkerDisposeSuccessMessage
  | CPRWorkerErrorMessage
  | CPRWorkerLifecycleMessage;

type CPRWorkerExpectedResponseType = Exclude<
  CPRWorkerResponseMessage['type'],
  'ERROR' | 'WORKER_LIFECYCLE'
>;

type PendingCPRWorkerRequest = {
  expectedTypes: Set<CPRWorkerExpectedResponseType>;
  requestType: string;
  resolve: (value: CPRWorkerResponseMessage) => void;
  reject: (error: Error) => void;
  timeoutId?: number;
  retryIntervalId?: number;
};

interface CPRWorkerSession {
  worker: Worker;
  volumeKey: string;
  sessionKey: string;
  workerEntryUrl: string;
  workerRuntimeUrl: string;
  workerCreationMode: 'prewarm' | 'lazy-init';
  pendingRequests: Map<string, PendingCPRWorkerRequest>;
  workerEntryReadyPromise: Promise<void>;
  workerEntryReady: boolean;
  workerEntryReadyError?: Error;
  bootstrapPromise: Promise<void>;
  bootstrapReady: boolean;
  bootstrapProbeFailed: boolean;
  bootstrapProbeError?: Error;
  initPromise?: Promise<void>;
  initCompleted?: boolean;
  lastTimedOutRequestType?: string;
  fatalWorkerError?: Error;
  workerLifecycleEventCount: number;
  workerLastLifecycleStage?: string;
  isTerminating: boolean;
  terminatePromise?: Promise<void>;
  cleanupListeners: () => void;
  revokeWorkerRuntimeUrl?: () => void;
  rejectPendingRequests: (
    error: Error,
    predicate?: (requestId: string, request: PendingCPRWorkerRequest) => boolean
  ) => void;
}

type BufferedCPRWorkerEvent =
  | {
      kind: 'message';
      event: MessageEvent<CPRWorkerResponseMessage>;
    }
  | {
      kind: 'error';
      event: ErrorEvent;
    }
  | {
      kind: 'messageerror';
      event: MessageEvent;
    };

let activeCPRWorkerSession: CPRWorkerSession | null = null;
let cprWorkerRequestCounter = 0;
const CPR_WORKER_ENTRY_SPECIFIER = './cprWorker.ts';

function isCPRWorkerInitializationRequestType(requestType: string): boolean {
  return requestType === 'BOOTSTRAP_CHECK' || requestType === 'INIT_VOLUME';
}

function buildCPRWorkerRequestError(params: {
  session: CPRWorkerSession;
  requestType: string;
  message: string;
  stage?: string | null;
}): Error {
  const baseMessage = params.message.trim();
  const fatalMessage = params.session.fatalWorkerError?.message?.trim() ?? null;
  if (!isCPRWorkerInitializationRequestType(params.requestType)) {
    return new Error(baseMessage);
  }

  const initStage =
    params.requestType === 'BOOTSTRAP_CHECK' ? 'bootstrap' : 'volume initialization';
  const stageSuffix = params.stage ? ` Stage: ${params.stage}.` : '';
  const rootCauseSuffix =
    fatalMessage && fatalMessage !== baseMessage ? ` Root cause: ${fatalMessage}` : '';
  const lifecycleSuffix =
    params.session.workerLifecycleEventCount === 0
      ? ` Worker entry ${params.session.workerEntryUrl} emitted no lifecycle event before failure; this points to worker script load or entry execution failure.`
      : params.session.workerLastLifecycleStage
        ? ` Last worker lifecycle stage: ${params.session.workerLastLifecycleStage}.`
        : '';
  return new Error(
    `[CPR] CPR worker failed to initialize during ${initStage}.${stageSuffix} ${baseMessage}${rootCauseSuffix}${lifecycleSuffix}`.trim()
  );
}

function computeCPRWorkerInitTimeoutMs(
  scalarData: ArrayLike<number> & {
    byteLength?: number;
    BYTES_PER_ELEMENT?: number;
  }
): number {
  const estimatedByteLength = estimateCPRWorkerPayloadByteLength(scalarData);
  const megaBytes = estimatedByteLength / (1024 * 1024);

  // Fresh module-worker startup plus transferring a large volume payload is
  // materially slower than the old voxel-count heuristic implied.
  return Math.max(CPR_WORKER_INIT_TIMEOUT_MS, Math.min(60000, 10000 + megaBytes * 180));
}

function estimateCPRWorkerPayloadByteLength(
  scalarData: ArrayLike<number> & {
    byteLength?: number;
    BYTES_PER_ELEMENT?: number;
  }
): number {
  const safeLength =
    Number.isFinite((scalarData as { length?: unknown })?.length) &&
    Number((scalarData as { length?: number }).length) > 0
      ? Number((scalarData as { length: number }).length)
      : 0;
  const safeBytesPerElement =
    Number.isFinite((scalarData as { BYTES_PER_ELEMENT?: unknown })?.BYTES_PER_ELEMENT) &&
    Number((scalarData as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT) > 0
      ? Number((scalarData as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT)
      : 4;
  return Number.isFinite((scalarData as { byteLength?: unknown })?.byteLength) &&
    Number((scalarData as { byteLength?: number }).byteLength) > 0
    ? Number((scalarData as { byteLength: number }).byteLength)
    : safeLength * safeBytesPerElement;
}

function computeCPRWorkerRenderTimeoutMs(params: {
  scalarData: ArrayLike<number> & {
    byteLength?: number;
    BYTES_PER_ELEMENT?: number;
  };
  panoWidth: number;
  panoHeight: number;
  frameCount: number;
  slabSamples: number;
  renderBackend?: 'gpu' | 'cpu';
  reconstructionMode?: PanoReconstructionMode;
  debugRunId?: string;
}): number {
  const megaBytes = estimateCPRWorkerPayloadByteLength(params.scalarData) / (1024 * 1024);
  const pixelCount = Math.max(1, params.panoWidth * params.panoHeight);
  const megaSampleWork =
    (pixelCount * Math.max(1, Math.min(64, Math.floor(params.slabSamples)))) / 1_000_000;
  const normalizedFrameCount = Math.max(1, params.frameCount) / 500;
  const backendOverheadMs = params.renderBackend === 'gpu' ? 12000 : 6000;
  const debugOverheadMs = params.debugRunId ? 20000 : 0;
  const reconstructionOverheadMs =
    params.reconstructionMode && params.reconstructionMode !== 'legacy' ? 10000 : 0;

  // GPU debug renders spend time inside the worker after the request is posted:
  // readback, support/tone diagnostics, and debug sidecar packaging all happen
  // before SUCCESS can be emitted back to the main thread.
  return Math.max(
    CPR_WORKER_RENDER_TIMEOUT_MS,
    Math.min(
      120000,
      Math.round(
        12000 +
          megaBytes * 120 +
          megaSampleWork * 2500 +
          normalizedFrameCount * 2000 +
          backendOverheadMs +
          debugOverheadMs +
          reconstructionOverheadMs
      )
    )
  );
}

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

function resolveBundledCPRWorkerEntryUrl(): string {
  // Keep the worker specifier inline so webpack's worker transform can rewrite it
  // to a served bundle URL instead of leaving it as a raw source-file URL.
  return new URL('./cprWorker.ts', import.meta.url).toString();
}

function createInstrumentedCPRWorker(params: {
  mode: 'prewarm' | 'lazy-init';
  volumeKey: string;
}): {
  worker: Worker;
  workerEntryUrl: string;
  workerRuntimeUrl: string;
  revokeWorkerRuntimeUrl: () => void;
  workerEntryReadyPromise: Promise<void>;
  handoffBufferedEvents: (handlers: {
    onMessage: (event: MessageEvent<CPRWorkerResponseMessage>) => void;
    onError: (event: ErrorEvent) => void;
    onMessageError: (event: MessageEvent) => void;
  }) => void;
} {
  const workerEntryUrl = resolveBundledCPRWorkerEntryUrl();
  const workerRuntimeUrl = workerEntryUrl;
  const revokeWorkerRuntimeUrl = () => {};
  console.log(
    '[CPR-WORKER-CONSTRUCTOR-JSON]',
    JSON.stringify({
      stage: 'before-create',
      mode: params.mode,
      volumeKey: params.volumeKey,
      workerEntrySpecifier: CPR_WORKER_ENTRY_SPECIFIER,
      workerEntryUrl,
      workerRuntimeUrl,
      baseImportMetaUrl: import.meta.url,
      workerType: 'module',
      bundlerStrategy: 'webpack-inline-new-worker-url-direct',
    })
  );

  try {
    const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), {
      type: 'module',
    });
    const bufferedEvents: BufferedCPRWorkerEvent[] = [];
    let handoffHandlers:
      | {
          onMessage: (event: MessageEvent<CPRWorkerResponseMessage>) => void;
          onError: (event: ErrorEvent) => void;
          onMessageError: (event: MessageEvent) => void;
        }
      | null = null;
    let workerEntryReadyResolved = false;
    let workerEntryReadyRejected = false;
    let resolveWorkerEntryReady!: () => void;
    let rejectWorkerEntryReady!: (error: Error) => void;
    const workerEntryReadyPromise = new Promise<void>((resolve, reject) => {
      resolveWorkerEntryReady = resolve;
      rejectWorkerEntryReady = reject;
    });
    const markWorkerEntryReady = () => {
      if (workerEntryReadyResolved || workerEntryReadyRejected) {
        return;
      }
      workerEntryReadyResolved = true;
      resolveWorkerEntryReady();
    };
    const markWorkerEntryFailed = (error: Error) => {
      if (workerEntryReadyResolved || workerEntryReadyRejected) {
        return;
      }
      workerEntryReadyRejected = true;
      rejectWorkerEntryReady(error);
    };
    const bufferOrDispatchMessage = (event: MessageEvent<CPRWorkerResponseMessage>) => {
      const data = event.data;
      if (
        data &&
        typeof data === 'object' &&
        data.type === 'WORKER_LIFECYCLE' &&
        data.scope === 'bootstrap' &&
        data.stage === 'worker-bootstrap-script-loaded'
      ) {
        markWorkerEntryReady();
      }
      if (handoffHandlers) {
        handoffHandlers.onMessage(event);
        return;
      }
      bufferedEvents.push({
        kind: 'message',
        event,
      });
    };
    const bufferOrDispatchError = (event: ErrorEvent) => {
      markWorkerEntryFailed(
        new Error(
          [
            '[cprWorker] Early worker error before session listener attachment.',
            event.message ? `message=${event.message}` : null,
            event.filename ? `file=${event.filename}` : null,
            Number.isFinite(event.lineno) && event.lineno > 0 ? `line=${event.lineno}` : null,
            Number.isFinite(event.colno) && event.colno > 0 ? `col=${event.colno}` : null,
          ]
            .filter(Boolean)
            .join(' ')
        )
      );
      if (handoffHandlers) {
        handoffHandlers.onError(event);
        return;
      }
      bufferedEvents.push({
        kind: 'error',
        event,
      });
    };
    const bufferOrDispatchMessageError = (event: MessageEvent) => {
      markWorkerEntryFailed(
        new Error('[cprWorker] Early worker messageerror before session listener attachment.')
      );
      if (handoffHandlers) {
        handoffHandlers.onMessageError(event);
        return;
      }
      bufferedEvents.push({
        kind: 'messageerror',
        event,
      });
    };
    worker.addEventListener('message', bufferOrDispatchMessage as EventListener);
    worker.addEventListener('error', bufferOrDispatchError as EventListener);
    worker.addEventListener('messageerror', bufferOrDispatchMessageError as EventListener);
    const handoffBufferedEvents = (handlers: {
      onMessage: (event: MessageEvent<CPRWorkerResponseMessage>) => void;
      onError: (event: ErrorEvent) => void;
      onMessageError: (event: MessageEvent) => void;
    }) => {
      handoffHandlers = handlers;
      worker.removeEventListener('message', bufferOrDispatchMessage as EventListener);
      worker.removeEventListener('error', bufferOrDispatchError as EventListener);
      worker.removeEventListener('messageerror', bufferOrDispatchMessageError as EventListener);
      for (const bufferedEvent of bufferedEvents.splice(0)) {
        if (bufferedEvent.kind === 'message') {
          handlers.onMessage(bufferedEvent.event);
          continue;
        }
        if (bufferedEvent.kind === 'error') {
          handlers.onError(bufferedEvent.event);
          continue;
        }
        handlers.onMessageError(bufferedEvent.event);
      }
    };
    console.log(
      '[CPR-WORKER-CONSTRUCTOR-JSON]',
      JSON.stringify({
        stage: 'constructor-succeeded',
        mode: params.mode,
        volumeKey: params.volumeKey,
        workerEntrySpecifier: CPR_WORKER_ENTRY_SPECIFIER,
        workerEntryUrl,
        workerRuntimeUrl,
        workerType: 'module',
        bundlerStrategy: 'webpack-inline-new-worker-url-direct',
      })
    );
    return {
      worker,
      workerEntryUrl,
      workerRuntimeUrl,
      revokeWorkerRuntimeUrl,
      workerEntryReadyPromise,
      handoffBufferedEvents,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    revokeWorkerRuntimeUrl();
    console.error(
      '[CPR-WORKER-CONSTRUCTOR-JSON]',
      JSON.stringify({
        stage: 'constructor-failed',
        mode: params.mode,
        volumeKey: params.volumeKey,
        workerEntrySpecifier: CPR_WORKER_ENTRY_SPECIFIER,
        workerEntryUrl,
        workerRuntimeUrl,
        workerType: 'module',
        bundlerStrategy: 'webpack-inline-new-worker-url-direct',
        message,
      })
    );
    throw new Error(
      `[CPR] CPR worker failed to initialize. Worker constructor failed for runtime ${workerRuntimeUrl} (entry ${workerEntryUrl}). ${message}`
    );
  }
}

function beginCPRWorkerVolumeInit(params: {
  workerSession: CPRWorkerSession;
  initPayload: Record<string, unknown>;
  transferList?: Transferable[];
  timeoutMs: number;
}): Promise<void> {
  const { workerSession, initPayload, transferList, timeoutMs } = params;
  if (workerSession.initCompleted) {
    return Promise.resolve();
  }
  if (workerSession.initPromise) {
    return workerSession.initPromise;
  }

  const initPromise = postMessageToCPRWorker<CPRWorkerInitSuccessMessage>(workerSession, initPayload, {
    expectedTypes: ['INIT_SUCCESS'],
    transferList,
    timeoutMs,
  })
    .then(() => {
      workerSession.initCompleted = true;
      workerSession.bootstrapReady = true;
      workerSession.bootstrapProbeFailed = false;
      workerSession.bootstrapProbeError = undefined;
      console.log('[CPR-WORKER-INIT-ACK-JSON]', {
        sessionKey: workerSession.sessionKey,
        volumeKey: workerSession.volumeKey,
        workerEntryUrl: workerSession.workerEntryUrl,
        bootstrapReady: workerSession.bootstrapReady,
        bootstrapProbeFailed: workerSession.bootstrapProbeFailed,
        workerLifecycleEventCount: workerSession.workerLifecycleEventCount,
        workerLastLifecycleStage: workerSession.workerLastLifecycleStage ?? null,
      });
      if (workerSession.bootstrapProbeFailed) {
        console.warn('[CPR] Worker bootstrap probe failed, but direct INIT_VOLUME succeeded.', {
          sessionKey: workerSession.sessionKey,
          volumeKey: workerSession.volumeKey,
          error:
            workerSession.bootstrapProbeError instanceof Error
              ? workerSession.bootstrapProbeError.message
              : null,
        });
      }
    });

  workerSession.initPromise = initPromise
    .catch(error => {
      workerSession.initCompleted = false;
      console.error('[CPR-WORKER-INIT-FAILED-JSON]', {
        sessionKey: workerSession.sessionKey,
        volumeKey: workerSession.volumeKey,
        workerEntryUrl: workerSession.workerEntryUrl,
        error: error instanceof Error ? error.message : String(error),
        fatalWorkerError: workerSession.fatalWorkerError?.message ?? null,
        workerLifecycleEventCount: workerSession.workerLifecycleEventCount,
        workerLastLifecycleStage: workerSession.workerLastLifecycleStage ?? null,
      });
      throw error;
    })
    .finally(() => {
      workerSession.initPromise = undefined;
    });

  return workerSession.initPromise;
}

function createCPRWorkerSession(
  worker: Worker,
  volumeKey: string,
  sessionKey: string,
  workerEntryUrl: string,
  workerRuntimeUrl: string,
  workerCreationMode: 'prewarm' | 'lazy-init',
  workerEntryReadyPromise: Promise<void>,
  handoffBufferedEvents: (handlers: {
    onMessage: (event: MessageEvent<CPRWorkerResponseMessage>) => void;
    onError: (event: ErrorEvent) => void;
    onMessageError: (event: MessageEvent) => void;
  }) => void
): CPRWorkerSession {
  const session: CPRWorkerSession = {
    worker,
    volumeKey,
    sessionKey,
    workerEntryUrl,
    workerRuntimeUrl,
    workerCreationMode,
    pendingRequests: new Map(),
    workerEntryReadyPromise,
    workerEntryReady: false,
    workerEntryReadyError: undefined,
    bootstrapPromise: Promise.resolve(),
    bootstrapReady: false,
    bootstrapProbeFailed: false,
    bootstrapProbeError: undefined,
    initPromise: undefined,
    initCompleted: false,
    lastTimedOutRequestType: undefined,
    fatalWorkerError: undefined,
    workerLifecycleEventCount: 0,
    workerLastLifecycleStage: undefined,
    isTerminating: false,
    cleanupListeners: () => {},
    revokeWorkerRuntimeUrl: undefined,
    rejectPendingRequests: () => {},
  };

  session.workerEntryReadyPromise = session.workerEntryReadyPromise
    .then(() => {
      session.workerEntryReady = true;
      session.workerEntryReadyError = undefined;
    })
    .catch(error => {
      const readyError = error instanceof Error ? error : new Error(String(error));
      session.workerEntryReady = false;
      session.workerEntryReadyError = readyError;
      throw readyError;
    });

  const recordFatalWorkerError = (error: Error, source: string) => {
    session.fatalWorkerError = error;
    if (!session.isTerminating) {
      console.error('[CPR-WORKER-FATAL-ERROR-JSON]', {
        source,
        sessionKey,
        volumeKey,
        workerEntryUrl: session.workerEntryUrl,
        workerRuntimeUrl: session.workerRuntimeUrl,
        mode: session.workerCreationMode,
        message: error.message,
      });
    }
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
      if (request.retryIntervalId != null) {
        window.clearInterval(request.retryIntervalId);
      }
      request.reject(error);
    }
  };

  const rejectPendingRequestsForFatalWorkerError = (error: Error, stage: string) => {
    for (const [requestId, request] of Array.from(session.pendingRequests.entries())) {
      session.pendingRequests.delete(requestId);
      if (request.timeoutId != null) {
        window.clearTimeout(request.timeoutId);
      }
      if (request.retryIntervalId != null) {
        window.clearInterval(request.retryIntervalId);
      }
      request.reject(
        buildCPRWorkerRequestError({
          session,
          requestType: request.requestType,
          message: error.message,
          stage,
        })
      );
    }
  };

  void session.workerEntryReadyPromise.catch(error => {
    const readyError = error instanceof Error ? error : new Error(String(error));
    recordFatalWorkerError(readyError, 'worker-entry-ready-failed');
    rejectPendingRequestsForFatalWorkerError(readyError, 'worker-entry-ready-failed');
  });

  const handleMessage = (event: MessageEvent<CPRWorkerResponseMessage>) => {
    const data = event.data;
    if (!data || typeof data !== 'object') {
      return;
    }

    if (data.type === 'WORKER_LIFECYCLE') {
      session.workerLifecycleEventCount += 1;
      session.workerLastLifecycleStage = `${data.scope}:${data.stage}`;
      console.log(
        '[CPR-WORKER-LIFECYCLE-EVENT-JSON]',
        JSON.stringify({
          sessionKey,
          volumeKey,
          workerEntryUrl: session.workerEntryUrl,
          workerRuntimeUrl: session.workerRuntimeUrl,
          mode: session.workerCreationMode,
          scope: data.scope,
          stage: data.stage,
          detail: data.detail ?? null,
        })
      );
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
    if (pendingRequest.retryIntervalId != null) {
      window.clearInterval(pendingRequest.retryIntervalId);
    }

    if (data.type === 'ERROR') {
      console.error(
        '[CPR-WORKER-ERROR-RESPONSE-JSON]',
        JSON.stringify({
          sessionKey,
          volumeKey,
          workerEntryUrl: session.workerEntryUrl,
          workerRuntimeUrl: session.workerRuntimeUrl,
          mode: session.workerCreationMode,
          requestType: pendingRequest.requestType,
          requestId,
          stage: data.stage ?? null,
          message: data.message,
        })
      );
      pendingRequest.reject(
        buildCPRWorkerRequestError({
          session,
          requestType: pendingRequest.requestType,
          message: `[cprWorker] ${data.message}`,
          stage: data.stage ?? null,
        })
      );
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

    session.lastTimedOutRequestType = undefined;
    pendingRequest.resolve(data);
  };

  const handleError = (event: ErrorEvent) => {
    console.error(
      '[CPR-WORKER-ERROR-EVENT-JSON]',
      JSON.stringify({
        sessionKey,
        volumeKey,
        workerEntryUrl: session.workerEntryUrl,
        workerRuntimeUrl: session.workerRuntimeUrl,
        mode: session.workerCreationMode,
        type: event.type ?? 'error',
        message: event.message ?? null,
        filename: event.filename ?? null,
        lineno: Number.isFinite(event.lineno) ? event.lineno : null,
        colno: Number.isFinite(event.colno) ? event.colno : null,
        errorName: event.error instanceof Error ? event.error.name : null,
        errorMessage: event.error instanceof Error ? event.error.message : null,
        errorStack: event.error instanceof Error ? event.error.stack ?? null : null,
      })
    );
    const messageParts = [
      event.message || 'unknown worker error',
      event.filename ? `file=${event.filename}` : null,
      Number.isFinite(event.lineno) && event.lineno > 0 ? `line=${event.lineno}` : null,
      Number.isFinite(event.colno) && event.colno > 0 ? `col=${event.colno}` : null,
    ].filter(Boolean);
    const error = new Error(
      `[cprWorker] Uncaught worker error: ${messageParts.join(' ')}`
    );
    recordFatalWorkerError(error, 'worker-error-event');
    rejectPendingRequestsForFatalWorkerError(error, 'worker-error-event');
  };
  const handleMessageError = (event: MessageEvent) => {
    console.error(
      '[CPR-WORKER-MESSAGEERROR-EVENT-JSON]',
      JSON.stringify({
        sessionKey,
        volumeKey,
        workerEntryUrl: session.workerEntryUrl,
        workerRuntimeUrl: session.workerRuntimeUrl,
        mode: session.workerCreationMode,
        type: event.type ?? 'messageerror',
        origin: event.origin ?? null,
        lastEventId: event.lastEventId ?? null,
        dataType: event.data == null ? null : typeof event.data,
      })
    );
    const error = new Error('[cprWorker] Worker message deserialization failed.');
    recordFatalWorkerError(error, 'worker-messageerror-event');
    rejectPendingRequestsForFatalWorkerError(error, 'worker-messageerror-event');
  };

  worker.addEventListener('message', handleMessage as EventListener);
  worker.addEventListener('error', handleError as EventListener);
  worker.addEventListener('messageerror', handleMessageError as EventListener);
  handoffBufferedEvents({
    onMessage: handleMessage,
    onError: handleError,
    onMessageError: handleMessageError,
  });

  session.cleanupListeners = () => {
    worker.removeEventListener('message', handleMessage as EventListener);
    worker.removeEventListener('error', handleError as EventListener);
    worker.removeEventListener('messageerror', handleMessageError as EventListener);
  };
  session.rejectPendingRequests = rejectPendingRequests;
  session.bootstrapPromise = Promise.resolve();

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
    const pendingRequestTypes = Array.from(session.pendingRequests.values()).map(
      request => request.requestType
    );
    const shouldAttemptGracefulDispose =
      pendingRequestTypes.length === 0 &&
      !session.lastTimedOutRequestType &&
      (session.bootstrapReady || session.initCompleted === true);

    session.rejectPendingRequests(
      new Error('[cprWorker] Worker session terminated during teardown.')
    );

    try {
      if (shouldAttemptGracefulDispose) {
        await postMessageToCPRWorker<CPRWorkerDisposeSuccessMessage>(
          session,
          {
            type: 'DISPOSE' as const,
          },
          {
            expectedTypes: ['DISPOSE_SUCCESS'],
            timeoutMs: CPR_WORKER_DISPOSE_TIMEOUT_MS,
          }
        );
      }
    } catch (disposeError) {
      console.warn('[CPR] Failed to dispose CPR worker before termination.', disposeError);
    } finally {
      session.rejectPendingRequests(
        new Error('[cprWorker] Worker session terminated before pending request completion.')
      );
      session.cleanupListeners();
      session.worker.terminate();
      session.revokeWorkerRuntimeUrl?.();
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
    if (activeCPRWorkerSession.initCompleted) {
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

    if (activeCPRWorkerSession.initPromise) {
      await activeCPRWorkerSession.initPromise;
      console.log(
        '[CPR-WORKER-PREWARM-JSON]',
        JSON.stringify({
          runId: debugRunId ?? null,
          status: 'awaited-existing-init',
          volumeId: volume.volumeId ?? null,
          requestedPanoHeight: requestedPanoHeight ?? null,
          applyModalityLut,
          allowStoredValueNormalization,
        })
      );
      return;
    }
  }

  await terminateCPRWorkerSession();
  const {
    worker,
    workerEntryUrl,
    workerRuntimeUrl,
    revokeWorkerRuntimeUrl,
    workerEntryReadyPromise,
    handoffBufferedEvents,
  } =
    createInstrumentedCPRWorker({
    mode: 'prewarm',
    volumeKey,
  });
  const sessionKey = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const workerSession = createCPRWorkerSession(
    worker,
    volumeKey,
    sessionKey,
    workerEntryUrl,
    workerRuntimeUrl,
    'prewarm',
    workerEntryReadyPromise,
    handoffBufferedEvents
  );
  workerSession.revokeWorkerRuntimeUrl = revokeWorkerRuntimeUrl;
  activeCPRWorkerSession = workerSession;
  console.log('[CPR-WORKER-SESSION-CREATED-JSON]', {
    sessionKey,
    volumeKey,
    mode: 'prewarm',
    workerEntry: CPR_WORKER_ENTRY_SPECIFIER,
    workerEntryUrl,
    workerRuntimeUrl,
  });
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
  const initTimeoutMs = computeCPRWorkerInitTimeoutMs(initScalarData);

  try {
    await beginCPRWorkerVolumeInit({
      workerSession,
      initPayload,
      transferList,
      timeoutMs: initTimeoutMs,
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
    retryIntervalMs?: number;
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
  if (session.fatalWorkerError && requestType !== 'DISPOSE') {
    return Promise.reject(
      buildCPRWorkerRequestError({
        session,
        requestType,
        message: session.fatalWorkerError.message,
        stage: 'worker-fatal-error-before-request',
      })
    );
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
        if (liveRequest.retryIntervalId != null) {
          window.clearInterval(liveRequest.retryIntervalId);
        }
        session.lastTimedOutRequestType = liveRequest.requestType;
        const timeoutMessage =
          session.fatalWorkerError?.message ??
          `[cprWorker] Timed out waiting for ${Array.from(expectedTypes).join('/')} from ${requestType}.`;
        liveRequest.reject(
          buildCPRWorkerRequestError({
            session,
            requestType: liveRequest.requestType,
            message: timeoutMessage,
            stage: session.fatalWorkerError ? 'worker-fatal-error-before-ack' : 'timeout',
          })
        );
      }, options.timeoutMs);
    }

    session.pendingRequests.set(requestId, pendingRequest);
    if (
      options.retryIntervalMs &&
      options.retryIntervalMs > 0 &&
      (!options.transferList || options.transferList.length === 0)
    ) {
      pendingRequest.retryIntervalId = window.setInterval(() => {
        const liveRequest = session.pendingRequests.get(requestId);
        if (!liveRequest || session.isTerminating) {
          if (liveRequest?.retryIntervalId != null) {
            window.clearInterval(liveRequest.retryIntervalId);
          }
          return;
        }

        try {
          console.log(
            '[CPR-WORKER-POSTMESSAGE-JSON]',
            JSON.stringify({
              stage: 'retry-post',
              requestType,
              requestId,
              sessionKey: session.sessionKey,
              volumeKey: session.volumeKey,
              workerEntryUrl: session.workerEntryUrl,
              mode: session.workerCreationMode,
              transferListLength: options.transferList?.length ?? 0,
            })
          );
          session.worker.postMessage(message);
        } catch (retryError) {
          session.pendingRequests.delete(requestId);
          if (liveRequest.timeoutId != null) {
            window.clearTimeout(liveRequest.timeoutId);
          }
          if (liveRequest.retryIntervalId != null) {
            window.clearInterval(liveRequest.retryIntervalId);
          }
          liveRequest.reject(
            retryError instanceof Error ? retryError : new Error(String(retryError))
          );
        }
      }, options.retryIntervalMs);
    }

    try {
      console.log(
        '[CPR-WORKER-POSTMESSAGE-JSON]',
        JSON.stringify({
          stage: 'initial-post',
          requestType,
          requestId,
          sessionKey: session.sessionKey,
          volumeKey: session.volumeKey,
          workerEntryUrl: session.workerEntryUrl,
          mode: session.workerCreationMode,
          transferListLength: options.transferList?.length ?? 0,
        })
      );
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
      if (pendingRequest.retryIntervalId != null) {
        window.clearInterval(pendingRequest.retryIntervalId);
      }
      reject(
        buildCPRWorkerRequestError({
          session,
          requestType,
          message: postError instanceof Error ? postError.message : String(postError),
          stage: 'postMessage-failed',
        })
      );
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
  debugScalarSamplingMode?:
    | 'current'
    | 'lut-only'
    | 'no-stored-value-normalization'
    | 'raw-stored-values-debug';
  verticalHalfMm?: number;
  verticalCenterOffsetMm?: number;
  rigidVerticalSliceMode?: boolean;
  debugRunId?: string;
  attemptLabel?: string;
  reconstructionMode?: PanoReconstructionMode;
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
      debugScalarSamplingMode,
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
        debugScalarSamplingMode: debugScalarSamplingMode ?? 'current',
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
      console.log(
        '[CPR-NORMALIZATION-AUDIT-JSON]',
        JSON.stringify({
          stage: 'orchestrator-launch',
          runId: debugRunId ?? null,
          attemptLabel: attemptLabel ?? null,
          sourceVolumeId: sourceVolumeId ?? null,
          transferredBuffer: {
            materializedNormalizedBufferBeforeWorker: false,
            transferPayloadKind: 'raw-source-scalars',
            scalarType,
            rawScalarRange: {
              min: scalarMin,
              max: scalarMax,
            },
            normalizedScalarRangeBeforeWorker: null,
            note:
              'The main thread transfers raw source scalarData to the worker. No standalone normalized scalar buffer is materialized before worker init.',
          },
          normalizationPath: {
            effectiveIsPreScaled,
            rawIsPreScaled,
            allowStoredValueNormalization,
            hasUnsignedPackedArtifact,
            hasBitDepthRangeMismatch,
            applyModalityLut,
            rescaleSlope,
            rescaleIntercept,
            bitsStored,
            bitsAllocated,
            highBit,
            pixelRepresentation,
            scalarMaxExceedsNominalBitsStoredRange:
              scalarMax > nominalStoredMax,
            nominalStoredMax,
            nominalSignedMin,
            unsignedPackedArtifactBranchCondition:
              '!effectiveIsPreScaled && scalarData instanceof Int16Array && nominalBitsStored < 16 && pixelRepresentation === 0 && (scalarMin < -1 || scalarMax > nominalStoredMax + 8)',
            overflowHandlingPath:
              'useCPROrchestrator heuristics only flag and forward raw data; packed-value correction happens later inside cprScalarPolicy.resolveStoredValueNormalizationPolicy in the worker.',
            modalityLutApplicationOrder:
              'No modality LUT is applied on the main thread before worker transfer.',
          },
        })
      );

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

      const needsNewWorkerSession =
        !activeCPRWorkerSession || activeCPRWorkerSession.volumeKey !== volumeKey;
      if (needsNewWorkerSession) {
        await terminateCPRWorkerSession();
        const {
          worker,
          workerEntryUrl,
          workerRuntimeUrl,
          revokeWorkerRuntimeUrl,
          workerEntryReadyPromise,
          handoffBufferedEvents,
        } =
          createInstrumentedCPRWorker({
          mode: 'lazy-init',
          volumeKey,
        });
        const sessionKey = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        const workerSession = createCPRWorkerSession(
          worker,
          volumeKey,
          sessionKey,
          workerEntryUrl,
          workerRuntimeUrl,
          'lazy-init',
          workerEntryReadyPromise,
          handoffBufferedEvents
        );
        workerSession.revokeWorkerRuntimeUrl = revokeWorkerRuntimeUrl;
        activeCPRWorkerSession = workerSession;
        console.log('[CPR-WORKER-SESSION-CREATED-JSON]', {
          sessionKey,
          volumeKey,
          mode: 'lazy-init',
          workerEntry: CPR_WORKER_ENTRY_SPECIFIER,
          workerEntryUrl,
          workerRuntimeUrl,
        });
      }

      if (!activeCPRWorkerSession?.initCompleted) {
        const workerSession = activeCPRWorkerSession;
        if (!workerSession) {
          throw new Error('[cprWorker] Worker session was not initialized.');
        }
        const isSharedArrayBuffer = scalarData.buffer instanceof SharedArrayBuffer;
        const initScalarData = isSharedArrayBuffer
          ? scalarData
          : (scalarData.slice(0) as Float32Array | Int16Array);
        const initPayload = {
          type: 'INIT_VOLUME' as const,
          sessionKey: workerSession.sessionKey,
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
        const initTimeoutMs = computeCPRWorkerInitTimeoutMs(initScalarData);
        try {
          await beginCPRWorkerVolumeInit({
            workerSession,
            initPayload,
            transferList,
            timeoutMs: initTimeoutMs,
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
      const renderTimeoutMs = computeCPRWorkerRenderTimeoutMs({
        scalarData,
        panoWidth,
        panoHeight,
        frameCount: serializedFrames.length,
        slabSamples,
        renderBackend,
        reconstructionMode,
        debugRunId,
      });
      if (debugRunId) {
        console.log(
          `${logPrefix} [CPR-WORKER-RENDER-TIMEOUT-JSON]`,
          JSON.stringify({
            panoWidth,
            panoHeight,
            frameCount: serializedFrames.length,
            slabSamples,
            renderBackend,
            reconstructionMode: reconstructionMode ?? 'legacy',
            scalarBytes: estimateCPRWorkerPayloadByteLength(scalarData),
            timeoutMs: renderTimeoutMs,
          })
        );
      }

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
          debugScalarSamplingMode,
          debugRunId,
          attemptLabel,
          reconstructionMode,
          allowLegacyFallback,
        },
        {
          expectedTypes: ['SUCCESS'],
          timeoutMs: renderTimeoutMs,
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
  const desiredVerticalHalfMm = Math.max(15, Math.min(35, params.baseVerticalHalfMm));
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
      verticalHalfMm: Math.max(14, Math.min(33, roundToStep(desiredVerticalHalfMm - 1.5, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.5 : 0.35,
        Math.min(
          averageProjection ? 15 : 0.9,
          roundToStep(
            averageProjection
              ? Math.max(Math.min(baseSlabHalfMm, 15), Math.min(desiredSlabHalfMm, 15), 1.5)
              : Math.min(baseSlabHalfMm, desiredSlabHalfMm),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 41), Math.min(desiredSamples, 41), 9)
          : Math.min(baseSamples, desiredSamples),
        averageProjection ? 9 : 3,
        averageProjection ? 41 : 11
      ),
    },
    {
      label: 'balanced',
      verticalHalfMm: Math.max(15, Math.min(35, roundToStep(desiredVerticalHalfMm, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.75 : 0.5,
        Math.min(
          averageProjection ? 15 : 1.4,
          roundToStep(
            averageProjection
              ? Math.max(Math.min(baseSlabHalfMm, 15), Math.min(desiredSlabHalfMm, 15), 1.75)
              : Math.max(desiredSlabHalfMm, Math.min(baseSlabHalfMm, 1.4)),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 41), Math.min(desiredSamples, 41), 11)
          : Math.max(baseSamples, desiredSamples),
        averageProjection ? 11 : 5,
        averageProjection ? 41 : 15
      ),
    },
    {
      label: 'broad',
      verticalHalfMm: Math.max(16, Math.min(37, roundToStep(desiredVerticalHalfMm + 2, 0.5))),
      slabHalfThicknessMm: Math.max(
        averageProjection ? 1.9 : 0.9,
        Math.min(
          averageProjection ? 15 : 2.2,
          roundToStep(
            Math.max(
              averageProjection
                ? Math.min(baseSlabHalfMm, 15)
                : baseSlabHalfMm,
              averageProjection
                ? Math.min(desiredSlabHalfMm + 0.15, 15)
                : desiredSlabHalfMm + 0.35
            ),
            averageProjection ? 0.1 : 0.05
          )
        )
      ),
      slabSamples: toOddSampleCount(
        averageProjection
          ? Math.max(Math.min(baseSamples, 41), Math.min(desiredSamples + 2, 41), 11)
          : Math.max(baseSamples, desiredSamples + 2),
        averageProjection ? 11 : 7,
        averageProjection ? 41 : 19
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
  panoHeight: requestedPanoHeight = 800,
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
      const projectedSpline = {
        points: rawControlPointsWorld,
        diagnostics: {
          projected: false,
          maxDistanceMm: 0,
          meanDistanceMm: 0,
        },
      };
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
      const sanitizedSpline = {
        points: rawControlPoints,
        diagnostics: {
          inputCount: rawControlPoints.length,
          outputCount: rawControlPoints.length,
          consecutiveDuplicateRemoved: 0,
          loopbackToStartRemoved: 0,
          terminalClosureRemoved: false,
          dedupeThresholdMm: 0,
          closureThresholdMm: 0,
          suspiciousJumpThresholdMm: 0,
          medianEdgeLengthMm: 0,
          maxEdgeLengthMm: 0,
          suspiciousJumpCount: 0,
          severeJumpDetected: false,
        },
      };
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
      const baseVerticalHalfMm = CPR_PANO_FIXED_VERTICAL_HALF_MM;
      const neutralVerticalCenterOffsetMm = 0;
      const rigidSliceVerticalCenterOffsetMm = 0;
      const rigidVerticalSliceModeEnabled = false;
      const mildSuperiorCenterOffsetMm = 1.4;
      const strongSuperiorCenterOffsetMm = 2.6;
      const superiorMandibularCenterOffsetMm = -2.4;
      const subtleMandibularCenterOffsetMm = -3.8;
      const balancedMediumCenterOffsetMm = -4.4;
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
      const balancedMediumSlabHalfThicknessMm = 1.6;
      const balancedMediumSlabSamples = 13;
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
      const pickReadableWorkerVerticalHalfMm = (
        presetSelection: ReturnType<typeof selectReadableVtkPanoPreset>,
        label: VtkPanoPresetCandidate['label'],
        fallback: number
      ): number =>
        presetSelection.candidates.find(candidate => candidate.label === label)?.verticalHalfMm ??
        presetSelection.selected.verticalHalfMm ??
        fallback;
      const focusedMeanWorkerPreset = selectReadableVtkPanoPreset({
        baseVerticalHalfMm,
        requestedSlabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
        requestedSlabSamples: focusedMeanSlabSamples,
        minSpacing,
        aggregation: 'MEAN',
      });
      const balancedMeanWorkerPreset = selectReadableVtkPanoPreset({
        baseVerticalHalfMm,
        requestedSlabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
        requestedSlabSamples: balancedMeanSlabSamples,
        minSpacing,
        aggregation: 'MEAN',
      });
      const broadMeanWorkerPreset = selectReadableVtkPanoPreset({
        baseVerticalHalfMm,
        requestedSlabHalfThicknessMm: broadMeanSlabHalfThicknessMm,
        requestedSlabSamples: broadMeanSlabSamples,
        minSpacing,
        aggregation: 'MEAN',
      });
      const thinnerVerticalHalfMm = pickReadableWorkerVerticalHalfMm(
        focusedMeanWorkerPreset,
        'focused',
        baseVerticalHalfMm
      );
      const narrowVerticalHalfMm = thinnerVerticalHalfMm;
      const toothBandVerticalHalfMm = thinnerVerticalHalfMm;
      const mediumVerticalHalfMm = pickReadableWorkerVerticalHalfMm(
        balancedMeanWorkerPreset,
        'balanced',
        baseVerticalHalfMm
      );
      const balancedMediumVerticalHalfMm = mediumVerticalHalfMm;
      const broadVerticalHalfMm = pickReadableWorkerVerticalHalfMm(
        broadMeanWorkerPreset,
        'broad',
        mediumVerticalHalfMm
      );
      const rootedMediumVerticalHalfMm = toothBandVerticalHalfMm;
      const minimumPanoHeightPx = clampPanoDimension(
        Math.round((CPR_PANO_FIXED_VERTICAL_HALF_MM * 2) / minSpacing)
      );

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
        verticalHalfMm: mediumVerticalHalfMm,
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
        debugScalarSamplingMode:
          | 'current'
          | 'lut-only'
          | 'no-stored-value-normalization'
          | 'raw-stored-values-debug';
        toothBandStageDiagnostics?: ToothBandStageDiagnostics | null;
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
        const requestedDebugScalarSamplingMode =
          overrides.debugScalarSamplingMode === 'lut-only' ||
          overrides.debugScalarSamplingMode === 'no-stored-value-normalization' ||
          overrides.debugScalarSamplingMode === 'raw-stored-values-debug'
            ? overrides.debugScalarSamplingMode
            : 'current';
        const actualVertHalfMm = toPositiveFinite(
          overrides.verticalHalfMm,
          toPositiveFinite(workerInput.verticalHalfMm, baseVerticalHalfMm)
        );
        const idealPanoHeight = Math.round((actualVertHalfMm * 2) / minSpacing);
        const autoDerivedPanoHeight = clampPanoDimension(Math.max(2, idealPanoHeight));
        const finalPanoHeight = clampPanoDimension(
          Math.max(
            2,
            Number.isFinite(Number(overrides.panoHeight))
              ? Number(overrides.panoHeight)
              : requestedPanoHeightPx
          )
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
            autoDerivedPanoHeight,
            slabHalfThicknessMm: requestedSlabHalfThicknessMm,
            slabSamples: requestedSlabSamples,
            verticalHalfMm: actualVertHalfMm,
            verticalCenterOffsetMm: requestedVerticalCenterOffsetMm,
            rigidVerticalSliceMode: requestedRigidVerticalSliceMode,
            debugScalarSamplingMode: requestedDebugScalarSamplingMode,
          })
        );
        if (finalPanoHeight !== autoDerivedPanoHeight) {
          console.debug(
            '[CPR-PANO-HEIGHT-FIX-JSON]',
            JSON.stringify({
              runId: debugRunId,
              label,
              requestedPanoHeight: requestedPanoHeightPx,
              autoDerivedPanoHeight,
              finalPanoHeight,
              source: 'runWorkerAttempt-finalPanoHeight-override',
            })
          );
        }
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
          debugScalarSamplingMode: requestedDebugScalarSamplingMode,
        });
        const result = rawResult;
        const hasIdentityRescaleMetadata =
          Math.abs(result.rescaleSlope - 1) <= 1e-6 && Math.abs(result.rescaleIntercept) <= 1e-6;
        const classifiedIntensityDomain = classifySyntheticCprIntensityDomain({
          modalityLutApplied: result.modalityLutApplied,
          effectiveIsPreScaled: result.effectiveIsPreScaled,
          minValue: result.minValue,
          maxValue: result.maxValue,
        });
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
        const toothBandStageDiagnostics = buildToothBandStageDiagnostics({
          buffer: summaryPixelData,
          width: finalPanoWidth,
          height: finalPanoHeight,
          debugMaps: result.debugMaps,
          analysisCenterRow: summaryAnalysisCenterRow ?? undefined,
          summary,
        });
        const supportSurfaceMetrics = extractPhase4QualityGateMetrics(summary, result.workerDebugPayload);
        const renderSupportMode = supportSurfaceMetrics.renderSupportMode;
        const intensityDomain = resolveSyntheticCprIntensityDomainForRenderMode(
          classifiedIntensityDomain,
          renderSupportMode
        );
        const huDomain = isSyntheticCprHuDomain(intensityDomain);
        const convertedToHu = false;
        const rescaleSkippedAsUnsafe = intensityDomain === 'unknown';
        const isDualArchProjection = isDualArchProjectionRenderMode(renderSupportMode);
        const workerVoi =
          isDualArchProjection || isNativeDisplayPanoRenderMode(renderSupportMode)
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
        const workerScalarSamplingDiagnostic =
          workerDiagnostic?.scalarSampling && typeof workerDiagnostic.scalarSampling === 'object'
            ? (workerDiagnostic.scalarSampling as Record<string, unknown>)
            : null;
        const routeDiagnostic = resolveWorkerDisplayRouteDiagnostic(result.workerDebugPayload);
        const workerOutputSelection =
          workerDiagnostic?.outputSelection && typeof workerDiagnostic.outputSelection === 'object'
            ? (workerDiagnostic.outputSelection as Record<string, unknown>)
            : null;
        const resolvedReconstructionMode =
          toNonEmptyString(workerDiagnostic?.reconstructionMode) ||
          toNonEmptyString(workerDiagnostic?.pipelineMode) ||
          routeDiagnostic.pipelineMode;
        const workerVirtualPanoAccepted =
          readPhase4DiagnosticBoolean(workerOutputSelection?.virtualPanoAccepted) === true;
        const workerVirtualPanoRejected =
          readPhase4DiagnosticBoolean(workerOutputSelection?.virtualPanoRejected) === true;
        const workerVirtualPanoAcceptedByLowerBandTolerance =
          readPhase4DiagnosticBoolean(workerOutputSelection?.acceptedByLowerBandTolerance) === true;
        const workerVirtualPanoAcceptedByToothBandTolerance =
          readPhase4DiagnosticBoolean(workerOutputSelection?.acceptedByToothBandTolerance) === true;
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
          Math.max(0, actualVertHalfMm - 40) / 2.0 +
          Math.max(
            0,
            requestedSlabHalfThicknessMm - (requestedAggregation === 'MIP' ? 0.8 : 15.5)
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
          ? Math.max(0, actualVertHalfMm - 38) *
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
        const isRadiographVirtualPano =
          isVirtualPanoLikeCandidateMode(resolvedReconstructionMode) &&
          (toNonEmptyString(supportSurfaceMetrics.rendererVariant) ===
            'virtual-panoramic-radiograph' ||
            toNonEmptyString(supportSurfaceMetrics.pipelineVariant) ===
              'virtualPanoramicRadiograph' ||
            isRadiographVirtualPanoRenderMode(renderSupportMode));
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
          isVirtualPanoLikeCandidateMode(resolvedReconstructionMode) &&
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
        const workerAcceptedVirtualPanoStructurallyUsable =
          isVirtualPanoLikeCandidateMode(resolvedReconstructionMode) &&
          routeDiagnostic.backend === 'cpu' &&
          workerVirtualPanoAccepted &&
          !workerVirtualPanoRejected &&
          (workerVirtualPanoAcceptedByLowerBandTolerance ||
            workerVirtualPanoAcceptedByToothBandTolerance ||
            ((summary?.lowerBandBrightFraction ?? Number.POSITIVE_INFINITY) <= 0.14 &&
              (summary?.lowerBandP50 ?? Number.POSITIVE_INFINITY) <= -300 &&
              toothBandContrastRange >= 500 &&
              (summary?.toothBandBlackClipFraction ?? Number.POSITIVE_INFINITY) <= 0.12 &&
              (supportSurfaceMetrics.supportUnstableColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.28 &&
              (supportSurfaceMetrics.supportAmbiguousColumnFraction ?? Number.POSITIVE_INFINITY) <= 0.68 &&
              (supportSurfaceMetrics.supportScoreGapP50 ?? 0) >= 0.03 &&
              (supportSurfaceMetrics.supportBestDepthDriftP95Mm ?? Number.POSITIVE_INFINITY) <= 1.3));
        const hardRejectReason =
          requestedRenderBackend === 'gpu' &&
          routeDiagnostic.backend === 'gpu' &&
          phase2GatePassed === false
            ? 'gpu-phase2-gate-failed'
            : !baseHardRejectReason && unstableSupportSurface
              ? 'support-surface-instability'
            : !baseHardRejectReason &&
                !isRadiographVirtualPano &&
                unstableVirtualPanoSupportSurface &&
                !workerAcceptedVirtualPanoStructurallyUsable
              ? 'virtual-pano-support-instability'
            : !baseHardRejectReason && excessiveCenterDrift
              ? 'vertical-center-drift'
              : !baseHardRejectReason &&
                  !!summary &&
                  requestedAggregation === 'MEAN' &&
                  actualVertHalfMm > 40 &&
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
          backgroundBands: summary?.backgroundBands,
          toothBandHoleFraction: summary?.toothBandHoleFraction,
          toothBandBlackClipFraction: summary?.toothBandBlackClipFraction,
          toothBandRetainedWeightP10: summary?.toothBandRetainedWeightP10,
          toothBandRetainedWeightP50: summary?.toothBandRetainedWeightP50,
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
            debugScalarSamplingMode: requestedDebugScalarSamplingMode,
            actualDebugScalarSamplingMode:
              toNonEmptyString(workerScalarSamplingDiagnostic?.debugScalarSamplingMode) ?? null,
            normalizationSignature:
              toNonEmptyString(workerScalarSamplingDiagnostic?.normalizationSignature) ?? null,
            modalityLutApplied: result.modalityLutApplied,
            requestedModalityLutApplied: result.requestedModalityLutApplied,
            storedValueNormalizationApplied: result.storedValueNormalizationApplied,
            effectiveIsPreScaled: result.effectiveIsPreScaled,
            outputSignature,
          })
        );
        if (
          toothBandStageDiagnostics &&
          (label === CPR_TARGET_TOOTH_DEBUG_LABEL || requestedDebugScalarSamplingMode !== 'current')
        ) {
          console.debug(
            '[CPR-TOOTH-STAGE-JSON]',
            JSON.stringify({
              runId: debugRunId,
              label,
              debugScalarSamplingMode: requestedDebugScalarSamplingMode,
              modalityLutApplied: result.modalityLutApplied,
              requestedModalityLutApplied: result.requestedModalityLutApplied,
              storedValueNormalizationApplied: result.storedValueNormalizationApplied,
              effectiveIsPreScaled: result.effectiveIsPreScaled,
              diagnostics: toothBandStageDiagnostics,
            })
          );
        }
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
            `bgBand05=${summary?.backgroundBands?.dominantOutlierBand05 ?? 'na'} ` +
            `bgBand10=${summary?.backgroundBands?.dominantOutlierBand10 ?? 'na'} ` +
            `blackClip=${formatReadablePanoValue(supportSurfaceMetrics.blackClipFraction, 3)} ` +
            `toothHole=${formatReadablePanoValue(summary?.toothBandHoleFraction, 3)} ` +
            `toothBlackClip=${formatReadablePanoValue(summary?.toothBandBlackClipFraction, 3)} ` +
            `toothRetainedP10=${formatReadablePanoValue(summary?.toothBandRetainedWeightP10, 3)} ` +
            `toothRetainedP50=${formatReadablePanoValue(summary?.toothBandRetainedWeightP50, 3)} ` +
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
            toothBandHoleFraction: summary?.toothBandHoleFraction ?? null,
            toothBandBlackClipFraction: summary?.toothBandBlackClipFraction ?? null,
            toothBandRetainedWeightP10: summary?.toothBandRetainedWeightP10 ?? null,
            toothBandRetainedWeightP50: summary?.toothBandRetainedWeightP50 ?? null,
            lowerBandP50: summary?.lowerBandP50 ?? null,
            lowerBandBrightFraction: summary?.lowerBandBrightFraction ?? null,
            backgroundToneSampleCount: summary?.backgroundToneSampleCount ?? null,
            backgroundToneP95: summary?.backgroundToneP95 ?? null,
            backgroundToneP99: summary?.backgroundToneP99 ?? null,
            backgroundToneMax: summary?.backgroundToneMax ?? null,
            backgroundOutlierFraction05: summary?.backgroundOutlierFraction05 ?? null,
            backgroundOutlierFraction10: summary?.backgroundOutlierFraction10 ?? null,
            backgroundBands: summary?.backgroundBands ?? null,
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
        if (summary?.backgroundBands) {
          console.log(
            '[CPR-BACKGROUND-LEAK-BANDS-JSON]',
            JSON.stringify({
              runId: debugRunId,
              label,
              dominantOutlierBand05: summary.backgroundBands.dominantOutlierBand05,
              dominantOutlierBand10: summary.backgroundBands.dominantOutlierBand10,
              backgroundBands: summary.backgroundBands,
            })
          );
        }

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
          debugScalarSamplingMode: requestedDebugScalarSamplingMode,
          toothBandStageDiagnostics,
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
      type EvaluatedPanoAttempt = Awaited<ReturnType<typeof runWorkerAttempt>>;
      const getFiniteAttemptMetric = (
        value: number | null | undefined,
        fallback: number
      ): number => (Number.isFinite(value) ? Number(value) : fallback);
      const shouldAbortVirtualPanoRetryLadder = (): {
        reason: string;
        clusteredLabels: string[];
      } | null => {
        const recentAttempts = attemptResults.slice(-3) as EvaluatedPanoAttempt[];
        if (recentAttempts.length < 3) {
          return null;
        }

        const isVirtualPanoInstabilityAttempt = (attempt: EvaluatedPanoAttempt): boolean =>
          attempt.resolvedBackend === 'cpu' &&
          isVirtualPanoLikeCandidateMode(attempt.reconstructionMode) &&
          attempt.hardRejectReason === 'virtual-pano-support-instability' &&
          !attempt.qualityGatePassed;

        if (!recentAttempts.every(isVirtualPanoInstabilityAttempt)) {
          return null;
        }

        const hasSameRejectShape = recentAttempts.every(
          attempt =>
            attempt.qualityGateRejectReasons.includes('off-trough-energy-too-high') &&
            attempt.qualityGateRejectReasons.includes('lower-band-bright-fraction-too-high') &&
            attempt.qualityGateRejectReasons.includes('lower-band-mean-too-high') &&
            attempt.qualityGateRejectReasons.includes('lower-suppression-ratio-too-high')
        );
        if (!hasSameRejectShape) {
          return null;
        }

        const metricRange = (values: number[]): number =>
          values.length > 0 ? Math.max(...values) - Math.min(...values) : 0;

        const ambiguousFractions = recentAttempts.map(attempt =>
          getFiniteAttemptMetric(attempt.qualityGateMetrics.supportAmbiguousColumnFraction, 1)
        );
        const scoreGaps = recentAttempts.map(attempt =>
          getFiniteAttemptMetric(attempt.qualityGateMetrics.supportScoreGapP50, 0)
        );
        const depthStdValues = recentAttempts.map(attempt =>
          getFiniteAttemptMetric(attempt.qualityGateMetrics.supportDepthStdMm, 0)
        );
        const unstableFractions = recentAttempts.map(attempt =>
          getFiniteAttemptMetric(attempt.qualityGateMetrics.supportUnstableColumnFraction, 0)
        );
        const lowerBandBrightFractions = recentAttempts.map(attempt =>
          getFiniteAttemptMetric(attempt.summary?.lowerBandBrightFraction, 0)
        );
        const qualityScores = recentAttempts.map(attempt => attempt.qualityScore);

        if (
          Math.min(...ambiguousFractions) < 0.5 ||
          Math.max(...scoreGaps) > 0.035 ||
          Math.min(...depthStdValues) < 0.75
        ) {
          return null;
        }

        const hasFlatSupportSurfaceFailure =
          Math.min(...ambiguousFractions) >= 0.9 &&
          Math.max(...scoreGaps) <= 0.015 &&
          Math.min(...depthStdValues) >= 1.0 &&
          Math.min(...unstableFractions) >= 0.24 &&
          metricRange(scoreGaps) <= 0.004 &&
          metricRange(depthStdValues) <= 0.5 &&
          metricRange(unstableFractions) <= 0.14 &&
          metricRange(qualityScores) <= 3.5;

        if (hasFlatSupportSurfaceFailure) {
          return {
            reason: 'retry-virtual-pano-flat-support-surface',
            clusteredLabels: recentAttempts.map(attempt => attempt.label),
          };
        }

        if (
          metricRange(ambiguousFractions) > 0.08 ||
          metricRange(scoreGaps) > 0.01 ||
          metricRange(depthStdValues) > 0.25 ||
          metricRange(lowerBandBrightFractions) > 0.08 ||
          metricRange(qualityScores) > 1.25
        ) {
          return null;
        }

        return {
          reason: 'retry-virtual-pano-support-instability-cluster',
          clusteredLabels: recentAttempts.map(attempt => attempt.label),
        };
      };
      type Phase2BaseSelectionAssessment = {
        attempt: EvaluatedPanoAttempt;
        phase0Rejected: boolean;
        seedEligible: boolean;
        structuralRejectReasons: string[];
        score: number;
        priorityIndex: number;
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
      const phase2PriorityIndexByLabel = new Map(
        CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS.map((label, index) => [label, index] as const)
      );
      const collectPhase2SeedStructuralRejectReasons = (attempt: EvaluatedPanoAttempt): string[] => {
        const reasons: string[] = [];
        if (attempt.hardRejectReason === 'support-surface-instability') {
          reasons.push('hard-reject:support-surface-instability');
        } else if (attempt.hardRejectReason) {
          reasons.push(`hard-reject:${attempt.hardRejectReason}`);
        }
        for (const rejectReason of attempt.qualityGateRejectReasons) {
          if (
            rejectReason === 'tooth-band-hole-fraction-too-high' ||
            rejectReason === 'tooth-band-black-clip-too-high' ||
            rejectReason === 'tooth-band-retained-weight-collapsed' ||
            rejectReason === 'middle-band-leakage-dominant'
          ) {
            reasons.push(rejectReason);
          }
        }
        if ((attempt.summary?.toothBandHoleFraction ?? 0) > CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION) {
          reasons.push('tooth-band-hole-fraction-too-high');
        }
        if (
          (attempt.summary?.toothBandBlackClipFraction ?? 0) >
          CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION
        ) {
          reasons.push('tooth-band-black-clip-too-high');
        }
        if (
          (attempt.summary?.toothBandRetainedWeightP10 ?? 1) <=
          CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX
        ) {
          reasons.push('tooth-band-retained-weight-collapsed');
        }
        if (hasPhase4DominantMiddleBandLeakage(attempt.summary)) {
          reasons.push('middle-band-leakage-dominant');
        }
        return Array.from(new Set(reasons));
      };
      const assessPhase2BaseAttempt = (attempt: EvaluatedPanoAttempt): Phase2BaseSelectionAssessment => {
        const summary = attempt.summary;
        const absVerticalCenterOffsetMm = Math.abs(attempt.verticalCenterOffsetMm);
        const lowerBandBrightFraction = summary?.lowerBandBrightFraction ?? 0;
        const lowerBandP50 = summary?.lowerBandP50 ?? -650;
        const backgroundOutlierFraction05 = summary?.backgroundOutlierFraction05 ?? 0;
        const backgroundOutlierFraction10 = summary?.backgroundOutlierFraction10 ?? 0;
        const toothBandContrastRange = summary ? summary.toothBandP90 - summary.toothBandP10 : 0;
        const toothBandDamagePenalty = computePhase4ToothBandDamagePenalty({
          toothBandHoleFraction: summary?.toothBandHoleFraction ?? null,
          toothBandBlackClipFraction: summary?.toothBandBlackClipFraction ?? null,
          toothBandRetainedWeightP10: summary?.toothBandRetainedWeightP10 ?? null,
          toothBandRetainedWeightP50: summary?.toothBandRetainedWeightP50 ?? null,
        });
        const middleBandLeakPenalty = computePhase4MiddleBandLeakPenalty(summary);
        const balancedMeanRecoveryBonus = computePhase4BalancedMeanRecoveryBonus({
          isGpuMeanAttempt:
            attempt.requestedBackend === 'gpu' &&
            attempt.resolvedBackend === 'gpu' &&
            attempt.aggregation === 'MEAN',
          summary,
          actualVertHalfMm: attempt.actualVertHalfMm,
          verticalCenterOffsetMm: attempt.verticalCenterOffsetMm,
          slabHalfThicknessMm: attempt.slabHalfThicknessMm,
        });
        const centerPenalty =
          absVerticalCenterOffsetMm * 3.2 +
          Math.max(0, absVerticalCenterOffsetMm - 3.5) * 4.5;
        const lowerBandPenalty =
          Math.max(0, lowerBandBrightFraction - 0.03) * 16 +
          Math.max(0, lowerBandP50 + 420) / 42;
        const backgroundPenalty =
          Math.max(0, backgroundOutlierFraction05 - 0.16) * 22 +
          Math.max(0, backgroundOutlierFraction10 - 0.07) * 28;
        const slabPenalty = Math.max(0, attempt.slabHalfThicknessMm - 1.6) * 1.8;
        const contrastReward = Math.max(0, toothBandContrastRange - 560) / 120;
        const structuralRejectReasons = collectPhase2SeedStructuralRejectReasons(attempt);
        const phase0Rejected = !attempt.qualityGatePassed || !!attempt.hardRejectReason;
        const seedEligible = !attempt.hardRejectReason && structuralRejectReasons.length === 0;
        const phase0RejectPenalty =
          (attempt.qualityGatePassed ? 0 : 180) +
          (attempt.hardRejectReason ? 420 : 0) +
          structuralRejectReasons.length * 180;
        return {
          attempt,
          phase0Rejected,
          seedEligible,
          structuralRejectReasons,
          score:
            centerPenalty +
            lowerBandPenalty +
            backgroundPenalty +
            slabPenalty +
            toothBandDamagePenalty +
            middleBandLeakPenalty +
            phase0RejectPenalty -
            contrastReward -
            balancedMeanRecoveryBonus,
          priorityIndex:
            phase2PriorityIndexByLabel.get(
              attempt.label as (typeof CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS)[number]
            ) ?? Number.POSITIVE_INFINITY,
        };
      };
      const collectPhase2BaseSelectionAssessments = (): Phase2BaseSelectionAssessment[] =>
        attemptResults
          .filter((attempt): attempt is EvaluatedPanoAttempt => attempt.aggregation === 'MEAN')
          .map(assessPhase2BaseAttempt);
      const selectPreferredMeanAttemptForPhase2 = (): Awaited<
        ReturnType<typeof runWorkerAttempt>
      > | null => {
        const phase2BaseAssessments = collectPhase2BaseSelectionAssessments();
        if (!phase2BaseAssessments.length) {
          return null;
        }

        const eligibleAssessments = phase2BaseAssessments.filter(assessment => assessment.seedEligible);
        if (!eligibleAssessments.length) {
          return null;
        }

        let selected = eligibleAssessments[0];
        for (let index = 1; index < eligibleAssessments.length; index++) {
          const candidate = eligibleAssessments[index];
          if (
            candidate.score < selected.score - 1e-6 ||
            (Math.abs(candidate.score - selected.score) <= 1e-6 &&
              candidate.priorityIndex < selected.priorityIndex)
          ) {
            selected = candidate;
          }
        }

        return selected.attempt;
      };
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
            'middle-band-leakage-dominant',
            'tooth-band-holes',
            'tooth-band-black-clipped',
            'tooth-band-retention-low',
            'tooth-band-retention-collapsed',
          ]);
          for (const riskFlag of attempt.supportSurfaceRiskFlags) {
            if (blockingRiskFlags.has(riskFlag)) {
              reasons.push(`risk-flag:${riskFlag}`);
            }
          }
          if ((attempt.summary?.toothBandHoleFraction ?? 0) > CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION) {
            reasons.push('tooth-band-hole-fraction-high');
          }
          if (
            (attempt.summary?.toothBandBlackClipFraction ?? 0) >
            CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION
          ) {
            reasons.push('tooth-band-black-clip-high');
          }
          if (
            (attempt.summary?.toothBandRetainedWeightP10 ?? 1) <=
            CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX
          ) {
            reasons.push('tooth-band-retained-weight-collapsed');
          }
          if (hasPhase4DominantMiddleBandLeakage(attempt.summary)) {
            reasons.push('middle-band-leakage-dominant');
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
      const shouldUseFastWorkerReconPhase2Seed =
        CPR_PANO_DISPLAY_PATH_DEFAULT === 'worker-recon' && !CPR_PANO_QUALITY_FIRST_MODE;

      await workerWarmupPromise;
      const primaryAttemptLabel = shouldUseFastWorkerReconPhase2Seed
        ? 'retry-mean-balanced-neutral'
        : 'primary-mean-toothband-narrow';
      const primaryAttemptOverrides = shouldUseFastWorkerReconPhase2Seed
        ? {
            renderBackend: 'cpu' as const,
            reconstructionMode: 'virtualPano' as const,
            allowLegacyFallback: false,
            modalityLutOverride: true,
            verticalHalfMm: narrowVerticalHalfMm,
            verticalCenterOffsetMm: neutralVerticalCenterOffsetMm,
            slabHalfThicknessMm: balancedMeanSlabHalfThicknessMm,
            slabSamples: balancedMeanSlabSamples,
            aggregation: 'MEAN' as const,
          }
        : {
            modalityLutOverride: true,
            verticalHalfMm: toothBandVerticalHalfMm,
            verticalCenterOffsetMm: subtleMandibularCenterOffsetMm,
            slabHalfThicknessMm: focusedMeanSlabHalfThicknessMm,
            slabSamples: focusedMeanSlabSamples,
            aggregation: 'MEAN' as const,
          };
      let bestAttempt = await runWorkerAttempt(primaryAttemptLabel, primaryAttemptOverrides);
      recordAttempt(bestAttempt);
      const primaryAttemptEscalationReasons = collectAttemptEscalationReasons(bestAttempt);
      logAttemptEscalation('primary-attempt', bestAttempt, primaryAttemptEscalationReasons);

      if (isGoodEnoughPanoAttempt(bestAttempt)) {
        earlyExitReason = 'primary-good-enough';
      } else if (shouldUseFastWorkerReconPhase2Seed) {
        earlyExitReason = 'worker-recon-fast-phase2-seed';
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
            verticalHalfMm: balancedMediumVerticalHalfMm,
            verticalCenterOffsetMm: balancedMediumCenterOffsetMm,
            slabHalfThicknessMm: balancedMediumSlabHalfThicknessMm,
            slabSamples: balancedMediumSlabSamples,
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
          const virtualPanoRetryAbort = shouldAbortVirtualPanoRetryLadder();
          if (virtualPanoRetryAbort) {
            earlyExitReason = `${virtualPanoRetryAbort.reason}:${virtualPanoRetryAbort.clusteredLabels.join(',')}`;
            console.debug(
              '[CPR-VIRTUAL-PANO-RETRY-EARLY-EXIT-JSON]',
              JSON.stringify({
                runId: debugRunId,
                reason: virtualPanoRetryAbort.reason,
                clusteredLabels: virtualPanoRetryAbort.clusteredLabels,
                bestAttemptLabel: bestAttempt.label,
                bestAttemptQualityScore: bestAttempt.qualityScore,
                bestAttemptHardRejectReason: bestAttempt.hardRejectReason,
              })
            );
            break;
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
      const phase2BaseAssessments = collectPhase2BaseSelectionAssessments();
      const phase2SeedEligibleBaseAssessments = phase2BaseAssessments.filter(
        assessment => assessment.seedEligible
      );
      const phase2BaseSelectionBlockedReason = !phase2BaseAssessments.length
        ? 'no-mean-attempt'
        : !phase2SeedEligibleBaseAssessments.length
          ? 'no-phase2-seed-eligible-mean-attempt'
          : null;
      const phase2NoBaseAttemptSkipReason =
        phase2BaseSelectionBlockedReason === 'no-phase2-seed-eligible-mean-attempt'
          ? 'NO_PHASE2_SEED_ELIGIBLE_MEAN_ATTEMPT'
          : 'NO_MEAN_ATTEMPT';
      const phase2BaseAssessmentByLabel = new Map(
        phase2BaseAssessments.map(assessment => [assessment.attempt.label, assessment] as const)
      );
      const fastPhase2SeedAssessment =
        bestAttempt.aggregation === 'MEAN' ? assessPhase2BaseAttempt(bestAttempt) : null;
      const phase2BaseAttempt =
        shouldUseFastWorkerReconPhase2Seed &&
        bestAttempt.aggregation === 'MEAN' &&
        fastPhase2SeedAssessment?.seedEligible
          ? bestAttempt
          : selectPreferredMeanAttemptForPhase2();
      const phase2BaseAssessment =
        phase2BaseAttempt ? phase2BaseAssessmentByLabel.get(phase2BaseAttempt.label) ?? null : null;
      console.log(
        '[CPR-PHASE2-BASE-SELECTION-JSON]',
        JSON.stringify({
          runId: debugRunId,
          selectedLabel: phase2BaseAttempt?.label ?? null,
          selectedPhase0QualityGatePassed: phase2BaseAttempt?.qualityGatePassed ?? null,
          selectedPhase0RejectReasons: phase2BaseAttempt?.qualityGateRejectReasons ?? [],
          selectedStructuralRejectReasons: phase2BaseAssessment?.structuralRejectReasons ?? [],
          selectedEligibleForPhase2Seed: phase2BaseAssessment?.seedEligible ?? null,
          selectedPhase2SeedScore: phase2BaseAssessment?.score ?? null,
          eligibleCandidateCount: phase2SeedEligibleBaseAssessments.length,
          ineligibleCandidateCount:
            phase2BaseAssessments.length - phase2SeedEligibleBaseAssessments.length,
          selectionBlockedReason: phase2BaseSelectionBlockedReason,
          prioritizedLabels: CPR_PHASE2_VIRTUAL_PANO_BASE_LABELS,
          candidates: phase2BaseAssessments.map(assessment => ({
            label: assessment.attempt.label,
            phase0QualityGatePassed: assessment.attempt.qualityGatePassed,
            phase0RejectReasons: assessment.attempt.qualityGateRejectReasons,
            hardRejectReason: assessment.attempt.hardRejectReason,
            structuralRejectReasons: assessment.structuralRejectReasons,
            seedEligible: assessment.seedEligible,
            score: assessment.score,
            priorityIndex: assessment.priorityIndex,
          })),
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
        phase1VirtualPano.skippedReason = phase2NoBaseAttemptSkipReason;
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
        intensityDomain: SyntheticCprIntensityDomain;
        huDomain: boolean;
        usedAsDisplayedOutput: boolean;
        borderlineAcceptedReason: string | null;
        workerAcceptedForOutput: boolean;
        orchestratorGatePassed: boolean | null;
        orchestratorRejectReasons: string[];
        displayDecisionReason: string | null;
        phase0SeedQualityGatePassed: boolean | null;
        phase0SeedRejectReasons: string[];
        phase0SeedStructuralRejectReasons: string[];
        phase0SeedEligible: boolean | null;
        revivedRejectedFamily: boolean;
        revivedRejectedFamilyRulePassed: boolean | null;
        revivedRejectedFamilyRejectReasons: string[];
        phase2RenderFamily: string | null;
        phase2CandidateSource: Phase4QualityGateCandidateSource | null;
      } = {
        executed: false,
        skippedReason: null,
        error: null,
        timingMs: null,
        diagnostics: null,
        summary: null,
        voi: null,
        intensityDomain: 'unknown',
        huDomain: false,
        usedAsDisplayedOutput: false,
        borderlineAcceptedReason: null,
        workerAcceptedForOutput: false,
        orchestratorGatePassed: null,
        orchestratorRejectReasons: [],
        displayDecisionReason: null,
        phase0SeedQualityGatePassed: phase2BaseAttempt?.qualityGatePassed ?? null,
        phase0SeedRejectReasons: phase2BaseAttempt?.qualityGateRejectReasons.slice() ?? [],
        phase0SeedStructuralRejectReasons: phase2BaseAssessment?.structuralRejectReasons ?? [],
        phase0SeedEligible: phase2BaseAssessment?.seedEligible ?? null,
        revivedRejectedFamily: false,
        revivedRejectedFamilyRulePassed: null,
        revivedRejectedFamilyRejectReasons: [],
        phase2RenderFamily: null,
        phase2CandidateSource: null,
      };
      let phase2WorkerResult: CPRWorkerLaunchResult | null = null;
      if (!phase2BaseAttempt) {
        phase2VirtualPano.skippedReason = phase2NoBaseAttemptSkipReason;
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
            reconstructionMode: getPanoV2Phase0ReconstructionMode(),
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
          const classifiedPhase2IntensityDomain = classifySyntheticCprIntensityDomain({
            modalityLutApplied: phase2Result.modalityLutApplied,
            effectiveIsPreScaled: phase2Result.effectiveIsPreScaled,
            minValue: phase2Result.minValue,
            maxValue: phase2Result.maxValue,
          });
          const phase2IntensityDomain = resolveSyntheticCprIntensityDomainForRenderMode(
            classifiedPhase2IntensityDomain,
            phase2RenderSupportMode
          );
          const phase2HuDomain = isSyntheticCprHuDomain(phase2IntensityDomain);
          const phase2FusionBypass =
            phase2RenderSupportMode === 'panoV2Fusion' &&
            phase2RenderDiagnostics?.renderBypass === true;
          const phase2WorkerVoi =
            phase2FusionBypass ||
            (!isDualArchProjectionRenderMode(phase2RenderSupportMode) &&
            !isNativeDisplayPanoRenderMode(phase2RenderSupportMode))
              ? createPanoVoiFromWindowLevel(phase2Result.windowWidth, phase2Result.windowCenter)
              : null;
          const phase2AdaptiveVoi = computeAdaptivePanoVoi(
            phase2Summary,
            phase2Result.minValue,
            phase2Result.maxValue,
            phase2RenderSupportMode
          );
          const phase2Voi = phase2WorkerVoi ?? phase2AdaptiveVoi;

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
          phase2VirtualPano.intensityDomain = phase2IntensityDomain;
          phase2VirtualPano.huDomain = phase2HuDomain;
          phase2VirtualPano.phase2RenderFamily = phase2RenderSupportMode;
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
      const phase2RejectedSeedRevivalDecision =
        phase2BaseAttempt && explicitPhase2QualityGateCandidate
          ? (() => {
              const derivedFromPhase0RejectedFamily =
                !phase2BaseAttempt.qualityGatePassed || !!phase2BaseAttempt.hardRejectReason;
              const seedStructuralRejectReasons = phase2BaseAssessment?.structuralRejectReasons ?? [];
              const rejectReasons: string[] = [];
              if (explicitPhase2QualityGateCandidate.hardRejectReason) {
                rejectReasons.push(
                  `phase2-hard-reject:${explicitPhase2QualityGateCandidate.hardRejectReason}`
                );
              }
              if (
                seedStructuralRejectReasons.includes('tooth-band-hole-fraction-too-high') &&
                (explicitPhase2QualityGateCandidate.metrics.toothBandHoleFraction ?? 0) >
                  CPR_TOOTH_BAND_SEVERE_HOLE_FRACTION
              ) {
                rejectReasons.push('phase2-tooth-band-holes-not-cleared');
              }
              if (
                seedStructuralRejectReasons.includes('tooth-band-black-clip-too-high') &&
                (explicitPhase2QualityGateCandidate.metrics.toothBandBlackClipFraction ?? 0) >
                  CPR_TOOTH_BAND_SEVERE_BLACK_CLIP_FRACTION
              ) {
                rejectReasons.push('phase2-tooth-band-black-clip-not-cleared');
              }
              if (
                seedStructuralRejectReasons.includes('tooth-band-retained-weight-collapsed') &&
                (explicitPhase2QualityGateCandidate.metrics.toothBandRetainedWeightP10 ?? 1) <=
                  CPR_TOOTH_BAND_SEVERE_RETAINED_WEIGHT_P10_MAX
              ) {
                rejectReasons.push('phase2-tooth-band-retention-collapse-not-cleared');
              }
              if (
                seedStructuralRejectReasons.includes('middle-band-leakage-dominant') &&
                hasPhase4DominantMiddleBandLeakage(phase2VirtualPano.summary)
              ) {
                rejectReasons.push('phase2-middle-band-leakage-not-cleared');
              }
              if (
                seedStructuralRejectReasons.includes('hard-reject:support-surface-instability') &&
                explicitPhase2QualityGateCandidate.hardRejectReason === 'support-surface-instability'
              ) {
                rejectReasons.push('phase2-support-surface-instability-not-cleared');
              }
              return {
                derivedFromPhase0RejectedFamily,
                seedStructuralRejectReasons,
                pass:
                  explicitPhase2QualityGateCandidate.pass &&
                  (!derivedFromPhase0RejectedFamily || rejectReasons.length === 0),
                rejectReasons,
              };
            })()
          : null;
      if (explicitPhase2QualityGateCandidate) {
        phase2VirtualPano.orchestratorGatePassed = explicitPhase2QualityGateCandidate.pass;
        phase2VirtualPano.orchestratorRejectReasons =
          explicitPhase2QualityGateCandidate.rejectReasons.slice();
        phase2VirtualPano.phase2CandidateSource = explicitPhase2QualityGateCandidate.candidateSource;
      }
      if (phase2RejectedSeedRevivalDecision) {
        phase2VirtualPano.revivedRejectedFamily =
          phase2RejectedSeedRevivalDecision.derivedFromPhase0RejectedFamily;
        phase2VirtualPano.revivedRejectedFamilyRulePassed = phase2RejectedSeedRevivalDecision.pass;
        phase2VirtualPano.revivedRejectedFamilyRejectReasons =
          phase2RejectedSeedRevivalDecision.rejectReasons.slice();
      }
      if (phase2VirtualPano.workerAcceptedForOutput && explicitPhase2QualityGateCandidate) {
        const separateRejectedSeedRulePass =
          phase2RejectedSeedRevivalDecision?.pass ?? explicitPhase2QualityGateCandidate.pass;
        phase2VirtualPano.usedAsDisplayedOutput =
          explicitPhase2QualityGateCandidate.pass && separateRejectedSeedRulePass;
        phase2VirtualPano.displayDecisionReason =
          explicitPhase2QualityGateCandidate.pass && separateRejectedSeedRulePass
            ? phase2RejectedSeedRevivalDecision?.derivedFromPhase0RejectedFamily
              ? 'accepted-by-worker-and-orchestrator-after-rejected-seed-revival-rule'
              : 'accepted-by-worker-and-orchestrator'
            : explicitPhase2QualityGateCandidate.pass
              ? 'rejected-by-rejected-seed-revival-rule'
              : 'rejected-by-orchestrator-gate';
      } else if (phase2VirtualPano.workerAcceptedForOutput) {
        phase2VirtualPano.displayDecisionReason = 'accepted-by-worker';
      } else if (phase2VirtualPano.executed) {
        phase2VirtualPano.displayDecisionReason = 'rejected-by-worker';
      }
      // Preview fallback display is disabled, so do not spend time rendering a rigid-slice preview.
      const shouldRunPhase2RigidSlicePreview = false;
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
      const phase2DisplayedOutputMode: PanoDisplayedOutputMode =
        phase2WorkerResult
          ? resolveDisplayedOutputModeFromWorkerResult({
              routeDiagnostic: resolveWorkerDisplayRouteDiagnostic(
                phase2WorkerResult.workerDebugPayload
              ),
              workerDebugPayload: phase2WorkerResult.workerDebugPayload,
            })
          : 'virtualPanoPhase2';
      let selectedDisplayedOutputMode: PanoDisplayedOutputMode =
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
              attempt.resolvedBackend === 'cpu' &&
              isVirtualPanoLikeCandidateMode(attempt.reconstructionMode)
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
      const shouldPreferPhase2BasePreviewFallback = false;
      const shouldPreferPhase2RigidSlicePreviewFallback = false;
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
        ? phase2DisplayedOutputMode
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
          selectedPhase0QualityGatePassed: phase2BaseAttempt?.qualityGatePassed ?? null,
          selectedPhase0RejectReasons: phase2BaseAttempt?.qualityGateRejectReasons ?? [],
          selectedPhase0StructuralRejectReasons: phase2BaseAssessment?.structuralRejectReasons ?? [],
          selectedEligibleForPhase2Seed: phase2BaseAssessment?.seedEligible ?? null,
          workerAcceptedForOutput: phase2VirtualPano.workerAcceptedForOutput,
          orchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
          orchestratorRejectReasons: phase2VirtualPano.orchestratorRejectReasons,
          displayDecisionReason: phase2VirtualPano.displayDecisionReason,
          revivedRejectedFamily: phase2VirtualPano.revivedRejectedFamily,
          revivedRejectedFamilyRulePassed: phase2VirtualPano.revivedRejectedFamilyRulePassed,
          revivedRejectedFamilyRejectReasons: phase2VirtualPano.revivedRejectedFamilyRejectReasons,
          phase2RenderFamily: phase2VirtualPano.phase2RenderFamily,
          phase2CandidateSource: phase2VirtualPano.phase2CandidateSource,
          qualityGateCandidate: explicitPhase2QualityGateCandidate,
        })
      );
      console.log(
        '[CPR-PHASE2-SEED-PROVENANCE-JSON]',
        JSON.stringify({
          runId: debugRunId,
          seedLabel: phase2BaseAttempt?.label ?? null,
          seedPhase0QualityGatePassed: phase2BaseAttempt?.qualityGatePassed ?? null,
          seedPhase0RejectReasons: phase2BaseAttempt?.qualityGateRejectReasons ?? [],
          seedPhase0StructuralRejectReasons: phase2BaseAssessment?.structuralRejectReasons ?? [],
          seedEligibleForPhase2: phase2BaseAssessment?.seedEligible ?? null,
          seedPhase2Score: phase2BaseAssessment?.score ?? null,
          phase2Executed: phase2VirtualPano.executed,
          phase2RenderFamily: phase2VirtualPano.phase2RenderFamily,
          phase2CandidateSource: phase2VirtualPano.phase2CandidateSource,
          phase2WorkerAccepted: phase2VirtualPano.workerAcceptedForOutput,
          phase2OrchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
          phase2DisplayDecisionReason: phase2VirtualPano.displayDecisionReason,
          revivedRejectedFamily: phase2VirtualPano.revivedRejectedFamily,
          revivedRejectedFamilyRulePassed: phase2VirtualPano.revivedRejectedFamilyRulePassed,
          revivedRejectedFamilyRejectReasons: phase2VirtualPano.revivedRejectedFamilyRejectReasons,
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
      const forceDisplayGpuCandidateEvenIfRejected =
        shouldForceDisplayGpuCandidateEvenIfRejected();
      let forcedDisplayedAttemptForDebug:
        | Awaited<ReturnType<typeof runWorkerAttempt>>
        | null = null;
      let forcedDisplayedQualityGateCandidate: Phase4QualityGateCandidate | null = null;
      let forcedDisplayedAttemptBlockedReason: string | null = null;
      let forcedDisplayedAttemptBlockedReasons: string[] = [];
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
                'worker-cpu-virtual-panoramic-radiograph' ||
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
        shouldForceDisplayRejectedPanoV2FusionForDebug() &&
        !!phase2WorkerResult &&
        phase2VirtualPano.executed &&
        !phase2VirtualPano.usedAsDisplayedOutput &&
        phase2VirtualPano.phase2RenderFamily === 'panoV2Fusion';
      if (
        shouldDisplayPhase2DiagnosticDraft &&
        explicitPhase2QualityGateCandidate &&
        phase2BaseAttempt
      ) {
        const phase2DraftDisplayReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'debug-force-display-rejected-pano-v2-fusion-after-orchestrator-reject'
          : 'debug-force-display-rejected-pano-v2-fusion-after-worker-reject';
        selectedAttempt = phase2BaseAttempt;
        selectedAttemptOverrideReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'phase2-quality-gate-rejected-display-pano-v2-fusion-debug-draft'
          : 'phase2-worker-rejected-display-pano-v2-fusion-debug-draft';
        selectedAttemptRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
          selectedAttempt.workerDebugPayload
        );
        phase2VirtualPano.usedAsDisplayedOutput = true;
        phase2VirtualPano.displayDecisionReason = phase2DraftDisplayReason;
        selectedDisplayedOutputMode = phase2DisplayedOutputMode;
        selectedDisplayedSourceLabel = phase2BaseAttempt.label;
        selectedDisplayedSourceAggregation = phase2BaseAttempt.aggregation;
        selectedQualityGateCandidate = explicitPhase2QualityGateCandidate;
        qualityGateSelectionReason = phase2VirtualPano.workerAcceptedForOutput
          ? 'phase2-pano-v2-fusion-debug-draft-displayed-after-gate-reject'
          : 'phase2-pano-v2-fusion-debug-draft-displayed-after-worker-reject';
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
      const phase2DegradedPreviewAttempt =
        explicitPhase2QualityGateCandidate &&
        phase2WorkerResult &&
        phase2VirtualPano.summary &&
        phase2VirtualPano.voi &&
        phase2BaseAttempt
          ? {
              ...phase2BaseAttempt,
              result: phase2WorkerResult,
              workerDebugPayload: phase2WorkerResult.workerDebugPayload ?? null,
              summary: phase2VirtualPano.summary,
              voi: phase2VirtualPano.voi,
              qualityBase: explicitPhase2QualityGateCandidate.qualityBase,
              qualityScore: explicitPhase2QualityGateCandidate.qualityScore,
              hardRejectReason: explicitPhase2QualityGateCandidate.hardRejectReason,
              qualityGatePassed: explicitPhase2QualityGateCandidate.pass,
              qualityGateRejectReasons:
                explicitPhase2QualityGateCandidate.rejectReasons.slice(),
              qualityGateMetrics: explicitPhase2QualityGateCandidate.metrics,
              supportSurfaceRiskSummary:
                explicitPhase2QualityGateCandidate.supportSurfaceRiskSummary,
              supportSurfaceBaselineFingerprint:
                explicitPhase2QualityGateCandidate.supportSurfaceRiskSummary
                  .baselineFingerprint,
              supportSurfaceRiskLevel:
                explicitPhase2QualityGateCandidate.supportSurfaceRiskSummary.riskLevel,
              supportSurfaceRiskFlags:
                explicitPhase2QualityGateCandidate.supportSurfaceRiskSummary.riskFlags.slice(),
              requestedBackend:
                explicitPhase2QualityGateCandidate.requestedBackend ??
                phase2BaseAttempt.requestedBackend,
              resolvedBackend:
                explicitPhase2QualityGateCandidate.backend === 'gpu' ? 'gpu' : 'cpu',
              pipelineMode: explicitPhase2QualityGateCandidate.pipelineMode,
              fallbackReason: explicitPhase2QualityGateCandidate.fallbackReason,
              reconstructionMode: explicitPhase2QualityGateCandidate.reconstructionMode,
            }
          : null;
      if (phase2DegradedPreviewAttempt) {
        displaySelectionBreakdownByAttemptLabel.set(
          phase2DegradedPreviewAttempt.label,
          buildPhase4DisplaySelectionBreakdown({
            qualityGatePassed: phase2DegradedPreviewAttempt.qualityGatePassed,
            qualityBase: phase2DegradedPreviewAttempt.qualityBase,
            qualityScore: phase2DegradedPreviewAttempt.qualityScore,
            detailReward: phase2DegradedPreviewAttempt.detailReward,
            hardRejectReason: phase2DegradedPreviewAttempt.hardRejectReason,
            summary: phase2DegradedPreviewAttempt.summary,
            metrics: phase2DegradedPreviewAttempt.qualityGateMetrics,
            requestedBackend: phase2DegradedPreviewAttempt.requestedBackend,
            resolvedBackend: phase2DegradedPreviewAttempt.resolvedBackend,
            aggregation: phase2DegradedPreviewAttempt.aggregation,
            actualVertHalfMm: phase2DegradedPreviewAttempt.actualVertHalfMm,
            verticalCenterOffsetMm: phase2DegradedPreviewAttempt.verticalCenterOffsetMm,
            slabHalfThicknessMm: phase2DegradedPreviewAttempt.slabHalfThicknessMm,
            slabSamples: phase2DegradedPreviewAttempt.slabSamples,
            duplicateOfLabel: duplicateOfLabelByAttemptLabel.get(phase2DegradedPreviewAttempt.label) ?? null,
          })
        );
      }
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
      const topDisplayRankedGpuAttempt =
        attemptResults
          .filter(
            attempt =>
              attempt.requestedBackend === 'gpu' && attempt.resolvedBackend === 'gpu'
          )
          .slice()
          .sort(compareDisplayRankedAttempts)[0] ?? topGpuAttempt;
      const displayRankedGpuQualityGateCandidate = topDisplayRankedGpuAttempt
        ? buildPhase4QualityGateCandidate({
            attemptLabel: topDisplayRankedGpuAttempt.label,
            displayedPath: 'worker-recon',
            sourceVolumeId,
            summary: topDisplayRankedGpuAttempt.summary,
            qualityBase: topDisplayRankedGpuAttempt.qualityBase,
            qualityScore: topDisplayRankedGpuAttempt.qualityScore,
            hardRejectReason: topDisplayRankedGpuAttempt.hardRejectReason,
            workerDebugPayload: topDisplayRankedGpuAttempt.workerDebugPayload ?? null,
          })
        : null;
      if (
        topGpuAttempt &&
        topDisplayRankedGpuAttempt &&
        topGpuAttempt.label !== topDisplayRankedGpuAttempt.label
      ) {
        console.debug(
          '[CPR-GPU-SELECTION-INCONSISTENCY-JSON]',
          JSON.stringify({
            runId: debugRunId,
            qualityRankedGpuLabel: topGpuAttempt.label,
            displayRankedGpuLabel: topDisplayRankedGpuAttempt.label,
            qualityRankedGpuQualityScore: topGpuAttempt.qualityScore,
            displayRankedGpuQualityScore: topDisplayRankedGpuAttempt.qualityScore,
            qualityRankedGpuDisplayScore:
              displaySelectionBreakdownByAttemptLabel.get(topGpuAttempt.label)?.displaySelectionScore ??
              null,
            displayRankedGpuDisplayScore:
              displaySelectionBreakdownByAttemptLabel.get(topDisplayRankedGpuAttempt.label)
                ?.displaySelectionScore ?? null,
            qualityRankedGpuBlackClip:
              topGpuAttempt.summary?.toothBandBlackClipFraction ??
              topGpuAttempt.qualityGateMetrics?.toothBandBlackClipFraction,
            displayRankedGpuBlackClip:
              topDisplayRankedGpuAttempt.summary?.toothBandBlackClipFraction ??
              topDisplayRankedGpuAttempt.qualityGateMetrics?.toothBandBlackClipFraction,
            qualityRankedGpuHoleFraction:
              topGpuAttempt.summary?.toothBandHoleFraction ??
              topGpuAttempt.qualityGateMetrics?.toothBandHoleFraction,
            displayRankedGpuHoleFraction:
              topDisplayRankedGpuAttempt.summary?.toothBandHoleFraction ??
              topDisplayRankedGpuAttempt.qualityGateMetrics?.toothBandHoleFraction,
            qualityRankedGpuRetainedWeightP50:
              topGpuAttempt.summary?.toothBandRetainedWeightP50 ??
              topGpuAttempt.qualityGateMetrics?.toothBandRetainedWeightP50,
            displayRankedGpuRetainedWeightP50:
              topDisplayRankedGpuAttempt.summary?.toothBandRetainedWeightP50 ??
              topDisplayRankedGpuAttempt.qualityGateMetrics?.toothBandRetainedWeightP50,
          })
        );
      }
      const degradedPreviewAttempts = phase2DegradedPreviewAttempt
        ? [...attemptResults, phase2DegradedPreviewAttempt]
        : attemptResults;
      const degradedPreviewCandidateOptions = degradedPreviewAttempts
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
        option =>
          !option.candidate.pass &&
          !option.degradedPreviewAssessment.catastrophic &&
          !option.degradedPreviewAssessment.blockedAsDegradedPreview
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
      const degradedPreviewOptionByCanonicalLabel = new Map<
        string,
        (typeof degradedPreviewCandidateOptions)[number]
      >();
      for (const option of degradedPreviewCandidateOptions) {
        const canonicalLabel =
          duplicateOfLabelByAttemptLabel.get(option.attempt.label) ?? option.attempt.label;
        const existing = degradedPreviewOptionByCanonicalLabel.get(canonicalLabel);
        if (!existing || compareDisplayRankedAttempts(option.attempt, existing.attempt) < 0) {
          degradedPreviewOptionByCanonicalLabel.set(canonicalLabel, option);
        }
      }
      const degradedPreviewUniqueOptions = Array.from(
        degradedPreviewOptionByCanonicalLabel.values()
      );
      const degradedPreviewRankedOptions = degradedPreviewUniqueOptions
        .slice()
        .sort((a, b) => compareDisplayRankedAttempts(a.attempt, b.attempt));
      const degradedPreviewFallbackChoice =
        !selectedQualityGateCandidate.pass
          ? degradedPreviewRankedDisplayableOptions[0] ?? null
          : null;
      const degradedPreviewEmergencyChoice =
        !selectedQualityGateCandidate.pass && !degradedPreviewFallbackChoice
          ? degradedPreviewRankedOptions[0] ?? null
          : null;
      const degradedPreviewSelectedChoice =
        degradedPreviewFallbackChoice ?? degradedPreviewEmergencyChoice;
      const degradedPreviewCatastrophicOptions = degradedPreviewCandidateOptions.filter(
        option => option.degradedPreviewAssessment.catastrophic
      );
      const degradedPreviewBlockedOptions = degradedPreviewCandidateOptions.filter(
        option => option.degradedPreviewAssessment.blockedAsDegradedPreview
      );
      const degradedPreviewCandidateByAttemptLabel = new Map(
        degradedPreviewCandidateOptions.map(option => [
          option.attempt.label,
          {
            candidate: option.candidate,
            displayable:
              option.candidate.pass ||
              (!option.degradedPreviewAssessment.catastrophic &&
                !option.degradedPreviewAssessment.blockedAsDegradedPreview),
            catastrophicReasons: option.degradedPreviewAssessment.catastrophicReasons,
            blockedAsDegradedPreview:
              option.degradedPreviewAssessment.blockedAsDegradedPreview,
            blockedAsDegradedPreviewReasons:
              option.degradedPreviewAssessment.blockedAsDegradedPreviewReasons,
          },
        ])
      );
      const topDisplayRankedGpuDisplayDiagnostic = topDisplayRankedGpuAttempt
        ? degradedPreviewCandidateByAttemptLabel.get(topDisplayRankedGpuAttempt.label) ?? null
        : null;
      const canForceDisplayTopDisplayRankedGpuCandidate =
        !!forceDisplayGpuCandidateEvenIfRejected &&
        !!topDisplayRankedGpuAttempt &&
        !!displayRankedGpuQualityGateCandidate &&
        !!topDisplayRankedGpuAttempt.voi &&
        topDisplayRankedGpuAttempt.result.pixelData.length ===
          topDisplayRankedGpuAttempt.result.width * topDisplayRankedGpuAttempt.result.height &&
        (displayRankedGpuQualityGateCandidate.pass ||
          !!topDisplayRankedGpuDisplayDiagnostic?.displayable);
      forcedDisplayedAttemptForDebug = canForceDisplayTopDisplayRankedGpuCandidate
        ? topDisplayRankedGpuAttempt
        : null;
      forcedDisplayedQualityGateCandidate =
        forcedDisplayedAttemptForDebug && displayRankedGpuQualityGateCandidate
          ? displayRankedGpuQualityGateCandidate
          : null;
      if (
        forceDisplayGpuCandidateEvenIfRejected &&
        topDisplayRankedGpuAttempt &&
        displayRankedGpuQualityGateCandidate &&
        !forcedDisplayedAttemptForDebug
      ) {
        forcedDisplayedAttemptBlockedReason = topDisplayRankedGpuDisplayDiagnostic?.catastrophicReasons
          ?.length
          ? 'force-display-blocked-catastrophic-gpu-candidate'
          : topDisplayRankedGpuDisplayDiagnostic?.blockedAsDegradedPreview
            ? 'force-display-blocked-structurally-rejected-gpu-candidate'
            : 'force-display-blocked-gpu-candidate-not-displayable';
        forcedDisplayedAttemptBlockedReasons = Array.from(
          new Set([
            ...(topDisplayRankedGpuDisplayDiagnostic?.blockedAsDegradedPreviewReasons ?? []),
            ...(topDisplayRankedGpuDisplayDiagnostic?.catastrophicReasons ?? []),
          ])
        );
        console.warn(
          `[CPR][${debugRunId}] Force-display GPU override was requested but the top display-ranked GPU candidate is blocked from direct display.`,
          {
            requestedLabel: topDisplayRankedGpuAttempt.label,
            blockedReason: forcedDisplayedAttemptBlockedReason,
            blockedReasons: forcedDisplayedAttemptBlockedReasons,
          }
        );
        console.log(
          '[CPR-FORCED-GPU-DISPLAY-BLOCKED-JSON]',
          JSON.stringify({
            runId: debugRunId,
            requestedLabel: topDisplayRankedGpuAttempt.label,
            blockedReason: forcedDisplayedAttemptBlockedReason,
            blockedReasons: forcedDisplayedAttemptBlockedReasons,
            candidateRejectReasons: displayRankedGpuQualityGateCandidate.rejectReasons,
          })
        );
      }
      const buildDisplayRankingLogRow = (
        attempt: Awaited<ReturnType<typeof runWorkerAttempt>>
      ): Record<string, unknown> => {
        const breakdown = displaySelectionBreakdownByAttemptLabel.get(attempt.label);
        const displayDiagnostic = degradedPreviewCandidateByAttemptLabel.get(attempt.label) ?? null;
        const displayable = displayDiagnostic?.displayable ?? attempt.qualityGatePassed;
        const winnerBreakdown = degradedPreviewSelectedChoice
          ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewSelectedChoice.attempt.label) ??
            null
          : null;

        return {
          label: attempt.label,
          displayable,
          rejectReasons: attempt.qualityGateRejectReasons,
          degradedPreviewBlocked:
            displayDiagnostic?.blockedAsDegradedPreview ?? false,
          degradedPreviewBlockedReasons:
            displayDiagnostic?.blockedAsDegradedPreviewReasons ?? [],
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
                balancedMeanRecoveryBonus: breakdown.balancedMeanRecoveryBonus,
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
                  deltaLegacyQualityScore:
                    attempt.qualityScore - degradedPreviewSelectedChoice!.attempt.qualityScore,
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
          gpuDisplayWinner: summarizePhase4QualityGateCandidate(displayRankedGpuQualityGateCandidate),
          cpu: summarizePhase4QualityGateCandidate(cpuQualityGateCandidate),
          legacy: summarizePhase4QualityGateCandidate(legacyQualityGateCandidate),
        },
        degradedPreview: {
          available: !!degradedPreviewSelectedChoice,
          displayableAvailable: !!degradedPreviewFallbackChoice,
          emergencyAvailable: !!degradedPreviewEmergencyChoice,
          legacyPolicyAllowed: CPR_PANO_ALLOW_DEGRADED_FIRST_RUN_DISPLAY,
          selectionMode: degradedPreviewFallbackChoice
            ? 'displayable'
            : degradedPreviewEmergencyChoice
              ? 'emergency'
              : null,
          selectedAttemptLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
          selectedCandidate: summarizePhase4QualityGateCandidate(
            degradedPreviewSelectedChoice?.candidate ?? null
          ),
          selectedRejectReasons: degradedPreviewSelectedChoice?.candidate.rejectReasons ?? [],
          displayableCandidateCount: degradedPreviewUniqueDisplayableOptions.length,
          candidateCount: degradedPreviewUniqueOptions.length,
          duplicateSignatureCollapsedCount:
            degradedPreviewDisplayableOptions.length - degradedPreviewUniqueDisplayableOptions.length,
          selectedDisplaySelectionScore:
            degradedPreviewSelectedChoice &&
            displaySelectionBreakdownByAttemptLabel.has(degradedPreviewSelectedChoice.attempt.label)
              ? displaySelectionBreakdownByAttemptLabel.get(
                  degradedPreviewSelectedChoice.attempt.label
                )?.displaySelectionScore ?? null
              : null,
          blockedCandidateCount: degradedPreviewBlockedOptions.length,
          blockedCandidates: degradedPreviewBlockedOptions.slice(0, 8).map(option => ({
            attemptLabel: option.attempt.label,
            blockedReasons: option.degradedPreviewAssessment.blockedAsDegradedPreviewReasons,
            rejectReasons: option.candidate.rejectReasons,
          })),
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
            secondary: 'worker-cpu-virtual-pano-or-radiograph-or-arch-guided-synthetic',
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
          qualityRankedGpuLabel: topGpuAttempt?.label ?? null,
          displayRankedGpuLabel: topDisplayRankedGpuAttempt?.label ?? null,
          finalSelectedOutput: qualityGateFallbackSummary.finalSelectedOutput,
          candidates: qualityGateFallbackSummary.candidates,
          degradedPreview: qualityGateFallbackSummary.degradedPreview,
          debugOverrides: {
            forceDisplayGpuCandidateEvenIfRejected,
            forceDisplayGpuCandidateActive: !!forcedDisplayedAttemptForDebug,
            forcedDisplayedAttemptLabel: forcedDisplayedAttemptForDebug?.label ?? null,
            forcedDisplayBlockedReason: forcedDisplayedAttemptBlockedReason,
            forcedDisplayBlockedReasons: forcedDisplayedAttemptBlockedReasons,
          },
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
            degradedPreviewAvailable: !!degradedPreviewSelectedChoice,
            degradedPreviewAttemptLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
            degradedPreviewRejectReasons:
              degradedPreviewSelectedChoice?.candidate.rejectReasons ?? [],
            degradedPreviewBlockedCandidateCount: degradedPreviewBlockedOptions.length,
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
          winnerLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
          winnerDisplaySelectionScore: degradedPreviewSelectedChoice
            ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewSelectedChoice.attempt.label)
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
          qualityRankedGpuLabel: topGpuAttempt?.label ?? null,
          displayRankedGpuLabel: topDisplayRankedGpuAttempt?.label ?? null,
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
          degradedDisplayRankingTopLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
          degradedDisplayRankingTopScore: degradedPreviewSelectedChoice
            ? displaySelectionBreakdownByAttemptLabel.get(degradedPreviewSelectedChoice.attempt.label)
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
          rankedAttempts: rankedAttempts.map(attempt => ({
            label: attempt.label,
            checksum: attempt.outputSignature.checksum,
            phase0QualityGatePassed: attempt.qualityGatePassed,
            phase0RejectReasons: attempt.qualityGateRejectReasons,
            supportP50: attempt.supportSurfaceRiskSummary.stability.supportConfidenceP50,
            bgOutlier05: attempt.summary?.backgroundOutlierFraction05 ?? null,
            bgOutlier10: attempt.summary?.backgroundOutlierFraction10 ?? null,
            blackClip: attempt.supportSurfaceRiskSummary.background.blackClipFraction,
            toothBandHoleFraction:
              attempt.supportSurfaceRiskSummary.anatomy.toothBandHoleFraction,
            toothBandBlackClipFraction:
              attempt.supportSurfaceRiskSummary.anatomy.toothBandBlackClipFraction,
            toothBandRetainedWeightP10:
              attempt.supportSurfaceRiskSummary.anatomy.toothBandRetainedWeightP10,
            toothBandRetainedWeightP50:
              attempt.supportSurfaceRiskSummary.anatomy.toothBandRetainedWeightP50,
            lowerBandBrightFraction: attempt.summary?.lowerBandBrightFraction ?? null,
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
            qualityRankedGpuPrimary: attempt.label === topGpuAttempt?.label,
            displayRankedGpuWinner: attempt.label === topDisplayRankedGpuAttempt?.label,
            duplicateOfLabel: duplicateOfLabelByAttemptLabel.get(attempt.label) ?? null,
            phase2SeedSelected: attempt.label === phase2BaseAttempt?.label,
            phase2SeedEligible: phase2BaseAssessmentByLabel.get(attempt.label)?.seedEligible ?? null,
            phase2SeedStructuralRejectReasons:
              phase2BaseAssessmentByLabel.get(attempt.label)?.structuralRejectReasons ?? [],
            phase2DisplayedDerived: attempt.label === phase2BaseAttempt?.label && phase2VirtualPano.executed,
            phase2DisplayedUsedAsOutput:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.usedAsDisplayedOutput
                : null,
            phase2DisplayedWorkerAccepted:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.workerAcceptedForOutput
                : null,
            phase2DisplayedOrchestratorGatePassed:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.orchestratorGatePassed
                : null,
            phase2DisplayedDecisionReason:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.displayDecisionReason
                : null,
            phase2DisplayedRenderFamily:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.phase2RenderFamily
                : null,
            phase2DisplayedCandidateSource:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.phase2CandidateSource
                : null,
            phase2DisplayedDerivedFromPhase0RejectedFamily:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.revivedRejectedFamily
                : null,
            phase2DisplayedRevivedSeedRulePassed:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.revivedRejectedFamilyRulePassed
                : null,
            phase2DisplayedRevivedSeedRejectReasons:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.revivedRejectedFamilyRejectReasons
                : [],
            phase2DisplayedBgOutlier05:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.summary?.backgroundOutlierFraction05 ?? null
                : null,
            phase2DisplayedBgOutlier10:
              attempt.label === phase2BaseAttempt?.label
                ? phase2VirtualPano.summary?.backgroundOutlierFraction10 ?? null
                : null,
            phase2DisplayedBlackClip:
              attempt.label === phase2BaseAttempt?.label
                ? explicitPhase2QualityGateCandidate?.supportSurfaceRiskSummary.background.blackClipFraction ??
                  null
                : null,
            phase2DisplayedToothHole:
              attempt.label === phase2BaseAttempt?.label
                ? explicitPhase2QualityGateCandidate?.metrics.toothBandHoleFraction ?? null
                : null,
            phase2DisplayedToothBlackClip:
              attempt.label === phase2BaseAttempt?.label
                ? explicitPhase2QualityGateCandidate?.metrics.toothBandBlackClipFraction ?? null
                : null,
            catastrophicReasons:
              degradedPreviewCandidateByAttemptLabel.get(attempt.label)?.catastrophicReasons ?? [],
          })),
        })
      );

      if (shouldExportCprAttemptReport()) {
        try {
          const targetNormalizationDiagnosticAttempts: Array<{
            matrixMode:
              | 'current'
              | 'lut-only'
              | 'no-stored-value-normalization'
              | 'raw-stored-values-debug';
            attempt: Awaited<ReturnType<typeof runWorkerAttempt>>;
          }> = [];
          const targetToothDebugBaseAttempt =
            attemptResults.find(attempt => attempt.label === CPR_TARGET_TOOTH_DEBUG_LABEL) ?? null;
          if (
            targetToothDebugBaseAttempt &&
            targetToothDebugBaseAttempt.requestedBackend === 'gpu' &&
            targetToothDebugBaseAttempt.resolvedBackend === 'gpu'
          ) {
            targetNormalizationDiagnosticAttempts.push({
              matrixMode: 'current',
              attempt: targetToothDebugBaseAttempt,
            });
            for (const matrixMode of [
              'lut-only',
              'no-stored-value-normalization',
              'raw-stored-values-debug',
            ] as const) {
              const diagnosticAttempt = await runWorkerAttempt(
                `${CPR_TARGET_TOOTH_DEBUG_LABEL}-diag-${matrixMode}`,
                {
                  ...targetToothDebugBaseAttempt.requestOverrides,
                  modalityLutOverride:
                    matrixMode === 'raw-stored-values-debug'
                      ? false
                      : matrixMode === 'lut-only'
                        ? true
                        : targetToothDebugBaseAttempt.requestOverrides.modalityLutOverride,
                  debugScalarSamplingMode: matrixMode,
                }
              );
              targetNormalizationDiagnosticAttempts.push({
                matrixMode,
                attempt: diagnosticAttempt,
              });
            }
            console.debug(
              '[CPR-TARGET-NORMALIZATION-MATRIX-JSON]',
              JSON.stringify({
                runId: debugRunId,
                targetLabel: CPR_TARGET_TOOTH_DEBUG_LABEL,
                rows: targetNormalizationDiagnosticAttempts.map(entry => ({
                  matrixMode: entry.matrixMode,
                  label:
                    entry.matrixMode === 'current'
                      ? `${entry.attempt.label}-diag-current`
                      : entry.attempt.label,
                  modalityLutApplied: entry.attempt.result.modalityLutApplied,
                  requestedModalityLutApplied: entry.attempt.result.requestedModalityLutApplied,
                  storedValueNormalizationApplied:
                    entry.attempt.result.storedValueNormalizationApplied,
                  supportP50:
                    entry.attempt.supportSurfaceRiskSummary.stability.supportConfidenceP50,
                  blackClipFraction:
                    entry.attempt.supportSurfaceRiskSummary.background.blackClipFraction,
                  toothBandHoleFraction: entry.attempt.summary?.toothBandHoleFraction ?? null,
                  toothBandBlackClipFraction:
                    entry.attempt.summary?.toothBandBlackClipFraction ?? null,
                  toothBandRetainedWeightP10:
                    entry.attempt.summary?.toothBandRetainedWeightP10 ?? null,
                  bg05: entry.attempt.summary?.backgroundOutlierFraction05 ?? null,
                  bg10: entry.attempt.summary?.backgroundOutlierFraction10 ?? null,
                  stageHint: entry.attempt.toothBandStageDiagnostics?.stageHint ?? null,
                  stageEvidence: entry.attempt.toothBandStageDiagnostics?.stageEvidence ?? [],
                })),
              })
            );
          }
          const attemptLookupByLabel = new Map(
            [...attemptResults, ...targetNormalizationDiagnosticAttempts.map(entry => entry.attempt)].map(
              attempt => [attempt.label, attempt] as const
            )
          );
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
                topDisplayRankedGpuAttempt?.label,
                'primary-mean-toothband-narrow',
                'retry-force-lut-mean-narrow-no-normalization',
                'retry-mean-balanced-medium-strong-bias',
                ...targetNormalizationDiagnosticAttempts.map(entry => entry.attempt.label),
                ...CPR_DEBUG_DISPLAY_SELECTION_FOCUS_LABELS,
                'retry-cpu-virtual-pano-fallback',
              ].filter(label => attemptLookupByLabel.has(label))
            )
          );
          const reportDisplayMode = forcedDisplayedAttemptForDebug
            ? 'forced-debug-gpu'
            : phase2VirtualPano.usedAsDisplayedOutput
              ? 'accepted-phase2'
              : canPreservePreviousPanoDisplay
                ? 'preserve-previous-accepted'
                : selectedQualityGateCandidate.pass
                  ? 'accepted-ranked-top'
                  : degradedPreviewFallbackChoice
                    ? 'degraded-preview'
                    : degradedPreviewEmergencyChoice
                      ? 'emergency-debug-preview'
                      : 'blocked';
          const reportDisplayedLabel = forcedDisplayedAttemptForDebug
            ? forcedDisplayedAttemptForDebug.label
            : phase2VirtualPano.usedAsDisplayedOutput
              ? phase2BaseAttempt?.label ?? selectedAttempt.label
              : canPreservePreviousPanoDisplay
                ? preservedPreviousPanoImageId ?? 'previous-accepted-pano'
                : degradedPreviewSelectedChoice?.attempt.label ?? selectedAttempt.label;
          const reportDisplayedAccepted =
            reportDisplayMode === 'preserve-previous-accepted' ||
            (phase2VirtualPano.usedAsDisplayedOutput
              ? explicitPhase2QualityGateCandidate?.pass === true
              : !forcedDisplayedAttemptForDebug && selectedQualityGateCandidate.pass);
          const reportDisplayedRejectReasons = forcedDisplayedAttemptForDebug
            ? forcedDisplayedQualityGateCandidate?.rejectReasons ??
              displayRankedGpuQualityGateCandidate?.rejectReasons ??
              []
            : phase2VirtualPano.usedAsDisplayedOutput
              ? explicitPhase2QualityGateCandidate?.rejectReasons ?? []
              : degradedPreviewSelectedChoice
                ? degradedPreviewSelectedChoice.candidate.rejectReasons
                : selectedQualityGateCandidate.rejectReasons;
          const buildAttemptReportRow = (
            attempt: any,
            options: {
              diagnosticOnly: boolean;
              normalizationMatrixMode: string | null;
              emergencyDebugPreviewDisplayed: boolean;
              labelOverride?: string;
            }
          ) => {
            const reportCandidate = buildPhase4QualityGateCandidate({
              attemptLabel: attempt?.label,
              displayedPath: 'worker-recon',
              sourceVolumeId,
              summary: attempt?.summary,
              qualityBase: attempt?.qualityBase,
              qualityScore: attempt?.qualityScore,
              hardRejectReason: attempt?.hardRejectReason,
              workerDebugPayload: attempt?.workerDebugPayload ?? null,
            });
            const stageDiagnostics = attempt?.toothBandStageDiagnostics ?? null;
            const stageEvidence = Array.isArray(stageDiagnostics?.stageEvidence)
              ? stageDiagnostics.stageEvidence.join('; ')
              : '';
            const backgroundBands =
              reportCandidate.supportSurfaceRiskSummary.background.backgroundBands ?? null;
            const anatomyRisk = reportCandidate.supportSurfaceRiskSummary.anatomy;
            const displaySelectionBreakdown = options.diagnosticOnly
              ? null
              : displaySelectionBreakdownByAttemptLabel.get(attempt?.label);
            const degradedPreviewCandidate = options.diagnosticOnly
              ? null
              : degradedPreviewCandidateByAttemptLabel.get(attempt?.label);
            const phase2Assessment = options.diagnosticOnly
              ? null
              : phase2BaseAssessmentByLabel.get(attempt?.label);
            const isPhase2Seed = !options.diagnosticOnly && attempt?.label === phase2BaseAttempt?.label;

            return {
              label: options.labelOverride ?? attempt?.label ?? '',
              checksum: attempt?.outputSignature?.checksum ?? null,
              diagnosticOnly: options.diagnosticOnly,
              normalizationMatrixMode: options.normalizationMatrixMode,
              requestedScalarSamplingMode: attempt?.debugScalarSamplingMode ?? null,
              modalityLutApplied: attempt?.result?.modalityLutApplied ?? null,
              requestedModalityLutApplied: attempt?.result?.requestedModalityLutApplied ?? null,
              storedValueNormalizationApplied:
                attempt?.result?.storedValueNormalizationApplied ?? null,
              candidateSource: reportCandidate.candidateSource,
              rendererVariant: reportCandidate.metrics.rendererVariant,
              renderSupportMode: reportCandidate.metrics.renderSupportMode,
              metricStage: reportCandidate.metricStage,
              workerBranchSelected: reportCandidate.workerBranchSelected,
              workerQcAccepted: reportCandidate.workerQcAccepted,
              workerQcStage: reportCandidate.workerQcStage,
              rejectReasonsStage: reportCandidate.rejectReasonsStage,
              orchestratorAccepted: reportCandidate.orchestratorAccepted,
              emergencyDebugPreviewDisplayed: options.emergencyDebugPreviewDisplayed,
              phase0QualityGatePassed: attempt?.qualityGatePassed ?? null,
              phase0RejectReasons: attempt?.qualityGateRejectReasons ?? [],
              supportP50: reportCandidate.supportSurfaceRiskSummary.stability.supportConfidenceP50,
              bgOutlier05: attempt?.summary?.backgroundOutlierFraction05 ?? null,
              bgOutlier10: attempt?.summary?.backgroundOutlierFraction10 ?? null,
              bgDominantBand05: backgroundBands?.dominantOutlierBand05 ?? null,
              bgDominantBand10: backgroundBands?.dominantOutlierBand10 ?? null,
              bgTopContribution05: backgroundBands?.top?.backgroundOutlierContribution05 ?? null,
              bgMiddleContribution05:
                backgroundBands?.middle?.backgroundOutlierContribution05 ?? null,
              bgBottomContribution05:
                backgroundBands?.bottom?.backgroundOutlierContribution05 ?? null,
              bgTopContribution10: backgroundBands?.top?.backgroundOutlierContribution10 ?? null,
              bgMiddleContribution10:
                backgroundBands?.middle?.backgroundOutlierContribution10 ?? null,
              bgBottomContribution10:
                backgroundBands?.bottom?.backgroundOutlierContribution10 ?? null,
              blackClip: reportCandidate.supportSurfaceRiskSummary.background.blackClipFraction,
              toothBandHoleFraction: anatomyRisk.toothBandHoleFraction,
              toothBandBlackClipFraction: anatomyRisk.toothBandBlackClipFraction,
              toothBandRetainedWeightP10: anatomyRisk.toothBandRetainedWeightP10,
              toothBandRetainedWeightP50: anatomyRisk.toothBandRetainedWeightP50,
              toothBandRetainedWeightRowP10: attempt?.summary?.toothBandRetainedWeightRowP10 ?? null,
              toothBandRetainedWeightRowP50: attempt?.summary?.toothBandRetainedWeightRowP50 ?? null,
              toothBandRetainedWeightRowP90: attempt?.summary?.toothBandRetainedWeightRowP90 ?? null,
              toothBandRetainedWeightColumnP10:
                attempt?.summary?.toothBandRetainedWeightColumnP10 ?? null,
              toothBandRetainedWeightColumnP50:
                attempt?.summary?.toothBandRetainedWeightColumnP50 ?? null,
              toothBandRetainedWeightColumnP90:
                attempt?.summary?.toothBandRetainedWeightColumnP90 ?? null,
              lowerBandBrightFraction: reportCandidate.metrics.lowerBandBrightFraction,
              stageHint: stageDiagnostics?.stageHint ?? null,
              stageEvidence,
              admissionAccumulationP50: stageDiagnostics?.admissionAccumulationP50 ?? null,
              toneSuppressedAccumulationP50:
                stageDiagnostics?.toneSuppressedAccumulationP50 ?? null,
              toneStageSuppressionP90: stageDiagnostics?.toneStageSuppressionP90 ?? null,
              invalidSupportBlackoutP90: stageDiagnostics?.invalidSupportBlackoutP90 ?? null,
              admissionOnlyHuP50: stageDiagnostics?.admissionOnlyHuP50 ?? null,
              toneBypassHuP50: stageDiagnostics?.toneBypassHuP50 ?? null,
              finalPostToneHuP50: stageDiagnostics?.finalPostToneHuP50 ?? null,
              rejectReasons: reportCandidate.rejectReasons,
              displayable: options.diagnosticOnly
                ? reportCandidate.pass
                : degradedPreviewCandidate?.displayable ?? reportCandidate.pass,
              legacyQualityScore: attempt?.qualityScore ?? null,
              legacyQualityBase: attempt?.qualityBase ?? null,
              displaySelectionScore: options.diagnosticOnly
                ? null
                : displaySelectionBreakdown?.displaySelectionScore ?? null,
              preferredTightRootFamily: options.diagnosticOnly
                ? false
                : displaySelectionBreakdown?.preferredTightRootFamily ?? false,
              effectiveTroughHalfWidthP50Mm: options.diagnosticOnly
                ? null
                : displaySelectionBreakdown?.effectiveTroughHalfWidthP50Mm ?? null,
              participatingSamplesP50: options.diagnosticOnly
                ? null
                : displaySelectionBreakdown?.participatingSamplesP50 ?? null,
              qualityRankedGpuPrimary: !options.diagnosticOnly && attempt?.label === topGpuAttempt?.label,
              displayRankedGpuWinner:
                !options.diagnosticOnly && attempt?.label === topDisplayRankedGpuAttempt?.label,
              duplicateOfLabel: options.diagnosticOnly
                ? null
                : duplicateOfLabelByAttemptLabel.get(attempt?.label) ?? null,
              phase2SeedSelected: isPhase2Seed,
              phase2SeedEligible: options.diagnosticOnly ? null : phase2Assessment?.seedEligible ?? null,
              phase2SeedStructuralRejectReasons: options.diagnosticOnly
                ? ''
                : Array.isArray(phase2Assessment?.structuralRejectReasons)
                ? phase2Assessment.structuralRejectReasons.join(', ')
                : '',
              phase2DisplayedDerived: isPhase2Seed && phase2VirtualPano.executed,
              phase2DisplayedUsedAsOutput: isPhase2Seed
                ? phase2VirtualPano.usedAsDisplayedOutput
                : null,
              phase2DisplayedWorkerAccepted: isPhase2Seed
                ? phase2VirtualPano.workerAcceptedForOutput
                : null,
              phase2DisplayedOrchestratorGatePassed: isPhase2Seed
                ? phase2VirtualPano.orchestratorGatePassed
                : null,
              phase2DisplayedDecisionReason: isPhase2Seed
                ? phase2VirtualPano.displayDecisionReason
                : null,
              phase2DisplayedRenderFamily: isPhase2Seed
                ? phase2VirtualPano.phase2RenderFamily
                : null,
              phase2DisplayedCandidateSource: isPhase2Seed
                ? phase2VirtualPano.phase2CandidateSource
                : null,
              phase2DisplayedDerivedFromPhase0RejectedFamily: isPhase2Seed
                ? phase2VirtualPano.revivedRejectedFamily
                : null,
              phase2DisplayedRevivedSeedRulePassed: isPhase2Seed
                ? phase2VirtualPano.revivedRejectedFamilyRulePassed
                : null,
              phase2DisplayedRevivedSeedRejectReasons: isPhase2Seed
                ? phase2VirtualPano.revivedRejectedFamilyRejectReasons.join(', ')
                : '',
              phase2DisplayedBgOutlier05: isPhase2Seed
                ? phase2VirtualPano.summary?.backgroundOutlierFraction05 ?? null
                : null,
              phase2DisplayedBgOutlier10: isPhase2Seed
                ? phase2VirtualPano.summary?.backgroundOutlierFraction10 ?? null
                : null,
              phase2DisplayedBlackClip: isPhase2Seed
                ? explicitPhase2QualityGateCandidate?.supportSurfaceRiskSummary.background
                    .blackClipFraction ?? null
                : null,
              phase2DisplayedToothHole: isPhase2Seed
                ? explicitPhase2QualityGateCandidate?.metrics.toothBandHoleFraction ?? null
                : null,
              phase2DisplayedToothBlackClip: isPhase2Seed
                ? explicitPhase2QualityGateCandidate?.metrics.toothBandBlackClipFraction ?? null
                : null,
            };
          };
          const rankedAttemptReportRows = rankedAttempts.map(attempt =>
            buildAttemptReportRow(attempt, {
              diagnosticOnly: false,
              normalizationMatrixMode:
                attempt.label === CPR_TARGET_TOOTH_DEBUG_LABEL ? 'current' : null,
              emergencyDebugPreviewDisplayed:
                reportDisplayMode === 'emergency-debug-preview' &&
                reportDisplayedLabel === attempt.label,
            })
          );
          const normalizationDiagnosticRows = targetNormalizationDiagnosticAttempts.map(entry =>
            buildAttemptReportRow(entry.attempt, {
              diagnosticOnly: true,
              normalizationMatrixMode: entry.matrixMode,
              emergencyDebugPreviewDisplayed: false,
              labelOverride:
                entry.matrixMode === 'current'
                  ? `${entry.attempt.label}-diag-current`
                  : entry.attempt.label,
            })
          );
          const reportRows = [...rankedAttemptReportRows, ...normalizationDiagnosticRows];
          const reportImages: Parameters<typeof downloadCprDebugArtifacts>[0]['images'] = [];
          type SupportFormationSummary = {
            columnSupportConfidenceP10?: number | null;
            columnSupportConfidenceP50?: number | null;
            columnSupportConfidenceP90?: number | null;
            columnSupportSpreadP10Mm?: number | null;
            columnSupportSpreadP50Mm?: number | null;
            columnSupportSpreadP90Mm?: number | null;
            columnSupportDensityP10?: number | null;
            columnSupportDensityP50?: number | null;
            columnSupportDensityP90?: number | null;
            rowSupportConfidenceP10?: number | null;
            rowSupportConfidenceP50?: number | null;
            rowSupportConfidenceP90?: number | null;
            rowSupportSpreadP10Mm?: number | null;
            rowSupportSpreadP50Mm?: number | null;
            rowSupportSpreadP90Mm?: number | null;
            rowSupportDensityP10?: number | null;
            rowSupportDensityP50?: number | null;
            rowSupportDensityP90?: number | null;
            dominantDensePeakGateP50?: number | null;
            dominantDensePeakGateP90?: number | null;
            toothBandStructureGuardP50?: number | null;
            toothBandStructureGuardP90?: number | null;
            protectedAmbiguousBroadSupportPenaltyGateP50?: number | null;
            protectedAmbiguousBroadSupportPenaltyGateP90?: number | null;
            supportValidityP10?: number | null;
            supportValidityP50?: number | null;
            supportValidityP90?: number | null;
            rawScoreGapP50?: number | null;
            falseSupportVetoFraction?: number | null;
            rowBackgroundVetoFraction?: number | null;
            supportVetoTriggeredFraction?: number | null;
          };
          type RawSupportFormationSummary = {
            rawScoreGapP10?: number | null;
            rawScoreGapP90?: number | null;
          };
          const readAttemptSupportFormationDiagnostics = (
            workerDebugPayload: CPRWorkerLaunchResult['workerDebugPayload'] | null | undefined
          ): {
            supportFormation: SupportFormationSummary | null;
            rawSupportFormation: RawSupportFormationSummary | null;
          } => {
            const diagnostic = readPhase4DiagnosticRecord(workerDebugPayload?.diagnostic);
            const gpuRender = readPhase4DiagnosticRecord(diagnostic?.gpuRender);

            return {
              supportFormation:
                (readPhase4DiagnosticRecord(gpuRender?.supportFormation) ??
                  readPhase4DiagnosticRecord(diagnostic?.supportFormation)) as SupportFormationSummary | null,
              rawSupportFormation:
                (readPhase4DiagnosticRecord(gpuRender?.rawSupportFormation) ??
                  readPhase4DiagnosticRecord(diagnostic?.rawSupportFormation)) as RawSupportFormationSummary | null,
            };
          };
          const targetPhase2SeedAssessment =
            phase2BaseAssessmentByLabel.get(CPR_TARGET_TOOTH_DEBUG_LABEL) ?? null;
          const reportSummarySections = [
            {
              title: 'Normal Winner',
              lines: [
                `acceptedWinnerLabel=${selectedQualityGateCandidate.pass ? selectedAttempt.label : 'none'}`,
                `rankedTopLabel=${selectedAttempt.label}`,
                `rankedTopAccepted=${selectedQualityGateCandidate.pass ? 'yes' : 'no'}`,
                `rankedTopRejectReasons=${
                  selectedQualityGateCandidate.rejectReasons.join(', ') || 'none'
                }`,
                `selectionReason=${qualityGateSelectionReason}`,
              ],
            },
            {
              title: 'Degraded Preview',
              lines: [
                `winnerLabel=${degradedPreviewSelectedChoice?.attempt.label ?? 'none'}`,
                `selectionMode=${
                  degradedPreviewFallbackChoice
                    ? 'displayable'
                    : degradedPreviewEmergencyChoice
                      ? 'emergency'
                      : 'none'
                }`,
                `winnerRejectReasons=${
                  degradedPreviewSelectedChoice?.candidate.rejectReasons.join(', ') || 'none'
                }`,
                `winnerBlockedAsDegraded=${
                  degradedPreviewSelectedChoice?.degradedPreviewAssessment.blockedAsDegradedPreview
                    ? 'yes'
                    : 'no'
                }`,
                `winnerBlockedReasons=${
                  degradedPreviewSelectedChoice?.degradedPreviewAssessment.blockedAsDegradedPreviewReasons.join(
                    ', '
                  ) || 'none'
                }`,
              ],
            },
            {
              title: 'Displayed Result',
              lines: [
                `displayMode=${reportDisplayMode}`,
                `displayedLabel=${reportDisplayedLabel}`,
                `displayedAccepted=${reportDisplayedAccepted ? 'yes' : 'no'}`,
                `displayedRejectReasons=${reportDisplayedRejectReasons.join(', ') || 'none'}`,
                `forceDisplayRequested=${forceDisplayGpuCandidateEvenIfRejected ? 'yes' : 'no'}`,
                `forceDisplayActive=${forcedDisplayedAttemptForDebug ? 'yes' : 'no'}`,
                `forceDisplayBlockedReason=${forcedDisplayedAttemptBlockedReason ?? 'none'}`,
                `forceDisplayBlockedReasons=${
                  forcedDisplayedAttemptBlockedReasons.join(', ') || 'none'
                }`,
              ],
            },
            {
              title: 'Phase-2 Seed',
              lines: [
                `selectedSeedLabel=${phase2BaseAttempt?.label ?? 'none'}`,
                `selectionBlockedReason=${phase2BaseSelectionBlockedReason ?? 'none'}`,
                `targetSeedLabel=${CPR_TARGET_TOOTH_DEBUG_LABEL}`,
                `targetSeedEligible=${targetPhase2SeedAssessment?.seedEligible ? 'yes' : 'no'}`,
                `targetStructuralRejects=${
                  targetPhase2SeedAssessment?.structuralRejectReasons.join(', ') || 'none'
                }`,
              ],
            },
          ];
          const {
            supportFormation: rankedSupportFormation,
            rawSupportFormation: rankedRawSupportFormation,
          } = readAttemptSupportFormationDiagnostics(topGpuAttempt?.workerDebugPayload ?? null);
          reportSummarySections.push(
            {
              title: 'Support Formation',
              lines: [
                `rankedTopLabel=${topGpuAttempt?.label ?? 'none'}`,
                `columnSupportConfidenceP10/P50/P90=${rankedSupportFormation?.columnSupportConfidenceP10 ?? 'na'}/${rankedSupportFormation?.columnSupportConfidenceP50 ?? 'na'}/${rankedSupportFormation?.columnSupportConfidenceP90 ?? 'na'}`,
                `columnSupportSpreadP10/P50/P90=${rankedSupportFormation?.columnSupportSpreadP10Mm ?? 'na'}/${rankedSupportFormation?.columnSupportSpreadP50Mm ?? 'na'}/${rankedSupportFormation?.columnSupportSpreadP90Mm ?? 'na'}`,
                `columnSupportDensityP10/P50/P90=${rankedSupportFormation?.columnSupportDensityP10 ?? 'na'}/${rankedSupportFormation?.columnSupportDensityP50 ?? 'na'}/${rankedSupportFormation?.columnSupportDensityP90 ?? 'na'}`,
                `rowSupportConfidenceP10/P50/P90=${rankedSupportFormation?.rowSupportConfidenceP10 ?? 'na'}/${rankedSupportFormation?.rowSupportConfidenceP50 ?? 'na'}/${rankedSupportFormation?.rowSupportConfidenceP90 ?? 'na'}`,
                `rowSupportSpreadP10/P50/P90=${rankedSupportFormation?.rowSupportSpreadP10Mm ?? 'na'}/${rankedSupportFormation?.rowSupportSpreadP50Mm ?? 'na'}/${rankedSupportFormation?.rowSupportSpreadP90Mm ?? 'na'}`,
                `rowSupportDensityP10/P50/P90=${rankedSupportFormation?.rowSupportDensityP10 ?? 'na'}/${rankedSupportFormation?.rowSupportDensityP50 ?? 'na'}/${rankedSupportFormation?.rowSupportDensityP90 ?? 'na'}`,
              ],
            },
            {
              title: 'Support Gates',
              lines: [
                `dominantDensePeakGateP50/P90=${rankedSupportFormation?.dominantDensePeakGateP50 ?? 'na'}/${rankedSupportFormation?.dominantDensePeakGateP90 ?? 'na'}`,
                `toothBandStructureGuardP50/P90=${rankedSupportFormation?.toothBandStructureGuardP50 ?? 'na'}/${rankedSupportFormation?.toothBandStructureGuardP90 ?? 'na'}`,
                `protectedAmbiguousPenaltyP50/P90=${rankedSupportFormation?.protectedAmbiguousBroadSupportPenaltyGateP50 ?? 'na'}/${rankedSupportFormation?.protectedAmbiguousBroadSupportPenaltyGateP90 ?? 'na'}`,
                `supportValidityP10/P50/P90=${rankedSupportFormation?.supportValidityP10 ?? 'na'}/${rankedSupportFormation?.supportValidityP50 ?? 'na'}/${rankedSupportFormation?.supportValidityP90 ?? 'na'}`,
                `rawScoreGapP10/P50/P90=${rankedRawSupportFormation?.rawScoreGapP10 ?? 'na'}/${rankedSupportFormation?.rawScoreGapP50 ?? 'na'}/${rankedRawSupportFormation?.rawScoreGapP90 ?? 'na'}`,
                `falseSupportVetoFraction=${rankedSupportFormation?.falseSupportVetoFraction ?? 'na'}`,
                `rowBackgroundVetoFraction=${rankedSupportFormation?.rowBackgroundVetoFraction ?? 'na'}`,
                `supportVetoTriggeredFraction=${rankedSupportFormation?.supportVetoTriggeredFraction ?? 'na'}`,
              ],
            }
          );
          const formatCandidateMetric = (
            value: number | null | undefined,
            digits = 3
          ): string => formatReadablePanoValue(value, digits);
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
          const buildRowMeanPlotImage = (
            pixels: Float32Array | Uint16Array,
            width: number,
            height: number
          ): { pixels: Float32Array; width: number; height: number } | null => {
            if (!pixels?.length || width <= 0 || height <= 0) {
              return null;
            }
            const rowMeans = new Float32Array(height);
            let minMean = Number.POSITIVE_INFINITY;
            let maxMean = Number.NEGATIVE_INFINITY;
            for (let row = 0; row < height; row++) {
              let rowSum = 0;
              let rowCount = 0;
              for (let col = 0; col < width; col++) {
                const value = Number(pixels[row * width + col]);
                if (!Number.isFinite(value)) {
                  continue;
                }
                rowSum += value;
                rowCount++;
              }
              const rowMean = rowCount > 0 ? rowSum / rowCount : 0;
              rowMeans[row] = rowMean;
              minMean = Math.min(minMean, rowMean);
              maxMean = Math.max(maxMean, rowMean);
            }
            const plotHeight = 96;
            const plotWidth = Math.max(height, 8);
            const plotPixels = new Float32Array(plotWidth * plotHeight);
            const span = Math.max(1e-6, maxMean - minMean);
            let previousY = plotHeight - 1;
            for (let row = 0; row < height; row++) {
              const x = Math.min(plotWidth - 1, row);
              const normalized = Math.max(
                0,
                Math.min(1, (rowMeans[row] - minMean) / span)
              );
              const y = Math.max(
                0,
                Math.min(plotHeight - 1, plotHeight - 1 - Math.round(normalized * (plotHeight - 1)))
              );
              const startY = Math.min(previousY, y);
              const endY = Math.max(previousY, y);
              for (let lineY = startY; lineY <= endY; lineY++) {
                plotPixels[lineY * plotWidth + x] = 1;
              }
              plotPixels[(plotHeight - 1) * plotWidth + x] = Math.max(
                plotPixels[(plotHeight - 1) * plotWidth + x],
                0.18
              );
              previousY = y;
            }
            return {
              pixels: plotPixels,
              width: plotWidth,
              height: plotHeight,
            };
          };
          const appendRowMeanPlotImage = (
            title: string,
            filenameStem: string,
            pixels: Float32Array | Uint16Array,
            width: number,
            height: number,
            caption?: string | null
          ) => {
            const plot = buildRowMeanPlotImage(pixels, width, height);
            if (!plot) {
              return;
            }
            appendReportImage(title, filenameStem, plot.pixels, plot.width, plot.height, 'unit', 0, 1, caption);
          };
          const buildCandidateReportCaption = (
            attempt: Awaited<ReturnType<typeof runWorkerAttempt>> | null,
            candidate: Phase4QualityGateCandidate | null
          ): string => {
            const backgroundBands = candidate?.supportSurfaceRiskSummary.background.backgroundBands;
            const attemptSupportFormation = readAttemptSupportFormationDiagnostics(
              attempt?.workerDebugPayload ?? null
            ).supportFormation;
            const localizedReadout = candidate?.localizedReadout;
            const formatBackgroundBandLine = (
              thresholdLabel: '05' | '10'
            ): string => {
              if (!backgroundBands) {
                return `bg${thresholdLabel}Bands=na`;
              }
              const contributionKey =
                thresholdLabel === '05'
                  ? 'backgroundOutlierContribution05'
                  : 'backgroundOutlierContribution10';
              const dominantKey =
                thresholdLabel === '05'
                  ? backgroundBands.dominantOutlierBand05
                  : backgroundBands.dominantOutlierBand10;
              return [
                `bg${thresholdLabel}Bands`,
                `top=${formatCandidateMetric(backgroundBands.top[contributionKey] * 100, 1)}%`,
                `mid=${formatCandidateMetric(backgroundBands.middle[contributionKey] * 100, 1)}%`,
                `bot=${formatCandidateMetric(backgroundBands.bottom[contributionKey] * 100, 1)}%`,
                `dom=${dominantKey}`,
              ].join(' ');
            };
            return [
              `label=${attempt?.label ?? 'none'}`,
              `pass=${candidate?.pass ? 'yes' : 'no'}`,
              `candidateSource=${candidate?.candidateSource ?? 'unknown'}`,
              `backend=${candidate?.backend ?? attempt?.resolvedBackend ?? 'unknown'}`,
              `requestedBackend=${candidate?.requestedBackend ?? attempt?.requestedBackend ?? 'unknown'}`,
              `pipelineMode=${candidate?.pipelineMode ?? attempt?.pipelineMode ?? 'unknown'}`,
              `reconstructionMode=${candidate?.reconstructionMode ?? attempt?.reconstructionMode ?? 'unknown'}`,
              `rendererVariant=${candidate?.metrics.rendererVariant ?? 'unknown'}`,
              `renderSupportMode=${candidate?.metrics.renderSupportMode ?? 'unknown'}`,
              `fallbackReason=${candidate?.fallbackReason ?? attempt?.fallbackReason ?? 'none'}`,
              `displayedPath=${candidate?.displayedPath ?? 'worker-recon'}`,
              `workerBranchSelected=${candidate?.workerBranchSelected === null ? 'na' : candidate?.workerBranchSelected ? 'yes' : 'no'}`,
              `workerQcAccepted=${candidate?.workerQcAccepted === null ? 'na' : candidate?.workerQcAccepted ? 'yes' : 'no'}`,
              `workerQcStage=${candidate?.workerQcStage ?? 'na'}`,
              `metricStage=${candidate?.metricStage ?? 'na'}`,
              `rejectStage=${candidate?.rejectReasonsStage ?? 'na'}`,
              `orchestratorAccepted=${candidate?.orchestratorAccepted ? 'yes' : 'no'}`,
              `qualityBase=${formatCandidateMetric(candidate?.qualityBase ?? attempt?.qualityBase, 2)}`,
              `qualityScore=${formatCandidateMetric(candidate?.qualityScore ?? attempt?.qualityScore, 2)}`,
              `supportP50=${formatCandidateMetric(candidate?.metrics.supportConfidenceP50, 3)}`,
              `supportPathP50=${formatCandidateMetric(candidate?.metrics.supportPathConfidenceP50, 3)}`,
              `supportAmbiguousFrac=${formatCandidateMetric(
                candidate?.metrics.supportAmbiguousColumnFraction,
                3
              )}`,
              `supportScoreGapP50=${formatCandidateMetric(
                candidate?.metrics.supportScoreGapP50,
                3
              )}`,
              `detailHuUpperP50=${formatCandidateMetric(candidate?.metrics.upperDetailHuP50, 1)}`,
              `detailHuLowerP50=${formatCandidateMetric(candidate?.metrics.lowerDetailHuP50, 1)}`,
              `detailSampleFraction=${formatCandidateMetric(candidate?.metrics.detailSampleFractionMean, 3)}`,
              `contextBlendMean=${formatCandidateMetric(candidate?.metrics.contextBlendMean, 3)}`,
              `contextWeightFractionMean=${formatCandidateMetric(
                candidate?.metrics.contextWeightFractionMean,
                3
              )}`,
              `shadowLift=${formatCandidateMetric(candidate?.metrics.shadowLiftMean, 3)}`,
              `supportValidityP50=${formatCandidateMetric(
                attemptSupportFormation?.supportValidityP50,
                3
              )}`,
              `dominantDensePeakGateP50=${formatCandidateMetric(
                attemptSupportFormation?.dominantDensePeakGateP50,
                3
              )}`,
              `protectedAmbiguousPenaltyP50=${formatCandidateMetric(
                attemptSupportFormation?.protectedAmbiguousBroadSupportPenaltyGateP50,
                3
              )}`,
              `rawScoreGapP50=${formatCandidateMetric(
                attemptSupportFormation?.rawScoreGapP50,
                3
              )}`,
              `bg05=${formatCandidateMetric(attempt?.summary?.backgroundOutlierFraction05, 3)}`,
              `bg10=${formatCandidateMetric(attempt?.summary?.backgroundOutlierFraction10, 3)}`,
              formatBackgroundBandLine('05'),
              formatBackgroundBandLine('10'),
              `blackClip=${formatCandidateMetric(
                candidate?.supportSurfaceRiskSummary.background.blackClipFraction,
                3
              )}`,
              `toothHole=${formatCandidateMetric(candidate?.metrics.toothBandHoleFraction, 3)}`,
              `toothBlackClip=${formatCandidateMetric(
                candidate?.metrics.toothBandBlackClipFraction,
                3
              )}`,
              `toothRetainedP10=${formatCandidateMetric(
                candidate?.metrics.toothBandRetainedWeightP10,
                3
              )}`,
              `toothRetainedP50=${formatCandidateMetric(
                candidate?.metrics.toothBandRetainedWeightP50,
                3
              )}`,
              `scalarMode=${attempt?.debugScalarSamplingMode ?? 'current'}`,
              `stageHint=${
                candidate?.candidateSource === 'worker-cpu-virtual-panoramic-radiograph'
                  ? candidate?.metricStage ?? 'na'
                  : attempt?.toothBandStageDiagnostics?.stageHint ?? 'na'
              }`,
              `lowerBandBright=${formatCandidateMetric(candidate?.metrics.lowerBandBrightFraction, 3)}`,
              `lowerBandP50=${formatCandidateMetric(candidate?.metrics.lowerBandP50, 1)}`,
              `toothContrast=${formatCandidateMetric(candidate?.metrics.toothBandContrastRange, 1)}`,
              `focalSharpnessCenterThirdP50=${formatCandidateMetric(
                candidate?.metrics.focalSharpnessCenterThirdP50,
                3
              )}`,
              `interToothValleyContrast=${formatCandidateMetric(
                candidate?.metrics.interToothValleyContrast,
                1
              )}`,
              `intraToothGradation=${formatCandidateMetric(
                candidate?.metrics.intraToothGradationScore,
                1
              )}`,
              `crownHighlightSaturation=${formatCandidateMetric(
                candidate?.metrics.crownHighlightSaturationFraction,
                3
              )}`,
              `occlusalDarkCap=${formatCandidateMetric(
                candidate?.metrics.occlusalDarkCapFraction,
                3
              )}`,
              `offTroughEnergy[top,mid,bot]=${[
                formatCandidateMetric(candidate?.metrics.offTroughEnergyTopRatio, 3),
                formatCandidateMetric(candidate?.metrics.offTroughEnergyMiddleRatio, 3),
                formatCandidateMetric(candidate?.metrics.offTroughEnergyBottomRatio, 3),
              ].join(',')}`,
              `lowerField[spec,streak]=${[
                formatCandidateMetric(candidate?.metrics.lowerFieldSpeckleFraction, 3),
                formatCandidateMetric(candidate?.metrics.lowerFieldVerticalStreakScore, 3),
              ].join(',')}`,
              `underRootSmear=${formatCandidateMetric(candidate?.metrics.underRootVerticalSmearScore, 3)}`,
              `hypothesisSwitch=${formatCandidateMetric(candidate?.metrics.hypothesisSwitchFraction, 3)}`,
              `finalDisplayClipping[low,high,hist]=${[
                formatCandidateMetric(candidate?.metrics.finalDisplayLowClipFraction, 3),
                formatCandidateMetric(candidate?.metrics.finalDisplayHighClipFraction, 3),
                formatCandidateMetric(candidate?.metrics.finalDisplayHistogramOccupancy, 3),
              ].join(',')}`,
              `panelLeak[rawZeroBright,outsideP95,gapP95,boundaryJump]=${[
                formatCandidateMetric(candidate?.metrics.rawZeroButFinalBrightFraction, 3),
                formatCandidateMetric(candidate?.metrics.outsideRowsBrightnessP95, 3),
                formatCandidateMetric(candidate?.metrics.gapRowsBrightnessP95, 3),
                formatCandidateMetric(candidate?.metrics.bandBoundaryJumpMean, 3),
              ].join(',')}`,
              localizedReadout
                ? `mapReadout.centerTeeth sharp=${formatCandidateMetric(
                    localizedReadout.centerTeeth.focalSharpnessP50,
                    3
                  )} raw=${formatCandidateMetric(
                    localizedReadout.centerTeeth.rawProjectedAttenuationP50,
                    3
                  )} post=${formatCandidateMetric(
                    localizedReadout.centerTeeth.postSuppressionP50,
                    3
                  )} preNorm=${formatCandidateMetric(
                    localizedReadout.centerTeeth.preNormalizeCompositeP50,
                    3
                  )} postNorm=${formatCandidateMetric(
                    localizedReadout.centerTeeth.postNormalizeDisplayP50,
                    3
                  )} final=${formatCandidateMetric(
                    localizedReadout.centerTeeth.finalDisplayP50,
                    1
                  )} bgPresent=${formatCandidateMetric(
                    localizedReadout.centerTeeth.backgroundPresentationP50,
                    3
                  )} signal=${formatCandidateMetric(
                    localizedReadout.centerTeeth.hasSignalFraction,
                    3
                  )} context=${formatCandidateMetric(
                    localizedReadout.centerTeeth.contextContributionP50,
                    3
                  )} fill=${formatCandidateMetric(
                    localizedReadout.centerTeeth.backgroundFillContributionP50,
                    3
                  )} switch=${formatCandidateMetric(
                    localizedReadout.centerTeeth.hypothesisSwitchFraction,
                    3
                  )}`
                : null,
              localizedReadout
                ? `mapReadout.upperCloudBand suppress=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.outOfTroughSuppressionP50,
                    3
                  )} raw=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.rawProjectedAttenuationP50,
                    3
                  )} post=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.postSuppressionP50,
                    3
                  )} preNorm=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.preNormalizeCompositeP50,
                    3
                  )} postNorm=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.postNormalizeDisplayP50,
                    3
                  )} final=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.finalDisplayP50,
                    1
                  )} bgPresent=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.backgroundPresentationP50,
                    3
                  )} signal=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.hasSignalFraction,
                    3
                  )} context=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.contextContributionP50,
                    3
                  )} fill=${formatCandidateMetric(
                    localizedReadout.upperCloudBand.backgroundFillContributionP50,
                    3
                  )}`
                : null,
              localizedReadout
                ? `mapReadout.lowerGranularField suppress=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.outOfTroughSuppressionP50,
                    3
                  )} raw=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.rawProjectedAttenuationP50,
                    3
                  )} post=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.postSuppressionP50,
                    3
                  )} preNorm=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.preNormalizeCompositeP50,
                    3
                  )} postNorm=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.postNormalizeDisplayP50,
                    3
                  )} final=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.finalDisplayP50,
                    1
                  )} bgPresent=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.backgroundPresentationP50,
                    3
                  )} signal=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.hasSignalFraction,
                    3
                  )} context=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.contextContributionP50,
                    3
                  )} fill=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.backgroundFillContributionP50,
                    3
                  )} speckle=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.speckleFraction,
                    3
                  )} streak=${formatCandidateMetric(
                    localizedReadout.lowerGranularField.verticalStreakScore,
                    3
                  )}`
                : null,
              `borderline=${candidate?.borderlineAcceptedReason ?? 'none'}`,
              `hardReject=${candidate?.hardRejectReason ?? attempt?.hardRejectReason ?? 'none'}`,
              `rejects=${candidate?.rejectReasons.join(', ') || 'none'}`,
              `workerRejects=${candidate?.workerRejectReasons.join(', ') || 'none'}`,
              `risk=${candidate?.supportSurfaceRiskSummary.riskLevel ?? 'unknown'}`,
              `riskFlags=${candidate?.supportSurfaceRiskSummary.riskFlags.join(', ') || 'none'}`,
              `baseline=${candidate?.supportSurfaceRiskSummary.baselineFingerprint ?? 'none'}`,
            ]
              .filter((line): line is string => !!line)
              .join('\n');
          };
          const comparisonAttemptSpecs = [
            {
              prefix: 'gpu-primary-candidate',
              title: `GPU Primary Candidate: ${topGpuAttempt?.label ?? 'none'}`,
              attempt: topGpuAttempt,
              candidate: gpuQualityGateCandidate,
            },
            ...(topDisplayRankedGpuAttempt &&
            topDisplayRankedGpuAttempt.label !== topGpuAttempt?.label
              ? [
                  {
                    prefix: 'gpu-display-ranked-winner',
                    title: `Display-ranked GPU Winner: ${topDisplayRankedGpuAttempt.label}`,
                    attempt: topDisplayRankedGpuAttempt,
                    candidate: displayRankedGpuQualityGateCandidate,
                  },
                ]
              : []),
            {
              prefix: 'cpu-fallback-candidate',
              title: `CPU Fallback Candidate: ${topCpuVirtualPanoAttempt?.label ?? 'none'}`,
              attempt: topCpuVirtualPanoAttempt,
              candidate: cpuQualityGateCandidate,
            },
          ];
          for (const spec of comparisonAttemptSpecs) {
            if (!spec.attempt) {
              continue;
            }
            appendReportImage(
              spec.title,
              `${spec.prefix}-${spec.attempt.label}`,
              spec.attempt.result.pixelData,
              spec.attempt.result.width,
              spec.attempt.result.height,
              'hu',
              spec.attempt.voi?.lower ?? spec.attempt.result.minValue,
              spec.attempt.voi?.upper ?? spec.attempt.result.maxValue,
              buildCandidateReportCaption(spec.attempt, spec.candidate)
            );
          }
          if (phase2WorkerResult && phase2BaseAttempt && phase2VirtualPano.summary && phase2VirtualPano.voi) {
            appendReportImage(
              `Phase-2 Displayed Candidate: ${phase2BaseAttempt.label}`,
              `phase2-displayed-${phase2BaseAttempt.label}`,
              phase2WorkerResult.pixelData,
              phase2WorkerResult.width,
              phase2WorkerResult.height,
              'hu',
              phase2VirtualPano.voi.lower ?? phase2WorkerResult.minValue,
              phase2VirtualPano.voi.upper ?? phase2WorkerResult.maxValue,
              [
                `phase0SeedLabel=${phase2BaseAttempt.label}`,
                `phase0SeedPass=${phase2BaseAttempt.qualityGatePassed ? 'yes' : 'no'}`,
                `phase0SeedRejects=${phase2BaseAttempt.qualityGateRejectReasons.join(', ') || 'none'}`,
                `phase0SeedStructuralRejects=${
                  phase2BaseAssessment?.structuralRejectReasons.join(', ') || 'none'
                }`,
                `phase2WorkerAccepted=${phase2VirtualPano.workerAcceptedForOutput ? 'yes' : 'no'}`,
                `phase2OrchestratorGate=${
                  phase2VirtualPano.orchestratorGatePassed === null
                    ? 'na'
                    : phase2VirtualPano.orchestratorGatePassed
                      ? 'pass'
                      : 'fail'
                }`,
                `phase2UsedAsOutput=${phase2VirtualPano.usedAsDisplayedOutput ? 'yes' : 'no'}`,
                `phase2Decision=${phase2VirtualPano.displayDecisionReason ?? 'none'}`,
                `phase2RenderFamily=${phase2VirtualPano.phase2RenderFamily ?? 'none'}`,
                `phase2CandidateSource=${phase2VirtualPano.phase2CandidateSource ?? 'none'}`,
                `derivedFromPhase0RejectedFamily=${
                  phase2VirtualPano.revivedRejectedFamily ? 'yes' : 'no'
                }`,
                `rejectedSeedRulePassed=${
                  phase2VirtualPano.revivedRejectedFamilyRulePassed === null
                    ? 'na'
                    : phase2VirtualPano.revivedRejectedFamilyRulePassed
                      ? 'yes'
                      : 'no'
                }`,
                `rejectedSeedRuleRejects=${
                  phase2VirtualPano.revivedRejectedFamilyRejectReasons.join(', ') || 'none'
                }`,
                `bg05=${formatCandidateMetric(phase2VirtualPano.summary.backgroundOutlierFraction05, 3)}`,
                `bg10=${formatCandidateMetric(phase2VirtualPano.summary.backgroundOutlierFraction10, 3)}`,
                `blackClip=${formatCandidateMetric(
                  explicitPhase2QualityGateCandidate?.supportSurfaceRiskSummary.background
                    .blackClipFraction,
                  3
                )}`,
                `toothHole=${formatCandidateMetric(
                  explicitPhase2QualityGateCandidate?.metrics.toothBandHoleFraction,
                  3
                )}`,
                `toothBlackClip=${formatCandidateMetric(
                  explicitPhase2QualityGateCandidate?.metrics.toothBandBlackClipFraction,
                  3
                )}`,
              ].join('\n')
            );
          }
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
                `toothHole=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.anatomy.toothBandHoleFraction,
                  3
                )}`,
                `toothBlackClip=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.anatomy.toothBandBlackClipFraction,
                  3
                )}`,
                `toothRetainedP10=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.anatomy.toothBandRetainedWeightP10,
                  3
                )}`,
                `toothRetainedP50=${formatReadablePanoValue(
                  attempt.supportSurfaceRiskSummary.anatomy.toothBandRetainedWeightP50,
                  3
                )}`,
                `scalarMode=${attempt.debugScalarSamplingMode}`,
                `stageHint=${
                  reportRows.find(row => row.label === label)?.metricStage ??
                  attempt.toothBandStageDiagnostics?.stageHint ??
                  'na'
                }`,
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
                attempt.label === topGpuAttempt?.label ? 'qualityRankedGpuPrimary=yes' : null,
                attempt.label === topDisplayRankedGpuAttempt?.label
                  ? 'displayRankedGpuWinner=yes'
                  : null,
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
              prefix: 'quality-ranked-gpu',
              title: `Quality-ranked GPU Primary Candidate: ${topGpuAttempt?.label ?? 'none'}`,
              attempt: topGpuAttempt,
              candidate: gpuQualityGateCandidate,
            },
            ...(topDisplayRankedGpuAttempt &&
            topDisplayRankedGpuAttempt.label !== topGpuAttempt?.label
              ? [
                  {
                    prefix: 'display-ranked-gpu',
                    title: `Display-ranked GPU Winner: ${topDisplayRankedGpuAttempt.label}`,
                    attempt: topDisplayRankedGpuAttempt,
                    candidate: displayRankedGpuQualityGateCandidate,
                  },
                ]
              : []),
            {
              prefix: 'cpu-fallback',
              title: `CPU Virtual Pano Fallback: ${topCpuVirtualPanoAttempt?.label ?? 'none'}`,
              attempt: topCpuVirtualPanoAttempt,
              candidate: cpuQualityGateCandidate,
            },
            ...targetNormalizationDiagnosticAttempts.map(entry => ({
              prefix: `target-normalization-${entry.matrixMode}`,
              title: `Target Family Normalization ${entry.matrixMode}: ${entry.attempt.label}`,
              attempt: entry.attempt,
              candidate: buildPhase4QualityGateCandidate({
                attemptLabel: entry.attempt.label,
                displayedPath: 'worker-recon',
                sourceVolumeId,
                summary: entry.attempt.summary,
                qualityBase: entry.attempt.qualityBase,
                qualityScore: entry.attempt.qualityScore,
                hardRejectReason: entry.attempt.hardRejectReason,
                workerDebugPayload: entry.attempt.workerDebugPayload ?? null,
              }),
            })),
          ];
          const sidecarMapSpecs: Array<{
            key: keyof NonNullable<PanoImagePayload['debugMaps']>;
            mode: 'hu' | 'unit' | 'signed' | 'positive';
            title: string;
          }> = [
            { key: 'renderBranchMap', mode: 'positive', title: 'renderBranchMap' },
            {
              key: 'selectedSupportHypothesisMap',
              mode: 'positive',
              title: 'selectedSupportHypothesisMap',
            },
            { key: 'focalTroughSharpnessMap', mode: 'unit', title: 'focalTroughSharpnessMap' },
            {
              key: 'outOfTroughSuppressionMap',
              mode: 'unit',
              title: 'outOfTroughSuppressionMap',
            },
            {
              key: 'rawProjectedAttenuationMap',
              mode: 'positive',
              title: 'rawProjectedAttenuationMap',
            },
            {
              key: 'upperProjectedAttenuationMap',
              mode: 'positive',
              title: 'upperProjectedAttenuationMap.png',
            },
            {
              key: 'lowerProjectedAttenuationMap',
              mode: 'positive',
              title: 'lowerProjectedAttenuationMap.png',
            },
            { key: 'upperBandRawOdMap', mode: 'positive', title: 'upperBandRawOD.png' },
            { key: 'lowerBandRawOdMap', mode: 'positive', title: 'lowerBandRawOD.png' },
            { key: 'gapBandRawOdMap', mode: 'positive', title: 'gapBandRawOD.png' },
            { key: 'outsideRowsRawOdMap', mode: 'positive', title: 'outsideRowsRawOD.png' },
            {
              key: 'displayBackgroundOdMap',
              mode: 'positive',
              title: 'displayBackgroundOD.png',
            },
            { key: 'bandMaskUpperMap', mode: 'unit', title: 'bandMaskUpper.png' },
            { key: 'bandMaskGapMap', mode: 'unit', title: 'bandMaskGap.png' },
            { key: 'bandMaskLowerMap', mode: 'unit', title: 'bandMaskLower.png' },
            { key: 'bandMaskOutsideMap', mode: 'unit', title: 'bandMaskOutside.png' },
            {
              key: 'normalizationEligibleMaskMap',
              mode: 'unit',
              title: 'normalizationEligibleMask.png',
            },
            {
              key: 'preNormalizeCompositeOdMap',
              mode: 'positive',
              title: 'preToneCompositeOD.png',
            },
            {
              key: 'postNormalizeDisplayMap',
              mode: 'unit',
              title: 'postNormalizeDisplay.png',
            },
            {
              key: 'displayAnatomyMap',
              mode: 'unit',
              title: 'displayAnatomy.png',
            },
            {
              key: 'backgroundPresentationMap',
              mode: 'unit',
              title: 'backgroundPresentation.png',
            },
            {
              key: 'finalCompositeDisplayMap',
              mode: 'unit',
              title: 'finalDisplay.png',
            },
            {
              key: 'contextContributionMap',
              mode: 'positive',
              title: 'contextContributionMap',
            },
            {
              key: 'backgroundFillContributionMap',
              mode: 'positive',
              title: 'backgroundFillContributionMap',
            },
            { key: 'finalDisplayImageMap', mode: 'hu', title: 'finalDisplayImageMap' },
            { key: 'panoV2FusionImageMap', mode: 'hu', title: 'panoV2FusionImageMap' },
            { key: 'panoV2UpperLayer1Map', mode: 'hu', title: 'panoV2UpperLayer1Map' },
            { key: 'panoV2UpperLayer2Map', mode: 'hu', title: 'panoV2UpperLayer2Map' },
            { key: 'panoV2UpperLayer3Map', mode: 'hu', title: 'panoV2UpperLayer3Map' },
            { key: 'panoV2LowerLayer1Map', mode: 'hu', title: 'panoV2LowerLayer1Map' },
            { key: 'panoV2LowerLayer2Map', mode: 'hu', title: 'panoV2LowerLayer2Map' },
            { key: 'panoV2LowerLayer3Map', mode: 'hu', title: 'panoV2LowerLayer3Map' },
            { key: 'rawSupportPeakDominanceMap', mode: 'unit', title: 'rawSupportPeakDominanceMap' },
            { key: 'rawSupportPeakValidityMap', mode: 'unit', title: 'rawSupportPeakValidityMap' },
            { key: 'rawSupportSecondPeakRatioMap', mode: 'unit', title: 'rawSupportSecondPeakRatioMap' },
            { key: 'rawSupportPeakAmbiguityMap', mode: 'unit', title: 'rawSupportPeakAmbiguityMap' },
            { key: 'rawSupportScoreGapMap', mode: 'unit', title: 'rawSupportScoreGapMap' },
            { key: 'rawSupportDenseFractionMap', mode: 'unit', title: 'rawSupportDenseFractionMap' },
            {
              key: 'rawSupportPeakHuSupportGateMap',
              mode: 'unit',
              title: 'rawSupportPeakHuSupportGateMap',
            },
            {
              key: 'rawSupportDominantPeakOffsetMap',
              mode: 'signed',
              title: 'rawSupportDominantPeakOffsetMap',
            },
            {
              key: 'rawSupportSecondaryPeakOffsetMap',
              mode: 'signed',
              title: 'rawSupportSecondaryPeakOffsetMap',
            },
            { key: 'toothBandPriorMap', mode: 'unit', title: 'toothBandPriorMap' },
            {
              key: 'dominantDensePeakGateMap',
              mode: 'unit',
              title: 'dominantDensePeakGateMap',
            },
            {
              key: 'toothBandStructureGuardMap',
              mode: 'unit',
              title: 'toothBandStructureGuardMap',
            },
            {
              key: 'ambiguousBroadSupportPenaltyGateMap',
              mode: 'unit',
              title: 'ambiguousBroadSupportPenaltyGateMap',
            },
            {
              key: 'protectedAmbiguousBroadSupportPenaltyGateMap',
              mode: 'unit',
              title: 'protectedAmbiguousBroadSupportPenaltyGateMap',
            },
            { key: 'structuralSupportGateMap', mode: 'unit', title: 'structuralSupportGateMap' },
            { key: 'peakStructureValidityMap', mode: 'unit', title: 'peakStructureValidityMap' },
            { key: 'supportValidityMap', mode: 'unit', title: 'supportValidityMap' },
            { key: 'rowConfidenceGateMap', mode: 'unit', title: 'rowConfidenceGateMap' },
            {
              key: 'falseSupportConfidenceGateMap',
              mode: 'unit',
              title: 'falseSupportConfidenceGateMap',
            },
            {
              key: 'falseSupportDensityGateMap',
              mode: 'unit',
              title: 'falseSupportDensityGateMap',
            },
            {
              key: 'falseSupportSpreadGateMap',
              mode: 'unit',
              title: 'falseSupportSpreadGateMap',
            },
            { key: 'falseSupportVetoMap', mode: 'unit', title: 'falseSupportVetoMap' },
            {
              key: 'rowBackgroundDensityGateMap',
              mode: 'unit',
              title: 'rowBackgroundDensityGateMap',
            },
            {
              key: 'rowBackgroundSpreadGateMap',
              mode: 'unit',
              title: 'rowBackgroundSpreadGateMap',
            },
            {
              key: 'rowBackgroundPeakHuGateMap',
              mode: 'unit',
              title: 'rowBackgroundPeakHuGateMap',
            },
            {
              key: 'rowBackgroundEdgeGateMap',
              mode: 'unit',
              title: 'rowBackgroundEdgeGateMap',
            },
            { key: 'rowBackgroundVetoMap', mode: 'unit', title: 'rowBackgroundVetoMap' },
            {
              key: 'supportVetoTriggeredMap',
              mode: 'unit',
              title: 'supportVetoTriggeredMap',
            },
            { key: 'supportConfidenceMap', mode: 'unit', title: 'supportConfidenceMap' },
            { key: 'supportDepthMap', mode: 'signed', title: 'supportDepthMap' },
            { key: 'supportSpreadMap', mode: 'positive', title: 'supportSpreadMap' },
            { key: 'supportDensityMap', mode: 'unit', title: 'supportDensityMap' },
            {
              key: 'continuityExpandedTroughHalfWidthMap',
              mode: 'positive',
              title: 'continuityExpandedTroughHalfWidthMap',
            },
            {
              key: 'backgroundTroughNarrowGateMap',
              mode: 'unit',
              title: 'backgroundTroughNarrowGateMap',
            },
            { key: 'dominantToothBandGateMap', mode: 'unit', title: 'dominantToothBandGateMap' },
            { key: 'broadWeakToothBandGateMap', mode: 'unit', title: 'broadWeakToothBandGateMap' },
            {
              key: 'toothContinuityAdmissionGateMap',
              mode: 'unit',
              title: 'toothContinuityAdmissionGateMap',
            },
            { key: 'totalAttenuationMap', mode: 'positive', title: 'totalAttenuationMap' },
            {
              key: 'admissionAccumulationMap',
              mode: 'positive',
              title: 'admissionAccumulationMap',
            },
            {
              key: 'toneSuppressedAccumulationMap',
              mode: 'positive',
              title: 'toneSuppressedAccumulationMap',
            },
            {
              key: 'participatingSampleCountMap',
              mode: 'positive',
              title: 'participatingSampleCountMap',
            },
            {
              key: 'preToneAccumulationMap',
              mode: 'positive',
              title: 'preToneAccumulationMap-legacy-tone-suppressed',
            },
            { key: 'toneResponseMap', mode: 'unit', title: 'toneResponseMap' },
            {
              key: 'retainedSampleMaskMap',
              mode: 'unit',
              title: 'retainedSampleMaskMap',
            },
            { key: 'middleBandLeakMap', mode: 'unit', title: 'middleBandLeakMap' },
            {
              key: 'admissionMiddleBandLeakMap',
              mode: 'unit',
              title: 'admissionMiddleBandLeakMap',
            },
            {
              key: 'toneStageSuppressionMap',
              mode: 'unit',
              title: 'toneStageSuppressionMap',
            },
            { key: 'blackClipMap', mode: 'unit', title: 'blackClipMap' },
            { key: 'invalidSupportBlackoutMap', mode: 'unit', title: 'invalidSupportBlackoutMap' },
            { key: 'admissionOnlyHuMap', mode: 'hu', title: 'admissionOnlyHuMap' },
            { key: 'toneBypassHuMap', mode: 'hu', title: 'toneBypassHuMap' },
            { key: 'backgroundLeakToneMap', mode: 'unit', title: 'backgroundLeakToneMap' },
            {
              key: 'backgroundLeakOutlier05Map',
              mode: 'unit',
              title: 'backgroundLeakOutlier05Map',
            },
            {
              key: 'backgroundLeakOutlier10Map',
              mode: 'unit',
              title: 'backgroundLeakOutlier10Map',
            },
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
                  `scalarMode=${attempt.debugScalarSamplingMode}`,
                  `stageHint=${
                    spec.candidate?.metricStage ?? attempt.toothBandStageDiagnostics?.stageHint ?? 'na'
                  }`,
                  `rejects=${spec.candidate?.rejectReasons.join(', ') || 'none'}`,
                  `workerRejects=${spec.candidate?.workerRejectReasons.join(', ') || 'none'}`,
                ].join('\n')
              );
            }
            const rowMeanPlotSpecs: Array<{
              key: keyof NonNullable<PanoImagePayload['debugMaps']>;
              title: string;
              filenameStem: string;
            }> = [
              {
                key: 'upperBandRawOdMap',
                title: 'rowMeanRawUpper.png',
                filenameStem: 'row-mean-raw-upper',
              },
              {
                key: 'lowerBandRawOdMap',
                title: 'rowMeanRawLower.png',
                filenameStem: 'row-mean-raw-lower',
              },
              {
                key: 'gapBandRawOdMap',
                title: 'rowMeanRawGap.png',
                filenameStem: 'row-mean-raw-gap',
              },
              {
                key: 'displayBackgroundOdMap',
                title: 'rowMeanDisplayBackground.png',
                filenameStem: 'row-mean-display-background',
              },
              {
                key: 'preNormalizeCompositeOdMap',
                title: 'rowMeanPreNormalizeComposite.png',
                filenameStem: 'row-mean-pre-normalize-composite',
              },
              {
                key: 'postNormalizeDisplayMap',
                title: 'rowMeanPostNormalizeDisplay.png',
                filenameStem: 'row-mean-post-normalize-display',
              },
              {
                key: 'displayAnatomyMap',
                title: 'rowMeanDisplayAnatomy.png',
                filenameStem: 'row-mean-display-anatomy',
              },
              {
                key: 'backgroundPresentationMap',
                title: 'rowMeanBackgroundPresentation.png',
                filenameStem: 'row-mean-background-presentation',
              },
              {
                key: 'finalCompositeDisplayMap',
                title: 'rowMeanFinalDisplay.png',
                filenameStem: 'row-mean-final-display',
              },
            ];
            for (const plotSpec of rowMeanPlotSpecs) {
              const pixels = attempt.result.debugMaps?.[plotSpec.key];
              if (!pixels) {
                continue;
              }
              appendRowMeanPlotImage(
                `${spec.title} ${plotSpec.title}`,
                `${spec.prefix}-${attempt.label}-${plotSpec.filenameStem}`,
                pixels,
                attempt.result.width,
                attempt.result.height,
                [
                  `label=${attempt.label}`,
                  `backend=${attempt.resolvedBackend}`,
                  `reconstructionMode=${attempt.reconstructionMode}`,
                  `stageHint=${
                    spec.candidate?.metricStage ?? attempt.toothBandStageDiagnostics?.stageHint ?? 'na'
                  }`,
                ].join('\n')
              );
            }
          }
          const emitDebugArtifacts = () => {
            const reportProbeRows = (panoDebugProbeClicksByRun.get(debugRunId) ?? []).map(
              (sample, index) => ({
                clickIndex: index + 1,
                imageId: sample.imageId ?? null,
                col: sample.col,
                row: sample.row,
                mappingMode: sample.mappingMode ?? null,
                displayedPath: sample.displayedPath ?? null,
                backend: sample.backend ?? null,
                pipelineMode: sample.pipelineMode ?? null,
                reconstructionMode: sample.reconstructionMode ?? null,
                rawSupportDepthMm: sample.rawSupportDepthMm ?? null,
                rawSupportPeakDominance: sample.rawSupportPeakDominance ?? null,
                rawSupportPeakValidity: sample.rawSupportPeakValidity ?? null,
                rawSupportSecondPeakRatio: sample.rawSupportSecondPeakRatio ?? null,
                rawSupportPeakAmbiguity: sample.rawSupportPeakAmbiguity ?? null,
                rawSupportScoreGap: sample.rawSupportScoreGap ?? null,
                rawSupportDenseFraction: sample.rawSupportDenseFraction ?? null,
                rawSupportPeakHuSupportGate: sample.rawSupportPeakHuSupportGate ?? null,
                supportCenterMm: sample.supportDepthMm ?? null,
                supportSpreadMm: sample.supportSpreadMm ?? null,
                supportValidity: sample.supportValidity ?? null,
                supportDensity: sample.supportDensity ?? null,
                supportConfidence: sample.supportConfidence ?? null,
                dominantDensePeakGate: sample.dominantDensePeakGate ?? null,
                toothBandStructureGuard: sample.toothBandStructureGuard ?? null,
                protectedAmbiguousBroadSupportPenaltyGate:
                  sample.protectedAmbiguousBroadSupportPenaltyGate ?? null,
                falseSupportVeto: sample.falseSupportVeto ?? null,
                rowBackgroundVeto: sample.rowBackgroundVeto ?? null,
                supportVetoTriggered: sample.supportVetoTriggered ?? null,
                effectiveTroughHalfWidthMm: sample.effectiveTroughHalfWidthMm ?? null,
                continuityExpandedTroughHalfWidthMm:
                  sample.continuityExpandedTroughHalfWidthMm ?? null,
                dominantToothBandGate: sample.dominantToothBandGate ?? null,
                broadWeakToothBandGate: sample.broadWeakToothBandGate ?? null,
                toothContinuityAdmissionGate: sample.toothContinuityAdmissionGate ?? null,
                admissionAccumulation: sample.admissionAccumulation ?? null,
                preToneAccumulation: sample.preToneAccumulation ?? null,
                blackClip: sample.blackClip ?? null,
                retainedSampleMask: sample.retainedSampleMask ?? null,
                middleBandLeak: sample.middleBandLeak ?? null,
                renderBranchCode: sample.renderBranchCode ?? null,
                selectedSupportHypothesis: sample.selectedSupportHypothesis ?? null,
                focalTroughSharpness: sample.focalTroughSharpness ?? null,
                outOfTroughSuppression: sample.outOfTroughSuppression ?? null,
                rawProjectedAttenuation: sample.rawProjectedAttenuation ?? null,
                finalDisplayImage: sample.finalDisplayImage ?? null,
                holeMetricWouldCount: sample.holeMetricWouldCount,
                holeMetricReasons: sample.holeMetricReasons ?? [],
              })
            );
            downloadCprDebugArtifacts({
              title: `CPR Attempt Debug Report ${debugRunId}`,
              runId: debugRunId,
              rows: reportRows,
              probeRows: reportProbeRows,
              images: reportImages,
              summarySections: reportSummarySections,
            });
          };
          panoDebugArtifactExportersByRun.set(debugRunId, emitDebugArtifacts);
          emitDebugArtifacts();
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
          degradedPreviewSelectedChoice?.candidate.rejectReasons ??
          selectedQualityGateCandidate.rejectReasons;
        const usingEmergencyDegradedPreviewFallback = !!degradedPreviewEmergencyChoice;
        const qualityGateFallbackReason = forcedDisplayedAttemptForDebug
          ? 'quality-gate-failed-debug-forcing-gpu-candidate'
          : canPreservePreviousPanoDisplay
            ? 'quality-gate-blocked-keeping-current-accepted-pano'
            : shouldDisplayPhase2DiagnosticDraft
              ? 'quality-gate-failed-displaying-primary-diagnostic-draft'
              : degradedPreviewSelectedChoice
                ? usingEmergencyDegradedPreviewFallback
                  ? 'quality-gate-failed-displaying-emergency-debug-pano'
                  : 'quality-gate-failed-displaying-ranked-degraded-pano'
                : selectedAttemptDisplayable
                  ? 'quality-gate-blocked-ranked-top-not-allowed-by-policy'
                  : 'quality-gate-blocked-ranked-top-not-displayable';
        const qualityGateFallbackMessage = forcedDisplayedAttemptForDebug
          ? `[CPR] Quality gate rejected the normal winner "${selectedAttempt.label}". Debug override is forcing GPU candidate "${forcedDisplayedAttemptForDebug.label}" onto the display. Normal reject reasons: ${formatPhase4RejectReasonsForMessage(
              selectedQualityGateCandidate.rejectReasons
            )}. GPU reject reasons: ${formatPhase4RejectReasonsForMessage(
              forcedDisplayedQualityGateCandidate?.rejectReasons ?? []
            )}.`
          : canPreservePreviousPanoDisplay
            ? `[CPR] Quality gate rejected the new panoramic output. Keeping the current accepted pano. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                selectedQualityGateCandidate.rejectReasons
              )}.`
            : shouldDisplayPhase2DiagnosticDraft
              ? `[CPR] Quality gate rejected the panoramic output. Displaying the primary virtual pano draft from "${phase2BaseAttempt?.label ?? selectedAttempt.label}" because no previously accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                  selectedQualityGateCandidate.rejectReasons
                )}.`
              : degradedPreviewSelectedChoice
                ? usingEmergencyDegradedPreviewFallback
                  ? `[CPR] Quality gate rejected the panoramic output. Displaying emergency debug pano "${degradedPreviewSelectedChoice.attempt.label}" because no accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                      degradedPreviewRejectReasons
                    )}.`
                  : `[CPR] Quality gate rejected the panoramic output. Displaying degraded pano "${degradedPreviewSelectedChoice.attempt.label}" because no accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                      degradedPreviewRejectReasons
                    )}.`
                : selectedAttemptDisplayable
                  ? `[CPR] Quality gate rejected the panoramic output. Blocking display of best-ranked pano "${selectedAttempt.label}" because it failed QC and no previously accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                      selectedQualityGateCandidate.rejectReasons
                    )}.`
                  : `[CPR] Quality gate rejected the panoramic output and the best-ranked pano "${selectedAttempt.label}" is not displayable. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                      selectedQualityGateCandidate.rejectReasons
                    )}. Catastrophic reasons: ${
                      selectedAttemptCatastrophicReasons.join(', ') || 'none'
                    }.`;
        const qualityGateCanContinue =
          !!forcedDisplayedAttemptForDebug ||
          canPreservePreviousPanoDisplay ||
          shouldDisplayPhase2DiagnosticDraft ||
          !!degradedPreviewSelectedChoice;

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
          degradedPreviewAttemptLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
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
            degradedPreviewAttemptLabel: degradedPreviewSelectedChoice?.attempt.label ?? null,
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

      const usingForcedGpuDisplayForDebug =
        !!forcedDisplayedAttemptForDebug && !!forcedDisplayedQualityGateCandidate;
      const degradedPreviewDisplayChoice =
        !usingForcedGpuDisplayForDebug &&
        !temporarilyPinnedDisplayedAttempt &&
        !canPreservePreviousPanoDisplay &&
        !shouldDisplayPhase2DiagnosticDraft &&
        !selectedQualityGateCandidate.pass
          ? degradedPreviewSelectedChoice
          : null;
      const usingDegradedPreviewFallback = !!degradedPreviewDisplayChoice;
      const usingEmergencyDegradedPreviewFallback =
        !!degradedPreviewDisplayChoice &&
        degradedPreviewDisplayChoice === degradedPreviewEmergencyChoice;
      const displayedAttempt =
        forcedDisplayedAttemptForDebug ??
        temporarilyPinnedDisplayedAttempt ??
        degradedPreviewDisplayChoice?.attempt ??
        selectedAttempt;
      let displayedQualityGateCandidate =
        forcedDisplayedQualityGateCandidate ??
        degradedPreviewDisplayChoice?.candidate ??
        selectedQualityGateCandidate;
      let qualityGateDisplaySelectionReason = usingForcedGpuDisplayForDebug
        ? 'debug-force-display-gpu-candidate-even-if-rejected'
        : usingDegradedPreviewFallback
          ? usingEmergencyDegradedPreviewFallback
            ? 'quality-gate-failed-displaying-emergency-debug-pano'
            : 'quality-gate-failed-displaying-ranked-degraded-pano'
          : qualityGateSelectionReason;
      let panoWorkerResult = displayedAttempt.result;
      let panoDebugSummary = displayedAttempt.summary;
      let adaptiveVoi = displayedAttempt.voi;
      const displayedAttemptRouteDiagnostic = resolveWorkerDisplayRouteDiagnostic(
        displayedAttempt.workerDebugPayload
      );
      let displayedOutputMode: PanoDisplayedOutputMode =
        usingForcedGpuDisplayForDebug || usingDegradedPreviewFallback
          ? resolveDisplayedOutputModeFromWorkerResult({
              routeDiagnostic: displayedAttemptRouteDiagnostic,
              workerDebugPayload: displayedAttempt.workerDebugPayload,
            })
          : selectedDisplayedOutputMode;
      let displayedSourceLabel = displayedAttempt.label;
      let displayedSourceAggregation = displayedAttempt.aggregation;
      let selectedPanoWidth = displayedAttempt.panoWidth;
      let selectedPanoHeight = displayedAttempt.panoHeight;
      let selectedActualVertHalfMm = displayedAttempt.actualVertHalfMm;
      let selectedColumnPixelSpacing = displayedAttempt.columnPixelSpacing;
      let selectedRowPixelSpacing = displayedAttempt.rowPixelSpacing;
      const displayUsesPhase2VirtualPano = Boolean(
        !usingForcedGpuDisplayForDebug &&
        !usingDegradedPreviewFallback &&
        phase2VirtualPano.usedAsDisplayedOutput &&
        phase2WorkerResult &&
        phase2VirtualPano.summary &&
        phase2VirtualPano.voi &&
        phase2BaseAttempt
      );
      const displayingRejectedPhase2VirtualPanoForDebug = Boolean(
        displayUsesPhase2VirtualPano &&
          phase2VirtualPano.displayDecisionReason &&
          phase2VirtualPano.displayDecisionReason.startsWith(
            'debug-force-display-rejected-pano-v2-fusion'
          )
      );
      if (displayUsesPhase2VirtualPano && phase2BaseAttempt) {
        panoWorkerResult = phase2WorkerResult;
        panoDebugSummary = phase2VirtualPano.summary;
        adaptiveVoi = phase2VirtualPano.voi;
        displayedOutputMode = phase2DisplayedOutputMode;
        displayedSourceLabel = phase2BaseAttempt.label;
        displayedSourceAggregation = phase2BaseAttempt.aggregation;
        selectedPanoWidth = phase2BaseAttempt.panoWidth;
        selectedPanoHeight = phase2BaseAttempt.panoHeight;
        selectedActualVertHalfMm = phase2BaseAttempt.actualVertHalfMm;
        selectedColumnPixelSpacing = phase2BaseAttempt.columnPixelSpacing;
        selectedRowPixelSpacing = phase2BaseAttempt.rowPixelSpacing;
        displayedQualityGateCandidate =
          explicitPhase2QualityGateCandidate ?? selectedQualityGateCandidate;
        qualityGateDisplaySelectionReason =
          phase2VirtualPano.displayDecisionReason ===
          'accepted-by-worker-and-orchestrator-after-rejected-seed-revival-rule'
            ? 'phase2-virtual-pano-accepted-after-rejected-seed-revival-rule'
            : phase2VirtualPano.displayDecisionReason?.startsWith(
                  'debug-force-display-rejected-pano-v2-fusion'
                )
              ? 'debug-force-display-rejected-pano-v2-fusion'
            : 'phase2-virtual-pano-accepted';
      }
      const displayIntensityDomain = displayUsesPhase2VirtualPano
        ? phase2VirtualPano.intensityDomain
        : displayedAttempt.intensityDomain;
      const displayHuDomain = displayUsesPhase2VirtualPano
        ? phase2VirtualPano.huDomain
        : displayedAttempt.huDomain;
      const displayingQcFailedRankedTop =
        !usingForcedGpuDisplayForDebug &&
        !usingDegradedPreviewFallback &&
        !temporarilyPinnedDisplayedAttempt &&
        !displayUsesPhase2VirtualPano &&
        shouldPreservePreviousPanoDisplay &&
        !canPreservePreviousPanoDisplay &&
        !displayedQualityGateCandidate.pass;
      const displayedWorkerDebugPayload = displayUsesPhase2VirtualPano
        ? phase2WorkerResult?.workerDebugPayload ?? null
        : displayedAttempt.workerDebugPayload ?? null;
      const displayedRouteDiagnostic = displayUsesPhase2VirtualPano
        ? resolveWorkerDisplayRouteDiagnostic(displayedWorkerDebugPayload)
        : displayedAttemptRouteDiagnostic;
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
        (displayUsesPhase2VirtualPano
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
          : displayUsesPhase2VirtualPano
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
            label: displayUsesPhase2VirtualPano
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
            label: displayUsesPhase2VirtualPano
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
            qualityStatus:
              usingForcedGpuDisplayForDebug ||
              displayingRejectedPhase2VirtualPanoForDebug ||
              usingDegradedPreviewFallback ||
              displayingQcFailedRankedTop
                ? 'degraded'
                : 'accepted',
            qualityGatePassed: displayedQualityGateCandidate.pass,
            qualityGateSelectionReason: qualityGateDisplaySelectionReason,
            qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
            workerTimingMs: displayedWorkerTimingMs,
            outputSignature: displayedOutputSignature,
          },
        })
      );
      console.log(`[CPR][${debugRunId}] selected pano candidate`, {
        label: displayUsesPhase2VirtualPano ? displayedSourceLabel : displayedAttempt.label,
        displayedAttemptLabel: displayedAttempt.label,
        forceDisplayGpuCandidateEvenIfRejected: usingForcedGpuDisplayForDebug,
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
        qualityGateSelectionReason: qualityGateDisplaySelectionReason,
        qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
        qualityGatePassed: displayedQualityGateCandidate.pass,
        qualityStatus:
          usingForcedGpuDisplayForDebug ||
          displayingRejectedPhase2VirtualPanoForDebug ||
          usingDegradedPreviewFallback ||
          displayingQcFailedRankedTop
            ? 'degraded'
            : 'accepted',
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

          const preservedPayloadForDisplay =
            !usingForcedGpuDisplayForDebug && canPreservePreviousPanoDisplay
            ? preservedPreviousPanoPayload
            : null;
          const displayingDegradedPanoPreview = usingDegradedPreviewFallback;
          const qualityStatusForDisplay: PanoQualityStatus =
            usingForcedGpuDisplayForDebug ||
            displayingRejectedPhase2VirtualPanoForDebug ||
            displayingDegradedPanoPreview ||
            displayingQcFailedRankedTop
              ? 'degraded'
              : 'accepted';
          const qualityMessageForDisplay = usingForcedGpuDisplayForDebug
            ? `[CPR] Debug override is forcing the GPU pano candidate "${displayedAttempt.label}" for inspection. Normal selection chose "${selectedAttempt.label}". GPU reject reasons: ${formatPhase4RejectReasonsForMessage(
                displayedQualityGateCandidate.rejectReasons
              )}.`
            : displayingRejectedPhase2VirtualPanoForDebug
              ? `[CPR] Debug override is forcing the rejected pano-v2 fusion output for inspection. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                  displayedQualityGateCandidate.rejectReasons
                )}.`
            : displayingDegradedPanoPreview
              ? usingEmergencyDegradedPreviewFallback
                ? `[CPR] Displaying emergency debug pano from "${displayedAttempt.label}" because no accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                    displayedQualityGateCandidate.rejectReasons
                  )}.`
                : `[CPR] Displaying degraded pano from "${displayedAttempt.label}" because no accepted pano is available. Reject reasons: ${formatPhase4RejectReasonsForMessage(
                    displayedQualityGateCandidate.rejectReasons
                  )}.`
            : displayingQcFailedRankedTop
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
              qualityGateSelectionReason: qualityGateDisplaySelectionReason,
              qualityGateMessage: qualityMessageForDisplay,
              holeMetricBlackClipThreshold: CPR_TOOTH_BAND_BLACK_CLIP_THRESHOLD,
              holeMetricRetainedWeightMax: CPR_TOOTH_BAND_HOLE_RETAINED_WEIGHT_MAX,
              holeMetricPreToneThreshold:
                displayedAttempt.summary?.toothBandHolePreToneThreshold ?? null,
              holeMetricLeakMin: CPR_TOOTH_BAND_HOLE_LEAK_MIN,
            },
            width: panoWorkerResult.width,
            height: panoWorkerResult.height,
            minValue: panoWorkerResult.minValue,
            maxValue: panoWorkerResult.maxValue,
            qualityGatePassed: displayedQualityGateCandidate.pass,
            qualityStatus: qualityStatusForDisplay,
            qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
            qualityGateSelectionReason: qualityGateDisplaySelectionReason,
            qualityGateMessage: qualityMessageForDisplay,
            huDomain: displayHuDomain,
            intensityDomain: displayIntensityDomain,
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
                selectionReason: qualityGateDisplaySelectionReason,
                selectedAttemptLabel: selectedAttempt.label,
                selectedHardRejectReason: selectedAttempt.hardRejectReason,
                preservedPreviousPanoImageId,
              }
            );
            console.log(
              '[CPR-PRESERVE-PREVIOUS-PANO-JSON]',
              JSON.stringify({
                runId: debugRunId,
                selectionReason: qualityGateDisplaySelectionReason,
                selectedAttemptLabel: selectedAttempt.label,
                selectedHardRejectReason: selectedAttempt.hardRejectReason,
                preservedPreviousPanoImageId,
                selectedDespiteGateFailure: !selectedQualityGateCandidate.pass,
              })
            );
          }
          if (usingForcedGpuDisplayForDebug) {
            console.warn(
              `[CPR][${debugRunId}] forcing rejected GPU pano candidate onto the display for direct evaluation`,
              {
                forcedAttemptLabel: displayedAttempt.label,
                normalSelectionLabel: selectedAttempt.label,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
              }
            );
            console.log(
              '[CPR-FORCED-GPU-DISPLAY-JSON]',
              JSON.stringify({
                runId: debugRunId,
                forcedAttemptLabel: displayedAttempt.label,
                normalSelectionLabel: selectedAttempt.label,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
                selectionReason: qualityGateDisplaySelectionReason,
              })
            );
          }
          if (displayingDegradedPanoPreview) {
            console.warn(
              `[CPR][${debugRunId}] displaying degraded pano preview because no accepted pano is available`,
              {
                selectionReason: qualityGateDisplaySelectionReason,
                degradedAttemptLabel: displayedAttempt.label,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
                emergencyPreview: usingEmergencyDegradedPreviewFallback,
              }
            );
            console.log(
              '[CPR-DISPLAY-DEGRADED-PREVIEW-JSON]',
              JSON.stringify({
                runId: debugRunId,
                selectionReason: qualityGateDisplaySelectionReason,
                degradedAttemptLabel: displayedAttempt.label,
                selectedAttemptLabel: selectedAttempt.label,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
                qualityStatus: qualityStatusForDisplay,
                selectionMode: usingEmergencyDegradedPreviewFallback ? 'emergency' : 'displayable',
              })
            );
          }
          if (displayingQcFailedRankedTop) {
            console.warn(
              `[CPR][${debugRunId}] displaying best-ranked pano despite QC failure because no previously accepted pano is available`,
              {
                selectionReason: qualityGateDisplaySelectionReason,
                selectedAttemptLabel: displayedAttempt.label,
                selectedHardRejectReason: displayedAttempt.hardRejectReason,
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
              }
            );
            console.log(
              '[CPR-DISPLAY-QC-FAILED-RANKED-TOP-JSON]',
              JSON.stringify({
                runId: debugRunId,
                selectionReason: qualityGateDisplaySelectionReason,
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
            '[CPR-FINAL-DISPLAYED-IMAGE-JSON]',
            JSON.stringify({
              runId: debugRunId,
              displayedLabel: displayUsesPhase2VirtualPano
                ? displayedSourceLabel
                : displayedAttempt.label,
              displayedSourceLabel,
              displayedSourceAggregation,
              displayedOutputMode,
              displayUsesPhase2VirtualPano,
              usingForcedGpuDisplayForDebug,
              usingDegradedPreviewFallback,
              usingEmergencyDegradedPreviewFallback,
              displayingQcFailedRankedTop,
              qualityStatus: qualityStatusForDisplay,
              qualityGatePassed: displayedQualityGateCandidate.pass,
              qualityGateSelectionReason: qualityGateDisplaySelectionReason,
              qualityGateRejectReasons: displayedQualityGateCandidate.rejectReasons,
              phase0AttemptTableCandidate: {
                label: displayUsesPhase2VirtualPano
                  ? phase2BaseAttempt?.label ?? null
                  : displayedAttempt.label,
                qualityGatePassed: displayUsesPhase2VirtualPano
                  ? phase2BaseAttempt?.qualityGatePassed ?? null
                  : displayedAttempt.qualityGatePassed,
                rejectReasons: displayUsesPhase2VirtualPano
                  ? phase2BaseAttempt?.qualityGateRejectReasons ?? []
                  : displayedAttempt.qualityGateRejectReasons,
                hardRejectReason: displayUsesPhase2VirtualPano
                  ? phase2BaseAttempt?.hardRejectReason ?? null
                  : displayedAttempt.hardRejectReason,
                structuralRejectReasons: displayUsesPhase2VirtualPano
                  ? phase2BaseAssessment?.structuralRejectReasons ?? []
                  : phase2BaseAssessmentByLabel.get(displayedAttempt.label)?.structuralRejectReasons ??
                    [],
              },
              phase2DisplayedCandidate: {
                executed: phase2VirtualPano.executed,
                skippedReason: phase2VirtualPano.skippedReason,
                error: phase2VirtualPano.error,
                seedLabel: phase2BaseAttempt?.label ?? null,
                seedEligible: phase2VirtualPano.phase0SeedEligible,
                seedRejectReasons: phase2VirtualPano.phase0SeedRejectReasons,
                seedStructuralRejectReasons: phase2VirtualPano.phase0SeedStructuralRejectReasons,
                workerAcceptedForOutput: phase2VirtualPano.workerAcceptedForOutput,
                orchestratorGatePassed: phase2VirtualPano.orchestratorGatePassed,
                orchestratorRejectReasons: phase2VirtualPano.orchestratorRejectReasons,
                displayDecisionReason: phase2VirtualPano.displayDecisionReason,
                renderFamily: phase2VirtualPano.phase2RenderFamily,
                candidateSource: phase2VirtualPano.phase2CandidateSource,
                derivedFromPhase0RejectedFamily: phase2VirtualPano.revivedRejectedFamily,
                rejectedSeedRulePassed: phase2VirtualPano.revivedRejectedFamilyRulePassed,
                rejectedSeedRuleRejectReasons: phase2VirtualPano.revivedRejectedFamilyRejectReasons,
              },
              finalDisplayedSummary: panoDebugSummary
                ? {
                    sampledCount: panoDebugSummary.sampledCount,
                    backgroundOutlierFraction05:
                      panoDebugSummary.backgroundOutlierFraction05 ?? null,
                    backgroundOutlierFraction10:
                      panoDebugSummary.backgroundOutlierFraction10 ?? null,
                    lowerBandBrightFraction: panoDebugSummary.lowerBandBrightFraction ?? null,
                    blackClipFraction:
                      displayedQualityGateCandidate.supportSurfaceRiskSummary.background
                        .blackClipFraction,
                    toothBandHoleFraction:
                      displayedQualityGateCandidate.metrics.toothBandHoleFraction,
                    toothBandBlackClipFraction:
                      displayedQualityGateCandidate.metrics.toothBandBlackClipFraction,
                    toothBandRetainedWeightP10:
                      displayedQualityGateCandidate.metrics.toothBandRetainedWeightP10,
                    toothBandRetainedWeightP50:
                      displayedQualityGateCandidate.metrics.toothBandRetainedWeightP50,
                  }
                : null,
              displayedImageDerivedFromPhase0RejectedFamily: displayUsesPhase2VirtualPano
                ? phase2VirtualPano.revivedRejectedFamily
                : !displayedAttempt.qualityGatePassed || !!displayedAttempt.hardRejectReason,
            })
          );
          console.log(
            '[CPR-FINAL-WINNER-STATUS-JSON]',
            JSON.stringify({
              runId: debugRunId,
              normalAcceptedWinner: selectedQualityGateCandidate.pass
                ? {
                    label: selectedAttempt.label,
                    qualityGatePassed: true,
                    rejectReasons: [],
                    selectionReason: qualityGateSelectionReason,
                  }
                : null,
              normalRejectedTopCandidate: selectedQualityGateCandidate.pass
                ? null
                : {
                    label: selectedAttempt.label,
                    qualityGatePassed: false,
                    rejectReasons: selectedQualityGateCandidate.rejectReasons,
                    selectionReason: qualityGateSelectionReason,
                  },
              degradedPreviewWinner: degradedPreviewSelectedChoice
                ? {
                    label: degradedPreviewSelectedChoice.attempt.label,
                    selectionMode: degradedPreviewFallbackChoice
                      ? 'displayable'
                      : degradedPreviewEmergencyChoice
                        ? 'emergency'
                        : 'unknown',
                    rejectReasons: degradedPreviewSelectedChoice.candidate.rejectReasons,
                    blockedAsDegradedPreview:
                      degradedPreviewSelectedChoice.degradedPreviewAssessment
                        .blockedAsDegradedPreview,
                    blockedAsDegradedPreviewReasons:
                      degradedPreviewSelectedChoice.degradedPreviewAssessment
                        .blockedAsDegradedPreviewReasons,
                  }
                : null,
              displayedImage: {
                label: displayUsesPhase2VirtualPano
                  ? displayedSourceLabel
                  : displayedAttempt.label,
                accepted: displayedQualityGateCandidate.pass,
                mode: usingForcedGpuDisplayForDebug
                  ? 'forced-debug-gpu'
                  : usingDegradedPreviewFallback
                    ? usingEmergencyDegradedPreviewFallback
                      ? 'emergency-debug-preview'
                      : 'degraded-preview'
                    : displayUsesPhase2VirtualPano
                      ? 'accepted-phase2'
                      : !!preservedPayloadForDisplay
                        ? 'preserve-previous-accepted'
                        : displayingQcFailedRankedTop
                          ? 'qc-failed-ranked-top'
                          : 'accepted-ranked-top',
                rejectReasons: displayedQualityGateCandidate.rejectReasons,
                qualityStatus: qualityStatusForDisplay,
              },
              phase2SeedStatus: {
                selectedLabel: phase2BaseAttempt?.label ?? null,
                selectionBlockedReason: phase2BaseSelectionBlockedReason,
                targetLabel: CPR_TARGET_TOOTH_DEBUG_LABEL,
                targetSeedEligible:
                  phase2BaseAssessmentByLabel.get(CPR_TARGET_TOOTH_DEBUG_LABEL)?.seedEligible ?? null,
                targetStructuralRejectReasons:
                  phase2BaseAssessmentByLabel.get(CPR_TARGET_TOOTH_DEBUG_LABEL)
                    ?.structuralRejectReasons ?? [],
              },
              forceDisplayOverride: {
                requested: forceDisplayGpuCandidateEvenIfRejected,
                active: usingForcedGpuDisplayForDebug,
                blockedReason: forcedDisplayedAttemptBlockedReason,
                blockedReasons: forcedDisplayedAttemptBlockedReasons,
              },
            })
          );
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
