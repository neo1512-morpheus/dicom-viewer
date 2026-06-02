/**
 * cprGpuRenderer.ts
 * GPU-accelerated panoramic CPR rendering using WebGL2.
 * Replaces CPU trilinear sampling with hardware 3D texture interpolation.
 */

// ─── Types ───────────────────────────────────────────────────────────
export interface GpuPanoInput {
  scalarData: Float32Array | Int16Array | Uint16Array;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
  worldToIndex?: number[] | null;

  /** Per-column spline frames (position + slab normal). Length = panoWidth. */
  frames: Array<{
    position: [number, number, number];
    N_slab: [number, number, number];
    S?: [number, number, number];
  }>;

  panoWidth: number;
  panoHeight: number;
  verticalDir: [number, number, number];
  vertHalfMm: number;
  verticalCenterOffsetMm: number;
  slabHalfThicknessMm: number;
  slabSamples: number;

  rescaleSlope: number;
  rescaleIntercept: number;
  applyRescale: boolean;
  normalizationSignature?: string | null;

  /** Optional normalizer for packed stored values (bit alignment, sign extension). */
  normalizeStoredSample?: (value: number) => number;
  returnDebugSidecars?: boolean;
}

export interface GpuPanoDebugMaps {
  renderBranchMap?: Float32Array;
  panoV2FusionImageMap?: Float32Array;
  panoV2BypassMissMaskMap?: Float32Array;
  panoV2UpperLayer1Map?: Float32Array;
  panoV2UpperLayer2Map?: Float32Array;
  panoV2UpperLayer3Map?: Float32Array;
  panoV2LowerLayer1Map?: Float32Array;
  panoV2LowerLayer2Map?: Float32Array;
  panoV2LowerLayer3Map?: Float32Array;
  selectedSupportHypothesisMap?: Float32Array;
  focalTroughSharpnessMap?: Float32Array;
  outOfTroughSuppressionMap?: Float32Array;
  rawProjectedAttenuationMap?: Float32Array;
  upperProjectedAttenuationMap?: Float32Array;
  lowerProjectedAttenuationMap?: Float32Array;
  upperBandRawOdMap?: Float32Array;
  lowerBandRawOdMap?: Float32Array;
  gapBandRawOdMap?: Float32Array;
  outsideRowsRawOdMap?: Float32Array;
  displayBackgroundOdMap?: Float32Array;
  bandMaskUpperMap?: Float32Array;
  bandMaskGapMap?: Float32Array;
  bandMaskLowerMap?: Float32Array;
  bandMaskOutsideMap?: Float32Array;
  normalizationEligibleMaskMap?: Float32Array;
  preNormalizeCompositeOdMap?: Float32Array;
  postNormalizeDisplayMap?: Float32Array;
  displayAnatomyMap?: Float32Array;
  backgroundPresentationMap?: Float32Array;
  finalCompositeDisplayMap?: Float32Array;
  contextContributionMap?: Float32Array;
  backgroundFillContributionMap?: Float32Array;
  finalDisplayImageMap?: Float32Array;
  rawSupportDepthMap?: Float32Array;
  rawSupportConfidenceMap?: Float32Array;
  rawSupportSpreadMap?: Float32Array;
  rawSupportDensityMap?: Float32Array;
  rawSupportDenseFractionMap?: Float32Array;
  rawSupportPeakHuSupportGateMap?: Float32Array;
  rawSupportDominantPeakOffsetMap?: Float32Array;
  rawSupportSecondaryPeakOffsetMap?: Float32Array;
  rawSupportPeakDominanceMap?: Float32Array;
  rawSupportPeakValidityMap?: Float32Array;
  rawSupportPeakConflictMap?: Float32Array;
  rawSupportSecondPeakRatioMap?: Float32Array;
  rawSupportPeakSeparationMap?: Float32Array;
  rawSupportPeakAmbiguityMap?: Float32Array;
  rawSupportScoreGapMap?: Float32Array;
  rawSupportLocalJumpMap?: Float32Array;
  rawSupportContinuityMap?: Float32Array;
  toothBandPriorMap?: Float32Array;
  dominantDensePeakGateMap?: Float32Array;
  toothBandStructureGuardMap?: Float32Array;
  ambiguousBroadSupportPenaltyGateMap?: Float32Array;
  protectedAmbiguousBroadSupportPenaltyGateMap?: Float32Array;
  structuralSupportGateMap?: Float32Array;
  peakStructureValidityMap?: Float32Array;
  supportValidityMap?: Float32Array;
  rowConfidenceGateMap?: Float32Array;
  falseSupportConfidenceGateMap?: Float32Array;
  falseSupportDensityGateMap?: Float32Array;
  falseSupportSpreadGateMap?: Float32Array;
  falseSupportVetoMap?: Float32Array;
  rowBackgroundDensityGateMap?: Float32Array;
  rowBackgroundSpreadGateMap?: Float32Array;
  rowBackgroundPeakHuGateMap?: Float32Array;
  rowBackgroundEdgeGateMap?: Float32Array;
  rowBackgroundVetoMap?: Float32Array;
  supportVetoTriggeredMap?: Float32Array;
  supportFailureDisplayMap?: Float32Array;
  upperSupportDepthMap?: Float32Array;
  lowerSupportDepthMap?: Float32Array;
  upperSupportConfidenceMap?: Float32Array;
  lowerSupportConfidenceMap?: Float32Array;
  supportBlendMap?: Float32Array;
  supportDepthMap?: Float32Array;
  supportConfidenceMap?: Float32Array;
  supportSpreadMap?: Float32Array;
  supportDensityMap?: Float32Array;
  supportLocalJumpMap?: Float32Array;
  supportContinuityMap?: Float32Array;
  totalAttenuationMap?: Float32Array;
  admissionAccumulationMap?: Float32Array;
  toneSuppressedAccumulationMap?: Float32Array;
  fogAttenuationMap?: Float32Array;
  lowerPenaltyMap?: Float32Array;
  participatingSampleCountMap?: Float32Array;
  toneResponseMap?: Float32Array;
  preToneAccumulationMap?: Float32Array;
  retainedSampleMaskMap?: Float32Array;
  middleBandLeakMap?: Float32Array;
  admissionMiddleBandLeakMap?: Float32Array;
  invalidSupportBlackoutMap?: Float32Array;
  toneStageSuppressionMap?: Float32Array;
  blackClipMap?: Float32Array;
  troughHalfWidthMap?: Float32Array;
  effectiveTroughHalfWidthMap?: Float32Array;
  continuityExpandedTroughHalfWidthMap?: Float32Array;
  backgroundTroughNarrowGateMap?: Float32Array;
  dominantToothBandGateMap?: Float32Array;
  broadWeakToothBandGateMap?: Float32Array;
  toothContinuityAdmissionGateMap?: Float32Array;
  backgroundLeakToneMap?: Float32Array;
  backgroundLeakOutlier05Map?: Float32Array;
  backgroundLeakOutlier10Map?: Float32Array;
  admissionOnlyHuMap?: Float32Array;
  toneBypassHuMap?: Float32Array;
  detailHuMap?: Float32Array;
  contextHuMap?: Float32Array;
  contextBlendFactorMap?: Float32Array;
  columnSupportReliabilityMap?: Float32Array;
  upperDetailHuMap?: Float32Array;
  lowerDetailHuMap?: Float32Array;
}

export interface GpuSupportSurfaceDiagnostics {
  depthMinMm: number;
  depthMaxMm: number;
  depthStdMm: number;
  pathJumpP95Mm: number;
  pathConfidenceP10: number;
  pathConfidenceP50: number;
  pathConfidenceP90: number;
  localJumpP50Mm: number;
  localJumpP90Mm: number;
  localJumpP95Mm: number;
  localJumpOutlierFraction: number;
  lowConfidenceColumnFraction: number;
  lowContinuityColumnFraction: number;
  unstableColumnFraction: number;
  longestUnstableRunColumns: number;
  continuityP10: number;
  continuityP50: number;
  continuityP90: number;
  clampFraction: number;
  confidenceP10: number;
  confidenceP50: number;
  confidenceP90: number;
  spreadP50Mm: number;
  spreadP90Mm: number;
  densityP50: number;
  densityP90: number;
  ambiguousColumnFraction: number;
  forcedDriftFraction: number;
  bestDepthDriftP95Mm: number;
  scoreGapP50: number;
  selectedDepthFirst8Mm: number[];
}

export interface GpuSupportFormationDiagnostics {
  dominantDensePeakGateP50: number;
  dominantDensePeakGateP90: number;
  toothBandStructureGuardP50: number;
  toothBandStructureGuardP90: number;
  protectedAmbiguousBroadSupportPenaltyGateP50: number;
  protectedAmbiguousBroadSupportPenaltyGateP90: number;
  supportValidityP10: number;
  supportValidityP50: number;
  supportValidityP90: number;
  rawScoreGapP10: number;
  rawScoreGapP50: number;
  rawScoreGapP90: number;
  falseSupportVetoFraction: number;
  rowBackgroundVetoFraction: number;
  supportVetoTriggeredFraction: number;
  columnSupportConfidenceP10: number;
  columnSupportConfidenceP50: number;
  columnSupportConfidenceP90: number;
  columnSupportSpreadP10Mm: number;
  columnSupportSpreadP50Mm: number;
  columnSupportSpreadP90Mm: number;
  columnSupportDensityP10: number;
  columnSupportDensityP50: number;
  columnSupportDensityP90: number;
  rowSupportConfidenceP10: number;
  rowSupportConfidenceP50: number;
  rowSupportConfidenceP90: number;
  rowSupportSpreadP10Mm: number;
  rowSupportSpreadP50Mm: number;
  rowSupportSpreadP90Mm: number;
  rowSupportDensityP10: number;
  rowSupportDensityP50: number;
  rowSupportDensityP90: number;
}

export interface GpuSupportPeakDiagnostics {
  peakDominanceP50: number;
  peakDominanceP90: number;
  peakValidityP50: number;
  peakValidityP90: number;
  peakValidityP99: number;
  peakConflictP50: number;
  peakConflictP90: number;
  peakConflictP99: number;
  secondPeakRatioP50: number;
  secondPeakRatioP90: number;
  peakSeparationP50Mm: number;
  peakSeparationP90Mm: number;
  peakAmbiguityP50: number;
  peakAmbiguityP90: number;
  peakAmbiguityP99: number;
}

export interface GpuDrrDiagnostics {
  slabHalfThicknessMm: number;
  requestedSlabSamples: number;
  effectiveRaySampleCountMin: number;
  effectiveRaySampleCountP50: number;
  effectiveRaySampleCountP90: number;
  effectiveRaySampleCountMax: number;
  troughSigmaP10Mm: number;
  troughSigmaP50Mm: number;
  troughSigmaP90Mm: number;
  approxTroughHalfWidthP50Mm: number;
  effectiveTroughHalfWidthP10Mm: number;
  effectiveTroughHalfWidthP50Mm: number;
  effectiveTroughHalfWidthP90Mm: number;
  totalAttenuationP10: number;
  totalAttenuationP50: number;
  totalAttenuationP90: number;
  fogAttenuationP50: number;
  fogAttenuationP90: number;
  lowerPenaltyP50: number;
  lowerPenaltyP90: number;
  participatingSamplesP10: number;
  participatingSamplesP50: number;
  participatingSamplesP90: number;
  localTransmittanceP10: number;
  localTransmittanceP50: number;
  localTransmittanceP90: number;
  backgroundTroughNarrowGateP50: number;
  backgroundTroughNarrowGateP90: number;
  middleBandLeak?: GpuMiddleBandLeakDiagnostics;
  drrModel: {
    supportWeightPowerLowConfidence: number;
    supportWeightPowerHighConfidence: number;
    lowerPenaltyDenseScale: number;
    lowerPenaltyRowStart: number;
    lowerPenaltyRowEnd: number;
    focusedBalancedCenterOffsetMm: number;
    focusedBalancedCenterToleranceMm: number;
    focusedBalancedLeakContributionFloor: number;
    focusedBalancedMiddleLeakProtectScale: number;
    troughSigmaHardCapMm: number;
  };
}

export interface GpuMiddleBandLeakDiagnostics {
  sampleCount: number;
  preToneAccumulationP50: number;
  preToneAccumulationP90: number;
  preToneAccumulationP99: number;
  retainedSampleMaskP50: number;
  retainedSampleMaskP90: number;
  retainedSampleMaskFraction05: number;
  leakP50: number;
  leakP90: number;
  leakP99: number;
  leakFraction05: number;
  leakFraction10: number;
  rowRetainedSampleMaskP10: number;
  rowRetainedSampleMaskP50: number;
  rowRetainedSampleMaskP90: number;
  rowLeakP10: number;
  rowLeakP50: number;
  rowLeakP90: number;
  columnRetainedSampleMaskP10: number;
  columnRetainedSampleMaskP50: number;
  columnRetainedSampleMaskP90: number;
  columnLeakP10: number;
  columnLeakP50: number;
  columnLeakP90: number;
}

export interface GpuToneMapDiagnostics {
  inputAttenuationP01: number;
  inputAttenuationP50: number;
  inputAttenuationP99: number;
  fogAttenuationP50: number;
  fogAttenuationP90: number;
  toneResponseP01: number;
  toneResponseP50: number;
  toneResponseP99: number;
  outputHuP01: number;
  outputHuP50: number;
  outputHuP99: number;
  lowerPenaltyP50: number;
  lowerPenaltyP90: number;
  invalidSupportBlackoutP50: number;
  invalidSupportBlackoutP90: number;
  invalidSupportBlackoutP99: number;
  supportFailureDisplayP50?: number;
  supportFailureDisplayP90?: number;
  supportFailureDisplayP99?: number;
  temporaryDisplayMode?: string | null;
  blackClipFraction: number;
  whiteClipFraction: number;
  toneCurve: {
    exposureScale: number;
    sigmoidMidpoint: number;
    sigmoidSlope: number;
    sigmoidWhitePoint: number;
    postCurveGamma: number;
    outputHuMin: number;
    outputHuMax: number;
    lowConfidenceFogSuppression: number;
    highConfidenceFogSuppression: number;
    lowConfidenceFogRetention: number;
    highConfidenceFogRetention: number;
    lowConfidencePenaltySuppression: number;
    highConfidencePenaltySuppression: number;
    lowConfidencePenaltyRetention: number;
    highConfidencePenaltyRetention: number;
    focusedBalancedBackgroundProtectScale: number;
    focusedBalancedBlackoutProtectScale: number;
  };
}

export interface GpuPanoDiagnostics {
  expectedPipelineMode: 'multi-pass';
  phase2GatePassed: boolean;
  degradedModeReason?: string | null;
  rawSupportPeaks?: GpuSupportPeakDiagnostics;
  rawSupportSurface?: GpuSupportSurfaceDiagnostics;
  supportSurface?: GpuSupportSurfaceDiagnostics;
  rawSupportFormation?: GpuSupportFormationDiagnostics;
  supportFormation?: GpuSupportFormationDiagnostics;
  drr?: GpuDrrDiagnostics;
  toneMap?: GpuToneMapDiagnostics;
  sidecarMaps?: {
    debugEnabled: boolean;
    attachedMaps: string[];
    attachedByteLength: number;
  };
}

export interface GpuPanoResult {
  pixelData: Float32Array;
  meanMap: Float32Array;
  maxMap: Float32Array;
  sampleCountMap: Float32Array;
  width: number;
  height: number;
  minValue: number;
  maxValue: number;
  pipelineMode: 'single-pass' | 'multi-pass';
  debugMaps?: GpuPanoDebugMaps;
  diagnostics?: GpuPanoDiagnostics;
}

const GPU_DEBUG_MODE_OFF = 0;
const GPU_DEBUG_MODE_RAY_START = 1;
const GPU_DEBUG_MODE_RAY_DIRECTION = 2;
const GPU_DEBUG_MODE_SPLINE_VECTOR = 3;
const ACTIVE_GPU_DEBUG_MODE = GPU_DEBUG_MODE_OFF;
const EXPECTED_GPU_PIPELINE_MODE: 'multi-pass' = 'multi-pass';
const GPU_TEMP_DEBUG_DISPLAY_INVALID_SUPPORT_MAP = false;
const GPU_TEMP_DEBUG_DISPLAY_PEAK_AMBIGUITY_MAP = false;
const GPU_TEMP_DEBUG_DISPLAY_SUPPORT_FAILURE_MAP = false;
const SUPPORT_PATH_ROW_START_FRACTION = 0.14;
const SUPPORT_PATH_ROW_END_FRACTION = 0.76;
const TONE_RESPONSE_BLACK_CLIP_THRESHOLD = 0.02;
const TONE_RESPONSE_WHITE_CLIP_THRESHOLD = 0.98;
const GPU_SUPPORT_MODEL_PARAMS = {
  toothBandCenter: 0.46,
  toothBandInnerHalfWidth: 0.16,
  toothBandOuterHalfWidth: 0.38,
  inferiorPenaltyStart: 0.62,
  inferiorPenaltyEnd: 0.96,
  superiorPenaltyStart: 0.02,
  superiorPenaltyEnd: 0.18,
  densityConfidenceLow: 0.015,
  densityConfidenceHigh: 0.08,
  peakHuConfidenceLow: 150,
  peakHuConfidenceHigh: 900,
  peakDominanceValidityLow: 0.16,
  peakDominanceValidityHigh: 0.46,
  secondPeakRatioPenaltyLow: 0.50,
  secondPeakRatioPenaltyHigh: 0.80,
  secondPeakPenaltyFloor: 0.28,
  peakAmbiguityPenaltyLow: 0.08,
  peakAmbiguityPenaltyHigh: 0.34,
  peakAmbiguityPenaltyFloor: 0.16,
  nonDominantPeakDominanceLow: 0.22,
  nonDominantPeakDominanceHigh: 0.48,
  nonDominantSecondPeakRatioLow: 0.64,
  nonDominantSecondPeakRatioHigh: 0.88,
  nonDominantDensityLow: 0.10,
  nonDominantDensityHigh: 0.28,
  nonDominantDensityPenaltyFloor: 0.03,
  nonDominantConfidencePenaltyFloor: 0.02,
  dominantDensePeakCenterBlendLow: 0.18,
  dominantDensePeakCenterBlendHigh: 0.42,
  dominantDensePeakDensityLow: 0.10,
  dominantDensePeakDensityHigh: 0.34,
  dominantPeakSnapToothGuardLow: 0.04,
  dominantPeakSnapToothGuardHigh: 0.22,
  dominantPeakSnapGuardFloor: 0.12,
  nonDominantPeakSnapGuardFloor: 0.18,
  nonToothDominantValidityPenaltyFloor: 0.74,
  nonToothDominantDensityPenaltyFloor: 0.82,
  dominantDensePeakSpreadShrinkScale: 0.68,
  ambiguousBroadSpreadShrinkScale: 0.84,
  toothBoundarySecondPeakRatioLow: 0.74,
  toothBoundarySecondPeakRatioHigh: 0.94,
  toothBoundaryDensityLow: 0.08,
  toothBoundaryDensityHigh: 0.24,
  toothBoundarySpreadShrinkScale: 0.62,
  toothBoundaryValidityFloor: 0.30,
  toothBoundaryMidpointBlendScale: 0.72,
  toothBoundaryMidpointSpreadScale: 0.58,
  weakToothDensePeakValidityPenaltyFloor: 0.72,
  weakToothDensePeakDensityPenaltyFloor: 0.78,
  enamelBandLowHu: 350,
  enamelBandPeakHu: 550,
  enamelBandHighHu: 1850,
  enamelBandTailHu: 2500,
  inferiorOffsetBoostStartMm: -0.38,
  inferiorOffsetBoostEndMm: -0.04,
  inferiorOffsetBoostScale: 1.22,
  superiorOffsetPenaltyStartMm: 0.02,
  superiorOffsetPenaltyEndMm: 0.26,
  superiorOffsetPenaltyScale: 0.64,
  positiveDepthClampStartMm: 0.02,
  positiveDepthClampEndMm: 0.18,
  positiveDepthClampTargetMm: -0.2,
  positiveDepthClampConfidenceLow: 0.004,
  positiveDepthClampConfidenceHigh: 0.028,
  positiveDepthClampDensityLow: 0.03,
  positiveDepthClampDensityHigh: 0.14,
  continuityNeighborAgreementSigmaMm: 0.28,
  continuityOutlierStartMm: 0.22,
  continuityOutlierEndMm: 0.62,
  continuityBroadSpreadStartMm: 0.30,
  continuityBroadSpreadEndMm: 0.85,
  continuityDensityLow: 0.08,
  continuityDensityMid: 0.40,
  continuityHighDensityStart: 0.72,
  continuityHighDensityEnd: 0.98,
  continuityRegularizerStrength: 0.30,
};
const GPU_DRR_MODEL_PARAMS = {
  broadSigmaSlabScale: 0.34,
  broadSigmaNativePitchScale: 0.9,
  focusedSpreadScale: 0.66,
  focusedNativePitchBias: 0.22,
  focusedSigmaMinNativePitchScale: 0.72,
  focusedSigmaMaxSlabScale: 0.32,
  focusedSigmaMaxNativePitchScale: 0.92,
  confidenceBlendLow: 0.14,
  confidenceBlendHigh: 0.62,
  confidenceSigmaExpansion: 1.04,
  attenuationConfidenceLow: 0.12,
  attenuationConfidenceHigh: 0.58,
  supportWeightPowerLowConfidence: 1.82,
  supportWeightPowerHighConfidence: 1.14,
  lowerPenaltyDenseScale: 0.06,
  lowerPenaltyConfidenceLow: 0.1,
  lowerPenaltyConfidenceHigh: 0.55,
  lowerPenaltyRowStart: 0.62,
  lowerPenaltyRowEnd: 0.96,
  approxTroughBoundarySigmaMultiplier: 1.58,
  backgroundTroughNarrowScale: 0.16,
  lowConfidenceBackgroundContributionScale: 0.12,
  lowConfidenceAirContributionScale: 0.015,
  toothBandBackgroundProtectionFloor: 0.18,
  middleBandLeakContributionScale: 0.18,
  middleBandOffTroughRejectStart: 0.36,
  middleBandOffTroughRejectEnd: 0.78,
  focusedBalancedSlabProtectStartMm: 1.62,
  focusedBalancedSlabProtectEndMm: 1.88,
  focusedBalancedCenterOffsetMm: -4.4,
  focusedBalancedCenterToleranceMm: 0.75,
  focusedBalancedMiddleLeakProtectScale: 0.78,
  focusedBalancedLeakContributionFloor: 0.58,
  dominantToothSigmaShrinkScale: 0.80,
  broadWeakSigmaShrinkScale: 0.78,
  hardOffTroughLeakContributionFloor: 0.03,
  hardOffTroughLeakExtraScale: 0.28,
  trueToothContributionFloor: 0.80,
  toothContinuityAdmissionExpandScale: 1.18,
  toothContinuitySupportWeightFloor: 0.34,
  toothContinuityContributionFloor: 0.58,
  toothBandValidityTroughExpandScale: 1.12,
  toothBoundarySigmaShrinkScale: 0.66,
  residualMiddleLeakProtectScale: 0.52,
  weakValidatedMiddleAccumContributionScale: 0.34,
  toothBandRetentionProtectFloor: 0.28,
  toothBandRetentionParticipationLow: 0.10,
  toothBandRetentionParticipationHigh: 1.35,
  toothBandRetentionConfidenceLow: 0.03,
  toothBandRetentionConfidenceHigh: 0.16,
  toothBandRetentionDensityLow: 0.08,
  toothBandRetentionDensityHigh: 0.24,
  toothBandRetentionSpreadLow: 0.32,
  toothBandRetentionSpreadHigh: 0.82,
  troughSigmaHardCapMm: 0.48,
};
const GPU_TONE_MODEL_PARAMS = {
  lowConfidenceFogSuppression: 0.2,
  highConfidenceFogSuppression: 0.08,
  lowConfidenceFogRetention: 0.75,
  highConfidenceFogRetention: 0.88,
  lowConfidencePenaltySuppression: 0.38,
  highConfidencePenaltySuppression: 0.20,
  lowConfidencePenaltyRetention: 0.72,
  highConfidencePenaltyRetention: 0.86,
  attenuationConfidenceLow: 0.12,
  attenuationConfidenceHigh: 0.58,
  lowConfidenceAttenuationScale: 1.8,
  exposureScale: 6.5,
  sigmoidMidpoint: 0.18,
  sigmoidSlope: 3.8,
  sigmoidWhitePoint: 1.2,
  postCurveGamma: 1.0,
  outputHuMin: -760,
  outputHuMax: 1650,
  invalidSupportBlackoutConfidenceLow: 0.002,
  invalidSupportBlackoutConfidenceHigh: 0.012,
  invalidSupportBlackoutSpreadLow: 0.45,
  invalidSupportBlackoutSpreadHigh: 0.85,
  invalidSupportBlackoutDensityLow: 0.72,
  invalidSupportBlackoutDensityHigh: 0.98,
  invalidSupportBlackoutParticipationLow: 2.2,
  invalidSupportBlackoutParticipationHigh: 4.2,
  invalidSupportBlackoutScale: 0.005,
  focusedBalancedBackgroundProtectScale: 0.92,
  focusedBalancedBlackoutProtectScale: 0.82,
};

// ─── Shaders ─────────────────────────────────────────────────────────
function computeContinuityAdaptiveToleranceMm(spreadMm: number, density: number): number {
  const spreadGate = smoothstepNumber(
    GPU_SUPPORT_MODEL_PARAMS.continuityBroadSpreadStartMm,
    GPU_SUPPORT_MODEL_PARAMS.continuityBroadSpreadEndMm,
    spreadMm
  );
  const densityGate = smoothstepNumber(
    GPU_SUPPORT_MODEL_PARAMS.continuityDensityLow,
    GPU_SUPPORT_MODEL_PARAMS.continuityDensityMid,
    density
  );
  const highDensityGate = smoothstepNumber(
    GPU_SUPPORT_MODEL_PARAMS.continuityHighDensityStart,
    GPU_SUPPORT_MODEL_PARAMS.continuityHighDensityEnd,
    density
  );
  const baseToleranceMm = mixNumber(
    GPU_SUPPORT_MODEL_PARAMS.continuityOutlierStartMm,
    GPU_SUPPORT_MODEL_PARAMS.continuityOutlierEndMm,
    spreadGate * densityGate
  );
  const densityScale = 1 + 0.10 * densityGate + 0.08 * highDensityGate;
  return Math.max(baseToleranceMm * densityScale, 0.04);
}

function computeContinuityExcessJumpMm(
  jumpMm: number,
  spreadMm: number,
  density: number
): number {
  return Math.max(0, jumpMm - computeContinuityAdaptiveToleranceMm(spreadMm, density));
}

function computeContinuityFailureGate(
  jumpMm: number,
  spreadMm: number,
  density: number
): number {
  const excessJumpMm = computeContinuityExcessJumpMm(jumpMm, spreadMm, density);
  return smoothstepNumber(
    0,
    GPU_SUPPORT_MODEL_PARAMS.continuityNeighborAgreementSigmaMm,
    excessJumpMm
  );
}

const VERT_SRC = `#version 300 es
void main() {
  // Fullscreen triangle: 3 vertices cover the clip-space quad
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

const RAY_SHADER_COMMON_GLSL = `
uniform sampler3D uVolume;
uniform sampler2D uSplineData; // width=panoWidth, height=3; row0=pos, row1=slabDir, row2=verticalDir

uniform vec3 uVerticalDir;
uniform float uVertHalfMm;
uniform float uVertCenterOffsetMm;
uniform float uSlabHalfMm;
uniform int uSlabSamples;
uniform int uPanoWidth;
uniform int uPanoHeight;

uniform mat4 uWorldToIndex;
uniform vec3 uDims;

uniform float uRescaleSlope;
uniform float uRescaleIntercept;
uniform bool uApplyRescale;

vec3 encodeDirection(vec3 direction) {
  float directionLength = length(direction);
  if (directionLength <= 1e-5) {
    return vec3(0.5, 0.5, 0.5);
  }
  return clamp((direction / directionLength) * 0.5 + 0.5, 0.0, 1.0);
}

vec3 encodeIndexCoord(vec3 indexCoord) {
  vec3 safeDims = max(uDims, vec3(1.0));
  return clamp((indexCoord + 0.5) / safeDims, 0.0, 1.0);
}

float sampleHu(vec3 uvw) {
  float rawVal = texture(uVolume, uvw).r;
  return uApplyRescale ? rawVal * uRescaleSlope + uRescaleIntercept : rawVal;
}

float computeNativeSlabPitchMm(vec3 slabDirIndexPerMm) {
  float indexUnitsPerMm = length(slabDirIndexPerMm);
  if (indexUnitsPerMm <= 1e-5) {
    return 0.2;
  }
  return clamp(1.0 / indexUnitsPerMm, 0.05, 2.0);
}

int computeRaySampleCount(float slabWidthMm, float nativeSlabPitchMm, int requestedSampleCount) {
  const int MAX_SLAB = 64;
  int pitchAlignedSampleCount = 1;
  if (slabWidthMm > 1e-4) {
    float safeNativePitchMm = max(nativeSlabPitchMm, 1e-4);
    pitchAlignedSampleCount = min(
      MAX_SLAB,
      max(3, int(ceil(slabWidthMm / safeNativePitchMm)) + 1)
    );
  }
  return min(MAX_SLAB, max(max(requestedSampleCount, 1), pitchAlignedSampleCount));
}

vec3 sampleSplineRowLinear(float curveColumnCoord, int splineRow) {
  float safeMaxColumn = max(float(uPanoWidth - 1), 0.0);
  float clampedColumn = clamp(curveColumnCoord, 0.0, safeMaxColumn);
  float leftColumnFloat = floor(clampedColumn);
  int leftColumn = int(leftColumnFloat);
  int rightColumn = min(leftColumn + 1, max(uPanoWidth - 1, 0));
  float blend = clampedColumn - leftColumnFloat;
  vec3 leftValue = texelFetch(uSplineData, ivec2(leftColumn, splineRow), 0).xyz;
  vec3 rightValue = texelFetch(uSplineData, ivec2(rightColumn, splineRow), 0).xyz;
  return mix(leftValue, rightValue, blend);
}

float computeCurveColumnCoord(int col) {
  return clamp(float(col), 0.0, max(float(uPanoWidth - 1), 0.0));
}

void loadRayGeometry(
  float curveColumnCoord,
  int outputRow,
  out vec3 baseWorldPos,
  out vec3 slabDirWorld,
  out vec3 slabDirIndexPerMm,
  out vec3 baseIndex
) {
  int row = (uPanoHeight - 1) - outputRow;
  vec3 pos = sampleSplineRowLinear(curveColumnCoord, 0);
  vec3 slabDir = sampleSplineRowLinear(curveColumnCoord, 1);
  vec3 verticalDir = sampleSplineRowLinear(curveColumnCoord, 2);

  float slabDirLen = length(slabDir);
  slabDir = slabDirLen > 1e-5 ? slabDir / slabDirLen : vec3(0.0, 0.0, 1.0);

  float verticalDirLen = length(verticalDir);
  verticalDir = verticalDirLen > 1e-5 ? verticalDir / verticalDirLen : normalize(uVerticalDir);

  float panoHeightDen = max(1.0, float(uPanoHeight - 1));
  float vertStepMm = (uVertHalfMm * 2.0) / panoHeightDen;
  float vertOffsetMm = uVertCenterOffsetMm + (uVertHalfMm - float(row) * vertStepMm);
  baseWorldPos = pos + vertOffsetMm * verticalDir;
  slabDirWorld = slabDir;
  slabDirIndexPerMm = (uWorldToIndex * vec4(slabDir, 0.0)).xyz;
  baseIndex = (uWorldToIndex * vec4(baseWorldPos, 1.0)).xyz;
}

bool computeSampleUvw(
  vec3 baseWorldPos,
  vec3 slabDirWorld,
  float slabOffset,
  out vec3 uvw
) {
  vec3 sampleWorldPos = baseWorldPos + slabOffset * slabDirWorld;
  vec3 sampleIndex = (uWorldToIndex * vec4(sampleWorldPos, 1.0)).xyz;
  vec3 minIndex = vec3(-0.5);
  vec3 maxIndex = uDims - vec3(0.5);
  if (any(lessThan(sampleIndex, minIndex)) || any(greaterThan(sampleIndex, maxIndex))) {
    return false;
  }
  vec3 clampedIndex = clamp(sampleIndex, vec3(0.0), max(uDims - vec3(1.001), vec3(0.0)));
  uvw = (clampedIndex + vec3(0.5)) / uDims;
  return true;
}
`;

const ATTENUATION_MODEL_GLSL = `
float pseudoAttenuationFromHu(float hu) {
  float softTissue = 0.0065 * smoothstep(-50.0, 150.0, hu);
  float cancellousBone = 0.0220 * smoothstep(-120.0, 420.0, hu);
  float metalRollOff = 1.0 - smoothstep(1100.0, 1900.0, hu);
  float denseBone = 0.0600 * smoothstep(180.0, 1350.0, hu) * metalRollOff;
  float enamel = 0.1700 * smoothstep(850.0, 3200.0, hu) * metalRollOff;
  return softTissue + cancellousBone + denseBone + enamel;
}

float softFogAttenuationFromHu(float hu) {
  float airToSoft = 0.0028 * smoothstep(-950.0, -220.0, hu);
  float softToLowBone = 0.0045 * smoothstep(-220.0, 140.0, hu);
  return airToSoft + softToLowBone;
}

float enamelBandSupportFromHu(float hu) {
  float rise = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.enamelBandLowHu.toFixed(1)},
    ${GPU_SUPPORT_MODEL_PARAMS.enamelBandPeakHu.toFixed(1)},
    hu
  );
  float falloff = 1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.enamelBandHighHu.toFixed(1)},
      ${GPU_SUPPORT_MODEL_PARAMS.enamelBandTailHu.toFixed(1)},
      hu
  );
  return clamp(rise * falloff, 0.0, 1.0);
}

float rootSupportFromHu(float hu) {
  return smoothstep(60.0, 760.0, hu);
}

float dentinSupportFromHu(float hu) {
  return smoothstep(220.0, 1250.0, hu);
}

float toothCoreSupportFromHu(float hu) {
  float rootSupport = rootSupportFromHu(hu);
  float dentinSupport = dentinSupportFromHu(hu);
  return clamp(0.46 * rootSupport + 0.96 * dentinSupport, 0.0, 1.0);
}

float shellDominanceFromHu(float hu) {
  float enamelBandSupport = enamelBandSupportFromHu(hu);
  float toothCoreSupport = toothCoreSupportFromHu(hu);
  return clamp(enamelBandSupport - toothCoreSupport * 0.72, 0.0, 1.0);
}

float denseBiasFromHu(float hu) {
  float metalSuppressSupport = 1.0 - smoothstep(1100.0, 1900.0, hu);
  return smoothstep(500.0, 1800.0, hu) * metalSuppressSupport;
}

float supportResponseFromHu(float hu) {
  float toothCoreSupport = toothCoreSupportFromHu(hu);
  float enamelBandSupport = enamelBandSupportFromHu(hu);
  float shellDominance = shellDominanceFromHu(hu);
  float metalSuppressSupport = 1.0 - smoothstep(1100.0, 1900.0, hu);
  float enamelSupport = smoothstep(950.0, 1800.0, hu) * metalSuppressSupport;
  float denseBias = smoothstep(1100.0, 1800.0, hu) * metalSuppressSupport;
  float combined =
    0.90 * toothCoreSupport +
    0.46 * enamelBandSupport +
    0.16 * enamelSupport;
  combined *= mix(1.0, 0.70, shellDominance);
  return clamp(combined * mix(0.92, 1.06, denseBias), 0.0, 2.0);
}

float denseSupportFromHu(float hu) {
  float toothCoreSupport = toothCoreSupportFromHu(hu);
  float enamelBandSupport = enamelBandSupportFromHu(hu);
  float shellDominance = shellDominanceFromHu(hu);
  float metalSuppressSupport = 1.0 - smoothstep(1100.0, 1900.0, hu);
  float dentinCore = smoothstep(320.0, 1200.0, hu) * metalSuppressSupport;
  float enamelCore = smoothstep(760.0, 1500.0, hu) * metalSuppressSupport;
  float combined =
    0.66 * dentinCore +
    0.22 * toothCoreSupport +
    0.28 * enamelBandSupport +
    0.18 * enamelCore;
  combined *= mix(1.0, 0.74, shellDominance);
  return clamp(combined, 0.0, 1.0);
}
`;

const FOCAL_TROUGH_DRR_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

${RAY_SHADER_COMMON_GLSL}
uniform int uDebugMode;

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(-1000.0, 0.0, 0.0, 1.0);
    return;
  }
  float curveColumnCoord = computeCurveColumnCoord(col);

  vec3 baseWorldPos;
  vec3 slabDirWorld;
  vec3 slabDirIndexPerMm;
  vec3 baseIndex;
  loadRayGeometry(curveColumnCoord, outputRow, baseWorldPos, slabDirWorld, slabDirIndexPerMm, baseIndex);

  if (uDebugMode == 1) {
    fragColor = vec4(encodeIndexCoord(baseIndex), 1.0);
    return;
  }

  if (uDebugMode == 2) {
    fragColor = vec4(encodeDirection(slabDirIndexPerMm), 1.0);
    return;
  }

  if (uDebugMode == 3) {
    int splineRow = min(2, (outputRow * 3) / max(1, uPanoHeight));
    vec3 rawSpline = sampleSplineRowLinear(curveColumnCoord, splineRow);
    vec3 debugColor = splineRow == 0
      ? encodeIndexCoord((uWorldToIndex * vec4(rawSpline, 1.0)).xyz)
      : encodeDirection(rawSpline);
    fragColor = vec4(debugColor, 1.0);
    return;
  }

  const int MAX_SLAB = 64;
  float effectiveSlabHalfMm = max(uSlabHalfMm, 0.0);
  float slabWidthMm = effectiveSlabHalfMm * 2.0;
  float nativeSlabPitchMm = computeNativeSlabPitchMm(slabDirIndexPerMm);
  int raySampleCount = computeRaySampleCount(slabWidthMm, nativeSlabPitchMm, uSlabSamples);
  float slabStep = raySampleCount > 1 ? slabWidthMm / float(raySampleCount - 1) : 0.0;
  const float sigmaMm = 1.5;
  const float troughDenom = 2.0 * sigmaMm * sigmaMm;
  float drrAccum = 0.0;
  float troughWeightSum = 0.0;
  float rayMax = -3.402823e38;
  int validSampleCount = 0;

  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= raySampleCount) {
      break;
    }

    float slabOffset = raySampleCount > 1
      ? -effectiveSlabHalfMm + float(s) * slabStep
      : 0.0;
    vec3 uvw;
    if (!computeSampleUvw(baseWorldPos, slabDirWorld, slabOffset, uvw)) {
      continue;
    }

    float hu = sampleHu(uvw);
    float offsetMm = slabOffset;
    float wTrough = exp(-(offsetMm * offsetMm) / troughDenom);
    drrAccum += hu * wTrough;
    troughWeightSum += wTrough;
    rayMax = max(rayMax, hu);
    validSampleCount++;
  }

  float weightedMeanHu =
    validSampleCount > 0 && troughWeightSum > 1e-5
      ? drrAccum / troughWeightSum
      : -1000.0;
  float finalHu = weightedMeanHu;
  fragColor = vec4(finalHu, weightedMeanHu, rayMax, float(validSampleCount));
}
`;

const SUPPORT_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

${RAY_SHADER_COMMON_GLSL}
${ATTENUATION_MODEL_GLSL}

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(0.0, 0.0, 1.0, 0.0);
    return;
  }
  float curveColumnCoord = computeCurveColumnCoord(col);

  vec3 baseWorldPos;
  vec3 slabDirWorld;
  vec3 slabDirIndexPerMm;
  vec3 baseIndex;
  loadRayGeometry(curveColumnCoord, outputRow, baseWorldPos, slabDirWorld, slabDirIndexPerMm, baseIndex);
  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);
  float toothBandDistance = abs(displayRowNorm - ${GPU_SUPPORT_MODEL_PARAMS.toothBandCenter.toFixed(3)});
  float toothBandPrior =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth.toFixed(3)},
      toothBandDistance
    );
  float inferiorPenalty = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyStart.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyEnd.toFixed(3)},
    displayRowNorm
  );
  float superiorPenalty =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyStart.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyEnd.toFixed(3)},
      displayRowNorm
    );
  float rowPrior = clamp(
    mix(0.84, 1.08, toothBandPrior) *
    mix(1.0, 0.78, inferiorPenalty) *
    mix(0.90, 1.0, 1.0 - superiorPenalty),
    0.72,
    1.12
  );

  float nativeSlabPitchMm = computeNativeSlabPitchMm(slabDirIndexPerMm);
  float slabWidthMm = max(uSlabHalfMm * 2.0, 0.0);
  int raySampleCount = computeRaySampleCount(slabWidthMm, nativeSlabPitchMm, uSlabSamples);
  float slabStep = raySampleCount > 1 ? slabWidthMm / float(raySampleCount - 1) : 0.0;

  float supportMass = 0.0;
  float supportOffsetSum = 0.0;
  float supportOffsetSqSum = 0.0;
  float denseMass = 0.0;
  float bestSupportScore = 0.0;
  float bestSupportOffsetMm = 0.0;
  float top1Score = 0.0;
  float top1OffsetMm = 0.0;
  float top2Score = 0.0;
  float top2OffsetMm = 0.0;
  float peakHu = -1000.0;
  float prevSampleHu = 0.0;
  bool hasPrevSampleHu = false;
  bool hasValidSample = false;

  const int MAX_SLAB = 64;
  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= raySampleCount) {
      break;
    }

    float slabOffset = raySampleCount > 1 ? -uSlabHalfMm + float(s) * slabStep : 0.0;
    vec3 uvw;
    if (!computeSampleUvw(baseWorldPos, slabDirWorld, slabOffset, uvw)) {
      continue;
    }

    float hu = sampleHu(uvw);
    float huGradient = hasPrevSampleHu ? hu - prevSampleHu : 0.0;
    float edgeBonus = smoothstep(0.0, 150.0, huGradient);
    float gradientPenalty = 1.0 - smoothstep(-100.0, 0.0, huGradient);
    float supportResponse = supportResponseFromHu(hu);
    float denseSupport = denseSupportFromHu(hu);
    float enamelBandSupport = enamelBandSupportFromHu(hu);
    float toothCoreSupport = toothCoreSupportFromHu(hu);
    float shellDominance = shellDominanceFromHu(hu);
    float denseBias = denseBiasFromHu(hu);
    float toothCoreGate = smoothstep(0.18, 0.62, toothCoreSupport);
    float toothMaterialGate = clamp(
      mix(0.58, 1.0, toothCoreGate) * mix(1.0, 0.72, shellDominance),
      0.42,
      1.0
    );
    float coreDominance = clamp(
      toothCoreSupport - enamelBandSupport * 0.28,
      0.0,
      1.0
    );
    float inferiorOffsetGate =
      1.0 -
      smoothstep(
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostStartMm.toFixed(2)},
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostEndMm.toFixed(2)},
        slabOffset
      );
    float superiorOffsetGate = smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyStartMm.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyEndMm.toFixed(2)},
      slabOffset
    );
    float offsetPrior = clamp(
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostScale.toFixed(3)}, inferiorOffsetGate) *
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyScale.toFixed(3)}, superiorOffsetGate),
      0.28,
      1.34
    );
    float weightedSupport =
      supportResponse *
      mix(0.76, 1.16, supportResponse) *
      mix(0.90, 1.14, denseBias * toothMaterialGate) *
      mix(0.96, 1.10, enamelBandSupport * toothMaterialGate) *
      mix(0.92, 1.12, toothCoreGate) *
      mix(1.0, 0.68, shellDominance) *
      rowPrior *
      offsetPrior;
    supportMass += weightedSupport;
    supportOffsetSum += slabOffset * weightedSupport;
    supportOffsetSqSum += slabOffset * slabOffset * weightedSupport;
    denseMass += weightedSupport * denseSupport;
    float candidateScore =
      weightedSupport *
      mix(0.92, 1.18, denseBias * toothMaterialGate) *
      mix(0.96, 1.12, enamelBandSupport * toothMaterialGate) *
      mix(0.94, 1.16, coreDominance) *
      mix(1.0, 0.62, superiorOffsetGate);
    candidateScore *= mix(1.0, 0.72, shellDominance * (1.0 - toothBandPrior * 0.35));
    candidateScore *= (1.0 + edgeBonus * 0.7) * (1.0 - gradientPenalty * 0.5);
    float peakCandidateScore =
      candidateScore *
      mix(0.84, 1.10, denseSupport * toothMaterialGate);
    if (candidateScore > bestSupportScore) {
      bestSupportScore = candidateScore;
      bestSupportOffsetMm = slabOffset;
    }
    if (peakCandidateScore > top1Score) {
      top2Score = top1Score;
      top2OffsetMm = top1OffsetMm;
      top1Score = peakCandidateScore;
      top1OffsetMm = slabOffset;
    } else if (peakCandidateScore > top2Score) {
      top2Score = peakCandidateScore;
      top2OffsetMm = slabOffset;
    }
    peakHu = max(peakHu, hu);
    prevSampleHu = hu;
    hasPrevSampleHu = true;
    hasValidSample = true;
  }

  float defaultSpreadMm = max(uSlabHalfMm * 0.75, nativeSlabPitchMm);
  if (!hasValidSample || supportMass <= 1e-5) {
    fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  float supportCenterMm = supportOffsetSum / supportMass;
  float secondMoment = supportOffsetSqSum / supportMass;
  float varianceMm = max(secondMoment - supportCenterMm * supportCenterMm, nativeSlabPitchMm * nativeSlabPitchMm * 0.25);
  float supportSpreadMm = sqrt(varianceMm);
  float peakDominance = top1Score / max(supportMass, 1e-5);
  float secondPeakRatio = top2Score / max(top1Score, 1e-5);
  float peakSeparationMm = abs(top1OffsetMm - top2OffsetMm);
  float rawSupportDensity = denseMass / max(float(raySampleCount), 1.0);
  float secondPeakComparableGate = smoothstep(0.30, 0.72, secondPeakRatio);
  float separatedPeakGate = smoothstep(
    max(nativeSlabPitchMm * 1.2, 0.18),
    max(uSlabHalfMm * 0.30, nativeSlabPitchMm * 2.6),
    peakSeparationMm
  );
  float broadSupportGate = smoothstep(
    max(nativeSlabPitchMm * 1.6, 0.35),
    max(uSlabHalfMm * 0.42, nativeSlabPitchMm * 3.2),
    supportSpreadMm
  );
  float weakDominanceGate = 1.0 - smoothstep(0.18, 0.42, peakDominance);
  float nonDominantBroadSupportGate =
    broadSupportGate *
    (1.0 - smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceHigh.toFixed(2)},
      peakDominance
    )) *
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioHigh.toFixed(2)},
      secondPeakRatio
    ) *
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityHigh.toFixed(2)},
      rawSupportDensity
    );
  float peakAmbiguity =
    secondPeakComparableGate *
    separatedPeakGate *
    broadSupportGate *
    mix(0.35, 1.0, weakDominanceGate);
  float denseSupportFraction = denseMass / max(supportMass, 1e-5);
  float peakDominanceBlendGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakCenterBlendLow.toFixed(2)},
    ${GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakCenterBlendHigh.toFixed(2)},
    peakDominance
  );
  float densePeakBlendGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakDensityLow.toFixed(2)},
    ${GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakDensityHigh.toFixed(2)},
    denseSupportFraction
  );
  float peakHuSupportGate =
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceLow.toFixed(1)},
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceHigh.toFixed(1)},
      peakHu
    );
  float dominantDensePeakGate =
    peakDominanceBlendGate *
    densePeakBlendGate *
    (1.0 - smoothstep(0.12, 0.42, peakAmbiguity));
  float toothBandStructureGuard =
    toothBandPrior *
    peakHuSupportGate *
    smoothstep(0.04, 0.16, rawSupportDensity) *
    max(
      smoothstep(0.08, 0.24, peakDominance),
      smoothstep(0.10, 0.28, denseSupportFraction)
    );
  float compactToothBandStructureGuard =
    toothBandPrior *
    peakHuSupportGate *
    smoothstep(0.06, 0.20, rawSupportDensity) *
    max(
      smoothstep(0.10, 0.28, peakDominance),
      smoothstep(0.12, 0.32, denseSupportFraction)
    ) *
    (1.0 - broadSupportGate);
  float toothBandPeakValidityGuard = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapToothGuardLow.toFixed(2)},
    ${GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapToothGuardHigh.toFixed(2)},
    max(toothBandStructureGuard, compactToothBandStructureGuard)
  );
  float guardedDominantDensePeakGate =
    dominantDensePeakGate *
    mix(
      ${GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapGuardFloor.toFixed(2)},
      1.0,
      toothBandPeakValidityGuard
    );
  float weakToothDensePeakGate =
    toothBandPrior *
    dominantDensePeakGate *
    smoothstep(0.06, 0.18, rawSupportDensity) *
    (1.0 - toothBandPeakValidityGuard) *
    (1.0 - smoothstep(0.18, 0.40, toothBandStructureGuard));
  float ambiguousBroadSupportPenaltyGate =
    broadSupportGate *
    secondPeakComparableGate *
    (1.0 - dominantDensePeakGate) *
    mix(0.45, 1.0, weakDominanceGate);
  float protectedAmbiguousBroadSupportPenaltyGate =
    ambiguousBroadSupportPenaltyGate *
    (1.0 - toothBandStructureGuard * 0.72);
  float guardedPeakCenterBlendGate =
    smoothstep(0.12, 0.48, peakDominance) *
    mix(
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakSnapGuardFloor.toFixed(2)},
      1.0,
      toothBandPeakValidityGuard
    );
  float guardedNonDominantBroadSupportGate =
    nonDominantBroadSupportGate *
    mix(
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakSnapGuardFloor.toFixed(2)},
      1.0,
      toothBandPeakValidityGuard
    );
  supportCenterMm = mix(
    supportCenterMm,
    top1OffsetMm,
    max(
      max(guardedPeakCenterBlendGate, guardedDominantDensePeakGate),
      guardedNonDominantBroadSupportGate * 0.72
    )
  );
  supportSpreadMm = mix(
    supportSpreadMm,
    max(
      nativeSlabPitchMm * 0.90,
      supportSpreadMm *
        mix(
          ${GPU_SUPPORT_MODEL_PARAMS.ambiguousBroadSpreadShrinkScale.toFixed(2)},
          ${GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakSpreadShrinkScale.toFixed(2)},
          guardedDominantDensePeakGate
        )
    ),
    max(
      guardedDominantDensePeakGate,
      max(guardedNonDominantBroadSupportGate * 0.48, ambiguousBroadSupportPenaltyGate * 0.60)
    )
  );
  float structuralSupportGate = max(
    smoothstep(0.28, 0.62, denseSupportFraction),
    smoothstep(0.14, 0.42, peakDominance)
  );
  float spreadConfidence =
    1.0 - smoothstep(
      nativeSlabPitchMm * 1.5,
      max(uSlabHalfMm * 0.75, nativeSlabPitchMm * 2.8),
      supportSpreadMm
    );
  float broadWeakSupportGate =
    smoothstep(
      nativeSlabPitchMm * 1.8,
      max(uSlabHalfMm * 0.46, nativeSlabPitchMm * 3.4),
      supportSpreadMm
    ) *
    (1.0 - structuralSupportGate);
  float protectedBroadWeakSupportGate =
    broadWeakSupportGate *
    (1.0 - toothBandStructureGuard * 0.58);
  float toothBoundaryAmbiguityGate =
    toothBandStructureGuard *
    peakHuSupportGate *
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundaryDensityLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundaryDensityHigh.toFixed(2)},
      rawSupportDensity
    ) *
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundarySecondPeakRatioLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundarySecondPeakRatioHigh.toFixed(2)},
      secondPeakRatio
    ) *
    broadSupportGate *
    (1.0 - smoothstep(0.22, 0.48, peakDominance));
  float toothBoundaryMidpointGate =
    toothBoundaryAmbiguityGate *
    smoothstep(
      max(nativeSlabPitchMm * 1.4, 0.22),
      max(uSlabHalfMm * 0.26, nativeSlabPitchMm * 2.4),
      peakSeparationMm
    );
  supportCenterMm = mix(
    supportCenterMm,
    0.5 * (top1OffsetMm + top2OffsetMm),
    toothBoundaryMidpointGate *
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundaryMidpointBlendScale.toFixed(2)}
  );
  supportSpreadMm = mix(
    supportSpreadMm,
    max(
      nativeSlabPitchMm * 0.82,
      supportSpreadMm *
        ${GPU_SUPPORT_MODEL_PARAMS.toothBoundarySpreadShrinkScale.toFixed(2)}
    ),
    toothBoundaryAmbiguityGate
  );
  supportSpreadMm = mix(
    supportSpreadMm,
    min(
      supportSpreadMm,
      max(
        nativeSlabPitchMm * 0.78,
        peakSeparationMm *
          ${GPU_SUPPORT_MODEL_PARAMS.toothBoundaryMidpointSpreadScale.toFixed(2)}
      )
    ),
    toothBoundaryMidpointGate
  );
  float peakDominanceValidityGate =
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityHigh.toFixed(2)},
      peakDominance
    );
  float competingPeakPenalty =
    mix(
      1.0,
      ${GPU_SUPPORT_MODEL_PARAMS.secondPeakPenaltyFloor.toFixed(2)},
      smoothstep(
        ${GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyLow.toFixed(2)},
        ${GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyHigh.toFixed(2)},
        secondPeakRatio
      )
    );
  float ambiguityPenalty =
    mix(
      1.0,
      ${GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyFloor.toFixed(2)},
      smoothstep(
        ${GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyLow.toFixed(2)},
        ${GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyHigh.toFixed(2)},
        peakAmbiguity
      )
    );
  float peakStructureValidity =
    peakDominanceValidityGate *
    competingPeakPenalty *
    ambiguityPenalty;
  float nonToothDominantPeakGate =
    dominantDensePeakGate *
    (1.0 - toothBandPeakValidityGuard);
  float supportValidity =
    structuralSupportGate *
    peakHuSupportGate *
    clamp(spreadConfidence, 0.0, 1.0) *
    peakStructureValidity *
    mix(
      1.0,
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantConfidencePenaltyFloor.toFixed(2)},
      nonDominantBroadSupportGate
    );
  supportValidity *= mix(1.0, 0.52, protectedAmbiguousBroadSupportPenaltyGate);
  supportValidity *= mix(
    1.0,
    ${GPU_SUPPORT_MODEL_PARAMS.nonToothDominantValidityPenaltyFloor.toFixed(2)},
    nonToothDominantPeakGate
  );
  supportValidity *= mix(
    1.0,
    ${GPU_SUPPORT_MODEL_PARAMS.weakToothDensePeakValidityPenaltyFloor.toFixed(2)},
    weakToothDensePeakGate
  );
  supportValidity = max(
    supportValidity,
    toothBandStructureGuard * mix(0.22, 0.34, 1.0 - broadSupportGate)
  );
  supportValidity = max(
    supportValidity,
    max(toothBoundaryAmbiguityGate, toothBoundaryMidpointGate) *
      toothBandStructureGuard *
      ${GPU_SUPPORT_MODEL_PARAMS.toothBoundaryValidityFloor.toFixed(2)}
  );
  float supportDensity =
    rawSupportDensity *
    mix(0.10, 1.0, structuralSupportGate) *
    mix(1.0, 0.36, protectedBroadWeakSupportGate) *
    mix(0.18, 1.0, supportValidity) *
    mix(
      1.0,
      ${GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityPenaltyFloor.toFixed(2)},
      nonDominantBroadSupportGate
    );
  supportDensity *= mix(1.0, 0.40, protectedAmbiguousBroadSupportPenaltyGate);
  supportDensity *= mix(
    1.0,
    ${GPU_SUPPORT_MODEL_PARAMS.nonToothDominantDensityPenaltyFloor.toFixed(2)},
    nonToothDominantPeakGate
  );
  supportDensity *= mix(
    1.0,
    ${GPU_SUPPORT_MODEL_PARAMS.weakToothDensePeakDensityPenaltyFloor.toFixed(2)},
    weakToothDensePeakGate
  );
  supportDensity = max(
    supportDensity,
    rawSupportDensity * toothBandStructureGuard * 0.46
  );
  float rowConfidenceGate = clamp(
    mix(0.72, 1.14, toothBandPrior) *
    mix(1.0, 0.50, inferiorPenalty) *
    mix(0.88, 1.0, 1.0 - superiorPenalty),
    0.26,
    1.18
  );
  rowConfidenceGate = clamp(
    rowConfidenceGate * mix(1.0, 1.08, toothBandStructureGuard),
    0.26,
    1.22
  );
  float supportConfidence =
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.densityConfidenceLow.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.densityConfidenceHigh.toFixed(3)},
      supportDensity
    ) *
    supportValidity *
    rowConfidenceGate;
	  float falseSupportConfidenceGate =
	    1.0 - smoothstep(0.004, 0.016, supportConfidence);
	  float falseSupportDensityGate = smoothstep(0.04, 0.12, rawSupportDensity);
	  float falseSupportSpreadGate =
	    smoothstep(
	      max(nativeSlabPitchMm * 2.0, 0.45),
	      max(uSlabHalfMm * 0.42, nativeSlabPitchMm * 4.0),
	      supportSpreadMm
	    );
	  float falseSupportStructureWeakGate =
	    1.0 - smoothstep(0.16, 0.40, structuralSupportGate);
	  float falseSupportVeto =
	    falseSupportConfidenceGate *
	    falseSupportDensityGate *
	    falseSupportSpreadGate *
	    falseSupportStructureWeakGate;
	  float nonToothBandGate =
	    1.0 -
	    smoothstep(
	      0.06,
	      0.18,
	      toothBandPrior
	    );
	  float rowBackgroundDensityGate = smoothstep(0.04, 0.16, rawSupportDensity);
	  float rowBackgroundSpreadGate =
	    smoothstep(
	      max(nativeSlabPitchMm * 1.6, 0.34),
	      max(uSlabHalfMm * 0.34, nativeSlabPitchMm * 3.2),
	      supportSpreadMm
	    );
	  float rowBackgroundPeakHuGate = 1.0 - peakHuSupportGate;
	  float rowBackgroundEdgeGate = max(
	    1.0 - smoothstep(0.16, 0.32, displayRowNorm),
	    smoothstep(0.78, 0.94, displayRowNorm)
	  );
	  float rowBackgroundVeto =
	    nonToothBandGate *
	    falseSupportConfidenceGate *
	    rowBackgroundDensityGate *
	    rowBackgroundSpreadGate *
      mix(0.58, 1.0, rowBackgroundPeakHuGate) *
      mix(0.90, 1.28, rowBackgroundEdgeGate);
	  float vetoThreshold = mix(
	    0.46,
	    0.24,
	    1.0 - smoothstep(0.36, 0.78, peakDominance)
	  );
	  if (max(falseSupportVeto, rowBackgroundVeto) > vetoThreshold) {
	    fragColor = vec4(supportCenterMm, 0.0, 0.0, 0.0);
	    return;
	  }

  fragColor = vec4(
    supportCenterMm,
    clamp(supportConfidence, 0.0, 1.0),
    clamp(supportSpreadMm, nativeSlabPitchMm * 0.75, max(uSlabHalfMm, nativeSlabPitchMm)),
    clamp(supportDensity, 0.0, 1.0)
  );
}
`;

const SUPPORT_PEAK_DEBUG_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

${RAY_SHADER_COMMON_GLSL}
${ATTENUATION_MODEL_GLSL}

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(0.0);
    return;
  }

  float curveColumnCoord = computeCurveColumnCoord(col);
  vec3 baseWorldPos;
  vec3 slabDirWorld;
  vec3 slabDirIndexPerMm;
  vec3 baseIndex;
  loadRayGeometry(curveColumnCoord, outputRow, baseWorldPos, slabDirWorld, slabDirIndexPerMm, baseIndex);

  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);
  float toothBandDistance = abs(displayRowNorm - ${GPU_SUPPORT_MODEL_PARAMS.toothBandCenter.toFixed(3)});
  float toothBandPrior =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth.toFixed(3)},
      toothBandDistance
    );
  float inferiorPenalty = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyStart.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyEnd.toFixed(3)},
    displayRowNorm
  );
  float superiorPenalty =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyStart.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyEnd.toFixed(3)},
      displayRowNorm
    );
  float rowPrior = clamp(
    mix(0.84, 1.08, toothBandPrior) *
    mix(1.0, 0.78, inferiorPenalty) *
    mix(0.90, 1.0, 1.0 - superiorPenalty),
    0.72,
    1.12
  );

  float nativeSlabPitchMm = computeNativeSlabPitchMm(slabDirIndexPerMm);
  float slabWidthMm = max(uSlabHalfMm * 2.0, 0.0);
  int raySampleCount = computeRaySampleCount(slabWidthMm, nativeSlabPitchMm, uSlabSamples);
  float slabStep = raySampleCount > 1 ? slabWidthMm / float(raySampleCount - 1) : 0.0;

  float supportMass = 0.0;
  float supportOffsetSum = 0.0;
  float supportOffsetSqSum = 0.0;
  float top1Score = 0.0;
  float top1OffsetMm = 0.0;
  float top2Score = 0.0;
  float top2OffsetMm = 0.0;
  float prevSampleHu = 0.0;
  bool hasPrevSampleHu = false;
  bool hasValidSample = false;

  const int MAX_SLAB = 64;
  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= raySampleCount) {
      break;
    }

    float slabOffset = raySampleCount > 1 ? -uSlabHalfMm + float(s) * slabStep : 0.0;
    vec3 uvw;
    if (!computeSampleUvw(baseWorldPos, slabDirWorld, slabOffset, uvw)) {
      continue;
    }

    float hu = sampleHu(uvw);
    float huGradient = hasPrevSampleHu ? hu - prevSampleHu : 0.0;
    float edgeBonus = smoothstep(0.0, 150.0, huGradient);
    float gradientPenalty = 1.0 - smoothstep(-100.0, 0.0, huGradient);
    float supportResponse = supportResponseFromHu(hu);
    float denseSupport = denseSupportFromHu(hu);
    float enamelBandSupport = enamelBandSupportFromHu(hu);
    float toothCoreSupport = toothCoreSupportFromHu(hu);
    float shellDominance = shellDominanceFromHu(hu);
    float denseBias = denseBiasFromHu(hu);
    float toothCoreGate = smoothstep(0.18, 0.62, toothCoreSupport);
    float toothMaterialGate = clamp(
      mix(0.58, 1.0, toothCoreGate) * mix(1.0, 0.72, shellDominance),
      0.42,
      1.0
    );
    float coreDominance = clamp(
      toothCoreSupport - enamelBandSupport * 0.28,
      0.0,
      1.0
    );
    float inferiorOffsetGate =
      1.0 -
      smoothstep(
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostStartMm.toFixed(2)},
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostEndMm.toFixed(2)},
        slabOffset
      );
    float superiorOffsetGate = smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyStartMm.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyEndMm.toFixed(2)},
      slabOffset
    );
    float offsetPrior = clamp(
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostScale.toFixed(3)}, inferiorOffsetGate) *
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyScale.toFixed(3)}, superiorOffsetGate),
      0.28,
      1.34
    );
    float weightedSupport =
      supportResponse *
      mix(0.76, 1.16, supportResponse) *
      mix(0.90, 1.14, denseBias * toothMaterialGate) *
      mix(0.96, 1.10, enamelBandSupport * toothMaterialGate) *
      mix(0.92, 1.12, toothCoreGate) *
      mix(1.0, 0.68, shellDominance) *
      rowPrior *
      offsetPrior;
    supportMass += weightedSupport;
    supportOffsetSum += slabOffset * weightedSupport;
    supportOffsetSqSum += slabOffset * slabOffset * weightedSupport;

    float candidateScore =
      weightedSupport *
      mix(0.92, 1.18, denseBias * toothMaterialGate) *
      mix(0.96, 1.12, enamelBandSupport * toothMaterialGate) *
      mix(0.94, 1.16, coreDominance) *
      mix(1.0, 0.62, superiorOffsetGate);
    candidateScore *= mix(1.0, 0.72, shellDominance * (1.0 - toothBandPrior * 0.35));
    candidateScore *= (1.0 + edgeBonus * 0.7) * (1.0 - gradientPenalty * 0.5);
    candidateScore *= mix(0.84, 1.10, denseSupport * toothMaterialGate);

    if (candidateScore > top1Score) {
      top2Score = top1Score;
      top2OffsetMm = top1OffsetMm;
      top1Score = candidateScore;
      top1OffsetMm = slabOffset;
    } else if (candidateScore > top2Score) {
      top2Score = candidateScore;
      top2OffsetMm = slabOffset;
    }

    prevSampleHu = hu;
    hasPrevSampleHu = true;
    hasValidSample = true;
  }

  if (!hasValidSample || supportMass <= 1e-5 || top1Score <= 1e-6) {
    fragColor = vec4(0.0);
    return;
  }

  float supportCenterMm = supportOffsetSum / supportMass;
  float secondMoment = supportOffsetSqSum / supportMass;
  float varianceMm =
    max(secondMoment - supportCenterMm * supportCenterMm, nativeSlabPitchMm * nativeSlabPitchMm * 0.25);
  float supportSpreadMm = sqrt(varianceMm);
  float peakDominance = top1Score / max(supportMass, 1e-5);
  float secondPeakRatio = top2Score / max(top1Score, 1e-5);
  float peakSeparationMm = abs(top1OffsetMm - top2OffsetMm);

  float secondPeakComparableGate = smoothstep(0.30, 0.72, secondPeakRatio);
  float separatedPeakGate = smoothstep(
    max(nativeSlabPitchMm * 1.2, 0.18),
    max(uSlabHalfMm * 0.30, nativeSlabPitchMm * 2.6),
    peakSeparationMm
  );
  float broadSupportGate = smoothstep(
    max(nativeSlabPitchMm * 1.6, 0.35),
    max(uSlabHalfMm * 0.42, nativeSlabPitchMm * 3.2),
    supportSpreadMm
  );
  float weakDominanceGate = 1.0 - smoothstep(0.18, 0.42, peakDominance);
  float peakAmbiguity =
    secondPeakComparableGate *
    separatedPeakGate *
    broadSupportGate *
    mix(0.35, 1.0, weakDominanceGate);

  fragColor = vec4(
    clamp(peakDominance, 0.0, 1.0),
    clamp(secondPeakRatio, 0.0, 1.0),
    max(peakSeparationMm, 0.0),
    clamp(peakAmbiguity, 0.0, 1.0)
  );
}
`;

const SUPPORT_MODEL_DIAGNOSTIC_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

${RAY_SHADER_COMMON_GLSL}
${ATTENUATION_MODEL_GLSL}

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(0.0);
    return;
  }

  float curveColumnCoord = computeCurveColumnCoord(col);
  vec3 baseWorldPos;
  vec3 slabDirWorld;
  vec3 slabDirIndexPerMm;
  vec3 baseIndex;
  loadRayGeometry(curveColumnCoord, outputRow, baseWorldPos, slabDirWorld, slabDirIndexPerMm, baseIndex);

  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);
  float toothBandDistance = abs(displayRowNorm - ${GPU_SUPPORT_MODEL_PARAMS.toothBandCenter.toFixed(3)});
  float toothBandPrior =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth.toFixed(3)},
      toothBandDistance
    );
  float inferiorPenalty = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyStart.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyEnd.toFixed(3)},
    displayRowNorm
  );
  float superiorPenalty =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyStart.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyEnd.toFixed(3)},
      displayRowNorm
    );
  float rowPrior = clamp(
    mix(0.84, 1.08, toothBandPrior) *
    mix(1.0, 0.78, inferiorPenalty) *
    mix(0.90, 1.0, 1.0 - superiorPenalty),
    0.72,
    1.12
  );

  float nativeSlabPitchMm = computeNativeSlabPitchMm(slabDirIndexPerMm);
  float slabWidthMm = max(uSlabHalfMm * 2.0, 0.0);
  int raySampleCount = computeRaySampleCount(slabWidthMm, nativeSlabPitchMm, uSlabSamples);
  float slabStep = raySampleCount > 1 ? slabWidthMm / float(raySampleCount - 1) : 0.0;

  float supportMass = 0.0;
  float denseMass = 0.0;
  float top1Score = 0.0;
  float top1OffsetMm = 0.0;
  float top2Score = 0.0;
  float top2OffsetMm = 0.0;
  float peakHu = -1000.0;
  float prevSampleHu = 0.0;
  bool hasPrevSampleHu = false;
  bool hasValidSample = false;

  const int MAX_SLAB = 64;
  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= raySampleCount) {
      break;
    }

    float slabOffset = raySampleCount > 1 ? -uSlabHalfMm + float(s) * slabStep : 0.0;
    vec3 uvw;
    if (!computeSampleUvw(baseWorldPos, slabDirWorld, slabOffset, uvw)) {
      continue;
    }

    float hu = sampleHu(uvw);
    float huGradient = hasPrevSampleHu ? hu - prevSampleHu : 0.0;
    float edgeBonus = smoothstep(0.0, 150.0, huGradient);
    float gradientPenalty = 1.0 - smoothstep(-100.0, 0.0, huGradient);
    float supportResponse = supportResponseFromHu(hu);
    float denseSupport = denseSupportFromHu(hu);
    float enamelBandSupport = enamelBandSupportFromHu(hu);
    float toothCoreSupport = toothCoreSupportFromHu(hu);
    float shellDominance = shellDominanceFromHu(hu);
    float denseBias = denseBiasFromHu(hu);
    float toothCoreGate = smoothstep(0.18, 0.62, toothCoreSupport);
    float toothMaterialGate = clamp(
      mix(0.58, 1.0, toothCoreGate) * mix(1.0, 0.72, shellDominance),
      0.42,
      1.0
    );
    float coreDominance = clamp(
      toothCoreSupport - enamelBandSupport * 0.28,
      0.0,
      1.0
    );
    float inferiorOffsetGate =
      1.0 -
      smoothstep(
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostStartMm.toFixed(2)},
        ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostEndMm.toFixed(2)},
        slabOffset
      );
    float superiorOffsetGate = smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyStartMm.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyEndMm.toFixed(2)},
      slabOffset
    );
    float offsetPrior = clamp(
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.inferiorOffsetBoostScale.toFixed(3)}, inferiorOffsetGate) *
      mix(1.0, ${GPU_SUPPORT_MODEL_PARAMS.superiorOffsetPenaltyScale.toFixed(3)}, superiorOffsetGate),
      0.28,
      1.34
    );
    float weightedSupport =
      supportResponse *
      mix(0.76, 1.16, supportResponse) *
      mix(0.90, 1.14, denseBias * toothMaterialGate) *
      mix(0.96, 1.10, enamelBandSupport * toothMaterialGate) *
      mix(0.92, 1.12, toothCoreGate) *
      mix(1.0, 0.68, shellDominance) *
      rowPrior *
      offsetPrior;
    supportMass += weightedSupport;
    denseMass += weightedSupport * denseSupport;

    float candidateScore =
      weightedSupport *
      mix(0.92, 1.18, denseBias * toothMaterialGate) *
      mix(0.96, 1.12, enamelBandSupport * toothMaterialGate) *
      mix(0.94, 1.16, coreDominance) *
      mix(1.0, 0.62, superiorOffsetGate);
    candidateScore *= mix(1.0, 0.72, shellDominance * (1.0 - toothBandPrior * 0.35));
    candidateScore *= (1.0 + edgeBonus * 0.7) * (1.0 - gradientPenalty * 0.5);
    float peakCandidateScore = candidateScore * mix(0.84, 1.10, denseSupport * toothMaterialGate);

    if (peakCandidateScore > top1Score) {
      top2Score = top1Score;
      top2OffsetMm = top1OffsetMm;
      top1Score = peakCandidateScore;
      top1OffsetMm = slabOffset;
    } else if (peakCandidateScore > top2Score) {
      top2Score = peakCandidateScore;
      top2OffsetMm = slabOffset;
    }

    peakHu = max(peakHu, hu);
    prevSampleHu = hu;
    hasPrevSampleHu = true;
    hasValidSample = true;
  }

  if (!hasValidSample || supportMass <= 1e-5) {
    fragColor = vec4(0.0);
    return;
  }

  float denseSupportFraction = denseMass / max(supportMass, 1e-5);
  float peakHuSupportGate =
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceLow.toFixed(1)},
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceHigh.toFixed(1)},
      peakHu
    );

  fragColor = vec4(
    clamp(denseSupportFraction, 0.0, 1.0),
    clamp(peakHuSupportGate, 0.0, 1.0),
    top1OffsetMm,
    top2OffsetMm
  );
}
`;

const SUPPORT_SMOOTH_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uSupportData;
uniform int uPanoWidth;
uniform int uPanoHeight;

out vec4 fragColor;

float continuityAdaptiveToleranceMm(float spreadMm, float density) {
  float spreadGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.continuityBroadSpreadStartMm.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.continuityBroadSpreadEndMm.toFixed(3)},
    spreadMm
  );
  float densityGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.continuityDensityLow.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.continuityDensityMid.toFixed(3)},
    density
  );
  float highDensityGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.continuityHighDensityStart.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.continuityHighDensityEnd.toFixed(3)},
    density
  );
  float baseToleranceMm = mix(
    ${GPU_SUPPORT_MODEL_PARAMS.continuityOutlierStartMm.toFixed(3)},
    ${GPU_SUPPORT_MODEL_PARAMS.continuityOutlierEndMm.toFixed(3)},
    spreadGate * densityGate
  );
  float densityScale = 1.0 + 0.10 * densityGate + 0.08 * highDensityGate;
  return max(baseToleranceMm * densityScale, 0.04);
}

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(0.0);
    return;
  }

  ivec2 centerCoord = ivec2(col, outputRow);
  vec4 centerSample = texelFetch(uSupportData, centerCoord, 0);
  float centerDepthMm = centerSample.r;
  float centerConfidence = clamp(centerSample.g, 0.0, 1.0);
  float centerDensity = clamp(centerSample.a, 0.0, 1.0);
  float centerSpreadMm = max(centerSample.b, 0.0);
  float centerSupportStrength = max(centerConfidence, centerDensity * 0.35);

  float rawHorizontalJumpSum = 0.0;
  float rawHorizontalJumpCount = 0.0;
  float immediateNeighborDepthSum = 0.0;
  float immediateNeighborConfidenceSum = 0.0;
  float immediateNeighborWeightSum = 0.0;

  if (col > 0) {
    vec4 leftSample = texelFetch(uSupportData, ivec2(col - 1, outputRow), 0);
    float leftConfidence = clamp(leftSample.g, 0.0, 1.0);
    float leftDensity = clamp(leftSample.a, 0.0, 1.0);
    float leftSupportStrength = max(leftConfidence, leftDensity * 0.35);
    if (leftSupportStrength > 1e-4) {
      rawHorizontalJumpSum += abs(leftSample.r - centerDepthMm);
      rawHorizontalJumpCount += 1.0;
      float immediateNeighborWeight =
        sqrt(max(leftSupportStrength * max(centerSupportStrength, 1e-4), 0.0));
      immediateNeighborDepthSum += leftSample.r * immediateNeighborWeight;
      immediateNeighborConfidenceSum += leftConfidence * immediateNeighborWeight;
      immediateNeighborWeightSum += immediateNeighborWeight;
    }
  }

  if (col + 1 < uPanoWidth) {
    vec4 rightSample = texelFetch(uSupportData, ivec2(col + 1, outputRow), 0);
    float rightConfidence = clamp(rightSample.g, 0.0, 1.0);
    float rightDensity = clamp(rightSample.a, 0.0, 1.0);
    float rightSupportStrength = max(rightConfidence, rightDensity * 0.35);
    if (rightSupportStrength > 1e-4) {
      rawHorizontalJumpSum += abs(rightSample.r - centerDepthMm);
      rawHorizontalJumpCount += 1.0;
      float immediateNeighborWeight =
        sqrt(max(rightSupportStrength * max(centerSupportStrength, 1e-4), 0.0));
      immediateNeighborDepthSum += rightSample.r * immediateNeighborWeight;
      immediateNeighborConfidenceSum += rightConfidence * immediateNeighborWeight;
      immediateNeighborWeightSum += immediateNeighborWeight;
    }
  }

  float rawLocalJumpMm =
    rawHorizontalJumpCount > 0.0 ? rawHorizontalJumpSum / rawHorizontalJumpCount : 0.0;
  float rawContinuityExcessMm =
    max(0.0, rawLocalJumpMm - continuityAdaptiveToleranceMm(centerSpreadMm, centerDensity));
  float rawContinuityFailureGate = smoothstep(
    0.0,
    ${GPU_SUPPORT_MODEL_PARAMS.continuityNeighborAgreementSigmaMm.toFixed(3)},
    rawContinuityExcessMm
  );
  float continuityRegularizerBlend =
    rawContinuityFailureGate *
    ${GPU_SUPPORT_MODEL_PARAMS.continuityRegularizerStrength.toFixed(3)} *
    (1.0 - smoothstep(0.18, 0.48, centerSupportStrength));
  float centerWeight = mix(
    2.2 + centerSupportStrength * 3.2,
    1.4 + centerSupportStrength * 2.2,
    continuityRegularizerBlend
  );
  float weightedCenter = centerSample.r * centerWeight;
  float weightedConfidence = centerConfidence * centerWeight;
  float weightedSpread = centerSample.b * centerWeight;
  float weightedDensity = centerDensity * centerWeight;
  float weightSum = centerWeight;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -4; dx <= 4; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      ivec2 sampleCoord = ivec2(
        clamp(col + dx, 0, max(uPanoWidth - 1, 0)),
        clamp(outputRow + dy, 0, max(uPanoHeight - 1, 0))
      );
      vec4 sampleValue = texelFetch(uSupportData, sampleCoord, 0);
      float confidence = clamp(sampleValue.g, 0.0, 1.0);
      float sampleDensity = clamp(sampleValue.a, 0.0, 1.0);
      float sampleSupportStrength = max(confidence, sampleDensity * 0.35);
      if (sampleSupportStrength <= 1e-4) {
        continue;
      }
      float spatialWeight =
        exp(-0.5 * (float(dx * dx) / 1.35 + float(dy * dy) / 0.80));
      float depthDelta = sampleValue.r - centerDepthMm;
      float depthWeight = exp(-(depthDelta * depthDelta) / (2.0 * 0.55 * 0.55));
      float confidenceDelta = sampleSupportStrength - centerSupportStrength;
      float confidenceSigma = mix(0.42, 0.16, centerConfidence);
      float confidenceWeight =
        exp(-(confidenceDelta * confidenceDelta) / (2.0 * confidenceSigma * confidenceSigma));
      float sampleWeight =
        spatialWeight *
        depthWeight *
        confidenceWeight *
        sqrt(max(sampleSupportStrength * centerSupportStrength, 0.0));

      weightedCenter += sampleValue.r * sampleWeight;
      weightedConfidence += confidence * sampleWeight;
      weightedSpread += sampleValue.b * sampleWeight;
      weightedDensity += sampleDensity * sampleWeight;
      weightSum += sampleWeight;
    }
  }

  if (weightSum <= 1e-5) {
    fragColor = centerSample;
    return;
  }

  float smoothedCenterMm = weightedCenter / weightSum;
  float smoothedConfidence = clamp(weightedConfidence / weightSum, 0.0, 1.0);
  float smoothedSpreadMm = max(weightedSpread / weightSum, 0.1);
  float smoothedDensity = clamp(weightedDensity / weightSum, 0.0, 1.0);
  if (immediateNeighborWeightSum > 1e-4 && continuityRegularizerBlend > 1e-4) {
    float immediateNeighborDepthMm = immediateNeighborDepthSum / immediateNeighborWeightSum;
    float immediateNeighborConfidence = clamp(
      immediateNeighborConfidenceSum / immediateNeighborWeightSum,
      0.0,
      1.0
    );
    smoothedCenterMm = mix(smoothedCenterMm, immediateNeighborDepthMm, continuityRegularizerBlend);
    smoothedConfidence = mix(
      smoothedConfidence,
      min(smoothedConfidence, immediateNeighborConfidence),
      continuityRegularizerBlend * 0.35
    );
  }
  if (smoothedConfidence <= 1e-5 && smoothedDensity <= 1e-5) {
    fragColor = vec4(smoothedCenterMm, 0.0, 0.0, 0.0);
    return;
  }
  float positiveDepthGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampStartMm.toFixed(2)},
    ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampEndMm.toFixed(2)},
    smoothedCenterMm
  );
	  float lowConfidenceGate =
	    1.0 -
	    smoothstep(
	      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampConfidenceLow.toFixed(3)},
	      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampConfidenceHigh.toFixed(3)},
	      smoothedConfidence
	    );
	  float lowDensityGate =
	    1.0 -
	    smoothstep(
	      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampDensityLow.toFixed(2)},
	      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampDensityHigh.toFixed(2)},
	      smoothedDensity
	    );
	  float broadPositiveSupportGate = smoothstep(0.22, 0.70, smoothedSpreadMm);
	  float positiveDepthCorrection =
	    positiveDepthGate *
	    lowConfidenceGate *
	    lowDensityGate *
	    broadPositiveSupportGate;
	  smoothedCenterMm = mix(
	    smoothedCenterMm,
	    ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampTargetMm.toFixed(2)},
	    positiveDepthCorrection
  );
  smoothedSpreadMm = mix(smoothedSpreadMm, min(smoothedSpreadMm, 0.55), positiveDepthCorrection * 0.55);

  fragColor = vec4(
    smoothedCenterMm,
    smoothedConfidence,
    smoothedSpreadMm,
    smoothedDensity
  );
}
`;

const DRR_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

${RAY_SHADER_COMMON_GLSL}
${ATTENUATION_MODEL_GLSL}

uniform sampler2D uSupportData;

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(0.0);
    return;
  }
  float curveColumnCoord = computeCurveColumnCoord(col);

  vec3 baseWorldPos;
  vec3 slabDirWorld;
  vec3 slabDirIndexPerMm;
  vec3 baseIndex;
  loadRayGeometry(curveColumnCoord, outputRow, baseWorldPos, slabDirWorld, slabDirIndexPerMm, baseIndex);

  vec4 supportData = texelFetch(uSupportData, ivec2(col, outputRow), 0);
  float supportCenterMm = supportData.r;
  float supportConfidence = clamp(supportData.g, 0.0, 1.0);
  float supportDensity = clamp(supportData.a, 0.0, 1.0);
  float rawSupportSpreadMm = max(supportData.b, 0.0);
  bool noSupportRay =
    supportConfidence <= 1e-6 &&
    supportDensity <= 1e-6 &&
    rawSupportSpreadMm <= 1e-6;
  if (noSupportRay) {
    fragColor = vec4(0.0);
    return;
  }
  float supportSpreadMm = max(rawSupportSpreadMm, 0.1);
  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);
  float toothBandDistance =
    abs(displayRowNorm - ${GPU_SUPPORT_MODEL_PARAMS.toothBandCenter.toFixed(3)});
  float toothBandPrior =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth.toFixed(3)},
      toothBandDistance
    );

  float nativeSlabPitchMm = computeNativeSlabPitchMm(slabDirIndexPerMm);
  float slabWidthMm = max(uSlabHalfMm * 2.0, 0.0);
  int raySampleCount = computeRaySampleCount(slabWidthMm, nativeSlabPitchMm, uSlabSamples);
  float slabStep = raySampleCount > 1 ? slabWidthMm / float(raySampleCount - 1) : 0.0;

  float broadSigmaMm = max(
    uSlabHalfMm * ${GPU_DRR_MODEL_PARAMS.broadSigmaSlabScale.toFixed(3)},
    nativeSlabPitchMm * ${GPU_DRR_MODEL_PARAMS.broadSigmaNativePitchScale.toFixed(3)}
  );
  float focusedSigmaMm = clamp(
    supportSpreadMm * ${GPU_DRR_MODEL_PARAMS.focusedSpreadScale.toFixed(3)} +
      nativeSlabPitchMm * ${GPU_DRR_MODEL_PARAMS.focusedNativePitchBias.toFixed(3)},
    nativeSlabPitchMm * ${GPU_DRR_MODEL_PARAMS.focusedSigmaMinNativePitchScale.toFixed(3)},
    max(
      uSlabHalfMm * ${GPU_DRR_MODEL_PARAMS.focusedSigmaMaxSlabScale.toFixed(3)},
      nativeSlabPitchMm * ${GPU_DRR_MODEL_PARAMS.focusedSigmaMaxNativePitchScale.toFixed(3)}
    )
  );
  float supportSigmaMm = min(
    broadSigmaMm,
    mix(
      focusedSigmaMm,
      focusedSigmaMm * ${GPU_DRR_MODEL_PARAMS.confidenceSigmaExpansion.toFixed(3)},
      smoothstep(
        ${GPU_DRR_MODEL_PARAMS.confidenceBlendLow.toFixed(3)},
        ${GPU_DRR_MODEL_PARAMS.confidenceBlendHigh.toFixed(3)},
        supportConfidence
      )
    )
  );
  float sigmaHardCapMm = mix(
    0.60,
    ${GPU_DRR_MODEL_PARAMS.troughSigmaHardCapMm.toFixed(3)},
    smoothstep(0.18, 0.62, supportConfidence)
  );
  supportSigmaMm = min(supportSigmaMm, sigmaHardCapMm);
  float dominantToothBandGate =
    toothBandPrior *
    smoothstep(0.08, 0.24, supportDensity) *
    smoothstep(0.02, 0.12, supportConfidence);
  float broadWeakToothBandGate =
    toothBandPrior *
    smoothstep(
      max(nativeSlabPitchMm * 1.6, 0.34),
      max(uSlabHalfMm * 0.36, nativeSlabPitchMm * 3.0),
      supportSpreadMm
    ) *
    (1.0 - smoothstep(0.10, 0.26, supportDensity)) *
    (1.0 - smoothstep(0.02, 0.10, supportConfidence));
  supportSigmaMm = mix(
    supportSigmaMm,
    max(
      nativeSlabPitchMm * 0.76,
      supportSigmaMm * ${GPU_DRR_MODEL_PARAMS.dominantToothSigmaShrinkScale.toFixed(3)}
    ),
    dominantToothBandGate
  );
  supportSigmaMm = mix(
    supportSigmaMm,
    max(
      nativeSlabPitchMm * 0.68,
      supportSigmaMm * ${GPU_DRR_MODEL_PARAMS.broadWeakSigmaShrinkScale.toFixed(3)}
    ),
    broadWeakToothBandGate
  );
  float toothBoundaryRefineGate =
    toothBandPrior *
    smoothstep(0.08, 0.22, supportConfidence) *
    smoothstep(0.10, 0.24, supportDensity) *
    (1.0 - smoothstep(0.24, 0.48, supportDensity)) *
    smoothstep(
      max(nativeSlabPitchMm * 1.8, 0.38),
      max(uSlabHalfMm * 0.34, nativeSlabPitchMm * 3.0),
      supportSpreadMm
    );
  supportSigmaMm = mix(
    supportSigmaMm,
    max(
      nativeSlabPitchMm * 0.64,
      supportSigmaMm * ${GPU_DRR_MODEL_PARAMS.toothBoundarySigmaShrinkScale.toFixed(3)}
    ),
    toothBoundaryRefineGate
  );
  float sigmaSlabFloor = max(
    uSlabHalfMm * 0.18,
    nativeSlabPitchMm * 1.4
  );
  supportSigmaMm = max(supportSigmaMm, sigmaSlabFloor);
  float supportDenom = 2.0 * supportSigmaMm * supportSigmaMm;
  float backgroundNarrowConfidenceGate =
    1.0 - smoothstep(0.010, 0.050, supportConfidence);
  float backgroundNarrowDensityGate =
    1.0 - smoothstep(0.10, 0.28, supportDensity);
  float toothBandLeakRisk =
    (1.0 - smoothstep(0.03, 0.14, supportConfidence)) *
    (1.0 - smoothstep(0.08, 0.26, supportDensity)) *
    smoothstep(
      max(nativeSlabPitchMm * 1.5, 0.26),
      max(uSlabHalfMm * 0.34, nativeSlabPitchMm * 2.8),
      supportSpreadMm
    );
  float backgroundNarrowToothBandProtection =
    mix(
      1.0,
      mix(
        ${GPU_DRR_MODEL_PARAMS.toothBandBackgroundProtectionFloor.toFixed(3)},
        1.0,
        toothBandLeakRisk
      ),
      toothBandPrior
    );
  float backgroundTroughNarrowGate =
    backgroundNarrowConfidenceGate *
    backgroundNarrowDensityGate *
    backgroundNarrowToothBandProtection;
  float troughHalfWidthMm =
    supportSigmaMm *
    ${GPU_DRR_MODEL_PARAMS.approxTroughBoundarySigmaMultiplier.toFixed(3)} *
    mix(1.0, ${GPU_DRR_MODEL_PARAMS.backgroundTroughNarrowScale.toFixed(3)}, backgroundTroughNarrowGate);
  float troughSlabFloor = nativeSlabPitchMm * 1.0;
  troughHalfWidthMm = max(troughHalfWidthMm, troughSlabFloor);
  float toothContinuityAdmissionGate =
    toothBandPrior *
    smoothstep(0.06, 0.20, supportDensity) *
    smoothstep(0.02, 0.10, supportConfidence) *
    (1.0 - smoothstep(
      max(nativeSlabPitchMm * 2.2, 0.46),
      max(uSlabHalfMm * 0.50, nativeSlabPitchMm * 4.0),
      supportSpreadMm
    ));
  float continuityExpandedTroughHalfWidthMm =
    troughHalfWidthMm *
    mix(
      1.0,
      ${GPU_DRR_MODEL_PARAMS.toothContinuityAdmissionExpandScale.toFixed(3)},
      toothContinuityAdmissionGate
    );
  float confidenceGate = smoothstep(
    ${GPU_DRR_MODEL_PARAMS.attenuationConfidenceLow.toFixed(3)},
    ${GPU_DRR_MODEL_PARAMS.attenuationConfidenceHigh.toFixed(3)},
    supportConfidence
  );
  float lowerPenaltyRowGate = smoothstep(
    ${GPU_DRR_MODEL_PARAMS.lowerPenaltyRowStart.toFixed(3)},
    ${GPU_DRR_MODEL_PARAMS.lowerPenaltyRowEnd.toFixed(3)},
    displayRowNorm
  );
  float lowerPenaltyConfidenceGate =
    1.0 -
    smoothstep(
      ${GPU_DRR_MODEL_PARAMS.lowerPenaltyConfidenceLow.toFixed(3)},
      ${GPU_DRR_MODEL_PARAMS.lowerPenaltyConfidenceHigh.toFixed(3)},
      supportConfidence
    );
  float supportWeightPower = mix(
    1.150,
    ${GPU_DRR_MODEL_PARAMS.supportWeightPowerHighConfidence.toFixed(3)},
    confidenceGate
  );
  float offToothBandGate = 1.0 - smoothstep(0.12, 0.32, toothBandPrior);
  float lowConfidenceGate = 1.0 - smoothstep(0.02, 0.12, supportConfidence);
  float lowConfidenceBackgroundContributionGate =
    backgroundTroughNarrowGate *
    offToothBandGate;
  float superiorBackgroundRowGate = 1.0 - smoothstep(0.16, 0.32, displayRowNorm);
  float inferiorBackgroundRowGate = smoothstep(0.78, 0.94, displayRowNorm);
  float edgeBackgroundRowGate = max(superiorBackgroundRowGate, inferiorBackgroundRowGate);
  float totalAttenuation = 0.0;
  float fogAttenuation = 0.0;
  float lowerPenaltyAccum = 0.0;
  float participatingSamples = 0.0;
  bool hasValidSample = false;

  const int MAX_SLAB = 64;
  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= raySampleCount) {
      break;
    }

    float slabOffset = raySampleCount > 1 ? -uSlabHalfMm + float(s) * slabStep : 0.0;
    vec3 uvw;
    if (!computeSampleUvw(baseWorldPos, slabDirWorld, slabOffset, uvw)) {
      continue;
    }

    float supportDistanceMm = slabOffset - supportCenterMm;
    float absSupportDistanceMm = abs(supportDistanceMm);
    if (absSupportDistanceMm > continuityExpandedTroughHalfWidthMm) {
      continue;
    }

    float hu = sampleHu(uvw);
    float supportWeight = exp(-(supportDistanceMm * supportDistanceMm) / supportDenom);
    supportWeight = pow(clamp(supportWeight, 0.0, 1.0), supportWeightPower);
    float mu = pseudoAttenuationFromHu(hu);
    float muFog = softFogAttenuationFromHu(hu);
    float lowerPenalty = max(mu - muFog, 0.0) *
      ${GPU_DRR_MODEL_PARAMS.lowerPenaltyDenseScale.toFixed(3)} *
      lowerPenaltyRowGate *
      lowerPenaltyConfidenceGate;
    float toothStructureSupport = clamp(
      max(denseSupportFromHu(hu), supportResponseFromHu(hu) * 0.58),
      0.0,
      1.0
    );
    float diffuseNonDentalGate =
      1.0 - smoothstep(0.12, 0.46, toothStructureSupport);
    float airLikeGate = 1.0 - smoothstep(-280.0, 120.0, hu);
    float lowConfidenceDiffuseLeakGate =
      lowConfidenceBackgroundContributionGate *
      diffuseNonDentalGate;
    float lowConfidenceAirLeakGate =
      backgroundTroughNarrowGate *
      lowConfidenceGate *
      airLikeGate;
    float rowEdgeLeakGate =
      edgeBackgroundRowGate *
      offToothBandGate *
      (1.0 - smoothstep(0.10, 0.30, supportDensity));
    float aggressiveDiffuseLeakGate = max(
      lowConfidenceDiffuseLeakGate,
      rowEdgeLeakGate * diffuseNonDentalGate
    );
    float aggressiveAirLeakGate = max(
      lowConfidenceAirLeakGate,
      rowEdgeLeakGate * airLikeGate
    );
    float toothContinuityShoulderGate =
      continuityExpandedTroughHalfWidthMm > troughHalfWidthMm + 1e-4
        ? 1.0 - smoothstep(
            troughHalfWidthMm,
            continuityExpandedTroughHalfWidthMm,
            absSupportDistanceMm
          )
        : 0.0;
    float toothBoundaryContinuityProtect =
      toothContinuityAdmissionGate *
      toothContinuityShoulderGate *
      smoothstep(0.16, 0.42, toothStructureSupport) *
      (1.0 - smoothstep(0.020, 0.060, lowerPenalty));
    supportWeight = mix(
      supportWeight,
      max(
        supportWeight,
        ${GPU_DRR_MODEL_PARAMS.toothContinuitySupportWeightFloor.toFixed(3)}
      ),
      toothBoundaryContinuityProtect
    );
    float normalizedSupportDistance =
      absSupportDistanceMm / max(continuityExpandedTroughHalfWidthMm, 1e-4);
    float middleBandCoreGate = smoothstep(0.22, 0.58, toothBandPrior);
    float middleBandOffTroughGate = smoothstep(
      ${GPU_DRR_MODEL_PARAMS.middleBandOffTroughRejectStart.toFixed(3)},
      ${GPU_DRR_MODEL_PARAMS.middleBandOffTroughRejectEnd.toFixed(3)},
      normalizedSupportDistance
    );
    float focusedBalancedSlabGate =
      1.0 - smoothstep(
        ${GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectStartMm.toFixed(3)},
        ${GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectEndMm.toFixed(3)},
        uSlabHalfMm
      );
    float focusedBalancedCenterGate =
      1.0 - smoothstep(
        0.0,
        ${GPU_DRR_MODEL_PARAMS.focusedBalancedCenterToleranceMm.toFixed(3)},
        abs(
          uVertCenterOffsetMm -
          ${GPU_DRR_MODEL_PARAMS.focusedBalancedCenterOffsetMm.toFixed(3)}
        )
      );
    float focusedBalancedToothProtect =
      focusedBalancedSlabGate *
      focusedBalancedCenterGate *
      middleBandCoreGate *
      smoothstep(0.04, 0.18, supportDensity) *
      smoothstep(0.008, 0.050, supportConfidence);
    float focusedBalancedNearTroughProtect =
      focusedBalancedToothProtect *
      (1.0 - smoothstep(0.40, 0.88, normalizedSupportDistance)) *
      (1.0 - smoothstep(0.018, 0.055, lowerPenalty));
    float softTrueToothNearTroughProtect =
      middleBandCoreGate *
      smoothstep(0.16, 0.42, toothStructureSupport) *
      smoothstep(0.03, 0.14, supportDensity) *
      smoothstep(0.004, 0.040, supportConfidence) *
      (1.0 - smoothstep(0.46, 0.92, normalizedSupportDistance));
    float trueToothNearTroughProtect =
      middleBandCoreGate *
      smoothstep(0.22, 0.52, toothStructureSupport) *
      smoothstep(0.05, 0.18, supportDensity) *
      smoothstep(0.006, 0.060, supportConfidence) *
      (1.0 - smoothstep(0.42, 0.90, normalizedSupportDistance));
    float residualToothNearTroughProtect =
      middleBandCoreGate *
      smoothstep(0.06, 0.14, supportConfidence) *
      smoothstep(0.02, 0.08, supportDensity) *
      (1.0 - smoothstep(0.12, 0.30, supportDensity)) *
      (1.0 - smoothstep(0.50, 0.92, normalizedSupportDistance));
    float weakValidatedMiddleAccumGate =
      middleBandCoreGate *
      (1.0 - dominantToothBandGate) *
      (1.0 - toothContinuityAdmissionGate * 0.88) *
      smoothstep(0.04, 0.14, supportConfidence) *
      (1.0 - smoothstep(0.12, 0.24, supportConfidence)) *
      smoothstep(0.04, 0.12, supportDensity) *
      (1.0 - smoothstep(0.12, 0.24, supportDensity)) *
      (1.0 - smoothstep(0.42, 0.86, normalizedSupportDistance)) *
      (1.0 - smoothstep(0.012, 0.050, lowerPenalty));
    float middleBandLeakAdmissionGate =
      middleBandCoreGate *
      toothBandLeakRisk *
      diffuseNonDentalGate *
      (1.0 - smoothstep(0.012, 0.048, lowerPenalty)) *
      middleBandOffTroughGate;
    middleBandLeakAdmissionGate *=
      1.0 -
      focusedBalancedNearTroughProtect *
      ${GPU_DRR_MODEL_PARAMS.focusedBalancedMiddleLeakProtectScale.toFixed(3)};
    middleBandLeakAdmissionGate *=
      1.0 - softTrueToothNearTroughProtect * 0.32;
    middleBandLeakAdmissionGate *=
      1.0 -
      residualToothNearTroughProtect *
      ${GPU_DRR_MODEL_PARAMS.residualMiddleLeakProtectScale.toFixed(3)};
    float hardOffTroughMiddleLeakGate =
      middleBandLeakAdmissionGate *
      (1.0 - smoothstep(0.18, 0.44, toothStructureSupport)) *
      smoothstep(
        max(nativeSlabPitchMm * 1.6, 0.34),
        max(uSlabHalfMm * 0.30, nativeSlabPitchMm * 2.6),
        supportSpreadMm
      ) *
      (1.0 - smoothstep(0.08, 0.24, supportDensity)) *
      (1.0 - smoothstep(0.010, 0.060, supportConfidence));
    hardOffTroughMiddleLeakGate *=
      1.0 - softTrueToothNearTroughProtect * 0.72;
    hardOffTroughMiddleLeakGate *=
      1.0 - residualToothNearTroughProtect * 0.72;
    float toothBandLeakContributionFloor = mix(
      ${GPU_DRR_MODEL_PARAMS.middleBandLeakContributionScale.toFixed(3)},
      0.380,
      middleBandCoreGate
    );
    toothBandLeakContributionFloor = mix(
      toothBandLeakContributionFloor,
      ${GPU_DRR_MODEL_PARAMS.focusedBalancedLeakContributionFloor.toFixed(3)},
      focusedBalancedNearTroughProtect
    );
    toothBandLeakContributionFloor = mix(
      toothBandLeakContributionFloor,
      ${GPU_DRR_MODEL_PARAMS.hardOffTroughLeakContributionFloor.toFixed(3)},
      hardOffTroughMiddleLeakGate
    );
    float preReductionContributionScale = mix(
      1.0,
      ${GPU_DRR_MODEL_PARAMS.lowConfidenceBackgroundContributionScale.toFixed(3)},
      aggressiveDiffuseLeakGate
    );
    preReductionContributionScale *= mix(
      1.0,
      ${GPU_DRR_MODEL_PARAMS.lowConfidenceAirContributionScale.toFixed(3)},
      aggressiveAirLeakGate
    );
    preReductionContributionScale *= mix(
      1.0,
      toothBandLeakContributionFloor,
      middleBandLeakAdmissionGate
    );
    preReductionContributionScale *= mix(
      1.0,
      ${GPU_DRR_MODEL_PARAMS.hardOffTroughLeakExtraScale.toFixed(3)},
      hardOffTroughMiddleLeakGate
    );
    preReductionContributionScale *= mix(
      1.0,
      ${GPU_DRR_MODEL_PARAMS.weakValidatedMiddleAccumContributionScale.toFixed(3)},
      weakValidatedMiddleAccumGate
    );
    preReductionContributionScale = mix(
      preReductionContributionScale,
      max(preReductionContributionScale, 0.58),
      focusedBalancedNearTroughProtect
    );
    preReductionContributionScale = mix(
      preReductionContributionScale,
      max(preReductionContributionScale, 0.62),
      softTrueToothNearTroughProtect
    );
    preReductionContributionScale = mix(
      preReductionContributionScale,
      max(
        preReductionContributionScale,
        ${GPU_DRR_MODEL_PARAMS.toothContinuityContributionFloor.toFixed(3)}
      ),
      toothBoundaryContinuityProtect
    );
    preReductionContributionScale = mix(
      preReductionContributionScale,
      max(
        preReductionContributionScale,
        ${GPU_DRR_MODEL_PARAMS.trueToothContributionFloor.toFixed(3)}
      ),
      trueToothNearTroughProtect
    );
    float segmentLength = max(slabStep, nativeSlabPitchMm);

    totalAttenuation +=
      mu *
      supportWeight *
      segmentLength *
      mix(0.62, 1.0, confidenceGate) *
      preReductionContributionScale;
    fogAttenuation +=
      muFog *
      supportWeight *
      segmentLength *
      mix(0.22, 0.32, supportConfidence) *
      preReductionContributionScale;
    lowerPenaltyAccum += lowerPenalty * supportWeight * segmentLength;
    participatingSamples += supportWeight;
    hasValidSample = true;
  }

  if (!hasValidSample) {
    fragColor = vec4(0.0);
    return;
  }

  fragColor = vec4(
    max(totalAttenuation, 0.0),
    max(lowerPenaltyAccum, 0.0),
    max(fogAttenuation, 0.0),
    max(participatingSamples, 0.0)
  );
}
`;

const TONE_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uDrrData;
uniform sampler2D uSupportData;
uniform int uPanoWidth;
uniform int uPanoHeight;
uniform float uSlabHalfMm;
uniform float uVertCenterOffsetMm;

out vec4 fragColor;

float normalizedSigmoidTone(float exposureValue) {
  float midpoint = ${GPU_TONE_MODEL_PARAMS.sigmoidMidpoint.toFixed(3)};
  float slope = ${GPU_TONE_MODEL_PARAMS.sigmoidSlope.toFixed(3)};
  float whitePoint = ${GPU_TONE_MODEL_PARAMS.sigmoidWhitePoint.toFixed(3)};
  float safeValue = max(exposureValue, 0.0);
  float low = 1.0 / (1.0 + exp(-slope * (0.0 - midpoint)));
  float high = 1.0 / (1.0 + exp(-slope * (whitePoint - midpoint)));
  float curved = 1.0 / (1.0 + exp(-slope * (safeValue - midpoint)));
  return clamp((curved - low) / max(high - low, 1e-4), 0.0, 1.0);
}

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(-780.0, 0.0, 0.0, 1.0);
    return;
  }

  vec4 drr = texelFetch(uDrrData, ivec2(col, outputRow), 0);
  vec4 supportData = texelFetch(uSupportData, ivec2(col, outputRow), 0);
  float totalAttenuation = max(drr.r, 0.0);
  float lowerPenalty = max(drr.g, 0.0);
  float fogAttenuation = max(drr.b, 0.0);
  float participatingSamples = max(drr.a, 0.0);
  float supportConfidence = clamp(supportData.g, 0.0, 1.0);
  float supportSpreadMm = max(supportData.b, 0.0);
  float supportDensity = clamp(supportData.a, 0.0, 1.0);
  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);
  float inferiorPenalty = smoothstep(
    ${GPU_DRR_MODEL_PARAMS.lowerPenaltyRowStart.toFixed(3)},
    ${GPU_DRR_MODEL_PARAMS.lowerPenaltyRowEnd.toFixed(3)},
    displayRowNorm
  );

  float dehazedAttenuation = max(
    totalAttenuation -
      fogAttenuation *
        mix(
          ${GPU_TONE_MODEL_PARAMS.lowConfidenceFogSuppression.toFixed(3)},
          ${GPU_TONE_MODEL_PARAMS.highConfidenceFogSuppression.toFixed(3)},
          supportConfidence
        ),
    totalAttenuation *
      mix(
          ${GPU_TONE_MODEL_PARAMS.lowConfidenceFogRetention.toFixed(3)},
          ${GPU_TONE_MODEL_PARAMS.highConfidenceFogRetention.toFixed(3)},
        supportConfidence
      )
  );
  float retainedPenaltyFloor =
    dehazedAttenuation *
    mix(
      ${GPU_TONE_MODEL_PARAMS.lowConfidencePenaltyRetention.toFixed(3)},
      ${GPU_TONE_MODEL_PARAMS.highConfidencePenaltyRetention.toFixed(3)},
      supportConfidence
    );
  float gentlySuppressedAttenuation = max(
    dehazedAttenuation -
      lowerPenalty *
        mix(
          ${GPU_TONE_MODEL_PARAMS.lowConfidencePenaltySuppression.toFixed(3)},
          ${GPU_TONE_MODEL_PARAMS.highConfidencePenaltySuppression.toFixed(3)},
          supportConfidence
        ),
    retainedPenaltyFloor
  );
  float backgroundSuppressionConfidenceGate =
    1.0 - smoothstep(0.004, 0.020, supportConfidence);
  float lowDensityGate =
    1.0 - smoothstep(0.05, 0.16, supportDensity);
  float lowPenaltyGate =
    1.0 - smoothstep(0.008, 0.035, lowerPenalty);
  float backgroundSuppressionGate =
    backgroundSuppressionConfidenceGate *
    lowDensityGate *
    lowPenaltyGate;
  float focusedBalancedSlabGate =
    1.0 - smoothstep(
      ${GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectStartMm.toFixed(3)},
      ${GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectEndMm.toFixed(3)},
      uSlabHalfMm
    );
  float focusedBalancedCenterGate =
    1.0 - smoothstep(
      0.0,
      ${GPU_DRR_MODEL_PARAMS.focusedBalancedCenterToleranceMm.toFixed(3)},
      abs(
        uVertCenterOffsetMm -
        ${GPU_DRR_MODEL_PARAMS.focusedBalancedCenterOffsetMm.toFixed(3)}
      )
    );
  float focusedBalancedToneProtect =
    focusedBalancedSlabGate *
    focusedBalancedCenterGate *
    smoothstep(0.12, 0.24, displayRowNorm) *
    (1.0 - smoothstep(0.78, 0.90, displayRowNorm)) *
    smoothstep(0.04, 0.18, supportDensity) *
    smoothstep(0.008, 0.050, supportConfidence) *
    lowPenaltyGate;
  float weakToothProtect =
    smoothstep(0.002, 0.015, supportDensity) *
    smoothstep(1.8, 3.2, participatingSamples) *
    smoothstep(0.12, 0.22, displayRowNorm) *
    (1.0 - smoothstep(0.82, 0.92, displayRowNorm));
  backgroundSuppressionGate *=
    1.0 -
    min(
      1.0,
      max(
        weakToothProtect,
        focusedBalancedToneProtect *
          ${GPU_TONE_MODEL_PARAMS.focusedBalancedBackgroundProtectScale.toFixed(3)}
      )
    );
  // Keep the commit's tooth attenuation behavior and only suppress
  // the extremely low-confidence background cluster.
  gentlySuppressedAttenuation *= mix(1.0, 0.30, backgroundSuppressionGate);
  float inferiorSuppressionStrength = mix(
    0.98,
    0.60,
    smoothstep(0.06, 0.22, supportDensity) *
    (1.0 - smoothstep(0.12, 0.35, supportConfidence))
  );
  gentlySuppressedAttenuation *= mix(1.0, inferiorSuppressionStrength, inferiorPenalty);
  // bottomPenalty fires when 1-displayRowNorm > 0.72, i.e. displayRowNorm < 0.28
  // Due to GL Y-flip, displayRowNorm~0 = VISUAL TOP (Zone B: maxillary/sinus).
  // Unconditional 95% suppression — no teeth above the crowns.
  float bottomPenalty = smoothstep(0.72, 0.92,
    1.0 - displayRowNorm);
  gentlySuppressedAttenuation *= mix(1.0, 0.05, bottomPenalty);
  // topPenalty fires when displayRowNorm > 0.82
  // Due to GL Y-flip, displayRowNorm~1 = VISUAL BOTTOM (Zone A: mandible).
  // Conditional: preserve real tooth roots, suppress only low-confidence bone.
  float topPenalty = smoothstep(0.82, 0.96, displayRowNorm);
  float topSuppressionStrength = mix(
    0.98,
    0.58,
    (1.0 - smoothstep(0.08, 0.28, supportDensity)) *
    (1.0 - smoothstep(0.14, 0.40, supportConfidence))
  );
  gentlySuppressedAttenuation *= mix(1.0, topSuppressionStrength, topPenalty);
  float weakSupportConfidenceGate =
    1.0 - smoothstep(
      ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutConfidenceLow.toFixed(3)},
      ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutConfidenceHigh.toFixed(3)},
      supportConfidence
    );
  float broadSupportSpreadGate = smoothstep(
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutSpreadLow.toFixed(2)},
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutSpreadHigh.toFixed(2)},
    supportSpreadMm
  );
  float denseSupportGate = smoothstep(
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutDensityLow.toFixed(2)},
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutDensityHigh.toFixed(2)},
    supportDensity
  );
  float participationGate = smoothstep(
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutParticipationLow.toFixed(1)},
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutParticipationHigh.toFixed(1)},
    participatingSamples
  );
  float invalidSupportBlackoutGate =
    weakSupportConfidenceGate *
    broadSupportSpreadGate *
    denseSupportGate *
    participationGate;
  float focusedBalancedBlackoutProtect =
    focusedBalancedSlabGate *
    focusedBalancedCenterGate *
    smoothstep(0.16, 0.28, displayRowNorm) *
    (1.0 - smoothstep(0.76, 0.88, displayRowNorm)) *
    smoothstep(0.55, 0.92, supportDensity) *
    smoothstep(0.006, 0.030, supportConfidence);
  invalidSupportBlackoutGate *=
    1.0 -
    focusedBalancedBlackoutProtect *
      ${GPU_TONE_MODEL_PARAMS.focusedBalancedBlackoutProtectScale.toFixed(3)};
  float debugSupportValiditySignal = invalidSupportBlackoutGate;

  if (${GPU_TEMP_DEBUG_DISPLAY_INVALID_SUPPORT_MAP ? 'true' : 'false'}) {
    float debugFinalHu = mix(
      ${GPU_TONE_MODEL_PARAMS.outputHuMin.toFixed(1)},
      ${GPU_TONE_MODEL_PARAMS.outputHuMax.toFixed(1)},
      debugSupportValiditySignal
    );
    fragColor = vec4(
      debugFinalHu,
      debugSupportValiditySignal,
      lowerPenalty,
      debugSupportValiditySignal
    );
    return;
  }

  gentlySuppressedAttenuation *= mix(
    1.0,
    ${GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutScale.toFixed(3)},
    invalidSupportBlackoutGate
  );
  float radiographSignal = normalizedSigmoidTone(
    gentlySuppressedAttenuation * ${GPU_TONE_MODEL_PARAMS.exposureScale.toFixed(3)}
  );
  radiographSignal = pow(
    radiographSignal,
    ${GPU_TONE_MODEL_PARAMS.postCurveGamma.toFixed(3)}
  );

  float finalHu = mix(
    ${GPU_TONE_MODEL_PARAMS.outputHuMin.toFixed(1)},
    ${GPU_TONE_MODEL_PARAMS.outputHuMax.toFixed(1)},
    radiographSignal
  );
  fragColor = vec4(finalHu, gentlySuppressedAttenuation, lowerPenalty, radiographSignal);
}
`;

// ─── Cached GPU State ────────────────────────────────────────────────
let _canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
let _gl: WebGL2RenderingContext | null = null;
let _panoProgram: WebGLProgram | null = null;
let _supportProgram: WebGLProgram | null = null;
let _supportPeakDebugProgram: WebGLProgram | null = null;
let _supportModelDiagnosticProgram: WebGLProgram | null = null;
let _supportSmoothProgram: WebGLProgram | null = null;
let _drrProgram: WebGLProgram | null = null;
let _toneProgram: WebGLProgram | null = null;
let _vao: WebGLVertexArrayObject | null = null;

// Volume texture cache (avoid re-uploading for same volume)
let _volumeTex: WebGLTexture | null = null;
let _cachedVolumeId: string | null = null;
let _cachedVolumeNormalizationSignature: string | null = null;
let _hasFloatLinearFiltering = false;

let _splineTex: WebGLTexture | null = null;
let _supportTexA: WebGLTexture | null = null;
let _supportTexB: WebGLTexture | null = null;
let _supportPeakDebugTex: WebGLTexture | null = null;
let _drrTex: WebGLTexture | null = null;
let _supportFboA: WebGLFramebuffer | null = null;
let _supportFboB: WebGLFramebuffer | null = null;
let _supportPeakDebugFbo: WebGLFramebuffer | null = null;
let _drrFbo: WebGLFramebuffer | null = null;
let _fbo: WebGLFramebuffer | null = null;
let _fboTex: WebGLTexture | null = null;
let _fboWidth = 0;
let _fboHeight = 0;

// Uniform locations

// ─── Helpers ─────────────────────────────────────────────────────────
function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
}

function compileProgram(gl: WebGL2RenderingContext, fragSrc: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${log}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function setUniform1i(gl: WebGL2RenderingContext, program: WebGLProgram | null, name: string, value: number): void {
  if (!program) return;
  const location = gl.getUniformLocation(program, name);
  if (location !== null) {
    gl.uniform1i(location, value);
  }
}

function setUniform1f(gl: WebGL2RenderingContext, program: WebGLProgram | null, name: string, value: number): void {
  if (!program) return;
  const location = gl.getUniformLocation(program, name);
  if (location !== null) {
    gl.uniform1f(location, value);
  }
}

function setUniform3f(
  gl: WebGL2RenderingContext,
  program: WebGLProgram | null,
  name: string,
  x: number,
  y: number,
  z: number
): void {
  if (!program) return;
  const location = gl.getUniformLocation(program, name);
  if (location !== null) {
    gl.uniform3f(location, x, y, z);
  }
}

function setUniformMatrix4fv(
  gl: WebGL2RenderingContext,
  program: WebGLProgram | null,
  name: string,
  value: Float32Array
): void {
  if (!program) return;
  const location = gl.getUniformLocation(program, name);
  if (location !== null) {
    gl.uniformMatrix4fv(location, false, value);
  }
}

function bindCommonRayUniforms(
  gl: WebGL2RenderingContext,
  program: WebGLProgram | null,
  params: {
    verticalDir: [number, number, number];
    vertHalfMm: number;
    verticalCenterOffsetMm: number;
    slabHalfThicknessMm: number;
    slabSamples: number;
    panoWidth: number;
    panoHeight: number;
    worldToIndexMat: Float32Array;
    dimensions: [number, number, number];
    rescaleSlope: number;
    rescaleIntercept: number;
    applyRescale: boolean;
    debugMode?: number;
  }
): void {
  if (!program) return;
  setUniform1i(gl, program, 'uVolume', 0);
  setUniform1i(gl, program, 'uSplineData', 1);
  setUniform3f(gl, program, 'uVerticalDir', params.verticalDir[0], params.verticalDir[1], params.verticalDir[2]);
  setUniform1f(gl, program, 'uVertHalfMm', params.vertHalfMm);
  setUniform1f(gl, program, 'uVertCenterOffsetMm', params.verticalCenterOffsetMm);
  setUniform1f(gl, program, 'uSlabHalfMm', params.slabHalfThicknessMm);
  setUniform1i(gl, program, 'uSlabSamples', params.slabSamples);
  setUniform1i(gl, program, 'uPanoWidth', params.panoWidth);
  setUniform1i(gl, program, 'uPanoHeight', params.panoHeight);
  setUniformMatrix4fv(gl, program, 'uWorldToIndex', params.worldToIndexMat);
  setUniform3f(gl, program, 'uDims', params.dimensions[0], params.dimensions[1], params.dimensions[2]);
  setUniform1f(gl, program, 'uRescaleSlope', params.rescaleSlope);
  setUniform1f(gl, program, 'uRescaleIntercept', params.rescaleIntercept);
  setUniform1i(gl, program, 'uApplyRescale', params.applyRescale ? 1 : 0);
  if (typeof params.debugMode === 'number') {
    setUniform1i(gl, program, 'uDebugMode', params.debugMode);
  }
}

function drawFullscreenTriangle(gl: WebGL2RenderingContext): void {
  gl.bindVertexArray(_vao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindVertexArray(null);
}

function destroyPipelineTargets(gl: WebGL2RenderingContext): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindTexture(gl.TEXTURE_3D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindTexture(gl.TEXTURE_3D, null);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindTexture(gl.TEXTURE_3D, null);

  if (_supportTexA) gl.deleteTexture(_supportTexA);
  if (_supportTexB) gl.deleteTexture(_supportTexB);
  if (_supportPeakDebugTex) gl.deleteTexture(_supportPeakDebugTex);
  if (_drrTex) gl.deleteTexture(_drrTex);
  if (_fboTex) gl.deleteTexture(_fboTex);
  if (_supportFboA) gl.deleteFramebuffer(_supportFboA);
  if (_supportFboB) gl.deleteFramebuffer(_supportFboB);
  if (_supportPeakDebugFbo) gl.deleteFramebuffer(_supportPeakDebugFbo);
  if (_drrFbo) gl.deleteFramebuffer(_drrFbo);
  if (_fbo) gl.deleteFramebuffer(_fbo);

  _supportTexA = null;
  _supportTexB = null;
  _supportPeakDebugTex = null;
  _drrTex = null;
  _fboTex = null;
  _supportFboA = null;
  _supportFboB = null;
  _supportPeakDebugFbo = null;
  _drrFbo = null;
  _fbo = null;
  _fboWidth = 0;
  _fboHeight = 0;
}

function createFloatRenderTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number
): { texture: WebGLTexture; fbo: WebGLFramebuffer } {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(texture);
    throw new Error(`FBO incomplete: ${status}`);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { texture, fbo };
}

function ensurePipelineTargets(gl: WebGL2RenderingContext, w: number, h: number): void {
  const targetsReady =
    _supportTexA &&
    _supportTexB &&
    _supportPeakDebugTex &&
    _drrTex &&
    _fboTex &&
    _supportFboA &&
    _supportFboB &&
    _supportPeakDebugFbo &&
    _drrFbo &&
    _fbo &&
    _fboWidth === w &&
    _fboHeight === h;

  if (targetsReady) {
    return;
  }

  destroyPipelineTargets(gl);

  const supportA = createFloatRenderTarget(gl, w, h);
  const supportB = createFloatRenderTarget(gl, w, h);
  const supportPeakDebug = createFloatRenderTarget(gl, w, h);
  const drr = createFloatRenderTarget(gl, w, h);
  const finalTarget = createFloatRenderTarget(gl, w, h);

  _supportTexA = supportA.texture;
  _supportFboA = supportA.fbo;
  _supportTexB = supportB.texture;
  _supportFboB = supportB.fbo;
  _supportPeakDebugTex = supportPeakDebug.texture;
  _supportPeakDebugFbo = supportPeakDebug.fbo;
  _drrTex = drr.texture;
  _drrFbo = drr.fbo;
  _fboTex = finalTarget.texture;
  _fbo = finalTarget.fbo;
  _fboWidth = w;
  _fboHeight = h;
}

// Retain legacy DRR declarations as inert references during rollback so
// strict TS unused-symbol checks do not fail while the file is simplified.
void SUPPORT_FRAG_SRC;
void SUPPORT_SMOOTH_FRAG_SRC;
void DRR_FRAG_SRC;
void TONE_FRAG_SRC;
void ensurePipelineTargets;

function buildWorldToIndexMat4(
  origin: [number, number, number],
  spacing: [number, number, number],
  direction: number[]
): Float32Array {
  // direction is a 3×3 row-major matrix: dir[row*3+col]
  // invDir = transpose of direction (orthogonal matrix)
  // worldToIndex: voxel = invDir * (world - origin) / spacing
  // As column-major mat4 for WebGL:
  const id0 = direction[0], id1 = direction[3], id2 = direction[6]; // invDir row 0
  const id3 = direction[1], id4 = direction[4], id5 = direction[7]; // invDir row 1
  const id6 = direction[2], id7 = direction[5], id8 = direction[8]; // invDir row 2

  const sx = spacing[0], sy = spacing[1], sz = spacing[2];
  const ox = origin[0], oy = origin[1], oz = origin[2];

  // Translation: -invDir * origin / spacing
  const tx = -(id0 * ox + id1 * oy + id2 * oz) / sx;
  const ty = -(id3 * ox + id4 * oy + id5 * oz) / sy;
  const tz = -(id6 * ox + id7 * oy + id8 * oz) / sz;

  // Column-major mat4
  return new Float32Array([
    id0 / sx, id3 / sy, id6 / sz, 0,
    id1 / sx, id4 / sy, id7 / sz, 0,
    id2 / sx, id5 / sy, id8 / sz, 0,
    tx, ty, tz, 1,
  ]);
}

function coerceWorldToIndexMat4(worldToIndex?: number[] | null): Float32Array {
  if (!Array.isArray(worldToIndex) || worldToIndex.length < 16) {
    throw new Error('[CPR-GPU] Missing worldToIndex mat4; GPU pano rendering requires the worker transform.');
  }

  const mat = new Float32Array(16);
  for (let i = 0; i < 16; i++) {
    const value = Number(worldToIndex[i]);
    if (!Number.isFinite(value)) {
      throw new Error('[CPR-GPU] Invalid worldToIndex mat4; non-finite element encountered.');
    }
    mat[i] = value;
  }

  return mat;
}

void buildWorldToIndexMat4;

function computeMinMax(buffer: Float32Array): { minValue: number; maxValue: number } {
  let minValue = Infinity;
  let maxValue = -Infinity;

  for (let i = 0; i < buffer.length; i++) {
    const v = buffer[i];
    if (!Number.isFinite(v)) continue;
    if (v < minValue) minValue = v;
    if (v > maxValue) maxValue = v;
  }

  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return { minValue: 0, maxValue: 0 };
  }
  return { minValue, maxValue };
}

interface FramebufferReadback {
  channel0: Float32Array;
  channel1: Float32Array;
  channel2: Float32Array;
  channel3: Float32Array;
  rawMin: number;
  rawMax: number;
}

interface BufferSummary {
  sampledCount: number;
  min: number;
  max: number;
  mean: number;
  stdDev: number;
  p01: number;
  p10: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

function clampNumber(value: number, minValue: number, maxValue: number): number {
  return Math.min(maxValue, Math.max(minValue, value));
}

function mixNumber(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smoothstepNumber(edge0: number, edge1: number, value: number): number {
  if (Math.abs(edge1 - edge0) <= 1e-6) {
    return value < edge0 ? 0 : 1;
  }

  const t = clampNumber((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function percentileFromSorted(values: number[], q: number): number {
  if (!values.length) {
    return NaN;
  }

  const clampedQ = clampNumber(q, 0, 1);
  const position = clampedQ * (values.length - 1);
  const lowIndex = Math.floor(position);
  const highIndex = Math.ceil(position);
  if (lowIndex === highIndex) {
    return values[lowIndex];
  }

  const t = position - lowIndex;
  return values[lowIndex] * (1 - t) + values[highIndex] * t;
}

function roundFinite(value: number, fractionDigits = 3): number {
  return Number.isFinite(value) ? Number(value.toFixed(fractionDigits)) : 0;
}

function summarizeFiniteBuffer(buffer: Float32Array, maxSamples = 40000): BufferSummary | null {
  if (!buffer.length) {
    return null;
  }

  const samples: number[] = [];
  const step = Math.max(1, Math.floor(buffer.length / maxSamples));
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSquares = 0;

  for (let i = 0; i < buffer.length; i += step) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }

    samples.push(value);
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    sumSquares += value * value;
  }

  if (!samples.length || !Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }

  samples.sort((a, b) => a - b);
  const sampledCount = samples.length;
  const mean = sum / sampledCount;
  const variance = Math.max(sumSquares / sampledCount - mean * mean, 0);

  return {
    sampledCount,
    min,
    max,
    mean,
    stdDev: Math.sqrt(variance),
    p01: percentileFromSorted(samples, 0.01),
    p10: percentileFromSorted(samples, 0.1),
    p50: percentileFromSorted(samples, 0.5),
    p90: percentileFromSorted(samples, 0.9),
    p95: percentileFromSorted(samples, 0.95),
    p99: percentileFromSorted(samples, 0.99),
  };
}

function computeClampFraction(buffer: Float32Array, clampLimitMm: number): number {
  if (!buffer.length || !Number.isFinite(clampLimitMm) || clampLimitMm <= 0) {
    return 0;
  }

  const thresholdMm = Math.max(clampLimitMm * 0.96, clampLimitMm - 0.1);
  let finiteCount = 0;
  let clampedCount = 0;
  for (let i = 0; i < buffer.length; i++) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }

    finiteCount++;
    if (Math.abs(value) >= thresholdMm) {
      clampedCount++;
    }
  }

  return finiteCount > 0 ? clampedCount / finiteCount : 0;
}

function collapseSupportPathByColumn(
  depthMap: Float32Array,
  confidenceMap: Float32Array,
  width: number,
  height: number
): { pathDepthMm: Float32Array; pathConfidence: Float32Array } {
  const pathDepthMm = new Float32Array(width);
  const pathConfidence = new Float32Array(width);
  const startRow = Math.max(0, Math.floor(height * SUPPORT_PATH_ROW_START_FRACTION));
  const endRow = Math.min(height - 1, Math.ceil(height * SUPPORT_PATH_ROW_END_FRACTION));
  const fallbackRow = Math.max(0, Math.min(height - 1, Math.round((height - 1) * 0.5)));

  for (let col = 0; col < width; col++) {
    let weightedDepth = 0;
    let weightedConfidence = 0;
    let weightSum = 0;

    for (let row = startRow; row <= endRow; row++) {
      const index = row * width + col;
      const depth = Number(depthMap[index]);
      const confidence = clampNumber(Number(confidenceMap[index]), 0, 1);
      if (!Number.isFinite(depth) || !Number.isFinite(confidence)) {
        continue;
      }

      const weight = confidence > 1e-4 ? confidence * confidence : 0;
      if (weight <= 0) {
        continue;
      }

      weightedDepth += depth * weight;
      weightedConfidence += confidence * weight;
      weightSum += weight;
    }

    if (weightSum > 1e-5) {
      pathDepthMm[col] = weightedDepth / weightSum;
      pathConfidence[col] = weightedConfidence / weightSum;
      continue;
    }

    const fallbackIndex = fallbackRow * width + col;
    pathDepthMm[col] = Number(depthMap[fallbackIndex]) || 0;
    pathConfidence[col] = clampNumber(Number(confidenceMap[fallbackIndex]) || 0, 0, 1);
  }

  return { pathDepthMm, pathConfidence };
}

function collapseWeightedMapByColumn(
  map: Float32Array,
  confidenceMap: Float32Array,
  width: number,
  height: number
): Float32Array {
  const collapsed = new Float32Array(width);
  const startRow = Math.max(0, Math.floor(height * SUPPORT_PATH_ROW_START_FRACTION));
  const endRow = Math.min(height - 1, Math.ceil(height * SUPPORT_PATH_ROW_END_FRACTION));
  const fallbackRow = Math.max(0, Math.min(height - 1, Math.round((height - 1) * 0.5)));

  for (let col = 0; col < width; col++) {
    let weightedValue = 0;
    let weightSum = 0;

    for (let row = startRow; row <= endRow; row++) {
      const index = row * width + col;
      const value = Number(map[index]);
      const confidence = clampNumber(Number(confidenceMap[index]), 0, 1);
      if (!Number.isFinite(value) || !Number.isFinite(confidence)) {
        continue;
      }

      const weight = confidence > 1e-4 ? confidence * confidence : 0;
      if (weight <= 0) {
        continue;
      }

      weightedValue += value * weight;
      weightSum += weight;
    }

    if (weightSum > 1e-5) {
      collapsed[col] = weightedValue / weightSum;
      continue;
    }

    const fallbackIndex = fallbackRow * width + col;
    collapsed[col] = Number(map[fallbackIndex]) || 0;
  }

  return collapsed;
}

function collapseWeightedMapByRow(
  map: Float32Array,
  confidenceMap: Float32Array,
  width: number,
  height: number
): Float32Array {
  const collapsed = new Float32Array(height);

  for (let row = 0; row < height; row++) {
    let weightedValue = 0;
    let weightSum = 0;

    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      const value = Number(map[index]);
      const confidence = clampNumber(Number(confidenceMap[index]) || 0, 0, 1);
      if (!Number.isFinite(value) || !Number.isFinite(confidence)) {
        continue;
      }

      const weight = confidence > 1e-4 ? confidence * confidence : 0;
      if (weight <= 0) {
        continue;
      }

      weightedValue += value * weight;
      weightSum += weight;
    }

    collapsed[row] = weightSum > 1e-5 ? weightedValue / weightSum : 0;
  }

  return collapsed;
}

function computeDisplayRowNorm(row: number, height: number): number {
  return height > 1 ? ((height - 1) - row) / Math.max(height - 1, 1) : 0;
}

function computeAdjacentDeltaP95(values: Float32Array): number {
  if (values.length < 2) {
    return 0;
  }

  const deltas: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const current = Number(values[i]);
    const previous = Number(values[i - 1]);
    if (!Number.isFinite(current) || !Number.isFinite(previous)) {
      continue;
    }
    deltas.push(Math.abs(current - previous));
  }

  if (!deltas.length) {
    return 0;
  }

  deltas.sort((a, b) => a - b);
  return percentileFromSorted(deltas, 0.95);
}

function computeLongestRunByThreshold(
  values: Float32Array,
  predicate: (value: number) => boolean
): number {
  let longestRun = 0;
  let currentRun = 0;

  for (let i = 0; i < values.length; i++) {
    const value = Number(values[i]);
    if (Number.isFinite(value) && predicate(value)) {
      currentRun++;
      longestRun = Math.max(longestRun, currentRun);
    } else {
      currentRun = 0;
    }
  }

  return longestRun;
}

function multiplyDirectionByWorldToIndex(
  worldToIndex: Float32Array,
  directionVector: [number, number, number]
): [number, number, number] {
  return [
    worldToIndex[0] * directionVector[0] + worldToIndex[4] * directionVector[1] + worldToIndex[8] * directionVector[2],
    worldToIndex[1] * directionVector[0] + worldToIndex[5] * directionVector[1] + worldToIndex[9] * directionVector[2],
    worldToIndex[2] * directionVector[0] + worldToIndex[6] * directionVector[1] + worldToIndex[10] * directionVector[2],
  ];
}

function computeNativeSlabPitchMmForFrames(
  worldToIndex: Float32Array,
  frames: GpuPanoInput['frames']
): Float32Array {
  const nativePitchMmByCol = new Float32Array(frames.length);
  for (let col = 0; col < frames.length; col++) {
    const directionVector = frames[col]?.N_slab ?? [0, 0, 1];
    const indexVector = multiplyDirectionByWorldToIndex(worldToIndex, directionVector);
    const indexUnitsPerMm = Math.hypot(indexVector[0], indexVector[1], indexVector[2]);
    nativePitchMmByCol[col] =
      indexUnitsPerMm <= 1e-5
        ? 0.2
        : clampNumber(1 / indexUnitsPerMm, 0.05, 2.0);
  }
  return nativePitchMmByCol;
}

function computeRaySampleCountForJs(
  slabWidthMm: number,
  nativeSlabPitchMm: number,
  requestedSampleCount: number
): number {
  const maxSlabSamples = 64;
  let pitchAlignedSampleCount = 1;
  if (slabWidthMm > 1e-4) {
    const safeNativePitchMm = Math.max(nativeSlabPitchMm, 1e-4);
    pitchAlignedSampleCount = Math.min(
      maxSlabSamples,
      Math.max(3, Math.ceil(slabWidthMm / safeNativePitchMm) + 1)
    );
  }

  return Math.min(maxSlabSamples, Math.max(Math.max(requestedSampleCount, 1), pitchAlignedSampleCount));
}

function buildSupportSigmaMap(
  supportSpreadMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportDensityMap: Float32Array,
  nativePitchMmByCol: Float32Array,
  width: number,
  height: number,
  slabHalfThicknessMm: number
): Float32Array {
  const supportSigmaMap = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      const supportSpreadMm = Math.max(Number(supportSpreadMap[index]), 0.1);
      const supportConfidence = clampNumber(Number(supportConfidenceMap[index]), 0, 1);
      const supportDensity = clampNumber(Number(supportDensityMap[index]), 0, 1);
      const nativePitchMm = Number(nativePitchMmByCol[col]) || 0.2;
      const toothBandPrior = computeToothBandPriorForJs(row, height);
      const broadSigmaMm = Math.max(
        slabHalfThicknessMm * GPU_DRR_MODEL_PARAMS.broadSigmaSlabScale,
        nativePitchMm * GPU_DRR_MODEL_PARAMS.broadSigmaNativePitchScale
      );
      const focusedSigmaMm = clampNumber(
        supportSpreadMm * GPU_DRR_MODEL_PARAMS.focusedSpreadScale +
        nativePitchMm * GPU_DRR_MODEL_PARAMS.focusedNativePitchBias,
        nativePitchMm * GPU_DRR_MODEL_PARAMS.focusedSigmaMinNativePitchScale,
        Math.max(
          slabHalfThicknessMm * GPU_DRR_MODEL_PARAMS.focusedSigmaMaxSlabScale,
          nativePitchMm * GPU_DRR_MODEL_PARAMS.focusedSigmaMaxNativePitchScale
        )
      );
      const sigmaMix = smoothstepNumber(
        GPU_DRR_MODEL_PARAMS.confidenceBlendLow,
        GPU_DRR_MODEL_PARAMS.confidenceBlendHigh,
        supportConfidence
      );
      const sigmaHardCapMix = smoothstepNumber(0.18, 0.62, supportConfidence);
      const sigmaHardCapMm = mixNumber(
        0.60,
        GPU_DRR_MODEL_PARAMS.troughSigmaHardCapMm,
        sigmaHardCapMix
      );
      let supportSigmaMm = Math.min(
        sigmaHardCapMm,
        Math.min(
          broadSigmaMm,
          mixNumber(
            focusedSigmaMm,
            focusedSigmaMm * GPU_DRR_MODEL_PARAMS.confidenceSigmaExpansion,
            sigmaMix
          )
        )
      );
      const dominantToothBandGate =
        toothBandPrior *
        smoothstepNumber(0.08, 0.24, supportDensity) *
        smoothstepNumber(0.02, 0.12, supportConfidence);
      const broadWeakToothBandGate =
        toothBandPrior *
        smoothstepNumber(
          Math.max(nativePitchMm * 1.6, 0.34),
          Math.max(slabHalfThicknessMm * 0.36, nativePitchMm * 3.0),
          supportSpreadMm
        ) *
        (1 - smoothstepNumber(0.10, 0.26, supportDensity)) *
        (1 - smoothstepNumber(0.02, 0.10, supportConfidence));
      const toothBoundaryRefineGate =
        toothBandPrior *
        smoothstepNumber(0.08, 0.22, supportConfidence) *
        smoothstepNumber(0.10, 0.24, supportDensity) *
        (1 - smoothstepNumber(0.24, 0.48, supportDensity)) *
        smoothstepNumber(
          Math.max(nativePitchMm * 1.8, 0.38),
          Math.max(slabHalfThicknessMm * 0.34, nativePitchMm * 3.0),
          supportSpreadMm
        );
      supportSigmaMm = mixNumber(
        supportSigmaMm,
        Math.max(
          nativePitchMm * 0.76,
          supportSigmaMm * GPU_DRR_MODEL_PARAMS.dominantToothSigmaShrinkScale
        ),
        dominantToothBandGate
      );
      supportSigmaMm = mixNumber(
        supportSigmaMm,
        Math.max(
          nativePitchMm * 0.68,
          supportSigmaMm * GPU_DRR_MODEL_PARAMS.broadWeakSigmaShrinkScale
        ),
        broadWeakToothBandGate
      );
      supportSigmaMm = mixNumber(
        supportSigmaMm,
        Math.max(
          nativePitchMm * 0.64,
          supportSigmaMm * GPU_DRR_MODEL_PARAMS.toothBoundarySigmaShrinkScale
        ),
        toothBoundaryRefineGate
      );
      supportSigmaMap[index] = Math.max(
        Math.max(slabHalfThicknessMm * 0.18, nativePitchMm * 1.4),
        supportSigmaMm
      );
    }
  }

  return supportSigmaMap;
}

function buildLocalTransmittanceMap(totalAttenuationMap: Float32Array): Float32Array {
  const localTransmittanceMap = new Float32Array(totalAttenuationMap.length);
  for (let i = 0; i < totalAttenuationMap.length; i++) {
    const attenuation = Math.max(0, Number(totalAttenuationMap[i]));
    localTransmittanceMap[i] = Math.exp(-attenuation);
  }
  return localTransmittanceMap;
}

function buildBackgroundTroughNarrowGateMap(
  supportConfidenceMap: Float32Array,
  supportDensityMap: Float32Array,
  panoWidth: number,
  panoHeight: number
): Float32Array {
  const length = Math.min(supportConfidenceMap.length, supportDensityMap.length);
  const backgroundTroughNarrowGateMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const supportDensity = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const row = panoWidth > 0 ? Math.floor(i / panoWidth) : 0;
    const displayRowNorm =
      panoHeight > 1 ? ((panoHeight - 1) - row) / (panoHeight - 1) : 0;
    const toothBandDistance = Math.abs(displayRowNorm - GPU_SUPPORT_MODEL_PARAMS.toothBandCenter);
    const toothBandPrior =
      1 -
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth,
        GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth,
        toothBandDistance
      );
    const backgroundNarrowConfidenceGate =
      1 - smoothstepNumber(0.0004, 0.0040, supportConfidence);
    const backgroundNarrowDensityGate =
      1 - smoothstepNumber(0.12, 0.32, supportDensity);
    const backgroundNarrowToothBandProtection = mixNumber(1.0, 0.18, toothBandPrior);
    backgroundTroughNarrowGateMap[i] =
      backgroundNarrowConfidenceGate *
      backgroundNarrowDensityGate *
      backgroundNarrowToothBandProtection;
  }

  return backgroundTroughNarrowGateMap;
}

function buildInvalidSupportBlackoutMap(
  supportConfidenceMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportDensityMap: Float32Array,
  participatingSampleCountMap: Float32Array,
  panoWidth: number,
  panoHeight: number,
  slabHalfThicknessMm: number,
  verticalCenterOffsetMm: number
): Float32Array {
  const length = Math.min(
    supportConfidenceMap.length,
    supportSpreadMap.length,
    supportDensityMap.length,
    participatingSampleCountMap.length
  );
  const invalidSupportBlackoutMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const supportSpreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const supportDensity = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const participatingSamples = Math.max(0, Number(participatingSampleCountMap[i]) || 0);
    const row = panoWidth > 0 ? Math.floor(i / panoWidth) : 0;
    const displayRowNorm =
      panoHeight > 1 ? ((panoHeight - 1) - row) / (panoHeight - 1) : 0;
    const weakSupportConfidenceGate =
      1 -
      smoothstepNumber(
        GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutConfidenceLow,
        GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutConfidenceHigh,
        supportConfidence
      );
    const broadSupportSpreadGate = smoothstepNumber(
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutSpreadLow,
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutSpreadHigh,
      supportSpreadMm
    );
    const denseSupportGate = smoothstepNumber(
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutDensityLow,
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutDensityHigh,
      supportDensity
    );
    const participationGate = smoothstepNumber(
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutParticipationLow,
      GPU_TONE_MODEL_PARAMS.invalidSupportBlackoutParticipationHigh,
      participatingSamples
    );
    const focusedBalancedSlabGate =
      1 -
      smoothstepNumber(
        GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectStartMm,
        GPU_DRR_MODEL_PARAMS.focusedBalancedSlabProtectEndMm,
        slabHalfThicknessMm
      );
    const focusedBalancedCenterGate =
      1 -
      smoothstepNumber(
        0,
        GPU_DRR_MODEL_PARAMS.focusedBalancedCenterToleranceMm,
        Math.abs(verticalCenterOffsetMm - GPU_DRR_MODEL_PARAMS.focusedBalancedCenterOffsetMm)
      );
    const focusedBalancedBlackoutProtect =
      focusedBalancedSlabGate *
      focusedBalancedCenterGate *
      smoothstepNumber(0.16, 0.28, displayRowNorm) *
      (1 - smoothstepNumber(0.76, 0.88, displayRowNorm)) *
      smoothstepNumber(0.55, 0.92, supportDensity) *
      smoothstepNumber(0.006, 0.03, supportConfidence);
    invalidSupportBlackoutMap[i] =
      weakSupportConfidenceGate *
      broadSupportSpreadGate *
      denseSupportGate *
      participationGate *
      (1 -
        focusedBalancedBlackoutProtect *
          GPU_TONE_MODEL_PARAMS.focusedBalancedBlackoutProtectScale);
  }

  return invalidSupportBlackoutMap;
}

function buildBackgroundLeakToneMap(
  outputHuMap: Float32Array,
  toneResponseMap: Float32Array,
  supportConfidenceMap: Float32Array,
  lowerPenaltyMap: Float32Array,
  participatingSampleCountMap: Float32Array
): Float32Array {
  const length = Math.min(
    outputHuMap.length,
    toneResponseMap.length,
    supportConfidenceMap.length,
    lowerPenaltyMap.length,
    participatingSampleCountMap.length
  );
  const backgroundLeakToneMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const outputHu = Number(outputHuMap[i]);
    if (!Number.isFinite(outputHu) || outputHu > -300) {
      continue;
    }
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const lowerPenalty = Math.max(0, Number(lowerPenaltyMap[i]) || 0);
    const participatingSamples = Math.max(0, Number(participatingSampleCountMap[i]) || 0);
    if (
      supportConfidence > 0.03 ||
      lowerPenalty > 0.05 ||
      participatingSamples < 0.5
    ) {
      continue;
    }
    backgroundLeakToneMap[i] = clampNumber(Number(toneResponseMap[i]) || 0, 0, 1);
  }

  return backgroundLeakToneMap;
}

function computeDisplayRowNormForJs(row: number, height: number): number {
  return height > 1 ? ((height - 1) - row) / Math.max(height - 1, 1) : 0.5;
}

function computeToothBandPriorForJs(row: number, height: number): number {
  const displayRowNorm = computeDisplayRowNormForJs(row, height);
  const toothBandDistance = Math.abs(displayRowNorm - GPU_SUPPORT_MODEL_PARAMS.toothBandCenter);
  return (
    1 -
    smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth,
      GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth,
      toothBandDistance
    )
  );
}

function computeMiddleBandGateForJs(row: number, height: number): number {
  return smoothstepNumber(0.22, 0.58, computeToothBandPriorForJs(row, height));
}

function buildRetainedSampleMaskMap(
  participatingSampleCountMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportDensityMap: Float32Array,
  width: number,
  height: number
): Float32Array {
  const retainedSampleMaskMap = new Float32Array(participatingSampleCountMap.length);
  for (let i = 0; i < participatingSampleCountMap.length; i++) {
    const participatingSamples = Math.max(0, Number(participatingSampleCountMap[i]) || 0);
    const baseRetainedMask = smoothstepNumber(0.35, 2.6, participatingSamples);
    const row = Math.floor(i / Math.max(width, 1));
    const toothBandPrior = computeToothBandPriorForJs(row, height);
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const supportSpreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const supportDensity = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const compactToothSupportGate =
      toothBandPrior *
      smoothstepNumber(
        GPU_DRR_MODEL_PARAMS.toothBandRetentionConfidenceLow,
        GPU_DRR_MODEL_PARAMS.toothBandRetentionConfidenceHigh,
        supportConfidence
      ) *
      smoothstepNumber(
        GPU_DRR_MODEL_PARAMS.toothBandRetentionDensityLow,
        GPU_DRR_MODEL_PARAMS.toothBandRetentionDensityHigh,
        supportDensity
      ) *
      (1 -
        smoothstepNumber(
          GPU_DRR_MODEL_PARAMS.toothBandRetentionSpreadLow,
          GPU_DRR_MODEL_PARAMS.toothBandRetentionSpreadHigh,
          supportSpreadMm
        ));
    const compactToothParticipationGate = smoothstepNumber(
      GPU_DRR_MODEL_PARAMS.toothBandRetentionParticipationLow,
      GPU_DRR_MODEL_PARAMS.toothBandRetentionParticipationHigh,
      participatingSamples
    );
    const toothBandRetentionProtect =
      compactToothSupportGate *
      compactToothParticipationGate *
      GPU_DRR_MODEL_PARAMS.toothBandRetentionProtectFloor;
    retainedSampleMaskMap[i] = Math.max(baseRetainedMask, toothBandRetentionProtect);
  }

  return retainedSampleMaskMap;
}

function buildMiddleBandLeakMap(
  preToneAccumulationMap: Float32Array,
  retainedSampleMaskMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportDensityMap: Float32Array,
  lowerPenaltyMap: Float32Array,
  width: number,
  height: number
): Float32Array {
  const length = Math.min(
    preToneAccumulationMap.length,
    retainedSampleMaskMap.length,
    supportConfidenceMap.length,
    supportSpreadMap.length,
    supportDensityMap.length,
    lowerPenaltyMap.length
  );
  const middleBandLeakMap = new Float32Array(length);
  const preToneSummary = summarizeFiniteBuffer(preToneAccumulationMap);
  const preToneLeakLow = Math.max(0.002, (preToneSummary?.p50 ?? 0) * 0.55);
  const preToneLeakHigh = Math.max(preToneLeakLow + 0.01, (preToneSummary?.p90 ?? 0) * 0.85);

  for (let i = 0; i < length; i++) {
    const row = Math.floor(i / Math.max(width, 1));
    if (row >= height) {
      break;
    }
    const middleBandGate = computeMiddleBandGateForJs(row, height);
    if (middleBandGate <= 1e-4) {
      continue;
    }

    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const supportSpreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const supportDensity = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const lowerPenalty = Math.max(0, Number(lowerPenaltyMap[i]) || 0);
    const retainedSampleMask = clampNumber(Number(retainedSampleMaskMap[i]) || 0, 0, 1);
    const preToneAccumulation = Math.max(0, Number(preToneAccumulationMap[i]) || 0);
    const weakSupportGate = 1.0 - smoothstepNumber(0.035, 0.14, supportConfidence);
    const lowDensityGate = 1.0 - smoothstepNumber(0.10, 0.30, supportDensity);
    const broadSupportGate = smoothstepNumber(0.30, 0.80, supportSpreadMm);
    const lowPenaltyGate = 1.0 - smoothstepNumber(0.008, 0.040, lowerPenalty);
    const preToneGate = smoothstepNumber(preToneLeakLow, preToneLeakHigh, preToneAccumulation);
    const dominantToothBandGate =
      middleBandGate *
      smoothstepNumber(0.08, 0.24, supportDensity) *
      smoothstepNumber(0.02, 0.12, supportConfidence);
    const toothContinuityApproxGate =
      middleBandGate *
      smoothstepNumber(0.06, 0.20, supportDensity) *
      smoothstepNumber(0.02, 0.10, supportConfidence) *
      (1 - smoothstepNumber(0.46, 0.92, supportSpreadMm));
    const residualToothNearTroughProtect =
      middleBandGate *
      smoothstepNumber(0.06, 0.14, supportConfidence) *
      smoothstepNumber(0.02, 0.08, supportDensity) *
      (1 - smoothstepNumber(0.12, 0.30, supportDensity)) *
      (1 - smoothstepNumber(0.50, 0.92, supportSpreadMm));
    const diffuseLeakProxy =
      middleBandGate *
      retainedSampleMask *
      weakSupportGate *
      lowDensityGate *
      broadSupportGate *
      lowPenaltyGate *
      Math.max(preToneGate, retainedSampleMask * 0.35) *
      (1 - residualToothNearTroughProtect * GPU_DRR_MODEL_PARAMS.residualMiddleLeakProtectScale);
    const weakValidatedMiddleAccumProxy =
      middleBandGate *
      retainedSampleMask *
      (1 - dominantToothBandGate) *
      (1 - toothContinuityApproxGate * 0.88) *
      smoothstepNumber(0.04, 0.14, supportConfidence) *
      (1 - smoothstepNumber(0.12, 0.24, supportConfidence)) *
      smoothstepNumber(0.04, 0.12, supportDensity) *
      (1 - smoothstepNumber(0.12, 0.24, supportDensity)) *
      (1 - smoothstepNumber(0.42, 0.86, supportSpreadMm)) *
      lowPenaltyGate *
      Math.max(preToneGate, retainedSampleMask * 0.22);
    middleBandLeakMap[i] = Math.max(diffuseLeakProxy, weakValidatedMiddleAccumProxy);
  }

  return middleBandLeakMap;
}

function buildMiddleBandLeakDiagnostics(
  preToneAccumulationMap: Float32Array,
  retainedSampleMaskMap: Float32Array,
  middleBandLeakMap: Float32Array,
  width: number,
  height: number
): GpuMiddleBandLeakDiagnostics {
  const length = Math.min(
    preToneAccumulationMap.length,
    retainedSampleMaskMap.length,
    middleBandLeakMap.length,
    width * height
  );
  const preToneSamples: number[] = [];
  const retainedSamples: number[] = [];
  const leakSamples: number[] = [];
  const rowRetainedSums = new Float32Array(height);
  const rowLeakSums = new Float32Array(height);
  const rowWeightSums = new Float32Array(height);
  const columnRetainedSums = new Float32Array(width);
  const columnLeakSums = new Float32Array(width);
  const columnWeightSums = new Float32Array(width);

  for (let i = 0; i < length; i++) {
    const row = Math.floor(i / Math.max(width, 1));
    const col = i % Math.max(width, 1);
    const middleBandGate = computeMiddleBandGateForJs(row, height);
    if (middleBandGate <= 0.05) {
      continue;
    }

    const preToneAccumulation = Math.max(0, Number(preToneAccumulationMap[i]) || 0);
    const retainedSampleMask = clampNumber(Number(retainedSampleMaskMap[i]) || 0, 0, 1);
    const leak = clampNumber(Number(middleBandLeakMap[i]) || 0, 0, 1);
    preToneSamples.push(preToneAccumulation);
    retainedSamples.push(retainedSampleMask);
    leakSamples.push(leak);
    rowRetainedSums[row] += retainedSampleMask * middleBandGate;
    rowLeakSums[row] += leak * middleBandGate;
    rowWeightSums[row] += middleBandGate;
    columnRetainedSums[col] += retainedSampleMask * middleBandGate;
    columnLeakSums[col] += leak * middleBandGate;
    columnWeightSums[col] += middleBandGate;
  }

  const normalizeWeightedAxisMap = (
    sums: Float32Array,
    weightSums: Float32Array
  ): Float32Array => {
    const normalized = new Float32Array(sums.length);
    for (let index = 0; index < sums.length; index++) {
      normalized[index] =
        weightSums[index] > 0 ? sums[index] / Math.max(weightSums[index], 1e-5) : 0;
    }
    return normalized;
  };

  const preToneSummary = summarizeFiniteBuffer(Float32Array.from(preToneSamples));
  const retainedSummary = summarizeFiniteBuffer(Float32Array.from(retainedSamples));
  const leakSummary = summarizeFiniteBuffer(Float32Array.from(leakSamples));
  const rowRetainedSummary = summarizeFiniteBuffer(
    normalizeWeightedAxisMap(rowRetainedSums, rowWeightSums)
  );
  const rowLeakSummary = summarizeFiniteBuffer(normalizeWeightedAxisMap(rowLeakSums, rowWeightSums));
  const columnRetainedSummary = summarizeFiniteBuffer(
    normalizeWeightedAxisMap(columnRetainedSums, columnWeightSums)
  );
  const columnLeakSummary = summarizeFiniteBuffer(
    normalizeWeightedAxisMap(columnLeakSums, columnWeightSums)
  );

  return {
    sampleCount: preToneSamples.length,
    preToneAccumulationP50: roundFinite(preToneSummary?.p50 ?? 0),
    preToneAccumulationP90: roundFinite(preToneSummary?.p90 ?? 0),
    preToneAccumulationP99: roundFinite(preToneSummary?.p99 ?? 0),
    retainedSampleMaskP50: roundFinite(retainedSummary?.p50 ?? 0, 4),
    retainedSampleMaskP90: roundFinite(retainedSummary?.p90 ?? 0, 4),
    retainedSampleMaskFraction05: roundFinite(
      computeFractionByThreshold(Float32Array.from(retainedSamples), value => value > 0.05),
      4
    ),
    leakP50: roundFinite(leakSummary?.p50 ?? 0, 4),
    leakP90: roundFinite(leakSummary?.p90 ?? 0, 4),
    leakP99: roundFinite(leakSummary?.p99 ?? 0, 4),
    leakFraction05: roundFinite(
      computeFractionByThreshold(Float32Array.from(leakSamples), value => value > 0.05),
      4
    ),
    leakFraction10: roundFinite(
      computeFractionByThreshold(Float32Array.from(leakSamples), value => value > 0.1),
      4
    ),
    rowRetainedSampleMaskP10: roundFinite(rowRetainedSummary?.p10 ?? 0, 4),
    rowRetainedSampleMaskP50: roundFinite(rowRetainedSummary?.p50 ?? 0, 4),
    rowRetainedSampleMaskP90: roundFinite(rowRetainedSummary?.p90 ?? 0, 4),
    rowLeakP10: roundFinite(rowLeakSummary?.p10 ?? 0, 4),
    rowLeakP50: roundFinite(rowLeakSummary?.p50 ?? 0, 4),
    rowLeakP90: roundFinite(rowLeakSummary?.p90 ?? 0, 4),
    columnRetainedSampleMaskP10: roundFinite(columnRetainedSummary?.p10 ?? 0, 4),
    columnRetainedSampleMaskP50: roundFinite(columnRetainedSummary?.p50 ?? 0, 4),
    columnRetainedSampleMaskP90: roundFinite(columnRetainedSummary?.p90 ?? 0, 4),
    columnLeakP10: roundFinite(columnLeakSummary?.p10 ?? 0, 4),
    columnLeakP50: roundFinite(columnLeakSummary?.p50 ?? 0, 4),
    columnLeakP90: roundFinite(columnLeakSummary?.p90 ?? 0, 4),
  };
}

function buildThresholdMaskMap(sourceMap: Float32Array, threshold: number): Float32Array {
  const output = new Float32Array(sourceMap.length);
  for (let i = 0; i < sourceMap.length; i++) {
    output[i] = (Number(sourceMap[i]) || 0) > threshold ? 1 : 0;
  }
  return output;
}

function buildSupportPeakDiagnostics(
  peakDominanceMap: Float32Array,
  peakValidityMap: Float32Array,
  peakConflictMap: Float32Array,
  secondPeakRatioMap: Float32Array,
  peakSeparationMap: Float32Array,
  peakAmbiguityMap: Float32Array
): GpuSupportPeakDiagnostics {
  const peakDominanceSummary = summarizeFiniteBuffer(peakDominanceMap);
  const peakValiditySummary = summarizeFiniteBuffer(peakValidityMap);
  const peakConflictSummary = summarizeFiniteBuffer(peakConflictMap);
  const secondPeakRatioSummary = summarizeFiniteBuffer(secondPeakRatioMap);
  const peakSeparationSummary = summarizeFiniteBuffer(peakSeparationMap);
  const peakAmbiguitySummary = summarizeFiniteBuffer(peakAmbiguityMap);

  return {
    peakDominanceP50: roundFinite(peakDominanceSummary?.p50 ?? 0, 4),
    peakDominanceP90: roundFinite(peakDominanceSummary?.p90 ?? 0, 4),
    peakValidityP50: roundFinite(peakValiditySummary?.p50 ?? 0, 4),
    peakValidityP90: roundFinite(peakValiditySummary?.p90 ?? 0, 4),
    peakValidityP99: roundFinite(peakValiditySummary?.p99 ?? 0, 4),
    peakConflictP50: roundFinite(peakConflictSummary?.p50 ?? 0, 4),
    peakConflictP90: roundFinite(peakConflictSummary?.p90 ?? 0, 4),
    peakConflictP99: roundFinite(peakConflictSummary?.p99 ?? 0, 4),
    secondPeakRatioP50: roundFinite(secondPeakRatioSummary?.p50 ?? 0, 4),
    secondPeakRatioP90: roundFinite(secondPeakRatioSummary?.p90 ?? 0, 4),
    peakSeparationP50Mm: roundFinite(peakSeparationSummary?.p50 ?? 0),
    peakSeparationP90Mm: roundFinite(peakSeparationSummary?.p90 ?? 0),
    peakAmbiguityP50: roundFinite(peakAmbiguitySummary?.p50 ?? 0, 4),
    peakAmbiguityP90: roundFinite(peakAmbiguitySummary?.p90 ?? 0, 4),
    peakAmbiguityP99: roundFinite(peakAmbiguitySummary?.p99 ?? 0, 4),
  };
}

function buildRawSupportPeakValidityMap(
  peakDominanceMap: Float32Array,
  secondPeakRatioMap: Float32Array,
  peakAmbiguityMap: Float32Array
): Float32Array {
  const length = Math.min(
    peakDominanceMap.length,
    secondPeakRatioMap.length,
    peakAmbiguityMap.length
  );
  const peakValidityMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const peakDominance = clampNumber(Number(peakDominanceMap[i]) || 0, 0, 1);
    const secondPeakRatio = clampNumber(Number(secondPeakRatioMap[i]) || 0, 0, 1);
    const peakAmbiguity = clampNumber(Number(peakAmbiguityMap[i]) || 0, 0, 1);
    const peakDominanceValidityGate = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityLow,
      GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityHigh,
      peakDominance
    );
    const competingPeakPenalty = mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.secondPeakPenaltyFloor,
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyLow,
        GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyHigh,
        secondPeakRatio
      )
    );
    const ambiguityPenalty = mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyFloor,
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyLow,
        GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyHigh,
        peakAmbiguity
      )
    );
    peakValidityMap[i] = clampNumber(
      peakDominanceValidityGate * competingPeakPenalty * ambiguityPenalty,
      0,
      1
    );
  }

  return peakValidityMap;
}

function buildRawSupportPeakConflictMap(
  peakDominanceMap: Float32Array,
  secondPeakRatioMap: Float32Array,
  supportSpreadMap: Float32Array,
  rawSupportDensityMap: Float32Array,
  nativePitchMmByCol: Float32Array,
  width: number,
  height: number,
  slabHalfThicknessMm: number
): Float32Array {
  const length = Math.min(
    peakDominanceMap.length,
    secondPeakRatioMap.length,
    supportSpreadMap.length,
    rawSupportDensityMap.length
  );
  const peakConflictMap = new Float32Array(length);

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      if (index >= length) {
        break;
      }
      const peakDominance = clampNumber(Number(peakDominanceMap[index]) || 0, 0, 1);
      const secondPeakRatio = clampNumber(Number(secondPeakRatioMap[index]) || 0, 0, 1);
      const supportSpreadMm = Math.max(0, Number(supportSpreadMap[index]) || 0);
      const rawSupportDensity = clampNumber(Number(rawSupportDensityMap[index]) || 0, 0, 1);
      const nativePitchMm = Number(nativePitchMmByCol[col]) || 0.2;
      const broadSupportGate = smoothstepNumber(
        Math.max(nativePitchMm * 1.6, 0.35),
        Math.max(slabHalfThicknessMm * 0.42, nativePitchMm * 3.2),
        supportSpreadMm
      );
      peakConflictMap[index] = clampNumber(
        broadSupportGate *
        (1 -
          smoothstepNumber(
            GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceLow,
            GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceHigh,
            peakDominance
          )) *
        smoothstepNumber(
          GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioLow,
          GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioHigh,
          secondPeakRatio
        ) *
        smoothstepNumber(
          GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityLow,
          GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityHigh,
          rawSupportDensity
        ),
        0,
        1
      );
    }
  }

  return peakConflictMap;
}

function buildDebugDisplayHuFromUnitMap(unitMap: Float32Array): Float32Array {
  const output = new Float32Array(unitMap.length);
  for (let i = 0; i < unitMap.length; i++) {
    const value = clampNumber(Number(unitMap[i]) || 0, 0, 1);
    output[i] = mixNumber(
      GPU_TONE_MODEL_PARAMS.outputHuMin,
      GPU_TONE_MODEL_PARAMS.outputHuMax,
      value
    );
  }
  return output;
}

function normalizedSigmoidToneForJs(exposureValue: number): number {
  const midpoint = GPU_TONE_MODEL_PARAMS.sigmoidMidpoint;
  const slope = GPU_TONE_MODEL_PARAMS.sigmoidSlope;
  const whitePoint = GPU_TONE_MODEL_PARAMS.sigmoidWhitePoint;
  const safeValue = Math.max(exposureValue, 0);
  const low = 1 / (1 + Math.exp(-slope * (0 - midpoint)));
  const high = 1 / (1 + Math.exp(-slope * (whitePoint - midpoint)));
  const curved = 1 / (1 + Math.exp(-slope * (safeValue - midpoint)));
  return clampNumber((curved - low) / Math.max(high - low, 1e-4), 0, 1);
}

function buildPositiveDisplayHuMap(
  sourceMap: Float32Array,
  options?: {
    useLogCompression?: boolean;
  }
): Float32Array {
  const output = new Float32Array(sourceMap.length);
  const summary = summarizeFiniteBuffer(sourceMap);
  const pivot = Math.max(summary?.p99 ?? 0, summary?.p90 ?? 0, 1e-4);
  const logDenominator = Math.max(Math.log1p(pivot), 1e-4);
  const useLogCompression = options?.useLogCompression !== false;
  for (let i = 0; i < sourceMap.length; i++) {
    const value = Math.max(0, Number(sourceMap[i]) || 0);
    const normalized = useLogCompression
      ? clampNumber(Math.log1p(value) / logDenominator, 0, 1)
      : clampNumber(value / Math.max(pivot, 1e-4), 0, 1);
    output[i] = mixNumber(
      GPU_TONE_MODEL_PARAMS.outputHuMin,
      GPU_TONE_MODEL_PARAMS.outputHuMax,
      normalized
    );
  }
  return output;
}

function buildToneBypassHuMap(admissionAccumulationMap: Float32Array): Float32Array {
  const output = new Float32Array(admissionAccumulationMap.length);
  for (let i = 0; i < admissionAccumulationMap.length; i++) {
    const admission = Math.max(0, Number(admissionAccumulationMap[i]) || 0);
    let radiographSignal = normalizedSigmoidToneForJs(
      admission * GPU_TONE_MODEL_PARAMS.exposureScale
    );
    radiographSignal = Math.pow(radiographSignal, GPU_TONE_MODEL_PARAMS.postCurveGamma);
    output[i] = mixNumber(
      GPU_TONE_MODEL_PARAMS.outputHuMin,
      GPU_TONE_MODEL_PARAMS.outputHuMax,
      radiographSignal
    );
  }
  return output;
}

function buildBlackClipMap(toneResponseMap: Float32Array): Float32Array {
  const output = new Float32Array(toneResponseMap.length);
  for (let i = 0; i < toneResponseMap.length; i++) {
    output[i] =
      clampNumber(Number(toneResponseMap[i]) || 0, 0, 1) <= TONE_RESPONSE_BLACK_CLIP_THRESHOLD
        ? 1
        : 0;
  }
  return output;
}

function buildToneStageSuppressionMap(
  admissionAccumulationMap: Float32Array,
  toneSuppressedAccumulationMap: Float32Array
): Float32Array {
  const length = Math.min(admissionAccumulationMap.length, toneSuppressedAccumulationMap.length);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const admission = Math.max(0, Number(admissionAccumulationMap[i]) || 0);
    const suppressed = Math.max(0, Number(toneSuppressedAccumulationMap[i]) || 0);
    if (admission <= 1e-6) {
      output[i] = 0;
      continue;
    }
    output[i] = clampNumber((admission - suppressed) / Math.max(admission, 1e-5), 0, 1);
  }
  return output;
}

function buildSupportFailureDisplayMap(
  peakAmbiguityMap: Float32Array,
  rawSupportLocalJumpMap: Float32Array,
  rawSupportContinuityMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportDensityMap: Float32Array
): Float32Array {
  const length = Math.min(
    peakAmbiguityMap.length,
    rawSupportLocalJumpMap.length,
    rawSupportContinuityMap.length,
    supportSpreadMap.length,
    supportDensityMap.length
  );
  const supportFailureDisplayMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const peakAmbiguity = clampNumber(Number(peakAmbiguityMap[i]) || 0, 0, 1);
    const localJumpMm = Math.max(0, Number(rawSupportLocalJumpMap[i]) || 0);
    const continuity = clampNumber(Number(rawSupportContinuityMap[i]) || 0, 0, 1);
    const spreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const density = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const jumpGate = smoothstepNumber(
      0,
      GPU_SUPPORT_MODEL_PARAMS.continuityNeighborAgreementSigmaMm,
      computeContinuityExcessJumpMm(localJumpMm, spreadMm, density)
    );
    const continuityFailure = 1 - continuity;
    const continuityFailureGate = smoothstepNumber(0.08, 0.72, continuityFailure);
    const jumpFailure = jumpGate * mixNumber(0.35, 1.0, continuityFailureGate);
    supportFailureDisplayMap[i] = clampNumber(
      Math.max(peakAmbiguity, jumpFailure),
      0,
      1
    );
  }

  return supportFailureDisplayMap;
}

type SupportFormationDebugMaps = {
  toothBandPriorMap: Float32Array;
  rawSupportScoreGapMap: Float32Array;
  dominantDensePeakGateMap: Float32Array;
  toothBandStructureGuardMap: Float32Array;
  ambiguousBroadSupportPenaltyGateMap: Float32Array;
  protectedAmbiguousBroadSupportPenaltyGateMap: Float32Array;
  structuralSupportGateMap: Float32Array;
  peakStructureValidityMap: Float32Array;
  supportValidityMap: Float32Array;
  rowConfidenceGateMap: Float32Array;
  falseSupportConfidenceGateMap: Float32Array;
  falseSupportDensityGateMap: Float32Array;
  falseSupportSpreadGateMap: Float32Array;
  falseSupportVetoMap: Float32Array;
  rowBackgroundDensityGateMap: Float32Array;
  rowBackgroundSpreadGateMap: Float32Array;
  rowBackgroundPeakHuGateMap: Float32Array;
  rowBackgroundEdgeGateMap: Float32Array;
  rowBackgroundVetoMap: Float32Array;
  supportVetoTriggeredMap: Float32Array;
};

function buildSupportFormationDebugMaps(params: {
  rawSupportSpreadMap: Float32Array;
  rawSupportDensityMap: Float32Array;
  rawSupportPeakDominanceMap: Float32Array;
  rawSupportSecondPeakRatioMap: Float32Array;
  rawSupportPeakSeparationMap: Float32Array;
  rawSupportPeakAmbiguityMap: Float32Array;
  rawSupportDenseFractionMap: Float32Array;
  rawSupportPeakHuSupportGateMap: Float32Array;
  supportSpreadMap: Float32Array;
  supportConfidenceMap: Float32Array;
  nativePitchMmByCol: Float32Array;
  width: number;
  height: number;
  slabHalfThicknessMm: number;
}): SupportFormationDebugMaps {
  const {
    rawSupportSpreadMap,
    rawSupportDensityMap,
    rawSupportPeakDominanceMap,
    rawSupportSecondPeakRatioMap,
    rawSupportPeakSeparationMap,
    rawSupportPeakAmbiguityMap,
    rawSupportDenseFractionMap,
    rawSupportPeakHuSupportGateMap,
    supportSpreadMap,
    supportConfidenceMap,
    nativePitchMmByCol,
    width,
    height,
    slabHalfThicknessMm,
  } = params;
  const length = Math.min(
    rawSupportSpreadMap.length,
    rawSupportDensityMap.length,
    rawSupportPeakDominanceMap.length,
    rawSupportSecondPeakRatioMap.length,
    rawSupportPeakSeparationMap.length,
    rawSupportPeakAmbiguityMap.length,
    rawSupportDenseFractionMap.length,
    rawSupportPeakHuSupportGateMap.length,
    supportSpreadMap.length,
    supportConfidenceMap.length
  );

  const toothBandPriorMap = new Float32Array(length);
  const rawSupportScoreGapMap = new Float32Array(length);
  const dominantDensePeakGateMap = new Float32Array(length);
  const toothBandStructureGuardMap = new Float32Array(length);
  const ambiguousBroadSupportPenaltyGateMap = new Float32Array(length);
  const protectedAmbiguousBroadSupportPenaltyGateMap = new Float32Array(length);
  const structuralSupportGateMap = new Float32Array(length);
  const peakStructureValidityMap = new Float32Array(length);
  const supportValidityMap = new Float32Array(length);
  const rowConfidenceGateMap = new Float32Array(length);
  const falseSupportConfidenceGateMap = new Float32Array(length);
  const falseSupportDensityGateMap = new Float32Array(length);
  const falseSupportSpreadGateMap = new Float32Array(length);
  const falseSupportVetoMap = new Float32Array(length);
  const rowBackgroundDensityGateMap = new Float32Array(length);
  const rowBackgroundSpreadGateMap = new Float32Array(length);
  const rowBackgroundPeakHuGateMap = new Float32Array(length);
  const rowBackgroundEdgeGateMap = new Float32Array(length);
  const rowBackgroundVetoMap = new Float32Array(length);
  const supportVetoTriggeredMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const col = width > 0 ? i % width : 0;
    const row = width > 0 ? Math.floor(i / width) : 0;
    const displayRowNorm = computeDisplayRowNorm(row, height);
    const nativePitchMm = Math.max(0.01, Number(nativePitchMmByCol[col]) || 0.2);
    const rawSupportSpreadMm = Math.max(0, Number(rawSupportSpreadMap[i]) || 0);
    const supportSpreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const rawSupportDensity = clampNumber(Number(rawSupportDensityMap[i]) || 0, 0, 1);
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const peakDominance = clampNumber(Number(rawSupportPeakDominanceMap[i]) || 0, 0, 1);
    const secondPeakRatio = clampNumber(Number(rawSupportSecondPeakRatioMap[i]) || 0, 0, 1);
    const peakSeparationMm = Math.max(0, Number(rawSupportPeakSeparationMap[i]) || 0);
    const peakAmbiguity = clampNumber(Number(rawSupportPeakAmbiguityMap[i]) || 0, 0, 1);
    const denseSupportFraction = clampNumber(Number(rawSupportDenseFractionMap[i]) || 0, 0, 1);
    const peakHuSupportGate = clampNumber(Number(rawSupportPeakHuSupportGateMap[i]) || 0, 0, 1);

    const toothBandDistance = Math.abs(
      displayRowNorm - GPU_SUPPORT_MODEL_PARAMS.toothBandCenter
    );
    const toothBandPrior =
      1 -
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth,
        GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth,
        toothBandDistance
      );
    toothBandPriorMap[i] = clampNumber(toothBandPrior, 0, 1);

    const inferiorPenalty = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyStart,
      GPU_SUPPORT_MODEL_PARAMS.inferiorPenaltyEnd,
      displayRowNorm
    );
    const superiorPenalty =
      1 -
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyStart,
        GPU_SUPPORT_MODEL_PARAMS.superiorPenaltyEnd,
        displayRowNorm
      );
    const secondPeakComparableGate = smoothstepNumber(0.3, 0.72, secondPeakRatio);
    const broadSupportGate = smoothstepNumber(
      Math.max(nativePitchMm * 1.6, 0.35),
      Math.max(slabHalfThicknessMm * 0.42, nativePitchMm * 3.2),
      rawSupportSpreadMm
    );
    const weakDominanceGate = 1 - smoothstepNumber(0.18, 0.42, peakDominance);
    const peakDominanceBlendGate = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakCenterBlendLow,
      GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakCenterBlendHigh,
      peakDominance
    );
    const densePeakBlendGate = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakDensityLow,
      GPU_SUPPORT_MODEL_PARAMS.dominantDensePeakDensityHigh,
      denseSupportFraction
    );
    const dominantDensePeakGate =
      peakDominanceBlendGate *
      densePeakBlendGate *
      (1 - smoothstepNumber(0.12, 0.42, peakAmbiguity));

    rawSupportScoreGapMap[i] = clampNumber(
      peakDominance * (1 - secondPeakRatio),
      0,
      1
    );

    const toothBandStructureGuard =
      toothBandPrior *
      peakHuSupportGate *
      smoothstepNumber(0.04, 0.16, rawSupportDensity) *
      Math.max(
        smoothstepNumber(0.08, 0.24, peakDominance),
        smoothstepNumber(0.1, 0.28, denseSupportFraction)
      );
    toothBandStructureGuardMap[i] = clampNumber(toothBandStructureGuard, 0, 1);
    const compactToothBandStructureGuard =
      toothBandPrior *
      peakHuSupportGate *
      smoothstepNumber(0.06, 0.20, rawSupportDensity) *
      Math.max(
        smoothstepNumber(0.10, 0.28, peakDominance),
        smoothstepNumber(0.12, 0.32, denseSupportFraction)
      ) *
      (1 - broadSupportGate);
    const toothBandPeakValidityGuard = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapToothGuardLow,
      GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapToothGuardHigh,
      Math.max(toothBandStructureGuard, compactToothBandStructureGuard)
    );
    const guardedDominantDensePeakGate =
      dominantDensePeakGate *
      mixNumber(
        GPU_SUPPORT_MODEL_PARAMS.dominantPeakSnapGuardFloor,
        1,
        toothBandPeakValidityGuard
      );
    const weakToothDensePeakGate =
      toothBandPrior *
      dominantDensePeakGate *
      smoothstepNumber(0.06, 0.18, rawSupportDensity) *
      (1 - toothBandPeakValidityGuard) *
      (1 - smoothstepNumber(0.18, 0.40, toothBandStructureGuard));
    dominantDensePeakGateMap[i] = clampNumber(guardedDominantDensePeakGate, 0, 1);

    const ambiguousBroadSupportPenaltyGate =
      broadSupportGate *
      secondPeakComparableGate *
      (1 - dominantDensePeakGate) *
      mixNumber(0.45, 1, weakDominanceGate);
    ambiguousBroadSupportPenaltyGateMap[i] = clampNumber(
      ambiguousBroadSupportPenaltyGate,
      0,
      1
    );

    const protectedAmbiguousBroadSupportPenaltyGate =
      ambiguousBroadSupportPenaltyGate * (1 - toothBandStructureGuard * 0.72);
    protectedAmbiguousBroadSupportPenaltyGateMap[i] = clampNumber(
      protectedAmbiguousBroadSupportPenaltyGate,
      0,
      1
    );

    const structuralSupportGate = Math.max(
      smoothstepNumber(0.28, 0.62, denseSupportFraction),
      smoothstepNumber(0.14, 0.42, peakDominance)
    );
    structuralSupportGateMap[i] = clampNumber(structuralSupportGate, 0, 1);

    const nonDominantBroadSupportGate =
      broadSupportGate *
      (1 -
        smoothstepNumber(
          GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceLow,
          GPU_SUPPORT_MODEL_PARAMS.nonDominantPeakDominanceHigh,
          peakDominance
        )) *
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioLow,
        GPU_SUPPORT_MODEL_PARAMS.nonDominantSecondPeakRatioHigh,
        secondPeakRatio
      ) *
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityLow,
        GPU_SUPPORT_MODEL_PARAMS.nonDominantDensityHigh,
        rawSupportDensity
      );
    const spreadConfidence =
      1 -
      smoothstepNumber(
        nativePitchMm * 1.5,
        Math.max(slabHalfThicknessMm * 0.75, nativePitchMm * 2.8),
        supportSpreadMm
      );
    const peakDominanceValidityGate = smoothstepNumber(
      GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityLow,
      GPU_SUPPORT_MODEL_PARAMS.peakDominanceValidityHigh,
      peakDominance
    );
    const competingPeakPenalty = mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.secondPeakPenaltyFloor,
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyLow,
        GPU_SUPPORT_MODEL_PARAMS.secondPeakRatioPenaltyHigh,
        secondPeakRatio
      )
    );
    const ambiguityPenalty = mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyFloor,
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyLow,
        GPU_SUPPORT_MODEL_PARAMS.peakAmbiguityPenaltyHigh,
        peakAmbiguity
      )
    );
    const peakStructureValidity =
      peakDominanceValidityGate * competingPeakPenalty * ambiguityPenalty;
    peakStructureValidityMap[i] = clampNumber(peakStructureValidity, 0, 1);
    const nonToothDominantPeakGate =
      dominantDensePeakGate * (1 - toothBandPeakValidityGuard);
    const toothBoundaryAmbiguityGate =
      toothBandStructureGuard *
      peakHuSupportGate *
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.toothBoundaryDensityLow,
        GPU_SUPPORT_MODEL_PARAMS.toothBoundaryDensityHigh,
        rawSupportDensity
      ) *
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.toothBoundarySecondPeakRatioLow,
        GPU_SUPPORT_MODEL_PARAMS.toothBoundarySecondPeakRatioHigh,
        secondPeakRatio
      ) *
      broadSupportGate *
      (1 - smoothstepNumber(0.22, 0.48, peakDominance));
    const toothBoundaryMidpointGate =
      toothBoundaryAmbiguityGate *
      smoothstepNumber(
        Math.max(nativePitchMm * 1.4, 0.22),
        Math.max(slabHalfThicknessMm * 0.26, nativePitchMm * 2.4),
        peakSeparationMm
      );

    let supportValidity =
      structuralSupportGate *
      peakHuSupportGate *
      clampNumber(spreadConfidence, 0, 1) *
      peakStructureValidity *
      mixNumber(
        1,
        GPU_SUPPORT_MODEL_PARAMS.nonDominantConfidencePenaltyFloor,
        nonDominantBroadSupportGate
      );
    supportValidity *= mixNumber(
      1,
      0.52,
      protectedAmbiguousBroadSupportPenaltyGate
    );
    supportValidity *= mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.nonToothDominantValidityPenaltyFloor,
      nonToothDominantPeakGate
    );
    supportValidity *= mixNumber(
      1,
      GPU_SUPPORT_MODEL_PARAMS.weakToothDensePeakValidityPenaltyFloor,
      weakToothDensePeakGate
    );
    supportValidity = Math.max(
      supportValidity,
      toothBandStructureGuard * mixNumber(0.22, 0.34, 1 - broadSupportGate)
    );
    supportValidity = Math.max(
      supportValidity,
      Math.max(toothBoundaryAmbiguityGate, toothBoundaryMidpointGate) *
        toothBandStructureGuard *
        GPU_SUPPORT_MODEL_PARAMS.toothBoundaryValidityFloor
    );
    supportValidityMap[i] = clampNumber(supportValidity, 0, 1);

    let rowConfidenceGate = clampNumber(
      mixNumber(0.72, 1.14, toothBandPrior) *
        mixNumber(1, 0.5, inferiorPenalty) *
        mixNumber(0.88, 1, 1 - superiorPenalty),
      0.26,
      1.18
    );
    rowConfidenceGate = clampNumber(
      rowConfidenceGate * mixNumber(1, 1.08, toothBandStructureGuard),
      0.26,
      1.22
    );
    rowConfidenceGateMap[i] = rowConfidenceGate;

    const falseSupportConfidenceGate =
      1 - smoothstepNumber(0.004, 0.016, supportConfidence);
    const falseSupportDensityGate = smoothstepNumber(0.04, 0.12, rawSupportDensity);
    const falseSupportSpreadGate = smoothstepNumber(
      Math.max(nativePitchMm * 2.0, 0.45),
      Math.max(slabHalfThicknessMm * 0.42, nativePitchMm * 4.0),
      supportSpreadMm
    );
    const falseSupportStructureWeakGate =
      1 - smoothstepNumber(0.16, 0.4, structuralSupportGate);
    const falseSupportVeto =
      falseSupportConfidenceGate *
      falseSupportDensityGate *
      falseSupportSpreadGate *
      falseSupportStructureWeakGate;
    falseSupportConfidenceGateMap[i] = clampNumber(falseSupportConfidenceGate, 0, 1);
    falseSupportDensityGateMap[i] = clampNumber(falseSupportDensityGate, 0, 1);
    falseSupportSpreadGateMap[i] = clampNumber(falseSupportSpreadGate, 0, 1);
    falseSupportVetoMap[i] = clampNumber(falseSupportVeto, 0, 1);

    const nonToothBandGate = 1 - smoothstepNumber(0.06, 0.18, toothBandPrior);
    const rowBackgroundDensityGate = smoothstepNumber(0.04, 0.16, rawSupportDensity);
    const rowBackgroundSpreadGate = smoothstepNumber(
      Math.max(nativePitchMm * 1.6, 0.34),
      Math.max(slabHalfThicknessMm * 0.34, nativePitchMm * 3.2),
      supportSpreadMm
    );
    const rowBackgroundPeakHuGate = 1 - peakHuSupportGate;
    const rowBackgroundEdgeGate = Math.max(
      1 - smoothstepNumber(0.16, 0.32, displayRowNorm),
      smoothstepNumber(0.78, 0.94, displayRowNorm)
    );
    const rowBackgroundVeto =
      nonToothBandGate *
      falseSupportConfidenceGate *
      rowBackgroundDensityGate *
      rowBackgroundSpreadGate *
      mixNumber(0.58, 1, rowBackgroundPeakHuGate) *
      mixNumber(0.9, 1.28, rowBackgroundEdgeGate);
    rowBackgroundDensityGateMap[i] = clampNumber(rowBackgroundDensityGate, 0, 1);
    rowBackgroundSpreadGateMap[i] = clampNumber(rowBackgroundSpreadGate, 0, 1);
    rowBackgroundPeakHuGateMap[i] = clampNumber(rowBackgroundPeakHuGate, 0, 1);
    rowBackgroundEdgeGateMap[i] = clampNumber(rowBackgroundEdgeGate, 0, 1);
    rowBackgroundVetoMap[i] = clampNumber(rowBackgroundVeto, 0, 1);

    const vetoThreshold = mixNumber(
      0.46,
      0.24,
      1 - smoothstepNumber(0.36, 0.78, peakDominance)
    );
    supportVetoTriggeredMap[i] =
      Math.max(falseSupportVeto, rowBackgroundVeto) > vetoThreshold ? 1 : 0;
  }

  return {
    toothBandPriorMap,
    rawSupportScoreGapMap,
    dominantDensePeakGateMap,
    toothBandStructureGuardMap,
    ambiguousBroadSupportPenaltyGateMap,
    protectedAmbiguousBroadSupportPenaltyGateMap,
    structuralSupportGateMap,
    peakStructureValidityMap,
    supportValidityMap,
    rowConfidenceGateMap,
    falseSupportConfidenceGateMap,
    falseSupportDensityGateMap,
    falseSupportSpreadGateMap,
    falseSupportVetoMap,
    rowBackgroundDensityGateMap,
    rowBackgroundSpreadGateMap,
    rowBackgroundPeakHuGateMap,
    rowBackgroundEdgeGateMap,
    rowBackgroundVetoMap,
    supportVetoTriggeredMap,
  };
}

function buildSupportFormationDiagnostics(params: {
  supportConfidenceMap: Float32Array;
  supportSpreadMap: Float32Array;
  supportDensityMap: Float32Array;
  rawSupportScoreGapMap: Float32Array;
  dominantDensePeakGateMap: Float32Array;
  toothBandStructureGuardMap: Float32Array;
  protectedAmbiguousBroadSupportPenaltyGateMap: Float32Array;
  supportValidityMap: Float32Array;
  falseSupportVetoMap: Float32Array;
  rowBackgroundVetoMap: Float32Array;
  supportVetoTriggeredMap: Float32Array;
  rawSupportDepthMap: Float32Array;
  supportDepthMap: Float32Array;
  width: number;
  height: number;
}): GpuSupportFormationDiagnostics {
  const {
    supportConfidenceMap,
    supportSpreadMap,
    supportDensityMap,
    rawSupportScoreGapMap,
    dominantDensePeakGateMap,
    toothBandStructureGuardMap,
    protectedAmbiguousBroadSupportPenaltyGateMap,
    supportValidityMap,
    falseSupportVetoMap,
    rowBackgroundVetoMap,
    supportVetoTriggeredMap,
    rawSupportDepthMap,
    supportDepthMap,
    width,
    height,
  } = params;
  const columnConfidence = collapseWeightedMapByColumn(
    supportConfidenceMap,
    supportConfidenceMap,
    width,
    height
  );
  const columnSpread = collapseWeightedMapByColumn(
    supportSpreadMap,
    supportConfidenceMap,
    width,
    height
  );
  const columnDensity = collapseWeightedMapByColumn(
    supportDensityMap,
    supportConfidenceMap,
    width,
    height
  );
  const rowConfidence = collapseWeightedMapByRow(
    supportConfidenceMap,
    supportConfidenceMap,
    width,
    height
  );
  const rowSpread = collapseWeightedMapByRow(
    supportSpreadMap,
    supportConfidenceMap,
    width,
    height
  );
  const rowDensity = collapseWeightedMapByRow(
    supportDensityMap,
    supportConfidenceMap,
    width,
    height
  );
  const columnScoreGap = collapseWeightedMapByColumn(
    rawSupportScoreGapMap,
    supportConfidenceMap,
    width,
    height
  );
  const columnAmbiguityPenalty = collapseWeightedMapByColumn(
    protectedAmbiguousBroadSupportPenaltyGateMap,
    supportConfidenceMap,
    width,
    height
  );
  const depthDriftMap = new Float32Array(rawSupportDepthMap.length);
  for (let i = 0; i < depthDriftMap.length; i++) {
    depthDriftMap[i] = Math.abs(
      (Number(supportDepthMap[i]) || 0) - (Number(rawSupportDepthMap[i]) || 0)
    );
  }
  const columnDepthDrift = collapseWeightedMapByColumn(
    depthDriftMap,
    supportConfidenceMap,
    width,
    height
  );

  const dominantPeakSummary = summarizeFiniteBuffer(dominantDensePeakGateMap);
  const toothGuardSummary = summarizeFiniteBuffer(toothBandStructureGuardMap);
  const ambiguityPenaltySummary = summarizeFiniteBuffer(
    protectedAmbiguousBroadSupportPenaltyGateMap
  );
  const supportValiditySummary = summarizeFiniteBuffer(supportValidityMap);
  const rawScoreGapSummary = summarizeFiniteBuffer(rawSupportScoreGapMap);
  const columnConfidenceSummary = summarizeFiniteBuffer(columnConfidence);
  const columnSpreadSummary = summarizeFiniteBuffer(columnSpread);
  const columnDensitySummary = summarizeFiniteBuffer(columnDensity);
  const rowConfidenceSummary = summarizeFiniteBuffer(rowConfidence);
  const rowSpreadSummary = summarizeFiniteBuffer(rowSpread);
  const rowDensitySummary = summarizeFiniteBuffer(rowDensity);
  const columnScoreGapSummary = summarizeFiniteBuffer(columnScoreGap);
  const columnDepthDriftSummary = summarizeFiniteBuffer(columnDepthDrift);

  return {
    dominantDensePeakGateP50: roundFinite(dominantPeakSummary?.p50 ?? 0, 4),
    dominantDensePeakGateP90: roundFinite(dominantPeakSummary?.p90 ?? 0, 4),
    toothBandStructureGuardP50: roundFinite(toothGuardSummary?.p50 ?? 0, 4),
    toothBandStructureGuardP90: roundFinite(toothGuardSummary?.p90 ?? 0, 4),
    protectedAmbiguousBroadSupportPenaltyGateP50: roundFinite(
      ambiguityPenaltySummary?.p50 ?? 0,
      4
    ),
    protectedAmbiguousBroadSupportPenaltyGateP90: roundFinite(
      ambiguityPenaltySummary?.p90 ?? 0,
      4
    ),
    supportValidityP10: roundFinite(supportValiditySummary?.p10 ?? 0, 4),
    supportValidityP50: roundFinite(supportValiditySummary?.p50 ?? 0, 4),
    supportValidityP90: roundFinite(supportValiditySummary?.p90 ?? 0, 4),
    rawScoreGapP10: roundFinite(rawScoreGapSummary?.p10 ?? 0, 4),
    rawScoreGapP50: roundFinite(columnScoreGapSummary?.p50 ?? rawScoreGapSummary?.p50 ?? 0, 4),
    rawScoreGapP90: roundFinite(rawScoreGapSummary?.p90 ?? 0, 4),
    falseSupportVetoFraction: roundFinite(
      computeFractionByThreshold(falseSupportVetoMap, value => value > 0.42),
      4
    ),
    rowBackgroundVetoFraction: roundFinite(
      computeFractionByThreshold(rowBackgroundVetoMap, value => value > 0.42),
      4
    ),
    supportVetoTriggeredFraction: roundFinite(
      computeFractionByThreshold(supportVetoTriggeredMap, value => value >= 0.5),
      4
    ),
    columnSupportConfidenceP10: roundFinite(columnConfidenceSummary?.p10 ?? 0, 4),
    columnSupportConfidenceP50: roundFinite(columnConfidenceSummary?.p50 ?? 0, 4),
    columnSupportConfidenceP90: roundFinite(columnConfidenceSummary?.p90 ?? 0, 4),
    columnSupportSpreadP10Mm: roundFinite(columnSpreadSummary?.p10 ?? 0),
    columnSupportSpreadP50Mm: roundFinite(columnSpreadSummary?.p50 ?? 0),
    columnSupportSpreadP90Mm: roundFinite(columnSpreadSummary?.p90 ?? 0),
    columnSupportDensityP10: roundFinite(columnDensitySummary?.p10 ?? 0, 4),
    columnSupportDensityP50: roundFinite(columnDensitySummary?.p50 ?? 0, 4),
    columnSupportDensityP90: roundFinite(columnDensitySummary?.p90 ?? 0, 4),
    rowSupportConfidenceP10: roundFinite(rowConfidenceSummary?.p10 ?? 0, 4),
    rowSupportConfidenceP50: roundFinite(rowConfidenceSummary?.p50 ?? 0, 4),
    rowSupportConfidenceP90: roundFinite(rowConfidenceSummary?.p90 ?? 0, 4),
    rowSupportSpreadP10Mm: roundFinite(rowSpreadSummary?.p10 ?? 0),
    rowSupportSpreadP50Mm: roundFinite(rowSpreadSummary?.p50 ?? 0),
    rowSupportSpreadP90Mm: roundFinite(rowSpreadSummary?.p90 ?? 0),
    rowSupportDensityP10: roundFinite(rowDensitySummary?.p10 ?? 0, 4),
    rowSupportDensityP50: roundFinite(rowDensitySummary?.p50 ?? 0, 4),
    rowSupportDensityP90: roundFinite(rowDensitySummary?.p90 ?? 0, 4),
  };
}

function buildDrrSupportGuidanceMaps(
  supportSigmaMap: Float32Array,
  backgroundTroughNarrowGateMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportDensityMap: Float32Array,
  nativePitchMmByCol: Float32Array,
  panoWidth: number,
  panoHeight: number,
  slabHalfThicknessMm: number
): {
  effectiveTroughHalfWidthMap: Float32Array;
  continuityExpandedTroughHalfWidthMap: Float32Array;
  dominantToothBandGateMap: Float32Array;
  broadWeakToothBandGateMap: Float32Array;
  toothContinuityAdmissionGateMap: Float32Array;
} {
  const length = Math.min(
    supportSigmaMap.length,
    backgroundTroughNarrowGateMap.length,
    supportSpreadMap.length,
    supportConfidenceMap.length,
    supportDensityMap.length
  );
  const effectiveTroughHalfWidthMap = new Float32Array(length);
  const continuityExpandedTroughHalfWidthMap = new Float32Array(length);
  const dominantToothBandGateMap = new Float32Array(length);
  const broadWeakToothBandGateMap = new Float32Array(length);
  const toothContinuityAdmissionGateMap = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const rawSupportSpreadMm = Math.max(0, Number(supportSpreadMap[i]) || 0);
    const supportConfidence = clampNumber(Number(supportConfidenceMap[i]) || 0, 0, 1);
    const supportDensity = clampNumber(Number(supportDensityMap[i]) || 0, 0, 1);
    const noSupportRay =
      supportConfidence <= 1e-6 &&
      supportDensity <= 1e-6 &&
      rawSupportSpreadMm <= 1e-6;
    if (noSupportRay) {
      effectiveTroughHalfWidthMap[i] = 0;
      continue;
    }
    const supportSigmaMm = Math.max(0, Number(supportSigmaMap[i]) || 0);
    const backgroundTroughNarrowGate = clampNumber(
      Number(backgroundTroughNarrowGateMap[i]) || 0,
      0,
      1
    );
    const col = panoWidth > 0 ? i % panoWidth : 0;
    const row = panoWidth > 0 ? Math.floor(i / panoWidth) : 0;
    const nativePitchMm = Math.max(0.01, Number(nativePitchMmByCol[col]) || 0.2);
    const displayRowNorm = computeDisplayRowNorm(row, panoHeight);
    const toothBandDistance = Math.abs(
      displayRowNorm - GPU_SUPPORT_MODEL_PARAMS.toothBandCenter
    );
    const toothBandPrior =
      1 -
      smoothstepNumber(
        GPU_SUPPORT_MODEL_PARAMS.toothBandInnerHalfWidth,
        GPU_SUPPORT_MODEL_PARAMS.toothBandOuterHalfWidth,
        toothBandDistance
      );
    const dominantToothBandGate =
      toothBandPrior *
      smoothstepNumber(0.08, 0.24, supportDensity) *
      smoothstepNumber(0.02, 0.12, supportConfidence);
    dominantToothBandGateMap[i] = clampNumber(dominantToothBandGate, 0, 1);
    const broadWeakToothBandGate =
      toothBandPrior *
      smoothstepNumber(
        Math.max(nativePitchMm * 1.6, 0.34),
        Math.max(slabHalfThicknessMm * 0.36, nativePitchMm * 3.0),
        rawSupportSpreadMm
      ) *
      (1 - smoothstepNumber(0.10, 0.26, supportDensity)) *
      (1 - smoothstepNumber(0.02, 0.10, supportConfidence));
    broadWeakToothBandGateMap[i] = clampNumber(broadWeakToothBandGate, 0, 1);
    const troughHalfWidthMm =
      supportSigmaMm *
      GPU_DRR_MODEL_PARAMS.approxTroughBoundarySigmaMultiplier *
      mixNumber(1.0, GPU_DRR_MODEL_PARAMS.backgroundTroughNarrowScale, backgroundTroughNarrowGate);
    const troughSlabFloor = nativePitchMm * 1.0;
    const nominalTroughHalfWidthMm = Math.max(
      troughHalfWidthMm,
      troughSlabFloor
    );
    const toothContinuityAdmissionGate =
      toothBandPrior *
      smoothstepNumber(0.06, 0.20, supportDensity) *
      smoothstepNumber(0.02, 0.10, supportConfidence) *
      (1 -
        smoothstepNumber(
          Math.max(nativePitchMm * 2.2, 0.46),
          Math.max(slabHalfThicknessMm * 0.50, nativePitchMm * 4.0),
          rawSupportSpreadMm
        ));
    toothContinuityAdmissionGateMap[i] = clampNumber(toothContinuityAdmissionGate, 0, 1);
    const continuityExpandedTroughHalfWidthMm =
      nominalTroughHalfWidthMm *
      mixNumber(
        1.0,
        GPU_DRR_MODEL_PARAMS.toothContinuityAdmissionExpandScale,
        toothContinuityAdmissionGate
      );
    continuityExpandedTroughHalfWidthMap[i] = continuityExpandedTroughHalfWidthMm;
    const toothBandValidityExpandGate = clampNumber(
      Math.max(dominantToothBandGate, toothContinuityAdmissionGate) *
        (1 - broadWeakToothBandGate * 0.55),
      0,
      1
    );
    const toothBandProtectedTroughHalfWidthMm = Math.max(
      continuityExpandedTroughHalfWidthMm,
      nominalTroughHalfWidthMm *
        mixNumber(
          1.0,
          GPU_DRR_MODEL_PARAMS.toothBandValidityTroughExpandScale,
          toothBandValidityExpandGate
        )
    );
    effectiveTroughHalfWidthMap[i] = toothBandProtectedTroughHalfWidthMm;
  }

  return {
    effectiveTroughHalfWidthMap,
    continuityExpandedTroughHalfWidthMap,
    dominantToothBandGateMap,
    broadWeakToothBandGateMap,
    toothContinuityAdmissionGateMap,
  };
}

function computeFractionByThreshold(
  buffer: Float32Array,
  predicate: (value: number) => boolean
): number {
  if (!buffer.length) {
    return 0;
  }

  let finiteCount = 0;
  let matchedCount = 0;
  for (let i = 0; i < buffer.length; i++) {
    const value = Number(buffer[i]);
    if (!Number.isFinite(value)) {
      continue;
    }

    finiteCount++;
    if (predicate(value)) {
      matchedCount++;
    }
  }

  return finiteCount > 0 ? matchedCount / finiteCount : 0;
}

function sumByteLength(buffers: Array<Float32Array | undefined>): number {
  let total = 0;
  for (let i = 0; i < buffers.length; i++) {
    total += buffers[i]?.byteLength ?? 0;
  }
  return total;
}

function buildSupportContinuityMaps(
  supportDepthMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportDensityMap: Float32Array,
  supportSpreadMap: Float32Array,
  width: number,
  height: number
): {
  localJumpMap: Float32Array;
  continuityMap: Float32Array;
} {
  const planeSize = width * height;
  const localJumpMap = new Float32Array(planeSize);
  const continuityMap = new Float32Array(planeSize);

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const index = row * width + col;
      const centerDepth = Number(supportDepthMap[index]);
      const centerConfidence = clampNumber(Number(supportConfidenceMap[index]) || 0, 0, 1);
      const centerDensity = clampNumber(Number(supportDensityMap[index]) || 0, 0, 1);
      const centerSpreadMm = Math.max(0, Number(supportSpreadMap[index]) || 0);
      const centerSupportStrength = Math.max(centerConfidence, centerDensity * 0.35);

      if (!Number.isFinite(centerDepth) || centerSupportStrength <= 1e-4) {
        localJumpMap[index] = 0;
        continuityMap[index] = 1;
        continue;
      }

      let deltaSum = 0;
      let deltaCount = 0;

      if (col > 0) {
        const leftIndex = index - 1;
        const leftDepth = Number(supportDepthMap[leftIndex]);
        const leftConfidence = clampNumber(Number(supportConfidenceMap[leftIndex]) || 0, 0, 1);
        const leftDensity = clampNumber(Number(supportDensityMap[leftIndex]) || 0, 0, 1);
        const leftSupportStrength = Math.max(leftConfidence, leftDensity * 0.35);
        if (Number.isFinite(leftDepth) && leftSupportStrength > 1e-4) {
          deltaSum += Math.abs(centerDepth - leftDepth);
          deltaCount++;
        }
      }

      if (col + 1 < width) {
        const rightIndex = index + 1;
        const rightDepth = Number(supportDepthMap[rightIndex]);
        const rightConfidence = clampNumber(Number(supportConfidenceMap[rightIndex]) || 0, 0, 1);
        const rightDensity = clampNumber(Number(supportDensityMap[rightIndex]) || 0, 0, 1);
        const rightSupportStrength = Math.max(rightConfidence, rightDensity * 0.35);
        if (Number.isFinite(rightDepth) && rightSupportStrength > 1e-4) {
          deltaSum += Math.abs(centerDepth - rightDepth);
          deltaCount++;
        }
      }

      if (!deltaCount) {
        localJumpMap[index] = 0;
        continuityMap[index] = 1;
        continue;
      }

      const localJumpMm = deltaSum / deltaCount;
      const excessJumpMm = computeContinuityExcessJumpMm(
        localJumpMm,
        centerSpreadMm,
        centerDensity
      );
      localJumpMap[index] = excessJumpMm;
      continuityMap[index] = 1 - smoothstepNumber(
        0,
        GPU_SUPPORT_MODEL_PARAMS.continuityNeighborAgreementSigmaMm,
        excessJumpMm
      );
    }
  }

  return {
    localJumpMap,
    continuityMap,
  };
}

function buildSupportSurfaceDiagnostics(
  supportDepthMap: Float32Array,
  supportConfidenceMap: Float32Array,
  supportSpreadMap: Float32Array,
  supportDensityMap: Float32Array,
  width: number,
  height: number,
  slabHalfThicknessMm: number,
  localJumpMap?: Float32Array,
  continuityMap?: Float32Array,
  supportAmbiguityMap?: Float32Array,
  scoreGapMap?: Float32Array,
  rawSupportDepthMap?: Float32Array
): GpuSupportSurfaceDiagnostics {
  const collapsedPath = collapseSupportPathByColumn(supportDepthMap, supportConfidenceMap, width, height);
  const pathDepthSummary = summarizeFiniteBuffer(collapsedPath.pathDepthMm);
  const pathConfidenceSummary = summarizeFiniteBuffer(collapsedPath.pathConfidence);
  const confidenceSummary = summarizeFiniteBuffer(supportConfidenceMap);
  const spreadSummary = summarizeFiniteBuffer(supportSpreadMap);
  const densitySummary = summarizeFiniteBuffer(supportDensityMap);
  const resolvedLocalJumpMap =
    localJumpMap ??
    buildSupportContinuityMaps(
      supportDepthMap,
      supportConfidenceMap,
      supportDensityMap,
      supportSpreadMap,
      width,
      height
    ).localJumpMap;
  const resolvedContinuityMap =
    continuityMap ??
    buildSupportContinuityMaps(
      supportDepthMap,
      supportConfidenceMap,
      supportDensityMap,
      supportSpreadMap,
      width,
      height
    ).continuityMap;
  const collapsedContinuity = collapseWeightedMapByColumn(
    resolvedContinuityMap,
    supportConfidenceMap,
    width,
    height
  );
  const lowConfidenceColumnFraction = computeFractionByThreshold(
    collapsedPath.pathConfidence,
    value => value <= 0.08
  );
  const lowContinuityColumnFraction = computeFractionByThreshold(
    collapsedContinuity,
    value => value <= 0.45
  );
  const unstableColumnFraction = computeFractionByThreshold(
    collapsedContinuity,
    value => value <= 0.35
  );
  const longestUnstableRunColumns = computeLongestRunByThreshold(
    collapsedContinuity,
    value => value <= 0.35
  );
  const localJumpSummary = summarizeFiniteBuffer(resolvedLocalJumpMap);
  const continuitySummary = summarizeFiniteBuffer(resolvedContinuityMap);
  const collapsedAmbiguity =
    supportAmbiguityMap
      ? collapseWeightedMapByColumn(
          supportAmbiguityMap,
          supportConfidenceMap,
          width,
          height
        )
      : null;
  const collapsedScoreGap =
    scoreGapMap
      ? collapseWeightedMapByColumn(scoreGapMap, supportConfidenceMap, width, height)
      : null;
  const collapsedDepthDrift =
    rawSupportDepthMap
      ? (() => {
          const driftMap = new Float32Array(rawSupportDepthMap.length);
          for (let i = 0; i < driftMap.length; i++) {
            driftMap[i] = Math.abs(
              (Number(supportDepthMap[i]) || 0) - (Number(rawSupportDepthMap[i]) || 0)
            );
          }
          return collapseWeightedMapByColumn(driftMap, supportConfidenceMap, width, height);
        })()
      : null;
  const scoreGapSummary = collapsedScoreGap ? summarizeFiniteBuffer(collapsedScoreGap) : null;
  const depthDriftSummary = collapsedDepthDrift ? summarizeFiniteBuffer(collapsedDepthDrift) : null;

  return {
    depthMinMm: roundFinite(pathDepthSummary?.min ?? 0),
    depthMaxMm: roundFinite(pathDepthSummary?.max ?? 0),
    depthStdMm: roundFinite(pathDepthSummary?.stdDev ?? 0),
    pathJumpP95Mm: roundFinite(computeAdjacentDeltaP95(collapsedPath.pathDepthMm)),
    pathConfidenceP10: roundFinite(pathConfidenceSummary?.p10 ?? 0, 4),
    pathConfidenceP50: roundFinite(pathConfidenceSummary?.p50 ?? 0, 4),
    pathConfidenceP90: roundFinite(pathConfidenceSummary?.p90 ?? 0, 4),
    localJumpP50Mm: roundFinite(localJumpSummary?.p50 ?? 0),
    localJumpP90Mm: roundFinite(localJumpSummary?.p90 ?? 0),
    localJumpP95Mm: roundFinite(localJumpSummary?.p95 ?? 0),
    localJumpOutlierFraction: roundFinite(
      computeFractionByThreshold(resolvedLocalJumpMap, value => value >= 0.35),
      4
    ),
    lowConfidenceColumnFraction: roundFinite(lowConfidenceColumnFraction, 4),
    lowContinuityColumnFraction: roundFinite(lowContinuityColumnFraction, 4),
    unstableColumnFraction: roundFinite(unstableColumnFraction, 4),
    longestUnstableRunColumns,
    continuityP10: roundFinite(continuitySummary?.p10 ?? 0, 4),
    continuityP50: roundFinite(continuitySummary?.p50 ?? 0, 4),
    continuityP90: roundFinite(continuitySummary?.p90 ?? 0, 4),
    clampFraction: roundFinite(computeClampFraction(collapsedPath.pathDepthMm, slabHalfThicknessMm), 4),
    confidenceP10: roundFinite(confidenceSummary?.p10 ?? 0),
    confidenceP50: roundFinite(confidenceSummary?.p50 ?? 0),
    confidenceP90: roundFinite(confidenceSummary?.p90 ?? 0),
    spreadP50Mm: roundFinite(spreadSummary?.p50 ?? 0),
    spreadP90Mm: roundFinite(spreadSummary?.p90 ?? 0),
    densityP50: roundFinite(densitySummary?.p50 ?? 0),
    densityP90: roundFinite(densitySummary?.p90 ?? 0),
    ambiguousColumnFraction: roundFinite(
      collapsedAmbiguity
        ? computeFractionByThreshold(collapsedAmbiguity, value => value > 0.18)
        : 0,
      4
    ),
    forcedDriftFraction: roundFinite(
      collapsedDepthDrift
        ? computeFractionByThreshold(collapsedDepthDrift, value => value > 0.16)
        : 0,
      4
    ),
    bestDepthDriftP95Mm: roundFinite(depthDriftSummary?.p95 ?? 0),
    scoreGapP50: roundFinite(scoreGapSummary?.p50 ?? 0, 4),
    selectedDepthFirst8Mm: Array.from(
      collapsedPath.pathDepthMm.subarray(0, Math.min(8, collapsedPath.pathDepthMm.length))
    ).map(value => roundFinite(Number(value))),
  };
}

function buildDrrDiagnostics(
  totalAttenuationMap: Float32Array,
  fogAttenuationMap: Float32Array,
  lowerPenaltyMap: Float32Array,
  participatingSampleCountMap: Float32Array,
  preToneAccumulationMap: Float32Array,
  retainedSampleMaskMap: Float32Array,
  middleBandLeakMap: Float32Array,
  supportSigmaMap: Float32Array,
  effectiveTroughHalfWidthMap: Float32Array,
  backgroundTroughNarrowGateMap: Float32Array,
  localTransmittanceMap: Float32Array,
  nativePitchMmByCol: Float32Array,
  panoWidth: number,
  panoHeight: number,
  slabHalfThicknessMm: number,
  requestedSlabSamples: number
): GpuDrrDiagnostics {
  const raySampleCounts = new Float32Array(nativePitchMmByCol.length);
  const slabWidthMm = Math.max(0, slabHalfThicknessMm * 2);
  for (let col = 0; col < nativePitchMmByCol.length; col++) {
    raySampleCounts[col] = computeRaySampleCountForJs(
      slabWidthMm,
      nativePitchMmByCol[col],
      requestedSlabSamples
    );
  }

  const rayCountSummary = summarizeFiniteBuffer(raySampleCounts);
  const sigmaSummary = summarizeFiniteBuffer(supportSigmaMap);
  const effectiveTroughSummary = summarizeFiniteBuffer(effectiveTroughHalfWidthMap);
  const totalSummary = summarizeFiniteBuffer(totalAttenuationMap);
  const fogSummary = summarizeFiniteBuffer(fogAttenuationMap);
  const lowerPenaltySummary = summarizeFiniteBuffer(lowerPenaltyMap);
  const participatingSummary = summarizeFiniteBuffer(participatingSampleCountMap);
  const transmittanceSummary = summarizeFiniteBuffer(localTransmittanceMap);
  const backgroundTroughNarrowGateSummary = summarizeFiniteBuffer(backgroundTroughNarrowGateMap);
  const middleBandLeakDiagnostics = buildMiddleBandLeakDiagnostics(
    preToneAccumulationMap,
    retainedSampleMaskMap,
    middleBandLeakMap,
    panoWidth,
    panoHeight
  );

  return {
    slabHalfThicknessMm: roundFinite(slabHalfThicknessMm),
    requestedSlabSamples,
    effectiveRaySampleCountMin: roundFinite(rayCountSummary?.min ?? 0),
    effectiveRaySampleCountP50: roundFinite(rayCountSummary?.p50 ?? 0),
    effectiveRaySampleCountP90: roundFinite(rayCountSummary?.p90 ?? 0),
    effectiveRaySampleCountMax: roundFinite(rayCountSummary?.max ?? 0),
    troughSigmaP10Mm: roundFinite(sigmaSummary?.p10 ?? 0),
    troughSigmaP50Mm: roundFinite(sigmaSummary?.p50 ?? 0),
    troughSigmaP90Mm: roundFinite(sigmaSummary?.p90 ?? 0),
    approxTroughHalfWidthP50Mm: roundFinite(
      (sigmaSummary?.p50 ?? 0) * GPU_DRR_MODEL_PARAMS.approxTroughBoundarySigmaMultiplier
    ),
    effectiveTroughHalfWidthP10Mm: roundFinite(effectiveTroughSummary?.p10 ?? 0),
    effectiveTroughHalfWidthP50Mm: roundFinite(effectiveTroughSummary?.p50 ?? 0),
    effectiveTroughHalfWidthP90Mm: roundFinite(effectiveTroughSummary?.p90 ?? 0),
    totalAttenuationP10: roundFinite(totalSummary?.p10 ?? 0),
    totalAttenuationP50: roundFinite(totalSummary?.p50 ?? 0),
    totalAttenuationP90: roundFinite(totalSummary?.p90 ?? 0),
    fogAttenuationP50: roundFinite(fogSummary?.p50 ?? 0),
    fogAttenuationP90: roundFinite(fogSummary?.p90 ?? 0),
    lowerPenaltyP50: roundFinite(lowerPenaltySummary?.p50 ?? 0),
    lowerPenaltyP90: roundFinite(lowerPenaltySummary?.p90 ?? 0),
    participatingSamplesP10: roundFinite(participatingSummary?.p10 ?? 0),
    participatingSamplesP50: roundFinite(participatingSummary?.p50 ?? 0),
    participatingSamplesP90: roundFinite(participatingSummary?.p90 ?? 0),
    localTransmittanceP10: roundFinite(transmittanceSummary?.p10 ?? 0, 4),
    localTransmittanceP50: roundFinite(transmittanceSummary?.p50 ?? 0, 4),
    localTransmittanceP90: roundFinite(transmittanceSummary?.p90 ?? 0, 4),
    backgroundTroughNarrowGateP50: roundFinite(backgroundTroughNarrowGateSummary?.p50 ?? 0, 4),
    backgroundTroughNarrowGateP90: roundFinite(backgroundTroughNarrowGateSummary?.p90 ?? 0, 4),
    middleBandLeak: middleBandLeakDiagnostics,
    drrModel: {
      supportWeightPowerLowConfidence: GPU_DRR_MODEL_PARAMS.supportWeightPowerLowConfidence,
      supportWeightPowerHighConfidence: GPU_DRR_MODEL_PARAMS.supportWeightPowerHighConfidence,
      lowerPenaltyDenseScale: GPU_DRR_MODEL_PARAMS.lowerPenaltyDenseScale,
      lowerPenaltyRowStart: GPU_DRR_MODEL_PARAMS.lowerPenaltyRowStart,
      lowerPenaltyRowEnd: GPU_DRR_MODEL_PARAMS.lowerPenaltyRowEnd,
      focusedBalancedCenterOffsetMm: GPU_DRR_MODEL_PARAMS.focusedBalancedCenterOffsetMm,
      focusedBalancedCenterToleranceMm: GPU_DRR_MODEL_PARAMS.focusedBalancedCenterToleranceMm,
      focusedBalancedLeakContributionFloor:
        GPU_DRR_MODEL_PARAMS.focusedBalancedLeakContributionFloor,
      focusedBalancedMiddleLeakProtectScale:
        GPU_DRR_MODEL_PARAMS.focusedBalancedMiddleLeakProtectScale,
      troughSigmaHardCapMm: GPU_DRR_MODEL_PARAMS.troughSigmaHardCapMm,
    },
  };
}

function buildToneMapDiagnostics(
  inputAttenuationMap: Float32Array,
  fogAttenuationMap: Float32Array,
  lowerPenaltyMap: Float32Array,
  invalidSupportBlackoutMap: Float32Array,
  supportFailureDisplayMap: Float32Array,
  toneResponseMap: Float32Array,
  outputHuMap: Float32Array
): GpuToneMapDiagnostics {
  const inputSummary = summarizeFiniteBuffer(inputAttenuationMap);
  const fogSummary = summarizeFiniteBuffer(fogAttenuationMap);
  const penaltySummary = summarizeFiniteBuffer(lowerPenaltyMap);
  const invalidSupportBlackoutSummary = summarizeFiniteBuffer(invalidSupportBlackoutMap);
  const supportFailureDisplaySummary = summarizeFiniteBuffer(supportFailureDisplayMap);
  const toneSummary = summarizeFiniteBuffer(toneResponseMap);
  const outputSummary = summarizeFiniteBuffer(outputHuMap);

  return {
    inputAttenuationP01: roundFinite(inputSummary?.p01 ?? 0),
    inputAttenuationP50: roundFinite(inputSummary?.p50 ?? 0),
    inputAttenuationP99: roundFinite(inputSummary?.p99 ?? 0),
    fogAttenuationP50: roundFinite(fogSummary?.p50 ?? 0),
    fogAttenuationP90: roundFinite(fogSummary?.p90 ?? 0),
    toneResponseP01: roundFinite(toneSummary?.p01 ?? 0, 4),
    toneResponseP50: roundFinite(toneSummary?.p50 ?? 0, 4),
    toneResponseP99: roundFinite(toneSummary?.p99 ?? 0, 4),
    outputHuP01: roundFinite(outputSummary?.p01 ?? 0),
    outputHuP50: roundFinite(outputSummary?.p50 ?? 0),
    outputHuP99: roundFinite(outputSummary?.p99 ?? 0),
    lowerPenaltyP50: roundFinite(penaltySummary?.p50 ?? 0),
    lowerPenaltyP90: roundFinite(penaltySummary?.p90 ?? 0),
    invalidSupportBlackoutP50: roundFinite(invalidSupportBlackoutSummary?.p50 ?? 0, 4),
    invalidSupportBlackoutP90: roundFinite(invalidSupportBlackoutSummary?.p90 ?? 0, 4),
    invalidSupportBlackoutP99: roundFinite(invalidSupportBlackoutSummary?.p99 ?? 0, 4),
    supportFailureDisplayP50: roundFinite(supportFailureDisplaySummary?.p50 ?? 0, 4),
    supportFailureDisplayP90: roundFinite(supportFailureDisplaySummary?.p90 ?? 0, 4),
    supportFailureDisplayP99: roundFinite(supportFailureDisplaySummary?.p99 ?? 0, 4),
    temporaryDisplayMode: GPU_TEMP_DEBUG_DISPLAY_SUPPORT_FAILURE_MAP
      ? 'support-failure-map'
      : GPU_TEMP_DEBUG_DISPLAY_PEAK_AMBIGUITY_MAP
        ? 'peak-ambiguity-map'
        : GPU_TEMP_DEBUG_DISPLAY_INVALID_SUPPORT_MAP
          ? 'invalid-support-map'
          : null,
    blackClipFraction: roundFinite(
      computeFractionByThreshold(
        toneResponseMap,
        value => value <= TONE_RESPONSE_BLACK_CLIP_THRESHOLD
      ),
      4
    ),
    whiteClipFraction: roundFinite(
      computeFractionByThreshold(
        toneResponseMap,
        value => value >= TONE_RESPONSE_WHITE_CLIP_THRESHOLD
      ),
      4
    ),
    toneCurve: {
      exposureScale: GPU_TONE_MODEL_PARAMS.exposureScale,
      sigmoidMidpoint: GPU_TONE_MODEL_PARAMS.sigmoidMidpoint,
      sigmoidSlope: GPU_TONE_MODEL_PARAMS.sigmoidSlope,
      sigmoidWhitePoint: GPU_TONE_MODEL_PARAMS.sigmoidWhitePoint,
      postCurveGamma: GPU_TONE_MODEL_PARAMS.postCurveGamma,
      outputHuMin: GPU_TONE_MODEL_PARAMS.outputHuMin,
      outputHuMax: GPU_TONE_MODEL_PARAMS.outputHuMax,
      lowConfidenceFogSuppression: GPU_TONE_MODEL_PARAMS.lowConfidenceFogSuppression,
      highConfidenceFogSuppression: GPU_TONE_MODEL_PARAMS.highConfidenceFogSuppression,
      lowConfidenceFogRetention: GPU_TONE_MODEL_PARAMS.lowConfidenceFogRetention,
      highConfidenceFogRetention: GPU_TONE_MODEL_PARAMS.highConfidenceFogRetention,
      lowConfidencePenaltySuppression: GPU_TONE_MODEL_PARAMS.lowConfidencePenaltySuppression,
      highConfidencePenaltySuppression: GPU_TONE_MODEL_PARAMS.highConfidencePenaltySuppression,
      lowConfidencePenaltyRetention: GPU_TONE_MODEL_PARAMS.lowConfidencePenaltyRetention,
      highConfidencePenaltyRetention: GPU_TONE_MODEL_PARAMS.highConfidencePenaltyRetention,
      focusedBalancedBackgroundProtectScale:
        GPU_TONE_MODEL_PARAMS.focusedBalancedBackgroundProtectScale,
      focusedBalancedBlackoutProtectScale:
        GPU_TONE_MODEL_PARAMS.focusedBalancedBlackoutProtectScale,
    },
  };
}

function resetGpuRendererState(): void {
  if (_gl) {
    try {
      if (_volumeTex) _gl.deleteTexture(_volumeTex);
      if (_splineTex) _gl.deleteTexture(_splineTex);
      destroyPipelineTargets(_gl);
      if (_panoProgram) _gl.deleteProgram(_panoProgram);
      if (_supportProgram) _gl.deleteProgram(_supportProgram);
      if (_supportPeakDebugProgram) _gl.deleteProgram(_supportPeakDebugProgram);
      if (_supportModelDiagnosticProgram) _gl.deleteProgram(_supportModelDiagnosticProgram);
      if (_supportSmoothProgram) _gl.deleteProgram(_supportSmoothProgram);
      if (_drrProgram) _gl.deleteProgram(_drrProgram);
      if (_toneProgram) _gl.deleteProgram(_toneProgram);
      if (_vao) _gl.deleteVertexArray(_vao);
    } catch (cleanupError) {
      console.warn('[CPR-GPU] Failed to clean up WebGL resources.', cleanupError);
    }
  }

  const domCanvas = _canvas as (HTMLCanvasElement & { parentNode?: ParentNode | null }) | null;
  if (domCanvas?.parentNode?.removeChild) {
    try {
      domCanvas.parentNode.removeChild(domCanvas);
    } catch {
      // Ignore DOM cleanup failures during disposal.
    }
  }

  _volumeTex = null;
  _splineTex = null;
  _supportTexA = null;
  _supportTexB = null;
  _supportPeakDebugTex = null;
  _drrTex = null;
  _supportFboA = null;
  _supportFboB = null;
  _supportPeakDebugFbo = null;
  _drrFbo = null;
  _fboTex = null;
  _fbo = null;
  _panoProgram = null;
  _supportProgram = null;
  _supportPeakDebugProgram = null;
  _supportModelDiagnosticProgram = null;
  _supportSmoothProgram = null;
  _drrProgram = null;
  _toneProgram = null;
  _vao = null;
  _gl = null;
  _canvas = null;
  _cachedVolumeId = null;
  _cachedVolumeNormalizationSignature = null;
  _hasFloatLinearFiltering = false;
  _fboWidth = 0;
  _fboHeight = 0;
}

// ─── Init ────────────────────────────────────────────────────────────
function ensureGpuContext(): WebGL2RenderingContext {
  if (_gl) {
    let contextLost = false;
    try {
      contextLost = typeof _gl.isContextLost === 'function' && _gl.isContextLost();
    } catch {
      contextLost = true;
    }

    if (!contextLost) {
      return _gl;
    }

    console.warn('[CPR-GPU] WebGL context was lost. Recreating GPU renderer state.');
    resetGpuRendererState();
  }

  // Prefer OffscreenCanvas (no DOM element needed)
  try {
    _canvas = new OffscreenCanvas(1, 1);
    _gl = _canvas.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;
  } catch {
    _canvas = null;
    _gl = null;
  }

  const canUseDocument = typeof document !== 'undefined' && !!document?.createElement;
  if (!_gl && canUseDocument) {
    // Fallback: hidden DOM canvas
    const c = document.createElement('canvas');
    c.width = 1;
    c.height = 1;
    c.style.display = 'none';
    document.body?.appendChild(c);
    _canvas = c;
    _gl = c.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
    }) as WebGL2RenderingContext | null;
  }

  if (!_gl) {
    throw new Error('WebGL2 not available — cannot use GPU panoramic renderer.');
  }

  // Need float color buffer for HU output
  const ext = _gl.getExtension('EXT_color_buffer_float');
  if (!ext) {
    throw new Error('EXT_color_buffer_float not available — cannot render float FBO.');
  }

  // Need float-linear for hardware trilinear interpolation on R32F volume texture
  const floatLinearExt = _gl.getExtension('OES_texture_float_linear');
  if (!floatLinearExt) {
    throw new Error('OES_texture_float_linear not available — cannot do trilinear R32F volume filtering.');
  }
  _hasFloatLinearFiltering = true;

  _panoProgram = compileProgram(_gl, FOCAL_TROUGH_DRR_FRAG_SRC);
  _supportProgram = compileProgram(_gl, SUPPORT_FRAG_SRC);
  _supportPeakDebugProgram = compileProgram(_gl, SUPPORT_PEAK_DEBUG_FRAG_SRC);
  _supportModelDiagnosticProgram = compileProgram(_gl, SUPPORT_MODEL_DIAGNOSTIC_FRAG_SRC);
  _supportSmoothProgram = compileProgram(_gl, SUPPORT_SMOOTH_FRAG_SRC);
  _drrProgram = compileProgram(_gl, DRR_FRAG_SRC);
  _toneProgram = compileProgram(_gl, TONE_FRAG_SRC);

  // Empty VAO for fullscreen triangle (no attributes needed)
  _vao = _gl.createVertexArray()!;

  console.log('[CPR-GPU] WebGL2 context initialized.');
  return _gl;
}

// ─── Volume Texture Upload ──────────────────────────────────────────
function uploadVolumeTexture(
  gl: WebGL2RenderingContext,
  scalarData: Float32Array | Int16Array | Uint16Array,
  dims: [number, number, number],
  normalizationSignature?: string | null,
  normalizeStoredSample?: (value: number) => number,
  volumeId?: string
): void {
  const effectiveNormalizationSignature = normalizeStoredSample
    ? (normalizationSignature ?? 'normalized:unspecified')
    : 'raw';
  // Skip re-upload if same volume
  if (
    volumeId &&
    volumeId === _cachedVolumeId &&
    _cachedVolumeNormalizationSignature === effectiveNormalizationSignature &&
    _volumeTex
  ) {
    return;
  }

  if (_volumeTex) {
    gl.deleteTexture(_volumeTex);
  }

  const [nx, ny, nz] = dims;
  _volumeTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_3D, _volumeTex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  // Allocate 3D texture storage (R32F for full precision — avoids half-float quantization)
  const voxelCount = nx * ny * nz;
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

  if (scalarData.length !== voxelCount) {
    throw new Error(
      `[CPR-GPU] Volume upload length mismatch: expected ${voxelCount}, got ${scalarData.length}.`
    );
  }

  let uploadData: Float32Array;
  if (!normalizeStoredSample && scalarData instanceof Float32Array) {
    uploadData = scalarData;
  } else {
    uploadData = new Float32Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      const sourceValue = scalarData[i];
      uploadData[i] = normalizeStoredSample ? normalizeStoredSample(sourceValue) : sourceValue;
    }
  }

  const finalFloatUpload = new Float32Array(uploadData);

  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, finalFloatUpload);

  let uploadMin = Infinity;
  let uploadMax = -Infinity;
  const uploadStep = Math.max(1, Math.floor(finalFloatUpload.length / 20000));
  for (let i = 0; i < finalFloatUpload.length; i += uploadStep) {
    const value = finalFloatUpload[i];
    if (value < uploadMin) uploadMin = value;
    if (value > uploadMax) uploadMax = value;
  }
  console.log('[CPR-GPU] Volume texture upload stats (bulk domain, no rescale)', {
    nx,
    ny,
    nz,
    uploadMin,
    uploadMax,
    first5: Array.from(finalFloatUpload.subarray(0, 5)),
  });

  _cachedVolumeId = volumeId ?? null;
  _cachedVolumeNormalizationSignature = effectiveNormalizationSignature;
  console.log(
    `[CPR-GPU] Volume texture uploaded: ${nx}×${ny}×${nz} (R32F, ${effectiveNormalizationSignature})`
  );
}

// ─── Spline Data Texture ────────────────────────────────────────────
function uploadSplineTexture(
  gl: WebGL2RenderingContext,
  frames: GpuPanoInput['frames'],
  panoWidth: number,
  fallbackVerticalDir: [number, number, number]
): Array<{
  position: [number, number, number];
  slabDir: [number, number, number];
  verticalDir: [number, number, number];
}> {
  if (_splineTex) {
    gl.deleteTexture(_splineTex);
  }

  // Pack positions (row 0), slab normals (row 1), and per-column vertical
  // directions (row 2) into RGBA32F texture.
  const data = new Float32Array(panoWidth * 3 * 4); // RGBA × width × 3 rows

  let fallbackS = fallbackVerticalDir;
  const fallbackLen = Math.hypot(fallbackS[0], fallbackS[1], fallbackS[2]);
  if (fallbackLen > 1e-8) {
    fallbackS = [fallbackS[0] / fallbackLen, fallbackS[1] / fallbackLen, fallbackS[2] / fallbackLen];
  } else {
    fallbackS = [0, 0, 1];
  }

  // Flip-correct slab normals (same logic as CPU worker)
  let prevSlabDir: [number, number, number] | null = null;
  let prevVerticalDir: [number, number, number] | null = null;
  const debugVertices: Array<{
    position: [number, number, number];
    slabDir: [number, number, number];
    verticalDir: [number, number, number];
  }> = [];
  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const p = frame.position;

    // Normalize slab dir
    let sd = frame.N_slab;
    const len = Math.sqrt(sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]);
    if (len > 1e-8) {
      sd = [sd[0] / len, sd[1] / len, sd[2] / len];
    }
    // Flip correction
    if (prevSlabDir) {
      const dot = prevSlabDir[0] * sd[0] + prevSlabDir[1] * sd[1] + prevSlabDir[2] * sd[2];
      if (dot < 0) {
        sd = [-sd[0], -sd[1], -sd[2]];
      }
    }
    prevSlabDir = sd;

    let vd = frame.S ?? fallbackS;
    const verticalLen = Math.sqrt(vd[0] * vd[0] + vd[1] * vd[1] + vd[2] * vd[2]);
    if (verticalLen > 1e-8) {
      vd = [vd[0] / verticalLen, vd[1] / verticalLen, vd[2] / verticalLen];
    } else {
      vd = fallbackS;
    }
    if (prevVerticalDir) {
      const dot = prevVerticalDir[0] * vd[0] + prevVerticalDir[1] * vd[1] + prevVerticalDir[2] * vd[2];
      if (dot < 0) {
        vd = [-vd[0], -vd[1], -vd[2]];
      }
    }
    prevVerticalDir = vd;

    // Row 0: position
    const r0 = col * 4;
    data[r0] = p[0];
    data[r0 + 1] = p[1];
    data[r0 + 2] = p[2];
    data[r0 + 3] = 0;

    // Row 1: slab normal
    const r1 = (panoWidth + col) * 4;
    data[r1] = sd[0];
    data[r1 + 1] = sd[1];
    data[r1 + 2] = sd[2];
    data[r1 + 3] = 0;

    // Row 2: per-column vertical direction
    const r2 = (panoWidth * 2 + col) * 4;
    data[r2] = vd[0];
    data[r2 + 1] = vd[1];
    data[r2 + 2] = vd[2];
    data[r2 + 3] = 0;

    if (col < 3) {
      debugVertices.push({
        position: [p[0], p[1], p[2]],
        slabDir: [sd[0], sd[1], sd[2]],
        verticalDir: [vd[0], vd[1], vd[2]],
      });
    }
  }

  _splineTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, _splineTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, panoWidth, 3, 0, gl.RGBA, gl.FLOAT, data);
  return debugVertices;
}

// ─── FBO Setup ──────────────────────────────────────────────────────
function ensureFbo(gl: WebGL2RenderingContext, w: number, h: number): void {
  if (_fbo && _fboWidth === w && _fboHeight === h) return;

  if (_fbo) {
    gl.deleteFramebuffer(_fbo);
    gl.deleteTexture(_fboTex);
  }

  _fboTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, _fboTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  _fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, _fboTex, 0);

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`FBO incomplete: ${status}`);
  }

  _fboWidth = w;
  _fboHeight = h;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function readFramebufferChannels(
  gl: WebGL2RenderingContext,
  width: number,
  height: number
): FramebufferReadback {
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  const rgbaBuffer = new Float32Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, rgbaBuffer);
  const readbackError = gl.getError();
  if (readbackError !== gl.NO_ERROR) {
    throw new Error(`GPU pano readPixels failed with WebGL error ${readbackError}.`);
  }

  const channel0 = new Float32Array(width * height);
  const channel1 = new Float32Array(width * height);
  const channel2 = new Float32Array(width * height);
  const channel3 = new Float32Array(width * height);
  let rawMin = Infinity;
  let rawMax = -Infinity;

  for (let row = 0; row < height; row++) {
    const srcRow = height - 1 - row;
    for (let col = 0; col < width; col++) {
      const dstIndex = row * width + col;
      const srcIndex = (srcRow * width + col) * 4;
      const channel0Value = rgbaBuffer[srcIndex];
      channel0[dstIndex] = channel0Value;
      channel1[dstIndex] = rgbaBuffer[srcIndex + 1];
      channel2[dstIndex] = rgbaBuffer[srcIndex + 2];
      channel3[dstIndex] = rgbaBuffer[srcIndex + 3];
      if (Number.isFinite(channel0Value)) {
        if (channel0Value < rawMin) rawMin = channel0Value;
        if (channel0Value > rawMax) rawMax = channel0Value;
      }
    }
  }

  return {
    channel0,
    channel1,
    channel2,
    channel3,
    rawMin,
    rawMax,
  };
}

// ─── Main Render Function ───────────────────────────────────────────
function inpaintLowParticipation(
  pixelData: Float32Array,
  participationMap: Float32Array,
  width: number,
  height: number,
  participationThreshold: number,
  minValidNeighbors: number
): void {
  const output = new Float32Array(pixelData);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const idx = row * width + col;
      if (participationMap[idx] >= participationThreshold) continue;

      const neighborValues: number[] = [];
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -4; dx <= 4; dx++) {
          const nr = row + dy;
          const nc = col + dx;
          if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
          const nidx = nr * width + nc;
          if (participationMap[nidx] >= participationThreshold) {
            neighborValues.push(pixelData[nidx]);
          }
        }
      }
      if (neighborValues.length >= minValidNeighbors) {
        neighborValues.sort((a, b) => a - b);
        output[idx] = neighborValues[Math.floor(neighborValues.length / 2)];
      }
    }
  }
  pixelData.set(output);
}

export function renderPanoGpu(input: GpuPanoInput, volumeId?: string): GpuPanoResult {
  const gl = ensureGpuContext();
  const {
    scalarData, dimensions, worldToIndex,
    frames, panoWidth, panoHeight,
    verticalDir, vertHalfMm, verticalCenterOffsetMm,
    slabHalfThicknessMm, slabSamples,
    rescaleSlope, rescaleIntercept, applyRescale,
    normalizationSignature,
    normalizeStoredSample,
    returnDebugSidecars = false,
  } = input;
  const safePanoWidth = Math.max(1, Math.floor(Number(panoWidth) || 1));
  const safePanoHeight = Math.max(1, Math.floor(Number(panoHeight) || 1));
  const t0 = performance.now();
  const formatReadableGpuValue = (value: number | null | undefined, fractionDigits = 1): string =>
    Number.isFinite(value) ? Number(value).toFixed(fractionDigits) : 'na';
  const multiPassPipelineEnabled = !!(
    _supportProgram &&
    _supportSmoothProgram &&
    _drrProgram &&
    _toneProgram &&
    (
      (!GPU_TEMP_DEBUG_DISPLAY_PEAK_AMBIGUITY_MAP &&
        !GPU_TEMP_DEBUG_DISPLAY_SUPPORT_FAILURE_MAP) ||
      _supportPeakDebugProgram
    )
  );
  const pipelineMode: 'single-pass' | 'multi-pass' = multiPassPipelineEnabled ? 'multi-pass' : 'single-pass';
  const phase2GatePassed = pipelineMode === EXPECTED_GPU_PIPELINE_MODE;
  console.log(
    `[CPR-GPU-PIPELINE] mode=${pipelineMode} expected=${EXPECTED_GPU_PIPELINE_MODE} ` +
    `phase2Gate=${phase2GatePassed ? 'pass' : 'fail'} ` +
    `panoShader=${_panoProgram ? 'on' : 'off'} supportShader=${_supportProgram ? 'on' : 'off'} ` +
    `supportSmoothShader=${_supportSmoothProgram ? 'on' : 'off'} drrShader=${_drrProgram ? 'on' : 'off'} ` +
    `toneShader=${_toneProgram ? 'on' : 'off'} pano=${safePanoWidth}x${safePanoHeight} ` +
    `slabHalfMm=${formatReadableGpuValue(slabHalfThicknessMm)} slabSamples=${Math.max(1, Math.min(64, slabSamples | 0))}`
  );
  if (!phase2GatePassed) {
    console.warn(
      '[CPR-GPU] Multi-pass support-surface routing is not active. Phase 2 quality tuning should stop until the GPU pipeline is back on multi-pass.'
    );
  }

  uploadVolumeTexture(
    gl,
    scalarData,
    dimensions,
    normalizationSignature,
    normalizeStoredSample,
    volumeId
  );
  uploadSplineTexture(gl, frames, safePanoWidth, verticalDir);
  if (multiPassPipelineEnabled) {
    ensurePipelineTargets(gl, safePanoWidth, safePanoHeight);
  } else {
    ensureFbo(gl, safePanoWidth, safePanoHeight);
  }

  const w2iMat = coerceWorldToIndexMat4(worldToIndex ?? undefined);
  const requestedSlabSamples = Math.max(1, Math.min(64, slabSamples | 0));
  const commonUniforms = {
    verticalDir,
    vertHalfMm,
    verticalCenterOffsetMm,
    slabHalfThicknessMm,
    slabSamples: requestedSlabSamples,
    panoWidth: safePanoWidth,
    panoHeight: safePanoHeight,
    worldToIndexMat: w2iMat,
    dimensions,
    rescaleSlope,
    rescaleIntercept,
    applyRescale,
  };

  gl.viewport(0, 0, safePanoWidth, safePanoHeight);
  gl.disable(gl.BLEND);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_3D, _volumeTex);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, _splineTex);
  let pixelData: Float32Array;
  let meanMap: Float32Array;
  let maxMap: Float32Array;
  let sampleCountMap: Float32Array;
  let rawMin = Infinity;
  let rawMax = -Infinity;
  let rawSupportReadback: FramebufferReadback | null = null;
  let rawSupportPeakDebugReadback: FramebufferReadback | null = null;
  let supportModelDiagnosticReadback: FramebufferReadback | null = null;
  let supportReadback: FramebufferReadback | null = null;
  let drrReadback: FramebufferReadback | null = null;
  let debugMaps: GpuPanoDebugMaps | undefined;
  let diagnostics: GpuPanoDiagnostics = {
    expectedPipelineMode: EXPECTED_GPU_PIPELINE_MODE,
    phase2GatePassed,
    degradedModeReason: phase2GatePassed ? null : 'multi-pass-unavailable-single-pass-fallback',
  };

  if (multiPassPipelineEnabled) {
    if (
      !_supportFboA ||
      !_supportFboB ||
      !_supportPeakDebugFbo ||
      !_drrFbo ||
      !_fbo ||
      !_supportTexA ||
      !_supportTexB ||
      !_drrTex ||
      !_supportPeakDebugProgram ||
      !_supportModelDiagnosticProgram
    ) {
      throw new Error('[CPR-GPU] Multi-pass render targets were not initialized.');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportFboA);
    gl.useProgram(_supportProgram);
    bindCommonRayUniforms(gl, _supportProgram, commonUniforms);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportPeakDebugFbo);
    gl.useProgram(_supportPeakDebugProgram);
    bindCommonRayUniforms(gl, _supportPeakDebugProgram, commonUniforms);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportFboB);
    gl.useProgram(_supportSmoothProgram);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, _supportTexA);
    setUniform1i(gl, _supportSmoothProgram, 'uSupportData', 2);
    setUniform1i(gl, _supportSmoothProgram, 'uPanoWidth', safePanoWidth);
    setUniform1i(gl, _supportSmoothProgram, 'uPanoHeight', safePanoHeight);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _drrFbo);
    gl.useProgram(_drrProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, _volumeTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, _splineTex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, _supportTexB);
    bindCommonRayUniforms(gl, _drrProgram, commonUniforms);
    setUniform1i(gl, _drrProgram, 'uSupportData', 2);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
    gl.useProgram(_toneProgram);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, _drrTex);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, _supportTexB);
    setUniform1i(gl, _toneProgram, 'uDrrData', 2);
    setUniform1i(gl, _toneProgram, 'uSupportData', 3);
    setUniform1i(gl, _toneProgram, 'uPanoWidth', safePanoWidth);
    setUniform1i(gl, _toneProgram, 'uPanoHeight', safePanoHeight);
    setUniform1f(gl, _toneProgram, 'uSlabHalfMm', slabHalfThicknessMm);
    setUniform1f(gl, _toneProgram, 'uVertCenterOffsetMm', verticalCenterOffsetMm);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportFboA);
    rawSupportReadback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportPeakDebugFbo);
    rawSupportPeakDebugReadback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportPeakDebugFbo);
    gl.useProgram(_supportModelDiagnosticProgram);
    bindCommonRayUniforms(gl, _supportModelDiagnosticProgram, commonUniforms);
    drawFullscreenTriangle(gl);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportPeakDebugFbo);
    supportModelDiagnosticReadback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _supportFboB);
    supportReadback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _drrFbo);
    drrReadback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);

    gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
  } else {
    gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
    gl.useProgram(_panoProgram);
    bindCommonRayUniforms(gl, _panoProgram, {
      ...commonUniforms,
      debugMode: ACTIVE_GPU_DEBUG_MODE,
    });
    drawFullscreenTriangle(gl);
  }

  const readback = readFramebufferChannels(gl, safePanoWidth, safePanoHeight);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  pixelData = readback.channel0;
  if (multiPassPipelineEnabled && drrReadback) {
    inpaintLowParticipation(
      pixelData,
      drrReadback.channel3,
      safePanoWidth,
      safePanoHeight,
      1.5,
      2
    );
    inpaintLowParticipation(
      pixelData,
      drrReadback.channel3,
      safePanoWidth,
      safePanoHeight,
      1.5,
      2
    );
  }
  meanMap = readback.channel1;
  maxMap = readback.channel2;
  sampleCountMap = readback.channel3;
  rawMin = readback.rawMin;
  rawMax = readback.rawMax;
  console.log('[GPU-RAW-MINMAX]', {
    min: Number.isFinite(rawMin) ? rawMin : null,
    max: Number.isFinite(rawMax) ? rawMax : null,
  });

  if (
    multiPassPipelineEnabled &&
    rawSupportReadback &&
    rawSupportPeakDebugReadback &&
    supportModelDiagnosticReadback &&
    supportReadback &&
    drrReadback
  ) {
    const w2iMat = coerceWorldToIndexMat4(worldToIndex ?? undefined);
    const nativePitchMmByCol = computeNativeSlabPitchMmForFrames(w2iMat, frames);
    const supportSigmaMap = buildSupportSigmaMap(
      supportReadback.channel2,
      supportReadback.channel1,
      supportReadback.channel3,
      nativePitchMmByCol,
      safePanoWidth,
      safePanoHeight,
      slabHalfThicknessMm
    );
    const rawSupportContinuityMaps = buildSupportContinuityMaps(
      rawSupportReadback.channel0,
      rawSupportReadback.channel1,
      rawSupportReadback.channel3,
      rawSupportReadback.channel2,
      safePanoWidth,
      safePanoHeight
    );
    const supportContinuityMaps = buildSupportContinuityMaps(
      supportReadback.channel0,
      supportReadback.channel1,
      supportReadback.channel3,
      supportReadback.channel2,
      safePanoWidth,
      safePanoHeight
    );
    const localTransmittanceMap = buildLocalTransmittanceMap(drrReadback.channel0);
    const troughHalfWidthMap = new Float32Array(supportSigmaMap.length);
    for (let i = 0; i < supportSigmaMap.length; i++) {
      troughHalfWidthMap[i] =
        supportSigmaMap[i] * GPU_DRR_MODEL_PARAMS.approxTroughBoundarySigmaMultiplier;
    }
    const backgroundTroughNarrowGateMap = buildBackgroundTroughNarrowGateMap(
      supportReadback.channel1,
      supportReadback.channel3,
      safePanoWidth,
      safePanoHeight
    );
    const invalidSupportBlackoutMap = buildInvalidSupportBlackoutMap(
      supportReadback.channel1,
      supportReadback.channel2,
      supportReadback.channel3,
      drrReadback.channel3,
      safePanoWidth,
      safePanoHeight,
      slabHalfThicknessMm,
      verticalCenterOffsetMm
    );
    const supportFailureDisplayMap = buildSupportFailureDisplayMap(
      rawSupportPeakDebugReadback.channel3,
      rawSupportContinuityMaps.localJumpMap,
      rawSupportContinuityMaps.continuityMap,
      rawSupportReadback.channel2,
      rawSupportReadback.channel3
    );
    const backgroundLeakToneMap = buildBackgroundLeakToneMap(
      pixelData,
      sampleCountMap,
      supportReadback.channel1,
      drrReadback.channel1,
      drrReadback.channel3
    );
    const admissionAccumulationMap = drrReadback.channel0;
    const preToneAccumulationMap = meanMap;
    const toneSuppressedAccumulationMap = preToneAccumulationMap;
    const retainedSampleMaskMap = buildRetainedSampleMaskMap(
      drrReadback.channel3,
      supportReadback.channel1,
      supportReadback.channel2,
      supportReadback.channel3,
      safePanoWidth,
      safePanoHeight
    );
    const admissionMiddleBandLeakMap = buildMiddleBandLeakMap(
      admissionAccumulationMap,
      retainedSampleMaskMap,
      supportReadback.channel1,
      supportReadback.channel2,
      supportReadback.channel3,
      drrReadback.channel1,
      safePanoWidth,
      safePanoHeight
    );
    const middleBandLeakMap = buildMiddleBandLeakMap(
      preToneAccumulationMap,
      retainedSampleMaskMap,
      supportReadback.channel1,
      supportReadback.channel2,
      supportReadback.channel3,
      drrReadback.channel1,
      safePanoWidth,
      safePanoHeight
    );
    const backgroundLeakOutlier05Map = buildThresholdMaskMap(backgroundLeakToneMap, 0.05);
    const backgroundLeakOutlier10Map = buildThresholdMaskMap(backgroundLeakToneMap, 0.1);
    const toneStageSuppressionMap = buildToneStageSuppressionMap(
      admissionAccumulationMap,
      toneSuppressedAccumulationMap
    );
    const blackClipMap = buildBlackClipMap(sampleCountMap);
    const admissionOnlyHuMap = buildPositiveDisplayHuMap(admissionAccumulationMap, {
      useLogCompression: true,
    });
    const toneBypassHuMap = buildToneBypassHuMap(admissionAccumulationMap);
    const rawSupportPeakValidityMap = buildRawSupportPeakValidityMap(
      rawSupportPeakDebugReadback.channel0,
      rawSupportPeakDebugReadback.channel1,
      rawSupportPeakDebugReadback.channel3
    );
    const rawSupportPeakConflictMap = buildRawSupportPeakConflictMap(
      rawSupportPeakDebugReadback.channel0,
      rawSupportPeakDebugReadback.channel1,
      rawSupportReadback.channel2,
      rawSupportReadback.channel3,
      nativePitchMmByCol,
      safePanoWidth,
      safePanoHeight,
      slabHalfThicknessMm
    );
    const rawSupportScoreGapMap = new Float32Array(rawSupportPeakDebugReadback.channel0.length);
    for (let i = 0; i < rawSupportScoreGapMap.length; i++) {
      const peakDominance = clampNumber(Number(rawSupportPeakDebugReadback.channel0[i]) || 0, 0, 1);
      const secondPeakRatio = clampNumber(Number(rawSupportPeakDebugReadback.channel1[i]) || 0, 0, 1);
      rawSupportScoreGapMap[i] = clampNumber(peakDominance * (1 - secondPeakRatio), 0, 1);
    }
    const supportFormationMaps = buildSupportFormationDebugMaps({
      rawSupportSpreadMap: rawSupportReadback.channel2,
      rawSupportDensityMap: rawSupportReadback.channel3,
      rawSupportPeakDominanceMap: rawSupportPeakDebugReadback.channel0,
      rawSupportSecondPeakRatioMap: rawSupportPeakDebugReadback.channel1,
      rawSupportPeakSeparationMap: rawSupportPeakDebugReadback.channel2,
      rawSupportPeakAmbiguityMap: rawSupportPeakDebugReadback.channel3,
      rawSupportDenseFractionMap: supportModelDiagnosticReadback.channel0,
      rawSupportPeakHuSupportGateMap: supportModelDiagnosticReadback.channel1,
      supportSpreadMap: rawSupportReadback.channel2,
      supportConfidenceMap: rawSupportReadback.channel1,
      nativePitchMmByCol,
      width: safePanoWidth,
      height: safePanoHeight,
      slabHalfThicknessMm,
    });
    const smoothedSupportFormationMaps = buildSupportFormationDebugMaps({
      rawSupportSpreadMap: rawSupportReadback.channel2,
      rawSupportDensityMap: rawSupportReadback.channel3,
      rawSupportPeakDominanceMap: rawSupportPeakDebugReadback.channel0,
      rawSupportSecondPeakRatioMap: rawSupportPeakDebugReadback.channel1,
      rawSupportPeakSeparationMap: rawSupportPeakDebugReadback.channel2,
      rawSupportPeakAmbiguityMap: rawSupportPeakDebugReadback.channel3,
      rawSupportDenseFractionMap: supportModelDiagnosticReadback.channel0,
      rawSupportPeakHuSupportGateMap: supportModelDiagnosticReadback.channel1,
      supportSpreadMap: supportReadback.channel2,
      supportConfidenceMap: supportReadback.channel1,
      nativePitchMmByCol,
      width: safePanoWidth,
      height: safePanoHeight,
      slabHalfThicknessMm,
    });
    const drrSupportGuidanceMaps = buildDrrSupportGuidanceMaps(
      supportSigmaMap,
      backgroundTroughNarrowGateMap,
      supportReadback.channel2,
      supportReadback.channel1,
      supportReadback.channel3,
      nativePitchMmByCol,
      safePanoWidth,
      safePanoHeight,
      slabHalfThicknessMm
    );

    diagnostics = {
      ...diagnostics,
      rawSupportPeaks: buildSupportPeakDiagnostics(
        rawSupportPeakDebugReadback.channel0,
        rawSupportPeakValidityMap,
        rawSupportPeakConflictMap,
        rawSupportPeakDebugReadback.channel1,
        rawSupportPeakDebugReadback.channel2,
        rawSupportPeakDebugReadback.channel3
      ),
      rawSupportSurface: buildSupportSurfaceDiagnostics(
        rawSupportReadback.channel0,
        rawSupportReadback.channel1,
        rawSupportReadback.channel2,
        rawSupportReadback.channel3,
        safePanoWidth,
        safePanoHeight,
        slabHalfThicknessMm,
        rawSupportContinuityMaps.localJumpMap,
        rawSupportContinuityMaps.continuityMap,
        rawSupportPeakDebugReadback.channel3,
        rawSupportScoreGapMap
      ),
      supportSurface: buildSupportSurfaceDiagnostics(
        supportReadback.channel0,
        supportReadback.channel1,
        supportReadback.channel2,
        supportReadback.channel3,
        safePanoWidth,
        safePanoHeight,
        slabHalfThicknessMm,
        supportContinuityMaps.localJumpMap,
        supportContinuityMaps.continuityMap,
        supportFormationMaps.protectedAmbiguousBroadSupportPenaltyGateMap,
        rawSupportScoreGapMap,
        rawSupportReadback.channel0
      ),
      rawSupportFormation: buildSupportFormationDiagnostics({
        supportConfidenceMap: rawSupportReadback.channel1,
        supportSpreadMap: rawSupportReadback.channel2,
        supportDensityMap: rawSupportReadback.channel3,
        rawSupportScoreGapMap,
        dominantDensePeakGateMap: supportFormationMaps.dominantDensePeakGateMap,
        toothBandStructureGuardMap: supportFormationMaps.toothBandStructureGuardMap,
        protectedAmbiguousBroadSupportPenaltyGateMap:
          supportFormationMaps.protectedAmbiguousBroadSupportPenaltyGateMap,
        supportValidityMap: supportFormationMaps.supportValidityMap,
        falseSupportVetoMap: supportFormationMaps.falseSupportVetoMap,
        rowBackgroundVetoMap: supportFormationMaps.rowBackgroundVetoMap,
        supportVetoTriggeredMap: supportFormationMaps.supportVetoTriggeredMap,
        rawSupportDepthMap: rawSupportReadback.channel0,
        supportDepthMap: rawSupportReadback.channel0,
        width: safePanoWidth,
        height: safePanoHeight,
      }),
      supportFormation: buildSupportFormationDiagnostics({
        supportConfidenceMap: supportReadback.channel1,
        supportSpreadMap: supportReadback.channel2,
        supportDensityMap: supportReadback.channel3,
        rawSupportScoreGapMap,
        dominantDensePeakGateMap: smoothedSupportFormationMaps.dominantDensePeakGateMap,
        toothBandStructureGuardMap: smoothedSupportFormationMaps.toothBandStructureGuardMap,
        protectedAmbiguousBroadSupportPenaltyGateMap:
          smoothedSupportFormationMaps.protectedAmbiguousBroadSupportPenaltyGateMap,
        supportValidityMap: smoothedSupportFormationMaps.supportValidityMap,
        falseSupportVetoMap: smoothedSupportFormationMaps.falseSupportVetoMap,
        rowBackgroundVetoMap: smoothedSupportFormationMaps.rowBackgroundVetoMap,
        supportVetoTriggeredMap: smoothedSupportFormationMaps.supportVetoTriggeredMap,
        rawSupportDepthMap: rawSupportReadback.channel0,
        supportDepthMap: supportReadback.channel0,
        width: safePanoWidth,
        height: safePanoHeight,
      }),
      drr: buildDrrDiagnostics(
        drrReadback.channel0,
        drrReadback.channel2,
        drrReadback.channel1,
        drrReadback.channel3,
        preToneAccumulationMap,
        retainedSampleMaskMap,
        middleBandLeakMap,
        supportSigmaMap,
        drrSupportGuidanceMaps.effectiveTroughHalfWidthMap,
        backgroundTroughNarrowGateMap,
        localTransmittanceMap,
        nativePitchMmByCol,
        safePanoWidth,
        safePanoHeight,
        slabHalfThicknessMm,
        requestedSlabSamples
      ),
      toneMap: buildToneMapDiagnostics(
        meanMap,
        drrReadback.channel2,
        drrReadback.channel1,
        invalidSupportBlackoutMap,
        supportFailureDisplayMap,
        sampleCountMap,
        pixelData
      ),
    };

    if (returnDebugSidecars) {
      debugMaps = {
        rawSupportDepthMap: rawSupportReadback.channel0,
        rawSupportConfidenceMap: rawSupportReadback.channel1,
        rawSupportSpreadMap: rawSupportReadback.channel2,
        rawSupportDensityMap: rawSupportReadback.channel3,
        rawSupportDenseFractionMap: supportModelDiagnosticReadback.channel0,
        rawSupportPeakHuSupportGateMap: supportModelDiagnosticReadback.channel1,
        rawSupportDominantPeakOffsetMap: supportModelDiagnosticReadback.channel2,
        rawSupportSecondaryPeakOffsetMap: supportModelDiagnosticReadback.channel3,
        rawSupportPeakDominanceMap: rawSupportPeakDebugReadback.channel0,
        rawSupportPeakValidityMap,
        rawSupportPeakConflictMap,
        rawSupportSecondPeakRatioMap: rawSupportPeakDebugReadback.channel1,
        rawSupportPeakSeparationMap: rawSupportPeakDebugReadback.channel2,
        rawSupportPeakAmbiguityMap: rawSupportPeakDebugReadback.channel3,
        rawSupportScoreGapMap,
        rawSupportLocalJumpMap: rawSupportContinuityMaps.localJumpMap,
        rawSupportContinuityMap: rawSupportContinuityMaps.continuityMap,
        toothBandPriorMap: supportFormationMaps.toothBandPriorMap,
        dominantDensePeakGateMap: smoothedSupportFormationMaps.dominantDensePeakGateMap,
        toothBandStructureGuardMap: smoothedSupportFormationMaps.toothBandStructureGuardMap,
        ambiguousBroadSupportPenaltyGateMap:
          smoothedSupportFormationMaps.ambiguousBroadSupportPenaltyGateMap,
        protectedAmbiguousBroadSupportPenaltyGateMap:
          smoothedSupportFormationMaps.protectedAmbiguousBroadSupportPenaltyGateMap,
        structuralSupportGateMap: smoothedSupportFormationMaps.structuralSupportGateMap,
        peakStructureValidityMap: smoothedSupportFormationMaps.peakStructureValidityMap,
        supportValidityMap: smoothedSupportFormationMaps.supportValidityMap,
        rowConfidenceGateMap: smoothedSupportFormationMaps.rowConfidenceGateMap,
        falseSupportConfidenceGateMap:
          smoothedSupportFormationMaps.falseSupportConfidenceGateMap,
        falseSupportDensityGateMap: smoothedSupportFormationMaps.falseSupportDensityGateMap,
        falseSupportSpreadGateMap: smoothedSupportFormationMaps.falseSupportSpreadGateMap,
        falseSupportVetoMap: smoothedSupportFormationMaps.falseSupportVetoMap,
        rowBackgroundDensityGateMap:
          smoothedSupportFormationMaps.rowBackgroundDensityGateMap,
        rowBackgroundSpreadGateMap: smoothedSupportFormationMaps.rowBackgroundSpreadGateMap,
        rowBackgroundPeakHuGateMap: smoothedSupportFormationMaps.rowBackgroundPeakHuGateMap,
        rowBackgroundEdgeGateMap: smoothedSupportFormationMaps.rowBackgroundEdgeGateMap,
        rowBackgroundVetoMap: smoothedSupportFormationMaps.rowBackgroundVetoMap,
        supportVetoTriggeredMap: smoothedSupportFormationMaps.supportVetoTriggeredMap,
        supportFailureDisplayMap,
        supportDepthMap: supportReadback.channel0,
        supportConfidenceMap: supportReadback.channel1,
        supportSpreadMap: supportReadback.channel2,
        supportDensityMap: supportReadback.channel3,
        supportLocalJumpMap: supportContinuityMaps.localJumpMap,
        supportContinuityMap: supportContinuityMaps.continuityMap,
        totalAttenuationMap: drrReadback.channel0,
        admissionAccumulationMap,
        toneSuppressedAccumulationMap,
        fogAttenuationMap: drrReadback.channel2,
        lowerPenaltyMap: drrReadback.channel1,
        participatingSampleCountMap: drrReadback.channel3,
        toneResponseMap: sampleCountMap,
        preToneAccumulationMap,
        retainedSampleMaskMap,
        middleBandLeakMap,
        admissionMiddleBandLeakMap,
        invalidSupportBlackoutMap,
        toneStageSuppressionMap,
        blackClipMap,
        backgroundLeakToneMap,
        backgroundLeakOutlier05Map,
        backgroundLeakOutlier10Map,
        admissionOnlyHuMap,
        toneBypassHuMap,
        troughHalfWidthMap,
        effectiveTroughHalfWidthMap: drrSupportGuidanceMaps.effectiveTroughHalfWidthMap,
        continuityExpandedTroughHalfWidthMap:
          drrSupportGuidanceMaps.continuityExpandedTroughHalfWidthMap,
        backgroundTroughNarrowGateMap,
        dominantToothBandGateMap: drrSupportGuidanceMaps.dominantToothBandGateMap,
        broadWeakToothBandGateMap: drrSupportGuidanceMaps.broadWeakToothBandGateMap,
        toothContinuityAdmissionGateMap:
          drrSupportGuidanceMaps.toothContinuityAdmissionGateMap,
      };
    }

    if (GPU_TEMP_DEBUG_DISPLAY_SUPPORT_FAILURE_MAP) {
      pixelData = buildDebugDisplayHuFromUnitMap(supportFailureDisplayMap);
    } else if (GPU_TEMP_DEBUG_DISPLAY_PEAK_AMBIGUITY_MAP) {
      pixelData = buildDebugDisplayHuFromUnitMap(rawSupportPeakDebugReadback.channel3);
    }
  }

  diagnostics.sidecarMaps = {
    debugEnabled: returnDebugSidecars,
    attachedMaps: debugMaps
      ? Object.entries(debugMaps)
        .filter(([, value]) => !!value)
        .map(([name]) => name)
      : [],
    attachedByteLength: debugMaps
      ? sumByteLength([
        debugMaps.rawSupportDepthMap,
        debugMaps.rawSupportConfidenceMap,
        debugMaps.rawSupportSpreadMap,
        debugMaps.rawSupportDensityMap,
        debugMaps.rawSupportDenseFractionMap,
        debugMaps.rawSupportPeakHuSupportGateMap,
        debugMaps.rawSupportDominantPeakOffsetMap,
        debugMaps.rawSupportSecondaryPeakOffsetMap,
        debugMaps.rawSupportPeakDominanceMap,
        debugMaps.rawSupportPeakValidityMap,
        debugMaps.rawSupportPeakConflictMap,
        debugMaps.rawSupportSecondPeakRatioMap,
        debugMaps.rawSupportPeakSeparationMap,
        debugMaps.rawSupportPeakAmbiguityMap,
        debugMaps.rawSupportScoreGapMap,
        debugMaps.rawSupportLocalJumpMap,
        debugMaps.rawSupportContinuityMap,
        debugMaps.toothBandPriorMap,
        debugMaps.dominantDensePeakGateMap,
        debugMaps.toothBandStructureGuardMap,
        debugMaps.ambiguousBroadSupportPenaltyGateMap,
        debugMaps.protectedAmbiguousBroadSupportPenaltyGateMap,
        debugMaps.structuralSupportGateMap,
        debugMaps.peakStructureValidityMap,
        debugMaps.supportValidityMap,
        debugMaps.rowConfidenceGateMap,
        debugMaps.falseSupportConfidenceGateMap,
        debugMaps.falseSupportDensityGateMap,
        debugMaps.falseSupportSpreadGateMap,
        debugMaps.falseSupportVetoMap,
        debugMaps.rowBackgroundDensityGateMap,
        debugMaps.rowBackgroundSpreadGateMap,
        debugMaps.rowBackgroundPeakHuGateMap,
        debugMaps.rowBackgroundEdgeGateMap,
        debugMaps.rowBackgroundVetoMap,
        debugMaps.supportVetoTriggeredMap,
        debugMaps.supportFailureDisplayMap,
        debugMaps.supportDepthMap,
        debugMaps.supportConfidenceMap,
        debugMaps.supportSpreadMap,
        debugMaps.supportDensityMap,
        debugMaps.supportLocalJumpMap,
        debugMaps.supportContinuityMap,
        debugMaps.totalAttenuationMap,
        debugMaps.admissionAccumulationMap,
        debugMaps.toneSuppressedAccumulationMap,
        debugMaps.fogAttenuationMap,
        debugMaps.lowerPenaltyMap,
        debugMaps.participatingSampleCountMap,
        debugMaps.toneResponseMap,
        debugMaps.preToneAccumulationMap,
        debugMaps.retainedSampleMaskMap,
        debugMaps.middleBandLeakMap,
        debugMaps.admissionMiddleBandLeakMap,
        debugMaps.invalidSupportBlackoutMap,
        debugMaps.toneStageSuppressionMap,
        debugMaps.blackClipMap,
        debugMaps.backgroundLeakToneMap,
        debugMaps.backgroundLeakOutlier05Map,
        debugMaps.backgroundLeakOutlier10Map,
        debugMaps.admissionOnlyHuMap,
        debugMaps.toneBypassHuMap,
        debugMaps.troughHalfWidthMap,
        debugMaps.effectiveTroughHalfWidthMap,
        debugMaps.continuityExpandedTroughHalfWidthMap,
        debugMaps.backgroundTroughNarrowGateMap,
        debugMaps.dominantToothBandGateMap,
        debugMaps.broadWeakToothBandGateMap,
        debugMaps.toothContinuityAdmissionGateMap,
      ])
      : 0,
  };

  const { minValue, maxValue } = computeMinMax(pixelData);
  const elapsed = performance.now() - t0;
  console.log(
    `[CPR-GPU-RESULT] mode=${pipelineMode} ` +
    `rawMin=${formatReadableGpuValue(rawMin)} rawMax=${formatReadableGpuValue(rawMax)} ` +
    `finalMin=${formatReadableGpuValue(minValue)} finalMax=${formatReadableGpuValue(maxValue)} ` +
    `elapsedMs=${formatReadableGpuValue(elapsed, 0)} ` +
    `reduction=${multiPassPipelineEnabled ? 'support-surface-drr-tone' : 'gaussian-focal-trough-weighted-mean'} ` +
    `phase2Gate=${phase2GatePassed ? 'pass' : 'fail'}`
  );
  console.log(`[CPR-GPU] ${multiPassPipelineEnabled ? 'Multi-pass support-surface panoramic projection' : 'Single-pass continuous-geometry hybrid projection'} complete: ${safePanoWidth}x${safePanoHeight} in ${elapsed.toFixed(1)}ms`, {
    minValue,
    maxValue,
    slabHalfThicknessMm,
    slabSamples: requestedSlabSamples,
    reduction: multiPassPipelineEnabled
      ? 'support estimation + support smoothing + DRR attenuation + tone mapping'
      : 'gaussian focal-trough weighted accumulation (sigma 1.5 mm)',
    debugMode: ACTIVE_GPU_DEBUG_MODE,
    phase2GatePassed,
    sidecarMaps: diagnostics.sidecarMaps?.attachedMaps ?? [],
  });

  return {
    pixelData,
    meanMap,
    maxMap,
    sampleCountMap,
    width: safePanoWidth,
    height: safePanoHeight,
    minValue,
    maxValue,
    pipelineMode,
    debugMaps,
    diagnostics,
  };
}

// ─── GPU Availability Check ─────────────────────────────────────────
export function isGpuPanoAvailable(): boolean {
  try {
    const testCanvas = new OffscreenCanvas(1, 1);
    const gl = testCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
    if (!gl) return false;
    const ext = gl.getExtension('EXT_color_buffer_float');
    const floatLinear = gl.getExtension('OES_texture_float_linear');
    return !!(ext && floatLinear);
  } catch {
    try {
      if (typeof document === 'undefined' || !document.createElement) {
        return false;
      }
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2') as WebGL2RenderingContext | null;
      if (!gl) return false;
      const ext = gl.getExtension('EXT_color_buffer_float');
      const floatLinear = gl.getExtension('OES_texture_float_linear');
      return !!(ext && floatLinear);
    } catch {
      return false;
    }
  }
}

// ─── Cleanup ────────────────────────────────────────────────────────
export function disposeGpuPanoRenderer(): void {
  resetGpuRendererState();
}


