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
    supportDepthMap?: Float32Array;
    supportConfidenceMap?: Float32Array;
    supportSpreadMap?: Float32Array;
    supportDensityMap?: Float32Array;
    totalAttenuationMap?: Float32Array;
    fogAttenuationMap?: Float32Array;
    lowerPenaltyMap?: Float32Array;
    participatingSampleCountMap?: Float32Array;
    toneResponseMap?: Float32Array;
    troughHalfWidthMap?: Float32Array;
}

export interface GpuSupportSurfaceDiagnostics {
    depthMinMm: number;
    depthMaxMm: number;
    depthStdMm: number;
    pathJumpP95Mm: number;
    clampFraction: number;
    confidenceP10: number;
    confidenceP50: number;
    confidenceP90: number;
    spreadP50Mm: number;
    spreadP90Mm: number;
    densityP50: number;
    densityP90: number;
    selectedDepthFirst8Mm: number[];
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
    drrModel: {
        supportWeightPowerLowConfidence: number;
        supportWeightPowerHighConfidence: number;
        lowerPenaltyDenseScale: number;
        lowerPenaltyRowStart: number;
        lowerPenaltyRowEnd: number;
        troughSigmaHardCapMm: number;
    };
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
    };
}

export interface GpuPanoDiagnostics {
    expectedPipelineMode: 'multi-pass';
    phase2GatePassed: boolean;
    degradedModeReason?: string | null;
    supportSurface?: GpuSupportSurfaceDiagnostics;
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
const SUPPORT_PATH_ROW_START_FRACTION = 0.14;
const SUPPORT_PATH_ROW_END_FRACTION = 0.76;
const TONE_RESPONSE_BLACK_CLIP_THRESHOLD = 0.02;
const TONE_RESPONSE_WHITE_CLIP_THRESHOLD = 0.98;
const GPU_SUPPORT_MODEL_PARAMS = {
    toothBandCenter: 0.44,
    toothBandInnerHalfWidth: 0.16,
    toothBandOuterHalfWidth: 0.38,
    inferiorPenaltyStart: 0.62,
    inferiorPenaltyEnd: 0.96,
    superiorPenaltyStart: 0.02,
    superiorPenaltyEnd: 0.16,
    densityConfidenceLow: 0.04,
    densityConfidenceHigh: 0.18,
    peakHuConfidenceLow: 700,
    peakHuConfidenceHigh: 1700,
    enamelBandLowHu: 350,
    enamelBandPeakHu: 550,
    enamelBandHighHu: 1850,
    enamelBandTailHu: 2500,
    inferiorOffsetBoostStartMm: -0.38,
    inferiorOffsetBoostEndMm: -0.04,
    inferiorOffsetBoostScale: 1.22,
    superiorOffsetPenaltyStartMm: 0.02,
    superiorOffsetPenaltyEndMm: 0.26,
    superiorOffsetPenaltyScale: 0.52,
    positiveDepthClampStartMm: 0.02,
    positiveDepthClampEndMm: 0.22,
    positiveDepthClampTargetMm: -0.12,
    positiveDepthClampConfidenceLow: 0.10,
    positiveDepthClampConfidenceHigh: 0.30,
    positiveDepthClampDensityLow: 0.05,
    positiveDepthClampDensityHigh: 0.16,
};
const GPU_DRR_MODEL_PARAMS = {
    broadSigmaSlabScale: 0.34,
    broadSigmaNativePitchScale: 0.9,
    focusedSpreadScale: 0.72,
    focusedNativePitchBias: 0.22,
    focusedSigmaMinNativePitchScale: 0.72,
    focusedSigmaMaxSlabScale: 0.32,
    focusedSigmaMaxNativePitchScale: 0.92,
    confidenceBlendLow: 0.14,
    confidenceBlendHigh: 0.62,
    confidenceSigmaExpansion: 1.12,
    attenuationConfidenceLow: 0.12,
    attenuationConfidenceHigh: 0.58,
    supportWeightPowerLowConfidence: 1.55,
    supportWeightPowerHighConfidence: 1.08,
    lowerPenaltyDenseScale: 0.06,
    lowerPenaltyConfidenceLow: 0.1,
    lowerPenaltyConfidenceHigh: 0.55,
    lowerPenaltyRowStart: 0.62,
    lowerPenaltyRowEnd: 0.96,
    approxTroughBoundarySigmaMultiplier: 1.7,
    troughSigmaHardCapMm: 0.3,
};
const GPU_TONE_MODEL_PARAMS = {
    lowConfidenceFogSuppression: 0.12,
    highConfidenceFogSuppression: 0.05,
    lowConfidenceFogRetention: 0.82,
    highConfidenceFogRetention: 0.92,
    lowConfidencePenaltySuppression: 0.38,
    highConfidencePenaltySuppression: 0.20,
    lowConfidencePenaltyRetention: 0.72,
    highConfidencePenaltyRetention: 0.86,
    attenuationConfidenceLow: 0.12,
    attenuationConfidenceHigh: 0.58,
    lowConfidenceAttenuationScale: 1.8,
    exposureScale: 3.0,
    sigmoidMidpoint: 0.085,
    sigmoidSlope: 3.8,
    sigmoidWhitePoint: 1.2,
    postCurveGamma: 1.0,
    outputHuMin: -760,
    outputHuMax: 1650,
};

// ─── Shaders ─────────────────────────────────────────────────────────
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
  float softTissue = 0.0065 * smoothstep(-950.0, -180.0, hu);
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

float supportResponseFromHu(float hu) {
  float rootSupport = smoothstep(180.0, 900.0, hu);
  float dentinSupport = smoothstep(500.0, 1400.0, hu);
  float enamelBandSupport = enamelBandSupportFromHu(hu);
  float metalSuppressSupport = 1.0 - smoothstep(1100.0, 1900.0, hu);
  float enamelSupport = smoothstep(950.0, 1800.0, hu) * metalSuppressSupport;
  float denseBias = smoothstep(1100.0, 1800.0, hu) * metalSuppressSupport;
  float combined =
    0.24 * rootSupport +
    0.72 * dentinSupport +
    1.18 * enamelBandSupport +
    0.52 * enamelSupport;
  return clamp(combined * mix(0.96, 1.18, denseBias), 0.0, 2.0);
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
    mix(0.70, 1.18, toothBandPrior) *
    mix(1.0, 0.55, inferiorPenalty) *
    mix(0.84, 1.0, 1.0 - superiorPenalty),
    0.30,
    1.24
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
    float enamelBandSupport = enamelBandSupportFromHu(hu);
    float metalSuppressLoop =
      1.0 - smoothstep(1100.0, 1900.0, hu);
    float denseBias = smoothstep(500.0, 1800.0, hu) * metalSuppressLoop;
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
      supportResponse *
      mix(0.82, 1.42, denseBias) *
      mix(1.0, 1.55, enamelBandSupport) *
      rowPrior *
      offsetPrior;
    supportMass += weightedSupport;
    supportOffsetSum += slabOffset * weightedSupport;
    supportOffsetSqSum += slabOffset * slabOffset * weightedSupport;
    denseMass += weightedSupport * mix(0.32, 1.0, max(denseBias, enamelBandSupport));
    float candidateScore =
      weightedSupport *
      mix(0.78, 1.45, denseBias) *
      mix(1.0, 1.65, enamelBandSupport) *
      mix(1.0, 0.62, superiorOffsetGate);
    candidateScore *= (1.0 + edgeBonus * 0.7) * (1.0 - gradientPenalty * 0.5);
    if (candidateScore > bestSupportScore) {
      bestSupportScore = candidateScore;
      bestSupportOffsetMm = slabOffset;
    }
    peakHu = max(peakHu, hu);
    prevSampleHu = hu;
    hasPrevSampleHu = true;
    hasValidSample = true;
  }

  float defaultSpreadMm = max(uSlabHalfMm * 0.75, nativeSlabPitchMm);
  if (!hasValidSample || supportMass <= 1e-5) {
    fragColor = vec4(0.0, 0.0, defaultSpreadMm, 0.0);
    return;
  }

  float supportCenterMm = supportOffsetSum / supportMass;
  float secondMoment = supportOffsetSqSum / supportMass;
  float varianceMm = max(secondMoment - supportCenterMm * supportCenterMm, nativeSlabPitchMm * nativeSlabPitchMm * 0.25);
  float supportSpreadMm = sqrt(varianceMm);
  float peakDominance = bestSupportScore / max(supportMass, 1e-5);
  supportCenterMm = mix(
    supportCenterMm,
    bestSupportOffsetMm,
    smoothstep(0.16, 0.58, peakDominance)
  );
  float supportDensity = denseMass / max(float(raySampleCount), 1.0);
  float spreadConfidence =
    1.0 - smoothstep(
      nativeSlabPitchMm * 1.5,
      max(uSlabHalfMm * 0.55, nativeSlabPitchMm * 2.8),
      supportSpreadMm
    );
  float rowConfidenceGate = clamp(
    mix(0.72, 1.14, toothBandPrior) *
    mix(1.0, 0.50, inferiorPenalty) *
    mix(0.88, 1.0, 1.0 - superiorPenalty),
    0.26,
    1.18
  );
  float supportConfidence =
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.densityConfidenceLow.toFixed(3)},
      ${GPU_SUPPORT_MODEL_PARAMS.densityConfidenceHigh.toFixed(3)},
      supportDensity
    ) *
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceLow.toFixed(1)},
      ${GPU_SUPPORT_MODEL_PARAMS.peakHuConfidenceHigh.toFixed(1)},
      peakHu
    ) *
    clamp(spreadConfidence, 0.0, 1.0) *
    rowConfidenceGate;

  fragColor = vec4(
    supportCenterMm,
    clamp(supportConfidence, 0.0, 1.0),
    clamp(supportSpreadMm, nativeSlabPitchMm * 0.75, max(uSlabHalfMm, nativeSlabPitchMm)),
    clamp(supportDensity, 0.0, 1.0)
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

  float centerWeight = 2.2 + centerConfidence * 3.2;
  float weightedCenter = centerSample.r * centerWeight;
  float weightedConfidence = centerConfidence * centerWeight;
  float weightedSpread = centerSample.b * centerWeight;
  float weightedDensity = centerSample.a * centerWeight;
  float weightSum = centerWeight;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      ivec2 sampleCoord = ivec2(
        clamp(col + dx, 0, max(uPanoWidth - 1, 0)),
        clamp(outputRow + dy, 0, max(uPanoHeight - 1, 0))
      );
      vec4 sampleValue = texelFetch(uSupportData, sampleCoord, 0);
      float confidence = clamp(sampleValue.g, 0.0, 1.0);
      float spatialWeight =
        exp(-0.5 * (float(dx * dx) / 1.35 + float(dy * dy) / 0.80));
      float depthDelta = sampleValue.r - centerDepthMm;
      float depthWeight = exp(-(depthDelta * depthDelta) / (2.0 * 0.55 * 0.55));
      float confidenceDelta = confidence - centerConfidence;
      float confidenceWeight =
        exp(-(confidenceDelta * confidenceDelta) / (2.0 * 0.18 * 0.18));
      float sampleWeight =
        spatialWeight *
        depthWeight *
        confidenceWeight *
        max(sqrt(max(confidence * centerConfidence, 0.0)), 0.02);

      weightedCenter += sampleValue.r * sampleWeight;
      weightedConfidence += confidence * sampleWeight;
      weightedSpread += sampleValue.b * sampleWeight;
      weightedDensity += sampleValue.a * sampleWeight;
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
  float positiveDepthGate = smoothstep(
    ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampStartMm.toFixed(2)},
    ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampEndMm.toFixed(2)},
    smoothedCenterMm
  );
  float lowConfidenceGate =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampConfidenceLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampConfidenceHigh.toFixed(2)},
      smoothedConfidence
    );
  float lowDensityGate =
    1.0 -
    smoothstep(
      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampDensityLow.toFixed(2)},
      ${GPU_SUPPORT_MODEL_PARAMS.positiveDepthClampDensityHigh.toFixed(2)},
      smoothedDensity
    );
  float positiveDepthCorrection =
    positiveDepthGate * clamp(max(lowConfidenceGate, lowDensityGate * 0.9), 0.0, 1.0);
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
  float supportSpreadMm = max(supportData.b, 0.1);
  float displayRowNorm =
    float((uPanoHeight - 1) - outputRow) / max(float(uPanoHeight - 1), 1.0);

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
  float supportDenom = 2.0 * supportSigmaMm * supportSigmaMm;
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
    ${GPU_DRR_MODEL_PARAMS.supportWeightPowerLowConfidence.toFixed(3)},
    ${GPU_DRR_MODEL_PARAMS.supportWeightPowerHighConfidence.toFixed(3)},
    confidenceGate
  );
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

    float hu = sampleHu(uvw);
    float supportDistanceMm = slabOffset - supportCenterMm;
    float supportWeight = exp(-(supportDistanceMm * supportDistanceMm) / supportDenom);
    supportWeight = pow(clamp(supportWeight, 0.0, 1.0), supportWeightPower);
    float mu = pseudoAttenuationFromHu(hu);
    float muFog = softFogAttenuationFromHu(hu);
    float segmentLength = max(slabStep, nativeSlabPitchMm);
    float lowerPenalty = max(mu - muFog, 0.0) *
      ${GPU_DRR_MODEL_PARAMS.lowerPenaltyDenseScale.toFixed(3)} *
      lowerPenaltyRowGate *
      lowerPenaltyConfidenceGate;

    totalAttenuation += mu * supportWeight * segmentLength * mix(0.35, 1.0, confidenceGate);
    fogAttenuation += muFog * supportWeight * segmentLength * mix(0.14, 0.32, supportConfidence);
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
    1.0 - smoothstep(0.0004, 0.0016, supportConfidence);
  float lowDensityGate =
    1.0 - smoothstep(0.06, 0.18, supportDensity);
  float lowPenaltyGate =
    1.0 - smoothstep(0.010, 0.050, lowerPenalty);
  float stableParticipationGate =
    smoothstep(1.8, 2.8, participatingSamples);
  float backgroundSuppressionGate =
    backgroundSuppressionConfidenceGate *
    lowDensityGate *
    lowPenaltyGate *
    stableParticipationGate;
  // Keep the commit's tooth attenuation behavior and only suppress
  // the extremely low-confidence background cluster.
  gentlySuppressedAttenuation *= mix(1.0, 0.43, backgroundSuppressionGate);
  gentlySuppressedAttenuation *= mix(1.0, 0.96, inferiorPenalty * (1.0 - supportConfidence));

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
let _drrTex: WebGLTexture | null = null;
let _supportFboA: WebGLFramebuffer | null = null;
let _supportFboB: WebGLFramebuffer | null = null;
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
    if (_drrTex) gl.deleteTexture(_drrTex);
    if (_fboTex) gl.deleteTexture(_fboTex);
    if (_supportFboA) gl.deleteFramebuffer(_supportFboA);
    if (_supportFboB) gl.deleteFramebuffer(_supportFboB);
    if (_drrFbo) gl.deleteFramebuffer(_drrFbo);
    if (_fbo) gl.deleteFramebuffer(_fbo);

    _supportTexA = null;
    _supportTexB = null;
    _drrTex = null;
    _fboTex = null;
    _supportFboA = null;
    _supportFboB = null;
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
        _drrTex &&
        _fboTex &&
        _supportFboA &&
        _supportFboB &&
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
    const drr = createFloatRenderTarget(gl, w, h);
    const finalTarget = createFloatRenderTarget(gl, w, h);

    _supportTexA = supportA.texture;
    _supportFboA = supportA.fbo;
    _supportTexB = supportB.texture;
    _supportFboB = supportB.fbo;
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
            const nativePitchMm = Number(nativePitchMmByCol[col]) || 0.2;
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
            supportSigmaMap[index] = Math.min(
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

function buildSupportSurfaceDiagnostics(
    supportDepthMap: Float32Array,
    supportConfidenceMap: Float32Array,
    supportSpreadMap: Float32Array,
    supportDensityMap: Float32Array,
    width: number,
    height: number,
    slabHalfThicknessMm: number
): GpuSupportSurfaceDiagnostics {
    const collapsedPath = collapseSupportPathByColumn(supportDepthMap, supportConfidenceMap, width, height);
    const pathDepthSummary = summarizeFiniteBuffer(collapsedPath.pathDepthMm);
    const confidenceSummary = summarizeFiniteBuffer(supportConfidenceMap);
    const spreadSummary = summarizeFiniteBuffer(supportSpreadMap);
    const densitySummary = summarizeFiniteBuffer(supportDensityMap);

    return {
        depthMinMm: roundFinite(pathDepthSummary?.min ?? 0),
        depthMaxMm: roundFinite(pathDepthSummary?.max ?? 0),
        depthStdMm: roundFinite(pathDepthSummary?.stdDev ?? 0),
        pathJumpP95Mm: roundFinite(computeAdjacentDeltaP95(collapsedPath.pathDepthMm)),
        clampFraction: roundFinite(computeClampFraction(collapsedPath.pathDepthMm, slabHalfThicknessMm), 4),
        confidenceP10: roundFinite(confidenceSummary?.p10 ?? 0),
        confidenceP50: roundFinite(confidenceSummary?.p50 ?? 0),
        confidenceP90: roundFinite(confidenceSummary?.p90 ?? 0),
        spreadP50Mm: roundFinite(spreadSummary?.p50 ?? 0),
        spreadP90Mm: roundFinite(spreadSummary?.p90 ?? 0),
        densityP50: roundFinite(densitySummary?.p50 ?? 0),
        densityP90: roundFinite(densitySummary?.p90 ?? 0),
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
    supportSigmaMap: Float32Array,
    localTransmittanceMap: Float32Array,
    nativePitchMmByCol: Float32Array,
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
    const totalSummary = summarizeFiniteBuffer(totalAttenuationMap);
    const fogSummary = summarizeFiniteBuffer(fogAttenuationMap);
    const lowerPenaltySummary = summarizeFiniteBuffer(lowerPenaltyMap);
    const participatingSummary = summarizeFiniteBuffer(participatingSampleCountMap);
    const transmittanceSummary = summarizeFiniteBuffer(localTransmittanceMap);

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
        drrModel: {
            supportWeightPowerLowConfidence: GPU_DRR_MODEL_PARAMS.supportWeightPowerLowConfidence,
            supportWeightPowerHighConfidence: GPU_DRR_MODEL_PARAMS.supportWeightPowerHighConfidence,
            lowerPenaltyDenseScale: GPU_DRR_MODEL_PARAMS.lowerPenaltyDenseScale,
            lowerPenaltyRowStart: GPU_DRR_MODEL_PARAMS.lowerPenaltyRowStart,
            lowerPenaltyRowEnd: GPU_DRR_MODEL_PARAMS.lowerPenaltyRowEnd,
            troughSigmaHardCapMm: GPU_DRR_MODEL_PARAMS.troughSigmaHardCapMm,
        },
    };
}

function buildToneMapDiagnostics(
    inputAttenuationMap: Float32Array,
    fogAttenuationMap: Float32Array,
    lowerPenaltyMap: Float32Array,
    toneResponseMap: Float32Array,
    outputHuMap: Float32Array
): GpuToneMapDiagnostics {
    const inputSummary = summarizeFiniteBuffer(inputAttenuationMap);
    const fogSummary = summarizeFiniteBuffer(fogAttenuationMap);
    const penaltySummary = summarizeFiniteBuffer(lowerPenaltyMap);
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
    _drrTex = null;
    _supportFboA = null;
    _supportFboB = null;
    _drrFbo = null;
    _fboTex = null;
    _fbo = null;
    _panoProgram = null;
    _supportProgram = null;
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
    const multiPassPipelineEnabled = !!(_supportProgram && _supportSmoothProgram && _drrProgram && _toneProgram);
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
    let supportReadback: FramebufferReadback | null = null;
    let drrReadback: FramebufferReadback | null = null;
    let debugMaps: GpuPanoDebugMaps | undefined;
    let diagnostics: GpuPanoDiagnostics = {
        expectedPipelineMode: EXPECTED_GPU_PIPELINE_MODE,
        phase2GatePassed,
        degradedModeReason: phase2GatePassed ? null : 'multi-pass-unavailable-single-pass-fallback',
    };

    if (multiPassPipelineEnabled) {
        if (!_supportFboA || !_supportFboB || !_drrFbo || !_fbo || !_supportTexA || !_supportTexB || !_drrTex) {
            throw new Error('[CPR-GPU] Multi-pass render targets were not initialized.');
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, _supportFboA);
        gl.useProgram(_supportProgram);
        bindCommonRayUniforms(gl, _supportProgram, commonUniforms);
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
        drawFullscreenTriangle(gl);

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
    meanMap = readback.channel1;
    maxMap = readback.channel2;
    sampleCountMap = readback.channel3;
    rawMin = readback.rawMin;
    rawMax = readback.rawMax;
    console.log('[GPU-RAW-MINMAX]', {
        min: Number.isFinite(rawMin) ? rawMin : null,
        max: Number.isFinite(rawMax) ? rawMax : null,
    });

    if (multiPassPipelineEnabled && supportReadback && drrReadback) {
        const w2iMat = coerceWorldToIndexMat4(worldToIndex ?? undefined);
        const nativePitchMmByCol = computeNativeSlabPitchMmForFrames(w2iMat, frames);
        const supportSigmaMap = buildSupportSigmaMap(
            supportReadback.channel2,
            supportReadback.channel1,
            nativePitchMmByCol,
            safePanoWidth,
            safePanoHeight,
            slabHalfThicknessMm
        );
        const localTransmittanceMap = buildLocalTransmittanceMap(drrReadback.channel0);
        const troughHalfWidthMap = new Float32Array(supportSigmaMap.length);
        for (let i = 0; i < supportSigmaMap.length; i++) {
            troughHalfWidthMap[i] =
                supportSigmaMap[i] * GPU_DRR_MODEL_PARAMS.approxTroughBoundarySigmaMultiplier;
        }

        diagnostics = {
            ...diagnostics,
            supportSurface: buildSupportSurfaceDiagnostics(
                supportReadback.channel0,
                supportReadback.channel1,
                supportReadback.channel2,
                supportReadback.channel3,
                safePanoWidth,
                safePanoHeight,
                slabHalfThicknessMm
            ),
            drr: buildDrrDiagnostics(
                drrReadback.channel0,
                drrReadback.channel2,
                drrReadback.channel1,
                drrReadback.channel3,
                supportSigmaMap,
                localTransmittanceMap,
                nativePitchMmByCol,
                slabHalfThicknessMm,
                requestedSlabSamples
            ),
            toneMap: buildToneMapDiagnostics(
                meanMap,
                drrReadback.channel2,
                drrReadback.channel1,
                sampleCountMap,
                pixelData
            ),
        };

        if (returnDebugSidecars) {
            debugMaps = {
                supportDepthMap: supportReadback.channel0,
                supportConfidenceMap: supportReadback.channel1,
                supportSpreadMap: supportReadback.channel2,
                supportDensityMap: supportReadback.channel3,
                totalAttenuationMap: drrReadback.channel0,
                fogAttenuationMap: drrReadback.channel2,
                lowerPenaltyMap: drrReadback.channel1,
                participatingSampleCountMap: drrReadback.channel3,
                toneResponseMap: sampleCountMap,
                troughHalfWidthMap,
            };
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
                debugMaps.supportDepthMap,
                debugMaps.supportConfidenceMap,
                debugMaps.supportSpreadMap,
                debugMaps.supportDensityMap,
                debugMaps.totalAttenuationMap,
                debugMaps.fogAttenuationMap,
                debugMaps.lowerPenaltyMap,
                debugMaps.participatingSampleCountMap,
                debugMaps.toneResponseMap,
                debugMaps.troughHalfWidthMap,
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


