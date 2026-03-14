export type CprScalarArray = Float32Array | Int16Array | Uint16Array | ArrayLike<number>;

export interface EffectivePreScaledResolution {
  effectiveIsPreScaled: boolean;
  sampledMin: number;
  sampledMax: number;
  heuristicOverride: boolean;
}

export interface StoredValueNormalizationPolicy {
  safeSlope: number;
  safeIntercept: number;
  safeBitsStored: number;
  safeBitsAllocated: number;
  safeHighBit: number;
  safePixelRepresentation: 0 | 1 | null;
  effectiveIsPreScaled: boolean;
  heuristicOverride: boolean;
  sampledMin: number;
  sampledMax: number;
  unsignedPackedArtifactDetected: boolean;
  hasBitDepthRangeMismatch: boolean;
  shouldNormalizeStoredValues: boolean;
  normalizationSignature: string | null;
  normalizeStoredSample?: (value: number) => number;
}

export interface HuScalarTransform extends StoredValueNormalizationPolicy {
  shouldApplyModalityLut: boolean;
  safeInterpolationOobValue: number;
  normalizeSample(value: number): number;
}

export interface NormalizeScalarDataToHuResult {
  pixelData: Float32Array;
  transform: HuScalarTransform;
}

function isPackedIntegerArray(data: CprScalarArray): data is Int16Array | Uint16Array {
  return data instanceof Int16Array || data instanceof Uint16Array;
}

function clampInteger(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, Math.floor(value)));
}

export function resolveStoredBitCount(bitsStored: number | null | undefined, fallback = 16): number {
  return Number.isFinite(bitsStored) && Number(bitsStored) >= 1 && Number(bitsStored) <= 31
    ? Math.floor(Number(bitsStored))
    : fallback;
}

export function resolveStoredBitMask(bitsStored: number | null | undefined, fallback = 16): number {
  const safeBitsStored = resolveStoredBitCount(bitsStored, fallback);
  return safeBitsStored >= 31 ? 0x7fffffff : (1 << safeBitsStored) - 1;
}

export function decodeStoredScalarValue(
  rawValue: number,
  bitsStored: number | null | undefined,
  pixelRepresentation: number | null | undefined,
  fallbackBits = 16
): number {
  if (!Number.isFinite(rawValue)) {
    return Number.NaN;
  }

  const safeBitsStored = resolveStoredBitCount(bitsStored, fallbackBits);
  const maskedValue = Math.trunc(rawValue) & resolveStoredBitMask(safeBitsStored, fallbackBits);

  if (pixelRepresentation === 1) {
    const signBit = safeBitsStored >= 31 ? 0x40000000 : 1 << (safeBitsStored - 1);
    const signedRange = safeBitsStored >= 31 ? 0x80000000 : 1 << safeBitsStored;
    return maskedValue >= signBit ? maskedValue - signedRange : maskedValue;
  }

  return maskedValue;
}

export function sampleScalarRange(data: ArrayLike<number>): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  const step = Math.max(1, Math.floor(data.length / 4096));

  for (let i = 0; i < data.length; i += step) {
    const value = Number(data[i]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }

  return {
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
  };
}

export function isHuLikeScalarRange(min: number, max: number): boolean {
  return Number.isFinite(min) && Number.isFinite(max) && min >= -5000 && max <= 7000;
}

export function resolveEffectivePreScaledForInit(params: {
  scalarData: CprScalarArray;
  isPreScaled?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  pixelRepresentation?: number;
}): EffectivePreScaledResolution {
  const {
    scalarData,
    isPreScaled,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    pixelRepresentation,
  } = params;
  const safeSlope =
    Number.isFinite(rescaleSlope) && Math.abs(Number(rescaleSlope)) > 1e-8
      ? Number(rescaleSlope)
      : 1;
  const safeIntercept = Number.isFinite(rescaleIntercept) ? Number(rescaleIntercept) : 0;
  const hasNonIdentityRescale = Math.abs(safeSlope - 1) > 1e-6 || Math.abs(safeIntercept) > 1e-6;
  const sampledRange = sampleScalarRange(scalarData);
  const sampledMin = sampledRange.min;
  const sampledMax = sampledRange.max;
  const looksHuLike = isHuLikeScalarRange(sampledMin, sampledMax);
  const safeBitsStored = resolveStoredBitCount(bitsStored, 16);
  const isUnsigned = Number(pixelRepresentation) === 0;
  const storedMin = isUnsigned ? 0 : -(1 << Math.max(safeBitsStored - 1, 0));
  const storedMax =
    safeBitsStored >= 31
      ? Number.MAX_SAFE_INTEGER
      : isUnsigned
        ? (1 << safeBitsStored) - 1
        : (1 << Math.max(safeBitsStored - 1, 0)) - 1;
  const storedMargin = Math.max(8, safeBitsStored < 16 ? 16 : 64);
  const looksLikeStoredRange =
    Number.isFinite(sampledMin) &&
    Number.isFinite(sampledMax) &&
    sampledMin >= storedMin - storedMargin &&
    sampledMax <= storedMax + storedMargin;
  const looksImplausiblyWideRange =
    Number.isFinite(sampledMin) &&
    Number.isFinite(sampledMax) &&
    (sampledMin < -9000 || sampledMax > 14000);

  if (isPreScaled === true) {
    if (looksHuLike || !hasNonIdentityRescale) {
      return {
        effectiveIsPreScaled: true,
        sampledMin,
        sampledMax,
        heuristicOverride: false,
      };
    }

    // Some source volumes are flagged pre-scaled even though the cached Int16
    // payload still behaves like stored values. Fall back to stored-value
    // decoding when the sampled range is implausibly wide or still sits inside
    // the native bit-depth envelope.
    if (looksImplausiblyWideRange || looksLikeStoredRange) {
      return {
        effectiveIsPreScaled: false,
        sampledMin,
        sampledMax,
        heuristicOverride: true,
      };
    }

    return {
      effectiveIsPreScaled: true,
      sampledMin,
      sampledMax,
      heuristicOverride: false,
    };
  }

  if (!hasNonIdentityRescale) {
    return {
      effectiveIsPreScaled: false,
      sampledMin,
      sampledMax,
      heuristicOverride: false,
    };
  }

  const impossibleUnsignedHuRange = isUnsigned && sampledMin < -1;
  const floatHuPayload = scalarData instanceof Float32Array && looksHuLike;
  const likelyAlreadyScaled =
    looksHuLike && (floatHuPayload || impossibleUnsignedHuRange || !looksLikeStoredRange);

  return {
    effectiveIsPreScaled: likelyAlreadyScaled,
    sampledMin,
    sampledMax,
    heuristicOverride: likelyAlreadyScaled,
  };
}

export function resolveStoredValueNormalizationPolicy(params: {
  scalarData: CprScalarArray;
  isPreScaled?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  bitsAllocated?: number;
  highBit?: number;
  pixelRepresentation?: number;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
}): StoredValueNormalizationPolicy {
  const {
    scalarData,
    isPreScaled,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    bitsAllocated,
    highBit,
    pixelRepresentation,
    allowStoredValueNormalization,
    disableStoredValueNormalization,
  } = params;

  const safeSlope =
    Number.isFinite(rescaleSlope) && Math.abs(Number(rescaleSlope)) > 1e-8
      ? Number(rescaleSlope)
      : 1;
  const safeIntercept = Number.isFinite(rescaleIntercept) ? Number(rescaleIntercept) : 0;
  const safeBitsStored = resolveStoredBitCount(bitsStored, 16);
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && Number(bitsAllocated) >= safeBitsStored
      ? Math.floor(Number(bitsAllocated))
      : 16;
  const safeHighBit = clampInteger(
    Number.isFinite(highBit) ? Number(highBit) : safeBitsStored - 1,
    0,
    Math.max(0, safeBitsAllocated - 1)
  );
  const safePixelRepresentation =
    Number.isFinite(pixelRepresentation) &&
    (Number(pixelRepresentation) === 0 || Number(pixelRepresentation) === 1)
      ? (Number(pixelRepresentation) as 0 | 1)
      : null;
  const preScaledResolution = resolveEffectivePreScaledForInit({
    scalarData,
    isPreScaled,
    rescaleSlope: safeSlope,
    rescaleIntercept: safeIntercept,
    bitsStored: safeBitsStored,
    pixelRepresentation: safePixelRepresentation ?? undefined,
  });
  const effectiveIsPreScaled = preScaledResolution.effectiveIsPreScaled;
  const sampledMin = preScaledResolution.sampledMin;
  const sampledMax = preScaledResolution.sampledMax;
  const normalizationEligible =
    !effectiveIsPreScaled &&
    safeBitsStored < 16 &&
    safePixelRepresentation !== null &&
    isPackedIntegerArray(scalarData);
  const nominalStoredMax =
    safeBitsStored < 31 ? (1 << safeBitsStored) - 1 : Number.MAX_SAFE_INTEGER;
  const nominalSignedMin = -(1 << Math.max(safeBitsStored - 1, 0));
  const unsignedPackedArtifactDetected =
    normalizationEligible &&
    safePixelRepresentation === 0 &&
    (sampledMin < -1 || sampledMax > nominalStoredMax + 8);
  const hasBitDepthRangeMismatch =
    normalizationEligible &&
    (sampledMin < nominalSignedMin - 8 || sampledMax > nominalStoredMax + 8);
  const allowNormalizationDefault =
    unsignedPackedArtifactDetected ||
    hasBitDepthRangeMismatch ||
    (normalizationEligible &&
      safePixelRepresentation === 0 &&
      sampledMin >= 0 &&
      sampledMax <= nominalStoredMax);
  const shouldNormalizeStoredValues =
    normalizationEligible &&
    (disableStoredValueNormalization === true
      ? false
      : typeof allowStoredValueNormalization === 'boolean'
        ? allowStoredValueNormalization
        : allowNormalizationDefault);
  const normalizedBitsStored = shouldNormalizeStoredValues ? safeBitsStored : 0;
  const normalizedHighBit = shouldNormalizeStoredValues
    ? clampInteger(safeHighBit, 0, Math.max(0, safeBitsAllocated - 1))
    : 0;
  const bitAlignmentShift = shouldNormalizeStoredValues
    ? clampInteger(normalizedHighBit + 1 - normalizedBitsStored, -15, 15)
    : 0;
  const storedMask =
    shouldNormalizeStoredValues && normalizedBitsStored > 0 ? (1 << normalizedBitsStored) - 1 : 0;
  const storedSignBit =
    shouldNormalizeStoredValues && normalizedBitsStored > 0 ? 1 << (normalizedBitsStored - 1) : 0;
  const storedRange =
    shouldNormalizeStoredValues && normalizedBitsStored > 0 ? 1 << normalizedBitsStored : 0;
  const normalizeStoredSample = shouldNormalizeStoredValues
    ? (value: number): number => {
      if (!Number.isFinite(value)) {
        return 0;
      }
      const intValue = Math.round(value);
      const rawU16 = intValue & 0xffff;
      let aligned = rawU16;
      if (bitAlignmentShift > 0) {
        aligned = rawU16 >>> bitAlignmentShift;
      } else if (bitAlignmentShift < 0) {
        aligned = (rawU16 << -bitAlignmentShift) & 0xffff;
      }
      let normalized = aligned & storedMask;
      if (safePixelRepresentation === 1 && (normalized & storedSignBit) !== 0) {
        normalized -= storedRange;
      }
      return normalized;
    }
    : undefined;
  const normalizationSignature = shouldNormalizeStoredValues
    ? `packed:${normalizedBitsStored}:${normalizedHighBit}:${bitAlignmentShift}:${safePixelRepresentation ?? 'na'}`
    : null;

  return {
    safeSlope,
    safeIntercept,
    safeBitsStored,
    safeBitsAllocated,
    safeHighBit,
    safePixelRepresentation,
    effectiveIsPreScaled,
    heuristicOverride: preScaledResolution.heuristicOverride,
    sampledMin,
    sampledMax,
    unsignedPackedArtifactDetected,
    hasBitDepthRangeMismatch,
    shouldNormalizeStoredValues,
    normalizationSignature,
    normalizeStoredSample,
  };
}

export function createHuScalarTransform(params: {
  scalarData: CprScalarArray;
  isPreScaled?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  bitsAllocated?: number;
  highBit?: number;
  pixelRepresentation?: number;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
}): HuScalarTransform {
  const policy = resolveStoredValueNormalizationPolicy(params);
  const shouldApplyModalityLut =
    !policy.effectiveIsPreScaled &&
    (Math.abs(policy.safeSlope - 1) > 1e-6 || Math.abs(policy.safeIntercept) > 1e-6);
  const interpolationOobValue = shouldApplyModalityLut
    ? (-1000 - policy.safeIntercept) / policy.safeSlope
    : -1000;
  const safeInterpolationOobValue = Number.isFinite(interpolationOobValue)
    ? interpolationOobValue
    : -1000;
  const normalizeSample = (value: number): number => {
    if (!Number.isFinite(value)) {
      return policy.effectiveIsPreScaled ? -1000 : policy.safeIntercept;
    }

    if (policy.effectiveIsPreScaled) {
      return value;
    }

    const storedValue = policy.normalizeStoredSample
      ? policy.normalizeStoredSample(value)
      : decodeStoredScalarValue(
        value,
        policy.safeBitsStored,
        policy.safePixelRepresentation,
        16
      );
    const hu = storedValue * policy.safeSlope + policy.safeIntercept;
    return Number.isFinite(hu) ? hu : policy.safeIntercept;
  };

  return {
    ...policy,
    shouldApplyModalityLut,
    safeInterpolationOobValue,
    normalizeSample,
  };
}

export function normalizeScalarDataToHuFloat32(params: {
  scalarData: CprScalarArray;
  isPreScaled?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  bitsAllocated?: number;
  highBit?: number;
  pixelRepresentation?: number;
  allowStoredValueNormalization?: boolean;
  disableStoredValueNormalization?: boolean;
}): NormalizeScalarDataToHuResult {
  const transform = createHuScalarTransform(params);
  const pixelData = new Float32Array(params.scalarData.length);

  for (let i = 0; i < params.scalarData.length; i++) {
    pixelData[i] = transform.normalizeSample(Number(params.scalarData[i]));
  }

  return {
    pixelData,
    transform,
  };
}
