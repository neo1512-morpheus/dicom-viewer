export interface SyntheticCprVoiSettings {
  lower: number;
  upper: number;
  windowWidth: number;
  windowCenter: number;
}

export type SyntheticCprIntensityDomain = 'hu' | 'native' | 'unknown';

interface SyntheticCprVoiOptions {
  intensityDomain?: SyntheticCprIntensityDomain;
}

const CPR_SYNTHETIC_FALLBACK_WINDOW_WIDTH = 3500;
const CPR_SYNTHETIC_FALLBACK_WINDOW_CENTER = 850;

function clampNumber(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function isFiniteRange(minValue: number, maxValue: number): boolean {
  return Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue;
}

function looksLikeCalibratedHuRange(minValue: number, maxValue: number): boolean {
  return isFiniteRange(minValue, maxValue) && minValue <= -400 && maxValue >= 700;
}

function percentile(sortedValues: number[], quantile: number): number {
  if (sortedValues.length <= 0) {
    return 0;
  }

  if (sortedValues.length === 1) {
    return sortedValues[0];
  }

  const q = clampNumber(quantile, 0, 1);
  const index = q * (sortedValues.length - 1);
  const lowIndex = Math.floor(index);
  const highIndex = Math.min(sortedValues.length - 1, Math.ceil(index));
  if (lowIndex === highIndex) {
    return sortedValues[lowIndex];
  }

  const t = index - lowIndex;
  return sortedValues[lowIndex] * (1 - t) + sortedValues[highIndex] * t;
}

export function getFallbackSyntheticCprVoi(): SyntheticCprVoiSettings {
  const windowWidth = CPR_SYNTHETIC_FALLBACK_WINDOW_WIDTH;
  const windowCenter = CPR_SYNTHETIC_FALLBACK_WINDOW_CENTER;
  return {
    lower: windowCenter - windowWidth / 2,
    upper: windowCenter + windowWidth / 2,
    windowWidth,
    windowCenter,
  };
}

export function isSyntheticCprHuDomain(
  intensityDomain: SyntheticCprIntensityDomain | null | undefined
): boolean {
  return intensityDomain === 'hu';
}

export function classifySyntheticCprIntensityDomain(params: {
  modalityLutApplied?: boolean;
  effectiveIsPreScaled?: boolean;
  minValue: number;
  maxValue: number;
  sourceMinValue?: number;
  sourceMaxValue?: number;
}): SyntheticCprIntensityDomain {
  const {
    modalityLutApplied,
    effectiveIsPreScaled,
    minValue,
    maxValue,
    sourceMinValue,
    sourceMaxValue,
  } = params;

  if (modalityLutApplied === true) {
    return 'hu';
  }

  const effectiveMin = Number.isFinite(sourceMinValue) ? Number(sourceMinValue) : Number(minValue);
  const effectiveMax = Number.isFinite(sourceMaxValue) ? Number(sourceMaxValue) : Number(maxValue);

  if (effectiveIsPreScaled === true) {
    return looksLikeCalibratedHuRange(effectiveMin, effectiveMax) ||
      looksLikeCalibratedHuRange(Number(minValue), Number(maxValue))
      ? 'hu'
      : 'unknown';
  }

  return isFiniteRange(Number(minValue), Number(maxValue)) ? 'native' : 'unknown';
}

function buildSyntheticCprVoi(
  lower: number,
  upper: number,
  centerHint?: number,
  options?: SyntheticCprVoiOptions
): SyntheticCprVoiSettings {
  void options;
  const fallback = getFallbackSyntheticCprVoi();

  if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) {
    return fallback;
  }

  const rawSpan = Math.max(1, upper - lower);
  const padding = Math.max(30, rawSpan * 0.04);
  const paddedLower = lower - padding;
  const paddedUpper = upper + padding;
  const windowWidth = paddedUpper - paddedLower;
  const windowCenter = Number.isFinite(centerHint)
    ? Number(centerHint)
    : paddedLower + windowWidth / 2;

  return {
    lower: windowCenter - windowWidth / 2,
    upper: windowCenter + windowWidth / 2,
    windowWidth,
    windowCenter,
  };
}

export function createSyntheticCprVoiFromRange(
  minValue: number,
  maxValue: number,
  options?: SyntheticCprVoiOptions
): SyntheticCprVoiSettings {
  const safeMin = Number.isFinite(minValue) ? Number(minValue) : Number.NaN;
  const safeMax = Number.isFinite(maxValue) ? Number(maxValue) : Number.NaN;
  return buildSyntheticCprVoi(safeMin, safeMax, undefined, options);
}

export function createSyntheticCprVoiFromBuffer(
  values: ArrayLike<number> | null | undefined,
  options?: SyntheticCprVoiOptions
): SyntheticCprVoiSettings {
  if (!values || !Number.isFinite(values.length) || values.length <= 0) {
    return getFallbackSyntheticCprVoi();
  }

  const sampled: number[] = [];
  const step = Math.max(1, Math.floor(values.length / 8192));

  for (let i = 0; i < values.length; i += step) {
    const value = Number(values[i]);
    if (Number.isFinite(value)) {
      sampled.push(value);
    }
  }

  if (sampled.length <= 0) {
    return getFallbackSyntheticCprVoi();
  }

  sampled.sort((a, b) => a - b);
  const lower = percentile(sampled, 0.02);
  const upper = percentile(sampled, 0.985);
  const centerHint = percentile(sampled, 0.5);
  return buildSyntheticCprVoi(lower, upper, centerHint, options);
}
