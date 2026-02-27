/**
 * cprWorker.ts
 * Web Worker: Generates the 2D panoramic image from a 3D CBCT volume.
 */

interface WorkerFrame {
  position: [number, number, number];
  N_slab: [number, number, number];
}

interface CPRWorkerInput {
  scalarData: Float32Array | Int16Array;
  isSharedArrayBuffer: boolean;
  dimensions: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[];
  verticalDir?: [number, number, number];
  frames: WorkerFrame[];
  panoWidth: number;
  panoHeight: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  aggregation: 'MIP' | 'MEAN';
  applyModalityLut?: boolean;
  rescaleSlope?: number;
  rescaleIntercept?: number;
  bitsStored?: number;
  pixelRepresentation?: number;
  debugRunId?: string;
}

interface CPRWorkerSuccess {
  type: 'SUCCESS';
  pixelData: Float32Array;
  panoWidth: number;
  panoHeight: number;
  minValue: number;
  maxValue: number;
}

interface CPRWorkerError {
  type: 'ERROR';
  message: string;
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-10) {
    return [0, 0, 1];
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function worldToVoxel(
  wx: number,
  wy: number,
  wz: number,
  origin: [number, number, number],
  spacing: [number, number, number],
  invDir: number[]
): [number, number, number] {
  const rx = wx - origin[0];
  const ry = wy - origin[1];
  const rz = wz - origin[2];

  const vi = (invDir[0] * rx + invDir[1] * ry + invDir[2] * rz) / spacing[0];
  const vj = (invDir[3] * rx + invDir[4] * ry + invDir[5] * rz) / spacing[1];
  const vk = (invDir[6] * rx + invDir[7] * ry + invDir[8] * rz) / spacing[2];

  return [vi, vj, vk];
}

function invertMatrix3(m: number[]): number[] {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

function trilinear(
  data: Float32Array | Int16Array,
  dims: [number, number, number],
  vi: number,
  vj: number,
  vk: number,
  oobValue: number = -1000,
  normalizeSample?: (value: number) => number
): number {
  const [nx, ny, nz] = dims;

  if (vi < 0 || vj < 0 || vk < 0 || vi >= nx - 1 || vj >= ny - 1 || vk >= nz - 1) {
    const ci = Math.max(0, Math.min(nx - 1.001, vi));
    const cj = Math.max(0, Math.min(ny - 1.001, vj));
    const ck = Math.max(0, Math.min(nz - 1.001, vk));

    if (vi < -0.5 || vj < -0.5 || vk < -0.5 || vi > nx - 0.5 || vj > ny - 0.5 || vk > nz - 0.5) {
      return oobValue;
    }

    return trilinearCore(data, nx, ny, ci, cj, ck, normalizeSample);
  }

  return trilinearCore(data, nx, ny, vi, vj, vk, normalizeSample);
}

function trilinearCore(
  data: Float32Array | Int16Array,
  nx: number,
  ny: number,
  vi: number,
  vj: number,
  vk: number,
  normalizeSample?: (value: number) => number
): number {
  const i0 = Math.floor(vi);
  const j0 = Math.floor(vj);
  const k0 = Math.floor(vk);
  const i1 = i0 + 1;
  const j1 = j0 + 1;
  const k1 = k0 + 1;

  const fi = vi - i0;
  const fj = vj - j0;
  const fk = vk - k0;

  const sliceStride = nx * ny;

  const c000 = k0 * sliceStride + j0 * nx + i0;
  const c100 = k0 * sliceStride + j0 * nx + i1;
  const c010 = k0 * sliceStride + j1 * nx + i0;
  const c110 = k0 * sliceStride + j1 * nx + i1;
  const c001 = k1 * sliceStride + j0 * nx + i0;
  const c101 = k1 * sliceStride + j0 * nx + i1;
  const c011 = k1 * sliceStride + j1 * nx + i0;
  const c111 = k1 * sliceStride + j1 * nx + i1;

  const read = normalizeSample || ((value: number) => value);
  const v000 = read(data[c000]);
  const v100 = read(data[c100]);
  const v010 = read(data[c010]);
  const v110 = read(data[c110]);
  const v001 = read(data[c001]);
  const v101 = read(data[c101]);
  const v011 = read(data[c011]);
  const v111 = read(data[c111]);

  const c00 = v000 + fi * (v100 - v000);
  const c10 = v010 + fi * (v110 - v010);
  const c01 = v001 + fi * (v101 - v001);
  const c11 = v011 + fi * (v111 - v011);

  const c0 = c00 + fj * (c10 - c00);
  const c1 = c01 + fj * (c11 - c01);

  return c0 + fk * (c1 - c0);
}

function generatePanorama(input: CPRWorkerInput): {
  pixelData: Float32Array;
  minValue: number;
  maxValue: number;
  lutSamplePreview: number[];
  outputPixelPreview: number[];
  modalityLutApplied: boolean;
  storedValueNormalizationApplied: boolean;
} {
  const {
    scalarData,
    dimensions,
    spacing,
    origin,
    direction,
    verticalDir,
    frames,
    panoWidth,
    panoHeight,
    slabHalfThicknessMm,
    slabSamples,
    aggregation,
    rescaleSlope,
    rescaleIntercept,
    bitsStored,
    pixelRepresentation,
  } = input;

  const invDir = invertMatrix3(direction);
  const pixelData = new Float32Array(panoWidth * panoHeight);
  let minValue = Infinity;
  let maxValue = -Infinity;
  const aggregationMode = aggregation === 'MEAN' ? 'MEAN' : 'MIP';
  const isMeanAggregation = aggregationMode === 'MEAN';
  const safeSlope =
    Number.isFinite(rescaleSlope) && Math.abs(Number(rescaleSlope)) > 1e-8 ? Number(rescaleSlope) : 1;
  const safeIntercept = Number.isFinite(rescaleIntercept) ? Number(rescaleIntercept) : 0;
  const safeBitsStored =
    Number.isFinite(bitsStored) && Number(bitsStored) >= 1 ? Math.floor(Number(bitsStored)) : null;
  const safePixelRepresentation =
    Number.isFinite(pixelRepresentation) && (Number(pixelRepresentation) === 0 || Number(pixelRepresentation) === 1)
      ? Number(pixelRepresentation)
      : null;
  const shouldNormalizeStoredValues =
    safeBitsStored !== null &&
    safeBitsStored < 16 &&
    safePixelRepresentation !== null &&
    scalarData instanceof Int16Array;
  const normalizedBitsStored = shouldNormalizeStoredValues ? (safeBitsStored as number) : 0;
  const storedMask = shouldNormalizeStoredValues ? (1 << normalizedBitsStored) - 1 : 0;
  const storedSignBit = shouldNormalizeStoredValues ? 1 << (normalizedBitsStored - 1) : 0;
  const storedRange = shouldNormalizeStoredValues ? 1 << normalizedBitsStored : 0;

  const normalizeStoredSample = shouldNormalizeStoredValues
    ? (value: number): number => {
        if (!Number.isFinite(value)) {
          return 0;
        }
        const intValue = Math.round(value);
        const rawU16 = intValue & 0xffff;
        let normalized = rawU16 & storedMask;
        if (safePixelRepresentation === 1 && (normalized & storedSignBit) !== 0) {
          normalized -= storedRange;
        }
        return normalized;
      }
    : undefined;
  // Enforce modality LUT conversion whenever metadata is non-identity.
  // This prevents CPR pano output from staying in stored-value space.
  const shouldApplyModalityLut = safeSlope !== 1 || safeIntercept !== 0;
  const lutSamplePreview: number[] = [];

  // Prefer precomputed vertical direction from orchestrator to keep main thread and worker
  // in exact agreement. Fallback to direction matrix K-axis if missing.
  const effectiveVerticalDir =
    Array.isArray(verticalDir) && verticalDir.length >= 3
      ? normalize3([verticalDir[0], verticalDir[1], verticalDir[2]])
      : normalize3([direction[6] ?? 0, direction[7] ?? 0, direction[8] ?? 1]);

  const vertHalfMm = 15.0;
  const panoHeightDen = Math.max(1, panoHeight - 1);
  const vertStepMm = (vertHalfMm * 2) / panoHeightDen;

  const slabSampleCount = Math.max(1, Math.floor(slabSamples));
  const slabStepMm = slabSampleCount > 1 ? (slabHalfThicknessMm * 2) / (slabSampleCount - 1) : 0;

  for (let col = 0; col < panoWidth; col++) {
    const frame = frames[col];
    const [px, py, pz] = frame.position;
    const [nx, ny, nz_slab] = frame.N_slab;

    for (let row = 0; row < panoHeight; row++) {
      const vertOffsetMm = -vertHalfMm + row * vertStepMm;

      const bx = px + vertOffsetMm * effectiveVerticalDir[0];
      const by = py + vertOffsetMm * effectiveVerticalDir[1];
      const bz = pz + vertOffsetMm * effectiveVerticalDir[2];

      let accumulator = isMeanAggregation ? 0 : -Infinity;

      for (let s = 0; s < slabSampleCount; s++) {
        const slabOffset = slabSampleCount > 1 ? -slabHalfThicknessMm + s * slabStepMm : 0;

        const sx = bx + slabOffset * nx;
        const sy = by + slabOffset * ny;
        const sz = bz + slabOffset * nz_slab;

        const [vi, vj, vk] = worldToVoxel(sx, sy, sz, origin, spacing, invDir);
        let sample = trilinear(scalarData, dimensions, vi, vj, vk, -1000, normalizeStoredSample);
        if (shouldApplyModalityLut) {
          sample = sample * safeSlope + safeIntercept;
          if (lutSamplePreview.length < 5 && Number.isFinite(sample)) {
            lutSamplePreview.push(sample);
          }
        }
        if (!Number.isFinite(sample)) {
          sample = -1000;
        }

        if (!isMeanAggregation) {
          if (sample > accumulator) {
            accumulator = sample;
          }
        } else {
          accumulator += sample;
        }
      }

      const pixelValueRaw = isMeanAggregation ? accumulator / slabSampleCount : accumulator;
      const pixelValue = Number.isFinite(pixelValueRaw) ? pixelValueRaw : -1000;

      const pixelIndex = row * panoWidth + col;
      pixelData[pixelIndex] = pixelValue;

      if (pixelValue < minValue) {
        minValue = pixelValue;
      }
      if (pixelValue > maxValue) {
        maxValue = pixelValue;
      }
    }
  }

  const outputPixelPreview = Array.from(pixelData.subarray(0, Math.min(5, pixelData.length)));
  return {
    pixelData,
    minValue,
    maxValue,
    lutSamplePreview,
    outputPixelPreview,
    modalityLutApplied: shouldApplyModalityLut,
    storedValueNormalizationApplied: shouldNormalizeStoredValues,
  };
}

// eslint-disable-next-line no-restricted-globals
self.onmessage = function (event: MessageEvent<CPRWorkerInput>) {
  try {
    const input = event.data;
    const workerTag = input.debugRunId ? `[cprWorker:${input.debugRunId}]` : '[cprWorker]';
    const safeSlope =
      Number.isFinite(input.rescaleSlope) && Math.abs(Number(input.rescaleSlope)) > 1e-8
        ? Number(input.rescaleSlope)
        : 1;
    const safeIntercept = Number.isFinite(input.rescaleIntercept)
      ? Number(input.rescaleIntercept)
      : 0;
    const modalityLutFlagMismatch = !input.applyModalityLut && (safeSlope !== 1 || safeIntercept !== 0);

    if (!input.scalarData || input.scalarData.length === 0) {
      throw new Error('Received empty or null scalar data.');
    }
    if (!input.frames || input.frames.length === 0) {
      throw new Error('Received empty frames array.');
    }

    const bufferType = input.isSharedArrayBuffer ? 'SharedArrayBuffer' : 'ArrayBuffer (cloned)';
    console.debug(`${workerTag} Starting panorama generation. Buffer: ${bufferType}`);
    console.debug(`${workerTag} Volume dims: ${input.dimensions}  Frames: ${input.frames.length}`);
    console.debug(
      `${workerTag} Output: ${input.panoWidth}x${input.panoHeight}  Slab: ${input.slabSamples} samples`
    );

    const start = performance.now();
    const {
      pixelData,
      minValue,
      maxValue,
      lutSamplePreview,
      outputPixelPreview,
      modalityLutApplied,
      storedValueNormalizationApplied,
    } = generatePanorama(input);
    const elapsed = (performance.now() - start).toFixed(0);

    console.debug(
      `${workerTag} Done in ${elapsed}ms. Range: [${minValue.toFixed(0)}, ${maxValue.toFixed(0)}] HU`
    );
    if (modalityLutApplied) {
      console.log(
        `${workerTag} LUT preview (first ${lutSamplePreview.length} converted samples):`,
        lutSamplePreview
      );
    }
    if (storedValueNormalizationApplied) {
      console.log(`${workerTag} Applied bitsStored/pixelRepresentation normalization before interpolation.`, {
        bitsStored: input.bitsStored,
        pixelRepresentation: input.pixelRepresentation,
      });
    }
    console.log(
      `${workerTag} Output preview (first ${outputPixelPreview.length} pano pixels):`,
      outputPixelPreview
    );
    if (modalityLutFlagMismatch) {
      console.warn(
        `${workerTag} applyModalityLut flag was false, but non-identity rescale metadata was detected. ` +
          'LUT conversion was enforced in worker.'
      );
    }

    const response: CPRWorkerSuccess = {
      type: 'SUCCESS',
      pixelData,
      panoWidth: input.panoWidth,
      panoHeight: input.panoHeight,
      minValue,
      maxValue,
    };

    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response, [pixelData.buffer]);
  } catch (err) {
    const response: CPRWorkerError = {
      type: 'ERROR',
      message: err instanceof Error ? err.message : String(err),
    };
    // eslint-disable-next-line no-restricted-globals
    (self as unknown as Worker).postMessage(response);
  }
};
