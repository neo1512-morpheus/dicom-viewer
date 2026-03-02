/**
 * cprGpuRenderer.ts
 * GPU-accelerated panoramic CPR rendering using WebGL2.
 * Replaces CPU trilinear sampling with hardware 3D texture interpolation.
 */

// ─── Types ───────────────────────────────────────────────────────────
export interface GpuPanoInput {
    scalarData: Float32Array | Int16Array;
    dimensions: [number, number, number];
    spacing: [number, number, number];
    origin: [number, number, number];
    direction: number[];
    worldToIndex?: number[] | null;

    /** Per-column spline frames (position + slab normal). Length = panoWidth. */
    frames: Array<{
        position: [number, number, number];
        N_slab: [number, number, number];
    }>;

    panoWidth: number;
    panoHeight: number;
    verticalDir: [number, number, number];
    vertHalfMm: number;
    verticalCenterOffsetMm: number;
    slabHalfThicknessMm: number;
    slabSamples: number;
    gaussSigma: number; // σ for Gaussian slab weighting

    rescaleSlope: number;
    rescaleIntercept: number;
    applyRescale: boolean;

    /** Optional normalizer for packed stored values (bit alignment, sign extension). */
    normalizeStoredSample?: (value: number) => number;
}

export interface GpuPanoResult {
    pixelData: Float32Array;
    width: number;
    height: number;
    minValue: number;
    maxValue: number;
}

// ─── Shaders ─────────────────────────────────────────────────────────
const VERT_SRC = `#version 300 es
void main() {
  // Fullscreen triangle: 3 vertices cover the clip-space quad
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
precision highp sampler2D;

uniform sampler3D uVolume;
uniform sampler2D uSplineData; // width=panoWidth, height=2; row0=pos, row1=slabDir

uniform vec3 uVerticalDir;
uniform float uVertHalfMm;
uniform float uVertCenterOffsetMm;
uniform float uSlabHalfMm;
uniform int uSlabSamples;
uniform float uGaussSigma;
uniform int uPanoWidth;
uniform int uPanoHeight;

uniform mat4 uWorldToIndex;
uniform vec3 uDims;

// Rescale: applied AFTER trilinear sampling (matches Cornerstone GPU pipeline)
uniform float uRescaleSlope;
uniform float uRescaleIntercept;
uniform bool uApplyRescale;

out vec4 fragColor;

void main() {
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  if (col >= uPanoWidth || row >= uPanoHeight) {
    fragColor = vec4(-1000.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 pos = texelFetch(uSplineData, ivec2(col, 0), 0).xyz;
  vec3 slabDir = texelFetch(uSplineData, ivec2(col, 1), 0).xyz;

  float panoHeightDen = max(1.0, float(uPanoHeight - 1));
  float vertStepMm = (uVertHalfMm * 2.0) / panoHeightDen;
  float vertOffsetMm = uVertCenterOffsetMm + (uVertHalfMm - float(row) * vertStepMm);

  vec3 baseWorld = pos + vertOffsetMm * uVerticalDir;

  float gaussSigmaSq2 = 2.0 * uGaussSigma * uGaussSigma;
  float slabStep = uSlabSamples > 1
    ? (uSlabHalfMm * 2.0) / float(uSlabSamples - 1)
    : 0.0;

  // Edge margin: reject samples within 1.5 voxels of volume boundary
  vec3 edgeMargin = 1.5 / uDims;
  vec3 uvwMin = edgeMargin;
  vec3 uvwMax = vec3(1.0) - edgeMargin;

  float huSum = 0.0;
  float wSum = 0.0;
  float huSamples[32];
  float wSamples[32];
  int validCount = 0;

  const int MAX_SLAB = 32;
  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= uSlabSamples) break;

    float slabOffset = uSlabSamples > 1
      ? -uSlabHalfMm + float(s) * slabStep
      : 0.0;

    vec3 sampleWorld = baseWorld + slabOffset * slabDir;
    vec4 voxel4 = uWorldToIndex * vec4(sampleWorld, 1.0);
    vec3 uvw = (voxel4.xyz + 0.5) / uDims;

    if (any(lessThan(uvw, uvwMin)) || any(greaterThan(uvw, uvwMax))) {
      continue;
    }

    // HARDWARE TRILINEAR on raw stored values
    float rawVal = texture(uVolume, uvw).r;

    // Apply rescale AFTER interpolation (like Cornerstone GPU pipeline)
    float hu = uApplyRescale ? rawVal * uRescaleSlope + uRescaleIntercept : rawVal;

    // Clamp to practical dental HU band to suppress outlier-driven speckle.
    hu = clamp(hu, -1500.0, 5000.0);

    float gw = exp(-(slabOffset * slabOffset) / gaussSigmaSq2);
    huSum += hu * gw;
    wSum += gw;
    huSamples[validCount] = hu;
    wSamples[validCount] = gw;
    validCount++;
  }

  float finalHu = -1000.0;
  if (validCount > 0) {
    // Insertion sort by HU for robust trimmed mean aggregation.
    for (int i = 1; i < MAX_SLAB; i++) {
      if (i >= validCount) break;
      float keyHu = huSamples[i];
      float keyW = wSamples[i];
      int j = i - 1;
      for (int k = 0; k < MAX_SLAB; k++) {
        if (j < 0) break;
        if (huSamples[j] <= keyHu) break;
        huSamples[j + 1] = huSamples[j];
        wSamples[j + 1] = wSamples[j];
        j--;
      }
      huSamples[j + 1] = keyHu;
      wSamples[j + 1] = keyW;
    }

    int trimCount = validCount >= 7 ? max(1, int(floor(float(validCount) * 0.15))) : 0;
    int keepLo = trimCount;
    int keepHi = validCount - trimCount;

    float robustHuSum = 0.0;
    float robustWSum = 0.0;
    for (int i = 0; i < MAX_SLAB; i++) {
      if (i < keepLo || i >= keepHi) continue;
      robustHuSum += huSamples[i] * wSamples[i];
      robustWSum += wSamples[i];
    }

    finalHu = robustWSum > 0.0
      ? (robustHuSum / robustWSum)
      : (wSum > 0.0 ? huSum / wSum : -1000.0);
  }
  fragColor = vec4(finalHu, 0.0, 0.0, 1.0);
}
`;

// ─── Cached GPU State ────────────────────────────────────────────────
let _canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
let _gl: WebGL2RenderingContext | null = null;
let _program: WebGLProgram | null = null;
let _vao: WebGLVertexArrayObject | null = null;

// Volume texture cache (avoid re-uploading for same volume)
let _volumeTex: WebGLTexture | null = null;
let _cachedVolumeId: string | null = null;

let _splineTex: WebGLTexture | null = null;
let _fbo: WebGLFramebuffer | null = null;
let _fboTex: WebGLTexture | null = null;
let _fboWidth = 0;
let _fboHeight = 0;

// Uniform locations
let _uVolume = -1;
let _uSplineData = -1;
let _uVerticalDir = -1;
let _uVertHalfMm = -1;
let _uVertCenterOffsetMm = -1;
let _uSlabHalfMm = -1;
let _uSlabSamples = -1;
let _uGaussSigma = -1;
let _uPanoWidth = -1;
let _uPanoHeight = -1;
let _uWorldToIndex = -1;
let _uDims = -1;
let _uRescaleSlope = -1;
let _uRescaleIntercept = -1;
let _uApplyRescale = -1;

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

function applyLightBilateralDenoise(
    pixelData: Float32Array,
    width: number,
    height: number,
    blendWeight = 0.5,
    sigmaRange = 280,
    passes = 2
): void {
    if (width < 3 || height < 3 || !pixelData.length || passes <= 0) return;

    const blend = Math.max(0, Math.min(0.9, blendWeight));
    const keep = 1 - blend;
    const sigmaDen = 2 * sigmaRange * sigmaRange;
    const neighbors: Array<[number, number, number]> = [
        [-1, -1, 0.45], [0, -1, 0.7], [1, -1, 0.45],
        [-1, 0, 0.7], [1, 0, 0.7],
        [-1, 1, 0.45], [0, 1, 0.7], [1, 1, 0.45],
    ];

    for (let pass = 0; pass < passes; pass++) {
        const src = new Float32Array(pixelData);
        for (let row = 1; row < height - 1; row++) {
            const rowOffset = row * width;
            for (let col = 1; col < width - 1; col++) {
                const idx = rowOffset + col;
                const center = src[idx];
                if (!Number.isFinite(center)) continue;

                let weightedSum = center;
                let weightTotal = 1;

                for (let n = 0; n < neighbors.length; n++) {
                    const [dx, dy, spatialWeight] = neighbors[n];
                    const sample = src[idx + dy * width + dx];
                    if (!Number.isFinite(sample)) continue;
                    const diff = sample - center;
                    const rangeWeight = Math.exp(-(diff * diff) / sigmaDen);
                    const w = spatialWeight * rangeWeight;
                    weightedSum += sample * w;
                    weightTotal += w;
                }

                if (weightTotal <= 0) continue;
                const filtered = weightedSum / weightTotal;
                pixelData[idx] = keep * center + blend * filtered;
            }
        }
    }
}

function applyMildUnsharpMask(
    pixelData: Float32Array,
    width: number,
    height: number,
    amount = 0.16,
    thresholdHu = 42
): void {
    if (width < 3 || height < 3 || !pixelData.length || amount <= 0) return;

    const boost = Math.max(0, Math.min(0.45, amount));
    const threshold = Math.max(0, thresholdHu);
    const minHu = -1500;
    const maxHu = 5000;
    const src = new Float32Array(pixelData);

    for (let row = 1; row < height - 1; row++) {
        const rowOffset = row * width;
        for (let col = 1; col < width - 1; col++) {
            const idx = rowOffset + col;
            const center = src[idx];
            if (!Number.isFinite(center)) continue;

            const blurred =
                (src[idx] * 4 +
                    (src[idx - 1] + src[idx + 1] + src[idx - width] + src[idx + width]) * 2 +
                    (src[idx - width - 1] +
                        src[idx - width + 1] +
                        src[idx + width - 1] +
                        src[idx + width + 1])) / 16;

            if (!Number.isFinite(blurred)) continue;

            const detail = center - blurred;
            if (Math.abs(detail) < threshold) continue;

            const sharpened = center + detail * boost;
            pixelData[idx] = Math.max(minHu, Math.min(maxHu, sharpened));
        }
    }
}

// ─── Init ────────────────────────────────────────────────────────────
function ensureGpuContext(): WebGL2RenderingContext {
    if (_gl) return _gl;

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

    if (!_gl) {
        // Fallback: hidden DOM canvas
        const c = document.createElement('canvas');
        c.width = 1;
        c.height = 1;
        c.style.display = 'none';
        document.body.appendChild(c);
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
        console.warn('[CPR-GPU] OES_texture_float_linear not available — R32F LINEAR filtering may fallback to NEAREST.');
    }

    // Compile program
    const vs = compileShader(_gl, _gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(_gl, _gl.FRAGMENT_SHADER, FRAG_SRC);
    _program = _gl.createProgram()!;
    _gl.attachShader(_program, vs);
    _gl.attachShader(_program, fs);
    _gl.linkProgram(_program);
    if (!_gl.getProgramParameter(_program, _gl.LINK_STATUS)) {
        const log = _gl.getProgramInfoLog(_program);
        throw new Error(`Program link error: ${log}`);
    }
    _gl.deleteShader(vs);
    _gl.deleteShader(fs);

    // Cache uniform locations
    _uVolume = _gl.getUniformLocation(_program, 'uVolume') as number;
    _uSplineData = _gl.getUniformLocation(_program, 'uSplineData') as number;
    _uVerticalDir = _gl.getUniformLocation(_program, 'uVerticalDir') as number;
    _uVertHalfMm = _gl.getUniformLocation(_program, 'uVertHalfMm') as number;
    _uVertCenterOffsetMm = _gl.getUniformLocation(_program, 'uVertCenterOffsetMm') as number;
    _uSlabHalfMm = _gl.getUniformLocation(_program, 'uSlabHalfMm') as number;
    _uSlabSamples = _gl.getUniformLocation(_program, 'uSlabSamples') as number;
    _uGaussSigma = _gl.getUniformLocation(_program, 'uGaussSigma') as number;
    _uPanoWidth = _gl.getUniformLocation(_program, 'uPanoWidth') as number;
    _uPanoHeight = _gl.getUniformLocation(_program, 'uPanoHeight') as number;
    _uWorldToIndex = _gl.getUniformLocation(_program, 'uWorldToIndex') as number;
    _uDims = _gl.getUniformLocation(_program, 'uDims') as number;
    _uRescaleSlope = _gl.getUniformLocation(_program, 'uRescaleSlope') as number;
    _uRescaleIntercept = _gl.getUniformLocation(_program, 'uRescaleIntercept') as number;
    _uApplyRescale = _gl.getUniformLocation(_program, 'uApplyRescale') as number;

    // Empty VAO for fullscreen triangle (no attributes needed)
    _vao = _gl.createVertexArray()!;

    console.log('[CPR-GPU] WebGL2 context initialized.');
    return _gl;
}

// ─── Volume Texture Upload ──────────────────────────────────────────
function uploadVolumeTexture(
    gl: WebGL2RenderingContext,
    scalarData: Float32Array | Int16Array,
    dims: [number, number, number],
    volumeId?: string
): void {
    // Skip re-upload if same volume
    if (volumeId && volumeId === _cachedVolumeId && _volumeTex) {
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
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.R32F, nx, ny, nz);

    // Upload RAW stored values slice-by-slice (NO rescale — rescale is done in shader)
    const sliceSize = nx * ny;
    const sliceFloat = new Float32Array(sliceSize);

    for (let k = 0; k < nz; k++) {
        const offset = k * sliceSize;
        for (let i = 0; i < sliceSize; i++) {
            sliceFloat[i] = scalarData[offset + i];
        }
        gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, k, nx, ny, 1, gl.RED, gl.FLOAT, sliceFloat);

        // Diagnostic: log middle slice stats
        if (k === Math.floor(nz / 2)) {
            let sliceMin = Infinity, sliceMax = -Infinity;
            for (let i = 0; i < sliceSize; i++) {
                if (sliceFloat[i] < sliceMin) sliceMin = sliceFloat[i];
                if (sliceFloat[i] > sliceMax) sliceMax = sliceFloat[i];
            }
            console.log('[CPR-GPU] Volume texture MIDDLE slice stats (raw, no rescale)', {
                sliceK: k, sliceMin, sliceMax,
                first5: Array.from(sliceFloat.subarray(0, 5)),
            });
        }
    }

    _cachedVolumeId = volumeId ?? null;
    console.log(`[CPR-GPU] Volume texture uploaded: ${nx}×${ny}×${nz} (R32F, raw values)`);
}

// ─── Spline Data Texture ────────────────────────────────────────────
function uploadSplineTexture(
    gl: WebGL2RenderingContext,
    frames: GpuPanoInput['frames'],
    panoWidth: number
): void {
    if (_splineTex) {
        gl.deleteTexture(_splineTex);
    }

    // Pack positions (row 0) and slab normals (row 1) into RGBA32F texture
    // Width = panoWidth, Height = 2
    const data = new Float32Array(panoWidth * 2 * 4); // RGBA × width × 2 rows

    // Flip-correct slab normals (same logic as CPU worker)
    let prevSlabDir: [number, number, number] | null = null;
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
    }

    _splineTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, _splineTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, panoWidth, 2, 0, gl.RGBA, gl.FLOAT, data);
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

// ─── Main Render Function ───────────────────────────────────────────
export function renderPanoGpu(input: GpuPanoInput, volumeId?: string): GpuPanoResult {
    const gl = ensureGpuContext();
    const {
        scalarData, dimensions, spacing, origin, direction, worldToIndex,
        frames, panoWidth, panoHeight,
        verticalDir, vertHalfMm, verticalCenterOffsetMm,
        slabHalfThicknessMm, slabSamples, gaussSigma,
        rescaleSlope, rescaleIntercept, applyRescale,
        normalizeStoredSample,
    } = input;
    const t0 = performance.now();

    // 1. Upload volume texture with RAW values (cached by volumeId)
    uploadVolumeTexture(gl, scalarData, dimensions, volumeId);

    // 2. Upload spline data
    uploadSplineTexture(gl, frames, panoWidth);

    // 3. Ensure FBO at correct size
    ensureFbo(gl, panoWidth, panoHeight);

    // 4. Compute worldToIndex mat4
    let w2iMat: Float32Array;
    if (worldToIndex && worldToIndex.length >= 16) {
        // VTK.js mat4 is column-major
        w2iMat = new Float32Array(worldToIndex);
    } else {
        w2iMat = buildWorldToIndexMat4(origin, spacing, direction);
    }

    // 5. Render
    gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
    gl.viewport(0, 0, panoWidth, panoHeight);
    gl.useProgram(_program);

    // Bind volume texture to unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, _volumeTex);
    gl.uniform1i(_uVolume, 0);

    // Bind spline data texture to unit 1
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, _splineTex);
    gl.uniform1i(_uSplineData, 1);

    // Set uniforms
    gl.uniform3f(_uVerticalDir, verticalDir[0], verticalDir[1], verticalDir[2]);
    gl.uniform1f(_uVertHalfMm, vertHalfMm);
    gl.uniform1f(_uVertCenterOffsetMm, verticalCenterOffsetMm);
    gl.uniform1f(_uSlabHalfMm, slabHalfThicknessMm);
    gl.uniform1i(_uSlabSamples, slabSamples);
    gl.uniform1f(_uGaussSigma, gaussSigma);
    gl.uniform1i(_uPanoWidth, panoWidth);
    gl.uniform1i(_uPanoHeight, panoHeight);
    gl.uniformMatrix4fv(_uWorldToIndex, false, w2iMat);
    gl.uniform3f(_uDims, dimensions[0], dimensions[1], dimensions[2]);
    gl.uniform1f(_uRescaleSlope, rescaleSlope);
    gl.uniform1f(_uRescaleIntercept, rescaleIntercept);
    gl.uniform1i(_uApplyRescale, applyRescale ? 1 : 0);

    // Draw fullscreen triangle
    gl.bindVertexArray(_vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    // 6. Read back pixels
    const rgba = new Float32Array(panoWidth * panoHeight * 4);
    gl.readPixels(0, 0, panoWidth, panoHeight, gl.RGBA, gl.FLOAT, rgba);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Extract R channel into output Float32Array
    const pixelData = new Float32Array(panoWidth * panoHeight);
    for (let i = 0; i < panoWidth * panoHeight; i++) {
        pixelData[i] = rgba[i * 4];
    }

    // Post-process in HU domain to reduce speckle while keeping tooth edges defined.
    applyLightBilateralDenoise(pixelData, panoWidth, panoHeight, 0.5, 280, 2);
    applyMildUnsharpMask(pixelData, panoWidth, panoHeight, 0.16, 42);
    const { minValue, maxValue } = computeMinMax(pixelData);

    const elapsed = performance.now() - t0;
    console.log(`[CPR-GPU] Render complete: ${panoWidth}x${panoHeight} in ${elapsed.toFixed(1)}ms`, {
        minValue,
        maxValue,
    });

    return { pixelData, width: panoWidth, height: panoHeight, minValue, maxValue };
}

// ─── GPU Availability Check ─────────────────────────────────────────
export function isGpuPanoAvailable(): boolean {
    try {
        const testCanvas = new OffscreenCanvas(1, 1);
        const gl = testCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
        if (!gl) return false;
        const ext = gl.getExtension('EXT_color_buffer_float');
        // OES_texture_float_linear needed for R32F + LINEAR trilinear filtering
        const floatLinear = gl.getExtension('OES_texture_float_linear');
        return !!(ext && floatLinear);
    } catch {
        try {
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
    if (_gl) {
        if (_volumeTex) _gl.deleteTexture(_volumeTex);
        if (_splineTex) _gl.deleteTexture(_splineTex);
        if (_fboTex) _gl.deleteTexture(_fboTex);
        if (_fbo) _gl.deleteFramebuffer(_fbo);
        if (_program) _gl.deleteProgram(_program);
        if (_vao) _gl.deleteVertexArray(_vao);
    }
    _volumeTex = null;
    _splineTex = null;
    _fboTex = null;
    _fbo = null;
    _program = null;
    _vao = null;
    _gl = null;
    _canvas = null;
    _cachedVolumeId = null;
    _fboWidth = 0;
    _fboHeight = 0;
}


