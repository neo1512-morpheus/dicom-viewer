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
}

export interface GpuPanoResult {
    pixelData: Float32Array;
    width: number;
    height: number;
    minValue: number;
    maxValue: number;
}

const GPU_DEBUG_MODE_OFF = 0;
const GPU_DEBUG_MODE_RAY_START = 1;
const GPU_DEBUG_MODE_RAY_DIRECTION = 2;
const GPU_DEBUG_MODE_SPLINE_VECTOR = 3;
const ACTIVE_GPU_DEBUG_MODE = GPU_DEBUG_MODE_OFF;

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

// Rescale: applied AFTER trilinear sampling (matches Cornerstone GPU pipeline)
uniform float uRescaleSlope;
uniform float uRescaleIntercept;
uniform bool uApplyRescale;
uniform int uDebugMode;

out vec4 fragColor;

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

void main() {
  int col = int(gl_FragCoord.x);
  int outputRow = int(gl_FragCoord.y);
  if (col >= uPanoWidth || outputRow >= uPanoHeight) {
    fragColor = vec4(-1000.0, 0.0, 0.0, 1.0);
    return;
  }

  int row = (uPanoHeight - 1) - outputRow;
  vec3 pos = texelFetch(uSplineData, ivec2(col, 0), 0).xyz;
  vec3 slabDir = texelFetch(uSplineData, ivec2(col, 1), 0).xyz;
  vec3 verticalDir = texelFetch(uSplineData, ivec2(col, 2), 0).xyz;

  float slabDirLen = length(slabDir);
  if (slabDirLen > 1e-5) {
    slabDir /= slabDirLen;
  } else {
    slabDir = vec3(0.0, 0.0, 1.0);
  }

  float verticalDirLen = length(verticalDir);
  if (verticalDirLen > 1e-5) {
    verticalDir /= verticalDirLen;
  } else {
    verticalDir = normalize(uVerticalDir);
  }

  float panoHeightDen = max(1.0, float(uPanoHeight - 1));
  float vertStepMm = (uVertHalfMm * 2.0) / panoHeightDen;
  float vertOffsetMm = uVertCenterOffsetMm + (uVertHalfMm - float(row) * vertStepMm);
  vec3 posIndex = (uWorldToIndex * vec4(pos, 1.0)).xyz;
  vec3 slabDirIndexPerMm = (uWorldToIndex * vec4(slabDir, 0.0)).xyz;
  vec3 verticalDirIndexPerMm = (uWorldToIndex * vec4(verticalDir, 0.0)).xyz;
  vec3 baseIndex = posIndex + vertOffsetMm * verticalDirIndexPerMm;

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
    vec3 rawSpline = texelFetch(uSplineData, ivec2(col, splineRow), 0).xyz;
    vec3 debugColor = splineRow == 0
      ? encodeIndexCoord((uWorldToIndex * vec4(rawSpline, 1.0)).xyz)
      : encodeDirection(rawSpline);
    fragColor = vec4(debugColor, 1.0);
    return;
  }

  float slabStep = uSlabSamples > 1
    ? (uSlabHalfMm * 2.0) / float(uSlabSamples - 1)
    : 0.0;

  vec3 clampMin = 0.5 / uDims;
  vec3 clampMax = vec3(1.0) - 0.5 / uDims;

  float finalHu = -1000.0;
  float sumHu = 0.0;
  float totalWeight = 0.0;
  const int MAX_SLAB = 64;
  const float TROUGH_SIGMA_MM = 1.0;
  const float TROUGH_DENOM = 2.0 * TROUGH_SIGMA_MM * TROUGH_SIGMA_MM;

  for (int s = 0; s < MAX_SLAB; s++) {
    if (s >= uSlabSamples) {
      break;
    }

    float slabOffset = uSlabSamples > 1
      ? -uSlabHalfMm + float(s) * slabStep
      : 0.0;

    vec3 sampleIndex = baseIndex + slabOffset * slabDirIndexPerMm;
    vec3 uvw = (sampleIndex + vec3(0.5)) / uDims;

    if (any(lessThan(uvw, clampMin)) || any(greaterThan(uvw, clampMax))) {
      continue;
    }

    float rawVal = texture(uVolume, uvw).r;
    float hu = uApplyRescale ? rawVal * uRescaleSlope + uRescaleIntercept : rawVal;
    hu = clamp(hu, -3000.0, 10000.0);
    float troughWeight = exp(-(slabOffset * slabOffset) / TROUGH_DENOM);
    sumHu += hu * troughWeight;
    totalWeight += troughWeight;
  }

  if (totalWeight > 1e-5) {
    finalHu = sumHu / totalWeight;
  } else {
    finalHu = -1000.0;
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
let _cachedVolumeNormalizationSignature: string | null = null;
let _hasFloatLinearFiltering = false;

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
let _uPanoWidth = -1;
let _uPanoHeight = -1;
let _uWorldToIndex = -1;
let _uDims = -1;
let _uRescaleSlope = -1;
let _uRescaleIntercept = -1;
let _uApplyRescale = -1;
let _uDebugMode = -1;

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

function resetGpuRendererState(): void {
    if (_gl) {
        try {
            if (_volumeTex) _gl.deleteTexture(_volumeTex);
            if (_splineTex) _gl.deleteTexture(_splineTex);
            if (_fboTex) _gl.deleteTexture(_fboTex);
            if (_fbo) _gl.deleteFramebuffer(_fbo);
            if (_program) _gl.deleteProgram(_program);
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
    _fboTex = null;
    _fbo = null;
    _program = null;
    _vao = null;
    _gl = null;
    _canvas = null;
    _cachedVolumeId = null;
    _cachedVolumeNormalizationSignature = null;
    _hasFloatLinearFiltering = false;
    _fboWidth = 0;
    _fboHeight = 0;
    _uVolume = -1;
    _uSplineData = -1;
    _uVerticalDir = -1;
    _uVertHalfMm = -1;
    _uVertCenterOffsetMm = -1;
    _uSlabHalfMm = -1;
    _uSlabSamples = -1;
    _uPanoWidth = -1;
    _uPanoHeight = -1;
    _uWorldToIndex = -1;
    _uDims = -1;
    _uRescaleSlope = -1;
    _uRescaleIntercept = -1;
    _uApplyRescale = -1;
    _uDebugMode = -1;
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
    _uPanoWidth = _gl.getUniformLocation(_program, 'uPanoWidth') as number;
    _uPanoHeight = _gl.getUniformLocation(_program, 'uPanoHeight') as number;
    _uWorldToIndex = _gl.getUniformLocation(_program, 'uWorldToIndex') as number;
    _uDims = _gl.getUniformLocation(_program, 'uDims') as number;
    _uRescaleSlope = _gl.getUniformLocation(_program, 'uRescaleSlope') as number;
    _uRescaleIntercept = _gl.getUniformLocation(_program, 'uRescaleIntercept') as number;
    _uApplyRescale = _gl.getUniformLocation(_program, 'uApplyRescale') as number;
    _uDebugMode = _gl.getUniformLocation(_program, 'uDebugMode') as number;

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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
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

// ─── Main Render Function ───────────────────────────────────────────
export function renderPanoGpu(input: GpuPanoInput, volumeId?: string): GpuPanoResult {
    const gl = ensureGpuContext();
    const {
        scalarData, dimensions, spacing, origin, direction, worldToIndex,
        frames, panoWidth, panoHeight,
        verticalDir, vertHalfMm, verticalCenterOffsetMm,
        slabHalfThicknessMm, slabSamples,
        rescaleSlope, rescaleIntercept, applyRescale,
        normalizationSignature,
        normalizeStoredSample,
    } = input;
    const safePanoWidth = Math.max(1, Math.floor(Number(panoWidth) || 1));
    const safePanoHeight = Math.max(1, Math.floor(Number(panoHeight) || 1));
    const t0 = performance.now();

    // 1. Upload volume texture with RAW values (cached by volumeId)
    uploadVolumeTexture(
        gl,
        scalarData,
        dimensions,
        normalizationSignature,
        normalizeStoredSample,
        volumeId
    );

    // 2. Upload spline data
    const splineDebugVertices = uploadSplineTexture(gl, frames, safePanoWidth, verticalDir);

    // 3. Ensure FBO at correct size
    ensureFbo(gl, safePanoWidth, safePanoHeight);

    // 4. Compute worldToIndex mat4
    const w2iMat = buildWorldToIndexMat4(origin, spacing, direction);

    // 5. Render
    gl.bindFramebuffer(gl.FRAMEBUFFER, _fbo);
    gl.viewport(0, 0, safePanoWidth, safePanoHeight);
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
    const effectiveSlabSamples = Math.max(1, Math.min(64, slabSamples | 0));
    gl.uniform1i(_uSlabSamples, effectiveSlabSamples);
    gl.uniform1i(_uPanoWidth, safePanoWidth);
    gl.uniform1i(_uPanoHeight, safePanoHeight);
    gl.uniformMatrix4fv(_uWorldToIndex, false, w2iMat);
    gl.uniform3f(_uDims, dimensions[0], dimensions[1], dimensions[2]);
    gl.uniform1f(_uRescaleSlope, rescaleSlope);
    gl.uniform1f(_uRescaleIntercept, rescaleIntercept);
    gl.uniform1i(_uApplyRescale, applyRescale ? 1 : 0);
    gl.uniform1i(_uDebugMode, ACTIVE_GPU_DEBUG_MODE);

    const splineFlattenedSample: number[] = [];
    for (let i = 0; i < Math.min(2, splineDebugVertices.length); i++) {
        const vertex = splineDebugVertices[i];
        splineFlattenedSample.push(
            vertex.position[0],
            vertex.position[1],
            vertex.position[2],
            0,
            vertex.slabDir[0],
            vertex.slabDir[1],
            vertex.slabDir[2],
            0,
            vertex.verticalDir[0],
            vertex.verticalDir[1],
            vertex.verticalDir[2],
            0
        );
    }

    console.error(
        `[FINAL-FAILSAFE-LOGS]\n${JSON.stringify(
            {
                debugMode: ACTIVE_GPU_DEBUG_MODE,
                sentWorldToIndex: Array.from(w2iMat),
                incomingWorldToIndex: worldToIndex ? Array.from(worldToIndex) : null,
                uDims: [dimensions[0], dimensions[1], dimensions[2]],
                spacing: [spacing[0], spacing[1], spacing[2]],
                origin: [origin[0], origin[1], origin[2]],
                panoWidth: safePanoWidth,
                panoHeight: safePanoHeight,
                slabHalfThicknessMm,
                slabSamples: effectiveSlabSamples,
                splineFirstTwoIndicesFlattened: splineFlattenedSample,
            },
            null,
            2
        )}`
    );

    // Draw fullscreen triangle
    gl.bindVertexArray(_vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    // 6. Read back pixels
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    const rgbaBuffer = new Float32Array(safePanoWidth * safePanoHeight * 4);
    gl.readPixels(0, 0, safePanoWidth, safePanoHeight, gl.RGBA, gl.FLOAT, rgbaBuffer);
    const readbackError = gl.getError();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (readbackError !== gl.NO_ERROR) {
        throw new Error(`GPU pano readPixels failed with WebGL error ${readbackError}.`);
    }

    // Flip rows to match Cornerstone's top-to-bottom pixel layout.
    const pixelData = new Float32Array(safePanoWidth * safePanoHeight);
    for (let row = 0; row < safePanoHeight; row++) {
        const srcRow = safePanoHeight - 1 - row;
        for (let col = 0; col < safePanoWidth; col++) {
            const dstIndex = row * safePanoWidth + col;
            const srcIndex = (srcRow * safePanoWidth + col) * 4;
            pixelData[dstIndex] = rgbaBuffer[srcIndex];
        }
    }

    console.log('[GPU-READ-ERROR]', readbackError);
    console.log('[GPU-RAW-PIXELS-SAMPLE]', Array.from(pixelData.slice(0, 10)));
    let rawMin = Infinity;
    let rawMax = -Infinity;
    for (let i = 0; i < pixelData.length; i++) {
        const value = pixelData[i];
        if (!Number.isFinite(value)) {
            continue;
        }
        if (value < rawMin) rawMin = value;
        if (value > rawMax) rawMax = value;
    }
    console.log('[GPU-RAW-MINMAX]', {
        min: Number.isFinite(rawMin) ? rawMin : null,
        max: Number.isFinite(rawMax) ? rawMax : null,
    });

    // No post-processing: strict GPU MIP result is read back directly.
    const { minValue, maxValue } = computeMinMax(pixelData);

    const elapsed = performance.now() - t0;
    console.log(`[CPR-GPU] Render complete: ${safePanoWidth}x${safePanoHeight} in ${elapsed.toFixed(1)}ms`, {
        minValue,
        maxValue,
    });

    return { pixelData, width: safePanoWidth, height: safePanoHeight, minValue, maxValue };
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


