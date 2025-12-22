/**
 * GPU Capability Detection Utility
 * Detects weak GPUs and stores result for use throughout the application
 */

export interface GPUCapability {
    isWeakGPU: boolean;
    renderer: string;
    vendor: string;
}

// Known weak GPU patterns
const WEAK_GPU_PATTERNS = [
    'Intel(R) HD Graphics',
    'Intel(R) UHD Graphics',
    'Intel HD Graphics',
    'Intel UHD Graphics',
    'SwiftShader',
    'llvmpipe',
    'Microsoft Basic Render',
    'Software Rasterizer',
];

// Known good integrated GPUs (modern, capable of volume rendering)
const GOOD_INTEGRATED_PATTERNS = [
    'Intel Iris',
    'Intel(R) Iris',
    'Apple M1',
    'Apple M2',
    'Apple M3',
];

let cachedResult: GPUCapability | null = null;

/**
 * Detects GPU capability by checking WebGL renderer info
 */
export function detectGPUCapability(): GPUCapability {
    // Return cached result if already detected
    if (cachedResult) {
        return cachedResult;
    }

    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

        if (!gl) {
            console.warn('[GPU Check] No WebGL support detected');
            cachedResult = {
                isWeakGPU: true,
                renderer: 'No WebGL',
                vendor: 'Unknown',
            };
            return cachedResult;
        }

        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const renderer = debugInfo
            ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
            : 'Unknown';
        const vendor = debugInfo
            ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
            : 'Unknown';

        // Check if this is a known good integrated GPU first
        const isGoodIntegrated = GOOD_INTEGRATED_PATTERNS.some(pattern =>
            renderer.includes(pattern)
        );

        // Check if this is a weak GPU
        const isWeakGPU = !isGoodIntegrated && WEAK_GPU_PATTERNS.some(pattern =>
            renderer.includes(pattern)
        );

        cachedResult = {
            isWeakGPU,
            renderer,
            vendor,
        };

        console.log(`[GPU Check] Detected: ${renderer} (${vendor}) - ${isWeakGPU ? 'WEAK' : 'CAPABLE'}`);

        return cachedResult;
    } catch (error) {
        console.error('[GPU Check] Error detecting GPU:', error);
        cachedResult = {
            isWeakGPU: true,
            renderer: 'Detection Failed',
            vendor: 'Unknown',
        };
        return cachedResult;
    }
}

/**
 * Quick check if GPU is weak (uses cached result)
 */
export function isWeakGPU(): boolean {
    return detectGPUCapability().isWeakGPU;
}
