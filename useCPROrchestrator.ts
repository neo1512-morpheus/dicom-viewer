/**
 * useCPROrchestrator.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * React hook that wires the full CPR pipeline together.
 *
 * SEQUENCE OVERVIEW
 * ─────────────────
 *
 *  User clicks "Done"
 *       │
 *       ▼
 *  1. Extract SplineROI annotation → raw world-space control points
 *       │
 *       ▼
 *  2. Build Catmull-Rom spline → arc-length LUT → equidistant positions + tangents
 *       │
 *       ▼
 *  3. Build RMF frames (Bishop) → N_camera, N_slab, S per frame
 *       │
 *       ▼
 *  4. cprStateService.setArchData(controlPoints, frames, volumeId)
 *       │
 *       ◄── SAFE POINT: data is now in singleton, survives HP switch ──►
 *       │
 *       ▼
 *  5. commandsManager.runCommand('setHangingProtocol', { protocolId: 'cpr', stageIndex: 1 })
 *       │   ← viewports are destroyed and rebuilt here ─────────────────────────
 *       │
 *       ▼
 *  6. Wait for HANGING_PROTOCOL_APPLIED event
 *       │
 *       ▼
 *  7. Spin up cprWorker with volume scalar data (SAB zero-copy or clone fallback)
 *       │
 *       ▼
 *  8. Worker returns pixelData
 *       │
 *       ▼
 *  9. setPanoImagePayload(PANO_IMAGE_ID, payload)
 *       │
 *       ▼
 * 10. panoViewport.setStack([PANO_IMAGE_ID]) → triggers panoImageLoader
 *       │
 *       ▼
 * 11. Register CPR_CROSSSECTION_SYNC + initialize cross-section camera at frame[0]
 *
 * USAGE
 * ─────
 * Mount this hook in the CPR mode's main layout component, or in a dedicated
 * "CPR Toolbar" component that renders the "Done" button.
 *
 *   const { onDone, isGenerating, error } = useCPROrchestrator({ servicesManager, commandsManager });
 *   <button onClick={onDone} disabled={isGenerating}>
 *     {isGenerating ? 'Generating...' : 'Done'}
 *   </button>
 */

import { useCallback, useRef, useState } from 'react';
import * as cornerstoneTools from '@cornerstonejs/tools';
import * as cornerstone from '@cornerstonejs/core';
import { cprStateService } from './CPRStateService';
import { setPanoImagePayload, PANO_IMAGE_ID, clearPanoImageCache } from './panoImageLoader';
import { buildRMFFrames } from './cprMath';
import type { CPRFrame } from './cprMath';

// ─────────────────────────────────────────────────────────────────────────────
// CATMULL-ROM SPLINE WITH ARC-LENGTH REPARAMETERIZATION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Evaluates a single Catmull-Rom segment at parameter t ∈ [0,1].
 * Uses the standard alpha=0.5 (centripetal) formulation.
 * P0, P1 = segment endpoints; Pm1, P2 = neighboring control points.
 */
function catmullRomPoint(
  Pm1: [number, number, number],
  P0:  [number, number, number],
  P1:  [number, number, number],
  P2:  [number, number, number],
  t: number
): [number, number, number] {
  const t2 = t * t;
  const t3 = t2 * t;

  // Standard Catmull-Rom matrix coefficients
  const b0 = -0.5 * t3 + t2 - 0.5 * t;
  const b1 =  1.5 * t3 - 2.5 * t2 + 1.0;
  const b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  const b3 =  0.5 * t3 - 0.5 * t2;

  return [
    b0 * Pm1[0] + b1 * P0[0] + b2 * P1[0] + b3 * P2[0],
    b0 * Pm1[1] + b1 * P0[1] + b2 * P1[1] + b3 * P2[1],
    b0 * Pm1[2] + b1 * P0[2] + b2 * P1[2] + b3 * P2[2],
  ];
}

/**
 * Evaluates the tangent of a Catmull-Rom segment at parameter t.
 * Returns a non-normalized direction vector.
 */
function catmullRomTangent(
  Pm1: [number, number, number],
  P0:  [number, number, number],
  P1:  [number, number, number],
  P2:  [number, number, number],
  t: number
): [number, number, number] {
  const t2 = t * t;

  const d0 = -1.5 * t2 + 2.0 * t - 0.5;
  const d1 =  4.5 * t2 - 5.0 * t;
  const d2 = -4.5 * t2 + 4.0 * t + 0.5;
  const d3 =  1.5 * t2 - 1.0 * t;

  return [
    d0 * Pm1[0] + d1 * P0[0] + d2 * P1[0] + d3 * P2[0],
    d0 * Pm1[1] + d1 * P0[1] + d2 * P1[1] + d3 * P2[1],
    d0 * Pm1[2] + d1 * P0[2] + d2 * P1[2] + d3 * P2[2],
  ];
}

/** Euclidean distance between two 3D points */
function dist3(a: [number, number, number], b: [number, number, number]): number {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Normalizes a 3D vector in-place, returns it */
function normalize3(v: [number, number, number]): [number, number, number] {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len < 1e-10) return [1, 0, 0]; // degenerate fallback
  return [v[0] / len, v[1] / len, v[2] / len];
}

/**
 * Adds phantom endpoints to a control-point array so Catmull-Rom evaluates
 * correctly at the very first and very last user-drawn points.
 *
 * Without phantoms, the spline either clamps (losing molar endpoints) or
 * throws an index error. This is the standard fix.
 */
function addPhantomEndpoints(
  pts: [number, number, number][]
): [number, number, number][] {
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

/**
 * Builds an arc-length parameterized spline.
 *
 * 1. Subdivides the Catmull-Rom curve into FINE_STEPS points to build an LUT
 * 2. Binary-searches the LUT to find equidistant sample positions
 * 3. Returns arrays of positions and tangents at N_SAMPLES equidistant points
 *
 * @param rawPoints   - User control points in world space (min 2 required)
 * @param nSamples    - Number of equidistant output samples (= panoWidth)
 * @returns { positions, tangents } both of length nSamples
 */
function buildArcLengthSpline(
  rawPoints: [number, number, number][],
  nSamples: number
): {
  positions: [number, number, number][];
  tangents:  [number, number, number][];
} {
  if (rawPoints.length < 2) {
    throw new Error('[buildArcLengthSpline] Need at least 2 control points.');
  }

  const extended = addPhantomEndpoints(rawPoints);
  const nSegments = extended.length - 3; // number of Catmull-Rom segments

  // ── Step 1: Build a fine arc-length LUT ──────────────────────────────────
  const FINE_STEPS = 2000; // more steps = more accurate equidistance
  const lutT: number[] = [0];       // parameter t along the full curve [0…nSegments]
  const lutArc: number[] = [0];     // cumulative arc length at each fine point

  let prevPt = catmullRomPoint(
    extended[0], extended[1], extended[2], extended[3], 0
  );

  for (let seg = 0; seg < nSegments; seg++) {
    const Pm1 = extended[seg];
    const P0  = extended[seg + 1];
    const P1  = extended[seg + 2];
    const P2  = extended[seg + 3];

    for (let step = 1; step <= FINE_STEPS / nSegments; step++) {
      const localT = step / (FINE_STEPS / nSegments);
      const globalT = seg + localT;
      const pt = catmullRomPoint(Pm1, P0, P1, P2, localT);

      const arcLen = lutArc[lutArc.length - 1] + dist3(prevPt, pt);
      lutT.push(globalT);
      lutArc.push(arcLen);
      prevPt = pt;
    }
  }

  const totalArcLength = lutArc[lutArc.length - 1];

  // ── Step 2: Sample at equidistant arc-length intervals ───────────────────
  const positions: [number, number, number][] = [];
  const tangents:  [number, number, number][] = [];

  for (let i = 0; i < nSamples; i++) {
    const targetArc = (i / (nSamples - 1)) * totalArcLength;

    // Binary search in lutArc for targetArc
    let lo = 0, hi = lutArc.length - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (lutArc[mid] < targetArc) lo = mid; else hi = mid;
    }

    // Interpolate globalT between lo and hi
    const arcRange = lutArc[hi] - lutArc[lo];
    const frac = arcRange < 1e-10 ? 0 : (targetArc - lutArc[lo]) / arcRange;
    const globalT = lutT[lo] + frac * (lutT[hi] - lutT[lo]);

    // Map globalT back to segment index and local t
    const seg = Math.min(Math.floor(globalT), nSegments - 1);
    const localT = globalT - seg;

    const Pm1 = extended[seg];
    const P0  = extended[seg + 1];
    const P1  = extended[seg + 2];
    const P2  = extended[seg + 3];

    positions.push(catmullRomPoint(Pm1, P0, P1, P2, localT));
    tangents.push(normalize3(catmullRomTangent(Pm1, P0, P1, P2, localT)));
  }

  return { positions, tangents };
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER LAUNCH HELPER
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Launches the CPR Web Worker and returns a Promise that resolves
 * with the panoramic pixel data.
 *
 * SHARED ARRAY BUFFER STRATEGY
 * ────────────────────────────
 * The codebase investigation confirmed that COOP/COEP headers are set in
 * _headers, netlify.toml, and serve.json. SharedArrayBuffer will be available
 * in all deployment targets.
 *
 * We still check at runtime and clone as a safe fallback — defensive coding.
 * We NEVER use the transfer list for regular ArrayBuffers (would detach the
 * Cornerstone3D volume cache and corrupt MPR viewports).
 */
function launchCPRWorker(params: {
  volume: cornerstone.Types.IImageVolume;
  frames: CPRFrame[];
  panoWidth: number;
  panoHeight: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  aggregation: 'MIP' | 'MEAN';
}): Promise<{ pixelData: Float32Array; width: number; height: number; minValue: number; maxValue: number }> {
  return new Promise((resolve, reject) => {
    // Inline worker via blob URL — avoids Webpack/Vite worker config requirements.
    // Replace with new Worker(new URL('./cprWorker.ts', import.meta.url)) if your
    // bundler is configured to handle worker imports.
    const worker = new Worker(new URL('./cprWorker.ts', import.meta.url), { type: 'module' });

    const { volume, frames, panoWidth, panoHeight, slabHalfThicknessMm, slabSamples, aggregation } = params;

    // Extract raw scalar data from the cached Cornerstone3D volume
    const scalarData = volume.imageData.getPointData().getScalars().getData() as Float32Array | Int16Array;
    const isSharedArrayBuffer = scalarData.buffer instanceof SharedArrayBuffer;

    // For non-SAB buffers: clone to avoid detaching the volume cache.
    // For SAB buffers: pass the typed array directly — zero copy, zero risk.
    const dataToSend = isSharedArrayBuffer
      ? scalarData
      : (scalarData.slice(0) as Float32Array | Int16Array);

    // Serialize frames to plain objects (typed arrays can't cross worker boundary
    // unless transferred — we keep them as plain number arrays for simplicity)
    const serializedFrames = frames.map(f => ({
      position: Array.from(f.position) as [number, number, number],
      N_slab:   Array.from(f.N_slab)   as [number, number, number],
    }));

    worker.onmessage = (event) => {
      worker.terminate();
      const data = event.data;

      if (data.type === 'ERROR') {
        reject(new Error(`[cprWorker] ${data.message}`));
        return;
      }

      resolve({
        pixelData: data.pixelData,
        width: data.panoWidth,
        height: data.panoHeight,
        minValue: data.minValue,
        maxValue: data.maxValue,
      });
    };

    worker.onerror = (err) => {
      worker.terminate();
      reject(new Error(`[cprWorker] Uncaught worker error: ${err.message}`));
    };

    // Post message — empty transfer list in all cases.
    // SAB: data is already shared, no transfer needed.
    // Clone: we own the clone, but keeping it un-transferred lets the worker
    // reference it safely without main-thread interference.
    worker.postMessage({
      scalarData: dataToSend,
      isSharedArrayBuffer,
      dimensions: volume.imageData.getDimensions(),
      spacing: volume.imageData.getSpacing(),
      origin: volume.imageData.getOrigin(),
      direction: volume.imageData.getDirection(),
      frames: serializedFrames,
      panoWidth,
      panoHeight,
      slabHalfThicknessMm,
      slabSamples,
      aggregation,
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// CROSS-SECTION SYNCHRONIZER SETUP
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Initializes the cross-section viewport camera at frame index 0
 * and registers the CPR_CROSSSECTION_SYNC group.
 * Called after Stage 2 HP is applied and viewports are ready.
 */
function initializeCrossSection(
  frames: CPRFrame[],
  servicesManager: AppTypes.ServicesManager
): void {
  const { cornerstoneViewportService, syncGroupService } = servicesManager.services;

  const crossViewport = cornerstoneViewportService.getCornerstoneViewport('cpr-crosssection');
  if (!crossViewport) {
    console.error('[CPR] cpr-crosssection viewport not found after HP switch.');
    return;
  }

  // Set initial camera to first frame
  const firstFrame = frames[0];
  crossViewport.setCamera({
    focalPoint:      Array.from(firstFrame.position) as [number, number, number],
    viewPlaneNormal: Array.from(firstFrame.N_camera) as [number, number, number],
    viewUp:          Array.from(firstFrame.S)        as [number, number, number],
    parallelScale: 20,
    parallelProjection: true,
  });
  crossViewport.render();

  // Add cross-section viewport to the CPR sync group
  // Sync type is unique — confirmed no collision with 'voi', 'imageSlice', 'camera'
  syncGroupService.addViewportToSyncGroup(
    'cpr-crosssection',
    crossViewport.getRenderingEngine().id,
    {
      type: 'CPR_CROSSSECTION_SYNC',
      id: 'cpr-crosssection-sync',
      source: false,
      target: true,
    }
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN HOOK
// ─────────────────────────────────────────────────────────────────────────────

interface UseCPROrchestratorProps {
  servicesManager: AppTypes.ServicesManager;
  commandsManager: AppTypes.CommandsManager;
  /** Number of columns in the panoramic output. Default: 800 */
  panoWidth?: number;
  /** Number of rows in the panoramic output. Default: 400 */
  panoHeight?: number;
  /** Slab half-thickness in mm. Default: 7mm = 14mm total slab */
  slabHalfThicknessMm?: number;
  /** Number of samples across the slab. Odd number preferred. Default: 21 */
  slabSamples?: number;
  /** Initial aggregation mode. Can be toggled by toolbar button later. */
  aggregation?: 'MIP' | 'MEAN';
}

interface UseCPROrchestratorReturn {
  /** Call this when the user clicks "Done" after drawing the arch */
  onDone: () => Promise<void>;
  /** Call this when the slider moves to index i */
  onSliderChange: (frameIndex: number) => void;
  isGenerating: boolean;
  error: string | null;
}

export function useCPROrchestrator({
  servicesManager,
  commandsManager,
  panoWidth = 800,
  panoHeight = 400,
  slabHalfThicknessMm = 7,
  slabSamples = 21,
  aggregation = 'MIP',
}: UseCPROrchestratorProps): UseCPROrchestratorReturn {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hpListenerRef = useRef<(() => void) | null>(null);

  // ── onDone: The full CPR trigger chain ──────────────────────────────────
  const onDone = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const {
        cornerstoneViewportService,
        hangingProtocolService,
      } = servicesManager.services;

      // ── STEP 1: Get the axial viewport element ───────────────────────────
      const axialViewport = cornerstoneViewportService.getCornerstoneViewport('cpr-axial');
      if (!axialViewport) throw new Error('cpr-axial viewport not found.');

      const axialElement = axialViewport.element;

      // ── STEP 2: Extract SplineROI annotation from the axial viewport ─────
      // getAnnotations returns ALL annotations for the tool on this element.
      // We want the most recently completed one.
      const annotations = cornerstoneTools.annotation.state.getAnnotations(
        'SplineROI',
        axialElement
      );

      if (!annotations || annotations.length === 0) {
        throw new Error('No SplineROI annotation found. Please draw the jaw arch first.');
      }

      // Most recently drawn annotation is last in the array
      const latestAnnotation = annotations[annotations.length - 1];

      // SplineROITool stores handles as an array of world-space points
      // Shape: annotation.data.handles.points = Vec3[]
      const rawControlPoints: [number, number, number][] =
        latestAnnotation.data.handles.points.map(
          (p: { x: number; y: number; z: number }) => [p.x, p.y, p.z]
        );

      if (rawControlPoints.length < 2) {
        throw new Error('Arch annotation needs at least 2 points.');
      }

      // ── STEP 3: Get the source volume ID ────────────────────────────────
      // The axial viewport is a VolumeViewport — get the first loaded volume
      const volumeActors = (axialViewport as cornerstone.Types.IVolumeViewport).getActors();
      if (!volumeActors || volumeActors.length === 0) {
        throw new Error('No volume loaded in cpr-axial viewport.');
      }
      const sourceVolumeId = volumeActors[0].uid;

      // ── STEP 4: Build spline + RMF frames ───────────────────────────────
      // This runs synchronously on the main thread and is fast (~1–5ms).
      // The heavy work is in the Worker (Step 7).
      const { positions, tangents } = buildArcLengthSpline(rawControlPoints, panoWidth);
      const frames = buildRMFFrames(positions, tangents);

      // ── STEP 5: Persist arch data BEFORE the HP switch ─────────────────
      // This is the critical state-preservation step. The HP switch below
      // will destroy the axial viewport and its tool annotations.
      // After this call, all CPR data lives in the singleton and is safe.
      cprStateService.setArchData(rawControlPoints, frames, sourceVolumeId);

      // ── STEP 6: Switch to Stage 2 (CPR Result — 1×2 layout) ─────────────
      // This destroys the current viewports. The CPRStateService singleton
      // is unaffected. The Worker (Step 7) reads from the singleton.

      // Register a one-time listener for PROTOCOL_APPLIED before switching.
      // This fires after Cornerstone3D has finished creating the new viewports.
      const onProtocolApplied = async () => {
        // Clean up listener immediately — only run once
        hangingProtocolService.unsubscribe(
          hangingProtocolService.EVENTS.PROTOCOL_APPLIED,
          hpListenerRef.current!
        );
        hpListenerRef.current = null;

        try {
          // ── STEP 7: Get volume from Cornerstone cache ──────────────────
          const volume = cornerstone.cache.getVolume(sourceVolumeId);
          if (!volume) throw new Error(`Volume ${sourceVolumeId} not found in cache.`);

          // ── STEP 8: Launch the Web Worker ─────────────────────────────
          // Main thread is free during this step. UI remains interactive.
          const result = await launchCPRWorker({
            volume,
            frames,
            panoWidth,
            panoHeight,
            slabHalfThicknessMm,
            slabSamples,
            aggregation,
          });

          // ── STEP 9: Cache the panoramic image for the image loader ─────
          clearPanoImageCache(); // clear any previous session's pano
          setPanoImagePayload(PANO_IMAGE_ID, {
            pixelData: result.pixelData,
            width: result.width,
            height: result.height,
            minValue: result.minValue,
            maxValue: result.maxValue,
          });

          // ── STEP 10: Push the pano image into the StackViewport ────────
          const panoViewport = cornerstoneViewportService.getCornerstoneViewport('cpr-pano');
          if (!panoViewport) throw new Error('cpr-pano viewport not ready.');

          await (panoViewport as cornerstone.Types.IStackViewport).setStack([PANO_IMAGE_ID]);
          panoViewport.render();

          // ── STEP 11: Initialize cross-section camera + sync group ──────
          initializeCrossSection(frames, servicesManager);

        } catch (innerErr) {
          const msg = innerErr instanceof Error ? innerErr.message : String(innerErr);
          console.error('[CPR] Pipeline failed after HP switch:', msg);
          setError(msg);
        } finally {
          setIsGenerating(false);
        }
      };

      // Store ref so we can unsubscribe if needed
      hpListenerRef.current = onProtocolApplied;
      hangingProtocolService.subscribe(
        hangingProtocolService.EVENTS.PROTOCOL_APPLIED,
        onProtocolApplied
      );

      // NOW switch the layout — viewports are destroyed after this call
      commandsManager.runCommand('setHangingProtocol', {
        protocolId: 'cpr',
        stageIndex: 1, // Stage 1 = CPR Result (0-indexed). Stage 0 = arch drawing.
      });

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[CPR] onDone failed:', msg);
      setError(msg);
      setIsGenerating(false);
    }
  }, [servicesManager, commandsManager, panoWidth, panoHeight, slabHalfThicknessMm, slabSamples, aggregation]);

  // ── onSliderChange: Called on every slider tick ─────────────────────────
  // This is O(1) — just a frame array lookup and a camera update.
  // No recomputation, no Worker, no async.
  const onSliderChange = useCallback((frameIndex: number) => {
    if (!cprStateService.hasData()) return;

    const frames = cprStateService.getFrames();
    const clampedIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
    const frame = frames[clampedIndex];

    const { cornerstoneViewportService } = servicesManager.services;
    const crossViewport = cornerstoneViewportService.getCornerstoneViewport('cpr-crosssection');
    if (!crossViewport) return;

    crossViewport.setCamera({
      focalPoint:      Array.from(frame.position) as [number, number, number],
      // N_camera: full 3D RMF normal — NOT N_slab (which is Z-zeroed for sampling only)
      viewPlaneNormal: Array.from(frame.N_camera) as [number, number, number],
      // S: RMF up-vector — prevents gimbal lock at steep ramus angles
      viewUp:          Array.from(frame.S)        as [number, number, number],
      parallelScale: 20,
      parallelProjection: true,
    });

    crossViewport.render();
  }, [servicesManager]);

  return { onDone, onSliderChange, isGenerating, error };
}

// ─────────────────────────────────────────────────────────────────────────────
// ALTERNATIVE: ANNOTATION_COMPLETED EVENT LISTENER (non-hook pattern)
// ─────────────────────────────────────────────────────────────────────────────
//
// If you prefer event-driven over a "Done" button, use this pattern inside
// your mode's onModeEnter() callback. This fires when the user double-clicks
// to complete the SplineROI annotation.
//
// IMPORTANT: This does NOT replace the "Done" button approach above.
// The ANNOTATION_COMPLETED event fires immediately when the annotation is
// closed — before the user has a chance to review or adjust the arch.
// A dedicated "Done" button gives the user a confirmation step, which is
// preferred in clinical workflows. Use whichever your UX requires.
//
// Usage in modes/cpr/src/index.ts → onModeEnter:
//
//   import { Enums as csToolsEnums } from '@cornerstonejs/tools';
//   import * as cornerstone from '@cornerstonejs/core';
//
//   // Inside onModeEnter:
//   const unsubscribe = cornerstone.eventTarget.addEventListener(
//     csToolsEnums.Events.ANNOTATION_COMPLETED,
//     (event: CustomEvent) => {
//       const { annotation } = event.detail;
//       if (annotation.metadata.toolName !== 'SplineROI') return;
//
//       // Annotation just completed — show a "Generate Pano" button in the UI
//       // rather than auto-triggering, to give the user a review step.
//       // Set a flag in CPRStateService or a React state atom to enable the button.
//       console.log('[CPR] Arch annotation completed. Ready to generate panoramic.');
//     }
//   );
//
//   // Store unsubscribe in a ref and call it in onModeExit:
//   _annotationCompleteUnsubscribe = unsubscribe;
//
// ─────────────────────────────────────────────────────────────────────────────
