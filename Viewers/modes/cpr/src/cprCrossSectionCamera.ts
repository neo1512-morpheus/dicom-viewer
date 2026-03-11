import { vec3 } from 'gl-matrix';

import type { CPRFrame } from './cprMath';

type CameraVector = [number, number, number];

type PreviousCameraLike = {
  viewPlaneNormal?: ArrayLike<number> | null;
  viewUp?: ArrayLike<number> | null;
  position?: ArrayLike<number> | null;
  focalPoint?: ArrayLike<number> | null;
  parallelScale?: number | null;
} | null | undefined;

const CROSSSECTION_CLINICAL_VERTICAL_HEIGHT_MM = 45;
const CROSSSECTION_FIXED_PARALLEL_SCALE = CROSSSECTION_CLINICAL_VERTICAL_HEIGHT_MM / 2;
const CROSSSECTION_RIGID_CAMERA_DISTANCE_MULTIPLIER = 8;

function normalizeCameraVector(
  value: CameraVector,
  fallback: CameraVector
): vec3 {
  const out = vec3.fromValues(
    Number(value[0] ?? fallback[0]),
    Number(value[1] ?? fallback[1]),
    Number(value[2] ?? fallback[2])
  );

  if (!Number.isFinite(out[0]) || !Number.isFinite(out[1]) || !Number.isFinite(out[2])) {
    vec3.set(out, fallback[0], fallback[1], fallback[2]);
  }
  if (vec3.length(out) < 1e-6) {
    vec3.set(out, fallback[0], fallback[1], fallback[2]);
  }

  vec3.normalize(out, out);
  return out;
}

function toCameraVector(value: vec3): CameraVector {
  return [value[0], value[1], value[2]];
}

function projectPerpendicular(
  value: vec3,
  axis: vec3,
  fallback: CameraVector
): vec3 {
  const projection = vec3.scale(vec3.create(), axis, vec3.dot(value, axis));
  const out = vec3.subtract(vec3.create(), value, projection);

  if (vec3.length(out) < 1e-6) {
    const fallbackVec = normalizeCameraVector(fallback, [0, 0, 1]);
    const fallbackProjection = vec3.scale(vec3.create(), axis, vec3.dot(fallbackVec, axis));
    vec3.subtract(out, fallbackVec, fallbackProjection);
  }

  if (vec3.length(out) < 1e-6) {
    const alternate = Math.abs(axis[2]) < 0.9 ? vec3.fromValues(0, 0, 1) : vec3.fromValues(0, 1, 0);
    const alternateProjection = vec3.scale(vec3.create(), axis, vec3.dot(alternate, axis));
    vec3.subtract(out, alternate, alternateProjection);
  }

  if (vec3.length(out) < 1e-6) {
    vec3.set(out, 1, 0, 0);
  }

  vec3.normalize(out, out);
  return out;
}

function pickStablePerpendicular(axis: vec3): vec3 {
  const unitAxis = normalizeCameraVector(toCameraVector(axis), [1, 0, 0]);
  const candidates = [
    vec3.fromValues(1, 0, 0),
    vec3.fromValues(0, 1, 0),
    vec3.fromValues(0, 0, 1),
  ];

  let best = candidates[0];
  let smallestAlignment = Math.abs(vec3.dot(unitAxis, best));

  for (let i = 1; i < candidates.length; i++) {
    const alignment = Math.abs(vec3.dot(unitAxis, candidates[i]));
    if (alignment < smallestAlignment) {
      smallestAlignment = alignment;
      best = candidates[i];
    }
  }

  return projectPerpendicular(best, unitAxis, [0, 1, 0]);
}

export function buildCrossSectionBasis(frame: CPRFrame): {
  viewPlaneNormal: CameraVector;
  viewUp: CameraVector;
} {
  const normal = normalizeCameraVector(
    [frame.T[0], frame.T[1], frame.T[2]],
    [1, 0, 0]
  );
  const frameRight = normalizeCameraVector(
    [frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]],
    [0, 1, 0]
  );
  const frameUp = normalizeCameraVector(
    [frame.S[0], frame.S[1], frame.S[2]],
    [0, 0, 1]
  );

  // Match the synthetic image basis exactly:
  // - plane normal follows the arch tangent T
  // - horizontal axis follows the stable RMF N_camera vector
  // - vertical axis follows the stable RMF S vector
  let right = projectPerpendicular(frameRight, normal, [0, 1, 0]);
  if (vec3.length(right) < 1e-6) {
    right = pickStablePerpendicular(normal);
  } else {
    vec3.normalize(right, right);
  }
  if (vec3.dot(right, frameRight) < 0) {
    vec3.scale(right, right, -1);
  }

  let up = projectPerpendicular(frameUp, normal, [0, 0, 1]);
  if (vec3.length(up) < 1e-6) {
    up = vec3.cross(vec3.create(), normal, right);
    if (vec3.length(up) < 1e-6) {
      up = pickStablePerpendicular(normal);
    }
  }
  if (vec3.length(up) < 1e-6) {
    up = projectPerpendicular(vec3.fromValues(0, 0, 1), normal, [0, 0, 1]);
  }
  vec3.normalize(up, up);

  if (vec3.dot(up, frameUp) < 0) {
    vec3.scale(right, right, -1);
    vec3.scale(up, up, -1);
  }

  const actualRight = vec3.cross(vec3.create(), up, normal);
  if (vec3.length(actualRight) >= 1e-6) {
    vec3.normalize(actualRight, actualRight);
    if (vec3.dot(actualRight, frameRight) < 0) {
      vec3.scale(up, up, -1);
    }
  }

  return {
    viewPlaneNormal: toCameraVector(normal),
    viewUp: toCameraVector(up),
  };
}

export function buildCrossSectionCameraForFrame(
  frame: CPRFrame,
  previousCamera?: PreviousCameraLike,
  verticalCenterOffsetMm = 0
): {
  focalPoint: CameraVector;
  position: CameraVector;
  viewPlaneNormal: CameraVector;
  viewUp: CameraVector;
  parallelScale: number;
  clippingRange: [number, number];
  parallelProjection: true;
  flipHorizontal: false;
  flipVertical: false;
} {
  const { viewPlaneNormal, viewUp } = buildCrossSectionBasis(frame);
  const parallelScale = CROSSSECTION_FIXED_PARALLEL_SCALE;
  const cameraDistance = parallelScale * CROSSSECTION_RIGID_CAMERA_DISTANCE_MULTIPLIER;
  const clippingRange: [number, number] = [cameraDistance - 200, cameraDistance + 200];
  const centerOffsetMm = Number.isFinite(verticalCenterOffsetMm) ? verticalCenterOffsetMm : 0;
  const focalPoint: CameraVector = [
    frame.position[0] + viewUp[0] * centerOffsetMm,
    frame.position[1] + viewUp[1] * centerOffsetMm,
    frame.position[2] + viewUp[2] * centerOffsetMm,
  ];
  const position: CameraVector = [
    focalPoint[0] + viewPlaneNormal[0] * cameraDistance,
    focalPoint[1] + viewPlaneNormal[1] * cameraDistance,
    focalPoint[2] + viewPlaneNormal[2] * cameraDistance,
  ];

  console.log(
    `[CPR-DEBUG] buildCrossSectionCameraForFrame ${JSON.stringify({
      frameIndex: frame.index,
      previousParallelScale: Number.isFinite(previousCamera?.parallelScale as number)
        ? Number(previousCamera?.parallelScale)
        : null,
      centerOffsetMm,
      fixedVerticalFieldOfViewMm: CROSSSECTION_CLINICAL_VERTICAL_HEIGHT_MM,
      outputParallelScale: parallelScale,
      clippingRange,
      focalPoint,
      position,
    })}`
  );

  return {
    focalPoint,
    position,
    viewPlaneNormal,
    viewUp,
    parallelScale,
    clippingRange,
    parallelProjection: true,
    flipHorizontal: false,
    flipVertical: false,
  };
}
