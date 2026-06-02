export type Point3 = [number, number, number];

export interface CPRFrame {
  index: number;
  position: Point3;
  T: Point3;
  N_camera: Point3;
  N_slab: Point3;
  S: Point3;
}

const EPS = 1e-8;

function dot(a: Point3, b: Point3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Point3, b: Point3): Point3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(v: Point3): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: Point3, fallback: Point3 = [1, 0, 0]): Point3 {
  const n = norm(v);
  if (!Number.isFinite(n) || n < EPS) {
    return fallback;
  }

  return [v[0] / n, v[1] / n, v[2] / n];
}

function scale(v: Point3, s: number): Point3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function add(a: Point3, b: Point3): Point3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function projectPerpendicular(v: Point3, axis: Point3): Point3 {
  const axisNorm = normalize(axis);
  const along = scale(axisNorm, dot(v, axisNorm));
  return [v[0] - along[0], v[1] - along[1], v[2] - along[2]];
}

function rotateAroundAxis(v: Point3, axis: Point3, angle: number): Point3 {
  const k = normalize(axis);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  const term1 = scale(v, cosA);
  const term2 = scale(cross(k, v), sinA);
  const term3 = scale(k, dot(k, v) * (1 - cosA));

  return add(add(term1, term2), term3);
}

function negate(v: Point3): Point3 {
  return [-v[0], -v[1], -v[2]];
}

function chooseInitialNormal(t0: Point3): Point3 {
  const upA: Point3 = [0, 0, 1];
  const upB: Point3 = [0, 1, 0];

  let n = cross(upA, t0);
  if (norm(n) < EPS) {
    n = cross(upB, t0);
  }

  return normalize(n, [1, 0, 0]);
}

function pickStablePerpendicular(axis: Point3): Point3 {
  const unitAxis = normalize(axis, [1, 0, 0]);
  const basis: Point3[] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  let bestBasis = basis[0];
  let smallestAlignment = Math.abs(dot(unitAxis, bestBasis));

  for (let i = 1; i < basis.length; i++) {
    const alignment = Math.abs(dot(unitAxis, basis[i]));
    if (alignment < smallestAlignment) {
      smallestAlignment = alignment;
      bestBasis = basis[i];
    }
  }

  const projected = projectPerpendicular(bestBasis, unitAxis);
  if (norm(projected) >= EPS) {
    return normalize(projected, [0, 1, 0]);
  }

  const fallbackA = cross(unitAxis, [0, 1, 0]);
  if (norm(fallbackA) >= EPS) {
    return normalize(fallbackA, [1, 0, 0]);
  }

  return normalize(cross(unitAxis, [1, 0, 0]), [0, 1, 0]);
}

function orthonormalizeFrame(
  tangent: Point3,
  candidateN: Point3,
  candidateS: Point3,
  previousN: Point3,
  previousS: Point3
): { N: Point3; S: Point3 } {
  const T = normalize(tangent, [1, 0, 0]);

  let N = projectPerpendicular(candidateN, T);
  if (norm(N) < EPS) {
    N = projectPerpendicular(candidateS, T);
  }
  if (norm(N) < EPS) {
    N = pickStablePerpendicular(T);
  } else {
    N = normalize(N, pickStablePerpendicular(T));
  }

  let S = cross(T, N);
  if (norm(S) < EPS) {
    N = pickStablePerpendicular(T);
    S = cross(T, N);
  }
  S = normalize(S, pickStablePerpendicular(T));
  N = normalize(cross(S, T), N);

  // Keep frame orientation continuous to prevent 180-degree flips.
  // Use S (vertical/binormal) as the stability anchor — on horizontal dental
  // arches S starts as [0,0,1] and should remain stable throughout.
  // N_camera legitimately rotates ~180° on U-shaped arches as T follows the
  // curve, so using N in this check would falsely trigger a flip that corrupts
  // the vertical direction, causing cross-section mirror-flips and panoramic
  // sampling distortion.
  if (dot(S, previousS) < 0) {
    N = negate(N);
    S = negate(S);
  }

  return { N, S };
}

function computeRigidSlabNormal(tangent: Point3, verticalDir: Point3 | null): Point3 {
  const T = normalize(tangent, [1, 0, 0]);
  const referenceVertical: Point3 = verticalDir
    ? normalize(verticalDir, [0, 0, 1])
    : [0, 0, 1];
  const fallback = pickStablePerpendicular(T);
  const slabNormal = projectPerpendicular(cross(T, referenceVertical), T);

  if (norm(slabNormal) < EPS) {
    return fallback;
  }

  return normalize(slabNormal, fallback);
}

function coercePoint3(v: Point3): Point3 {
  return [v[0], v[1], v[2]];
}

export function buildRMFFrames(
  positions: Point3[],
  tangents: Point3[],
  verticalDir?: [number, number, number]
): CPRFrame[] {
  if (!Array.isArray(positions) || !Array.isArray(tangents)) {
    throw new Error('[cprMath] positions and tangents must be arrays.');
  }

  if (positions.length === 0 || tangents.length === 0) {
    throw new Error('[cprMath] positions and tangents cannot be empty.');
  }

  if (positions.length !== tangents.length) {
    throw new Error('[cprMath] positions and tangents length mismatch.');
  }

  const count = positions.length;
  const frames: CPRFrame[] = new Array(count);
  const normalizedVerticalDir = verticalDir ? normalize(coercePoint3(verticalDir), [0, 0, 1]) : null;
  const normalizedPositions = positions.map(coercePoint3);

  const T0 = normalize(coercePoint3(tangents[0]), [1, 0, 0]);
  const initialProjectedVertical = normalizedVerticalDir
    ? projectPerpendicular(normalizedVerticalDir, T0)
    : null;
  const initialS =
    initialProjectedVertical && norm(initialProjectedVertical) >= EPS
      ? normalize(initialProjectedVertical, pickStablePerpendicular(T0))
      : normalize(cross(T0, chooseInitialNormal(T0)), pickStablePerpendicular(T0));
  const initialN =
    initialProjectedVertical && norm(initialProjectedVertical) >= EPS
      ? normalize(cross(T0, initialS), chooseInitialNormal(T0))
      : chooseInitialNormal(T0);
  const initialBasis = orthonormalizeFrame(T0, initialN, initialS, initialN, initialS);
  let N = initialBasis.N;
  let S = initialBasis.S;

  let prevT = T0;
  const prevNslab = computeRigidSlabNormal(T0, normalizedVerticalDir);

  frames[0] = {
    index: 0,
    position: normalizedPositions[0],
    T: T0,
    N_camera: N,
    N_slab: prevNslab,
    S,
  };

  for (let i = 1; i < count; i++) {
    const T = normalize(coercePoint3(tangents[i]), prevT);
    const axis = cross(prevT, T);
    const axisLen = norm(axis);
    const tangentDot = Math.max(-1, Math.min(1, dot(prevT, T)));

    let transportedN = N;
    let transportedS = S;
    if (axisLen >= EPS) {
      const angle = Math.atan2(axisLen, tangentDot);
      transportedN = rotateAroundAxis(N, axis, angle);
      transportedS = rotateAroundAxis(S, axis, angle);
    } else if (tangentDot < -0.9999) {
      const halfTurnAxis = pickStablePerpendicular(prevT);
      transportedN = rotateAroundAxis(N, halfTurnAxis, Math.PI);
      transportedS = rotateAroundAxis(S, halfTurnAxis, Math.PI);
    }

    const basis = orthonormalizeFrame(T, transportedN, transportedS, N, S);
    const Ni = basis.N;
    const Si = basis.S;

    const Nslab = computeRigidSlabNormal(T, normalizedVerticalDir);

    frames[i] = {
      index: i,
      position: normalizedPositions[i],
      T,
      N_camera: Ni,
      N_slab: Nslab,
      S: Si,
    };

    prevT = T;
    N = Ni;
    S = Si;
  }

  return frames;
}
