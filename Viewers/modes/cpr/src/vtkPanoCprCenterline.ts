import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';

import type { CPRFrame, Point3 } from './cprMath';

const BASIS_EPSILON = 1e-6;
const ORTHOGONALITY_TOLERANCE = 1e-4;
const UNIT_LENGTH_TOLERANCE = 1e-4;

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

function length(v: Point3): number {
  return Math.sqrt(dot(v, v));
}

function scale(v: Point3, factor: number): Point3 {
  return [v[0] * factor, v[1] * factor, v[2] * factor];
}

function subtract(a: Point3, b: Point3): Point3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function assertFiniteVector(vector: Point3, label: string, frameIndex: number): void {
  if (
    !Number.isFinite(vector[0]) ||
    !Number.isFinite(vector[1]) ||
    !Number.isFinite(vector[2])
  ) {
    throw new Error(`[CPR] ${label} contains a non-finite value at frame ${frameIndex}.`);
  }
}

function normalizeStrict(vector: Point3, label: string, frameIndex: number): Point3 {
  assertFiniteVector(vector, label, frameIndex);
  const vectorLength = length(vector);
  if (!Number.isFinite(vectorLength) || vectorLength <= BASIS_EPSILON) {
    throw new Error(
      `[CPR] ${label} is degenerate at frame ${frameIndex}; cannot build VTK CPR centerline.`
    );
  }

  return [vector[0] / vectorLength, vector[1] / vectorLength, vector[2] / vectorLength];
}

function projectPerpendicular(vector: Point3, axis: Point3): Point3 {
  return subtract(vector, scale(axis, dot(vector, axis)));
}

function validateBasis(
  tangent: Point3,
  slabNormal: Point3,
  vertical: Point3,
  frameIndex: number
): void {
  const tangentLength = length(tangent);
  const slabLength = length(slabNormal);
  const verticalLength = length(vertical);

  if (
    Math.abs(tangentLength - 1) > UNIT_LENGTH_TOLERANCE ||
    Math.abs(slabLength - 1) > UNIT_LENGTH_TOLERANCE ||
    Math.abs(verticalLength - 1) > UNIT_LENGTH_TOLERANCE
  ) {
    throw new Error(
      `[CPR] Frame ${frameIndex} produced a non-unit VTK CPR orientation basis.`
    );
  }

  const dotSnT = Math.abs(dot(slabNormal, tangent));
  const dotSnS = Math.abs(dot(slabNormal, vertical));
  const dotST = Math.abs(dot(vertical, tangent));
  if (
    dotSnT > ORTHOGONALITY_TOLERANCE ||
    dotSnS > ORTHOGONALITY_TOLERANCE ||
    dotST > ORTHOGONALITY_TOLERANCE
  ) {
    throw new Error(
      `[CPR] Frame ${frameIndex} produced a non-orthogonal VTK CPR orientation basis.`
    );
  }

  const handedness = dot(cross(vertical, slabNormal), tangent);
  if (handedness < 1 - ORTHOGONALITY_TOLERANCE) {
    throw new Error(
      `[CPR] Frame ${frameIndex} produced a left-handed VTK CPR orientation basis.`
    );
  }
}

function buildOrthonormalOrientation(frame: CPRFrame, frameIndex: number): [Point3, Point3, Point3] {
  const tangent = normalizeStrict(frame.T, 'frame.T', frameIndex);
  const sourceVertical = normalizeStrict(frame.S, 'frame.S', frameIndex);
  const sourceSlabNormal = normalizeStrict(frame.N_slab, 'frame.N_slab', frameIndex);

  const projectedVertical = projectPerpendicular(sourceVertical, tangent);
  let vertical = normalizeStrict(
    projectedVertical,
    'frame.S projected perpendicular to frame.T',
    frameIndex
  );
  let slabNormal = normalizeStrict(cross(tangent, vertical), 'cross(frame.T, frame.S)', frameIndex);

  if (dot(slabNormal, sourceSlabNormal) < 0) {
    slabNormal = scale(slabNormal, -1);
  }

  vertical = normalizeStrict(cross(slabNormal, tangent), 'cross(frame.N_slab, frame.T)', frameIndex);
  validateBasis(tangent, slabNormal, vertical, frameIndex);

  return [slabNormal, vertical, tangent];
}

function assertFrameCount(frames: CPRFrame[]): void {
  if (!Array.isArray(frames) || frames.length < 2) {
    throw new Error('[CPR] At least two CPR frames are required to build a VTK pano centerline.');
  }
}

export function computePanoCenterlineLengthMm(frames: CPRFrame[]): number {
  assertFrameCount(frames);

  let totalLengthMm = 0;
  for (let index = 1; index < frames.length; index++) {
    const previous = frames[index - 1].position;
    const current = frames[index].position;
    assertFiniteVector(previous, 'frame.position', index - 1);
    assertFiniteVector(current, 'frame.position', index);
    totalLengthMm += length(subtract(current, previous));
  }

  return totalLengthMm;
}

export function buildVtkPanoCenterline(frames: CPRFrame[]): vtkPolyData {
  assertFrameCount(frames);

  const polyData = vtkPolyData.newInstance();
  const points = vtkPoints.newInstance();
  const lines = vtkCellArray.newInstance();
  const orientationArray = vtkDataArray.newInstance({
    name: 'Orientation',
    numberOfComponents: 9,
    values: new Float32Array(frames.length * 9),
  });

  const pointValues = new Float32Array(frames.length * 3);
  const lineConnectivity = new Uint32Array(frames.length + 1);
  lineConnectivity[0] = frames.length;

  for (let index = 0; index < frames.length; index++) {
    const frame = frames[index];
    assertFiniteVector(frame.position, 'frame.position', index);

    pointValues[index * 3] = frame.position[0];
    pointValues[index * 3 + 1] = frame.position[1];
    pointValues[index * 3 + 2] = frame.position[2];
    lineConnectivity[index + 1] = index;

    const [slabNormal, vertical, tangent] = buildOrthonormalOrientation(frame, index);
    const orientationTuple: [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ] = [
      vertical[0],
      vertical[1],
      vertical[2],
      slabNormal[0],
      slabNormal[1],
      slabNormal[2],
      tangent[0],
      tangent[1],
      tangent[2],
    ];
    if (index === 0) {
      const handedness = dot(cross(vertical, slabNormal), tangent);
      const sourceVertical = normalizeStrict(frame.S, 'frame.S source log', index);
      const sourceSlabNormal = normalizeStrict(frame.N_slab, 'frame.N_slab source log', index);
      const sourceTangent = normalizeStrict(frame.T, 'frame.T source log', index);
      console.log(
        `DIAG-TRIPWIRE: vtk-orientation-tuple-0: [${orientationTuple.join(', ')}]`
      );
      console.log(
          `DIAG-TRIPWIRE: vtk-basis-check-0 det=${handedness.toFixed(6)} dotWidthS=${dot(
            vertical,
            sourceVertical
          ).toFixed(6)} dotProjSlab=${dot(slabNormal, sourceSlabNormal).toFixed(
            6
          )} dotTravelT=${dot(tangent, sourceTangent).toFixed(6)} widthAxis=[${vertical.join(
            ', '
          )}] projAxis=[${slabNormal.join(', ')}] travelAxis=[${tangent.join(', ')}]`
        );
      }
    orientationArray.setTuple(index, orientationTuple);
  }

  points.setData(pointValues, 3);
  lines.setData(lineConnectivity);

  polyData.setPoints(points);
  polyData.setLines(lines);
  polyData.getPointData().addArray(orientationArray);

  return polyData;
}
