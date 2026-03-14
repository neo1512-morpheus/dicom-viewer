import { cache, metaData, type Types } from '@cornerstonejs/core';
import type vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import type vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import type vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import type vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArrayFactory from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import '@kitware/vtk.js/Rendering/Profiles/Volume';
import vtkImageCPRMapper from '@kitware/vtk.js/Rendering/Core/ImageCPRMapper';
import { ProjectionMode } from '@kitware/vtk.js/Rendering/Core/ImageCPRMapper/Constants';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';

import type { CPRFrame } from './cprMath';
import { normalizeScalarDataToHuFloat32 } from './cprScalarPolicy';
import { buildVtkPanoCenterline, computePanoCenterlineLengthMm } from './vtkPanoCprCenterline';

const VTK_PANO_ACTOR_UID_PREFIX = 'cpr-vtk-pano-actor';
const VTK_PANO_HOST_ATTRIBUTE = 'data-cpr-vtk-pano-host';
const VTK_PANO_SCREEN_VIEW_UP: [number, number, number] = [1, 0, 0];
type CprSourceVolume = Types.IImageVolume & {
  imageIds?: string[];
  isPreScaled?: boolean;
};
type VtkGenericRenderWindowInstance = ReturnType<typeof vtkGenericRenderWindow.newInstance>;
type VtkRendererInstance = ReturnType<VtkGenericRenderWindowInstance['getRenderer']>;
type VtkRenderWindowInstance = ReturnType<VtkGenericRenderWindowInstance['getRenderWindow']>;
type VtkImageDataInstance = ReturnType<typeof vtkImageData.newInstance>;
type VtkScalarRange = ReturnType<vtkDataArray['getRange']> | null;

type HostedPanoViewportElement = HTMLElement & {
  __cprVtkPanoHost?: {
    actorUID: string;
    resetCamera(): void;
    render(): void;
    resize(): void;
    captureCameraSyncBaseline(): void;
    syncCamera(
      syncState:
        | {
            panDeltaPx?: ArrayLike<number> | null;
            zoomRatio?: number | null;
            viewportHeightPx?: number | null;
          }
        | null
        | undefined
    ): void;
    clearCameraSyncBaseline(): void;
    dispose(): void;
  };
};

export interface AttachVtkPanoCprArgs {
  viewport: Types.IVolumeViewport;
  sourceVolumeId: string;
  frames: CPRFrame[];
  verticalHalfMm: number;
  slabHalfThicknessMm: number;
  slabSamples: number;
  projectionMode?: 'AVERAGE' | 'MAX' | 'MIN';
  initialWindowWidth: number;
  initialWindowCenter: number;
  runId?: string;
}

export interface AttachedVtkPanoCpr {
  actorUID: string;
  updateWindowLevel(windowWidth: number, windowCenter: number): void;
  captureCameraSyncBaseline(): void;
  syncCamera(
    syncState:
      | {
          panDeltaPx?: ArrayLike<number> | null;
          zoomRatio?: number | null;
          viewportHeightPx?: number | null;
        }
      | null
      | undefined
  ): void;
  clearCameraSyncBaseline(): void;
  resize(): void;
  dispose(): void;
}

function toRunLabel(runId?: string): string {
  return runId || 'na';
}

function requirePositiveNumber(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`[CPR] ${label} must be a positive finite number.`);
  }

  return Number(value);
}

function requireFiniteNumber(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new Error(`[CPR] ${label} must be a finite number.`);
  }

  return Number(value);
}

function toProjectionMode(projectionMode: AttachVtkPanoCprArgs['projectionMode']): ProjectionMode {
  switch (projectionMode) {
    case 'MAX':
      return ProjectionMode.MAX;
    case 'MIN':
      return ProjectionMode.MIN;
    case 'AVERAGE':
    default:
      return ProjectionMode.AVERAGE;
  }
}

function getProjectionModeLabel(projectionMode: AttachVtkPanoCprArgs['projectionMode']): string {
  return projectionMode || 'AVERAGE';
}

function getElementClassName(element: Element): string {
  const rawClassName = (element as HTMLElement | SVGElement).className;
  if (typeof rawClassName === 'string') {
    return rawClassName;
  }

  if (rawClassName && typeof rawClassName.baseVal === 'string') {
    return rawClassName.baseVal;
  }

  return '';
}

function isOverlayLikeElement(element: Element): boolean {
  return (
    element instanceof SVGElement ||
    element.classList.contains('svg-layer') ||
    element.getAttribute('data-svg-layer') === 'true' ||
    getElementClassName(element).includes('overlay')
  );
}

function getSourceVolumeModalityRescale(sourceVolume: CprSourceVolume): {
  slope: number;
  intercept: number;
} {
  const imageIds = Array.isArray(sourceVolume?.imageIds) ? sourceVolume.imageIds : [];
  const firstImageId = imageIds.length ? imageIds[0] : undefined;

  if (!firstImageId) {
    return { slope: 1, intercept: 0 };
  }

  const modalityLut = metaData.get('modalityLutModule', firstImageId) as
    | {
        rescaleSlope?: number;
        rescaleIntercept?: number;
      }
    | undefined;
  const slope = Number(modalityLut?.rescaleSlope);
  const intercept = Number(modalityLut?.rescaleIntercept);

  return {
    slope: Number.isFinite(slope) && Math.abs(slope) > 1e-8 ? slope : 1,
    intercept: Number.isFinite(intercept) ? intercept : 0,
  };
}

function getSourceVolumePixelStorage(sourceVolume: CprSourceVolume): {
  bitsStored: number;
  bitsAllocated: number;
  highBit: number;
  pixelRepresentation: number;
} {
  const imageIds = Array.isArray(sourceVolume?.imageIds) ? sourceVolume.imageIds : [];
  const firstImageId = imageIds.length ? imageIds[0] : undefined;

  if (!firstImageId) {
    return {
      bitsStored: 16,
      bitsAllocated: 16,
      highBit: 15,
      pixelRepresentation: 1,
    };
  }

  const imagePixelModule = metaData.get('imagePixelModule', firstImageId) as
    | {
        bitsStored?: number;
        bitsAllocated?: number;
        highBit?: number;
        pixelRepresentation?: number;
      }
    | undefined;

  const bitsStored = Number(imagePixelModule?.bitsStored);
  const bitsAllocated = Number(imagePixelModule?.bitsAllocated);
  const highBit = Number(imagePixelModule?.highBit);
  const pixelRepresentation = Number(imagePixelModule?.pixelRepresentation);
  const safeBitsStored =
    Number.isFinite(bitsStored) && bitsStored >= 1 ? Math.floor(bitsStored) : 16;
  const safeBitsAllocated =
    Number.isFinite(bitsAllocated) && bitsAllocated >= safeBitsStored
      ? Math.floor(bitsAllocated)
      : 16;
  const safeHighBit =
    Number.isFinite(highBit) && highBit >= 0 ? Math.floor(highBit) : safeBitsStored - 1;

  return {
    bitsStored: safeBitsStored,
    bitsAllocated: safeBitsAllocated,
    highBit: Math.max(0, Math.min(safeBitsAllocated - 1, safeHighBit)),
    pixelRepresentation:
      Number.isFinite(pixelRepresentation) &&
      (pixelRepresentation === 0 || pixelRepresentation === 1)
        ? pixelRepresentation
        : 1,
  };
}

interface NormalizedVtkPanoSource {
  imageData: VtkImageDataInstance;
  scalarRange: VtkScalarRange;
  dataType: string;
  effectiveIsPreScaled: boolean;
  heuristicOverride: boolean;
  storedValueNormalizationApplied: boolean;
  normalizationSignature: string | null;
  unsignedPackedArtifactDetected: boolean;
  sourceScalarRange: VtkScalarRange;
  averageProjectionFilter: string | null;
}

function createNormalizedHuPanoSource(
  sourceVolume: CprSourceVolume,
  runLabel: string,
  projectionMode: AttachVtkPanoCprArgs['projectionMode']
): NormalizedVtkPanoSource {
  const sourceImageData = sourceVolume?.imageData;
  const sourceScalars = sourceImageData?.getPointData?.().getScalars?.();

  if (!sourceImageData || !sourceScalars) {
    throw new Error(
      '[CPR] Source volume imageData scalars are unavailable for VTK pano normalization.'
    );
  }

  const sourceScalarData = sourceScalars.getData?.() as ArrayLike<number> | undefined;
  if (!sourceScalarData || sourceScalarData.length === 0) {
    throw new Error('[CPR] Source volume scalar data is empty for VTK pano normalization.');
  }

  const sourceScalarRange = sourceScalars.getRange?.() || null;
  const sourceDataType = sourceScalars.getDataType?.() || 'unknown';
  const rawSourceIsPreScaled = !!sourceVolume.isPreScaled;
  const { slope, intercept } = getSourceVolumeModalityRescale(sourceVolume);
  const pixelStorage = getSourceVolumePixelStorage(sourceVolume);
  const normalized = normalizeScalarDataToHuFloat32({
    scalarData: sourceScalarData,
    isPreScaled: rawSourceIsPreScaled,
    rescaleSlope: slope,
    rescaleIntercept: intercept,
    bitsStored: pixelStorage.bitsStored,
    bitsAllocated: pixelStorage.bitsAllocated,
    highBit: pixelStorage.highBit,
    pixelRepresentation: pixelStorage.pixelRepresentation,
  });
  const cleanedHuScalarData = normalized.pixelData;
  let averageProjectionFilter: string | null = null;

  if (projectionMode === 'AVERAGE') {
    averageProjectionFilter = 'winsorized-average-hu[-1000,1800]';
    for (let i = 0; i < cleanedHuScalarData.length; i++) {
      const value = cleanedHuScalarData[i];
      if (!Number.isFinite(value)) {
        cleanedHuScalarData[i] = -1000;
        continue;
      }
      if (value < -1000) {
        cleanedHuScalarData[i] = -1000;
      } else if (value > 1800) {
        cleanedHuScalarData[i] = 1800;
      }
    }
  }

  const normalizedImageData = vtkImageData.newInstance();
  normalizedImageData.setDimensions(sourceImageData.getDimensions());
  normalizedImageData.setOrigin(sourceImageData.getOrigin());
  normalizedImageData.setSpacing(sourceImageData.getSpacing());
  normalizedImageData.setDirection(sourceImageData.getDirection());

  const normalizedScalars = vtkDataArrayFactory.newInstance({
    name: sourceScalars.getName?.() || 'Pixels',
    numberOfComponents: 1,
    values: cleanedHuScalarData,
  });
  normalizedImageData.getPointData().setScalars(normalizedScalars);
  normalizedImageData.modified();

  const normalizedScalarRange = normalizedScalars.getRange?.() || null;
  const normalizedDataType = normalizedScalars.getDataType?.() || 'Float32Array';

  console.log(
    `DIAG-TRIPWIRE: vtk-normalized-source run=${runLabel} sourceRange=${JSON.stringify(
      sourceScalarRange
    )} normalizedRange=${JSON.stringify(normalizedScalarRange)} sourceDataType=${sourceDataType} normalizedDataType=${normalizedDataType} sourceIsPreScaled=${rawSourceIsPreScaled} effectiveIsPreScaled=${
      normalized.transform.effectiveIsPreScaled
    } heuristicOverride=${normalized.transform.heuristicOverride} slope=${slope} intercept=${intercept} bitsStored=${
      pixelStorage.bitsStored
    } bitsAllocated=${pixelStorage.bitsAllocated} highBit=${pixelStorage.highBit} pixelRepresentation=${
      pixelStorage.pixelRepresentation
    } storedValueNormalizationApplied=${normalized.transform.shouldNormalizeStoredValues} normalizationSignature=${
      normalized.transform.normalizationSignature ?? 'none'
    } unsignedPackedArtifactDetected=${normalized.transform.unsignedPackedArtifactDetected} averageProjectionFilter=${
      averageProjectionFilter ?? 'none'
    }`
  );

  return {
    imageData: normalizedImageData,
    scalarRange: normalizedScalarRange,
    dataType: normalizedDataType,
    effectiveIsPreScaled: normalized.transform.effectiveIsPreScaled,
    heuristicOverride: normalized.transform.heuristicOverride,
    storedValueNormalizationApplied: normalized.transform.shouldNormalizeStoredValues,
    normalizationSignature: normalized.transform.normalizationSignature,
    unsignedPackedArtifactDetected: normalized.transform.unsignedPackedArtifactDetected,
    sourceScalarRange,
    averageProjectionFilter,
  };
}

function deleteVtkObject(vtkObject: { delete?: () => void } | null | undefined): void {
  vtkObject?.delete?.();
}

function toFiniteVector3(
  value: ArrayLike<number> | null | undefined
): [number, number, number] | null {
  if (!value || value.length < 3) {
    return null;
  }

  const x = Number(value[0]);
  const y = Number(value[1]);
  const z = Number(value[2]);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return null;
  }

  return [x, y, z];
}

function toFiniteVector2(value: ArrayLike<number> | null | undefined): [number, number] | null {
  if (!value || value.length < 2) {
    return null;
  }

  const x = Number(value[0]);
  const y = Number(value[1]);

  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  return [x, y];
}

function add3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract3(
  a: [number, number, number],
  b: [number, number, number]
): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale3(vector: [number, number, number], scalar: number): [number, number, number] {
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
}

function cross3(
  a: [number, number, number],
  b: [number, number, number]
): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function transformPointMat4(
  matrix: ArrayLike<number> | null | undefined,
  point: [number, number, number]
): [number, number, number] | null {
  if (!matrix || matrix.length < 16) {
    return null;
  }

  const x = point[0];
  const y = point[1];
  const z = point[2];

  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ];
}

function formatVector(value: ArrayLike<number> | null | undefined): string {
  if (!value || value.length < 3) {
    return 'null';
  }

  return `[${Number(value[0])}, ${Number(value[1])}, ${Number(value[2])}]`;
}

function isIndexPointInsideVolume(
  indexPoint: [number, number, number] | null,
  dimensions: ArrayLike<number> | null | undefined
): boolean {
  if (!indexPoint || !dimensions || dimensions.length < 3) {
    return false;
  }

  return (
    indexPoint[0] >= 0 &&
    indexPoint[1] >= 0 &&
    indexPoint[2] >= 0 &&
    indexPoint[0] <= Number(dimensions[0]) - 1 &&
    indexPoint[1] <= Number(dimensions[1]) - 1 &&
    indexPoint[2] <= Number(dimensions[2]) - 1
  );
}

function normalize3(
  vector: [number, number, number],
  fallback: [number, number, number]
): [number, number, number] {
  const length = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
  if (!Number.isFinite(length) || length < 1e-8) {
    return fallback;
  }

  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function getViewportElement(viewport: Types.IVolumeViewport): HostedPanoViewportElement {
  const element = viewport.element as HostedPanoViewportElement | undefined;
  if (!element) {
    throw new Error('[CPR] cpr-pano viewport element is not available.');
  }

  return element;
}

function getCustomPanoActorUIDs(viewport: Types.IVolumeViewport): string[] {
  return viewport
    .getActors()
    .map(actorEntry => actorEntry.uid)
    .filter(actorUID => actorUID.startsWith(VTK_PANO_ACTOR_UID_PREFIX));
}

export function detachVtkPanoCpr(viewport: Types.IVolumeViewport, actorUID?: string): void {
  const viewportElement = viewport.element as HostedPanoViewportElement | undefined;
  const hostedPano = viewportElement?.__cprVtkPanoHost;
  if (hostedPano && (!actorUID || hostedPano.actorUID === actorUID)) {
    hostedPano.dispose();
  } else if (viewportElement) {
    viewportElement
      .querySelectorAll(`[${VTK_PANO_HOST_ATTRIBUTE}]`)
      .forEach(hostedContainer => hostedContainer.parentNode?.removeChild(hostedContainer));
  }

  const actorUIDs = actorUID ? [actorUID] : getCustomPanoActorUIDs(viewport);
  if (!actorUIDs.length) {
    return;
  }

  viewport.removeActors(actorUIDs);
  viewport.render();
  console.log(
    `CPR-VTK-PANO-DETACH run=na actorUIDs=${actorUIDs.join('|')} viewportClass=${
      viewport.constructor?.name || 'unknown'
    }`
  );
}

export async function attachVtkPanoCpr(args: AttachVtkPanoCprArgs): Promise<AttachedVtkPanoCpr> {
  const runLabel = toRunLabel(args.runId);
  const viewportElement = getViewportElement(args.viewport);
  const totalWidthMm = requirePositiveNumber(args.verticalHalfMm * 2, 'vertical field of view');
  const totalSlabThicknessMm = requirePositiveNumber(
    args.slabHalfThicknessMm * 2,
    'projection slab thickness'
  );
  const slabSamples = Math.max(
    1,
    Math.round(requirePositiveNumber(args.slabSamples, 'slabSamples'))
  );
  const initialWindowWidth = requirePositiveNumber(args.initialWindowWidth, 'initialWindowWidth');
  const initialWindowCenter = requireFiniteNumber(args.initialWindowCenter, 'initialWindowCenter');

  const sourceVolume = cache.getVolume(args.sourceVolumeId) as CprSourceVolume | undefined;
  if (!sourceVolume?.imageData) {
    throw new Error(`[CPR] Source volume ${args.sourceVolumeId} is not available in cache.`);
  }
  const sourceScalars = sourceVolume.imageData.getPointData().getScalars();
  const sourceScalarRange = sourceScalars?.getRange?.() || null;
  const sourceDataType = sourceScalars?.getDataType?.() || 'unknown';
  const sourceIsPreScaled = !!sourceVolume.isPreScaled;
  const sourceDimensions = sourceVolume.imageData.getDimensions?.() || null;
  const sourceOrigin = sourceVolume.imageData.getOrigin?.() || null;
  const sourceSpacing = sourceVolume.imageData.getSpacing?.() || null;
  const sourceDirection = sourceVolume.imageData.getDirection?.() || null;
  const sourceBounds = sourceVolume.imageData.getBounds?.() || null;
  const sourceWorldToIndex = sourceVolume.imageData.getWorldToIndex?.() || null;
  const { slope: sourceRescaleSlope, intercept: sourceRescaleIntercept } =
    getSourceVolumeModalityRescale(sourceVolume);
  const appliedWindowWidth = initialWindowWidth;
  const appliedWindowCenter = initialWindowCenter;
  console.log(
    `DIAG-TRIPWIRE: vtk-initial-wl run=${runLabel} rawWindowWidth=${initialWindowWidth} rawWindowCenter=${initialWindowCenter} slope=${sourceRescaleSlope} intercept=${sourceRescaleIntercept} normalizedHuWindowWidth=${appliedWindowWidth} normalizedHuWindowCenter=${appliedWindowCenter}`
  );
  console.log(
    `DIAG-TRIPWIRE: vtk-volume-geometry run=${runLabel} dims=${JSON.stringify(
      sourceDimensions ? Array.from(sourceDimensions) : null
    )} origin=${JSON.stringify(sourceOrigin ? Array.from(sourceOrigin) : null)} spacing=${JSON.stringify(
      sourceSpacing ? Array.from(sourceSpacing) : null
    )} direction=${JSON.stringify(sourceDirection ? Array.from(sourceDirection) : null)} bounds=${JSON.stringify(
      sourceBounds ? Array.from(sourceBounds) : null
    )}`
  );

  if (args.frames.length > 0) {
    const frame0 = args.frames[0];
    const frame0Position = toFiniteVector3(frame0.position);
    const frame0Vertical = toFiniteVector3(frame0.S);
    const frame0Projection = toFiniteVector3(frame0.N_slab);
    if (frame0Position && frame0Vertical && frame0Projection) {
      const halfVerticalMm = requirePositiveNumber(args.verticalHalfMm, 'verticalHalfMm');
      const halfSlabMm = requirePositiveNumber(args.slabHalfThicknessMm, 'slabHalfThicknessMm');
      const widthTop = add3(frame0Position, scale3(frame0Vertical, halfVerticalMm));
      const widthBottom = add3(frame0Position, scale3(frame0Vertical, -halfVerticalMm));
      const slabStart = add3(frame0Position, scale3(frame0Projection, -halfSlabMm));
      const slabEnd = add3(frame0Position, scale3(frame0Projection, halfSlabMm));
      const frame0Index = transformPointMat4(sourceWorldToIndex, frame0Position);
      const widthTopIndex = transformPointMat4(sourceWorldToIndex, widthTop);
      const widthBottomIndex = transformPointMat4(sourceWorldToIndex, widthBottom);
      const slabStartIndex = transformPointMat4(sourceWorldToIndex, slabStart);
      const slabEndIndex = transformPointMat4(sourceWorldToIndex, slabEnd);
      console.log(
        `DIAG-TRIPWIRE: vtk-frame0-sampling run=${runLabel} worldPos=${formatVector(
          frame0Position
        )} indexPos=${formatVector(frame0Index)} indexInside=${isIndexPointInsideVolume(
          frame0Index,
          sourceDimensions
        )} widthTopIndex=${formatVector(widthTopIndex)} widthTopInside=${isIndexPointInsideVolume(
          widthTopIndex,
          sourceDimensions
        )} widthBottomIndex=${formatVector(
          widthBottomIndex
        )} widthBottomInside=${isIndexPointInsideVolume(
          widthBottomIndex,
          sourceDimensions
        )} slabStartIndex=${formatVector(
          slabStartIndex
        )} slabStartInside=${isIndexPointInsideVolume(
          slabStartIndex,
          sourceDimensions
        )} slabEndIndex=${formatVector(slabEndIndex)} slabEndInside=${isIndexPointInsideVolume(
          slabEndIndex,
          sourceDimensions
        )}`
      );
    }
  }

  detachVtkPanoCpr(args.viewport);

  const centerline = buildVtkPanoCenterline(args.frames);
  const centerlinePoints = centerline.getPoints() as vtkPoints | null;
  const centerlineLines = centerline.getLines() as vtkCellArray | null;
  const orientationArray = centerline
    .getPointData()
    .getArrayByName('Orientation') as vtkDataArray | null;
  const centerlineLengthMm = computePanoCenterlineLengthMm(args.frames);

  console.log(
    `CPR-VTK-CENTERLINE run=${runLabel} frameCount=${args.frames.length} pointCount=${
      centerline.getNumberOfPoints?.() ?? args.frames.length
    } lengthMm=${centerlineLengthMm.toFixed(2)}`
  );

  const mapper = vtkImageCPRMapper.newInstance();
  const imageSlice = vtkImageSlice.newInstance();
  const actorUID = `${VTK_PANO_ACTOR_UID_PREFIX}-${args.runId || Date.now().toString(36)}`;
  let normalizedPanoSource: NormalizedVtkPanoSource | null = null;
  let disposed = false;
  let hostContainer: HTMLDivElement | null = null;
  let genericRenderWindow: VtkGenericRenderWindowInstance | null = null;
  let renderer: VtkRendererInstance | null = null;
  let renderWindow: VtkRenderWindowInstance | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let domLayerObserver: MutationObserver | null = null;
  let resizeAnimationFrame: number | null = null;
  let restoreViewportPosition: string | null = null;
  const managedLayerStyles = new Map<
    HTMLElement | SVGElement,
    {
      position: string;
      zIndex: string;
      opacity: string;
      inset: string;
      pointerEvents: string;
    }
  >();
  let cameraSyncBaseline: {
    focalPoint: [number, number, number];
    position: [number, number, number];
    viewUp: [number, number, number];
    parallelScale: number;
  } | null = null;

  const getInternalViewportElement = (): HTMLDivElement | null =>
    viewportElement.querySelector('div.viewport-element');

  const getNativeCornerstoneCanvas = (
    internalViewportElement?: HTMLDivElement | null
  ): HTMLCanvasElement | null =>
    (internalViewportElement || getInternalViewportElement())?.querySelector(
      'canvas.cornerstone-canvas'
    ) || null;

  const applyPointerEventPassthrough = () => {
    if (!hostContainer) {
      return;
    }

    hostContainer.style.pointerEvents = 'none';

    hostContainer.querySelectorAll('*').forEach(node => {
      if (node instanceof HTMLElement) {
        node.style.pointerEvents = 'none';
      }
    });
  };

  const setManagedStyles = (
    element: HTMLElement | SVGElement,
    styles: Partial<
      Pick<CSSStyleDeclaration, 'position' | 'zIndex' | 'opacity' | 'inset' | 'pointerEvents'>
    >
  ) => {
    if (!managedLayerStyles.has(element)) {
      managedLayerStyles.set(element, {
        position: element.style.position,
        zIndex: element.style.zIndex,
        opacity: element.style.opacity,
        inset: element.style.inset,
        pointerEvents: element.style.pointerEvents,
      });
    }

    if (styles.position !== undefined) {
      element.style.position = styles.position;
    }
    if (styles.zIndex !== undefined) {
      element.style.zIndex = styles.zIndex;
    }
    if (styles.opacity !== undefined) {
      element.style.opacity = styles.opacity;
    }
    if (styles.inset !== undefined) {
      element.style.inset = styles.inset;
    }
    if (styles.pointerEvents !== undefined) {
      element.style.pointerEvents = styles.pointerEvents;
    }
  };

  const restoreManagedStyles = () => {
    managedLayerStyles.forEach((styles, element) => {
      element.style.position = styles.position;
      element.style.zIndex = styles.zIndex;
      element.style.opacity = styles.opacity;
      element.style.inset = styles.inset;
      element.style.pointerEvents = styles.pointerEvents;
    });
    managedLayerStyles.clear();
  };

  const applyLayerSandwich = () => {
    if (!hostContainer || disposed) {
      return;
    }

    const internalViewportElement = getInternalViewportElement();
    if (!internalViewportElement) {
      return;
    }

    const nativeCanvas = getNativeCornerstoneCanvas(internalViewportElement);
    const viewportWrapper = viewportElement.parentElement;

    if (nativeCanvas) {
      // Keep Cornerstone's source canvas alive for tool state, but make it invisible.
      setManagedStyles(nativeCanvas, {
        position: 'absolute',
        inset: '0',
        zIndex: '0',
        opacity: '0',
      });
    }

    Array.from(internalViewportElement.children).forEach(child => {
      if (child === hostContainer || child === nativeCanvas) {
        return;
      }

      if (isOverlayLikeElement(child)) {
        setManagedStyles(child as HTMLElement | SVGElement, {
          position: 'absolute',
          inset: '0',
          zIndex: '10',
          pointerEvents: 'none',
        });
      }
    });

    setManagedStyles(hostContainer, {
      position: 'absolute',
      inset: '0',
      zIndex: '1',
      opacity: '1',
      pointerEvents: 'none',
    });

    if (viewportWrapper) {
      Array.from(viewportWrapper.children).forEach(child => {
        if (!(child instanceof HTMLElement) || child === viewportElement) {
          return;
        }

        if (child.classList.contains('noselect')) {
          setManagedStyles(child, {
            position: 'absolute',
            inset: '0',
            zIndex: '10',
            pointerEvents: 'none',
          });
        }
      });
    }

    applyPointerEventPassthrough();
  };

  const logLayerStack = () => {
    if (!hostContainer || disposed) {
      return;
    }

    const internalViewportElement = getInternalViewportElement();
    if (!internalViewportElement) {
      return;
    }

    const canvases = Array.from(
      internalViewportElement.querySelectorAll('canvas')
    ) as HTMLCanvasElement[];
    const layers = canvases.map((canvas, index) => {
      const parentElement = canvas.parentElement;
      const style = window.getComputedStyle(canvas);
      return {
        index,
        className: canvas.className || '',
        parentTag: parentElement?.tagName?.toLowerCase?.() || 'none',
        parentClassName: parentElement ? getElementClassName(parentElement) : '',
        isNativeCanvas: canvas === getNativeCornerstoneCanvas(internalViewportElement),
        isHostedVtkCanvas: parentElement === hostContainer,
        opacity: style.opacity,
        zIndex: style.zIndex,
        visibility: style.visibility,
        display: style.display,
        width: canvas.width,
        height: canvas.height,
      };
    });

    const childOrder = Array.from(internalViewportElement.children).map(child => {
      if (child === hostContainer) {
        return 'vtk-host';
      }

      return `${child.tagName.toLowerCase()}${
        getElementClassName(child) ? `.${getElementClassName(child).replace(/\s+/g, '.')}` : ''
      }`;
    });

    console.log(
      `DIAG-TRIPWIRE: vtk-layer-stack run=${runLabel} childOrder=${childOrder.join(
        ' > '
      )} canvases=${JSON.stringify(layers)}`
    );
  };

  const startDomLayerObserver = () => {
    if (disposed || typeof MutationObserver === 'undefined') {
      return;
    }

    domLayerObserver?.disconnect();
    domLayerObserver = new MutationObserver(() => {
      applyLayerSandwich();
    });
    const internalViewportElement = getInternalViewportElement();
    if (internalViewportElement) {
      domLayerObserver.observe(internalViewportElement, { childList: true });
    } else {
      domLayerObserver.observe(viewportElement, { childList: true, subtree: true });
    }
    if (viewportElement.parentElement) {
      domLayerObserver.observe(viewportElement.parentElement, { childList: true });
    }
  };

  const resetHostedCamera = () => {
    const activeCamera = renderer?.getActiveCamera?.();
    activeCamera?.setParallelProjection?.(true);
    renderer?.resetCamera?.();
    activeCamera?.setViewUp?.(
      VTK_PANO_SCREEN_VIEW_UP[0],
      VTK_PANO_SCREEN_VIEW_UP[1],
      VTK_PANO_SCREEN_VIEW_UP[2]
    );
    activeCamera?.orthogonalizeViewUp?.();
    renderer?.resetCameraClippingRange?.();
    console.log(
      `DIAG-TRIPWIRE: vtk-camera-viewup run=${runLabel} viewUp=[${VTK_PANO_SCREEN_VIEW_UP.join(', ')}]`
    );
    const focalPoint = toFiniteVector3(activeCamera?.getFocalPoint?.());
    const position = toFiniteVector3(activeCamera?.getPosition?.());
    const viewUp = toFiniteVector3(activeCamera?.getViewUp?.());
    const parallelScale = Number(activeCamera?.getParallelScale?.());
    console.log(
      `DIAG-TRIPWIRE: vtk-camera-state run=${runLabel} position=${JSON.stringify(
        position
      )} focalPoint=${JSON.stringify(focalPoint)} viewUp=${JSON.stringify(
        viewUp
      )} parallelScale=${Number.isFinite(parallelScale) ? parallelScale : 'na'}`
    );
  };

  const captureCameraSyncBaseline = () => {
    if (disposed) {
      return;
    }

    const activeCamera = renderer?.getActiveCamera?.();
    if (!activeCamera) {
      cameraSyncBaseline = null;
      return;
    }

    const focalPoint = toFiniteVector3(activeCamera.getFocalPoint?.());
    const position = toFiniteVector3(activeCamera.getPosition?.());
    const rawViewUp = toFiniteVector3(activeCamera.getViewUp?.());
    const parallelScale = Number(activeCamera.getParallelScale?.());

    if (
      !focalPoint ||
      !position ||
      !rawViewUp ||
      !Number.isFinite(parallelScale) ||
      parallelScale <= 0
    ) {
      cameraSyncBaseline = null;
      return;
    }

    const viewDir = normalize3(subtract3(focalPoint, position), [0, 0, 1]);
    let viewUp = normalize3(rawViewUp, [0, 1, 0]);
    const viewRight = normalize3(cross3(viewDir, viewUp), [1, 0, 0]);
    viewUp = normalize3(cross3(viewRight, viewDir), [0, 1, 0]);

    cameraSyncBaseline = {
      focalPoint,
      position,
      viewUp,
      parallelScale,
    };
  };

  const clearCameraSyncBaseline = () => {
    cameraSyncBaseline = null;
  };

  const resizeHostedRenderWindow = (resetCameraToFit = false) => {
    if (disposed) {
      return;
    }

    genericRenderWindow?.resize?.();
    applyPointerEventPassthrough();
    applyLayerSandwich();

    if (resetCameraToFit) {
      resetHostedCamera();
      captureCameraSyncBaseline();
    } else {
      renderer?.resetCameraClippingRange?.();
    }

    renderWindow?.render?.();
  };

  const syncHostedCamera = (
    syncState:
      | {
          panDeltaPx?: ArrayLike<number> | null;
          zoomRatio?: number | null;
          viewportHeightPx?: number | null;
        }
      | null
      | undefined
  ) => {
    if (disposed || !syncState) {
      return;
    }

    const activeCamera = renderer?.getActiveCamera?.();
    if (!activeCamera || !cameraSyncBaseline) {
      return;
    }

    const panDeltaPx = toFiniteVector2(syncState.panDeltaPx);
    const zoomRatio = Number(syncState.zoomRatio);
    const viewportHeightPx = Number(syncState.viewportHeightPx);

    if (!panDeltaPx || !Number.isFinite(zoomRatio) || zoomRatio <= 0) {
      return;
    }

    const safeViewportHeightPx = Math.max(
      1,
      Number.isFinite(viewportHeightPx)
        ? viewportHeightPx
        : Number(hostContainer?.clientHeight || viewportElement.clientHeight || 1)
    );
    const nextParallelScale = cameraSyncBaseline.parallelScale / zoomRatio;
    const worldUnitsPerPixel = (2 * nextParallelScale) / safeViewportHeightPx;
    const viewDir = normalize3(
      subtract3(cameraSyncBaseline.focalPoint, cameraSyncBaseline.position),
      [0, 0, 1]
    );
    const viewRight = normalize3(cross3(viewDir, cameraSyncBaseline.viewUp), [1, 0, 0]);
    const viewUp = normalize3(cross3(viewRight, viewDir), cameraSyncBaseline.viewUp);
    const translation = add3(
      scale3(viewRight, -panDeltaPx[0] * worldUnitsPerPixel),
      scale3(viewUp, panDeltaPx[1] * worldUnitsPerPixel)
    );
    const nextFocalPoint = add3(cameraSyncBaseline.focalPoint, translation);
    const nextPosition = add3(cameraSyncBaseline.position, translation);

    activeCamera.setParallelProjection?.(true);
    activeCamera.setViewUp(
      cameraSyncBaseline.viewUp[0],
      cameraSyncBaseline.viewUp[1],
      cameraSyncBaseline.viewUp[2]
    );
    activeCamera.setFocalPoint(nextFocalPoint[0], nextFocalPoint[1], nextFocalPoint[2]);
    activeCamera.setPosition(nextPosition[0], nextPosition[1], nextPosition[2]);
    activeCamera.setParallelScale(nextParallelScale);

    renderer?.resetCameraClippingRange?.();
    renderWindow?.render?.();
  };

  const disposeConstructed = () => {
    if (disposed) {
      return;
    }

    disposed = true;

    if (resizeAnimationFrame !== null) {
      window.cancelAnimationFrame(resizeAnimationFrame);
      resizeAnimationFrame = null;
    }

    resizeObserver?.disconnect();
    resizeObserver = null;
    domLayerObserver?.disconnect();
    domLayerObserver = null;

    if (viewportElement.__cprVtkPanoHost?.actorUID === actorUID) {
      delete viewportElement.__cprVtkPanoHost;
    }

    if (renderer && imageSlice) {
      renderer.removeActor?.(imageSlice);
      renderWindow?.render?.();
    }

    genericRenderWindow?.setContainer?.(null);
    deleteVtkObject(genericRenderWindow);

    if (hostContainer?.parentNode) {
      hostContainer.parentNode.removeChild(hostContainer);
    }
    restoreManagedStyles();

    if (restoreViewportPosition !== null && viewportElement.style.position === 'relative') {
      viewportElement.style.position = restoreViewportPosition;
    }

    deleteVtkObject(centerlinePoints);
    deleteVtkObject(centerlineLines);
    deleteVtkObject(orientationArray);
    deleteVtkObject(centerline);
    deleteVtkObject(mapper);
    deleteVtkObject(imageSlice);
    deleteVtkObject(normalizedPanoSource?.imageData);
    normalizedPanoSource = null;
    clearCameraSyncBaseline();
    args.viewport.render();
  };

  try {
    normalizedPanoSource = createNormalizedHuPanoSource(sourceVolume, runLabel, args.projectionMode);

    mapper.setImageData(normalizedPanoSource.imageData);
    mapper.setCenterlineData(centerline as vtkPolyData);
    mapper.setTangentDirection([1, 0, 0]);
    mapper.setBitangentDirection([0, 1, 0]);
    mapper.setNormalDirection([0, 0, 1]);
    mapper.setOrientationArrayName('Orientation');
    mapper.setUseUniformOrientation(false);
    mapper.useStraightenedMode();
    mapper.setProjectionMode(toProjectionMode(args.projectionMode));
    mapper.setProjectionSlabThickness(totalSlabThicknessMm);
    mapper.setProjectionSlabNumberOfSamples(slabSamples);
    mapper.setWidth(totalWidthMm);

    imageSlice.setMapper(mapper as unknown as Parameters<typeof imageSlice.setMapper>[0]);
    const property = imageSlice.getProperty();
    property.setRGBTransferFunction(0, null);
    property.setUseLookupTableScalarRange(false);
    console.log(
      `DIAG-TRIPWIRE: vtk-property-apply run=${runLabel} isPreScaled=${sourceIsPreScaled} normalizedHuSource=true finalWindowWidth=${appliedWindowWidth} finalWindowCenter=${appliedWindowCenter}`
    );
    property.setColorWindow(appliedWindowWidth);
    property.setColorLevel(appliedWindowCenter);

    if (window.getComputedStyle(viewportElement).position === 'static') {
      restoreViewportPosition = viewportElement.style.position;
      viewportElement.style.position = 'relative';
    }

    hostContainer = document.createElement('div');
    hostContainer.setAttribute(VTK_PANO_HOST_ATTRIBUTE, 'true');
    hostContainer.style.position = 'absolute';
    hostContainer.style.inset = '0';
    hostContainer.style.pointerEvents = 'none';
    hostContainer.style.background = '#000';
    hostContainer.style.zIndex = '1';

    const internalViewportElement = getInternalViewportElement();
    const nativeCanvas = getNativeCornerstoneCanvas(internalViewportElement);
    if (!internalViewportElement || !nativeCanvas) {
      throw new Error(
        '[CPR] cpr-pano internal Cornerstone canvas is not available for hosted VTK pano insertion.'
      );
    }

    const internalChildren = Array.from(internalViewportElement.children);
    const firstOverlayLikeChild =
      internalChildren.find(
        child => child !== hostContainer && child !== nativeCanvas && isOverlayLikeElement(child)
      ) || null;
    const insertionReference = firstOverlayLikeChild || nativeCanvas.nextSibling;

    internalViewportElement.insertBefore(hostContainer, insertionReference);
    console.log(
      `DIAG-TRIPWIRE: vtk-dom-inserted. vtkContainer is firstChild: ${
        internalViewportElement.firstChild === hostContainer
      } hostAfterCanvas=${hostContainer.previousSibling === nativeCanvas} childOrder=${Array.from(
        internalViewportElement.childNodes
      )
        .map(node => {
          if (node === hostContainer) {
            return 'vtk-host';
          }

          if (node instanceof Element) {
            const className = getElementClassName(node);
            return `${node.tagName.toLowerCase()}${
              className ? `.${String(className).replace(/\s+/g, '.')}` : ''
            }`;
          }

          return node.nodeName;
        })
        .join(' > ')}`
    );

    genericRenderWindow = vtkGenericRenderWindow.newInstance({
      background: [0, 0, 0],
      listenWindowResize: false,
    });
    genericRenderWindow.setContainer(hostContainer);
    renderer = genericRenderWindow.getRenderer();
    renderWindow = genericRenderWindow.getRenderWindow();
    applyPointerEventPassthrough();
    applyLayerSandwich();
    startDomLayerObserver();

    renderer.addActor(imageSlice);
    resizeHostedRenderWindow(true);
    logLayerStack();
    args.viewport.render();
    console.log('MAPPED RANGE:', normalizedPanoSource.scalarRange);
    console.log(
      `DIAG-TRIPWIRE: vtk-scalar-range run=${runLabel} sourceRange=${JSON.stringify(
        sourceScalarRange
      )} normalizedRange=${JSON.stringify(normalizedPanoSource.scalarRange)} sourceDataType=${sourceDataType} normalizedDataType=${
        normalizedPanoSource.dataType
      } isPreScaled=${sourceIsPreScaled} effectiveIsPreScaled=${normalizedPanoSource.effectiveIsPreScaled} heuristicOverride=${
        normalizedPanoSource.heuristicOverride
      } storedValueNormalizationApplied=${normalizedPanoSource.storedValueNormalizationApplied} normalizationSignature=${
        normalizedPanoSource.normalizationSignature ?? 'none'
      } unsignedPackedArtifactDetected=${normalizedPanoSource.unsignedPackedArtifactDetected}`
    );
    console.log('CPR-VTK-PANO-RANGE', {
      runId: runLabel,
      sourceVolumeId: args.sourceVolumeId,
      sourceScalarRange: normalizedPanoSource.sourceScalarRange,
      normalizedScalarRange: normalizedPanoSource.scalarRange,
      sourceDataType,
      normalizedDataType: normalizedPanoSource.dataType,
      sourceIsPreScaled,
      effectiveIsPreScaled: normalizedPanoSource.effectiveIsPreScaled,
      heuristicOverride: normalizedPanoSource.heuristicOverride,
      storedValueNormalizationApplied: normalizedPanoSource.storedValueNormalizationApplied,
      normalizationSignature: normalizedPanoSource.normalizationSignature,
      unsignedPackedArtifactDetected: normalizedPanoSource.unsignedPackedArtifactDetected,
      sourceRescaleSlope,
      sourceRescaleIntercept,
      projectionMode: getProjectionModeLabel(args.projectionMode),
      requestedWindowWidth: initialWindowWidth,
      requestedWindowCenter: initialWindowCenter,
      appliedWindowWidth,
      appliedWindowCenter,
      translatedToStoredWindowLevel: false,
      normalizedHuSource: true,
      averageProjectionFilter: normalizedPanoSource.averageProjectionFilter,
    });

    viewportElement.__cprVtkPanoHost = {
      actorUID,
      resetCamera: () => {
        if (disposed) {
          return;
        }

        resizeHostedRenderWindow(true);
      },
      render: () => {
        if (disposed) {
          return;
        }

        resizeHostedRenderWindow(false);
      },
      resize: () => {
        resizeHostedRenderWindow(false);
      },
      captureCameraSyncBaseline: () => {
        captureCameraSyncBaseline();
      },
      syncCamera: syncState => {
        syncHostedCamera(syncState);
      },
      clearCameraSyncBaseline: () => {
        clearCameraSyncBaseline();
      },
      dispose: () => {
        if (disposed) {
          return;
        }

        console.log(
          `CPR-VTK-PANO-DETACH run=${runLabel} actorUID=${actorUID} viewportClass=${
            args.viewport.constructor?.name || 'unknown'
          } mode=hosted-child-render-window`
        );
        disposeConstructed();
      },
    };

    if (typeof ResizeObserver === 'function') {
      resizeObserver = new ResizeObserver(() => {
        resizeHostedRenderWindow(false);
      });
      resizeObserver.observe(getInternalViewportElement() || viewportElement);
    }

    resizeAnimationFrame = window.requestAnimationFrame(() => {
      resizeHostedRenderWindow(false);
    });

    console.log(
      `CPR-VTK-PANO-ACTOR run=${runLabel} actorUID=${actorUID} actorClass=${
        imageSlice.getClassName?.() || 'unknown'
      } mapperClass=${mapper.getClassName?.() || 'unknown'} projectionMode=${getProjectionModeLabel(
        args.projectionMode
      )} slabThicknessMm=${totalSlabThicknessMm.toFixed(2)} slabSamples=${slabSamples} widthMm=${totalWidthMm.toFixed(
        2
      )} sourceVolumeId=${args.sourceVolumeId} mode=hosted-child-render-window`
    );

    return {
      actorUID,
      updateWindowLevel(windowWidth: number, windowCenter: number): void {
        if (disposed) {
          return;
        }

        const nextWindowWidth = requirePositiveNumber(windowWidth, 'windowWidth');
        const nextWindowCenter = requireFiniteNumber(windowCenter, 'windowCenter');
        const nextAppliedWindowWidth = nextWindowWidth;
        const nextAppliedWindowCenter = nextWindowCenter;
        console.log(
          `DIAG-TRIPWIRE: vtk-live-wl run=${runLabel} rawWindowWidth=${nextWindowWidth} rawWindowCenter=${nextWindowCenter} slope=${sourceRescaleSlope} intercept=${sourceRescaleIntercept} normalizedHuWindowWidth=${nextAppliedWindowWidth} normalizedHuWindowCenter=${nextAppliedWindowCenter}`
        );
        const property = imageSlice.getProperty();
        property.setRGBTransferFunction(0, null);
        property.setUseLookupTableScalarRange(false);
        console.log(
          `DIAG-TRIPWIRE: vtk-property-apply-live run=${runLabel} isPreScaled=${sourceIsPreScaled} normalizedHuSource=true finalWindowWidth=${nextAppliedWindowWidth} finalWindowCenter=${nextAppliedWindowCenter}`
        );
        property.setColorWindow(nextAppliedWindowWidth);
        property.setColorLevel(nextAppliedWindowCenter);
        resizeHostedRenderWindow(false);
      },
      captureCameraSyncBaseline(): void {
        captureCameraSyncBaseline();
      },
      syncCamera(syncState): void {
        syncHostedCamera(syncState);
      },
      clearCameraSyncBaseline(): void {
        clearCameraSyncBaseline();
      },
      resize(): void {
        resizeHostedRenderWindow(false);
      },
      dispose(): void {
        if (disposed) {
          return;
        }

        viewportElement.__cprVtkPanoHost?.dispose();
        if (!disposed) {
          disposeConstructed();
        }
      },
    };
  } catch (error) {
    disposeConstructed();
    throw error;
  }
}
