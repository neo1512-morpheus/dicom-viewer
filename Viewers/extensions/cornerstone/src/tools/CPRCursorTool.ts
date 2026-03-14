import { getEnabledElement } from '@cornerstonejs/core';
import {
  AnnotationTool,
  Enums,
  annotation as cstAnnotation,
  cursors,
  drawing,
  state as cstState,
  utilities as cstUtils,
} from '@cornerstonejs/tools';

import { cprStateService } from '../../../../modes/cpr/src/CPRStateService';
import { emitCPRCrossSectionSync } from '../../../../modes/cpr/src/cprEvents';

const { hideElementCursor, resetElementCursor } = cursors.elementCursor;

type CanvasPoint = [number, number];
type CursorAnnotation = any;

class CPRCursorTool extends AnnotationTool {
  static toolName = 'CPRCursor';

  private editData: { annotation: CursorAnnotation } | null = null;

  constructor(
    toolProps = {},
    defaultToolProps = {
      supportedInteractionTypes: ['Mouse'],
      configuration: {
        lineColor: 'rgb(0, 255, 0)',
        lineWidth: 3,
        hitTolerance: 25,
      },
    }
  ) {
    super(toolProps, defaultToolProps);
  }

  filterInteractableAnnotationsForElement = (element, annotations) => {
    const enabledElement = getEnabledElement(element);
    const viewportId = enabledElement?.viewport?.id;

    if (!viewportId || !Array.isArray(annotations)) {
      return [];
    }

    return annotations.filter(annotation => annotation?.metadata?.viewportId === viewportId);
  };

  addNewAnnotation = evt => {
    const { element } = evt.detail;
    const annotation = element ? this._getOrCreateCursorAnnotation(element) : null;

    if (!annotation) {
      return null;
    }

    this.toolSelectedCallback(evt, annotation, 'Mouse');
    return annotation;
  };

  cancel = (element: HTMLDivElement) => {
    if (!this.editData) {
      return;
    }

    const { annotation } = this.editData;
    annotation.highlighted = false;
    if (annotation?.data?.handles) {
      annotation.data.handles.activeHandleIndex = null;
    }

    this._deactivateModify(element);
    resetElementCursor(element);
    this.editData = null;
    cstUtils.triggerAnnotationRender(element);

    return annotation.annotationUID;
  };

  handleSelectedCallback = evt => {
    this._beginInteraction(evt, this._getOrCreateCursorAnnotation(evt.detail.element));
  };

  toolSelectedCallback = evt => {
    this._beginInteraction(evt, this._getOrCreateCursorAnnotation(evt.detail.element));
  };

  isPointNearTool = (element, annotation, canvasCoords, proximity) => {
    const enabledElement = getEnabledElement(element);
    const frames = cprStateService.getFrames();

    if (!enabledElement || !annotation || frames.length === 0) {
      return false;
    }

    const line = this._getLineForFrame(
      enabledElement.viewport,
      cprStateService.getCurrentFrameIndex(),
      frames.length
    );
    if (!line) {
      return false;
    }

    const tolerance = Math.max(
      Number.isFinite(proximity) ? proximity : 0,
      Number(this.configuration.hitTolerance) || 0
    );

    return this._distanceToSegment(canvasCoords, line.start, line.end) <= tolerance;
  };

  getHandleNearImagePoint = (element, annotation, canvasCoords, proximity) => {
    if (!this.isPointNearTool(element, annotation, canvasCoords, proximity, 'mouse')) {
      if (annotation?.data?.handles) {
        annotation.data.handles.activeHandleIndex = null;
      }
      return;
    }

    if (annotation?.data?.handles) {
      annotation.data.handles.activeHandleIndex = 0;
      return annotation.data.handles.points?.[0] || ([0, 0, 0] as any);
    }
  };

  renderAnnotation = (enabledElement, svgDrawingHelper) => {
    const { viewport } = enabledElement;
    const frames = cprStateService.getFrames();
    const element = viewport.element as HTMLDivElement;

    if (!element || frames.length === 0) {
      return false;
    }

    const annotation = this._getOrCreateCursorAnnotation(element);
    if (!annotation) {
      return false;
    }

    const line = this._getLineForFrame(viewport, cprStateService.getCurrentFrameIndex(), frames.length);
    if (!line) {
      return false;
    }

    if (annotation?.data?.handles?.points) {
      annotation.data.handles.points[0] = line.centerWorld;
    }

    const lineOptions = {
      color: this.configuration.lineColor,
      width: this.configuration.lineWidth,
    };

    drawing.drawLine(
      svgDrawingHelper,
      annotation.annotationUID,
      'cursor-line',
      line.start,
      line.end,
      lineOptions
    );

    return true;
  };

  private _beginInteraction(evt, annotation: CursorAnnotation | null): void {
    const { element, currentPoints } = evt.detail;
    const canvasPoint = this._toCanvasPoint(currentPoints?.canvas);

    if (!annotation || !element || !canvasPoint) {
      return;
    }

    const updated = this._updateCursorFromCanvasPoint(element, canvasPoint);
    if (!updated) {
      return;
    }

    annotation.highlighted = true;
    if (annotation?.data?.handles) {
      annotation.data.handles.activeHandleIndex = 0;
    }

    this.editData = { annotation };
    this._activateModify(element);
    hideElementCursor(element);
    cstUtils.triggerAnnotationRender(element);
    evt.preventDefault();
  }

  private _dragCallback = evt => {
    if (!this.editData) {
      return;
    }

    const { element, currentPoints } = evt.detail;
    const canvasPoint = this._toCanvasPoint(currentPoints?.canvas);
    if (!element || !canvasPoint) {
      return;
    }

    if (!this._updateCursorFromCanvasPoint(element, canvasPoint)) {
      return;
    }

    this.editData.annotation.highlighted = true;
    evt.preventDefault();
    cstUtils.triggerAnnotationRender(element);
  };

  private _endCallback = evt => {
    const { element } = evt.detail;
    if (!element || !this.editData) {
      return;
    }

    const { annotation } = this.editData;
    annotation.highlighted = false;
    if (annotation?.data?.handles) {
      annotation.data.handles.activeHandleIndex = null;
    }

    this._deactivateModify(element);
    resetElementCursor(element);
    this.editData = null;
    evt.preventDefault();
    cstUtils.triggerAnnotationRender(element);
  };

  private _activateModify(element: HTMLDivElement): void {
    cstState.isInteractingWithTool = true;

    element.addEventListener(Enums.Events.MOUSE_UP, this._endCallback as EventListener);
    element.addEventListener(Enums.Events.MOUSE_DRAG, this._dragCallback as EventListener);
    element.addEventListener(Enums.Events.MOUSE_CLICK, this._endCallback as EventListener);
  }

  private _deactivateModify(element: HTMLDivElement): void {
    cstState.isInteractingWithTool = false;

    element.removeEventListener(Enums.Events.MOUSE_UP, this._endCallback as EventListener);
    element.removeEventListener(Enums.Events.MOUSE_DRAG, this._dragCallback as EventListener);
    element.removeEventListener(Enums.Events.MOUSE_CLICK, this._endCallback as EventListener);
  }

  private _getCursorAnnotation(element: HTMLDivElement): CursorAnnotation | null {
    const enabledElement = getEnabledElement(element);
    const viewportId = enabledElement?.viewport?.id;
    if (!viewportId) {
      return null;
    }

    const annotations = cstAnnotation.state.getAnnotations(this.getToolName(), element) || [];

    return annotations.find(annotation => annotation?.metadata?.viewportId === viewportId) || null;
  }

  private _getOrCreateCursorAnnotation(element: HTMLDivElement): CursorAnnotation | null {
    const existingAnnotation = this._getCursorAnnotation(element);
    if (existingAnnotation) {
      return existingAnnotation;
    }

    const enabledElement = getEnabledElement(element);
    if (!enabledElement) {
      return null;
    }

    const { viewport } = enabledElement;
    const toolClass = this.constructor as typeof AnnotationTool;
    const annotation = toolClass.createAnnotationForViewport(viewport, {
      metadata: {
        viewportId: viewport.id,
      },
      data: {
        viewportId: viewport.id,
        handles: {
          points: [[0, 0, 0]],
        },
      },
    });

    cstAnnotation.state.addAnnotation(annotation, element);
    return annotation;
  }

  private _updateCursorFromCanvasPoint(element: HTMLDivElement, canvasPoint: CanvasPoint): boolean {
    const enabledElement = getEnabledElement(element);
    if (!enabledElement) {
      return false;
    }

    const { viewport } = enabledElement;
    const viewportId = viewport.id;
    const frames = cprStateService.getFrames();
    if (frames.length < 2) {
      return false;
    }

    const frameIndex = this._canvasToFrameIndex(viewport, canvasPoint, frames.length);
    if (frameIndex == null) {
      return false;
    }

    const previousFrameIndex = cprStateService.getCurrentFrameIndex();
    cprStateService.setCurrentFrameIndex(frameIndex);

    if (frameIndex !== previousFrameIndex) {
      emitCPRCrossSectionSync({
        frameIndex,
        viewportId,
      });
    }

    return true;
  }

  private _toCanvasPoint(point: unknown): [number, number] | null {
    if (Array.isArray(point) && point.length >= 2) {
      const x = Number(point[0]);
      const y = Number(point[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
    }

    if (point && typeof point === 'object') {
      const maybePoint = point as { x?: number; y?: number };
      const x = Number(maybePoint.x);
      const y = Number(maybePoint.y);
      return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
    }

    return null;
  }

  private _canvasToImagePoint(
    viewport,
    canvasPoint: CanvasPoint
  ): {
    imageX: number;
    imageY: number;
    imageWidth: number;
    imageHeight: number;
  } | null {
    const imageData = viewport.getImageData?.();
    const dimensions = imageData?.dimensions;
    const worldToIndex = imageData?.imageData?.worldToIndex;

    if (
      !dimensions ||
      dimensions.length < 2 ||
      !Number.isFinite(dimensions[0]) ||
      !Number.isFinite(dimensions[1]) ||
      dimensions[0] <= 1 ||
      dimensions[1] <= 1 ||
      typeof worldToIndex !== 'function'
    ) {
      return null;
    }

    let worldPoint;
    let indexPoint;

    try {
      worldPoint = viewport.canvasToWorld(canvasPoint);
      indexPoint = worldToIndex(worldPoint);
    } catch {
      return null;
    }

    const imageX = Number(indexPoint?.[0]);
    const imageY = Number(indexPoint?.[1]);
    const imageWidth = Number(dimensions[0]);
    const imageHeight = Number(dimensions[1]);

    if (
      !Number.isFinite(imageX) ||
      !Number.isFinite(imageY) ||
      !Number.isFinite(imageWidth) ||
      !Number.isFinite(imageHeight)
    ) {
      return null;
    }

    const epsilon = 1e-3;
    if (
      imageX < -epsilon ||
      imageX > imageWidth - 1 + epsilon ||
      imageY < -epsilon ||
      imageY > imageHeight - 1 + epsilon
    ) {
      return null;
    }

    return {
      imageX: Math.max(0, Math.min(imageX, imageWidth - 1)),
      imageY: Math.max(0, Math.min(imageY, imageHeight - 1)),
      imageWidth,
      imageHeight,
    };
  }

  private _canvasToFrameIndex(
    viewport,
    canvasPoint: CanvasPoint,
    frameCount: number
  ): number | null {
    const imagePoint = this._canvasToImagePoint(viewport, canvasPoint);
    if (frameCount <= 1) {
      return 0;
    }

    let normalized: number | null = null;

    if (imagePoint) {
      normalized = imagePoint.imageX / Math.max(1, imagePoint.imageWidth - 1);
    } else {
      const viewportSize = this._getViewportCanvasSize(viewport);
      if (!viewportSize || viewportSize.width <= 1) {
        return null;
      }

      normalized = canvasPoint[0] / Math.max(1, viewportSize.width - 1);
    }

    return Math.max(0, Math.min(Math.round(normalized * (frameCount - 1)), frameCount - 1));
  }

  private _getLineForFrame(
    viewport,
    frameIndex: number,
    frameCount: number
  ): {
    start: CanvasPoint;
    end: CanvasPoint;
    canvasX: number;
    centerWorld: [number, number, number];
  } | null {
    if (frameCount <= 0) {
      return null;
    }

    const normalized =
      frameCount <= 1 ? 0 : Math.max(0, Math.min(frameIndex, frameCount - 1)) / (frameCount - 1);

    const imageData = viewport.getImageData?.();
    const dimensions = imageData?.dimensions;
    const worldFromIndex = imageData?.imageData?.indexToWorld;

    if (
      dimensions &&
      dimensions.length >= 2 &&
      Number.isFinite(dimensions[0]) &&
      Number.isFinite(dimensions[1]) &&
      dimensions[0] > 1 &&
      dimensions[1] > 1 &&
      typeof worldFromIndex === 'function'
    ) {
      try {
        const imageX = normalized * (dimensions[0] - 1);
        const topWorld = worldFromIndex([imageX, 0, 0]);
        const bottomWorld = worldFromIndex([imageX, dimensions[1] - 1, 0]);
        const centerWorld = worldFromIndex([imageX, (dimensions[1] - 1) / 2, 0]);
        const topCanvas = viewport.worldToCanvas(topWorld);
        const bottomCanvas = viewport.worldToCanvas(bottomWorld);

        if (
          topCanvas &&
          bottomCanvas &&
          centerWorld &&
          Number.isFinite(topCanvas[0]) &&
          Number.isFinite(topCanvas[1]) &&
          Number.isFinite(bottomCanvas[0]) &&
          Number.isFinite(bottomCanvas[1])
        ) {
          return {
            start: [topCanvas[0], topCanvas[1]],
            end: [bottomCanvas[0], bottomCanvas[1]],
            canvasX: topCanvas[0],
            centerWorld: [centerWorld[0], centerWorld[1], centerWorld[2]],
          };
        }
      } catch {
        // Fall back to element-relative coordinates when the hosted VTK pano
        // does not expose usable Cornerstone image/world transforms.
      }
    }

    const viewportSize = this._getViewportCanvasSize(viewport);
    if (!viewportSize) {
      return null;
    }

    const canvasX = normalized * Math.max(1, viewportSize.width - 1);

    return {
      start: [canvasX, 0],
      end: [canvasX, viewportSize.height],
      canvasX,
      centerWorld: [0, 0, 0],
    };
  }

  private _getViewportCanvasSize(
    viewport
  ): {
    width: number;
    height: number;
  } | null {
    const element = viewport?.element as HTMLDivElement | undefined;
    const rect = element?.getBoundingClientRect?.();
    const width = Number(rect?.width ?? element?.clientWidth);
    const height = Number(rect?.height ?? element?.clientHeight);

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 1 || height <= 1) {
      return null;
    }

    return { width, height };
  }

  private _distanceToSegment(point: CanvasPoint, start: CanvasPoint, end: CanvasPoint): number {
    const [px, py] = point;
    const [x1, y1] = start;
    const [x2, y2] = end;
    const dx = x2 - x1;
    const dy = y2 - y1;

    if (dx === 0 && dy === 0) {
      return Math.hypot(px - x1, py - y1);
    }

    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    const closestX = x1 + t * dx;
    const closestY = y1 + t * dy;

    return Math.hypot(px - closestX, py - closestY);
  }
}

export default CPRCursorTool;
