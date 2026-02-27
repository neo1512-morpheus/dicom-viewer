import { getEnabledElement } from '@cornerstonejs/core';
import { AnnotationDisplayTool, drawing, utilities as cstUtils } from '@cornerstonejs/tools';

import { cprStateService } from '../../../../modes/cpr/src/CPRStateService';
import { emitCPRCrossSectionSync } from '../../../../modes/cpr/src/cprEvents';

class CPRCursorTool extends AnnotationDisplayTool {
  static toolName = 'CPRCursor';

  private cursorByViewport = new Map<string, { frameIndex: number; canvasX: number }>();

  constructor(
    toolProps = {},
    defaultToolProps = {
      supportedInteractionTypes: ['Mouse'],
      configuration: {
        lineColor: 'rgb(0, 255, 0)',
        lineWidth: 1.5,
      },
    }
  ) {
    super(toolProps, defaultToolProps);
  }

  mouseMoveCallback = evt => {
    this._updateCursor(evt);
  };

  mouseDownCallback = evt => {
    this._updateCursor(evt);
  };

  mouseDragCallback = evt => {
    this._updateCursor(evt);
  };

  mouseUpCallback = evt => {
    this._updateCursor(evt);
  };

  renderAnnotation = (enabledElement, svgDrawingHelper) => {
    const { viewport } = enabledElement;
    const viewportId = viewport.id;
    const frames = cprStateService.getFrames();
    if (frames.length === 0) {
      return false;
    }

    let cursor = this.cursorByViewport.get(viewportId);
    if (!cursor) {
      cursor = { frameIndex: cprStateService.getCurrentFrameIndex(), canvasX: 0 };
      this.cursorByViewport.set(viewportId, cursor);
    }

    const line = this._getLineForFrame(viewport, cursor.frameIndex, frames.length);
    if (!line) {
      return false;
    }

    cursor.canvasX = line.canvasX;

    const annotationUID = `${this.getToolName()}-${viewportId}`;
    const lineUID = 'cursor-line';
    const lineOptions = {
      color: this.configuration.lineColor,
      width: this.configuration.lineWidth,
    };

    drawing.drawLine(
      svgDrawingHelper,
      annotationUID,
      lineUID,
      line.start,
      line.end,
      lineOptions
    );

    return true;
  };

  private _updateCursor(evt): void {
    const { element, currentPoints } = evt.detail;
    const canvasPoint = this._toCanvasPoint(currentPoints?.canvas);
    if (!element || !canvasPoint) {
      return;
    }

    const enabledElement = getEnabledElement(element);
    if (!enabledElement) {
      return;
    }

    const { viewport } = enabledElement;
    const viewportId = viewport.id;

    const frames = cprStateService.getFrames();
    if (frames.length < 2) {
      return;
    }

    const frameIndex = this._canvasToFrameIndex(
      viewport,
      element,
      canvasPoint as [number, number],
      frames.length
    );
    if (frameIndex == null) {
      return;
    }

    this.cursorByViewport.set(viewportId, {
      frameIndex,
      canvasX: canvasPoint[0],
    });
    cprStateService.setCurrentFrameIndex(frameIndex);
    emitCPRCrossSectionSync({
      frameIndex,
      viewportId,
    });

    cstUtils.triggerAnnotationRender(element);
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

  private _canvasToFrameIndex(
    viewport,
    element: HTMLDivElement,
    canvasPoint: [number, number],
    frameCount: number
  ): number | null {
    if (frameCount <= 1) {
      return 0;
    }

    let imageX = NaN;
    let imageWidth = NaN;

    try {
      const worldPoint = viewport.canvasToWorld(canvasPoint);
      const imageData = viewport.getImageData?.();
      const indexPoint = imageData?.imageData?.worldToIndex?.(worldPoint);

      if (indexPoint && Number.isFinite(indexPoint[0])) {
        imageX = Number(indexPoint[0]);
        imageWidth = Number(imageData?.dimensions?.[0]);
      }
    } catch {
      // Fallback mapping below.
    }

    if (!Number.isFinite(imageX) || !Number.isFinite(imageWidth) || imageWidth <= 1) {
      imageX = canvasPoint[0];
      imageWidth = element.clientWidth;
    }

    if (!Number.isFinite(imageWidth) || imageWidth <= 1) {
      return null;
    }

    const clampedX = Math.max(0, Math.min(imageX, imageWidth - 1));
    const normalized = clampedX / Math.max(1, imageWidth - 1);

    return Math.max(0, Math.min(Math.round(normalized * (frameCount - 1)), frameCount - 1));
  }

  private _getLineForFrame(
    viewport,
    frameIndex: number,
    frameCount: number
  ): { start: [number, number]; end: [number, number]; canvasX: number } | null {
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
        const topCanvas = viewport.worldToCanvas(topWorld);
        const bottomCanvas = viewport.worldToCanvas(bottomWorld);

        if (
          topCanvas &&
          bottomCanvas &&
          Number.isFinite(topCanvas[0]) &&
          Number.isFinite(topCanvas[1]) &&
          Number.isFinite(bottomCanvas[0]) &&
          Number.isFinite(bottomCanvas[1])
        ) {
          return {
            start: [topCanvas[0], topCanvas[1]],
            end: [bottomCanvas[0], bottomCanvas[1]],
            canvasX: topCanvas[0],
          };
        }
      } catch {
        // Fallback below.
      }
    }

    const element = viewport.element as HTMLDivElement;
    if (!element) {
      return null;
    }

    const fallbackX = normalized * Math.max(1, element.clientWidth - 1);
    return {
      start: [fallbackX, 0],
      end: [fallbackX, element.clientHeight],
      canvasX: fallbackX,
    };
  }
}

export default CPRCursorTool;
