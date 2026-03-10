import { PubSubService } from '@ohif/core';
// @ts-ignore
import * as OhifTypes from '@ohif/core/types';
import {
  RenderingEngine,
  StackViewport,
  Types,
  getRenderingEngine,
  utilities as csUtils,
  VolumeViewport,
  VolumeViewport3D,
  cache,
  Enums as csEnums,
  BaseVolumeViewport,
  eventTarget,
  metaData,
} from '@cornerstonejs/core';

import { utilities as csToolsUtils, Enums as csToolsEnums, annotation } from '@cornerstonejs/tools';
import { IViewportService } from './IViewportService';
import { RENDERING_ENGINE_ID } from './constants';
import ViewportInfo, { DisplaySetOptions, PublicViewportOptions } from './Viewport';
import { StackViewportData, VolumeViewportData } from '../../types/CornerstoneCacheService';
import { LutPresentation, PositionPresentation, Presentations } from '../../types/Presentation';

import JumpPresets from '../../utils/JumpPresets';

const EVENTS = {
  VIEWPORT_DATA_CHANGED: 'event::cornerstoneViewportService:viewportDataChanged',
  VIEWPORT_VOLUMES_CHANGED: 'event::cornerstoneViewportService:viewportVolumesChanged',
  VOLUME_LOADING_PROGRESS: 'event::cornerstoneViewportService:volumeLoadingProgress',
};

/**
 * Handles cornerstone viewport logic including enabling, disabling, and
 * updating the viewport.
 */
// @ts-ignore
class CornerstoneViewportService extends PubSubService implements IViewportService {
  static REGISTRATION = {
    name: 'cornerstoneViewportService',
    altName: 'CornerstoneViewportService',
    create: ({
      servicesManager,
    }: OhifTypes.Extensions.ExtensionParams): CornerstoneViewportService => {
      return new CornerstoneViewportService(servicesManager);
    },
  };

  renderingEngine: Types.IRenderingEngine | null;
  viewportsById: Map<string, ViewportInfo> = new Map();
  viewportGridResizeObserver: ResizeObserver | null;
  viewportsDisplaySets: Map<string, string[]> = new Map();
  beforeResizePositionPresentations: Map<string, PositionPresentation> = new Map();

  // Some configs
  enableResizeDetector: true;
  resizeRefreshRateMs: 200;
  resizeRefreshMode: 'debounce';
  servicesManager: AppTypes.ServicesManager = null;

  resizeQueue = [];
  viewportResizeTimer = null;
  gridResizeDelay = 50;
  gridResizeTimeOut = null;

  // Strict hydration queue to prevent memory spikes for large volumes
  private static _hydrationQueue: Promise<void> = Promise.resolve();

  // Track volume loading progress for UI feedback
  volumeLoadingProgress: Map<string, {
    percentComplete: number,
    totalImages: number,
    loadedImages: number
  }> = new Map();

  constructor(servicesManager: AppTypes.ServicesManager) {
    super(EVENTS);
    this.renderingEngine = null;
    this.viewportGridResizeObserver = null;
    this.servicesManager = servicesManager;

    // Global listener for volume loading progress
    eventTarget.addEventListener('IMAGE_VOLUME_LOADING_PROGRESS', (evt: any) => {
      const { volumeId, framesLoaded, totalFrames } = evt.detail;
      const percentComplete = Math.round((framesLoaded / totalFrames) * 100);

      const prevProgress = this.volumeLoadingProgress.get(volumeId);
      const prevPercent = prevProgress ? prevProgress.percentComplete : -1;

      // Throttle: only update if percent changed and is a multiple of 25 (or first/last)
      if (percentComplete !== prevPercent && (percentComplete % 25 === 0 || percentComplete === 100)) {
        this.volumeLoadingProgress.set(volumeId, {
          percentComplete,
          loadedImages: framesLoaded,
          totalImages: totalFrames
        });

        this._broadcastEvent(EVENTS.VOLUME_LOADING_PROGRESS, {
          volumeId,
          percentComplete,
          loadedImages: framesLoaded,
          totalImages: totalFrames
        });
      }
    });

    // Clear progress on completion
    eventTarget.addEventListener('IMAGE_VOLUME_LOADING_COMPLETED', (evt: any) => {
      const { volumeId } = evt.detail;
      this.volumeLoadingProgress.delete(volumeId);
      this._broadcastEvent(EVENTS.VOLUME_LOADING_PROGRESS, { volumeId, percentComplete: 100 });
    });

    // =========================================================================
    // [GEMINI FIX] Metadata Proxy for Parallel Cache (#2d)
    // =========================================================================
    // The Measurement Service needs metadata (SeriesInstanceUID, etc.) to save annotations.
    // Since we appended '#2d' to the imageIDs for the Parallel Cache, standard lookups fail.
    // We register a high-priority provider (10000) to intercept these requests, strip the suffix,
    // and redirect them to the original image metadata.
    // =========================================================================
    metaData.addProvider((type: string, imageId: string) => {
      if (imageId && imageId.endsWith('#2d')) {
        const cleanImageId = imageId.replace('#2d', '');
        // Redirect to the original ID which exists in the store
        return metaData.get(type, cleanImageId);
      }
      // Return undefined to let standard providers handle non-#2d images
      return undefined;
    }, 10000);
    console.log('[GEMINI FIX] Registered #2d metadata proxy provider');

    // Note: Measurement label render fix is now in initMeasurementService.ts
    // (see MEASUREMENT_UPDATED subscriber with renderingEngine.render() call)
  }

  private isWebGLUnavailableError(error: unknown): boolean {
    const message =
      (typeof error === 'object' && error && 'message' in error
        ? String((error as { message?: unknown }).message)
        : String(error ?? '')) || '';

    const stack =
      (typeof error === 'object' && error && 'stack' in error
        ? String((error as { stack?: unknown }).stack)
        : '') || '';

    const text = `${message}\n${stack}`;

    return (
      text.includes('Cannot create proxy with a non-object as target or handler') ||
      text.includes('get3DContext') ||
      text.includes('getRenderWindow') ||
      text.includes('Cannot read properties of undefined (reading \'values\')') ||
      text.includes('WebGL') ||
      text.includes('RenderWindow.js') ||
      text.includes('RenderingEngine.ts:1093') ||
      text.includes('RenderingEngine.ts:1174')
    );
  }

  /**
   * Adds the HTML element to the viewportService
   * @param {*} viewportId
   * @param {*} elementRef
   */
  public enableViewport(viewportId: string, elementRef: HTMLDivElement): void {
    const viewportInfo = new ViewportInfo(viewportId);
    viewportInfo.setElement(elementRef);
    this.viewportsById.set(viewportId, viewportInfo);
  }

  public getViewportIds(): string[] {
    return Array.from(this.viewportsById.keys());
  }

  /**
   * It retrieves the renderingEngine if it does exist, or creates one otherwise
   * @returns {RenderingEngine} rendering engine
   */
  public getRenderingEngine() {
    // get renderingEngine from cache if it exists
    const renderingEngine = getRenderingEngine(RENDERING_ENGINE_ID);

    if (renderingEngine && !renderingEngine.hasBeenDestroyed) {
      this.renderingEngine = renderingEngine;
      return this.renderingEngine;
    }

    if (renderingEngine?.hasBeenDestroyed) {
      console.warn(
        '[CornerstoneViewportService] Found destroyed rendering engine in cache, recreating'
      );
    }

    this.renderingEngine = new RenderingEngine(RENDERING_ENGINE_ID);

    return this.renderingEngine;
  }

  /**
   * It triggers the resize on the rendering engine, and renders the viewports
   *
   * @param isGridResize - if the resize is triggered by a grid resize
   * this is used to avoid double resize of the viewports since if the
   * grid is resized, all viewports will be resized so there is no need
   * to resize them individually which will get triggered by their
   * individual resize observers
   */
  public resize(isGridResize = false) {
    // if there is a grid resize happening, it means the viewport grid
    // has been manipulated (e.g., panels closed, added, etc.) and we need
    // to resize all viewports, so we will add a timeout here to make sure
    // we don't double resize the viewports when viewports in the grid are
    // resized individually
    if (isGridResize) {
      this.performResize();
      this.resetGridResizeTimeout();
      this.resizeQueue = [];
      clearTimeout(this.viewportResizeTimer);
    } else {
      this.enqueueViewportResizeRequest();
    }
  }

  /**
   * Removes the viewport from cornerstone, and destroys the rendering engine
   */
  public destroy() {
    this._removeResizeObserver();
    this.viewportGridResizeObserver = null;
    try {
      this.renderingEngine?.destroy?.();
    } catch (e) {
      console.warn('Rendering engine not destroyed', e);
    }
    this.viewportsDisplaySets.clear();
    this.renderingEngine = null;
    cache.purgeCache();
  }

  /**
   * Disables the viewport inside the renderingEngine, if no viewport is left
   * it destroys the renderingEngine.
   *
   * This is called when the element goes away entirely - with new viewportId's
   * created for every new viewport, this will be called whenever the set of
   * viewports is changed, but NOT when the viewport position changes only.
   *
   * @param viewportId - The viewportId to disable
   */
  public disableElement(viewportId: string): void {
    try {
      const viewport = this.renderingEngine?.getViewport?.(viewportId);
      if (viewport) {
        this.renderingEngine?.disableElement(viewportId);
      }
    } catch (error) {
      console.warn(`[CornerstoneViewportService] Failed to disable viewport ${viewportId}`, error);
    }

    // clean up
    this.viewportsById.delete(viewportId);
    this.viewportsDisplaySets.delete(viewportId);
  }

  /**
   * Sets the presentations for a given viewport. Presentations is an object
   * that can define the lut or position for a viewport.
   *
   * @param viewportId - The ID of the viewport.
   * @param presentations - The presentations to apply to the viewport.
   */
  public setPresentations(viewportId: string, presentations?: Presentations): void {
    const viewport = this.getCornerstoneViewport(viewportId) as
      | Types.IStackViewport
      | Types.IVolumeViewport;

    if (!viewport) {
      return;
    }

    if (!presentations) {
      return;
    }

    const { lutPresentation, positionPresentation } = presentations;
    if (lutPresentation) {
      const { presentation } = lutPresentation;
      if (viewport instanceof BaseVolumeViewport) {
        if (presentation instanceof Map) {
          presentation.forEach((properties, volumeId) => {
            viewport.setProperties(properties, volumeId);
          });
        } else {
          viewport.setProperties(presentation);
        }
      } else {
        viewport.setProperties(presentation);
      }
    }

    if (positionPresentation) {
      const { viewPlaneNormal, viewUp, zoom, pan } = positionPresentation.presentation;
      viewport.setCamera({ viewPlaneNormal, viewUp });

      if (zoom !== undefined) {
        viewport.setZoom(zoom);
      }

      if (pan !== undefined) {
        viewport.setPan(pan);
      }
    }
  }

  /**
   * Retrieves the position presentation information for a given viewport.
   * @param viewportId The ID of the viewport.
   * @returns The position presentation object containing various properties
   * such as ID, viewport type, initial image index, view plane normal, view up, zoom, and pan.
   */
  public getPositionPresentation(viewportId: string): PositionPresentation {
    const viewportInfo = this.viewportsById.get(viewportId);
    if (!viewportInfo) {
      return;
    }

    const presentationIds = viewportInfo.getPresentationIds();

    if (!presentationIds) {
      return;
    }

    const { positionPresentationId } = presentationIds;

    const csViewport = this.getCornerstoneViewport(viewportId);
    if (!csViewport) {
      return;
    }

    const { viewPlaneNormal, viewUp } = csViewport.getCamera();
    const initialImageIndex = csViewport.getCurrentImageIdIndex() || 0;
    const zoom = csViewport.getZoom();
    const pan = csViewport.getPan();

    return {
      id: positionPresentationId,
      viewportType: viewportInfo.getViewportType(),
      presentation: {
        initialImageIndex,
        viewUp,
        viewPlaneNormal,
        zoom,
        pan,
      },
    };
  }

  /**
   * Retrieves the LUT (Lookup Table) presentation for a given viewport.
   * @param viewportId The ID of the viewport.
   * @returns The LUT presentation object, or undefined if the viewport does not exist.
   */
  public getLutPresentation(viewportId: string): LutPresentation {
    const viewportInfo = this.viewportsById.get(viewportId);
    if (!viewportInfo) {
      return;
    }

    const presentationIds = viewportInfo.getPresentationIds();

    if (!presentationIds) {
      return;
    }

    const { lutPresentationId } = presentationIds;

    const csViewport = this.getCornerstoneViewport(viewportId) as
      | Types.IStackViewport
      | Types.IVolumeViewport;

    if (!csViewport) {
      return;
    }

    const cleanProperties = properties => {
      if (properties.isComputedVOI) {
        delete properties.voiRange;
        delete properties.VOILUTFunction;
      }
      return properties;
    };

    const presentation =
      csViewport instanceof BaseVolumeViewport
        ? new Map()
        : cleanProperties(csViewport.getProperties());

    if (presentation instanceof Map) {
      csViewport.getActors().forEach(({ uid: volumeId }) => {
        const properties = cleanProperties(csViewport.getProperties(volumeId));
        presentation.set(volumeId, properties);
      });
    }

    return {
      id: lutPresentationId,
      viewportType: viewportInfo.getViewportType(),
      presentation,
    };
  }

  /**
   * Retrieves the presentations for a given viewport.
   * @param viewportId - The ID of the viewport.
   * @returns The presentations for the viewport.
   */
  public getPresentations(viewportId: string): Presentations {
    const viewportInfo = this.viewportsById.get(viewportId);
    if (!viewportInfo) {
      return;
    }

    const positionPresentation = this.getPositionPresentation(viewportId);
    const lutPresentation = this.getLutPresentation(viewportId);

    return {
      positionPresentation,
      lutPresentation,
    };
  }

  /**
   * Stores the presentation state for a given viewport inside the
   * stateSyncService. This is used to persist the presentation state
   * across different scenarios e.g., when the viewport is changing the
   * display set, or when the viewport is moving to a different layout.
   *
   * @param viewportId The ID of the viewport.
   */
  public storePresentation({ viewportId }) {
    let presentations = null as Presentations;
    try {
      presentations = this.getPresentations(viewportId);
      if (!presentations?.positionPresentation && !presentations?.lutPresentation) {
        return;
      }
    } catch (error) {
      console.warn(error);
      return;
    }

    const { stateSyncService, syncGroupService } = this.servicesManager.services;

    const synchronizers = syncGroupService.getSynchronizersForViewport(viewportId);

    const { positionPresentationStore, synchronizersStore, lutPresentationStore } =
      stateSyncService.getState();

    const { lutPresentation, positionPresentation } = presentations;
    const { id: positionPresentationId } = positionPresentation;
    const { id: lutPresentationId } = lutPresentation;

    const updateStore = (store, id, value) => ({ ...store, [id]: value });

    const newState = {} as { [key: string]: any };

    if (lutPresentationId) {
      newState.lutPresentationStore = updateStore(
        lutPresentationStore,
        lutPresentationId,
        lutPresentation
      );
    }

    if (positionPresentationId) {
      newState.positionPresentationStore = updateStore(
        positionPresentationStore,
        positionPresentationId,
        positionPresentation
      );
    }

    if (synchronizers?.length) {
      newState.synchronizersStore = updateStore(
        synchronizersStore,
        viewportId,
        synchronizers.map(synchronizer => ({
          id: synchronizer.id,
          sourceViewports: [...synchronizer.getSourceViewports()],
          targetViewports: [...synchronizer.getTargetViewports()],
        }))
      );
    }

    stateSyncService.store(newState);
  }

  /**
   * Sets the viewport data for a viewport.
   * @param viewportId - The ID of the viewport to set the data for.
   * @param viewportData - The viewport data to set.
   * @param publicViewportOptions - The public viewport options.
   * @param publicDisplaySetOptions - The public display set options.
   * @param presentations - The presentations to set.
   */
  public setViewportData(
    viewportId: string,
    viewportData: StackViewportData | VolumeViewportData,
    publicViewportOptions: PublicViewportOptions,
    publicDisplaySetOptions: DisplaySetOptions[],
    presentations?: Presentations
  ): void {
    let renderingEngine = this.getRenderingEngine();

    // This is the old viewportInfo, which may have old options but we might be
    // using its viewport (same viewportId as the new viewportInfo)
    const viewportInfo = this.viewportsById.get(viewportId);

    if (!viewportInfo) {
      // This can happen when setViewportData is deferred and the viewport unmounts first.
      console.warn(
        `[CornerstoneViewportService] Skipping setViewportData: viewport ${viewportId} is not enabled`
      );
      return;
    }

    // We should store the presentation for the current viewport since we can't only
    // rely to store it WHEN the viewport is disabled since we might keep around the
    // same viewport/element and just change the viewportData for it (drag and drop etc.)
    // the disableElement storePresentation handle would not be called in this case
    // and we would lose the presentation.
    this.storePresentation({ viewportId: viewportInfo.getViewportId() });

    // override the viewportOptions and displaySetOptions with the public ones
    // since those are the newly set ones, we set them here so that it handles defaults
    const displaySetOptions = viewportInfo.setPublicDisplaySetOptions(publicDisplaySetOptions);
    const viewportOptions = viewportInfo.setPublicViewportOptions(publicViewportOptions);

    const element = viewportInfo.getElement();
    const type = viewportInfo.getViewportType();
    const background = viewportInfo.getBackground();
    const orientation = viewportInfo.getOrientation();
    const displayArea = viewportInfo.getDisplayArea();

    const viewportInput: Types.PublicViewportInput = {
      viewportId,
      element,
      type,
      defaultOptions: {
        background,
        orientation,
        displayArea,
      },
    };

    // Rendering Engine Id set should happen before enabling the element
    // since there are callbacks that depend on the renderingEngine id
    // Todo: however, this is a limitation which means that we can't change
    // the rendering engine id for a given viewport which might be a super edge
    // case
    viewportInfo.setRenderingEngineId(renderingEngine.id);

    // Todo: this is not optimal at all, we are re-enabling the already enabled
    // element which is not what we want. But enabledElement as part of the
    // renderingEngine is designed to be used like this. This will trigger
    // ENABLED_ELEMENT again and again, which will run onEnableElement callbacks
    try {
      renderingEngine.enableElement(viewportInput);
    } catch (enableError) {
      if (this.isWebGLUnavailableError(enableError)) {
        return;
      }

      console.warn(
        `[CornerstoneViewportService] Failed to enable viewport ${viewportId} on existing engine, attempting recovery`,
        enableError
      );

      try {
        this.renderingEngine = new RenderingEngine(RENDERING_ENGINE_ID);
        renderingEngine = this.renderingEngine;
        renderingEngine.enableElement(viewportInput);
      } catch (recoveryError) {
        if (this.isWebGLUnavailableError(recoveryError)) {
          return;
        }

        console.warn(
          `[CornerstoneViewportService] Recovery failed while enabling viewport ${viewportId}`,
          recoveryError
        );
        return;
      }
    }

    viewportInfo.setViewportOptions(viewportOptions);
    viewportInfo.setDisplaySetOptions(displaySetOptions);
    viewportInfo.setViewportData(viewportData);
    viewportInfo.setViewportId(viewportId);

    this.viewportsById.set(viewportId, viewportInfo);

    // FIX: Prevent 3D/Volume settings from corrupting 2D/Stack viewports
    if (presentations?.lutPresentation) {
      const currentViewportType = viewportInfo.getViewportType(); // 'stack' or 'volume'

      // If the saved state is from a different viewport type (e.g. Volume -> Stack), discard it.
      if (presentations.lutPresentation.viewportType &&
        presentations.lutPresentation.viewportType !== currentViewportType) {
        console.log(
          `[CornerstoneViewportService] Discarding LUT presentation due to type mismatch: ${presentations.lutPresentation.viewportType} !== ${currentViewportType}`
        );
        presentations.lutPresentation = undefined;
      }
    }

    let viewport: Types.IViewport | null = null;
    try {
      viewport = renderingEngine.getViewport(viewportId);
    } catch (error) {
      console.warn(
        `[CornerstoneViewportService] Failed to fetch viewport ${viewportId} after enableElement`,
        error
      );
      return;
    }

    if (!viewport) {
      console.warn(
        `[CornerstoneViewportService] No viewport returned for ${viewportId} after enableElement`
      );
      return;
    }

    const displaySetPromise = this._setDisplaySets(
      viewport,
      viewportData,
      viewportInfo,
      presentations
    );

    // The broadcast event here ensures that listeners have a valid, up to date
    // viewport to access.  Doing it too early can result in exceptions or
    // invalid data.
    displaySetPromise
      .then(() => {
        this._broadcastEvent(this.EVENTS.VIEWPORT_DATA_CHANGED, {
          viewportData,
          viewportId,
        });

        // Batch render after viewport is fully initialized - allows progressive loading
        requestAnimationFrame(() => {
          try {
            renderingEngine.render();
          } catch (renderError) {
            if (!this.isWebGLUnavailableError(renderError)) {
              console.warn(
                `[CornerstoneViewportService] Failed to render viewport ${viewportId} after data change`,
                renderError
              );
            }
          }
        });
      })
      .catch(error => {
        if (!this.isWebGLUnavailableError(error)) {
          console.warn(
            `[CornerstoneViewportService] Failed while setting display sets for viewport ${viewportId}`,
            error
          );
        }
      });
  }

  /**
   * Retrieves the Cornerstone viewport with the specified ID.
   *
   * @param viewportId - The ID of the viewport.
   * @returns The Cornerstone viewport object if found, otherwise null.
   */
  public getCornerstoneViewport(viewportId: string): Types.IViewport | null {
    const viewportInfo = this.getViewportInfo(viewportId);

    if (!viewportInfo) {
      return null;
    }

    const renderingEngine =
      (getRenderingEngine(RENDERING_ENGINE_ID) as Types.IRenderingEngine | undefined) ??
      this.renderingEngine;

    if (!renderingEngine || renderingEngine.hasBeenDestroyed) {
      return null;
    }

    this.renderingEngine = renderingEngine;

    try {
      const viewport = renderingEngine.getViewport(viewportId);
      return viewport ?? null;
    } catch (error) {
      console.warn(
        `[CornerstoneViewportService] Failed to get viewport ${viewportId} from rendering engine`,
        error
      );
      return null;
    }
  }

  /**
   * Retrieves the viewport information for a given viewport ID. The viewport information
   * is the OHIF construct that holds different options and data for a given viewport and
   * is different from the cornerstone viewport.
   *
   * @param viewportId The ID of the viewport.
   * @returns The viewport information.
   */
  public getViewportInfo(viewportId: string): ViewportInfo {
    return this.viewportsById.get(viewportId);
  }

  /**
   * Looks through the viewports to see if the specified measurement can be
   * displayed in one of the viewports.
   *
   * @param measurement
   *          The measurement that is desired to view.
   * @param activeViewportId - the index that was active at the time the jump
   *          was initiated.
   * @return the viewportId that the measurement should be displayed in.
   */
  public getViewportIdToJump(
    activeViewportId: string,
    displaySetInstanceUID: string,
    cameraProps: unknown
  ): string {
    const viewportInfo = this.getViewportInfo(activeViewportId);

    if (viewportInfo.getViewportType() === csEnums.ViewportType.VOLUME_3D) {
      return null;
    }

    const { referencedImageId } = cameraProps as any;
    if (viewportInfo?.contains(displaySetInstanceUID, referencedImageId)) {
      return activeViewportId;
    }

    return (
      Array.from(this.viewportsById.values()).find(viewportInfo =>
        viewportInfo.contains(displaySetInstanceUID, referencedImageId)
      )?.getViewportId() ?? null
    );
  }

  private async _setStackViewport(
    viewport: Types.IStackViewport,
    viewportData: StackViewportData,
    viewportInfo: ViewportInfo,
    presentations: Presentations = {}
  ): Promise<void> {
    const displaySetOptions = viewportInfo.getDisplaySetOptions();

    const displaySetInstanceUIDs = viewportData.data.map(data => data.displaySetInstanceUID);

    // based on the cache service construct always the first one is the non-overlay
    // and the rest are overlays

    this.viewportsDisplaySets.set(viewport.id, [...displaySetInstanceUIDs]);

    const { initialImageIndex, imageIds } = viewportData.data[0];

    let initialImageIndexToUse =
      (presentations?.positionPresentation as any)?.initialImageIndex ?? initialImageIndex;

    if (initialImageIndexToUse === undefined || initialImageIndexToUse === null) {
      initialImageIndexToUse = this._getInitialImageIndexForViewport(viewportInfo, imageIds) || 0;
    }

    const properties = { ...(presentations.lutPresentation as any)?.properties };
    if (!(presentations.lutPresentation as any)?.properties) {
      const { voi, voiInverted, colormap } = displaySetOptions[0];
      if (voi && (voi.windowWidth || voi.windowCenter)) {
        const { lower, upper } = csUtils.windowLevel.toLowHighRange(
          voi.windowWidth,
          voi.windowCenter
        );
        properties.voiRange = { lower, upper };
      }

      if (voiInverted !== undefined) {
        properties.invert = voiInverted;
      }

      if (colormap !== undefined) {
        properties.colormap = colormap;
      }

      // FIX: Pass VOILUTFunction through to viewport for 16-bit stability
      // Note: Using raw string 'LINEAR_EXACT' since enum may not exist in all CS versions
      properties.VOILUTFunction = 'LINEAR_EXACT';
    }

    this._handleOverlays(viewport);

    console.log('[DEBUG] _setStackViewport called, setting stack with', imageIds.length, 'images');
    const isCPRPanoViewport = viewport.id === 'cpr-pano';
    const isCPRCrossSectionViewport = viewport.id === 'cpr-crosssection';
    if (isCPRPanoViewport) {
      console.log('[CPR-TRACE] CornerstoneViewportService entering _setStackViewport for cpr-pano', {
        initialImageIndexToUse,
        firstInputImageId: imageIds?.[0],
        hasLutPresentation: !!(presentations?.lutPresentation as any)?.properties,
        requestedVOI: properties?.voiRange,
        requestedColormap: (properties as any)?.colormap,
      });
    }

    // Avoid #2d parallel cache rewriting on synthetic CPR stacks to keep their custom schemes intact.
    const shouldUseParallel2dCache = !isCPRPanoViewport && !isCPRCrossSectionViewport;
    const stackImageIds = shouldUseParallel2dCache
      ? imageIds.map((imageId: string) => {
          if (imageId.endsWith('#2d')) {
            return imageId;
          }
          return `${imageId}#2d`;
        })
      : imageIds;

    if (shouldUseParallel2dCache) {
      console.log(
        `[GEMINI FIX] Parallel Cache: Using #2d suffixed imageIds (${stackImageIds.length} images)`
      );
    } else {
      console.log('[CPR-TRACE] synthetic CPR stack using original imageIds (no #2d suffix)', {
        firstStackImageId: stackImageIds?.[0],
        viewportId: viewport.id,
      });
    }

    if (viewport.id === 'cpr-pano') {
      const firstImageId = stackImageIds?.[0] ?? '';
      if (!firstImageId.startsWith('pano://')) {
        console.log(
          '[CPR] _setStackViewport: blocking non-pano:// stack load into cpr-pano. Orchestrator will set the correct pano:// stack.'
        );
        return Promise.resolve();
      }
    }

    if (viewport.id === 'cpr-crosssection') {
      const firstImageId = stackImageIds?.[0] ?? '';
      if (!firstImageId.startsWith('cross://')) {
        console.log(
          '[CPR] _setStackViewport: blocking non-cross:// stack load into cpr-crosssection. Orchestrator will set the correct cross:// stack.'
        );
        return Promise.resolve();
      }
    }

    // Load the image stack with parallel cache IDs
    return viewport.setStack(stackImageIds, initialImageIndexToUse).then(() => {
      console.log('[FIX] Stack loaded, now applying properties and recalculating VOI');
      if (isCPRPanoViewport) {
        const loadedImageIds = viewport.getImageIds?.() || [];
        const loadedIndex = Number(viewport.getCurrentImageIdIndex?.());
        const safeLoadedIndex =
          Number.isFinite(loadedIndex) && loadedImageIds.length
            ? Math.max(0, Math.min(loadedImageIds.length - 1, Math.floor(loadedIndex)))
            : 0;
        console.log('[CPR-TRACE] cpr-pano after setStack in CornerstoneViewportService', {
          loadedImageCount: loadedImageIds.length,
          currentImageIndex: loadedIndex,
          currentImageId: loadedImageIds[safeLoadedIndex],
          propertiesBeforeApply: viewport.getProperties?.(),
        });
      }

      // Apply the properties that were passed in
      viewport.setProperties({ ...properties });
      if (isCPRPanoViewport) {
        console.log('[CPR-TRACE] cpr-pano after setProperties in CornerstoneViewportService', {
          appliedProperties: properties,
          liveProperties: viewport.getProperties?.(),
        });
      }

      if (!isCPRPanoViewport && !isCPRCrossSectionViewport) {
        // [GEMINI FIX] Recalculate VOI based on ACTUAL pixel data (Corrected for Slope/Intercept)
        try {
          const imageData = viewport.getImageData();
          // SAFETY: Fetch slope/intercept from the wrapper, as VTK imageData doesn't have them
          const { slope = 1, intercept = 0 } = viewport.getCornerstoneImage() || {};

          if (imageData && imageData.scalarData) {
            const scalarData = imageData.scalarData;
            let scalarMin = Infinity;
            let scalarMax = -Infinity;

            // Sample the data for performance
            const step = Math.max(1, Math.floor(scalarData.length / 10000));
            for (let i = 0; i < scalarData.length; i += step) {
              // FIX: Apply Modality LUT (Slope/Intercept) to get real HU values
              const val = scalarData[i] * slope + intercept;
              if (val < scalarMin) scalarMin = val;
              if (val > scalarMax) scalarMax = val;
            }

            const range = scalarMax - scalarMin;
            const padding = range * 0.1; // 10% padding
            viewport.setProperties({
              voiRange: {
                lower: scalarMin - padding,
                upper: scalarMax + padding
              }
            });
            console.log(`[GEMINI FIX] Corrected VOI to actual data range (HU): ${scalarMin} - ${scalarMax}`);
          }
        } catch (error) {
          console.warn('[GEMINI FIX] Failed to calculate auto VOI:', error);
        }
      }

      // [FIX #2 - STEP 4] Final render
      viewport.render();
      if (isCPRPanoViewport) {
        const loadedImageIds = viewport.getImageIds?.() || [];
        const loadedIndex = Number(viewport.getCurrentImageIdIndex?.());
        const safeLoadedIndex =
          Number.isFinite(loadedIndex) && loadedImageIds.length
            ? Math.max(0, Math.min(loadedImageIds.length - 1, Math.floor(loadedIndex)))
            : 0;
        console.log('[CPR-TRACE] cpr-pano after render in CornerstoneViewportService', {
          currentImageIndex: loadedIndex,
          currentImageId: loadedImageIds[safeLoadedIndex],
          liveProperties: viewport.getProperties?.(),
        });
        window.setTimeout(() => {
          const delayedImageIds = viewport.getImageIds?.() || [];
          const delayedIndex = Number(viewport.getCurrentImageIdIndex?.());
          const safeDelayedIndex =
            Number.isFinite(delayedIndex) && delayedImageIds.length
              ? Math.max(0, Math.min(delayedImageIds.length - 1, Math.floor(delayedIndex)))
              : 0;
          console.log('[CPR-TRACE] cpr-pano delayed snapshot (+600ms) in CornerstoneViewportService', {
            currentImageIndex: delayedIndex,
            currentImageId: delayedImageIds[safeDelayedIndex],
            liveProperties: viewport.getProperties?.(),
          });
        }, 600);
      }

      // Log final state
      const propsAfterFix = viewport.getProperties();
      console.log('=== [FIX] FINAL STATE ===');
      console.log('voiRange:', JSON.stringify(propsAfterFix.voiRange));
      console.log('VOILUTFunction:', propsAfterFix.VOILUTFunction);
    });
  }

  private _getInitialImageIndexForViewport(
    viewportInfo: ViewportInfo,
    imageIds?: string[]
  ): number {
    const initialImageOptions = viewportInfo.getInitialImageOptions();

    if (!initialImageOptions) {
      return;
    }

    const { index, preset } = initialImageOptions;
    const viewportType = viewportInfo.getViewportType();

    let numberOfSlices;
    if (viewportType === csEnums.ViewportType.STACK) {
      numberOfSlices = imageIds.length;
    } else if (viewportType === csEnums.ViewportType.ORTHOGRAPHIC) {
      const viewport = this.getCornerstoneViewport(viewportInfo.getViewportId()) as unknown as Types.IVolumeViewport;
      const imageSliceData = csUtils.getImageSliceDataForVolumeViewport(viewport);

      if (!imageSliceData) {
        return;
      }

      ({ numberOfSlices } = imageSliceData);
    } else {
      return;
    }

    return this._getInitialImageIndex(numberOfSlices, index, preset);
  }

  _getInitialImageIndex(numberOfSlices: number, imageIndex?: number, preset?: JumpPresets): number {
    const lastSliceIndex = numberOfSlices - 1;

    if (imageIndex !== undefined) {
      return csToolsUtils.clip(imageIndex, 0, lastSliceIndex);
    }

    if (preset === JumpPresets.First) {
      return 0;
    }

    if (preset === JumpPresets.Last) {
      return lastSliceIndex;
    }

    if (preset === JumpPresets.Middle) {
      // Note: this is a simple but yet very important formula.
      // since viewport reset works with the middle slice
      // if the below formula is not correct, on a viewport reset
      // it will jump to a different slice than the middle one which
      // was the initial slice, and we have some tools such as Crosshairs
      // which rely on a relative camera modifications and those will break.
      return lastSliceIndex % 2 === 0 ? lastSliceIndex / 2 : (lastSliceIndex + 1) / 2;
    }

    return 0;
  }



  async _setVolumeViewport(
    viewport: Types.IVolumeViewport,
    viewportData: VolumeViewportData,
    viewportInfo: ViewportInfo,
    presentations: Presentations = {}
  ): Promise<void> {
    // TODO: We need to overhaul the way data sources work so requests can be made
    // async. I think we should follow the image loader pattern which is async and
    // has a cache behind it.
    // The problem is that to set this volume, we need the metadata, but the request is
    // already in-flight, and the promise is not cached, so we have no way to wait for
    // it and know when it has fully arrived.
    // loadStudyMetadata(StudyInstanceUID) => Promise([instances for study])
    // loadSeriesMetadata(StudyInstanceUID, SeriesInstanceUID) => Promise([instances for series])
    // If you call loadStudyMetadata and it's not in the DicomMetadataStore cache, it should fire
    // a request through the data source?
    // (This call may or may not create sub-requests for series metadata)
    const volumeInputArray = [];
    const displaySetOptionsArray = viewportInfo.getDisplaySetOptions();
    const { hangingProtocolService } = this.servicesManager.services;

    const volumeToLoad = [];
    const displaySetInstanceUIDs = [];

    for (const [index, data] of Array.from(viewportData.data.entries())) {
      const { volume, imageIds, displaySetInstanceUID } = data;

      displaySetInstanceUIDs.push(displaySetInstanceUID);

      if (!volume) {
        console.log('Volume display set not found');
        continue;
      }

      volumeToLoad.push(volume);

      const displaySetOptions = displaySetOptionsArray[index];
      const { volumeId } = volume;

      volumeInputArray.push({
        imageIds,
        volumeId,
        blendMode: displaySetOptions.blendMode,
        slabThickness: this._getSlabThickness(displaySetOptions, volumeId),
      });
    }

    this.viewportsDisplaySets.set(viewport.id, displaySetInstanceUIDs);

    // Set volumes and allow progressive rendering as slices arrive
    // Use the hydration queue to ensure viewports are initialized one by one
    // This is the "Crash Killer" for large 600-slice volumes.
    CornerstoneViewportService._hydrationQueue = CornerstoneViewportService._hydrationQueue.then(async () => {
      console.log(`[Hydration] Queue starting for viewport: ${viewport.id}`);

      // Small "breathing room" (micro-task) before starting the next viewport
      // This allows the browser to run Garbage Collection and update the UI/Progress Bar.
      await new Promise(resolve => setTimeout(resolve, 100));

      const volumesNotLoaded = volumeToLoad.filter(volume => !volume.loadStatus.loaded);

      if (volumesNotLoaded.length) {
        if (hangingProtocolService.getShouldPerformCustomImageLoad()) {
          console.log(`[Hydration] Running custom image load strategy for: ${viewport.id}`);
          await hangingProtocolService.runImageLoadStrategy({
            viewportId: viewport.id,
            volumeInputArray,
          });
        } else {
          // Start loading volumes sequentially within this viewport's turn
          volumesNotLoaded.forEach(volume => {
            if (!volume.loadStatus.loading) {
              console.log(`[Hydration] Triggering volume.load() for: ${volume.volumeId}`);
              volume.load();
            }
          });
        }
      }

      return this.setVolumesForViewport(viewport, volumeInputArray, presentations);
    }).catch(err => {
      console.error(`[Hydration] Error in queue for viewport ${viewport.id}:`, err);
    });

    return CornerstoneViewportService._hydrationQueue;
  }

  public async setVolumesForViewport(viewport, volumeInputArray, presentations) {
    const { displaySetService, toolGroupService, viewportGridService } =
      this.servicesManager.services;

    const viewportInfo = this.getViewportInfo(viewport.id);
    const viewportId = viewport.id;

    const displaySetOptions = viewportInfo.getDisplaySetOptions();
    const displaySetUIDs = viewportGridService.getDisplaySetsUIDsForViewport(viewport.id);
    const displaySet = displaySetService.getDisplaySetByUID(displaySetUIDs[0]);
    const displaySetModality = (displaySet as any)?.Modality;

    // Todo: use presentations states
    const volumesProperties = volumeInputArray.map((volumeInput, index) => {
      const { volumeId } = volumeInput;
      const displaySetOption = displaySetOptions[index];
      const { voi, voiInverted, colormap, displayPreset } = displaySetOption;
      const properties: any = {};

      if (voi && (voi.windowWidth || voi.windowCenter)) {
        const { lower, upper } = csUtils.windowLevel.toLowHighRange(
          voi.windowWidth,
          voi.windowCenter
        );
        properties.voiRange = { lower, upper };
      }

      if (voiInverted !== undefined) {
        properties.invert = voiInverted;
      }

      if (colormap !== undefined) {
        properties.colormap = colormap;
      }

      if (displayPreset !== undefined) {
        properties.preset = displayPreset[displaySetModality] || displayPreset['default'];
      }

      return { properties, volumeId };
    });

    try {
      await viewport.setVolumes(volumeInputArray);
    } catch (err) {
      console.error(`Error setting volumes for viewport ${viewportId}:`, err);
    }

    volumesProperties.forEach(({ properties, volumeId }) => {
      viewport.setProperties(properties, volumeId);
    });

    this.setPresentations(viewport.id, presentations);

    this._handleOverlays(viewport);

    const toolGroup = toolGroupService.getToolGroupForViewport(viewport.id);
    if (toolGroup && (toolGroup as any).id && toolGroup.hasTool?.('SegmentationDisplay')) {
      csToolsUtils.segmentation.triggerSegmentationRender((toolGroup as any).id);
    }

    const imageIndex = this._getInitialImageIndexForViewport(viewportInfo);

    if (imageIndex !== undefined) {
      csToolsUtils.jumpToSlice(viewport.element, {
        imageIndex,
      });
    }

    // Stage 1: Initial render after volume assignment (shows viewport structure)
    requestAnimationFrame(() => {
      try {
        viewport.render();
      } catch (renderError) {
        if (!this.isWebGLUnavailableError(renderError)) {
          console.warn(
            `[CornerstoneViewportService] Failed initial volume render for viewport ${viewportId}`,
            renderError
          );
        }
      }
    });

    // Final render logic
    const triggerFinalRender = () => {
      requestAnimationFrame(() => {
        try {
          viewport.render();
        } catch (renderError) {
          if (!this.isWebGLUnavailableError(renderError)) {
            console.warn(
              `[CornerstoneViewportService] Failed final volume render for viewport ${viewportId}`,
              renderError
            );
          }
        }
      });
    };

    // Stage 2: Final render when volume data fully loads (fixes black screens)
    const onVolumeLoadComplete = (evt) => {
      const { volumeId } = evt.detail;

      // Check if this event is for one of our volumes
      const isOurVolume = volumeInputArray.some(v => v.volumeId === volumeId);

      if (isOurVolume) {
        triggerFinalRender();

        // Cleanup: Remove listener after first successful render
        eventTarget.removeEventListener(
          csEnums.Events.IMAGE_VOLUME_LOADING_COMPLETED,
          onVolumeLoadComplete
        );
      }
    };

    // Check if ALL volumes are already loaded (Cache hit)
    const allLoaded = volumeInputArray.every(v => {
      const vol = cache.getVolume(v.volumeId);
      return vol?.loadStatus?.loaded;
    });

    if (allLoaded) {
      // Already loaded, just trigger final render
      triggerFinalRender();
    } else {
      // Attach listener to global eventTarget for future completion
      eventTarget.addEventListener(
        csEnums.Events.IMAGE_VOLUME_LOADING_COMPLETED,
        onVolumeLoadComplete
      );
    }

    this._broadcastEvent(this.EVENTS.VIEWPORT_VOLUMES_CHANGED, {
      viewportInfo,
    });
  }

  private _handleOverlays(viewport: Types.IStackViewport | Types.IVolumeViewport) {
    const { displaySetService } = this.servicesManager.services;

    // load any secondary displaySets
    const displaySetInstanceUIDs = this.viewportsDisplaySets.get(viewport.id);

    // can be SEG or RTSTRUCT for now
    const overlayDisplaySet = displaySetInstanceUIDs
      .map(displaySetService.getDisplaySetByUID)
      .find(displaySet => (displaySet as any)?.isOverlayDisplaySet);
    if (overlayDisplaySet) {
      this.addOverlayRepresentationForDisplaySet(overlayDisplaySet, viewport);
    } else {
      // If the displaySet is not a SEG displaySet we assume it is a primary displaySet
      // and we can look into hydrated segmentations to check if any of them are
      // associated with the primary displaySet

      // get segmentations only returns the hydrated segmentations
      this._addSegmentationRepresentationToToolGroupIfNecessary(displaySetInstanceUIDs, viewport);
    }
  }

  private _addSegmentationRepresentationToToolGroupIfNecessary(
    displaySetInstanceUIDs: string[],
    viewport: any
  ) {
    const { segmentationService, toolGroupService } = this.servicesManager.services;

    const toolGroup = toolGroupService.getToolGroupForViewport(viewport.id);
    if (!toolGroup || !toolGroup.hasTool?.('SegmentationDisplay')) {
      return;
    }

    // this only returns hydrated segmentations
    const segmentations = segmentationService.getSegmentations();

    for (const segmentation of segmentations) {
      const toolGroupSegmentationRepresentations =
        segmentationService.getSegmentationRepresentationsForToolGroup(toolGroup.id) || [];

      // if there is already a segmentation representation for this segmentation
      // for this toolGroup, don't bother at all
      const isSegmentationInToolGroup = toolGroupSegmentationRepresentations.find(
        representation => representation.segmentationId === segmentation.id
      );

      if (isSegmentationInToolGroup) {
        continue;
      }

      // otherwise, check if the hydrated segmentations are in the same FrameOfReferenceUID
      // as the primary displaySet, if so add the representation (since it was not there)
      const { id: segDisplaySetInstanceUID } = segmentation;
      let segFrameOfReferenceUID = this._getFrameOfReferenceUID(segDisplaySetInstanceUID);

      if (!segFrameOfReferenceUID) {
        // if the segmentation displaySet does not have a FrameOfReferenceUID, we might check the
        // segmentation itself maybe it has a FrameOfReferenceUID
        const { FrameOfReferenceUID } = segmentation;
        if (FrameOfReferenceUID) {
          segFrameOfReferenceUID = FrameOfReferenceUID;
        }
      }

      if (!segFrameOfReferenceUID) {
        return;
      }

      let shouldDisplaySeg = false;

      for (const displaySetInstanceUID of displaySetInstanceUIDs) {
        const primaryFrameOfReferenceUID = this._getFrameOfReferenceUID(displaySetInstanceUID);

        if (segFrameOfReferenceUID === primaryFrameOfReferenceUID) {
          shouldDisplaySeg = true;
          break;
        }
      }

      if (!shouldDisplaySeg) {
        return;
      }

      segmentationService.addSegmentationRepresentationToToolGroup(
        toolGroup.id,
        segmentation.id,
        false, // already hydrated,
        segmentation.type
      );
    }
  }

  private addOverlayRepresentationForDisplaySet(displaySet: any, viewport: any) {
    const { segmentationService, toolGroupService } = this.servicesManager.services;

    const { referencedVolumeId } = displaySet;
    const segmentationId = displaySet.displaySetInstanceUID;

    const toolGroup = toolGroupService.getToolGroupForViewport(viewport.id);
    if (!toolGroup || !toolGroup.hasTool?.('SegmentationDisplay')) {
      return;
    }

    const representationType =
      referencedVolumeId && cache.getVolume(referencedVolumeId) !== undefined
        ? csToolsEnums.SegmentationRepresentations.Labelmap
        : csToolsEnums.SegmentationRepresentations.Contour;

    segmentationService.addSegmentationRepresentationToToolGroup(
      (toolGroup as any).id,
      segmentationId,
      false,
      representationType
    );
  }

  // Todo: keepCamera is an interim solution until we have a better solution for
  // keeping the camera position when the viewport data is changed
  public updateViewport(viewportId: string, viewportData, keepCamera = false) {
    const viewportInfo = this.getViewportInfo(viewportId);
    const viewport = this.getCornerstoneViewport(viewportId);
    const viewportCamera = viewport.getCamera();

    let displaySetPromise;

    if (viewport instanceof VolumeViewport || viewport instanceof VolumeViewport3D) {
      displaySetPromise = this._setVolumeViewport(viewport, viewportData, viewportInfo).then(() => {
        if (keepCamera) {
          viewport.setCamera(viewportCamera);
          viewport.render();
        }
      });
    }

    if (viewport instanceof StackViewport) {
      displaySetPromise = this._setStackViewport(viewport, viewportData, viewportInfo);
    }

    displaySetPromise
      .then(() => {
        this._broadcastEvent(this.EVENTS.VIEWPORT_DATA_CHANGED, {
          viewportData,
          viewportId,
        });
      })
      .catch(error => {
        if (!this.isWebGLUnavailableError(error)) {
          console.warn(
            `[CornerstoneViewportService] Failed to update viewport ${viewportId}`,
            error
          );
        }
      });
  }

  _setDisplaySets(
    viewport: Types.IViewport,
    viewportData: StackViewportData | VolumeViewportData,
    viewportInfo: ViewportInfo,
    presentations: Presentations = {}
  ): Promise<void> {
    if (viewport instanceof StackViewport) {
      return this._setStackViewport(
        viewport,
        viewportData as StackViewportData,
        viewportInfo,
        presentations
      );
    }

    if ([VolumeViewport, VolumeViewport3D].some(type => viewport instanceof type)) {
      return this._setVolumeViewport(
        viewport as Types.IVolumeViewport,
        viewportData as VolumeViewportData,
        viewportInfo,
        presentations
      );
    }

    throw new Error('Unknown viewport type');
  }

  /**
   * Removes the resize observer from the viewport element
   */
  _removeResizeObserver() {
    if (this.viewportGridResizeObserver) {
      this.viewportGridResizeObserver.disconnect();
    }
  }

  _getSlabThickness(displaySetOptions, volumeId) {
    const { blendMode } = displaySetOptions;
    if (blendMode === undefined || displaySetOptions.slabThickness === undefined) {
      return;
    }

    // if there is a slabThickness set as a number then use it
    if (typeof displaySetOptions.slabThickness === 'number') {
      return displaySetOptions.slabThickness;
    }

    if (displaySetOptions.slabThickness.toLowerCase() === 'fullvolume') {
      // calculate the slab thickness based on the volume dimensions
      const imageVolume = cache.getVolume(volumeId);

      const { dimensions } = imageVolume;
      const slabThickness = Math.sqrt(
        dimensions[0] * dimensions[0] +
        dimensions[1] * dimensions[1] +
        dimensions[2] * dimensions[2]
      );

      return slabThickness;
    }
  }

  _getFrameOfReferenceUID(displaySetInstanceUID) {
    const { displaySetService } = this.servicesManager.services;
    const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);

    if (!displaySet) {
      return;
    }

    if ((displaySet as any).frameOfReferenceUID) {
      return (displaySet as any).frameOfReferenceUID;
    }

    if ((displaySet as any).Modality === 'SEG') {
      const { instance } = displaySet as any;
      return instance.FrameOfReferenceUID;
    }

    if ((displaySet as any).Modality === 'RTSTRUCT') {
      const { instance } = displaySet as any;
      return instance.ReferencedFrameOfReferenceSequence.FrameOfReferenceUID;
    }

    const { images } = displaySet as any;
    if (images && images.length) {
      return images[0].FrameOfReferenceUID;
    }
  }

  private enqueueViewportResizeRequest() {
    this.resizeQueue.push(false); // false indicates viewport resize

    clearTimeout(this.viewportResizeTimer);
    this.viewportResizeTimer = setTimeout(() => {
      this.processViewportResizeQueue();
    }, this.gridResizeDelay);
  }

  private processViewportResizeQueue() {
    const isGridResizeInQueue = this.resizeQueue.some(isGridResize => isGridResize);
    if (this.resizeQueue.length > 0 && !isGridResizeInQueue && !this.gridResizeTimeOut) {
      this.performResize();
    }

    // Clear the queue after processing viewport resizes
    this.resizeQueue = [];
  }

  private performResize() {
    const isImmediate = false;

    try {
      const renderingEngine =
        (getRenderingEngine(RENDERING_ENGINE_ID) as Types.IRenderingEngine | undefined) ??
        this.renderingEngine;

      if (!renderingEngine || renderingEngine.hasBeenDestroyed) {
        return;
      }

      this.renderingEngine = renderingEngine;

      let viewports = [];
      try {
        viewports = renderingEngine.getViewports?.() || [];
      } catch (getViewportsError) {
        if (!this.isWebGLUnavailableError(getViewportsError)) {
          console.warn('Caught resize exception while querying viewports', getViewportsError);
        }
        return;
      }

      if (!Array.isArray(viewports) || viewports.length === 0) {
        return;
      }

      // Store the current position presentations for each viewport.
      viewports.forEach(({ id }) => {
        const presentation = this.getPositionPresentation(id);
        this.beforeResizePositionPresentations.set(id, presentation);
      });

      // Use requestAnimationFrame to move heavy render operations off main thread
      requestAnimationFrame(() => {
        try {
          renderingEngine.resize(isImmediate);
        } catch (resizeError) {
          if (!this.isWebGLUnavailableError(resizeError)) {
            console.warn('Caught resize exception during first resize pass', resizeError);
          }
          return;
        }

        // Reset the camera for viewports that should reset their camera on resize,
        // which means only those viewports that have a zoom level of 1.
        this.beforeResizePositionPresentations.forEach((positionPresentation, viewportId) => {
          this.setPresentations(viewportId, { positionPresentation });
        });

        // Single render after resize and presentation updates
        requestAnimationFrame(() => {
          try {
            renderingEngine.resize(isImmediate);
            renderingEngine.render();
          } catch (renderError) {
            if (!this.isWebGLUnavailableError(renderError)) {
              console.warn('Caught resize exception during second resize pass', renderError);
            }
          }
        });
      });
    } catch (e) {
      // This can happen if the resize is too close to navigation or shutdown
      if (!this.isWebGLUnavailableError(e)) {
        console.warn('Caught resize exception', e);
      }
    }
  }

  private resetGridResizeTimeout() {
    clearTimeout(this.gridResizeTimeOut);
    this.gridResizeTimeOut = setTimeout(() => {
      this.gridResizeTimeOut = null;
    }, this.gridResizeDelay);
  }
}

export default CornerstoneViewportService;
