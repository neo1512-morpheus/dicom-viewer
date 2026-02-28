import React, { useEffect, useRef, useCallback, useState } from 'react';
import ReactResizeDetector from 'react-resize-detector';
import PropTypes from 'prop-types';
import * as cs3DTools from '@cornerstonejs/tools';
import {
  Enums,
  eventTarget,
  getEnabledElement,
  StackViewport,
  utilities as csUtils,
} from '@cornerstonejs/core';
import { MeasurementService } from '@ohif/core';
import { Notification, useViewportDialog, AllInOneMenu } from '@ohif/ui';
import { IStackViewport, IVolumeViewport } from '@cornerstonejs/core/dist/esm/types';

import { setEnabledElement } from '../state';

import './OHIFCornerstoneViewport.css';
import CornerstoneOverlays from './Overlays/CornerstoneOverlays';
import getSOPInstanceAttributes from '../utils/measurementServiceMappings/utils/getSOPInstanceAttributes';
import CinePlayer from '../components/CinePlayer';
import { Types } from '@ohif/core';

import OHIFViewportActionCorners from '../components/OHIFViewportActionCorners';
import { getWindowLevelActionMenu } from '../components/WindowLevelActionMenu/getWindowLevelActionMenu';
import { useAppConfig } from '@state';
import { useCPROrchestrator } from '../../../../modes/cpr/src/useCPROrchestrator';

import { LutPresentation, PositionPresentation } from '../types/Presentation';

const isKnownWebGLUnavailableError = (error: unknown) => {
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
    text.includes('Cannot read properties of undefined (reading \'values\')') ||
    text.includes('get3DContext') ||
    text.includes('getRenderWindow') ||
    text.includes('RenderWindow.js') ||
    text.includes('RenderingEngine.ts:1093') ||
    text.includes('RenderingEngine.ts:1174') ||
    text.includes('WebGL')
  );
};

function CPRDoneAction({ servicesManager, commandsManager, viewportId }) {
  const { onDone, onRedraw, isGenerating, error } = useCPROrchestrator({
    servicesManager,
    commandsManager,
    sourceViewportId: viewportId,
    panoWidth: 800,
    panoHeight: 400,
    slabHalfThicknessMm: 7,
    slabSamples: 21,
    aggregation: 'MEAN',
  });
  const [hasSpline, setHasSpline] = useState(false);

  const setActiveAxialViewport = useCallback(() => {
    const { cornerstoneViewportService, viewportGridService } = servicesManager.services;

    const directViewport = cornerstoneViewportService.getCornerstoneViewport('cpr-axial');
    if (directViewport) {
      commandsManager.runCommand('setViewportActive', { viewportId: directViewport.id });
      return;
    }

    const gridViewports = viewportGridService.getState().viewports;
    for (const [gridViewportId, gridViewport] of gridViewports.entries()) {
      if (gridViewport?.viewportOptions?.viewportId === 'cpr-axial') {
        commandsManager.runCommand('setViewportActive', { viewportId: gridViewportId });
        return;
      }
    }

    commandsManager.runCommand('setViewportActive', { viewportId });
  }, [servicesManager, commandsManager, viewportId]);

  const onButtonMouseDown = useCallback(evt => {
    evt.preventDefault();
    evt.stopPropagation();
  }, []);

  useEffect(() => {
    const updateSplineState = () => {
      const allAnnotations = cs3DTools.annotation.state.getAllAnnotations?.() || [];
      const splineCount = allAnnotations.filter(
        annotation => annotation?.metadata?.toolName === 'SplineROI'
      ).length;
      setHasSpline(splineCount > 0);
    };

    updateSplineState();
    const intervalId = window.setInterval(updateSplineState, 300);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [servicesManager, viewportId]);

  const onPan = useCallback(() => {
    setActiveAxialViewport();
    commandsManager.runCommand('setToolActive', { toolName: 'Pan', toolGroupId: 'mpr' });
  }, [commandsManager, setActiveAxialViewport]);

  const onSpline = useCallback(() => {
    setActiveAxialViewport();
    commandsManager.runCommand('setToolActive', { toolName: 'SplineROI', toolGroupId: 'mpr' });
  }, [commandsManager, setActiveAxialViewport]);

  const onRedrawClick = useCallback(() => {
    void onRedraw();
  }, [onRedraw]);

  const onDoneMouseDown = useCallback(
    evt => {
      evt.preventDefault();
      evt.stopPropagation();
    },
    []
  );

  return (
    <div className="flex max-w-[320px] flex-col gap-1 rounded bg-black/60 px-2 py-1 text-white">
      <div className="text-[11px] leading-tight text-cyan-100">
        {hasSpline
          ? 'Arch ready. Press Done to generate CPR panels.'
          : 'Draw Spline ROI on axial, then press Done.'}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="rounded border border-white/30 px-2 py-1 text-xs"
          onMouseDown={onButtonMouseDown}
          onClick={onPan}
        >
          Pan
        </button>
        <button
          type="button"
          className="rounded border border-white/30 px-2 py-1 text-xs"
          onMouseDown={onButtonMouseDown}
          onClick={onSpline}
        >
          Spline
        </button>
        <button
          type="button"
          className="rounded border border-white/30 px-2 py-1 text-xs"
          onMouseDown={onButtonMouseDown}
          onClick={onRedrawClick}
        >
          Redraw
        </button>
        <button
          type="button"
          className="rounded border border-white/40 px-2 py-1 text-xs disabled:opacity-50"
          onMouseDown={onDoneMouseDown}
          onClick={() => {
            void onDone();
          }}
          disabled={isGenerating || !hasSpline}
        >
          {isGenerating ? 'Generating...' : 'Done'}
        </button>
      </div>
      {error ? <span className="max-w-[300px] truncate text-[11px] text-red-300">{error}</span> : null}
    </div>
  );
}

function VolumeLoadingOverlay({ progress }) {
  if (!progress || progress.percentComplete >= 100) {
    return null;
  }

  return (
    <div className="volume-loading-overlay">
      <div className="loading-content">
        <div className="loading-spinner"></div>
        <div className="loading-text">
          Loading Volume: {progress.percentComplete}%
        </div>
        <div className="loading-subtext">
          {progress.loadedImages} / {progress.totalImages} slices
        </div>
        <div className="loading-bar-container">
          <div
            className="loading-bar-fill"
            style={{ width: `${progress.percentComplete}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
}

const STACK = 'stack';

/**
 * Caches the jump to measurement operation, so that if display set is shown,
 * it can jump to the measurement.
 */
let cacheJumpToMeasurementEvent;

function areEqual(prevProps, nextProps) {
  if (nextProps.needsRerendering) {
    return false;
  }

  if (prevProps.displaySets.length !== nextProps.displaySets.length) {
    return false;
  }

  if (prevProps.viewportOptions.orientation !== nextProps.viewportOptions.orientation) {
    return false;
  }

  if (prevProps.viewportOptions.toolGroupId !== nextProps.viewportOptions.toolGroupId) {
    return false;
  }

  if (prevProps.viewportOptions.viewportType !== nextProps.viewportOptions.viewportType) {
    return false;
  }

  if (nextProps.viewportOptions.needsRerendering) {
    return false;
  }

  const prevDisplaySets = prevProps.displaySets;
  const nextDisplaySets = nextProps.displaySets;

  if (prevDisplaySets.length !== nextDisplaySets.length) {
    return false;
  }

  for (let i = 0; i < prevDisplaySets.length; i++) {
    const prevDisplaySet = prevDisplaySets[i];

    const foundDisplaySet = nextDisplaySets.find(
      nextDisplaySet =>
        nextDisplaySet.displaySetInstanceUID === prevDisplaySet.displaySetInstanceUID
    );

    if (!foundDisplaySet) {
      return false;
    }

    // check they contain the same image
    if (foundDisplaySet.images?.length !== prevDisplaySet.images?.length) {
      return false;
    }

    // check if their imageIds are the same
    if (foundDisplaySet.images?.length) {
      for (let j = 0; j < foundDisplaySet.images.length; j++) {
        if (foundDisplaySet.images[j].imageId !== prevDisplaySet.images[j].imageId) {
          return false;
        }
      }
    }
  }

  return true;
}

// Todo: This should be done with expose of internal API similar to react-vtkjs-viewport
// Then we don't need to worry about the re-renders if the props change.
const OHIFCornerstoneViewport = React.memo((props: withAppTypes) => {
  const {
    displaySets,
    dataSource,
    viewportOptions,
    displaySetOptions,
    servicesManager,
    commandsManager,
    onElementEnabled,
    // eslint-disable-next-line react/prop-types
    onElementDisabled,
    isJumpToMeasurementDisabled,
    // Note: you SHOULD NOT use the initialImageIdOrIndex for manipulation
    // of the imageData in the OHIFCornerstoneViewport. This prop is used
    // to set the initial state of the viewport's first image to render
    // eslint-disable-next-line react/prop-types
    initialImageIndex,
    // if the viewport is part of a hanging protocol layout
    // we should not really rely on the old synchronizers and
    // you see below we only rehydrate the synchronizers if the viewport
    // is not part of the hanging protocol layout. HPs should
    // define their own synchronizers. Since the synchronizers are
    // viewportId dependent and
    // eslint-disable-next-line react/prop-types
    isHangingProtocolLayout,
  } = props;

  const viewportId = viewportOptions.viewportId;

  if (!viewportId) {
    throw new Error('Viewport ID is required');
  }

  // Preserve explicit stack viewports (e.g. CPR pano) and only auto-promote when not pinned.
  const hasDynamicReconstructable = displaySets.some(ds => ds.isDynamicVolume && ds.isReconstructable);
  const isExplicitStackViewport = viewportOptions.viewportType === 'stack' || viewportId === 'cpr-pano';
  viewportOptions.viewportType =
    hasDynamicReconstructable && !isExplicitStackViewport
      ? 'volume'
      : viewportOptions.viewportType;

  const [scrollbarHeight, setScrollbarHeight] = useState('100px');
  const [enabledVPElement, setEnabledVPElement] = useState(null);
  const elementRef = useRef();
  const [appConfig] = useAppConfig();

  const {
    measurementService,
    displaySetService,
    toolbarService,
    toolGroupService,
    syncGroupService,
    cornerstoneViewportService,
    cornerstoneCacheService,
    viewportGridService,
    stateSyncService,
    viewportActionCornersService,
  } = servicesManager.services;

  const [loadingProgress, setLoadingProgress] = useState(null);
  const [viewportDialogState] = useViewportDialog();
  // useCallback for scroll bar height calculation
  const setImageScrollBarHeight = useCallback(() => {
    const scrollbarHeight = `${elementRef.current.clientHeight - 40}px`;
    setScrollbarHeight(scrollbarHeight);
  }, [elementRef]);

  // useCallback for onResize
  const onResize = useCallback(() => {
    if (elementRef.current) {
      cornerstoneViewportService.resize();
      setImageScrollBarHeight();
    }
  }, [elementRef]);

  const cleanUpServices = useCallback(
    viewportInfo => {
      const renderingEngineId = viewportInfo.getRenderingEngineId();
      const syncGroups = viewportInfo.getSyncGroups();

      try {
        toolGroupService.removeViewportFromToolGroup(viewportId, renderingEngineId);
      } catch (error) {
        if (!isKnownWebGLUnavailableError(error)) {
          console.warn('Failed to remove viewport from tool group during cleanup', error);
        }
      }

      try {
        syncGroupService.removeViewportFromSyncGroup(viewportId, renderingEngineId, syncGroups);
      } catch (error) {
        if (!isKnownWebGLUnavailableError(error)) {
          console.warn('Failed to remove viewport from sync group during cleanup', error);
        }
      }

      viewportActionCornersService.clear(viewportId);
    },
    [viewportId]
  );

  const elementEnabledHandler = useCallback(
    evt => {
      // check this is this element reference and return early if doesn't match
      if (evt.detail.element !== elementRef.current) {
        return;
      }

      const { viewportId, element } = evt.detail;
      const viewportInfo = cornerstoneViewportService.getViewportInfo(viewportId);
      setEnabledElement(viewportId, element);
      setEnabledVPElement(element);

      const renderingEngineId = viewportInfo.getRenderingEngineId();
      const toolGroupId = viewportInfo.getToolGroupId();
      const syncGroups = viewportInfo.getSyncGroups();

      toolGroupService.addViewportToToolGroup(viewportId, renderingEngineId, toolGroupId);

      syncGroupService.addViewportToSyncGroup(viewportId, renderingEngineId, syncGroups);

      const synchronizersStore = stateSyncService.getState().synchronizersStore;

      if (synchronizersStore?.[viewportId]?.length && !isHangingProtocolLayout) {
        // If the viewport used to have a synchronizer, re apply it again
        _rehydrateSynchronizers(synchronizersStore, viewportId, syncGroupService);
      }

      if (onElementEnabled) {
        onElementEnabled(evt);
      }
    },
    [viewportId, onElementEnabled, toolGroupService]
  );

  // disable the element upon unmounting
  useEffect(() => {
    cornerstoneViewportService.enableViewport(viewportId, elementRef.current);

    eventTarget.addEventListener(Enums.Events.ELEMENT_ENABLED, elementEnabledHandler);

    setImageScrollBarHeight();

    return () => {
      const viewportInfo = cornerstoneViewportService.getViewportInfo(viewportId);

      if (!viewportInfo) {
        return;
      }

      cornerstoneViewportService.storePresentation({ viewportId });

      // This should be done after the store presentation since synchronizers
      // will get cleaned up and they need the viewportInfo to be present
      cleanUpServices(viewportInfo);

      if (onElementDisabled) {
        onElementDisabled(viewportInfo);
      }

      cornerstoneViewportService.disableElement(viewportId);

      eventTarget.removeEventListener(Enums.Events.ELEMENT_ENABLED, elementEnabledHandler);
    };
  }, [viewportId, elementEnabledHandler, cornerstoneViewportService]);

  // =====================================================
  // WebGL CONTEXT LOSS LISTENERS
  // =====================================================
  useEffect(() => {
    // Wait for viewport to be enabled (canvas to exist)
    if (!enabledVPElement) return;

    const canvas = elementRef.current?.querySelector('canvas');
    if (!canvas) {
      console.log('[GPU] No canvas found yet, waiting...');
      return;
    }

    console.log('[GPU] Attaching WebGL context loss listener to canvas');

    const handleContextLost = (event: Event) => {
      event.preventDefault();
    };

    const handleContextRestored = () => {
      console.log('[GPU] WebGL context restored.');
      try {
        cornerstoneViewportService.resize(true);
      } catch (e) {
        console.warn('[GPU] Could not resize after context restore:', e);
      }
    };

    canvas.addEventListener('webglcontextlost', handleContextLost);
    canvas.addEventListener('webglcontextrestored', handleContextRestored);

    return () => {
      canvas.removeEventListener('webglcontextlost', handleContextLost);
      canvas.removeEventListener('webglcontextrestored', handleContextRestored);
    };
  }, [viewportId, enabledVPElement, cornerstoneViewportService, servicesManager]);

  // Listen for volume loading progress
  useEffect(() => {
    const VOLUME_LOADING_PROGRESS = 'event::cornerstoneViewportService:volumeLoadingProgress';

    const onProgress = (evt) => {
      if (!evt || !evt.detail) return;
      const { volumeId, percentComplete, loadedImages, totalImages } = evt.detail;

      // Check if this volume belongs to our displaySets
      if (!displaySets || !displaySets.length) return;

      const displaySetInstanceUIDs = displaySets.map(ds => ds.displaySetInstanceUID);
      // Robust matching: Check if any of our displaySetUIDs are part of the volumeId
      const isOurVolume = displaySetInstanceUIDs.some(uid => volumeId.includes(uid));

      if (isOurVolume) {
        setLoadingProgress({ percentComplete, loadedImages, totalImages });
      }
    };

    // Robust subscription handling: checks if return value is a function or an object
    const subscription = cornerstoneViewportService.subscribe(
      VOLUME_LOADING_PROGRESS,
      onProgress
    );

    return () => {
      if (typeof subscription === 'function') {
        subscription();
      } else if (typeof subscription === 'object' && typeof subscription.unsubscribe === 'function') {
        subscription.unsubscribe();
      }
    };
  }, [displaySets, cornerstoneViewportService]);

  // subscribe to displaySet metadata invalidation (updates)
  // Currently, if the metadata changes we need to re-render the display set
  // for it to take effect in the viewport. As we deal with scaling in the loading,
  // we need to remove the old volume from the cache, and let the
  // viewport to re-add it which will use the new metadata. Otherwise, the
  // viewport will use the cached volume and the new metadata will not be used.
  // Note: this approach does not actually end of sending network requests
  // and it uses the network cache
  useEffect(() => {
    // Robust subscription handling: checks if return value is a function or an object
    const subscription = displaySetService.subscribe(
      displaySetService.EVENTS.DISPLAY_SET_SERIES_METADATA_INVALIDATED,
      async ({
        displaySetInstanceUID: invalidatedDisplaySetInstanceUID,
        invalidateData,
      }: Types.DisplaySetSeriesMetadataInvalidatedEvent) => {
        if (!invalidateData) {
          return;
        }

        const viewportInfo = cornerstoneViewportService.getViewportInfo(viewportId);

        if (viewportInfo.hasDisplaySet(invalidatedDisplaySetInstanceUID)) {
          const viewportData = viewportInfo.getViewportData();
          const newViewportData = await cornerstoneCacheService.invalidateViewportData(
            viewportData,
            invalidatedDisplaySetInstanceUID,
            dataSource,
            displaySetService
          );

          const keepCamera = true;
          cornerstoneViewportService.updateViewport(viewportId, newViewportData, keepCamera);
        }
      }
    );
    return () => {
      if (typeof subscription === 'function') {
        subscription();
      } else if (typeof subscription === 'object' && typeof subscription.unsubscribe === 'function') {
        subscription.unsubscribe();
      }
    };
  }, [viewportId]);

  useEffect(() => {
    // handle the default viewportType to be stack
    if (!viewportOptions.viewportType) {
      viewportOptions.viewportType = STACK;
    }
    let isCancelled = false;
    let deferredViewportDataTimeoutId: number | null = null;

    const loadViewportData = async () => {
      const viewportData = await cornerstoneCacheService.createViewportData(
        displaySets,
        viewportOptions,
        dataSource,
        initialImageIndex
      );
      if (isCancelled) {
        return;
      }

      // The presentation state will have been stored previously by closing
      // a viewport.  Otherwise, this viewport will be unchanged and the
      // presentation information will be directly carried over.
      const state = stateSyncService.getState();
      const lutPresentationStore = state.lutPresentationStore as LutPresentation;
      const positionPresentationStore = state.positionPresentationStore as PositionPresentation;

      const { presentationIds } = viewportOptions;
      const presentations = {
        positionPresentation: positionPresentationStore[presentationIds?.positionPresentationId],
        lutPresentation: lutPresentationStore[presentationIds?.lutPresentationId],
      };
      let measurement;
      if (cacheJumpToMeasurementEvent?.viewportId === viewportId) {
        measurement = cacheJumpToMeasurementEvent.measurement;
        // Delete the position presentation so that viewport navigates direct
        presentations.positionPresentation = null;
        cacheJumpToMeasurementEvent = null;
      }

      // Note: This is a hack to get the grid to re-render the OHIFCornerstoneViewport component
      // Used for segmentation hydration right now, since the logic to decide whether
      // a viewport needs to render a segmentation lives inside the CornerstoneViewportService
      // so we need to re-render (force update via change of the needsRerendering) so that React
      // does the diffing and decides we should render this again (although the id and element has not changed)
      // so that the CornerstoneViewportService can decide whether to render the segmentation or not. Not that we reached here we can turn it off.
      if (viewportOptions.needsRerendering) {
        viewportOptions.needsRerendering = false;
      }

      // Defer the heavy setViewportData call to allow React to finish the mount
      // and keep the UI responsive (e.g. allow the layout menu to close)
      deferredViewportDataTimeoutId = window.setTimeout(() => {
        if (isCancelled) {
          return;
        }
        if (!cornerstoneViewportService.getViewportInfo(viewportId)) {
          return;
        }
        cornerstoneViewportService.setViewportData(
          viewportId,
          viewportData,
          viewportOptions,
          displaySetOptions,
          presentations
        );
      }, 0);
      if (measurement) {
        cs3DTools.annotation.selection.setAnnotationSelected(measurement.uid);
      }
    };

    loadViewportData();
    return () => {
      isCancelled = true;
      if (deferredViewportDataTimeoutId !== null) {
        window.clearTimeout(deferredViewportDataTimeoutId);
      }
    };
  }, [viewportOptions, displaySets, dataSource]);

  /**
   * There are two scenarios for jump to click
   * 1. Current viewports contain the displaySet that the annotation was drawn on
   * 2. Current viewports don't contain the displaySet that the annotation was drawn on
   * and we need to change the viewports displaySet for jumping.
   * Since measurement_jump happens via events and listeners, the former case is handled
   * by the measurement_jump direct callback, but the latter case is handled first by
   * the viewportGrid to set the correct displaySet on the viewport, AND THEN we check
   * the cache for jumping to see if there is any jump queued, then we jump to the correct slice.
   */
  useEffect(() => {
    if (isJumpToMeasurementDisabled) {
      return;
    }

    const unsubscribeFromJumpToMeasurementEvents = _subscribeToJumpToMeasurementEvents(
      measurementService,
      displaySetService,
      elementRef,
      viewportId,
      displaySets,
      viewportGridService,
      cornerstoneViewportService
    );

    _checkForCachedJumpToMeasurementEvents(
      measurementService,
      displaySetService,
      elementRef,
      viewportId,
      displaySets,
      viewportGridService,
      cornerstoneViewportService
    );

    return () => {
      unsubscribeFromJumpToMeasurementEvents();
    };
  }, [displaySets, elementRef, viewportId]);

  // Set up the window level action menu in the viewport action corners.
  useEffect(() => {
    // Doing an === check here because the default config value when not set is true
    if (appConfig.addWindowLevelActionMenu === false) {
      return;
    }

    // TODO: In the future we should consider using the customization service
    // to determine if and in which corner various action components should go.
    const wlActionMenu = getWindowLevelActionMenu({
      viewportId,
      element: elementRef.current,
      displaySets,
      servicesManager,
      commandsManager,
      verticalDirection: AllInOneMenu.VerticalDirection.TopToBottom,
      horizontalDirection: AllInOneMenu.HorizontalDirection.RightToLeft,
    });

    viewportActionCornersService.setComponent({
      viewportId,
      id: 'windowLevelActionMenu',
      component: wlActionMenu,
      location: viewportActionCornersService.LOCATIONS.topRight,
      indexPriority: -100,
    });
  }, [displaySets, viewportId, viewportActionCornersService, servicesManager, commandsManager]);

  // Inject CPR Done action only on cpr-axial viewport.
  useEffect(() => {
    if (viewportId !== 'cpr-axial') {
      return;
    }

    viewportActionCornersService.setComponent({
      viewportId,
      id: 'cprDoneAction',
      component: (
        <CPRDoneAction
          servicesManager={servicesManager}
          commandsManager={commandsManager}
          viewportId={viewportId}
        />
      ),
      location: viewportActionCornersService.LOCATIONS.topLeft,
      indexPriority: -200,
    });
  }, [viewportId, viewportActionCornersService, servicesManager, commandsManager]);

  return (
    <React.Fragment>
      <div className="viewport-wrapper">
        <ReactResizeDetector
          onResize={onResize}
          targetRef={elementRef.current}
        />
        <div
          className="cornerstone-viewport-element"
          style={{ height: '100%', width: '100%' }}
          onContextMenu={e => e.preventDefault()}
          onMouseDown={e => e.preventDefault()}
          onWheel={
            viewportId === 'cpr-crosssection'
              ? e => {
                  e.preventDefault();
                  e.stopPropagation();
                }
              : undefined
          }
          ref={elementRef}
        ></div>
        <CornerstoneOverlays
          viewportId={viewportId}
          toolBarService={toolbarService}
          element={elementRef.current}
          scrollbarHeight={scrollbarHeight}
          servicesManager={servicesManager}
        />
        <CinePlayer
          enabledVPElement={enabledVPElement}
          viewportId={viewportId}
          servicesManager={servicesManager}
        />
        <VolumeLoadingOverlay progress={loadingProgress} />
      </div>
      {/* top offset of 24px to account for ViewportActionCorners. */}
      <div className="absolute top-[24px] w-full">
        {viewportDialogState.viewportId === viewportId && (
          <Notification
            id="viewport-notification"
            message={viewportDialogState.message}
            type={viewportDialogState.type}
            actions={viewportDialogState.actions}
            onSubmit={viewportDialogState.onSubmit}
            onOutsideClick={viewportDialogState.onOutsideClick}
            onKeyPress={viewportDialogState.onKeyPress}
          />
        )}
      </div>
      {/* The OHIFViewportActionCorners follows the viewport in the DOM so that it is naturally at a higher z-index.*/}
      <OHIFViewportActionCorners viewportId={viewportId} />
    </React.Fragment>
  );
}, areEqual);

function _subscribeToJumpToMeasurementEvents(
  measurementService,
  displaySetService,
  elementRef,
  viewportId,
  displaySets,
  viewportGridService,
  cornerstoneViewportService
) {
  // Robust subscription handling: checks if return value is a function or an object
  const subscription = measurementService.subscribe(
    MeasurementService.EVENTS.JUMP_TO_MEASUREMENT_VIEWPORT,
    props => {
      cacheJumpToMeasurementEvent = props;
      const { viewportId: jumpId, measurement, isConsumed } = props;
      if (!measurement || isConsumed) {
        return;
      }
      if (cacheJumpToMeasurementEvent.cornerstoneViewport === undefined) {
        // Decide on which viewport should handle this
        cacheJumpToMeasurementEvent.cornerstoneViewport =
          cornerstoneViewportService.getViewportIdToJump(
            jumpId,
            measurement.displaySetInstanceUID,
            {
              referencedImageId:
                measurement.referencedImageId || measurement.metadata?.referencedImageId,
            }
          );
      }
      if (cacheJumpToMeasurementEvent.cornerstoneViewport !== viewportId) {
        return;
      }
      _jumpToMeasurement(
        measurement,
        elementRef,
        viewportId,
        measurementService,
        displaySetService,
        viewportGridService,
        cornerstoneViewportService
      );
    }
  );

  // Return a safe cleanup function that handles both types
  return () => {
    if (typeof subscription === 'function') {
      subscription();
    } else if (typeof subscription === 'object' && typeof subscription.unsubscribe === 'function') {
      subscription.unsubscribe();
    }
  };
}

// Check if there is a queued jumpToMeasurement event
function _checkForCachedJumpToMeasurementEvents(
  measurementService,
  displaySetService,
  elementRef,
  viewportId,
  displaySets,
  viewportGridService,
  cornerstoneViewportService
) {
  if (!cacheJumpToMeasurementEvent) {
    return;
  }
  if (cacheJumpToMeasurementEvent.isConsumed) {
    cacheJumpToMeasurementEvent = null;
    return;
  }
  const displaysUIDs = displaySets.map(displaySet => displaySet.displaySetInstanceUID);
  if (!displaysUIDs?.length) {
    return;
  }

  // Jump to measurement if the measurement exists
  const { measurement } = cacheJumpToMeasurementEvent;
  if (measurement && elementRef) {
    if (displaysUIDs.includes(measurement?.displaySetInstanceUID)) {
      _jumpToMeasurement(
        measurement,
        elementRef,
        viewportId,
        measurementService,
        displaySetService,
        viewportGridService,
        cornerstoneViewportService
      );
    }
  }
}

function _jumpToMeasurement(
  measurement,
  targetElementRef,
  viewportId,
  measurementService,
  displaySetService,
  viewportGridService,
  cornerstoneViewportService
) {
  const targetElement = targetElementRef.current;
  const { displaySetInstanceUID, SOPInstanceUID, frameNumber } = measurement;

  if (!SOPInstanceUID) {
    console.warn('cannot jump in a non-acquisition plane measurements yet');
    return;
  }

  const referencedDisplaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);

  // Todo: setCornerstoneMeasurementActive should be handled by the toolGroupManager
  //  to set it properly
  // setCornerstoneMeasurementActive(measurement);

  viewportGridService.setActiveViewportId(viewportId);

  const enabledElement = getEnabledElement(targetElement);

  if (enabledElement) {
    // See how the jumpToSlice() of Cornerstone3D deals with imageIdx param.
    const viewport = enabledElement.viewport as IStackViewport | IVolumeViewport;

    let imageIdIndex = 0;
    let viewportCameraDirectionMatch = true;

    if (viewport instanceof StackViewport) {
      const imageIds = viewport.getImageIds();
      imageIdIndex = imageIds.findIndex(imageId => {
        const { SOPInstanceUID: aSOPInstanceUID, frameNumber: aFrameNumber } =
          getSOPInstanceAttributes(imageId);
        return aSOPInstanceUID === SOPInstanceUID && (!frameNumber || frameNumber === aFrameNumber);
      });
    } else {
      // for volume viewport we can't rely on the imageIdIndex since it can be
      // a reconstructed view that doesn't match the original slice numbers etc.
      const { viewPlaneNormal: measurementViewPlane } = measurement.metadata;
      imageIdIndex = referencedDisplaySet.images.findIndex(
        i => i.SOPInstanceUID === SOPInstanceUID
      );

      // the index is reversed in the volume viewport
      // imageIdIndex = referencedDisplaySet.images.length - 1 - imageIdIndex;

      const { viewPlaneNormal: viewportViewPlane } = viewport.getCamera();

      // should compare abs for both planes since the direction can be flipped
      if (
        measurementViewPlane &&
        !csUtils.isEqual(measurementViewPlane.map(Math.abs), viewportViewPlane.map(Math.abs))
      ) {
        viewportCameraDirectionMatch = false;
      }
    }

    if (!viewportCameraDirectionMatch || imageIdIndex === -1) {
      return;
    }

    cs3DTools.utilities.jumpToSlice(targetElement, {
      imageIndex: imageIdIndex,
    });

    cs3DTools.annotation.selection.setAnnotationSelected(measurement.uid);
    // Jump to measurement consumed, remove.
    cacheJumpToMeasurementEvent?.consume?.();
    cacheJumpToMeasurementEvent = null;
  }
}

function _rehydrateSynchronizers(
  synchronizersStore: { [key: string]: unknown },
  viewportId: string,
  syncGroupService: any
) {
  synchronizersStore[viewportId].forEach(synchronizerObj => {
    if (!synchronizerObj.id) {
      return;
    }

    const { id, sourceViewports, targetViewports } = synchronizerObj;

    const synchronizer = syncGroupService.getSynchronizer(id);

    if (!synchronizer) {
      return;
    }

    const sourceViewportInfo = sourceViewports.find(
      sourceViewport => sourceViewport.viewportId === viewportId
    );

    const targetViewportInfo = targetViewports.find(
      targetViewport => targetViewport.viewportId === viewportId
    );

    const isSourceViewportInSynchronizer = synchronizer
      .getSourceViewports()
      .find(sourceViewport => sourceViewport.viewportId === viewportId);

    const isTargetViewportInSynchronizer = synchronizer
      .getTargetViewports()
      .find(targetViewport => targetViewport.viewportId === viewportId);

    // if the viewport was previously a source viewport, add it again
    if (sourceViewportInfo && !isSourceViewportInSynchronizer) {
      synchronizer.addSource({
        viewportId: sourceViewportInfo.viewportId,
        renderingEngineId: sourceViewportInfo.renderingEngineId,
      });
    }

    // if the viewport was previously a target viewport, add it again
    if (targetViewportInfo && !isTargetViewportInSynchronizer) {
      synchronizer.addTarget({
        viewportId: targetViewportInfo.viewportId,
        renderingEngineId: targetViewportInfo.renderingEngineId,
      });
    }
  });
}

// Component displayName
OHIFCornerstoneViewport.displayName = 'OHIFCornerstoneViewport';

OHIFCornerstoneViewport.defaultProps = {
  isJumpToMeasurementDisabled: false,
};

OHIFCornerstoneViewport.propTypes = {
  displaySets: PropTypes.array.isRequired,
  dataSource: PropTypes.object.isRequired,
  viewportOptions: PropTypes.object,
  displaySetOptions: PropTypes.arrayOf(PropTypes.any),
  servicesManager: PropTypes.object.isRequired,
  onElementEnabled: PropTypes.func,
  isJumpToMeasurementDisabled: PropTypes.bool,
  // Note: you SHOULD NOT use the initialImageIdOrIndex for manipulation
  // of the imageData in the OHIFCornerstoneViewport. This prop is used
  // to set the initial state of the viewport's first image to render
  initialImageIdOrIndex: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
};

export default OHIFCornerstoneViewport;
