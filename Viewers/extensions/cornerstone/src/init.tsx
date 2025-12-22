import OHIF, { Types, errorHandler } from '@ohif/core';
import React from 'react';

import * as cornerstone from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import {
  init as cs3DInit,
  eventTarget,
  EVENTS,
  metaData,
  volumeLoader,
  imageLoadPoolManager,
  getEnabledElement,
  Settings,
  utilities as csUtilities,
  Enums as csEnums,
} from '@cornerstonejs/core';
import {
  cornerstoneStreamingImageVolumeLoader,
  cornerstoneStreamingDynamicImageVolumeLoader,
} from '@cornerstonejs/streaming-image-volume-loader';

import initWADOImageLoader from './initWADOImageLoader';
import initCornerstoneTools from './initCornerstoneTools';

import { connectToolsToMeasurementService } from './initMeasurementService';
import initCineService from './initCineService';
import interleaveCenterLoader from './utils/interleaveCenterLoader';
import nthLoader from './utils/nthLoader';
import interleaveTopToBottom from './utils/interleaveTopToBottom';
import initContextMenu from './initContextMenu';
import initDoubleClick from './initDoubleClick';
import initViewTiming from './utils/initViewTiming';
import { colormaps } from './utils/colormaps';
import { detectGPUCapability } from './utils/gpuCapabilityCheck';

const { registerColormap } = csUtilities.colormap;

// TODO: Cypress tests are currently grabbing this from the window?
window.cornerstone = cornerstone;
window.cornerstoneTools = cornerstoneTools;
/**
 *
 */
export default async function init({
  servicesManager,
  commandsManager,
  extensionManager,
  appConfig,
}: Types.Extensions.ExtensionParams): Promise<void> {
  // =====================================================
  // GPU CAPABILITY DETECTION - For logging/debugging only
  // No behavior changes - uniform treatment for all users
  // =====================================================
  const gpuCapability = detectGPUCapability();

  // Store GPU info for debugging (NOT used to change behavior)
  appConfig.gpuRenderer = gpuCapability.renderer;
  appConfig.isLowEndGPU = gpuCapability.isWeakGPU;

  console.log('[GPU Info] Detected:', gpuCapability.renderer);
  if (gpuCapability.isWeakGPU) {
    console.log('[GPU Info] Note: This is an integrated/low-end GPU. Large 3D scans may be slow.');
  }
  // =====================================================

  // Note: this should run first before initializing the cornerstone
  // DO NOT CHANGE THE ORDER
  const value = appConfig.useSharedArrayBuffer;
  let sharedArrayBufferDisabled = false;

  if (value === 'AUTO') {
    cornerstone.setUseSharedArrayBuffer(csEnums.SharedArrayBufferModes.AUTO);
  } else if (value === 'FALSE' || value === false) {
    cornerstone.setUseSharedArrayBuffer(csEnums.SharedArrayBufferModes.FALSE);
    sharedArrayBufferDisabled = true;
  } else {
    cornerstone.setUseSharedArrayBuffer(csEnums.SharedArrayBufferModes.TRUE);
  }

  await cs3DInit({
    rendering: {
      preferSizeOverAccuracy: appConfig.preferSizeOverAccuracy ?? true, // Default to memory-efficient 16-bit textures
      useNorm16Texture: Boolean(appConfig.useNorm16Texture),
    },
  });

  // For debugging e2e tests that are failing on CI
  cornerstone.setUseCPURendering(Boolean(appConfig.useCPURendering));

  cornerstone.setConfiguration({
    ...cornerstone.getConfiguration(),
    rendering: {
      ...cornerstone.getConfiguration().rendering,
      strictZSpacingForVolumeViewport: appConfig.strictZSpacingForVolumeViewport,
    },
  });

  // For debugging large datasets, otherwise prefer the defaults
  const { maxCacheSize } = appConfig;
  if (maxCacheSize) {
    cornerstone.cache.setMaxCacheSize(maxCacheSize);
  }

  initCornerstoneTools();

  Settings.getRuntimeSettings().set('useCursors', Boolean(appConfig.useCursors));

  const {
    userAuthenticationService,
    customizationService,
    uiModalService,
    uiNotificationService,
    cornerstoneViewportService,
    hangingProtocolService,
    viewportGridService,
    stateSyncService,
  } = servicesManager.services;

  // =====================================================
  // GLOBAL WebGL ERROR DETECTION
  // Catches WebGL errors from VTK.js and other sources
  // =====================================================
  let hasShownGPUWarning = false;

  const showGPUWarning = () => {
    if (hasShownGPUWarning) return;
    hasShownGPUWarning = true;
    console.warn('[GPU] Showing GPU memory warning to user');
    uiNotificationService.show({
      title: 'GPU Memory Limit Reached',
      message: 'This 3D scan is too large for your graphics card. Try closing other browser tabs or reloading the page.',
      type: 'warning',
      duration: 15000,
    });
  };

  // Listen for WebGL context loss on ALL canvases (including VTK.js internal ones)
  const attachCanvasListeners = () => {
    const canvases = document.querySelectorAll('canvas');
    canvases.forEach((canvas) => {
      if (!canvas.dataset.gpuListenerAttached) {
        canvas.dataset.gpuListenerAttached = 'true';
        canvas.addEventListener('webglcontextlost', (e) => {
          e.preventDefault();
          console.warn('[GPU] WebGL context lost on canvas');
          showGPUWarning();
        });
      }
    });
  };

  // Use MutationObserver to catch dynamically created canvases
  const observer = new MutationObserver(() => {
    attachCanvasListeners();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // Also attach to existing canvases
  attachCanvasListeners();

  // Global error handler for uncaught WebGL errors
  window.addEventListener('error', (event) => {
    const errorMsg = event.message || '';
    if (
      errorMsg.includes('WebGL') ||
      errorMsg.includes('CONTEXT_LOST') ||
      errorMsg.includes('GPU')
    ) {
      console.warn('[GPU] Caught global WebGL error:', errorMsg);
      showGPUWarning();
    }
  });
  // =====================================================

  window.services = servicesManager.services;
  window.extensionManager = extensionManager;
  window.commandsManager = commandsManager;

  if (
    appConfig.showWarningMessageForCrossOrigin &&
    !window.crossOriginIsolated &&
    !sharedArrayBufferDisabled
  ) {
    uiNotificationService.show({
      title: 'Cross Origin Isolation',
      message:
        'Cross Origin Isolation is not enabled, read more about it here: https://docs.ohif.org/faq/',
      type: 'warning',
    });
  }

  if (appConfig.showCPUFallbackMessage && cornerstone.getShouldUseCPURendering()) {
    _showCPURenderingModal(uiModalService, hangingProtocolService);
  }

  // Stores a map from `lutPresentationId` to a Presentation object so that
  // an OHIFCornerstoneViewport can be redisplayed with the same LUT
  stateSyncService.register('lutPresentationStore', { clearOnModeExit: true });

  // Stores synchronizers state to be restored
  stateSyncService.register('synchronizersStore', { clearOnModeExit: true });

  // Stores a map from `positionPresentationId` to a Presentation object so that
  // an OHIFCornerstoneViewport can be redisplayed with the same position
  stateSyncService.register('positionPresentationStore', {
    clearOnModeExit: true,
  });

  // Stores the entire ViewportGridService getState when toggling to one up
  // (e.g. via a double click) so that it can be restored when toggling back.
  stateSyncService.register('toggleOneUpViewportGridStore', {
    clearOnModeExit: true,
  });

  const labelmapRepresentation = cornerstoneTools.Enums.SegmentationRepresentations.Labelmap;
  const contourRepresentation = cornerstoneTools.Enums.SegmentationRepresentations.Contour;

  cornerstoneTools.segmentation.config.setGlobalRepresentationConfig(labelmapRepresentation, {
    fillAlpha: 0.5,
    fillAlphaInactive: 0.2,
    outlineOpacity: 1,
    outlineOpacityInactive: 0.65,
  });

  cornerstoneTools.segmentation.config.setGlobalRepresentationConfig(contourRepresentation, {
    renderFill: false,
  });

  const metadataProvider = OHIF.classes.MetadataProvider;

  volumeLoader.registerVolumeLoader(
    'cornerstoneStreamingImageVolume',
    cornerstoneStreamingImageVolumeLoader
  );

  volumeLoader.registerVolumeLoader(
    'cornerstoneStreamingDynamicImageVolume',
    cornerstoneStreamingDynamicImageVolumeLoader
  );

  hangingProtocolService.registerImageLoadStrategy('interleaveCenter', interleaveCenterLoader);
  hangingProtocolService.registerImageLoadStrategy('interleaveTopToBottom', interleaveTopToBottom);
  hangingProtocolService.registerImageLoadStrategy('nth', nthLoader);

  // add metadata providers
  metaData.addProvider(
    csUtilities.calibratedPixelSpacingMetadataProvider.get.bind(
      csUtilities.calibratedPixelSpacingMetadataProvider
    )
  ); // this provider is required for Calibration tool
  metaData.addProvider(metadataProvider.get.bind(metadataProvider), 9999);

  imageLoadPoolManager.maxNumRequests = {
    interaction: appConfig?.maxNumRequests?.interaction || 100,
    thumbnail: appConfig?.maxNumRequests?.thumbnail || 75,
    prefetch: appConfig?.maxNumRequests?.prefetch || 10,
  };

  initWADOImageLoader(userAuthenticationService, appConfig, extensionManager);

  /* Measurement Service */
  this.measurementServiceSource = connectToolsToMeasurementService(servicesManager);

  initCineService(servicesManager);

  // When a custom image load is performed, update the relevant viewports
  hangingProtocolService.subscribe(
    hangingProtocolService.EVENTS.CUSTOM_IMAGE_LOAD_PERFORMED,
    volumeInputArrayMap => {
      for (const entry of volumeInputArrayMap.entries()) {
        const [viewportId, volumeInputArray] = entry;
        const viewport = cornerstoneViewportService.getCornerstoneViewport(viewportId);

        const ohifViewport = cornerstoneViewportService.getViewportInfo(viewportId);

        const { lutPresentationStore, positionPresentationStore } = stateSyncService.getState();
        const { presentationIds } = ohifViewport.getViewportOptions();
        const presentations = {
          positionPresentation: positionPresentationStore[presentationIds?.positionPresentationId],
          lutPresentation: lutPresentationStore[presentationIds?.lutPresentationId],
        };

        cornerstoneViewportService.setVolumesForViewport(viewport, volumeInputArray, presentations);
      }
    }
  );

  // resize the cornerstone viewport service when the grid size changes
  // IMPORTANT: this should happen outside of the OHIFCornerstoneViewport
  // since it will trigger a rerender of each viewport and each resizing
  // the offscreen canvas which would result in a performance hit, this should
  // done only once per grid resize here. Doing it once here, allows us to reduce
  // the refreshRage(in ms) to 10 from 50. I tried with even 1 or 5 ms it worked fine
  viewportGridService.subscribe(viewportGridService.EVENTS.GRID_SIZE_CHANGED, () => {
    cornerstoneViewportService.resize(true);
  });

  initContextMenu({
    cornerstoneViewportService,
    customizationService,
    commandsManager,
  });

  initDoubleClick({
    customizationService,
    commandsManager,
  });

  /**
   * Runs error handler for failed requests.
   * @param event
   */
  const imageLoadFailedHandler = ({ detail }) => {
    const handler = errorHandler.getHTTPErrorHandler();
    handler(detail.error);
  };

  eventTarget.addEventListener(EVENTS.STACK_VIEWPORT_NEW_STACK, evt => {
    const { element } = evt.detail;
    cornerstoneTools.utilities.stackContextPrefetch.enable(element);
  });
  eventTarget.addEventListener(EVENTS.IMAGE_LOAD_FAILED, imageLoadFailedHandler);
  eventTarget.addEventListener(EVENTS.IMAGE_LOAD_ERROR, imageLoadFailedHandler);

  function elementEnabledHandler(evt) {
    const { element } = evt.detail;

    element.addEventListener(EVENTS.CAMERA_RESET, evt => {
      const { element } = evt.detail;
      const { viewportId } = getEnabledElement(element);
      commandsManager.runCommand('resetCrosshairs', { viewportId });
    });

    // eventTarget.addEventListener(EVENTS.STACK_VIEWPORT_NEW_STACK, toolbarEventListener);

    initViewTiming({ element });
  }

  function elementDisabledHandler(evt) {
    const { element } = evt.detail;

    // element.removeEventListener(EVENTS.CAMERA_RESET, resetCrosshairs);

    // TODO - consider removing the callback when all elements are gone
    // eventTarget.removeEventListener(
    //   EVENTS.STACK_VIEWPORT_NEW_STACK,
    //   newStackCallback
    // );
  }

  eventTarget.addEventListener(EVENTS.ELEMENT_ENABLED, elementEnabledHandler.bind(null));

  eventTarget.addEventListener(EVENTS.ELEMENT_DISABLED, elementDisabledHandler.bind(null));
  colormaps.forEach(registerColormap);

  // Event listener
  eventTarget.addEventListenerDebounced(
    EVENTS.ERROR_EVENT,
    ({ detail }) => {
      uiNotificationService.show({
        title: detail.type,
        message: detail.message,
        type: 'error',
      });
    },
    1000
  );
}

function CPUModal() {
  return (
    <div>
      <p>
        Your computer does not have enough GPU power to support the default GPU rendering mode. OHIF
        has switched to CPU rendering mode. Please note that CPU rendering does not support all
        features such as Volume Rendering, Multiplanar Reconstruction, and Segmentation Overlays.
      </p>
    </div>
  );
}

function _showCPURenderingModal(uiModalService, hangingProtocolService) {
  const callback = progress => {
    if (progress === 100) {
      uiModalService.show({
        content: CPUModal,
        title: 'OHIF Fell Back to CPU Rendering',
      });

      return true;
    }
  };

  const { unsubscribe } = hangingProtocolService.subscribe(
    hangingProtocolService.EVENTS.PROTOCOL_CHANGED,
    () => {
      const done = callback(100);

      if (done) {
        unsubscribe();
      }
    }
  );
}
