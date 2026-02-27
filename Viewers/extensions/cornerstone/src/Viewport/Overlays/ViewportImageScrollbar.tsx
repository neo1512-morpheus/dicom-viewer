import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import { Enums, Types, eventTarget, utilities } from '@cornerstonejs/core';
import { utilities as csToolsUtils } from '@cornerstonejs/tools';
import { ImageScrollbar } from '@ohif/ui';
import { cprStateService } from '../../../../../modes/cpr/src/CPRStateService';
import { CPR_CROSSSECTION_SYNC_EVENT, emitCPRCrossSectionSync } from '../../../../../modes/cpr/src/cprEvents';

function CornerstoneImageScrollbar({
  viewportData,
  viewportId,
  element,
  imageSliceData,
  setImageSliceData,
  scrollbarHeight,
  servicesManager,
}: withAppTypes) {
  const { cineService, cornerstoneViewportService } = servicesManager.services;

  const getLogicalViewportId = () => {
    const viewportInfo = cornerstoneViewportService.getViewportInfo(viewportId);
    return viewportInfo?.viewportOptions?.viewportId || viewportId;
  };

  const isCPRPanoViewport = () => getLogicalViewportId() === 'cpr-pano';
  const isCPRCrossSectionViewport = () => getLogicalViewportId() === 'cpr-crosssection';

  const onImageScrollbarChange = (imageIndex, viewportId) => {
    if (isCPRCrossSectionViewport() && cprStateService.hasData()) {
      const frameCount = cprStateService.getFrames().length;
      if (frameCount > 0) {
        const frameIndex = Math.max(0, Math.min(Math.round(imageIndex), frameCount - 1));
        cprStateService.setCurrentFrameIndex(frameIndex);
        emitCPRCrossSectionSync({ frameIndex, viewportId: 'cpr-crosssection' });
      }
      return;
    }

    if (isCPRPanoViewport()) {
      return;
    }

    const viewport = cornerstoneViewportService.getCornerstoneViewport(viewportId);

    const { isCineEnabled } = cineService.getState();

    if (isCineEnabled) {
      // on image scrollbar change, stop the CINE if it is playing
      cineService.stopClip(element, { viewportId });
      cineService.setCine({ id: viewportId, isPlaying: false });
    }

    csToolsUtils.jumpToSlice(viewport.element, {
      imageIndex,
      debounceLoading: true,
    });
  };

  useEffect(() => {
    if (!viewportData) {
      return;
    }

    const viewport = cornerstoneViewportService.getCornerstoneViewport(viewportId);

    if (!viewport) {
      return;
    }

    if (viewportData.viewportType === Enums.ViewportType.STACK) {
      if (isCPRPanoViewport()) {
        setImageSliceData({ imageIndex: 0, numberOfSlices: 0 });
        return;
      }

      const imageIndex = viewport.getCurrentImageIdIndex();

      setImageSliceData({
        imageIndex: imageIndex,
        numberOfSlices: viewportData.data[0].imageIds.length,
      });

      return;
    }

    if (viewportData.viewportType === Enums.ViewportType.ORTHOGRAPHIC) {
      if (isCPRCrossSectionViewport() && cprStateService.hasData()) {
        const frames = cprStateService.getFrames();
        setImageSliceData({
          imageIndex: cprStateService.getCurrentFrameIndex(),
          numberOfSlices: frames.length,
        });
        return;
      }

      const sliceData = utilities.getImageSliceDataForVolumeViewport(
        viewport as Types.IVolumeViewport
      );

      if (!sliceData) {
        return;
      }

      const { imageIndex, numberOfSlices } = sliceData;
      setImageSliceData({ imageIndex, numberOfSlices });
    }
  }, [viewportId, viewportData]);

  useEffect(() => {
    if (viewportData?.viewportType !== Enums.ViewportType.STACK) {
      return;
    }

    if (isCPRPanoViewport()) {
      return;
    }

    const updateStackIndex = event => {
      const { newImageIdIndex } = event.detail;
      // find the index of imageId in the imageIds
      setImageSliceData({
        imageIndex: newImageIdIndex,
        numberOfSlices: viewportData.data[0].imageIds.length,
      });
    };

    element.addEventListener(Enums.Events.STACK_VIEWPORT_SCROLL, updateStackIndex);

    return () => {
      element.removeEventListener(Enums.Events.STACK_VIEWPORT_SCROLL, updateStackIndex);
    };
  }, [viewportData, element]);

  useEffect(() => {
    if (viewportData?.viewportType !== Enums.ViewportType.ORTHOGRAPHIC) {
      return;
    }

    if (isCPRCrossSectionViewport()) {
      return;
    }

    const updateVolumeIndex = event => {
      const { imageIndex, numberOfSlices } = event.detail;
      // find the index of imageId in the imageIds
      setImageSliceData({ imageIndex, numberOfSlices });
    };

    element.addEventListener(Enums.Events.VOLUME_NEW_IMAGE, updateVolumeIndex);

    return () => {
      element.removeEventListener(Enums.Events.VOLUME_NEW_IMAGE, updateVolumeIndex);
    };
  }, [viewportData, element]);

  useEffect(() => {
    if (!isCPRCrossSectionViewport()) {
      return;
    }

    const onCPRSync = (evt: Event) => {
      const detail = (evt as CustomEvent<{ frameIndex: number }>).detail;
      if (!detail || !Number.isFinite(detail.frameIndex)) {
        return;
      }

      const frames = cprStateService.getFrames();
      if (!frames.length) {
        return;
      }

      const frameIndex = Math.max(0, Math.min(Math.round(detail.frameIndex), frames.length - 1));
      setImageSliceData({
        imageIndex: frameIndex,
        numberOfSlices: frames.length,
      });
    };

    eventTarget.addEventListener(CPR_CROSSSECTION_SYNC_EVENT, onCPRSync);

    return () => {
      eventTarget.removeEventListener(CPR_CROSSSECTION_SYNC_EVENT, onCPRSync);
    };
  }, [viewportId]);

  if (isCPRPanoViewport()) {
    return null;
  }

  return (
    <ImageScrollbar
      onChange={evt => onImageScrollbarChange(evt, viewportId)}
      max={imageSliceData.numberOfSlices ? imageSliceData.numberOfSlices - 1 : 0}
      height={scrollbarHeight}
      value={imageSliceData.imageIndex}
    />
  );
}

CornerstoneImageScrollbar.propTypes = {
  viewportData: PropTypes.object,
  viewportId: PropTypes.string.isRequired,
  element: PropTypes.instanceOf(Element),
  scrollbarHeight: PropTypes.string,
  imageSliceData: PropTypes.object.isRequired,
  setImageSliceData: PropTypes.func.isRequired,
  servicesManager: PropTypes.object.isRequired,
};

export default CornerstoneImageScrollbar;
