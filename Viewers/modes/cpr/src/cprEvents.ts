import { eventTarget } from '@cornerstonejs/core';

export const CPR_CROSSSECTION_SYNC_EVENT = 'CPR_CROSSSECTION_SYNC';
export const CPR_PANO_HOST_ATTACHED_EVENT = 'CPR_PANO_HOST_ATTACHED';

export interface CPRCrossSectionSyncDetail {
  frameIndex: number;
  viewportId?: string;
}

export interface CPRPanoHostAttachedDetail {
  actorUID?: string;
  runId?: string;
  viewportId?: string;
}

export function emitCPRCrossSectionSync(detail: CPRCrossSectionSyncDetail): void {
  console.log('[CPR-CROSSSECTION-SYNC]', detail);
  eventTarget.dispatchEvent(new CustomEvent(CPR_CROSSSECTION_SYNC_EVENT, { detail }));
}

export function emitCPRPanoHostAttached(detail: CPRPanoHostAttachedDetail): void {
  console.log('[CPR-PANO-HOST-ATTACHED]', detail);
  eventTarget.dispatchEvent(new CustomEvent(CPR_PANO_HOST_ATTACHED_EVENT, { detail }));
}
