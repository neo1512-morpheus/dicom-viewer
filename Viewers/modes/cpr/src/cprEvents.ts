import { eventTarget } from '@cornerstonejs/core';

export const CPR_CROSSSECTION_SYNC_EVENT = 'CPR_CROSSSECTION_SYNC';

export interface CPRCrossSectionSyncDetail {
  frameIndex: number;
  viewportId?: string;
}

export function emitCPRCrossSectionSync(detail: CPRCrossSectionSyncDetail): void {
  eventTarget.dispatchEvent(new CustomEvent(CPR_CROSSSECTION_SYNC_EVENT, { detail }));
}
