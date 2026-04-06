export type WorkerCompatiblePanoReconstructionMode =
  | 'legacy'
  | 'virtualPanoPhase1'
  | 'virtualPano';

export type PanoReconstructionMode = WorkerCompatiblePanoReconstructionMode | 'virtualPanoV2';

export type PanoDisplayedOutputMode =
  | 'legacy'
  | 'workerGpuPhase2'
  | 'virtualPanoPhase2'
  | 'virtualPanoV2Phase0';

export const PANO_V2_RECONSTRUCTION_MODE = 'virtualPanoV2';
export const PANO_V2_DISPLAYED_OUTPUT_MODE = 'virtualPanoV2Phase0';

export function isWorkerCompatiblePanoReconstructionMode(
  value: unknown
): value is WorkerCompatiblePanoReconstructionMode {
  return value === 'legacy' || value === 'virtualPanoPhase1' || value === 'virtualPano';
}

export function isPanoV2ReconstructionMode(
  value: unknown
): value is typeof PANO_V2_RECONSTRUCTION_MODE {
  return value === PANO_V2_RECONSTRUCTION_MODE;
}

export function resolvePanoReconstructionMode(value: unknown): PanoReconstructionMode {
  return isWorkerCompatiblePanoReconstructionMode(value) || isPanoV2ReconstructionMode(value)
    ? value
    : 'legacy';
}

export function resolveWorkerCompatiblePanoReconstructionMode(
  value: unknown
): WorkerCompatiblePanoReconstructionMode {
  const resolvedMode = resolvePanoReconstructionMode(value);
  return resolvedMode === PANO_V2_RECONSTRUCTION_MODE ? 'virtualPano' : resolvedMode;
}

export function isVirtualPanoLikeReconstructionMode(value: unknown): boolean {
  const resolvedMode = resolvePanoReconstructionMode(value);
  return resolvedMode === 'virtualPano' || resolvedMode === PANO_V2_RECONSTRUCTION_MODE;
}
