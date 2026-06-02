import {
  PANO_V2_RECONSTRUCTION_MODE,
  type PanoDisplayedOutputMode,
  type PanoReconstructionMode,
  isPanoV2ReconstructionMode,
  isVirtualPanoLikeReconstructionMode,
} from './panoV2Types';

export function getPanoV2Phase0ReconstructionMode(): PanoReconstructionMode {
  return PANO_V2_RECONSTRUCTION_MODE;
}

export function resolvePanoDisplayedOutputMode(params: {
  backend: 'gpu' | 'cpu';
  phase2GatePassed: boolean | null;
  reconstructionMode: string | null;
  pipelineMode: string | null;
}): PanoDisplayedOutputMode {
  const resolvedReconstructionMode = params.reconstructionMode || params.pipelineMode || 'legacy';

  if (params.backend === 'cpu' && isPanoV2ReconstructionMode(resolvedReconstructionMode)) {
    return 'virtualPanoV2Phase0';
  }

  if (params.backend === 'cpu' && isVirtualPanoLikeReconstructionMode(resolvedReconstructionMode)) {
    return 'virtualPanoPhase2';
  }

  if (params.backend === 'gpu' && params.phase2GatePassed !== false) {
    return 'workerGpuPhase2';
  }

  return 'legacy';
}

export function isVirtualPanoLikeCandidateMode(value: unknown): boolean {
  return isVirtualPanoLikeReconstructionMode(value);
}
