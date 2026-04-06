import {
  PANO_V2_DISPLAYED_OUTPUT_MODE,
  type PanoReconstructionMode,
  resolvePanoReconstructionMode,
  resolveWorkerCompatiblePanoReconstructionMode,
} from './panoV2Types';

export const PANO_V2_IMPLEMENTATION_PHASE = 4;

export interface PanoV2WorkerRouteDecision {
  requestedReconstructionMode: PanoReconstructionMode;
  delegatedReconstructionMode: ReturnType<typeof resolveWorkerCompatiblePanoReconstructionMode>;
  usesPanoV2Bridge: boolean;
}

export function resolvePanoV2WorkerRoute(value: unknown): PanoV2WorkerRouteDecision {
  const requestedReconstructionMode = resolvePanoReconstructionMode(value);
  const delegatedReconstructionMode =
    resolveWorkerCompatiblePanoReconstructionMode(requestedReconstructionMode);

  return {
    requestedReconstructionMode,
    delegatedReconstructionMode,
    usesPanoV2Bridge: requestedReconstructionMode === 'virtualPanoV2',
  };
}

export function getPanoV2WorkerRendererFamily(
  baseRendererFamily: string,
  routeDecision: PanoV2WorkerRouteDecision
): string {
  return routeDecision.usesPanoV2Bridge
    ? `pano-v2-phase${PANO_V2_IMPLEMENTATION_PHASE}:${baseRendererFamily}`
    : baseRendererFamily;
}

export function getPanoV2WorkerRouteReason(
  baseRouteReason: string,
  routeDecision: PanoV2WorkerRouteDecision
): string {
  return routeDecision.usesPanoV2Bridge
    ? `pano-v2-phase${PANO_V2_IMPLEMENTATION_PHASE}:${baseRouteReason}`
    : baseRouteReason;
}

export function applyPanoV2WorkerRouteDiagnostic(
  diagnosticPayload: Record<string, unknown>,
  routeDecision: PanoV2WorkerRouteDecision
): Record<string, unknown> {
  if (!routeDecision.usesPanoV2Bridge) {
    return diagnosticPayload;
  }

  const outputSelection =
    diagnosticPayload.outputSelection && typeof diagnosticPayload.outputSelection === 'object'
      ? {
          ...(diagnosticPayload.outputSelection as Record<string, unknown>),
          requestedReconstructionMode: routeDecision.requestedReconstructionMode,
          candidateReconstructionMode: routeDecision.requestedReconstructionMode,
          selectedReconstructionMode: routeDecision.requestedReconstructionMode,
          panoV2DelegatedReconstructionMode: routeDecision.delegatedReconstructionMode,
          panoV2DisplayedOutputMode: PANO_V2_DISPLAYED_OUTPUT_MODE,
        }
      : diagnosticPayload.outputSelection;
  const geometryPhase =
    diagnosticPayload.panoV2Geometry &&
    typeof diagnosticPayload.panoV2Geometry === 'object' &&
    typeof (diagnosticPayload.panoV2Geometry as { phase?: unknown }).phase === 'number'
      ? Number((diagnosticPayload.panoV2Geometry as { phase: number }).phase)
      : 0;
  const volumePhase =
    diagnosticPayload.panoV2Volume &&
    typeof diagnosticPayload.panoV2Volume === 'object' &&
    typeof (diagnosticPayload.panoV2Volume as { phase?: unknown }).phase === 'number'
      ? Number((diagnosticPayload.panoV2Volume as { phase: number }).phase)
      : 0;
  const layersPhase =
    diagnosticPayload.panoV2Layers &&
    typeof diagnosticPayload.panoV2Layers === 'object' &&
    typeof (diagnosticPayload.panoV2Layers as { phase?: unknown }).phase === 'number'
      ? Number((diagnosticPayload.panoV2Layers as { phase: number }).phase)
      : 0;
  const fusionPhase =
    diagnosticPayload.panoV2Fusion &&
    typeof diagnosticPayload.panoV2Fusion === 'object' &&
    typeof (diagnosticPayload.panoV2Fusion as { phase?: unknown }).phase === 'number'
      ? Number((diagnosticPayload.panoV2Fusion as { phase: number }).phase)
      : 0;
  const effectivePhase = Math.max(
    PANO_V2_IMPLEMENTATION_PHASE,
    geometryPhase,
    volumePhase,
    layersPhase,
    fusionPhase
  );
  const rendererFamily =
    fusionPhase >= 4 ? 'pano-v2-fusion' : (diagnosticPayload.rendererFamily as string | undefined);

  return {
    ...diagnosticPayload,
    ...(rendererFamily ? { rendererFamily } : {}),
    pipelineMode: routeDecision.requestedReconstructionMode,
    reconstructionMode: routeDecision.requestedReconstructionMode,
    requestedReconstructionMode: routeDecision.requestedReconstructionMode,
    candidateReconstructionMode: routeDecision.requestedReconstructionMode,
    outputSelection,
    panoV2Route: {
      enabled: true,
      phase: effectivePhase,
      dispatchMode: 'bridge',
      requestedReconstructionMode: routeDecision.requestedReconstructionMode,
      delegatedReconstructionMode: routeDecision.delegatedReconstructionMode,
      displayedOutputMode: PANO_V2_DISPLAYED_OUTPUT_MODE,
    },
  };
}
