import type { CPRFrame, Point3 } from './cprMath';

interface CPRArchData {
  controlPoints: Point3[];
  frames: CPRFrame[];
  sourceVolumeId: string;
  archAnnotationUID?: string | null;
  currentFrameIndex: number;
  crossSectionVerticalCenterOffsetMm: number;
  crossSectionVerticalCenterOffsetsMm: number[];
}

class CPRStateService {
  private archData: CPRArchData | null = null;
  private axialTransitioning = false;
  private listeners = new Set<() => void>();

  private notify(): void {
    this.listeners.forEach(listener => {
      try {
        listener();
      } catch (error) {
        console.warn('[CPR] Failed to notify CPR state listener.', error);
      }
    });
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  setArchData(
    controlPoints: Point3[],
    frames: CPRFrame[],
    sourceVolumeId: string,
    archAnnotationUID: string | null = null,
    crossSectionVerticalCenterOffsetMm = 0,
    crossSectionVerticalCenterOffsetsMm: number[] = []
  ): void {
    const safeCrossSectionVerticalCenterOffsetMm = Number.isFinite(crossSectionVerticalCenterOffsetMm)
      ? Number(crossSectionVerticalCenterOffsetMm)
      : 0;
    const safeCrossSectionVerticalCenterOffsetsMm = frames.map((_, index) => {
      const offset = Number(crossSectionVerticalCenterOffsetsMm[index]);
      return Number.isFinite(offset) ? offset : safeCrossSectionVerticalCenterOffsetMm;
    });

    this.archData = {
      controlPoints: controlPoints.map(p => [p[0], p[1], p[2]]),
      frames: frames.map(frame => ({
        ...frame,
        position: [frame.position[0], frame.position[1], frame.position[2]],
        T: [frame.T[0], frame.T[1], frame.T[2]],
        N_camera: [frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]],
        N_slab: [frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]],
        S: [frame.S[0], frame.S[1], frame.S[2]],
      })),
      sourceVolumeId,
      archAnnotationUID,
      currentFrameIndex: Math.max(0, Math.floor(frames.length / 2)),
      crossSectionVerticalCenterOffsetMm: safeCrossSectionVerticalCenterOffsetMm,
      crossSectionVerticalCenterOffsetsMm: safeCrossSectionVerticalCenterOffsetsMm,
    };
  }

  hasData(): boolean {
    return !!this.archData && this.archData.frames.length > 0;
  }

  clear(): void {
    this.archData = null;
    this.setAxialTransitioning(false);
  }

  isAxialTransitioning(): boolean {
    return this.axialTransitioning;
  }

  setAxialTransitioning(nextValue: boolean): void {
    const safeNextValue = Boolean(nextValue);
    if (this.axialTransitioning === safeNextValue) {
      return;
    }

    this.axialTransitioning = safeNextValue;
    this.notify();
  }

  getControlPoints(): Point3[] {
    if (!this.archData) {
      return [];
    }

    return this.archData.controlPoints.map(p => [p[0], p[1], p[2]]);
  }

  getFrames(): CPRFrame[] {
    if (!this.archData) {
      return [];
    }

    return this.archData.frames.map(frame => ({
      ...frame,
      position: [frame.position[0], frame.position[1], frame.position[2]],
      T: [frame.T[0], frame.T[1], frame.T[2]],
      N_camera: [frame.N_camera[0], frame.N_camera[1], frame.N_camera[2]],
      N_slab: [frame.N_slab[0], frame.N_slab[1], frame.N_slab[2]],
      S: [frame.S[0], frame.S[1], frame.S[2]],
    }));
  }

  getSourceVolumeId(): string | null {
    return this.archData?.sourceVolumeId ?? null;
  }

  getArchAnnotationUID(): string | null {
    return this.archData?.archAnnotationUID ?? null;
  }

  getCurrentFrameIndex(): number {
    if (!this.archData) {
      return 0;
    }

    const maxIndex = Math.max(0, this.archData.frames.length - 1);
    return Math.max(0, Math.min(this.archData.currentFrameIndex, maxIndex));
  }

  setCurrentFrameIndex(frameIndex: number): void {
    if (!this.archData) {
      return;
    }

    const maxIndex = Math.max(0, this.archData.frames.length - 1);
    this.archData.currentFrameIndex = Math.max(0, Math.min(Math.round(frameIndex), maxIndex));
  }

  getCrossSectionVerticalCenterOffsetMm(frameIndex?: number): number {
    if (
      this.archData &&
      typeof frameIndex === 'number' &&
      Number.isFinite(frameIndex) &&
      this.archData.crossSectionVerticalCenterOffsetsMm.length > 0
    ) {
      const maxIndex = Math.max(0, this.archData.crossSectionVerticalCenterOffsetsMm.length - 1);
      const safeIndex = Math.max(0, Math.min(Math.round(frameIndex), maxIndex));
      const localOffset = this.archData.crossSectionVerticalCenterOffsetsMm[safeIndex];
      if (Number.isFinite(localOffset)) {
        return Number(localOffset);
      }
    }

    const offset = this.archData?.crossSectionVerticalCenterOffsetMm;
    return Number.isFinite(offset) ? Number(offset) : 0;
  }

  getCrossSectionVerticalCenterOffsetsMm(): number[] {
    if (!this.archData) {
      return [];
    }

    return this.archData.crossSectionVerticalCenterOffsetsMm.map(offset =>
      Number.isFinite(offset) ? Number(offset) : 0
    );
  }
}

export const cprStateService = new CPRStateService();
