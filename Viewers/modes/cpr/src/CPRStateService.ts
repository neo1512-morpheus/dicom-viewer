import type { CPRFrame, Point3 } from './cprMath';

interface CPRArchData {
  controlPoints: Point3[];
  frames: CPRFrame[];
  sourceVolumeId: string;
  archAnnotationUID?: string | null;
  currentFrameIndex: number;
}

class CPRStateService {
  private archData: CPRArchData | null = null;

  setArchData(
    controlPoints: Point3[],
    frames: CPRFrame[],
    sourceVolumeId: string,
    archAnnotationUID: string | null = null
  ): void {
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
    };
  }

  hasData(): boolean {
    return !!this.archData && this.archData.frames.length > 0;
  }

  clear(): void {
    this.archData = null;
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
}

export const cprStateService = new CPRStateService();
