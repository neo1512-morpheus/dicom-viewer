# OHIF DICOM JSON Data Source Standard
The backend must return a JSON object with this TypeScript structure.
**CRITICAL:** The fields marked with (*) are required for 3D MPR rendering.

interface OHIFStudy {
  studies: Array<{
    StudyInstanceUID: string;
    StudyDescription: string;
    StudyDate: string;
    StudyTime: string;
    PatientName: string;
    PatientID: string;
    AccessionNumber: string;
    PatientAge: string;
    PatientSex: string;
    series: Array<{
      SeriesInstanceUID: string;
      SeriesDescription: string;
      SeriesNumber: number;
      Modality: string;
      SliceThickness: number;
      SeriesDate?: string;     // Added
      NumInstances?: number;   // Added
      instances: Array<{
        metadata: {
          SOPInstanceUID: string;
          SeriesInstanceUID: string;
          StudyInstanceUID: string;
          InstanceNumber: number;
          
          // --- IMAGE DATA ---
          Rows: number;
          Columns: number;
          BitsAllocated: number;
          BitsStored: number;
          PixelRepresentation: number;
          SamplesPerPixel: number;        // * Required
          PhotometricInterpretation: string; // * Required (e.g., MONOCHROME2)
          HighBit: number;                // * Required
          SOPClassUID: string;            // * Required (CT Image Storage)

          // --- 3D GEOMETRY (CRITICAL FOR MPR) ---
          PixelSpacing: [number, number];       // * [RowSpacing, ColSpacing]
          ImageOrientationPatient: number[];    // * [1,0,0,0,1,0] (Axial)
          ImagePositionPatient: number[];       // * [x, y, z] (Must change per slice)
          FrameOfReferenceUID: string;          // * Links slices together

          // --- WINDOWING ---
          WindowCenter: number;
          WindowWidth: number;
        };
        url: string; // Must use `dicomweb:` prefix + Signed URL
      }>;
    }>;
  }>;
}

## Defaults for CBCT Prototype (If parsing fails):
- Modality: "CT"
- SOPClassUID: "1.2.840.10008.5.1.4.1.1.2" (CT Image Storage)
- PixelSpacing: [0.5, 0.5]
- ImageOrientationPatient: [1, 0, 0, 0, 1, 0]
- FrameOfReferenceUID: Use StudyInstanceUID
- ImagePositionPatient: [0, 0, index * 0.5] (Stacking logic)