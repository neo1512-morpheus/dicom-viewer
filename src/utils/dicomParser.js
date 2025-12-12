import dcmjs from 'dcmjs';
import JSZip from 'jszip';

export async function parseDicomFiles(inputFiles) {
  const parsedFiles = [];
  const processedBuffers = [];

  console.log("Processing input files:", inputFiles);

  // --- 1. Unzip ---
  for (const file of inputFiles) {
    if (file.name.toLowerCase().endsWith('.zip')) {
      console.log(`Unzipping ${file.name}...`);
      try {
        const zip = new JSZip();
        const zipContent = await zip.loadAsync(file);
        
        const entries = Object.entries(zipContent.files);
        for (const [filename, zipEntry] of entries) {
          if (!zipEntry.dir && !filename.includes('__MACOSX') && !filename.startsWith('.')) {
            const buffer = await zipEntry.async('arraybuffer');
            const cleanName = filename.split('/').pop();
            processedBuffers.push({ name: cleanName, buffer });
          }
        }
      } catch (err) {
        console.error("Error unzipping file:", err);
        throw new Error("Failed to unzip. Is it a valid zip?");
      }
    } else {
      processedBuffers.push({ name: file.name, buffer: await file.arrayBuffer() });
    }
  }

  if (processedBuffers.length === 0) throw new Error("No files found inside the Zip!");

  console.log(`Parsing ${processedBuffers.length} potential DICOM files...`);

  const studyStruct = {
    StudyInstanceUID: null,
    StudyDescription: 'CBCT Scan',
    StudyDate: new Date().toISOString().replace(/-|:|T|Z|\./g, '').substring(0, 8),
    StudyTime: '120000',
    PatientName: 'Anonymous',
    PatientID: 'Unknown',
    series: {} 
  };

  // Helper to safely read numbers
  const getNumber = (val, def) => (val !== undefined && val !== null ? Number(val) : def);

  for (let i = 0; i < processedBuffers.length; i++) {
    const { name, buffer } = processedBuffers[i];
    try {
      const dicomData = dcmjs.data.DicomMessage.readFile(buffer);
      const dataset = dcmjs.data.DicomMetaDictionary.naturalizeDataset(dicomData.dict);

      // UIDs
      const sopInstanceUid = dataset.SOPInstanceUID || dcmjs.data.DicomMetaDictionary.uid();
      const seriesInstanceUid = dataset.SeriesInstanceUID || dcmjs.data.DicomMetaDictionary.uid();
      const studyInstanceUid = dataset.StudyInstanceUID || dcmjs.data.DicomMetaDictionary.uid();
      
      // Study & Patient Info
      if (!studyStruct.StudyInstanceUID) {
        studyStruct.StudyInstanceUID = studyInstanceUid;
        studyStruct.PatientID = dataset.PatientID || `PID-${Math.floor(Math.random() * 10000)}`;
        studyStruct.PatientName = dataset.PatientName ? (dataset.PatientName.Alphabetic || dataset.PatientName) : 'Anonymous';
      }

      // Series Info
      if (!studyStruct.series[seriesInstanceUid]) {
        studyStruct.series[seriesInstanceUid] = {
          SeriesInstanceUID: seriesInstanceUid,
          SeriesDescription: dataset.SeriesDescription || 'CBCT Series',
          SeriesNumber: getNumber(dataset.SeriesNumber, 1),
          Modality: dataset.Modality || 'CT',
          instances: []
        };
      }

      // --- CRITICAL FIX: Z-Position Logic ---
      // We check if ImagePositionPatient is missing or empty.
      // If missing, we manually stack them: [0, 0, index * spacing]
      const spacing = dataset.PixelSpacing || [0.5, 0.5];
      let imagePos = dataset.ImagePositionPatient;
      
      // Use 'i' (loop index) as a fallback InstanceNumber if tag is missing
      const instanceNum = getNumber(dataset.InstanceNumber, i + 1);

      // If Position is missing (common cause of distortion), calculate it
      if (!imagePos || imagePos.length !== 3) {
         // Assume 0.5mm slice thickness for stacking
         imagePos = [0, 0, instanceNum * 0.5];
      }

      studyStruct.series[seriesInstanceUid].instances.push({
        metadata: {
          SOPInstanceUID: sopInstanceUid,
          SeriesInstanceUID: seriesInstanceUid,
          StudyInstanceUID: studyInstanceUid,
          InstanceNumber: instanceNum,
          
          // Image Specs
          Rows: getNumber(dataset.Rows, 512),
          Columns: getNumber(dataset.Columns, 512),
          BitsAllocated: getNumber(dataset.BitsAllocated, 16),
          BitsStored: getNumber(dataset.BitsStored, 12),
          PixelRepresentation: getNumber(dataset.PixelRepresentation, 0),
          SamplesPerPixel: getNumber(dataset.SamplesPerPixel, 1),
          PhotometricInterpretation: dataset.PhotometricInterpretation || 'MONOCHROME2',
          SOPClassUID: dataset.SOPClassUID || '1.2.840.10008.5.1.4.1.1.2',
          
          // Geometry
          PixelSpacing: spacing,
          ImageOrientationPatient: dataset.ImageOrientationPatient || [1, 0, 0, 0, 1, 0],
          ImagePositionPatient: imagePos, // <--- Using our robust position
          FrameOfReferenceUID: dataset.FrameOfReferenceUID || studyInstanceUid,
          
          // Contrast / Hounsfield Units (Fixes gray/noise issues)
          RescaleIntercept: getNumber(dataset.RescaleIntercept, -1024),
          RescaleSlope: getNumber(dataset.RescaleSlope, 1),
          WindowCenter: getNumber(dataset.WindowCenter, 400),
          WindowWidth: getNumber(dataset.WindowWidth, 2000),
          
          // Syntax (Helps decoder)
          TransferSyntaxUID: dicomData.meta.TransferSyntaxUID || "1.2.840.10008.1.2.1"
        },
        url: name, 
        fileBlob: new Blob([buffer]) 
      });

      // Add to upload list
      parsedFiles.push({
        file: new File([buffer], name),
        path: `${studyStruct.PatientID}/${name}`, 
        patientId: studyStruct.PatientID
      });

    } catch (err) {
      // Skip bad files
    }
  }

  // Final Sorting (Critical for 3D)
  const finalSeries = Object.values(studyStruct.series).map(s => {
    s.instances.sort((a, b) => a.metadata.InstanceNumber - b.metadata.InstanceNumber);
    return s;
  });

  return { 
    manifest: { studies: [{ ...studyStruct, series: finalSeries }] }, 
    parsedFiles 
  };
}