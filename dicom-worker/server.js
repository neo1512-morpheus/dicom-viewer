require('dotenv').config(); // Loads .env if present (local dev), Docker injects env vars directly
const express = require('express');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const AdmZip = require('adm-zip');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');
const dcmjs = require('dcmjs');

const execPromise = util.promisify(exec);
const app = express();
app.use(express.json());

// Validate required R2 environment variables
if (!process.env.R2_ENDPOINT) {
    throw new Error('R2_ENDPOINT is missing. Set it in your .env or pass via --env-file.');
}
if (!process.env.R2_ACCESS_KEY_ID || !process.env.R2_SECRET_ACCESS_KEY) {
    throw new Error('R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY is missing.');
}
if (!process.env.R2_BUCKET_NAME) {
    throw new Error('R2_BUCKET_NAME is missing.');
}

// R2/S3 Client for uploads
const s3Client = new S3Client({
    region: 'auto',
    endpoint: process.env.R2_ENDPOINT,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    },
    forcePathStyle: true
});

app.post('/webhook', async (req, res) => {
    // 1. Input Normalization: Support both manual test and webhook
    const filename = req.body.filename || (req.body.record && req.body.record.name);
    if (!filename) {
        return res.status(400).json({ error: 'No filename provided. Send either "filename" or "record.name".' });
    }
    if (!filename.endsWith('.zip')) return res.status(200).send('Ignored');
    console.log(`🚀 Processing ZIP: ${filename}`);

    const localFilePath = path.join('/mnt/inbox', filename);

    try {
        // Input: Read from local volume mount only
        if (!fs.existsSync(localFilePath)) {
            return res.status(404).json({ error: `File not found in /mnt/inbox: ${filename}` });
        }

        console.log('📂 Found file in volume mount. Processing locally...');
        const fileData = fs.readFileSync(localFilePath);

        // Construct the public base URL for OHIF compatibility (uses public r2.dev URL, not private S3 API)
        const studyId = filename.replace('.zip', '');
        const publicBaseUrl = `${process.env.R2_PUBLIC_URL}/compressed/${studyId}`;

        const zipPath = `/tmp/${path.basename(filename)}`;
        fs.writeFileSync(zipPath, fileData);

        const extractPath = `/tmp/extracted_${Date.now()}`;
        const zip = new AdmZip(zipPath);
        zip.extractAllTo(extractPath, true);

        const files = fs.readdirSync(extractPath);

        // Manifest Initialization
        const studyStruct = {
            StudyInstanceUID: null,
            StudyDescription: 'CBCT Scan',
            StudyDate: new Date().toISOString().replace(/-|:|T|Z|\./g, '').substring(0, 8),
            StudyTime: '120000',
            PatientName: 'Anonymous',
            PatientID: 'Unknown',
            series: {}
        };

        for (const file of files) {
            const input = path.join(extractPath, file);

            if (file.endsWith('.dcm')) {
                // 1. Read & Parse Metadata
                const fileBuffer = fs.readFileSync(input);
                let dataset = null;

                try {
                    const dicomData = dcmjs.data.DicomMessage.readFile(fileBuffer.buffer);
                    dataset = dcmjs.data.DicomMetaDictionary.naturalizeDataset(dicomData.dict);
                } catch (e) {
                    console.error(`Error parsing DICOM ${file}:`, e);
                    continue;
                }

                // 2. Populate Study Info (first file wins)
                if (!studyStruct.StudyInstanceUID) {
                    studyStruct.StudyInstanceUID = dataset.StudyInstanceUID;
                    studyStruct.StudyDescription = dataset.StudyDescription || 'CBCT Scan';
                    studyStruct.StudyDate = dataset.StudyDate;
                    studyStruct.StudyTime = dataset.StudyTime;
                    studyStruct.PatientID = dataset.PatientID || 'Unknown';
                    studyStruct.PatientName = dataset.PatientName
                        ? (dataset.PatientName.Alphabetic || dataset.PatientName)
                        : 'Anonymous';
                }

                // 3. Populate Series & Instance
                const seriesUID = dataset.SeriesInstanceUID || 'unknown-series';
                if (!studyStruct.series[seriesUID]) {
                    studyStruct.series[seriesUID] = {
                        SeriesInstanceUID: seriesUID,
                        SeriesDescription: dataset.SeriesDescription || 'Series',
                        SeriesNumber: dataset.SeriesNumber || 0,
                        Modality: dataset.Modality || 'CT',
                        instances: []
                    };
                }

                // 4. Compress Strategy (with Fallback)
                const output = path.join(extractPath, `compressed_${file}`);

                try {
                    // Run Compression (J2K High Quality)
                    await execPromise(`gdcmconv --j2k --lossy -q 90 ${input} ${output}`);

                    // Verify
                    const outStats = fs.statSync(output);

                    // Poison Pill: <3KB (3072 bytes)
                    if (outStats.size < 3072) {
                        throw new Error(`Output too small (<3KB: ${outStats.size}B)`);
                    }

                    const inputStats = fs.statSync(input);
                    const ratio = ((1 - (outStats.size / inputStats.size)) * 100).toFixed(2);
                    console.log(`✅ Compressed: ${(inputStats.size / 1024).toFixed(2)}KB -> ${(outStats.size / 1024).toFixed(2)}KB (Ratio: ${ratio}%)`);

                } catch (err) {
                    console.warn(`⚠️ Compression issue with ${file}: ${err.message}. Uploading original.`);
                    // Fallback: Copy input to output location so the rest of the script works
                    fs.copyFileSync(input, output);
                }

                // 5. Upload to R2
                const uploadPath = filename.replace('.zip', '') + `/${file}`;
                const uploadBuffer = fs.readFileSync(output);

                try {
                    await s3Client.send(new PutObjectCommand({
                        Bucket: process.env.R2_BUCKET_NAME,
                        Key: `compressed/${uploadPath}`,
                        Body: uploadBuffer,
                        ContentType: 'application/dicom'
                    }));
                } catch (uploadError) {
                    console.error(`❌ R2 upload failed for ${uploadPath}:`, uploadError.message);
                    throw uploadError;
                }

                // 6. Add to Manifest with FORCE-INJECTED TransferSyntaxUID
                studyStruct.series[seriesUID].instances.push({
                    metadata: {
                        ...dataset,
                        TransferSyntaxUID: '1.2.840.10008.1.2.4.91'
                    },
                    url: `dicomweb:${publicBaseUrl}/${file}`
                });

                console.log(`✅ Processed: ${file}`);
            }
        }

        // Finalize Manifest
        const finalSeries = Object.values(studyStruct.series);
        const manifest = {
            studies: [
                {
                    ...studyStruct,
                    series: finalSeries
                }
            ]
        };

        // Upload Manifest to R2
        const manifestPath = filename.replace('.zip', '') + '/dicom_manifest.json';
        try {
            await s3Client.send(new PutObjectCommand({
                Bucket: process.env.R2_BUCKET_NAME,
                Key: `compressed/${manifestPath}`,
                Body: JSON.stringify(manifest, null, 2),
                ContentType: 'application/json'
            }));
        } catch (uploadError) {
            console.error(`❌ R2 manifest upload failed for ${manifestPath}:`, uploadError.message);
            throw uploadError;
        }

        console.log(`📜 Manifest Generated: ${manifestPath}`);

        // Cleanup
        fs.rmSync(extractPath, { recursive: true, force: true });
        fs.unlinkSync(zipPath);

        // Delete raw file from volume mount after successful processing
        if (fs.existsSync(localFilePath)) {
            fs.unlinkSync(localFilePath);
            console.log(`🗑️ Deleted local file: ${localFilePath}`);
        }

        res.status(200).json({ status: 'success' });
    } catch (err) {
        console.error('❌ Error:', err);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Worker listening on port ${PORT}`));
