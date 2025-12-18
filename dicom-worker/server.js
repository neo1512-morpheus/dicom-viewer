const express = require('express');
const { createClient } = require('@supabase/supabase-js');
const AdmZip = require('adm-zip');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const util = require('util');
const dcmjs = require('dcmjs');
require('dotenv').config();

const execPromise = util.promisify(exec);
const app = express();
app.use(express.json());

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);

app.post('/webhook', async (req, res) => {
    // 1. Input Normalization: Support both manual test and Supabase webhook
    const filename = req.body.filename || (req.body.record && req.body.record.name);
    if (!filename) {
        return res.status(400).json({ error: 'No filename provided. Send either "filename" or "record.name".' });
    }
    if (!filename.endsWith('.zip')) return res.status(200).send('Ignored');
    console.log(`🚀 Processing ZIP: ${filename}`);

    // Track if we used local file (for cleanup later)
    let usedLocalFile = false;
    const localFilePath = path.join('/mnt/inbox', filename);

    try {
        let fileData;

        // 2. Hybrid Input Source: Check local volume mount first
        if (fs.existsSync(localFilePath)) {
            console.log('📂 Found file in volume mount. Processing locally...');
            fileData = fs.readFileSync(localFilePath);
            usedLocalFile = true;
        } else {
            // Fallback: Download from Supabase
            console.log('☁️ File not local. Downloading from Supabase...');
            const { data: supabaseData, error: dlError } =
                await supabase.storage.from('scans').download(filename);
            if (dlError) throw dlError;
            fileData = Buffer.from(await supabaseData.arrayBuffer());
        }

        // Construct the public base URL for OHIF compatibility
        const projectRef = process.env.SUPABASE_URL.split('https://')[1].split('.')[0];
        const publicBaseUrl = `https://${projectRef}.supabase.co/storage/v1/object/public/scans/compressed/${filename.replace('.zip', '')}`;

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
                // Use J2K (16-bit preservation) with multi-layer lossy compression to achieve ~10:1 ratio
                // This reduces a 120MB scan to ~12MB, matching the competitor's network footprint.
                const output = path.join(extractPath, `compressed_${file}`);

                try {
                    // 1. Run Compression (J2K High Quality)
                    await execPromise(`gdcmconv --j2k --lossy -q 90 ${input} ${output}`);

                    // 2. verify
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

                // 5. Upload (Original or Compressed)
                const uploadPath = filename.replace('.zip', '') + `/${file}`;
                const uploadBuffer = fs.readFileSync(output);
                await supabase.storage.from('scans').upload(
                    `compressed/${uploadPath}`,
                    uploadBuffer,
                    {
                        contentType: 'application/dicom',
                        upsert: true
                    }
                );

                // 6. Add to Manifest with FORCE-INJECTED TransferSyntaxUID
                studyStruct.series[seriesUID].instances.push({
                    metadata: {
                        ...dataset, // Copies all existing tags
                        // FORCE INJECT: Ensures TransferSyntaxUID is at top level for viewer
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

        // Upload Manifest
        const manifestPath = filename.replace('.zip', '') + '/dicom_manifest.json';
        await supabase.storage.from('scans').upload(
            `compressed/${manifestPath}`,
            JSON.stringify(manifest, null, 2),
            {
                contentType: 'application/json',
                upsert: true
            }
        );

        console.log(`📜 Manifest Generated: ${manifestPath}`);

        // Cleanup
        fs.rmSync(extractPath, { recursive: true, force: true });
        fs.unlinkSync(zipPath);

        // 3. Cleanup: Delete raw file from volume mount if it was used
        if (usedLocalFile && fs.existsSync(localFilePath)) {
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
