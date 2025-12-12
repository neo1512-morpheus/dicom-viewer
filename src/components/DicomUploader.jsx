import React, { useState } from 'react';
import { supabase } from '../lib/supabaseClient';
import { parseDicomFiles } from '../utils/dicomParser';

// YOUR DEPLOYED EDGE FUNCTION URL
const EDGE_FUNCTION_URL = "https://qwcezedtdjurtyelbeeg.supabase.co/functions/v1/serve-dicom-json";

export default function DicomUploader() {
  const [status, setStatus] = useState('idle'); // idle, parsing, uploading, complete, error
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [patientId, setPatientId] = useState(null); 

  const handleFolderSelect = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    try {
      setStatus('parsing');
      setMessage('Parsing DICOM files... This happens locally in your browser.');
      
      // 1. Parse Files locally
      const { manifest, parsedFiles } = await parseDicomFiles(files);
      
      if (parsedFiles.length === 0) {
        throw new Error('No valid DICOM files found.');
      }

      // Capture the ID locally (Fix for the bug)
      const pid = parsedFiles[0].patientId;
      setPatientId(pid); 

      setStatus('uploading');
      setMessage(`Uploading ${parsedFiles.length} files for Patient: ${pid}...`);

      // 2. Upload Files with Concurrency Limit (3)
      const CONCURRENCY_LIMIT = 3;
      let completedUploads = 0;

      const uploadFile = async (item) => {
        const { file, path } = item;
        const { error } = await supabase.storage
          .from('scans')
          .upload(path, file, {
            cacheControl: '3600',
            upsert: true,
          });

        if (error) throw error;
        
        completedUploads++;
        setProgress(Math.round((completedUploads / (parsedFiles.length + 1)) * 100));
      };

      const pool = [];
      for (const item of parsedFiles) {
        const p = uploadFile(item).then(() => {
          pool.splice(pool.indexOf(p), 1);
        });
        pool.push(p);
        if (pool.length >= CONCURRENCY_LIMIT) {
          await Promise.race(pool);
        }
      }
      await Promise.all(pool);

      // 3. Upload Manifest
      // FIX: Use local 'pid' variable, not the state 'patientId'
      if (!pid) throw new Error("Patient ID could not be generated from files.");

      const manifestPath = `${pid}/dicom_manifest.json`;
      const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
      
      const { error: manifestError } = await supabase.storage
        .from('scans')
        .upload(manifestPath, blob, {
          contentType: 'application/json',
          upsert: true
        });

      if (manifestError) throw manifestError;

      completedUploads++;
      setProgress(100);
      setStatus('complete');
      setMessage('Upload Complete! You can now view the scan.');

    } catch (error) {
      console.error('Upload failed:', error);
      setStatus('error');
      setMessage(`Upload failed: ${error.message}`);
    }
  };

  const handleOpenViewer = () => {
    if (!patientId) return;
    
    // Construct the inner URL (Your Edge Function + Patient ID)
    const sourceUrl = `${EDGE_FUNCTION_URL}?patientId=${patientId}`;
    
    // Construct the OHIF Viewer URL
    // Add '&hangingProtocolId=mpr' to the end
const viewerUrl = `https://viewer.ohif.org/viewer/dicomjson?url=${encodeURIComponent(sourceUrl)}&hangingProtocolId=mpr&prefetch=false`;
    
    window.open(viewerUrl, '_blank');
  };

  return (
    <div className="p-6 max-w-xl mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-xl font-bold text-gray-900">DICOM Upload</h2>
      <p className="text-gray-600 text-sm">
        Select a Zip file containing .dcm files. They will be parsed locally and securely uploaded.
      </p>

      <div className="flex items-center justify-center w-full">
        <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg className="w-8 h-8 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
            </svg>
            <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload Zip</span></p>
            <p className="text-xs text-gray-500">DICOM Zip (.zip)</p>
          </div>
          <input 
            id="dropzone-file" 
            type="file" 
            className="hidden" 
            accept=".zip"
            onChange={handleFolderSelect}
            disabled={status === 'parsing' || status === 'uploading'}
          />
        </label>
      </div>

      {status !== 'idle' && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm font-medium text-gray-900">
            <span>{status === 'parsing' ? 'Parsing...' : status === 'uploading' ? 'Uploading...' : status === 'complete' ? 'Done' : 'Error'}</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`bg-blue-600 h-2.5 rounded-full transition-all duration-300 ${status === 'error' ? 'bg-red-600' : ''}`} 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-500">{message}</p>
          
          {/* Real View Button */}
          {status === 'complete' && (
             <button
               onClick={handleOpenViewer}
               className="mt-4 px-4 py-2 bg-green-600 text-white font-bold rounded hover:bg-green-700 w-full shadow-lg transition-transform transform hover:scale-105"
             >
               Open 3D Viewer
             </button>
          )}
        </div>
      )}
    </div>
  );
}