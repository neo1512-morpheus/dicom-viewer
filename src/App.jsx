import React from 'react';
import DicomUploader from './components/DicomUploader';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800">DICOM Viewer Prototype</h1>
        <p className="text-gray-600 mt-2 text-center">Secure Client-Side Parsing & Upload</p>
      </header>
      
      <main className="w-full max-w-4xl px-4">
        <DicomUploader />
      </main>
    </div>
  );
}

export default App;