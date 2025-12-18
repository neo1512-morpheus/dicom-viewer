import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Register DICOM Service Worker for caching
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then(reg => console.log('✅ DICOM Service Worker Registered', reg.scope))
      .catch(err => console.log('❌ Service Worker Failed', err));
  });
}

console.log("🚨 TRACER: Main App JS is mounting. Current URL:", window.location.href);

createRoot(document.getElementById('root')).render(
  // <StrictMode>
  <App />
  // </StrictMode>
)
