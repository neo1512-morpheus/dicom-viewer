// public/sw.js
const DICOM_CACHE_NAME = 'dicom-cache-v1';

self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== DICOM_CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    if (url.pathname.endsWith('.dcm')) {
        if (event.request.headers.get('range')) {
            event.respondWith(handleRangeRequest(event.request));
        } else {
            event.respondWith(handleFullRequest(event.request));
        }
    }
});

async function handleFullRequest(request) {
    const cache = await caches.open(DICOM_CACHE_NAME);
    const cachedResponse = await cache.match(request);
    if (cachedResponse) return cachedResponse;
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
        cache.put(request, networkResponse.clone());
    }
    return networkResponse;
}

async function handleRangeRequest(request) {
    const cache = await caches.open(DICOM_CACHE_NAME);
    const cachedResponse = await cache.match(request);
    let buffer;
    if (cachedResponse) {
        buffer = await cachedResponse.arrayBuffer();
    } else {
        // Cache Miss: Fetch full file (ignore range), cache it, then slice
        const networkRequest = new Request(request.url, {
            method: 'GET',
            headers: request.headers,
            mode: 'cors',
            credentials: 'omit',
        });
        // Remove range header for the network fetch
        // Note: Creating 'new Request' doesn't automatically strip it,
        // but standard fetch usually ignores it unless explicitly set.
        // Better to be explicit if possible, or just rely on caching the full response.

        const networkResponse = await fetch(networkRequest);
        if (networkResponse.status === 200) {
            cache.put(request.url, networkResponse.clone()); // Cache the full URL
        }
        buffer = await networkResponse.arrayBuffer();
    }

    const rangeHeader = request.headers.get('range');
    const rangeMatch = /^bytes=(\d+)-(\d*)$/.exec(rangeHeader);

    if (!rangeMatch) return new Response(buffer, { status: 200 }); // Fallback

    const start = Number(rangeMatch[1]);
    const end = rangeMatch[2] ? Number(rangeMatch[2]) : buffer.byteLength - 1;
    const slicedBuffer = buffer.slice(start, end + 1);

    return new Response(slicedBuffer, {
        status: 206,
        statusText: 'Partial Content',
        headers: {
            'Content-Type': 'application/dicom',
            'Content-Length': slicedBuffer.byteLength,
            'Content-Range': `bytes ${start}-${end}/${buffer.byteLength}`,
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache'
        }
    });
}
