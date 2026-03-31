import { createManifestSession } from './manifestSession';

const manifestSession = createManifestSession();

self.onmessage = async event => {
  const { requestId, type, payload } = event.data || {};

  try {
    let responsePayload;

    switch (type) {
      case 'bootstrap': {
        responsePayload = await manifestSession.request('bootstrap', payload);
        break;
      }
      case 'seriesMetadata': {
        responsePayload = await manifestSession.request('seriesMetadata', payload);
        break;
      }
      default:
        throw new Error(`Unsupported worker request type: ${type}`);
    }

    self.postMessage({
      requestId,
      payload: responsePayload,
    });
  } catch (error) {
    self.postMessage({
      requestId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
