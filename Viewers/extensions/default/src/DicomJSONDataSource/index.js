import { DicomMetadataStore, IWebApiDataSource } from '@ohif/core';
import OHIF from '@ohif/core';

import getImageId from '../DicomWebDataSource/utils/getImageId';
import getDirectURL from '../utils/getDirectURL';
import { createManifestSession, createSeriesKey } from './manifestSession';

const metadataProvider = OHIF.classes.MetadataProvider;

const mappings = {
  studyInstanceUid: 'StudyInstanceUID',
  patientId: 'PatientID',
};

let _store = {
  urls: [],
  studyInstanceUIDMap: new Map(),
};

function wrapSequences(obj) {
  return Object.keys(obj).reduce(
    (acc, key) => {
      if (typeof obj[key] === 'object' && obj[key] !== null) {
        acc[key] = wrapSequences(obj[key]);
      } else {
        acc[key] = obj[key];
      }
      if (key.endsWith('Sequence')) {
        acc[key] = OHIF.utils.addAccessors(acc[key]);
      }
      return acc;
    },
    Array.isArray(obj) ? [] : {}
  );
}

const getMetaDataByURL = url => {
  return _store.urls.find(metaData => metaData.url === url);
};

const findStudies = (key, value) => {
  const studies = [];
  _store.urls.forEach(metaData => {
    metaData.studies.forEach(aStudy => {
      if (aStudy[key] === value) {
        studies.push(aStudy);
      }
    });
  });
  return studies;
};

function createWorkerSession() {
  if (typeof Worker === 'undefined') {
    return createManifestSession();
  }

  let worker;
  try {
    worker = new Worker(new URL('./manifestWorker.js', import.meta.url), { type: 'module' });
  } catch (error) {
    console.warn('[DicomJSON] Falling back to main-thread manifest loading.', error);
    return createManifestSession();
  }

  const pendingRequests = new Map();
  let nextRequestId = 0;

  const rejectAll = error => {
    pendingRequests.forEach(({ reject }) => reject(error));
    pendingRequests.clear();
  };

  worker.onmessage = event => {
    const { requestId, payload, error } = event.data || {};
    const pendingRequest = pendingRequests.get(requestId);

    if (!pendingRequest) {
      return;
    }

    pendingRequests.delete(requestId);

    if (error) {
      pendingRequest.reject(new Error(error));
      return;
    }

    pendingRequest.resolve(payload);
  };

  worker.onerror = event => {
    rejectAll(new Error(`[DicomJSON Worker] ${event.message || 'Worker error'}`));
    worker.terminate();
  };

  return {
    request(type, payload) {
      const requestId = nextRequestId++;
      return new Promise((resolve, reject) => {
        pendingRequests.set(requestId, { resolve, reject });
        worker.postMessage({ requestId, type, payload });
      });
    },
    terminate() {
      rejectAll(new Error('[DicomJSON Worker] Worker session terminated.'));
      worker.terminate();
    },
  };
}

function createDeferredSeriesLoad(metadata, processFunction) {
  let internalPromise;
  let startResolve;
  let startReject;
  const startCompletionPromise = new Promise((resolve, reject) => {
    startResolve = resolve;
    startReject = reject;
  });
  const pendingThenCalls = [];
  const pendingCatchCalls = [];
  const pendingFinallyCalls = [];

  const flushPendingHandlers = () => {
    pendingThenCalls.forEach(([onFulfilled, onRejected]) => {
      internalPromise.then(onFulfilled, onRejected);
    });
    pendingCatchCalls.forEach(onRejected => {
      internalPromise.catch(onRejected);
    });
    pendingFinallyCalls.forEach(onFinally => {
      internalPromise.finally(onFinally);
    });
  };

  return {
    metadata,
    getCompletionPromise() {
      return startCompletionPromise;
    },
    start() {
      if (!internalPromise) {
        internalPromise = processFunction();
        internalPromise.then(startResolve, startReject);
        flushPendingHandlers();
      }

      return internalPromise;
    },
    then(onFulfilled, onRejected) {
      if (internalPromise) {
        return internalPromise.then(onFulfilled, onRejected);
      }

      pendingThenCalls.push([onFulfilled, onRejected]);
    },
    catch(onRejected) {
      if (internalPromise) {
        return internalPromise.catch(onRejected);
      }

      pendingCatchCalls.push(onRejected);
    },
    finally(onFinally) {
      if (internalPromise) {
        return internalPromise.finally(onFinally);
      }

      pendingFinallyCalls.push(onFinally);
    },
  };
}

function addImageIdMapping({ imageId, StudyInstanceUID, SeriesInstanceUID, naturalizedDicom }) {
  const TransferSyntaxUID =
    naturalizedDicom.TransferSyntaxUID || naturalizedDicom['00020010']?.Value?.[0];

  metadataProvider.addImageIdToUIDs(imageId, {
    StudyInstanceUID,
    SeriesInstanceUID,
    SOPInstanceUID: naturalizedDicom.SOPInstanceUID,
    TransferSyntaxUID,
  });
}

function processSeriesInstances({ study, seriesSummary, instances, madeInClient }) {
  const seenSOPInstanceUIDs = new Set();
  const naturalizedInstances = [];
  let duplicateCount = 0;

  instances.forEach(instance => {
    if (!instance?.url || !instance?.metadata) {
      return;
    }

    const sopUID = instance.metadata.SOPInstanceUID;
    if (sopUID && seenSOPInstanceUIDs.has(sopUID)) {
      duplicateCount += 1;
      return;
    }

    if (sopUID) {
      seenSOPInstanceUIDs.add(sopUID);
    }

    const modifiedMetadata = wrapSequences(instance.metadata);
    const imageId = instance.url;

    addImageIdMapping({
      imageId,
      StudyInstanceUID: study.StudyInstanceUID,
      SeriesInstanceUID: seriesSummary.SeriesInstanceUID,
      naturalizedDicom: modifiedMetadata,
    });

    const naturalizedInstance = {
      ...modifiedMetadata,
      url: imageId,
      imageId,
      ...seriesSummary,
      ...study,
    };

    delete naturalizedInstance.instances;
    delete naturalizedInstance.series;

    naturalizedInstances.push(naturalizedInstance);
  });

  if (duplicateCount > 0) {
    console.warn(
      `[DicomJSON] Skipped ${duplicateCount} duplicate instances in series ${seriesSummary.SeriesInstanceUID}`
    );
  }

  if (naturalizedInstances.length > 0) {
    DicomMetadataStore.addInstances(naturalizedInstances, madeInClient);
  }

  return naturalizedInstances;
}

function setStudyLoadedFlag(StudyInstanceUID, madeInClient = false) {
  const study = DicomMetadataStore.getStudy(StudyInstanceUID, madeInClient);
  if (study) {
    study.isLoaded = true;
  }
}

function createDicomJSONApi(dicomJsonConfig) {
  const implementation = {
    initialize: async ({ query, url }) => {
      if (!url) {
        url = query.get('url');
      }

      const cachedMetaData = getMetaDataByURL(url);
      if (cachedMetaData) {
        return cachedMetaData.studies.map(aStudy => aStudy.StudyInstanceUID);
      }

      const workerSession = createWorkerSession();

      console.time('[DIAG] Manifest Bootstrap');
      const bootstrapData = await workerSession.request('bootstrap', { url });
      console.timeEnd('[DIAG] Manifest Bootstrap');

      const metaData = {
        url,
        studies: bootstrapData.studies || [],
        workerSession,
        seriesLoadPromises: new Map(),
      };

      _store.urls.push(metaData);
      _store.studyInstanceUIDMap.set(
        url,
        bootstrapData.studyInstanceUIDs || metaData.studies.map(study => study.StudyInstanceUID)
      );

      return _store.studyInstanceUIDMap.get(url);
    },
    query: {
      studies: {
        mapParams: () => {},
        search: async param => {
          const [key, value] = Object.entries(param)[0];
          const mappedParam = mappings[key];
          const studies = findStudies(mappedParam, value);

          return studies.map(aStudy => {
            return {
              accession: aStudy.AccessionNumber,
              date: aStudy.StudyDate,
              description: aStudy.StudyDescription,
              instances: aStudy.NumInstances,
              modalities: aStudy.Modalities,
              mrn: aStudy.PatientID,
              patientName: aStudy.PatientName,
              studyInstanceUid: aStudy.StudyInstanceUID,
              NumInstances: aStudy.NumInstances,
              time: aStudy.StudyTime,
            };
          });
        },
        processResults: () => {
          console.warn(' DICOMJson QUERY processResults not implemented');
        },
      },
      series: {
        search: () => {
          console.warn(' DICOMJson QUERY SERIES SEARCH not implemented');
        },
      },
      instances: {
        search: () => {
          console.warn(' DICOMJson QUERY instances SEARCH not implemented');
        },
      },
    },
    retrieve: {
      directURL: params => {
        return getDirectURL(dicomJsonConfig, params);
      },
      series: {
        metadata: async ({
          StudyInstanceUID,
          madeInClient = false,
          customSort,
          returnPromises = false,
        } = {}) => {
          if (!StudyInstanceUID) {
            throw new Error('Unable to query for SeriesMetadata without StudyInstanceUID');
          }

          const study = findStudies('StudyInstanceUID', StudyInstanceUID)[0];
          if (!study) {
            throw new Error(`Unable to find study ${StudyInstanceUID} in the cached DICOM JSON data.`);
          }

          const metaData = _store.urls.find(entry => {
            return entry.studies.some(aStudy => aStudy.StudyInstanceUID === StudyInstanceUID);
          });

          if (!metaData) {
            throw new Error(`Unable to find the backing manifest for study ${StudyInstanceUID}.`);
          }

          const series = customSort ? customSort([...study.series]) : [...study.series];
          const seriesSummaryMetadata = series.map(seriesEntry => {
            return {
              StudyInstanceUID: study.StudyInstanceUID,
              ...seriesEntry,
            };
          });

          DicomMetadataStore.addSeriesMetadata(seriesSummaryMetadata, madeInClient);

          const loadSeriesMetadata = async seriesSummary => {
            const seriesKey = createSeriesKey(StudyInstanceUID, seriesSummary.SeriesInstanceUID);
            if (metaData.seriesLoadPromises.has(seriesKey)) {
              return metaData.seriesLoadPromises.get(seriesKey);
            }

            const seriesLoadPromise = metaData.workerSession
              .request('seriesMetadata', {
                url: metaData.url,
                StudyInstanceUID,
                SeriesInstanceUID: seriesSummary.SeriesInstanceUID,
              })
              .then(seriesPayload => {
                const payloadStudy = {
                  ...study,
                  ...(seriesPayload.study || {}),
                };
                const payloadSeries = {
                  ...seriesSummary,
                  ...(seriesPayload.series || {}),
                };

                return processSeriesInstances({
                  study: payloadStudy,
                  seriesSummary: payloadSeries,
                  instances: seriesPayload.instances || [],
                  madeInClient,
                });
              })
              .catch(error => {
                metaData.seriesLoadPromises.delete(seriesKey);
                throw error;
              });

            metaData.seriesLoadPromises.set(seriesKey, seriesLoadPromise);
            return seriesLoadPromise;
          };

          const deferredSeriesPromises = seriesSummaryMetadata.map(seriesSummary => {
            return createDeferredSeriesLoad(seriesSummary, async () => {
              return loadSeriesMetadata(seriesSummary);
            });
          });

          Promise.all(deferredSeriesPromises.map(promise => promise.getCompletionPromise()))
            .then(() => {
              setStudyLoadedFlag(StudyInstanceUID, madeInClient);
            })
            .catch(() => {});

          if (returnPromises) {
            return deferredSeriesPromises;
          }

          await Promise.all(deferredSeriesPromises.map(promise => promise.start()));
          setStudyLoadedFlag(StudyInstanceUID, madeInClient);

          return seriesSummaryMetadata;
        },
      },
    },
    store: {
      dicom: () => {
        console.warn(' DICOMJson store dicom not implemented');
      },
    },
    getImageIdsForDisplaySet(displaySet) {
      const images = displaySet.images;
      const imageIds = [];

      if (!images) {
        return imageIds;
      }

      displaySet.images.forEach(instance => {
        const NumberOfFrames = instance.NumberOfFrames;

        if (NumberOfFrames > 1) {
          for (let i = 0; i < NumberOfFrames; i++) {
            const imageId = getImageId({
              instance,
              frame: i,
              config: dicomJsonConfig,
            });
            imageIds.push(imageId);
          }
        } else {
          const imageId = getImageId({ instance, config: dicomJsonConfig });
          imageIds.push(imageId);
        }
      });

      return imageIds;
    },
    getImageIdsForInstance({ instance, frame }) {
      return getImageId({ instance, frame });
    },
    getStudyInstanceUIDs: ({ query }) => {
      const url = query.get('url');
      return _store.studyInstanceUIDMap.get(url);
    },
  };

  return IWebApiDataSource.create(implementation);
}

export { createDicomJSONApi };
