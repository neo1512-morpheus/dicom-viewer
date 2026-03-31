const manifestCache = new Map();

export function createSeriesKey(studyInstanceUID, seriesInstanceUID) {
  return `${studyInstanceUID}::${seriesInstanceUID}`;
}

function toAbsoluteUrl(baseUrl, maybeRelativeUrl) {
  if (!maybeRelativeUrl) {
    return maybeRelativeUrl;
  }

  try {
    return new URL(maybeRelativeUrl, baseUrl).toString();
  } catch (error) {
    return maybeRelativeUrl;
  }
}

function uniqueValues(values) {
  return values.filter((value, index) => value && values.indexOf(value) === index);
}

function buildStudySummary(study, manifestUrl) {
  const sourceSeries = Array.isArray(study.series) ? study.series : [];
  const series = sourceSeries.map((seriesEntry, seriesIndex) => {
    const metadataUrl = seriesEntry.metadataUrl
      ? toAbsoluteUrl(manifestUrl, seriesEntry.metadataUrl)
      : null;

    return {
      SeriesInstanceUID: seriesEntry.SeriesInstanceUID || `series-${seriesIndex + 1}`,
      SeriesDescription: seriesEntry.SeriesDescription || 'Series',
      SeriesNumber: seriesEntry.SeriesNumber ?? seriesIndex + 1,
      Modality: seriesEntry.Modality || 'OT',
      NumInstances:
        seriesEntry.NumInstances ??
        (Array.isArray(seriesEntry.instances) ? seriesEntry.instances.length : 0),
      metadataUrl,
    };
  });

  const numInstances =
    study.NumInstances ??
    series.reduce((total, seriesEntry) => total + (seriesEntry.NumInstances || 0), 0);
  const modalities = study.Modalities || uniqueValues(series.map(seriesEntry => seriesEntry.Modality));

  return {
    StudyInstanceUID: study.StudyInstanceUID,
    StudyDescription: study.StudyDescription,
    StudyDate: study.StudyDate,
    StudyTime: study.StudyTime,
    PatientName: study.PatientName,
    PatientID: study.PatientID,
    AccessionNumber: study.AccessionNumber,
    NumInstances: numInstances,
    Modalities: modalities,
    series,
  };
}

function buildCacheEntry(manifestUrl, manifest) {
  const sourceStudies = Array.isArray(manifest?.studies) ? manifest.studies : [];
  const studies = [];
  const seriesSources = new Map();

  sourceStudies.forEach((study, studyIndex) => {
    const studySummary = buildStudySummary(study, manifestUrl);
    const sourceSeries = Array.isArray(study.series) ? study.series : [];

    studySummary.series.forEach((seriesSummary, seriesIndex) => {
      const sourceSeriesEntry = sourceSeries[seriesIndex] || {};
      const key = createSeriesKey(studySummary.StudyInstanceUID, seriesSummary.SeriesInstanceUID);

      if (seriesSummary.metadataUrl) {
        seriesSources.set(key, {
          kind: 'remote',
          metadataUrl: seriesSummary.metadataUrl,
          studySummary,
          seriesSummary,
        });
        return;
      }

      seriesSources.set(key, {
        kind: 'embedded',
        instances: Array.isArray(sourceSeriesEntry.instances) ? sourceSeriesEntry.instances : [],
        studySummary,
        seriesSummary,
      });
    });

    studies.push({
      ...studySummary,
      StudyInstanceUID: studySummary.StudyInstanceUID || `study-${studyIndex + 1}`,
    });
  });

  return {
    manifestFormatVersion: manifest?.manifestFormatVersion || 1,
    studies,
    studyInstanceUIDs: studies.map(study => study.StudyInstanceUID),
    seriesSources,
  };
}

async function getManifestEntry(url) {
  if (manifestCache.has(url)) {
    return manifestCache.get(url);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest: ${response.status} ${response.statusText}`);
  }

  const manifest = await response.json();
  const entry = buildCacheEntry(url, manifest);
  manifestCache.set(url, entry);
  return entry;
}

async function loadSeriesPayload(url, StudyInstanceUID, SeriesInstanceUID) {
  const entry = await getManifestEntry(url);
  const key = createSeriesKey(StudyInstanceUID, SeriesInstanceUID);
  const source = entry.seriesSources.get(key);

  if (!source) {
    throw new Error(`Series ${SeriesInstanceUID} was not found in manifest.`);
  }

  if (source.kind === 'embedded') {
    return {
      study: source.studySummary,
      series: source.seriesSummary,
      instances: source.instances,
    };
  }

  const response = await fetch(source.metadataUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch series metadata: ${response.status} ${response.statusText}`);
  }

  const seriesPayload = await response.json();

  return {
    study: {
      ...source.studySummary,
      ...(seriesPayload.study || {}),
    },
    series: {
      ...source.seriesSummary,
      ...(seriesPayload.series || {}),
    },
    instances: Array.isArray(seriesPayload.instances) ? seriesPayload.instances : [],
  };
}

export function createManifestSession() {
  return {
    async request(type, payload) {
      switch (type) {
        case 'bootstrap': {
          const entry = await getManifestEntry(payload.url);
          return {
            manifestFormatVersion: entry.manifestFormatVersion,
            studies: entry.studies,
            studyInstanceUIDs: entry.studyInstanceUIDs,
          };
        }
        case 'seriesMetadata':
          return loadSeriesPayload(payload.url, payload.StudyInstanceUID, payload.SeriesInstanceUID);
        default:
          throw new Error(`Unsupported manifest request type: ${type}`);
      }
    },
    terminate() {},
  };
}
