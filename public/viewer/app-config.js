window.config = {
    routerBasename: "/",
    extensions: [],
    modes: [],
    customizationService: {},
    showStudyList: true,
    maxNumberOfWebWorkers: 3,
    showWarningMessageForCrossOrigin: true,
    showCPUFallbackMessage: true,
    showLoadingIndicator: true,
    useNorm16Texture: true,
    strictZSpacingForVolumeViewport: false,
    preferSizeOverAccuracy: true, // Enable memory-efficient 16-bit textures
    groupEnabledModesFirst: true,
    maxCacheSize: 1.5 * 1024 * 1024 * 1024, // 1.5GB - Safer limit for integrated GPUs with shared VRAM
    studyPrefetcher: {
        enabled: true,
        displaySetCount: 1,
        maxNumPrefetchRequests: 3, // Further reduced to prevent main thread/memory saturation
    },
    maxNumRequests: {
        interaction: 100,
        thumbnail: 50,
        prefetch: 3 // Tightened to give priority to interaction requests
    },
    defaultDataSourceName: "dicomweb",
    // DISABLE the investigational use disclaimer popup
    investigationalUseDialog: {
        option: "never"
    },
    // White Labeling - Custom Logo with left margin
    whiteLabeling: {
        createLogoComponentFn: function (React) {
            return React.createElement('img', {
                src: '/viewer/assets/oroscan-logo.png',
                alt: 'Oroscan',
                style: {
                    height: '32px',
                    width: 'auto',
                    marginLeft: '16px'
                }
            });
        }
    },
    dataSources: [
        {
            namespace: "@ohif/extension-default.dataSourcesModule.dicomweb",
            sourceName: "dicomweb",
            configuration: {
                friendlyName: "AWS S3 Static wado server",
                name: "aws",
                wadoUriRoot: "https://d33do7qe4w26qo.cloudfront.net/dicomweb",
                qidoRoot: "https://d33do7qe4w26qo.cloudfront.net/dicomweb",
                wadoRoot: "https://d33do7qe4w26qo.cloudfront.net/dicomweb",
                qidoSupportsIncludeField: false,
                imageRendering: "wadors",
                thumbnailRendering: "wadors",
                enableStudyLazyLoad: true,
                supportsFuzzyMatching: false,
                supportsWildcard: true,
                staticWado: true,
                singlepart: "bulkdata,video",
                bulkDataURI: {
                    enabled: true,
                    relativeResolution: "studies"
                },
                omitQuotationForMultipartRequest: true,
                retrieve: {
                    stages: [
                        {
                            id: 'initialImages',
                            positions: [0.5, 0, -1], // Middle, first, last slice
                            priority: 10,
                            requestType: 'INTERACTION',
                        },
                        {
                            id: 'quarterResolution',
                            decimate: 4, // Every 4th slice
                            offset: 3,
                            priority: 9,
                            retrieveType: 'multipleFast',
                        },
                        // ---------------------------------------------------------
                        // 🛑 DISABLED: fullResolution stage was causing GPU memory crash
                        // This was the "Silent Killer" - loading ALL 600 slices at once
                        // Now high-res slices load on-demand when scrolling stops
                        // ---------------------------------------------------------
                        // {
                        //     id: 'fullResolution',
                        //     decimate: 1, // Fill gaps
                        //     priority: 8,
                        //     retrieveType: 'default',
                        // },
                    ],
                }
            }
        },
        {
            namespace: "@ohif/extension-default.dataSourcesModule.dicomweb",
            sourceName: "dicomweb2",
            configuration: {
                friendlyName: "AWS S3 Static wado secondary server",
                name: "aws",
                wadoUriRoot: "https://d28o5kq0jsoob5.cloudfront.net/dicomweb",
                qidoRoot: "https://d28o5kq0jsoob5.cloudfront.net/dicomweb",
                wadoRoot: "https://d28o5kq0jsoob5.cloudfront.net/dicomweb",
                qidoSupportsIncludeField: false,
                supportsReject: false,
                imageRendering: "wadors",
                thumbnailRendering: "wadors",
                enableStudyLazyLoad: true,
                supportsFuzzyMatching: false,
                supportsWildcard: true,
                staticWado: true,
                singlepart: "bulkdata,video",
                bulkDataURI: {
                    enabled: true,
                    relativeResolution: "studies"
                },
                omitQuotationForMultipartRequest: true
            }
        },
        {
            namespace: "@ohif/extension-default.dataSourcesModule.dicomwebproxy",
            sourceName: "dicomwebproxy",
            configuration: {
                friendlyName: "dicomweb delegating proxy",
                name: "dicomwebproxy"
            }
        },
        {
            namespace: "@ohif/extension-default.dataSourcesModule.dicomjson",
            sourceName: "dicomjson",
            configuration: {
                friendlyName: "dicom json",
                name: "json"
            }
        },
        {
            namespace: "@ohif/extension-default.dataSourcesModule.dicomlocal",
            sourceName: "dicomlocal",
            configuration: {
                friendlyName: "dicom local"
            }
        }
    ],
    httpErrorHandler: function (error) {
        console.warn(error.status);
        console.warn("test, navigate to https://ohif.org/");
    },
    hotkeys: [
        { commandName: "incrementActiveViewport", label: "Next Viewport", keys: ["right"] },
        { commandName: "decrementActiveViewport", label: "Previous Viewport", keys: ["left"] },
        { commandName: "rotateViewportCW", label: "Rotate Right", keys: ["r"] },
        { commandName: "rotateViewportCCW", label: "Rotate Left", keys: ["l"] },
        { commandName: "invertViewport", label: "Invert", keys: ["i"] },
        { commandName: "flipViewportHorizontal", label: "Flip Horizontally", keys: ["h"] },
        { commandName: "flipViewportVertical", label: "Flip Vertically", keys: ["v"] },
        { commandName: "scaleUpViewport", label: "Zoom In", keys: ["+"] },
        { commandName: "scaleDownViewport", label: "Zoom Out", keys: ["-"] },
        { commandName: "fitViewportToWindow", label: "Zoom to Fit", keys: ["="] },
        { commandName: "resetViewport", label: "Reset", keys: ["space"] },
        { commandName: "nextImage", label: "Next Image", keys: ["down"] },
        { commandName: "previousImage", label: "Previous Image", keys: ["up"] },
        { commandName: "setToolActive", commandOptions: { toolName: "Zoom" }, label: "Zoom", keys: ["z"] },
        { commandName: "windowLevelPreset1", label: "W/L Preset 1", keys: ["1"] },
        { commandName: "windowLevelPreset2", label: "W/L Preset 2", keys: ["2"] },
        { commandName: "windowLevelPreset3", label: "W/L Preset 3", keys: ["3"] },
        { commandName: "windowLevelPreset4", label: "W/L Preset 4", keys: ["4"] },
        { commandName: "windowLevelPreset5", label: "W/L Preset 5", keys: ["5"] },
        { commandName: "windowLevelPreset6", label: "W/L Preset 6", keys: ["6"] },
        { commandName: "windowLevelPreset7", label: "W/L Preset 7", keys: ["7"] },
        { commandName: "windowLevelPreset8", label: "W/L Preset 8", keys: ["8"] },
        { commandName: "windowLevelPreset9", label: "W/L Preset 9", keys: ["9"] }
    ]
};