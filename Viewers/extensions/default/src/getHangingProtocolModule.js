import hpMNGrid from './hpMNGrid';
import hpMNCompare from './hpCompare';

const defaultProtocol = {
  id: 'default',
  locked: true,
  imageLoadStrategy: 'default',
  // Don't store this hanging protocol as it applies to the currently active
  // display set by default
  // cacheId: null,
  name: 'Default',
  createdDate: '2021-02-23T19:22:08.894Z',
  modifiedDate: '2023-04-01',
  availableTo: {},
  editableBy: {},
  protocolMatchingRules: [],
  toolGroupIds: ['default'],
  hpInitiationCriteria: { minSeriesLoaded: 1 },
  // -1 would be used to indicate active only, whereas other values are
  // the number of required priors referenced - so 0 means active with
  // 0 or more priors.
  numberOfPriorsReferenced: 0,
  // Default viewport is used to define the viewport when
  // additional viewports are added using the layout tool
  defaultViewport: {
    viewportOptions: {
      viewportType: 'stack',
      toolGroupId: 'default',
      allowUnmatchedView: true,
    },
    displaySets: [
      {
        id: 'defaultDisplaySetId',
        matchedDisplaySetsIndex: -1,
        options: {
          // FIX: Force Exact Math for new viewports
          voiLUTFunction: 'LINEAR_EXACT',
        },
      },
    ],
  },
  displaySetSelectors: {
    defaultDisplaySetId: {
      // Matches displaysets, NOT series
      seriesMatchingRules: [
        // Try to match series with images by default, to prevent weird display
        // on SEG/SR containing studies
        {
          attribute: 'numImageFrames',
          constraint: {
            greaterThan: { value: 0 },
          },
        },
        // This display set will select the specified items by preference
        // It has no affect if nothing is specified in the URL.
        {
          attribute: 'isDisplaySetFromUrl',
          weight: 10,
          constraint: {
            equals: true,
          },
        },
      ],
      // Can be used to select matching studies
      // studyMatchingRules: [],
    },
  },
  stages: [
    {
      name: 'default',
      viewportStructure: {
        layoutType: 'grid',
        properties: {
          rows: 1,
          columns: 1,
        },
      },
      viewports: [
        {
          viewportOptions: {
            viewportType: 'stack',
            viewportId: 'default',
            toolGroupId: 'default',
            // This will specify the initial image options index if it matches in the URL
            // and will otherwise not specify anything.
            initialImageOptions: {
              custom: 'sopInstanceLocation',
            },
            // Other options for initialImageOptions, which can be included in the default
            // custom attribute, or can be provided directly.
            //   index: 180,
            //   preset: 'middle', // 'first', 'last', 'middle'
            // },
          },
          displaySets: [
            {
              id: 'defaultDisplaySetId',
              options: {
                // FIX: Force Exact Math for 2D to prevent binary thresholding on 16-bit data
                voiLUTFunction: 'LINEAR_EXACT',
              },
            },
          ],
        },
      ],
      createdDate: '2021-02-23T18:32:42.850Z',
    },
  ],
};

// Axial 1x1 Volume Protocol
const hpAxial = {
  id: 'hpAxial',
  name: 'Axial',
  icon: 'layout-advanced-axial-primary',
  isPreset: true,
  locked: true,
  imageLoadStrategy: 'nth',
  protocolMatchingRules: [],
  numberOfPriorsReferenced: 0,
  displaySetSelectors: {
    activeDisplaySet: {
      seriesMatchingRules: [
        {
          weight: 1,
          attribute: 'isReconstructable',
          constraint: {
            equals: {
              value: true,
            },
          },
          required: true,
        },
      ],
    },
  },
  stages: [
    {
      name: 'axial',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 1 } },
      viewports: [
        {
          viewportOptions: {
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'axial',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
  ],
};

// Coronal 1x1 Volume Protocol
const hpCoronal = {
  id: 'hpCoronal',
  name: 'Coronal',
  icon: 'layout-common-1x1',
  isPreset: true,
  locked: true,
  imageLoadStrategy: 'nth',
  protocolMatchingRules: [],
  numberOfPriorsReferenced: 0,
  displaySetSelectors: {
    activeDisplaySet: {
      seriesMatchingRules: [
        {
          weight: 1,
          attribute: 'isReconstructable',
          constraint: {
            equals: {
              value: true,
            },
          },
          required: true,
        },
      ],
    },
  },
  stages: [
    {
      name: 'coronal',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 1 } },
      viewports: [
        {
          viewportOptions: {
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'coronal',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
  ],
};

// Sagittal 1x1 Volume Protocol
const hpSagittal = {
  id: 'hpSagittal',
  name: 'Sagittal',
  icon: 'layout-common-1x1',
  isPreset: true,
  locked: true,
  imageLoadStrategy: 'nth',
  protocolMatchingRules: [],
  numberOfPriorsReferenced: 0,
  displaySetSelectors: {
    activeDisplaySet: {
      seriesMatchingRules: [
        {
          weight: 1,
          attribute: 'isReconstructable',
          constraint: {
            equals: {
              value: true,
            },
          },
          required: true,
        },
      ],
    },
  },
  stages: [
    {
      name: 'sagittal',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 1 } },
      viewports: [
        {
          viewportOptions: {
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'sagittal',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
  ],
};

// 3D Only 1x1 Volume3D Protocol
const hp3D = {
  id: 'hp3D',
  name: '3D Only',
  icon: 'layout-advanced-3d-only',
  isPreset: true,
  locked: true,
  imageLoadStrategy: 'interleaveCenter',
  protocolMatchingRules: [],
  numberOfPriorsReferenced: 0,
  displaySetSelectors: {
    activeDisplaySet: {
      seriesMatchingRules: [
        {
          weight: 1,
          attribute: 'isReconstructable',
          constraint: {
            equals: {
              value: true,
            },
          },
          required: true,
        },
      ],
    },
  },
  stages: [
    {
      name: '3d',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 1 } },
      viewports: [
        {
          viewportOptions: {
            toolGroupId: 'volume3d',
            viewportType: 'volume3d',
            orientation: 'coronal',
            customViewportProps: {
              hideOverlays: true,
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
              options: {
                displayPreset: {
                  CT: 'CT-Bone',
                  MR: 'MR-Default',
                  default: 'CT-Bone',
                },
              },
            },
          ],
        },
      ],
    },
  ],
};

// CPR protocol with 2 stages:
// stage 0 = setup (draw spline on axial)
// stage 1 = CPR view (axial + pano + cross-section)
const hpCPR = {
  id: 'cpr',
  name: 'CPR',
  icon: 'layout-advanced-mpr',
  isPreset: true,
  locked: true,
  imageLoadStrategy: 'nth',
  protocolMatchingRules: [],
  numberOfPriorsReferenced: 0,
  displaySetSelectors: {
    activeDisplaySet: {
      seriesMatchingRules: [
        {
          weight: 1,
          attribute: 'isReconstructable',
          constraint: {
            equals: {
              value: true,
            },
          },
          required: true,
        },
      ],
    },
  },
  stages: [
    {
      name: 'cpr-setup',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 1 } },
      viewports: [
        {
          viewportOptions: {
            viewportId: 'cpr-axial',
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'axial',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
    {
      name: 'cpr-view',
      viewportStructure: { layoutType: 'grid', properties: { rows: 1, columns: 3 } },
      viewports: [
        {
          viewportOptions: {
            viewportId: 'cpr-axial',
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'axial',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
        {
          viewportOptions: {
            viewportId: 'cpr-pano',
            toolGroupId: 'cprPano',
            viewportType: 'stack',
            allowUnmatchedView: true,
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
              options: {
              },
            },
          ],
        },
        {
          viewportOptions: {
            viewportId: 'cpr-crosssection',
            toolGroupId: 'cprCrossSection',
            viewportType: 'volume',
            orientation: 'coronal',
            initialImageOptions: {
              preset: 'middle',
            },
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
            },
          ],
        },
      ],
    },
  ],
};

function getHangingProtocolModule() {
  return [
    {
      name: defaultProtocol.id,
      protocol: defaultProtocol,
    },
    // Create a MxN hanging protocol available by default
    {
      name: hpMNGrid.id,
      protocol: hpMNGrid,
    },
    // Create a MxN comparison hanging protocol available by default
    {
      name: hpMNCompare.id,
      protocol: hpMNCompare,
    },
    // Custom 1x1 volume protocols
    {
      name: hpAxial.id,
      protocol: hpAxial,
    },
    {
      name: hpCoronal.id,
      protocol: hpCoronal,
    },
    {
      name: hpSagittal.id,
      protocol: hpSagittal,
    },
    {
      name: hp3D.id,
      protocol: hp3D,
    },
    {
      name: hpCPR.id,
      protocol: hpCPR,
    },
  ];
}

export default getHangingProtocolModule;
