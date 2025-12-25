import { Types } from '@ohif/core';

export const mpr: Types.HangingProtocol.Protocol = {
  id: 'mpr',
  name: 'MPR',
  locked: true,
  icon: 'layout-advanced-mpr',
  isPreset: true,
  createdDate: '2021-02-23',
  modifiedDate: '2023-08-15',
  availableTo: {},
  editableBy: {},
  // Unknown number of priors referenced - so just match any study
  numberOfPriorsReferenced: 0,
  protocolMatchingRules: [],
  imageLoadStrategy: 'nth',
  callbacks: {},
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
      name: 'MPR 1x3',
      viewportStructure: {
        layoutType: 'grid',
        properties: {
          rows: 1,
          columns: 3,
          layoutOptions: [
            {
              x: 0,
              y: 0,
              width: 1 / 3,
              height: 1,
            },
            {
              x: 1 / 3,
              y: 0,
              width: 1 / 3,
              height: 1,
            },
            {
              x: 2 / 3,
              y: 0,
              width: 1 / 3,
              height: 1,
            },
          ],
        },
      },
      viewports: [
        {
          viewportOptions: {
            viewportId: 'mpr-axial',
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'axial',
            voiLUTFunction: 'LINEAR_EXACT',
            initialImageOptions: {
              preset: 'middle',
            },
            syncGroups: [
              {
                type: 'voi',
                id: 'mpr',
                source: true,
                target: true,
                options: {
                  syncColormap: true,
                },
              },
            ],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
              options: {
                blendMode: 'avg',
                slabThickness: 0.5,
                // FIX: Force Bone Window to prevent gray look on CBCT
                voi: {
                  windowWidth: 4000,
                  windowCenter: 1000,
                },
              },
            },
          ],
        },
        {
          viewportOptions: {
            viewportId: 'mpr-sagittal',
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'sagittal',
            voiLUTFunction: 'LINEAR_EXACT',
            initialImageOptions: {
              preset: 'middle',
            },
            syncGroups: [
              {
                type: 'voi',
                id: 'mpr',
                source: true,
                target: true,
                options: {
                  syncColormap: true,
                },
              },
            ],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
              options: {
                blendMode: 'avg',
                slabThickness: 0.5,
                // FIX: Force Bone Window to prevent gray look on CBCT
                voi: {
                  windowWidth: 4000,
                  windowCenter: 1000,
                },
              },
            },
          ],
        },
        {
          viewportOptions: {
            viewportId: 'mpr-coronal',
            toolGroupId: 'mpr',
            viewportType: 'volume',
            orientation: 'coronal',
            voiLUTFunction: 'LINEAR_EXACT',
            initialImageOptions: {
              preset: 'middle',
            },
            syncGroups: [
              {
                type: 'voi',
                id: 'mpr',
                source: true,
                target: true,
                options: {
                  syncColormap: true,
                },
              },
            ],
          },
          displaySets: [
            {
              id: 'activeDisplaySet',
              options: {
                blendMode: 'avg',
                slabThickness: 0.5,
                // FIX: Force Bone Window to prevent gray look on CBCT
                voi: {
                  windowWidth: 4000,
                  windowCenter: 1000,
                },
              },
            },
          ],
        },
      ],
    },
  ],
};
