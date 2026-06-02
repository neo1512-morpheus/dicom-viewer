[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# Oroscan Viewer - OHIF-based dental CBCT viewer with Curved Planar Reformation (CPR) and Panoramic Reconstruction

Oroscan Viewer is built on OHIF, the Open Health Imaging Foundation viewer. OHIF is a zero-footprint, browser-based medical imaging viewer for DICOM and DICOMweb workflows, with a configurable and extensible application architecture. The upstream OHIF project has about 4,200 GitHub stars and is used across medical imaging research workflows, including work associated with Harvard Medical School, Massachusetts General Hospital, and National Cancer Institute-funded imaging projects.

This project extends OHIF for dental CBCT imaging with Curved Planar Reformation (CPR) and panoramic reconstruction. CPR is clinically important in dental imaging because it lets users trace the dental arch and inspect curved anatomy as a continuous panoramic view rather than disconnected axial, sagittal, and coronal slices. OHIF does not natively include a dedicated dental CPR/panoramic reconstruction workflow, so this repository adds that missing workflow on top of the OHIF viewer architecture.

## Features

- CPR viewport for dental arch-based review
- GPU rendering path for interactive reconstruction
- Panoramic reconstruction from CBCT volume data
- VTK-native panoramic CPR rendering
- Noise reduction and background suppression experiments for dental pano quality
- 2D and 3D DICOM viewing through the OHIF framework
- JPEG 2000 compression/transcoding support through the DICOM worker
- Cloudflare R2 storage integration for DICOM-derived assets

## Branch Guide

| Branch | Contents |
| --- | --- |
| `main` | Public-ready base branch after cleanup and merge review. |
| `feature/cpr-volume-viewport` | Earlier CPR volume viewport work already present on GitHub. |
| `feature/virtual-pano-reconstruction` | Virtual panoramic reconstruction experiments. |
| `feature/vtk-native-pano` | VTK-native panoramic CPR rendering work. |
| `feature/vtk-pano-noise-reduction` | Stable CPR quality improvements through the April 8 black-patch reduction work. |
| `feature/cpr-gpu-render-finalized` | GPU CPR rendering finalization work. |
| `wip/cpr-may-improvements` | Preserved May experimental MIP and multi-method comparison work; intentionally not public-ready yet. |

## Tech Stack

| Area | Technology |
| --- | --- |
| Medical image viewer | OHIF v3.8-based viewer |
| Runtime | Node.js, Yarn workspaces |
| Containerization | Docker |
| Storage | Cloudflare R2 |
| DICOM transcoding | `gdcmconv` |
| Deployment | Netlify |

## Setup

### Prerequisites

- Node.js 16 or newer
- Yarn 1.x
- Docker, when using the DICOM worker or containerized services
- Access to a DICOMweb source or static DICOM JSON data
- Cloudflare R2 credentials, if using R2-backed storage

### Environment Variables

Copy `.env.example` to `.env` and fill in only the values required by the services you run.

```env
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
R2_ACCOUNT_ID=<cloudflare-account-id>
R2_ACCESS_KEY_ID=<r2-access-key-id>
R2_SECRET_ACCESS_KEY=<r2-secret-access-key>
R2_BUCKET_NAME=<r2-bucket-name>
R2_PUBLIC_URL=https://<public-r2-domain>

PERCY_TOKEN=<percy-token>
PUBLIC_URL=/demo/
APP_CONFIG=config/netlify.js
USE_HASH_ROUTER=false
```

### Local Development

```powershell
cd Viewers
yarn install --frozen-lockfile
yarn dev
```

### Docker Worker

```powershell
cd dicom-worker
docker build -t oroscan-dicom-worker .
docker run --env-file ../.env -p 3001:3001 oroscan-dicom-worker
```

## Screenshots

[Screenshot: CPR viewport showing dental CBCT scan]

## Contributing

Contributions are welcome for CPR reconstruction quality, OHIF integration, DICOM data-source compatibility, documentation, and deployment hardening. Please open an issue or pull request with a focused description of the clinical or technical workflow being improved.
