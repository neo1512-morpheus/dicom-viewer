# Contributing

Thank you for considering a contribution to Oroscan Viewer. This repository extends OHIF for dental CBCT workflows, so changes should stay focused on medical imaging correctness, viewer stability, and maintainable OHIF integration.

## Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Add the upstream repository as a remote.

```powershell
git clone https://github.com/<your-username>/dicom-viewer.git
cd dicom-viewer
git remote add upstream https://github.com/neo1512-morpheus/dicom-viewer.git
```

## Local Setup

Install the OHIF workspace dependencies from the `Viewers` directory.

```powershell
cd Viewers
yarn install --frozen-lockfile
yarn dev
```

Copy `.env.example` to `.env` at the repository root when running services that need Cloudflare R2 or deployment-specific settings.

## Docker

The DICOM worker can be built and run with Docker.

```powershell
cd dicom-worker
docker build -t oroscan-dicom-worker .
docker run --env-file ../.env -p 3001:3001 oroscan-dicom-worker
```

## Pull Requests

1. Create a feature branch from `main`.
2. Keep changes focused and avoid committing generated viewer bundles unless the change specifically requires them.
3. Include a clear description of the workflow affected by the change.
4. Include screenshots or short notes for CPR, panoramic reconstruction, viewport, or rendering changes.
5. Run the relevant build or test command before opening the pull request.

## Coding Standards

- Follow the existing OHIF TypeScript and React patterns.
- Keep CPR reconstruction changes scoped to `Viewers/modes/cpr` unless an integration point requires otherwise.
- Prefer typed arrays and explicit numeric bounds for volume-processing code.
- Avoid adding patient data, private DICOM files, generated bundles, local debug logs, or credentials to commits.
- Use environment variables for secrets and document required variables in `.env.example`.
