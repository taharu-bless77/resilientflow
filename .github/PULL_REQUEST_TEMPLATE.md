<!-- PR テンプレート: 作成時に自動で本文が入ります -->
<!-- Title は手動で入力しても構いません -->

## Summary

Create a clean branch excluding `node_modules` for safe review.

## What I changed
- Added source files and frontend `src` (no `node_modules`).
- Kept `frontend/package.json` and Storybook config.

## How to review
- Run tests: `pytest -q` (backend)
- Frontend: `cd frontend && npm install && npm run storybook` to run Storybook locally

## Notes
- This branch intentionally excludes `frontend/node_modules/` and any large binaries.
- The original `planA` branch contains an Electron binary exceeding GitHub's 100MB limit; consider cleaning history or using Git LFS.

---
Title suggestion: feat(clean): add source and frontend src (no node_modules)
Body suggestion: Create a clean branch excluding node_modules for safe review.
