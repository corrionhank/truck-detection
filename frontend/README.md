# Truck Detection — Data Console

React + Vite + TypeScript frontend, using the **SFS design system** (plain CSS custom properties in
`src/styles.css` — no Tailwind, no component library).

**Data-only.** The modeling views (Results / Models / Inference) were archived during the deliberate modeling
rebuild — the model-capable console is preserved at [`../archive/frontend/App.tsx`](../archive/frontend/App.tsx).
This console shows the real annotation dataset; re-add the modeling views when the new pipeline exists.

## Run

```sh
cd frontend
npm install
npm run dev        # http://localhost:5173
```

- `npm run build` — type-check (`tsc --noEmit`) + production build to `dist/`
- `npm run preview` — serve the built app

## What's here

- `src/styles.css` — the design tokens (`:root` + `[data-theme="dark"]`) and the
  universal components (buttons, inputs, card, statusline, badge-state, meter,
  segmented control, selectable list) ported verbatim from the style guide, plus a
  small app-shell layout.
- `src/useTheme.ts` — light/dark toggle; writes `data-theme` on `<html>`, persists to
  `localStorage`. The pre-paint script in `index.html` prevents a flash on reload.
- `src/App.tsx` — the shell: dark-chrome topbar (wordmark + theme toggle) and two views:
  **Dataset** (live label counts from `GET /api/dataset`, per-scene coverage) and **Spec**
  (the annotation contract, rendered from `src/docs/annotations-spec.md`).

Data is live from the backend (`python3 backend/server.py` on :8787; Vite proxies `/api`). Some CSS classes in
`styles.css` (e.g. `.model`, `.metric`) are now unused by the data-only console but kept for the model-view
rebuild.
