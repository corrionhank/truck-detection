# Truck Detection — Web Console

React + Vite + TypeScript frontend, using the **SFS design system** (plain CSS custom properties in
`src/styles.css` — no Tailwind, no component library).

Five tabs backed by the Flask API (`python3 backend/server.py` on :8787; Vite proxies `/api` + `/outputs`):
**Dataset** (live label counts), **Results** (cross-validation + full-scene numbers), **Models** (browse /
activate / archive / annotate models, view methodology cards), **Inference** (run the active model on a scene),
**Spec** (the annotation contract). The *training* scripts behind the models are archived (`archive/src/`), but
inference and the registry are live, so the console runs the existing models.

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
- `src/App.tsx` — the shell (dark-chrome topbar + theme toggle) and the five tab views
  (Dataset / Results / Models / Inference / Spec), built from the universal components. Data is
  live from the backend; the Spec tab renders `src/docs/annotations-spec.md`.
