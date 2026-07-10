# Truck Detection — Web Console

Basic React + Vite + TypeScript frontend for the ML training/inference + data-storage
app, using the **SFS design system** (plain CSS custom properties in `src/styles.css` —
no Tailwind, no component library).

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
- `src/App.tsx` — the shell: dark-chrome topbar (wordmark + theme toggle) and three
  views (Dataset / Training / Inference) built from the universal components.

**All data shown is mock/placeholder** — there's no backend yet. Swap the `SCENES` /
`RUNS` constants in `App.tsx` for real data (e.g. counts from `Annotations-RGB.gpkg`
and training runs) when the API exists.
