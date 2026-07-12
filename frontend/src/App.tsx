import { useEffect, useState } from 'react'
import { Moon, Sun, Truck, Play, Database, Activity, ScanLine, FileText, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from './useTheme'
import annotationsSpec from './docs/annotations-spec.md?raw'

type Tab = 'dataset' | 'training' | 'inference' | 'spec'
type RunState = 's-success' | 's-running' | 's-queued' | 's-failed'

type Scene = { name: string; vehicles: number }
type Dataset = {
  vehicles: number
  echoes: number
  scenes_labelled: number
  per_scene: Record<string, { vehicles: number; echoes: number }>
}
type Detection = { score: number; red_utm: [number, number]; keypoints_px: number[][] }
type DetectResult = {
  scene: string
  count: number
  stride: number
  thresh: number
  detections: Detection[]
  gt: { labelled: number; recall: number; near_label: number; elsewhere: number } | null
  montage_url: string
  preview_url: string
}

const VEHICLE_TARGET = 150 // vehicles needed before the model can generalise (Adamiak used ~1000+)

/* Training runs are illustrative only — real training runs from the CLI (see MODEL.md). */
const RUNS: { id: string; name: string; state: RunState; label: string; pct: number; meta: string }[] = [
  { id: 'r3', name: 'kprcnn-r50-jitter · run 002', state: 's-success', label: 'done', pct: 100, meta: 'final loss 1.32 · 30 epochs (CPU)' },
  { id: 'r1', name: 'kprcnn-r50 · run 001', state: 's-success', label: 'done', pct: 100, meta: 'final loss 1.38 · overfit demo' },
]

export default function App() {
  const { theme, toggle } = useTheme()
  const [tab, setTab] = useState<Tab>('dataset')
  const [scenes, setScenes] = useState<Scene[]>([])

  useEffect(() => {
    fetch('/api/scenes')
      .then((r) => r.json())
      .then((d) => setScenes(d.scenes))
      .catch(() => setScenes([]))
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <Truck size={18} /> SFS <span className="wm-dim">Truck Detection</span>
        </div>
        <div className="topbar-right">
          <button className="icon-btn" onClick={toggle} title="Toggle theme" aria-label="Toggle theme">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <main className="content">
        <div className="view">
          <div className="section-label">ML console</div>
          <h1 className="page-title">Truck Detection</h1>

          <div className="segmented" style={{ maxWidth: 440 }}>
            <button className={tab === 'dataset' ? 'active' : ''} onClick={() => setTab('dataset')}>
              <Database size={14} /> Dataset
            </button>
            <button className={tab === 'training' ? 'active' : ''} onClick={() => setTab('training')}>
              <Activity size={14} /> Training
            </button>
            <button className={tab === 'inference' ? 'active' : ''} onClick={() => setTab('inference')}>
              <ScanLine size={14} /> Inference
            </button>
            <button className={tab === 'spec' ? 'active' : ''} onClick={() => setTab('spec')}>
              <FileText size={14} /> Spec
            </button>
          </div>

          {tab === 'dataset' && <DatasetView totalScenes={scenes.length} />}
          {tab === 'training' && <TrainingView />}
          {tab === 'inference' && <InferenceView scenes={scenes} />}
          {tab === 'spec' && <SpecView />}
        </div>
      </main>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="card stat">
      <div className="section-label">{label}</div>
      <b>{value}</b>
    </div>
  )
}

function DatasetView({ totalScenes }: { totalScenes: number }) {
  const [data, setData] = useState<Dataset | null>(null)
  const [err, setErr] = useState(false)

  useEffect(() => {
    fetch('/api/dataset')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setErr(true))
  }, [])

  if (err) return <div className="card">Backend not reachable — start it with <code>python3 backend/server.py</code></div>
  if (!data) return <div className="card">Loading dataset…</div>

  const pct = Math.min(100, Math.round((data.vehicles / VEHICLE_TARGET) * 100))
  const rows = Object.entries(data.per_scene)

  return (
    <>
      <div className="stat-row">
        <Stat label="Scenes" value={totalScenes || '—'} />
        <Stat label="Scenes labeled" value={data.scenes_labelled} />
        <Stat label="Vehicles labeled" value={data.vehicles} />
        <Stat label="Echoes (keypoints)" value={data.echoes} />
      </div>

      <div className="card">
        <div className="section-label">Labeling progress</div>
        <div className="row-between">
          <span className="quota-num">
            {data.vehicles}
            <span className="quota-sub"> / {VEHICLE_TARGET} vehicles</span>
          </span>
          <span className="hint">{pct}% toward first-model target · {data.echoes} echoes (3 per vehicle)</span>
        </div>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className="card">
        <div className="section-label">Labeled scenes</div>
        <div className="list">
          {rows.map(([name, s]) => (
            <div key={name} className="scene-btn">
              <span className="mono">{name}</span>
              <span className="hint">{s.vehicles} vehicles · {s.echoes} echoes</span>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

function TrainingView() {
  return (
    <>
      <div className="row-between">
        <div className="section-label">Runs</div>
        <button className="primary" style={{ height: 36 }} disabled title="Training runs from the CLI — see MODEL.md">
          <Play size={14} /> New run
        </button>
      </div>
      <p className="hint" style={{ marginTop: 0 }}>
        Illustrative — training runs from the terminal (<code>src/train_keypoint_rcnn_jitter.py</code>). The
        Inference tab uses the finished weights.
      </p>
      <div className="list">
        {RUNS.map((r) => (
          <div key={r.id} className="card run">
            <div className="row-between">
              <span style={{ fontWeight: 600 }}>{r.name}</span>
              <span className={`badge-state ${r.state}`}>{r.label}</span>
            </div>
            <div className="meter">
              <div className="meter-fill" style={{ width: `${r.pct}%` }} />
            </div>
            <div className="statusline">{r.meta}</div>
          </div>
        ))}
      </div>
    </>
  )
}

function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  )
}

function SpecView() {
  return (
    <div className="card">
      <Markdown>{annotationsSpec}</Markdown>
    </div>
  )
}

function InferenceView({ scenes }: { scenes: Scene[] }) {
  const [scene, setScene] = useState('')
  const [thresh, setThresh] = useState(0.5)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (!scene && scenes.length) setScene(scenes[0].name)
  }, [scenes, scene])

  const run = async () => {
    setBusy(true); setError(null); setResult(null)
    try {
      const r = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene, thresh }),
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data.error || r.statusText)
      setResult(data)
      setNonce((n) => n + 1)
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card" style={{ maxWidth: 900 }}>
      <div className="section-label">Run detection</div>
      <p className="hint" style={{ margin: 0 }}>
        Runs the trained Keypoint R-CNN (<code>keypoint_rcnn_echo_jitter.pt</code>) across the whole scene.
        Takes ~1&nbsp;min on CPU.
      </p>

      <label className="field-label">Scene</label>
      <select value={scene} onChange={(e) => setScene(e.target.value)} disabled={busy}>
        {scenes.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name}{s.vehicles ? ` · ${s.vehicles} labeled` : ' · unlabeled'}
          </option>
        ))}
      </select>

      <label className="field-label">Confidence threshold: {thresh.toFixed(2)}</label>
      <input
        type="range" min={0.3} max={0.9} step={0.05} value={thresh}
        onChange={(e) => setThresh(parseFloat(e.target.value))} disabled={busy}
        style={{ width: '100%' }}
      />

      <button className="primary wide" onClick={run} style={{ marginTop: 12 }} disabled={busy || !scene}>
        {busy ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
        {busy ? ' Running… (~1 min)' : ' Run detection'}
      </button>

      {error && (
        <div className="statusline err" style={{ marginTop: 10 }}>
          <code>[err]</code> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 16 }}>
          <div className="row-between">
            <span style={{ fontWeight: 600 }}>
              {result.count} detections · stride {result.stride} · thresh {result.thresh}
            </span>
            {result.gt && (
              <span className="hint">
                recall {result.gt.recall}/{result.gt.labelled} labeled · {result.gt.near_label} near-label ·{' '}
                {result.gt.elsewhere} elsewhere
              </span>
            )}
          </div>
          {result.gt && (
            <p className="hint" style={{ marginTop: 4 }}>
              "elsewhere" ≠ false positives: most scenes are only partly labeled, so many are real unlabeled trucks.
            </p>
          )}
          <img
            src={`${result.montage_url}?v=${nonce}`}
            alt="detections"
            style={{ width: '100%', borderRadius: 8, marginTop: 8, border: '1px solid var(--border, #ddd)' }}
          />
        </div>
      )}
    </div>
  )
}
