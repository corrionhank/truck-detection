import { useEffect, useState } from 'react'
import { Moon, Sun, Truck, Play, Database, BarChart3, Layers, ScanLine, FileText, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from './useTheme'
import annotationsSpec from './docs/annotations-spec.md?raw'

type Tab = 'dataset' | 'results' | 'models' | 'inference' | 'spec'

type ModelEntry = {
  id: string
  name: string
  weights: string
  status: 'active' | 'archived'
  created: string
  arch: { backbone?: string; anchors: string; classes?: number; keypoints?: number }
  train: { vehicles?: number; scenes?: string[]; epochs?: number; aug?: string; device?: string; script?: string }
  metrics: Record<string, string | number | null>
  notes: string
}
type Registry = { active: string; models: ModelEntry[] }

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
  model_id?: string
  model_name?: string
}

export default function App() {
  const { theme, toggle } = useTheme()
  const [tab, setTab] = useState<Tab>('dataset')
  const [scenes, setScenes] = useState<Scene[]>([])
  const [registry, setRegistry] = useState<Registry | null>(null)
  const refreshModels = () =>
    fetch('/api/models').then((r) => r.json()).then(setRegistry).catch(() => setRegistry(null))

  useEffect(() => {
    fetch('/api/scenes').then((r) => r.json()).then((d) => setScenes(d.scenes)).catch(() => setScenes([]))
    refreshModels()
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

          <div className="segmented" style={{ maxWidth: 560 }}>
            <button className={tab === 'dataset' ? 'active' : ''} onClick={() => setTab('dataset')}>
              <Database size={14} /> Dataset
            </button>
            <button className={tab === 'results' ? 'active' : ''} onClick={() => setTab('results')}>
              <BarChart3 size={14} /> Results
            </button>
            <button className={tab === 'models' ? 'active' : ''} onClick={() => setTab('models')}>
              <Layers size={14} /> Models
            </button>
            <button className={tab === 'inference' ? 'active' : ''} onClick={() => setTab('inference')}>
              <ScanLine size={14} /> Inference
            </button>
            <button className={tab === 'spec' ? 'active' : ''} onClick={() => setTab('spec')}>
              <FileText size={14} /> Spec
            </button>
          </div>

          {tab === 'dataset' && <DatasetView totalScenes={scenes.length} />}
          {tab === 'results' && <ResultsView />}
          {tab === 'models' && <ModelsView registry={registry} refresh={refreshModels} onRun={() => setTab('inference')} />}
          {tab === 'inference' && <InferenceView scenes={scenes} registry={registry} />}
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

  const rows = Object.entries(data.per_scene).sort((a, b) => b[1].vehicles - a[1].vehicles)
  const maxV = Math.max(1, ...rows.map(([, s]) => s.vehicles))
  const centralia = rows
    .filter(([n]) => n.includes('Centralia'))
    .reduce((a, [, s]) => a + s.vehicles, 0)
  const concPct = Math.round((100 * centralia) / Math.max(1, data.vehicles))

  return (
    <>
      <div className="stat-row">
        <Stat label="Labeled scenes" value={data.scenes_labelled} />
        <Stat label="Vehicles" value={data.vehicles} />
        <Stat label="Echoes (keypoints)" value={data.echoes} />
        <Stat label="Scenes on disk" value={totalScenes || '—'} />
      </div>

      <div className="card">
        <div className="row-between">
          <div className="section-label">Labels per scene</div>
          <span className="hint">{concPct}% in the Centralia / south-I-5 corridor</span>
        </div>
        <div className="list" style={{ gap: 8, marginTop: 4 }}>
          {rows.map(([name, s]) => (
            <div key={name}>
              <div className="row-between" style={{ marginBottom: 3 }}>
                <span className="mono" style={{ fontSize: 13 }}>{name}</span>
                <span className="hint">{s.vehicles} veh · {s.echoes} echoes</span>
              </div>
              <div className="meter" style={{ height: 6 }}>
                <div className="meter-fill" style={{ width: `${(100 * s.vehicles) / maxV}%` }} />
              </div>
            </div>
          ))}
        </div>
        <p className="hint" style={{ marginTop: 12 }}>
          Coverage is volume-rich but concentrated — the next labeling is worth more on new corridors
          (Bellingham, Stanwood, Seattle, more I-90) than on more Centralia. Each vehicle = 3 keypoints
          (blue → red → green).
        </p>
      </div>
    </>
  )
}

function ResultsView() {
  const cv = [
    { scene: 'Centralia_01', recall: '58 / 58', pct: 100, err: '0.9' },
    { scene: 'Centralia_02', recall: '54 / 54', pct: 100, err: '0.8' },
    { scene: 'Tacoma-Centralia_01', recall: '74 / 94', pct: 79, err: '1.2' },
    { scene: 'Tacoma-Centralia_02', recall: '92 / 101', pct: 91, err: '1.0' },
  ]
  return (
    <>
      <div className="card">
        <div className="section-label">Model</div>
        <p style={{ margin: '6px 0 0' }}>
          <b>Keypoint R-CNN</b> (ResNet-50 + FPN), fine-tuned on the 339-vehicle set. Predicts 3 keypoints
          per vehicle — the blue/red/green moving-echo streak. Trained on CPU (the Apple GPU / MPS diverges).
        </p>
      </div>

      <div className="stat-row">
        <Stat label="Within-corridor recall" value="91%" />
        <Stat label="Keypoint error" value="~1 px" />
        <Stat label="Full-scene recall" value="40%" />
        <Stat label="Full-scene precision" value="68%" />
      </div>

      <div className="card">
        <div className="section-label">Within-corridor generalization</div>
        <p className="hint" style={{ marginTop: 2 }}>
          Leave-one-scene-out CV on the 4 Centralia scenes — train on 3, test on the unseen one
          (centered chip per labeled vehicle).
        </p>
        <div className="list" style={{ gap: 8 }}>
          {cv.map((r) => (
            <div key={r.scene}>
              <div className="row-between" style={{ marginBottom: 3 }}>
                <span className="mono" style={{ fontSize: 13 }}>{r.scene}</span>
                <span className="hint">recall {r.recall} ({r.pct}%) · {r.err} px</span>
              </div>
              <div className="meter" style={{ height: 6 }}>
                <div className="meter-fill" style={{ width: `${r.pct}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-label">Full-scene deployment (held-out Tacoma-Centralia_01)</div>
        <p style={{ margin: '6px 0' }}>
          Sliding-window detection over the whole raw scene: <b>38 true positives</b>, <b>18 false positives</b>,
          <b> 56 missed</b> → 40% recall, 68% precision.
        </p>
        <p className="hint" style={{ margin: 0 }}>
          The gap from 91% is the deployment reality: the model <b>recognizes</b> a centered echo well but is
          weaker at <b>finding</b> them across a full scene. False positives are bright road/lane-paint features
          (not off-road hallucinations); most misses are real echoes in <b>dense traffic</b>, where the
          detection dedup merges neighboring trucks — a method fix, not a data problem.
        </p>
      </div>
    </>
  )
}

function ModelsView({ registry, refresh, onRun }: { registry: Registry | null; refresh: () => void; onRun: () => void }) {
  const [busy, setBusy] = useState(false)
  if (!registry) return <div className="card">Loading models… (start the backend if this hangs)</div>

  const post = async (url: string, body: object) => {
    setBusy(true)
    await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    await refresh()
    setBusy(false)
  }
  const sorted = [...registry.models].sort(
    (a, b) => (a.id === registry.active ? -1 : b.id === registry.active ? 1 : 0))

  return (
    <>
      <p className="hint" style={{ marginTop: 0 }}>
        Every trained model — config, results, and findings. The <b>active</b> model runs in the Inference tab.
        This log is <code>models/registry.json</code> (committed); weights live in <code>weights/</code>.
      </p>
      <div className="list">
        {sorted.map((m) => {
          const isActive = m.id === registry.active
          const nums = Object.entries(m.metrics || {}).filter(([, v]) => typeof v === 'number')
          return (
            <div key={m.id} className={`card model${isActive ? ' model-active' : ''}`}>
              <div className="row-between">
                <div><span style={{ fontWeight: 600 }}>{m.name}</span> <span className="mono hint">{m.id}</span></div>
                <span className={`badge-state ${isActive ? 's-running' : 's-queued'}`}>
                  {isActive ? 'ACTIVE' : m.status}
                </span>
              </div>
              <div className="model-meta">
                <span>{m.train.vehicles ?? '?'} vehicles</span>
                <span>{(m.train.scenes || []).length} scenes</span>
                <span>anchors: {m.arch.anchors}</span>
                <span>aug: {m.train.aug || 'none'}</span>
                <span>{m.train.epochs}ep · {m.train.device}</span>
                <span className="hint">{m.created}</span>
              </div>
              {nums.length > 0 && (
                <div className="model-meta">
                  {nums.map(([k, v]) => <span key={k}><b>{k.replace(/_/g, ' ')}</b> {v}</span>)}
                </div>
              )}
              <NotesEditor model={m} onSave={(notes) => post(`/api/models/${m.id}`, { notes })} />
              <div className="row-between" style={{ marginTop: 4, alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: 8 }}>
                  {isActive
                    ? <button className="primary" style={{ height: 30 }} onClick={onRun}><Play size={13} /> Run inference</button>
                    : <button className="primary" style={{ height: 30 }} disabled={busy} onClick={() => post('/api/models/active', { id: m.id })}>Set active</button>}
                  <button className="ghost" style={{ height: 30 }} disabled={busy}
                    onClick={() => post(`/api/models/${m.id}`, { status: m.status === 'archived' ? 'active' : 'archived' })}>
                    {m.status === 'archived' ? 'Unarchive' : 'Archive'}
                  </button>
                </div>
                <span className="mono hint">{m.weights}</span>
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}

function NotesEditor({ model, onSave }: { model: ModelEntry; onSave: (n: string) => void }) {
  const [notes, setNotes] = useState(model.notes)
  const [editing, setEditing] = useState(false)
  useEffect(() => { setNotes(model.notes) }, [model.notes])
  if (!editing)
    return (
      <div className="model-notes" onClick={() => setEditing(true)} title="Click to edit">
        {model.notes || <span className="hint">click to add notes…</span>}
      </div>
    )
  return (
    <div>
      <textarea className="notes-area" value={notes} onChange={(e) => setNotes(e.target.value)} rows={5} />
      <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
        <button className="primary" style={{ height: 28 }} onClick={() => { onSave(notes); setEditing(false) }}>Save</button>
        <button className="ghost" style={{ height: 28 }} onClick={() => { setNotes(model.notes); setEditing(false) }}>Cancel</button>
      </div>
    </div>
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

function InferenceView({ scenes, registry }: { scenes: Scene[]; registry: Registry | null }) {
  const [scene, setScene] = useState('')
  const [modelId, setModelId] = useState('')
  const [thresh, setThresh] = useState(0.5)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (!scene && scenes.length) setScene(scenes[0].name)
  }, [scenes, scene])
  useEffect(() => {
    if (!modelId && registry) setModelId(registry.active)
  }, [registry, modelId])

  const run = async () => {
    setBusy(true); setError(null); setResult(null)
    try {
      const r = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene, thresh, model_id: modelId }),
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
        Slides the selected model across a whole scene and maps detections back to map coordinates.
        Takes ~1–2&nbsp;min on CPU. Lower the threshold for higher recall (and more false positives).
      </p>

      <label className="field-label">Model</label>
      <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={busy}>
        {registry?.models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}{m.id === registry.active ? ' · active' : ` · ${m.status}`}
          </option>
        ))}
      </select>

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
        {busy ? ' Running… (~1–2 min)' : ' Run detection'}
      </button>

      {error && (
        <div className="statusline err" style={{ marginTop: 10 }}>
          <code>[err]</code> {error}
        </div>
      )}

      {result && <ResultPanel result={result} nonce={nonce} />}
    </div>
  )
}

function Metric({ label, value, sub }: { label: string; value: number | string; sub?: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

function ResultPanel({ result, nonce }: { result: DetectResult; nonce: number }) {
  const scores = result.detections.map((d) => d.score)
  const lo = scores.length ? Math.min(...scores) : 0
  const hi = scores.length ? Math.max(...scores) : 0
  const med = scores.length ? [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)] : 0
  const g = result.gt
  const recall = g && g.labelled ? g.recall / g.labelled : null
  const precision = g && result.count ? g.near_label / result.count : null
  const missed = g ? g.labelled - g.recall : null
  const f1 = recall != null && precision != null && recall + precision > 0
    ? (2 * recall * precision) / (recall + precision) : null
  const countErr = g && g.labelled ? (result.count - g.labelled) / g.labelled : null
  const pct = (x: number) => `${Math.round(x * 100)}%`
  const signed = (x: number) => `${x >= 0 ? '+' : ''}${Math.round(x * 100)}%`

  return (
    <div style={{ marginTop: 16 }}>
      <div className="metric-row">
        <Metric label="Detected" value={result.count} sub={`stride ${result.stride} · thr ${result.thresh}`} />
        {g && <Metric label="Recall" value={pct(recall!)} sub={`${g.recall}/${g.labelled} found`} />}
        {g && <Metric label="Precision*" value={pct(precision!)} sub={`${g.near_label}/${result.count} on-label`} />}
        {g && f1 != null && <Metric label="F1*" value={pct(f1)} sub="precision · recall" />}
        {g && <Metric label="Missed (FN)" value={missed!} sub="labeled, undetected" />}
        {g && <Metric label="False alarms*" value={g.elsewhere} sub="detections off-label" />}
        {g && countErr != null && (
          <Metric label="Count error*" value={signed(countErr)} sub={`${result.count} vs ${g.labelled} labeled`} />
        )}
        <Metric label="Confidence" value={`${lo.toFixed(2)}–${hi.toFixed(2)}`} sub={`median ${med.toFixed(2)}`} />
      </div>
      {g && (
        <p className="hint" style={{ marginTop: 8 }}>
          *Precision, F1, false alarms, and count error are affected by partial labeling — {g.elsewhere}{' '}
          detection{g.elsewhere === 1 ? '' : 's'} fell off any label, but scenes aren't exhaustively labeled, so
          some are real unlabeled trucks, not errors. For traffic monitoring, a stable, calibratable count error
          matters more than per-truck recall (Van Etten 2024).
        </p>
      )}
      <img
        src={`${result.montage_url}?v=${nonce}`}
        alt="detections"
        style={{ width: '100%', borderRadius: 8, marginTop: 8, border: '1px solid var(--border, #ddd)' }}
      />
    </div>
  )
}
