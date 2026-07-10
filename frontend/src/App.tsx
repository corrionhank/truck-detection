import { useState } from 'react'
import { Moon, Sun, Truck, Play, Database, Activity, ScanLine, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from './useTheme'
import annotationsSpec from './docs/annotations-spec.md?raw'

type Tab = 'dataset' | 'training' | 'inference' | 'spec'
type RunState = 's-success' | 's-running' | 's-queued' | 's-failed'

/* Mock data — no backend yet. Reflects the truck-detection project state. */
const SCENES = [
  { name: 'Bellingham_01_20260425', vehicles: 7 },
  { name: 'Centralia_02_20260511', vehicles: 6 },
  { name: 'Stanwood_08_20260504', vehicles: 3 },
  { name: 'Seattle_01_20260502', vehicles: 0 },
  { name: 'Ellensburg_01_20260504', vehicles: 0 },
]
const VEHICLES = 50
const TARGET = 150

const RUNS: { id: string; name: string; state: RunState; label: string; pct: number; meta: string }[] = [
  { id: 'r3', name: 'kprcnn-r50 · run 003', state: 's-running', label: 'running', pct: 62, meta: 'epoch 12/20 · loss 4.81' },
  { id: 'r2', name: 'kprcnn-r50 · run 002', state: 's-success', label: 'done', pct: 100, meta: 'final loss 3.92 · 2.4 h' },
  { id: 'r1', name: 'kprcnn-r50 · run 001', state: 's-failed', label: 'failed', pct: 100, meta: 'OOM at epoch 3' },
]

export default function App() {
  const { theme, toggle } = useTheme()
  const [tab, setTab] = useState<Tab>('dataset')

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

          {tab === 'dataset' && <DatasetView />}
          {tab === 'training' && <TrainingView />}
          {tab === 'inference' && <InferenceView />}
          {tab === 'spec' && <SpecView />}
        </div>
      </main>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="card stat">
      <div className="section-label">{label}</div>
      <b>{value}</b>
    </div>
  )
}

function DatasetView() {
  const [selected, setSelected] = useState<string | null>(null)
  const labeledScenes = SCENES.filter((s) => s.vehicles > 0).length
  const pct = Math.min(100, Math.round((VEHICLES / TARGET) * 100))

  return (
    <>
      <div className="stat-row">
        <Stat label="Scenes" value={SCENES.length} />
        <Stat label="Scenes labeled" value={labeledScenes} />
        <Stat label="Vehicles labeled" value={VEHICLES} />
        <Stat label="Target" value={TARGET} />
      </div>

      <div className="card">
        <div className="section-label">Labeling progress</div>
        <div className="row-between">
          <span className="quota-num">
            {VEHICLES}
            <span className="quota-sub"> / {TARGET}</span>
          </span>
          <span className="hint">{pct}% toward first-model target</span>
        </div>
        <div className="meter">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>

      <div className="card">
        <div className="section-label">Scenes</div>
        <div className="list">
          {SCENES.map((s) => (
            <button
              key={s.name}
              className={`scene-btn${selected === s.name ? ' on' : ''}`}
              onClick={() => setSelected(s.name)}
            >
              <span className="mono">{s.name}</span>
              <span className="hint">{s.vehicles > 0 ? `${s.vehicles} vehicles labeled` : 'not labeled'}</span>
            </button>
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
        <button className="primary" style={{ height: 36 }}>
          <Play size={14} /> New run
        </button>
      </div>
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

function InferenceView() {
  const [scene, setScene] = useState(SCENES[0].name)
  const [status, setStatus] = useState<{ kind: 'ok' | 'err' | 'info'; msg: string } | null>(null)

  const run = () => {
    setStatus({ kind: 'info', msg: 'running…' })
    window.setTimeout(
      () => setStatus({ kind: 'ok', msg: `detected 0 echoes in ${scene} (stub — no trained model yet)` }),
      600,
    )
  }

  return (
    <div className="card" style={{ maxWidth: 560 }}>
      <div className="section-label">Run detection</div>
      <p className="hint" style={{ margin: 0 }}>
        Pick a scene and run the detector. Wires the UI end to end; returns a stub until a model exists.
      </p>
      <label className="field-label">Scene</label>
      <select value={scene} onChange={(e) => setScene(e.target.value)}>
        {SCENES.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name}
          </option>
        ))}
      </select>
      <button className="primary wide" onClick={run} style={{ marginTop: 12 }}>
        <Play size={14} /> Run detection
      </button>
      {status && (
        <div className={`statusline ${status.kind}`} style={{ marginTop: 10 }}>
          <code>[{status.kind}]</code> {status.msg}
        </div>
      )}
    </div>
  )
}
