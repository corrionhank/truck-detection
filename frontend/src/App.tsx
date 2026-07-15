import { useEffect, useState } from 'react'
import { Moon, Sun, Truck, Database, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from './useTheme'
import annotationsSpec from './docs/annotations-spec.md?raw'

// DATA-ONLY console. The modeling views (Results / Models / Inference) and their
// backend endpoints were archived during the deliberate modeling rebuild — the
// model-capable version is preserved at archive/frontend/App.tsx. See archive/README.md.

type Tab = 'dataset' | 'spec'

type Scene = { name: string; vehicles: number }
type Dataset = {
  vehicles: number
  echoes: number
  scenes_labelled: number
  per_scene: Record<string, { vehicles: number; echoes: number }>
}

export default function App() {
  const { theme, toggle } = useTheme()
  const [tab, setTab] = useState<Tab>('dataset')
  const [scenes, setScenes] = useState<Scene[]>([])

  useEffect(() => {
    fetch('/api/scenes').then((r) => r.json()).then((d) => setScenes(d.scenes)).catch(() => setScenes([]))
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
          <div className="section-label">Data console</div>
          <h1 className="page-title">Truck Detection</h1>

          <div className="segmented" style={{ maxWidth: 280 }}>
            <button className={tab === 'dataset' ? 'active' : ''} onClick={() => setTab('dataset')}>
              <Database size={14} /> Dataset
            </button>
            <button className={tab === 'spec' ? 'active' : ''} onClick={() => setTab('spec')}>
              <FileText size={14} /> Spec
            </button>
          </div>

          {tab === 'dataset' && <DatasetView totalScenes={scenes.length} />}
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
