import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
        <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
          </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
                  </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
