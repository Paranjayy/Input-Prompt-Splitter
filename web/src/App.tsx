import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
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
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')
  const [isDragOver, setIsDragOver] = useState(false)

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

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
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
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
      overlap,
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

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
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
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!file) return

    // Validate file type
    const allowedTypes = ['text/plain', 'text/markdown', 'text/x-markdown']
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(txt|md|markdown)$/i)) {
      alert('Please upload a .txt or .md file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB')
      return
    }

    try {
      const content = await file.text()
      setText(content)
      setParts([]) // Clear previous results
    } catch (error) {
      alert('Error reading file: ' + error.message)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <div style={{ marginBottom: 12 }}>
            <input
              type="file"
              accept=".txt,.md,.markdown"
              onChange={handleFileInputChange}
              style={{ marginRight: 12 }}
            />
            <span style={{ fontSize: '0.9em', color: '#666' }}>
              or drag & drop a .txt/.md file below
            </span>
          </div>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              position: 'relative',
              border: isDragOver ? '2px dashed #007bff' : '1px solid #ccc',
              borderRadius: 4,
              backgroundColor: isDragOver ? '#f0f8ff' : '#fafafa',
              transition: 'all 0.2s ease'
            }}
          >
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={10}
              style={{
                width: '100%',
                border: 'none',
                backgroundColor: 'transparent',
                resize: 'vertical'
              }}
              placeholder="Paste your long text here, or drag & drop a .txt/.md file"
            />
            {isDragOver && (
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                color: '#007bff',
                fontWeight: 'bold',
                pointerEvents: 'none',
                borderRadius: 4
              }}>
                📄 Drop file here
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Split by number of words (whitespace-delimited)"><input type="radio" checked={unit === 'words'} onChange={() => setUnit('words')} /> Words</label>
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
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                <button onClick={downloadAll}>📦 Download as ZIP</button>
                <button onClick={downloadMarkdown}>📝 Download as Markdown</button>
              </div>
              <div style={{ display: 'grid', gap: 8 }}>
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
