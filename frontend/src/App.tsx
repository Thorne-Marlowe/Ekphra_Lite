import { useEffect, useRef, useState } from 'react'
import { Download, Image as ImageIcon, Pencil, RotateCcw, Save } from 'lucide-react'
import { StoryboardRequestController } from './storyboardRequest'

type Dialogue = { character_id?: string; text?: string; emotion?: string }
type PanelMeta = { id: string; x: number; y: number; width: number; height: number; prompt: string; dialogues: Dialogue[] }
type PageMeta = { url: string; pageWidth: number; pageHeight: number; panels: PanelMeta[] }
type Profile = { id: string; label: string; description: string }
const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export default function App() {
  const [idea, setIdea] = useState('')
  const [draft, setDraft] = useState('')
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [profile, setProfile] = useState('')
  const [profileError, setProfileError] = useState('')
  const [loading, setLoading] = useState(false)
  const [storyboardLoading, setStoryboardLoading] = useState(false)
  const [message, setMessage] = useState('Create a storyboard, review it, then render approved panels through ComfyUI.')
  const [pages, setPages] = useState<string[]>([])
  const [history, setHistory] = useState<string[]>([])
  const [meta, setMeta] = useState<Record<string, PageMeta>>({})
  const [editing, setEditing] = useState<{ pageUrl: string; panel: PanelMeta } | null>(null)
  const [panelPrompt, setPanelPrompt] = useState('')
  const [dialogues, setDialogues] = useState('[]')
  const [editMode, setEditMode] = useState<'panel' | 'dialogue'>('panel')
  const [steps, setSteps] = useState('')
  const [guidance, setGuidance] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [seed, setSeed] = useState('')
  const storyboardRequest = useRef(new StoryboardRequestController())

  const api = async (path: string, body?: unknown, signal?: AbortSignal) => {
    const response = await fetch(`${API_BASE}${path}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal } : { signal })
    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }
    return response.json()
  }

  const loadMeta = async (pageUrl: string) => {
    const data = await api(`/api/page-meta?url=${encodeURIComponent(pageUrl)}`) as PageMeta
    setMeta(previous => ({ ...previous, [pageUrl]: data }))
    return data
  }

  useEffect(() => {
    void api('/api/capabilities').then(data => {
      const items = Array.isArray(data.profiles) ? data.profiles as Profile[] : []
      setProfiles(items)
      setProfile(items[0]?.id || '')
      setProfileError(data.profileError || '')
    }).catch(error => setProfileError(String(error)))
    void api('/api/history').then(data => setHistory((data.history || []).map((url: string) => `${API_BASE}${url}`))).catch(() => undefined)
  }, [])

  const createStoryboard = async () => {
    if (!idea.trim()) return
    const { controller, id: requestId } = storyboardRequest.current.start()
    setStoryboardLoading(true)
    setLoading(true); setMessage('Creating storyboard with the configured local LLM…')
    try {
      const data = await api('/api/storyboards', { prompt: idea, pageCount: 1 }, controller.signal)
      if (!storyboardRequest.current.isCurrent(requestId)) return
      setDraft(JSON.stringify(data.storyboard, null, 2))
      setMessage('Storyboard ready. Review or edit the JSON, then approve it for ComfyUI rendering.')
    } catch (error) {
      if (!storyboardRequest.current.isCurrent(requestId)) return
      if (error instanceof DOMException && error.name === 'AbortError') setMessage('Storyboard generation cancelled.')
      else setMessage(`Storyboard error: ${error}`)
    } finally {
      if (storyboardRequest.current.isCurrent(requestId)) {
        storyboardRequest.current.finish(requestId)
        setStoryboardLoading(false)
        setLoading(false)
      }
    }
  }

  const cancelStoryboard = () => {
    if (!storyboardRequest.current.cancel()) return
    setStoryboardLoading(false)
    setLoading(false)
    setMessage('Storyboard generation cancelled.')
  }

  const parseDraft = () => {
    try { return JSON.parse(draft) } catch { throw new Error('Storyboard JSON is invalid. Fix it before rendering.') }
  }

  const render = async () => {
    if (!profile) { setMessage('Select a renderer profile after configuring renderer_profiles.json.'); return }
    setLoading(true); setMessage('Submitting approved storyboard panels to ComfyUI…')
    try {
      const data = await api('/api/render-storyboard', {
        storyboard: parseDraft(), rendererProfile: profile, steps: steps ? Number(steps) : null,
        guidance: guidance ? Number(guidance) : null, negativePrompt, seed: seed ? Number(seed) : null,
      })
      const urls = data.pages.map((url: string) => `${API_BASE}${url}`)
      setPages(urls); setHistory(previous => [...urls, ...previous.filter(item => !urls.includes(item))])
      await Promise.all(data.pages.map((url: string) => loadMeta(url)))
      setMessage(`Rendered ${urls.length} comic page(s) with ${profile}.`)
    } catch (error) { setMessage(`Render error: ${error}`) } finally { setLoading(false) }
  }

  const openEditor = (pageUrl: string, panel: PanelMeta) => {
    setEditing({ pageUrl, panel }); setPanelPrompt(panel.prompt); setDialogues(JSON.stringify(panel.dialogues, null, 2)); setEditMode('panel')
  }

  const regenerate = async () => {
    if (!editing || !profile) return
    let parsedDialogues: Dialogue[]
    try { parsedDialogues = JSON.parse(dialogues); if (!Array.isArray(parsedDialogues)) throw new Error() } catch { setMessage('Dialogue must be a JSON array.'); return }
    setLoading(true)
    try {
      const data = await api('/api/regenerate-panel', {
        pageUrl: editing.pageUrl, panelId: editing.panel.id, panelPrompt, dialogues: parsedDialogues, mode: editMode,
        rendererProfile: profile, steps: steps ? Number(steps) : null, guidance: guidance ? Number(guidance) : null,
        negativePrompt, seed: seed ? Number(seed) : null,
      })
      const fresh = `${API_BASE}${data.url}?v=${Date.now()}`
      setPages(previous => previous.map(url => url.split('?')[0] === `${API_BASE}${editing.pageUrl}` ? fresh : url))
      await loadMeta(data.url)
      setEditing(null); setMessage(editMode === 'panel' ? 'Panel regenerated through ComfyUI.' : 'Dialogue recomposed without sending an image job.')
    } catch (error) { setMessage(`Panel update error: ${error}`) } finally { setLoading(false) }
  }

  const download = (url: string) => { const link = document.createElement('a'); link.href = url; link.download = 'comic-page.png'; link.click() }
  const save = async (url: string) => { try { await api('/api/save-page', { pageUrl: new URL(url).pathname }); setMessage('Comic page saved in outputs/saved.') } catch (error) { setMessage(`Save error: ${error}`) } }

  return <main className="min-h-screen bg-gray-50 p-6 text-gray-900">
    <header className="mx-auto mb-6 max-w-7xl"><h1 className="text-2xl font-semibold">Ekphra — FLUX.2 Comic Studio</h1><p className="mt-1 text-sm text-gray-600">{message}</p></header>
    <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[minmax(360px,0.8fr)_minmax(460px,1.2fr)]">
      <section className="space-y-4 rounded-lg border bg-white p-5 shadow-sm">
        <label className="block text-sm font-medium">Prose source</label>
        <textarea value={idea} onChange={event => setIdea(event.target.value)} className="h-32 w-full rounded border p-3 text-sm" placeholder="Paste the scene or several paragraphs of prose…" />
        <div className="flex gap-2"><button disabled={loading || !idea.trim()} onClick={() => void createStoryboard()} className="rounded bg-gray-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"><Pencil className="mr-2 inline h-4 w-4" />Create storyboard</button>{storyboardLoading && <button onClick={cancelStoryboard} className="rounded border border-gray-400 px-4 py-2 text-sm font-medium">Cancel</button>}</div>
        <label className="block pt-3 text-sm font-medium">Approved storyboard JSON</label>
        <textarea value={draft} onChange={event => setDraft(event.target.value)} className="h-[28rem] w-full rounded border p-3 font-mono text-xs" placeholder="The reviewed storyboard appears here before any images are generated." />
      </section>
      <section className="space-y-4 rounded-lg border bg-white p-5 shadow-sm">
        <h2 className="font-semibold">ComfyUI renderer</h2>
        {profileError && <p className="rounded bg-amber-50 p-3 text-sm text-amber-800">{profileError}</p>}
        <label className="block text-sm">Workflow profile<select value={profile} onChange={event => setProfile(event.target.value)} className="mt-1 w-full rounded border p-2"><option value="">Select profile</option>{profiles.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        {profiles.find(item => item.id === profile)?.description && <p className="text-xs text-gray-500">{profiles.find(item => item.id === profile)?.description}</p>}
        <div className="grid grid-cols-2 gap-3 text-sm"><label>Steps<input value={steps} onChange={event => setSteps(event.target.value)} className="mt-1 w-full rounded border p-2" placeholder="Workflow default" /></label><label>Guidance<input value={guidance} onChange={event => setGuidance(event.target.value)} className="mt-1 w-full rounded border p-2" placeholder="Workflow default" /></label><label>Seed<input value={seed} onChange={event => setSeed(event.target.value)} className="mt-1 w-full rounded border p-2" placeholder="Workflow default" /></label><label>Negative prompt<input value={negativePrompt} onChange={event => setNegativePrompt(event.target.value)} className="mt-1 w-full rounded border p-2" placeholder="If profile maps it" /></label></div>
        <button disabled={loading || !draft.trim() || !profile} onClick={() => void render()} className="rounded bg-indigo-700 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"><ImageIcon className="mr-2 inline h-4 w-4" />Render approved storyboard</button>
        <h2 className="pt-4 font-semibold">History</h2><div className="flex gap-2 overflow-x-auto">{history.map(url => <button key={url} onClick={() => { setPages([url]); void loadMeta(new URL(url).pathname) }} className="h-20 w-14 shrink-0 overflow-hidden rounded border"><img className="h-full w-full object-cover" src={url} /></button>)}</div>
      </section>
    </div>
    <section className="mx-auto mt-6 max-w-5xl space-y-6">{pages.length ? pages.map((url, index) => {
      const pageUrl = new URL(url).pathname; const page = meta[pageUrl]
      return <div key={url} className="relative rounded border bg-white p-3 shadow-sm"><div className="mb-3 flex justify-end gap-2"><button onClick={() => void save(url)} className="rounded border px-3 py-1 text-sm"><Save className="mr-1 inline h-4 w-4" />Save</button><button onClick={() => download(url)} className="rounded border px-3 py-1 text-sm"><Download className="mr-1 inline h-4 w-4" />Download</button></div><div className="relative"><img src={url} alt={`Generated comic page ${index + 1}`} className="w-full" />{page?.panels.map(panel => <button key={panel.id} onClick={() => openEditor(pageUrl, panel)} className="absolute rounded-full bg-gray-900 p-2 text-white" style={{ left: `${((panel.x + panel.width - 52) / page.pageWidth) * 100}%`, top: `${((panel.y + 8) / page.pageHeight) * 100}%` }} title={`Edit ${panel.id}`}><RotateCcw className="h-4 w-4" /></button>)}</div></div>
    }) : <div className="py-20 text-center text-gray-400">Your approved comic page will appear here.</div>}</section>
    {editing && <div className="fixed inset-0 flex items-center justify-center bg-black/50 p-4"><div className="w-full max-w-2xl rounded-lg bg-white p-5"><div className="mb-3 flex gap-2"><button className={`rounded px-3 py-2 text-sm ${editMode === 'panel' ? 'bg-gray-900 text-white' : 'border'}`} onClick={() => setEditMode('panel')}>Image prompt</button><button className={`rounded px-3 py-2 text-sm ${editMode === 'dialogue' ? 'bg-gray-900 text-white' : 'border'}`} onClick={() => setEditMode('dialogue')}>Dialogue only</button></div>{editMode === 'panel' ? <textarea value={panelPrompt} onChange={event => setPanelPrompt(event.target.value)} className="h-40 w-full rounded border p-3" /> : <textarea value={dialogues} onChange={event => setDialogues(event.target.value)} className="h-40 w-full rounded border p-3 font-mono text-xs" />}<div className="mt-4 flex justify-end gap-2"><button onClick={() => setEditing(null)} className="rounded border px-3 py-2 text-sm">Cancel</button><button disabled={loading} onClick={() => void regenerate()} className="rounded bg-gray-900 px-3 py-2 text-sm text-white">{editMode === 'panel' ? 'Regenerate panel' : 'Update bubbles'}</button></div></div></div>}
  </main>
}
