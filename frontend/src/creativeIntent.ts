export const GENRE_OPTIONS = [
  ['comedy', 'Comedy'], ['drama', 'Drama'], ['mystery', 'Mystery'], ['romance', 'Romance'],
  ['slice of life', 'Slice of Life'], ['horror', 'Horror'], ['thriller', 'Thriller'],
  ['fantasy', 'Fantasy'], ['science fiction', 'Science Fiction'], ['action', 'Action'], ['adventure', 'Adventure'],
] as const

export const TONE_OPTIONS = [
  ['lighthearted', 'Lighthearted'], ['serious', 'Serious'], ['suspenseful', 'Suspenseful'], ['dark', 'Dark'],
  ['whimsical', 'Whimsical'], ['melancholic', 'Melancholic'], ['hopeful', 'Hopeful'], ['romantic', 'Romantic'],
  ['reflective', 'Reflective'], ['tense', 'Tense'],
] as const

export const RESOLUTION_OPTIONS = [
  ['auto', 'Auto'], ['punchline', 'Punchline'], ['cliffhanger', 'Cliffhanger'], ['revelation', 'Revelation'],
  ['transition', 'Transition'], ['emotional_beat', 'Emotional Beat'], ['deliberate_nonresolution', 'Deliberate Nonresolution'],
] as const

type Resolution = (typeof RESOLUTION_OPTIONS)[number][0]

export type CreativeIntentSelection = {
  genre: string
  tone: string
  narrativeResolution: Resolution
}

export type EffectiveCreativeIntent = {
  effective?: Record<string, unknown>
  source?: Record<string, unknown>
}

export function buildCreativeIntent(selection: CreativeIntentSelection): Record<string, string> | undefined {
  const intent: Record<string, string> = {}
  const genre = selection.genre.trim()
  const tone = selection.tone.trim()
  if (genre) intent.genre = genre
  if (tone) intent.tone = tone
  if (selection.narrativeResolution !== 'auto') intent.narrative_resolution = selection.narrativeResolution
  return Object.keys(intent).length ? intent : undefined
}

export function buildStoryboardRequest(
  prompt: string,
  pageCount: number,
  selection: CreativeIntentSelection,
): { prompt: string; pageCount: number; creativeIntent?: Record<string, string> } {
  const creativeIntent = buildCreativeIntent(selection)
  return { prompt, pageCount, ...(creativeIntent ? { creativeIntent } : {}) }
}

export function effectiveCreativeIntentEntries(value: unknown): Array<{ label: string; value: string; source?: string }> {
  if (!value || typeof value !== 'object') return []
  const intent = value as EffectiveCreativeIntent
  if (!intent.effective || typeof intent.effective !== 'object') return []
  const labels: Record<string, string> = { genre: 'Genre', tone: 'Tone', narrative_resolution: 'Resolution' }
  return Object.entries(intent.effective).flatMap(([key, rawValue]) => {
    if (!(key in labels) || typeof rawValue !== 'string' || !rawValue.trim()) return []
    const source = intent.source && typeof intent.source[key] === 'string' ? intent.source[key] : undefined
    return [{ label: labels[key], value: rawValue, source }]
  })
}
