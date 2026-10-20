export const ART_DIRECTION_PRESETS = [
  ['contemporary_western_commercial_comic', 'Contemporary Western Commercial Comic'],
  ['cinematic_graphic_novel', 'Cinematic Graphic Novel'],
  ['black_and_white_manga', 'Black-and-White Manga'],
  ['european_ligne_claire', 'European Ligne Claire'],
  ['flat_color_webcomic', 'Flat-Color Webcomic'],
  ['painterly_graphic_novel', 'Painterly Graphic Novel'],
] as const

export type ArtDirectionSelection = { presetId: string; customPrompt: string }

export function buildArtDirection(selection: ArtDirectionSelection): Record<string, string> | undefined {
  const presetId = selection.presetId.trim()
  const customPrompt = selection.customPrompt.trim()
  if (!presetId && !customPrompt) return undefined
  return {
    ...(presetId ? { preset_id: presetId } : {}),
    ...(customPrompt ? { custom_prompt: customPrompt } : {}),
  }
}

export function resolvedArtDirection(value: unknown): string | null {
  if (!value || typeof value !== 'object') return null
  const prompt = (value as { resolved_prompt?: unknown }).resolved_prompt
  return typeof prompt === 'string' && prompt.trim() ? prompt : null
}
