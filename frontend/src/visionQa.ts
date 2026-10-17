/** Derive a server-issued job ID from the existing completed-page URL format. */
export function jobIdFromCompletedPageUrl(url: string): string | null {
  try {
    const filename = new URL(url).pathname.split('/').pop() || ''
    const match = /^comic_page_([A-Za-z0-9][A-Za-z0-9_-]{0,99})_\d+\.png$/.exec(filename)
    return match?.[1] || null
  } catch {
    return null
  }
}

export type VisionQaRequestToken = { jobId: string; selectionId: number; requestId: number }

/**
 * Tracks independent manual QA runs and stale report reads without assuming an
 * aborted browser request can stop local model inference.
 */
export class VisionQaRequestTracker {
  private selectionId = 0
  private selectedJobId: string | null = null
  private nextRequestId = 0
  private activeRuns = new Map<string, VisionQaRequestToken>()
  private latestListId = 0
  private latestDetailId = 0

  selectComic(jobId: string | null): number {
    this.selectionId += 1
    this.selectedJobId = jobId
    this.latestListId = ++this.nextRequestId
    this.latestDetailId = ++this.nextRequestId
    return this.selectionId
  }

  currentSelectedJobId(): string | null {
    return this.selectedJobId
  }

  beginRun(jobId: string): VisionQaRequestToken | null {
    if (this.activeRuns.has(jobId)) return null
    const token = { jobId, selectionId: this.selectionId, requestId: ++this.nextRequestId }
    this.activeRuns.set(jobId, token)
    // A manual run produces a newer report than any outstanding list/detail read.
    this.latestListId = ++this.nextRequestId
    this.latestDetailId = ++this.nextRequestId
    return token
  }

  finishRun(token: VisionQaRequestToken): boolean {
    if (this.activeRuns.get(token.jobId)?.requestId !== token.requestId) return false
    this.activeRuns.delete(token.jobId)
    return true
  }

  isRunCurrent(token: VisionQaRequestToken): boolean {
    return this.activeRuns.get(token.jobId)?.requestId === token.requestId
      && this.selectionId === token.selectionId && this.selectedJobId === token.jobId
  }

  isRunActive(jobId: string | null): boolean {
    return Boolean(jobId && this.activeRuns.has(jobId))
  }

  beginList(jobId: string): VisionQaRequestToken {
    const token = { jobId, selectionId: this.selectionId, requestId: ++this.nextRequestId }
    this.latestListId = token.requestId
    return token
  }

  isListCurrent(token: VisionQaRequestToken): boolean {
    return this.latestListId === token.requestId
      && this.selectionId === token.selectionId && this.selectedJobId === token.jobId
  }

  beginDetail(jobId: string): VisionQaRequestToken {
    const token = { jobId, selectionId: this.selectionId, requestId: ++this.nextRequestId }
    this.latestDetailId = token.requestId
    return token
  }

  isDetailCurrent(token: VisionQaRequestToken): boolean {
    return this.latestDetailId === token.requestId
      && this.selectionId === token.selectionId && this.selectedJobId === token.jobId
  }
}
