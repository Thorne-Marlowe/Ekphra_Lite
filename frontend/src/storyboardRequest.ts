export type StartedStoryboardRequest = { id: number; controller: AbortController }

/** Tracks the one cancellable storyboard request owned by the composer UI. */
export class StoryboardRequestController {
  private currentId = 0
  private controller: AbortController | null = null

  start(): StartedStoryboardRequest {
    this.controller?.abort()
    this.currentId += 1
    this.controller = new AbortController()
    return { id: this.currentId, controller: this.controller }
  }

  isCurrent(id: number): boolean {
    return id === this.currentId
  }

  finish(id: number): void {
    if (this.isCurrent(id)) this.controller = null
  }

  cancel(): boolean {
    if (!this.controller) return false
    // Invalidate before aborting so an already-resolved fetch is stale.
    this.currentId += 1
    this.controller.abort()
    this.controller = null
    return true
  }
}
