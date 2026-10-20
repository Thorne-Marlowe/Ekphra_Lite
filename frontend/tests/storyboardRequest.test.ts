import assert from 'node:assert/strict'
import test from 'node:test'
import { StoryboardRequestController } from '../src/storyboardRequest.ts'
import { jobIdFromCompletedPageUrl, VisionQaRequestTracker } from '../src/visionQa.ts'
import { buildCreativeIntent, buildStoryboardRequest, effectiveCreativeIntentEntries } from '../src/creativeIntent.ts'

test('creative controls default to an omitted intent and preserve explicit selections', () => {
  assert.equal(buildCreativeIntent({ genre: '', tone: '', narrativeResolution: 'auto' }), undefined)
  assert.deepEqual(buildCreativeIntent({ genre: 'mystery', tone: '', narrativeResolution: 'auto' }), { genre: 'mystery' })
  assert.deepEqual(buildCreativeIntent({ genre: '', tone: 'serious', narrativeResolution: 'auto' }), { tone: 'serious' })
  assert.deepEqual(buildCreativeIntent({ genre: '', tone: '', narrativeResolution: 'cliffhanger' }), { narrative_resolution: 'cliffhanger' })
  assert.deepEqual(buildCreativeIntent({ genre: '', tone: 'suspenseful', narrativeResolution: 'revelation' }), {
    tone: 'suspenseful', narrative_resolution: 'revelation',
  })
  assert.deepEqual(buildStoryboardRequest('A quiet scene.', 2, { genre: '', tone: '', narrativeResolution: 'auto' }), {
    prompt: 'A quiet scene.', pageCount: 2,
  })
})

test('effective creative intent display tolerates missing and partial storyboard metadata', () => {
  assert.deepEqual(effectiveCreativeIntentEntries(undefined), [])
  assert.deepEqual(effectiveCreativeIntentEntries({ effective: { genre: 'mystery', narrative_resolution: 'revelation' }, source: {
    genre: 'user_selection', narrative_resolution: 'planner_inference',
  } }), [
    { label: 'Genre', value: 'mystery', source: 'user_selection' },
    { label: 'Resolution', value: 'revelation', source: 'planner_inference' },
  ])
})

test('cancellation aborts the active request and invalidates its result', () => {
  const requests = new StoryboardRequestController()
  const active = requests.start()

  assert.equal(requests.cancel(), true)
  assert.equal(active.controller.signal.aborted, true)
  assert.equal(requests.isCurrent(active.id), false)
})

test('a stale storyboard response cannot replace a newer request', () => {
  const requests = new StoryboardRequestController()
  const first = requests.start()
  const second = requests.start()

  assert.equal(first.controller.signal.aborted, true)
  assert.equal(requests.isCurrent(first.id), false)
  assert.equal(requests.isCurrent(second.id), true)
})

test('vision QA derives only a safe job ID from a completed comic page URL', () => {
  assert.equal(jobIdFromCompletedPageUrl('http://127.0.0.1:8000/outputs/comic_page_job_alpha_1.png'), 'job_alpha')
  assert.equal(jobIdFromCompletedPageUrl('http://127.0.0.1:8000/outputs/comic_page_../escape_1.png'), null)
  assert.equal(jobIdFromCompletedPageUrl('http://127.0.0.1:8000/outputs/other.png'), null)
})

test('Comic A completing cannot clear Comic B active QA state', () => {
  const requests = new VisionQaRequestTracker()
  requests.selectComic('comic_a')
  const runA = requests.beginRun('comic_a')!
  requests.selectComic('comic_b')
  const runB = requests.beginRun('comic_b')!

  assert.equal(requests.isRunCurrent(runA), false)
  assert.equal(requests.finishRun(runA), true)
  assert.equal(requests.isRunActive('comic_b'), true)
  assert.equal(requests.isRunCurrent(runB), true)
})

test('an older report-list response is stale after a manual QA run starts', () => {
  const requests = new VisionQaRequestTracker()
  requests.selectComic('comic_a')
  const olderList = requests.beginList('comic_a')
  requests.beginRun('comic_a')

  assert.equal(requests.isListCurrent(olderList), false)
})

test('an older report-detail response is stale after a newer report selection', () => {
  const requests = new VisionQaRequestTracker()
  requests.selectComic('comic_a')
  const olderDetail = requests.beginDetail('comic_a')
  const newerDetail = requests.beginDetail('comic_a')

  assert.equal(requests.isDetailCurrent(olderDetail), false)
  assert.equal(requests.isDetailCurrent(newerDetail), true)
})

test('duplicate manual QA runs for the same comic are refused', () => {
  const requests = new VisionQaRequestTracker()
  requests.selectComic('comic_a')
  const first = requests.beginRun('comic_a')

  assert.ok(first)
  assert.equal(requests.beginRun('comic_a'), null)
  assert.equal(requests.isRunActive('comic_a'), true)
})

test('switching comics invalidates stale QA findings and report reads', () => {
  const requests = new VisionQaRequestTracker()
  requests.selectComic('comic_a')
  const run = requests.beginRun('comic_a')!
  const list = requests.beginList('comic_a')
  const detail = requests.beginDetail('comic_a')
  requests.selectComic('comic_b')

  assert.equal(requests.isRunCurrent(run), false)
  assert.equal(requests.isListCurrent(list), false)
  assert.equal(requests.isDetailCurrent(detail), false)
})
