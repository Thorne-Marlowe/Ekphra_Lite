import assert from 'node:assert/strict'
import test from 'node:test'
import { StoryboardRequestController } from '../src/storyboardRequest.ts'

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
