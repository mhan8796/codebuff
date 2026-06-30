import { describe, expect, it } from 'bun:test'

import { OpenAICompatibleChatLanguageModel } from './openai-compatible-chat-language-model'

import type { LanguageModelV2StreamPart } from '@ai-sdk/provider'
import type { FetchFunction } from '@ai-sdk/provider-utils'

/** Build a fake SSE Response the way `createEventSourceResponseHandler` expects:
 *  each event is one `data: <json>\n\n` line, finished with `data: [DONE]\n\n`. */
function sseResponse(events: object[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(event)}\n\n`),
        )
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  })
}

async function readAllParts(
  stream: ReadableStream<LanguageModelV2StreamPart>,
): Promise<LanguageModelV2StreamPart[]> {
  const reader = stream.getReader()
  const parts: LanguageModelV2StreamPart[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  return parts
}

/** Build a `typeof fetch`-shaped mock by attaching a no-op `preconnect`.
 *  Bun's `FetchFunction` resolves to `typeof globalThis.fetch`, whose TypeScript
 *  type also exposes static members like `preconnect` — a plain async function
 *  doesn't satisfy that shape under strict TS, hence the shim. */
function mockFetch(events: object[]): FetchFunction {
  const stub = ((..._args: Parameters<typeof fetch>) =>
    Promise.resolve(sseResponse(events))) as typeof fetch
  return Object.assign(stub, { preconnect: () => {} }) as FetchFunction
}

function makeModel(fetch: FetchFunction) {
  return new OpenAICompatibleChatLanguageModel('test-model', {
    provider: 'test.chat',
    url: ({ modelId, path }) =>
      `https://example.test${path}?model=${encodeURIComponent(modelId)}`,
    headers: () => ({}),
    fetch,
  })
}

describe('OpenAICompatibleChatLanguageModel.doStream', () => {
  it('emits text-delta stream parts and a finish part for valid chunks', async () => {
    const model = makeModel(
      mockFetch([
        {
          id: 'chunk-1',
          model: 'test-model',
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chunk-2',
          model: 'test-model',
          choices: [
            {
              index: 0,
              delta: { content: ' world' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chunk-3',
          model: 'test-model',
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
          },
        },
      ]),
    )
    const { stream } = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    })

    const parts = await readAllParts(stream)

    // First emitted stream part must be a stream-start.
    expect(parts[0].type).toBe('stream-start')

    // Concatenate every text-delta — recovers the full text from a stream split
    // across two chunks of delta.content.
    const textDeltas = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'text-delta' }> =>
        p.type === 'text-delta',
    )
    expect(textDeltas.map((p) => p.delta).join('')).toBe('Hello world')

    // The finish part must carry the OpenAI "stop" reason and the parsed usage.
    const finishes = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'finish' }> =>
        p.type === 'finish',
    )
    expect(finishes.length).toBe(1)
    expect(finishes[0].finishReason).toBe('stop')
    expect(finishes[0].usage.outputTokens).toBe(2)
    expect(finishes[0].usage.inputTokens).toBe(1)
    expect(finishes[0].usage.totalTokens).toBe(3)
  })

  it('emits an error stream part when the server sends an error chunk', async () => {
    const model = makeModel(
      mockFetch([
        {
          error: {
            message: 'rate limited',
            type: 'rate_limit_error',
            code: 'rate_limit',
          },
        },
      ]),
    )
    const { stream } = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    })

    const parts = await readAllParts(stream)

    // The transform branch `'error' in value` should forward the message into
    // an `{ type: 'error', error: <message> }` stream part and set finishReason
    // to 'error'. This is the second arm of the chunk-schema union whose
    // type safety the original "MUST FIX" TODO was about.
    const errorParts = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'error' }> =>
        p.type === 'error',
    )
    expect(errorParts.length).toBe(1)
    expect(errorParts[0].error).toBe('rate limited')

    const finishes = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'finish' }> =>
        p.type === 'finish',
    )
    expect(finishes.length).toBe(1)
    expect(finishes[0].finishReason).toBe('error')
  })

  it('emits tool-call stream parts when the server sends a streaming tool call', async () => {
    const model = makeModel(
      mockFetch([
        {
          id: 'chunk-1',
          model: 'test-model',
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_1',
                    function: { name: 'search', arguments: '{"q":' },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chunk-2',
          model: 'test-model',
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: { arguments: '"codebuff"}' },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        },
      ]),
    )
    const { stream } = await model.doStream({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    })

    const parts = await readAllParts(stream)

    // Two-part streaming arguments get merged into a single tool-call part
    // when the second chunk closes the JSON argument.
    const toolInputStarts = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'tool-input-start' }> =>
        p.type === 'tool-input-start',
    )
    expect(toolInputStarts.length).toBe(1)
    expect(toolInputStarts[0].toolName).toBe('search')

    const toolCalls = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'tool-call' }> =>
        p.type === 'tool-call',
    )
    expect(toolCalls.length).toBe(1)
    expect(toolCalls[0].toolName).toBe('search')
    expect(toolCalls[0].input).toBe('{"q":"codebuff"}')

    // OpenAI's wire-level 'tool_calls' is mapped to AI SDK's 'tool-calls'.
    const finishes = parts.filter(
      (p): p is Extract<LanguageModelV2StreamPart, { type: 'finish' }> =>
        p.type === 'finish',
    )
    expect(finishes.length).toBe(1)
    expect(finishes[0].finishReason).toBe('tool-calls')
  })
})
