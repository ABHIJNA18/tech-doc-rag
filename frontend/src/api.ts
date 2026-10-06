// Talking to the backend.
//
// If VITE_API_URL is set (in frontend/.env.local, or in the Vercel
// project settings), questions are sent to `${VITE_API_URL}/api/ask`.
// If it is not set, a placeholder answer is returned, so the UI can
// be developed and demoed before the backend exists.

import type { AskResponse } from './types'

const API_URL = import.meta.env.VITE_API_URL

export async function askQuestion(question: string): Promise<AskResponse> {
  if (!API_URL) {
    return placeholderAnswer(question)
  }

  const response = await fetch(`${API_URL.replace(/\/$/, '')}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`The server answered with status ${response.status}`)
  }

  return (await response.json()) as AskResponse
}

// A fixed example answer in the real response shape. The citation is
// a real chunk from the index, so the UI shows realistic sources.
async function placeholderAnswer(question: string): Promise<AskResponse> {
  await new Promise((resolve) => setTimeout(resolve, 800)) // feels like a request

  return {
    placeholder: true,
    answer:
      `The backend is not connected yet, so this is an example answer (you asked: "${question}"). ` +
      'A 429 RATE_LIMIT_EXCEEDED error in the AI Model Hub means your contract\'s rate limit is exhausted; ' +
      'retry using exponential backoff [1]. Error responses always carry an HTTP status and a JSON body ' +
      'with an errorCode and a message [2].',
    citations: [
      {
        number: 1,
        chunkId: 'ai:ai-model-hub:error-codes:chunk:1',
        title: 'Error Codes',
        breadcrumb: ['AI Model Hub'],
        headingPath: ['Error Codes', 'API errors'],
        url: 'https://docs.ionos.com/cloud/ai/ai-model-hub/error-codes',
        text: '| 429 - RATE_LIMIT_EXCEEDED | Client | Cause: Your contract\'s rate limit is exhausted. Solution: Retry using exponential backoff. ... |',
      },
      {
        number: 2,
        chunkId: 'ai:ai-model-hub:error-codes:chunk:0',
        title: 'Error Codes',
        breadcrumb: ['AI Model Hub'],
        headingPath: ['Error Codes', 'Error format'],
        url: 'https://docs.ionos.com/cloud/ai/ai-model-hub/error-codes',
        text: 'Every error response carries an HTTP status and a JSON body listing one or more messages. Each message has a machine-readable errorCode and a human-readable message.',
      },
    ],
  }
}
