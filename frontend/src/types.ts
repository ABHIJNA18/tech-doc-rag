// Shared TypeScript types.
//
// AskResponse is the contract between this frontend and the Python
// backend (FastAPI, POST /api/ask). The backend will return exactly
// this shape; until it exists, src/api.ts returns a placeholder
// answer with the same shape.

// One source the answer cites, e.g. [1].
export interface Citation {
  number: number // the [n] marker used in the answer text
  chunkId: string
  title: string // page title
  breadcrumb: string[] // parent pages, e.g. ["AI Model Hub"]
  headingPath: string[] // section path on the page
  url: string // link to the IONOS documentation page
  text: string // the exact chunk text the answer drew from
}

export interface AskResponse {
  answer: string // answer text with [n] citation markers
  citations: Citation[]
  placeholder?: boolean // true while the backend is not connected
}

// One message in the chat.
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
  placeholder?: boolean
  error?: boolean
}
