// The whole chat page: header, message list, question box.
//
// State lives here: the list of messages and whether an answer is
// loading. Sending a question adds the user message, calls the API
// (src/api.ts) and adds the assistant's answer (or an error).

import { useEffect, useRef, useState } from 'react'
import { askQuestion } from './api'
import ChatInput from './components/ChatInput'
import EmptyState from './components/EmptyState'
import MessageBubble from './components/MessageBubble'
import type { ChatMessage } from './types'

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSend(question: string) {
    setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: 'user', text: question }])
    setLoading(true)

    try {
      const response = await askQuestion(question)
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: response.answer,
          citations: response.citations,
          meta: response.meta,
          placeholder: response.placeholder,
        },
      ])
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: `Sorry, something went wrong: ${error instanceof Error ? error.message : 'unknown error'}`,
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-dvh flex-col bg-gradient-to-br from-violet-50 via-sky-50 to-emerald-50">
      <header className="border-b border-violet-100 bg-white/70 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-violet-400 via-sky-400 to-emerald-300 text-xl text-white shadow-sm">✦</div>
            <div>
            <h1 className="text-xl font-bold text-slate-900">Tech Doc RAG</h1>
            <p className="text-sm text-slate-500">Question answering over the IONOS Cloud docs</p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => setMessages([])}
              className="rounded-lg px-2 py-1 text-sm text-violet-600 hover:bg-violet-50"
            >
              New chat
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-4 px-4 py-6">
          {messages.length === 0 && <EmptyState onPick={handleSend} />}

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-sm border border-violet-100 bg-white/90 px-4 py-3 text-violet-400 shadow-sm animate-pulse">
                Searching the docs…
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-violet-100 bg-white/50 backdrop-blur">
        <div className="mx-auto max-w-4xl px-4 py-3">
          <ChatInput disabled={loading} onSend={handleSend} />
          <p className="mt-2 text-center text-xs text-slate-400">
            Unofficial portfolio project. Answers may be wrong; check the cited sources.
          </p>
        </div>
      </footer>
    </div>
  )
}
