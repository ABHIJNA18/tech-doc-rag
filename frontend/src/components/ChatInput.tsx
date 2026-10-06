// The question box at the bottom. Enter sends, Shift+Enter adds a
// new line. Disabled while an answer is loading.

import { useState } from 'react'
import type { KeyboardEvent } from 'react'

interface Props {
  disabled: boolean
  onSend: (question: string) => void
}

export default function ChatInput({ disabled, onSend }: Props) {
  const [text, setText] = useState('')

  function send() {
    const question = text.trim()
    if (!question || disabled) return
    onSend(question)
    setText('')
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-violet-200 bg-white/90 p-2 shadow-sm focus-within:border-violet-400 focus-within:ring-2 focus-within:ring-violet-100">
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        maxLength={1000}
        placeholder="Ask about the IONOS Cloud documentation…"
        className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-slate-800 outline-none placeholder:text-slate-400"
      />
      <button
        type="button"
        onClick={send}
        disabled={disabled || !text.trim()}
        className="rounded-xl bg-gradient-to-r from-violet-500 to-sky-500 px-4 py-2 text-sm font-medium text-white shadow-sm hover:from-violet-600 hover:to-sky-600 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300"
      >
        {disabled ? 'Thinking…' : 'Ask'}
      </button>
    </div>
  )
}
