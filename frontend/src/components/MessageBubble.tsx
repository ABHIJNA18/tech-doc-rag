// One chat message. User questions are right-aligned bubbles;
// assistant answers show the answer text with clickable citations
// and the list of sources below it.

import { useState } from 'react'
import type { ChatMessage } from '../types'
import AnswerText from './AnswerText'
import SourceCard from './SourceCard'

interface Props {
  message: ChatMessage
}

export default function MessageBubble({ message }: Props) {
  // Which source is highlighted after clicking a [n] marker.
  const [activeSource, setActiveSource] = useState<number | null>(null)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-gradient-to-br from-violet-500 to-sky-500 px-4 py-2.5 text-white whitespace-pre-wrap shadow-sm">
          {message.text}
        </div>
      </div>
    )
  }

  const citations = message.citations ?? []

  function showSource(number: number) {
    setActiveSource(number)
    const citation = citations.find((c) => c.number === number)
    if (citation) {
      document.getElementById(`source-${citation.chunkId}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }

  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[85%] rounded-2xl rounded-bl-sm border px-4 py-3 ${
          message.error ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-violet-100 bg-white/90 text-slate-800 shadow-sm'
        }`}
      >
        {message.placeholder && (
          <span className="mb-2 inline-block rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
            Placeholder · backend not connected yet
          </span>
        )}

        <AnswerText text={message.text} onCitationClick={showSource} />

        {citations.length > 0 && (
          <div className="mt-4">
            <p className="mb-2 text-xs font-semibold tracking-wide text-violet-500 uppercase">Sources</p>
            <div className="space-y-2">
              {citations.map((citation) => (
                <SourceCard
                  key={citation.chunkId}
                  citation={citation}
                  highlighted={activeSource === citation.number}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
