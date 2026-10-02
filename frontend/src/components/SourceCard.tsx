// One cited source: where it comes from (breadcrumb and section),
// a link to the documentation page, and the exact text the answer
// drew from (collapsed by default). Tinted in its citation's colour.

import { useState } from 'react'
import type { Citation } from '../types'
import { citationColor } from './citationColors'

interface Props {
  citation: Citation
  highlighted: boolean
}

export default function SourceCard({ citation, highlighted }: Props) {
  const [expanded, setExpanded] = useState(false)

  const color = citationColor(citation.number)
  const location = [...citation.breadcrumb, ...citation.headingPath].join(' › ')

  return (
    <div
      id={`source-${citation.chunkId}`}
      className={`rounded-xl border p-3 text-sm transition-all ${color.card} ${highlighted ? `ring-2 ${color.ring}` : ''}`}
    >
      <div className="flex items-start gap-2">
        <span
          className={`mt-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-xs font-semibold ${color.badge}`}
        >
          {citation.number}
        </span>
        <div className="min-w-0 flex-1">
          <a
            href={citation.url}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-slate-800 hover:text-violet-700 hover:underline"
          >
            {location}
          </a>
          <p className="truncate text-xs text-slate-400">{citation.url}</p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="mt-2 text-xs font-medium text-slate-500 hover:text-slate-800"
      >
        {expanded ? 'Hide source text' : 'Show source text'}
      </button>

      {expanded && (
        <p className="mt-2 max-h-60 overflow-y-auto rounded-lg bg-white/80 p-2 text-xs leading-relaxed whitespace-pre-wrap text-slate-600">
          {citation.text}
        </p>
      )}
    </div>
  )
}
