// Renders an answer and turns its [n] citation markers into small
// clickable, colour-coded buttons that highlight the matching source.

import { citationColor } from './citationColors'

interface Props {
  text: string
  onCitationClick: (number: number) => void
}

export default function AnswerText({ text, onCitationClick }: Props) {
  // "... backoff [1]. Error ..." -> ["... backoff ", "1", ". Error ..."]
  // Odd positions are the numbers inside [ ].
  const parts = text.split(/\[(\d+)\]/)

  return (
    <p className="leading-relaxed whitespace-pre-wrap">
      {parts.map((part, index) =>
        index % 2 === 1 ? (
          <button
            key={index}
            type="button"
            onClick={() => onCitationClick(Number(part))}
            className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-xs font-semibold align-text-top transition-colors ${
              citationColor(Number(part)).badge
            }`}
            title={`Show source ${part}`}
          >
            {part}
          </button>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </p>
  )
}
