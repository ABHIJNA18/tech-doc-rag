// Renders an answer as Markdown (lists, `code`, **bold**, tables, ...)
// and turns its [n] citation markers into small clickable,
// colour-coded buttons that highlight the matching source.
//
// How the citation buttons work: before rendering, each [n] marker is
// rewritten into a Markdown link [n](#cite-n). The Markdown renderer
// then calls our `a` component for every link, which draws a citation
// button for #cite-n links and a normal link for everything else.
// Raw HTML in the answer is not rendered (react-markdown default).

import type { ReactNode } from 'react'
import Markdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { citationColor } from './citationColors'

const CITATION_LINK_PREFIX = '#cite-'

// Code blocks (```...```) and inline code (`...`) are left untouched,
// so a literal "[1]" inside code stays as written.
const CODE_SPANS_RE = /(```[\s\S]*?```|`[^`\n]*`)/

// [3] -> [3](#cite-3). The (?!\() skips real Markdown links like [text](url).
const CITATION_MARKER_RE = /\[(\d+)\](?!\()/g

function markCitationsAsLinks(answerText: string): string {
  return answerText
    .split(CODE_SPANS_RE)
    .map((part, index) =>
      index % 2 === 1 ? part : part.replace(CITATION_MARKER_RE, `[$1](${CITATION_LINK_PREFIX}$1)`),
    )
    .join('')
}

interface CitationButtonProps {
  citationNumber: number
  onCitationClick: (citationNumber: number) => void
}

function CitationButton({ citationNumber, onCitationClick }: CitationButtonProps) {
  return (
    <button
      type="button"
      onClick={() => onCitationClick(citationNumber)}
      className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-xs font-semibold align-text-top transition-colors ${
        citationColor(citationNumber).badge
      }`}
      title={`Show source ${citationNumber}`}
    >
      {citationNumber}
    </button>
  )
}

// How each Markdown element is drawn (Tailwind resets browser styles,
// so lists, headings and code get their look here).
function buildMarkdownComponents(onCitationClick: (citationNumber: number) => void): Components {
  return {
    a: ({ href, children }) => {
      if (href?.startsWith(CITATION_LINK_PREFIX)) {
        const citationNumber = Number(href.slice(CITATION_LINK_PREFIX.length))
        return <CitationButton citationNumber={citationNumber} onCitationClick={onCitationClick} />
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" className="text-violet-700 underline hover:text-violet-900">
          {children}
        </a>
      )
    },
    p: ({ children }) => <p className="mb-3 leading-relaxed last:mb-0">{children}</p>,
    ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-6 last:mb-0">{children}</ul>,
    ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-6 last:mb-0">{children}</ol>,
    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
    h1: ({ children }) => <h3 className="mt-3 mb-2 text-lg font-semibold first:mt-0">{children}</h3>,
    h2: ({ children }) => <h3 className="mt-3 mb-2 text-lg font-semibold first:mt-0">{children}</h3>,
    h3: ({ children }) => <h4 className="mt-3 mb-2 font-semibold first:mt-0">{children}</h4>,
    // Inline code: `RATE_LIMIT_EXCEEDED`
    code: ({ children }) => (
      <code className="rounded bg-violet-50 px-1 py-0.5 font-mono text-[0.9em] text-violet-800">{children}</code>
    ),
    // Code blocks: the <code> inside <pre> drops the inline styling.
    pre: ({ children }) => (
      <pre className="mb-3 overflow-x-auto rounded-lg bg-slate-900 p-3 text-sm text-slate-100 last:mb-0 [&_code]:bg-transparent [&_code]:p-0 [&_code]:text-inherit">
        {children}
      </pre>
    ),
    blockquote: ({ children }) => (
      <blockquote className="mb-3 border-l-4 border-violet-200 pl-3 text-slate-600 last:mb-0">{children}</blockquote>
    ),
    table: ({ children }) => (
      <div className="mb-3 overflow-x-auto last:mb-0">
        <table className="w-full border-collapse text-sm">{children}</table>
      </div>
    ),
    th: ({ children }) => (
      <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left font-semibold">{children}</th>
    ),
    td: ({ children }) => <td className="border border-slate-200 px-2 py-1 align-top">{children}</td>,
  }
}

interface AnswerTextProps {
  text: string
  onCitationClick: (citationNumber: number) => void
}

export default function AnswerText({ text, onCitationClick }: AnswerTextProps): ReactNode {
  return (
    <div className="text-slate-800">
      <Markdown remarkPlugins={[remarkGfm]} components={buildMarkdownComponents(onCitationClick)}>
        {markCitationsAsLinks(text)}
      </Markdown>
    </div>
  )
}
