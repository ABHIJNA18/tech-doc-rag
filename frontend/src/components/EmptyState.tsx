// Shown before the first question: what the app does and a few
// example questions to click.

const EXAMPLES = [
  { question: 'What does a 429 error mean in the AI Model Hub?', color: 'border-violet-200 bg-violet-50 hover:bg-violet-100' },
  { question: 'What is the maximum number of replicas in an In-Memory DB replica set?', color: 'border-sky-200 bg-sky-50 hover:bg-sky-100' },
  { question: 'What happens if I forget my backup encryption password?', color: 'border-emerald-200 bg-emerald-50 hover:bg-emerald-100' },
  { question: 'What is ACPI?', color: 'border-amber-200 bg-amber-50 hover:bg-amber-100' },
]

interface Props {
  onPick: (question: string) => void
}

export default function EmptyState({ onPick }: Props) {
  return (
    <div className="mx-auto mt-12 max-w-2xl text-center">
      <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-400 via-sky-400 to-emerald-300 text-3xl shadow-sm">✦</div>
      <h2 className="text-3xl font-bold text-slate-800">Ask the documentation</h2>
      <p className="mt-3 text-lg text-slate-500">
        Answers are generated from the public IONOS Cloud documentation, with citations pointing to the exact
        source passages.
      </p>
      <div className="mt-6 grid gap-2 sm:grid-cols-2">
        {EXAMPLES.map(({ question, color }) => (
          <button
            key={question}
            type="button"
            onClick={() => onPick(question)}
            className={`rounded-xl border px-3 py-2.5 text-left text-sm text-slate-700 transition-colors ${color}`}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}
