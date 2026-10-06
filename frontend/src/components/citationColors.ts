// Soft pastel colours per citation number, so a [n] marker in the
// answer and its source card share a colour. Tailwind needs complete
// class names written out, so they are listed here, not built.

const PALETTE = [
  { badge: 'bg-violet-100 text-violet-700 hover:bg-violet-200', card: 'border-violet-200 bg-violet-50/60', ring: 'ring-violet-200 border-violet-400' },
  { badge: 'bg-sky-100 text-sky-700 hover:bg-sky-200', card: 'border-sky-200 bg-sky-50/60', ring: 'ring-sky-200 border-sky-400' },
  { badge: 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200', card: 'border-emerald-200 bg-emerald-50/60', ring: 'ring-emerald-200 border-emerald-400' },
  { badge: 'bg-amber-100 text-amber-700 hover:bg-amber-200', card: 'border-amber-200 bg-amber-50/60', ring: 'ring-amber-200 border-amber-400' },
  { badge: 'bg-rose-100 text-rose-700 hover:bg-rose-200', card: 'border-rose-200 bg-rose-50/60', ring: 'ring-rose-200 border-rose-400' },
]

export function citationColor(number: number) {
  return PALETTE[(number - 1) % PALETTE.length]
}
