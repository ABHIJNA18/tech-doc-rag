# Frontend: Tech Doc RAG chat UI

React + TypeScript (Vite) + Tailwind CSS chat interface for the RAG
backend. Ask a question, get an answer with `[n]` citations, and see
the exact source passages with links to the IONOS documentation.

Until the backend exists, the app returns a **placeholder answer**
(marked as such in the UI).

## Run locally

```bash
cd frontend
npm install      # first time only
npm run dev      # http://localhost:5180
```

## Connect to the backend

Create `frontend/.env.local` (git-ignored):

```
VITE_API_URL=http://localhost:8000
```

Without `VITE_API_URL`, the placeholder answer is used. The API
contract (`POST /api/ask` → `AskResponse`) is defined in
`src/types.ts`.

## Structure

```
src/
  main.tsx                 entry point: renders <App />
  App.tsx                  page layout and chat state
  api.ts                   askQuestion(): backend call or placeholder
  types.ts                 shared types / API contract
  components/
    EmptyState.tsx         intro + example questions
    MessageBubble.tsx      one chat message (question or answer)
    AnswerText.tsx         answer text with clickable [n] markers
    SourceCard.tsx         one cited source (location, link, text)
    ChatInput.tsx          question box
```

## Deploy on Vercel

- Import the GitHub repository and set **Root Directory** to `frontend`.
- Vercel detects Vite: build command `npm run build`, output `dist`.
- Add the environment variable `VITE_API_URL` (the public backend URL)
  in the project settings. It is read at build time, so redeploy after
  changing it.
