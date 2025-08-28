More features you can add
Smart chunking at sentence/paragraph/token boundaries (avoid mid-word splits)
Chunk overlap for context continuity between parts
Model presets (e.g., GPT-4o mini budgets), auto-select encoding
Max-reply token budget helper and visual progress bar
Paste/upload files (txt, md, pdf) and batch splitting
Export all chunks as ZIP or Markdown bundle with ToC
Per-chunk header/footer templates; custom instruction templates
“Send mode” shortcuts for popular UIs (e.g., Conversational instructions)
Advanced safety trims (e.g., strip images/HTML, normalize whitespace)
Dark mode and responsive UX polish; keyboard shortcuts
Shareable links with prefilled settings (prompt in localStorage)
REST API for programmatic splitting; minimal CLI
Multi-language tokenization hints and right-to-left support
Ignore code fences or keep code blocks intact when splitting
Vercel/Render deploy button and health checks
Better repo name ideas
PromptChunker
TokenTailor
ChunkSmith
PromptParcel
TokenSlicer
ChunkCourier
PromptSpooler
TokenBreaker
ChunkRelay
PromptChopper
SplitGenius
PromptSlicer
TokenSplitter
ChunkForge
PromptParceler
If you want, I can rename the repo and update badges/URLs.
I implemented token counting/splitting, updated api/index.py, api/templates/index.html, and static/scripts.js, added tests, pinned Werkzeug, installed deps, ran tests (7 passed), committed and pushed, and created ARCHIVE_COMMITS.md.
Roadmap (short-term)
- Token-aware smart splitting at boundaries
- Chunk overlap slider (0–50 tokens)
- Custom per-chunk prepend/append templates (UI fields, saved to localStorage)
- Preset templates gallery (QA, Summarization, Translation, Coding)
- Export: ZIP of .txt parts and single Markdown with ToC
- File upload (.txt/.md first), drag-and-drop

Roadmap (mid-term)
- Model profiles: context window + reply budget guidance
- Shareable links (encode settings + hash into URL)
- REST API and CLI
- Sentence tokenizer fallback when tiktoken not available
- Theme (dark/light/system) and keyboard shortcuts

Roadmap (stretch)
- Multi-file project splitter with dependency-aware ordering
- Ignore/keep code fences and HTML blocks heuristics
- Auto-detect language and RTL-aware counting
- Cloud deploy template and health endpoint