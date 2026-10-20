# Handover for Jules

Read `PROJECT-SUMMARY.md` first — it is the cold-start document. `DESIGN.md` holds the
design; `docs/eval/baseline-v1.6.json` is the retrieval baseline. This file only says what
an agent needs to know before touching the code.

## What this is

A rewrite of KnowNote onto a Rust backend. The React UI is reused unchanged; the plan is a
Tauri shell using the OS webview, with the Electron app staying functional until the shell
lands. The rewrite's bar is not "it works" — it is **the same measurements as the original**:
same chunks, same blocks, same embeddings, same retrieval results.

The branch you are handed is `main` of `Maynul-haque/KnowNote`, a fork. Upstream is
`mrsibe/KnowNote`; `main` there is untouched and must stay untouched. Do not push upstream.

## Where the work stands

| Area | State |
|---|---|
| Chunker, block builder, Markdown loader, PDF loader | done, parity-fixture proven |
| Retrieval + eval harness | done, reproduces the committed baseline exactly |
| Tiered store (fast index / bulk content drive) | done and tested |
| DOCX, PPTX, HTML loaders | **not done** — unzip + `quick-xml`; `scraper` + readability |
| LLM protocol adapters, streaming, citation resolver | **not done** |
| Tauri shell (`src-tauri/` + `window.api` shim) | **not done** |
| MCP server | **not done** |
| Content-drive settings UI + migration command | **not done** |
| `.apkg` export, package pruning | **not done** |

## Commands (Linux)

```bash
cargo test --workspace                # always-on suite, no model or PDFium needed
cargo test --workspace -- --ignored   # needs the env vars below
npm ci
npm test && npm run typecheck && npm run check:design
npm run chunk:parity                  # regenerates the fixture from the TypeScript chunker
```

`scripts/jules-setup.sh` installs Rust and Node 24 and warms both caches. Run it in the
environment-setup field, not as part of your change.

### The gates that need local files

Two suites are `#[ignore]`d because they need artifacts that are not in the repository:

| Gate | Needs |
|---|---|
| `knownote-eval --test baseline` | `KNOWNOTE_EVAL_MODEL_CACHE=<dir with the pinned model>` and `KNOWNOTE_ONNXRUNTIME=<path to onnxruntime>` |
| `pdf_loader_parity` (the pin replay) | `KNOWNOTE_PDFIUM=<directory holding the PDFium shared library>` |

Without them, run `cargo test --workspace`. **CI is the authority** for those gates —
`verify.yml` provisions the model, and the packaged-app gate runs it inside the AppImage.
Do not weaken or delete an ignored test to make a local run green.

## Invariants — do not regress

1. `chunk.content` and `block.text` are exactly `content[start..end]` of the canonical text.
2. Re-indexing rebuilds derived data in place; a stored `documentId` keeps resolving.
3. One `vec0` table per notebook; vectors from two embedding spaces are never compared.
4. A citation resolves to a page + character span, not to a chunk.
5. The renderer never sees a filesystem path; bytes are served by document id.
6. Provenance travels with retrieval results; hydration is batched.
7. The MCP surface has no write tool and no path-bearing result.
8. Say what is not there. A stale "not done yet" line is as much a bug as a false claim.
9. Offsets are stored in **bytes**, not UTF-16 units, while chunk boundaries keep the
   original's UTF-16 size comparisons. Do not "fix" one into the other.

## Ground rules

- **Parity fixtures are pins, not snapshots.** `crates/knownote-core/tests/fixtures/*.json`
  are generated from the TypeScript implementation; a diff means the port changed, and the
  loaded output is what is wrong. Regenerate only with the documented script.
- **Never commit** `target/`, `node_modules/`, `out/`, `dist/`, `build/`, or `*.AppImage`
  (they are ignored — keep it that way).
- `unsafe_code` is forbidden workspace-wide. The PDFium high-level API is safe.
- Embed a document's chunks in **one batch**. One-at-a-time embedding produces a different
  index (see the trap in `PROJECT-SUMMARY.md` §6, item 4).
- **Do not tag a release.** `release.yml` publishes to this fork and will build and upload
  Windows, macOS and Linux artifacts. Tagging is a human decision.
- Keep commits in the existing style: `type(scope): lowercase summary`, body explaining what
  and how it was proven.
- If a test can only pass on one platform, say so in the commit; a form that works on exactly
  one machine is the failure mode this project keeps hitting.

## Good first task

The DOCX/PPTX/HTML loaders are the next unblocked piece: `crates/knownote-ingest/` has the
Markdown and PDF loaders to copy from, and `scripts/gen-chunk-parity.mjs` is how a new loader
gets its TypeScript-equivalence fixture. PDF is the worked example of the full pattern.
