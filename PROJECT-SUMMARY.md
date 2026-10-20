# Project summary — the Rust rewrite

**What this is:** a handover document. It says what we are trying to do, what is already
done and proven, what is not done, and how to pick it up cold.

**Last updated:** 2026-10-19, at commit `f6d376f` on `feature/knownote-rust`.

---

## 1. What we want

Three things, in the order they were asked for:

1. **Rewrite the project in Rust.**
2. **Make it run on Linux as well as Windows.**
3. **Keep the file chunks on a hard drive (HDD) instead of the SSD.**

### The decisions behind those

**The UI stays React.** The brief said "rewrite the whole project in Rust", but the app is
Electron + React with 13,672 lines of components, a Tiptap editor, a React Flow mind map and
27 Radix primitives. Rewriting all of that into a native Rust GUI is a 2–3× larger project
and would regress the interface. Instead the **backend is Rust** and the **frontend is
reused untouched** behind a Tauri shell — Tauri uses the OS webview, so Windows and Linux
both work with no extra effort. The only TypeScript that changes is the IPC layer.

**The bulk tier is configurable, the index tier is not.** The chunks, the document blocks
and the extracted text live on a drive the user picks. The vector index, the full-text index
and the chunk *index* stay on the fast drive, because a dense query touches them on every
search and a brute-force vector scan over 150 MB on a spinning disk is seconds.

**Linux support is a CI change, not a port.** The Electron app was already ~95% Linux-ready
— `electron-builder.yml` had a `linux:` block and `verify.yml` already packaged and
smoke-tested the app on `ubuntu-latest`. The only reason there was no Linux *release* was
one line in the release workflow's matrix.

### What "done" has to mean here

A rewrite that merely *works* is not the bar. The bar is that the Rust implementation
**reproduces the original's measurements exactly** — same chunks, same blocks, same
embeddings, same retrieval results, same eval baseline. That is what the parity work
below is for, and it is the reason this rewrite is tractable at all.

---

## 2. Where the code is

| | |
|---|---|
| Local clone | `F:\knownote` |
| Upstream (read-only) | `origin` → `https://github.com/mrsibe/KnowNote.git` |
| Our fork | `fork` → `https://github.com/Maynul-haque/KnowNote.git` |
| Branch | `feature/knownote-rust` (5 commits ahead of `origin/main`) |
| Upstream `main` | `afbd1a5` — **untouched** |
| Tag | `v1.5.3-rust.1` (triggers the release workflow) |
| Fork visibility | public, a registered fork of `mrsibe/KnowNote` |

`git fetch upstream` does not exist here; use `git fetch origin` to pull upstream changes
and `git fetch fork` to sync with the fork.

### The commits

```
f6d376f  chore(release): 1.5.3-rust.1
5d81754  chore: write the parity fixture one case per line
879964c  feat: the Rust retrieval reproduces the committed eval baseline exactly
f362c6d  feat: add the Markdown loader, verified against the original
1821d61  feat: add a Rust core with a provably exact chunker, and a Linux release build
```

---

## 3. How to pick this up cold

### The toolchain this machine now has

| Tool | Version | Why it is needed |
|---|---|---|
| Rust | 1.99.1 (`x86_64-pc-windows-msvc`) | the rewrite |
| Visual Studio Build Tools 2022 | 17.14 | **the MSVC linker.** Rust on Windows cannot link without it |
| Node | 24.20.0 | the existing Electron app and the parity generator |
| Git | 2.55.0 | — |
| GitHub CLI | 2.102.0 | fork and release; token has `gist, read:org, repo, workflow` |

### Running cargo on this machine

`cargo` needs the MSVC environment, which a normal shell does not have. Use the wrapper:

```
C:\Users\Saikot\AppData\Local\Temp\opencode\cargo.cmd <cargo args>
```

It calls `VsDevCmd.bat` and adds the Windows SDK's `um\x64` and `ucrt\x64` library paths
explicitly — `VsDevCmd` alone does not find an SDK that was still being installed, and the
linker then fails with `cannot open input file 'kernel32.lib'`.

If that wrapper is gone, recreate it (contents in `C:\Users\Saikot\AppData\Local\Temp\opencode\`).

### The two local caches the verification needs

| What | Where | Size |
|---|---|---|
| The pinned embedding model | `C:\Users\Saikot\AppData\Local\Temp\opencode\models` | 140 MB |
| ONNX Runtime 1.30.0 (win-x64) | `C:\Users\Saikot\AppData\Local\Temp\opencode\onnxruntime-130\onnxruntime-win-x64-1.30.0\lib\onnxruntime.dll` | 16 MB |

The model was fetched by our own downloader (`cargo run -p knownote-embed --example
download_model -- <dir>`), which verifies every file against the manifest's SHA256. The
runtime is the official Microsoft CPU-only build — 17 MB against the 67 MB DirectML build the
Electron app currently ships.

### The verification gates

```bash
# the Rust test suite (21 tests, no model needed)
cargo.cmd test --workspace

# the parity fixture, regenerated from the TypeScript implementation
npm run chunk:parity

# the baseline gate - needs the model and the runtime
KNOWNOTE_EVAL_MODEL_CACHE=<models dir> \
KNOWNOTE_ONNXRUNTIME=<onnxruntime.dll> \
  cargo.cmd test -p knownote-eval --test baseline -- --ignored

# the retrieval harness on its own
cargo.cmd run -p knownote-eval -- \
  --corpus eval/corpus --questions eval/questions.jsonl --splits eval/splits.json \
  --models <models dir> --runtime <onnxruntime.dll>

# the existing Electron app, untouched
npm test && npm run typecheck && npm run check:design
```

---

## 4. What is done, and how it was proven

**7,234 lines of Rust across 6 crates, 35 files. 21 tests passing, zero warnings.**

| Crate | Lines | What it holds |
|---|---|---|
| `knownote-core` | 2,533 | the chunker, the block builder, the retrieval contract, the fusion, the text helpers. Pure — no I/O, no Electron, no SQLite |
| `knownote-store` | 1,815 | the tiered store: the schema, the layout, the write path, the retrievers |
| `knownote-eval` | 1,413 | the harness, the metrics, the baseline gate |
| `knownote-embed` | 890 | the pinned model, the downloader, the ONNX Runtime backend, the remote backend |
| `knownote-ingest` | 458 | the loaders. Markdown only so far |
| `knownote-embedding` | 125 | the embedding contract — the trait and the space identity |

### 4.1 The chunker is byte-identical to the original

`scripts/gen-chunk-parity.mjs` runs the **TypeScript** chunker over the 21-document eval
corpus and 12 adversarial inputs, across eight option sets, and writes the result to
`crates/knownote-core/tests/fixtures/chunk-parity.json`. The Rust test replays it and
asserts identical output.

**33 cases, 8,409 chunks — identical.** The inputs include CJK, mixed CJK/Latin,
astral-plane characters, code blocks, tables, empty and whitespace-only input, long single
lines, repeated blank lines and trailing whitespace.

### 4.2 The block builder matches

Same fixture, same comparison: the same blocks, with the same kind, page, level, text and
offsets.

### 4.3 The Markdown loader matches

Same canonical text, same title, same section tree — levels, titles and offsets — for every
corpus document.

### 4.4 The retrieval reproduces the committed eval baseline exactly

`docs/eval/baseline-v1.6.json`, all nine metrics, to six decimals:

| Metric | Rust | Baseline |
|---|---|---|
| Recall@1 | 0.594340 | 0.59434 |
| Recall@5 | 0.877358 | 0.877358 |
| Recall@10 | 0.919811 | 0.919811 |
| MRR | 0.754755 | 0.754755 |
| nDCG@10 | 0.780449 | 0.780449 |
| Hit rate@5 | 0.905660 | 0.90566 |
| MAP@10 | 0.728040 | 0.72804 |
| Context precision@3 | 0.308176 | 0.308176 |
| Context recall@3 | 0.816038 | 0.816038 |

**All 78 questions match individually** — the same passages retrieved, the same ranks, the
same blocks matched. Same 21 documents, 53 chunks. Every query class matches.

### 4.5 The tiered store works

Two tiers, split by *how often something is read* rather than by what it is about:

```
fast drive   knownote.db      workspace: notebooks, sessions, messages, notes, items
                             + chunk_ref  (the chunk index: document, offsets, pages, tokens)
                             + chunks_fts (the FTS5 index)
                             + vec_<notebook> (one vec0 table per notebook)
your drive   content.db       chunk_text, document_blocks, chunk_blocks,
                             document_content, library_source_content, sources/
```

A dense query never reads chunk text: it embeds the question, scans the vectors, and only
then fetches the text of the handful of chunks that survived. Six tests cover it, including
a move of the bulk tier to another drive.

### 4.6 Linux support is written, and has now been exercised

- A `release-linux` job in `release.yml`: build → package → smoke test the packaged app on
  Linux → upload AppImage and deb to the same draft release.
- It **does not gate publication** — a Linux-only failure must not hold the Windows and
  macOS users hostage — but the failed job is still visible in the run, and a soft check
  names any missing platform artifact.
- `fail-fast: false` on the existing matrix, so a cancelled sibling cannot leave the draft
  half-populated.
- The publish target follows the workflow's own repository, so the same file works in a
  fork and upstream.
- `README.md` and `README_CN.md` document the Linux artifacts; `PRODUCT.md` updated.
- **The first run proved the app itself works on Linux**: the packaged app was built and
  the smoke test passed all its checks on `ubuntu-latest`. The only thing that failed was
  the upload, for the reason in §6.9.

### 4.7 Three pre-existing CI breakages were found and fixed

See §6.

---

## 5. What is not done

Ordered by how much it unblocks.

| # | What | Notes |
|---|---|---|
| 1 | **The remaining loaders** — PDF, DOCX, PPTX, HTML | PDF is the app's most important input format. `pdfium-render` needs a C++ toolchain, which this machine now has. DOCX/PPTX are unzip + `quick-xml`; HTML is `scraper` + readability. |
| 2 | **The LLM protocol adapters** | OpenAI Completions, OpenAI Responses, Anthropic Messages, Google Generative AI, and streaming. `rig` or raw `reqwest`. The chat turn lifecycle and the citation resolver come with it. |
| 3 | **The Tauri shell** | `src-tauri/`, one Rust command per IPC channel, and a `window.api` shim that keeps the renderer's shape identical so the 13,672 lines of React keep working untouched. |
| 4 | **The MCP server** | `rmcp` is a direct port: 5 read-only tools, same provenance shape. |
| 5 | **The settings screen for the content drive** | The store supports it; the UI to pick the path and migrate does not exist. |
| 6 | **The .apkg export** | `anki-apkg-export` is a JS library. A `.apkg` is a ZIP with an SQLite `collection.anki2` plus media — reimplementable with `rusqlite` + `zip`. |
| 7 | **Pruning the package** | The Linux build ships ONNX Runtime for every platform plus a 260 MB CUDA provider bundle. `docs/dependency-audit.md` names exactly which files to drop. Would roughly halve the download — but it is a cross-platform packaging change with a real risk of shipping a broken app, so it needs its own change with its own rollback. |

### The next task

**The PDF loader.** It is the last big parser and the most important input format. It needs
`pdfium-render`, which compiles PDFium from source — a long first build.

Note the one thing the PDF loader must preserve: the page boundaries and the character
offsets. A PDF's blocks carry a page number and a normalised bounding box, and that is what
lets a citation point at *page 7, paragraph 3* rather than at a chunk. The
`multipage.pdf` fixture in `test/fixtures/` is the test, and the parity fixture should
grow a PDF case the same way the Markdown one did.

---

## 6. Traps

The things that cost time. They are all still true, and each one will bite again.

1. **Electron 44 dropped its `postinstall`.** It exposes the download as an `install-electron`
   binary instead, so `npm ci` no longer fetches the runtime — and electron-builder needs
   `node_modules/electron/dist` to package, and the smoke test needs it to run. A clean
   checkout failed at `build:unpack`. Fixed with a root `postinstall`.

2. **npm 11 blocks dependency install scripts by default.** `better-sqlite3`,
   `onnxruntime-node`, `esbuild`, `protobufjs` and `electron-winstaller` now need approval.
   The approvals are in `package.json` under `allowScripts`. Without them the CI fails on a
   clean machine.

3. **GitHub's OAuth token needs the `workflow` scope** to modify `.github/workflows/*`. The
   first push was rejected with *"refusing to allow an OAuth App to create or update
   workflow"* until `gh auth refresh -h github.com -s workflow` was run.

4. **A batched embedding is not the same as an unpadded one.** The difference is around
   `1e-2` — not float noise — and it moved the ranking of 12 of the 78 questions. The app
   embeds a document's chunks in *one batch*, so the port has to as well; embedding them one
   at a time produces a **different index from the same document**. The tokenizer needs
   `PaddingStrategy::BatchLongest` and `TruncationStrategy::LongestFirst` set explicitly.

5. **The `sqlite-vec` extension must be registered before any connection is opened.**
   SQLite's auto-extension only applies to connections opened afterwards. Registering it
   after `Connection::open` gives you *"no such module: vec0"*.

6. **The SQL identifier whitelist is strict.** `vec_<notebookId>` is interpolated into SQL,
   so the notebook id has to match `^[A-Za-z_][A-Za-z0-9_]*$` — a hyphen is not allowed, and
   SQLite would not quote it. The eval notebook is therefore `evalnotebook`.

7. **The version guard has two modes.** On a tag push it is in *tag mode* and the tag must
   equal `v{package.json version}` exactly. `1.5.3-rust.1` is valid and sorts below
   `1.5.3`, which is why the fork uses a prerelease: it can never collide with upstream's
   own version line.

8. **The parity fixture must stay one case per line.** Pretty-printing put every number on
   its own line — 166,360 lines for 33 cases. Git stores that, but every regeneration
   produced a diff of the same size. One case per line: 39 lines, and a changed case is one
   changed line.

9. **electron-builder's publish target must follow the workflow's repository.**
   `electron-builder.yml` names `owner: MrSibe`, which is right there and wrong in a fork.
   The first Linux run *built both artifacts* and then failed on the **upload** with a 403,
   because the token has no write access to somebody else's repository. The workflow now
   passes `--config.publish.owner="$PUBLISH_OWNER" --config.publish.repo="$PUBLISH_REPO"`,
   where those come from `github.repository_owner` and `github.event.repository.name` — so
   the same file publishes to the fork in a fork and to upstream in upstream.

   electron-builder supports dot-notation config overrides on the command line
   (`--config.publish.owner=…`); they merge over `electron-builder.yml` rather than
   replacing it. That is what makes the override possible without a second config file.

10. **A bare `$NAME` in a `run:` block is not an environment variable on Windows.**
    The Windows runner uses PowerShell, where `$NAME` is a PowerShell variable; the
    environment is `$env:NAME`. The Linux job published to the fork and the Windows job
    failed with *"the variable '$PUBLISH_OWNER' cannot be retrieved because it has not
    been set"*. Use `${{ env.NAME }}` — the workflow substitutes it before any shell
    sees it, so the command reads the same on bash and on PowerShell. A form that works
    on exactly one platform is the hardest kind of failure to notice.

### Releasing from the fork

```bash
npm version 1.5.3-rust.2 --no-git-tag-version   # package.json and package-lock.json
git commit -am "chore(release): 1.5.3-rust.2"
git push fork feature/knownote-rust
git tag -a v1.5.3-rust.2 -m "KnowNote 1.5.3-rust.2"
git push fork v1.5.3-rust.2
```

The version guard runs in *tag mode* on a tag push and requires the tag to equal
`v{package.json version}` exactly. A prerelease suffix is valid and sorts below the
release, which is why the fork uses one: it can never collide with upstream's own version
line.

The artifacts land on a **draft** release in the fork, and the `publish` job promotes it to
public once every platform has built. The Linux job deliberately does not gate that
promotion.

---

## 7. The invariants that must not regress

Carried over from the original repository, and asserted by the tests:

1. `chunk.content` and `block.text` are always exactly `content[start..end]` of the
   canonical document text.
2. Re-indexing rebuilds the derived layer in place; it never creates a second
   `documents` row, so a citation that stored a `documentId` stays valid.
3. One `vec0` table per notebook, each carrying its own width — vectors from two embedding
   spaces are never compared.
4. A citation resolves to a source location (page + character span), not to a chunk.
5. The renderer never sees a filesystem path; bytes are served by document id.
6. Provenance travels with the retrieval result; hydration is batch.
7. The MCP surface has no write tool, no model-calling tool, and no result that reports a
   filesystem path.
8. Say what is not there. A claim ahead of the code is a bug — and so is a stale "not here
   yet" line in the README.

---

## 8. One deliberate deviation from the original

**Offsets are bytes, not UTF-16 code units.** A JavaScript string is indexed by UTF-16 unit
and a Rust string by byte. The port keeps the *size* comparisons in UTF-16 so the chunk
boundaries are the original's, and only the stored numbers change unit. The chunk
*contents* are therefore identical — which is what the parity test asserts.

Consequences:

- A database migration is mechanical and checkable: for every chunk, the old UTF-16 range
  and the new byte range must select the same characters. That is an assertion, not a hope.
- Citations that already point at a page and a character span keep pointing there, because
  the characters did not move.

There is also one **bug in the original** that the port does not reproduce: the original
computes a nested section's `content` by slicing the parent's content with the child's
absolute offsets, which mixes two coordinate systems and comes back empty. The offsets,
which are the part the pipeline reads, were correct in both.

---

## 9. Where the plan lives

`C:\Users\Saikot\.opencode\plan\knownote-rust-rewrite.md` — the full plan, with the phase
breakdown, the crate-by-crate design and the reasoning behind the tier split.

The phases:

| Phase | State |
|---|---|
| 0. Linux release build | **written; a tagged run was triggered** |
| 1. Rust core, loaders, embedding, retrieval, eval | **chunking, blocks, Markdown, embedding and retrieval done and proven**; the other loaders, the LLM adapters and the MCP server are not |
| 2. Tauri shell | not started |
| 3. Tiered storage | **the store, the split and the retrievers are done and tested**; the settings screen and the migration command are not |
| 4. MCP and packaging | not started |
