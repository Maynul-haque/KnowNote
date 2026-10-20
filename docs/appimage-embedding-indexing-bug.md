# AppImage PDF indexing fails with "Not Found" — findings

Status: **investigation record**. This document records what was observed on a
real Linux installation, the evidence that narrows the failure to one stage, and
the changes that are proposed to fix it. It is **not** the contract; the
normative design is [DESIGN.md](../DESIGN.md).

Reported symptom: the Linux AppImage imports a PDF, the document never becomes
indexed, and the app shows the bare error **`Not Found`**.

---

## 1. Where the evidence came from

The failing installation lives on the same machine as this repository, so the
report was reproduced **from the user's own data** rather than guessed at:

| Artifact | Path |
| --- | --- |
| App profile | `/home/saikot/.config/knownote/` |
| App database | `/home/saikot/.config/knownote/knownote.db` |
| Local copies | `/home/saikot/.config/knownote/knowledge-files/` |
| Model cache | `/home/saikot/.config/knownote/models/` |
| Reported AppImage | `knownote-1.5.3-rust.1.AppImage` (repo root, 490 MB) |

All inspection was read-only (`sqlite3 "file:…?mode=ro"`, `7z l`, `asar list`).

---

## 2. What actually failed

The document row (`knownote.db`):

| Column | Value |
| --- | --- |
| `id` | `doc_1792434732710_fvy1353` |
| `title` | `Sonia thesis.pdf` |
| `status` | `failed` |
| `error_message` | `Not Found` (exactly 9 bytes — `4E6F7420466F756E64`) |
| `content` | 94,800 chars — **the PDF parsed successfully** |
| `structure` | 464,686 chars — structure built |
| `chunk_count` | `0` |
| `mime_type` | `application/pdf` |
| `local_file_path` | `…/knowledge-files/lib_1792434732887_k5nifgo.pdf` (exists, 2.6 MB) |
| `source_uri` | `/mnt/proton/Thesis/…/Sonia thesis.pdf` |

Both ingestion attempts died the same way (`ingestion_runs`):

```
run_1792434732886_g7snd28 | import  | failed | progress 20 | Not Found
run_1792434755618_2xxbc2l | reindex | failed | progress 20 | Not Found
```

### 2.1 Progress 20 is the embedding stage

`src/main/services/KnowledgeService.ts` advances the run through fixed
progress values, and `failRun` (`src/main/services/ingestion/IngestionRuns.ts`)
overwrites the stage but **keeps the progress**, so the last progress value
identifies the stage that failed:

| Progress | Stage | Line |
| --- | --- | --- |
| 0 | `copying` | `KnowledgeService.ts:598` |
| 5 | `parsing` | `KnowledgeService.ts:606` |
| 10 | `chunking` | `KnowledgeService.ts:748` |
| **20** | **`embedding`** | **`KnowledgeService.ts:761`** |
| 85 | `finalizing` | `KnowledgeService.ts:779` |

**So: copy → parse → chunk all succeeded, and the run died inside
`embedDocumentChunks()` (`KnowledgeService.ts:1260-1279`).** The PDF is fine.
The local copy is fine. The model is next.

---

## 3. What was ruled out

### 3.1 The model cache is complete and byte-exact

`src/main/embedding/localModel.ts` pins the manifest. On disk
(`models/Xenova/multilingual-e5-small/761b726dd34fb83930e26aab4e9ac3899aa1fa78/`):

| File | Manifest size | On disk |
| --- | --- | --- |
| `onnx/model_quantized.onnx` | 118,308,185 | 118,308,185 ✓ |
| `tokenizer.json` | 17,082,730 | 17,082,730 ✓ |
| `sentencepiece.bpe.model` | 5,069,051 | 5,069,051 ✓ |
| `config.json` | 658 | 658 ✓ |
| `quant_config.json` | 674 | 674 ✓ |
| `tokenizer_config.json` | 443 | 443 ✓ |
| `special_tokens_map.json` | 167 | 167 ✓ |

The revisionless aliases (`models/Xenova/multilingual-e5-small/<file>`) that
`ModelRegistry.ensureRevisionlessAliases()` creates are present too.

### 3.2 The model loads and embeds correctly outside Electron

Running the app's exact `env` configuration against the real cache:

```js
env.cacheDir = '/home/saikot/.config/knownote/models'
env.allowRemoteModels = false
env.allowLocalModels = true
env.useFSCache = true
pipeline('feature-extraction', 'Xenova/multilingual-e5-small', {
  dtype: 'q8', revision: '761b726dd34fb83930e26aab4e9ac3899aa1fa78'
})
// → RESULT OK, dims = 384
```

The cache layout matches what transformers.js looks for
(`ModelRegistry.ts:6-13` documents this: `<cacheDir>/<model_id>/<revision>/<file>`,
which is `buildResourcePaths()`'s `proposedCacheKey` when `revision !== 'main'`).

### 3.3 Native modules are packaged correctly

Inside the AppImage, `resources/app.asar.unpacked/node_modules/` contains
`onnxruntime-node` (with `bin/napi-v6/linux/x64/`), `sharp`, `@img`,
`sqlite-vec`, `sqlite-vec-linux-x64`, `better-sqlite3` — and
`resources/pdfjs/{cmaps,standard_fonts,wasm}` is present. Nothing is missing.

---

## 4. Where the string "Not Found" comes from

There is **no** `Not Found` anywhere in the application source. Searching the
built artifact (`out/main/`, including the bundled dependencies) gives exactly
one hit:

```
out/main/node_modules/@huggingface/transformers/src/utils/hub/FileResponse.js:50
out/main/node_modules/@huggingface/transformers/dist/transformers.node.mjs:6535
```

```js
// @huggingface/transformers — src/utils/hub/FileResponse.js
} else {
  this.status = 404;
  this.statusText = 'Not Found';
  this.body = null;
}
```

That branch is taken **only when `fs.existsSync(filePath)` is false** — i.e.
transformers.js was asked for a model file and could not find it.

The message survives to the database as a bare `Not Found` because the embedding
worker forwards only the message string
(`src/main/embedding/embeddingWorker.ts:72`):

```js
message: error instanceof Error ? error.message : String(error)
```

and `KnowledgeService.ingestFile()` stores it verbatim
(`KnowledgeService.ts:674-706`).

No other runtime dependency contains the string: `@huggingface/tokenizers`,
`@huggingface/jinja`, `onnxruntime-node`, `onnxruntime-common`, `sharp`,
`sqlite-vec`, `better-sqlite3`, `pdfjs-dist`, `drizzle-orm`, `jinx`, `long`,
`protobufjs`, `numjs`, `ndarray*` were all checked.

**Conclusion: the embedding worker inside the AppImage cannot see the local
model file.**

---

## 5. Why the AppImage and not the unpacked build

### 5.1 The AppImage is never verified

`.github/workflows/release.yml:359-362` builds and smoke-tests the **unpacked**
build only:

```yaml
- name: Build and smoke test the unpacked app
  run: |
    npm run build:unpack
    xvfb-run -a npm run smoke:packaged
```

`scripts/smoke-packaged.mjs` looks for `dist/*-unpacked/<exe>` and never touches
`dist/*.AppImage`. The AppImage is then built and uploaded
(`release.yml:379-383`) with **no** execution of any kind. The release
workflow's own comment says the unpacked smoke test exists because "running the
real executable is what catches a native addon or module-resolution problem that
a successful package does not reveal" — that reasoning was never applied to the
artifact users actually download on Linux.

### 5.2 The worker entry lives inside `app.asar` in the AppImage

`WorkerEmbeddingBackend.resolveWorkerPath()` (`src/main/embedding/WorkerEmbeddingBackend.ts:124-130`):

```ts
return join(__dirname, 'embeddingWorker.js')
```

`electron-builder.yml:31-43` unpacks native modules, `resources/**` and the
pdfjs assets — but **not** the built worker. Confirmed by listing the AppImage:

```
/out/main/index.js
/out/main/embeddingWorker.js        ← inside app.asar, NOT in app.asar.unpacked
```

So in the two builds the worker starts from different kinds of path:

| Build | `__dirname` | Worker entry |
| --- | --- | --- |
| `dist/linux-unpacked` | `…/resources/app/out/main` | a real file on disk |
| AppImage | `/tmp/.mount_knownote*/resources/app.asar/out/main` | a path **inside the asar archive** |

The worker then resolves `@huggingface/transformers` (an ESM package) from
inside the same archive via `await import('@huggingface/transformers')`
(`src/main/embedding/LocalEmbeddingBackend.ts:59-65`).

**This is the one code path in the embedding stack that differs between the
build CI exercises and the build users download.**

### 5.3 A latent config bug that makes the failure possible

`LocalEmbeddingBackend.createExtractor()` (`src/main/embedding/LocalEmbeddingBackend.ts:164-183`)
configures the cache but **never sets `env.localModelPath`**:

```ts
env.cacheDir = this.cacheDir
env.allowRemoteModels = false
env.allowLocalModels = true
env.useFSCache = true
```

transformers.js computes the local path as
`pathJoin(env.localModelPath, requestURL)`, and `env.localModelPath` defaults to
`<transformers.js package dir>/models/` (`@huggingface/transformers/src/env.js:161`).
That directory is inside `node_modules` — inside `app.asar` in the AppImage —
and can never contain the user's model. The app only works today because
`ModelRegistry.ensureRevisionlessAliases()` duplicates the files at
`<cacheDir>/<model_id>/<file>`, and because the `FileCache` lookup happens to
fall through to `proposedCacheKey`.

The second, decisive bug: **`env.cacheDir` is set on the `env` object of the
module instance that is loaded in the worker thread.** If the worker resolves a
*different* copy of `@huggingface/transformers` than the one the main process
configured, the two `env` objects are distinct, the cache directory is unset,
and `FileCache` is constructed with `undefined` — every model file then resolves
as missing, which is exactly the `statusText = 'Not Found'` branch.

---

## 6. Proposed fix

### 6.1 Make the error actionable (the actual user-facing bug)

A nine-byte `Not Found` is not a diagnosis. The worker should report *which*
file it could not find, and the service should say *which* model and cache
directory were in play.

- `src/main/embedding/embeddingWorker.ts` — include the model id, the cache
  directory and the missing path in the error message it posts back.
- `src/main/services/EmbeddingService.ts` — wrap the local backend call so the
  message names the pinned model (`Xenova/multilingual-e5-small` @ `761b726…`)
  and `<userData>/models`, and suggests re-downloading the model from Settings.

### 6.2 Point transformers.js at the real cache

- `src/main/embedding/LocalEmbeddingBackend.ts` — set `env.localModelPath` to
  the cache directory as well as `env.cacheDir`, so the local lookup resolves
  against `<userData>/models/<model_id>/…` instead of a package-relative
  directory that can never hold the user's model. This also makes the
  revisionless-alias workaround in `ModelRegistry` a belt-and-braces measure
  rather than the thing the whole path depends on.

### 6.3 Keep the worker thread out of the archive

- `electron-builder.yml` — add `out/main/embeddingWorker.js` to `asarUnpack`
  so `worker_threads` always gets a real file, exactly as in the unpacked
  build. Add a comment explaining why, next to the existing native-module
  notes.

### 6.4 Verify the artifact users actually download

- `.github/workflows/release.yml` — after packaging the AppImage, run the
  packaged smoke test against the AppImage itself (not only against
  `dist/linux-unpacked`), under `xvfb-run`. Extend
  `scripts/smoke-packaged.mjs` with an AppImage target, and add a smoke check
  that **loads the local embedding model in the worker thread**, because that
  is the step that is currently unverified in every packaged build.
- Make the AppImage smoke test a hard gate for the Linux job, the way the
  macOS signature verification gates its job.

### 6.5 Fix the two "not found" red herrings in the reader

- `src/main/protocol/documentProtocol.ts:49,54,66` — the 404 body is the
  literal `Not found`, which is what a developer sees in DevTools when the
  preview fails. The reader shows `readerPdfUnavailable` ("This PDF could not
  be opened. The original file may be missing."), which is a *different*
  failure from the one in this report. The two should not be confusable:
  - `getDocumentLocalFilePath()` (`KnowledgeService.ts:1466-1474`) reads only
    `documents.local_file_path` and never falls back to `sourceUri`, unlike
    `knowledgeHandlers.ts:484` (`doc.localFilePath || doc.sourceUri`). A
    document whose local copy is gone but whose original file is still on disk
    therefore 404s instead of serving the original.
  - The reader reports the same message whether the fetch 404'd or pdf.js
    itself failed to parse. The status should distinguish them.

---

## 7. Verification

1. Re-import the failing document on a build with the fix:
   ```sql
   SELECT title, status, error_message FROM documents;
   ```
   The document must reach `indexed` with `chunk_count > 0`.
2. `xvfb-run -a` the packaged smoke test against the **AppImage**, not only the
   unpacked build, and have it exercise the local embedding model in the
   worker thread.
3. Re-run the plain-Node model load from §3.2 to confirm the cache layout is
   still correct after `env.localModelPath` changes.

---

## 8. Summary

The PDF was never the problem. It parsed (94,800 characters) and chunked
successfully; the run died at progress 20, the embedding stage, with the error
`Not Found` — the bare `statusText` transformers.js assigns to a model file it
cannot find on disk. The model cache is complete and loads correctly outside
Electron, the native modules are packaged correctly, and the one code path that
differs between the CI-verified build and the downloaded AppImage is the
embedding worker thread, whose entry point and ESM dependencies are loaded
from **inside** `app.asar`. That path is never exercised by CI, and the
failure is reported to the user as an untraceable nine-byte string.
