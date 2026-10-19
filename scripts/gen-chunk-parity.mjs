#!/usr/bin/env node
/**
 * Dump the TypeScript chunker's output, so the Rust port can be compared to it
 * directly instead of being taken on trust.
 *
 * A port of a chunker is easy to get *nearly* right. The rules that decide
 * where a chunk starts and ends are subtle: a heading is atomic, a block is a
 * boundary, a page breaks a chunk, the overlap aligns to unit starts. A
 * difference in any one of them changes the retrieval results, and nothing in
 * the test suite would notice - the chunker still satisfies its own contract
 * while quietly answering a different question.
 *
 * So this script runs the real TypeScript implementation over the real corpus
 * and the awkward cases, and writes the result to
 * `crates/knownote-core/tests/fixtures/chunk-parity.json`. The Rust test
 * `crates/knownote-core/tests/parity.rs` reads that file and asserts the port
 * produces byte-identical output.
 *
 * Offsets are written twice: the UTF-16 code-unit offsets the TypeScript
 * implementation produces, and the byte offsets the Rust port produces. They
 * are the same positions in the same characters; only the unit differs, and
 * the byte form is what the port stores.
 *
 * Usage:
 *   node --experimental-transform-types --import ./test/ts-resolve.mjs \
 *     scripts/gen-chunk-parity.mjs
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { ChunkingService } from '../src/main/services/ChunkingService.ts'
import {
  assignBlockIds,
  buildDocumentBlocks
} from '../src/main/services/blocks/documentBlocks.ts'
import { MarkdownLoader } from '../src/main/services/loaders/MarkdownLoader.ts'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const OUT = join(ROOT, 'crates', 'knownote-core', 'tests', 'fixtures', 'chunk-parity.json')

/** The byte offset of a UTF-16 code-unit index into `text`. */
function byteOffsetOf(text, utf16Index) {
  return Buffer.byteLength(text.slice(0, utf16Index), 'utf8')
}

/** The option sets every case is run with. */
const OPTION_SETS = [
  { name: 'default', options: {} },
  { name: 'small', options: { chunkSize: 40, chunkOverlap: 5 } },
  { name: 'tiny', options: { chunkSize: 20, chunkOverlap: 5 } },
  { name: 'overlap-heavy', options: { chunkSize: 120, chunkOverlap: 80 } },
  { name: 'no-overlap', options: { chunkSize: 200, chunkOverlap: 0 } },
  { name: 'respect-headings', options: { chunkSize: 200, chunkOverlap: 20, respectHeadings: true } },
  { name: 'span-pages', options: { chunkSize: 2000, allowSpanPages: true } },
  { name: 'min-chunk', options: { chunkSize: 50, minChunkSize: 500 } }
]

/** The awkward inputs, alongside the corpus. */
const SYNTHETIC = [
  { name: 'empty', content: '' },
  { name: 'whitespace-only', content: '   \n\n  \t ' },
  { name: 'single-char', content: 'a' },
  { name: 'no-trailing-newline', content: '# Heading\n\nBody text.' },
  { name: 'cjk', content: '第一段中文内容。\n\n第二段中文内容，包含标点。\n\n第三段。' },
  {
    name: 'mixed-cjk-latin',
    content: '# 标题 Mixed\n\n这是中文和 English 混排的段落。\n\nAnother paragraph, in English only.'
  },
  {
    name: 'astral',
    content: '# Emoji 🌍 heading\n\nA paragraph with an astral character 🧪 inside it.\n\nAnother one 🚀 here.'
  },
  {
    name: 'code-and-table',
    content:
      '# Code\n\n```js\nconst x = 1\n  const y = 2\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nAfter the table.'
  },
  {
    name: 'long-single-line',
    content: 'word '.repeat(400)
  },
  {
    name: 'no-blank-lines',
    content: 'Line one.\nLine two.\nLine three.\nLine four.'
  },
  {
    name: 'repeated-blank-lines',
    content: 'First.\n\n\n\nSecond.\n\n\n\nThird.'
  },
  {
    name: 'trailing-whitespace',
    content: '# Heading   \n\nParagraph with trailing spaces   \n\nAnother one\t\n'
  }
]

/** Serialise a section tree, recording the offsets in both units. */
function serialiseSections(sections, content) {
  return sections.map((section) => ({
    level: section.level,
    title: section.title,
    // UTF-16 offsets, as the TypeScript loader produces them.
    startOffsetUtf16: section.startOffset,
    // Byte offsets, which is what the Rust port stores.
    startOffset: byteOffsetOf(content, section.startOffset),
    children: serialiseSections(section.children ?? [], content)
  }))
}

/** Serialise the chunks of one run. */
function serialiseChunks(content, chunks, blocksById) {
  return chunks.map((chunk) => ({
    index: chunk.index,
    content: chunk.content,
    // UTF-16 offsets, as the TypeScript implementation produces them.
    startOffsetUtf16: chunk.startOffset,
    endOffsetUtf16: chunk.endOffset,
    // Byte offsets, which is what the Rust port stores.
    startOffset: byteOffsetOf(content, chunk.startOffset),
    endOffset: byteOffsetOf(content, chunk.endOffset),
    tokenCount: chunk.tokenCount,
    pageStart: chunk.pageStart,
    pageEnd: chunk.pageEnd,
    blockSpans: chunk.blockSpans.map((span) => {
      const blockText = blocksById.get(span.blockId) ?? ''
      return {
        blockId: span.blockId,
        startInBlockUtf16: span.startInBlock,
        endInBlockUtf16: span.endInBlock,
        startInBlock: byteOffsetOf(blockText, span.startInBlock),
        endInBlock: byteOffsetOf(blockText, span.endInBlock)
      }
    })
  }))
}

async function main() {
  const service = new ChunkingService()
  const cases = []

  // The corpus: the documents the committed eval baseline was measured on.
  const corpusDir = join(ROOT, 'eval', 'corpus')
  const loader = new MarkdownLoader()

  for (const file of readdirSync(corpusDir).filter((name) => name.endsWith('.md')).sort()) {
    const path = join(corpusDir, file)
    const raw = readFileSync(path, 'utf8')
    const result = await loader.loadFromPath(path)
    const blocks = assignBlockIds(
      'doc_parity',
      buildDocumentBlocks({ content: result.content, structure: result.structure })
    )

    cases.push({
      name: `corpus/${file}`,
      content: result.content,
      // What the loader produced, so the Rust loader can be compared against
      // this one rather than only against the chunker that consumes it.
      loader: {
        source: raw,
        content: result.content,
        title: result.title ?? null,
        sections: serialiseSections(result.structure?.sections ?? [], result.content)
      },
      blocks: blocks.map((block) => ({
        id: block.id,
        kind: block.kind,
        page: block.page,
        level: block.level,
        text: block.text,
        startOffset: byteOffsetOf(result.content, block.startOffset),
        endOffset: byteOffsetOf(result.content, block.endOffset)
      })),
      runs: OPTION_SETS.map(({ name, options }) => ({
        mode: 'blocks',
        name,
        options,
        chunks: serialiseChunks(
          result.content,
          service.chunkBlocks(result.content, blocks, options),
          new Map(blocks.map((block) => [block.id, block.text]))
        )
      }))
    })
  }

  // The synthetic cases. Some of them have no blocks at all, which is what
  // exercises the character-window fallback.
  for (const synthetic of SYNTHETIC) {
    const blocks = assignBlockIds(
      'doc_parity',
      buildDocumentBlocks({ content: synthetic.content, structure: { type: 'flat' } })
    )

    cases.push({
      name: `synthetic/${synthetic.name}`,
      content: synthetic.content,
      blocks: blocks.map((block) => ({
        id: block.id,
        kind: block.kind,
        page: block.page,
        level: block.level,
        text: block.text,
        startOffset: byteOffsetOf(synthetic.content, block.startOffset),
        endOffset: byteOffsetOf(synthetic.content, block.endOffset)
      })),
      runs: OPTION_SETS.map(({ name, options }) => ({
        mode: 'blocks',
        name,
        options,
        chunks: serialiseChunks(
          synthetic.content,
          service.chunkBlocks(synthetic.content, blocks, options),
          new Map(blocks.map((block) => [block.id, block.text]))
        )
      })).concat(
        OPTION_SETS.map(({ name, options }) => ({
          mode: 'fallback',
          name,
          options,
          chunks: serialiseChunks(
            synthetic.content,
            service.chunk(synthetic.content, options),
            new Map()
          )
        }))
      )
    })
  }

  const payload = {
    generatedBy: 'scripts/gen-chunk-parity.mjs',
    note:
      'Offsets are the UTF-16 code-unit offsets the TypeScript chunker produces ' +
      'and the byte offsets the Rust port stores. Both describe the same position ' +
      'in the same characters.',
    cases
  }

  mkdirSync(join(OUT, '..'), { recursive: true })
  // One case per line.
  //
  // The file is several megabytes, and pretty-printing it put every number on
  // its own line: 166,000 lines for 33 cases. Git stores that fine, but every
  // regeneration produced a diff of the same size, which makes the branch
  // unreviewable and the history noise.
  //
  // This is still one valid JSON document - whitespace between array elements
  // is allowed - so nothing has to change on the reading side. A case that
  // changed is now one changed line.
  const body = [
    '{',
    `  "generatedBy": ${JSON.stringify(payload.generatedBy)},`,
    `  "note": ${JSON.stringify(payload.note)},`,
    '  "cases": [',
    payload.cases.map((entry) => `    ${JSON.stringify(entry)}`).join(',\n'),
    '  ]',
    '}',
    ''
  ].join('\n')

  writeFileSync(OUT, body, 'utf8')

  const chunkCount = cases.reduce(
    (total, entry) =>
      total + entry.runs.reduce((sum, run) => sum + run.chunks.length, 0),
    0
  )
  console.log(
    `[parity] wrote ${cases.length} cases, ${chunkCount} chunks -> ${OUT}`
  )
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
