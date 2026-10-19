#!/usr/bin/env node
/**
 * Embed texts with the built-in local model, the way the app does, and print
 * the vectors.
 *
 * This is the other half of the embedding comparison: the Rust port runs the
 * same model through ONNX Runtime directly, and the two have to produce the
 * same vectors, or the retrieval does not reproduce the baseline.
 *
 * Usage:
 *   node --experimental-transform-types scripts/embed-vectors.mjs <models-dir> <text> [text ...]
 */

import { pipeline, env } from '@huggingface/transformers'

const modelsDir = process.argv[2]
const texts = process.argv.slice(3)

if (!modelsDir || texts.length === 0) {
  console.error('usage: embed-vectors <models-dir> <text> [text ...]')
  process.exit(1)
}

env.cacheDir = modelsDir
env.allowRemoteModels = false
env.allowLocalModels = true
env.useFSCache = true

const extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', {
  dtype: 'q8',
  revision: '761b726dd34fb83930e26aab4e9ac3899aa1fa78'
})

// The documents are embedded in one batch, the way the app embeds a
// document's chunks. The questions are embedded one at a time, the way a
// query is embedded. Both matter: a padded batch and an unpadded one give
// different vectors, so comparing the two has to compare the same batching.
const batched = await extractor(
  texts.map((text) => `passage: ${text}`),
  { pooling: 'mean', normalize: true, truncation: true, max_length: 512 }
)
batched.tolist().forEach((vector, index) => {
  console.log(
    `document\t${texts[index].replace(/\n/g, ' ')}\t${vector.map((value) => value.toFixed(9)).join(',')}`
  )
})

for (const text of texts) {
  const output = await extractor([`query: ${text}`], {
    pooling: 'mean',
    normalize: true,
    truncation: true,
    max_length: 512
  })
  const vector = output.tolist()[0]
  console.log(
    `query\t${text.replace(/\n/g, ' ')}\t${vector.map((value) => value.toFixed(9)).join(',')}`
  )
}
