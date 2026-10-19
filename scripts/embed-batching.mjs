#!/usr/bin/env node
/**
 * Does batching change the numbers?
 *
 * The app embeds a document's chunks in one batch; the Rust port embeds them
 * one at a time. If the ONNX graph picks different kernels for a batch of N
 * than for a batch of 1, the vectors differ in the last bits — and a corpus
 * with near-duplicate documents has enough near-ties for that to move a
 * ranking.
 *
 * Usage:
 *   node scripts/embed-batching.mjs <models-dir>
 */

import { pipeline, env } from '@huggingface/transformers'

const modelsDir = process.argv[2]
if (!modelsDir) {
  console.error('usage: embed-batching <models-dir>')
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

const texts = [
  'The first paragraph of the document.',
  'A second paragraph with different words.',
  'A third paragraph, close in meaning to the first.'
]

const options = { pooling: 'mean', normalize: true, truncation: true, max_length: 512 }

const individually = []
for (const text of texts) {
  const output = await extractor([`passage: ${text}`], options)
  individually.push(output.tolist()[0])
}

const batched = (await extractor(texts.map((text) => `passage: ${text}`), options)).tolist()

for (let index = 0; index < texts.length; index++) {
  const alone = individually[index]
  const together = batched[index]
  let maxDifference = 0
  for (let dimension = 0; dimension < alone.length; dimension++) {
    const difference = Math.abs(alone[dimension] - together[dimension])
    if (difference > maxDifference) maxDifference = difference
  }
  console.log(`text ${index}: max abs difference between alone and batched = ${maxDifference.toExponential(3)}`)
}
