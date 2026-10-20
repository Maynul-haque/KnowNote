import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  LocalEmbeddingBackend,
  applyEmbeddingPrefix,
  type FeatureExtractionPipeline,
  type TransformersModuleLoader
} from '../src/main/embedding/LocalEmbeddingBackend.ts'
import { LOCAL_EMBEDDING_MODEL } from '../src/main/embedding/localModel.ts'

/**
 * 本地后端要保证两件事：
 * 1. E5 的 query / document 前缀必须真的加上（不加会掉检索效果）；
 * 2. 推理时 transformers.js 不触网（allowRemoteModels=false），权重只从预置缓存读。
 */

interface Capture {
  texts: string[]
  options: Parameters<FeatureExtractionPipeline>[1] | undefined
}

function makeLoader(captures: Capture[], pipelineOptions: Record<string, unknown>[] = []) {
  const env = {
    cacheDir: null as string | null,
    localModelPath: null as string | null,
    allowRemoteModels: true,
    allowLocalModels: false,
    useFSCache: false
  }

  const loader: TransformersModuleLoader = async () => ({
    env,
    pipeline: async (_task, _model, options) => {
      pipelineOptions.push(options)
      const extractor: FeatureExtractionPipeline = async (texts, opts) => {
        captures.push({ texts, options: opts })
        return { tolist: () => texts.map(() => [0.1, 0.2, 0.3]) }
      }
      return extractor
    }
  })

  return { loader, env }
}

test('query 与 document 使用各自的 E5 前缀', () => {
  assert.equal(applyEmbeddingPrefix('hello', 'query'), 'query: hello')
  assert.equal(applyEmbeddingPrefix('hello', 'document'), 'passage: hello')
})

test('embedBatch 加前缀、传参正确并回填维度', async () => {
  const captures: Capture[] = []
  const { loader } = makeLoader(captures)
  const backend = new LocalEmbeddingBackend({
    cacheDir: '/tmp/knownote-models',
    loadTransformers: loader
  })

  const results = await backend.embedBatch(['a', 'b'], 'query')

  assert.deepEqual(captures[0].texts, ['query: a', 'query: b'])
  assert.deepEqual(captures[0].options, {
    pooling: 'mean',
    normalize: true,
    truncation: true,
    max_length: 512
  })
  assert.equal(results.length, 2)
  assert.equal(results[0].dimensions, 3)
  assert.equal(results[0].model, LOCAL_EMBEDDING_MODEL.id)
  assert.ok(results[0].embedding instanceof Float32Array)
})

test('embed(document) 使用 passage 前缀', async () => {
  const captures: Capture[] = []
  const { loader } = makeLoader(captures)
  const backend = new LocalEmbeddingBackend({
    cacheDir: '/tmp/knownote-models',
    loadTransformers: loader
  })

  await backend.embed('doc', 'document')
  assert.deepEqual(captures[0].texts, ['passage: doc'])
})

test('加载 pipeline 时关闭远端模型并指向缓存目录', async () => {
  const captures: Capture[] = []
  const pipelineOptions: Record<string, unknown>[] = []
  const { loader, env } = makeLoader(captures, pipelineOptions)
  const backend = new LocalEmbeddingBackend({
    cacheDir: '/tmp/knownote-models',
    loadTransformers: loader
  })

  await backend.embed('x', 'document')

  assert.equal(env.cacheDir, '/tmp/knownote-models')
  assert.equal(env.allowRemoteModels, false)
  assert.equal(env.allowLocalModels, true)
  // The local lookup must resolve against the cache dir too. transformers.js' own
  // default is a `models/` directory inside its package — inside app.asar once
  // packaged — where the user's weights can never be; that default is what turned
  // every model file into "Not Found" in the AppImage
  // (docs/appimage-embedding-indexing-bug.md §5.3).
  assert.equal(env.localModelPath, '/tmp/knownote-models')
  assert.equal(pipelineOptions[0].dtype, LOCAL_EMBEDDING_MODEL.dtype)
  assert.equal(pipelineOptions[0].revision, LOCAL_EMBEDDING_MODEL.revision)
})

// A bare `Not Found` is not a diagnosis: embeddingWorker forwards error.message
// verbatim into documents.error_message, so the message the loader produced is
// what the user's own database stores (docs/appimage-embedding-indexing-bug.md §4).
// This asserts the three facts the error had lost: the model, the cache directory
// and whether the weights are actually on disk.
test('模型加载失败时错误消息带上模型、缓存目录与权重文件诊断', async () => {
  const loader: TransformersModuleLoader = async () => {
    throw new Error('Not Found')
  }
  const backend = new LocalEmbeddingBackend({ cacheDir: '/tmp/x', loadTransformers: loader })

  await assert.rejects(backend.embed('x', 'document'), (error: Error) => {
    const message = String(error.message)
    assert.ok(message.includes('Not Found'), 'the original reason must survive in the message')
    assert.ok(
      message.includes(LOCAL_EMBEDDING_MODEL.id),
      'the message must name the pinned model'
    )
    assert.ok(message.includes('/tmp/x'), 'the message must name the model cache directory')
    assert.ok(
      message.includes('missing from disk'),
      'a model file that is not there must be reported as missing, not as unreadable'
    )
    return true
  })
})

test('空输入不触发 pipeline 加载', async () => {
  let loaded = false
  const loader: TransformersModuleLoader = async () => {
    loaded = true
    return {
      env: {
        cacheDir: null,
        localModelPath: null,
        allowRemoteModels: true,
        allowLocalModels: false,
        useFSCache: false
      },
      pipeline: async () => async () => ({ tolist: () => [] })
    }
  }

  const backend = new LocalEmbeddingBackend({ cacheDir: '/tmp/x', loadTransformers: loader })
  assert.deepEqual(await backend.embedBatch([], 'query'), [])
  assert.equal(loaded, false)
})
