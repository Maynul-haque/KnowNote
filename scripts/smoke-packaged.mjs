#!/usr/bin/env node
/**
 * Runs the packaged application's `--smoke-test` and propagates its exit code.
 *
 * `npm run build` and `npm run build:unpack` both succeed on an artifact that
 * cannot start - that is exactly how `Cannot find module 'script-loader!sql.js'`
 * and `ReferenceError: DOMMatrix is not defined` reached a package. This script
 * closes that gap by running the real executable, which means the checks run
 * against the real app.asar, the real native addons and the real module
 * resolution.
 *
 * Usage:
 *   npm run build:unpack && npm run smoke:packaged
 *   npx electron-builder --linux AppImage --publish never && npm run smoke:appimage
 *
 * Modes:
 *   default        runs the unpacked build electron-builder produced, as before.
 *   --appimage     instead runs the `dist/*.AppImage` artifact users actually
 *                  download. The AppImage is extracted with its built-in
 *                  `--appimage-extract` (no FUSE involved) and the extracted
 *                  AppRun is launched; app.asar and app.asar.unpacked boundaries
 *                  are exactly the packaged ones. Linux only: the unpacked gate
 *                  already covers the other platforms.
 *
 * Env:
 *   SMOKE_TEST_TIMEOUT_MS  how long to wait for the app to finish
 *                          (default 120s, 300s when the model cache is set)
 *   SMOKE_MODEL_CACHE      path to the pinned embedding model as downloaded by
 *                          `node scripts/eval.mjs --prepare`. When set, the
 *                          model is copied into the throwaway profile and the
 *                          smoke test runs the worker-thread embedding check.
 *                          --appimage refuses to run without it: a gate that
 *                          skips its strongest check cannot be a gate.
 */

import { spawn } from 'node:child_process'
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

const DIST_DIR = resolve('dist')
const APPIMAGE_FLAG = '--appimage'
const MODEL_SOURCE = process.env.SMOKE_MODEL_CACHE

// Loading the real model inside a worker thread is the slowest check (ONNX cold
// start on any disk), so model-mode runs get a proportionate default timeout.
const DEFAULT_TIMEOUT_MS = MODEL_SOURCE ? 300_000 : 120_000
const TIMEOUT_MS = Number(process.env.SMOKE_TEST_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS)

const fail = (message) => {
  console.error(`[smoke:packaged] ${message}`)
  process.exit(1)
}

/** electron-builder.yml is a flat file for the keys we need; avoid a YAML dep. */
function readYamlScalar(file, key) {
  const match = new RegExp(`^${key}:\\s*(.+?)\\s*$`, 'm').exec(readFileSync(file, 'utf8'))
  return match?.[1]
}

/**
 * @returns {string} the `name` field of the package, used for the on-disk binary
 */
function readAppName() {
  const file = resolve('package.json')
  try {
    return JSON.parse(readFileSync(file, 'utf8')).name
  } catch (error) {
    console.error(`[smoke:packaged] cannot read ${file}: ${error.message}`)
    process.exit(1)
  }
}

/**
 * Locates the unpacked executable electron-builder produced for this platform.
 *
 * Mirrors electron-builder's layout: the on-disk name is `name` from
 * package.json for Linux/Windows (it defaults `executableName` to it, and
 * electron-builder.yml sets it explicitly to the same value) and `productName`
 * for the macOS bundle.
 *
 * @returns {string[]} candidate executable paths, most likely first
 */
function findExecutable() {
  const name = readAppName()
  const productName = readYamlScalar(resolve('electron-builder.yml'), 'productName') ?? name
  const roots = existsSync(DIST_DIR)
    ? readdirSync(DIST_DIR)
        .map((entry) => join(DIST_DIR, entry))
        .filter((path) => statSync(path).isDirectory())
    : []

  if (process.platform === 'darwin') {
    const binary = join('Contents', 'MacOS', productName)
    return roots
      .filter((dir) => /[\\/]mac/.test(dir))
      .map((dir) => join(dir, `${productName}.app`, binary))
  }

  if (process.platform === 'win32') {
    return roots.filter((dir) => dir.endsWith('-unpacked')).map((dir) => join(dir, `${name}.exe`))
  }

  return roots.filter((dir) => dir.endsWith('-unpacked')).map((dir) => join(dir, name))
}

/**
 * The revision this checkout's `src/main/embedding/localModel.ts` pins, read from
 * its source so the manifest stays the single source of truth. A smoke gate that
 * provisioned a *stale* revision would fail every run and nobody would trust it.
 */
function readPinnedRevision() {
  const source = readFileSync(resolve('src/main/embedding/localModel.ts'), 'utf8')
  const match = /revision:\s*'([^']+)'/.exec(source)
  return match?.[1]
}

/** dist/*.AppImage paths, listing nothing when no package ran yet. */
function findAppImages() {
  return existsSync(DIST_DIR)
    ? readdirSync(DIST_DIR)
        .filter((entry) => entry.endsWith('.AppImage'))
        .map((entry) => join(DIST_DIR, entry))
    : []
}

function spawnCapture(command, args, cwd) {
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, { cwd, stdio: 'ignore' })
    child.on('error', (error) => {
      console.error(`[smoke:packaged] could not run ${command}: ${error.message}`)
      resolvePromise(false)
    })
    child.on('exit', (code) => resolvePromise(code === 0))
  })
}

const useAppImage = process.argv.includes(APPIMAGE_FLAG)
const candidates = useAppImage ? [] : findExecutable()

let scratchDir
let executable

if (useAppImage) {
  if (process.platform !== 'linux') {
    fail(
      'the AppImage is the Linux release artifact; run the unpacked gate (`npm run smoke:packaged`) on this platform instead'
    )
  }

  if (!MODEL_SOURCE) {
    fail('the AppImage gate must exercise the embedding worker, which needs SMOKE_MODEL_CACHE')
  }

  const images = findAppImages()
  if (images.length === 0) {
    fail('no dist/*.AppImage found - package it first: npx electron-builder --linux AppImage --publish never')
  }
  if (images.length > 1) {
    fail(`dist/ holds more than one AppImage, refusing to guess:\n${images.map((path) => `- ${path}`).join('\n')}`)
  }

  // The AppImage runtime can extract itself with no FUSE mounting involved,
  // which is both more deterministic on runners and closer to how the artifact
  // is inspected after the fact (docs/appimage-embedding-indexing-bug.md §1).
  scratchDir = mkdtempSync(join(tmpdir(), 'knownote-smoke-appimage-'))
  console.log(`[smoke:packaged] extracting ${images[0]} with --appimage-extract`)
  const extracted = await spawnCapture(images[0], ['--appimage-extract'], scratchDir)
  const appRun = join(scratchDir, 'squashfs-root', 'AppRun')
  if (!extracted || !existsSync(appRun)) {
    fail(`--appimage-extract did not produce a runnable AppRun under ${scratchDir}`)
  }
  executable = appRun
} else {
  const unpackedExecutable = candidates.find((candidate) => existsSync(candidate))
  if (!unpackedExecutable) {
    console.error('[smoke:packaged] could not find an unpacked build of the app.')
    console.error('[smoke:packaged] looked for:')
    for (const candidate of candidates.length ? candidates : ['(nothing - is dist/ missing?)']) {
      console.error(`[smoke:packaged]   ${candidate}`)
    }
    console.error('[smoke:packaged] run `npm run build:unpack` first.')
    process.exit(1)
  }
  executable = unpackedExecutable
}

console.log(`[smoke:packaged] launching ${executable}`)

// Run against a throwaway profile: the app opens and migrates a real SQLite
// database on startup, and a smoke test must never touch a developer's own
// ~/.config/knownote.
const dataDir = mkdtempSync(join(tmpdir(), 'knownote-smoke-'))

const args = [
  '--smoke-test',
  `--user-data-dir=${dataDir}`,
  // The document-import checks read these. They live in the repo rather than in
  // the package, so point the packaged app at them explicitly.
  `--smoke-fixtures=${resolve('test/fixtures')}`
]

if (MODEL_SOURCE) {
  const revision = readPinnedRevision()
  const primaryWeights = revision
    ? join(MODEL_SOURCE, 'Xenova', 'multilingual-e5-small', revision, 'onnx', 'model_quantized.onnx')
    : null
  if (!revision || !existsSync(primaryWeights)) {
    fail(
      `SMOKE_MODEL_CACHE (${MODEL_SOURCE}) does not hold the pinned model` +
        `${revision ? ` (${revision})` : ''}; provision it first: node scripts/eval.mjs --prepare`
    )
  }
  // Copy, never point at: a smoke test must not mutate the cache (transformers.js
  // writes unresolved partials and aliases into the directory it reads).
  cpSync(MODEL_SOURCE, join(dataDir, 'models'), { recursive: true })
  args.push(`--smoke-model-cache=${join(dataDir, 'models')}`)
  console.log(`[smoke:packaged] model check enabled from ${MODEL_SOURCE}`)
}

// The Chromium sandbox needs either a setuid helper or user namespaces, and
// neither is dependable on CI runners: containers run as root, and the unpacked
// chrome-sandbox is not setuid. The smoke test only loads our own code, so this
// is safe there - and it keeps `smoke:packaged` working on plain `ubuntu-latest`.
const isRoot = typeof process.getuid === 'function' && process.getuid() === 0
if (isRoot || process.env.CI) {
  args.push('--no-sandbox')
}

const cleanup = () => {
  rmSync(dataDir, { recursive: true, force: true })
  // The extracted AppImage is close to 1 GB; leaving it in /tmp after every
  // dispatch run would eat runner disk across attempts.
  if (scratchDir) {
    rmSync(scratchDir, { recursive: true, force: true })
  }
}

const child = spawn(executable, args, {
  stdio: 'inherit',
  env: { ...process.env, ELECTRON_DISABLE_SECURITY_WARNINGS: '1' }
})

const timeout = setTimeout(() => {
  console.error(`[smoke:packaged] timed out after ${TIMEOUT_MS}ms, killing the app`)
  child.kill('SIGKILL')
}, TIMEOUT_MS)

child.on('error', (error) => {
  clearTimeout(timeout)
  cleanup()
  console.error('[smoke:packaged] failed to launch:', error.message)
  process.exit(1)
})

child.on('exit', (code, signal) => {
  clearTimeout(timeout)
  cleanup()

  if (signal) {
    console.error(`[smoke:packaged] app was killed by ${signal}`)
    process.exit(1)
  }

  if (code !== 0) {
    console.error(`[smoke:packaged] FAIL - app exited with code ${code}`)
    process.exit(code ?? 1)
  }

  console.log('[smoke:packaged] PASS - packaged app started and self-checked')
  process.exit(0)
})
