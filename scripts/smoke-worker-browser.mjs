// Real browser Worker + ONNX WASM smoke test. Requires an installed Chrome.
import { build } from 'esbuild';
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root = fileURLToPath(new URL('../', import.meta.url));
const temp = await mkdtemp(join(tmpdir(), 'webinfer-worker-'));
const ortDist = dirname(fileURLToPath(import.meta.resolve('onnxruntime-web/wasm')));
let browser, server;
try {
  await build({ stdin: { resolveDir: root, contents: "import 'webinfer/onnx-worker';" }, bundle: true,
    format: 'esm', platform: 'browser', target: 'es2022', outfile: join(temp, 'worker.js') });
  await build({ stdin: { resolveDir: root, contents: `
    export { createInferenceEngine } from 'webinfer/core';
    export { WorkerRuntime } from 'webinfer/adapters';
    export { WebInferTensor } from 'webinfer/core';
  ` }, bundle: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: join(temp, 'client.js') });
  server = createServer(async (req, res) => {
    try {
      const name = new URL(req.url, 'http://localhost').pathname.split('/').at(-1);
      if (name === '') { res.setHeader('content-type', 'text/html'); res.end('<!doctype html><title>WebInfer worker test</title><body>Worker inference test</body>'); return; }
      if (name === 'favicon.ico') { res.writeHead(204).end(); return; }
      const wasm = name.endsWith('.wasm');
      if (!['client.js', 'worker.js', 'ort-wasm-simd-threaded.wasm'].includes(name)) { res.writeHead(404).end(); return; }
      res.setHeader('content-type', wasm ? 'application/wasm' : 'text/javascript');
      res.end(await readFile(join(wasm ? ortDist : temp, name)));
    } catch (error) { res.writeHead(500).end(String(error)); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const { createInferenceEngine, WorkerRuntime, WebInferTensor } = await import('/client.js');
    const scalar = (n, value) => [n * 8, value];
    const message = (n, bytes) => [n * 8 + 2, bytes.length, ...bytes];
    const string = (n, value) => message(n, [...new TextEncoder().encode(value)]);
    const type = message(1, [...scalar(1, 7), ...message(2, message(1, scalar(1, 2)))]);
    const value = name => [...string(1, name), ...message(2, type)];
    const graph = [...message(1, [...string(1, 'input'), ...string(2, 'output'), ...string(4, 'Identity')]),
      ...string(2, 'identity'), ...message(11, value('input')), ...message(12, value('output'))];
    const data = new Uint8Array([...scalar(1, 8), ...message(7, graph), ...message(8, scalar(2, 13))]).buffer;
    let workerCount = 0;
    const engine = createInferenceEngine({ backends: [{ type: 'wasm', create: memory => {
      workerCount++;
      return new WorkerRuntime(new Worker('/worker.js', { type: 'module' }), memory);
    }}] });
    let ticks = 0;
    const heartbeat = setInterval(() => ticks++, 1);
    try {
      const model = await engine.loadModelFromBuffer(data);
      const input = new WebInferTensor(new BigInt64Array([9007199254740993n, -7n]), [2], 'int64');
      const outputs = await engine.runInferenceNamed(model, new Map([['input', input]]));
      const result = { values: [...outputs[0].data].map(String), dtype: outputs[0].dtype, workerCount, ticks, inputIntact: input.data[0].toString() };
      outputs.forEach(t => t.dispose()); input.dispose();
      return result;
    } finally { clearInterval(heartbeat); await engine.dispose(); }
  });
  assert.deepEqual(result.values, ['9007199254740993', '-7']);
  assert.equal(result.dtype, 'int64'); assert.equal(result.workerCount, 1);
  assert.equal(result.inputIntact, '9007199254740993'); assert.ok(result.ticks > 0);
  assert.deepEqual(errors, []);
  console.log('PASS: Chrome module Worker → real ONNX WASM → int64 output; input preserved; page heartbeat active');
} finally {
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
  await rm(temp, { recursive: true, force: true });
}
