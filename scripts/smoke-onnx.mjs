// Real ONNX WASM inference, without downloaded models or generated test outputs.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmBinary = await readFile(new URL('./ort-wasm-simd-threaded.wasm', import.meta.resolve('onnxruntime-web/wasm')));
import { createInferenceEngine } from '../dist/core/engine.js';
import { createONNXRuntime } from '../dist/backends/onnx.js';
import { WebInferTensor } from '../dist/core/tensor.js';

// Minimal ONNX ModelProto: Identity(input[2]) -> output[2], opset 13.
const varint = value => { const bytes = []; do { let byte = value & 127; value >>>= 7; if (value) byte |= 128; bytes.push(byte); } while (value); return bytes; };
const scalar = (field, value) => [...varint(field * 8), ...varint(value)];
const message = (field, bytes) => [...varint(field * 8 + 2), ...varint(bytes.length), ...bytes];
const string = (field, value) => message(field, [...new TextEncoder().encode(value)]);
const identity = dtype => {
  const type = message(1, [...scalar(1, dtype), ...message(2, message(1, scalar(1, 2)))]);
  const value = name => [...string(1, name), ...message(2, type)];
  const graph = [...message(1, [...string(1, 'input'), ...string(2, 'output'), ...string(4, 'Identity')]),
    ...string(2, 'identity-smoke'), ...message(11, value('input')), ...message(12, value('output'))];
  return new Uint8Array([...scalar(1, 8), ...message(7, graph), ...message(8, scalar(2, 13))]).buffer;
};
const first = createInferenceEngine({ backends: [{ type: 'wasm', create: createONNXRuntime }] });
const second = createInferenceEngine({ backends: [{ type: 'wasm', create: createONNXRuntime }] });
try {
  const floatModel = await first.loadModelFromBuffer(identity(1));
  const intModel = await second.loadModelFromBuffer(identity(7));
  const floatInput = new WebInferTensor([1.25, -2.5], [2]);
  const floatOutput = await first.runInference(floatModel, [floatInput]);
  assert.deepEqual([...floatOutput[0].data], [1.25, -2.5]);
  floatInput.dispose(); floatOutput.forEach(t => t.dispose());
  await first.dispose();
  const intInput = new WebInferTensor(new BigInt64Array([9007199254740993n, -7n]), [2], 'int64');
  const intOutput = await second.runInferenceNamed(intModel, new Map([['input', intInput]]));
  assert.equal(intOutput[0].dtype, 'int64');
  assert.deepEqual([...intOutput[0].data], [9007199254740993n, -7n]);
  intInput.dispose(); intOutput.forEach(t => t.dispose());
  console.log('PASS: real ONNX WASM float32/int64 inference and independent engine disposal');
} finally { await Promise.all([first.dispose(), second.dispose()]); }
