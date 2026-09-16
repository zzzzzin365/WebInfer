import { describe, it, expect, vi, beforeEach } from 'vitest';
const ort = vi.hoisted(() => ({ sessions: [] as any[], tensors: [] as any[], gate: undefined as Promise<void> | undefined }));
vi.mock('onnxruntime-web/wasm', () => ({
  env: {},
  Tensor: class {
    dispose = vi.fn();
    constructor(public type: string, public data: any, public dims: number[]) { ort.tensors.push(this); }
  },
  InferenceSession: { create: vi.fn(async () => {
    const session = { inputNames: ['input'], outputNames: ['output'], release: vi.fn(async () => {}),
      run: vi.fn(async (feeds: any) => {
        await ort.gate;
        const output = { type: feeds.input.type, data: feeds.input.data.slice(), dims: feeds.input.dims, dispose: vi.fn() };
        ort.tensors.push(output);
        return { output };
      }) };
    ort.sessions.push(session); return session;
  }) },
}));
import { ONNXRuntime } from '../../src/backends/onnx';
import { WorkerRuntime } from '../../src/adapters/worker-runtime';
import { createONNXWorkerHandler } from '../../src/adapters/onnx-worker-handler';
import { WebInferTensor } from '../../src/core/tensor';
import { serializeTensor, deserializeTensor } from '../../src/core/worker';
import { createInferenceEngine } from '../../src/core/engine';

beforeEach(() => { ort.sessions.length = 0; ort.tensors.length = 0; ort.gate = undefined; });

describe('ONNX resource ownership', () => {
  it('releases only its own sessions and preserves int64 output without precision loss', async () => {
    const a = new ONNXRuntime(), b = new ONNXRuntime();
    const ma = await a.loadModel(new ArrayBuffer(1)), mb = await b.loadModel(new ArrayBuffer(1));
    const input = new WebInferTensor(new BigInt64Array([9007199254740993n]), [1], 'int64');
    const output = await b.run(mb, [input]);
    expect(output[0].data[0]).toBe(9007199254740993n);
    expect(output[0].dtype).toBe('int64');
    expect(ort.tensors.every(t => t.dispose.mock.calls.length === 1)).toBe(true);
    ma.dispose(); await a.dispose();
    expect(ort.sessions[0].release).toHaveBeenCalledTimes(1);
    expect(ort.sessions[1].release).not.toHaveBeenCalled();
    expect(mb.isLoaded).toBe(true);
    mb.dispose(); await b.dispose(); expect(ort.sessions[1].release).toHaveBeenCalledTimes(1);
  });
  it('defers session.release until in-flight inference settles', async () => {
    let finish!: () => void;
    ort.gate = new Promise<void>(r => { finish = r; });
    const runtime = new ONNXRuntime(), model = await runtime.loadModel(new ArrayBuffer(1));
    const pending = runtime.run(model, [new WebInferTensor([1], [1])]);
    model.dispose(); const closing = runtime.dispose();
    await Promise.resolve(); expect(ort.sessions[0].release).not.toHaveBeenCalled();
    finish(); await pending; await closing;
    expect(ort.sessions[0].release).toHaveBeenCalledTimes(1);
  });
  it('preserves typed array byte offsets in worker transfers', async () => {
    for (const data of [new BigInt64Array([0n, 9007199254740993n, 2n]).subarray(1, 2), new Int32Array([1, -5, 3]).subarray(1, 2), new Uint8Array([1,255,3]).subarray(1,2)]) {
      const dtype = data instanceof BigInt64Array ? 'int64' : data instanceof Int32Array ? 'int32' : 'uint8';
      const original = new WebInferTensor(data, [1], dtype);
      const decoded = await deserializeTensor(serializeTensor(original));
      expect(decoded.data[0]).toBe(data[0]); expect(decoded.dtype).toBe(dtype);
      expect(original.data.byteLength).toBe(data.byteLength);
    }
  });
});

describe('worker runtime protocol', () => {
  it('loads, runs named inputs and disposes through the actual worker handler', async () => {
    const worker = { onmessage: undefined as any, onerror: undefined as any, onmessageerror: undefined as any,
      terminate: vi.fn(), postMessage: (message: any) => { void handle(message); } };
    const handle = createONNXWorkerHandler(response => queueMicrotask(() => worker.onmessage({ data: response })));
    const engine = createInferenceEngine({ backends: [{ type: 'wasm', create: memory => new WorkerRuntime(worker as any, memory) }] });
    const model = await engine.loadModelFromBuffer(new ArrayBuffer(1));
    const output = await engine.runInferenceNamed(model, new Map([['input', new WebInferTensor(new BigInt64Array([42n]), [1], 'int64')]]));
    expect(output[0].data[0]).toBe(42n);
    await engine.dispose();
    expect(ort.sessions[0].release).toHaveBeenCalledTimes(1);
    expect(worker.terminate).toHaveBeenCalledTimes(1);
  });
  it('rejects pending requests on a worker crash', async () => {
    const worker = { onmessage: undefined as any, onerror: undefined as any, onmessageerror: undefined as any, postMessage: vi.fn(), terminate: vi.fn() };
    const runtime = new WorkerRuntime(worker as any);
    const init = runtime.initialize();
    const check = expect(init).rejects.toThrow('crashed');
    worker.onerror(); await check;
    await runtime.dispose();
  });
});
