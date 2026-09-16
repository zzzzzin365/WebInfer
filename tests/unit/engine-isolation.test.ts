import { describe, it, expect, vi } from 'vitest';
import { createInferenceEngine } from '../../src/core/engine';
import { RuntimeManager, LoadedModelImpl } from '../../src/core/runtime';
import { InferenceScheduler } from '../../src/core/scheduler';
import { TaskScope } from '../../src/core/task-scope';
import { WebInferTensor } from '../../src/core/tensor';
import { registerAllBackends } from '../../src/backends';
import type { Runtime } from '../../src/core/types';

const deferred = <T = void>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
};
function runtime(): Runtime {
  return { name: 'wasm', capabilities: { concurrency: true, quantization: true, float16: false,
    dynamicShapes: true, maxBatchSize: 1, availableMemory: 1024 },
    initialize: vi.fn(async () => {}), isAvailable: async () => true,
    loadModel: vi.fn(async () => new LoadedModelImpl({ name: 'test', version: '1', inputs: [], outputs: [],
      sizeBytes: 1, quantization: 'float32', format: 'onnx' }, 'wasm', vi.fn())),
    run: vi.fn(async () => [new WebInferTensor([1], [1])]), dispose: vi.fn() };
}

describe('instance ownership', () => {
  it('coalesces concurrent runtime initialization and preserves custom registration', async () => {
    const manager = new RuntimeManager(), backend = runtime();
    const create = vi.fn(() => backend);
    manager.register('wasm', create);
    registerAllBackends(manager); registerAllBackends(manager);
    const results = await Promise.all([manager.getRuntime(), manager.getRuntime()]);
    expect(results).toEqual([backend, backend]);
    expect(create).toHaveBeenCalledTimes(1);
    expect(backend.initialize).toHaveBeenCalledTimes(1);
    await manager.disposeAll();
  });
  it('rejects a foreign model and disposing one engine leaves the other usable', async () => {
    const a = runtime(), b = runtime();
    const first = createInferenceEngine({ backends: [{ type: 'wasm', create: () => a }] });
    const second = createInferenceEngine({ backends: [{ type: 'wasm', create: () => b }] });
    const modelA = await first.loadModelFromBuffer(new ArrayBuffer(1));
    const modelB = await second.loadModelFromBuffer(new ArrayBuffer(1));
    await expect(first.runInference(modelB, [])).rejects.toThrow('another engine');
    await first.dispose();
    expect(modelA.isLoaded).toBe(false);
    expect(b.dispose).not.toHaveBeenCalled();
    expect((await second.runInference(modelB, []))[0].toFloat32Array()[0]).toBe(1);
    await second.dispose();
  });
  it('waits for a running inference before model/session disposal', async () => {
    const gate = deferred(), started = deferred(), backend = runtime();
    backend.run = vi.fn(async () => { started.resolve(); await gate.promise; return []; });
    const engine = createInferenceEngine({ backends: [{ type: 'wasm', create: () => backend }] });
    const model = await engine.loadModelFromBuffer(new ArrayBuffer(1));
    const inference = engine.runInference(model, []);
    await started.promise;
    const closing = engine.dispose();
    expect(model.isLoaded).toBe(true);
    expect(backend.dispose).not.toHaveBeenCalled();
    gate.resolve(); await inference; await closing;
    expect(model.isLoaded).toBe(false);
    await expect(engine.runInference(model, [])).rejects.toThrow('disposed');
  });
});

describe('global scheduling and scoped lifetimes', () => {
  it('compares priority across models without preempting a running task', async () => {
    const scheduler = new InferenceScheduler({ maxConcurrentTasks: 1 });
    const gate = deferred(), order: string[] = [];
    const first = scheduler.execute('running', async () => { order.push('running'); await gate.promise; });
    const low = scheduler.execute('background', async () => { order.push('low'); }, { priority: 'low' });
    const high = scheduler.execute('foreground', async () => { order.push('high'); }, { priority: 'high' });
    expect(order).toEqual(['running']); gate.resolve();
    await Promise.all([first, low, high]);
    expect(order).toEqual(['running', 'high', 'low']);
    expect(scheduler.getStats().totalTasks).toBe(0);
    scheduler.dispose();
  });
  it('refills a free slot without waiting for an unrelated slow model', async () => {
    const scheduler = new InferenceScheduler({ maxConcurrentTasks: 2 });
    const slow = deferred(), started = deferred();
    const first = scheduler.execute('slow', () => slow.promise);
    const second = scheduler.execute('fast', async () => {});
    const third = scheduler.execute('third', async () => { started.resolve(); });
    await started.promise;
    expect(scheduler.getStats().runningTasks).toBeGreaterThanOrEqual(1);
    slow.resolve(); await Promise.all([first, second, third]); scheduler.dispose();
  });
  it('cancels queued work, discards running output, and holds resources until completion', async () => {
    const scheduler = new InferenceScheduler({ maxConcurrentTasks: 1 });
    const scope = new TaskScope('asset:1'), gate = deferred(), started = deferred();
    const resource = { dispose: vi.fn() }, output = { dispose: vi.fn() }, queued = vi.fn();
    scope.track(resource);
    const running = scope.run(context => scheduler.execute('running', async () => {
      started.resolve(); await gate.promise; return output;
    }, context, value => value.dispose()));
    const runningCheck = expect(running).rejects.toThrow();
    await started.promise;
    const waiting = scope.run(context => scheduler.execute('queued', queued, context));
    const waitingCheck = expect(waiting).rejects.toThrow();
    await Promise.resolve();
    const closing = scope.dispose();
    expect(resource.dispose).not.toHaveBeenCalled();
    expect(scheduler.getStats().runningTasks).toBe(1);
    gate.resolve(); await Promise.all([closing, runningCheck, waitingCheck]);
    expect(queued).not.toHaveBeenCalled();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
    expect(output.dispose).toHaveBeenCalledTimes(1);
    await scope.dispose(); expect(resource.dispose).toHaveBeenCalledTimes(1);
    scheduler.dispose();
  });
});
