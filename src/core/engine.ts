import { RuntimeManager } from './runtime.js';
import { InferenceScheduler } from './scheduler.js';
import { MemoryManager } from './memory.js';
import { throwIfAborted, type ExecutionContext, type InferenceClient, type LoadOptions } from './inference-client.js';
import type { LoadedModel, Runtime, RuntimeType, SchedulerOptions, Tensor } from './types.js';

export interface EngineOptions {
  backends: Array<{ type: RuntimeType; create: (memory: MemoryManager) => Runtime }>;
  scheduler?: SchedulerOptions;
}

/** Owns runtimes and models. Callers own inference input/output tensors. */
export class InferenceEngine implements InferenceClient {
  readonly runtimes = new RuntimeManager();
  readonly scheduler: InferenceScheduler;
  readonly memory = new MemoryManager();
  private readonly models = new Map<LoadedModel, Runtime>();
  private readonly pending = new Set<Promise<unknown>>();
  private closed = false;
  private disposing?: Promise<void>;

  constructor(options: EngineOptions) {
    this.scheduler = new InferenceScheduler(options.scheduler);
    for (const backend of options.backends) this.runtimes.register(backend.type, () => backend.create(this.memory));
  }

  private operation<T>(fn: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('Engine is disposed'));
    const promise = fn();
    this.pending.add(promise);
    void promise.then(() => this.pending.delete(promise), () => this.pending.delete(promise));
    return promise;
  }

  loadModel(url: string, options: LoadOptions = {}): Promise<LoadedModel> {
    return this.operation(async () => {
      const { loadModelData } = await import('../utils/model-loader.js');
      const data = await loadModelData(url, {
        cache: options.cache, resumable: options.resumable, chunkSize: options.chunkSize,
        forceDownload: options.forceDownload,
        onProgress: options.onProgress ? p => options.onProgress!(p.percent / 100) : undefined,
      });
      return this.load(data, options);
    });
  }
  loadModelFromBuffer(data: ArrayBuffer, options: LoadOptions = {}): Promise<LoadedModel> {
    return this.operation(() => this.load(data, options));
  }
  private async load(data: ArrayBuffer, options: LoadOptions): Promise<LoadedModel> {
    const runtime = await this.runtimes.getRuntime(options.runtime ?? 'auto');
    const model = await runtime.loadModel(data, options);
    this.models.set(model, runtime);
    return model;
  }
  runInference(model: LoadedModel, inputs: Tensor[], context: ExecutionContext = {}): Promise<Tensor[]> {
    return this.run(model, runtime => runtime.run(model, inputs), context);
  }
  runInferenceNamed(model: LoadedModel, inputs: Map<string, Tensor>, context: ExecutionContext = {}): Promise<Tensor[]> {
    return this.run(model, runtime => {
      if (!runtime.runNamed) throw new Error('Runtime does not support named inputs');
      return runtime.runNamed(model, inputs);
    }, context);
  }
  private run(model: LoadedModel, fn: (runtime: Runtime) => Promise<Tensor[]>, context: ExecutionContext): Promise<Tensor[]> {
    return this.operation(async () => {
      throwIfAborted(context.signal);
      const runtime = this.models.get(model);
      if (!runtime || !model.isLoaded) throw new Error('Model is disposed or belongs to another engine');
      return this.scheduler.execute(model.id, () => {
        if (!model.isLoaded) throw new Error('Model has been disposed');
        return fn(runtime);
      }, context, outputs => outputs.forEach(t => t.dispose()));
    });
  }
  dispose(): Promise<void> {
    if (this.disposing) return this.disposing;
    this.closed = true;
    this.scheduler.dispose();
    this.disposing = (async () => {
      await Promise.allSettled(this.pending);
      for (const model of this.models.keys()) model.dispose();
      this.models.clear();
      await this.runtimes.disposeAll();
      this.memory.dispose();
    })();
    return this.disposing;
  }
}
export function createInferenceEngine(options: EngineOptions): InferenceEngine {
  return new InferenceEngine(options);
}
