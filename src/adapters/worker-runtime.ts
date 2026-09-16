import type { LoadedModel, ModelLoadOptions, ModelMetadata, Runtime, RuntimeCapabilities, Tensor } from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { MemoryManager } from '../core/memory.js';
import { deserializeTensor, serializeTensor, type SerializedTensor } from '../core/worker.js';

export interface RuntimeWorkerRequest {
  id: number;
  type: 'init' | 'load' | 'run' | 'unload' | 'dispose';
  data?: ArrayBuffer;
  options?: ModelLoadOptions;
  modelId?: string;
  inputs?: Array<[string, SerializedTensor]>;
}
export interface RuntimeWorkerResponse { id: number; value?: unknown; error?: string }

/** Inject a bundled module Worker. No CDN, default URL or application globals. */
export class WorkerRuntime implements Runtime {
  readonly name = 'wasm' as const;
  readonly capabilities: RuntimeCapabilities = { concurrency: false, quantization: true, float16: false,
    dynamicShapes: true, maxBatchSize: 32, availableMemory: 512 * 1024 * 1024 };
  private nextId = 0;
  private readonly requests = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();
  private readonly models = new Map<LoadedModel, string>();
  private readonly releases = new Set<Promise<unknown>>();
  private init?: Promise<void>;
  private closed = false;
  private disposing?: Promise<void>;
  constructor(private readonly worker: Worker, private readonly memory = new MemoryManager()) {
    worker.onmessage = (event: MessageEvent<RuntimeWorkerResponse>) => {
      const response = event.data, pending = this.requests.get(response.id);
      if (!pending) return;
      this.requests.delete(response.id);
      if (response.error) pending.reject(new Error(response.error));
      else pending.resolve(response.value);
    };
    worker.onerror = () => this.fail(new Error('Inference worker crashed; recreate the runtime and reload models'));
    worker.onmessageerror = () => this.fail(new Error('Invalid worker message'));
  }
  private fail(error: Error): void {
    this.closed = true;
    for (const request of this.requests.values()) request.reject(error);
    this.requests.clear();
    this.worker.terminate();
  }
  private request<T>(payload: Omit<RuntimeWorkerRequest, 'id'>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('Worker runtime is closed'));
    const id = ++this.nextId;
    return new Promise<T>((resolve, reject) => {
      this.requests.set(id, { resolve: value => resolve(value as T), reject });
      const transfers: ArrayBuffer[] = payload.data ? [payload.data] : payload.inputs?.map(([, input]) => input.data) ?? [];
      try { this.worker.postMessage({ ...payload, id }, transfers); }
      catch (error) { this.requests.delete(id); reject(error); }
    });
  }
  async isAvailable(): Promise<boolean> { return !this.closed; }
  initialize(): Promise<void> { return this.init ??= this.request<void>({ type: 'init' }); }
  async loadModel(data: ArrayBuffer, options: ModelLoadOptions = {}): Promise<LoadedModel> {
    await this.initialize();
    const { onProgress: _onProgress, ...transferOptions } = options;
    const result = await this.request<{ id: string; metadata: ModelMetadata }>({ type: 'load', data: data.slice(0), options: transferOptions });
    const model = new LoadedModelImpl(result.metadata, 'wasm', () => {
      this.models.delete(model);
      const release = this.request({ type: 'unload', modelId: result.id });
      this.releases.add(release);
      void release.then(() => this.releases.delete(release), () => undefined);
    }, this.memory);
    this.models.set(model, result.id);
    this.memory.trackModel(model, () => model.dispose());
    return model;
  }
  run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]> {
    if (inputs.length !== model.metadata.inputs.length) return Promise.reject(new Error('Incorrect input count'));
    return this.runNamed(model, new Map(model.metadata.inputs.map((input, i) => [input.name, inputs[i]!])));
  }
  async runNamed(model: LoadedModel, inputs: Map<string, Tensor>): Promise<Tensor[]> {
    const modelId = this.models.get(model);
    if (!modelId || !model.isLoaded) throw new Error('Model does not belong to this worker');
    const outputs = await this.request<SerializedTensor[]>({ type: 'run', modelId,
      inputs: [...inputs].map(([name, tensor]) => [name, serializeTensor(tensor)]) });
    return Promise.all(outputs.map(deserializeTensor));
  }
  dispose(): Promise<void> {
    return this.disposing ??= (async () => {
      try {
        if (!this.closed) {
          for (const model of this.models.keys()) model.dispose();
          await Promise.allSettled(this.releases);
          await this.request({ type: 'dispose' });
        }
      } finally { this.fail(new Error('Worker runtime disposed')); }
    })();
  }
}
