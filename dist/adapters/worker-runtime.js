import { LoadedModelImpl } from '../core/runtime.js';
import { MemoryManager } from '../core/memory.js';
import { deserializeTensor, serializeTensor } from '../core/worker.js';
/** Inject a bundled module Worker. No CDN, default URL or application globals. */
export class WorkerRuntime {
    worker;
    memory;
    name = 'wasm';
    capabilities = { concurrency: false, quantization: true, float16: false,
        dynamicShapes: true, maxBatchSize: 32, availableMemory: 512 * 1024 * 1024 };
    nextId = 0;
    requests = new Map();
    models = new Map();
    releases = new Set();
    init;
    closed = false;
    disposing;
    constructor(worker, memory = new MemoryManager()) {
        this.worker = worker;
        this.memory = memory;
        worker.onmessage = (event) => {
            const response = event.data, pending = this.requests.get(response.id);
            if (!pending)
                return;
            this.requests.delete(response.id);
            if (response.error)
                pending.reject(new Error(response.error));
            else
                pending.resolve(response.value);
        };
        worker.onerror = () => this.fail(new Error('Inference worker crashed; recreate the runtime and reload models'));
        worker.onmessageerror = () => this.fail(new Error('Invalid worker message'));
    }
    fail(error) {
        this.closed = true;
        for (const request of this.requests.values())
            request.reject(error);
        this.requests.clear();
        this.worker.terminate();
    }
    request(payload) {
        if (this.closed)
            return Promise.reject(new Error('Worker runtime is closed'));
        const id = ++this.nextId;
        return new Promise((resolve, reject) => {
            this.requests.set(id, { resolve: value => resolve(value), reject });
            const transfers = payload.data ? [payload.data] : payload.inputs?.map(([, input]) => input.data) ?? [];
            try {
                this.worker.postMessage({ ...payload, id }, transfers);
            }
            catch (error) {
                this.requests.delete(id);
                reject(error);
            }
        });
    }
    async isAvailable() { return !this.closed; }
    initialize() { return this.init ??= this.request({ type: 'init' }); }
    async loadModel(data, options = {}) {
        await this.initialize();
        const { onProgress: _onProgress, ...transferOptions } = options;
        const result = await this.request({ type: 'load', data: data.slice(0), options: transferOptions });
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
    run(model, inputs) {
        if (inputs.length !== model.metadata.inputs.length)
            return Promise.reject(new Error('Incorrect input count'));
        return this.runNamed(model, new Map(model.metadata.inputs.map((input, i) => [input.name, inputs[i]])));
    }
    async runNamed(model, inputs) {
        const modelId = this.models.get(model);
        if (!modelId || !model.isLoaded)
            throw new Error('Model does not belong to this worker');
        const outputs = await this.request({ type: 'run', modelId,
            inputs: [...inputs].map(([name, tensor]) => [name, serializeTensor(tensor)]) });
        return Promise.all(outputs.map(deserializeTensor));
    }
    dispose() {
        return this.disposing ??= (async () => {
            try {
                if (!this.closed) {
                    for (const model of this.models.keys())
                        model.dispose();
                    await Promise.allSettled(this.releases);
                    await this.request({ type: 'dispose' });
                }
            }
            finally {
                this.fail(new Error('Worker runtime disposed'));
            }
        })();
    }
}
//# sourceMappingURL=worker-runtime.js.map