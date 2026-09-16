import type { LoadedModel, ModelLoadOptions, Runtime, RuntimeCapabilities, Tensor } from '../core/types.js';
import { MemoryManager } from '../core/memory.js';
import { type SerializedTensor } from '../core/worker.js';
export interface RuntimeWorkerRequest {
    id: number;
    type: 'init' | 'load' | 'run' | 'unload' | 'dispose';
    data?: ArrayBuffer;
    options?: ModelLoadOptions;
    modelId?: string;
    inputs?: Array<[string, SerializedTensor]>;
}
export interface RuntimeWorkerResponse {
    id: number;
    value?: unknown;
    error?: string;
}
/** Inject a bundled module Worker. No CDN, default URL or application globals. */
export declare class WorkerRuntime implements Runtime {
    private readonly worker;
    private readonly memory;
    readonly name: "wasm";
    readonly capabilities: RuntimeCapabilities;
    private nextId;
    private readonly requests;
    private readonly models;
    private readonly releases;
    private init?;
    private closed;
    private disposing?;
    constructor(worker: Worker, memory?: MemoryManager);
    private fail;
    private request;
    isAvailable(): Promise<boolean>;
    initialize(): Promise<void>;
    loadModel(data: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    runNamed(model: LoadedModel, inputs: Map<string, Tensor>): Promise<Tensor[]>;
    dispose(): Promise<void>;
}
//# sourceMappingURL=worker-runtime.d.ts.map