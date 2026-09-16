import { RuntimeManager } from './runtime.js';
import { InferenceScheduler } from './scheduler.js';
import { MemoryManager } from './memory.js';
import { type ExecutionContext, type InferenceClient, type LoadOptions } from './inference-client.js';
import type { LoadedModel, Runtime, RuntimeType, SchedulerOptions, Tensor } from './types.js';
export interface EngineOptions {
    backends: Array<{
        type: RuntimeType;
        create: (memory: MemoryManager) => Runtime;
    }>;
    scheduler?: SchedulerOptions;
}
/** Owns runtimes and models. Callers own inference input/output tensors. */
export declare class InferenceEngine implements InferenceClient {
    readonly runtimes: RuntimeManager;
    readonly scheduler: InferenceScheduler;
    readonly memory: MemoryManager;
    private readonly models;
    private readonly pending;
    private closed;
    private disposing?;
    constructor(options: EngineOptions);
    private operation;
    loadModel(url: string, options?: LoadOptions): Promise<LoadedModel>;
    loadModelFromBuffer(data: ArrayBuffer, options?: LoadOptions): Promise<LoadedModel>;
    private load;
    runInference(model: LoadedModel, inputs: Tensor[], context?: ExecutionContext): Promise<Tensor[]>;
    runInferenceNamed(model: LoadedModel, inputs: Map<string, Tensor>, context?: ExecutionContext): Promise<Tensor[]>;
    private run;
    dispose(): Promise<void>;
}
export declare function createInferenceEngine(options: EngineOptions): InferenceEngine;
//# sourceMappingURL=engine.d.ts.map