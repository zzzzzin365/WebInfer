import type { LoadedModel, ModelLoadOptions, RuntimeType, TaskPriority, Tensor } from './types.js';
/** Per-call context: never stored as mutable state on a shared pipeline. */
export interface ExecutionContext {
    priority?: TaskPriority;
    signal?: AbortSignal;
    scopeId?: string;
}
export type LoadOptions = ModelLoadOptions & {
    runtime?: RuntimeType;
    cache?: boolean;
    resumable?: boolean;
    chunkSize?: number;
    forceDownload?: boolean;
};
export interface InferenceClient {
    loadModel(url: string, options?: LoadOptions): Promise<LoadedModel>;
    loadModelFromBuffer(data: ArrayBuffer, options?: LoadOptions): Promise<LoadedModel>;
    runInference(model: LoadedModel, inputs: Tensor[], context?: ExecutionContext): Promise<Tensor[]>;
    runInferenceNamed(model: LoadedModel, inputs: Map<string, Tensor>, context?: ExecutionContext): Promise<Tensor[]>;
}
export declare function throwIfAborted(signal?: AbortSignal): void;
//# sourceMappingURL=inference-client.d.ts.map