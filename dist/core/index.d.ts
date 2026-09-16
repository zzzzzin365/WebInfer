/**
 * WebInfer - Core Module Exports
 */
export * from './types.js';
export { WebInferTensor, tensor, zeros, ones, full, random, randn, arange, linspace, eye, add, sub, mul, div, matmul, softmax, relu, sigmoid, tanh, sum, mean, argmax, concat, } from './tensor.js';
export { InferenceScheduler, getScheduler, setScheduler, configureScheduler, } from './scheduler.js';
export { MemoryManager, MemoryScope, ModelCache, withMemoryScope, withMemoryScopeSync, getMemoryManager, getMemoryStats, release, gc, } from './memory.js';
export { RuntimeManager, LoadedModelImpl, loadModel, loadModelFromBuffer, runInference, runBatchInference, getRuntimeManager, registerRuntime, getBestRuntime, getAvailableRuntimes, } from './runtime.js';
export { registerPlugin, getPluginPipeline, getPluginMiddleware, listPlugins, unregisterPlugin, type WebInferPlugin, type PluginPipelineEntry, type PluginBackendEntry, type PluginMiddleware, } from './plugin.js';
export { getDeviceProfile, recommendQuantization, recommendModelVariant, resetDeviceProfile, type DeviceProfile, type DeviceTier, type ModelRecommendation, } from './device-profiler.js';
export { InferenceWorker, WorkerPool, getWorkerPool, runInWorker, isWorkerSupported, serializeTensor, deserializeTensor, type WorkerMessage, type WorkerMessageType, type LoadModelRequest, type InferenceRequest, type SerializedTensor, type WorkerPoolOptions, } from './worker.js';
export { InferenceEngine, createInferenceEngine, type EngineOptions } from './engine.js';
export type { InferenceClient, ExecutionContext, LoadOptions } from './inference-client.js';
export { TaskScope } from './task-scope.js';
//# sourceMappingURL=index.d.ts.map