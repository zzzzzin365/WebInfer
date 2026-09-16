/**
 * WebInfer - Runtime Management
 *
 * Manages runtime backends and automatic selection.
 * Provides unified interface for different compute backends.
 */
import { WebInferError, ErrorCodes, } from './types.js';
import { getScheduler } from './scheduler.js';
import { getMemoryManager } from './memory.js';
// ============================================================================
// Runtime Registry
// ============================================================================
/**
 * Runtime priority order (higher priority first)
 */
const RUNTIME_PRIORITY = ['webgpu', 'webnn', 'wasm'];
// ============================================================================
// Runtime Manager
// ============================================================================
/**
 * RuntimeManager - Manages runtime selection and lifecycle
 *
 * Features:
 * - Automatic best runtime selection
 * - Runtime registration
 * - Capability detection
 * - Fallback handling
 */
export class RuntimeManager {
    static instance = null;
    listeners = new Map();
    defaultRuntime = 'auto';
    runtimeFactories = new Map();
    runtimeInstances = new Map();
    initializing = new Map();
    constructor() { }
    has(type) { return this.runtimeFactories.has(type); }
    /**
     * Get singleton instance
     */
    static getInstance() {
        if (!RuntimeManager.instance) {
            RuntimeManager.instance = new RuntimeManager();
        }
        return RuntimeManager.instance;
    }
    /**
     * Register a runtime factory
     */
    register(type, factory) {
        if (this.runtimeInstances.has(type) || this.initializing.has(type)) {
            throw new Error(`Runtime '${type}' is already active`);
        }
        this.runtimeFactories.set(type, factory);
    }
    /**
     * Get a runtime instance
     */
    async getRuntime(type = 'auto') {
        if (type === 'auto') {
            return this.getBestRuntime();
        }
        const existing = this.runtimeInstances.get(type);
        if (existing)
            return existing;
        const pending = this.initializing.get(type);
        if (pending)
            return pending;
        const factory = this.runtimeFactories.get(type);
        if (!factory)
            throw new WebInferError(`Runtime '${type}' is not registered`, ErrorCodes.RUNTIME_NOT_AVAILABLE);
        const promise = (async () => {
            const runtime = factory();
            try {
                if (!await runtime.isAvailable())
                    throw new Error(`Runtime '${type}' is not available`);
                await runtime.initialize();
                this.runtimeInstances.set(type, runtime);
                this.emit('runtime:ready', { runtime: type });
                return runtime;
            }
            catch (error) {
                await runtime.dispose();
                throw error;
            }
        })();
        this.initializing.set(type, promise);
        try {
            return await promise;
        }
        finally {
            this.initializing.delete(type);
        }
    }
    /**
     * Get the best available runtime
     */
    async getBestRuntime() {
        for (const type of RUNTIME_PRIORITY) {
            try {
                if (this.has(type))
                    return await this.getRuntime(type);
            }
            catch {
                // Try next runtime
                continue;
            }
        }
        throw new WebInferError('No runtime available. Please ensure WebGPU, WebNN, or WASM is supported.', ErrorCodes.RUNTIME_NOT_AVAILABLE, { triedRuntimes: RUNTIME_PRIORITY });
    }
    /**
     * Check which runtimes are available
     */
    async detectAvailableRuntimes() {
        const results = new Map();
        for (const type of RUNTIME_PRIORITY) {
            const factory = this.runtimeFactories.get(type);
            if (!factory) {
                results.set(type, false);
                continue;
            }
            try {
                const existing = this.runtimeInstances.get(type);
                const runtime = existing ?? factory();
                try {
                    results.set(type, await runtime.isAvailable());
                }
                finally {
                    if (!existing)
                        await runtime.dispose();
                }
            }
            catch {
                results.set(type, false);
            }
        }
        return results;
    }
    /**
     * Get capabilities of a runtime
     */
    async getCapabilities(type) {
        const runtime = await this.getRuntime(type);
        return runtime.capabilities;
    }
    /**
     * Set default runtime
     */
    setDefaultRuntime(type) {
        this.defaultRuntime = type;
    }
    /**
     * Get default runtime type
     */
    getDefaultRuntimeType() {
        return this.defaultRuntime;
    }
    /**
     * Dispose a specific runtime
     */
    async disposeRuntime(type) {
        await this.initializing.get(type)?.catch(() => undefined);
        const runtime = this.runtimeInstances.get(type);
        if (runtime) {
            await runtime.dispose();
            this.runtimeInstances.delete(type);
        }
    }
    /**
     * Dispose all runtimes
     */
    async disposeAll() {
        await Promise.allSettled(this.initializing.values());
        await Promise.all([...this.runtimeInstances.keys()].map(type => this.disposeRuntime(type)));
    }
    /**
     * Add event listener
     */
    on(event, listener) {
        let listeners = this.listeners.get(event);
        if (!listeners) {
            listeners = new Set();
            this.listeners.set(event, listeners);
        }
        listeners.add(listener);
    }
    /**
     * Remove event listener
     */
    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }
    /**
     * Emit event
     */
    emit(type, data) {
        const event = {
            type,
            timestamp: Date.now(),
            data,
        };
        const listeners = this.listeners.get(type);
        if (listeners) {
            for (const listener of listeners) {
                try {
                    listener(event);
                }
                catch (error) {
                    console.error('Error in event listener:', error);
                }
            }
        }
    }
}
// ============================================================================
// Model Loader
// ============================================================================
/**
 * Model instance counter
 */
let modelIdCounter = 0;
/**
 * Generate unique model ID
 */
function generateModelId() {
    return `model_${++modelIdCounter}_${Date.now().toString(36)}`;
}
/**
 * LoadedModelImpl - Implementation of LoadedModel interface
 */
export class LoadedModelImpl {
    memory;
    id;
    metadata;
    runtime;
    _isLoaded = true;
    _dispose;
    constructor(metadata, runtime, dispose, memory = getMemoryManager()) {
        this.memory = memory;
        this.id = generateModelId();
        this.metadata = metadata;
        this.runtime = runtime;
        this._dispose = dispose;
    }
    get isLoaded() {
        return this._isLoaded;
    }
    dispose() {
        if (this._isLoaded) {
            this._isLoaded = false;
            this._dispose();
            this.memory.untrack(this.id);
        }
    }
}
// ============================================================================
// Model Loading Functions
// ============================================================================
/**
 * Load model from URL with advanced loading support
 * (caching, sharding, resume download)
 */
export async function loadModel(url, options = {}) {
    const manager = RuntimeManager.getInstance();
    const runtime = await manager.getRuntime(options.runtime ?? 'auto');
    // Import model loader dynamically to avoid circular dependencies
    const { loadModelData } = await import('../utils/model-loader.js');
    // Use advanced model loader with caching and resume support
    const modelData = await loadModelData(url, {
        cache: options.cache ?? true,
        resumable: options.resumable ?? true,
        chunkSize: options.chunkSize,
        forceDownload: options.forceDownload,
        onProgress: options.onProgress ? (progress) => {
            options.onProgress(progress.percent / 100);
        } : undefined,
    });
    // Load into runtime
    const model = await runtime.loadModel(modelData, options);
    return model;
}
/**
 * Load model from ArrayBuffer
 */
export async function loadModelFromBuffer(data, options = {}) {
    const manager = RuntimeManager.getInstance();
    const runtime = await manager.getRuntime(options.runtime ?? 'auto');
    return runtime.loadModel(data, options);
}
// ============================================================================
// Inference Functions
// ============================================================================
/**
 * Run inference on a model
 */
export async function runInference(model, inputs, context = {}) {
    if (!model.isLoaded) {
        throw new WebInferError('Model has been disposed', ErrorCodes.MODEL_NOT_LOADED, { modelId: model.id });
    }
    const manager = RuntimeManager.getInstance();
    const runtime = await manager.getRuntime(model.runtime);
    // Use scheduler for execution
    const scheduler = getScheduler();
    return scheduler.execute(model.id, () => runtime.run(model, inputs), context, outputs => outputs.forEach(t => t.dispose()));
}
/**
 * Run inference with named inputs
 */
export async function runInferenceNamed(model, namedInputs, context = {}) {
    if (!model.isLoaded) {
        throw new WebInferError('Model has been disposed', ErrorCodes.MODEL_NOT_LOADED, { modelId: model.id });
    }
    const manager = RuntimeManager.getInstance();
    const runtime = await manager.getRuntime(model.runtime);
    // Check if runtime supports named inputs
    if (!('runNamed' in runtime)) {
        throw new WebInferError('Runtime does not support named inputs', ErrorCodes.INFERENCE_FAILED, { modelId: model.id });
    }
    // Use scheduler for execution
    const scheduler = getScheduler();
    return scheduler.execute(model.id, () => runtime.runNamed(model, namedInputs), context, outputs => outputs.forEach(t => t.dispose()));
}
/**
 * Run inference with batch processing
 */
export async function runBatchInference(model, batches, context = {}) {
    return Promise.all(batches.map(inputs => runInference(model, inputs, context)));
}
// ============================================================================
// Convenience Functions
// ============================================================================
/**
 * Get runtime manager instance
 */
export function getRuntimeManager() {
    return RuntimeManager.getInstance();
}
/**
 * Register a runtime
 */
export function registerRuntime(type, factory) {
    RuntimeManager.getInstance().register(type, factory);
}
/**
 * Get the best available runtime
 */
export async function getBestRuntime() {
    return RuntimeManager.getInstance().getBestRuntime();
}
/**
 * Check available runtimes
 */
export async function getAvailableRuntimes() {
    return RuntimeManager.getInstance().detectAvailableRuntimes();
}
//# sourceMappingURL=runtime.js.map