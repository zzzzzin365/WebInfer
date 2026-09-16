/**
 * WebInfer - ONNX Runtime Backend
 *
 * Uses onnxruntime-web for real ONNX model inference.
 * onnxruntime-web is an optional peer dependency loaded dynamically.
 */
import { WebInferError, ErrorCodes, } from '../core/types.js';
import { LoadedModelImpl } from '../core/runtime.js';
import { WebInferTensor } from '../core/tensor.js';
import { MemoryManager } from '../core/memory.js';
// Lazy-loaded onnxruntime-web module
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort = null;
async function getOrt() {
    if (ort)
        return ort;
    try {
        // Import the WASM-only sub-path so Vite rewrites the bare specifier
        // to ort.wasm.bundle.min.mjs. This avoids loading the JSEP/WebGPU
        // worker module (jsep.mjs) that ort.bundle.min.mjs eagerly fetches
        // whenever navigator.gpu exists — which causes a 404 in dev servers
        // that restrict ES module imports from /public.
        ort = await import('onnxruntime-web/wasm');
        return ort;
    }
    catch {
        return null;
    }
}
/**
 * Check whether onnxruntime-web is importable.
 */
export async function isOnnxAvailable() {
    return (await getOrt()) != null;
}
// ============================================================================
// ONNX Runtime Implementation
// ============================================================================
/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 */
export class ONNXRuntime {
    memory;
    name = 'wasm'; // Register as wasm since it's the fallback
    sessionStore = new Map();
    releases = new Set();
    constructor(memory = new MemoryManager()) {
        this.memory = memory;
    }
    initialized = false;
    executionProvider = 'wasm';
    get capabilities() {
        return {
            concurrency: true,
            quantization: true,
            float16: this.executionProvider === 'webgpu',
            dynamicShapes: true,
            maxBatchSize: 32,
            availableMemory: 512 * 1024 * 1024, // 512MB
        };
    }
    /**
     * Check if ONNX Runtime is available (peer dependency installed)
     */
    async isAvailable() {
        return isOnnxAvailable();
    }
    /**
     * Initialize the ONNX runtime
     */
    async initialize() {
        if (this.initialized)
            return;
        const ortModule = await getOrt();
        if (!ortModule) {
            throw new WebInferError('onnxruntime-web is not installed. Install it with: npm install onnxruntime-web', ErrorCodes.RUNTIME_NOT_AVAILABLE);
        }
        // Configure WASM backend for browser use.
        // numThreads=1 disables multi-threading so ort only needs the plain
        // .wasm binary — the worker .mjs file is never requested, which avoids
        // Vite's restriction on importing files from /public as ES modules.
        // Consumers should copy onnxruntime-web/dist/*.wasm to public/ort/.
        if (typeof window !== 'undefined' && ortModule.env?.wasm) {
            ortModule.env.wasm.wasmPaths ??= '/ort/';
            ortModule.env.wasm.numThreads = 1;
        }
        this.initialized = true;
    }
    /**
     * Load a model from ArrayBuffer
     */
    async loadModel(modelData, options = {}) {
        if (!this.initialized) {
            await this.initialize();
        }
        try {
            const ortModule = await getOrt();
            if (!ortModule) {
                throw new Error('onnxruntime-web is not installed');
            }
            // WASM-only execution provider — WebGPU acceleration can be added
            // later via the dedicated WebGPURuntime backend.
            const sessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            };
            const modelBytes = new Uint8Array(modelData);
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const session = await ortModule.InferenceSession.create(modelBytes, sessionOptions);
            // Get input/output names
            const inputNames = session.inputNames;
            const outputNames = session.outputNames;
            // Generate model ID
            const modelId = `onnx_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
            // Store session
            this.sessionStore.set(modelId, {
                session,
                inputNames: [...inputNames],
                outputNames: [...outputNames],
                active: new Set(),
            });
            // Create metadata
            const metadata = {
                name: options.metadata?.name ?? 'onnx-model',
                version: '1.0.0',
                inputs: inputNames.map((name) => ({
                    name,
                    dtype: 'float32',
                    shape: [-1], // Dynamic shape
                })),
                outputs: outputNames.map((name) => ({
                    name,
                    dtype: 'float32',
                    shape: [-1],
                })),
                sizeBytes: modelData.byteLength,
                quantization: options.quantization ?? 'float32',
                format: 'onnx',
            };
            // Create model instance
            const model = new LoadedModelImpl(metadata, 'wasm', () => { void this.unloadModel(modelId); }, this.memory);
            // Override the ID to match our stored session
            Object.defineProperty(model, 'id', { value: modelId, writable: false });
            // Track in memory manager
            this.memory.trackModel(model, () => model.dispose());
            return model;
        }
        catch (error) {
            throw new WebInferError(`Failed to load ONNX model: ${error instanceof Error ? error.message : String(error)}`, ErrorCodes.MODEL_LOAD_FAILED, { error });
        }
    }
    /**
     * Run inference
     */
    async run(model, inputs) {
        const data = this.sessionStore.get(model.id);
        if (!data)
            throw new Error(`ONNX session not found for model ${model.id}`);
        if (inputs.length !== data.inputNames.length)
            throw new Error('Incorrect ONNX input count');
        return this.runNamed(model, new Map(data.inputNames.map((name, i) => [name, inputs[i]])));
    }
    async runNamed(model, inputs) {
        const data = this.sessionStore.get(model.id);
        if (!data || !model.isLoaded)
            throw new Error(`ONNX session not found for model ${model.id}`);
        const operation = this.execute(data, inputs);
        data.active.add(operation);
        try {
            return await operation;
        }
        finally {
            data.active.delete(operation);
        }
    }
    async execute(data, inputs) {
        const ortModule = await getOrt();
        const feeds = {};
        let results = {};
        const outputs = [];
        try {
            for (const name of data.inputNames) {
                const input = inputs.get(name);
                if (!input)
                    throw new Error(`Missing ONNX input '${name}'`);
                feeds[name] = new ortModule.Tensor(input.dtype, input.data, [...input.shape]);
            }
            results = await data.session.run(feeds);
            for (const name of data.outputNames) {
                const output = results[name];
                if (!output)
                    throw new Error(`Missing ONNX output '${name}'`);
                // Copy before releasing the ORT-owned tensor; preserve integer outputs.
                outputs.push(new WebInferTensor(output.data.slice(), Array.from(output.dims), output.type));
            }
            return outputs;
        }
        catch (error) {
            outputs.forEach(t => t.dispose());
            throw new WebInferError(`ONNX inference failed: ${String(error)}`, ErrorCodes.INFERENCE_FAILED, { error });
        }
        finally {
            for (const tensor of [...Object.values(feeds), ...Object.values(results)])
                tensor.dispose?.();
        }
    }
    unloadModel(modelId) {
        const data = this.sessionStore.get(modelId);
        if (!data)
            return Promise.resolve();
        this.sessionStore.delete(modelId);
        const release = (async () => {
            await Promise.allSettled(data.active);
            await data.session.release();
        })();
        this.releases.add(release);
        // Attach a handler for callers using the synchronous model.dispose API.
        void release.then(() => this.releases.delete(release), () => undefined);
        return release;
    }
    async dispose() {
        for (const id of this.sessionStore.keys())
            void this.unloadModel(id);
        const releases = [...this.releases];
        await Promise.all(releases);
        this.releases.clear();
        this.initialized = false;
    }
}
export function createONNXRuntime(memory) {
    return new ONNXRuntime(memory);
}
//# sourceMappingURL=onnx.js.map