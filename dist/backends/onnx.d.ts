/**
 * WebInfer - ONNX Runtime Backend
 *
 * Uses onnxruntime-web for real ONNX model inference.
 * onnxruntime-web is an optional peer dependency loaded dynamically.
 */
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
import { MemoryManager } from '../core/memory.js';
/**
 * Check whether onnxruntime-web is importable.
 */
export declare function isOnnxAvailable(): Promise<boolean>;
/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 */
export declare class ONNXRuntime implements Runtime {
    private readonly memory;
    readonly name: RuntimeType;
    private readonly sessionStore;
    private readonly releases;
    constructor(memory?: MemoryManager);
    private initialized;
    private executionProvider;
    get capabilities(): RuntimeCapabilities;
    /**
     * Check if ONNX Runtime is available (peer dependency installed)
     */
    isAvailable(): Promise<boolean>;
    /**
     * Initialize the ONNX runtime
     */
    initialize(): Promise<void>;
    /**
     * Load a model from ArrayBuffer
     */
    loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    /**
     * Run inference
     */
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    runNamed(model: LoadedModel, inputs: Map<string, Tensor>): Promise<Tensor[]>;
    private execute;
    private unloadModel;
    dispose(): Promise<void>;
}
export declare function createONNXRuntime(memory?: MemoryManager): Runtime;
//# sourceMappingURL=onnx.d.ts.map