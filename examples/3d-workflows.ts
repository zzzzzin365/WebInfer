/** Application composition example. Supply real Viewer and backend service adapters. */
import { createInferenceEngine, TaskScope } from '../src/core/index.js';
import { createONNXRuntime } from '../src/backends/onnx.js';
import { createPipelineInferenceAdapters, WorkerRuntime, type PipelineAdapterOptions } from '../src/adapters/index.js';
import { submitGenerationRequest, ViewerAnalysisController, type AgentGateway, type AssetRepository,
  type GenerationRequest, type GenerationService, type ReferenceUploader, type ViewerAdapter } from '../src/workflows-3d/index.js';

export function create3DApplication(ports: {
  viewer: ViewerAdapter;
  assets: AssetRepository;
  generation: GenerationService;
  agent: AgentGateway;
  uploader: ReferenceUploader;
  onAnalysisError(error: unknown): void;
  /** Prefer a bundled module worker. Omission runs inference on the current thread. */
  createWorker?: () => Worker;
}, models: Omit<PipelineAdapterOptions, 'engine'>) {
  const engine = createInferenceEngine({
    backends: [{ type: 'wasm', create: memory => ports.createWorker
      ? new WorkerRuntime(ports.createWorker(), memory) : createONNXRuntime(memory) }],
    scheduler: { maxConcurrentTasks: 1, maxConcurrentPerModel: 1 },
  });
  const inference = createPipelineInferenceAdapters({ ...models, engine });
  const controller = new ViewerAnalysisController(ports.viewer, inference.assetClassifier, ports.assets, ports.onAnalysisError);
  const requests = new TaskScope('generation-requests');
  return {
    viewerAnalysis: controller,
    submit(request: GenerationRequest, reference?: Parameters<typeof submitGenerationRequest>[2]) {
      return requests.run(context => submitGenerationRequest({ ...ports,
        classifier: inference.intentClassifier, segmenter: inference.segmenter,
      }, request, reference, context), 'high');
    },
    async dispose() {
      try {
        await Promise.all([controller.dispose(), requests.dispose()]);
        await inference.dispose();
      } finally { await engine.dispose(); }
    },
  };
}
