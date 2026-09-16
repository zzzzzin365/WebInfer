import { type InferenceClient } from '../core/inference-client.js';
import type { AssetClassifier, IntentClassifier, Segmenter } from '../workflows-3d/types.js';
export interface PipelineAdapterOptions {
    engine: InferenceClient;
    assetModel: string;
    assetLabels: string[];
    analyzerVersion: string;
    /** NLI model must match the pipeline's tokenizer and entailment output convention. */
    intentModel?: string;
    segmentationModels?: {
        encoder: string;
        decoder: string;
    };
}
/** Lazy, single-flight model initialization; application owns the shared model lifetime. */
export declare function createPipelineInferenceAdapters(options: PipelineAdapterOptions): {
    segmenter: Segmenter;
    intentClassifier: IntentClassifier;
    assetClassifier: AssetClassifier;
    dispose(): Promise<void>;
};
//# sourceMappingURL=pipeline-inference.d.ts.map