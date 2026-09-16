/**
 * WebInfer - Feature Extraction Pipeline
 *
 * Extract embeddings/features from text using sentence-transformer models.
 */
import { PipelineConfig, PipelineOptions } from '../core/types.js';
import { WebInferTensor } from '../core/tensor.js';
import { BasePipeline, FeatureExtractionResult } from './base.js';
export interface FeatureExtractionOptions extends PipelineOptions {
    pooling?: 'mean' | 'max' | 'cls' | 'none';
    normalize?: boolean;
    outputDim?: number;
}
export declare class FeatureExtractionPipeline extends BasePipeline<string | string[], FeatureExtractionResult | FeatureExtractionResult[]> {
    private tokenizer;
    private onnxModel;
    private embeddingDim;
    private modelUrl;
    private tokenizerUrl;
    constructor(config: PipelineConfig, embeddingDim?: number);
    initialize(): Promise<void>;
    dispose(): void;
    run(input: string | string[], options?: FeatureExtractionOptions): Promise<FeatureExtractionResult | FeatureExtractionResult[]>;
    protected preprocess(input: string | string[]): Promise<WebInferTensor[]>;
    private runInference;
    protected postprocess(outputs: WebInferTensor[], options?: FeatureExtractionOptions): Promise<FeatureExtractionResult>;
    private extractCLSEmbedding;
    private meanPooling;
    private maxPooling;
    private normalizeVector;
}
export declare function createFeatureExtractionPipeline(config?: Partial<PipelineConfig>): FeatureExtractionPipeline;
//# sourceMappingURL=feature-extraction.d.ts.map