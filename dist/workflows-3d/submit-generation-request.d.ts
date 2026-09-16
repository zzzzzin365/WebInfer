import { type ExecutionContext } from '../core/inference-client.js';
import { type RoutingPolicy } from './route-generation-request.js';
import type { AgentGateway, AssetRepository, ForegroundPrompt, GenerationRequest, GenerationService, IntentClassifier, PixelImage, ReferenceUploader, Segmenter } from './types.js';
export interface GenerationDependencies {
    classifier: IntentClassifier;
    segmenter: Segmenter;
    uploader: ReferenceUploader;
    generation: GenerationService;
    agent: AgentGateway;
    assets: AssetRepository;
    routingPolicy?: RoutingPolicy;
}
/** Application orchestration. Route classification itself never submits a job. */
export declare function submitGenerationRequest(deps: GenerationDependencies, request: GenerationRequest, reference?: {
    sourceUrl: string;
    image: PixelImage;
    prompts: ForegroundPrompt[];
}, context?: ExecutionContext): Promise<{
    decision: import("./types.js").RouteDecision;
    job: import("./types.js").GenerationJob;
}>;
//# sourceMappingURL=submit-generation-request.d.ts.map