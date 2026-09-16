import { type ExecutionContext } from '../core/inference-client.js';
import type { GenerationRequest, IntentClassifier, RouteDecision } from './types.js';
export interface RoutingPolicy {
    minConfidence: number;
    minMargin: number;
}
/** Conservative prototype policy; tune with labeled business requests. */
export declare function routeGenerationRequest(classifier: IntentClassifier, request: GenerationRequest, context?: ExecutionContext, policy?: RoutingPolicy): Promise<RouteDecision>;
//# sourceMappingURL=route-generation-request.d.ts.map