import { throwIfAborted, type ExecutionContext } from '../core/inference-client.js';
import type { GenerationRequest, IntentClassifier, RouteDecision } from './types.js';

export interface RoutingPolicy { minConfidence: number; minMargin: number }
/** Conservative prototype policy; tune with labeled business requests. */
export async function routeGenerationRequest(classifier: IntentClassifier, request: GenerationRequest,
  context: ExecutionContext = {}, policy: RoutingPolicy = { minConfidence: 0.85, minMargin: 0.2 }): Promise<RouteDecision> {
  throwIfAborted(context.signal);
  const fallback = (reason: string): RouteDecision => ({ route: 'agent', confidence: 0, reason });
  if (!request.standalone || request.assets.length || request.context?.trim()) return fallback('context-required');
  if (!request.text.trim() || request.attachments.length > 1 || request.attachments.some(a => a.kind !== 'image' || !a.url)) return fallback('incomplete-or-unsupported-input');
  if (!Number.isFinite(policy.minConfidence) || policy.minConfidence < 0 || policy.minConfidence > 1 ||
    !Number.isFinite(policy.minMargin) || policy.minMargin < 0 || policy.minMargin > 1) throw new Error('Invalid routing policy');
  try {
    const scores = await classifier.classify(request, { ...context, priority: context.priority ?? 'high' });
    throwIfAborted(context.signal);
    if (scores.length !== 3 || new Set(scores.map(s => s.route)).size !== 3 || scores.some(s =>
      !['text-to-3d', 'image-to-3d', 'agent'].includes(s.route) || !Number.isFinite(s.score) || s.score < 0 || s.score > 1)) return fallback('invalid-classifier-output');
    const ranked = [...scores].sort((a, b) => b.score - a.score);
    const best = ranked[0]!, second = ranked[1]!;
    if (best.route === 'agent' || best.score < policy.minConfidence || best.score - second.score < policy.minMargin) return fallback('complex-or-uncertain');
    const expected = request.attachments.length ? 'image-to-3d' : 'text-to-3d';
    if (best.route !== expected) return fallback('input-route-mismatch');
    return { route: best.route, confidence: best.score, reason: 'confident-standalone-request' };
  } catch {
    throwIfAborted(context.signal);
    return fallback('classifier-failed');
  }
}
