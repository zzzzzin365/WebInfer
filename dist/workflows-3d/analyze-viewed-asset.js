import { throwIfAborted } from '../core/inference-client.js';
/** Analysis only; persistence is a separate, version-checked operation. */
export async function analyzeViewedAsset(viewer, classifier, asset, context = {}, minConfidence = 0.7) {
    const execution = { ...context, priority: context.priority ?? 'low' };
    throwIfAborted(execution.signal);
    const snapshot = await viewer.capture(asset, execution);
    try {
        throwIfAborted(execution.signal);
        const result = await classifier.classify(snapshot.image, execution);
        throwIfAborted(execution.signal);
        const valid = Number.isFinite(result.confidence) && result.confidence >= 0 && result.confidence <= 1;
        return { asset: { ...asset }, category: valid && result.confidence >= minConfidence ? result.category : null,
            confidence: valid ? result.confidence : 0, sourceView: snapshot.view, analyzerVersion: classifier.version };
    }
    finally {
        snapshot.dispose();
    }
}
//# sourceMappingURL=analyze-viewed-asset.js.map