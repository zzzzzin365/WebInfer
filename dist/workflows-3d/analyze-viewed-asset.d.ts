import { type ExecutionContext } from '../core/inference-client.js';
import type { AssetAnalysis, AssetClassifier, AssetRef, ViewerAdapter } from './types.js';
/** Analysis only; persistence is a separate, version-checked operation. */
export declare function analyzeViewedAsset(viewer: ViewerAdapter, classifier: AssetClassifier, asset: AssetRef, context?: ExecutionContext, minConfidence?: number): Promise<AssetAnalysis>;
//# sourceMappingURL=analyze-viewed-asset.d.ts.map