import type { AssetClassifier, AssetRef, AssetRepository, ViewerAdapter } from './types.js';
/** Latest request wins. Interaction stops new background work and invalidates old results. */
export declare class ViewerAnalysisController {
    private readonly viewer;
    private readonly classifier;
    private readonly repository;
    private readonly onError;
    private readonly delayMs;
    private asset;
    private scope?;
    private revision;
    private interacting;
    private closed;
    private timer?;
    private readonly cleanups;
    private readonly unsubscribe;
    constructor(viewer: ViewerAdapter, classifier: AssetClassifier, repository: AssetRepository, onError: (error: unknown) => void, delayMs?: number);
    setAsset(asset: AssetRef | null): void;
    private invalidate;
    requestAnalysis(): void;
    dispose(): Promise<void>;
}
//# sourceMappingURL=viewer-analysis-controller.d.ts.map