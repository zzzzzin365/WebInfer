import { TaskScope } from '../core/task-scope.js';
import { analyzeViewedAsset } from './analyze-viewed-asset.js';
/** Latest request wins. Interaction stops new background work and invalidates old results. */
export class ViewerAnalysisController {
    viewer;
    classifier;
    repository;
    onError;
    delayMs;
    asset = null;
    scope;
    revision = 0;
    interacting = false;
    closed = false;
    timer;
    cleanups = new Set();
    unsubscribe;
    constructor(viewer, classifier, repository, onError, delayMs = 50) {
        this.viewer = viewer;
        this.classifier = classifier;
        this.repository = repository;
        this.onError = onError;
        this.delayMs = delayMs;
        this.unsubscribe = [viewer.onAssetChange(asset => this.setAsset(asset)),
            viewer.onInteractionChange(active => {
                this.interacting = active;
                this.invalidate();
                if (!active)
                    this.requestAnalysis();
            })];
    }
    setAsset(asset) {
        if (this.closed)
            return;
        this.invalidate();
        this.asset = asset ? { ...asset } : null;
        this.requestAnalysis();
    }
    invalidate() {
        this.revision++;
        clearTimeout(this.timer);
        if (this.scope) {
            const cleanup = this.scope.dispose();
            this.cleanups.add(cleanup);
            void cleanup.then(() => this.cleanups.delete(cleanup));
            this.scope = undefined;
        }
    }
    requestAnalysis() {
        if (this.closed || !this.asset)
            return;
        this.invalidate();
        if (this.interacting)
            return;
        const asset = { ...this.asset }, revision = this.revision;
        this.timer = setTimeout(() => {
            const scope = new TaskScope(JSON.stringify([asset.id, asset.version, revision]));
            this.scope = scope;
            const isCurrent = () => !this.closed && scope.active && this.revision === revision;
            void scope.run(async (context) => {
                const analysis = await analyzeViewedAsset(this.viewer, this.classifier, asset, context);
                if (isCurrent())
                    await this.repository.updateAnalysis(asset, analysis, { signal: context.signal, isCurrent });
            }, 'low').catch(error => {
                if (scope.active)
                    this.onError(error);
            }).finally(() => {
                if (this.scope === scope)
                    this.scope = undefined;
                return scope.dispose();
            });
        }, this.delayMs);
    }
    async dispose() {
        if (!this.closed) {
            this.closed = true;
            this.unsubscribe.forEach(unsubscribe => unsubscribe());
            this.invalidate();
        }
        await Promise.all(this.cleanups);
    }
}
//# sourceMappingURL=viewer-analysis-controller.js.map