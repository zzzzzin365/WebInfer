import { TaskScope } from '../core/task-scope.js';
import { analyzeViewedAsset } from './analyze-viewed-asset.js';
import type { AssetClassifier, AssetRef, AssetRepository, ViewerAdapter } from './types.js';

/** Latest request wins. Interaction stops new background work and invalidates old results. */
export class ViewerAnalysisController {
  private asset: AssetRef | null = null;
  private scope?: TaskScope;
  private revision = 0;
  private interacting = false;
  private closed = false;
  private timer?: ReturnType<typeof setTimeout>;
  private readonly cleanups = new Set<Promise<void>>();
  private readonly unsubscribe: Array<() => void>;
  constructor(private readonly viewer: ViewerAdapter, private readonly classifier: AssetClassifier,
    private readonly repository: AssetRepository, private readonly onError: (error: unknown) => void,
    private readonly delayMs = 50) {
    this.unsubscribe = [viewer.onAssetChange(asset => this.setAsset(asset)),
      viewer.onInteractionChange(active => {
        this.interacting = active;
        this.invalidate();
        if (!active) this.requestAnalysis();
      })];
  }
  setAsset(asset: AssetRef | null): void {
    if (this.closed) return;
    this.invalidate();
    this.asset = asset ? { ...asset } : null;
    this.requestAnalysis();
  }
  private invalidate(): void {
    this.revision++;
    clearTimeout(this.timer);
    if (this.scope) {
      const cleanup = this.scope.dispose();
      this.cleanups.add(cleanup);
      void cleanup.then(() => this.cleanups.delete(cleanup));
      this.scope = undefined;
    }
  }
  requestAnalysis(): void {
    if (this.closed || !this.asset) return;
    this.invalidate();
    if (this.interacting) return;
    const asset = { ...this.asset }, revision = this.revision;
    this.timer = setTimeout(() => {
      const scope = new TaskScope(JSON.stringify([asset.id, asset.version, revision]));
      this.scope = scope;
      const isCurrent = () => !this.closed && scope.active && this.revision === revision;
      void scope.run(async context => {
        const analysis = await analyzeViewedAsset(this.viewer, this.classifier, asset, context);
        if (isCurrent()) await this.repository.updateAnalysis(asset, analysis, { signal: context.signal, isCurrent });
      }, 'low').catch(error => {
        if (scope.active) this.onError(error);
      }).finally(() => {
        if (this.scope === scope) this.scope = undefined;
        return scope.dispose();
      });
    }, this.delayMs);
  }
  async dispose(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      this.unsubscribe.forEach(unsubscribe => unsubscribe());
      this.invalidate();
    }
    await Promise.all(this.cleanups);
  }
}
