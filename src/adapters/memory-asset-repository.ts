import type { AnalysisWriteGuard, AssetAnalysis, AssetRef, AssetRepository } from '../workflows-3d/types.js';

/** Demo/reference adapter. Volatile storage, not a replacement for the Library service. */
export class MemoryAssetRepository implements AssetRepository {
  private readonly analyses = new Map<string, AssetAnalysis>();
  private key(asset: AssetRef): string { return JSON.stringify([asset.id, asset.version]); }
  async updateAnalysis(asset: AssetRef, analysis: AssetAnalysis, guard: AnalysisWriteGuard): Promise<boolean> {
    if (guard.signal?.aborted || !guard.isCurrent()) return false;
    if (this.key(asset) !== this.key(analysis.asset)) throw new Error('Analysis asset version mismatch');
    this.analyses.set(this.key(asset), structuredClone(analysis));
    return true;
  }
  async getAnalysis(asset: AssetRef): Promise<AssetAnalysis | undefined> {
    const analysis = this.analyses.get(this.key(asset));
    return analysis ? structuredClone(analysis) : undefined;
  }
}
