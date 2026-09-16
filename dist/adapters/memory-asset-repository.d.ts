import type { AnalysisWriteGuard, AssetAnalysis, AssetRef, AssetRepository } from '../workflows-3d/types.js';
/** Demo/reference adapter. Volatile storage, not a replacement for the Library service. */
export declare class MemoryAssetRepository implements AssetRepository {
    private readonly analyses;
    private key;
    updateAnalysis(asset: AssetRef, analysis: AssetAnalysis, guard: AnalysisWriteGuard): Promise<boolean>;
    getAnalysis(asset: AssetRef): Promise<AssetAnalysis | undefined>;
}
//# sourceMappingURL=memory-asset-repository.d.ts.map