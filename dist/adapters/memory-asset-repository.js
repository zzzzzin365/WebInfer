/** Demo/reference adapter. Volatile storage, not a replacement for the Library service. */
export class MemoryAssetRepository {
    analyses = new Map();
    key(asset) { return JSON.stringify([asset.id, asset.version]); }
    async updateAnalysis(asset, analysis, guard) {
        if (guard.signal?.aborted || !guard.isCurrent())
            return false;
        if (this.key(asset) !== this.key(analysis.asset))
            throw new Error('Analysis asset version mismatch');
        this.analyses.set(this.key(asset), structuredClone(analysis));
        return true;
    }
    async getAnalysis(asset) {
        const analysis = this.analyses.get(this.key(asset));
        return analysis ? structuredClone(analysis) : undefined;
    }
}
//# sourceMappingURL=memory-asset-repository.js.map