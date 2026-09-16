import type { ExecutionContext } from '../core/inference-client.js';

export interface AssetRef { id: string; version: string }
export interface PixelImage { width: number; height: number; data: Uint8ClampedArray }
export interface ForegroundPrompt { x: number; y: number; label: 0 | 1 }
export interface ForegroundMask { width: number; height: number; data: Uint8Array }
export interface GenerationRequest {
  text: string;
  attachments: Array<{ kind: 'image' | 'other'; url: string }>;
  assets: AssetRef[];
  context?: string;
  /** Explicit input from the application; unknown also routes to the Agent. */
  standalone?: boolean;
}
export type GenerationRoute = 'text-to-3d' | 'image-to-3d' | 'agent';
export interface RouteDecision { route: GenerationRoute; confidence: number; reason: string }
export interface AssetAnalysis {
  asset: AssetRef;
  category: string | null;
  confidence: number;
  sourceView: string;
  analyzerVersion: string;
}
export interface Snapshot { image: PixelImage; view: string; dispose(): void }
export interface GenerationJob { id: string }
export interface Segmenter {
  segment(image: PixelImage, prompts: ForegroundPrompt[], context: ExecutionContext): Promise<ForegroundMask>;
}
export interface IntentClassifier {
  classify(request: GenerationRequest, context: ExecutionContext): Promise<Array<{ route: GenerationRoute; score: number }>>;
}
export interface AssetClassifier {
  version: string;
  classify(image: PixelImage, context: ExecutionContext): Promise<{ category: string; confidence: number }>;
}
export interface ViewerAdapter {
  /** Capture after render is stable; reject if the requested asset is no longer displayed. */
  capture(asset: AssetRef, context: ExecutionContext): Promise<Snapshot>;
  onAssetChange(listener: (asset: AssetRef | null) => void): () => void;
  onInteractionChange(listener: (active: boolean) => void): () => void;
}
export interface AnalysisWriteGuard {
  signal?: AbortSignal;
  /** Must be checked at the write point, not only before asynchronous IO. */
  isCurrent(): boolean;
}
export interface AssetRepository {
  /** Must bind writes to id + version. Remote implementations need conditional writes. */
  updateAnalysis(asset: AssetRef, analysis: AssetAnalysis, guard: AnalysisWriteGuard): Promise<boolean>;
  getAnalysis(asset: AssetRef): Promise<AssetAnalysis | undefined>;
}
export interface GenerationService {
  textTo3D(input: { text: string }, context: ExecutionContext): Promise<GenerationJob>;
  imageTo3D(input: { text: string; imageUrl: string }, context: ExecutionContext): Promise<GenerationJob>;
}
export interface AgentGateway {
  submit(request: GenerationRequest, analysis: AssetAnalysis[], context: ExecutionContext): Promise<GenerationJob>;
}
export interface ReferenceUploader {
  upload(image: PixelImage, context: ExecutionContext): Promise<string>;
}
