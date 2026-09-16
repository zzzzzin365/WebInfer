import { type ExecutionContext } from '../core/inference-client.js';
import type { ForegroundPrompt, PixelImage, Segmenter } from './types.js';
export type PreparationResult = {
    status: 'prepared';
    original: PixelImage;
    image: PixelImage;
} | {
    status: 'failed';
    original: PixelImage;
    error: Error;
};
/** Interactive foreground extraction; keeps the original intact on failure. */
export declare function prepareReferenceImage(segmenter: Segmenter, original: PixelImage, prompts: ForegroundPrompt[], context?: ExecutionContext, maxDimension?: number): Promise<PreparationResult>;
//# sourceMappingURL=prepare-reference-image.d.ts.map