import { throwIfAborted, type ExecutionContext } from '../core/inference-client.js';
import type { ForegroundPrompt, PixelImage, Segmenter } from './types.js';

export type PreparationResult =
  | { status: 'prepared'; original: PixelImage; image: PixelImage }
  | { status: 'failed'; original: PixelImage; error: Error };

/** Interactive foreground extraction; keeps the original intact on failure. */
export async function prepareReferenceImage(
  segmenter: Segmenter, original: PixelImage, prompts: ForegroundPrompt[],
  context: ExecutionContext = {}, maxDimension = 1024,
): Promise<PreparationResult> {
  const execution = { ...context, priority: context.priority ?? 'high' as const };
  throwIfAborted(execution.signal);
  try {
    const { width, height, data } = original;
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
      data.length !== width * height * 4 || !Number.isInteger(maxDimension) || maxDimension <= 0) throw new Error('Invalid image dimensions');
    if (!prompts.some(p => p.label === 1) || prompts.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y) ||
      p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) throw new Error('A foreground point is required');
    const mask = await segmenter.segment(original, prompts, execution);
    throwIfAborted(execution.signal);
    if (!Number.isInteger(mask.width) || !Number.isInteger(mask.height) || mask.width <= 0 || mask.height <= 0 ||
      mask.data.length !== mask.width * mask.height) throw new Error('Invalid foreground mask');
    const alphaAt = (x: number, y: number) => mask.data[Math.floor(y * mask.height / height) * mask.width + Math.floor(x * mask.width / width)]!;
    let left = width, top = height, right = -1, bottom = -1;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      if (alphaAt(x, y) && data[(y * width + x) * 4 + 3]) {
        left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y);
      }
    }
    if (right < left) throw new Error('No foreground found');
    const cropWidth = right - left + 1, cropHeight = bottom - top + 1;
    const scale = Math.min(1, maxDimension / Math.max(cropWidth, cropHeight));
    const outWidth = Math.max(1, Math.round(cropWidth * scale)), outHeight = Math.max(1, Math.round(cropHeight * scale));
    const pixels = new Uint8ClampedArray(outWidth * outHeight * 4);
    for (let y = 0; y < outHeight; y++) for (let x = 0; x < outWidth; x++) {
      const sx = left + Math.min(cropWidth - 1, Math.floor(x / scale));
      const sy = top + Math.min(cropHeight - 1, Math.floor(y / scale));
      const source = (sy * width + sx) * 4, target = (y * outWidth + x) * 4;
      pixels.set(data.subarray(source, source + 4), target);
      pixels[target + 3] = Math.round(data[source + 3]! * alphaAt(sx, sy) / 255);
    }
    return { status: 'prepared', original, image: { width: outWidth, height: outHeight, data: pixels } };
  } catch (error) {
    throwIfAborted(execution.signal);
    return { status: 'failed', original, error: error instanceof Error ? error : new Error(String(error)) };
  }
}
