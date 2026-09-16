import { throwIfAborted, type InferenceClient } from '../core/inference-client.js';
import { ImageSegmentationPipeline } from '../pipelines/image-segmentation.js';
import { ImageClassificationPipeline } from '../pipelines/image-classification.js';
import { ZeroShotClassificationPipeline } from '../pipelines/zero-shot-classification.js';
import type { AssetClassifier, GenerationRoute, IntentClassifier, PixelImage, Segmenter } from '../workflows-3d/types.js';

export interface PipelineAdapterOptions {
  engine: InferenceClient;
  assetModel: string;
  assetLabels: string[];
  analyzerVersion: string;
  /** NLI model must match the pipeline's tokenizer and entailment output convention. */
  intentModel?: string;
  segmentationModels?: { encoder: string; decoder: string };
}
const intentLabels: Record<GenerationRoute, string> = {
  'text-to-3d': 'a standalone request to generate a new 3D object from text',
  'image-to-3d': 'a standalone request to generate a new 3D object from a reference image',
  agent: 'a complex operation requiring existing assets, conversation context or multiple steps',
};
const toImageData = (image: PixelImage) => new ImageData(new Uint8ClampedArray(image.data), image.width, image.height);

/** Lazy, single-flight model initialization; application owns the shared model lifetime. */
export function createPipelineInferenceAdapters(options: PipelineAdapterOptions): {
  segmenter: Segmenter; intentClassifier: IntentClassifier; assetClassifier: AssetClassifier; dispose(): Promise<void>;
} {
  const sam = new ImageSegmentationPipeline({ engine: options.engine, task: 'image-segmentation', model: 'default' });
  if (options.segmentationModels) sam.setModelUrls(options.segmentationModels.encoder, options.segmentationModels.decoder);
  const intent = new ZeroShotClassificationPipeline({ engine: options.engine, task: 'zero-shot-classification', model: options.intentModel ?? 'default' });
  const asset = new ImageClassificationPipeline({ engine: options.engine, task: 'image-classification', model: options.assetModel }, options.assetLabels);
  let samInit: Promise<void> | undefined, intentInit: Promise<void> | undefined, assetInit: Promise<void> | undefined;
  let samTail: Promise<unknown> = Promise.resolve();
  let closed = false;
  const pending = new Set<Promise<unknown>>();
  const run = <T>(fn: () => Promise<T>): Promise<T> => {
    if (closed) return Promise.reject(new Error('Pipeline adapters are disposed'));
    const promise = fn(); pending.add(promise);
    void promise.then(() => pending.delete(promise), () => pending.delete(promise));
    return promise;
  };
  return {
    segmenter: { segment: (image, prompts, context) => run(() => {
      // SAM holds embeddings between encoder and decoder: serialize the entire pair.
      const operation = samTail.catch(() => undefined).then(async () => {
        throwIfAborted(context.signal);
        await (samInit ??= sam.loadModels().catch(error => { sam.dispose(); samInit = undefined; throw error; }));
        throwIfAborted(context.signal);
        try {
          const result = await sam.run(toImageData(image), { ...context, points: prompts });
          return { width: result.width, height: result.height, data: result.mask };
        } finally { sam.clearImage(); }
      });
      samTail = operation;
      return operation;
    }) },
    intentClassifier: { classify: (request, context) => run(async () => {
      throwIfAborted(context.signal);
      await (intentInit ??= intent.initialize().catch(error => { intentInit = undefined; throw error; }));
      throwIfAborted(context.signal);
      const result = await intent.classify(request.text, Object.values(intentLabels), context);
      if (Array.isArray(result)) throw new Error('Expected one intent result');
      return (Object.keys(intentLabels) as GenerationRoute[]).map(route => ({ route,
        score: result.scores[result.labels.indexOf(intentLabels[route])] ?? 0 }));
    }) },
    assetClassifier: { version: options.analyzerVersion, classify: (image, context) => run(async () => {
      throwIfAborted(context.signal);
      await (assetInit ??= asset.initialize().catch(error => { assetInit = undefined; throw error; }));
      throwIfAborted(context.signal);
      const result = await asset.run(toImageData(image), context);
      if (Array.isArray(result)) throw new Error('Expected one asset result');
      return { category: result.label, confidence: result.score };
    }) },
    async dispose() {
      closed = true;
      await Promise.allSettled(pending);
      sam.dispose(); intent.dispose(); asset.dispose();
    },
  };
}
