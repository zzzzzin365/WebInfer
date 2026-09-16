/**
 * WebInfer - Zero-shot Classification Pipeline
 * 
 * Classify text into any set of labels without fine-tuning,
 * using a real NLI (Natural Language Inference) model.
 */

import { BasePipeline, PipelineResult, registerPipeline } from './base.js';
import { WebInferTensor, softmax } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions, LoadedModel } from '../core/types.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';

// ============================================================================
// Default Model (DistilBART fine-tuned on MNLI)
// ============================================================================

const DEFAULT_MODELS = {
  model: 'https://huggingface.co/Xenova/nli-deberta-v3-small/resolve/main/onnx/model_quantized.onnx',
  tokenizer: 'https://huggingface.co/Xenova/nli-deberta-v3-small/resolve/main/tokenizer.json',
};

// NLI output indices: [contradiction, neutral, entailment]
const ENTAILMENT_IDX = 2;

// ============================================================================
// Types
// ============================================================================

export interface ZeroShotClassificationOptions extends PipelineOptions {
  multiLabel?: boolean;
  hypothesisTemplate?: string;
}

export interface ZeroShotClassificationResult extends PipelineResult {
  sequence: string;
  labels: string[];
  scores: number[];
}

export interface ZeroShotInput {
  text: string | string[];
  candidateLabels: string[];
}

// ============================================================================
// Zero-shot Classification Pipeline
// ============================================================================

export class ZeroShotClassificationPipeline extends BasePipeline<
  ZeroShotInput,
  ZeroShotClassificationResult | ZeroShotClassificationResult[]
> {
  private tokenizer: Tokenizer | null = null;
  private onnxModel: LoadedModel | null = null;
  private hypothesisTemplate: string = 'This text is about {label}.';
  private modelUrl: string;
  private tokenizerUrl: string;

  constructor(config?: PipelineConfig) {
    super(config ?? {
      task: 'zero-shot-classification',
      model: 'default',
    });
    this.modelUrl = (config?.model && config.model !== 'default') ? config.model : DEFAULT_MODELS.model;
    this.tokenizerUrl = DEFAULT_MODELS.tokenizer;
  }

  override async initialize(): Promise<void> {
    if (!this.tokenizer) {
      this.tokenizer = await Tokenizer.fromUrl(this.tokenizerUrl);
    }

    if (!this.onnxModel) {
      const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
      this.onnxModel = await this.inference.loadModelFromBuffer(modelData, { runtime: this.config.runtime });
    }
    this.isReady = true;
  }

  setTokenizer(tokenizer: Tokenizer): void {
    this.tokenizer = tokenizer;
  }

  async classify(
    text: string | string[],
    candidateLabels: string[],
    options?: ZeroShotClassificationOptions
  ): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]> {
    return this.run({ text, candidateLabels }, options);
  }

  override async run(
    input: ZeroShotInput,
    options?: PipelineOptions
  ): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]> {
    await this.initialize();

    const { text, candidateLabels } = input;
    const opts = options as ZeroShotClassificationOptions ?? {};
    const texts = Array.isArray(text) ? text : [text];
    const template = opts.hypothesisTemplate ?? this.hypothesisTemplate;
    const multiLabel = opts.multiLabel ?? false;

    const results = await Promise.all(
      texts.map(t => this.classifySingle(t, candidateLabels, template, multiLabel, options))
    );

    return Array.isArray(text) ? results : results[0]!;
  }

  private async classifySingle(
    text: string,
    candidateLabels: string[],
    template: string,
    multiLabel: boolean,
    options?: PipelineOptions
  ): Promise<ZeroShotClassificationResult> {
    const startTime = performance.now();

    const hypotheses = candidateLabels.map(label =>
      template.replace('{label}', label)
    );

    const scores: number[] = [];

    for (const hypothesis of hypotheses) {
      const score = await this.scoreHypothesis(text, hypothesis, options);
      scores.push(score);
    }

    let normalizedScores: number[];

    if (multiLabel) {
      normalizedScores = scores.map(s => 1 / (1 + Math.exp(-s)));
    } else {
      const tensor = new WebInferTensor(new Float32Array(scores), [scores.length], 'float32');
      const probs = softmax(tensor);
      normalizedScores = Array.from(probs.toFloat32Array());
      probs.dispose(); tensor.dispose();
    }

    const indexed = candidateLabels.map((label, i) => ({
      label,
      score: normalizedScores[i] ?? 0,
    }));
    indexed.sort((a, b) => b.score - a.score);

    return {
      sequence: text,
      labels: indexed.map(i => i.label),
      scores: indexed.map(i => i.score),
      processingTime: performance.now() - startTime,
    };
  }

  /**
   * Score a single hypothesis using the real NLI ONNX model.
   * Returns the entailment logit.
   */
  private async scoreHypothesis(premise: string, hypothesis: string, options?: PipelineOptions): Promise<number> {
    const encoded = this.tokenizer!.encode(premise, {
      textPair: hypothesis,
      addSpecialTokens: true,
      maxLength: 512,
      truncation: true,
      returnAttentionMask: true,
    });

    const inputIds = new WebInferTensor(
      BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
      [1, encoded.inputIds.length],
      'int64'
    );
    const attentionMask = new WebInferTensor(
      BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))),
      [1, encoded.attentionMask.length],
      'int64'
    );

    const namedInputs = new Map<string, WebInferTensor>();
    namedInputs.set('input_ids', inputIds);
    namedInputs.set('attention_mask', attentionMask);

    let outputs: import('../core/types.js').Tensor[] = [];
    try {
      outputs = await this.inference.runInferenceNamed(this.onnxModel!, namedInputs, options);
      return (outputs[0] as WebInferTensor).toFloat32Array()[ENTAILMENT_IDX] ?? 0;
    } finally {
      inputIds.dispose(); attentionMask.dispose(); outputs.forEach(t => t.dispose());
    }
  }

  override dispose(): void {
    this.onnxModel?.dispose();
    this.onnxModel = null;
    super.dispose();
  }

  protected async preprocess(
    input: ZeroShotInput
  ): Promise<WebInferTensor[]> {
    const { text, candidateLabels } = input;
    const firstText = Array.isArray(text) ? text[0] ?? '' : text;
    const firstLabel = candidateLabels[0] ?? '';

    const encoded = this.tokenizer!.encode(firstText, {
      textPair: this.hypothesisTemplate.replace('{label}', firstLabel),
      addSpecialTokens: true,
      maxLength: 512,
    });

    return [new WebInferTensor(
      BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
      [1, encoded.inputIds.length],
      'int64'
    )];
  }

  protected async postprocess(
    _outputs: WebInferTensor[],
    _options?: PipelineOptions
  ): Promise<ZeroShotClassificationResult | ZeroShotClassificationResult[]> {
    return {
      sequence: '',
      labels: [],
      scores: [],
    };
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createZeroShotClassificationPipeline(
  config?: PipelineConfig
): ZeroShotClassificationPipeline {
  return new ZeroShotClassificationPipeline(config);
}

registerPipeline('zero-shot-classification', (config) => new ZeroShotClassificationPipeline(config));
