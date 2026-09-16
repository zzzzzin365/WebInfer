# transformers.js Adapter

Use WebInfer as an orchestration layer on top of [transformers.js](https://huggingface.co/docs/transformers.js) to access 1000+ HuggingFace models with scheduling, caching, and memory management.

## Installation

```bash
npm install webinfer @xenova/transformers
```

## Setup

```typescript
import { pipeline as tfPipeline } from '@xenova/transformers';
import { useTransformersBackend, pipeline, configureScheduler } from 'webinfer';

// Register transformers.js as the inference backend
useTransformersBackend({
  pipelineFactory: tfPipeline,
  device: 'webgpu',    // GPU acceleration
  dtype: 'fp16',       // Half precision for speed
});

// Optional: configure WebInfer scheduling
configureScheduler({
  maxConcurrentTasks: 4,
  maxConcurrentPerModel: 1,
  maxRetries: 2,
  circuitBreaker: true,
});
```

## Usage

After setup, use the standard WebInfer API. All inference calls go through transformers.js but with WebInfer orchestration:

```typescript
// Sentiment analysis with scheduling + caching
const classifier = await pipeline('text-classification', {
  model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
});

const result = await classifier.run('I love WebInfer!');
```

## Why Use the Adapter?

| Scenario | transformers.js alone | With WebInfer adapter |
|----------|----------------------|--------------------------|
| Run 5 models at once | Uncontrolled memory | Scheduled with limits |
| Same input repeated | Recomputed | Cached |
| Model download interrupted | Restart | Resume from last chunk |
| Task cancellation | Not possible | `task.cancel()` |
| Performance monitoring | Manual | Built-in dashboard |

## Advanced: Direct Pipeline Access

For advanced use, access the transformers.js pipeline directly:

```typescript
import { getTransformersAdapter } from 'webinfer';

const adapter = getTransformersAdapter();
if (adapter) {
  const modelId = await adapter.loadPipeline(
    'text-classification',
    'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
  );

  const result = await adapter.runDirect(modelId, 'Hello world');
}
```
