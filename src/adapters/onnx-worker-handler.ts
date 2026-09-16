import { ONNXRuntime } from '../backends/onnx.js';
import type { LoadedModel } from '../core/types.js';
import { deserializeTensor, serializeTensor } from '../core/worker.js';
import type { RuntimeWorkerRequest, RuntimeWorkerResponse } from './worker-runtime.js';

/** Serializes messages so unload/dispose cannot overtake an in-flight session.run. */
export function createONNXWorkerHandler(send: (response: RuntimeWorkerResponse, transfers?: ArrayBuffer[]) => void) {
  const runtime = new ONNXRuntime();
  const models = new Map<string, LoadedModel>();
  let tail = Promise.resolve();
  return (message: RuntimeWorkerRequest): Promise<void> => {
    tail = tail.then(async () => {
      try {
        let value: unknown;
        let transfers: ArrayBuffer[] = [];
        switch (message.type) {
          case 'init': await runtime.initialize(); break;
          case 'load': {
            if (!message.data) throw new Error('Model data required');
            const model = await runtime.loadModel(message.data, message.options);
            models.set(model.id, model);
            value = { id: model.id, metadata: model.metadata };
            break;
          }
          case 'run': {
            const model = models.get(message.modelId!);
            if (!model) throw new Error('Worker model not found');
            const inputs = new Map(await Promise.all((message.inputs ?? []).map(async ([name, tensor]) => [name, await deserializeTensor(tensor)] as const)));
            try {
              const outputs = await runtime.runNamed(model, inputs);
              try {
                const serialized = outputs.map(serializeTensor);
                value = serialized; transfers = serialized.map(t => t.data);
              } finally { outputs.forEach(t => t.dispose()); }
            } finally { inputs.forEach(t => t.dispose()); }
            break;
          }
          case 'unload': models.get(message.modelId!)?.dispose(); models.delete(message.modelId!); break;
          case 'dispose':
            for (const model of models.values()) model.dispose();
            models.clear(); await runtime.dispose(); break;
          default: throw new Error('Unknown worker operation');
        }
        send({ id: message.id, value }, transfers);
      } catch (error) { send({ id: message.id, error: String(error) }); }
    });
    return tail;
  };
}
