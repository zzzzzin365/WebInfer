import { createONNXWorkerHandler } from './onnx-worker-handler.js';
import type { RuntimeWorkerRequest } from './worker-runtime.js';
const endpoint = self as unknown as DedicatedWorkerGlobalScope;
const handle = createONNXWorkerHandler((response, transfers) => endpoint.postMessage(response, transfers ?? []));
endpoint.onmessage = (event: MessageEvent<RuntimeWorkerRequest>) => { void handle(event.data); };
