import { createONNXWorkerHandler } from './onnx-worker-handler.js';
const endpoint = self;
const handle = createONNXWorkerHandler((response, transfers) => endpoint.postMessage(response, transfers ?? []));
endpoint.onmessage = (event) => { void handle(event.data); };
//# sourceMappingURL=onnx-worker-entry.js.map