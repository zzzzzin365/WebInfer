import type { RuntimeWorkerRequest, RuntimeWorkerResponse } from './worker-runtime.js';
/** Serializes messages so unload/dispose cannot overtake an in-flight session.run. */
export declare function createONNXWorkerHandler(send: (response: RuntimeWorkerResponse, transfers?: ArrayBuffer[]) => void): (message: RuntimeWorkerRequest) => Promise<void>;
//# sourceMappingURL=onnx-worker-handler.d.ts.map