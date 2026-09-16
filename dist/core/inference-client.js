export function throwIfAborted(signal) {
    if (signal?.aborted)
        throw new DOMException('Inference cancelled', 'AbortError');
}
//# sourceMappingURL=inference-client.js.map