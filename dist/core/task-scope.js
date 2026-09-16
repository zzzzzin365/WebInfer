import { MemoryScope } from './memory.js';
import { throwIfAborted } from './inference-client.js';
/** Generic lifetime: invalidation is immediate, cleanup waits for running work. */
export class TaskScope {
    id;
    controller = new AbortController();
    resources = new MemoryScope();
    pending = new Set();
    closing;
    constructor(id) {
        this.id = id;
    }
    get active() { return !this.controller.signal.aborted; }
    context(priority = 'normal') {
        return { scopeId: this.id, signal: this.controller.signal, priority };
    }
    track(resource) {
        if (!this.active && this.pending.size === 0) {
            resource.dispose();
            throwIfAborted(this.controller.signal);
        }
        return this.resources.track(resource);
    }
    run(fn, priority = 'normal') {
        if (!this.active)
            return Promise.reject(new DOMException('Scope is closed', 'AbortError'));
        const promise = Promise.resolve().then(async () => {
            throwIfAborted(this.controller.signal);
            const result = await fn(this.context(priority));
            throwIfAborted(this.controller.signal);
            return result;
        });
        this.pending.add(promise);
        void promise.then(() => this.pending.delete(promise), () => this.pending.delete(promise));
        return promise;
    }
    dispose() {
        if (this.closing)
            return this.closing;
        this.controller.abort();
        this.closing = (async () => {
            await Promise.allSettled(this.pending);
            this.resources.dispose();
        })();
        return this.closing;
    }
}
//# sourceMappingURL=task-scope.js.map