import { MemoryScope } from './memory.js';
import { throwIfAborted, type ExecutionContext } from './inference-client.js';
import type { TaskPriority } from './types.js';

/** Generic lifetime: invalidation is immediate, cleanup waits for running work. */
export class TaskScope {
  private readonly controller = new AbortController();
  private readonly resources = new MemoryScope();
  private readonly pending = new Set<Promise<unknown>>();
  private closing?: Promise<void>;
  constructor(readonly id: string) {}
  get active(): boolean { return !this.controller.signal.aborted; }
  context(priority: TaskPriority = 'normal'): ExecutionContext {
    return { scopeId: this.id, signal: this.controller.signal, priority };
  }
  track<T extends { dispose(): void }>(resource: T): T {
    if (!this.active && this.pending.size === 0) { resource.dispose(); throwIfAborted(this.controller.signal); }
    return this.resources.track(resource);
  }
  run<T>(fn: (context: ExecutionContext) => Promise<T>, priority: TaskPriority = 'normal'): Promise<T> {
    if (!this.active) return Promise.reject(new DOMException('Scope is closed', 'AbortError'));
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
  dispose(): Promise<void> {
    if (this.closing) return this.closing;
    this.controller.abort();
    this.closing = (async () => {
      await Promise.allSettled(this.pending);
      this.resources.dispose();
    })();
    return this.closing;
  }
}
