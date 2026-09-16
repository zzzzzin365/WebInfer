import { type ExecutionContext } from './inference-client.js';
import type { TaskPriority } from './types.js';
/** Generic lifetime: invalidation is immediate, cleanup waits for running work. */
export declare class TaskScope {
    readonly id: string;
    private readonly controller;
    private readonly resources;
    private readonly pending;
    private closing?;
    constructor(id: string);
    get active(): boolean;
    context(priority?: TaskPriority): ExecutionContext;
    track<T extends {
        dispose(): void;
    }>(resource: T): T;
    run<T>(fn: (context: ExecutionContext) => Promise<T>, priority?: TaskPriority): Promise<T>;
    dispose(): Promise<void>;
}
//# sourceMappingURL=task-scope.d.ts.map