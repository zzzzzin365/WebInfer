import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { MemoryManager, MemoryScope } from '../../src/core/memory';

describe('dependency and resource boundaries', () => {
  it('keeps core independent of pipelines, backends and 3D workflows', () => {
    for (const name of readdirSync(resolve('src/core')).filter(name => name.endsWith('.ts'))) {
      const source = readFileSync(resolve('src/core', name), 'utf8');
      expect(source, name).not.toMatch(/(?:from\s*|import\s*\()['"]\.\.\/(?:pipelines|backends|adapters|workflows-3d)\//);
    }
  });
  it('keeps 3D workflows independent of concrete adapters and network calls', () => {
    for (const name of readdirSync(resolve('src/workflows-3d')).filter(name => name.endsWith('.ts'))) {
      const source = readFileSync(resolve('src/workflows-3d', name), 'utf8');
      expect(source, name).not.toMatch(/(?:from\s*|import\s*\()['"]\.\.\/(?:pipelines|backends|adapters)\//);
      expect(source, name).not.toMatch(/\bfetch\s*\(/);
    }
  });
  it('disposing an independent manager does not reset the legacy singleton', () => {
    const legacy = MemoryManager.getInstance();
    new MemoryManager().dispose();
    expect(MemoryManager.getInstance()).toBe(legacy);
  });
  it('cleans every child scope even as children detach from the parent', () => {
    const parent = new MemoryScope();
    const disposed: number[] = [];
    for (const i of [1, 2, 3]) parent.createChild().track({ dispose: () => disposed.push(i) });
    parent.dispose(); parent.dispose();
    expect(disposed).toEqual([1, 2, 3]);
  });
});
