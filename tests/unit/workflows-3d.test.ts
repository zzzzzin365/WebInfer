import { describe, it, expect, vi } from 'vitest';
import { prepareReferenceImage, routeGenerationRequest, submitGenerationRequest, ViewerAnalysisController } from '../../src/workflows-3d';
import { MemoryAssetRepository } from '../../src/adapters/memory-asset-repository';
import type { GenerationRequest, IntentClassifier, ViewerAdapter, PixelImage } from '../../src/workflows-3d';

const image: PixelImage = { width: 2, height: 2, data: new Uint8ClampedArray([255,0,0,255, 0,255,0,255, 0,0,255,255, 255,255,255,255]) };
const request: GenerationRequest = { text: 'Generate a chair', attachments: [], assets: [], standalone: true };
const classifier: IntentClassifier = { classify: vi.fn(async () => [
  { route: 'text-to-3d', score: .95 }, { route: 'image-to-3d', score: .02 }, { route: 'agent', score: .03 },
]) };
const segmenter = { segment: vi.fn(async () => ({ width: 2, height: 2, data: new Uint8Array([0,255,0,0]) })) };

describe('3D workflows', () => {
  it('composites and crops the foreground without mutating the original', async () => {
    const result = await prepareReferenceImage(segmenter, image, [{ x: .75, y: .25, label: 1 }]);
    expect(result.status).toBe('prepared');
    if (result.status === 'prepared') expect(result.image).toEqual({ width: 1, height: 1, data: new Uint8ClampedArray([0,255,0,255]) });
    expect(image.data[3]).toBe(255);
    expect(segmenter.segment).toHaveBeenLastCalledWith(image, expect.anything(), expect.objectContaining({ priority: 'high' }));
  });
  it('preserves the original on segmentation failure and rejects cancellation', async () => {
    const failed = await prepareReferenceImage({ segment: async () => { throw new Error('offline'); } }, image, [{ x: .5, y: .5, label: 1 }]);
    expect(failed).toMatchObject({ status: 'failed', original: image });
    const abort = new AbortController(); abort.abort();
    await expect(prepareReferenceImage(segmenter, image, [], { signal: abort.signal })).rejects.toThrow();
  });
  it('routes only confident standalone requests directly', async () => {
    expect((await routeGenerationRequest(classifier, request)).route).toBe('text-to-3d');
    for (const patch of [{ standalone: false }, { context: 'previous chair' }, { assets: [{ id: 'a', version: '1' }] },
      { attachments: [{ kind: 'image' as const, url: 'reference' }] }]) {
      expect((await routeGenerationRequest(classifier, { ...request, ...patch })).route).toBe('agent');
    }
    const uncertain: IntentClassifier = { classify: async () => [{ route: 'text-to-3d', score: .5 }, { route: 'image-to-3d', score: .1 }, { route: 'agent', score: .4 }] };
    expect((await routeGenerationRequest(uncertain, request)).route).toBe('agent');
    expect((await routeGenerationRequest({ classify: async () => { throw new Error('offline'); } }, request)).route).toBe('agent');
  });
  it('submits the processed upload and never invokes the Agent on the direct image path', async () => {
    const imageClassifier: IntentClassifier = { classify: async () => [{ route: 'text-to-3d', score: .01 }, { route: 'image-to-3d', score: .98 }, { route: 'agent', score: .01 }] };
    const generation = { textTo3D: vi.fn(), imageTo3D: vi.fn(async () => ({ id: 'job' })) };
    const uploader = { upload: vi.fn(async () => 'processed.png') }, agent = { submit: vi.fn() };
    const deps = { classifier: imageClassifier, segmenter, uploader, generation, agent, assets: new MemoryAssetRepository() };
    const input = { ...request, attachments: [{ kind: 'image' as const, url: 'original.png' }] };
    await submitGenerationRequest(deps, input, { sourceUrl: 'original.png', image, prompts: [{ x: .75, y: .25, label: 1 }] });
    expect(uploader.upload.mock.calls[0][0]).toMatchObject({ width: 1, height: 1 });
    expect(generation.imageTo3D).toHaveBeenCalledWith({ text: request.text, imageUrl: 'processed.png' }, {});
    expect(agent.submit).not.toHaveBeenCalled();
    await expect(submitGenerationRequest({ ...deps, segmenter: { segment: async () => { throw new Error('bad mask'); } } }, input,
      { sourceUrl: 'original.png', image, prompts: [{ x: .5, y: .5, label: 1 }] })).rejects.toThrow('bad mask');
    expect(generation.imageTo3D).toHaveBeenCalledTimes(1);
  });
  it('provides only the attached asset version metadata to the Agent', async () => {
    const assets = new MemoryAssetRepository();
    const asset = { id: 'a', version: '2' };
    await assets.updateAnalysis(asset, { asset, category: 'chair', confidence: .9, sourceView: 'front', analyzerVersion: '1' }, { isCurrent: () => true });
    const agent = { submit: vi.fn(async () => ({ id: 'agent-job' })) };
    await submitGenerationRequest({ classifier, segmenter, uploader: { upload: vi.fn() }, generation: { textTo3D: vi.fn(), imageTo3D: vi.fn() }, agent, assets }, { ...request, assets: [asset] });
    expect(agent.submit.mock.calls[0][1]).toEqual([expect.objectContaining({ category: 'chair', asset })]);
    expect(await assets.getAnalysis({ id: 'a', version: '1' })).toBeUndefined();
  });
});

describe('Viewer lifecycle', () => {
  it('discards A after switching to B, merges interactions, releases both snapshots and unsubscribes', async () => {
    let changed!: Parameters<ViewerAdapter['onAssetChange']>[0];
    let interaction!: (active: boolean) => void;
    let finishA!: (value: { category: string; confidence: number }) => void;
    const gateA = new Promise<{ category: string; confidence: number }>(r => { finishA = r; });
    const disposeA = vi.fn(), disposeB = vi.fn(), unsubscribe = vi.fn();
    const viewer: ViewerAdapter = {
      capture: vi.fn(async asset => ({ image, view: asset.id, dispose: asset.id === 'a' ? disposeA : disposeB })),
      onAssetChange: listener => { changed = listener; return unsubscribe; },
      onInteractionChange: listener => { interaction = listener; return unsubscribe; },
    };
    let calls = 0;
    const analyze = { version: '1', classify: vi.fn(async () => ++calls === 1 ? gateA : { category: 'table', confidence: .9 }) };
    const repository = new MemoryAssetRepository(), onError = vi.fn();
    const controller = new ViewerAnalysisController(viewer, analyze, repository, onError, 0);
    changed({ id: 'a', version: '1' });
    await vi.waitFor(() => expect(analyze.classify).toHaveBeenCalledTimes(1));
    interaction(true);
    changed({ id: 'b', version: '1' }); controller.requestAnalysis(); controller.requestAnalysis();
    expect(viewer.capture).toHaveBeenCalledTimes(1);
    interaction(false);
    finishA({ category: 'chair', confidence: .9 });
    await vi.waitFor(async () => expect((await repository.getAnalysis({ id: 'b', version: '1' }))?.category).toBe('table'));
    expect(await repository.getAnalysis({ id: 'a', version: '1' })).toBeUndefined();
    expect(disposeA).toHaveBeenCalledTimes(1); expect(disposeB).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    await controller.dispose(); expect(unsubscribe).toHaveBeenCalledTimes(2);
  });
  it('checks staleness at the actual write point', async () => {
    const repository = new MemoryAssetRepository(), asset = { id: 'a', version: '1' };
    const result = await repository.updateAnalysis(asset, { asset, category: 'chair', confidence: .9, sourceView: 'front', analyzerVersion: '1' }, { isCurrent: () => false });
    expect(result).toBe(false); expect(await repository.getAnalysis(asset)).toBeUndefined();
  });
});
