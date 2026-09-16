import { throwIfAborted } from '../core/inference-client.js';
import { prepareReferenceImage } from './prepare-reference-image.js';
import { routeGenerationRequest } from './route-generation-request.js';
/** Application orchestration. Route classification itself never submits a job. */
export async function submitGenerationRequest(deps, request, reference, context = {}) {
    const decision = await routeGenerationRequest(deps.classifier, request, context, deps.routingPolicy);
    throwIfAborted(context.signal);
    if (decision.route === 'agent') {
        const analysis = await Promise.all(request.assets.map(asset => deps.assets.getAnalysis(asset)));
        throwIfAborted(context.signal);
        return { decision, job: await deps.agent.submit(request, analysis.filter(a => a !== undefined), context) };
    }
    if (decision.route === 'text-to-3d')
        return { decision, job: await deps.generation.textTo3D({ text: request.text }, context) };
    if (!reference || request.attachments[0]?.url !== reference.sourceUrl)
        throw new Error('Reference image and foreground prompts are required');
    const prepared = await prepareReferenceImage(deps.segmenter, reference.image, reference.prompts, context);
    if (prepared.status === 'failed')
        throw prepared.error;
    const imageUrl = await deps.uploader.upload(prepared.image, context);
    throwIfAborted(context.signal);
    return { decision, job: await deps.generation.imageTo3D({ text: request.text, imageUrl }, context) };
}
//# sourceMappingURL=submit-generation-request.js.map