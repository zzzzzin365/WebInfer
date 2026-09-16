# 3D 业务接入与低耦合重构

WebInfer 负责端侧推理。3D 生成服务、复杂任务 Agent、Viewer 截图和资产存储由应用提供；本仓库没有这些生产服务的实现或凭证。

## 分层与入口

| 层 | 目录 | 职责 |
| --- | --- | --- |
| 核心 | `src/core` | 独立引擎、后端注册、通用任务调度、资源作用域 |
| 模型管线 | `src/pipelines` | 模型加载、预处理、推理、后处理、管线组合 |
| 3D 流程 | `src/workflows-3d` | 前景处理、请求路由、资产分析与生命周期协调 |
| 适配实现 | `src/adapters` | Pipeline 接入、Worker ONNX 执行、示例内存存储 |
| 应用组装 | `examples/3d-workflows.ts` | 创建引擎，注入外部能力，连接生命周期 |

核心不导入业务流程或具体后端；业务流程不导入适配实现，不调用 fetch，不访问 Viewer 内部 renderer。接口定义在 `workflows-3d/types.ts`，适配器实现这些接口。`core` 历史 WorkerPool API 仍保留作兼容入口；新应用使用显式 WorkerRuntime，避免旧的内嵌 CDN Worker。

## 独立引擎

```ts
import { createInferenceEngine } from 'webinfer/core';
import { createONNXRuntime } from 'webinfer/backends';
import { pipeline } from 'webinfer/pipelines';

const engine = createInferenceEngine({
  backends: [{ type: 'wasm', create: createONNXRuntime }],
  scheduler: { maxConcurrentTasks: 1 },
});
const classifier = await pipeline('image-classification', {
  engine,
  model: '/models/asset-classifier.onnx', // 由应用部署适配模型
  labels: ['chair', 'table'],             // 必须匹配训练输出顺序
});
try {
  await classifier.run(image, { priority: 'low', signal: abortController.signal });
} finally {
  classifier.dispose();
  await engine.dispose();
}
```

每个引擎拥有独立的注册表、Runtime、调度器和资源管理器。加载的模型不能交给另一个引擎执行。相同后端并发初始化只创建一次。ONNX session 按 Runtime 存储；`session.release()` 等运行中的任务结束后执行。应用销毁时应 `await engine.dispose()`。

ONNX 模块代码、ORT 环境配置和浏览器下载缓存仍可共享，不代表模型 session 全局共享。WASM 部署配置属于应用级；浏览器默认 `/ort/`，应用预先设置的 `ort.env.wasm.wasmPaths` 不会被覆盖。

## 三条业务流程

1. `prepareReferenceImage()`：要求用户点选前景，分割后合成透明背景图、裁剪和限制尺寸。原图不被修改。失败返回 `failed` 与原图；取消则拒绝 Promise。它不是自动主体检测，也不是像素降噪。
2. `routeGenerationRequest()`：返回决定，不提交请求。只有应用明确声明 `standalone: true`、参数完整、无关联资产或对话依赖，且分类置信度与领先分差足够，才直达生成服务。失败、未知、歧义统一交给 Agent。默认 0.85/0.2 是原型配置，未经业务样本标定。
3. `analyzeViewedAsset()`：通过 ViewerAdapter 获取渲染稳定后的截图，只返回类别、置信度、视角和模型版本；低置信度类别为 null。不把截图分类冒充完整 3D 质量检查。

`submitGenerationRequest()` 负责独立的提交步骤：文字生成调用 GenerationService；图片生成先处理并上传，再提交处理后的 URL；复杂任务读取所 attach 的资产版本信息，交给 AgentGateway。分割或上传失败不会偷偷提交原图。服务鉴权、参数校验、任务去重、供应侧密钥全部留在业务后端。

`createPipelineInferenceAdapters()` 将现有真实 Pipeline 接到业务接口。它延迟加载并复用模型，同一个 SAM 的编码和解码串行执行，结束后释放图片 embedding。替换分类模型必须同时检查 tokenizer、预后处理及标签约定，不能只换 URL。首次加载可能下载大模型；下载暂不支持通过业务 AbortSignal 中断，加载完成后会检查取消状态。

## 优先级、取消与资产切换

- Runtime 入口接收 `priority / signal / scopeId`。Pipeline 按每次调用传递，不在共享实例上存储可变的执行上下文。
- 调度器跨模型比较待执行任务：critical → high → normal → low。已经运行的任务不抢占；有空闲槽便补位，不再等待同一批次最慢的模型。
- `TaskScope` 是通用资源作用域。关闭立即使 signal 失效、取消排队；在途任务结束后才释放显式 track 的资源。底层无法中止的推理会完成，但结果丢弃并清理。
- `ViewerAnalysisController` 监听应用提供的资产变化和交互事件；合并连续请求，交互期间暂停启动分析，切换后旧结果失效。
- `AssetRepository.updateAnalysis()` 必须按资产 ID + version 写入，并在真正写入时检查 guard。远端存储必须实现条件写入/版本约束，单靠前端在 fetch 前检查不能保证一致性。示例 MemoryAssetRepository 在同步写点检查，仅用于演示，不持久化。
- Viewer 仍负责几何、材质、纹理和 WebGL 资源；WebInfer 只清理自身持有的资源。不承诺消除全部 OOM。

## Worker 接入

应用构建工具将 `src/adapters/onnx-worker-entry.ts`（发布产物为 `dist/adapters/onnx-worker-entry.js`）打包为 module worker；包含可选依赖 onnxruntime-web，按所安装的版本部署对应 WASM 文件。应用也可新建一个 Worker 入口文件，内容为 `import 'webinfer/onnx-worker'`，交给构建工具打包。随后注入 Worker 工厂：

```ts
backends: [{
  type: 'wasm',
  create: memory => new WorkerRuntime(new Worker(workerBundleUrl, { type: 'module' }), memory),
}]
```

调度保留在调用线程，模型和 ONNX 推理在 Worker。协议支持具名输入，按原始 dtype 复制并传输 Tensor，不分离调用者的输入 buffer。Worker 崩溃时拒绝在途请求；必须新建引擎并重载模型，不把原模型 ID 随意转发到另一 Worker。预处理和后处理仍可能在主线程，Worker 不等于整个业务流程无主线程开销。

## 兼容与迁移

- `pipeline(task)` 继续使用兼容默认引擎；只在未注册默认后端时补充注册，不覆盖自定义后端。
- 导入 `webinfer/backends` 不再自动注册。旧式 `loadModel()` 调用需先显式 `registerAllBackends()`，新代码推荐 createInferenceEngine。
- `compose/parallel` 移到 pipelines 层，包根导出不变。原 `webinfer/core` 的组合导入改用 `webinfer` 或 `webinfer/pipelines`。
- `Runtime.dispose()` 可返回 Promise；`RuntimeManager.disposeAll/disposeRuntime` 现在可等待清理完成。

## 验证与尚未验收的内容

执行：`npm test`、`npm run build`、`node scripts/smoke-onnx.mjs`、`node scripts/smoke-worker-browser.mjs`。最后一项需要本机 Chrome。

重构前基线：143 项通过，1 项因同步函数使用 Promise 断言失败；TypeScript 通过。重构修正该断言，以及一个实际没有占满并发槽的取消测试。

本次 `npm test`：12 个测试文件、166 项全部通过；构建通过。

验证覆盖引擎/session 隔离、全局优先级、在途清理、Worker 协议、整数精度、前景合成、保守路由、资产版本校验和切换清理。两个 smoke 分别执行真实 ONNX WASM、真实 Chrome Worker + ONNX WASM，使用自包含 Identity 小模型，不代表业务模型质量验证。

未验收：生产生成服务与 Agent 接入、真实 Library 落库、真实 Viewer 事件与截图接入、业务分类/分割模型准确率、Viewer 帧率、大 GLB 连续切换资源趋势。原生 WebGPU 后端仍是骨架；本轮未宣称支持自动 WebGPU/WASM 模型切换。
