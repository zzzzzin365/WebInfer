# 安装

## 通过包管理器安装

### npm

```bash
npm install webinfer
```

### yarn

```bash
yarn add webinfer
```

### pnpm

```bash
pnpm add webinfer
```

## 通过 CDN 使用

```html
<script type="module">
  import * as WebInfer from 'https://cdn.jsdelivr.net/npm/webinfer/dist/webinfer.browser.min.js';
  
  // 使用 WebInfer
  const pipeline = await WebInfer.pipeline('text-classification');
</script>
```

## 浏览器兼容性

| 浏览器 | WebGPU | WebNN | WASM |
|--------|--------|-------|------|
| Chrome 113+ | ✅ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox 118+ | ⚠️ | ❌ | ✅ |
| Safari 17+ | ⚠️ | ❌ | ✅ |

## TypeScript 支持

WebInfer 使用 TypeScript 编写，提供完整的类型定义：

```typescript
import { pipeline, WebInferTensor, Tokenizer } from 'webinfer';
import type { PipelineOptions, TextClassificationResult } from 'webinfer';
```

## 下一步

- [快速入门](./quickstart.md)
- [核心概念](./concepts.md)
