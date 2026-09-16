# 文本分类教程

本教程将介绍如何使用 WebInfer 进行文本分类任务，如情感分析。

## 基本用法

### 1. 创建 Pipeline

```typescript
import { pipeline } from 'webinfer';

const classifier = await pipeline('text-classification');
```

### 2. 运行分类

```typescript
const result = await classifier.run('I love this product!');
console.log(result);
// { label: 'positive', score: 0.98 }
```

## 批量分类

一次性处理多个文本：

```typescript
const texts = [
  'Great product, highly recommended!',
  'Terrible experience, never again.',
  'It was okay, nothing special.',
];

const results = await classifier.run(texts);
results.forEach((result, i) => {
  console.log(`${texts[i]}: ${result.label} (${result.score.toFixed(2)})`);
});
```

## 使用自定义模型

```typescript
const classifier = await pipeline('text-classification', {
  modelId: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
});
```

## 获取多个结果

使用 `topK` 参数获取多个分类结果：

```typescript
const results = await classifier.run('The movie was interesting', {
  topK: 3
});
// 返回前 3 个可能的分类
```

## 多语言支持

```typescript
// 使用多语言模型
const classifier = await pipeline('text-classification', {
  modelId: 'nlptown/bert-base-multilingual-uncased-sentiment'
});

// 支持多种语言
const results = await classifier.run([
  'This is great!',      // English
  'C\'est magnifique!',  // French
  '太棒了！',             // Chinese
]);
```

## 实时应用示例

### 评论情感分析

```typescript
import { pipeline } from 'webinfer';

async function analyzeComments(comments: string[]) {
  const classifier = await pipeline('text-classification');
  
  const results = await classifier.run(comments);
  
  const summary = {
    positive: 0,
    negative: 0,
    neutral: 0,
  };
  
  results.forEach(r => {
    if (r.score > 0.7) {
      summary[r.label.toLowerCase()]++;
    } else {
      summary.neutral++;
    }
  });
  
  console.log('评论分析:', summary);
  
  classifier.dispose();
}
```

### 表单验证

```typescript
async function validateFeedback(text: string): Promise<boolean> {
  const classifier = await pipeline('text-classification');
  const result = await classifier.run(text);
  
  // 拒绝过于负面的内容
  if (result.label === 'negative' && result.score > 0.9) {
    return false;
  }
  return true;
}
```

## 性能优化

### 预加载模型

```typescript
import { preloadModel, pipeline } from 'webinfer';

// 页面加载时预加载
preloadModel('https://example.com/model.onnx');

// 用户点击时立即可用
button.onclick = async () => {
  const classifier = await pipeline('text-classification');
  // 模型已预加载，立即可用
};
```

### 复用 Pipeline

```typescript
// ❌ 不好：每次都创建新 Pipeline
async function classify(text: string) {
  const classifier = await pipeline('text-classification');
  const result = await classifier.run(text);
  classifier.dispose();
  return result;
}

// ✅ 好：复用 Pipeline
let classifier: TextClassificationPipeline | null = null;

async function classify(text: string) {
  if (!classifier) {
    classifier = await pipeline('text-classification');
  }
  return classifier.run(text);
}
```

## 错误处理

```typescript
try {
  const result = await classifier.run(text);
  console.log(result);
} catch (error) {
  if (error.code === 'MODEL_NOT_FOUND') {
    console.error('模型未找到');
  } else if (error.code === 'INFERENCE_FAILED') {
    console.error('推理失败:', error.message);
  }
}
```

## 完整示例

```html
<!DOCTYPE html>
<html>
<head>
  <title>情感分析</title>
</head>
<body>
  <textarea id="input" placeholder="输入要分析的文本..."></textarea>
  <button id="analyze">分析</button>
  <div id="result"></div>

  <script type="module">
    import { pipeline } from 'https://cdn.jsdelivr.net/npm/webinfer/dist/webinfer.browser.min.js';
    
    let classifier = null;
    
    document.getElementById('analyze').onclick = async () => {
      const text = document.getElementById('input').value;
      const resultDiv = document.getElementById('result');
      
      if (!text) return;
      
      resultDiv.textContent = '分析中...';
      
      try {
        if (!classifier) {
          classifier = await pipeline('text-classification');
        }
        
        const result = await classifier.run(text);
        
        const emoji = result.label === 'positive' ? '😊' : '😔';
        resultDiv.textContent = `${emoji} ${result.label} (${(result.score * 100).toFixed(1)}%)`;
      } catch (error) {
        resultDiv.textContent = '分析失败: ' + error.message;
      }
    };
  </script>
</body>
</html>
```

## 下一步

- [特征提取](./feature-extraction.md)
- [图像分类](./image-classification.md)
- [API 参考](../api/pipeline.md)
