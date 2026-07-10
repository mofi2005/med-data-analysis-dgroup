<script setup>
import { onMounted, ref } from 'vue'
import { getHealth } from '@/api'

const health = ref(null)
const error = ref('')

onMounted(async () => {
  try {
    health.value = await getHealth()
  } catch (e) {
    error.value = e.message
  }
})

const apis = [
  ['POST', '/api/data/upload', '文件上传'],
  ['GET', '/api/data/dictionary/match', '字典匹配'],
  ['POST', '/api/data/dictionary/map_fields', '字段映射'],
  ['POST', '/api/nlp/extract', 'NLP 三元组抽取'],
  ['GET', '/api/labs/trend', '生化折线图'],
  ['GET', '/api/stats/describe', '描述统计'],
  ['POST', '/api/stats/difference', '组间差异'],
  ['POST', '/api/stats/correlation', '相关性热力图'],
  ['POST', '/api/models/train', '模型训练'],
  ['GET', '/api/models/list', '模型列表'],
  ['GET', '/api/models/{id}/importance', '特征重要性'],
  ['POST', '/api/multimodal/route_and_predict', '自适应路由'],
  ['POST', '/api/multimodal/align_features', '多模态对齐'],
  ['GET', '/api/reports/download', '报告下载'],
  ['POST', '/api/mlops/trigger_evolution_demo', 'MLOps 演示'],
]
</script>

<template>
  <p class="page-desc">后端健康状态与 16 个 RESTful API 导航</p>

  <div v-if="error" class="alert alert-error">后端未连接：{{ error }}。请先运行 uvicorn src.backend.main:app --port 8000</div>
  <div v-else-if="health" class="alert alert-success">{{ health.message }}</div>

  <div class="card">
    <h3>快速启动</h3>
    <pre class="json"># 终端 1
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000

# 终端 2
cd frontend-vue && npm install && npm run dev</pre>
  </div>

  <div class="card">
    <h3>API 全景（组员1 中台）</h3>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <thead>
        <tr style="background:#eef4fb; text-align:left;">
          <th style="padding:8px;">Method</th>
          <th style="padding:8px;">Path</th>
          <th style="padding:8px;">说明</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, i) in apis" :key="i" style="border-top:1px solid #e5e7eb;">
          <td style="padding:8px;"><code>{{ row[0] }}</code></td>
          <td style="padding:8px;"><code>{{ row[1] }}</code></td>
          <td style="padding:8px;">{{ row[2] }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
