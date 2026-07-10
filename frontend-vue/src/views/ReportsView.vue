<script setup>
import { computed, ref } from 'vue'
import { reportDownloadUrl } from '@/api'

const patientId = ref('p001')
const reportType = ref('html')

const iframeSrc = computed(() => reportDownloadUrl(patientId.value, reportType.value))
</script>

<template>
  <p class="page-desc">GET /api/reports/download — HTML 报告 iframe 嵌入</p>

  <div class="card">
    <div class="grid-2">
      <div class="form-row">
        <label>patient_id</label>
        <input v-model="patientId" />
      </div>
      <div class="form-row">
        <label>report_type</label>
        <select v-model="reportType">
          <option value="html">html</option>
          <option value="json">json</option>
        </select>
      </div>
    </div>
    <a :href="iframeSrc" target="_blank" rel="noopener">在新标签页打开报告</a>
  </div>

  <div class="card" v-if="reportType === 'html'">
    <h3>临床决策报告预览</h3>
    <iframe class="iframe-report" :src="iframeSrc" title="临床报告" />
  </div>

  <div class="card" v-else>
    <p>JSON 格式请使用上方链接在新标签页查看，或对接 axios 下载解析。</p>
  </div>
</template>
