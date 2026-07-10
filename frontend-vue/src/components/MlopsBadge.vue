<script setup>
defineProps({
  enhanced: { type: Boolean, default: false },
})
</script>

<template>
  <div v-if="enhanced" class="badge-mlops">🏅 MLOps Enhanced · 当前诊断模型已自动更新至最新版本</div>
  <div v-else class="badge-muted">ℹ️ 使用注册表初始战绩（可触发 AI 进化演示）</div>
</template>
