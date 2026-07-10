<script setup>
import { ref } from 'vue'
import { uploadDataset, initDictionary, matchDictionary, mapFields } from '@/api'

const file = ref(null)
const rawField = ref('白细胞')
const datasetId = ref('')
const uploadResult = ref(null)
const matchResult = ref(null)
const mapResult = ref(null)
const error = ref('')

function onFileChange(e) {
  file.value = e.target.files?.[0] || null
}

async function doUpload() {
  error.value = ''
  if (!file.value) return
  try {
    uploadResult.value = await uploadDataset(file.value)
    datasetId.value = uploadResult.value.dataset_id
  } catch (e) {
    error.value = e.message
  }
}

async function doInitDict() {
  try {
    uploadResult.value = await initDictionary()
  } catch (e) {
    error.value = e.message
  }
}

async function doMatch() {
  try {
    matchResult.value = await matchDictionary(rawField.value)
  } catch (e) {
    error.value = e.message
  }
}

async function doMap() {
  try {
    mapResult.value = await mapFields({
      dataset_id: datasetId.value,
      mappings: [{ raw_name: '空腹血溏', standard_name: 'GLU' }],
    })
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">POST /api/data/upload · GET match · POST map_fields</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="grid-2">
    <div class="card">
      <h3>文件上传</h3>
      <input type="file" accept=".csv,.xlsx,.xls,.json" @change="onFileChange" />
      <div style="margin-top:12px;">
        <button class="btn btn-primary" @click="doUpload">上传到 /api/data/upload</button>
        <button class="btn" @click="doInitDict">初始化字典</button>
      </div>
      <pre v-if="uploadResult" class="json">{{ JSON.stringify(uploadResult, null, 2) }}</pre>
    </div>

    <div class="card">
      <h3>字典匹配 & 映射</h3>
      <div class="form-row">
        <label>原始字段名</label>
        <input v-model="rawField" />
      </div>
      <button class="btn btn-primary" @click="doMatch">GET /dictionary/match</button>
      <button class="btn" @click="doMap">POST /dictionary/map_fields</button>
      <pre v-if="matchResult" class="json">{{ JSON.stringify(matchResult, null, 2) }}</pre>
      <pre v-if="mapResult" class="json">{{ JSON.stringify(mapResult, null, 2) }}</pre>
    </div>
  </div>
</template>
