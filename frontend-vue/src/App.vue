<script setup>
import { computed, onMounted, ref } from "vue";
import LabsTrendChart from "./components/LabsTrendChart.vue";
import StatsHeatmap from "./components/StatsHeatmap.vue";
import MlopsScoresChart from "./components/MlopsScoresChart.vue";
import * as api from "./api";

const sections = [
  { id: "overview", label: "总览 & 健康检查" },
  { id: "data", label: "数据与字典 (4 API)" },
  { id: "nlp", label: "NLP 结构化" },
  { id: "labs", label: "生化时序折线图" },
  { id: "stats", label: "统计与热力图" },
  { id: "models", label: "模型训练管理" },
  { id: "multimodal", label: "多模态 & MLOps" },
  { id: "reports", label: "临床报告 iframe" },
];

const active = ref("overview");
const health = ref(null);
const error = ref("");
const loading = ref(false);

// shared state
const datasetId = ref("");
const patientId = ref("p001");
const itemName = ref("GLU");
const rawField = ref("血糖");
const nlpText = ref("患者既往有高血压病史，近期出现咳嗽、乏力，血糖偏高。");
const corrVars = ref("WBC,HGB,PLT,ALT,CEA,GLU");

const dataResult = ref(null);
const nlpResult = ref(null);
const trendResult = ref(null);
const describeResult = ref(null);
const diffResult = ref(null);
const corrResult = ref(null);
const trainResult = ref(null);
const modelsList = ref(null);
const importanceResult = ref(null);
const alignResult = ref(null);

const modelId = ref("demo_vue_model");
const taskType = ref("classification");

const glu = ref(9.8);
const cea = ref(3.5);
const timeSeriesSteps = ref(5);
const nlpEntities = ref(4);
const missingRate = ref(0.1);

const prediction = ref(null);
const scoresBefore = ref(null);
const evolutionMsg = ref("");

const reportPatientId = ref("p001");
const reportUrl = computed(() => api.reportDownloadUrl(reportPatientId.value, "html"));

const routePayload = computed(() => ({
  patient_profile: {
    patient_id: patientId.value,
    time_series_steps: timeSeriesSteps.value,
    nlp_entities: nlpEntities.value,
    missing_rate: missingRate.value,
  },
  clinical_data: { GLU: glu.value, CEA: cea.value },
}));

async function run(fn) {
  error.value = "";
  loading.value = true;
  try {
    await fn();
  } catch (e) {
    error.value = e?.response?.data?.detail || e.message || String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  run(async () => {
    health.value = await api.getHealth();
  });
});

function onFileChange(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  run(async () => {
    dataResult.value = await api.uploadData(file);
    datasetId.value = dataResult.value.dataset_id || dataResult.value.id || "";
  });
}

async function doInitDict() {
  await run(async () => {
    dataResult.value = await api.initDictionary();
  });
}

async function doMatchField() {
  await run(async () => {
    dataResult.value = await api.matchDictionary(rawField.value);
  });
}

async function doMapFields() {
  await run(async () => {
    dataResult.value = await api.mapFields([
      { raw_name: rawField.value, standard_name: "GLU" },
    ]);
  });
}

async function doNlpExtract() {
  await run(async () => {
    nlpResult.value = await api.extractNlp(nlpText.value);
  });
}

async function doLabTrend() {
  await run(async () => {
    trendResult.value = await api.getLabTrend(patientId.value, itemName.value);
  });
}

async function doDescribe() {
  await run(async () => {
    describeResult.value = await api.describeStats(datasetId.value || undefined);
  });
}

async function doDifference() {
  await run(async () => {
    diffResult.value = await api.differenceTest({
      group_variable: "label",
      target_variable: "CEA",
      test_method: "t_test",
    });
  });
}

async function doCorrelation() {
  await run(async () => {
    const variables = corrVars.value.split(",").map((s) => s.trim()).filter(Boolean);
    corrResult.value = await api.correlationMatrix({ variables, method: "pearson" });
  });
}

async function doTrain() {
  await run(async () => {
    trainResult.value = await api.trainModel({
      model_id: modelId.value,
      task_type: taskType.value,
      target_variable: "label",
      features: ["CEA", "WBC", "ALT", "GLU", "age"],
    });
  });
}

async function doListModels() {
  await run(async () => {
    modelsList.value = await api.listModels();
  });
}

async function doImportance() {
  await run(async () => {
    importanceResult.value = await api.modelImportance(modelId.value);
  });
}

async function doAlignFeatures() {
  await run(async () => {
    alignResult.value = await api.alignFeatures({
      patient_id: patientId.value,
      case_id: `${patientId.value}_c1`,
      group_b_quality: { artifact_level: "low", quality_score: 0.95 },
      group_e_radiomics: { texture_mean: 15.3, glcm_entropy: 3.45 },
      d_group_nlp: { entity_count: nlpEntities.value, negation_count: 1 },
      d_group_stats: { GLU_slope: 0.64, Creatinine_mean: 88 },
    });
  });
}

async function doPredict() {
  await run(async () => {
    prediction.value = await api.routeAndPredict(routePayload.value);
    scoresBefore.value = null;
  });
}

async function doEvolution() {
  await run(async () => {
    if (prediction.value?.mlops_live_scores) {
      scoresBefore.value = { ...prediction.value.mlops_live_scores };
    }
    const evo = await api.triggerEvolutionDemo();
    evolutionMsg.value = evo.message || "进化完成";
    prediction.value = await api.routeAndPredict(routePayload.value);
  });
}
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <h1>D组医学数据中台</h1>
      <p>Vue 3 + ECharts · 16 RESTful API</p>
      <button
        v-for="s in sections"
        :key="s.id"
        class="nav-btn"
        :class="{ active: active === s.id }"
        @click="active = s.id"
      >
        {{ s.label }}
      </button>
    </aside>

    <main class="main">
      <div v-if="health" class="card">
        <span class="status-ok">● 后端在线</span>
        <span style="margin-left: 12px; color: var(--muted)">{{ health.message }}</span>
      </div>
      <div v-if="error" class="card status-err">{{ error }}</div>
      <div v-if="loading" class="card">加载中…</div>

      <!-- 总览 -->
      <section v-show="active === 'overview'" class="card">
        <h2>API 全景（PDF 交付）</h2>
        <table class="simple">
          <thead>
            <tr><th>模块</th><th>接口</th></tr>
          </thead>
          <tbody>
            <tr><td>健康</td><td>GET /api/health</td></tr>
            <tr><td>数据</td><td>POST /api/data/upload · init_default · match · map_fields</td></tr>
            <tr><td>NLP</td><td>POST /api/nlp/extract</td></tr>
            <tr><td>生化</td><td>GET /api/labs/trend</td></tr>
            <tr><td>统计</td><td>GET describe · POST difference · POST correlation</td></tr>
            <tr><td>模型</td><td>POST train · GET list · GET {id}/importance</td></tr>
            <tr><td>多模态</td><td>POST route_and_predict · align_features</td></tr>
            <tr><td>MLOps</td><td>POST /api/mlops/trigger_evolution_demo</td></tr>
            <tr><td>报告</td><td>GET /api/reports/download</td></tr>
          </tbody>
        </table>
      </section>

      <!-- 数据 -->
      <section v-show="active === 'data'" class="card">
        <h2>数据接入与字典</h2>
        <div class="grid-2">
          <div class="field">
            <label>上传 CSV / Excel</label>
            <input type="file" accept=".csv,.xlsx,.json" @change="onFileChange" />
          </div>
          <div class="field">
            <label>dataset_id</label>
            <input v-model="datasetId" placeholder="上传后自动填充" />
          </div>
        </div>
        <div class="grid-2">
          <div class="field">
            <label>原始字段名 (match / map)</label>
            <input v-model="rawField" />
          </div>
        </div>
        <button class="btn btn-primary" @click="doInitDict">初始化标准字典</button>
        <button class="btn btn-ghost" @click="doMatchField">匹配字段</button>
        <button class="btn btn-ghost" @click="doMapFields">批量映射</button>
        <pre v-if="dataResult" class="json">{{ JSON.stringify(dataResult, null, 2) }}</pre>
      </section>

      <!-- NLP -->
      <section v-show="active === 'nlp'" class="card">
        <h2>NLP 病历结构化</h2>
        <div class="field">
          <label>病历文本</label>
          <textarea v-model="nlpText" rows="5" />
        </div>
        <button class="btn btn-primary" @click="doNlpExtract">POST /api/nlp/extract</button>
        <pre v-if="nlpResult" class="json">{{ JSON.stringify(nlpResult, null, 2) }}</pre>
      </section>

      <!-- 生化折线 P1 -->
      <section v-show="active === 'labs'" class="card">
        <h2>生化时序 · ECharts 折线图</h2>
        <div class="grid-3">
          <div class="field"><label>patient_id</label><input v-model="patientId" /></div>
          <div class="field">
            <label>item_name</label>
            <select v-model="itemName">
              <option>GLU</option><option>WBC</option><option>HGB</option>
              <option>Creatinine</option><option>CEA</option><option>ALT</option>
            </select>
          </div>
        </div>
        <button class="btn btn-primary" @click="doLabTrend">GET /api/labs/trend</button>
        <LabsTrendChart :trend="trendResult" />
      </section>

      <!-- 统计 P1 -->
      <section v-show="active === 'stats'" class="card">
        <h2>统计学检验 & 相关性热力图</h2>
        <button class="btn btn-primary" @click="doDescribe">描述性统计</button>
        <button class="btn btn-ghost" @click="doDifference">组间差异 T 检验</button>
        <div class="field" style="margin-top: 12px">
          <label>热力图变量（逗号分隔）</label>
          <input v-model="corrVars" />
        </div>
        <button class="btn btn-primary" @click="doCorrelation">生成相关性热力图</button>
        <pre v-if="describeResult" class="json">{{ JSON.stringify(describeResult, null, 2) }}</pre>
        <pre v-if="diffResult" class="json">{{ JSON.stringify(diffResult, null, 2) }}</pre>
        <StatsHeatmap :result="corrResult" />
      </section>

      <!-- 模型 -->
      <section v-show="active === 'models'" class="card">
        <h2>机器学习建模</h2>
        <div class="grid-3">
          <div class="field"><label>model_id</label><input v-model="modelId" /></div>
          <div class="field">
            <label>task_type</label>
            <select v-model="taskType">
              <option value="classification">classification</option>
              <option value="regression">regression</option>
              <option value="clustering">clustering</option>
            </select>
          </div>
        </div>
        <button class="btn btn-primary" @click="doTrain">POST /api/models/train</button>
        <button class="btn btn-ghost" @click="doListModels">GET /api/models/list</button>
        <button class="btn btn-ghost" @click="doImportance">特征重要性</button>
        <pre v-if="trainResult" class="json">{{ JSON.stringify(trainResult, null, 2) }}</pre>
        <pre v-if="modelsList" class="json">{{ JSON.stringify(modelsList, null, 2) }}</pre>
        <pre v-if="importanceResult" class="json">{{ JSON.stringify(importanceResult, null, 2) }}</pre>
      </section>

      <!-- 多模态 MLOps P0 -->
      <section v-show="active === 'multimodal'" class="card">
        <h2>多模态路由 & MLOps 自进化</h2>
        <div class="grid-3">
          <div class="field"><label>GLU</label><input v-model.number="glu" type="number" step="0.1" /></div>
          <div class="field"><label>CEA</label><input v-model.number="cea" type="number" step="0.1" /></div>
          <div class="field"><label>缺失率</label><input v-model.number="missingRate" type="number" step="0.05" min="0" max="1" /></div>
        </div>
        <button class="btn btn-primary" @click="doPredict">🔮 route_and_predict</button>
        <button class="btn btn-accent" @click="doEvolution">🚀 trigger_evolution_demo</button>
        <button class="btn btn-ghost" @click="doAlignFeatures">align_features</button>

        <div v-if="prediction">
          <div v-if="prediction.mlops_enhanced" class="badge-mlops">
            🏅 MLOps Enhanced · 当前诊断模型已自动更新至最新版本
          </div>
          <div v-else class="badge-muted">使用注册表初始战绩（可点击进化演示）</div>
          <div class="metrics">
            <div class="metric"><div class="label">选中模型</div><div class="value">{{ prediction.selected_model }}</div></div>
            <div class="metric"><div class="label">路由得分</div><div class="value">{{ prediction.winning_score }}</div></div>
            <div class="metric"><div class="label">引擎</div><div class="value" style="font-size:14px">{{ prediction.engine_mode }}</div></div>
          </div>
          <MlopsScoresChart :scores="prediction.mlops_live_scores || {}" :previous-scores="scoresBefore" />
          <p><strong>路由理由：</strong>{{ prediction.routing_reason }}</p>
          <p v-if="evolutionMsg" class="status-ok">{{ evolutionMsg }}</p>
        </div>
        <pre v-if="alignResult" class="json">{{ JSON.stringify(alignResult, null, 2) }}</pre>
      </section>

      <!-- 报告 iframe P1 -->
      <section v-show="active === 'reports'" class="card">
        <h2>临床决策报告 · iframe 嵌入</h2>
        <div class="field">
          <label>patient_id</label>
          <input v-model="reportPatientId" />
        </div>
        <p>
          <a :href="reportUrl" target="_blank" rel="noopener">新窗口打开报告</a>
        </p>
        <iframe class="report-frame" :src="reportUrl" title="临床报告" />
      </section>
    </main>
  </div>
</template>
