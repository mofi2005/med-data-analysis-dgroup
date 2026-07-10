import axios from "axios";

const http = axios.create({
  baseURL: "",
  timeout: 60000,
});

export async function getHealth() {
  const { data } = await http.get("/api/health");
  return data;
}

export async function uploadData(file, description = "") {
  const form = new FormData();
  form.append("file", file);
  if (description) form.append("description", description);
  const { data } = await http.post("/api/data/upload", form);
  return data;
}

export async function initDictionary() {
  const { data } = await http.post("/api/data/dictionary/init_default");
  return data;
}

export async function matchDictionary(rawField) {
  const { data } = await http.get("/api/data/dictionary/match", {
    params: { raw_field: rawField },
  });
  return data;
}

export async function mapFields(mappings) {
  const { data } = await http.post("/api/data/dictionary/map_fields", { mappings });
  return data;
}

export async function extractNlp(text) {
  const { data } = await http.post("/api/nlp/extract", { text });
  return data;
}

export async function getLabTrend(patientId, itemName) {
  const { data } = await http.get("/api/labs/trend", {
    params: { patient_id: patientId, item_name: itemName },
  });
  return data;
}

export async function describeStats(datasetId) {
  const { data } = await http.get("/api/stats/describe", {
    params: datasetId ? { dataset_id: datasetId } : {},
  });
  return data;
}

export async function differenceTest(payload) {
  const { data } = await http.post("/api/stats/difference", payload);
  return data;
}

export async function correlationMatrix(payload) {
  const { data } = await http.post("/api/stats/correlation", payload);
  return data;
}

export async function trainModel(payload) {
  const { data } = await http.post("/api/models/train", payload);
  return data;
}

export async function listModels() {
  const { data } = await http.get("/api/models/list");
  return data;
}

export async function modelImportance(modelId) {
  const { data } = await http.get(`/api/models/${modelId}/importance`);
  return data;
}

export async function routeAndPredict(payload) {
  const { data } = await http.post("/api/multimodal/route_and_predict", payload);
  return data;
}

export async function alignFeatures(payload) {
  const { data } = await http.post("/api/multimodal/align_features", payload);
  return data;
}

export async function triggerEvolutionDemo() {
  const { data } = await http.post("/api/mlops/trigger_evolution_demo");
  return data;
}

export function reportDownloadUrl(patientId, reportType = "html") {
  const params = new URLSearchParams({
    patient_id: patientId,
    report_type: reportType,
  });
  return `/api/reports/download?${params.toString()}`;
}
