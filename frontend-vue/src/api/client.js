import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 60000,
});

export const getHealth = () => api.get("/health");

export const uploadDataset = (file, sourceType = "D组") => {
  const form = new FormData();
  form.append("file", file);
  form.append("source_type", sourceType);
  return api.post("/data/upload", form);
};

export const matchDictionary = (fieldName) =>
  api.get("/data/dictionary/match", { params: { field_name: fieldName } });

export const mapFields = (mappings) =>
  api.post("/data/dictionary/map_fields", { mappings });

export const extractNlp = (text) => api.post("/nlp/extract", { text });

export const getLabTrend = (patientId, itemName) =>
  api.get("/labs/trend", { params: { patient_id: patientId, item_name: itemName } });

export const describeStats = (datasetId) =>
  api.get("/stats/describe", { params: datasetId ? { dataset_id: datasetId } : {} });

export const statsDifference = (payload) => api.post("/stats/difference", payload);

export const statsCorrelation = (variables, method = "pearson") =>
  api.post("/stats/correlation", { variables, method });

export const trainModel = (payload) => api.post("/models/train", payload);

export const listModels = () => api.get("/models/list");

export const getModelImportance = (modelId, topK = 10) =>
  api.get(`/models/${modelId}/importance`, { params: { top_k: topK } });

export const routeAndPredict = (payload) => api.post("/multimodal/route_and_predict", payload);

export const alignFeatures = (payload) => api.post("/multimodal/align_features", payload);

export const triggerEvolutionDemo = () => api.post("/mlops/trigger_evolution_demo");

export const reportDownloadUrl = (patientId, reportType = "html") =>
  `/api/reports/download?patient_id=${encodeURIComponent(patientId)}&report_type=${reportType}`;

export default api;
