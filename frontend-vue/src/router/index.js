import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import DataView from '@/views/DataView.vue'
import NlpView from '@/views/NlpView.vue'
import LabsView from '@/views/LabsView.vue'
import StatsView from '@/views/StatsView.vue'
import ModelsView from '@/views/ModelsView.vue'
import MultimodalView from '@/views/MultimodalView.vue'
import ReportsView from '@/views/ReportsView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView, meta: { title: '总览' } },
  { path: '/data', name: 'data', component: DataView, meta: { title: '数据与字典' } },
  { path: '/nlp', name: 'nlp', component: NlpView, meta: { title: 'NLP 抽取' } },
  { path: '/labs', name: 'labs', component: LabsView, meta: { title: '生化时序' } },
  { path: '/stats', name: 'stats', component: StatsView, meta: { title: '统计分析' } },
  { path: '/models', name: 'models', component: ModelsView, meta: { title: '模型训练' } },
  { path: '/multimodal', name: 'multimodal', component: MultimodalView, meta: { title: '多模态 & MLOps' } },
  { path: '/reports', name: 'reports', component: ReportsView, meta: { title: '临床报告' } },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
