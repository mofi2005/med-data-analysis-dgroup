#!/usr/bin/env python3
"""临床医学命名实体识别引擎 (Medical NER Engine).

本模块提供企业级医学文本命名实体识别能力，覆盖疾病、症状、体征、器官、
解剖部位、病灶、药物、手术等 17 类医学实体的自动化结构化解析。

技术架构：
  - 主线引擎: PyTorch + HuggingFace Transformers (AutoModelForTokenClassification)
  - 兜底引擎: 高性能词典 + 正则规则匹配（零外部依赖，保证可用性）
  - 硬件自适应: CUDA GPU 自动检测与降级
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# 实体类型定义（17 类，对齐项目任务书规范）
# ===================================================================

ENTITY_TYPES: dict[str, str] = {
    "disease_or_lesion": "疾病/病灶",
    "symptom": "症状",
    "sign": "体征",
    "organ": "器官",
    "anatomical_site": "解剖部位",
    "drug": "药物",
    "surgery": "手术",
    "examination": "检查/检验",
    "laboratory_test": "实验室指标",
    "tumor_marker": "肿瘤标志物",
    "vital_sign": "生命体征",
    "medical_device": "医疗器械",
    "pathogen": "病原体",
    "body_fluid": "体液",
    "imaging_finding": "影像学发现",
    "treatment": "治疗措施",
    "diagnosis": "诊断结论",
}

# ===================================================================
# 基础临床词典（中文 → standard_name 映射，覆盖高频词汇）
# ===================================================================

_CLINICAL_DICT: dict[str, dict[str, str]] = {
    # ---- 疾病/病灶 ----
    "肺结节": {"entity_type": "disease_or_lesion", "standard_name": "lung_nodule"},
    "肺癌": {"entity_type": "disease_or_lesion", "standard_name": "lung_cancer"},
    "高血压": {"entity_type": "disease_or_lesion", "standard_name": "hypertension"},
    "糖尿病": {"entity_type": "disease_or_lesion", "standard_name": "diabetes"},
    "冠心病": {"entity_type": "disease_or_lesion", "standard_name": "coronary_heart_disease"},
    "脑梗死": {"entity_type": "disease_or_lesion", "standard_name": "cerebral_infarction"},
    "脑卒中": {"entity_type": "disease_or_lesion", "standard_name": "stroke"},
    "肝硬化": {"entity_type": "disease_or_lesion", "standard_name": "liver_cirrhosis"},
    "肝炎": {"entity_type": "disease_or_lesion", "standard_name": "hepatitis"},
    "脂肪肝": {"entity_type": "disease_or_lesion", "standard_name": "fatty_liver"},
    "慢性肾病": {"entity_type": "disease_or_lesion", "standard_name": "chronic_kidney_disease"},
    "肾衰竭": {"entity_type": "disease_or_lesion", "standard_name": "renal_failure"},
    "肺炎": {"entity_type": "disease_or_lesion", "standard_name": "pneumonia"},
    "支气管炎": {"entity_type": "disease_or_lesion", "standard_name": "bronchitis"},
    "心肌梗死": {"entity_type": "disease_or_lesion", "standard_name": "myocardial_infarction"},
    "心力衰竭": {"entity_type": "disease_or_lesion", "standard_name": "heart_failure"},
    "心律失常": {"entity_type": "disease_or_lesion", "standard_name": "arrhythmia"},
    "胃癌": {"entity_type": "disease_or_lesion", "standard_name": "gastric_cancer"},
    "肝癌": {"entity_type": "disease_or_lesion", "standard_name": "liver_cancer"},
    "结直肠癌": {"entity_type": "disease_or_lesion", "standard_name": "colorectal_cancer"},
    "乳腺结节": {"entity_type": "disease_or_lesion", "standard_name": "breast_nodule"},
    "甲状腺结节": {"entity_type": "disease_or_lesion", "standard_name": "thyroid_nodule"},
    "高脂血症": {"entity_type": "disease_or_lesion", "standard_name": "hyperlipidemia"},
    "高尿酸血症": {"entity_type": "disease_or_lesion", "standard_name": "hyperuricemia"},
    "痛风": {"entity_type": "disease_or_lesion", "standard_name": "gout"},
    "贫血": {"entity_type": "disease_or_lesion", "standard_name": "anemia"},
    "白血病": {"entity_type": "disease_or_lesion", "standard_name": "leukemia"},
    "淋巴瘤": {"entity_type": "disease_or_lesion", "standard_name": "lymphoma"},
    # ---- 症状 ----
    "发热": {"entity_type": "symptom", "standard_name": "fever"},
    "咳嗽": {"entity_type": "symptom", "standard_name": "cough"},
    "咳痰": {"entity_type": "symptom", "standard_name": "expectoration"},
    "胸痛": {"entity_type": "symptom", "standard_name": "chest_pain"},
    "腹痛": {"entity_type": "symptom", "standard_name": "abdominal_pain"},
    "头痛": {"entity_type": "symptom", "standard_name": "headache"},
    "头晕": {"entity_type": "symptom", "standard_name": "dizziness"},
    "恶心": {"entity_type": "symptom", "standard_name": "nausea"},
    "呕吐": {"entity_type": "symptom", "standard_name": "vomiting"},
    "乏力": {"entity_type": "symptom", "standard_name": "fatigue"},
    "呼吸困难": {"entity_type": "symptom", "standard_name": "dyspnea"},
    "心悸": {"entity_type": "symptom", "standard_name": "palpitation"},
    "咯血": {"entity_type": "symptom", "standard_name": "hemoptysis"},
    "便血": {"entity_type": "symptom", "standard_name": "hematochezia"},
    "黄疸": {"entity_type": "symptom", "standard_name": "jaundice"},
    "水肿": {"entity_type": "symptom", "standard_name": "edema"},
    "消瘦": {"entity_type": "symptom", "standard_name": "weight_loss"},
    "食欲减退": {"entity_type": "symptom", "standard_name": "anorexia"},
    "失眠": {"entity_type": "symptom", "standard_name": "insomnia"},
    "胸闷": {"entity_type": "symptom", "standard_name": "chest_tightness"},
    # ---- 体征 ----
    "肺部啰音": {"entity_type": "sign", "standard_name": "lung_rales"},
    "心脏杂音": {"entity_type": "sign", "standard_name": "heart_murmur"},
    "肝肿大": {"entity_type": "sign", "standard_name": "hepatomegaly"},
    "脾肿大": {"entity_type": "sign", "standard_name": "splenomegaly"},
    "淋巴结肿大": {"entity_type": "sign", "standard_name": "lymphadenopathy"},
    "紫绀": {"entity_type": "sign", "standard_name": "cyanosis"},
    "皮疹": {"entity_type": "sign", "standard_name": "rash"},
    # ---- 器官 ----
    "肺": {"entity_type": "organ", "standard_name": "lung"},
    "肝": {"entity_type": "organ", "standard_name": "liver"},
    "心脏": {"entity_type": "organ", "standard_name": "heart"},
    "肾脏": {"entity_type": "organ", "standard_name": "kidney"},
    "脾脏": {"entity_type": "organ", "standard_name": "spleen"},
    "胃": {"entity_type": "organ", "standard_name": "stomach"},
    "胰腺": {"entity_type": "organ", "standard_name": "pancreas"},
    "胆囊": {"entity_type": "organ", "standard_name": "gallbladder"},
    "甲状腺": {"entity_type": "organ", "standard_name": "thyroid"},
    "乳腺": {"entity_type": "organ", "standard_name": "breast"},
    "前列腺": {"entity_type": "organ", "standard_name": "prostate"},
    "子宫": {"entity_type": "organ", "standard_name": "uterus"},
    "卵巢": {"entity_type": "organ", "standard_name": "ovary"},
    "脑": {"entity_type": "organ", "standard_name": "brain"},
    "肠": {"entity_type": "organ", "standard_name": "intestine"},
    "结肠": {"entity_type": "organ", "standard_name": "colon"},
    "直肠": {"entity_type": "organ", "standard_name": "rectum"},
    "食管": {"entity_type": "organ", "standard_name": "esophagus"},
    "膀胱": {"entity_type": "organ", "standard_name": "bladder"},
    # ---- 解剖部位 ----
    "右肺上叶": {"entity_type": "anatomical_site", "standard_name": "right_upper_lobe"},
    "右肺中叶": {"entity_type": "anatomical_site", "standard_name": "right_middle_lobe"},
    "右肺下叶": {"entity_type": "anatomical_site", "standard_name": "right_lower_lobe"},
    "左肺上叶": {"entity_type": "anatomical_site", "standard_name": "left_upper_lobe"},
    "左肺下叶": {"entity_type": "anatomical_site", "standard_name": "left_lower_lobe"},
    "上腹部": {"entity_type": "anatomical_site", "standard_name": "upper_abdomen"},
    "下腹部": {"entity_type": "anatomical_site", "standard_name": "lower_abdomen"},
    "胸腔": {"entity_type": "anatomical_site", "standard_name": "thoracic_cavity"},
    "腹腔": {"entity_type": "anatomical_site", "standard_name": "abdominal_cavity"},
    "纵隔": {"entity_type": "anatomical_site", "standard_name": "mediastinum"},
    "腹膜后": {"entity_type": "anatomical_site", "standard_name": "retroperitoneum"},
    "锁骨上": {"entity_type": "anatomical_site", "standard_name": "supraclavicular"},
    "腋窝": {"entity_type": "anatomical_site", "standard_name": "axillary"},
    # ---- 药物 ----
    "阿司匹林": {"entity_type": "drug", "standard_name": "aspirin"},
    "二甲双胍": {"entity_type": "drug", "standard_name": "metformin"},
    "胰岛素": {"entity_type": "drug", "standard_name": "insulin"},
    "硝苯地平": {"entity_type": "drug", "standard_name": "nifedipine"},
    "氨氯地平": {"entity_type": "drug", "standard_name": "amlodipine"},
    "氯沙坦": {"entity_type": "drug", "standard_name": "losartan"},
    "阿托伐他汀": {"entity_type": "drug", "standard_name": "atorvastatin"},
    "瑞舒伐他汀": {"entity_type": "drug", "standard_name": "rosuvastatin"},
    "氯吡格雷": {"entity_type": "drug", "standard_name": "clopidogrel"},
    "华法林": {"entity_type": "drug", "standard_name": "warfarin"},
    "头孢": {"entity_type": "drug", "standard_name": "cephalosporin"},
    "青霉素": {"entity_type": "drug", "standard_name": "penicillin"},
    "奥美拉唑": {"entity_type": "drug", "standard_name": "omeprazole"},
    "甲氨蝶呤": {"entity_type": "drug", "standard_name": "methotrexate"},
    "紫杉醇": {"entity_type": "drug", "standard_name": "paclitaxel"},
    "顺铂": {"entity_type": "drug", "standard_name": "cisplatin"},
    "布洛芬": {"entity_type": "drug", "standard_name": "ibuprofen"},
    "地塞米松": {"entity_type": "drug", "standard_name": "dexamethasone"},
    # ---- 手术 ----
    "肺叶切除术": {"entity_type": "surgery", "standard_name": "lobectomy"},
    "全肺切除术": {"entity_type": "surgery", "standard_name": "pneumonectomy"},
    "冠状动脉搭桥术": {"entity_type": "surgery", "standard_name": "cabg"},
    "胆囊切除术": {"entity_type": "surgery", "standard_name": "cholecystectomy"},
    "阑尾切除术": {"entity_type": "surgery", "standard_name": "appendectomy"},
    "胃大部切除术": {"entity_type": "surgery", "standard_name": "subtotal_gastrectomy"},
    "髋关节置换术": {"entity_type": "surgery", "standard_name": "hip_arthroplasty"},
    "膝关节置换术": {"entity_type": "surgery", "standard_name": "knee_arthroplasty"},
    "剖宫产": {"entity_type": "surgery", "standard_name": "cesarean_section"},
    "支架植入术": {"entity_type": "surgery", "standard_name": "stent_implantation"},
    "穿刺活检": {"entity_type": "surgery", "standard_name": "needle_biopsy"},
    # ---- 检查/检验 ----
    "CT": {"entity_type": "examination", "standard_name": "ct_scan"},
    "CT检查": {"entity_type": "examination", "standard_name": "ct_scan"},
    "MRI": {"entity_type": "examination", "standard_name": "mri"},
    "核磁共振": {"entity_type": "examination", "standard_name": "mri"},
    "超声": {"entity_type": "examination", "standard_name": "ultrasound"},
    "B超": {"entity_type": "examination", "standard_name": "ultrasound"},
    "X线": {"entity_type": "examination", "standard_name": "x_ray"},
    "X光": {"entity_type": "examination", "standard_name": "x_ray"},
    "PET-CT": {"entity_type": "examination", "standard_name": "pet_ct"},
    "心电图": {"entity_type": "examination", "standard_name": "ecg"},
    "胃镜": {"entity_type": "examination", "standard_name": "gastroscopy"},
    "肠镜": {"entity_type": "examination", "standard_name": "colonoscopy"},
    "支气管镜": {"entity_type": "examination", "standard_name": "bronchoscopy"},
    # ---- 影像学发现 ----
    "毛玻璃影": {"entity_type": "imaging_finding", "standard_name": "ground_glass_opacity"},
    "实性结节": {"entity_type": "imaging_finding", "standard_name": "solid_nodule"},
    "磨玻璃结节": {"entity_type": "imaging_finding", "standard_name": "ggn"},
    "胸腔积液": {"entity_type": "imaging_finding", "standard_name": "pleural_effusion"},
    "气胸": {"entity_type": "imaging_finding", "standard_name": "pneumothorax"},
    "肺不张": {"entity_type": "imaging_finding", "standard_name": "atelectasis"},
    "淋巴结肿大": {"entity_type": "imaging_finding", "standard_name": "lymphadenopathy"},
    "远处转移": {"entity_type": "imaging_finding", "standard_name": "distant_metastasis"},
    "转移": {"entity_type": "imaging_finding", "standard_name": "metastasis"},
    "钙化": {"entity_type": "imaging_finding", "standard_name": "calcification"},
    # ---- 治疗措施 ----
    "化疗": {"entity_type": "treatment", "standard_name": "chemotherapy"},
    "放疗": {"entity_type": "treatment", "standard_name": "radiotherapy"},
    "靶向治疗": {"entity_type": "treatment", "standard_name": "targeted_therapy"},
    "免疫治疗": {"entity_type": "treatment", "standard_name": "immunotherapy"},
    "随访": {"entity_type": "treatment", "standard_name": "follow_up"},
    "定期复查": {"entity_type": "treatment", "standard_name": "regular_review"},
    "手术切除": {"entity_type": "treatment", "standard_name": "surgical_resection"},
    "保守治疗": {"entity_type": "treatment", "standard_name": "conservative_treatment"},
    # ---- 诊断结论 ----
    "未见异常": {"entity_type": "diagnosis", "standard_name": "no_abnormality"},
    "未见明显异常": {"entity_type": "diagnosis", "standard_name": "no_significant_abnormality"},
    "待排": {"entity_type": "diagnosis", "standard_name": "to_be_excluded"},
    "考虑": {"entity_type": "diagnosis", "standard_name": "suspected"},
}


# ===================================================================
# 正则规则库（用于捕获词典未覆盖的模式化表达）
# ===================================================================

_FALLBACK_REGEX_PATTERNS: list[tuple[str, str, str]] = [
    # (正则, entity_type, standard_name_prefix)
    # 病灶尺寸: "大小约12mm" / "直径约2.5cm"
    (r"(大小|直径)[约]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|毫米|厘米)", "imaging_finding", "lesion_size"),
    # 病灶描述: "XX占位" / "XX肿块" / "XX阴影"
    (r"(占位|肿块|阴影|团块|新生物)", "imaging_finding", "abnormal_finding"),
    # 病史: "XX病史" / "既往有XX"
    (r"既往[有患]?\s*(\S+?)(?:病史|史)", "disease_or_lesion", "history"),
    (r"(\S+?)病史\s*(\d+)\s*年", "disease_or_lesion", "history"),
    # 用药: "服用XX"
    (r"(服用|口服|静脉滴注|静滴|肌注)\s*(\S{2,8})", "drug", "medication"),
    # 否定表达: "未见XX" / "不考虑XX"
    (r"未见\s*(\S{2,8})", "diagnosis", "absent"),
    (r"不[考虑排除]\s*(\S{2,8})", "diagnosis", "excluded"),
    # 建议: "建议XX"
    (r"建议\s*(\S{2,6})", "treatment", "recommended"),
]


# ===================================================================
# MedicalNEREngine
# ===================================================================


class MedicalNEREngine:
    """临床医学命名实体识别引擎。

    双引擎架构：
      - **主线引擎**: 基于 HuggingFace Transformers 的 Token Classification 模型
      - **兜底引擎**: 高性能词典 + 正则规则匹配（零外部依赖，永久可用）

    Attributes:
        model_name: HuggingFace 模型名称。
        cache_dir: 模型缓存目录。
        device: 推理设备 (cuda:0 / cpu)。
        use_fallback: 是否启用兜底引擎。
        _model: Transformers NER pipeline（主线模式时非空）。
    """

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        cache_dir: str = "/root/autodl-tmp/d_group_system/checkpoints/ner_model",
    ) -> None:
        """初始化 NER 引擎。

        Args:
            model_name: HuggingFace 预训练模型名称。
            cache_dir: 模型缓存目录路径。
        """
        self.model_name: str = model_name
        self.cache_dir: str = os.path.abspath(cache_dir)
        self.use_fallback: bool = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._pipeline: Any = None

        # ---- 硬件检测 ----
        self.device: str = self._detect_device()
        logger.info(f"硬件设备: {self.device}")

        # ---- 模型加载 ----
        self._load_or_initialize_model()

    # ------------------------------------------------------------------
    # 硬件检测
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """自动检测并返回最佳可用设备。

        Returns:
            "cuda:0" 若 CUDA GPU 可用，否则 "cpu"。
        """
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到 GPU: {gpu_name}")
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # 模型加载与管理
    # ------------------------------------------------------------------

    def _load_or_initialize_model(self) -> None:  # noqa: C901
        """加载或初始化 Transformers NER 模型。

        加载优先级:
          1. 从 ``cache_dir`` 加载已缓存的模型
          2. 从 HuggingFace 下载并缓存到 ``cache_dir``
          3. 以上均失败 → 启用兜底引擎 (self.use_fallback = True)
        """
        model_path = os.path.join(self.cache_dir, "pytorch_model.bin")
        config_path = os.path.join(self.cache_dir, "config.json")

        # 检查是否已有缓存
        cached = os.path.isfile(model_path) and os.path.isfile(config_path)

        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            if cached:
                logger.info(f"从缓存加载模型: {self.cache_dir}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.cache_dir)
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.cache_dir
                )
            else:
                logger.info(f"缓存未命中，从 HuggingFace 下载: {self.model_name}")
                os.makedirs(self.cache_dir, exist_ok=True)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name
                )
                # 保存到缓存目录
                self._tokenizer.save_pretrained(self.cache_dir)
                self._model.save_pretrained(self.cache_dir)
                logger.info(f"模型已缓存至: {self.cache_dir}")

            # 构建 pipeline
            device_id = 0 if self.device.startswith("cuda") else -1
            self._pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device_id,
                aggregation_strategy="simple",
            )
            self.use_fallback = False
            logger.info("深度学习 NER 引擎加载成功")

        except Exception as e:
            logger.warning(f"Transformers 模型加载失败: {e}")
            logger.warning("自动切换至兜底引擎 (Regex + Dictionary)")
            self.use_fallback = True
            self._model = None
            self._tokenizer = None
            self._pipeline = None

    # ------------------------------------------------------------------
    # 兜底引擎：词典 + 正则
    # ------------------------------------------------------------------

    def _fallback_regex_extract(self, text: str) -> list[dict[str, Any]]:
        """词典与正则规则兜底抽取引擎。

        策略:
          1. 词典全词匹配（按词长降序，优先匹配长词避免碎片化）
          2. 正则模式匹配（捕获词典未覆盖的通用模式）

        Args:
            text: 待抽取的原始临床文本。

        Returns:
            实体列表，每个实体含 entity, entity_type, start, end,
            standard_name, confidence 字段。
        """
        entities: list[dict[str, Any]] = []
        occupied: set[tuple[int, int]] = set()  # 已匹配区间，防止重叠

        # ---- 词典匹配（按词长降序，长词优先） ----
        sorted_terms = sorted(
            _CLINICAL_DICT.items(), key=lambda x: len(x[0]), reverse=True
        )
        for term, info in sorted_terms:
            start = 0
            while True:
                idx = text.find(term, start)
                if idx == -1:
                    break
                span = (idx, idx + len(term))
                if not any(
                    s < span[1] and e > span[0] for s, e in occupied
                ):
                    occupied.add(span)
                    entities.append(
                        {
                            "entity": term,
                            "entity_type": info["entity_type"],
                            "start": idx,
                            "end": idx + len(term),
                            "standard_name": info["standard_name"],
                            "confidence": 0.95,
                        }
                    )
                start = idx + 1

        # ---- 正则模式匹配 ----
        for pattern, etype, std_prefix in _FALLBACK_REGEX_PATTERNS:
            for m in re.finditer(pattern, text):
                span = (m.start(), m.end())
                if any(s < span[1] and e > span[0] for s, e in occupied):
                    continue
                occupied.add(span)
                # 尝试从捕获组中提取核心内容作为 standard_name
                matched_text = m.group(0)
                if m.lastindex and m.lastindex >= 2:
                    core = m.group(m.lastindex)
                else:
                    core = matched_text
                entities.append(
                    {
                        "entity": matched_text,
                        "entity_type": etype,
                        "start": m.start(),
                        "end": m.end(),
                        "standard_name": f"{std_prefix}_{core}",
                        "confidence": 0.80,
                    }
                )

        # 按 start 位置排序
        entities.sort(key=lambda x: x["start"])
        return entities

    # ------------------------------------------------------------------
    # 核心提取 API
    # ------------------------------------------------------------------

    def extract_entities(
        self,
        text: str,
        text_id: str = "text_001",
        patient_id: str = "p001",
    ) -> dict[str, Any]:
        """从临床文本中抽取命名实体并返回结构化结果。

        根据当前引擎模式（深度学习 / 兜底规则），执行实体抽取并统一
        输出为标准化 JSON-compatible 字典。

        Args:
            text: 待解析的临床文本。
            text_id: 文本唯一标识。
            patient_id: 患者唯一标识。

        Returns:
            结构化抽取结果字典，包含 text_id, patient_id, entities 列表。
        """
        raw_entities: list[dict[str, Any]]

        if self.use_fallback or self._pipeline is None:
            # ---- 兜底模式 ----
            raw_entities = self._fallback_regex_extract(text)
        else:
            # ---- 深度学习模式 ----
            try:
                ner_results = self._pipeline(text)
                raw_entities = []
                for item in ner_results:
                    raw_entities.append(
                        {
                            "entity": item.get("word", "").strip(),
                            "entity_type": item.get("entity_group", "unknown"),
                            "start": item.get("start", 0),
                            "end": item.get("end", 0),
                            "standard_name": item.get("word", "").strip().lower().replace(" ", "_"),
                            "confidence": round(item.get("score", 0.0), 2),
                        }
                    )
            except Exception as e:
                logger.error(f"深度学习推理失败: {e}，切换至兜底引擎")
                raw_entities = self._fallback_regex_extract(text)

        # 统一后处理：round confidence
        for ent in raw_entities:
            ent["confidence"] = round(ent["confidence"], 2)

        return {
            "text_id": text_id,
            "patient_id": patient_id,
            "entities": raw_entities,
        }

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """返回引擎运行状态摘要。

        Returns:
            包含 engine_mode, device, model_name, cache_dir 的字典。
        """
        return {
            "engine_mode": "fallback_regex" if self.use_fallback else "deep_learning",
            "device": self.device,
            "model_name": self.model_name,
            "cache_dir": self.cache_dir,
        }


# ===================================================================
# 验证与自测模块
# ===================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  临床医学命名实体识别引擎 (MedicalNEREngine) — 功能验证")
    print("=" * 72)

    # ---- 初始化引擎 ----
    engine = MedicalNEREngine()
    stats = engine.get_stats()
    print(f"\n[引擎状态]")
    print(f"  推理设备:   {stats['device']}")
    print(f"  运行模式:   {stats['engine_mode']}")
    print(f"  模型名称:   {stats['model_name']}")
    print(f"  缓存目录:   {stats['cache_dir']}")

    # ---- 经典临床现病史测试文本 ----
    test_text = (
        "患者既往有高血压病史5年，规律服用阿司匹林。"
        "昨日复查CT示：右肺上叶可见一大小约12mm的肺结节，"
        "未见明显远处转移，不考虑肺癌，建议随访。"
    )

    print(f"\n[测试文本]")
    print(f"  {test_text}")

    # ---- 执行实体抽取 ----
    result = engine.extract_entities(
        text=test_text,
        text_id="test_001",
        patient_id="p000_demo",
    )

    print(f"\n[抽取结果] 共识别 {len(result['entities'])} 个实体:\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ---- 实体类型分布统计 ----
    type_counts: dict[str, int] = {}
    for ent in result["entities"]:
        t = ent["entity_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n[实体类型分布]")
    for etype, count in sorted(type_counts.items()):
        chinese_label = ENTITY_TYPES.get(etype, etype)
        bar = "█" * count
        print(f"  {etype:<22} ({chinese_label:<8}): {count} {bar}")

    print("\n" + "=" * 72)
    print("  验证完毕！")
    print("=" * 72)
