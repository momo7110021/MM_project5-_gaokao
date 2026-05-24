# 业务解释模板

> Project 5 高考志愿填报多目标推荐模型 — 业务解释规范
>
> 版本: v3.0.0 | 适用端: 家长端 / 咨询师端 / 系统后台端

---

## 1. 解释原则

1. **不承诺录取**：录取概率为基于历史数据的估计值，不构成录取保证。
2. **不夸大就业**：就业评分为基于公开数据的参考值，不同学校、地区和个人情况存在差异。
3. **不替代官方政策**：所有推荐仅供参考，最终决策以各省教育考试院官方信息为准。
4. **高风险结果必须提示人工复核**：review_required=True 时必须明确建议咨询专业人士。
5. **推荐理由必须可追溯到模型字段**：每条解释对应的模型输出字段和计算依据需在后台端可查。

---

## 2. 禁止表述

以下表述在任何端、任何上下文中均**禁止出现**：

- "保证录取"
- "一定能上" / "肯定录取"
- "绝对安全"
- "稳赚就业" / "稳就业"
- "未来一定高薪" / "必定高薪"
- "该专业一定适合" / "绝对适合您"
- "不需要复核"
- "录取概率 100%" / "100%录取"
- "薪资最低 XXXX 元"（应表述为"参考约 XXXX 元"）

---

## 3. 家长端解释模板

**语言要求**：通俗、克制、自然，不使用过多模型术语，一句话能说出推荐理由。

### 3.1 录取概率解释模板

> 报考{学校名称}的{专业名称}专业，预估录取概率约 XX%（区间 XX%-XX%），属于{冲/稳/保/垫}范围。
> {补充说明：如"该志愿有较大希望被录取" / "该志愿录取难度较大，建议搭配保底志愿"}

**示例**：

> 报考A理工大学的计算机科学与技术专业，预估录取概率约65%（区间50%-85%），属于"保"范围。该志愿有较大把握被录取，建议结合个人兴趣和职业规划综合考虑。

### 3.2 冲稳保垫解释模板

| 等级 | 家长端表述 |
|------|-----------|
| rush (冲) | 录取概率偏低，是较高目标，可以尝试。建议同时填报足够多的保底志愿。 |
| stable (稳) | 录取概率中等，在可争取范围内，有较大概率录取。 |
| safe (保) | 录取概率较高，基本能够被录取，是稳妥选择。 |
| bottom (垫) | 录取概率很高，作为兜底志愿，确保不被滑档。 |

### 3.3 专业就业解释模板

> {专业名称}的综合就业景气度为{较好/中等/需关注}（{绿色/黄色/红色}标签）。
> 近年就业落实率约 XX%，相关岗位月薪参考约 XXXX 元（不同城市和行业有差异）。
> {如红色}该专业就业竞争较激烈，建议结合个人兴趣和家庭资源综合评估。

**注意**：薪资必须使用"参考约"等不确定措辞，不得承诺具体收入。

### 3.4 风险提示模板

> 经评估，您的志愿表综合风险为{低/中等/较高/高}。
> 主要风险点：{简述1-2条最相关的风险原因}。
> 建议{具体建议}。

### 3.5 人工复核提示模板

> 注意：部分志愿的历史数据较少或波动较大，预测结果不确定性较高，建议咨询专业人士进行人工复核。

---

## 4. 咨询师端解释模板

**语言要求**：专业、结构化，包含模型依据、风险触发条件和可调整项。

### 4.1 录取概率解释模板

> 录取概率: {admit_probability} ({probability_interval})
> 推荐等级: {recommendation_tier}
> 不确定性: {uncertainty_level}
> 历史数据年数: {n_years_data}
> 分项概率: LR={p_lr} XGB={p_xgb} Bayes={p_bayes} MC={p_mc}
> Top影响因素: {feature1}({value}), {feature2}({value}), {feature3}({value})

### 4.2 志愿优化解释模板

> 方案类型: {plan_type}
> 阈值配置: rush<{threshold_rush} stable<{threshold_stable} bottom>={threshold_bottom}
> 目标比例: 冲{ratio_rush} 稳{ratio_stable} 保{ratio_safe} 垫{ratio_bottom}
> 实际结果: 冲{rush_count} 稳{stable_count} 保{safe_count} 垫{bottom_count}
> 硬约束满足: 排斥专业已过滤 / 选科匹配 / 预算约束 / 地域偏好

### 4.3 风险解释模板

> 综合风险评分: {overall_risk_score} ({risk_level})
> 滑档风险: {slip_risk.score} {trigger_reason} | 退档风险: {withdrawal_risk.score} ...
> 识别的问题: {Q1-Q10 列表}
> 修改建议: {modification_suggestion}

### 4.4 就业评价解释模板

> 专业: {major_name} career_score={career_score} ({red_yellow_green_label})
> 薪资评分={salary_score} 稳定度={stability_score} 成长性={growth_score} 读研价值={postgraduate_value_score} 考公适配度={civil_service_score}
> 冷门风险={cold_major_risk_score} 风险类型={risk_types}
> 数据可靠性={data_reliability_level} 趋势={trend_label}
> 注意: sentiment_warning_score 仅作为辅助预警，不作为核心就业评价依据。

---

## 5. 系统后台解释模板

**格式要求**：结构化 dict/JSON，便于追溯和审核。

### 5.1 通用结构

```json
{
  "model": "模型名称",
  "method": "建模方法",
  "parameters": {},
  "weights": {},
  "source_table": "数据来源表名",
  "result": {},
  "review_required": true,
  "warnings": []
}
```

### 5.2 模型二（录取概率）后台解释示例

```json
{
  "model": "模型二: 录取概率预测",
  "method": "multi_model_fusion(LR+XGBoost+Bayes+MC)",
  "fusion_weights": {"lr": 0.15, "xgb": 0.35, "bayes": 0.20, "mc": 0.30},
  "thresholds": {"rush": 0.20, "stable": 0.45, "safe": 0.70, "bottom": 0.88},
  "mc_iterations": 10000,
  "is_trained": true,
  "training_meta": {"is_simulated": true, "n_train_samples": 5000},
  "result": {
    "admit_probability": 0.6521,
    "probability_interval": [0.5000, 0.8500],
    "recommendation_tier": "safe"
  },
  "review_required": false
}
```

### 5.3 模型五（就业评价）后台解释示例

```json
{
  "model": "模型五: 专业就业景气度评价",
  "method": "AHP + Entropy + TOPSIS + KMeans clustering",
  "ahp_weights": {"employment_rate": 0.13, "average_salary": 0.20, ...},
  "combined_weights": {"employment_rate": 0.14, ...},
  "is_simulated": true,
  "result": {
    "career_score": 0.8521,
    "red_yellow_green_label": "green",
    "salary_source_explanation": {"is_simulated": true, ...}
  }
}
```

---

## 6. 五个核心模型解释模板

### 6.1 模型一：分数—位次等效换算

**家长端**："您今年考了 X 分，在全省排第 Y 名。相当于往年约 Z 分（区间 A-B 分）。这是因为每年试卷难度和考生人数不同，不能直接用今年分数比较往年录取分数。位次比分数更稳定。"

**咨询师端**："等效分 Z，区间[A,B]，置信度: {confidence_level}。换算方法: 分位数映射+线差修正(theta=0.3)+异常年份降权(tau=1000)。异常年份: {list}。当前数据可用性: {review_required?}"

**后台端**：`{"method":"quantile_mapping+batch_line_correction+abnormal_year_downweight", "parameters":{"theta":0.3,"tau":1000}, "source_table":"segment_table"}`

### 6.2 模型二：录取概率预测

**家长端**："报考{学校}的{专业}，预估录取概率约 X%（区间 A%-B%），属于{冲/稳/保/垫}。{补充说明}"

**咨询师端**：含概率区间、分项概率、Top特征、省份/科类校正、小样本警告

**后台端**：含融合权重、阈值配置、训练元信息、特征重要性

### 6.3 模型三：志愿组合优化

**家长端**："为您推荐 Y 个志愿：冲 X1 个（录取概率较低）、稳 X2 个、保 X3 个、垫 X4 个（兜底）。已根据您的排斥专业、预算和偏好城市进行过滤。"

**咨询师端**：含方案类型、阈值、硬约束检查结果、建议调整项\"

**后台端**：含目标函数权重、硬约束校验清单、局部搜索迭代次数

### 6.4 模型四：风险评估

**家长端**："经评估，您的志愿表综合风险为{低/中等/较高}。主要风险：{简述}。建议{建议}。"

**咨询师端**：含六类风险评分、十个问题扫描结果、具体修改建议

**后台端**：含蒙特卡洛迭代次数、风险矩阵权重、风险触发阈值

### 6.5 模型五：就业景气度评价

**家长端**："{专业}的综合就业景气度为{较好/中等/需关注}（{绿色/黄色/红色}标签）。近年就业落实率约 X%，参考月薪约 XXXX 元（不同城市有差异）。"

**咨询师端**：含 career_score、五维细分评分、红黄绿标签、趋势分析、数据可靠性

**后台端**：含 AHP/熵权权重、TOPSIS 正负理想解距离、KMeans 聚类中心

---

## 7. 完整推荐解释示例

> 报考A理工大学的计算机科学与技术专业，预估录取概率约65%（区间50%-85%），属于"保"等级。
>
> 该专业综合就业景气度为"较好"（绿色标签），近年就业落实率约94%，相关岗位月薪参考约8000-9000元（不同城市和行业有差异，数据来自就业质量报告和招聘网站公开数据）。
>
> 经评估，您的志愿表综合风险为中等。主要风险：保底志愿偏少。建议增加2-3个录取概率高于88%的兜底志愿。
>
> 以上分析基于历史数据和公开就业数据，不构成录取或就业保证。最终决策请结合个人情况并咨询专业人士。

**关联字段**：
- admit_probability: 0.65 | recommendation_tier: "safe" | career_score: 0.85
- red_yellow_green_label: "green" | overall_risk_score: 0.18 (medium)
- review_required: false
