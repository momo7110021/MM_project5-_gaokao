# 数据字典

> Project 5 高考志愿填报多目标推荐模型 — 数据字典
>
> 版本: v3.0.0 | 与 src/crawler.py 和 pipeline.py 字段保持一致

---

## 数据采集说明

- 七类数据分别由 `src/crawler.py` 中对应的七类采集器负责采集
- 所有采集数据必须携带 `source_trace` 追溯字段
- 无法自动采集的数据通过 `manual_import_template()` 人工导入
- 模拟数据由 `src/data_generator.py` 生成，仅用于教学、测试和流程验证，不作为真实填报依据
- 真实上线需要人工校验公开数据源

---

## source_trace 追溯字段说明

所有数据表均携带以下 5 个追溯字段：

| 字段名 | 类型 | 含义 | 示例 |
|--------|------|------|------|
| `source_url` | VARCHAR(500) | 数据来源 URL 或文件路径 | `http://www.hebeea.edu.cn/2024/segment.html` |
| `source_name` | VARCHAR(100) | 数据来源标识（采集器名称） | `segment_table` |
| `crawl_time` | DATETIME | 数据采集时间 | `2026-05-15 20:00:00` |
| `data_version` | VARCHAR(20) | 数据版本号 | `v1.0` |
| `source_type` | VARCHAR(20) | 来源类型（html/pdf/excel/csv/manual/simulated/api/dynamic） | `html` |

---

## 表 1：segment_table（一分一段表）

- **业务用途**：模型一 分数—位次等效换算（核心输入）
- **对应采集器**：`SegmentTableCrawler`（src/crawler.py）
- **更新频率**：每年高考出分后更新
- **公开来源**：各省教育考试院官网、阳光高考平台

| 字段名 | 类型 | 含义 | 必填 | 数据来源 | 缺失值处理 | 进入模型 |
|--------|------|------|------|----------|-----------|----------|
| `province` | VARCHAR(20) | 省份名称 | 是 | 爬取 | 不可缺失 | 模型一 |
| `year` | SMALLINT | 高考年份 | 是 | 爬取 | 不可缺失 | 模型一 |
| `subject_type` | VARCHAR(20) | 科类（物理类/历史类） | 是 | 爬取 | 不可缺失 | 模型一 |
| `score` | SMALLINT | 分数（0-750） | 是 | 爬取 | 不可缺失 | 模型一 |
| `segment_count` | INT | 本段人数 | 是 | 爬取 | 相邻插值 | 模型一 |
| `cumulative_count` | INT | 累计人数 | 是 | 爬取 | 相邻插值 | 模型一 |
| `rank` | INT | 位次 | 否 | 计算 | 从cumulative_count计算 | 模型一 |
| `percentile` | DECIMAL(10,6) | 分位点 | 否 | 计算 | `cumulative_count/total` | 模型一 |
| `batch_line` | SMALLINT | 批次线分数 | 是 | 爬取 | 不可缺失 | 模型一 |
| `batch_line_type` | VARCHAR(20) | 批次线类型 | 否 | 爬取 | 标记未知 | 模型一 |
| `total_exam_count` | INT | 该科类考生总数 | 否 | 爬取 | `cumulative_count.max()` | 模型一 |

---

## 表 2：school_info（院校基本信息表）

- **业务用途**：院校筛选、城市价值评估
- **对应采集器**：`AdmissionLineCrawler`（部分字段）、人工维护
- **更新频率**：年度更新（院校合并/更名时）

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `school_code` | VARCHAR(20) | 院校标准代码 | 是 | 模型二、模型三 |
| `school_name` | VARCHAR(100) | 院校标准名称 | 是 | 模型二、模型三 |
| `province` | VARCHAR(20) | 所在省份 | 否 | 城市评估 |
| `city` | VARCHAR(50) | 所在城市 | 否 | 模型三、城市评估 |
| `school_level` | VARCHAR(20) | 层次（985/211/双一流/普通） | 否 | 模型二（热度校正） |
| `is_public` | TINYINT | 是否公办（0/1） | 否 | 模型三（预算匹配） |
| `school_type` | VARCHAR(20) | 类型（综合/理工/师范/医药等） | 否 | 模型二（参考） |

---

## 表 3：school_admission_line（院校投档线表）

- **业务用途**：模型二 录取概率预测（核心输入）、模型三 候选志愿构建
- **对应采集器**：`AdmissionLineCrawler`
- **更新频率**：每年投档结束后更新

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `province` | VARCHAR(20) | 生源省份 | 是 | 模型二 |
| `year` | SMALLINT | 年份 | 是 | 模型二 |
| `batch` | VARCHAR(20) | 批次（本科批/提前批） | 是 | 模型二 |
| `subject_type` | VARCHAR(20) | 科类 | 是 | 模型二 |
| `school_code` | VARCHAR(20) | 院校代码 | 是 | 模型二、模型三 |
| `school_name` | VARCHAR(100) | 院校名称 | 否 | 模型二、模型三 |
| `major_group_code` | VARCHAR(30) | 专业组代码 | 否 | 模型二、模型四 |
| `min_admission_score` | SMALLINT | 最低投档分 | 是 | 模型二、模型三 |
| `min_admission_rank` | INT | 最低投档位次 | 是 | 模型二 |
| `plan_count` | INT | 招生计划数 | 否 | 模型二（计划变化率） |
| `admission_count` | INT | 实际录取人数 | 否 | 模型四（退档风险） |

---

## 表 4：major_admission（专业录取数据表）

- **业务用途**：模型二 录取概率预测（核心输入）、模型四 调剂量风险评估
- **对应采集器**：`MajorAdmissionCrawler`

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `school_code` | VARCHAR(20) | 院校代码 | 是 | 模型二 |
| `major_code` | VARCHAR(20) | 专业代码 | 是 | 模型二、模型五 |
| `major_name` | VARCHAR(100) | 专业名称 | 否 | 模型二、模型三、模型五 |
| `major_group_code` | VARCHAR(30) | 专业组代码 | 否 | 模型二、模型四 |
| `province` | VARCHAR(20) | 生源省份 | 是 | 模型二 |
| `year` | SMALLINT | 年份 | 是 | 模型二 |
| `min_admission_score` | SMALLINT | 专业最低录取分 | 是 | 模型二 |
| `min_admission_rank` | INT | 专业最低录取位次 | 是 | 模型二 |
| `avg_score` | SMALLINT | 平均分 | 否 | 模型二 |
| `max_score` | SMALLINT | 最高分 | 否 | 模型二 |
| `plan_count` | INT | 招生计划数 | 否 | 模型二 |
| `admission_count` | INT | 实际录取人数 | 否 | 模型四 |
| `subject_requirement` | VARCHAR(200) | 选科要求 | 否 | 模型三（硬约束） |
| `adjustment_rule` | VARCHAR(200) | 调剂量规则 | 否 | 模型四 |

---

## 表 5：admission_plan（招生计划表）

- **业务用途**：模型三 志愿组合硬约束（选科、学费、身体条件、单科限制）
- **对应采集器**：`EnrollmentPlanCrawler`

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `school_code` | VARCHAR(20) | 院校代码 | 是 | 模型三 |
| `major_code` | VARCHAR(20) | 专业代码 | 是 | 模型三 |
| `major_name` | VARCHAR(100) | 专业名称 | 否 | 模型三 |
| `major_group_code` | VARCHAR(30) | 专业组代码 | 否 | 模型三 |
| `province` | VARCHAR(20) | 招生省份 | 是 | 模型三 |
| `year` | SMALLINT | 招生年份 | 是 | 模型三 |
| `plan_count` | INT | 计划数 | 是 | 模型三 |
| `tuition` | DECIMAL(10,2) | 学费（元/年） | 否 | 模型三（预算约束） |
| `duration` | TINYINT | 学制（年） | 否 | 模型三 |
| `subject_requirement` | VARCHAR(200) | 选科要求 | 否 | 模型三（选科约束） |
| `is_sino_foreign` | TINYINT | 是否中外合作 | 否 | 模型三 |
| `is_normal_major` | TINYINT | 是否师范类 | 否 | 模型三 |
| `is_medical_major` | TINYINT | 是否医学类 | 否 | 模型三 |
| `single_subject_limit` | VARCHAR(200) | 单科成绩限制 | 否 | 模型三（硬约束） |
| `physical_limit` | VARCHAR(200) | 身体条件限制 | 否 | 模型三（硬约束） |
| `remark` | VARCHAR(500) | 备注 | 否 | 模型三（参考） |

---

## 表 6：major_employment（专业就业数据表）

- **业务用途**：模型五 专业就业景气度评价（核心输入）
- **对应采集器**：`MajorEmploymentCrawler`
- **更新频率**：年度更新（就业质量报告发布后）

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `major_code` | VARCHAR(20) | 专业代码 | 是 | 模型五 |
| `major_name` | VARCHAR(100) | 专业名称 | 否 | 模型五 |
| `employment_rate` | DECIMAL(5,4) | 就业落实率（含升学） | 否 | 模型五(AHP) |
| `postgraduate_rate` | DECIMAL(5,4) | 国内升学率 | 否 | 模型五(AHP)、读研价值 |
| `civil_service_post_count` | INT | 近三年考公岗位数 | 否 | 模型五(AHP)、考公适配度 |
| `average_salary` | DECIMAL(10,2) | 平均月薪（元） | 否 | 模型五(AHP)、薪资评分 |
| `median_salary` | DECIMAL(10,2) | 中位数月薪（元） | 否 | 模型五(AHP)、薪资评分 |
| `job_count` | INT | 招聘岗位数 | 否 | 模型五(AHP) |
| `job_growth_rate` | DECIMAL(6,4) | 岗位同比增长率 | 否 | 模型五(AHP)、成长性 |
| `industry_growth_score` | DECIMAL(4,2) | 行业成长性(1-10) | 否 | 模型五(AHP) |
| `stability_score` | DECIMAL(4,2) | 就业稳定性(1-10) | 否 | 模型五(AHP)、稳定度 |
| `sentiment_warning_score` | DECIMAL(4,2) | 社媒舆情预警(0-1) | 否 | 模型五（仅辅助预警，不参与TOPSIS） |
| `data_year` | SMALLINT | 数据年份 | 是 | 模型五 |
| `sample_size` | INT | 样本量 | 否 | 模型五（数据可靠性评估） |
| `industry_distribution` | TEXT | 行业分布(JSON) | 否 | 参考 |
| `main_employment_city` | TEXT | 主要就业城市(JSON) | 否 | 参考 |

---

## 表 7：city_data（城市产业数据表）

- **业务用途**：城市价值评估、城市—专业—产业匹配（扩展模型）
- **对应采集器**：`CityDataCrawler`

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `city` | VARCHAR(50) | 城市名称 | 是 | 城市评估 |
| `province` | VARCHAR(20) | 所在省份 | 否 | 城市评估 |
| `gdp` | DECIMAL(15,2) | GDP（亿元） | 否 | 城市评估 |
| `gdp_per_capita` | DECIMAL(10,2) | 人均GDP（元） | 否 | 城市评估 |
| `tertiary_industry_ratio` | DECIMAL(5,4) | 第三产业占比 | 否 | 城市评估 |
| `key_industries` | TEXT | 重点产业(JSON) | 否 | 城市评估 |
| `high_tech_company_count` | INT | 高新企业数 | 否 | 城市评估 |
| `listed_company_count` | INT | 上市公司数 | 否 | 城市评估 |
| `related_job_count` | INT | 相关岗位数 | 否 | 城市—专业匹配 |
| `average_salary` | DECIMAL(10,2) | 城市平均月薪 | 否 | 城市评估 |
| `living_cost` | DECIMAL(10,2) | 月均生活成本 | 否 | 城市评估 |
| `distance_from_home` | INT | 距考生家乡距离(km) | 否 | 家庭匹配 |
| `transport_score` | DECIMAL(3,1) | 交通便利度(1-10) | 否 | 城市评估 |

---

## 表 8：candidate_profile（考生画像表）

- **业务用途**：考生个性化约束和偏好（全部模型共用）
- **对应采集器**：`CandidateProfileCollector`（前端问卷/人工录入/CSV导入）
- **注意**：此表数据由用户在前端填写，不涉及公开爬取

| 字段名 | 类型 | 含义 | 必填 | 进入模型 |
|--------|------|------|------|----------|
| `candidate_id` | VARCHAR(32) | 考生唯一标识 | 是 | 全部模型 |
| `province` | VARCHAR(20) | 考生省份 | 是 | 全部模型 |
| `year` | SMALLINT | 高考年份 | 是 | 全部模型 |
| `subject_type` | VARCHAR(20) | 科类 | 是 | 模型一、二、三 |
| `score` | SMALLINT | 总分 | 是 | 模型一、二 |
| `rank` | INT | 位次 | 是 | 模型一、二 |
| `interest_direction` | VARCHAR(500) | 兴趣方向(JSON) | 否 | 模型二（M_fit） |
| `strong_subjects` | VARCHAR(200) | 优势科目(JSON) | 否 | 模型三（选科约束） |
| `excluded_majors` | VARCHAR(1000) | 排斥专业(JSON) | 否 | 模型三（硬约束） |
| `preferred_cities` | VARCHAR(1000) | 偏好城市(JSON) | 否 | 模型三（地域约束） |
| `family_budget` | DECIMAL(10,2) | 家庭预算上限(元) | 否 | 模型三（预算约束） |
| `risk_preference` | VARCHAR(10) | 风险偏好 | 是 | 模型三（方案选择） |
| `accept_adjustment` | TINYINT | 是否接受调剂量 | 否 | 模型四（调剂量风险） |
| `accept_sino_foreign` | TINYINT | 是否接受中外合作 | 否 | 模型三 |
| `accept_far_city` | TINYINT | 是否接受远距离城市 | 否 | 模型三（地域约束） |
| `employment_first` | TINYINT | 是否就业优先 | 否 | 模型三（效用权重参考） |
| `postgraduate_first` | TINYINT | 是否读研优先 | 否 | 模型五（读研价值权重） |

---

## 表 9：model_output（模型输出结果表）

- **业务用途**：存储每次模型运行的完整输出结果（可追溯、可审核）
- **来源**：`pipeline.generate_output()` 统一生成

| 字段名 | 类型 | 含义 |
|--------|------|------|
| `id` | BIGINT | 自增主键 |
| `candidate_id` | VARCHAR(32) | 考生ID |
| `model_version` | VARCHAR(20) | 模型版本号 |
| `school_code` | VARCHAR(20) | 院校代码 |
| `major_code` | VARCHAR(20) | 专业代码 |
| `admit_probability` | DECIMAL(6,4) | 录取概率 |
| `recommendation_tier` | VARCHAR(10) | 推荐等级 |
| `career_score` | DECIMAL(4,2) | 就业评分 |
| `city_score` | DECIMAL(4,2) | 城市评分 |
| `overall_utility` | DECIMAL(8,4) | 综合效用 |
| `risk_level` | VARCHAR(10) | 风险等级 |
| `explanation` | TEXT | 家长端解释 |
| `modification_suggestion` | TEXT | 修改建议 |
| `review_required` | TINYINT | 是否需复核 |
| `create_time` | DATETIME | 生成时间 |

---

## 模拟数据说明

当前 `src/data_generator.py` 生成的数据集仅用于：

1. 教学和学术展示
2. 代码测试和流程验证
3. pipeline 联调和接口调试

**模拟数据不能作为真实高考志愿填报依据**。真实上线前必须：

1. 使用 `src/crawler.py` 各采集器从公开来源爬取真实数据
2. 经过 `src/cleaner.py` 六级清洗流水线处理
3. 人工校验关键字段（分数线、位次、就业数据等）
4. 按省份、科类、选科组合分层后重新训练和回测模型
