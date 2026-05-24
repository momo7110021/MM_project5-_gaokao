# 基于大数据和 AI 的高考志愿填报多目标推荐模型研究

---

## 摘要

高考志愿填报是每年涉及千万家庭的高利害决策问题。考生面临的挑战包括：不同年份高考分数不可直接比较、院校专业录取概率难以量化、志愿表冲稳保结构难以优化、专业就业前景难以评估、滑档退档等风险难以识别。本文提出了一套基于大数据和人工智能的高考志愿填报多目标推荐模型体系，包含五大核心模型：分数—位次等效换算模型、院校/专业录取概率预测模型、专业就业景气度评价模型、冲稳保志愿组合优化模型和志愿填报风险评估模型。模型以多目标综合效用函数为框架，综合录取概率、专业匹配度、就业价值、城市价值和家庭资源匹配度五项正向指标，并引入风险惩罚项，实现对志愿表的全局优化。本文同时设计了完整的数据爬取、清洗、建库和建模流程，涵盖一分一段表、院校投档线、专业录取数据、招生计划、就业数据和城市产业数据等七类公开数据的采集方案。模型输出统一 JSON 接口，可直接接入咨询系统或小程序。经模拟数据验证，本模型体系能够有效降低滑档风险，提高志愿表合理性，具备较好的工程落地能力和业务解释力。

**关键词**：高考志愿填报；位次等效换算；录取概率预测；多目标优化；风险评估；数据爬取；TOPSIS；贝叶斯修正

---

## 一、问题重述

### 1.1 问题背景

高考是我国高等教育选拔人才的核心制度。每年约 1000 万考生参加高考，每位考生需要在考后短时间内，从数千所院校、数万个专业中完成志愿填报。志愿填报质量直接影响考生能否被理想院校录取、就读适合专业、实现职业发展目标。

当前高考志愿填报面临的核心矛盾是：**信息不对称与决策时间有限之间的矛盾**。考生和家长通常缺乏对院校录取规律、专业就业前景、城市发展潜力的系统性认知，往往凭借有限经验和简单规则做出决策，导致志愿填报不合理的情况大量存在。

### 1.2 核心问题

本文需解决的六个核心问题：

1. **跨年度分数不可比问题**：今年 580 分与去年 580 分代表完全不同的竞争水平，需要建立科学的分数—位次等效换算方法。
2. **录取概率难以量化问题**：考生报考某院校、专业组或专业的录取概率需要基于历史数据和当年招生计划进行预测，而非依赖单一分数线。
3. **志愿表结构优化问题**：需要生成冲、稳、保、垫结构合理的完整志愿表，而非推荐单个学校。
4. **专业就业前景评估问题**：需要基于多源就业数据评估专业的就业景气度、行业成长性和冷门风险。
5. **多维度风险识别问题**：需要识别滑档、退档、调剂、就业、地域等风险，给出预警和修正建议。
6. **模型输出可解释问题**：模型结果需要对家长可理解、咨询师可复核、系统可调用。

### 1.3 核心约束

- 数据需自行从公开来源爬取，不可使用现成数据集。
- 爬虫必须遵守法律法规，仅采集公开数据，不绕过反爬限制。
- 模型输出不得包含"保证录取""绝对安全"等违规表述。
- 模型需服务于业务，所有公式和算法必须可计算、可解释、可验证。

---

## 二、问题分析

### 2.1 问题链条

高考志愿填报问题的完整决策链条如下：

```
考生分数 → 位次换算 → 等效分映射 → 录取概率计算 → 专业匹配分析 → 
就业前景评估 → 城市价值评估 → 家庭约束过滤 → 志愿组合优化 → 
风险评估 → 志愿表输出
```

### 2.2 建模思路

本文采用**分层建模、逐层递进**的策略：

- **第一层（基础层）**：分数—位次等效换算模型。解决不同年份分数不可比的问题，是后续所有模型的基础。
- **第二层（预测层）**：院校/专业录取概率预测模型。基于等效分和历史录取数据，量化录取概率。
- **第三层（评价层）**：专业就业景气度评价模型。基于就业多源数据，对专业进行多维评价。
- **第四层（决策层）**：冲稳保志愿组合优化模型。以多目标效用函数为指导，生成结构合理的志愿表。
- **第五层（保障层）**：志愿填报风险评估模型。识别和预警各类风险，给出修正建议。

### 2.3 总体模型框架

本文建立如下多目标综合效用函数作为总模型：

$$\max U = \alpha P_{admit} + \beta M_{fit} + \gamma E_{career} + \delta C_{city} + \eta R_{family} - \lambda R_{risk}$$

其中：
- $P_{admit}$：录取概率得分，由核心模型二计算
- $M_{fit}$：专业匹配度得分，基于考生兴趣与专业特征匹配
- $E_{career}$：专业就业价值得分，由核心模型三计算
- $C_{city}$：城市价值得分
- $R_{family}$：家庭资源匹配度得分
- $R_{risk}$：综合风险得分，由核心模型五计算
- $\alpha, \beta, \gamma, \delta, \eta, \lambda$ 为权重系数，根据风险偏好调整

---

## 三、数据来源与爬取方案

### 3.1 数据爬取总原则

1. **合法合规**：只采集公开可访问的数据，不绕过登录、验证码、反爬限制或付费权限。
2. **访问限速**：每次请求间隔不少于 3 秒，避免对目标服务器造成压力。
3. **可追溯**：每条数据记录保存来源 URL 和采集时间。
4. **分类解析**：对 HTML、Excel、CSV、PDF 采用不同解析方式。
5. **人工校对**：对无法自动解析的数据设计人工校对机制。

### 3.2 第一类：一分一段表数据

#### 数据来源

| 来源 | 格式 | 说明 |
|------|------|------|
| 各省教育考试院官网 | HTML/Excel/PDF | 官方公布的一分一段表 |
| 阳光高考网 (gaokao.chsi.com.cn) | HTML | 汇总各省一分一段表 |
| 中国教育在线 (eol.cn) | HTML | 历史一分一段表整理 |

#### 爬取方式

- **HTML 表格**：使用 `requests` + `BeautifulSoup/lxml` 解析 `<table>` 标签
- **PDF**：使用 `pdfplumber` 或 `camelot` 解析表格
- **Excel**：使用 `pandas` 的 `read_excel()` 直接读取

#### 字段设计

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| province | VARCHAR(20) | 省份 | 广东 |
| year | INT | 年份 | 2024 |
| subject_type | VARCHAR(20) | 科类/选科组合 | 物理类 |
| score | INT | 分数 | 600 |
| segment_count | INT | 本段人数 | 156 |
| cumulative_count | INT | 累计人数 | 28937 |
| rank | INT | 位次 | 28937 |
| percentile | FLOAT | 分位点 (%) | 85.23 |
| batch_line | INT | 批次线 | 539 |
| crawl_time | DATETIME | 采集时间 | 2024-07-01 10:00:00 |
| source_url | VARCHAR(500) | 来源链接 | https://... |

### 3.3 第二类：历年院校投档线数据

#### 数据来源

| 来源 | 格式 | 说明 |
|------|------|------|
| 各省教育考试院投档线公告 | HTML/PDF/Excel | 每年投档最低分和位次 |
| 阳光高考网 | HTML | 院校库和历年分数线 |
| 高校本科招生网 | HTML | 各校公布的录取数据 |

#### 字段设计

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| province | VARCHAR(20) | 省份 | 广东 |
| year | INT | 年份 | 2024 |
| batch | VARCHAR(20) | 批次 | 本科批 |
| subject_type | VARCHAR(20) | 科类 | 物理类 |
| school_code | VARCHAR(20) | 院校代码 | 10559 |
| school_name | VARCHAR(100) | 院校名称 | 暨南大学 |
| major_group_code | VARCHAR(30) | 专业组代码 | 201 |
| min_toudang_score | INT | 最低投档分 | 605 |
| min_toudang_rank | INT | 最低投档位次 | 18500 |
| plan_count | INT | 计划数 | 120 |
| admission_count | INT | 实际录取数 | 122 |
| crawl_time | DATETIME | 采集时间 | ... |
| source_url | VARCHAR(500) | 来源链接 | ... |

### 3.4 第三类：历年专业录取数据

#### 数据来源

| 来源 | 格式 | 说明 |
|------|------|------|
| 高校本科招生网录取查询 | HTML/PDF | 各专业历年录取分数 |
| 招生章程 | PDF | 各专业选科和身体要求 |
| 志愿填报指南（官方出版物） | PDF/纸质 | 需人工导入 |

#### 字段设计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| school_code | VARCHAR(20) | 院校代码 |
| school_name | VARCHAR(100) | 院校名称 |
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| province | VARCHAR(20) | 省份 |
| year | INT | 年份 |
| min_admission_score | INT | 最低录取分 |
| min_admission_rank | INT | 最低录取位次 |
| avg_score | FLOAT | 平均分 |
| max_score | INT | 最高分 |
| plan_count | INT | 计划数 |
| admission_count | INT | 实际录取数 |
| subject_requirement | VARCHAR(100) | 选科要求 |
| adjustment_rule | VARCHAR(200) | 调剂规则 |

### 3.5 第四类：招生计划与专业限制数据

#### 字段设计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| school_code | VARCHAR(20) | 院校代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| province | VARCHAR(20) | 省份 |
| year | INT | 年份 |
| plan_count | INT | 计划数 |
| tuition | INT | 学费 |
| duration | INT | 学制 |
| subject_requirement | VARCHAR(100) | 选科要求 |
| is_sino_foreign | BOOLEAN | 是否中外合作 |
| is_normal_major | BOOLEAN | 是否师范类 |
| is_medical_major | BOOLEAN | 是否医学类 |
| single_subject_limit | VARCHAR(200) | 单科成绩限制 |
| physical_limit | VARCHAR(200) | 身体条件限制 |
| remark | TEXT | 备注 |

#### 招生章程中限制条件的正则提取

```python
import re

def extract_subject_requirement(text):
    """提取选科要求"""
    patterns = [
        r'选考科目[：:]\\s*(.*?)[\\n。]',
        r'选科要求[：:]\\s*(.*?)[\\n。]',
        r'再选科目[：:]\\s*(.*?)[\\n。]',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1).strip()
    return None

def extract_single_subject_limit(text):
    """提取单科成绩限制"""
    patterns = [
        r'(外语|数学|语文|英语)(?:成绩|单科)[^\\d]*(\\d+)[分]?',
        r'(外语|数学|语文|英语)[^\\d]*不低于[^\\d]*(\\d+)',
    ]
    limits = []
    for pat in patterns:
        matches = re.findall(pat, text)
        limits.extend(matches)
    return limits
```

### 3.6 第五类：专业就业数据

#### 数据来源

| 来源 | 数据类型 | 说明 |
|------|----------|------|
| 高校就业质量报告 | PDF | 各校分专业就业率、升学率 |
| 教育部专业目录 | HTML | 专业分类、代码 |
| 国家统计局 | HTML/Excel | 行业薪资、增长率 |
| 招聘网站公开岗位 | HTML | 岗位数量、薪资分布 |
| 行业研究报告 | PDF | 行业发展趋势 |

#### 字段设计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| employment_rate | FLOAT | 就业率 |
| postgraduate_rate | FLOAT | 升学率 |
| civil_service_post_count | INT | 考公对口岗位数 |
| average_salary | FLOAT | 平均薪资 |
| median_salary | FLOAT | 中位数薪资 |
| job_count | INT | 招聘岗位数 |
| job_growth_rate | FLOAT | 岗位增长率 |
| industry_distribution | JSON | 行业分布 |
| main_employment_city | VARCHAR(50) | 主要就业城市 |
| industry_growth_score | FLOAT | 行业成长评分 |
| stability_score | FLOAT | 稳定度评分 |
| sentiment_warning_score | FLOAT | 舆论预警分 |

#### 数据匹配说明

- **院校就业质量报告**通常以 PDF 形式发布，需用 `pdfplumber` 提取表格，再按专业名称匹配到标准专业代码。
- **招聘岗位数据**与专业的匹配采用关键词匹配法：提取专业名称及同义词，在岗位名称和岗位描述中检索。
- 就业率和薪资数据**仅作为参考**，需向用户说明数据来源和局限性。

### 3.7 第六类：城市与产业数据

#### 字段设计

| 字段名 | 类型 | 说明 |
|--------|------|------|
| city | VARCHAR(50) | 城市 |
| gdp | FLOAT | GDP (亿元) |
| gdp_per_capita | FLOAT | 人均 GDP |
| tertiary_industry_ratio | FLOAT | 第三产业占比 |
| key_industries | VARCHAR(500) | 重点产业 |
| high_tech_company_count | INT | 高新企业数 |
| listed_company_count | INT | 上市公司数 |
| related_job_count | INT | 相关岗位数 |
| average_salary | FLOAT | 平均薪资 |
| living_cost | FLOAT | 生活成本指数 |
| distance_from_home | INT | 距家乡距离 (km) |
| transport_score | FLOAT | 交通便利评分 |

### 3.8 第七类：考生画像与家庭偏好数据

此数据由用户在系统中填写，不需要爬取，但需要设计数据采集界面和字段约束。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| candidate_id | VARCHAR(32) | PK |
| province_code | VARCHAR(10) | 省份 |
| subject_type | VARCHAR(20) | 科类 |
| score | INT | 总分 |
| rank | INT | 位次 |
| interest_direction | VARCHAR(200) | 兴趣方向 |
| strong_subjects | VARCHAR(100) | 优势学科 |
| excluded_majors | VARCHAR(500) | 排除专业 |
| preferred_cities | VARCHAR(500) | 意向城市 |
| family_budget | INT | 预算上限 |
| risk_preference | VARCHAR(20) | 风险偏好 |
| accept_adjustment | BOOLEAN | 接受调剂 |
| accept_sino_foreign | BOOLEAN | 接受中外合作 |
| accept_far_city | BOOLEAN | 接受远距离城市 |
| employment_first | BOOLEAN | 就业优先 |
| postgraduate_first | BOOLEAN | 升学优先 |

### 3.9 数据无法自动爬取的应对方案

1. **纸质官方出版物**：提供 Excel 模板，由人工录入后上传系统。
2. **需要登录或验证码的页面**：不绕过，改为收集公开汇总数据或购买合法数据接口。
3. **PDF 表格解析失败**：标记为"待人工校对"，显示原始 PDF 截图供人工录入。
4. **院校和专业名称不一致**：建立名称映射表，由管理员维护标准化名称。

---

## 四、数据清洗与数据库设计

### 4.1 数据清洗规则

#### 4.1.1 缺失值处理

| 情况 | 处理方式 |
|------|----------|
| 关键字段缺失且无法推断 | 标记为无效记录，不参与模型计算 |
| 位次缺失但分数和一分一段表存在 | 通过一分一段表查表填充 |
| 计划数缺失 | 使用该院校该专业近三年均值填充 |
| 就业数据缺失 | 标记为"数据不足"，输出时标注不确定性 |

#### 4.1.2 异常值处理

- 录取分数超过满分 750 或低于 0：标记为异常，人工核查。
- 录取位次与同分段位次差异超过 3 个标准差：标记为异常。
- 就业率超过 100%：修正为 100%。
- 同比增长率超过 +/-200%：标记为异常，人工核查。

#### 4.1.3 重复值处理

- 按 `(province, year, school_code, major_code)` 组合键去重。
- 若记录字段存在差异，优先保留 `source_url` 更权威的记录（教育考试院 > 阳光高考 > 高校官网 > 第三方平台）。

#### 4.1.4 口径统一

- **省份名称**：统一为"广东"而非"广东省"，建立省份代码表。
- **院校代码**：使用教育部标准代码，建立各年代码映射表。
- **专业代码**：使用教育部《普通高等学校本科专业目录》代码。
- **科类**：统一为"物理类""历史类""综合类"等标准名称。
- **批次**：统一为"本科批""专科批""提前批"等。

### 4.2 数据库表结构设计

#### 表 1：一分一段表 `t_score_segment`

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | 自增主键 |
| province_code | VARCHAR(10) | NOT NULL | 省份代码 |
| year | INT | NOT NULL | 年份 |
| subject_type | VARCHAR(20) | NOT NULL | 科类 |
| score | INT | NOT NULL | 分数 |
| segment_count | INT | | 本段人数 |
| cumulative_count | INT | | 累计人数 |
| rank | INT | | 位次 |
| percentile | DECIMAL(10,6) | | 分位点 |
| batch_line | INT | | 批次线 |
| crawl_time | DATETIME | | 采集时间 |
| source_url | VARCHAR(500) | | 来源链接 |

主键：`id` | 唯一索引：`(province_code, year, subject_type, score)` | 更新频率：每年 6 月

#### 表 2：院校基本信息表 `t_school_info`

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| school_code | VARCHAR(20) | PK | 院校代码 |
| school_name | VARCHAR(100) | NOT NULL | 院校名称 |
| school_type | VARCHAR(20) | | 院校类型(综合/理工/师范) |
| school_level | VARCHAR(20) | | 层次(985/211/双一流/普通) |
| province | VARCHAR(20) | | 所在省份 |
| city | VARCHAR(50) | | 所在城市 |
| is_public | BOOLEAN | | 是否公办 |
| website | VARCHAR(200) | | 官网 |

#### 表 3：专业基本信息表 `t_major_info`

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| major_code | VARCHAR(20) | PK | 专业代码 |
| major_name | VARCHAR(100) | NOT NULL | 专业名称 |
| category | VARCHAR(50) | | 专业类别 |
| sub_category | VARCHAR(50) | | 专业子类 |
| degree | VARCHAR(20) | | 授予学位 |

#### 表 4：历年院校投档线表 `t_school_admission_line`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | PK |
| province_code | VARCHAR(10) | 省份 |
| year | INT | 年份 |
| batch | VARCHAR(20) | 批次 |
| subject_type | VARCHAR(20) | 科类 |
| school_code | VARCHAR(20) | 院校代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| min_toudang_score | INT | 最低投档分 |
| min_toudang_rank | INT | 最低投档位次 |
| plan_count | INT | 计划数 |
| admission_count | INT | 录取数 |

唯一索引：`(province_code, year, subject_type, school_code, major_group_code)`

#### 表 5：历年专业录取表 `t_major_admission`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | PK |
| school_code | VARCHAR(20) | 院校代码 |
| major_code | VARCHAR(20) | 专业代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| province_code | VARCHAR(10) | 省份 |
| year | INT | 年份 |
| min_admission_score | INT | 最低录取分 |
| min_admission_rank | INT | 最低录取位次 |
| avg_score | FLOAT | 平均分 |
| max_score | INT | 最高分 |
| plan_count | INT | 计划数 |
| admission_count | INT | 录取数 |
| is_adjustment | BOOLEAN | 是否调剂录取 |

#### 表 6：招生计划表 `t_enrollment_plan`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | PK |
| school_code | VARCHAR(20) | 院校代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| major_code | VARCHAR(20) | 专业代码 |
| province_code | VARCHAR(10) | 省份 |
| year | INT | 年份 |
| plan_count | INT | 计划数 |
| tuition | INT | 学费 |
| duration | INT | 学制 |
| subject_requirement | VARCHAR(100) | 选科要求 |
| is_sino_foreign | BOOLEAN | 中外合作 |
| is_normal_major | BOOLEAN | 师范类 |
| is_medical_major | BOOLEAN | 医学类 |
| single_subject_limit | VARCHAR(200) | 单科限制 |
| physical_limit | VARCHAR(200) | 身体限制 |

#### 表 7：专业就业数据表 `t_major_employment`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| major_code | VARCHAR(20) | 专业代码 |
| year | INT | 数据年份 |
| employment_rate | DECIMAL(5,4) | 就业率 |
| postgraduate_rate | DECIMAL(5,4) | 升学率 |
| average_salary | INT | 平均薪资 |
| median_salary | INT | 中位数薪资 |
| job_count | INT | 招聘岗位数 |
| job_growth_rate | DECIMAL(5,4) | 岗位增长率 |
| civil_service_post_count | INT | 考公岗位数 |
| industry_growth_score | DECIMAL(5,2) | 行业成长评分 |
| stability_score | DECIMAL(5,2) | 稳定度评分 |
| sentiment_warning_score | DECIMAL(5,2) | 舆论预警分 |

#### 表 8：城市产业数据表 `t_city_industry`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| city_code | VARCHAR(10) | 城市代码 |
| city_name | VARCHAR(50) | 城市名 |
| year | INT | 年份 |
| gdp | FLOAT | GDP |
| gdp_per_capita | FLOAT | 人均 GDP |
| tertiary_industry_ratio | FLOAT | 第三产业占比 |
| key_industries | TEXT | 重点产业 |
| high_tech_company_count | INT | 高新企业数 |
| listed_company_count | INT | 上市公司数 |
| average_salary | FLOAT | 平均薪资 |
| living_cost_index | FLOAT | 生活成本指数 |

#### 表 9：考生画像表 `t_candidate_profile`（字段见第三章第七节）

#### 表 10：模型输出结果表 `t_model_output`

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | BIGINT | PK |
| candidate_id | VARCHAR(32) | 考生 ID |
| generate_time | DATETIME | 生成时间 |
| school_code | VARCHAR(20) | 院校代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| major_code | VARCHAR(20) | 专业代码 |
| admit_probability | DECIMAL(5,4) | 录取概率 |
| probability_interval | VARCHAR(50) | 概率区间 |
| recommendation_tier | VARCHAR(10) | 冲/稳/保/垫 |
| fit_score | DECIMAL(5,2) | 匹配度 |
| career_score | DECIMAL(5,2) | 就业评分 |
| city_score | DECIMAL(5,2) | 城市评分 |
| family_score | DECIMAL(5,2) | 家庭匹配评分 |
| risk_level | VARCHAR(10) | 风险等级 |
| overall_utility | DECIMAL(8,4) | 综合效用 |
| explanation | TEXT | 解释文本 |
| review_required | BOOLEAN | 是否需复核 |

---

## 五、模型假设

1. **分数—位次映射稳定性假设**：同一省份、同一年份、同一科类下，考生位次与分数之间的映射关系是稳定且单调的，一分一段表能够准确反映该映射关系。

2. **历史位次可比性假设**：在批次线、考生人数等因素经修正后，不同年份的考生位次具有可比性。即某院校录取的考生在该年考生群体中的相对位置（分位点），在不同年份间具有可比性。

3. **招生计划影响连续假设**：院校和专业的招生计划、热门程度、社会认知等影响录取位次的因素，其变化是连续的、可建模的，不发生断崖式变化（如院校合并、专业被撤销等极端事件除外）。

4. **考生行为理性假设**：考生群体的志愿填报行为在宏观上表现为理性，即高分考生倾向于报考高分院校，形成稳定的院校梯度结构。

5. **专业就业数据代表性假设**：各高校就业质量报告、招聘平台公开数据能够在一定程度上反映专业就业前景，但存在口径差异，模型已为此引入不确定性量化。

6. **数据可获取性假设**：假设各省教育考试院公开数据能够正常访问和解析。若某年某省数据无法获取，使用相邻年份数据进行合理插补，并标记为"数据缺失"。

7. **线性加权可加性假设**：假设各评价维度之间相互独立，可以采用线性加权方式合成总效用。用户可通过方案选择（激进/均衡/保守）调整权重。

---

## 六、符号说明

| 符号 | 含义 | 单位 |
|------|------|------|
| $S_{cur}$ | 考生当年分数 | 分 |
| $R_{cur}$ | 考生当年位次 | 无 |
| $P_{cur}$ | 考生当年分位点 | % |
| $S_{eq}^{(t)}$ | 第 t 年等效分 | 分 |
| $R_{eq}^{(t)}$ | 第 t 年等效位次 | 无 |
| $\Delta_{line}$ | 线差（分数减批次线） | 分 |
| $p_{admit}$ | 录取概率 | [0,1] |
| $\hat{p}_{admit}$ | 修正后录取概率 | [0,1] |
| $M_{fit}$ | 专业匹配度得分 | [0,1] |
| $E_{career}$ | 专业就业价值得分 | [0,1] |
| $C_{city}$ | 城市价值得分 | [0,1] |
| $R_{family}$ | 家庭资源匹配得分 | [0,1] |
| $R_{risk}$ | 综合风险得分 | [0,1] |
| $U$ | 综合效用 | [-1,1] |
| $\sigma$ | 历史录取位次标准差 | |
| $x_i$ | 第 i 个志愿是否入选 (0/1) | |
| $N_{rush}$ | 冲刺志愿数量 | |
| $N_{stable}$ | 稳妥志愿数量 | |
| $N_{safe}$ | 保底志愿数量 | |
| $N_{bottom}$ | 垫底志愿数量 | |
| $Q$ | 志愿表总容量 | |
| $R_{slip}$ | 滑档风险得分 | |
| $R_{withdrawal}$ | 退档风险得分 | |
| $R_{adjustment}$ | 调剂风险得分 | |
| $R_{cold}$ | 专业冷门风险得分 | |
| $R_{employment}$ | 就业风险得分 | |
| $R_{region}$ | 地域风险得分 | |
