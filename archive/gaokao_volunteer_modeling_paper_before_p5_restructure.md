# 基于大数据和 AI 的高考志愿填报多目标推荐模型研究

---

## 摘要

高考志愿填报是影响考生未来发展的关键决策环节，但每年全国约千万考生面临信息不对称、数据量庞大、政策复杂等困难。本文以"基于大数据和 AI 的高考志愿填报系统"为研究对象，提出了一套完整、可计算、可解释、可接入系统的数学建模方案。

本文首先设计了覆盖七大类数据（一分一段表、院校投档线、专业录取数据、招生计划、专业就业数据、城市产业数据、考生画像数据）的爬取、清洗与建库方案，明确了每类数据的来源、字段和清洗规则。在此基础上，本文构建了五个核心模型：(1) 基于分位数映射与线差修正的分数—位次等效换算模型，将跨年度不可直接比较的原始分数统一转换为可比等效分；(2) 基于 XGBoost 与贝叶斯修正的院校/专业录取概率预测模型，输出带有置信区间的录取概率估计；(3) 基于 AHP-熵权-TOPSIS 组合赋权的专业就业景气度评价模型，从多维度评价专业就业价值并输出红黄绿风险标签；(4) 基于整数规划的冲稳保志愿组合优化模型，在众多候选志愿中生成结构合理的志愿表；(5) 基于蒙特卡洛模拟和风险矩阵的志愿填报风险评估模型，定量识别滑档、退档、调剂等六类风险。

本文进一步以多目标综合效用函数 `max U = alpha*P_admit + beta*M_fit + gamma*E_career + delta*C_city + eta*R_family - lambda*Risk` 为总框架，融合五个子模型，针对激进型、均衡型、保守型三类风险偏好产出个性化志愿方案。最后，本文给出了统一 JSON 接口设计、Python 爬虫与算法伪代码、技术验收与业务验收指标体系，以及模型推广与改进方向。

**关键词**：高考志愿填报；录取概率预测；位次换算；就业景气度；多目标优化；风险评估；机器学习

---

## 论文大纲

- 摘要与关键词
- 一、问题重述
- 二、问题分析
- 三、数据来源与爬取方案
- 四、数据清洗与数据库设计
- 五、模型假设
- 六、符号说明
- 七、分数—位次等效换算模型（核心模型一）
- 八、院校/专业录取概率预测模型（核心模型二）
- 九、专业就业景气度评价模型（核心模型三）
- 十、冲稳保志愿组合优化模型（核心模型四）
- 十一、志愿填报风险评估模型（核心模型五）
- 十二、多目标综合推荐模型
- 十三、系统接口与输出设计
- 十四、模型评价与验收标准
- 十五、模型优缺点分析
- 十六、模型推广与改进方向
- 参考文献
- 附录


---

## 一、问题重述

### 1.1 问题背景

高考是中国大陆规模最大、影响最深远的教育选拔考试。每年约 1000 万考生在知晓成绩和位次后，需要在短短数日内完成志愿填报。不同省份的志愿填报规则各不相同（如"院校+专业组""专业+院校"等模式），且每年招生计划、报考人数、专业热度均会发生变化，使得历史数据的直接参考价值有限。

当前高考志愿填报面临六大核心问题：

(1) **分数不可直接跨年比较**。今年 580 分与去年 580 分因试题难度、考生人数、招生计划等差异，代表的竞争力不同。

(2) **录取概率缺乏科学估计**。多数考生和家长凭"感觉"或单一的历史最低分数据填报，无法定量衡量录取可能性。

(3) **志愿表结构难以优化**。缺乏系统方法确保志愿表具备合理的"冲、稳、保、垫"梯度。

(4) **专业就业前景缺乏量化评估**。专业选择往往依赖名称直觉或碎片化信息，缺乏多维度的就业景气度量化评价。

(5) **风险识别不充分**。滑档、退档、调剂至冷门专业、就业困难等风险缺乏系统性的识别和预警机制。

(6) **个性化推荐能力不足**。考生的兴趣方向、家庭预算、地域偏好、风险承受能力等个性化因素难以融入统一的推荐框架。

### 1.2 问题目标

本文旨在建立一套完整的数学模型体系，实现以下目标：

(1) 判断考生报考某院校、专业组或专业的录取概率；
(2) 解决不同年份高考分数不可直接比较的问题；
(3) 生成冲、稳、保、垫结构合理的志愿表；
(4) 评估专业就业前景和专业冷门风险；
(5) 识别滑档、退档、调剂、就业、地域等风险；
(6) 输出家长能听懂、咨询师能复核、系统能调用的解释结果。

---

## 二、问题分析

### 2.1 核心矛盾分析

本问题的核心矛盾在于 **信息不充分** 与 **决策高利害** 之间的张力。考生需要在有限认知范围内做出影响深远的选择，而决策所需的信息（历史录取数据、专业就业数据、政策变动等）分散在不同渠道，难以系统整合。

### 2.2 问题分解

根据问题的层次性，将总目标分解为五个子问题：

(1) **等效分转换子问题**。给定考生当年分数与位次，如何将其映射为历史年份的等效分？核心挑战在于：不同年份试卷难度、考生规模、招生计划不同，原始分数不具可比性。

(2) **录取概率估计子问题**。给定考生等效位次和目标院校/专业的历史录取数据，如何估计录取概率？核心挑战在于：历史数据有限（通常 3-5 年），招生计划变动频繁，专业热度动态变化。

(3) **专业价值评价子问题**。如何从就业率、薪资水平、行业增长率、稳定性等多维度综合评价一个专业的就业价值？核心挑战在于：薪资数据口径不统一，就业报告覆盖不全，社媒舆情存在偏差。

(4) **志愿组合优化子问题**。如何在大量候选志愿中选择一组满足约束且总体效用最大的志愿组合？核心挑战在于：这是一个多目标、多约束的组合优化问题，需要平衡录取概率与个人偏好。

(5) **风险预警子问题**。如何从不确定性角度识别志愿方案中的潜在风险？核心挑战在于：风险来源多元，需要将不确定性定量化并映射为可操作的预警信息。

### 2.3 总体建模路线

本文采取"自底向上、分层建模、总分结合"的思路：

- **第一层**：建立分数等效换算模型，将原始分数统一为可比指标；
- **第二层**：基于可比指标，分别建立录取概率预测、就业景气度评价、城市价值评估等专项模型；
- **第三层**：将专项模型的结果融入多目标优化框架，生成个性化志愿方案；
- **第四层**：对方案进行风险扫描，给出预警和修正建议；
- **输出层**：将结果转化为家长可理解的自然语言解释和 JSON 结构化数据。

### 2.4 多目标综合效用函数

总模型框架为如下的多目标综合效用函数：

```
max U = alpha*P_admit + beta*M_fit + gamma*E_career + delta*C_city + eta*R_family - lambda*Risk
```

其中各符号含义如下：

| 符号 | 含义 | 业务解释 | 所需数据 |
|------|------|---------|---------|
| P_admit | 录取概率 | 考生被该志愿录取的可能性 | 历史投档线、专业录取数据、招生计划 |
| M_fit | 专业匹配度 | 专业与考生兴趣、优势科目的匹配程度 | 考生画像、专业目录 |
| E_career | 就业价值 | 专业毕业后的就业前景 | 就业率、薪资、岗位数量、行业增长数据 |
| C_city | 城市价值 | 院校所在城市的发展水平 | GDP、产业结构、生活成本 |
| R_family | 家庭资源匹配 | 考生家庭偏好和约束满足度 | 家庭预算、地域偏好、排斥专业 |
| Risk | 综合风险 | 滑档、退档、调剂等风险总和 | 历史录取波动数据、招生章程 |
| alpha,beta,gamma,delta,eta,lambda | 权重系数 | 不同风险偏好的权重 | 由考生 risk_preference 决定 |

各指标均归一化至 [0, 1] 区间。权重的确定将在第十二章详述。


---

## 三、数据来源与爬取方案

### 3.1 爬取原则

本文设计的数据采集方案严格遵循以下原则：

1. **只采集公开数据**。所有数据来源于官方网站、公开公告和合规公开信息。
2. **不绕过登录、验证码、反爬限制或付费权限**。对于需要登录才能访问的数据，通过人工辅助录入方式补全。
3. **设置访问间隔**。每次请求间隔不少于 3 秒，避免对目标服务器造成负担。
4. **保留数据来源链接和采集时间**。每条数据记录原始 URL 和抓取时间戳。
5. **对 PDF、Excel、网页表格采用不同解析方式**。
6. **对无法解析的数据设计人工校对机制**。
7. **遵守 robots.txt 协议**。在爬取前检查目标网站的 robots.txt，遵循其禁止和允许规则。
8. **避免违法违规爬虫行为**。不使用任何绕过安全措施的技术手段，不采集用户个人隐私信息。

### 3.2 数据分类与采集方案

#### 3.2.1 第一类：一分一段表数据

**用途**：分数—位次等效换算模型的核心输入。

**来源**：各省教育考试院官方网站（如河北省教育考试院）、省级招生考试信息网、阳光高考平台（https://gaokao.chsi.com.cn）、省级教育考试院发布的官方 PDF 或 Excel 公告。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| province | VARCHAR(20) | 省份名称 |
| year | SMALLINT | 年份 |
| subject_type | VARCHAR(20) | 科类，"物理类"/"历史类"/"理科"/"文科" |
| score | SMALLINT | 高考分数 |
| segment_count | INT | 本段人数 |
| cumulative_count | INT | 累计人数 |
| rank | INT | 位次 |
| percentile | DECIMAL(10,6) | 分位点 |
| batch_line | SMALLINT | 该批次分数线 |
| source_url | VARCHAR(500) | 数据来源链接 |
| crawl_time | DATETIME | 数据采集时间 |

**爬取方式**：
- HTML 表格（如阳光高考）：使用 requests + BeautifulSoup 解析 table 标签
- Excel 文件：使用 pandas.read_excel() 读取
- PDF 文件（最常见）：使用 pdfplumber 或 camelot 解析表格
- 动态网页：使用 Selenium 等待页面加载后提取

**清洗规则**：
- 缺失值：关键字段（总分、累计人数）缺失比例超过 5% 的行标记"待人工审核"；零星缺失利用相邻分数段线性插值补全
- 异常值：分数超过 0-750 范围的标记为异常；累计人数不单调递增的标记异常
- 重复值：按 (province, year, subject_type, score) 去重
- 口径不一致：建立全省份科类标准化映射表

**不可自动爬取的应对**：若 PDF 表格格式复杂、行列识别错误率高，则：(1) 优先寻找同一数据的 Excel 版本；(2) 若无 Excel 版本，设计人工对照校正流程，两名录入员独立录入后交叉比对。

#### 3.2.2 第二类：历年院校投档线数据

**来源**：省教育考试院发布的各批次投档线公告、高校本科招生网、阳光高考平台院校信息库。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| province | VARCHAR(20) | 省份 |
| year | SMALLINT | 年份 |
| batch | VARCHAR(20) | 批次 |
| subject_type | VARCHAR(20) | 科类 |
| school_code | VARCHAR(20) | 院校代码 |
| school_name | VARCHAR(100) | 院校名称 |
| major_group_code | VARCHAR(30) | 专业组代码（新高考省份） |
| min_admission_score | SMALLINT | 最低投档分 |
| min_admission_rank | INT | 最低投档位次 |
| plan_count | INT | 招生计划数 |
| admission_count | INT | 实际录取人数 |
| source_url | VARCHAR(500) | 来源链接 |
| crawl_time | DATETIME | 采集时间 |

**爬取方式**：HTML 网页公告使用 requests + BeautifulSoup；Excel 文件使用 pandas.read_excel()；PDF 投档线表使用 pdfplumber + 正则提取。

**清洗规则**：
- 院校名称标准化：建立院校名称清洗映射表（如"北京大学"与"北京大学(校本部)"统一为"北京大学"）
- 院校代码统一：以教育部公布的院校代码为准，建立年份-代码对照表
- 投档分与位次一致性检验：投档分与当年一分一段表的对应位次关系应一致

#### 3.2.3 第三类：历年专业录取数据

**来源**：高校本科招生网"历年录取分数"查询系统、省级招办发布的分专业录取情况、高校招生章程、志愿填报指南（纸质版需人工录入）。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| school_code | VARCHAR(20) | 院校代码 |
| school_name | VARCHAR(100) | 院校名称 |
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| province | VARCHAR(20) | 生源省份 |
| year | SMALLINT | 年份 |
| min_admission_score | SMALLINT | 专业最低录取分 |
| min_admission_rank | INT | 专业最低录取位次 |
| avg_score | SMALLINT | 平均分 |
| max_score | SMALLINT | 最高分 |
| plan_count | INT | 该专业招生计划数 |
| admission_count | INT | 该专业实际录取人数 |
| subject_requirement | VARCHAR(200) | 选科要求 |
| adjustment_rule | VARCHAR(200) | 调剂规则说明 |
| source_url | VARCHAR(500) | 来源链接 |

**爬取方式**：高校招生网查询系统通过 POST 请求模拟查询；历年录取分数 PDF 汇总使用 pdfplumber 解析；动态加载页面使用 Selenium 模拟选择后获取结果。

**清洗规则**：专业名称标准化（关联教育部《普通高等学校本科专业目录》）；min <= avg <= max 逻辑校验；专业最低分不应低于院校投档线（除非有特殊招生政策）。

#### 3.2.4 第四类：招生计划与专业限制数据

**来源**：省教育考试院发布的《普通高等学校招生计划》、高校招生章程、阳光高考平台"招生章程"栏目。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| school_code | VARCHAR(20) | 院校代码 |
| major_group_code | VARCHAR(30) | 专业组代码 |
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| province | VARCHAR(20) | 招生省份 |
| year | SMALLINT | 招生年份 |
| plan_count | INT | 计划数 |
| tuition | DECIMAL(10,2) | 学费（元/年） |
| duration | TINYINT | 学制（年） |
| subject_requirement | VARCHAR(200) | 选科要求 |
| is_sino_foreign | TINYINT | 是否中外合作办学 |
| is_normal_major | TINYINT | 是否师范类专业 |
| is_medical_major | TINYINT | 是否医学类专业 |
| single_subject_limit | VARCHAR(200) | 单科成绩限制 |
| physical_limit | VARCHAR(200) | 身体条件限制 |
| remark | VARCHAR(500) | 备注说明 |

**爬取方式**：招生章程 HTML 使用 requests + BeautifulSoup 提取，结合正则表达式匹配特殊限制（如"英语单科成绩不低于120分"）；招生计划 PDF 使用 pdfplumber 解析。


#### 3.2.5 第五类：专业就业数据

**来源**：各高校《毕业生就业质量年度报告》（PDF）、教育部《普通高等学校本科专业目录》、国家统计局"全国城镇单位就业人员平均工资"、人社部"最缺工"职业排行、主流招聘网站（BOSS直聘、智联招聘等）公开的行业报告、公务员招录职位表。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| major_code | VARCHAR(20) | 专业代码 |
| major_name | VARCHAR(100) | 专业名称 |
| employment_rate | DECIMAL(5,4) | 就业率（含升学） |
| postgraduate_rate | DECIMAL(5,4) | 国内升学率 |
| civil_service_post_count | INT | 近三年国考省考对应岗位数 |
| average_salary | DECIMAL(10,2) | 近三年平均月薪（元） |
| median_salary | DECIMAL(10,2) | 中位数月薪（元） |
| job_count | INT | 招聘网站近一年岗位发布量 |
| job_growth_rate | DECIMAL(6,4) | 岗位数量同比增长率 |
| industry_distribution | TEXT | 行业分布（JSON） |
| main_employment_city | TEXT | 主要就业城市（JSON） |
| industry_growth_score | DECIMAL(4,2) | 行业成长性评分（1-10） |
| stability_score | DECIMAL(4,2) | 就业稳定性评分（1-10） |
| sentiment_warning_score | DECIMAL(4,2) | 社媒舆情预警分（0-1，越高越危险） |
| source_url | VARCHAR(500) | 来源链接 |
| data_year | SMALLINT | 数据年份 |
| crawl_time | DATETIME | 采集时间 |

**口径说明**：就业率采用高校就业质量报告中的"毕业去向落实率"口径。薪资综合高校就业报告、国家统计局分行业工资、招聘网站公开数据，取近三年中位数，注明不同来源的薪资口径差异。

#### 3.2.6 第六类：城市与产业数据

**来源**：国家统计局城市GDP、《中国城市统计年鉴》、各城市统计局年度公报、上市公司年报（巨潮资讯网）、高新技术企业认定管理工作网、生活成本指数、交通便利度数据。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| city | VARCHAR(50) | 城市名称 |
| gdp | DECIMAL(15,2) | GDP（亿元） |
| gdp_per_capita | DECIMAL(10,2) | 人均GDP（元） |
| tertiary_industry_ratio | DECIMAL(5,4) | 第三产业占比 |
| key_industries | TEXT | 重点产业（JSON数组） |
| high_tech_company_count | INT | 高新技术企业数量 |
| listed_company_count | INT | 上市公司数量 |
| related_job_count | INT | 与选定专业相关的岗位数 |
| average_salary | DECIMAL(10,2) | 城市平均月薪（元） |
| living_cost | DECIMAL(10,2) | 月均生活成本（元） |
| distance_from_home | INT | 距考生家庭所在城市的距离（km） |
| transport_score | DECIMAL(3,1) | 交通便利度评分（1-10） |
| source_url | VARCHAR(500) | 来源链接 |
| data_year | SMALLINT | 数据年份 |

#### 3.2.7 第七类：考生画像与家庭偏好数据

**来源**：通过系统前端问卷/表单由考生和家长手动填写（非爬取数据）。

**字段设计**：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| candidate_id | VARCHAR(32) | 考生唯一标识（UUID） |
| province | VARCHAR(20) | 考生所在省份 |
| subject_type | VARCHAR(20) | 选科组合 |
| score | SMALLINT | 高考总分 |
| rank | INT | 全省位次 |
| interest_direction | VARCHAR(500) | 兴趣方向（JSON数组） |
| strong_subjects | VARCHAR(200) | 优势科目（JSON数组） |
| excluded_majors | VARCHAR(1000) | 排斥专业（JSON数组） |
| preferred_cities | VARCHAR(1000) | 偏好城市（JSON数组） |
| family_budget | DECIMAL(10,2) | 家庭年预算上限（元） |
| risk_preference | VARCHAR(10) | "aggressive"/"balanced"/"conservative" |
| accept_adjustment | TINYINT | 是否接受调剂 |
| accept_sino_foreign | TINYINT | 是否接受中外合作 |
| accept_far_city | TINYINT | 是否接受远距离城市 |
| employment_first | TINYINT | 是否就业优先 |
| postgraduate_first | TINYINT | 是否读研优先 |


---

## 四、数据清洗与数据库设计

### 4.1 数据清洗总框架

所有原始数据进入数据库前，统一经过以下六级清洗流水线：

1. **格式统一**：统一编码（UTF-8）、统一日期格式（YYYY-MM-DD）、统一数值精度。
2. **缺失值处理**：关键字段（score, rank, school_code, year）不允许缺失，缺失则丢弃或标记人工补全；次要字段缺失时对照同类数据均值/中位数填充并标记。
3. **异常值检测**：使用 IQR 方法或业务规则（如分数必须在 0-750 之间）检测。
4. **一致性校验**：跨表关联校验（如投档线表中院校-省份-年份的最低分应与专业录取表中同一院校-省份-年份各专业最低分的最小值一致或相近）。
5. **去重**：按业务主键组合去重。
6. **标准化**：院校名称、专业名称、省份名称等文本字段标准化。

### 4.2 口径统一规则

**院校代码统一**：以教育部每年公布的《全国普通高等学校名单》中的院校标识码为基准，建立院校代码-名称-曾用名映射表。

**专业代码统一**：以教育部《普通高等学校本科专业目录（2024年版）》中的专业代码为准，将各高校招生时的专业名称通过模糊匹配和人工审核映射到标准专业代码。

**科类统一**：传统高考省份"理科"映射为"物理类"、"文科"映射为"历史类"；新高考省份保持"物理类""历史类"或具体选科组合。

**年份统一**：年份字段统一为 SMALLINT，表示高考年份。

### 4.3 数据库表设计

#### 表 1：segment_table（一分一段表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | 自增主键 |
| province | VARCHAR(20) | NOT NULL | 省份 |
| year | SMALLINT | NOT NULL | 年份 |
| subject_type | VARCHAR(20) | NOT NULL | 科类 |
| score | SMALLINT | NOT NULL | 分数 |
| segment_count | INT | NOT NULL | 本段人数 |
| cumulative_count | INT | NOT NULL | 累计人数 |
| rank | INT | NOT NULL | 位次 |
| percentile | DECIMAL(10,6) | | 分位点 |
| batch_line | SMALLINT | | 批次线 |
| total_exam_count | INT | | 该科类考生总数 |
| source_url | VARCHAR(500) | | 来源 |
| crawl_time | DATETIME | NOT NULL | 采集时间 |
| review_status | TINYINT | DEFAULT 0 | 0=未审核,1=已通过,2=异常标记 |

- **主键**：id
- **唯一约束**：(province, year, subject_type, score)
- **数据来源**：省教育考试院
- **更新频率**：每年高考出分后更新
- **用途**：分数-位次等效换算

#### 表 2：school_info（院校基本信息表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| school_code | VARCHAR(20) | PK | 院校标准代码 |
| school_name | VARCHAR(100) | NOT NULL | 院校标准名称 |
| alias_list | VARCHAR(500) | | 曾用名/简称（JSON） |
| province | VARCHAR(20) | | 所在省份 |
| city | VARCHAR(50) | | 所在城市 |
| school_level | VARCHAR(20) | | "985"/"211"/"双一流"/"普通本科"/"专科" |
| is_public | TINYINT | | 是否公办 |
| school_type | VARCHAR(20) | | "综合"/"理工"/"师范"/"医药"/等 |
| moe_code | VARCHAR(20) | | 教育部标识码 |
| update_time | DATETIME | | 更新时间 |

#### 表 3：major_info（专业基本信息表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| major_code | VARCHAR(20) | PK | 专业标准代码 |
| major_name | VARCHAR(100) | NOT NULL | 专业标准名称 |
| major_category | VARCHAR(50) | | 专业大类 |
| discipline | VARCHAR(50) | | 学科门类 |
| degree | VARCHAR(20) | | 授予学位 |
| is_new_major | TINYINT | | 是否近年新增专业 |
| update_time | DATETIME | | 更新时间 |


#### 表 4：school_admission_line（历年院校投档线表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | |
| province | VARCHAR(20) | NOT NULL | 生源省份 |
| year | SMALLINT | NOT NULL | 年份 |
| batch | VARCHAR(20) | NOT NULL | 批次 |
| subject_type | VARCHAR(20) | NOT NULL | 科类 |
| school_code | VARCHAR(20) | NOT NULL, FK | 院校代码 |
| major_group_code | VARCHAR(30) | | 专业组代码 |
| min_admission_score | SMALLINT | NOT NULL | 最低投档分 |
| min_admission_rank | INT | | 最低投档位次 |
| plan_count | INT | | 计划数 |
| admission_count | INT | | 实际录取人数 |
| source_url | VARCHAR(500) | | 来源 |
| crawl_time | DATETIME | | 采集时间 |

- **唯一约束**：(province, year, batch, subject_type, school_code, major_group_code)

#### 表 5：major_admission（历年专业录取表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | |
| school_code | VARCHAR(20) | FK | |
| major_code | VARCHAR(20) | FK | |
| major_group_code | VARCHAR(30) | | |
| province | VARCHAR(20) | NOT NULL | |
| year | SMALLINT | NOT NULL | |
| min_admission_score | SMALLINT | | |
| min_admission_rank | INT | | |
| avg_score | SMALLINT | | |
| max_score | SMALLINT | | |
| plan_count | INT | | |
| admission_count | INT | | |
| subject_requirement | VARCHAR(200) | | |
| adjustment_rule | VARCHAR(200) | | |
| source_url | VARCHAR(500) | | |
| crawl_time | DATETIME | | |

- **唯一约束**：(school_code, major_code, major_group_code, province, year)

#### 表 6：admission_plan（招生计划表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | |
| school_code | VARCHAR(20) | FK | |
| major_group_code | VARCHAR(30) | | |
| major_code | VARCHAR(20) | FK | |
| province | VARCHAR(20) | NOT NULL | |
| year | SMALLINT | NOT NULL | |
| plan_count | INT | | |
| tuition | DECIMAL(10,2) | | 学费 |
| duration | TINYINT | | 学制 |
| subject_requirement | VARCHAR(200) | | 选科要求 |
| is_sino_foreign | TINYINT | | |
| is_normal_major | TINYINT | | |
| is_medical_major | TINYINT | | |
| single_subject_limit | VARCHAR(200) | | 单科限制 |
| physical_limit | VARCHAR(200) | | 身体限制 |
| remark | VARCHAR(500) | | |
| source_url | VARCHAR(500) | | |

#### 表 7：major_employment（专业就业数据表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| major_code | VARCHAR(20) | PK（与data_year联合） | |
| major_name | VARCHAR(100) | | |
| employment_rate | DECIMAL(5,4) | | |
| postgraduate_rate | DECIMAL(5,4) | | |
| civil_service_post_count | INT | | |
| average_salary | DECIMAL(10,2) | | |
| median_salary | DECIMAL(10,2) | | |
| job_count | INT | | |
| job_growth_rate | DECIMAL(6,4) | | |
| industry_distribution | TEXT | | JSON |
| main_employment_city | TEXT | | JSON |
| industry_growth_score | DECIMAL(4,2) | | |
| stability_score | DECIMAL(4,2) | | |
| sentiment_warning_score | DECIMAL(4,2) | | |
| data_year | SMALLINT | NOT NULL | 数据年份 |
| source_url | VARCHAR(500) | | |
| crawl_time | DATETIME | | |

#### 表 8：city_data（城市产业数据表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| city | VARCHAR(50) | PK（与data_year联合） | |
| gdp | DECIMAL(15,2) | | |
| gdp_per_capita | DECIMAL(10,2) | | |
| tertiary_industry_ratio | DECIMAL(5,4) | | |
| key_industries | TEXT | | JSON |
| high_tech_company_count | INT | | |
| listed_company_count | INT | | |
| related_job_count | INT | | |
| average_salary | DECIMAL(10,2) | | |
| living_cost | DECIMAL(10,2) | | |
| distance_from_home | INT | | |
| transport_score | DECIMAL(3,1) | | |
| data_year | SMALLINT | | |
| source_url | VARCHAR(500) | | |

#### 表 9：candidate_profile（考生画像表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| candidate_id | VARCHAR(32) | PK | UUID |
| province | VARCHAR(20) | NOT NULL | |
| subject_type | VARCHAR(20) | NOT NULL | |
| score | SMALLINT | NOT NULL | |
| rank | INT | NOT NULL | |
| interest_direction | VARCHAR(500) | | |
| strong_subjects | VARCHAR(200) | | |
| excluded_majors | VARCHAR(1000) | | |
| preferred_cities | VARCHAR(1000) | | |
| family_budget | DECIMAL(10,2) | | |
| risk_preference | VARCHAR(10) | | |
| accept_adjustment | TINYINT | | |
| accept_sino_foreign | TINYINT | | |
| accept_far_city | TINYINT | | |
| employment_first | TINYINT | | |
| postgraduate_first | TINYINT | | |
| create_time | DATETIME | NOT NULL | |

#### 表 10：model_output（模型输出结果表）

| 字段名 | 类型 | 约束 | 说明 |
|-------|------|------|------|
| id | BIGINT | PK, AUTO_INCREMENT | |
| candidate_id | VARCHAR(32) | FK | |
| model_version | VARCHAR(20) | NOT NULL | 模型版本号 |
| province | VARCHAR(20) | | |
| school_code | VARCHAR(20) | | |
| major_group_code | VARCHAR(30) | | |
| major_code | VARCHAR(20) | | |
| admit_probability | DECIMAL(6,4) | | 录取概率 |
| prob_lower | DECIMAL(6,4) | | 概率下限 |
| prob_upper | DECIMAL(6,4) | | 概率上限 |
| recommendation_tier | VARCHAR(10) | | "rush"/"stable"/"safe"/"bottom" |
| fit_score | DECIMAL(4,2) | | 专业匹配度 |
| career_score | DECIMAL(4,2) | | 就业价值评分 |
| city_score | DECIMAL(4,2) | | 城市价值评分 |
| family_score | DECIMAL(4,2) | | 家庭匹配评分 |
| risk_level | VARCHAR(10) | | "low"/"medium"/"high"/"critical" |
| risk_reason | TEXT | | 风险原因 |
| overall_utility | DECIMAL(8,4) | | 综合效用值 |
| explanation | TEXT | | 家长端解释文本 |
| modification_suggestion | TEXT | | 调整建议 |
| review_required | TINYINT | | 是否需要人工复核 |
| create_time | DATETIME | NOT NULL | |


---

## 五、模型假设

为建立数学模型，本文做出以下基本假设：

**假设 1（历史可参考性假设）**：在无重大政策变动的条件下，历史年份的录取数据可作为当前年份录取概率估计的参考依据。该假设是全部概率预测模型的基础。

**假设 2（位次稳定性假设）**：同一高校同一专业在相近年份的录取位次具有较强的统计稳定性，其波动服从某种概率分布（如正态分布或 t 分布）。该假设保证了历史数据的使用合理性。

**假设 3（考生行为理性假设）**：考生填报志愿时主要依据分数位次、院校声誉、专业热度等可观测因素，不存在大规模的非理性填报行为。该假设确保模型统计规律的有效性。

**假设 4（数据独立性假设）**：各个志愿的录取事件之间在给定考生位次条件下相互独立。该假设简化了联合概率的计算。

**假设 5（招生章程合规假设）**：招生章程中公布的选科要求、身体条件限制、单科成绩限制等是严格的约束条件，不符合约束的考生在该专业/专业组中没有录取资格。

**假设 6（就业数据时效性假设）**：近三年的专业就业数据可以较好地反映未来 4-6 年（即考生在本科或本硕阶段毕业后）的就业市场状况，但需承认该假设存在较大不确定性。

**假设 7（考生偏好稳定性假设）**：考生在问卷中填写的偏好信息（兴趣方向、排斥专业、偏好城市等）真实反映了其意愿，且在志愿填报期间不发生重大变化。

**假设 8（城市发展连续性假设）**：城市的经济与产业数据在短期内具有连续性，当前的 GDP、产业结构等指标可以反映未来数年的城市吸引力。

---

## 六、符号说明

| 符号 | 含义 | 单位/取值范围 | 首次出现位置 |
|------|------|-------------|------------|
| S | 考生当年原始分数 | 0-750 | 核心模型一 |
| S_t | 第 t 年的分数 | 0-750 | 核心模型一 |
| R | 考生当年位次 | 正整数 | 核心模型一 |
| R_t | 第 t 年的位次 | 正整数 | 核心模型一 |
| P_ct | 分位点（percentile） | [0, 1] | 核心模型一 |
| S_eq^{(t)} | 第 t 年的等效分 | 0-750 | 核心模型一 |
| delta_B | 批次线差（分数-批次线） | 实数 | 核心模型一 |
| w_t | 第 t 年的权重 | [0, 1] | 核心模型一 |
| sigma_t | 第 t 年的波动异常度 | >=0 | 核心模型一 |
| E_min^{(k)}(t) | 第 t 年第 k 个院校的最低录取位次 | 正整数 | 核心模型二 |
| R_cand | 考生位次 | 正整数 | 核心模型二 |
| gap_k | 第 k 个志愿的位次差 | 整数 | 核心模型二 |
| P_admit^{(k)} | 第 k 个志愿的录取概率 | [0, 1] | 核心模型二 |
| plan_t^{(k)} | 第 k 个志愿第 t 年的招生计划数 | 正整数 | 核心模型二 |
| lambda_pop | 院校/专业热度因子 | >=1 | 核心模型二 |
| phi_unc | 小样本不确定性修正因子 | [0, 1] | 核心模型二 |
| E_career | 就业价值评分 | [0, 1] | 核心模型三 |
| w_j^AHP | AHP 权重 | [0, 1] | 核心模型三 |
| w_j^entropy | 熵权法权重 | [0, 1] | 核心模型三 |
| w_j^comb | 组合权重 | [0, 1] | 核心模型三 |
| C_i | TOPSIS 中第 i 个专业的综合得分 | [0, 1] | 核心模型三 |
| x_i | 决策变量，是否选入第 i 个志愿 | {0, 1} | 核心模型四 |
| alpha, beta, gamma, delta, eta, lambda | 效用函数的权重系数 | [0, 1] | 核心模型四 |
| N_rush, N_stable, N_safe, N_bottom | 各层次志愿数量 | 正整数 | 核心模型四 |
| N_max | 志愿表总容量 | 正整数 | 核心模型四 |
| Risk | 综合风险值 | [0, 1] | 核心模型五 |
| R_slip | 滑档风险 | [0, 1] | 核心模型五 |
| R_withdrawal | 退档风险 | [0, 1] | 核心模型五 |
| R_adjust | 调剂风险 | [0, 1] | 核心模型五 |
| R_cold | 专业冷门风险 | [0, 1] | 核心模型五 |
| R_employment | 就业风险 | [0, 1] | 核心模型五 |
| R_region | 地域风险 | [0, 1] | 核心模型五 |


---

## 七、分数—位次等效换算模型（核心模型一）

### 7.1 业务问题说明

高考成绩公布后，考生获得的是当年的分数 S 和在全省的位次 R。但在参考往年录取数据时，用今年 580 分直接去对标去年的最低录取分 580 分是不合理的，因为两年之间的试卷难度、考生人数、招生计划均不同。必须将考生当年的分数和位次映射为历史年份中的**等效分**，才能进行有意义的比较。

本模型的核心业务问题是：**给定考生当年分数 S 和位次 R，求出该考生在目标历史年份 t 中的等效分数 S_eq^{(t)} 和等效位次区间。**

### 7.2 所需爬取数据

第一类数据：一分一段表（segment_table），包含近 5 年各科类各分数段的累计人数和位次数据。批次线数据（各省教育考试院发布）。

### 7.3 数据来源与字段设计

见 3.2.1 节。

### 7.4 数据清洗与口径统一

1. 从 segment_table 中读取近 N 年（默认 N=5）的一分一段数据。
2. 检查同一 (province, year, subject_type) 下累计人数是否单调不减。
3. 若某年份数据存在大段缺失，标记该年份异常并降权。
4. 统一批次线口径：使用该科类本科批次线作为线差基准。

### 7.5 符号说明

| 符号 | 含义 |
|------|------|
| S | 考生当年分数 |
| R | 考生当年位次 |
| N_total | 当年该科类考生总数 |
| P_ct(S) | 分位点，P_ct(S) = CumCount(S) / N_total |
| T | 历史参考年份集合，默认 T = {t1, t2, ..., t5}（近5年） |
| S_eq^{(t)} | 第 t 年的等效分 |
| B^{(t)} | 第 t 年的批次线 |
| delta_B^{(t)} | 第 t 年的线差，S_eq^{(t)} - B^{(t)} |
| w_t | 第 t 年数据权重 |
| sigma_t | 第 t 年波动异常度 |
| CI_95%(t) | 第 t 年等效分的 95%置信区间 |

### 7.6 模型假设

1. **分位点守恒假设**：同一考生群体在排除了不同年份考生规模差异后，其在考生群体中的相对位置（分位点）具有跨年度可比性。
2. **相邻分数段均匀分布假设**：在缺少精确数据的两个已知分位点之间，考生分布近似为连续均匀分布，可以使用线性插值。
3. **近 5 年参考窗口假设**：5 年内的高考制度和高校招生格局没有发生足以推翻历史数据参考价值的重大变化。
4. **批次线稳定性假设**：批次线在多年度之间可比较，线差修正可以有效消除不同年份试题难度的整体漂移。

### 7.7 模型建立

#### 步骤 1：确定考生当前分位点

根据当年一分一段表：

```
P_ct(S) = CumulativeCount(S) / N_total
```

若考生分数 S 恰好对应一行一分一段数据，则直接取该行的 cumulative_count / total。若 S 未精确匹配某个分数段（例如省级一分一段表不以 1 分为间隔），则通过相邻已知分位点的线性插值计算。

#### 步骤 2：分位数映射

在目标年份 t 的一分一段表中，寻找与 P_ct(S) 最接近的累计人数对应的分数：

```
S_eq^{(t)} = argmin_s | CumCount_t(s) / N_total,t - P_ct(S) |
```

#### 步骤 3：线性插值

由于一分一段表是离散的，需要在两个相邻分数段之间做线性插值：

设当前分位点 p 落在分数 s1 和 s2（s1 < s2）对应的分位点 p1 和 p2（p1 <= p <= p2）之间，则等效分：

```
S_eq^{(t)} = s1 + (p - p1) / (p2 - p1) * (s2 - s1)
```

#### 步骤 4：线差修正

计算考生的批次线差：delta_B^{(t)} = S_eq^{(t)} - B^{(t)}

对于目标年份，如果该年份的批次线 B^{(t)} 与当前年份 B^{(0)} 差距较大（例如当批次线变动超过 30 分），说明命题难度或招生规模发生了整体偏移。此时采用"分位点为主、线差为辅"的策略进行修正：

```
S_eq_adjusted^{(t)} = S_eq^{(t)} + theta * [(B^{(0)} - B^{(t)}) * (1 - P_ct(S))]
```

其中 theta 为修正系数（建议 theta = 0.3），(1 - P_ct(S)) 起调节作用：高分考生受批次线影响较小，低分考生受影响较大。

#### 步骤 5：异常年份降权

定义第 t 年的波动异常度：

```
sigma_t = 1/(|T|-1) * sum_{t'!=t} | delta_R^{(t)} - delta_R^{(t')} |
```

其中 delta_R^{(t)} 为第 t 年某个参考点（如前 20% 分位点）的位次值。

权重：

```
w_t = exp(-sigma_t / tau) / sum_{t'} exp(-sigma_{t'} / tau)
```

其中 tau 为温度参数（建议 tau = 1000）。

同时设置绝对阈值：若 sigma_t > 3 * median(sigma)，标记该年为**异常年份**，系统输出警告且不将其作为主要参考依据。

#### 步骤 6：多年度加权等效分

```
S_eq^* = sum_{t in T} w_t * S_eq^{(t)}
```

#### 步骤 7：置信区间估计

等效分在历史年份上的波动可用标准差来衡量：

```
sigma_S = sqrt( sum_{t in T} w_t * (S_eq^{(t)} - S_eq^*)^2 / sum_{t in T} w_t )
```

95% 置信区间：

```
S_eq^* ± 1.96 * sigma_S
```

#### 步骤 8：Fallback 规则

当某目标年份的数据严重不足时：
- 回退到最近可用年份的等效分；
- 标注 confidence_level = "low"；
- 增加 abnormal_year_warning = True。

### 7.8 模型求解

**算法 1：分数-位次等效换算算法（伪代码）**

```
输入: candidate_score S, candidate_rank R, current_year y0,
      目标年份列表 T, 当前批次线 B0
输出: 等效分区间、置信度

1. 计算分位点 Pct = R / N_total
2. 初始化结果列表 results = []
3. FOR t IN T:
   a. 读取第 t 年的一分一段表
   b. 在表中找到包含 Pct 的分数区间 [s1, s2]
   c. 线性插值: Seq = s1 + (Pct - p1) / (p2 - p1) * (s2 - s1)
   d. 读取 Bt = 第 t 年批次线
   e. 线差修正: Seq_adj = Seq + theta * ((B0 - Bt) * (1 - Pct))
   f. 结果存入 results
4. 计算各年波动异常度 sigma_t
5. 计算各年权重 w_t = softmax(-sigma_t / tau)
6. 识别异常年份（sigma_t > 3 * median(sigma)），标记警告
7. 计算加权等效分 Seq_star = sum(w_t * Seq_t)
8. 计算标准差 sigma_S
9. 计算置信区间: [Seq_star - 1.96*sigma_S, Seq_star + 1.96*sigma_S]
10. 判断 confidence_level:
    - sigma_S <= 3: "high"
    - 3 < sigma_S <= 8: "medium"
    - sigma_S > 8: "low"
11. 返回结果
```

### 7.9 输出结果（模拟数据示例）

| 字段 | 示例值 |
|------|--------|
| candidate_id | "cand_2024_hebei_001" |
| current_score | 580 |
| current_rank | 15823 |
| current_percentile | 0.0468 |
| target_year | 2023 |
| equivalent_rank | 15420 |
| equivalent_score | 576 |
| equivalent_score_interval | [570, 582] |
| confidence_level | "high" |
| abnormal_year_warning | False |

### 7.10 评价指标

| 指标 | 计算方式 | 目标值 |
|------|---------|--------|
| 等效分误差 | 历史留一验证 MAE | < 3 分 |
| 置信区间覆盖率 | 等效分落在 95% CI 内的比例 | > 90% |
| 异常年份识别率 | 人工标记异常年份被正确识别的比例 | > 80% |

### 7.11 业务解释

**家长端**：
"您今年考了 580 分，在全省理科生中排第 15823 名。相当于您在去年考了大约 576 分（区间为 570-582 分）。这个等效分是用来和往年各学校的录取分数做比较的基础。"

**咨询师端**：
"等效分换算为概率预测提供了统一的比较基准。本次换算置信度为高，各年等效分波动在 6 分以内，说明参考价值较强。需要注意的是，2021 年数据波动较大已被标记为异常年份，在后续概率预测中该年份的权重较低。"

**系统后台端**：
等效分数据存入 model_output 表；confidence_level 字段用于决定后续模型是否触发人工复核；abnormal_year_warning 为 True 时，在概率预测模型中对该年份自动降权。

### 7.12 局限性与改进方向

1. **分位点守恒假设的局限**：当高校扩招或缩招幅度较大，或考生人数发生突变时，同一分位点对应的录取竞争力可能发生变化。改进方向：引入招生计划变化率修正因子。
2. **线性插值精度**：如果一分一段表非 1 分间隔（如 5 分一段），等效分的精度会降低。改进方向：引入 Beta 分布拟合分数分布实现更精细的插值。
3. **批次线修正的局限**：批次线的划定本身受政策影响，不一定完全反映命题难度变化。改进方向：引入更多校标点（如 985 高校中位投档分）进行多锚点修正。


---

## 八、院校/专业录取概率预测模型（核心模型二）

### 8.1 业务问题说明

给定考生的等效位次 R_cand 和等效分 S_eq，以及目标院校 k（或专业 j）在近几年的历史录取数据，估计考生被该院校/专业录取的概率 P_admit。

核心业务问题：**"我报这个学校/这个专业，被录取的概率有多大？"** 并且 **"这个概率靠谱吗？"**（需要给出置信区间）

### 8.2 所需爬取数据

第二类数据：历年院校投档线（school_admission_line 表）；第三类数据：历年专业录取数据（major_admission 表）；第四类数据：招生计划数据（admission_plan 表）；第一类数据：一分一段表。

### 8.3 数据来源与字段设计

见 3.2.2、3.2.3、3.2.4 节。

### 8.4 数据清洗与口径统一

1. 从 school_admission_line 和 major_admission 表中提取近 N 年（默认 N=5）数据。
2. 确保 min_admission_score 与 min_admission_rank 的对应关系与当年一分一段表一致。
3. 招生计划为 0 的记录排除在训练样本之外。
4. 新开设专业（仅有 1-2 年数据）标记为"小样本高不确定性"。

### 8.5 符号说明

| 符号 | 含义 |
|------|------|
| x_k | 第 k 个候选志愿的特征向量 |
| y_k | 二值标签：1=录取成功, 0=录取失败 |
| f(x) | 录取概率预测函数 |
| mu_k | 预测概率值 |
| sigma_k | 概率不确定性 |
| R_cand | 考生位次 |
| E_min^{(k)}(t) | 志愿 k 在第 t 年的最低录取位次 |
| gap_k(t) | 位次差，R_cand - E_min^{(k)}(t)，正值表示考生位次更优 |
| delta_plan_k(t) | 招生计划同比变化率 |
| lambda_pop^{(k)} | 热度因子 |
| v_k(t) | 录取位次波动率 |
| tau_k | 小样本修正因子 |

### 8.6 模型假设

1. 在给定考生位次和院校特征的情况下，录取概率服从某种可学习的统计规律。
2. 投档线和专业录取线对应的位次是历史考生真实选择行为的结果，可以作为模型学习的标签。
3. 近 5 年的数据对当前预测具有递减的参考价值（越近年份越重要）。
4. 同一个专业组内各专业的调剂风险主要取决于该专业组内是否有明显冷门专业。

### 8.7 模型建立

本文采用**多模型融合**策略建立录取概率预测模型，融合五种方法以互补短长。

#### 8.7.1 特征工程

对每个候选志愿构建如下特征向量：

| 特征 | 含义 | 构造方式 |
|------|------|---------|
| x1: rank_gap_mean | 考生位次与历年最低录取位次的平均差值 | mean(gap_k(t)) |
| x2: rank_gap_std | gap 的标准差 | std(gap_k(t)) |
| x3: rank_gap_min | gap 的最小值（最危险年份） | min(gap_k(t)) |
| x4: rank_gap_trend | gap 的趋势（正=变难,负=变易） | gap 的线性回归斜率 |
| x5: score_diff_mean | 等效分与历年最低录取分的平均差值 | 与 x1 类似但用分数 |
| x6: plan_count_mean | 近3年平均招生计划数 | mean(plan_t) |
| x7: plan_change_rate | 招生计划同比变化率 | (plan_current - plan_last) / plan_last |
| x8: rank_volatility | 录取位次波动系数 | std(rank_t) / mean(rank_t) |
| x9: school_popularity | 院校热度标度 | 基于搜索引擎指数或历年报考热度排名 |
| x10: major_popularity | 专业热度标度 | 基于专业的年度报考热度变化 |
| x11: years_with_data | 有录取数据的年份数 | 反映数据充分性 |
| x12: is_new_major | 是否为新增专业 | 0/1 |
| x13: subject_match | 选科是否完全匹配 | 0/1 |
| x14: percentile_rank | 考生分位点 | 见核心模型一 |

#### 8.7.2 方法一：Logistic 回归基准模型

以历史数据构造训练集：对于每个 (candidate_simulated, school, year)，当 candidate_rank < min_admission_rank 时标签为 1（可录取），否则为 0。

```
log(P_admit / (1 - P_admit)) = beta0 + sum_{i=1}^{m} beta_i * x_i
P_admit = 1 / (1 + exp(-(beta0 + sum beta_i * x_i)))
```

Logistic 回归提供可解释的基准概率和特征重要性。

#### 8.7.3 方法二：XGBoost 梯度提升模型

使用 XGBoost 作为非线性模型补充。XGBoost 具有以下优势：自动捕获特征间的非线性交互；对缺失值有良好处理；天然支持特征重要性排序。

建议参数：n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, objective='binary:logistic', eval_metric='auc'。

#### 8.7.4 方法三：贝叶斯修正

对于数据极度稀疏的小样本专业（数据年数 <= 2），直接使用频率或统计模型估计的概率误差较大。引入贝叶斯先验：

假设录取事件服从 Beta-Binomial 模型：

```
P_admit^Bayes = (alpha0 + successes_sim) / (alpha0 + beta0 + trials_sim)
```

其中 (alpha0, beta0) 为先验参数，从同层次同类型专业的总体统计中估计（例如用矩估计法）。

先验估算：对同一院校层次（如 211 院校）内的所有专业，计算它们的平均录取概率作为先验均值，并据此反推 (alpha0, beta0)。

#### 8.7.5 方法四：蒙特卡洛模拟

对每个候选志愿进行 M 次（默认 M=10000）模拟：

1. 从录取位次的历史分布中抽样（假设服从正态分布或 t 分布，小样本时用 Bootstrap 抽样）。
2. 每次抽样得到一个模拟的最低录取位次 E_min^sim。
3. 计算 P_admit^MC = (1/M) * sum_m 1[R_cand <= E_min^{sim,m}]。
4. 同时得到概率的置信区间。

对于招生计划发生变化的场景，需要将计划变化率纳入模拟：

```
E_min^{sim,m}(t) = E_min^hist(t) * (1 + eta * delta_plan_k(t) + epsilon_m)
```

其中 epsilon_m ~ N(0, sigma_vol^2) 捕捉随机波动。

#### 8.7.6 方法五：小样本不确定性修正

定义数据充分性因子：

```
tau_k = 1 / (1 + exp(-(n_k - 3)))
```

其中 n_k 为有历史录取数据的年数。

对于 n_k <= 2 的专业：概率估计以贝叶斯修正结果为准；不确定性区间扩大为原来的 (1 + (3 - n_k)) 倍；触发 review_required = True。

#### 8.7.7 模型融合

最终概率采用加权融合：

```
P_admit^{(k)} = gamma1 * P_LR + gamma2 * P_XGB + gamma3 * P_Bayes + gamma4 * P_MC
```

推荐默认权重：gamma = [0.15, 0.35, 0.20, 0.30]，在实际部署时可通过交叉验证动态调整。


### 8.8 模型求解

**算法 2：录取概率预测算法（伪代码）**

```
输入: candidate_features, school_code, major_code (optional),
      province, current_year, historical_data
输出: admit_probability, probability_interval, recommendation_tier, top_features

1. 特征构造:
   - 读取近5年该院校/专业的历史录取位次
   - 计算 rank_gap_mean, rank_gap_std, rank_gap_min, rank_gap_trend
   - 读取招生计划变化率
   - 计算录取位次波动系数
   - 构造完整特征向量 x

2. Logistic 回归预测: P_LR = sigmoid(beta^T * x)
3. XGBoost 预测: P_XGB = xgb.predict_proba(x)
4. 贝叶斯修正:
   - 若 n_years <= 2: 计算先验参数 (alpha0, beta0)
   - P_Bayes = (alpha0 + simulated_successes) / (alpha0 + beta0 + simulated_trials)
5. 蒙特卡洛模拟:
   - 重复 M 次:
     a. 从历史位次分布中抽样得到 simulated_rank
     b. 应用计划变化修正
     c. 判断: I = (candidate_rank <= simulated_rank)
   - P_MC = mean(I), CI = [percentile(I, 2.5), percentile(I, 97.5)]
6. 小样本修正:
   - tau = 1 / (1 + exp(-(n_years - 3)))
   - 调整概率区间宽度
7. 模型融合: P_final = 0.15*P_LR + 0.35*P_XGB + 0.20*P_Bayes + 0.30*P_MC
8. 确定 recommendation_tier:
   - P >= 0.85 -> "bottom" (垫)
   - 0.60 <= P < 0.85 -> "safe" (保)
   - 0.30 <= P < 0.60 -> "stable" (稳)
   - P < 0.30 -> "rush" (冲)
9. 识别 top_features (对概率影响最大的3个特征)
10. 返回结果
```

### 8.9 输出结果

| 字段 | 示例值 |
|------|--------|
| school_code | "10001" |
| major_code | "080901" |
| admit_probability | 0.68 |
| probability_interval | [0.58, 0.78] |
| recommendation_tier | "stable" |
| top_features | ["rank_gap_mean", "rank_volatility", "plan_change_rate"] |
| uncertainty_level | "medium" |
| review_required | False |

### 8.10 评价指标

| 指标 | 目标值 |
|------|--------|
| AUC（ROC 曲线下面积） | > 0.85 |
| Brier Score（校准度） | < 0.15 |
| 概率校准曲线斜率和截距 | 斜率接近 1，截距接近 0 |
| 留一省份交叉验证 AUC | > 0.80 |
| 模型稳定性（各年 AUC 标准差） | < 0.03 |
| 小样本专业概率区间覆盖率 | > 80% |

### 8.11 业务解释

**家长端**：
"根据近 5 年数据综合分析，您报考 A 大学计算机科学与技术专业的预估录取概率约为 68%（区间 58%-78%），属于'稳'的范畴。该校该专业历年录取位次在 12000-15000 名之间，您当前的位次 14000 名处于中等偏上位置，有较大希望被录取。"

**咨询师端**：
"候选志愿 A 的位次差均值为 +1200 名，录取概率 68%，建议列为'稳'类。该专业近 3 年招生计划稳定，录取位次波动系数 0.08（较低），数据充分（5 年数据），预测可信度中高。需要注意该专业为热门专业，近两年报考热度上升，建议同时填报 1-2 个'保'类志愿。"

**系统后台端**：
概率 < 0.15 或 > 0.95 时检查是否存在极端特征；uncertainty_level = "high" 时触发 review_required；小样本专业自动触发贝叶斯修正和人工复核；特征重要性排序可用于后续模型的解释性报告。

### 8.12 局限性与改进方向

1. **数据充分性问题**：新开设专业仅 1-2 年数据，贝叶斯先验的选择对结果影响较大。改进方向：利用相似专业（如计算机科学与技术->数据科学）的数据构造更合理的层次先验。
2. **"只看最低分"的局限**：仅使用最低录取位次作为标签丢失了录取分布在最低分以上的信息。改进方向：若能获得各专业录取的平均分或中位分，可构造更精细的序数回归模型。
3. **专业热度的动态性**：专业热度随就业市场、社会舆论变化较快，模型难以捕捉未来的热度变化。改进方向：引入时间序列预测模型（如 Prophet）对专业热度进行趋势外推。
4. **录取概率阈值设定的主观性**：冲/稳/保/垫的概率阈值是基于专家经验设定的，不同年份、不同省份的最佳阈值可能不同。改进方向：通过历史回测，动态寻优各概率阈值。

---

## 九、专业就业景气度评价模型（核心模型三）

### 9.1 业务问题说明

"这个专业好不好就业？""四年/七年后这个专业还吃香吗？""这个专业考公务员的机会多不多？"——这些都是考生和家长在填报志愿时的核心关切。

本模型的核心业务问题是：**给定一个本科专业，从就业率、薪资水平、行业成长性、稳定性、考公适配度、冷门风险等多维度综合评价其就业景气度，并输出易于理解的红/黄/绿标签。**

### 9.2 所需爬取数据

第五类数据：专业就业数据（major_employment 表）；第六类数据：城市产业数据（city_data 表）；公务员招录职位表（Excel/CSV）。

### 9.3 数据来源与字段设计

见 3.2.5、3.2.6 节。

### 9.4 数据清洗与口径统一

1. 就业率数据统一使用"毕业去向落实率"口径。
2. 薪资数据按来源分为三个层次：第一层（高可信）高校官方就业报告；第二层（中可信）国家统计局分行业工资；第三层（低可信）招聘网站公开报告。
3. 专业名称标准化：与 major_info 表关联。
4. 冷门专业薪资数据缺失时，使用同一专业大类下其他专业的平均薪资并标记。
5. 社媒舆情数据仅从公开数据平台获取，评分越低越好（负面舆情多则 score 高）。

### 9.5 符号说明

| 符号 | 含义 |
|------|------|
| z_ij | 第 i 个专业在第 j 个指标上的取值（已归一化） |
| w_j | 第 j 个指标的权重 |
| C_i | 第 i 个专业的综合就业景气度评分 |
| A_AHP | AHP 判断矩阵 |
| CI | 一致性指标 |
| CR | 一致性比率 |
| e_j | 第 j 个指标的信息熵 |
| d_j | 第 j 个指标的差异系数 |
| D_i^+ | 第 i 个专业到正理想解的距离 |
| D_i^- | 第 i 个专业到负理想解的距离 |
| L_i | 第 i 个专业的聚类类别标签 |
| trend_ij | 指标的时间趋势斜率 |

### 9.6 模型假设

1. 就业质量报告的数据具有较高可信度，但承认部分高校可能对就业数据进行美化。
2. 薪资数据的一、二、三层来源可以互补，但需要分层加权使用。
3. 社媒舆情可以有效反映公众对该专业的信心和担忧，但只能作为辅助预警信号。
4. 当前（近 3 年）的就业数据结构在未来 4-6 年内将大致保持，但行业变革可能打破此假设。


### 9.7 模型建立

本模型采用**AHP-熵权-TOPSIS 组合评价方法**，辅以聚类分析和趋势分析。

#### 9.7.1 评价指标体系

一级指标及下属二级指标：

| 一级指标 | 二级指标 | 编号 | 方向 |
|---------|---------|------|------|
| 就业实现 | 就业率 | z1 | 正向 |
| | 升学率 | z2 | 正向 |
| 薪资水平 | 平均月薪 | z3 | 正向 |
| | 中位数月薪 | z4 | 正向 |
| 岗位供给 | 岗位数量 | z5 | 正向 |
| | 岗位增长率 | z6 | 正向 |
| 行业前景 | 行业成长性评分 | z7 | 正向 |
| 稳定性 | 稳定性评分 | z8 | 正向 |
| 考公适配 | 可报考公务员岗位数 | z9 | 正向 |
| 风险预警 | 社媒舆情预警分 | z10 | 负向（需反向归一化） |

#### 9.7.2 步骤一：指标归一化

对正向指标（越大越好）：

```
z_ij^norm = (z_ij - min_j(z_ij)) / (max_j(z_ij) - min_j(z_ij))
```

对负向指标（社媒舆情预警分，越小越好）：

```
z_ij^norm = (max_j(z_ij) - z_ij) / (max_j(z_ij) - min_j(z_ij))
```

#### 9.7.3 步骤二：AHP 层次分析法赋权

构建一级指标判断矩阵。以就业实现、薪资水平、岗位供给、行业前景、稳定性、考公适配、风险预警为 7 个准则层。

**判断矩阵示例**（基于专家评估，使用 1-9 标度法）：

| | 就业实现 | 薪资 | 岗位供给 | 行业前景 | 稳定性 | 考公适配 | 风险预警 |
|---|---------|------|---------|---------|------|---------|---------|
| 就业实现 | 1 | 1/2 | 1 | 1/2 | 2 | 3 | 2 |
| 薪资 | 2 | 1 | 2 | 1 | 3 | 4 | 3 |
| 岗位供给 | 1 | 1/2 | 1 | 1/2 | 2 | 3 | 2 |
| 行业前景 | 2 | 1 | 2 | 1 | 3 | 4 | 3 |
| 稳定性 | 1/2 | 1/3 | 1/2 | 1/3 | 1 | 2 | 1 |
| 考公适配 | 1/3 | 1/4 | 1/3 | 1/4 | 1/2 | 1 | 1/2 |
| 风险预警 | 1/2 | 1/3 | 1/2 | 1/3 | 1 | 2 | 1 |

通过特征值法求得权重向量 w^AHP，并计算一致性比率 CR < 0.1 以确保判断矩阵一致性可接受。

#### 9.7.4 步骤三：熵权法赋权

熵权法基于数据本身的信息量确定权重，避免主观偏差。

对第 j 个指标，信息熵：

```
e_j = -k * sum_{i=1}^{n} p_ij * ln(p_ij)
```

其中 p_ij = z_ij^norm / sum(z_ij^norm)，k = 1 / ln(n)。

差异系数：d_j = 1 - e_j

熵权：w_j^entropy = d_j / sum_j d_j

#### 9.7.5 步骤四：组合权重

```
w_j^comb = (w_j^AHP * w_j^entropy) / sum_j (w_j^AHP * w_j^entropy)
```

这种乘法合成法兼顾了主观经验和客观数据两个维度。

#### 9.7.6 步骤五：TOPSIS 综合评价

构建加权归一化矩阵：

```
v_ij = w_j^comb * z_ij^norm
```

正理想解：V_j^+ = max_i(v_ij)；负理想解：V_j^- = min_i(v_ij)

到正理想解的距离：D_i^+ = sqrt(sum_j (v_ij - V_j^+)^2)

到负理想解的距离：D_i^- = sqrt(sum_j (v_ij - V_j^-)^2)

综合得分：C_i = D_i^- / (D_i^+ + D_i^-)

C_i 越接近 1，表示该专业的就业景气度越好。

#### 9.7.7 步骤六：聚类分析与标签生成

对 C_i 排名进行三分类聚类（K-Means, K=3），按聚类中心从高到低分别标记为：

- **绿色（绿灯）**：就业景气度好，风险低
- **黄色（黄灯）**：就业景气度中等，需关注变化趋势
- **红色（红灯）**：就业景气度较差，冷门风险高

**五维细分标签**：

1. **本科直接就业价值**（就业率、薪资、岗位供给，对升学率降权）
2. **读研后就业价值**（就业率、薪资 x (1 + 升学率调节因子)）
3. **考公考编适配度**（可报考公务员岗位数和稳定性评分）
4. **行业成长性**（岗位增长率和行业增长评分）
5. **专业冷门风险**（综合低频的就业率、薪资低和岗位少三个信号）

#### 9.7.8 步骤七：时间序列趋势分析

对每个指标计算近 3 年的线性趋势斜率：

```
trend_j = sum_{t=1}^{3} (t - t_bar)(y_ij^{(t)} - y_bar_ij) / sum_{t=1}^{3} (t - t_bar)^2
```

定义趋势标签：trend_j > 0.05 为"上升趋势"↑；-0.05 < trend_j < 0.05 为"稳定"→；trend_j < -0.05 为"下降趋势"↓。当关键指标（如薪资、岗位增长率）出现下降趋势时，在风险预警中标注。

### 9.8 模型求解

**算法 3：专业就业景气度评价算法（伪代码）**

```
输入: major_code_list, 近3年 major_employment 数据, AHP判断矩阵
输出: career_score, sub_scores, red_yellow_green_label, trend_info

1. 数据准备:
   - 读取近3年就业数据，对每年取各指标值
   - 计算10个二级指标的近3年均值作为评价基准
   - 计算各指标趋势斜率

2. 指标归一化:
   - 正向指标: z_norm = (z - min) / (max - min)
   - 负向指标(舆情): z_norm = (max - z) / (max - min)

3. AHP 赋权:
   - 构造判断矩阵
   - 计算特征向量 -> w_AHP
   - 验证 CR < 0.1

4. 熵权法赋权:
   - 计算各指标信息熵 e_j
   - 计算差异系数 d_j = 1 - e_j
   - 计算熵权 w_entropy_j

5. 组合权重: w_comb_j = (w_AHP_j * w_entropy_j) / sum(w_AHP * w_entropy)

6. TOPSIS 综合评价:
   - 加权归一化: v_ij = w_comb_j * z_norm_ij
   - 确定正负理想解
   - 计算各专业到理想解的距离
   - 计算综合得分 C_i

7. 聚类标签:
   - K-Means (K=3) 对 C_i 聚类
   - 按聚类中心排序: 高->绿色, 中->黄色, 低->红色

8. 五维分项评分（分别计算五种场景下的得分）

9. 趋势分析:
   - 计算各指标3年趋势斜率
   - 生成趋势标签

10. 返回结果
```


### 9.9 输出结果（模拟数据示例）

| 字段 | 示例值 |
|------|--------|
| major_code | "080901" |
| major_name | "计算机科学与技术" |
| career_score | 0.82 |
| employment_level | "高" |
| salary_score | 0.88 |
| growth_score | 0.75 |
| stability_score | 0.70 |
| postgraduate_value_score | 0.85 |
| civil_service_score | 0.45 |
| major_risk_label | "low" |
| red_yellow_green_label | "green" |
| trend_direction | "stable" |
| data_quality | "high" |

### 9.10 评价指标

| 指标 | 说明 |
|------|------|
| 各维度评分与专家评估的相关系数 | 验证模型有效性 |
| 分类一致性 | 聚类结果与专家分类的一致性 |
| 趋势方向准确性 | 模型预测的趋势方向与下一年实际趋势的吻合率 |
| AHP 一致性比率 CR | < 0.1 |

### 9.11 业务解释

**家长端**：
"计算机科学与技术专业就业景气度评价为'绿灯'（好），总评分 82 分。近三年该专业的平均就业率为 94.3%，平均月薪约 8500 元，相关岗位充足，行业处于上升期。需要注意：该专业考公务员的适配岗位较少。如果您有明确的考公计划，可以同时关注电子信息类、管理类等考公适配度更高的专业。"

**咨询师端**：
"专业就业景气度评价显示：计算机类（绿灯），career_score=0.82，薪资和能力突出，岗位供给充足，冷门风险极低。但升学竞争激烈。哲学类（黄灯），career_score=0.38，就业率和薪资较低，但考公适配度中等。建议就业优先型考生重点关注绿灯专业，保守型考生在稳保志愿中避开红灯专业。"

**系统后台端**：major_risk_label 纳入风险评估模型的 R_cold 风险计算；red_yellow_green_label 用于志愿表可视化展示；趋势下降信号触发预警标记。

### 9.12 局限性与改进方向

1. **就业质量报告的可信度问题**：部分高校可能对就业数据进行美化，需要交叉验证。改进方向：引入多源数据交叉验证机制。
2. **薪资口径不统一**：不同来源的薪资数据口径不一致。改进方向：建立标准化的薪资口径转换系数。
3. **专业名称与岗位匹配的模糊性**：招聘网站上岗位名称与本科专业名称并非一一对应。改进方向：使用 NLP 语义匹配技术（如 BERT 文本相似度）提高匹配精度。
4. **社媒舆情的局限性**：仅在特定信息茧房内传播，不代表全社会共识。本文已将社媒舆情仅作为辅助预警信号，权重较低。

---

## 十、冲稳保志愿组合优化模型（核心模型四）

### 10.1 业务问题说明

前面各模型分别解决了"这个志愿好不好""这个志愿能不能上"的问题。但在实际志愿填报中，考生最终提交的是一张包含多个志愿的**志愿表**，而非单个志愿。

本章的核心业务问题是：**在满足所有约束条件的前提下，从大量候选志愿（可能多达几百个）中选出一组（通常 30-96 个，依省份规则而定）最优志愿，构成一张结构合理、梯度清晰、风险可控的志愿表。**

"结构合理"的核心标准是"冲、稳、保、垫"的合理搭配：
- **冲（rush）**：录取概率较低（如 10%-30%），但院校/专业质量高，值得一搏
- **稳（stable）**：录取概率中等（30%-60%），有较大把握被录取
- **保（safe）**：录取概率较高（60%-85%），基本能确保被录取
- **垫（bottom）**：录取概率很高（>85%），为兜底志愿，确保有学上

### 10.2 所需数据

本模型主要使用核心模型一、二、三的输出结果以及考生画像中的约束条件。

### 10.3 符号说明

| 符号 | 含义 |
|------|------|
| C = {c1, c2, ..., cN} | 候选志愿集合 |
| x_i in {0, 1} | 决策变量，第 i 个志愿是否入选 |
| N_max | 志愿表最大容量 |
| P_admit^{(i)} | 第 i 个志愿的录取概率 |
| M_fit^{(i)} | 第 i 个志愿的专业匹配度 |
| E_career^{(i)} | 第 i 个志愿的就业价值评分 |
| C_city^{(i)} | 第 i 个志愿的城市价值评分 |
| R_family^{(i)} | 第 i 个志愿的家庭资源匹配度 |
| Risk^{(i)} | 第 i 个志愿的综合风险 |
| U_i | 第 i 个志愿的综合效用 |

### 10.4 模型假设

1. 各个志愿的录取事件在给定考生位次条件下相互独立。
2. 志愿表中的志愿按概率从低到高排列（冲->稳->保->垫的梯度）。
3. 考生对志愿的满意度是各维度效用的线性加权和。
4. 考生填写的约束条件（排斥专业、预算上限、地域限制等）是刚性的，不可违反。

### 10.5 模型建立

#### 10.5.1 综合效用函数

对每个候选志愿 i，其综合效用：

```
U_i = alpha*P_admit^{(i)} + beta*M_fit^{(i)} + gamma*E_career^{(i)}
      + delta*C_city^{(i)} + eta*R_family^{(i)} - lambda*Risk^{(i)}
```

其中所有指标均已归一化至 [0, 1] 区间。

#### 10.5.2 优化目标

```
max  sum_{i=1}^{N} x_i * U_i
```

#### 10.5.3 约束条件

**(1) 志愿数量约束**：sum(x_i) = N_max

**(2) 冲稳保垫比例约束**：

```
N_rush = sum_{i in C_rush} x_i >= floor(N_max * r_rush^min)
N_safe = sum_{i in C_safe} x_i >= floor(N_max * r_safe^min)
N_bottom = sum_{i in C_bottom} x_i >= floor(N_max * r_bottom^min)
```

其中 C_rush = {i | P_admit^{(i)} < 0.30}，C_stable = {i | 0.30 <= P_admit^{(i)} < 0.60} 等。

**(3) 排斥专业约束**：x_i = 0, for all i where major_code_i in M_excluded

**(4) 选科约束**：x_i = 0, for all i where subject_requirement_i not subset of candidate_subjects

**(5) 预算约束**：x_i = 0, for all i where tuition_i > family_budget

**(6) 地域约束**：若 accept_far_city=0，x_i = 0, for all i where distance_i > distance_threshold

**(7) 身体/单科约束**：x_i = 0, for all i where constraints_violated

**(8) 保底志愿最低数量**：N_bottom >= N_bottom^min

**(9) 极高风险志愿上限**：sum_{i where Risk^{(i)} > threshold} x_i <= N_high_risk^max

**(10) 调剂风险触发人工复核**：对于存在调剂风险的专业组，标记 review_required = True。

#### 10.5.4 三种方案的权重设置

| 风险偏好 | alpha(录取概率) | beta(匹配度) | gamma(就业) | delta(城市) | eta(家庭) | lambda(风险惩罚) | 冲:稳:保:垫 |
|---------||---|---|---|---|---|---|---|
| **激进型** | 0.15 | 0.25 | 0.30 | 0.15 | 0.15 | 0.10 | 4:3:2:1 |
| **均衡型** | 0.25 | 0.25 | 0.20 | 0.10 | 0.15 | 0.05 | 2:3:3:2 |
| **保守型** | 0.35 | 0.15 | 0.10 | 0.10 | 0.20 | 0.10 | 1:2:3:4 |

**说明**：激进型录取概率权重偏低（强调"冲"好学校好专业），就业权重高，冲的比例最大。均衡型各方面权重大致均衡。保守型录取概率权重最高（强调"稳"和"保"），保和垫的比例最大。

#### 10.5.5 求解方法：整数规划 + 启发式贪心搜索

该问题是一个典型的**0-1 整数规划问题**。本文采用**分层贪心 + 局部搜索**的启发式方法：

**分层贪心初始化**：
1. 将所有候选志愿按 P_admit 分为四层。
2. 在每一层内，按 U_i 降序排列。
3. 从每一层中选取 Top-K 个志愿，满足比例约束。
4. 若某层候选数量不足，从相邻层中补充。

**局部搜索优化**：在初始解的基础上，执行 2-opt 交换操作：随机选择一个已入选志愿，尝试用剩余候选志愿中的高分志愿替换，若替换后目标函数提升且约束满足，则接受替换。


### 10.6 模型求解

**算法 4：冲稳保志愿组合优化算法（伪代码）**

```
输入: 候选志愿列表 candidates, 考生画像 profile, 权重方案 plan_type
输出: 志愿表 volunteer_list

1. 候选志愿预处理:
   FOR each candidate c in candidates:
   a. 应用硬约束过滤:
      - 排斥专业过滤
      - 选科不匹配过滤
      - 预算超标过滤
      - 身体/单科限制过滤
      - 地域限制过滤（若 accept_far_city=0）
   b. 计算 U_c = alpha*P_admit + beta*M_fit + gamma*E_career
                  + delta*C_city + eta*R_family - lambda*Risk
   c. 标记 tier: rush/stable/safe/bottom

2. 根据 plan_type 设定参数:
   - 权重向量 [alpha, beta, gamma, delta, eta, lambda]
   - 比例约束 [r_rush, r_stable, r_safe, r_bottom]
   - N_max（志愿表容量，依省份规则）

3. 分层贪心初始化:
   FOR tier IN [rush, stable, safe, bottom]:
   a. 筛选该 tier 的候选志愿
   b. 按 U 降序排序
   c. 计算该 tier 的配额: quota_tier = round(N_max * r_tier)
   d. 取 Top-quota_tier 个志愿
   e. 若候选不足: 从相邻 tier 补充

4. 局部搜索优化（迭代 K 次）:
   FOR iter = 1 TO K:
   a. 随机选择当前志愿表中的一个志愿 c_out
   b. 从未入选的候选志愿中选择 U 最高的志愿 c_in
      (确保 c_in 满足所有硬约束)
   c. 若替换后志愿表仍满足比例约束且 sum(U) 提升:
      执行替换
   d. 记录最优解

5. 志愿表排序与校验:
   - 按 admission_probability 升序排列（冲->稳->保->垫）
   - 检查是否有 user-excluded 专业混入
   - 检查调剂风险是否触发人工复核
   - 检查所有约束是否满足

6. 输出志愿表和评估结果
```

### 10.7 输出结果（模拟数据示例）

| 字段 | 示例值 |
|------|--------|
| candidate_id | "cand_2024_hebei_001" |
| plan_type | "balanced" |
| volunteer_list | [{id, school, major, prob, tier, utility}, ...] |
| rush_count | 8 |
| stable_count | 12 |
| safe_count | 12 |
| bottom_count | 8 |
| overall_score | 0.764 |
| overall_risk_level | "medium" |
| review_required | False |
| adjustment_suggestion | "建议将第32志愿替换为录取概率更高的志愿" |

### 10.8 评价指标

| 指标 | 说明 |
|------|------|
| 优化目标函数值 | sum(U_i)，越高越好 |
| 约束满足率 | 100% 为必须 |
| 志愿表覆盖率 | 各层比例与目标比例的偏差 |
| 最差 case 录取概率 | 所有志愿被拒绝的概率连乘（越小越好） |
| 计算时间 | 应在 5 秒内完成 |

### 10.9 业务解释

**家长端**：
"根据您的分数和偏好，我们为您推荐以下志愿方案（均衡型）：前 8 个志愿为'冲'，包括 A 大学软件工程等；中间 12 个为'稳'；接着 12 个为'保'；最后 8 个为'垫'，确保有学上。整个志愿表的综合评分为 76.4 分，风险等级为中。需要注意：第 15 志愿的专业组内包含较冷门专业，若不接受调剂请注意风险。"

**咨询师端**：
"均衡型方案已生成，冲:稳:保:垫 = 8:12:12:8，志愿表总分 76.4。各志愿的效用分布合理，录取概率梯度平滑。已自动过滤 3 个超预算志愿和 2 个排斥专业。第 32 志愿（应用心理学）录取概率不确定性大（rank_volatility=0.22），建议人工复核。"

**系统后台端**：优化问题规模 N_candidates=215, 入选=40, 求解耗时 2.3s；约束全部满足；review_required 标记的志愿数量：2。

### 10.10 局限性与改进方向

1. **线性效用假设**：本文假设各指标对总体效用的贡献是线性的。实际上录取概率可能在某阈值附近存在边际效用突变。改进方向：引入分段效用函数或 S 形效用曲线。
2. **志愿表排序问题**：本文生成的志愿表按录取概率升序排列，但平行志愿的排序还受到偏好影响。改进方向：引入二次排序规则（先按偏好层、再按概率层）。
3. **遗传算法的可选项**：对于超大规模候选集（N>500），启发式求解可能陷入局部最优。改进方向：引入遗传算法（GA）或多目标进化算法（NSGA-II）。

---

## 十一、志愿填报风险评估模型（核心模型五）

### 11.1 业务问题说明

即使录取概率预测准确、志愿组合结构合理，志愿填报仍然存在多种不可忽视的风险。这些风险包括：所有志愿都没被录取（滑档）、虽被投档但因不满足专业录取条件被退回（退档）、被调剂到不喜欢的专业（调剂风险）、选到就业前景差的冷门专业（冷门风险）、就业困难（就业风险），以及去了一座不适合发展的城市（地域风险）。

核心业务问题：**评估一份志愿方案的综合风险水平，识别具体的高风险点，并给出修改建议。**

### 11.2 符号说明

| 符号 | 含义 |
|------|------|
| Risk_total | 综合风险评分（0-1，越高越危险） |
| R_slip | 滑档风险——全部志愿均未被录取的概率 |
| R_withdrawal | 退档风险——投档后因不符合条件被退回的概率 |
| R_adjust | 调剂风险——被调剂到冷门或排斥专业的概率 |
| R_cold | 专业冷门风险——专业就业前景差的风险 |
| R_employment | 就业风险——毕业后就业困难的风险 |
| R_region | 地域风险——城市不适合长期发展的风险 |
| vol_i | 第 i 个志愿的历史录取位次波动 |

### 11.3 模型建立与求解

#### 11.3.1 滑档风险 R_slip

滑档风险 = 志愿方案中所有志愿全不被录取的概率：

```
R_slip = prod_{i=1}^{N} (1 - P_admit^{(i)})
```

使用蒙特卡洛模拟法：
1. 模拟 M 次（M=10000）
2. 每次模拟中，为每个志愿从它的预测分布中抽样录取事件
3. 计算 M 次中至少一次被录取的比例
4. R_slip = 1 - P(至少一次录取)

若 R_slip > 0.05，说明保底志愿不足，需要添加更多高录取概率的志愿。

#### 11.3.2 退档风险 R_withdrawal

退档风险由以下因素决定：是否满足单科成绩要求、身体条件要求、投档比例（通常为 1:1.05 或 1:1.1）。

```
R_withdrawal^{(i)} = min(1, (admission_count_i - plan_count_i) / admission_count_i + 0.01)
```

若硬性条件不满足，R_withdrawal = 1（已在模型中过滤）。志愿表的退档风险取所有志愿退档风险的最大值。

#### 11.3.3 调剂风险 R_adjust

调剂危险的来源：考生虽被该专业组录取，但未被所填专业录取，从而被调剂到组内其他专业。

调剂危险度计算：若 P_admit^{(i)} < 0.4（冲志愿），调剂风险较高；若专业组内存在考生排斥的冷门专业，调剂风险升级；若考生 accept_adjustment=0 且调剂风险高，需要特别警告。

```
R_adjust = (1/N) * sum_i adjust_danger_i
```

#### 11.3.4 专业冷门风险 R_cold

使用核心模型三输出的标签映射为数值风险：红灯=0.8，黄灯=0.4，绿灯=0.1。以录取概率为权重加权计算志愿表的冷门风险。

#### 11.3.5 就业风险 R_employment 与地域风险 R_region

R_employment = 加权平均的 (1 - E_career^{(i)})

R_region = 加权平均的 (1 - C_city^{(i)})

同时考虑：若考生对城市有明确偏好但志愿表中偏好城市志愿比例过低，则添加额外地域错配惩罚。

#### 11.3.6 综合风险评分

```
Risk_total = 0.30*R_slip + 0.25*R_withdrawal + 0.20*R_adjust
             + 0.10*R_cold + 0.10*R_employment + 0.05*R_region
```

风险等级划分：

| Risk_total | 风险等级 |
|------------|---------|
| < 0.15 | 低 |
| 0.15 - 0.30 | 中 |
| 0.30 - 0.50 | 高 |
| > 0.50 | 极高 |


#### 11.3.7 十大典型问题识别规则

| 编号 | 问题描述 | 识别规则 |
|------|---------|---------|
| Q1 | 冲太多 | N_rush / N_max > 0.5 |
| Q2 | 稳太少 | N_stable < N_max * 0.2 |
| Q3 | 保底不足 | R_slip > 0.05 |
| Q4 | 冷热差异大 | 存在调剂危险度 > 0.7 的志愿 |
| Q5 | 调剂风险高 | R_adjust > 0.4 |
| Q6 | 排斥专业仍被纳入 | 志愿表中存在 M_excluded 中的专业 |
| Q7 | 就业风险未提示 | 志愿表中有红灯专业但未被警告 |
| Q8 | 城市/地域忽略 | 偏好城市覆盖率为 0 且距离超阈值 |
| Q9 | 小样本概率高估 | 存在小样本专业的概率估计且 sigma > 0.2 |
| Q10 | 历史波动大未复核 | 存在 rank_volatility > 0.25 但 review_required=False |

### 11.4 模型求解

**算法 5：风险评估算法（伪代码）**

```
输入: 志愿表 volunteer_list, 考生画像 profile, 各模型输出
输出: risk_scores, risk_level, risk_reasons, modification_suggestions

1. 计算滑档风险 R_slip:
   - 蒙特卡洛模拟 M 次
   - R_slip = 1 - P(至少一次录取)
   - 若 R_slip > 0.05: 触发 Q3 警告

2. 计算退档风险 R_withdrawal:
   FOR each volunteer v:
   - 检查单科/身体限制
   - 从历史数据计算超录比例
   R_withdrawal = max(R_withdrawal_i)

3. 计算调剂风险 R_adjust:
   FOR each volunteer v:
   - 若 P_admit 偏低 -> 调剂风险较高
   - 若专业组内含排斥专业 -> 风险升级
   - 若 accept_adjustment=0 -> 风险升级
   R_adjust = mean(adjust_danger_i)
   若 > 0.4: 触发 Q5 警告

4. 计算冷门风险 R_cold:
   P 加权平均冷门标签值

5. 计算就业风险 R_employment:
   P 加权平均 (1 - E_career)

6. 计算地域风险 R_region:
   P 加权平均 (1 - C_city)

7. 计算综合风险: Risk_total = 加权求和
   确定风险等级

8. 十大问题扫描:
   逐一检查 Q1-Q10, 生成 risk_reasons 列表

9. 生成修改建议:
   FOR each triggered problem:
   基于问题类型自动生成修改建议文本

10. 返回结果
```

### 11.5 输出结果（模拟数据示例）

| 字段 | 示例值 |
|------|--------|
| overall_risk_score | 0.22 |
| risk_level | "medium" |
| slip_risk | 0.008 |
| withdrawal_risk | 0.03 |
| adjustment_risk | 0.28 |
| cold_major_risk | 0.15 |
| employment_risk | 0.18 |
| region_risk | 0.12 |
| risk_reason | ["Q4: 第15志愿专业组冷热差异大(调剂危险度0.72)", "Q10: 第32志愿历史波动率达0.25未触发复核"] |
| modification_suggestion | "建议将第15志愿替换为专业组内部专业差异较小的同类院校；对第32志愿进行人工复核。" |
| review_required | True |

### 11.6 评价指标

| 指标 | 说明 |
|------|------|
| 风险预警准确率 | 模型标记的高风险志愿经咨询师确认确实存在问题的比例 |
| 虚假预警率 | 模型标记高风险但实际无风险的比例（越低越好） |
| 遗漏率 | 实际存在风险但模型未标记的比例（越低越好） |
| 滑档风险预测准确度 | 历史回测，比较预测滑档概率与实际滑档发生率 |
| 修改建议采纳率 | 用户/咨询师对修改建议的采纳比例 |

### 11.7 业务解释

**家长端**：
"您的志愿方案综合风险评为中等，具体来看：滑档风险很低（因为您有充足的保底志愿），退档风险也较低。需要特别注意的是，第 15 志愿的专业组内部各专业录取分差异较大，如果不幸被调剂，可能进入您不太喜欢的专业。建议考虑替换为专业组更'纯净'的志愿。"

**咨询师端**：
"风险评估结果：overall=0.22(中)。R_slip=0.008(低)，保底志愿充足。R_adjust=0.28(中偏高)，主因第15、23志愿专业组内部标准差大。R_employment=0.18(低)，表中无红灯专业。Q4(冷热差异)、Q10(波动未复核)触发。建议替换第15志愿或手动确认调剂风险接受。结论：需复核。"

**系统后台端**：review_required=True 的志愿需推送至咨询师待办列表；risk_reason 存入 model_output 表，包含可追溯的触发规则标识；修改建议以 JSON 结构化格式存储。

### 11.8 局限性与改进方向

1. **风险类别权重的经验性**：各类风险的权重基于专家经验设定，未经过大规模数据校准。改进方向：通过咨询师反馈数据用机器学习方法优化权重。
2. **招生章程提取的准确性**：单科限制和身体限制的正则表达式提取可能存在遗漏或误提取。改进方向：结合 NLP 实体识别技术提高抽取精度。
3. **调剂方向预测的困难**：专业组内具体调剂到哪个专业涉及微观的报考行为，模型难以精确预测。改进方向：仅做风险等级预警而不做具体调剂专业预测。


---

## 十二、多目标综合推荐模型

### 12.1 总体框架

本文的核心是将前述五个核心模型有机融合为一个统一的推荐系统。总模型框架为：

```
max U = alpha*P_admit + beta*M_fit + gamma*E_career + delta*C_city + eta*R_family - lambda*Risk
```

### 12.2 各指标计算方式与归一化

| 指标 | 业务含义 | 计算来源 | 归一化 |
|------|---------|---------|--------|
| P_admit | 录取概率 | 核心模型二输出 | 已为 [0,1] |
| M_fit | 专业匹配度 | 兴趣方向 + 优势科目与专业的相似度 | Min-Max 归一化 |
| E_career | 就业价值 | 核心模型三输出 | 已为 [0,1] |
| C_city | 城市价值 | 城市产业数据加权评分 | Min-Max 归一化 |
| R_family | 家庭匹配度 | 考生偏好 vs 志愿属性的吻合度 | 匹配度计算后归一化 |
| Risk | 综合风险 | 核心模型五输出 | 已为 [0,1] |

**M_fit 的计算**：

```
M_fit = 0.6 * |intersection(interest_direction, major_keywords)| / |interest_direction|
        + 0.4 * |intersection(strong_subjects, major_subjects)| / |strong_subjects|
```

**R_family 的计算**：

```
R_family = (1/K) * sum_{k=1}^{K} w_k * 1[preference_k satisfied]
```

其中子指标包括：城市偏好达成、预算合适度（tuition/budget）、距离接受度等。

**C_city 的计算**：

```
C_city = 0.25 * GDP_norm + 0.20 * gdp_per_capita_norm + 0.15 * tertiary_ratio_norm
         + 0.15 * high_tech_norm + 0.10 * salary_norm - 0.10 * living_cost_norm
         + 0.05 * transport_norm
```

### 12.3 推荐流程

```
1. 输入考生画像
2. 核心模型一：计算等效分/位次
3. 候选志愿集合生成（基于等效分 +/- delta 范围的院校和专业）
4. 核心模型二：为每个候选志愿计算录取概率 P_admit
5. 核心模型三：为每个候选志愿的专业计算就业评分 E_career
6. 计算 M_fit, C_city, R_family
7. 计算综合效用 U_i
8. 核心模型四：志愿组合优化（输出志愿表）
9. 核心模型五：志愿表风险评估
10. 生成最终推荐方案 + 解释文本 + JSON 输出
```

---

## 十三、系统接口与输出设计

### 13.1 统一 JSON 输出示例（基于模拟数据）

```json
{
  "meta": {
    "model_version": "v2.1.0",
    "generate_time": "2024-06-25 10:30:00",
    "data_refresh_date": "2024-06-20"
  },
  "candidate": {
    "candidate_id": "cand_2024_hebei_001",
    "province": "河北省",
    "subject_type": "物理类",
    "score": 620,
    "rank": 8500,
    "percentile": 0.0254,
    "equivalent_scores": {
      "2023": {
        "equivalent_score": 616,
        "equivalent_score_interval": [609, 623],
        "confidence_level": "high"
      },
      "2022": {
        "equivalent_score": 614,
        "equivalent_score_interval": [606, 622],
        "confidence_level": "high"
      },
      "2021": {
        "equivalent_score": 608,
        "equivalent_score_interval": [593, 623],
        "confidence_level": "medium",
        "abnormal_year_warning": true
      }
    },
    "risk_preference": "balanced"
  },
  "recommendation_plan": {
    "plan_type": "balanced",
    "volunteers": [
      {
        "volunteer_id": 1,
        "school_code": "10003",
        "school_name": "A理工大学",
        "major_group_code": "10003_001",
        "major_code": "080901",
        "major_name": "计算机科学与技术",
        "admit_probability": 0.25,
        "probability_interval": [0.18, 0.32],
        "recommendation_tier": "rush",
        "fit_score": 0.82,
        "career_score": 0.88,
        "city_score": 0.72,
        "family_score": 0.80,
        "risk_level": "medium",
        "risk_reason": "冲志愿概率偏低，录取不确定性较大",
        "overall_utility": 0.712,
        "explanation": "A理工大学计算机科学与技术专业为您的冲志愿。根据近5年数据分析，您的预估录取概率约为25%（区间18%-32%）。",
        "modification_suggestion": "",
        "review_required": false
      },
      {
        "volunteer_id": 2,
        "school_code": "10005",
        "school_name": "B大学",
        "major_group_code": "10005_002",
        "major_code": "080902",
        "major_name": "软件工程",
        "admit_probability": 0.45,
        "probability_interval": [0.38, 0.52],
        "recommendation_tier": "stable",
        "fit_score": 0.75,
        "career_score": 0.85,
        "city_score": 0.80,
        "family_score": 0.90,
        "risk_level": "low",
        "overall_utility": 0.786,
        "explanation": "B大学软件工程专业为您的稳志愿。录取概率约45%，属于有较大把握的范畴。",
        "review_required": false
      },
      {
        "volunteer_id": 32,
        "school_code": "10020",
        "school_name": "D大学",
        "major_group_code": "10020_005",
        "major_code": "071101",
        "major_name": "心理学",
        "admit_probability": 0.52,
        "probability_interval": [0.30, 0.74],
        "recommendation_tier": "stable",
        "fit_score": 0.55,
        "career_score": 0.42,
        "city_score": 0.65,
        "family_score": 0.60,
        "risk_level": "high",
        "risk_reason": "历史录取位次波动大(volatility=0.25)；小样本(仅2年数据)；专业组内含冷门专业，调剂风险高",
        "overall_utility": 0.542,
        "explanation": "D大学心理学专业存在较高不确定性。该专业仅2年录取数据，预测概率区间宽(30%-74%)。建议复核或替换。",
        "modification_suggestion": "建议替换为录取数据更充分(>=3年)的同类志愿。",
        "review_required": true
      }
    ],
    "statistics": {
      "rush_count": 8,
      "stable_count": 12,
      "safe_count": 12,
      "bottom_count": 8,
      "overall_score": 0.764,
      "overall_risk_level": "medium"
    },
    "risk_assessment": {
      "overall_risk_score": 0.22,
      "risk_level": "medium",
      "slip_risk": 0.008,
      "withdrawal_risk": 0.03,
      "adjustment_risk": 0.28,
      "cold_major_risk": 0.15,
      "employment_risk": 0.18,
      "region_risk": 0.12,
      "risk_reason": [
        "Q4: 第15志愿专业组冷热差异大(调剂危险度0.72)",
        "Q10: 第32志愿历史波动率达0.25"
      ],
      "modification_suggestion": "建议将第15志愿替换为专业组内部专业差异较小的同类院校；对第32志愿进行人工复核。",
      "review_required": true
    }
  }
}
```


---

## 十四、模型评价与验收标准

### 14.1 技术验收标准

| 编号 | 指标 | 目标值 | 测量方法 |
|------|------|--------|---------|
| T1 | 数据爬取成功率 | > 90% | (成功爬取的表数/目标表数) |
| T2 | 字段完整率 | > 85% | (非空字段数/总字段数) |
| T3 | 数据重复率 | < 2% | (重复行数/总行数) |
| T4 | 异常数据占比 | < 5% | (被标记异常的行数/总行数) |
| T5 | 数据源可追溯率 | 100% | (有 source_url 的行数/总行数) |
| T6 | 分数等效换算误差 (MAE) | < 3 分 | 留一验证 |
| T7 | 录取概率校准曲线 | 斜率 1+/-0.1, 截距 0+/-0.05 | 可靠性图 |
| T8 | AUC | > 0.85 | ROC 曲线 |
| T9 | Brier Score | < 0.15 | 概率预测误差 |
| T10 | 录取位次预测 MAE | 根据数据量浮动 | 留一年验证 |
| T11 | 录取位次预测 RMSE | 根据数据量浮动 | 留一年验证 |
| T12 | 模型稳定性 | 各年 AUC 标准差 < 0.03 | 逐年回测 |
| T13 | 接口响应时间 | < 5 秒 | 压力测试 |
| T14 | 同一输入结果可复现性 | 100% | 两次相同输入产出一致 |

### 14.2 业务验收标准

| 编号 | 指标 | 目标 | 测量方法 |
|------|------|------|---------|
| B1 | 咨询师使用意愿 | > 70% | 咨询师满意度问卷 |
| B2 | 家长理解度 | > 80% | "您是否理解了推荐理由"问卷 |
| B3 | 高风险志愿识别率 | > 90% | 与咨询师人工标注对照 |
| B4 | 人工复核触发率 | 10%-30%（适中） | 过高说明模型不可靠，过低说明漏检 |
| B5 | 违规表述检查 | 0 次 | 自动扫描"保证录取""一定录取"等关键词 |
| B6 | 咨询师效率提升 | 筛选时间减少 > 50% | 对比人工筛选耗时 |
| B7 | 志愿表安全性 | 滑档风险 < 5% | 历史回测 |
| B8 | 可交付报告 | 支持一键生成 PDF/WORD | 功能测试 |
| B9 | 系统可接入性 | 完整 JSON API | 集成测试 |
| B10 | 案例沉淀率 | 100% 填报案例保留 | 数据库记录统计 |

---

## 十五、模型优缺点分析

### 15.1 优点

1. **体系完整**：覆盖了从数据采集、清洗、建库到五个核心模型建模、再到系统接口输出的完整链路，形成了可落地实施的总体方案。
2. **模型服务业务**：每个模型都明确对应一个具体的业务问题，输出结果面向三类用户（家长、咨询师、系统后台）做了差异化解释。
3. **数据驱动与专家经验结合**：AHP 层次分析法引入专家经验，熵权法利用数据自身信息，TOPSIS 综合评价结合两者优势，避免了纯数据驱动或纯经验驱动的偏颇。
4. **风险视角全面**：覆盖滑档、退档、调剂、冷门、就业、地域六类风险，并设置了十大典型问题的自动识别规则。
5. **可解释性强**：每个模型输出都包含了面向非技术用户的解释文本，避免"黑箱模型"在志愿填报决策中引起不信任。
6. **可扩展性好**：模块化设计，各模型之间通过统一的数据接口耦合，可单独升级或替换。

### 15.2 缺点

1. **数据依赖性强**：模型效果严重依赖数据质量和数据覆盖度。对于录取数据不公开的省份、专业就业数据不完善的专业，模型效果会显著下降。
2. **部分参数基于经验设定**：AHP 判断矩阵、风险权重、概率阈值等参数依赖专家经验，缺乏大规模实证校准。
3. **专业热度变化的预测能力有限**：模型可以基于历史趋势外推，但无法预测技术突破、政策变动等带来的专业热度突变。
4. **计算资源需求较高**：蒙特卡洛模拟和整数规划的启发式搜索在大规模候选集下需要较高的计算资源。
5. **地区差异性处理不足**：各省高考政策和志愿填报规则差异较大，模型中的部分规则不具有全国通用性。
6. **未充分考虑平行志愿的排序效应**：平行志愿按分数优先、遵循志愿的原则投档，志愿的顺序既影响录取概率也影响考生的真实满意度，本文的概率独立性假设可能高估了后位志愿的录取机会。

### 15.3 主要业务风险提示

1. 模型输出的录取概率是**估计值**而非保证值。任何低于 100% 的录取概率都意味着存在不被录取的可能。
2. 就业景气度评价基于**历史数据**。在当前经济与技术快速变革的背景下，四年后的就业市场可能与当前显著不同。
3. 本文模型**不能替代专业咨询师的判断**。对于涉及考生人生重大抉择的建议，应由咨询师结合经验进行最终确认。
4. 系统输出的"冲、稳、保、垫"标签仅用于辅助决策参考，不构成推荐承诺。

---

## 十六、模型推广与改进方向

### 16.1 推广方向

1. **多省份适配**：当前模型以单一省份为基础设计，可推广为多省份通用模型。关键是建立各省高考政策的规则引擎，将志愿填报模式、批次设置、志愿数量等规则参数化。
2. **多轮动态推荐**：在志愿填报窗口期内，系统可以根据已填报考生的情况（通过公开的实时填报数据或模拟预测）动态调整推荐策略。
3. **产品化路径**：将模型封装为 SaaS 服务，通过 API 接入志愿填报 APP、小程序、线下咨询系统等多种终端。
4. **家长自助模式**：对低风险（如成绩优异、选择明确）的考生，提供全自动推荐模式；对高风险（如踩线生、有特殊需求）考生，引导至人工咨询师复核。

### 16.2 改进方向

1. **深度学习方法引入**：对于录取概率预测，尝试使用基于自注意力机制（Transformer）的时序预测模型，更好地捕捉录取位次的多年度变化模式。
2. **NLP 增强招生章程解析**：使用预训练语言模型（如 BERT）对招生章程进行细粒度信息抽取，提高选科限制、身体条件、单科限制的提取准确率。
3. **知识图谱构建**：构建"院校-专业-行业-城市-岗位"知识图谱，增强专业匹配度和就业评价的语义理解能力。
4. **强化学习优化志愿排列**：使用强化学习框架直接优化志愿表的排序，以最大化期望录取满意度为目标。
5. **联邦学习保护数据隐私**：在各咨询机构之间通过联邦学习方式共享模型参数而非原始数据，在保护数据隐私的前提下提升模型泛化能力。
6. **实时舆情监测**：接入更实时的就业市场数据流，动态更新专业就业景气度评价。
7. **A/B 测试驱动参数优化**：在实际使用中设计 A/B 测试框架，通过用户满意度和真实录取结果反馈，持续优化模型参数。

---

## 参考文献

[1] 中华人民共和国教育部. 普通高等学校本科专业目录(2024年版)[Z]. 2024.

[2] 教育部高校学生司. 普通高等学校招生工作规定[Z]. 2024.

[3] 各省(自治区、直辖市)教育考试院. 历年高考成绩一分一段表[EB/OL].

[4] 各省(自治区、直辖市)教育考试院. 历年普通高等学校招生各批次投档线公告[EB/OL].

[5] 教育部高校招生阳光工程指定平台(阳光高考). 院校库&招生章程[EB/OL]. https://gaokao.chsi.com.cn.

[6] 各普通高等学校. 毕业生就业质量年度报告[R]. 2020-2024.

[7] 国家统计局. 中国统计年鉴[R]. 2020-2024.

[8] 中华人民共和国人力资源和社会保障部. 全国招聘大于求职"最缺工"的100个职业排行[EB/OL]. 2020-2024.

[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]. KDD, 2016.

[10] Saaty T L. The Analytic Hierarchy Process[M]. McGraw-Hill, 1980.

[11] Hwang C L, Yoon K. Multiple Attribute Decision Making: Methods and Applications[M]. Springer, 1981.

[12] Shannon C E. A Mathematical Theory of Communication[J]. Bell System Technical Journal, 1948.

[13] 李航. 统计学习方法[M]. 清华大学出版社, 2019.

[14] 司守奎, 孙玺菁. 数学建模算法与应用(第3版)[M]. 国防工业出版社, 2021.

[15] Bishop C M. Pattern Recognition and Machine Learning[M]. Springer, 2006.



---

## 附录

### 附录 A：爬虫伪代码

#### A.1 通用爬虫类

```python
"""
高考数据爬虫框架（伪代码/代码框架）
注意：此为教学演示代码，实际部署需根据目标网站结构调整选择器和URL模式。
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
import time
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class GaoKaoDataCrawler:
    """高考数据爬虫基类"""

    def __init__(self, delay_seconds=3):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        self.delay = delay_seconds
        self.source_urls = {}

    def check_robots(self, base_url):
        """检查robots.txt规则"""
        robots_url = urljoin(base_url, "/robots.txt")
        try:
            resp = self.session.get(robots_url, timeout=10)
            if resp.status_code == 200:
                print(f"[INFO] robots.txt found at {robots_url}")
        except Exception:
            pass

    def fetch_html(self, url, max_retries=3):
        """获取静态HTML页面"""
        time.sleep(self.delay)
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=30, allow_redirects=True)
                resp.encoding = resp.apparent_encoding
                if resp.status_code == 200:
                    return resp.text
                elif resp.status_code in (404, 403):
                    print(f"[WARN] HTTP {resp.status_code} for {url}")
                    return None
                else:
                    time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[ERROR] Fetch attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

    def parse_html_table(self, html, table_index=0):
        """解析HTML中的table标签为DataFrame"""
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return None
        table = tables[table_index]
        rows = table.find_all("tr")
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        data = []
        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            data.append([col.get_text(strip=True) for col in cols])
        return pd.DataFrame(data, columns=headers)

    def parse_pdf_table(self, pdf_path, page_range=None):
        """解析PDF中的表格"""
        tables_list = []
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[page_range[0]:page_range[1]] if page_range else pdf.pages
            for page in pages:
                extracted = page.extract_tables()
                for table in extracted:
                    if table:
                        table = [row for row in table if any(cell for cell in row)]
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables_list.append(df)
        return pd.concat(tables_list, ignore_index=True) if tables_list else None

    def read_excel(self, file_path, sheet_name=0):
        """读取Excel文件"""
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def read_csv(self, file_path, encoding="utf-8"):
        """读取CSV文件"""
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="gbk")


class DynamicPageCrawler:
    """动态网页爬虫（使用Selenium）"""

    def __init__(self, chromedriver_path=None):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(options=self.options)

    def fetch_dynamic_table(self, url, wait_selector="table", timeout=10):
        """等待动态加载的表格出现后抓取"""
        self.driver.get(url)
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
        )
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if tables:
            rows = tables[0].find_all("tr")
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cols = row.find_all(["td", "th"])
                data.append([col.get_text(strip=True) for col in cols])
            return pd.DataFrame(data, columns=headers)
        return None

    def close(self):
        self.driver.quit()
```

#### A.2 数据清洗核心函数

```python
def clean_segment_data(df, province, year):
    """清洗一分一段表数据"""
    col_map = {
        "分数": "score", "分数段": "score", "本段人数": "segment_count",
        "累计人数": "cumulative_count", "位次": "rank", "批次线": "batch_line"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["score", "segment_count", "cumulative_count", "rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["score", "cumulative_count"]].isnull().mean().max() > 0.05:
        df["review_status"] = 2
    if "score" in df.columns:
        df = df[(df["score"] >= 0) & (df["score"] <= 750)]
    df = df.drop_duplicates(subset=["score"])
    df["province"] = province
    df["year"] = year
    df["crawl_time"] = pd.Timestamp.now()
    df = df.sort_values("score", ascending=False)
    total = df["cumulative_count"].max()
    df["percentile"] = df["cumulative_count"] / total
    return df


def standardize_school_name(name):
    """院校名称标准化"""
    aliases = {
        "北京大学": ["北京大学", "北京大学(校本部)", "Peking University"],
        "清华大学": ["清华大学", "Tsinghua University"],
    }
    for standard, variants in aliases.items():
        if name in variants:
            return standard
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name


def extract_admission_requirements(html_content):
    """从招生章程HTML中提取选科、身体、单科限制（正则表达式）"""
    requirements = {"single_subject_limit": [], "physical_limit": [], "subject_requirement": []}
    patterns_single = [
        r"(英语|数学|语文|物理|化学|生物)单科成绩(不|须)(低于|达到|不低于)\s*(\d+)\s*分",
        r"(英语|数学)高考成绩\s*(\d+)\s*分以上",
    ]
    for pattern in patterns_single:
        matches = re.findall(pattern, html_content)
        if matches:
            requirements["single_subject_limit"].extend(["".join(m) for m in matches])
    patterns_physical = [
        r"(不招|不宜报考).*?(色盲|色弱|高度近视)",
        r"身高(不低于|要求)\s*(\d+)\s*cm",
    ]
    for pattern in patterns_physical:
        matches = re.findall(pattern, html_content)
        if matches:
            requirements["physical_limit"].extend(["".join(m) for m in matches])
    patterns_subject = [
        r"选考科目要求[：:]\s*(.+)",
        r"选科要求[：:]\s*(.+)",
        r"首选科目[：:]\s*(.+)",
    ]
    for pattern in patterns_subject:
        match = re.search(pattern, html_content)
        if match:
            requirements["subject_requirement"].append(match.group(1).strip())
    return requirements


def validate_admission_data(df):
    """录取数据逻辑校验: min <= avg <= max"""
    errors = []
    if all(col in df.columns for col in ["min_admission_score", "avg_score", "max_score"]):
        mask = (df["min_admission_score"] > df["avg_score"]) | (df["avg_score"] > df["max_score"])
        if mask.any():
            errors.append(f"min<=avg<=max violation: {mask.sum()} rows")
    return errors


def save_to_database(df, table_name, db_connection):
    """将清洗后的数据入库"""
    df.to_sql(name=table_name, con=db_connection, if_exists="append", index=False, method="multi")
    print(f"[DB] {len(df)} rows inserted into {table_name}")
```


#### A.3 建模核心函数伪代码

```python
"""
建模算法核心函数（伪代码/代码框架）
注意：以下代码为算法逻辑示意，实际部署需补充完整实现。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats


def equivalent_score_conversion(candidate_score, candidate_rank, segment_df_list,
                                 current_batch_line, current_year, target_years,
                                 theta=0.3, tau=1000):
    """
    分数-位次等效换算（核心模型一）
    """
    results = {}
    for t_year, seg_df in zip(target_years, segment_df_list):
        total = seg_df["cumulative_count"].max()
        pct = candidate_rank / total
        scores = seg_df["score"].values
        pcts = seg_df["cumulative_count"].values / total
        idx = np.searchsorted(pcts, pct)
        if idx == 0:
            eq_score = scores[0]
        elif idx == len(scores):
            eq_score = scores[-1]
        else:
            eq_score = scores[idx - 1] + (pct - pcts[idx - 1]) / (pcts[idx] - pcts[idx - 1]) * (scores[idx] - scores[idx - 1])
        batch_line_t = seg_df["batch_line"].iloc[0] if "batch_line" in seg_df.columns else 0
        eq_score += theta * ((current_batch_line - batch_line_t) * (1 - pct))
        results[t_year] = {"equivalent_score": round(eq_score, 1), "batch_line": batch_line_t}

    eq_scores = np.array([r["equivalent_score"] for r in results.values()])
    sigma_t = np.abs(eq_scores[:, None] - eq_scores[None, :]).mean(axis=1)
    w_t = np.exp(-sigma_t / tau) / np.exp(-sigma_t / tau).sum()
    threshold = 3 * np.median(sigma_t)
    eq_star = np.sum(w_t * eq_scores)
    sigma_S = np.sqrt(np.sum(w_t * (eq_scores - eq_star) ** 2) / np.sum(w_t))
    ci_lower, ci_upper = eq_star - 1.96 * sigma_S, eq_star + 1.96 * sigma_S

    if sigma_S <= 3:
        confidence = "high"
    elif sigma_S <= 8:
        confidence = "medium"
    else:
        confidence = "low"

    abnormal_years = [year for year, sig in zip(target_years, sigma_t) if sig > threshold]
    return {
        "equivalent_score": round(eq_star, 1),
        "equivalent_interval": [round(ci_lower, 1), round(ci_upper, 1)],
        "confidence_level": confidence,
        "abnormal_year_warning": len(abnormal_years) > 0,
        "abnormal_years": abnormal_years,
    }


def predict_admission_probability(candidate_features, historical_data, model=None):
    """
    录取概率预测（核心模型二）
    """
    features = {}
    ranks = historical_data["min_admission_rank"].values
    features["rank_gap_mean"] = candidate_features["candidate_rank"] - ranks.mean()
    features["rank_gap_std"] = ranks.std()
    features["rank_gap_min"] = candidate_features["candidate_rank"] - ranks.min()
    features["rank_volatility"] = ranks.std() / ranks.mean() if ranks.mean() > 0 else 0

    x_vector = np.array(list(features.values())).reshape(1, -1)
    p_lr, p_xgb = 0, 0
    if model and "lr" in model:
        p_lr = model["lr"].predict_proba(x_vector)[0, 1]
    if model and "xgb" in model:
        p_xgb = model["xgb"].predict_proba(x_vector)[0, 1]

    n_years = historical_data.shape[0]
    if n_years <= 2:
        alpha0, beta0 = 2, 2
        p_bayes = (alpha0 + 1) / (alpha0 + beta0 + n_years)
    else:
        p_bayes = p_lr

    M = 10000
    if n_years >= 2:
        mc_samples = np.random.normal(ranks.mean(), ranks.std(), M)
        mc_results = np.mean(candidate_features["candidate_rank"] <= mc_samples)
        mc_ci = np.percentile(candidate_features["candidate_rank"] <= mc_samples, [2.5, 97.5])
        mc_ci_lower, mc_ci_upper = mc_ci[0], mc_ci[1]
    else:
        mc_results = p_bayes
        mc_ci_lower = max(0, p_bayes - 0.15)
        mc_ci_upper = min(1, p_bayes + 0.15)

    p_final = 0.15 * p_lr + 0.35 * p_xgb + 0.20 * p_bayes + 0.30 * mc_results

    if p_final >= 0.85:
        tier = "bottom"
    elif p_final >= 0.60:
        tier = "safe"
    elif p_final >= 0.30:
        tier = "stable"
    else:
        tier = "rush"

    return {
        "admit_probability": round(p_final, 4),
        "probability_interval": [round(mc_ci_lower, 4), round(mc_ci_upper, 4)],
        "recommendation_tier": tier,
        "uncertainty_level": "high" if n_years <= 2 else ("medium" if n_years <= 3 else "low"),
        "review_required": n_years <= 2,
    }
```


```python

def evaluate_major_career(major_employment_data, ahp_judgment_matrix=None):
    """
    专业就业景气度评价（核心模型三）
    """
    df = major_employment_data.copy()
    indicators = ["employment_rate", "postgraduate_rate", "average_salary",
                  "median_salary", "job_count", "job_growth_rate",
                  "industry_growth_score", "stability_score",
                  "civil_service_post_count", "sentiment_warning_score"]
    positive_indicators = indicators[:-1]
    negative_indicators = ["sentiment_warning_score"]

    df_norm = df.copy()
    for col in positive_indicators:
        if col in df.columns:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    for col in negative_indicators:
        if col in df.columns:
            df_norm[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min() + 1e-10)

    default_weights = {
        "employment_rate": 0.13, "postgraduate_rate": 0.07, "average_salary": 0.20,
        "median_salary": 0.15, "job_count": 0.13, "job_growth_rate": 0.10,
        "industry_growth_score": 0.10, "stability_score": 0.05,
        "civil_service_post_count": 0.05, "sentiment_warning_score": 0.02,
    }

    n = len(df_norm)
    p_ij = df_norm[indicators] / df_norm[indicators].sum()
    k = 1.0 / np.log(n)
    e_j = -k * np.sum(p_ij * np.log(p_ij + 1e-10), axis=0)
    d_j = 1 - e_j
    w_entropy = d_j / d_j.sum()

    w_ahp = np.array([default_weights.get(col, 0) for col in indicators])
    w_comb = (w_ahp * w_entropy.values) / (w_ahp * w_entropy.values).sum()

    v_ij = df_norm[indicators].values * w_comb
    v_plus = v_ij.max(axis=0)
    v_minus = v_ij.min(axis=0)
    d_plus = np.sqrt(np.sum((v_ij - v_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((v_ij - v_minus) ** 2, axis=1))
    c_i = d_minus / (d_plus + d_minus)

    thresholds = np.percentile(c_i, [33, 66])
    labels = np.where(c_i >= thresholds[1], "green",
                      np.where(c_i >= thresholds[0], "yellow", "red"))

    df["career_score"] = c_i
    df["red_yellow_green_label"] = labels
    return df


def optimize_volunteer_combination(candidates_df, profile, plan_type="balanced"):
    """
    冲稳保志愿组合优化（核心模型四）
    """
    weight_schemes = {
        "aggressive": (0.15, 0.25, 0.30, 0.15, 0.15, 0.10),
        "balanced": (0.25, 0.25, 0.20, 0.10, 0.15, 0.05),
        "conservative": (0.35, 0.15, 0.10, 0.10, 0.20, 0.10),
    }
    ratio_schemes = {
        "aggressive": (0.4, 0.3, 0.2, 0.1),
        "balanced": (0.2, 0.3, 0.3, 0.2),
        "conservative": (0.1, 0.2, 0.3, 0.4),
    }

    alpha, beta, gamma, delta, eta, lambd = weight_schemes[plan_type]
    r_rush, r_stable, r_safe, r_bottom = ratio_schemes[plan_type]
    N_max = 40

    df = candidates_df.copy()
    if "excluded_majors" in profile:
        df = df[~df["major_code"].isin(profile["excluded_majors"])]

    df["U"] = (alpha * df["P_admit"] + beta * df["M_fit"] +
               gamma * df["E_career"] + delta * df["C_city"] +
               eta * df["R_family"] - lambd * df["Risk"])

    tiers = {"rush": [], "stable": [], "safe": [], "bottom": []}
    for _, row in df.iterrows():
        p = row["P_admit"]
        if p < 0.30:
            tiers["rush"].append(row)
        elif p < 0.60:
            tiers["stable"].append(row)
        elif p < 0.85:
            tiers["safe"].append(row)
        else:
            tiers["bottom"].append(row)

    selected = []
    for tier_name, ratio in zip(["rush", "stable", "safe", "bottom"],
                                 [r_rush, r_stable, r_safe, r_bottom]):
        tier_df = pd.DataFrame(tiers[tier_name])
        if not tier_df.empty:
            tier_df = tier_df.sort_values("U", ascending=False)
            n_select = int(round(N_max * ratio))
            n_select = min(n_select, len(tier_df))
            selected.extend(tier_df.head(n_select).to_dict("records"))

    result_df = pd.DataFrame(selected)
    result_df = result_df.sort_values("P_admit")

    return {
        "volunteer_list": result_df,
        "rush_count": len(result_df[result_df["P_admit"] < 0.30]),
        "stable_count": len(result_df[(result_df["P_admit"] >= 0.30) & (result_df["P_admit"] < 0.60)]),
        "safe_count": len(result_df[(result_df["P_admit"] >= 0.60) & (result_df["P_admit"] < 0.85)]),
        "bottom_count": len(result_df[result_df["P_admit"] >= 0.85]),
        "overall_score": round(result_df["U"].sum() / len(result_df), 4),
    }
```


```python

def assess_volunteer_risks(volunteer_df, profile, M=10000):
    """
    志愿填报风险评估（核心模型五）
    """
    # 滑档风险: 蒙特卡洛模拟
    slip_simulations = []
    for _ in range(M):
        all_fail = True
        for _, row in volunteer_df.iterrows():
            if np.random.random() < row["P_admit"]:
                all_fail = False
                break
        slip_simulations.append(all_fail)
    R_slip = np.mean(slip_simulations)

    # 退档风险
    R_withdrawal = np.max([min(1, (row.get("admission_count", 1) - row.get("plan_count", 1)) /
                              max(row.get("admission_count", 1), 1) + 0.01)
                           for _, row in volunteer_df.iterrows()])

    # 调剂风险
    R_adjust = volunteer_df["adjust_danger"].mean() if "adjust_danger" in volunteer_df.columns else 0.1

    # 冷门/就业/地域风险
    weights = volunteer_df["P_admit"] / volunteer_df["P_admit"].sum()
    R_cold = np.average(volunteer_df.get("cold_risk", 0.1), weights=weights)
    R_employment = np.average(1 - volunteer_df.get("E_career", 0.5), weights=weights)
    R_region = np.average(1 - volunteer_df.get("C_city", 0.5), weights=weights)

    Risk_total = (0.30 * R_slip + 0.25 * R_withdrawal + 0.20 * R_adjust +
                  0.10 * R_cold + 0.10 * R_employment + 0.05 * R_region)

    # 十大问题扫描
    risk_reasons = []
    N_max = len(volunteer_df)
    N_rush = (volunteer_df["P_admit"] < 0.30).sum()
    if N_rush / N_max > 0.5:
        risk_reasons.append("Q1: 冲太多")
    if R_slip > 0.05:
        risk_reasons.append("Q3: 保底不足")

    if Risk_total > 0.50:
        risk_level = "critical"
    elif Risk_total > 0.30:
        risk_level = "high"
    elif Risk_total > 0.15:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "overall_risk_score": round(Risk_total, 4),
        "risk_level": risk_level,
        "slip_risk": round(R_slip, 4),
        "withdrawal_risk": round(R_withdrawal, 4),
        "adjustment_risk": round(R_adjust, 4),
        "cold_major_risk": round(R_cold, 4),
        "employment_risk": round(R_employment, 4),
        "region_risk": round(R_region, 4),
        "risk_reason": risk_reasons,
        "review_required": Risk_total > 0.20,
    }


def generate_output_json(candidate_id, profile, equivalent_scores,
                         volunteer_df, risk_assessment):
    """生成统一JSON输出"""
    import json
    from datetime import datetime

    output = {
        "meta": {
            "model_version": "v2.1.0",
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "candidate": {
            "candidate_id": candidate_id,
            "province": profile["province"],
            "subject_type": profile["subject_type"],
            "score": profile["score"],
            "rank": profile["rank"],
            "equivalent_scores": equivalent_scores,
            "risk_preference": profile["risk_preference"],
        },
        "recommendation_plan": {
            "plan_type": profile.get("plan_type", "balanced"),
            "volunteers": [],
            "statistics": {
                "rush_count": int((volunteer_df["P_admit"] < 0.30).sum()),
                "stable_count": int(((volunteer_df["P_admit"] >= 0.30) &
                                     (volunteer_df["P_admit"] < 0.60)).sum()),
                "safe_count": int(((volunteer_df["P_admit"] >= 0.60) &
                                   (volunteer_df["P_admit"] < 0.85)).sum()),
                "bottom_count": int((volunteer_df["P_admit"] >= 0.85).sum()),
                "overall_score": round(volunteer_df["U"].mean(), 4),
                "overall_risk_level": risk_assessment["risk_level"],
            },
            "risk_assessment": risk_assessment,
        },
    }

    for _, row in volunteer_df.iterrows():
        output["recommendation_plan"]["volunteers"].append({
            "volunteer_id": int(row.get("volunteer_id", 0)),
            "school_code": row["school_code"],
            "school_name": row["school_name"],
            "major_code": row["major_code"],
            "major_name": row["major_name"],
            "admit_probability": round(row["P_admit"], 4),
            "probability_interval": row.get("prob_interval", [0, 0]),
            "recommendation_tier": row["recommendation_tier"],
            "fit_score": round(row["M_fit"], 2),
            "career_score": round(row["E_career"], 2),
            "city_score": round(row["C_city"], 2),
            "family_score": round(row["R_family"], 2),
            "risk_level": row.get("risk_level", "low"),
            "overall_utility": round(row["U"], 4),
            "explanation": row.get("explanation", ""),
            "modification_suggestion": row.get("modification_suggestion", ""),
            "review_required": row.get("review_required", False),
        })

    return json.dumps(output, ensure_ascii=False, indent=2)
```

### 附录 B：测试案例字段（模拟数据）

| 字段 | 示例值 | 说明 |
|------|--------|------|
| candidate_id | cand_2024_hebei_001 | 模拟ID |
| province | 河北省 | 模拟省份 |
| subject_type | 物理类 | |
| score | 620 | 模拟分数 |
| rank | 8500 | 模拟位次 |
| risk_preference | balanced | |
| school_code | 10003 | 模拟院校代码 |
| school_name | A理工大学 | 模拟院校名 |
| major_code | 080901 | 计算机科学与技术 |
| admit_probability | 0.68 | 模拟录取概率 |
| recommendation_tier | stable | |
| career_score | 0.82 | |
| city_score | 0.72 | |
| risk_level | medium | |
| review_required | False | |

**声明**：以上测试案例中的院校名称、分数和位次均为模拟数据，仅用于展示模型输出格式和接口结构，不代表任何真实考生或真实院校的录取情况。

### 附录 C：数据来源链接汇总

| 数据类别 | 典型来源 | URL 示例 |
|---------|---------|---------|
| 一分一段表 | 各省教育考试院 | 各省教育考试院官网 |
| 院校投档线 | 省教育考试院 | 各省教育考试院官网 |
| 专业录取数据 | 高校本科招生网 | 各高校官网"本科招生"栏目 |
| 招生计划 | 省教育考试院 | 各省《招生计划》PDF公告 |
| 招生章程 | 阳光高考平台 | https://gaokao.chsi.com.cn |
| 专业目录 | 教育部 | http://www.moe.gov.cn |
| 就业质量报告 | 各高校就业指导中心 | 各高校官网 |
| 就业岗位数据 | 人社部 | http://www.mohrss.gov.cn |
| 城市经济数据 | 国家统计局 | https://data.stats.gov.cn |
| 公务员招录 | 国家公务员局 | http://bm.scs.gov.cn |

**注意**：以上 URL 为示意性地址，实际爬取时需以各网站最新发布地址为准，并在爬取前检查目标网站的 robots.txt 和访问条款。

---

**论文完**

*本文中的所有模拟数据仅供数学建模演示使用，不构成对任何考生、院校或专业的实际评价。模型输出的录取概率为基于历史数据和统计方法的估计值，不具有任何保证性质。考生和家长在实际志愿填报中应结合自身情况和官方发布的政策信息做出决策。*

