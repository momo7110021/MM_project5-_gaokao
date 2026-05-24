# 基于大数据和 AI 的高考志愿填报系统数学建模项目

> Project 5：基于大数据和 AI 的高考志愿填报多目标推荐模型研究
>
> 模型版本：v3.1.4 | 论文：`gaokao_volunteer_modeling_paper.md`

---

## 快速运行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
# 浏览器访问 http://localhost:8501
```

**推荐演示参数**：数据模式=模拟数据 Demo | 省份=广东省 | 分数=620 | 位次=11500 | 3+1+2 | 物理/化学/生物 | 均衡型

> 当前项目为数学建模论文和系统原型 demo，数据包含模拟数据和广东样例数据，不可直接作为真实高考志愿填报依据。

---

## 1. 项目简介

本项目是 **Project 5** 的数学建模与系统原型交付工程，目标是构建一个**可计算、可解释、可交付、可接入系统的**高考志愿填报多目标推荐模型。

核心原则：**模型必须服务业务。**

模型最终服务于：
- 录取概率判断
- 专业选择
- 城市选择
- 风险控制
- 志愿组合优化
- 咨询交付
- 小程序或咨询后台接入

每个子项目均包含：**输入数据 → 数学模型 → 评价指标 → 接口输出 → 业务解释**。

---

## 2. Project 5 对齐说明

| Project 5 要求 | 本项目完成情况 |
|---------------|--------------|
| 每子项目含输入数据 | 五模型 + 七类采集器均明确输入字段 |
| 每子项目含数学模型 | 等效分/Logistic+XGBoost+Bayes+MC/TOPSIS/蒙特卡洛/整数规划 |
| 每子项目含评价指标 | MAE/AUC/Brier/约束满足率/风险识别准确率 |
| 每子项目含接口输出 | 统一 JSON 标准(pipeline.py `generate_output()`) |
| 每子项目含业务解释 | 家长端/咨询师端/系统后台三类解释 |
| 优先完成五个核心模型 | 全部具备 Python 实现 + 论文详细描述 |
| 系统接口输出 + 人工复核 | `review_required` 字段 + `--test` 模式 |

### 五个核心模型

| 编号 | 模型名称 | 对应文件 |
|------|---------|---------|
| 模型一 | 分数—位次等效换算模型 | `src/equivalent_score.py` |
| 模型二 | 院校/专业录取概率预测模型 | `src/admission_probability.py` |
| 模型三 | 冲稳保志愿组合优化模型 | `src/volunteer_optimizer.py` |
| 模型四 | 志愿填报风险评估模型 | `src/risk_assessment.py` |
| 模型五 | 专业就业景气度评价模型 | `src/career_evaluation.py` |

---

## 3. 项目文件结构

```
gaokao_volunteer_modeling_paper.md    # 数学建模论文 (19章, P5标准结构)
main.py                               # 主入口 (支持 --plan/--test/--export-json)
requirements.txt                      # Python 依赖
Project5.pdf                          # Project 5 原始要求文档

src/
├── __init__.py                       # 包初始化
├── data_generator.py                 # 模拟数据生成器 (9类数据集, 仅供测试)
├── crawler.py                        # 数据爬虫与采集框架 (7类采集器 + BaseCrawler)
├── cleaner.py                        # 数据清洗与预处理 (6级流水线 + 13项规则)
├── equivalent_score.py              # 模型一：分数—位次等效换算
├── admission_probability.py         # 模型二：录取概率预测 (LR+XGBoost+Bayes+MC)
├── career_evaluation.py             # 模型五：专业就业景气度评价 (AHP+熵权+TOPSIS)
├── volunteer_optimizer.py           # 模型三：冲稳保志愿组合优化 (贪心+局部搜索)
├── risk_assessment.py               # 模型四：风险评估 (6类风险 + 10项问题扫描)
└── pipeline.py                       # 总流水线 (8步流程 + 统一JSON输出)

archive/                              # 旧版文件归档
├── gaokao_math_modeling_paper_old.md
└── gaokao_volunteer_modeling_paper_before_p5_restructure.md

tests/                                # 测试目录 (待新增)
docs/                                 # 文档目录 (待新增)
data/                                 # 数据目录 (空, 运行时生成模拟数据)
```

---

## 4. 核心模型说明

| 模型 | 业务问题 | 主要输入 | 主要输出 | 对应文件 |
|------|---------|---------|---------|---------|
| 模型一：等效分换算 | 跨年分数不可比 | 一分一段表(近5年)、批次线 | equivalent_score, equivalent_rank, confidence_level | `src/equivalent_score.py` |
| 模型二：录取概率预测 | 量化录取可能性 | 历年投档线、专业录取位次、招生计划变化率 | admit_probability, probability_interval, recommendation_tier | `src/admission_probability.py` |
| 模型三：志愿组合优化 | 生成结构合理志愿表 | 候选志愿(U值)、考生画像(硬约束) | volunteer_list, rush_count, stable_count, safe_count, bottom_count | `src/volunteer_optimizer.py` |
| 模型四：风险评估 | 识别六类风险 | 志愿表、考生画像 | slip_risk, withdrawal_risk, adjustment_risk, cold_major_risk, employment_risk, region_risk | `src/risk_assessment.py` |
| 模型五：就业景气度评价 | 多维评价专业就业价值 | 就业率、薪资、岗位数、增长率、稳定性、考公岗位 | career_score, red_yellow_green_label, 细分评分 | `src/career_evaluation.py` |

**总目标函数**：`max U = αP_admit + βM_fit + γE_career + δC_city + ηR_family − λRisk`

支持三种方案：激进型(4:3:2:1)、均衡型(2:3:3:2)、保守型(1:2:3:4)。

---

## 5. 数据来源与采集方式

### 七类数据采集器 (src/crawler.py)

| # | 采集器类 | 采集数据 | 格式 | 来源类型 |
|---|---------|---------|------|----------|
| 1 | `SegmentTableCrawler` | 一分一段表、批次线、考生人数 | HTML/PDF/Excel | 省教育考试院 |
| 2 | `AdmissionLineCrawler` | 历年院校投档线 | HTML/PDF/Excel | 省考试院投档公告 |
| 3 | `MajorAdmissionCrawler` | 历年专业录取数据 | HTML/PDF | 高校招生网 |
| 4 | `EnrollmentPlanCrawler` | 招生计划、选科要求、学费、学制、特殊限制 | HTML/PDF/Excel | 省招办公告 |
| 5 | `MajorEmploymentCrawler` | 就业率、薪资、考研率、考公岗位、招聘岗位 | PDF/HTML/Excel | 就业质量报告/统计局/招聘网站 |
| 6 | `CityDataCrawler` | 城市产业、GDP、岗位、薪资、生活成本 | HTML/Excel | 统计局/统计年鉴 |
| 7 | `CandidateProfileCollector` | 考生画像与家庭偏好 | JSON/CSV/手动 | 前端问卷录入 |

### 重要说明

- 当前 `crawler.py` 是**公开数据采集框架**，所有采集器类均实现了完整的采集方法
- `crawl_url_list()` 方法当前返回空列表（**占位**）
- **真实部署时需补充**各省教育考试院、高校招生网、就业质量报告等真实 URL
- 无法自动采集的数据使用 `manual_import_template()` 人工导入模板
- 模拟数据由 `data_generator.py` 统一管理，不与爬虫混用

### source_trace 追溯字段

所有采集数据均携带以下 5 个追溯字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `source_url` | 数据来源 URL 或文件路径 | `http://www.hebeea.edu.cn/2024/segment.html` |
| `source_name` | 数据来源标识 | `segment_table` |
| `crawl_time` | 采集时间 | `2026-05-15 20:00:00` |
| `data_version` | 数据版本号 | `v1.0` |
| `source_type` | 来源类型 | `html` / `pdf` / `excel` / `manual` / `simulated` |

---

## 6. 爬虫合规声明

本项目 `src/crawler.py` 严格遵守以下规则：

- 只采集公开可访问的数据
- 不绕过登录、验证码、反爬限制或付费权限
- 设置默认 3 秒请求间隔 (`delay_seconds=3`)
- 支持 `robots.txt` 检查 (`check_robots()`)
- 动态网页仅在合法公开访问条件下使用 Selenium（`DynamicPageCrawler` 上下文管理器）
- 所有采集方法均包含异常处理，不会因单次失败而崩溃
- 无法自动采集的数据通过 `manual_import_template()` 降级为人工导入
- **当前项目不包含任何绕过网站安全机制的代码**

---

## 7. 模拟数据与真实数据说明

### 模拟数据的角色

- 本项目主要使用 `src/data_generator.py` 生成 9 类模拟数据集（一分一段表、院校、专业、投档线、录取数据、招生计划、就业数据、城市数据、考生画像）
- 模拟数据仅在以下场景使用：教学演示、代码测试、流程验证、pipeline 联调
- 所有模拟数据输出均标注 `"data_disclaimer": "所有数据均为模拟数据，仅供教学和测试使用"`
- `data_generator.py` 文件头注释明确标注"所有数据仅为模拟数据"

### 真实数据要求

- 模拟数据**不能替代**真实高考志愿填报数据
- 真实上线前必须使用按省份、科类、选科组合分层后的**真实公开历史数据**重新训练、回测和校准
- 录取概率、就业评分、风险等级等模型输出**仅供测试参考，不构成真实录取或就业保证**
- 模型输出的概率上限为 0.99，绝不输出 100%

---

## 8. 安装依赖

```bash
# 克隆项目后
pip install -r requirements.txt
```

主要依赖：

| 依赖 | 用途 |
|------|------|
| `numpy`, `pandas` | 数据处理 |
| `scikit-learn` | Logistic 回归、KMeans 聚类 |
| `scipy` | 统计分布 (分数拟合) |
| `xgboost` | 梯度提升模型 |
| `requests`, `beautifulsoup4`, `lxml` | HTML 爬虫 |
| `pdfplumber` | PDF 表格解析 |
| `openpyxl` | Excel 读写 |
| `selenium` | 动态网页 (可选，需 ChromeDriver) |
| `matplotlib`, `seaborn` | 可视化 (可选) |
| `pytest` | 测试框架 |

> `selenium` 为可选依赖——仅在使用 `DynamicPageCrawler` 动态网页爬虫时需要，且需要额外安装 ChromeDriver。

## 16. 广东真实数据使用

`data/` 目录包含广东省 2021-2025 年真实高考数据（19 个 CSV）：

| 数据类别 | 文件数 | 来源 |
|----------|--------|------|
| 一分一段表 | 10 个 | 广东省教育考试院官方 PDF 解析 (2022/2024/2025 真实) |
| 本科投档线 | 4 个 | 广东省考试院投档公告 PDF 解析 (2023/2024 真实, 含 954 校 5325 条) |
| 院校/城市/就业等 | 5 个 | 从投档线提取 + 统计年鉴 + 麦可思报告 |

### 使用真实数据运行

```bash
# 终端版
python run_gd.py                          # 默认 620分/11500名/均衡
python run_gd.py 650 3000 aggressive       # 自定义分数位次方案
python run_gd.py 580 35000 --export        # 导出 JSON

# 网页版 — 自动检测 data/ 下真实数据
streamlit run streamlit_app.py             # 选广东省即可使用真实院校
```

### 切换到其他省份/模拟数据

删除 `data/` 下 `gd_` 前缀文件即可回退到模拟数据模式。添加其他省份 CSV（命名格式同 `gd_` 系列）即可扩展。详细数据清单见 `docs/data_dictionary.md`。

---

## 9. 运行方式

### 基础运行

```bash
# 默认均衡方案 (分数=620, 位次=8500, 河北省 物理类)
python main.py

# 指定方案类型
python main.py --plan aggressive     # 激进型
python main.py --plan balanced       # 均衡型 (默认)
python main.py --plan conservative   # 保守型

# 指定考生参数
python main.py --score 650 --rank 3000 --province 河北省 --subject 物理类

# 显示详细输出
python main.py --verbose
```

### 导出结果

```bash
# 导出完整 JSON
python main.py --export-json result.json

# 导出 Markdown 业务报告
python main.py --export-report report.md

# 传统输出方式 (兼容旧版)
python main.py --output result.json
```

### 运行测试

```bash
# 运行内置测试案例集 (7个案例)
python main.py --test

# (待新增) pytest 测试套件
# pytest tests/ -v
```

### 运行完整流水线

```bash
# 生成模拟数据并运行 (一步到位)
python main.py --plan balanced --score 620 --rank 8500 --verbose
```

### 单独运行某个模型

```bash
# 等效分换算
python -m src.equivalent_score

# 录取概率预测
python -m src.admission_probability

# 就业景气度评价
python -m src.career_evaluation

# 志愿组合优化
python -m src.volunteer_optimizer

# 风险评估
python -m src.risk_assessment

# 爬虫示例
python -m src.crawler
```

---

## 10. 输出格式

### JSON 统一输出

```json
{
  "meta": {
    "model_version": "v3.0.0",
    "generate_time": "2026-05-15 20:00:00",
    "project": "Project 5: 基于大数据和AI的高考志愿填报多目标推荐模型",
    "data_disclaimer": "所有数据均为模拟数据，仅供教学和测试使用"
  },
  "candidate": {
    "candidate_id": "...", "province": "河北省", "score": 620, "rank": 8500,
    "equivalent_score": 619.2, "equivalent_score_interval": [610.0, 628.4],
    "equivalent_rank": 8700, "confidence_level": "high", "risk_preference": "balanced",
    "source_trace": { "source_url": "", "source_name": "", "crawl_time": "...", "data_version": "simulated_v3.0", "source_type": "simulated" }
  },
  "recommendation_plan": {
    "plan_type": "balanced",
    "volunteers": [
      {
        "volunteer_id": 1, "school_code": "10003", "school_name": "A理工大学",
        "major_code": "080901", "major_name": "计算机科学与技术",
        "admit_probability": 0.65, "probability_interval": [0.50, 0.85],
        "recommendation_tier": "safe", "fit_score": 0.50, "career_score": 0.85,
        "city_score": 0.55, "family_score": 0.50, "risk_level": "low",
        "overall_utility": 0.28, "explanation": "报考A理工大学的计算机科学与技术专业...",
        "modification_suggestion": "该志愿较稳定...", "review_required": false,
        "source_trace": { "source_url": "", ... }
      }
    ],
    "statistics": {
      "rush_count": 8, "stable_count": 12, "safe_count": 12, "bottom_count": 8,
      "overall_score": 0.25, "overall_risk_level": "medium"
    },
    "risk_assessment": {
      "overall_risk_score": 0.18, "risk_level": "medium",
      "slip_risk": { "score": 0.05, "level": "low", "trigger_reason": "", ... },
      "withdrawal_risk": { ... }, "adjustment_risk": { ... },
      "cold_major_risk": { ... }, "employment_risk": { ... }, "region_risk": { ... },
      "risk_reason": ["Q2: 稳志愿过少..."], "modification_suggestion": "...",
      "review_required": false
    }
  }
}
```

详细 JSON schema 见论文第十四章。

### Markdown 报告

运行 `python main.py --export-report report.md` 可生成包含考生信息、等效分说明、志愿表、风险提醒的中文业务报告。

---

## 11. 测试

### 内置测试 (7 案例)

```bash
python main.py --test
```

覆盖：高分冲名校、中分均衡、低分保底、不接受调剂、排斥专业、地域约束、小样本专业。

### 待新增测试套件

论文第十六章设计了 14 个测试案例，覆盖 6 个测试文件。待 `tests/` 目录新建后使用 `pytest` 运行：

```bash
pytest tests/ -v
```

---

## 12. 业务解释模板

系统为每个核心模型输出三类解释：

| 端 | 目标受众 | 语言风格 | 内容要求 |
|----|---------|---------|---------|
| 家长端 | 考生及家长 | 通俗、克制 | 录取概率 + 等级含义 + 风险提示，不承诺录取 |
| 咨询师端 | 高考志愿咨询师 | 专业、可操作 | 模型依据 + 风险触发条件 + 可调整项 |
| 系统后台 | 系统开发者 | 结构化、可追溯 | 完整参数 + 数据来源 + 可审核字段 |

### 禁止出现的表述

- "保证录取" / "一定能上" / "绝对安全"
- "录取概率 100%"
- "稳赚就业" / "未来一定热门"
- "该专业一定适合"
- "不需要复核"

---

## 13. 已知限制

1. **模拟数据**：所有测试基于 `data_generator.py` 生成的模拟数据，不反映任何真实院校、专业或考生的实际情况。
2. **爬虫占位**：七类采集器的 `crawl_url_list()` 返回空列表，真实 URL 需要人工补充。
3. **模型训练**：`admission_probability.py` 的 `fit_all()` 方法使用模拟数据训练(训练集 AUC~0.94)，不代表真实泛化能力。
4. **省份校正**：省份/科类/选科组合校正是简化实现，真实上线需做完整分层回测。
5. **就业数据**：`career_evaluation.py` 的时间序列趋势分析当前依赖模拟单年数据。
6. **测试覆盖**：`tests/` 目录尚未创建，当前仅 `main.py --test` 提供 7 个内置案例。
7. **可视化**：可视化模块 (`visualizer.py`) 尚未实现。
8. **概率限制**：模型输出的录取概率上限为 0.99，绝不出 1.0。

---

## 14. 后续扩展方向

- 真实数据接入（替换爬虫占位 URL，接入真实就业质量报告和招聘数据）
- 按省份/科类/选科组合分层训练与回测
- 城市—专业—产业匹配模型（二期增强）
- 考生兴趣与专业 NLP 语义匹配（二期增强）
- 可视化看板（位次曲线、风险热力图、雷达图）
- A/B 测试校准效用函数权重
- 咨询师管理后台与小程序前端接入
- 多年度运营数据沉淀与模型迭代

---

## 15. 版本记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.1.0 | 2025 | 初版：五个核心模型基础实现 |
| v3.0.0 | 2026-05 | Project 5 标准重构：论文 19 章结构、pipeline 统一 JSON、三类业务解释、爬虫 7 类采集器、风险 10 项问题扫描、完整 source_trace 追溯 |
