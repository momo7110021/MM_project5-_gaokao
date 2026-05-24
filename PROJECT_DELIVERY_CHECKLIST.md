# Project 5 交付验收清单

> 基于大数据和 AI 的高考志愿填报多目标推荐模型研究
>
> 模型版本：v3.0.0 | 验收日期：2026-05-15

---

## 一、交付物清单

| # | 交付物 | 文件路径 | 状态 |
|---|--------|----------|------|
| 1 | 数学建模论文 (19章 P5标准) | `gaokao_volunteer_modeling_paper.md` | done |
| 2 | 项目说明文档 | `README.md` | done |
| 3 | Python 依赖配置 | `requirements.txt` | done |
| 4 | 主入口程序 | `main.py` | done |
| 5 | 数据爬虫与采集框架 (7类) | `src/crawler.py` | done |
| 6 | 数据清洗与预处理 (13项规则) | `src/cleaner.py` | done |
| 7 | 模拟数据生成器 (9类) | `src/data_generator.py` | done |
| 8 | 模型一：分数—位次等效换算 | `src/equivalent_score.py` | done |
| 9 | 模型二：录取概率预测 | `src/admission_probability.py` | done |
| 10 | 模型三：冲稳保志愿组合优化 | `src/volunteer_optimizer.py` | done |
| 11 | 模型四：志愿填报风险评估 | `src/risk_assessment.py` | done |
| 12 | 模型五：专业就业景气度评价 | `src/career_evaluation.py` | done |
| 13 | 总流水线与统一JSON输出 | `src/pipeline.py` | done |
| 14 | 可视化模块 (8类图表) | `src/visualizer.py` | done |
| 15 | 测试用例集 (58 tests, 7文件) | `tests/` | done |
| 16 | API 接口规范 (JSON) | `docs/api_spec.json` | done |
| 17 | 业务解释模板 | `docs/business_explanations.md` | done |
| 18 | 数据字典 (9表 + source_trace) | `docs/data_dictionary.md` | done |
| 19 | 评价报告模板 | `docs/evaluation_report_template.md` | done |
| 20 | 旧版文件归档 | `archive/` | done |
| 21 | 可视化输出 (6张PNG) | `output/figures/` | done |

---

## 二、P0 必改项验收

| # | P0 项 | 验收标准 | 结果 |
|---|-------|----------|------|
| 1 | 论文合并重构 | 删除 gaokao_math_modeling_paper.md，重构 volunteer 论文为 19 章 | PASS |
| 2 | 模型编号与 P5 对齐 | 模一=等效分/模二=录取概率/模三=志愿优化/模四=风险/模五=就业 | PASS |
| 3 | 爬虫 7 类数据采集器 | BaseCrawler + 7 子类，含 source_trace、人工导入模板 | PASS |
| 4 | 统一 JSON 输出 | pipeline.generate_output() 含 20+ 逐志愿字段 + source_trace | PASS |
| 5 | 冲稳保垫阈值一致 | volunteer_optimizer + admission_probability 均用 0.20/0.45/0.70/0.88 | PASS |
| 6 | 风险评估六类 + 十问题 | 六类独立 dict(score/level/trigger) + Q1-Q10 扫描 | PASS |
| 7 | 三类业务解释 | 每模型含 parent/consultant/backend 解释 | PASS |
| 8 | 测试案例集 | 58 tests, 14 案例, 6 文件 | PASS |
| 9 | README | 15 章节完整说明 | PASS |
| 10 | 禁止违规表述 | 论文/代码/解释中不含"保证录取"等 | PASS |

---

## 三、P1 优化项验收

| # | P1 项 | 验收标准 | 结果 |
|---|-------|----------|------|
| 1 | 社媒舆情仅辅助 | sentiment_warning_score 不在 INDICATORS_CORE 中 | PASS |
| 2 | 薪资口径说明 | explain_salary_source() + 论文第十二章说明 | PASS |
| 3 | 论文第七章总模型 | 独立章节含效用函数 + 权重表 + 三类解释 | PASS |
| 4 | 第十六章测试案例 | 14 案例表 + 测试文件映射 | PASS |
| 5 | 第十八章实施路线 | 三月三阶段计划 | PASS |
| 6 | 第十九章结论 | 已撰写 | PASS |
| 7 | 附录 A-E | 爬虫伪代码 + 建模伪代码 + JSON + 字典 + 测试表 | PASS |
| 8 | requirements.txt 补充 | openpyxl / pytest / scipy 已补充 | PASS |
| 9 | main.py --test/--export | 7 个内置测试案例 + JSON/md 导出 | PASS |
| 10 | 小样本专业处理 | n<=2 → review_required + CI 扩展 | PASS |
| 11 | 省份/科类校正 | province_subject_correction() 已实现 | PASS |
| 12 | 志愿优化全量硬约束 | 排斥/选科/学费/地域/身体/单科/高风险上限 | PASS |
| 13 | 可视化模块 | 8 类图表, 6 张 PNG 生成 | PASS |
| 14 | 文档补齐 | api_spec + business_explanations + data_dictionary + eval_report | PASS |

---

## 四、验证结果

| 验证项 | 命令 | 结果 |
|--------|------|------|
| pytest 全量测试 | `pytest tests/ -q` | **58 passed**, 0 failed, 23 warnings |
| main.py 运行 | `python main.py --plan balanced` | 正常运行，输出志愿表 |
| main.py --test | `python main.py --test` | 7 内置案例全部通过 |
| JSON 合法性 | `python -m json.tool docs/api_spec.json` | **Valid** |
| 可视化自检 | `python -m src.visualizer` | 6 PNG 生成, 2 skipped(缺列) |
| 论文章节 | 19 章 + 参考 + 附录 A-E | **完整** |
| 逐志愿字段 | volunteer_item 含 20 字段 + source_trace | **完整** |
| 风险评估字段 | 六类独立 dict + 十问题扫描 | **完整** |

---

## 五、项目文件树 (最终)

```
gaokao_volunteer_modeling_paper.md    # 论文 (19章)
README.md                             # 项目说明 (15章)
main.py                               # CLI 入口
requirements.txt                      # 依赖
Project5.pdf                          # 原始要求

src/
├── __init__.py
├── pipeline.py                       # 总流水线 + 统一 JSON
├── data_generator.py                 # 模拟数据 (9类)
├── crawler.py                        # 7 类采集器 + BaseCrawler
├── cleaner.py                        # 13 项清洗规则
├── equivalent_score.py              # 模型一
├── admission_probability.py         # 模型二
├── volunteer_optimizer.py           # 模型三
├── risk_assessment.py               # 模型四
├── career_evaluation.py             # 模型五
└── visualizer.py                     # 8 类图表

tests/
├── conftest.py                       # fixtures
├── test_equivalent_score.py         # 8 tests
├── test_admission_probability.py    # 13 tests
├── test_career_evaluation.py        # 10 tests
├── test_volunteer_optimizer.py      # 8 tests
├── test_risk_assessment.py          # 8 tests
└── test_pipeline.py                 # 11 tests

docs/
├── api_spec.json                     # API 接口规范
├── business_explanations.md          # 三类解释模板
├── data_dictionary.md                # 9 表数据字典
└── evaluation_report_template.md     # 11 章评价报告模板

archive/
├── gaokao_math_modeling_paper_old.md
└── gaokao_volunteer_modeling_paper_before_p5_restructure.md

output/figures/                       # 6 张可视化 PNG
data/                                 # (空，运行时生成模拟数据)
```

---

## 六、已知限制与后续建议

| 类别 | 限制 | 建议 |
|------|------|------|
| 数据 | crawler.py URL 列表为占位，模拟数据不能替代真实数据 | 补充各省教育考试院真实公开 URL |
| 训练 | admission_probability 用模拟数据训练(AUC~0.94 虚高) | 真实分层数据训练+交叉验证 |
| 就业 | career_evaluation 趋势分析缺多年数据 | 接入 3 年以上就业质量报告 |
| 分析 | visualizer 无真实录取标签，校准图不可用 | 收集真实录取结果后填入 |
| 部署 | 无前端/后端服务 | 接入小程序或咨询后台 API |
| 算法 | 权重需 A/B 测试校准 | 咨询师内测 + 人工复核反馈 |

---

## 七、交付结论

**Project 5 核心交付已闭环。**

- P0 必改项：10/10 通过
- P1 优化项：14/14 通过
- 测试：58/58 passed
- 论文：19 章完整结构
- 代码：10 个 src 模块 + 7 tests 文件
- 文档：README + 4 docs
- 可视化：6 张 PNG

**可交付。**
