"""
系统总流水线：串联所有模块，完成从数据输入到志愿方案输出的完整流程。

Project 5 统一输出主流程。
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

from .equivalent_score import EquivalentScoreConverter
from .admission_probability import AdmissionProbabilityPredictor
from .career_evaluation import CareerEvaluator
from .volunteer_optimizer import VolunteerOptimizer
from .risk_assessment import RiskAssessor


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class GaoKaoPipeline:
    """高考志愿填报系统总流水线 (Project 5 v3.0.0)"""

    MODEL_VERSION = "v3.0.0"

    def __init__(self):
        self.eq_converter = EquivalentScoreConverter()
        self.prob_predictor = AdmissionProbabilityPredictor()
        self.career_evaluator = CareerEvaluator()
        self.volunteer_optimizer = VolunteerOptimizer(n_max=40)
        self.risk_assessor = RiskAssessor()
        self.data = {}

    def load_data(self, data_dict):
        """加载数据"""
        self.data = data_dict

    # ============================================================
    # 八步核心流程
    # ============================================================

    def step1_equivalent_score(self, profile, target_years=None):
        """步骤1: 分数-位次等效换算"""
        segment_by_year = {}
        batch_lines = {}

        seg_table = self.data.get("segment_table")
        if seg_table is None:
            raise ValueError("缺少一分一段表数据")

        current_year = profile.get("year", 2024)
        segments = seg_table.groupby("year")

        for year, group in segments:
            segment_by_year[year] = group
            bl = group["batch_line"].iloc[0] if "batch_line" in group.columns else 0
            batch_lines[year] = bl

        if target_years is None:
            target_years = [y for y in segment_by_year.keys() if y != current_year]

        result = self.eq_converter.convert(
            segment_dfs=segment_by_year,
            candidate_score=profile["score"],
            candidate_rank=profile["rank"],
            current_year=current_year,
            target_years=target_years,
            batch_lines=batch_lines,
        )
        self.data["equivalent_result"] = result
        return result

    def step2_build_candidates(self, profile, score_range=30):
        """步骤2: 构建候选志愿集合"""
        eq_result = self.data.get("equivalent_result")
        if eq_result is None:
            raise ValueError("请先执行 step1_equivalent_score")

        eq_score = eq_result["equivalent_score"]
        # 等效分异常时回退到原始分数
        if eq_score < 100 or eq_score > 900:
            eq_score = profile.get("score", eq_score)
        plan_df = self.data.get("admission_plan")
        school_info_df = self.data.get("school_info")

        if plan_df is None:
            raise ValueError("缺少招生计划数据")

        school_line_df = self.data.get("school_admission_line")
        if school_line_df is not None:
            recent = school_line_df[school_line_df["year"] == school_line_df["year"].max()].copy()
            recent["min_admission_score"] = pd.to_numeric(recent["min_admission_score"], errors="coerce")

            # 主窗口: ±score_range 分
            candidate_diff = score_range
            candidates_main = recent[
                (recent["min_admission_score"] >= eq_score - candidate_diff) &
                (recent["min_admission_score"] <= eq_score + candidate_diff)
            ].merge(plan_df, on=["school_code", "province", "year"], how="inner", suffixes=("", "_plan"))

            # 扩展底池: -100分以下学校作为保/垫候选
            candidates_bottom = recent[
                (recent["min_admission_score"] >= eq_score - 100) &
                (recent["min_admission_score"] < eq_score - candidate_diff)
            ].merge(plan_df, on=["school_code", "province", "year"], how="inner", suffixes=("", "_plan"))

            # 扩展冲池: +100分以上学校作为冲候选
            candidates_rush = recent[
                (recent["min_admission_score"] > eq_score + candidate_diff) &
                (recent["min_admission_score"] <= eq_score + 100)
            ].merge(plan_df, on=["school_code", "province", "year"], how="inner", suffixes=("", "_plan"))

            candidates = candidates_main.copy()
            if not candidates_bottom.empty:
                candidates = pd.concat([candidates, candidates_bottom], ignore_index=True)
            if not candidates_rush.empty:
                candidates = pd.concat([candidates, candidates_rush], ignore_index=True)

            if len(candidates) == 0:
                candidates = recent[
                    (recent["min_admission_score"] >= eq_score - 60) &
                    (recent["min_admission_score"] <= eq_score + 60)
                ].merge(plan_df, on=["school_code", "province", "year"], how="inner", suffixes=("", "_plan"))

            if len(candidates) == 0:
                candidates = plan_df.copy()
        else:
            candidates = plan_df.copy()

        if school_info_df is not None and "school_code" in candidates.columns:
            # 只取 school_info 中 candidates 缺失的列，避免 school_name/city 列名碰撞
            sch_cols = ["school_code"]
            for c in ["school_name", "city"]:
                if c not in candidates.columns and c in school_info_df.columns:
                    sch_cols.append(c)
            if len(sch_cols) > 1:
                candidates = candidates.merge(
                    school_info_df[sch_cols], on="school_code", how="left")
            for c in ["school_name", "city"]:
                if c in candidates.columns:
                    candidates[c] = candidates[c].fillna("")

        candidates["P_admit"] = 0.5
        candidates["M_fit"] = 0.5
        candidates["E_career"] = 0.5
        candidates["C_city"] = 0.5
        candidates["R_family"] = 0.5
        candidates["Risk"] = 0.1
        candidates["review_required"] = False
        candidates["explanation"] = ""
        candidates["modification_suggestion"] = ""
        self.data["candidates"] = candidates
        return candidates

    def step3_admission_probability(self, profile):
        """步骤3: 计算每个候选志愿的录取概率"""
        candidates = self.data.get("candidates")
        if candidates is None:
            raise ValueError("请先执行 step2_build_candidates")

        major_admission = self.data.get("major_admission")
        prob_results = []

        for idx, row in candidates.iterrows():
            school_code = row.get("school_code", "")
            major_code = row.get("major_code", "")

            if major_admission is not None and school_code and major_code:
                hist = major_admission[
                    (major_admission["school_code"] == school_code) &
                    (major_admission["major_code"] == major_code) &
                    (major_admission["province"] == profile.get("province"))
                ].sort_values("year")

                if len(hist) > 0:
                    pred = self.prob_predictor.predict(
                        candidate_rank=profile["rank"],
                        historical_records=hist,
                    )
                else:
                    score_gap = row.get("min_admission_score", profile["score"]) - profile["score"]
                    p_est = self._estimate_from_score_gap(score_gap)
                    pred = {
                        "admit_probability": p_est,
                        "probability_interval": [max(0, p_est - 0.1), min(1, p_est + 0.1)],
                        "recommendation_tier": self.prob_predictor._classify_tier(p_est),
                        "top_features": [],
                        "uncertainty_level": "high",
                        "review_required": True,
                        "n_years_data": 0,
                        "component_probabilities": {},
                    }
            else:
                pred = {
                    "admit_probability": 0.5,
                    "probability_interval": [0.3, 0.7],
                    "recommendation_tier": "stable",
                    "top_features": [],
                    "uncertainty_level": "high",
                    "review_required": True,
                    "n_years_data": 0,
                    "component_probabilities": {},
                }

            prob_results.append(pred)

        candidates["P_admit"] = [r["admit_probability"] for r in prob_results]
        candidates["prob_lower"] = [r["probability_interval"][0] for r in prob_results]
        candidates["prob_upper"] = [r["probability_interval"][1] for r in prob_results]
        candidates["recommendation_tier"] = [r["recommendation_tier"] for r in prob_results]
        candidates["uncertainty_level"] = [r["uncertainty_level"] for r in prob_results]
        candidates["review_required"] = [r["review_required"] for r in prob_results]
        candidates["n_years_data"] = [r.get("n_years_data", 0) for r in prob_results]
        self.data["prob_results"] = prob_results
        return candidates

    def step4_career_evaluation(self):
        """步骤4: 专业就业景气度评价"""
        employment_df = self.data.get("major_employment")
        if employment_df is None:
            return None

        latest_year = employment_df["data_year"].max()
        latest_emp = employment_df[employment_df["data_year"] == latest_year]
        eval_result = self.career_evaluator.evaluate(latest_emp)
        self.data["career_eval"] = eval_result

        candidates = self.data.get("candidates")
        if candidates is not None and "major_code" in candidates.columns:
            career_scores = eval_result.set_index("major_code")["career_score"]
            candidates["E_career"] = candidates["major_code"].map(career_scores).fillna(0.5)
            if "red_yellow_green_label" in eval_result.columns:
                labels = eval_result.set_index("major_code")["red_yellow_green_label"]
                candidates["career_label"] = candidates["major_code"].map(labels).fillna("yellow")

        return eval_result

    def step5_city_evaluation(self, profile):
        """步骤5: 城市价值评估"""
        city_data = self.data.get("city_data")
        candidates = self.data.get("candidates")
        if city_data is None or candidates is None:
            return

        city_scores = {}
        for _, row in city_data.iterrows():
            city = row["city"]
            score = (
                0.25 * (row["gdp"] / city_data["gdp"].max()) +
                0.20 * (row["gdp_per_capita"] / city_data["gdp_per_capita"].max()) +
                0.15 * row["tertiary_industry_ratio"] +
                0.15 * (row["high_tech_company_count"] / max(city_data["high_tech_company_count"].max(), 1)) +
                0.10 * (row["average_salary"] / city_data["average_salary"].max()) -
                0.10 * (row["living_cost"] / city_data["living_cost"].max()) +
                0.05 * (row["transport_score"] / 10)
            )
            city_scores[city] = max(0, min(1, score))

        if "city" in candidates.columns:
            candidates["C_city"] = candidates["city"].map(city_scores).fillna(0.5)
        else:
            candidates["C_city"] = 0.5

    def step6_fit_and_family(self, profile):
        """步骤6: 计算专业匹配度和家庭资源匹配度"""
        candidates = self.data.get("candidates")
        if candidates is None:
            return

        interests = profile.get("interest_direction", [])
        excluded = profile.get("excluded_majors", [])
        preferred_cities = profile.get("preferred_cities", [])

        for idx, row in candidates.iterrows():
            m_fit = 0.5
            if "major_name" in row and interests:
                m_name = str(row["major_name"])
                kw_match = sum(1 for kw in interests if kw in m_name)
                m_fit = 0.3 + 0.7 * (kw_match / max(len(interests), 1))
            candidates.at[idx, "M_fit"] = max(0, min(1, m_fit))

        candidates["R_family"] = 0.5

        if "city" in candidates.columns and preferred_cities:
            candidates["R_family"] = (
                0.4 * candidates["city"].isin(preferred_cities).astype(float) + 0.2
            ) * 0.6 + 0.3

        if excluded and "major_code" in candidates.columns:
            mask = candidates["major_code"].isin(excluded)
            candidates = candidates[~mask]

        self.data["candidates"] = candidates

    def step7_optimize_volunteers(self, profile, plan_type="balanced"):
        """步骤7: 志愿组合优化"""
        candidates = self.data.get("candidates")
        if candidates is None:
            raise ValueError("缺少候选志愿数据")

        result = self.volunteer_optimizer.optimize(candidates, profile, plan_type)
        self.data["optimize_result"] = result
        self.data["plan_type"] = plan_type
        return result

    def step8_risk_assessment(self, profile):
        """步骤8: 风险评估"""
        optimize_result = self.data.get("optimize_result")
        if optimize_result is None:
            raise ValueError("请先执行 step7_optimize_volunteers")

        volunteer_table = optimize_result["volunteer_table"]
        risk_result = self.risk_assessor.assess(volunteer_table, profile)
        optimize_result["risk_assessment"] = risk_result
        self.data["optimize_result"] = optimize_result
        return risk_result

    # ============================================================
    # Project 5 统一输出构建方法
    # ============================================================

    def _add_source_trace(self, table_name=None):
        """构建 source_trace 字段，兼容模拟数据场景"""
        trace = {
            "source_url": "",
            "source_name": "",
            "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_version": "simulated_v3.0",
            "source_type": "simulated" if not table_name else "public",
        }
        if table_name and table_name in self.data:
            df = self.data[table_name]
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                if "source_url" in df.columns:
                    urls = df["source_url"].dropna().unique()
                    if len(urls) > 0:
                        trace["source_url"] = str(urls[0])
                if hasattr(df, "attrs") and "source_name" in df.attrs:
                    trace["source_name"] = df.attrs["source_name"]
        return trace

    def _build_candidate_block(self, profile, plan_type):
        """构建考生信息块"""
        eq_result = self.data.get("equivalent_result", {})
        block = {
            "candidate_id": str(profile.get("candidate_id", "")),
            "province": str(profile.get("province", "")),
            "year": int(profile.get("year", 2024)),
            "subject_type": str(profile.get("subject_type", "")),
            "score": int(profile.get("score", 0)),
            "rank": int(profile.get("rank", 0)),
            "equivalent_scores": eq_result.get("year_details", {}),
            "equivalent_score": eq_result.get("equivalent_score"),
            "equivalent_score_interval": eq_result.get("equivalent_score_interval"),
            "equivalent_rank": eq_result.get("equivalent_rank"),
            "confidence_level": eq_result.get("confidence_level", ""),
            "abnormal_year_warning": eq_result.get("abnormal_year_warning", False),
            "fallback_rule": eq_result.get("fallback_rule", ""),
            "risk_preference": str(plan_type),
            "source_trace": self._add_source_trace("segment_table"),
        }
        return block

    def _build_risk_for_volunteer(self, row, overall_risk):
        """为单个志愿构建风险信息（基于综合风险推断）"""
        p = row.get("P_admit", 0.5)
        tier = row.get("recommendation_tier", "stable")
        review = row.get("review_required", False)

        if tier == "rush":
            risk_level = "high"
            risk_reason = "冲刺志愿，录取概率较低"
        elif tier == "stable":
            risk_level = "medium"
            risk_reason = "稳妥志愿"
        elif tier == "bottom":
            risk_level = "low"
            risk_reason = "垫底志愿，几乎确保录取"
        else:
            risk_level = "low"
            risk_reason = "保底志愿"

        if review:
            risk_reason += "；数据不足需复核"

        return {"risk_level": risk_level, "risk_reason": risk_reason}

    def _build_default_explanation(self, row):
        """为单个志愿生成默认解释（兼容模型未实现三类解释的情况）"""
        p = row.get("P_admit", 0.5)
        tier = row.get("recommendation_tier", "stable")
        school = row.get("school_name", "该院校")
        major = row.get("major_name", "该专业")

        tier_names = {
            "rush": "冲（录取有一定难度，值得尝试）",
            "stable": "稳（有较大希望被录取）",
            "safe": "保（基本能够被录取）",
            "bottom": "垫（几乎确保录取，为兜底志愿）",
        }

        return (
            f"报考{school}的{major}专业，预估录取概率约{int(p*100)}%，"
            f"属于{tier_names.get(tier, tier)}。"
            f"建议结合个人兴趣和职业规划综合考虑。"
        )

    def _build_default_suggestion(self, row):
        """为单个志愿生成默认修改建议"""
        tier = row.get("recommendation_tier", "stable")
        review = row.get("review_required", False)

        suggestions_map = {
            "rush": "该志愿录取难度较大，建议仅作为冲刺志愿，同时确保有足够的保底志愿。",
            "stable": "该志愿可保留，建议关注该专业近年录取位次波动趋势。",
            "safe": "该志愿较稳定，可考虑作为主要选择。",
            "bottom": "该志愿兜底效果好，确保录取安全。",
        }
        suggestion = suggestions_map.get(tier, "")
        if review:
            suggestion += "该志愿数据年份较少，建议咨询师人工复核。"
        return suggestion

    def _build_volunteer_item(self, row, volunteer_id, risk_info):
        """构建单个志愿的完整输出项（Project 5 标准字段）"""
        item = {
            "volunteer_id": volunteer_id,
            "school_code": str(row.get("school_code", "")),
            "school_name": str(row.get("school_name", "")),
            "major_group_code": str(row.get("major_group_code", "")),
            "major_code": str(row.get("major_code", "")),
            "major_name": str(row.get("major_name", "")),
            "admit_probability": self._safe_float(row.get("P_admit", 0), 4),
            "probability_interval": [
                self._safe_float(row.get("prob_lower", 0), 4),
                self._safe_float(row.get("prob_upper", 1), 4),
            ],
            "recommendation_tier": str(row.get("recommendation_tier", "")),
            "fit_score": self._safe_float(row.get("M_fit", 0), 2),
            "career_score": self._safe_float(row.get("E_career", 0), 2),
            "city_score": self._safe_float(row.get("C_city", 0), 2),
            "family_score": self._safe_float(row.get("R_family", 0), 2),
            "risk_level": risk_info["risk_level"],
            "risk_reason": risk_info["risk_reason"],
            "overall_utility": self._safe_float(row.get("U", 0), 4),
            "explanation": str(row.get("explanation", "")) or self._build_default_explanation(row),
            "modification_suggestion": str(row.get("modification_suggestion", "")) or self._build_default_suggestion(row),
            "review_required": bool(row.get("review_required", False)),
            "source_trace": self._add_source_trace("admission_plan"),
        }
        return item

    def _build_risk_block(self, risk_result):
        """构建风险评估块，兼容新旧两种 risk_assessment 输出格式"""
        block = {
            "overall_risk_score": self._safe_float(risk_result.get("overall_risk_score", 0), 4),
            "risk_level": str(risk_result.get("risk_level", "low")),
            "risk_reason": [str(r) for r in risk_result.get("risk_reason", [])],
            "modification_suggestion": str(risk_result.get("modification_suggestion", "")),
            "review_required": bool(risk_result.get("review_required", False)),
        }

        # 六类分项风险:兼容旧格式(float)与新格式(dict)
        for risk_key in ["slip_risk", "withdrawal_risk", "adjustment_risk",
                          "cold_major_risk", "employment_risk", "region_risk"]:
            raw = risk_result.get(risk_key, {})
            if isinstance(raw, dict):
                block[risk_key] = {
                    "score": self._safe_float(raw.get("score", 0), 4),
                    "level": str(raw.get("level", "low")),
                    "trigger_reason": str(raw.get("trigger_reason", "")),
                    "modification_suggestion": str(raw.get("modification_suggestion", "")),
                    "review_required": bool(raw.get("review_required", False)),
                }
            else:
                block[risk_key] = {
                    "score": self._safe_float(raw, 4),
                    "level": "high" if self._safe_float(raw, 4) > 0.3 else "low",
                    "trigger_reason": "",
                    "modification_suggestion": "",
                    "review_required": self._safe_float(raw, 4) > 0.3,
                }

        return block

    def generate_output(self, profile, plan_type="balanced"):
        """
        生成统一 JSON 输出 (Project 5 标准)

        Returns:
            str: 缩进格式的 JSON 字符串
        """
        eq_result = self.data.get("equivalent_result", {})
        opt_result = self.data.get("optimize_result", {})
        risk_result = opt_result.get("risk_assessment", {})
        volunteer_table = opt_result.get("volunteer_table", pd.DataFrame())
        career_eval = self.data.get("career_eval")

        # 构建志愿列表
        volunteers_json = []
        if not volunteer_table.empty:
            for vol_id, (_, row) in enumerate(volunteer_table.iterrows(), start=1):
                risk_info = self._build_risk_for_volunteer(row, risk_result)
                item = self._build_volunteer_item(row, vol_id, risk_info)
                volunteers_json.append(item)

        # 构建考生信息块
        candidate_block = self._build_candidate_block(profile, plan_type)

        # 构建风险评估块
        risk_block = self._build_risk_block(risk_result)

        output = {
            "meta": {
                "model_version": self.MODEL_VERSION,
                "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "project": "Project 5: 基于大数据和AI的高考志愿填报多目标推荐模型",
                "data_disclaimer": "所有数据均为模拟数据，仅供教学和测试使用",
            },
            "candidate": candidate_block,
            "recommendation_plan": {
                "plan_type": str(plan_type),
                "volunteers": volunteers_json,
                "statistics": {
                    "rush_count": int(opt_result.get("rush_count", 0)),
                    "stable_count": int(opt_result.get("stable_count", 0)),
                    "safe_count": int(opt_result.get("safe_count", 0)),
                    "bottom_count": int(opt_result.get("bottom_count", 0)),
                    "overall_score": self._safe_float(opt_result.get("overall_score", 0), 4),
                    "overall_risk_level": str(risk_result.get("risk_level", "")),
                },
                "risk_assessment": risk_block,
            },
        }

        return json.dumps(output, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def generate_evaluation_report(self, profile, plan_type="balanced"):
        """
        生成模型评价报告 (Markdown 格式)

        Returns:
            str: Markdown 格式的评价报告
        """
        eq_result = self.data.get("equivalent_result", {})
        opt_result = self.data.get("optimize_result", {})
        risk_result = opt_result.get("risk_assessment", {})

        lines = [
            "# 高考志愿填报模型评价报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**模型版本**: {self.MODEL_VERSION}",
            "",
            "## 1. 考生信息",
            "",
            f"- 考生ID: {profile.get('candidate_id', 'N/A')}",
            f"- 省份: {profile.get('province', 'N/A')}",
            f"- 科类: {profile.get('subject_type', 'N/A')}",
            f"- 分数: {profile.get('score', 'N/A')}",
            f"- 位次: {profile.get('rank', 'N/A')}",
            f"- 方案类型: {plan_type}",
            "",
            "## 2. 等效分换算评价",
            "",
            f"- 加权等效分: {eq_result.get('equivalent_score', 'N/A')}",
            f"- 等效分区间: {eq_result.get('equivalent_score_interval', 'N/A')}",
            f"- 等效位次: {eq_result.get('equivalent_rank', 'N/A')}",
            f"- 置信度: {eq_result.get('confidence_level', 'N/A')}",
            f"- 异常年份警告: {eq_result.get('abnormal_year_warning', False)}",
            f"- Fallback规则: {eq_result.get('fallback_rule', 'N/A')}",
            f"- 需复核: {eq_result.get('review_required', False)}",
            "",
            "## 3. 录取概率预测评价",
            "",
            "| 指标 | 目标值 | 当前状态 |",
            "|------|--------|---------|",
            "| AUC | >0.85 | 模拟数据(需真实数据校准) |",
            "| Brier Score | <0.15 | 模拟数据(需真实数据校准) |",
            "",
            "## 4. 志愿表统计",
            "",
            f"- 冲: {opt_result.get('rush_count', 0)} 个",
            f"- 稳: {opt_result.get('stable_count', 0)} 个",
            f"- 保: {opt_result.get('safe_count', 0)} 个",
            f"- 垫: {opt_result.get('bottom_count', 0)} 个",
            f"- 综合评分: {opt_result.get('overall_score', 'N/A')}",
            "",
            "## 5. 风险评估",
            "",
            f"- 综合风险评分: {risk_result.get('overall_risk_score', 'N/A')}",
            f"- 风险等级: {risk_result.get('risk_level', 'N/A')}",
            f"- 修改建议: {risk_result.get('modification_suggestion', '无')}",
            f"- 需人工复核: {risk_result.get('review_required', False)}",
            "",
            "## 6. 评价结论",
            "",
            "当前模型基于模拟数据运行，所有概率估计和评分仅供参考。",
            "正式上线前必须使用真实省份/科类/选科组合分层数据重新训练和校准。",
            "",
            "---",
            "*此报告由系统自动生成，数据为模拟数据。*",
        ]
        return "\n".join(lines)

    def generate_business_report(self, profile, plan_type="balanced"):
        """
        生成面向业务的 Markdown 报告（家长/咨询师可读）

        Returns:
            str: Markdown 格式的业务报告
        """
        eq_result = self.data.get("equivalent_result", {})
        opt_result = self.data.get("optimize_result", {})
        risk_result = opt_result.get("risk_assessment", {})
        volunteer_table = opt_result.get("volunteer_table", pd.DataFrame())

        lines = [
            "# 高考志愿填报推荐报告",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 方案类型: **{plan_type}**",
            "",
            "## 一、考生基本信息",
            "",
            f"**省份**: {profile.get('province', 'N/A')}  "
            f"**科类**: {profile.get('subject_type', 'N/A')}  "
            f"**分数**: {profile.get('score', 'N/A')}  "
            f"**位次**: {profile.get('rank', 'N/A')}",
            "",
            "## 二、等效分说明",
            "",
        ]
        if self.eq_converter and eq_result:
            lines.append(self.eq_converter.generate_parent_explanation(eq_result) if eq_result else "等效分数据不完整。")
        lines.append("")

        lines.extend([
            "## 三、志愿表",
            "",
            "| 序号 | 院校 | 专业 | 概率 | 等级 | 就业 | 风险 |",
            "|------|------|------|------|------|------|------|",
        ])

        if not volunteer_table.empty:
            for vol_id, (_, row) in enumerate(volunteer_table.iterrows(), start=1):
                risk = self._build_risk_for_volunteer(row, risk_result)
                lines.append(
                    f"| {vol_id} | {row.get('school_name', '')} | {row.get('major_name', '')} | "
                    f"{int(row.get('P_admit', 0)*100)}% | {row.get('recommendation_tier', '')} | "
                    f"{self._safe_float(row.get('E_career', 0), 2)} | {risk['risk_level']} |"
                )
        else:
            lines.append("| - | 无候选志愿 | - | - | - | - | - |")

        lines.extend([
            "",
            f"**统计**: 冲{opt_result.get('rush_count', 0)}个 "
            f"稳{opt_result.get('stable_count', 0)}个 "
            f"保{opt_result.get('safe_count', 0)}个 "
            f"垫{opt_result.get('bottom_count', 0)}个",
            "",
            "## 四、风险提醒",
            "",
        ])

        risk_reasons = risk_result.get("risk_reason", [])
        if risk_reasons:
            for r in risk_reasons:
                lines.append(f"- {r}")
        else:
            lines.append("- 未检测到明显风险。")

        lines.extend([
            "",
            f"**综合风险等级**: {risk_result.get('risk_level', 'N/A')}  "
            f"**需人工复核**: {'是' if risk_result.get('review_required', False) else '否'}",
            "",
            f"**修改建议**: {risk_result.get('modification_suggestion', '无需修改')}",
            "",
            "---",
            "*本报告由系统基于模拟数据自动生成，不构成任何录取保证。*",
        ])

        return "\n".join(lines)

    def export_json(self, profile, plan_type="balanced", filepath="output.json"):
        """导出 JSON 到文件"""
        json_output = self.generate_output(profile, plan_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_output)
        return filepath

    def export_markdown_report(self, profile, plan_type="balanced",
                                filepath="report.md", report_type="business"):
        """导出 Markdown 报告到文件"""
        if report_type == "evaluation":
            report = self.generate_evaluation_report(profile, plan_type)
        else:
            report = self.generate_business_report(profile, plan_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        return filepath

    def generate_markdown_report(self, profile=None, plan_type="balanced"):
        """便捷方法: 返回 markdown 报告字符串（供 main.py 调用）"""
        if profile is None:
            profile = {}
        return self.generate_business_report(profile, plan_type)

    def run(self, profile, plan_type="balanced", target_years=None):
        """
        一键运行完整流程

        Args:
            profile: dict, 考生画像
            plan_type: str, "aggressive" / "balanced" / "conservative"
            target_years: list, 目标年份（默认从数据自动推断）

        Returns:
            str: JSON 格式的完整推荐结果
        """
        print("=" * 60)
        print(f"  高考志愿填报系统 {self.MODEL_VERSION} - {plan_type}方案")
        print("=" * 60)

        print("[1/8] 分数-位次等效换算...")
        self.step1_equivalent_score(profile, target_years)

        print("[2/8] 构建候选志愿集合...")
        self.step2_build_candidates(profile)

        print("[3/8] 计算录取概率...")
        self.step3_admission_probability(profile)

        print("[4/8] 专业就业景气度评价...")
        self.step4_career_evaluation()

        print("[5/8] 城市价值评估...")
        self.step5_city_evaluation(profile)

        print("[6/8] 专业匹配与家庭资源匹配度...")
        self.step6_fit_and_family(profile)

        print("[7/8] 志愿组合优化...")
        opt_result = self.step7_optimize_volunteers(profile, plan_type)
        print(f"  -> 冲{opt_result['rush_count']} 稳{opt_result['stable_count']} "
              f"保{opt_result['safe_count']} 垫{opt_result['bottom_count']}")

        print("[8/8] 风险评估...")
        risk_result = self.step8_risk_assessment(profile)
        print(f"  -> 风险等级: {risk_result['risk_level']}")

        json_output = self.generate_output(profile, plan_type)

        print("\n" + "=" * 60)
        print("  流程完成")
        print("=" * 60)

        return json_output

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _safe_float(val, precision=4):
        """安全转换为指定位数的浮点数"""
        try:
            return float(round(float(val), precision))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _estimate_from_score_gap(score_gap):
        """从分差粗略估计录取概率"""
        if score_gap > 30:
            return 0.90
        elif score_gap > 15:
            return 0.70 + (score_gap - 15) / 15 * 0.20
        elif score_gap > 0:
            return 0.35 + score_gap / 15 * 0.35
        elif score_gap > -15:
            return 0.10 + (score_gap + 15) / 15 * 0.25
        elif score_gap > -30:
            return 0.02 + (score_gap + 30) / 15 * 0.08
        else:
            return 0.01
