"""
核心模型五：专业就业景气度评价模型 (Project 5 编号)

功能：从多维度评价本科专业的就业景气度。
方法：AHP 层次分析法 + 熵权法 + TOPSIS 综合评价 + KMeans聚类 + 趋势分析

社媒舆情(sentiment_warning_score)仅作为辅助预警，不纳入核心 TOPSIS 计算。
薪资数据口径：综合高校就业质量报告、国家统计局分行业工资、招聘网站公开数据。
当前为模拟数据版本，仅用于教学、测试和流程验证；
真实上线前必须使用高校就业质量报告、招聘岗位数据、统计年鉴
和人工校验后的真实数据重新计算、回测和校准。
"""

import numpy as np
import pandas as pd


class CareerEvaluator:
    """专业就业景气度评价器 (Project 5 核心模型五)"""

    # 核心指标(参与TOPSIS)
    INDICATORS_CORE = [
        "employment_rate", "postgraduate_rate",
        "average_salary", "median_salary",
        "job_count", "job_growth_rate",
        "industry_growth_score", "stability_score",
        "civil_service_post_count",
    ]
    # 辅助预警指标(不参与TOPSIS,单独用于风险提示)
    INDICATOR_AUX = "sentiment_warning_score"

    POSITIVE_INDICATORS = INDICATORS_CORE
    NEGATIVE_INDICATORS = []

    ahp_weight_scheme_maps = {
        "employment_rate": 0.13, "postgraduate_rate": 0.07,
        "average_salary": 0.20, "median_salary": 0.15,
        "job_count": 0.13, "job_growth_rate": 0.10,
        "industry_growth_score": 0.10, "stability_score": 0.05,
        "civil_service_post_count": 0.07,
    }

    def __init__(self):
        self.w_comb = None
        self.v_plus = None
        self.v_minus = None
        self._meta = {
            "is_simulated": True,
            "note": "当前为模拟数据版本，仅用于教学、测试和流程验证。"
                    "真实上线前必须使用高校就业质量报告、招聘岗位数据、"
                    "统计年鉴和人工校验后的真实数据重新计算、回测和校准。",
        }

    # ================================================================
    # 数据预处理与归一化
    # ================================================================

    def normalize(self, df):
        """Min-Max 归一化（正向指标；舆情预警单独处理不参与归一化）"""
        df_norm = df.copy()
        for col in self.POSITIVE_INDICATORS:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                denom = col_max - col_min
                if denom > 1e-10:
                    df_norm[col] = (df[col] - col_min) / denom
                else:
                    df_norm[col] = 0.5
        for col in self.NEGATIVE_INDICATORS:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                denom = col_max - col_min
                if denom > 1e-10:
                    df_norm[col] = (col_max - df[col]) / denom
                else:
                    df_norm[col] = 0.5
        return df_norm

    # ================================================================
    # AHP + 熵权法 + TOPSIS
    # ================================================================

    def compute_entropy_weights(self, df_norm):
        """熵权法计算客观权重"""
        available = [c for c in self.INDICATORS_CORE if c in df_norm.columns]
        n = len(df_norm)
        p_ij = df_norm[available] / (df_norm[available].sum() + 1e-10)
        k = 1.0 / np.log(max(n, 2))
        e_j = -k * np.sum(p_ij * np.log(p_ij + 1e-10), axis=0)
        d_j = 1 - e_j
        return d_j / d_j.sum()

    def compute_combined_weights(self, df_norm):
        """计算组合权重（AHP × 熵权）"""
        available = [c for c in self.INDICATORS_CORE if c in df_norm.columns]
        w_ahp = np.array([
            self.ahp_weight_scheme_maps.get(c, 0.05) for c in available
        ])
        w_ahp = w_ahp / w_ahp.sum()
        w_entropy = self.compute_entropy_weights(df_norm)
        w_combined = w_ahp * w_entropy.values
        w_combined = w_combined / w_combined.sum()
        self.w_comb = dict(zip(available, w_combined))
        return w_combined

    def topsis_evaluate(self, df_norm):
        """TOPSIS 综合评价"""
        available = [c for c in self.INDICATORS_CORE if c in df_norm.columns]
        w = self.compute_combined_weights(df_norm)
        v_ij = df_norm[available].values * w
        self.v_plus = v_ij.max(axis=0)
        self.v_minus = v_ij.min(axis=0)
        d_plus = np.sqrt(np.sum((v_ij - self.v_plus) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((v_ij - self.v_minus) ** 2, axis=1))
        denom = d_plus + d_minus
        c_i = np.where(denom > 0, d_minus / denom, 0.5)
        return c_i

    # ================================================================
    # 标签生成
    # ================================================================

    def generate_labels(self, scores):
        """基于三分类聚类生成红黄绿标签"""
        if len(scores) < 3:
            return np.array(["green"] * len(scores))
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = kmeans.fit_predict(scores.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        center_order = np.argsort(centers)
        label_map = {
            center_order[0]: "red",
            center_order[1]: "yellow",
            center_order[2]: "green",
        }
        return np.array([label_map[l] for l in labels])

    def _employment_level(self, c_i):
        """基于 career_score 分档"""
        if c_i < 0.33:
            return "low"
        elif c_i < 0.66:
            return "medium"
        return "high"

    def _major_risk_label(self, c_i):
        """专业风险标签"""
        if c_i < 0.33:
            return "high_risk"
        elif c_i < 0.50:
            return "medium_risk"
        elif c_i < 0.66:
            return "low_risk"
        return "safe"

    def _detailed_risk_type(self, row, c_i):
        """进一步细分风险类型"""
        risks = []
        if row.get("employment_rate", 0) < 0.75:
            risks.append("low_employment_risk")
        if row.get("average_salary", 0) < 4500:
            risks.append("low_salary_risk")
        if row.get("job_growth_rate", 0) < -0.05:
            risks.append("low_growth_risk")
        if c_i < 0.33:
            risks.append("cold_major_risk")
        if row.get("sentiment_warning_score", 0) > 0.4:
            risks.append("sentiment_warning")
        if not risks:
            risks.append("no_significant_risk")
        return risks

    # ================================================================
    # 独立评分方法
    # ================================================================

    def calculate_career_score(self, row_series, df_norm_all):
        """综合计算 career_score(0-1)"""
        available = [c for c in self.INDICATORS_CORE if c in df_norm_all.columns]
        if not available:
            return 0.5
        if self.w_comb is None:
            return float(df_norm_all[available].mean(axis=1).iloc[0]) if len(df_norm_all) > 0 else 0.5
        w = np.array([self.w_comb.get(c, 0.05) for c in available])
        if w.sum() > 0:
            return float(np.dot(row_series[available].values, w) / w.sum())
        return 0.5

    def calculate_salary_score(self, row):
        """薪资评分(0-1)，结合平均值和中位数"""
        scores = []
        if "average_salary" in row:
            scores.append(float(row["average_salary"]))
        if "median_salary" in row:
            scores.append(float(row["median_salary"]))
        if not scores:
            return 0.5
        avg_sal = np.mean(scores)
        return min(1.0, max(0.0, avg_sal / 12000))

    def calculate_stability_score(self, row):
        """稳定性评分(0-1)，结合 stability_score 和考公适配度"""
        parts = []
        if "stability_score" in row:
            parts.append(min(1.0, float(row["stability_score"]) / 10))
        if "civil_service_post_count" in row:
            parts.append(min(1.0, float(row["civil_service_post_count"]) / 500))
        if not parts:
            return 0.5
        return float(np.mean(parts))

    def calculate_growth_score(self, row):
        """成长性评分(0-1)，结合岗位增长率和行业成长性"""
        parts = []
        if "job_growth_rate" in row:
            jgr = float(row["job_growth_rate"])
            parts.append(min(1.0, max(0.0, (jgr + 0.2) / 0.4)))
        if "industry_growth_score" in row:
            parts.append(min(1.0, float(row["industry_growth_score"]) / 10))
        if not parts:
            return 0.5
        return float(np.mean(parts))

    def calculate_postgraduate_value_score(self, row):
        """读研后就业价值(0-1)，区分本科直接就业和读研后就业价值

        升学率高、医学/法学/科研类专业的读研价值更高。
        """
        pg_rate = float(row.get("postgraduate_rate", 0))
        major_name = str(row.get("major_name", ""))
        base = min(1.0, pg_rate * 1.5 + 0.2)
        high_pg_majors = ["临床医学", "法学", "数学", "物理", "化学", "生物", "材料"]
        for kw in high_pg_majors:
            if kw in major_name:
                base = min(1.0, base + 0.15)
                break
        return round(base, 4)

    def calculate_civil_service_score(self, row):
        """考公考编适配度(0-1)"""
        cs_count = float(row.get("civil_service_post_count", 0))
        if cs_count <= 0:
            return 0.1
        return min(1.0, max(0.0, cs_count / 500))

    def calculate_major_cold_risk(self, row, c_i):
        """专业冷门风险(0-1)，1=最高风险"""
        risk = 0.0
        if row.get("employment_rate", 1) < 0.75:
            risk += 0.30
        if row.get("job_count", 0) < 5000:
            risk += 0.25
        if row.get("job_growth_rate", 0) < -0.05:
            risk += 0.25
        if c_i < 0.33:
            risk += 0.20
        return min(1.0, risk)

    # ================================================================
    # 时间序列趋势分析
    # ================================================================

    def time_series_trend_analysis(self, major_code, multi_year_df):
        """
        多年度就业趋势分析

        Args:
            major_code: 专业代码
            multi_year_df: DataFrame, 含多年就业数据(data_year列)

        Returns:
            dict: {trend_label, indicators_trend, confidence}

        当前为模拟数据接口，真实上线需接入多年真实就业数据。
        """
        if multi_year_df is None or len(multi_year_df) <= 1:
            return {
                "trend_label": "insufficient_data",
                "indicators_trend": {},
                "confidence": "low",
                "note": "当前为模拟数据/单年数据，趋势分析不可用。"
                       "真实上线需接入3年以上高校就业质量报告和招聘数据。",
            }

        major_data = multi_year_df[multi_year_df["major_code"] == major_code].sort_values("data_year")
        if len(major_data) <= 1:
            return {"trend_label": "insufficient_data", "indicators_trend": {}, "confidence": "low"}

        trend_indicators = {}
        for col in ["employment_rate", "average_salary", "job_count", "job_growth_rate"]:
            if col in major_data.columns:
                years = np.arange(len(major_data))
                vals = major_data[col].values.astype(float)
                if len(vals) >= 2:
                    slope = np.polyfit(years, vals, 1)[0]
                    mean_val = vals.mean() if vals.mean() != 0 else 1
                    trend_indicators[col] = {
                        "slope": round(float(slope), 4),
                        "direction": "up" if slope > 0 else "down" if slope < 0 else "stable",
                        "change_rate": round(float(slope / mean_val), 4),
                        "years_count": len(vals),
                    }

        if not trend_indicators:
            return {"trend_label": "no_data", "indicators_trend": {}, "confidence": "low"}

        downward = sum(1 for v in trend_indicators.values() if v["direction"] == "down")
        upward = sum(1 for v in trend_indicators.values() if v["direction"] == "up")
        total = len(trend_indicators)

        if downward > upward:
            label = "declining"
        elif upward > downward:
            label = "improving"
        else:
            label = "stable"

        n_years = max(len(major_data), 2)
        conf = "high" if n_years >= 4 else "medium" if n_years >= 3 else "low"

        return {
            "trend_label": label,
            "indicators_trend": trend_indicators,
            "confidence": conf,
        }

    # ================================================================
    # 薪资来源说明
    # ================================================================

    def explain_salary_source(self, row=None):
        """薪资数据来源与口径说明"""
        explanation = {
            "source": "综合高校就业质量报告、国家统计局分行业工资、招聘网站公开数据",
            "is_simulated": self._meta["is_simulated"],
            "salary_type": "average_salary(平均月薪) + median_salary(中位数月薪)",
            "unit": "元/月",
            "data_year": str(row.get("data_year", "N/A")) if row is not None else "N/A",
            "sample_note": "不同来源薪资口径可能存在差异(是否含五险一金、年终奖分摊等)。"
                           "模拟数据仅供测试，不代表真实薪资水平。",
            "disclaimer": "薪资数据仅为参考，不同城市、行业、企业类型差异较大，"
                          "不能直接承诺学生未来收入。",
        }
        if row is not None:
            if "average_salary" in row:
                explanation["average_salary"] = float(row["average_salary"])
            if "median_salary" in row:
                explanation["median_salary"] = float(row["median_salary"])
        return explanation

    # ================================================================
    # 三类业务解释
    # ================================================================

    def generate_parent_explanation(self, row):
        """家长端解释：通俗、克制、不承诺就业和薪资"""
        c_s = row.get("career_score", 0.5)
        label = row.get("red_yellow_green_label", "yellow")
        sal = row.get("average_salary", 0)
        emp_rate = row.get("employment_rate", 0)
        major = row.get("major_name", "该专业")

        label_cn = {"green": "较好", "yellow": "中等", "red": "需关注"}
        score_desc = "较高" if c_s > 0.6 else "中等" if c_s > 0.3 else "偏低"

        parts = [
            f"{major}的综合就业景气度为{label_cn.get(label, label)}（评分{score_desc}）。",
        ]
        if emp_rate > 0:
            parts.append(f"近年就业落实率约{int(emp_rate * 100)}%。")
        if sal > 0:
            parts.append(f"相关岗位月薪参考约{int(sal)}元（不同城市和行业有差异）。")

        if label == "red":
            parts.append("该专业就业竞争较激烈，建议结合个人兴趣和家庭资源综合评估。")
        if c_s < 0.3:
            parts.append("建议同时考虑就业前景更好的相关专业。")

        parts.append("以上数据为模拟参考，不同学校、地区和个人情况会有差异。")
        return "".join(parts)

    def generate_consultant_explanation(self, row):
        """咨询师端解释：指标分数、数据来源、风险标签、复核建议"""
        lines = [
            f"专业: {row.get('major_name', '')}({row.get('major_code', '')})",
            f"career_score: {row.get('career_score', 0):.4f}",
            f"标签: {row.get('red_yellow_green_label', '')}",
            f"employment_level: {row.get('employment_level', '')}",
            f"salary_score: {row.get('salary_score', 0):.4f}",
            f"stability_score: {row.get('stability_score', 0):.4f}",
            f"growth_score: {row.get('growth_score', 0):.4f}",
            f"postgraduate_value: {row.get('postgraduate_value_score', 0):.4f}",
            f"civil_service: {row.get('civil_service_score', 0):.4f}",
            f"major_risk_label: {row.get('major_risk_label', '')}",
            f"trend: {row.get('trend_label', 'N/A')}",
            f"data_reliability: {row.get('data_reliability_level', 'medium')}",
        ]
        if row.get("sentiment_warning_score", 0) > 0.3:
            lines.append(f"sentiment_warning: {row['sentiment_warning_score']:.2f}(非核心指标，仅辅助参考)")
        if row.get("review_required", False):
            lines.append("已触发人工复核，建议核实数据来源和时效性。")
        return "; ".join(lines)

    def generate_backend_explanation(self, row):
        """系统后台解释：结构化 dict"""
        return {
            "method": "AHP + Entropy + TOPSIS + KMeans clustering",
            "ahp_weights": self.ahp_weight_scheme_maps,
            "combined_weights": {k: round(float(v), 4) for k, v in (self.w_comb or {}).items()},
            "is_simulated": self._meta["is_simulated"],
            "note": self._meta["note"],
            "result": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                       for k, v in row.items()},
        }

    def generate_explanation(self, row):
        """统一解释入口，向后兼容 pipeline.py"""
        return self.generate_parent_explanation(row)

    # ================================================================
    # 数据可靠性评估
    # ================================================================

    def _assess_data_reliability(self, row):
        """评估数据可靠性"""
        reliability = "high"
        sample_size = row.get("sample_size", 100)
        if sample_size and sample_size < 30:
            reliability = "low"
        elif row.get("employment_rate", 0) == 0 and row.get("average_salary", 0) == 0:
            reliability = "low"
        if row.get("sentiment_warning_score", 0) > 0.5:
            reliability = "low"
        return reliability

    def _should_review(self, row, c_i):
        """判断是否需要人工复核"""
        if row.get("employment_rate", 0) == 0:
            return True
        if row.get("average_salary", 0) == 0:
            return True
        if row.get("sentiment_warning_score", 0) > 0.5:
            return True
        if row.get("red_yellow_green_label", "") == "red":
            return True
        if row.get("trend_label", "") == "declining":
            return True
        if row.get("sample_size", 100) < 30:
            return True
        return False

    # ================================================================
    # 主方法
    # ================================================================

    def evaluate_major(self, major_row, multi_year_df=None):
        """
        评价单个专业的就业景气度

        Args:
            major_row: Series, 单专业就业数据
            multi_year_df: DataFrame, 多年数据(可选,用于趋势分析)

        Returns:
            dict: 完整评价结果
        """
        row = major_row.copy()
        c_i = float(row.get("career_score", 0.5)) if "career_score" in row.index else 0.5

        trend_result = {"trend_label": "no_data", "indicators_trend": {}, "confidence": "low"}
        if multi_year_df is not None:
            major_code = row.get("major_code", "")
            trend_result = self.time_series_trend_analysis(major_code, multi_year_df)

        return {
            "major_code": str(row.get("major_code", "")),
            "major_name": str(row.get("major_name", "")),
            "career_score": round(c_i, 4),
            "employment_level": self._employment_level(c_i),
            "salary_score": round(self.calculate_salary_score(row), 4),
            "stability_score": round(self.calculate_stability_score(row), 4),
            "growth_score": round(self.calculate_growth_score(row), 4),
            "postgraduate_value_score": round(self.calculate_postgraduate_value_score(row), 4),
            "civil_service_score": round(self.calculate_civil_service_score(row), 4),
            "cold_major_risk_score": round(self.calculate_major_cold_risk(row, c_i), 4),
            "major_risk_label": self._major_risk_label(c_i),
            "risk_types": self._detailed_risk_type(row, c_i),
            "red_yellow_green_label": str(row.get("red_yellow_green_label", "yellow")),
            "sentiment_warning_score": round(float(row.get("sentiment_warning_score", 0)), 4),
            "trend_label": trend_result["trend_label"],
            "indicators_trend": trend_result["indicators_trend"],
            "salary_source_explanation": self.explain_salary_source(row),
            "data_reliability_level": self._assess_data_reliability(row),
            "explanation": self.generate_parent_explanation(row),
            "consultant_explanation": self.generate_consultant_explanation(row),
            "backend_explanation": self.generate_backend_explanation(row),
            "review_required": self._should_review(row, c_i),
        }

    def evaluate_batch(self, employment_df, multi_year_df=None):
        """
        批量评价多个专业的就业景气度

        Args:
            employment_df: DataFrame, 含所有就业指标的宽表
            multi_year_df: DataFrame, 多年数据(可选)

        Returns:
            DataFrame: 含各专业评分和标签
        """
        df = employment_df.copy()

        df_norm = self.normalize(df)
        c_i = self.topsis_evaluate(df_norm)
        df["career_score"] = c_i
        df["red_yellow_green_label"] = self.generate_labels(c_i)
        df["employment_level"] = [self._employment_level(v) for v in c_i]

        df["salary_score"] = df.apply(self.calculate_salary_score, axis=1)
        df["stability_score"] = df.apply(self.calculate_stability_score, axis=1)
        df["growth_score"] = df.apply(self.calculate_growth_score, axis=1)
        df["postgraduate_value_score"] = df.apply(self.calculate_postgraduate_value_score, axis=1)
        df["civil_service_score"] = df.apply(self.calculate_civil_service_score, axis=1)
        df["cold_major_risk_score"] = [
            self.calculate_major_cold_risk(row, c_i[i]) for i, (_, row) in enumerate(df.iterrows())
        ]
        df["major_risk_label"] = [self._major_risk_label(v) for v in c_i]

        if multi_year_df is not None:
            trends = {}
            for major_code in df["major_code"].unique():
                trends[major_code] = self.time_series_trend_analysis(major_code, multi_year_df)
            df["trend_label"] = df["major_code"].map(
                lambda mc: trends.get(mc, {}).get("trend_label", "no_data")
            )
        else:
            df["trend_label"] = "no_data"

        df["data_reliability_level"] = df.apply(self._assess_data_reliability, axis=1)
        df["review_required"] = [
            self._should_review(row, c_i[i]) for i, (_, row) in enumerate(df.iterrows())
        ]

        return df

    def evaluate(self, employment_df, multi_year_df=None):
        """
        主方法：专业就业景气度评价（向后兼容 pipeline.py）

        Args:
            employment_df: DataFrame, 含所有就业指标的宽表
            multi_year_df: DataFrame, 多年数据(可选)

        Returns:
            DataFrame: 含各专业评分、红黄绿标签和完整输出字段
        """
        return self.evaluate_batch(employment_df, multi_year_df)


if __name__ == "__main__":
    from data_generator import generate_major_employment

    print("=" * 50)
    print("  核心模型五：专业就业景气度评价模型 演示 (Project 5)")
    print("=" * 50)

    emp_df = generate_major_employment(data_years=[2023])
    evaluator = CareerEvaluator()
    result = evaluator.evaluate(emp_df)

    print(f"\n评价结果（前12个专业）:")
    display_cols = [
        "major_code", "major_name", "career_score",
        "red_yellow_green_label", "employment_level",
        "salary_score", "stability_score", "growth_score",
        "postgraduate_value_score", "civil_service_score",
        "cold_major_risk_score", "major_risk_label",
        "trend_label", "data_reliability_level", "review_required",
    ]
    available = [c for c in display_cols if c in result.columns]
    print(result[available].head(12).to_string(index=False))

    print(f"\n红黄绿分布:")
    print(result["red_yellow_green_label"].value_counts())

    top5 = result.nlargest(5, "career_score")
    print(f"\n就业景气度 TOP5:")
    for _, row in top5.iterrows():
        print(f"  {row['major_name']}: {row['career_score']:.3f} ({row['red_yellow_green_label']})")

    bottom5 = result.nsmallest(5, "career_score")
    print(f"\n就业景气度 BOTTOM5:")
    for _, row in bottom5.iterrows():
        print(f"  {row['major_name']}: {row['career_score']:.3f} ({row['red_yellow_green_label']})")

    # 单个专业评价演示
    print(f"\n单个专业评价示例 (计算机科学与技术):")
    cs_row = result[result["major_code"] == "080901"].iloc[0]
    single = evaluator.evaluate_major(cs_row)
    print(f"  career_score: {single['career_score']}")
    print(f"  salary_score: {single['salary_score']}")
    print(f"  growth_score: {single['growth_score']}")
    print(f"  postgraduate_value_score: {single['postgraduate_value_score']}")
    print(f"  civil_service_score: {single['civil_service_score']}")
    print(f"  risk_label: {single['major_risk_label']}")
    print(f"  risk_types: {single['risk_types']}")
    print(f"  review_required: {single['review_required']}")
    print(f"  家长解释: {single['explanation'][:100]}...")
    print(f"  薪资说明: {single['salary_source_explanation']['is_simulated']}")
