"""
可视化模块 (Project 5 v3.0.0)

为数学建模论文和咨询报告生成图表：
- 位次曲线 (模型一)
- 等效分对比 (模型一)
- 录取概率分布 (模型二)
- 概率校准图 (模型二评价)
- 志愿结构柱状图 (模型三)
- 风险热力图 (模型四)
- 专业就业雷达图 (模型五)
- 城市价值柱状图 (扩展模型)

依赖: matplotlib
所有图表基于模拟数据，仅供教学和测试使用。
"""

import os
import warnings
import numpy as np
import pandas as pd


def _setup_chinese_font():
    """尝试设置中文字体，失败时不报错"""
    try:
        import matplotlib
        from matplotlib.font_manager import FontProperties
        available = {f.name for f in FontProperties().get_fontManager().ttflist}
        for font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]:
            if font in available:
                matplotlib.rcParams["font.sans-serif"] = [font]
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_setup_chinese_font()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GaokaoVisualizer:
    """高考志愿填报可视化器"""

    TIER_COLORS = {"rush": "#E74C3C", "stable": "#F39C12", "safe": "#2ECC71", "bottom": "#3498DB"}
    TIER_LABELS = {"rush": "冲", "stable": "稳", "safe": "保", "bottom": "垫"}
    RISK_LABELS = {
        "slip_risk": "滑档风险", "withdrawal_risk": "退档风险",
        "adjustment_risk": "调剂风险", "cold_major_risk": "专业冷门风险",
        "employment_risk": "就业风险", "region_risk": "地域风险",
    }
    RADAR_LABELS = ["薪资", "稳定度", "成长性", "读研价值", "考公适配度"]

    def __init__(self, output_dir="output/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}

    # ================================================================
    # 1. 位次曲线 (模型一)
    # ================================================================

    def plot_rank_curve(self, segment_tables, province="", subject_type="",
                         title=None, filename="rank_curve.png"):
        """
        绘制分数—位次曲线，展示多年份对比。

        Args:
            segment_tables: dict {year: DataFrame} 或 DataFrame(含year列)
        Returns:
            dict: {status, path, warning}
        """
        if segment_tables is None:
            return self._skip(filename, "segment_tables is None")

        fig, ax = plt.subplots(figsize=(10, 6))
        years_data = self._normalize_segment_tables(segment_tables)

        if not years_data:
            plt.close(fig)
            return self._skip(filename, "no segment data")

        for year, df in years_data.items():
            if "score" in df.columns and "cumulative_count" in df.columns:
                ax.plot(df["score"], df["cumulative_count"], alpha=0.7, label=str(year))

        ax.set_xlabel("Score")
        ax.set_ylabel("Cumulative Count")
        t = title or f"Rank Curve — {province} {subject_type}"
        ax.set_title(t)
        ax.legend()
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        return self._save(fig, filename)

    def _normalize_segment_tables(self, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
        if isinstance(data, pd.DataFrame):
            if "year" in data.columns:
                return {int(y): g for y, g in data.groupby("year")}
            return {"data": data}
        return {}

    # ================================================================
    # 2. 等效分对比 (模型一)
    # ================================================================

    def plot_equivalent_score_comparison(self, equivalent_result,
                                           title=None, filename="equivalent_score_comparison.png"):
        """
        绘制不同年份等效分对比图（含误差线）。

        Args:
            equivalent_result: dict, 来自 equivalent_score.convert()
        Returns:
            dict: {status, path, warning}
        """
        if not equivalent_result or "year_details" not in equivalent_result:
            return self._skip(filename, "missing year_details in equivalent_result")

        details = equivalent_result["year_details"]
        if not details:
            return self._skip(filename, "year_details is empty")

        years = []
        scores = []
        for year_str, val in sorted(details.items()):
            years.append(str(year_str))
            scores.append(val.get("equivalent_score", 0))

        if not years:
            return self._skip(filename, "no year data")

        eq = equivalent_result.get("equivalent_score")
        interval = equivalent_result.get("equivalent_score_interval")
        confidence = equivalent_result.get("confidence_level", "")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(years, scores, color="#5DADE2", alpha=0.8, label="per-year eq score")
        if eq is not None:
            ax.axhline(y=eq, color="#E74C3C", linestyle="--", linewidth=2,
                        label=f"weighted avg: {eq}")
        if interval and len(interval) == 2:
            ax.fill_between([years[0], years[-1]], interval[0], interval[1],
                             alpha=0.15, color="#E74C3C", label=f"95% CI (conf: {confidence})")
        ax.set_ylabel("Equivalent Score")
        t = title or f"Equivalent Score Comparison (confidence: {confidence})"
        ax.set_title(t)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        return self._save(fig, filename)

    # ================================================================
    # 3. 录取概率分布 (模型二)
    # ================================================================

    def plot_probability_distribution(self, volunteers, title=None,
                                        filename="probability_distribution.png"):
        """
        绘制候选志愿录取概率分布，按 tier 着色。

        Args:
            volunteers: DataFrame 或 list[dict], 含 admit_probability / recommendation_tier
        Returns:
            dict: {status, path, warning}
        """
        df = self._to_dataframe(volunteers)
        if df is None or df.empty:
            return self._skip(filename, "volunteers is empty")

        if "admit_probability" not in df.columns:
            return self._skip(filename, "missing admit_probability")

        df = df.sort_values("admit_probability")
        tier_col = "recommendation_tier" if "recommendation_tier" in df.columns else None

        fig, ax = plt.subplots(figsize=(12, 5))
        if tier_col:
            for tier in ["rush", "stable", "safe", "bottom"]:
                mask = df[tier_col] == tier
                if mask.any():
                    ax.bar(range(mask.sum()), df.loc[mask, "admit_probability"],
                            color=self.TIER_COLORS.get(tier, "#999"),
                            label=self.TIER_LABELS.get(tier, tier),
                            alpha=0.85)
            ax.legend()
        else:
            ax.bar(range(len(df)), df["admit_probability"], color="#5DADE2", alpha=0.7)

        ax.set_xlabel("Volunteer Index")
        ax.set_ylabel("Admission Probability")
        ax.set_ylim(0, 1)
        t = title or "Admission Probability Distribution"
        ax.set_title(t)
        ax.grid(True, alpha=0.3, axis="y")

        return self._save(fig, filename)

    # ================================================================
    # 4. 概率校准图 (模型二评价)
    # ================================================================

    def plot_probability_calibration(self, predicted, actual=None, bins=10,
                                       title=None, filename="probability_calibration.png"):
        """
        绘制录取概率校准图。

        Args:
            predicted: array-like, 预测概率
            actual: array-like, 实际录取结果(0/1)。若为None，仅画空图+warning
        Returns:
            dict: {status, path, warning}
        """
        predicted = np.asarray(predicted, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")

        if actual is None:
            ax.text(0.5, 0.5, "No actual labels available\n(requires real admission data)",
                     ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_title("Calibration Plot — No Real Data")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Admission Rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return self._save(fig, filename,
                              warning="no actual labels; simulated data cannot produce calibration")
        else:
            actual = np.asarray(actual, dtype=float)
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = []
            bin_actuals = []
            for i in range(bins):
                mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_centers.append(predicted[mask].mean())
                    bin_actuals.append(actual[mask].mean())
            if bin_centers:
                ax.plot(bin_centers, bin_actuals, "o-", color="#E74C3C", linewidth=2, label="model")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Admission Rate")
            ax.set_title(title or "Probability Calibration")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            return self._save(fig, filename)

    # ================================================================
    # 5. 志愿结构柱状图 (模型三)
    # ================================================================

    def plot_volunteer_structure_bar(self, optimize_result, plan_type="",
                                       title=None, filename="volunteer_structure_bar.png"):
        """
        绘制冲/稳/保/垫志愿结构柱状图。

        Args:
            optimize_result: dict, 含 rush_count/stable_count/safe_count/bottom_count
                或 DataFrame(含 recommendation_tier 列)
        Returns:
            dict: {status, path, warning}
        """
        counts = {}
        if isinstance(optimize_result, dict):
            for k in ["rush_count", "stable_count", "safe_count", "bottom_count"]:
                counts[k.replace("_count", "")] = optimize_result.get(k, 0)
        elif isinstance(optimize_result, pd.DataFrame) and "recommendation_tier" in optimize_result.columns:
            vc = optimize_result["recommendation_tier"].value_counts()
            for t in ["rush", "stable", "safe", "bottom"]:
                counts[t] = int(vc.get(t, 0))
        else:
            return self._skip(filename, "unsupported input type for volunteer structure")

        if sum(counts.values()) == 0:
            return self._skip(filename, "all counts are zero")

        tiers = ["rush", "stable", "safe", "bottom"]
        values = [counts.get(t, 0) for t in tiers]
        colors = [self.TIER_COLORS[t] for t in tiers]
        labels = [self.TIER_LABELS[t] for t in tiers]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(tiers, values, color=colors, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(val), ha="center", fontsize=11, fontweight="bold")
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        t = title or f"Volunteer Structure — {plan_type}"
        ax.set_title(t)
        ax.grid(True, alpha=0.3, axis="y")

        return self._save(fig, filename)

    # ================================================================
    # 6. 风险热力图 (模型四)
    # ================================================================

    def plot_risk_heatmap(self, risk_assessment, title=None,
                            filename="risk_heatmap.png"):
        """
        绘制六类风险热力图。

        Args:
            risk_assessment: dict, 含 slip_risk/withdrawal_risk/... 可为 float 或
                            {score,level,trigger_reason,...}
        Returns:
            dict: {status, path, warning}
        """
        if not risk_assessment:
            return self._skip(filename, "risk_assessment is empty")

        risk_keys = ["slip_risk", "withdrawal_risk", "adjustment_risk",
                      "cold_major_risk", "employment_risk", "region_risk"]
        scores = []
        for key in risk_keys:
            val = risk_assessment.get(key, 0)
            if isinstance(val, dict):
                scores.append(float(val.get("score", 0)))
            else:
                scores.append(float(val))

        labels = [self.RISK_LABELS.get(k, k) for k in risk_keys]
        scores_arr = np.array(scores).reshape(1, -1)

        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(scores_arr, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

        for i, s in enumerate(scores):
            ax.text(i, 0, f"{s:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if s > 0.5 else "black")

        overall = risk_assessment.get("overall_risk_score",
                                        risk_assessment.get("risk_level", ""))
        t = title or f"Risk Heatmap (overall: {overall})"
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Risk Score")

        return self._save(fig, filename)

    # ================================================================
    # 7. 专业就业雷达图 (模型五)
    # ================================================================

    def plot_major_radar(self, major_data, title=None, filename="major_radar.png"):
        """
        绘制专业就业景气度雷达图。

        Args:
            major_data: dict 或 Series, 含 salary_score/stability_score/growth_score/
                        postgraduate_value_score/civil_service_score/career_score
        Returns:
            dict: {status, path, warning}
        """
        if major_data is None:
            return self._skip(filename, "major_data is None")

        keys = ["salary_score", "stability_score", "growth_score",
                "postgraduate_value_score", "civil_service_score"]
        values = []
        missing = []
        for key in keys:
            v = major_data.get(key, None) if isinstance(major_data, dict) else getattr(major_data, key, None)
            if v is None:
                missing.append(key)
                values.append(0)
            else:
                values.append(float(v))

        if all(v == 0 for v in values):
            return self._skip(filename, "all radar values are zero")

        values.append(values[0])
        n = len(keys)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="#3498DB")
        ax.plot(angles, values, "o-", linewidth=2, color="#3498DB")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.RADAR_LABELS[:n], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.grid(True)

        major_name = major_data.get("major_name", "") if isinstance(major_data, dict) else ""
        cs = major_data.get("career_score", "") if isinstance(major_data, dict) else ""
        t = title or f"Career Radar — {major_name} (score={cs})"
        ax.set_title(t, pad=20)

        warning = f"missing: {missing}" if missing else ""
        return self._save(fig, filename, warning=warning)

    # ================================================================
    # 8. 城市价值柱状图 (扩展模型)
    # ================================================================

    def plot_city_score_bar(self, city_data, title=None, filename="city_score_bar.png"):
        """
        绘制城市价值或匹配度柱状图。

        Args:
            city_data: DataFrame 或 list[dict], 含 city/city_score
        Returns:
            dict: {status, path, warning}
        """
        df = self._to_dataframe(city_data)
        if df is None or df.empty:
            return self._skip(filename, "city_data is empty")

        city_col = "city" if "city" in df.columns else None
        score_col = "C_city" if "C_city" in df.columns else "city_score" if "city_score" in df.columns else None

        if city_col is None or score_col is None:
            return self._skip(filename, f"missing columns: need city and city_score/C_city, got {list(df.columns)}")

        df = df.sort_values(score_col, ascending=True)
        top_n = min(15, len(df))
        df = df.tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
        bars = ax.barh(range(len(df)), df[score_col], color=colors, alpha=0.85)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df[city_col].values, fontsize=8)
        ax.set_xlabel("City Score")
        ax.set_xlim(0, 1)
        t = title or "City Value Score"
        ax.set_title(t)
        ax.grid(True, alpha=0.3, axis="x")

        return self._save(fig, filename)

    # ================================================================
    # 9. 批量生成
    # ================================================================

    def generate_all_figures(self, data_dict):
        """
        批量生成所有可视化图表。

        Args:
            data_dict: dict, 可包含:
                {"segment_table": ..., "equivalent_result": ..., "volunteers": ...,
                 "risk_assessment": ..., "career_result": ..., "city_data": ...,
                 "optimize_result": ..., "province": ..., "subject_type": ...}

        Returns:
            dict: {chart_name: {status, path, warning}}
        """
        self.results = {}

        seg = data_dict.get("segment_table")
        province = data_dict.get("province", "")
        subject = data_dict.get("subject_type", "")
        self.results["rank_curve"] = self.plot_rank_curve(seg, province, subject)

        eq_result = data_dict.get("equivalent_result")
        self.results["equiv_score_comparison"] = self.plot_equivalent_score_comparison(eq_result)

        volunteers = data_dict.get("volunteers")
        self.results["prob_distribution"] = self.plot_probability_distribution(volunteers)

        opt_result = data_dict.get("optimize_result")
        plan_type = data_dict.get("plan_type", "")
        self.results["volunteer_structure"] = self.plot_volunteer_structure_bar(opt_result, plan_type)

        risk = data_dict.get("risk_assessment")
        self.results["risk_heatmap"] = self.plot_risk_heatmap(risk)

        career = data_dict.get("career_result")
        self.results["major_radar"] = self.plot_major_radar(career if career is not None else {})

        city = data_dict.get("city_data")
        self.results["city_score_bar"] = self.plot_city_score_bar(city)

        return self.results

    # ================================================================
    # 工具方法
    # ================================================================

    def _save(self, fig, filename, warning=""):
        path = os.path.join(self.output_dir, filename)
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return {"status": "success", "path": path, "warning": warning}
        except Exception as e:
            plt.close(fig)
            return {"status": "error", "path": "", "warning": str(e)}

    def _skip(self, filename, reason):
        return {"status": "skipped", "path": "", "warning": reason}

    @staticmethod
    def _to_dataframe(data):
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list) and all(isinstance(d, dict) for d in data):
            return pd.DataFrame(data)
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("  GaokaoVisualizer — Min Self-Check")
    print("=" * 50)

    from src.data_generator import generate_all_data, generate_major_employment
    from src.equivalent_score import EquivalentScoreConverter

    viz = GaokaoVisualizer(output_dir="output/figures")

    # --- 1. rank_curve ---
    print("\n[1] plot_rank_curve...")
    data = generate_all_data()
    r = viz.plot_rank_curve(data["segment_table"], province="河北省", subject_type="物理类")
    print(f"    {r}")

    # --- 2. equivalent_score_comparison ---
    print("\n[2] plot_equivalent_score_comparison...")
    seg_table = data["segment_table"]
    segs = {int(y): g for y, g in seg_table.groupby("year")}
    conv = EquivalentScoreConverter()
    eq_result = conv.convert(
        segment_dfs=segs, candidate_score=620, candidate_rank=8500,
        current_year=2024, target_years=[2020, 2021, 2022, 2023],
        batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
    )
    r = viz.plot_equivalent_score_comparison(eq_result)
    print(f"    {r}")

    # --- 3. probability_distribution ---
    print("\n[3] plot_probability_distribution...")
    vols = [
        {"admit_probability": 0.12, "recommendation_tier": "rush"},
        {"admit_probability": 0.28, "recommendation_tier": "rush"},
        {"admit_probability": 0.55, "recommendation_tier": "stable"},
        {"admit_probability": 0.65, "recommendation_tier": "stable"},
        {"admit_probability": 0.78, "recommendation_tier": "safe"},
        {"admit_probability": 0.92, "recommendation_tier": "bottom"},
    ]
    r = viz.plot_probability_distribution(vols)
    print(f"    {r}")

    # --- 4. calibration (no real data) ---
    print("\n[4] plot_probability_calibration...")
    r = viz.plot_probability_calibration(predicted=[0.3, 0.5, 0.7])
    print(f"    {r}")

    # --- 5. volunteer_structure_bar ---
    print("\n[5] plot_volunteer_structure_bar...")
    opt = {"rush_count": 8, "stable_count": 12, "safe_count": 12, "bottom_count": 8}
    r = viz.plot_volunteer_structure_bar(opt, plan_type="balanced")
    print(f"    {r}")

    # --- 6. risk_heatmap ---
    print("\n[6] plot_risk_heatmap...")
    risk = {
        "overall_risk_score": 0.18, "risk_level": "medium",
        "slip_risk": {"score": 0.05}, "withdrawal_risk": {"score": 0.02},
        "adjustment_risk": {"score": 0.15}, "cold_major_risk": {"score": 0.10},
        "employment_risk": {"score": 0.08}, "region_risk": {"score": 0.12},
    }
    r = viz.plot_risk_heatmap(risk)
    print(f"    {r}")

    # --- 7. major_radar ---
    print("\n[7] plot_major_radar...")
    emp = generate_major_employment(data_years=[2023])
    from src.career_evaluation import CareerEvaluator
    ev = CareerEvaluator()
    career_result = ev.evaluate(emp)
    cs_row = career_result[career_result["major_code"] == "080901"].iloc[0]
    r = viz.plot_major_radar(cs_row.to_dict())
    print(f"    {r}")

    # --- 8. city_score_bar ---
    print("\n[8] plot_city_score_bar...")
    r = viz.plot_city_score_bar(data["city_data"])
    print(f"    {r}")

    # --- 9. generate_all_figures ---
    print("\n[9] generate_all_figures...")
    opt_df = pd.DataFrame(vols)
    opt_df["recommendation_tier"] = opt_df["admit_probability"].apply(
        lambda p: "bottom" if p >= 0.88 else "safe" if p >= 0.45 else "stable" if p >= 0.20 else "rush"
    )
    bundle = {
        "segment_table": data["segment_table"],
        "equivalent_result": eq_result,
        "volunteers": vols,
        "optimize_result": opt_df,
        "risk_assessment": risk,
        "career_result": cs_row.to_dict(),
        "city_data": data["city_data"],
        "province": "河北省", "subject_type": "物理类", "plan_type": "balanced",
    }
    all_r = viz.generate_all_figures(bundle)
    for k, v in all_r.items():
        print(f"    {k}: {v['status']}")

    # --- summary ---
    generated = [f for f in os.listdir(viz.output_dir) if f.endswith(".png")]
    print(f"\nGenerated {len(generated)} PNG files:")
    for f in sorted(generated):
        print(f"    {f}")
