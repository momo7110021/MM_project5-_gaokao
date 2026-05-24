"""
核心模型一：分数—位次等效换算模型

功能：将考生当年分数转换为历史年份的等效分。
方法：分位数映射 + 线性插值 + 线差修正 + 异常年份降权 + 置信区间估计
"""

import numpy as np
import pandas as pd


class EquivalentScoreConverter:
    """分数-位次等效换算器"""

    def __init__(self, theta=0.3, tau=1000):
        self.theta = theta
        self.tau = tau

    def compute_percentile(self, segment_df, score=None, rank=None):
        """根据分数或位次计算分位点"""
        if rank is not None:
            total = segment_df["cumulative_count"].max()
            if total is None or total == 0:
                raise ValueError("一分一段表缺少考生总数")
            return rank / total

        if score is not None:
            scores = segment_df["score"].values
            cumcounts = segment_df["cumulative_count"].values
            total = segment_df["cumulative_count"].max()
            if score >= scores[0]:
                return cumcounts[0] / total
            if score <= scores[-1]:
                return cumcounts[-1] / total
            idx = np.searchsorted(-scores, -score)
            if idx == 0:
                return cumcounts[0] / total
            if idx >= len(scores):
                return cumcounts[-1] / total
            s1, s2 = scores[idx - 1], scores[idx]
            c1, c2 = cumcounts[idx - 1], cumcounts[idx]
            frac = (score - s1) / (s2 - s1) if s2 != s1 else 0
            cum = c1 + frac * (c2 - c1)
            return cum / total

        raise ValueError("必须提供 score 或 rank")

    def percentile_to_score(self, segment_df, percentile):
        """将分位点映射回目标年份的分数（含线性插值）
        cumulative_count 在 score 降序下递增: score↓ cum↑ -> cum_pcts↑
        """
        df = segment_df.copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["cumulative_count"] = pd.to_numeric(df["cumulative_count"], errors="coerce")
        df = df.dropna(subset=["score", "cumulative_count"])
        df = df.groupby("score", as_index=False)["cumulative_count"].max()
        df = df.sort_values("score", ascending=False)

        scores = df["score"].to_numpy(dtype=float)
        cumcounts = df["cumulative_count"].to_numpy(dtype=float)
        total = cumcounts.max()
        if total <= 0:
            return float(np.nan)

        cum_pcts = cumcounts / total  # ascending: 0.0001 ... 1.0

        # small percentile → good rank → high score
        if percentile <= cum_pcts[0]:
            return float(np.clip(scores[0], 0, 750))
        # large percentile → poor rank → low score
        if percentile >= cum_pcts[-1]:
            return float(np.clip(scores[-1], 0, 750))

        idx = np.searchsorted(cum_pcts, percentile, side="left")
        if idx <= 0:
            return float(np.clip(scores[0], 0, 750))
        if idx >= len(scores):
            return float(np.clip(scores[-1], 0, 750))

        p1, p2 = cum_pcts[idx - 1], cum_pcts[idx]
        s1, s2 = scores[idx - 1], scores[idx]
        if abs(p2 - p1) < 1e-12:
            return float(np.clip(s1, 0, 750))

        eq_score = s1 + (percentile - p1) / (p2 - p1) * (s2 - s1)
        return float(np.clip(eq_score, 0, 750))

    def convert_single_year(self, segment_df_current, segment_df_target,
                             current_score, current_rank,
                             current_batch_line, target_batch_line):
        """将考生分数换算为单个目标年份的等效分"""
        percentile = self.compute_percentile(
            segment_df_current, score=current_score, rank=current_rank
        )
        eq_score_raw = self.percentile_to_score(segment_df_target, percentile)

        eq_score_adjusted = max(0.0, min(750.0, eq_score_raw + self.theta * (
            (current_batch_line - target_batch_line) * (1 - percentile)
        )))

        return {
            "equivalent_score": round(eq_score_adjusted, 1),
            "equivalent_score_raw": round(eq_score_raw, 1),
            "percentile": round(percentile, 6),
            "target_batch_line": float(target_batch_line),
        }

    def convert(self, segment_dfs, candidate_score, candidate_rank,
                current_year, target_years, batch_lines):
        """
        主方法：多年度加权等效分换算

        Args:
            segment_dfs: dict, {year: DataFrame} 一分一段表
            candidate_score: float, 考生当年分数
            candidate_rank: int, 考生当年位次
            current_year: int, 当前年份
            target_years: list, 目标年份列表
            batch_lines: dict, {year: float} 各年批次线

        Returns:
            dict: 包含等效分、置信区间、各年明细
        """
        current_seg = segment_dfs[current_year]
        current_bl = batch_lines.get(current_year, 0)

        year_results = {}
        for t_year in target_years:
            if t_year not in segment_dfs:
                continue
            year_results[t_year] = self.convert_single_year(
                current_seg, segment_dfs[t_year],
                candidate_score, candidate_rank,
                current_bl, batch_lines.get(t_year, 0)
            )

        if not year_results:
            return self._fallback_result(candidate_score, candidate_rank)

        eq_scores = np.array([
            r["equivalent_score"] for r in year_results.values()
        ])

        if len(eq_scores) >= 2:
            sigma_t = np.abs(eq_scores[:, None] - eq_scores[None, :]).mean(axis=1)
            w_t = np.exp(-sigma_t / self.tau)
            w_t = w_t / w_t.sum()
        else:
            sigma_t = np.array([0])
            w_t = np.array([1.0])

        eq_star = max(0.0, min(750.0, np.sum(w_t * eq_scores)))

        if len(eq_scores) >= 2:
            sigma_S = np.sqrt(
                np.sum(w_t * (eq_scores - eq_star) ** 2) / np.sum(w_t)
            )
        else:
            sigma_S = 0

        ci_lower = eq_star - 1.96 * sigma_S
        ci_upper = eq_star + 1.96 * sigma_S

        if sigma_S <= 3:
            confidence = "high"
        elif sigma_S <= 8:
            confidence = "medium"
        else:
            confidence = "low"

        median_sigma = np.median(sigma_t) if len(sigma_t) > 0 else 0
        abnormal_threshold = 3 * median_sigma + 1
        abnormal_years = [
            year for year, sig in zip(year_results.keys(), sigma_t)
            if sig > abnormal_threshold
        ]

        eq_rank = self.score_to_rank(segment_dfs.get(target_years[-1] if target_years else current_year, segment_dfs[current_year]), eq_star)
        fallback_triggered = len(year_results) == 0

        return {
            "candidate_score": candidate_score,
            "candidate_rank": candidate_rank,
            "current_percentile": year_results.get(
                list(year_results.keys())[0], {}
            ).get("percentile", 0),
            "equivalent_score": round(eq_star, 1),
            "equivalent_score_interval": [round(ci_lower, 1), round(ci_upper, 1)],
            "equivalent_rank": int(eq_rank) if eq_rank is not None else None,
            "confidence_level": confidence,
            "abnormal_year_warning": len(abnormal_years) > 0,
            "abnormal_years": abnormal_years,
            "fallback_rule": "insufficient_data" if fallback_triggered else "normal",
            "review_required": confidence == "low" or len(abnormal_years) > 0,
            "year_details": {
                year: {
                    "equivalent_score": res["equivalent_score"],
                    "equivalent_score_raw": res["equivalent_score_raw"],
                }
                for year, res in year_results.items()
            },
        }

    def score_to_rank(self, segment_df, eq_score):
        """从等效分反查位次"""
        if segment_df is None:
            return None
        scores = segment_df["score"].values
        cumcounts = segment_df["cumulative_count"].values
        if eq_score >= scores[0]:
            return int(cumcounts[0])
        if eq_score <= scores[-1]:
            return int(cumcounts[-1])
        idx = np.searchsorted(-scores, -eq_score)
        if idx >= len(scores):
            return int(cumcounts[-1])
        if idx == 0:
            return int(cumcounts[0])
        s1, s2 = scores[idx - 1], scores[idx]
        c1, c2 = cumcounts[idx - 1], cumcounts[idx]
        frac = (eq_score - s1) / (s2 - s1) if s2 != s1 else 0
        return int(c1 + frac * (c2 - c1))

    def evaluate_backtest_error(self, segment_dfs, test_year, test_score, test_rank, batch_lines):
        """留一验证：计算等效分回测误差"""
        target_years = [y for y in segment_dfs.keys() if y != test_year]
        result = self.convert(
            segment_dfs=segment_dfs,
            candidate_score=test_score,
            candidate_rank=test_rank,
            current_year=test_year,
            target_years=target_years,
            batch_lines=batch_lines,
        )
        actual_scores = [d["equivalent_score"] for d in result["year_details"].values()]
        if actual_scores:
            mae = np.mean([abs(s - test_score) for s in actual_scores])
            return {"mae": round(mae, 1), "target_year": test_year}
        return {"mae": None, "target_year": test_year}

    def generate_parent_explanation(self, result):
        """家长端解释：通俗、克制、不承诺"""
        eq = result["equivalent_score"]
        lo, hi = result["equivalent_score_interval"]
        conf = result["confidence_level"]
        rank = result.get("equivalent_rank")
        lines = [
            f"您今年考了{result['candidate_score']}分,在全省排第{result['candidate_rank']}名。",
            f"相当于往年的{round(eq)}分(区间 {round(lo)}-{round(hi)} 分)。",
            "这是因为每年试卷难度和考生人数不同,不能直接用今年分数比较往年录取分数。",
            "位次比分数更稳定,我们用您在全省考生中的相对位置来换算。",
        ]
        if rank:
            lines.append(f"等效位次约第{rank}名,这是和往年录取位次做比较的依据。")
        if conf != "high":
            lines.append(f"注意:今年数据波动较大,等效分置信度为{conf},建议咨询师复核。")
        return "".join(lines)

    def generate_consultant_explanation(self, result):
        """咨询师端解释：模型依据、风险条件、可调整项"""
        conf = result["confidence_level"]
        abnormal = result.get("abnormal_years", [])
        lines = [
            f"等效分: {result['equivalent_score']}, 区间: {result['equivalent_score_interval']}, "
            f"置信度: {conf}, 各年等效分: {result['year_details']}",
            f"换算方法: 分位数映射+线差修正(theta={self.theta})+异常年份降权(tau={self.tau})",
        ]
        if abnormal:
            lines.append(f"异常年份: {abnormal}(已自动降权,但建议复核该年份数据)")
        if conf == "low":
            lines.append("置信度低,建议: 1)扩大候选院校分数范围; 2)优先参考位次而非等效分; 3)增加保底志愿数量")
        if result.get("review_required"):
            lines.append("已触发人工复核标志,请确认各年一分一段表数据完整性")
        return "; ".join(lines)

    def generate_backend_explanation(self, result):
        """系统后台解释：结构化字段,便于追溯和审核"""
        return {
            "method": "quantile_mapping + batch_line_correction + abnormal_year_downweight",
            "parameters": {"theta": self.theta, "tau": self.tau},
            "source_table": "segment_table",
            "result": result,
        }

    def _fallback_result(self, score, rank):
        return {
            "candidate_score": score,
            "candidate_rank": rank,
            "current_percentile": 0,
            "equivalent_score": float(score),
            "equivalent_score_interval": [float(score), float(score)],
            "equivalent_rank": rank,
            "confidence_level": "low",
            "abnormal_year_warning": True,
            "abnormal_years": [],
            "fallback_rule": "insufficient_data",
            "review_required": True,
            "year_details": {},
        }


if __name__ == "__main__":
    from data_generator import generate_segment_table, generate_all_data

    print("=" * 50)
    print("  核心模型一：分数—位次等效换算模型 演示")
    print("=" * 50)

    data = generate_all_data(candidate_score=620, candidate_rank=8500)

    segment_by_year = {}
    batch_lines = {}
    for year in [2020, 2021, 2022, 2023, 2024]:
        seg_df = generate_segment_table(
            province="河北省", year=year, subject_type="物理类",
            total_exam_count=330000 + year * 2000,
            batch_line=430 + (year - 2020) * 2,
        )
        segment_by_year[year] = seg_df
        batch_lines[year] = 430 + (year - 2020) * 2

    converter = EquivalentScoreConverter()
    result = converter.convert(
        segment_dfs=segment_by_year,
        candidate_score=620,
        candidate_rank=8500,
        current_year=2024,
        target_years=[2020, 2021, 2022, 2023],
        batch_lines=batch_lines,
    )

    print(f"\n考生信息: 2024年 620分, 位次 8500")
    print(f"加权等效分: {result['equivalent_score']}")
    print(f"等效分区间: {result['equivalent_score_interval']}")
    print(f"置信度: {result['confidence_level']}")
    print(f"异常年份警告: {result['abnormal_year_warning']}")
    print(f"异常年份: {result['abnormal_years']}")
    print("\n各年明细:")
    for year, detail in result["year_details"].items():
        print(f"  {year}: 等效分 {detail['equivalent_score']}")
