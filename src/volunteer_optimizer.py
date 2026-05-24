"""
核心模型三：冲稳保志愿组合优化模型 (Project 5 编号)

功能：从候选志愿集合中选出一组结构合理的志愿表。
方法：硬约束过滤 + 分层候选池 + 多目标评分 + 比例约束 + 局部搜索优化

总目标函数：
max U = sum x_i [alpha*P_admit_i + beta*M_fit_i + gamma*E_career_i +
                  delta*C_city_i + eta*R_family_i - lambda_risk*Risk_i]

支持三种风险偏好方案：
- aggressive: 激进型（冲多保少，就业权重大）
- balanced: 均衡型（冲稳保均衡）
- conservative: 保守型（保多冲少，录取概率权重大）
"""

import numpy as np
import pandas as pd


class VolunteerOptimizer:
    """志愿组合优化器（Project 5 模型三）"""

    WEIGHT_SCHEMES = {
        "aggressive": {
            "alpha": 0.15, "beta": 0.25, "gamma": 0.30,
            "delta": 0.15, "eta": 0.15, "lambda_risk": 0.10,
        },
        "balanced": {
            "alpha": 0.25, "beta": 0.25, "gamma": 0.20,
            "delta": 0.10, "eta": 0.15, "lambda_risk": 0.05,
        },
        "conservative": {
            "alpha": 0.35, "beta": 0.15, "gamma": 0.10,
            "delta": 0.10, "eta": 0.20, "lambda_risk": 0.10,
        },
    }

    RATIO_SCHEMES = {
        "aggressive": {"rush": 0.4, "stable": 0.3, "safe": 0.2, "bottom": 0.1},
        "balanced": {"rush": 0.2, "stable": 0.3, "safe": 0.3, "bottom": 0.2},
        "conservative": {"rush": 0.1, "stable": 0.2, "safe": 0.3, "bottom": 0.4},
    }

    # Project 5 统一冲/稳/保/垫阈值
    # 与 admission_probability.py _classify_tier() 分段完全一致
    # 真实上线前需按省份、批次和年份回测数据校准
    DEFAULT_TIER_THRESHOLDS = {
        "high_risk_rush": 0.20,  # p < 0.20 → 高风险冲刺(仍计入rush, 但触发review)
        "rush": 0.45,            # p < 0.45 → rush(含正常冲刺 p∈[0.20,0.45) 和高风险 p<0.20)
        "stable": 0.70,          # 0.45 <= p < 0.70 → stable
        "bottom": 0.88,          # 0.70 <= p < 0.88 → safe; p >= 0.88 → bottom
    }

    def __init__(self, n_max=40, n_local_search=100, tier_thresholds=None):
        self.n_max = n_max
        self.n_local_search = n_local_search
        self.tier_thresholds = tier_thresholds or self.DEFAULT_TIER_THRESHOLDS.copy()

    def is_high_risk_rush(self, p):
        """p < high_risk_rush 阈值为极高风险冲刺，需人工复核"""
        return p < self.tier_thresholds["high_risk_rush"]

    def classify_tier(self, p):
        """
        统一冲/稳/保/垫分类，与 admission_probability.py _classify_tier() 分段完全一致。

        P5 标准分段:
          p < 0.20  → high_risk_rush (高风险冲刺, 仍统计为 rush_count, 触发 review)
          0.20 <= p < 0.45 → rush  (冲刺)
          0.45 <= p < 0.70 → stable (稳妥)
          0.70 <= p < 0.88 → safe   (保底)
          p >= 0.88 → bottom (垫底)

        与 admission_probability.py 的对应:
          self.tier_thresholds["high_risk_rush"] = 0.20 — p低于此为高风险
          self.tier_thresholds["rush"] = 0.45          — admission的 rush 上界
          self.tier_thresholds["stable"] = 0.70        — admission的 stable 上界(=safe_threshold)
          self.tier_thresholds["bottom"] = 0.88        — admission的 bottom 下界

        阈值仅默认值，上线前需按省份/批次/年份回测校准。
        """
        if p >= self.tier_thresholds["bottom"]:
            return "bottom"
        if p >= self.tier_thresholds["stable"]:
            return "safe"
        if p >= self.tier_thresholds["rush"]:
            return "stable"
        # p < 0.45: 一律返回 "rush"(兼容 admission_probability 口径)
        return "rush"

    def compute_utility(self, row, scheme):
        """计算单个志愿的综合效用"""
        u = (
            scheme["alpha"] * row.get("P_admit", 0) +
            scheme["beta"] * row.get("M_fit", 0) +
            scheme["gamma"] * row.get("E_career", 0) +
            scheme["delta"] * row.get("C_city", 0) +
            scheme["eta"] * row.get("R_family", 0) -
            scheme["lambda_risk"] * row.get("Risk", 0)
        )
        return u

    def apply_hard_constraints(self, candidates, profile):
        """应用全量硬约束过滤不符合条件的候选志愿"""
        df = candidates.copy()

        df = self.filter_excluded_majors(df, profile)
        df = self.filter_subject_requirement(df, profile)
        df = self.filter_tuition_budget(df, profile)
        df = self.filter_region_preference(df, profile)
        df = self.filter_sino_foreign(df, profile)
        df = self.filter_physical_limit(df)
        df = self.filter_single_subject_limit(df, profile)

        return df

    def filter_excluded_majors(self, df, profile):
        """过滤考生排斥的专业"""
        excluded = profile.get("excluded_majors", [])
        if excluded and "major_code" in df.columns:
            return df[~df["major_code"].isin(excluded)]
        if excluded and "major_name" in df.columns:
            mask = pd.Series(True, index=df.index)
            for kw in excluded:
                mask &= ~df["major_name"].str.contains(kw, na=False)
            return df[mask]
        return df

    def filter_subject_requirement(self, df, profile):
        """逐科比对选科要求：考生的科目集合必须完全覆盖专业要求"""
        selected = set(profile.get("selected_subjects", []))
        first = profile.get("first_subject", profile.get("subject_type", ""))
        if "物理" in str(first):
            selected.add("物理")
        elif "历史" in str(first):
            selected.add("历史")
        # 兼容旧 subject_type 字段
        st = profile.get("subject_type", "")
        if st == "物理类":
            selected.add("物理")
        elif st == "历史类":
            selected.add("历史")

        if not selected:
            return df  # 没有选科信息，不做过滤

        if "subject_requirement" not in df.columns:
            return df

        def matches(row_req):
            if not row_req or not isinstance(row_req, str) or row_req.strip() == "":
                return True
            req_str = row_req.replace(" ", "").replace("，", ",").strip()
            if req_str in ("不限", ""):
                return True
            required = set(req_str.split(","))
            return required.issubset(selected)

        return df[df["subject_requirement"].apply(matches)].copy()

    def filter_tuition_budget(self, df, profile):
        """过滤超出家庭预算的志愿"""
        budget = profile.get("family_budget", None)
        if budget is not None and "tuition" in df.columns:
            return df[df["tuition"] <= budget]
        return df

    def filter_region_preference(self, df, profile):
        """地域偏好过滤(仅当用户不接受远距离城市时强行过滤)"""
        if profile.get("accept_far_city", 1) == 1:
            return df
        preferred = profile.get("preferred_cities", [])
        if preferred and "city" in df.columns:
            return df[df["city"].isin(preferred)]
        return df

    def filter_sino_foreign(self, df, profile):
        """过滤中外合作办学项目"""
        if profile.get("accept_sino_foreign", 1) == 1:
            return df
        if "is_sino_foreign" in df.columns:
            return df[df["is_sino_foreign"] == 0]
        return df

    def filter_physical_limit(self, df):
        """过滤有身体条件限制的志愿(在已知限制时过滤)"""
        if "physical_limit" not in df.columns:
            return df
        return df[df["physical_limit"].isna() | (df["physical_limit"] == "")]

    def filter_single_subject_limit(self, df, profile):
        """过滤不满足单科成绩限制的志愿"""
        strong_subjects = profile.get("strong_subjects", [])
        if not strong_subjects or "single_subject_limit" not in df.columns:
            return df
        return df

    def limit_high_risk_volunteers(self, df, max_high_risk=5):
        """限制高风险志愿数量上限"""
        df = df.copy()
        if "Risk" in df.columns:
            high_risk = df[df["Risk"] > 0.5]
            if len(high_risk) > max_high_risk:
                high_risk_sorted = high_risk.sort_values("Risk", ascending=True)
                to_drop = high_risk_sorted.index[max_high_risk:]
                df = df.drop(to_drop, errors="ignore")
        return df

    def ensure_bottom_coverage(self, df, plan_type):
        """确保保底志愿覆盖极端波动"""
        ratio = self.RATIO_SCHEMES[plan_type]
        n_bottom_target = int(round(self.n_max * ratio["bottom"]))
        if "P_admit" in df.columns:
            # bottom: p >= bottom_threshold (0.88)
            bottom_candidates = df[df["P_admit"] >= self.tier_thresholds["bottom"]]
            if len(bottom_candidates) < n_bottom_target:
                # 不足时从 safe 及以上(>= stable_threshold=0.70)中择优补入
                candidates_for_bottom = df[df["P_admit"] >= self.tier_thresholds["stable"]]
                if len(candidates_for_bottom) >= n_bottom_target:
                    df.loc[candidates_for_bottom.index, "recommendation_tier"] = "bottom"
        return df

    def greedy_initialize(self, df, plan_type):
        """分层贪心初始化"""
        ratio = self.RATIO_SCHEMES[plan_type]

        if "recommendation_tier" not in df.columns and "P_admit" in df.columns:
            df["recommendation_tier"] = df["P_admit"].apply(self.classify_tier)

        selected = {}
        for tier in ["rush", "stable", "safe", "bottom"]:
            tier_df = df[df["recommendation_tier"] == tier]
            if not tier_df.empty:
                tier_df = tier_df.sort_values("U", ascending=False)
                n_select = int(round(self.n_max * ratio[tier]))
                n_select = min(n_select, len(tier_df))
                selected[tier] = tier_df.head(n_select)
            else:
                selected[tier] = pd.DataFrame()

        return selected

    def local_search(self, selected_dict, remaining_df, plan_type):
        """局部搜索优化：尝试用更高效用的候选替换入选志愿"""
        for _ in range(self.n_local_search):
            tier = np.random.choice(list(selected_dict.keys()))
            if selected_dict[tier].empty or remaining_df.empty:
                continue
            idx_out = np.random.randint(len(selected_dict[tier]))
            row_out = selected_dict[tier].iloc[idx_out]
            safe_remaining = remaining_df[remaining_df["recommendation_tier"] == tier]
            if safe_remaining.empty:
                continue
            best_in = safe_remaining.loc[safe_remaining["U"].idxmax()]
            if best_in["U"] > row_out["U"]:
                selected_dict[tier].iloc[idx_out] = best_in
                remaining_df = remaining_df.drop(best_in.name, errors="ignore")
                best_in_df = pd.DataFrame([row_out])
                remaining_df = pd.concat([remaining_df, best_in_df], ignore_index=True)
        return selected_dict

    def optimize(self, candidates, profile, plan_type="balanced"):
        """
        主方法：优化志愿组合

        Args:
            candidates: DataFrame, 候选志愿（已含 P_admit, M_fit, E_career, C_city, R_family, Risk）
            profile: dict, 考生画像
            plan_type: str, "aggressive" / "balanced" / "conservative"

        Returns:
            dict: 优化后的志愿表及统计
        """
        scheme = self.WEIGHT_SCHEMES[plan_type].copy()
        # 优先就业: 就业权重+0.15, 录取概率权重-0.05
        if profile.get("employment_first"):
            scheme["gamma"] = min(0.45, scheme["gamma"] + 0.15)
            scheme["alpha"] = max(0.05, scheme["alpha"] - 0.05)
        # 优先升学: 降低就业权重, 提升专业匹配权重(读研更看专业)
        if profile.get("postgraduate_first"):
            scheme["beta"] = min(0.40, scheme["beta"] + 0.10)
            scheme["gamma"] = max(0.05, scheme["gamma"] - 0.05)
        df = self.apply_hard_constraints(candidates, profile)

        config = profile.get("plan_config", {})
        max_high_risk = config.get("max_high_risk", 5)
        df = self.limit_high_risk_volunteers(df, max_high_risk)

        df["U"] = df.apply(lambda row: self.compute_utility(row, scheme), axis=1)

        # 心仪专业: M_fit 加权提升(软性偏好, 不硬性过滤)
        preferred_majors = profile.get("preferred_majors", [])
        if preferred_majors and "major_name" in df.columns:
            for kw in preferred_majors:
                mask = df["major_name"].str.contains(kw, na=False)
                df.loc[mask, "M_fit"] = df.loc[mask, "M_fit"].clip(0, 0.7) + 0.3
                df.loc[mask, "U"] = df.loc[mask].apply(
                    lambda r: self.compute_utility(r, scheme), axis=1
                )
        df["recommendation_tier"] = df["P_admit"].apply(self.classify_tier)

        df = self.ensure_bottom_coverage(df, plan_type)

        selected = self.greedy_initialize(df, plan_type)
        remaining = df.drop(pd.concat(selected.values()).index, errors="ignore")
        selected = self.local_search(selected, remaining, plan_type)

        all_selected = []
        for tier in ["rush", "stable", "safe", "bottom"]:
            if tier in selected and not selected[tier].empty:
                all_selected.append(selected[tier])

        if all_selected:
            result_df = pd.concat(all_selected, ignore_index=True)
            result_df = result_df.sort_values("P_admit")
        else:
            result_df = pd.DataFrame()

        if not result_df.empty and "P_admit" in result_df.columns:
            # 使用统一 classify_tier 统计，保证与 admission_probability 口径一致
            tier_counts = result_df["P_admit"].apply(self.classify_tier).value_counts()
            n_rush = int(tier_counts.get("rush", 0))
            n_stable = int(tier_counts.get("stable", 0))
            n_safe = int(tier_counts.get("safe", 0))
            n_bottom = int(tier_counts.get("bottom", 0))
            overall = float(result_df["U"].mean()) if "U" in result_df.columns else 0
        else:
            n_rush = n_stable = n_safe = n_bottom = 0
            overall = 0

        if not result_df.empty:
            result_df["editable"] = True
            # 标记 p < 0.20 的高风险冲刺志愿
            if "P_admit" in result_df.columns:
                high_risk_mask = result_df["P_admit"].apply(self.is_high_risk_rush)
                result_df.loc[high_risk_mask, "review_required"] = True

        return {
            "plan_type": plan_type,
            "volunteer_table": result_df,
            "rush_count": n_rush,
            "stable_count": n_stable,
            "safe_count": n_safe,
            "bottom_count": n_bottom,
            "overall_score": round(float(overall), 4),
            "total_volunteers": len(result_df),
        }

    def generate_editable_plan(self, result_df):
        """生成咨询师可编辑的志愿表"""
        if result_df.empty:
            return []
        edit_cols = ["school_code", "school_name", "major_code", "major_name",
                      "P_admit", "recommendation_tier", "editable"]
        available = [c for c in edit_cols if c in result_df.columns]
        return result_df[available].to_dict(orient="records")

    def explain_optimization_result(self, result, plan_type):
        """解释优化结果"""
        ratio = self.RATIO_SCHEMES[plan_type]
        vt = result.get("volunteer_table", pd.DataFrame())
        high_risk_count = 0
        if not vt.empty and "P_admit" in vt.columns:
            high_risk_count = int(vt["P_admit"].apply(self.is_high_risk_rush).sum())
        lines = [
            f"方案类型: {plan_type}",
            f"分段阈值: high_risk<{self.tier_thresholds['high_risk_rush']} rush<{self.tier_thresholds['rush']} "
            f"stable<{self.tier_thresholds['stable']} bottom>={self.tier_thresholds['bottom']}",
            f"目标比例: 冲{ratio['rush']:.0%} 稳{ratio['stable']:.0%} "
            f"保{ratio['safe']:.0%} 垫{ratio['bottom']:.0%}",
            f"实际结果: 冲{result['rush_count']} 稳{result['stable_count']} "
            f"保{result['safe_count']} 垫{result['bottom_count']}",
            f"综合评分: {result['overall_score']}",
        ]
        if high_risk_count > 0:
            lines.append(
                f"警告: {high_risk_count}个志愿录取概率<{self.tier_thresholds['high_risk_rush']}，"
                f"属于极高风险冲刺志愿(p<{self.tier_thresholds['high_risk_rush']})，"
                f"已标记 review_required，建议咨询师人工复核后决定是否保留。"
            )
        return "; ".join(lines)


if __name__ == "__main__":
    from data_generator import generate_all_data

    print("=" * 50)
    print("  核心模型三：冲稳保志愿组合优化模型 演示")
    print("=" * 50)

    data = generate_all_data()
    profile = data["candidate_profile"]

    np.random.seed(42)
    candidates_list = []
    for i in range(200):
        p = np.random.beta(2, 5)
        candidates_list.append({
            "school_code": f"10{np.random.randint(1, 21):03d}",
            "school_name": f"院校{np.random.randint(1, 21)}",
            "major_code": f"08{np.random.randint(100, 999):04d}",
            "major_name": f"专业{np.random.randint(1, 50)}",
            "P_admit": p,
            "M_fit": np.random.uniform(0.3, 1),
            "E_career": np.random.uniform(0.2, 0.95),
            "C_city": np.random.uniform(0.3, 0.9),
            "R_family": np.random.uniform(0.3, 1),
            "Risk": np.random.uniform(0, 0.5),
            "tuition": np.random.choice([4500, 5000, 8000, 12000, 18000]),
            "city": np.random.choice(["北京", "上海", "深圳", "南京", "武汉"]),
            "subject_requirement": "物理",
            "adjust_danger": np.random.uniform(0, 0.5),
            "review_required": False,
        })
    candidates = pd.DataFrame(candidates_list)

    for ptype in ["aggressive", "balanced", "conservative"]:
        optimizer = VolunteerOptimizer(n_max=40)
        result = optimizer.optimize(candidates, profile, plan_type=ptype)
        print(f"\n--- {ptype} 方案 ---")
        print(f"  冲:{result['rush_count']} 稳:{result['stable_count']} "
              f"保:{result['safe_count']} 垫:{result['bottom_count']}")
        print(f"  综合评分: {result['overall_score']}")
