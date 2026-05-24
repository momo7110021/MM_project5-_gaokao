"""
核心模型四：志愿填报风险评估模型 (Project 5 编号)

功能：评估志愿方案中的六类风险并给出修改建议。
方法：蒙特卡洛模拟 + 风险矩阵 + 十大典型问题自动识别

六类风险：
1. 滑档风险(slip_risk) - 所有志愿均未被录取的概率
2. 退档风险(withdrawal_risk) - 因体检/单科不达标被退档的概率
3. 调剂风险(adjustment_risk) - 被调剂到非期望专业的概率
4. 专业冷门风险(cold_major_risk) - 录取专业就业前景差的概率
5. 就业风险(employment_risk) - 整体志愿表就业前景低于预期的风险
6. 地域风险(region_risk) - 录取院校所在城市不符合偏好的风险
"""

import numpy as np
import pandas as pd


class RiskAssessor:
    """志愿填报风险评估器（Project 5 模型四）"""

    RISK_WEIGHTS = {
        "slip": 0.30,
        "withdrawal": 0.25,
        "adjustment": 0.20,
        "cold_major": 0.10,
        "employment": 0.10,
        "region": 0.05,
    }

    def __init__(self, mc_iterations=10000):
        self.mc_iterations = mc_iterations

    def _slip_risk(self, volunteer_df):
        """滑档风险：蒙特卡洛模拟所有志愿同时失败的概率"""
        if volunteer_df.empty:
            return 1.0, True, "志愿表为空，滑档风险极高"
        probs = volunteer_df["P_admit"].values
        n = len(probs)
        if n == 0:
            return 1.0, True, "志愿表为空，滑档风险极高"
        simulations = np.random.random((self.mc_iterations, n)) < probs
        any_success = simulations.any(axis=1)
        r_slip = float(1 - np.mean(any_success))
        need_warning = r_slip > 0.05
        reason = f"蒙特卡洛模拟({self.mc_iterations}次): 滑档概率{r_slip:.2%}" if need_warning else ""
        return r_slip, need_warning, reason

    def _withdrawal_risk(self, volunteer_df):
        """退档风险：基于超录比例和招生章程限制"""
        if volunteer_df.empty:
            return 0.5, True, "志愿表为空"
        score = 0.02
        if "admission_count" in volunteer_df.columns and "plan_count" in volunteer_df.columns:
            ratios = (volunteer_df["admission_count"] - volunteer_df["plan_count"]) / volunteer_df["admission_count"].clip(lower=1)
            score = float(max(ratios.max(), 0) + 0.02)
        has_physical_limit = ("physical_limit" in volunteer_df.columns and volunteer_df["physical_limit"].notna().any())
        if has_physical_limit:
            score += 0.05
        r_withdrawal = min(1.0, max(0, score))
        need_warning = r_withdrawal > 0.1
        reason = ("超录比例过大(退档风险)" if "admission_count" in volunteer_df.columns else "建议检查各志愿招生章程中的退档条款") if need_warning else ""
        return r_withdrawal, need_warning, reason

    def _adjustment_risk(self, volunteer_df, profile):
        """调剂风险：考虑专业组冷热差异和不接受调剂"""
        if volunteer_df.empty:
            return 0.5, True, "志愿表为空"
        if "adjust_danger" in volunteer_df.columns:
            r_adjust = float(volunteer_df["adjust_danger"].mean())
        else:
            low_prob_ratio = (volunteer_df["P_admit"] < 0.40).mean()
            r_adjust = low_prob_ratio * 0.5
        if profile.get("accept_adjustment", 1) == 0:
            r_adjust *= 1.5
        r_adjust = min(1.0, r_adjust)
        need_warning = r_adjust > 0.4 or (profile.get("accept_adjustment", 1) == 0 and r_adjust > 0.2)
        reason = ""
        if need_warning:
            if profile.get("accept_adjustment", 1) == 0:
                reason = "考生不接受调剂, 但部分志愿调剂风险较高"
            else:
                reason = "专业组内部冷热差异大,调剂风险较高"
        return r_adjust, need_warning, reason

    def _cold_major_risk(self, volunteer_df):
        """专业冷门风险：录取概率加权后的就业评分"""
        if volunteer_df.empty:
            return 0.5, True, "志愿表为空"
        if "E_career" in volunteer_df.columns and "P_admit" in volunteer_df.columns:
            p_sum = volunteer_df["P_admit"].sum()
            if p_sum > 0:
                weights = volunteer_df["P_admit"] / p_sum
                r_cold = float(1 - np.average(volunteer_df["E_career"], weights=weights))
            else:
                r_cold = float(1 - volunteer_df["E_career"].mean())
        else:
            r_cold = 0.2
        need_warning = r_cold > 0.5
        reason = f"志愿表中冷门专业加权平均就业评分低于{'%.0f'%(1-r_cold)}" if need_warning else ""
        return r_cold, need_warning, reason

    def _employment_risk(self, volunteer_df):
        """就业风险：整体就业评分评估"""
        if volunteer_df.empty:
            return 0.5, True, "志愿表为空"
        if "E_career" in volunteer_df.columns:
            r_emp = float(1 - volunteer_df["E_career"].mean())
        else:
            r_emp = 0.3
        need_warning = r_emp > 0.4
        reason = f"志愿表整体就业评分偏低({1-r_emp:.2f})" if need_warning else ""
        return r_emp, need_warning, reason

    def _region_risk(self, volunteer_df, profile):
        """地域风险：城市偏好匹配度"""
        if volunteer_df.empty:
            return 0.5, True, "志愿表为空"
        if "C_city" in volunteer_df.columns:
            r_region = float(1 - volunteer_df["C_city"].mean())
        else:
            r_region = 0.3
        preferred = profile.get("preferred_cities", [])
        if preferred and "city" in volunteer_df.columns:
            coverage = volunteer_df["city"].isin(preferred).mean()
            if coverage < 0.2:
                r_region += 0.2
                r_region = min(1.0, r_region)
        need_warning = r_region > 0.4
        reason = f"偏好城市覆盖率仅{coverage:.0%}" if (need_warning and preferred) else ""
        return r_region, need_warning, reason

    def scan_too_many_rush(self, volunteer_df):
        """Q1: 冲志愿过多"""
        if volunteer_df.empty:
            return None
        n_max = len(volunteer_df)
        n_rush = int((volunteer_df["P_admit"] < 0.30).sum())
        if n_rush / n_max > 0.5:
            return f"Q1: 冲志愿过多({n_rush}/{n_max}={n_rush/n_max:.0%})，可能导致录取概率不足"
        return None

    def scan_too_few_stable(self, volunteer_df):
        """Q2: 稳志愿过少"""
        if volunteer_df.empty:
            return None
        n_max = len(volunteer_df)
        n_stable = int(((volunteer_df["P_admit"] >= 0.30) & (volunteer_df["P_admit"] < 0.60)).sum())
        if n_stable < n_max * 0.2 and n_max > 0:
            return f"Q2: 稳志愿过少({n_stable}/{n_max}，不足20%)，志愿表梯度不合理"
        return None

    def scan_insufficient_bottom(self, volunteer_df, slip_risk):
        """Q3: 保底志愿不足"""
        if slip_risk > 0.05:
            return f"Q3: 保底志愿不足，滑档风险为{slip_risk:.1%}"
        return None

    def scan_major_group_hot_cold_gap(self, volunteer_df):
        """Q4: 专业组冷热差异过大"""
        if volunteer_df.empty or "adjust_danger" not in volunteer_df.columns:
            return None
        high_adj = volunteer_df[volunteer_df["adjust_danger"] > 0.7]
        if len(high_adj) > 0:
            return f"Q4: {len(high_adj)}个志愿专业组冷热差异过大（调剂危险度>0.7）"
        return None

    def scan_excluded_major_included(self, volunteer_df, profile):
        """Q5: 用户排斥专业被纳入"""
        excluded = profile.get("excluded_majors", [])
        if not excluded or "major_code" not in volunteer_df.columns:
            return None
        included = volunteer_df[volunteer_df["major_code"].isin(excluded)]
        if len(included) > 0:
            return f"Q5: {len(included)}个排斥专业被纳入志愿表: {list(included['major_code'])}"
        return None

    def scan_employment_risk_missing(self, volunteer_df):
        """Q6: 就业风险未提示"""
        if volunteer_df.empty:
            return None
        if "E_career" in volunteer_df.columns:
            low_emp = volunteer_df[volunteer_df["E_career"] < 0.3]
            if len(low_emp) > 0 and len(low_emp) >= len(volunteer_df) * 0.3:
                return f"Q6: {len(low_emp)}个专业就业评分<0.3且占{len(low_emp)/len(volunteer_df):.0%}，需就业风险提示"
        return None

    def scan_region_constraint_ignored(self, volunteer_df, profile):
        """Q7: 地域约束被忽略"""
        preferred = profile.get("preferred_cities", [])
        if not preferred or "city" not in volunteer_df.columns or volunteer_df.empty:
            return None
        coverage = volunteer_df["city"].isin(preferred).mean()
        if coverage < 0.3:
            return f"Q7: 偏好城市覆盖率仅{coverage:.0%}，地域约束可能被忽略"
        return None

    def scan_historical_volatility(self, volunteer_df):
        """Q8: 历史波动过大"""
        if volunteer_df.empty:
            return None
        vol_col = "rank_volatility"
        if vol_col in volunteer_df.columns:
            high_vol = volunteer_df[pd.to_numeric(volunteer_df[vol_col], errors="coerce") > 0.25]
            if len(high_vol) > 0:
                return f"Q8: {len(high_vol)}个志愿历史位次波动率>0.25，录取不确定性高"
        return None

    def scan_small_sample_overconfidence(self, volunteer_df):
        """Q9: 小样本专业概率过度乐观"""
        if volunteer_df.empty:
            return None
        if "review_required" in volunteer_df.columns:
            small = volunteer_df[volunteer_df["review_required"] == True]
            high_p = small[small["P_admit"] > 0.5] if "P_admit" in small.columns else pd.DataFrame()
            if len(high_p) > 0:
                return f"Q9: {len(high_p)}个小样本专业概率估计过度乐观(>50%)，建议扩充参考数据"
        return None

    def scan_adjustment_risk_warning(self, volunteer_df, profile):
        """Q10: 调剂风险高但未规避"""
        if profile.get("accept_adjustment", 1) == 0:
            return None
        if volunteer_df.empty:
            return None
        if "adjust_danger" in volunteer_df.columns:
            high = volunteer_df[volunteer_df["adjust_danger"] > 0.7]
            if len(high) > 3:
                return f"Q10: {len(high)}个志愿调剂风险>0.7且接受调剂，建议调整"
        return None

    def scan_typical_problems(self, volunteer_df, risk_scores, profile):
        """十大典型问题综合扫描"""
        problems = []
        scanners = [
            lambda: self.scan_too_many_rush(volunteer_df),
            lambda: self.scan_too_few_stable(volunteer_df),
            lambda: self.scan_insufficient_bottom(volunteer_df, risk_scores.get("slip", 0)),
            lambda: self.scan_major_group_hot_cold_gap(volunteer_df),
            lambda: self.scan_excluded_major_included(volunteer_df, profile),
            lambda: self.scan_employment_risk_missing(volunteer_df),
            lambda: self.scan_region_constraint_ignored(volunteer_df, profile),
            lambda: self.scan_historical_volatility(volunteer_df),
            lambda: self.scan_small_sample_overconfidence(volunteer_df),
            lambda: self.scan_adjustment_risk_warning(volunteer_df, profile),
        ]
        for scanner in scanners:
            result = scanner()
            if result:
                problems.append(result)
        return problems

    def _build_suggestions(self, slips, wds, adjs, clds, emps, rgis, profile):
        """根据各分项风险构建修改建议"""
        suggestions = []
        if slips[1]:
            suggestions.append("增加至少2个'保'或'垫'级别志愿，降低滑档风险")
        if wds[1]:
            suggestions.append("检查各志愿招生章程，确认满足单科成绩、身体条件等要求")
        if adjs[1]:
            if profile.get("accept_adjustment", 1) == 0:
                suggestions.append("您选择了不接受调剂，建议替换专业组内冷热差异大的志愿")
            else:
                suggestions.append("部分志愿专业组内冷热差异大，建议关注调剂规则")
        if clds[1]:
            suggestions.append("志愿表中包含冷门风险较高的专业，请确认是否接受")
        if emps[1]:
            suggestions.append("志愿表中部分专业就业前景需关注，如就业优先请重新考虑")
        if rgis[1]:
            suggestions.append("较多志愿不在偏好城市，如对地域敏感请调整")
        return suggestions

    def assess(self, volunteer_df, profile):
        """
        主方法：综合风险评估

        Args:
            volunteer_df: DataFrame, 志愿表
            profile: dict, 考生画像

        Returns:
            dict: 风险评分和修改建议
        """
        slips = self._slip_risk(volunteer_df)
        wds = self._withdrawal_risk(volunteer_df)
        adjs = self._adjustment_risk(volunteer_df, profile)
        clds = self._cold_major_risk(volunteer_df)
        emps = self._employment_risk(volunteer_df)
        rgis = self._region_risk(volunteer_df, profile)

        risk_total = (
            self.RISK_WEIGHTS["slip"] * slips[0] +
            self.RISK_WEIGHTS["withdrawal"] * wds[0] +
            self.RISK_WEIGHTS["adjustment"] * adjs[0] +
            self.RISK_WEIGHTS["cold_major"] * clds[0] +
            self.RISK_WEIGHTS["employment"] * emps[0] +
            self.RISK_WEIGHTS["region"] * rgis[0]
        )

        if risk_total > 0.50:
            risk_level = "critical"
        elif risk_total > 0.30:
            risk_level = "high"
        elif risk_total > 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        risk_scores_summary = {
            "slip": slips[0], "withdrawal": wds[0],
            "adjustment": adjs[0], "cold_major": clds[0],
            "employment": emps[0], "region": rgis[0],
        }
        problems = self.scan_typical_problems(volunteer_df, risk_scores_summary, profile)
        suggestions = self._build_suggestions(slips, wds, adjs, clds, emps, rgis, profile)

        return {
            "overall_risk_score": round(risk_total, 4),
            "risk_level": risk_level,
            "slip_risk": {
                "score": round(slips[0], 4), "level": "high" if slips[0] > 0.05 else "low",
                "trigger_reason": slips[2], "modification_suggestion": suggestions[0] if suggestions and slips[1] else "",
                "review_required": slips[1],
            },
            "withdrawal_risk": {
                "score": round(wds[0], 4), "level": "high" if wds[0] > 0.1 else "low",
                "trigger_reason": wds[2], "modification_suggestion": suggestions[1] if len(suggestions) > 1 and wds[1] else "",
                "review_required": wds[1],
            },
            "adjustment_risk": {
                "score": round(adjs[0], 4), "level": "high" if adjs[0] > 0.4 else "medium" if adjs[0] > 0.2 else "low",
                "trigger_reason": adjs[2], "modification_suggestion": suggestions[2] if len(suggestions) > 2 and adjs[1] else "",
                "review_required": adjs[1],
            },
            "cold_major_risk": {
                "score": round(clds[0], 4), "level": "high" if clds[0] > 0.5 else "low",
                "trigger_reason": clds[2], "modification_suggestion": suggestions[3] if len(suggestions) > 3 and clds[1] else "",
                "review_required": clds[1],
            },
            "employment_risk": {
                "score": round(emps[0], 4), "level": "high" if emps[0] > 0.4 else "low",
                "trigger_reason": emps[2], "modification_suggestion": suggestions[4] if len(suggestions) > 4 and emps[1] else "",
                "review_required": emps[1],
            },
            "region_risk": {
                "score": round(rgis[0], 4), "level": "high" if rgis[0] > 0.4 else "low",
                "trigger_reason": rgis[2], "modification_suggestion": suggestions[5] if len(suggestions) > 5 and rgis[1] else "",
                "review_required": rgis[1],
            },
            "risk_reason": problems,
            "modification_suggestion": "; ".join(suggestions) if suggestions else "无需修改",
            "review_required": risk_total > 0.20,
        }


if __name__ == "__main__":
    print("=" * 50)
    print("  核心模型四：志愿填报风险评估模型 演示")
    print("=" * 50)

    np.random.seed(42)
    volunteer_data = {
        "school_name": [f"院校{i}" for i in range(1, 41)],
        "major_name": [f"专业{i}" for i in range(1, 41)],
        "P_admit": np.clip(np.random.beta(3, 5, 40) + 0.1, 0.05, 0.95),
        "E_career": np.random.uniform(0.2, 0.95, 40),
        "C_city": np.random.uniform(0.3, 0.9, 40),
        "city": np.random.choice(["北京", "上海", "深圳", "武汉", "成都"], 40),
        "adjust_danger": np.random.uniform(0, 0.6, 40),
        "rank_volatility": np.random.uniform(0.05, 0.35, 40),
    }
    volunteer_df = pd.DataFrame(volunteer_data)

    profile = {
        "accept_adjustment": 0,
        "preferred_cities": ["北京", "上海", "深圳"],
        "excluded_majors": [],
    }

    assessor = RiskAssessor(mc_iterations=5000)
    result = assessor.assess(volunteer_df, profile)

    print(f"\n综合风险评分: {result['overall_risk_score']}")
    print(f"风险等级: {result['risk_level']}")
    print(f"需人工复核: {result['review_required']}")
    print(f"\n各分项风险:")
    for k in ["slip_risk", "withdrawal_risk", "adjustment_risk", "cold_major_risk", "employment_risk", "region_risk"]:
        info = result[k]
        print(f"  {k}: score={info['score']:.4f}, level={info['level']}, trigger={bool(info['trigger_reason'])}")

    print(f"\n识别的问题: ({len(result['risk_reason'])}条)")
    for r in result["risk_reason"]:
        print(f"  - {r}")

    print(f"\n修改建议:\n{result['modification_suggestion']}")
