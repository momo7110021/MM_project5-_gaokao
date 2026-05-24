"""
test_volunteer_optimizer.py
测试模型三：冲稳保志愿组合优化模型
Project 5 要求：硬约束过滤(排斥专业/选科/学费/地域/身体/单科)，
多目标评分，比例约束，三种方案，咨询师可编辑
当前使用模拟数据。
"""
import numpy as np
import pandas as pd
import pytest

from src.volunteer_optimizer import VolunteerOptimizer


def make_candidates(n=80, seed=42):
    """构造模拟候选志愿"""
    np.random.seed(seed)
    rows = []
    for i in range(n):
        p = np.clip(np.random.beta(3, 5) + 0.1, 0.05, 0.95)
        rows.append({
            "school_code": f"10{np.random.randint(1, 21):03d}",
            "school_name": f"院校{np.random.randint(1, 21)}",
            "major_code": f"080{np.random.randint(100, 999):03d}",
            "major_name": f"专业{np.random.randint(1, 30)}",
            "major_group_code": "",
            "P_admit": p,
            "prob_lower": max(0, p - 0.1),
            "prob_upper": min(1, p + 0.1),
            "recommendation_tier": "stable",
            "M_fit": np.random.uniform(0.3, 1),
            "E_career": np.random.uniform(0.2, 0.95),
            "C_city": np.random.uniform(0.3, 0.9),
            "R_family": np.random.uniform(0.3, 1),
            "Risk": np.random.uniform(0, 0.5),
            "U": np.random.uniform(0, 0.6),
            "tuition": np.random.choice([4500, 5000, 8000, 12000, 18000]),
            "city": np.random.choice(["北京", "上海", "深圳", "武汉", "成都"]),
            "subject_requirement": "物理",
            "adjust_danger": np.random.uniform(0, 0.6),
            "review_required": False,
        })
    return pd.DataFrame(rows)


class TestOptimizerOutput:
    """验证优化器输出字段"""

    def test_output_fields(self):
        candidates = make_candidates(80)
        opt = VolunteerOptimizer(n_max=40)
        profile = {"accept_adjustment": 1, "accept_far_city": 1}
        result = opt.optimize(candidates, profile, plan_type="balanced")
        required = [
            "plan_type", "volunteer_table", "rush_count", "stable_count",
            "safe_count", "bottom_count", "overall_score", "total_volunteers",
        ]
        for field in required:
            assert field in result, f"Missing: {field}"

    def test_counts_sum_to_total(self):
        candidates = make_candidates(80)
        opt = VolunteerOptimizer(n_max=40)
        result = opt.optimize(candidates, {}, plan_type="balanced")
        total = sum([result["rush_count"], result["stable_count"],
                      result["safe_count"], result["bottom_count"]])
        assert total <= result["total_volunteers"]


class TestConstraints:
    """硬约束过滤测试"""

    def test_excluded_major_filter(self):
        """排除专业应被过滤"""
        candidates = make_candidates(40)
        candidates.loc[0:2, "major_code"] = "010101"  # 哲学
        candidates.loc[0:2, "major_name"] = "哲学"
        opt = VolunteerOptimizer(n_max=30)
        profile = {"excluded_majors": ["010101"], "accept_adjustment": 1}
        result = opt.optimize(candidates, profile, plan_type="balanced")
        df = result["volunteer_table"]
        if not df.empty and "major_code" in df.columns:
            assert "010101" not in df["major_code"].values, "Excluded major should not appear"

    def test_budget_constraint(self):
        candidates = make_candidates(50)
        candidates.loc[0:5, "tuition"] = 25000  # 超预算
        opt = VolunteerOptimizer(n_max=30)
        profile = {"family_budget": 15000, "accept_adjustment": 1}
        result = opt.optimize(candidates, profile, plan_type="balanced")
        df = result["volunteer_table"]
        if not df.empty and "tuition" in df.columns:
            assert (df["tuition"] <= 15000).all(), "All volunteers should be within budget"

    def test_region_constraint(self):
        candidates = make_candidates(50)
        opt = VolunteerOptimizer(n_max=30)
        profile = {
            "preferred_cities": ["北京", "上海"],
            "accept_far_city": 0,  # 不接受远程
            "accept_adjustment": 1,
        }
        result = opt.optimize(candidates, profile, plan_type="balanced")
        df = result["volunteer_table"]
        if not df.empty and "city" in df.columns:
            assert df["city"].isin(["北京", "上海"]).all(), "Should only have preferred cities"


class TestPlanTypes:
    """三种方案测试"""

    def test_all_plan_types_run(self):
        candidates = make_candidates(80)
        opt = VolunteerOptimizer(n_max=40)
        for pt in ["aggressive", "balanced", "conservative"]:
            result = opt.optimize(candidates, {}, plan_type=pt)
            assert result["plan_type"] == pt
            assert result["rush_count"] + result["stable_count"] + \
                   result["safe_count"] + result["bottom_count"] > 0


class TestEditablePlan:
    """咨询师可编辑志愿表"""

    def test_generate_editable_plan(self):
        candidates = make_candidates(80)
        opt = VolunteerOptimizer(n_max=40)
        result = opt.optimize(candidates, {}, plan_type="balanced")
        df = result["volunteer_table"]
        if not df.empty:
            plan = opt.generate_editable_plan(df)
            assert isinstance(plan, list)
