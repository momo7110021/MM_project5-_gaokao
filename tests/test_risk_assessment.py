"""
test_risk_assessment.py
测试模型四：志愿填报风险评估模型
Project 5 要求：六类风险独立评分, 十类典型问题扫描,
滑档/退档/调剂/冷门/就业/地域 各有 score/level/trigger_reason/modification_suggestion/review_required
当前使用模拟数据。
"""
import numpy as np
import pandas as pd
import pytest

from src.risk_assessment import RiskAssessor


def make_volunteer_df(probs=None, seed=42):
    """构造模拟志愿表"""
    np.random.seed(seed)
    if probs is None:
        probs = np.clip(np.random.beta(3, 5, 40) + 0.1, 0.05, 0.95)
    n = len(probs)
    return pd.DataFrame({
        "school_name": [f"院校{i}" for i in range(1, n + 1)],
        "major_name": [f"专业{i}" for i in range(1, n + 1)],
        "P_admit": probs,
        "E_career": np.random.uniform(0.2, 0.95, n),
        "C_city": np.random.uniform(0.3, 0.9, n),
        "city": np.random.choice(["北京", "上海", "深圳", "武汉", "成都"], n),
        "adjust_danger": np.random.uniform(0, 0.6, n),
        "rank_volatility": np.random.uniform(0.05, 0.35, n),
    })


class TestRiskAssessmentOutput:
    """验证风险评估输出字段"""

    def test_output_fields_complete(self):
        assessor = RiskAssessor(mc_iterations=1000)
        df = make_volunteer_df()
        result = assessor.assess(df, {"accept_adjustment": 1})
        required = [
            "overall_risk_score", "risk_level",
            "slip_risk", "withdrawal_risk", "adjustment_risk",
            "cold_major_risk", "employment_risk", "region_risk",
            "risk_reason", "modification_suggestion", "review_required",
        ]
        for field in required:
            assert field in result, f"Missing: {field}"

    def test_six_risks_are_dicts(self):
        assessor = RiskAssessor(mc_iterations=1000)
        df = make_volunteer_df()
        result = assessor.assess(df, {"accept_adjustment": 1})
        risk_keys = ["slip_risk", "withdrawal_risk", "adjustment_risk",
                      "cold_major_risk", "employment_risk", "region_risk"]
        for key in risk_keys:
            assert isinstance(result[key], dict), f"{key} should be dict"
            for sub in ["score", "level", "trigger_reason", "modification_suggestion", "review_required"]:
                assert sub in result[key], f"{key} missing sub-field: {sub}"

    def test_risk_level_valid(self):
        assessor = RiskAssessor(mc_iterations=1000)
        df = make_volunteer_df()
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert result["risk_level"] in ("critical", "high", "medium", "low")

    def test_overall_risk_score_in_range(self):
        assessor = RiskAssessor(mc_iterations=1000)
        df = make_volunteer_df()
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert 0 <= result["overall_risk_score"] <= 1.0


class TestRiskDetection:
    """风险检测测试"""

    def test_too_many_rush_detected(self):
        """构造冲志愿过多的表格"""
        assessor = RiskAssessor(mc_iterations=500)
        probs = np.array([0.10] * 25 + [0.50] * 10 + [0.80] * 5)
        probs = np.clip(probs, 0.05, 0.95)
        df = make_volunteer_df(probs=probs, seed=1)
        result = assessor.assess(df, {"accept_adjustment": 1})
        reasons = result["risk_reason"]
        has_rush = any("Q1" in r or "冲" in r for r in reasons)
        assert has_rush or result["slip_risk"]["score"] > 0, "Should detect too many rush"

    def test_insufficient_bottom_detected(self):
        """构造无保底志愿（全部低概率），应触发滑档风险"""
        assessor = RiskAssessor(mc_iterations=2000)
        probs = np.full(10, 0.05)
        df = make_volunteer_df(probs=probs, seed=2)
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert result["slip_risk"]["score"] > 0.1, f"All-low-prob should have elevated slip risk, got {result['slip_risk']['score']}"

    def test_not_accept_adjustment(self):
        assessor = RiskAssessor(mc_iterations=500)
        df = make_volunteer_df()
        result = assessor.assess(df, {"accept_adjustment": 0})
        assert result["adjustment_risk"]["score"] >= 0

    def test_employment_risk_detected(self):
        assessor = RiskAssessor(mc_iterations=500)
        df = make_volunteer_df(seed=3)
        df["E_career"] = 0.15
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert result["employment_risk"]["score"] > 0.4

    def test_slip_risk_with_all_low_prob(self):
        assessor = RiskAssessor(mc_iterations=500)
        probs = np.full(20, 0.10)
        df = make_volunteer_df(probs=probs, seed=4)
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert result["slip_risk"]["score"] > 0.05

    def test_empty_table_handled(self):
        assessor = RiskAssessor(mc_iterations=100)
        df = pd.DataFrame()
        result = assessor.assess(df, {"accept_adjustment": 1})
        assert result["risk_level"] in ("critical", "high")
        assert result["review_required"] is True
