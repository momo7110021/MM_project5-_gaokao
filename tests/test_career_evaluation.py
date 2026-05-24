"""
test_career_evaluation.py
测试模型五：专业就业景气度评价模型
Project 5 要求：AHP+熵权+TOPSIS, 红黄绿标签, 细分评分, 社媒舆情仅辅助,
薪资口径说明, 三类解释
当前使用模拟数据。
"""
import numpy as np
import pandas as pd
import pytest

from src.career_evaluation import CareerEvaluator
from src.data_generator import generate_major_employment


class TestCareerEvaluationOutput:
    """验证就业评价输出字段"""

    def test_evaluate_returns_dataframe(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_batch_output_fields(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        required = [
            "career_score", "employment_level", "salary_score",
            "stability_score", "growth_score", "postgraduate_value_score",
            "civil_service_score", "cold_major_risk_score", "major_risk_label",
            "red_yellow_green_label", "trend_label",
            "data_reliability_level", "review_required",
        ]
        for field in required:
            assert field in result.columns, f"Missing field: {field}"

    def test_evaluate_major_output_fields(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        row = result[result["major_code"] == "080901"].iloc[0]
        single = ev.evaluate_major(row)
        required = [
            "career_score", "employment_level", "salary_score",
            "stability_score", "growth_score", "postgraduate_value_score",
            "civil_service_score", "cold_major_risk_score",
            "major_risk_label", "red_yellow_green_label",
            "salary_source_explanation", "data_reliability_level",
            "explanation", "consultant_explanation", "backend_explanation",
            "review_required",
        ]
        for field in required:
            assert field in single, f"Missing field in single: {field}"

    def test_career_score_range(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        assert result["career_score"].between(0, 1).all()
        assert result["salary_score"].between(0, 1).all()
        assert result["stability_score"].between(0, 1).all()
        assert result["growth_score"].between(0, 1).all()

    def test_red_yellow_green_label_valid(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        labels = result["red_yellow_green_label"].unique()
        for label in labels:
            assert label in ("green", "yellow", "red")


class TestSentimentNotCore:
    """舆情不参与核心评分"""

    def test_sentiment_not_in_core_indicators(self):
        ev = CareerEvaluator()
        assert "sentiment_warning_score" not in ev.INDICATORS_CORE
        assert ev.INDICATOR_AUX == "sentiment_warning_score"

    def test_sentiment_only_triggers_warning(self):
        """高舆情但核心指标好，career_score 不应大幅下降"""
        ev = CareerEvaluator()
        df = pd.DataFrame([
            {"major_code": "A", "major_name": "Good", "employment_rate": 0.95,
             "postgraduate_rate": 0.4, "average_salary": 10000, "median_salary": 9500,
             "job_count": 50000, "job_growth_rate": 0.1, "industry_growth_score": 9,
             "stability_score": 8, "civil_service_post_count": 200,
             "sentiment_warning_score": 0.9, "data_year": 2023},
            {"major_code": "B", "major_name": "GoodLowSent", "employment_rate": 0.95,
             "postgraduate_rate": 0.4, "average_salary": 10000, "median_salary": 9500,
             "job_count": 50000, "job_growth_rate": 0.1, "industry_growth_score": 9,
             "stability_score": 8, "civil_service_post_count": 200,
             "sentiment_warning_score": 0.05, "data_year": 2023},
        ])
        result = ev.evaluate(df)
        score_a = result[result["major_code"] == "A"]["career_score"].iloc[0]
        score_b = result[result["major_code"] == "B"]["career_score"].iloc[0]
        # 两个专业的 career_score 应接近（sentiment 不参与 TOPSIS）
        assert abs(score_a - score_b) < 0.15


class TestHighRiskMajor:
    """高风险专业测试"""

    def test_low_employment_red_label(self):
        ev = CareerEvaluator()
        df = pd.DataFrame([
            {"major_code": "Z", "major_name": "Bad", "employment_rate": 0.60,
             "postgraduate_rate": 0.1, "average_salary": 3000, "median_salary": 2800,
             "job_count": 2000, "job_growth_rate": -0.15, "industry_growth_score": 2,
             "stability_score": 3, "civil_service_post_count": 20,
             "sentiment_warning_score": 0.6, "data_year": 2023},
            {"major_code": "X", "major_name": "Med", "employment_rate": 0.80,
             "postgraduate_rate": 0.2, "average_salary": 5000, "median_salary": 4800,
             "job_count": 8000, "job_growth_rate": 0.0, "industry_growth_score": 5,
             "stability_score": 5, "civil_service_post_count": 100,
             "sentiment_warning_score": 0.2, "data_year": 2023},
        ])
        result = ev.evaluate(df)
        bad = result[result["major_code"] == "Z"].iloc[0]
        med = result[result["major_code"] == "X"].iloc[0]
        assert bad["career_score"] < med["career_score"], "Low employment major should score lower"

        # 高风险专业 review_required 为 True
        single = ev.evaluate_major(bad)
        assert single["review_required"] is True


class TestSalaryExplanation:
    """薪资来源说明测试"""

    def test_salary_source_is_simulated(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        row = result.iloc[0]
        single = ev.evaluate_major(row)
        assert single["salary_source_explanation"]["is_simulated"] is True
        assert "disclaimer" in single["salary_source_explanation"]


class TestNoForbiddenPhrases:
    """禁止表述检查"""

    def test_no_guarantee_in_explanation(self, sample_major_employment):
        ev = CareerEvaluator()
        result = ev.evaluate(sample_major_employment)
        row = result.iloc[0]
        single = ev.evaluate_major(row)
        forbidden = ["保证就业", "一定高薪", "稳赚", "该专业一定适合"]
        for phrase in forbidden:
            assert phrase not in single.get("explanation", ""), f"Forbidden: {phrase}"
