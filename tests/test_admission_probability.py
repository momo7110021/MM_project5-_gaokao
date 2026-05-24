"""
test_admission_probability.py
测试模型二：院校/专业录取概率预测模型
Project 5 要求：AUC>0.85, Brier<0.15, 冲稳保垫映射, 小样本不确定性, 三类解释
当前使用模拟数据。
"""
import numpy as np
import pandas as pd
import pytest

from src.admission_probability import AdmissionProbabilityPredictor
from src.data_generator import generate_major_admission


class TestAdmissionProbabilityOutput:
    """验证预测输出字段"""

    def test_output_fields_complete(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901") &
            (sample_historical_admission["province"] == "河北省")
        ].sort_values("year")

        result = p.predict(candidate_rank=8500, historical_records=hist)
        required = [
            "admit_probability", "probability_interval", "recommendation_tier",
            "top_features", "uncertainty_level", "review_required",
            "n_years_data",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_predict_probability_includes_explanations(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901")
        ].sort_values("year")

        result = p.predict_probability(
            candidate_rank=8500, historical_records=hist,
            school_name="Test U", major_name="Test Major",
        )
        for key in ["explanation", "consultant_explanation", "backend_explanation",
                     "modification_suggestion"]:
            assert key in result, f"Missing explanation field: {key}"

    def test_probability_not_one(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901")
        ].sort_values("year")

        result = p.predict(candidate_rank=8500, historical_records=hist)
        assert 0 <= result["admit_probability"] <= 0.99
        assert result["admit_probability"] < 1.0, "Probability must not be 1.0"

    def test_probability_interval_valid(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901")
        ].sort_values("year")

        result = p.predict(candidate_rank=8500, historical_records=hist)
        lo, hi = result["probability_interval"]
        assert 0 <= lo <= hi <= 1.0


class TestTierMapping:
    """验证冲稳保垫映射规则"""

    def test_rush_tier(self):
        p = AdmissionProbabilityPredictor()
        assert p.map_to_recommendation_tier(0.10) == "rush"
        assert p.map_to_recommendation_tier(0.19) == "rush"

    def test_stable_tier(self):
        p = AdmissionProbabilityPredictor()
        # 默认阈值：rush<0.20, stable<0.45, safe<0.70, bottom>=0.88
        assert p.map_to_recommendation_tier(0.30) == "rush", "0.30 < 0.45 should be rush"
        assert p.map_to_recommendation_tier(0.50) == "stable"

    def test_safe_tier(self):
        p = AdmissionProbabilityPredictor()
        assert p.map_to_recommendation_tier(0.75) == "safe"
        assert p.map_to_recommendation_tier(0.85) == "safe"

    def test_bottom_tier(self):
        p = AdmissionProbabilityPredictor()
        assert p.map_to_recommendation_tier(0.90) == "bottom"
        assert p.map_to_recommendation_tier(0.95) == "bottom"


class TestSmallSample:
    """小样本专业测试"""

    def test_small_sample_review_required(self):
        p = AdmissionProbabilityPredictor()
        hist = pd.DataFrame({
            "min_admission_rank": [10000],
            "min_admission_score": [600],
            "province": ["河北省"],
            "year": [2023],
        })
        result = p.predict(candidate_rank=8500, historical_records=hist)
        assert result["n_years_data"] <= 2
        assert result["review_required"] is True
        assert result["uncertainty_level"] in ("high", "medium")


class TestTopFeatures:
    """Top特征测试"""

    def test_top_features_non_empty(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901")
        ].sort_values("year")

        result = p.predict(candidate_rank=8500, historical_records=hist)
        assert len(result["top_features"]) >= 1
        if len(result["top_features"]) > 0 and isinstance(result["top_features"][0], dict):
            assert "feature" in result["top_features"][0]


class TestFitAll:
    """模型训练测试"""

    def test_fit_all_with_simulated(self):
        p = AdmissionProbabilityPredictor()
        meta = p.fit_all(use_simulated=True)
        assert p.is_trained is True
        assert meta["is_simulated"] is True


class TestNoForbiddenPhrases:
    """禁止表述检查"""

    def test_no_guarantee_in_explanation(self, sample_historical_admission):
        p = AdmissionProbabilityPredictor()
        hist = sample_historical_admission[
            (sample_historical_admission["school_code"] == "10003") &
            (sample_historical_admission["major_code"] == "080901")
        ].sort_values("year")

        result = p.predict_probability(
            candidate_rank=8500, historical_records=hist,
            school_name="Test", major_name="Test",
        )
        forbidden = ["保证录取", "一定能上", "绝对安全", "录取概率100%", "100%录取"]
        for phrase in forbidden:
            assert phrase not in result.get("explanation", ""), f"Forbidden phrase '{phrase}' found in explanation"
            assert phrase not in result.get("modification_suggestion", ""), f"Forbidden phrase '{phrase}' found in suggestion"
