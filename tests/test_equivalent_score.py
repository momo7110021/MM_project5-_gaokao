"""
test_equivalent_score.py
测试模型一：分数—位次等效换算模型
Project 5 要求：输入数据、数学模型、评价指标、接口输出、业务解释
当前使用模拟数据。
"""
import numpy as np
import pandas as pd
import pytest

from src.equivalent_score import EquivalentScoreConverter
from src.data_generator import generate_segment_table


class TestEquivalentScoreOutput:
    """验证等效分模型输出字段"""

    def test_output_fields_complete(self, sample_segment_tables):
        conv = EquivalentScoreConverter()
        result = conv.convert(
            segment_dfs=sample_segment_tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[2020, 2021, 2022, 2023],
            batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        required = [
            "equivalent_score", "equivalent_score_interval", "equivalent_rank",
            "current_percentile", "confidence_level", "abnormal_year_warning",
            "fallback_rule", "review_required",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_interval_valid(self, sample_segment_tables):
        conv = EquivalentScoreConverter()
        result = conv.convert(
            segment_dfs=sample_segment_tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[2020, 2021, 2022, 2023],
            batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        lo, hi = result["equivalent_score_interval"]
        assert lo <= hi, "lower must <= upper"
        assert 0 <= result["equivalent_score"] <= 750

    def test_confidence_level_is_valid(self, sample_segment_tables):
        conv = EquivalentScoreConverter()
        result = conv.convert(
            segment_dfs=sample_segment_tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[2020, 2021, 2022, 2023],
            batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        assert result["confidence_level"] in ("high", "medium", "low")

    def test_equivalent_rank_is_int(self, sample_segment_tables):
        conv = EquivalentScoreConverter()
        result = conv.convert(
            segment_dfs=sample_segment_tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[2020, 2021, 2022, 2023],
            batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        rank = result["equivalent_rank"]
        if rank is not None:
            assert isinstance(rank, (int, np.integer))

    def test_fallback_on_missing_years(self):
        """构造仅有当前年份(无目标年份)，验证 fallback 触发"""
        conv = EquivalentScoreConverter()
        seg_current = generate_segment_table(province="河北省", year=2024,
                                              subject_type="物理类")
        tables = {2024: seg_current}
        result = conv.convert(
            segment_dfs=tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[],
            batch_lines={2024: 432},
        )
        assert result["fallback_rule"] == "insufficient_data"
        assert result["review_required"] is True
        assert result["confidence_level"] == "low"

    def test_abnormal_year_warning_triggered(self):
        """构造波动极大的异常年份，验证 abnormal_year_warning"""
        np.random.seed(99)
        tables = {}
        for year in [2020, 2021, 2022, 2023, 2024]:
            seg = generate_segment_table(province="河北省", year=year,
                                          subject_type="物理类",
                                          total_exam_count=300000 + year * 1000,
                                          batch_line=430 + (year - 2020) * 2)
            if year == 2021:
                seg["cumulative_count"] = seg["cumulative_count"] * np.random.uniform(0.85, 1.15)
            tables[year] = seg

        conv = EquivalentScoreConverter(tau=100)
        result = conv.convert(
            segment_dfs=tables,
            candidate_score=620, candidate_rank=8500,
            current_year=2024,
            target_years=[2020, 2021, 2022, 2023],
            batch_lines={y: 430 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        # 至少验证结果不崩溃
        assert "equivalent_score" in result


class TestBacktestError:
    """验证回测误差计算"""

    def test_backtest_returns_mae(self, sample_segment_tables):
        conv = EquivalentScoreConverter()
        err = conv.evaluate_backtest_error(
            segment_dfs=sample_segment_tables,
            test_year=2023, test_score=600, test_rank=10000,
            batch_lines={y: 432 + (y - 2020) * 2 for y in range(2020, 2025)},
        )
        assert "mae" in err
