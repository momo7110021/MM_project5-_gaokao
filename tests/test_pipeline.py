"""
test_pipeline.py
测试总流水线：Project 5 统一 JSON 输出结构完整性
验证 meta/candidate/recommendation_plan 顶层字段,
volunteers 逐志愿 20+ 字段, source_trace 5 子字段, 禁止表述
当前使用模拟数据。
"""
import json
import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_all_data
from src.pipeline import GaoKaoPipeline


def run_pipeline(score=620, rank=8500, plan="balanced"):
    """运行完整流水线并返回解析后的 JSON"""
    data = generate_all_data(province="河北省", candidate_score=score, candidate_rank=rank)
    profile = data["candidate_profile"]
    profile.update({"plan_type": plan, "year": 2024, "province": "河北省"})
    p = GaoKaoPipeline()
    p.load_data({
        "segment_table": data["segment_table"],
        "school_info": data["school_info"],
        "major_info": data["major_info"],
        "school_admission_line": data["school_admission_line"],
        "major_admission": data["major_admission"],
        "admission_plan": data["admission_plan"],
        "major_employment": data["major_employment"],
        "city_data": data["city_data"],
    })
    output = p.run(profile=profile, plan_type=plan)
    return json.loads(output)


class TestPipelineOutputStructure:
    """顶层结构测试"""

    def test_top_level_keys(self):
        parsed = run_pipeline()
        for key in ["meta", "candidate", "recommendation_plan"]:
            assert key in parsed, f"Missing top-level key: {key}"

    def test_meta_fields(self):
        parsed = run_pipeline()
        assert "model_version" in parsed["meta"]
        assert "generate_time" in parsed["meta"]
        assert parsed["meta"]["model_version"] == "v3.0.0"


class TestCandidateBlock:
    """candidate 块字段测试"""

    def test_candidate_fields(self):
        parsed = run_pipeline()
        c = parsed["candidate"]
        required = [
            "candidate_id", "province", "year", "subject_type",
            "score", "rank", "equivalent_scores", "equivalent_score",
            "equivalent_score_interval", "equivalent_rank", "confidence_level",
            "abnormal_year_warning", "fallback_rule", "risk_preference",
            "source_trace",
        ]
        for field in required:
            assert field in c, f"Missing candidate field: {field}"

    def test_source_trace_in_candidate(self):
        parsed = run_pipeline()
        trace = parsed["candidate"]["source_trace"]
        for field in ["source_url", "source_name", "crawl_time", "data_version", "source_type"]:
            assert field in trace, f"Missing source_trace field: {field}"


class TestVolunteerItem:
    """volunteers 列表字段测试"""

    def test_volunteer_item_fields(self):
        parsed = run_pipeline()
        volunteers = parsed["recommendation_plan"]["volunteers"]
        assert len(volunteers) > 0, "Should have at least one volunteer"
        v = volunteers[0]
        required = [
            "volunteer_id", "school_code", "school_name",
            "major_group_code", "major_code", "major_name",
            "admit_probability", "probability_interval", "recommendation_tier",
            "fit_score", "career_score", "city_score", "family_score",
            "risk_level", "risk_reason", "overall_utility",
            "explanation", "modification_suggestion",
            "review_required", "source_trace",
        ]
        for field in required:
            assert field in v, f"Missing volunteer field: {field}"

    def test_source_trace_in_volunteer(self):
        parsed = run_pipeline()
        v = parsed["recommendation_plan"]["volunteers"][0]
        trace = v["source_trace"]
        for field in ["source_url", "source_name", "crawl_time", "data_version", "source_type"]:
            assert field in trace, f"Missing source_trace field in volunteer: {field}"


class TestStatisticsBlock:
    """statistics 块测试"""

    def test_statistics_fields(self):
        parsed = run_pipeline()
        stats = parsed["recommendation_plan"]["statistics"]
        required = ["rush_count", "stable_count", "safe_count", "bottom_count",
                     "overall_score", "overall_risk_level"]
        for field in required:
            assert field in stats, f"Missing stats field: {field}"


class TestRiskAssessmentBlock:
    """risk_assessment 块测试"""

    def test_risk_assessment_fields(self):
        parsed = run_pipeline()
        risk = parsed["recommendation_plan"]["risk_assessment"]
        required = [
            "overall_risk_score", "risk_level",
            "slip_risk", "withdrawal_risk", "adjustment_risk",
            "cold_major_risk", "employment_risk", "region_risk",
            "risk_reason", "modification_suggestion", "review_required",
        ]
        for field in required:
            assert field in risk, f"Missing risk field: {field}"

    def test_risk_sub_fields_are_dicts(self):
        parsed = run_pipeline()
        risk = parsed["recommendation_plan"]["risk_assessment"]
        for key in ["slip_risk", "withdrawal_risk", "adjustment_risk",
                     "cold_major_risk", "employment_risk", "region_risk"]:
            assert isinstance(risk[key], dict), f"{key} should be dict"
            for sub in ["score", "level"]:
                assert sub in risk[key], f"{key} missing {sub}"


class TestNoForbiddenPhrases:
    """禁止表述检查"""

    def test_no_forbidden_in_volunteers(self):
        parsed = run_pipeline()
        forbidden = ["保证录取", "一定能上", "绝对安全", "录取概率100%", "100%录取"]
        for v in parsed["recommendation_plan"]["volunteers"]:
            for phrase in forbidden:
                assert phrase not in v.get("explanation", ""), f"Forbidden: {phrase} in volunteer"
                assert phrase not in v.get("modification_suggestion", ""), f"Forbidden: {phrase} in suggestion"


class TestAllPlanTypes:
    """三种方案均能运行"""

    def test_all_plans(self):
        for pt in ["aggressive", "balanced", "conservative"]:
            parsed = run_pipeline(plan=pt)
            assert parsed["recommendation_plan"]["plan_type"] == pt


class TestMiscOutputs:
    """报告导出测试"""

    def test_business_report_generated(self):
        data = generate_all_data(province="河北省", candidate_score=620, candidate_rank=8500)
        profile = data["candidate_profile"]
        profile.update({"plan_type": "balanced", "year": 2024, "province": "河北省"})
        p = GaoKaoPipeline()
        p.load_data({
            "segment_table": data["segment_table"],
            "school_info": data["school_info"],
            "major_employment": data["major_employment"],
            "city_data": data["city_data"],
            "school_admission_line": data["school_admission_line"],
            "major_admission": data["major_admission"],
            "admission_plan": data["admission_plan"],
        })
        p.run(profile=profile, plan_type="balanced")
        report = p.generate_business_report(profile)
        assert len(report) > 100
        assert "志愿" in report
