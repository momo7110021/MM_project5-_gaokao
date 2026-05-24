"""
pytest fixtures for gaokao volunteer recommendation system tests.
All data is simulated. Tests validate model structure, output fields, and business rules.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    generate_segment_table,
    generate_school_info,
    generate_major_info,
    generate_major_admission,
    generate_admission_plan,
    generate_major_employment,
    generate_city_data,
    generate_candidate_profile,
    generate_all_data,
)


@pytest.fixture
def sample_segment_tables():
    """多年度一分一段表 dict: {year: DataFrame}"""
    tables = {}
    for year, total, bl in [(2020, 330000, 430), (2021, 332000, 432),
                               (2022, 334000, 434), (2023, 336000, 436),
                               (2024, 350000, 432)]:
        tables[year] = generate_segment_table(
            province="河北省", year=year, subject_type="物理类",
            total_exam_count=total, batch_line=bl,
        )
    return tables


@pytest.fixture
def sample_candidate():
    """默认考生画像"""
    return {
        "candidate_id": "test_001",
        "province": "河北省",
        "year": 2024,
        "subject_type": "物理类",
        "score": 620,
        "rank": 8500,
        "interest_direction": ["计算机", "电子信息"],
        "strong_subjects": ["数学", "物理"],
        "excluded_majors": [],
        "preferred_cities": ["北京", "上海", "深圳"],
        "family_budget": 20000,
        "risk_preference": "balanced",
        "accept_adjustment": 1,
        "accept_sino_foreign": 1,
        "accept_far_city": 1,
        "employment_first": 0,
        "postgraduate_first": 1,
    }


@pytest.fixture
def sample_historical_admission():
    """模拟历史录取数据"""
    return generate_major_admission(province="河北省", years=(2020, 2021, 2022, 2023))


@pytest.fixture
def sample_major_employment():
    """模拟专业就业数据"""
    return generate_major_employment(data_years=[2023])


@pytest.fixture
def sample_volunteer_candidates():
    """模拟候选志愿列表"""
    np.random.seed(42)
    rows = []
    for i in range(80):
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
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def sample_user_preference():
    """用户偏好过滤参数"""
    return {
        "excluded_majors": ["哲学", "历史学"],
        "family_budget": 15000,
        "preferred_cities": ["北京", "上海", "杭州", "深圳"],
        "accept_adjustment": 0,
        "accept_far_city": 1,
        "risk_preference": "balanced",
    }


@pytest.fixture
def full_data():
    """完整模拟数据集"""
    return generate_all_data(province="河北省", candidate_score=620, candidate_rank=8500)
