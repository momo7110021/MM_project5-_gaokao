"""
数据生成器：生成模拟数据集，用于演示和测试所有模型。
所有数据仅为模拟数据，不代表任何真实院校、专业或考生信息。

生成的数据类别：
1. 一分一段表 (segment_table)
2. 院校基本信息表 (school_info)
3. 专业基本信息表 (major_info)
4. 历年院校投档线表 (school_admission_line)
5. 历年专业录取表 (major_admission)
6. 招生计划表 (admission_plan)
7. 专业就业数据表 (major_employment)
8. 城市产业数据表 (city_data)
9. 考生画像表 (candidate_profile)
"""

import numpy as np
import pandas as pd
from itertools import product

np.random.seed(42)


def generate_segment_table(province="河北省", year=2024, subject_type="物理类",
                           total_exam_count=350000, batch_line=432):
    """生成模拟一分一段表"""
    mu, sigma = 450, 100
    scores = []
    s = 750
    cum = 0
    target_cum = total_exam_count

    while s >= 100:
        if s > 700:
            p = stats_pdf(s, mu, sigma, target_cum)
        else:
            p = stats_pdf(s, mu, sigma, target_cum)
        p = max(0, int(p))
        if s == 750:
            p = max(0, min(p, 50))
        seg = p
        cum += seg
        scores.append([s, seg, cum])
        s -= 1

    df = pd.DataFrame(scores, columns=["score", "segment_count", "cumulative_count"])
    total = df["cumulative_count"].max()
    df["rank"] = df["cumulative_count"]
    df["percentile"] = df["cumulative_count"] / total
    df["province"] = province
    df["year"] = year
    df["subject_type"] = subject_type
    df["total_exam_count"] = total_exam_count
    df["batch_line"] = batch_line
    df["batch_line_type"] = "本科线"
    df["source_url"] = f"http://www.example-exam.edu.cn/{year}/segment_{subject_type}.html"
    df["crawl_time"] = pd.Timestamp.now()
    df["review_status"] = 0
    return df.reset_index(drop=True)


def stats_pdf(score, mu, sigma, target_total):
    """模拟分数分布的近似PDF"""
    from scipy.stats import norm
    density = norm.pdf(score, mu, sigma)
    return density / norm.pdf(mu, mu, sigma) * target_total * 0.005


def generate_school_info():
    """生成模拟院校基本信息"""
    schools = [
        ("10001", "清华大学", "北京", "北京", "985", 1, "综合"),
        ("10002", "北京大学", "北京", "北京", "985", 1, "综合"),
        ("10003", "A理工大学", "河北", "石家庄", "211", 1, "理工"),
        ("10004", "B师范大学", "河北", "保定", "普通本科", 1, "师范"),
        ("10005", "C医科大学", "江苏", "南京", "211", 1, "医药"),
        ("10006", "D财经大学", "上海", "上海", "211", 1, "财经"),
        ("10007", "E工业大学", "湖北", "武汉", "985", 1, "理工"),
        ("10008", "F农业大学", "陕西", "杨凌", "985", 1, "农林"),
        ("10009", "G科技大学", "广东", "深圳", "普通本科", 0, "理工"),
        ("10010", "H政法大学", "北京", "北京", "211", 1, "政法"),
        ("10011", "I语言大学", "北京", "北京", "211", 1, "语言"),
        ("10012", "J邮电大学", "四川", "成都", "211", 1, "理工"),
        ("10013", "K交通大学", "上海", "上海", "985", 1, "理工"),
        ("10014", "L航空航天大学", "北京", "北京", "985", 1, "理工"),
        ("10015", "M海洋大学", "山东", "青岛", "985", 1, "综合"),
        ("10016", "N师范大学", "湖南", "长沙", "211", 1, "师范"),
        ("10017", "O建筑大学", "重庆", "重庆", "双一流", 1, "理工"),
        ("10018", "P中医药大学", "北京", "北京", "211", 1, "医药"),
        ("10019", "Q传媒大学", "北京", "北京", "211", 1, "艺术"),
        ("10020", "R体育大学", "北京", "北京", "211", 1, "体育"),
    ]
    df = pd.DataFrame(schools, columns=["school_code", "school_name", "province", "city",
                                         "school_level", "is_public", "school_type"])
    df["alias_list"] = df["school_name"].apply(lambda x: f'["{x}"]')
    df["moe_code"] = df["school_code"].apply(lambda x: f"10{x}000")
    df["update_time"] = pd.Timestamp.now()
    return df


def generate_major_info():
    """生成模拟专业基本信息"""
    majors = [
        ("080901", "计算机科学与技术", "计算机类", "工学", "工学学士"),
        ("080902", "软件工程", "计算机类", "工学", "工学学士"),
        ("080703", "通信工程", "电子信息类", "工学", "工学学士"),
        ("080201", "机械工程", "机械类", "工学", "工学学士"),
        ("080601", "电气工程及其自动化", "电气类", "工学", "工学学士"),
        ("081001", "土木工程", "土木类", "工学", "工学学士"),
        ("080301", "测控技术与仪器", "仪器类", "工学", "工学学士"),
        ("082801", "建筑学", "建筑类", "工学", "建筑学学士"),
        ("070101", "数学与应用数学", "数学类", "理学", "理学学士"),
        ("070201", "物理学", "物理学类", "理学", "理学学士"),
        ("070301", "化学", "化学类", "理学", "理学学士"),
        ("071001", "生物科学", "生物科学类", "理学", "理学学士"),
        ("050101", "汉语言文学", "中国语言文学类", "文学", "文学学士"),
        ("050201", "英语", "外国语言文学类", "文学", "文学学士"),
        ("120201", "工商管理", "工商管理类", "管理学", "管理学学士"),
        ("120203", "会计学", "工商管理类", "管理学", "管理学学士"),
        ("020301", "金融学", "金融学类", "经济学", "经济学学士"),
        ("030101", "法学", "法学类", "法学", "法学学士"),
        ("100201", "临床医学", "临床医学类", "医学", "医学学士"),
        ("101101", "护理学", "护理学类", "医学", "理学学士"),
        ("071101", "心理学", "心理学类", "理学", "理学学士"),
        ("030301", "社会学", "社会学类", "法学", "法学学士"),
        ("060101", "历史学", "历史学类", "历史学", "历史学学士"),
        ("010101", "哲学", "哲学类", "哲学", "哲学学士"),
        ("080401", "材料科学与工程", "材料类", "工学", "工学学士"),
        ("082503", "环境科学", "环境科学与工程类", "工学", "工学学士"),
    ]
    df = pd.DataFrame(majors, columns=["major_code", "major_name", "major_category",
                                        "discipline", "degree"])
    df["is_new_major"] = 0
    df.loc[df["major_code"].isin(["080901", "080902"]), "is_new_major"] = 0
    df["update_time"] = pd.Timestamp.now()
    return df


def generate_school_admission_line(province="河北省", years=(2020, 2021, 2022, 2023),
                                    subject_type="物理类", batch="本科批"):
    """生成模拟历年院校投档线"""
    data = []
    school_list = [f"100{i:02d}" for i in range(1, 21)]
    for year, school_code in product(years, school_list):
        school_idx = int(school_code[-2:])  # 1-20
        base_rank = int(np.random.normal(3000 + school_idx * 1000, 1500))
        base_score = max(400, min(700, 670 - school_idx * 12 + np.random.normal(0, 20)))
        plan_count = int(np.random.normal(80, 30))
        data.append([
            province, year, batch, subject_type, school_code,
            int(base_score), base_rank,
            plan_count, int(plan_count * np.random.uniform(0.95, 1.1)),
            f"http://www.example-exam.edu.cn/{year}/admission/{school_code}.html",
            pd.Timestamp.now()
        ])
    df = pd.DataFrame(data, columns=[
        "province", "year", "batch", "subject_type", "school_code",
        "min_admission_score", "min_admission_rank",
        "plan_count", "admission_count", "source_url", "crawl_time"
    ])
    df["min_admission_score"] = df["min_admission_score"].astype(int)
    return df


def generate_major_admission(province="河北省", years=(2020, 2021, 2022, 2023),
                              subject_type="物理类"):
    """生成模拟历年专业录取数据"""
    data = []
    school_list = [f"100{i:02d}" for i in range(1, 21)]
    major_list = [
        ("080901", "计算机科学与技术", "物理,化学"),
        ("080902", "软件工程", "物理,化学"),
        ("080703", "通信工程", "物理,化学"),
        ("080201", "机械工程", "物理"),
        ("080601", "电气工程及其自动化", "物理"),
        ("081001", "土木工程", "物理"),
        ("070101", "数学与应用数学", "物理"),
        ("120203", "会计学", "物理"),
        ("020301", "金融学", "物理"),
        ("030101", "法学", "物理"),
        ("100201", "临床医学", "物理,化学,生物"),
        ("071101", "心理学", "物理"),
    ]
    school_list = [f"100{i:02d}" for i in range(1, 21)]
    for year, school_code in product(years, school_list):
        school_idx = int(school_code[-2:])  # 1-20
        base_rank = int(np.random.normal(3000 + school_idx * 1000 + 1500, 2000))
        chosen_indices = np.random.choice(len(major_list), size=min(6, len(major_list)), replace=False)
        for idx in chosen_indices:
            m_code, m_name, sub_req = major_list[idx]
            rank_offset = np.random.normal(0, 1000)
            min_rank = base_rank + int(rank_offset)
            min_score = int(650 - min_rank / 500 + np.random.normal(0, 8))
            avg_score = int(min_score + np.random.uniform(5, 20))
            max_score = int(avg_score + np.random.uniform(5, 25))
            plan = int(np.random.normal(15, 8))
            data.append([
                school_code, m_code, m_name, "",
                province, year, min_score, min_rank,
                avg_score, max_score, plan,
                int(plan * np.random.uniform(0.9, 1.1)),
                sub_req, "",
                f"http://www.example-school.edu.cn/admission/{school_code}/{m_code}/{year}",
                pd.Timestamp.now()
            ])
    df = pd.DataFrame(data, columns=[
        "school_code", "major_code", "major_name", "major_group_code",
        "province", "year", "min_admission_score", "min_admission_rank",
        "avg_score", "max_score", "plan_count", "admission_count",
        "subject_requirement", "adjustment_rule", "source_url", "crawl_time",
    ])
    return df


def generate_admission_plan(province="河北省", year=2024):
    """生成模拟招生计划"""
    data = []
    school_list = [f"100{i:02d}" for i in range(1, 21)]
    major_list = [
        ("080901", "计算机科学与技术"),
        ("080902", "软件工程"),
        ("080703", "通信工程"),
        ("080201", "机械工程"),
        ("080601", "电气工程及其自动化"),
        ("081001", "土木工程"),
        ("070101", "数学与应用数学"),
        ("120203", "会计学"),
        ("020301", "金融学"),
        ("030101", "法学"),
        ("100201", "临床医学"),
        ("071101", "心理学"),
    ]
    for school_code in school_list:
        chosen_indices = np.random.choice(len(major_list), size=min(6, len(major_list)), replace=False)
        for idx in chosen_indices:
            m_code, m_name = major_list[idx]
            tuition = int(np.random.choice([4500, 5000, 5500, 6000, 6800, 8000, 12000, 18000]))
            is_sf = 1 if m_name in ("软件工程",) and tuition >= 12000 else 0
            is_normal = 1 if m_name in ("数学与应用数学", "汉语言文学") else 0
            is_medical = 1 if m_name in ("临床医学", "护理学") else 0
            sub_req = "物理,化学" if m_code.startswith("08") or m_code.startswith("07") else "物理"
            data.append([
                school_code, "", m_code, m_name, province, year,
                int(np.random.normal(8, 4)), tuition, 4,
                sub_req, is_sf, is_normal, is_medical,
                "", "", "",
                f"http://www.example-exam.edu.cn/plan/{year}/{school_code}.html",
                pd.Timestamp.now()
            ])
    df = pd.DataFrame(data, columns=[
        "school_code", "major_group_code", "major_code", "major_name",
        "province", "year", "plan_count", "tuition", "duration",
        "subject_requirement", "is_sino_foreign", "is_normal_major", "is_medical_major",
        "single_subject_limit", "physical_limit", "remark",
        "source_url", "crawl_time",
    ])
    return df


def generate_major_employment(data_years=(2021, 2022, 2023)):
    """生成模拟专业就业数据"""
    major_emp = {
        "080901": {"name": "计算机科学与技术", "emp": 0.94, "pg": 0.35, "sal": 8500, "jobs": 45000, "grow": 0.08, "ind": 8.5, "stab": 7.0, "cs": 120, "sent": 0.05},
        "080902": {"name": "软件工程", "emp": 0.93, "pg": 0.28, "sal": 9000, "jobs": 48000, "grow": 0.10, "ind": 8.8, "stab": 6.5, "cs": 100, "sent": 0.04},
        "080703": {"name": "通信工程", "emp": 0.90, "pg": 0.30, "sal": 7500, "jobs": 25000, "grow": 0.05, "ind": 7.8, "stab": 7.0, "cs": 180, "sent": 0.08},
        "080201": {"name": "机械工程", "emp": 0.88, "pg": 0.22, "sal": 6000, "jobs": 20000, "grow": -0.02, "ind": 6.0, "stab": 6.5, "cs": 150, "sent": 0.20},
        "080601": {"name": "电气工程及其自动化", "emp": 0.92, "pg": 0.25, "sal": 7000, "jobs": 18000, "grow": 0.03, "ind": 7.0, "stab": 7.5, "cs": 200, "sent": 0.06},
        "081001": {"name": "土木工程", "emp": 0.75, "pg": 0.18, "sal": 5500, "jobs": 12000, "grow": -0.10, "ind": 4.5, "stab": 5.5, "cs": 300, "sent": 0.45},
        "070101": {"name": "数学与应用数学", "emp": 0.78, "pg": 0.42, "sal": 5500, "jobs": 8000, "grow": 0.01, "ind": 5.5, "stab": 6.0, "cs": 250, "sent": 0.15},
        "120203": {"name": "会计学", "emp": 0.88, "pg": 0.15, "sal": 5500, "jobs": 30000, "grow": -0.03, "ind": 5.0, "stab": 8.0, "cs": 500, "sent": 0.12},
        "020301": {"name": "金融学", "emp": 0.85, "pg": 0.28, "sal": 7500, "jobs": 15000, "grow": 0.02, "ind": 6.5, "stab": 6.0, "cs": 350, "sent": 0.18},
        "030101": {"name": "法学", "emp": 0.72, "pg": 0.32, "sal": 5000, "jobs": 10000, "grow": 0.0, "ind": 4.5, "stab": 6.0, "cs": 800, "sent": 0.25},
        "100201": {"name": "临床医学", "emp": 0.96, "pg": 0.60, "sal": 6000, "jobs": 22000, "grow": 0.06, "ind": 7.5, "stab": 9.0, "cs": 100, "sent": 0.03},
        "071101": {"name": "心理学", "emp": 0.70, "pg": 0.38, "sal": 4800, "jobs": 5000, "grow": 0.02, "ind": 4.0, "stab": 5.0, "cs": 100, "sent": 0.30},
        "050101": {"name": "汉语言文学", "emp": 0.80, "pg": 0.25, "sal": 4500, "jobs": 12000, "grow": -0.05, "ind": 3.5, "stab": 6.5, "cs": 600, "sent": 0.10},
        "050201": {"name": "英语", "emp": 0.82, "pg": 0.22, "sal": 5000, "jobs": 15000, "grow": -0.08, "ind": 3.0, "stab": 5.0, "cs": 200, "sent": 0.35},
        "120201": {"name": "工商管理", "emp": 0.80, "pg": 0.18, "sal": 5500, "jobs": 20000, "grow": -0.02, "ind": 5.0, "stab": 6.0, "cs": 400, "sent": 0.22},
        "080401": {"name": "材料科学与工程", "emp": 0.82, "pg": 0.30, "sal": 6000, "jobs": 10000, "grow": 0.01, "ind": 5.5, "stab": 6.0, "cs": 80, "sent": 0.18},
        "082503": {"name": "环境科学", "emp": 0.75, "pg": 0.35, "sal": 5000, "jobs": 6000, "grow": -0.03, "ind": 4.5, "stab": 5.5, "cs": 150, "sent": 0.20},
        "010101": {"name": "哲学", "emp": 0.68, "pg": 0.40, "sal": 4200, "jobs": 3000, "grow": -0.05, "ind": 3.0, "stab": 5.0, "cs": 100, "sent": 0.40},
        "030301": {"name": "社会学", "emp": 0.72, "pg": 0.35, "sal": 4500, "jobs": 4000, "grow": 0.01, "ind": 3.5, "stab": 5.5, "cs": 80, "sent": 0.28},
    }
    data = []
    for year in data_years:
        for code, info in major_emp.items():
            sal_jitter = np.random.normal(0, 500)
            data.append([
                code, info["name"],
                min(1, info["emp"] + np.random.uniform(-0.03, 0.03)),
                min(1, info["pg"] + np.random.uniform(-0.05, 0.05)),
                info["cs"],
                info["sal"] + sal_jitter,
                info["sal"] + sal_jitter * 0.8,
                int(info["jobs"] * (1 + np.random.uniform(-0.1, 0.1))),
                info["grow"] + np.random.uniform(-0.02, 0.02),
                "[]", "[]",
                info["ind"] + np.random.uniform(-0.3, 0.3),
                info["stab"] + np.random.uniform(-0.3, 0.3),
                info["sent"] + np.random.uniform(-0.02, 0.02),
                year,
                f"http://www.example.edu.cn/employment/{code}/{year}",
                pd.Timestamp.now()
            ])
    df = pd.DataFrame(data, columns=[
        "major_code", "major_name",
        "employment_rate", "postgraduate_rate", "civil_service_post_count",
        "average_salary", "median_salary", "job_count", "job_growth_rate",
        "industry_distribution", "main_employment_city",
        "industry_growth_score", "stability_score", "sentiment_warning_score",
        "data_year", "source_url", "crawl_time"
    ])
    return df


def generate_city_data(data_year=2023):
    """生成模拟城市产业数据"""
    cities = [
        ("北京", 41610, 190000, 0.84, 28000, 450, 12000, 8000, 120),
        ("上海", 47218, 183000, 0.75, 22000, 380, 11000, 7500, 200),
        ("深圳", 34606, 185000, 0.62, 18000, 350, 10000, 7000, 300),
        ("广州", 30355, 152000, 0.73, 15000, 150, 9000, 6000, 500),
        ("武汉", 20011, 143000, 0.62, 12000, 80, 7500, 4500, 800),
        ("成都", 22074, 102000, 0.66, 10000, 100, 7500, 4200, 1100),
        ("南京", 17421, 178000, 0.63, 9000, 120, 8500, 5500, 900),
        ("杭州", 20059, 162000, 0.68, 11000, 200, 9000, 6000, 1200),
        ("西安", 12010, 92000, 0.65, 8000, 60, 6500, 4000, 1500),
        ("长沙", 14331, 138000, 0.58, 6000, 70, 7000, 3800, 1600),
        ("石家庄", 7534, 67000, 0.67, 3000, 20, 5500, 3200, 200),
        ("保定", 4012, 47000, 0.48, 1500, 8, 4500, 2500, 300),
    ]
    data = []
    for city, gdp, gdp_pc, t_ratio, ht_count, listed, salary, living, dist in cities:
        data.append([
            city, gdp, gdp_pc, t_ratio,
            f'["电子信息","装备制造"]',
            ht_count, listed, ht_count * 2,
            salary, living, 0, 8.0,
            data_year,
            f"http://www.stats.gov.cn/city/{city}",
            pd.Timestamp.now()
        ])
    df = pd.DataFrame(data, columns=[
        "city", "gdp", "gdp_per_capita", "tertiary_industry_ratio",
        "key_industries", "high_tech_company_count", "listed_company_count",
        "related_job_count", "average_salary", "living_cost",
        "distance_from_home", "transport_score",
        "data_year", "source_url", "crawl_time"
    ])
    return df


def generate_candidate_profile(candidate_id="cand_2024_hebei_001", province="河北省",
                                subject_type="物理类", score=620, rank=8500,
                                risk_preference="balanced"):
    """生成模拟考生画像"""
    return {
        "candidate_id": candidate_id,
        "province": province,
        "subject_type": subject_type,
        "score": score,
        "rank": rank,
        "interest_direction": ["计算机", "电子信息", "自动化", "数据科学"],
        "strong_subjects": ["数学", "物理"],
        "excluded_majors": ["哲学", "历史学", "心理学"],
        "preferred_cities": ["北京", "上海", "深圳", "杭州", "南京", "成都"],
        "family_budget": 20000,
        "risk_preference": risk_preference,
        "accept_adjustment": 0,
        "accept_sino_foreign": 1,
        "accept_far_city": 1,
        "employment_first": 0,
        "postgraduate_first": 1,
        "create_time": pd.Timestamp.now(),
    }


def generate_all_data(province="河北省", year=2024, subject_type="物理类",
                       candidate_score=620, candidate_rank=8500):
    """生成完整的模拟数据集"""
    print("=" * 60)
    print("  高考志愿填报系统 - 模拟数据生成器")
    print("  所有数据均为模拟数据，仅供教学和测试使用")
    print("=" * 60)

    data = {}

    print("\n[1/9] 生成一分一段表...")
    data["segment_table"] = generate_segment_table(province, year, subject_type)

    print("[2/9] 生成院校基本信息...")
    data["school_info"] = generate_school_info()

    print("[3/9] 生成专业基本信息...")
    data["major_info"] = generate_major_info()

    print("[4/9] 生成历年院校投档线...")
    data["school_admission_line"] = generate_school_admission_line(province)

    print("[5/9] 生成历年专业录取数据...")
    data["major_admission"] = generate_major_admission(province)

    print("[6/9] 生成招生计划...")
    data["admission_plan"] = generate_admission_plan(province, year)

    print("[7/9] 生成专业就业数据...")
    data["major_employment"] = generate_major_employment()

    print("[8/9] 生成城市产业数据...")
    data["city_data"] = generate_city_data()

    print("[9/9] 生成考生画像...")
    data["candidate_profile"] = generate_candidate_profile(
        score=candidate_score, rank=candidate_rank
    )

    print("\n数据生成完毕!")
    total_rows = sum(len(v) if isinstance(v, pd.DataFrame) else 1 for v in data.values())
    print(f"总计: {len(data)} 个数据集, {total_rows} 行数据")
    return data


if __name__ == "__main__":
    data = generate_all_data()
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"    字段: {list(df.columns)[:8]}...")
