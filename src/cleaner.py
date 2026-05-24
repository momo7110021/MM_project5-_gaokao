"""
数据清洗与预处理模块

清洗流水线：
1. 格式统一 (编码、日期、数值精度)
2. 缺失值处理
3. 异常值检测 (IQR方法 + 业务规则)
4. 一致性校验 (跨表关联校验)
5. 去重 (按业务主键)
6. 标准化 (院校名称、专业名称、省份)
"""

import re
import pandas as pd
import numpy as np


def standardize_school_name(name):
    """院校名称标准化"""
    if not isinstance(name, str):
        return name
    known_aliases = {
        "北京大学": ["北京大学", "北京大学(校本部)", "pku"],
        "清华大学": ["清华大学", "tsinghua"],
    }
    for standard, variants in known_aliases.items():
        if name in variants:
            return standard
    return re.sub(r"\(.*?\)", "", name).strip()


def standardize_major_name(name):
    """专业名称标准化"""
    if not isinstance(name, str):
        return name
    return re.sub(r"\(.*?方向\)|（.*?方向）", "", name).strip()


def standardize_province(name):
    """省份名称标准化"""
    if not isinstance(name, str):
        return name
    province_map = {
        "河北省": ["河北", "河北省"],
        "北京市": ["北京", "北京市"],
        "上海市": ["上海", "上海市"],
        "江苏省": ["江苏", "江苏省"],
        "浙江省": ["浙江", "浙江省"],
        "广东省": ["广东", "广东省"],
        "湖北省": ["湖北", "湖北省"],
        "湖南省": ["湖南", "湖南省"],
        "四川省": ["四川", "四川省"],
        "山东省": ["山东", "山东省"],
        "河南省": ["河南", "河南省"],
        "安徽省": ["安徽", "安徽省"],
        "福建省": ["福建", "福建省"],
        "江西省": ["江西", "江西省"],
        "辽宁省": ["辽宁", "辽宁省"],
        "吉林省": ["吉林", "吉林省"],
        "黑龙江省": ["黑龙江", "黑龙江省"],
        "山西省": ["山西", "山西省"],
        "陕西省": ["陕西", "陕西省"],
        "甘肃省": ["甘肃", "甘肃省"],
        "重庆市": ["重庆", "重庆市"],
        "天津市": ["天津", "天津市"],
        "广西": ["广西", "广西壮族自治区"],
        "内蒙古": ["内蒙古", "内蒙古自治区"],
        "新疆": ["新疆", "新疆维吾尔自治区"],
        "西藏": ["西藏", "西藏自治区"],
        "宁夏": ["宁夏", "宁夏回族自治区"],
        "贵州省": ["贵州", "贵州省"],
        "云南省": ["云南", "云南省"],
        "海南省": ["海南", "海南省"],
    }
    for standard, variants in province_map.items():
        if name in variants:
            return standard
    return name


def standardize_year(value):
    """年份标准化:统一为4位整数"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"(\d{4})", value)
        if m:
            return int(m.group(1))
    return None


def standardize_subject_type(value):
    """科类/选科组合标准化"""
    if not isinstance(value, str):
        return value
    subject_map = {
        "物理类": ["物理类", "理科", "理工类", "物理科目组合"],
        "历史类": ["历史类", "文科", "文史类", "历史科目组合"],
        "综合类": ["综合类", "不分文理", "3+3"],
    }
    for standard, variants in subject_map.items():
        if value in variants:
            return standard
    return value


def standardize_school_code(code):
    """院校代码标准化:统一为5位字符串"""
    if not isinstance(code, str):
        code = str(code)
    return code.zfill(5)


def standardize_major_code(code):
    """专业代码标准化:统一为6位字符串"""
    if not isinstance(code, str):
        code = str(code)
    return code.zfill(6)


def standardize_major_group_code(code):
    """专业组代码标准化"""
    if not isinstance(code, str):
        code = str(code)
    return code.zfill(3)


def validate_monotonic_cumulative(df):
    """校验累计人数单调不减(按分数降序排列时)"""
    if "cumulative_count" not in df.columns or "score" not in df.columns:
        return {"valid": False, "reason": "missing_columns"}
    df_sorted = df.sort_values("score", ascending=False)
    cum = df_sorted["cumulative_count"].values
    violations = int(np.sum(np.diff(cum) < 0))
    return {"valid": violations == 0, "violations": violations, "total_diff": len(cum) - 1}


def fix_rank_range(df):
    """修正位次区间异常(位次<=0或超过考生总数)"""
    if "rank" not in df.columns:
        return df
    total = df["cumulative_count"].max() if "cumulative_count" in df.columns else df["rank"].max()
    df = df.copy()
    df.loc[df["rank"] <= 0, "rank"] = int(total)
    df.loc[df["rank"] > total, "rank"] = int(total)
    return df


def preserve_source_trace(df, url=None, source_name=None, version=None):
    """保留数据来源追溯字段"""
    df = df.copy()
    if "source_url" not in df.columns and url:
        df["source_url"] = url
    if "source_name" not in df.columns and source_name:
        df["source_name"] = source_name
    if "crawl_time" not in df.columns:
        df["crawl_time"] = pd.Timestamp.now().isoformat()
    if "data_version" not in df.columns and version:
        df["data_version"] = version
    if "source_type" not in df.columns:
        df["source_type"] = "public"
    return df


def cross_table_validation_full(school_line_df, major_admission_df):
    """跨表一致性校验：专业最低分 >= 院校投档线（可容忍5分偏差）"""
    report = {"total": 0, "inconsistent": 0, "details": []}
    if "school_code" not in major_admission_df.columns or "school_code" not in school_line_df.columns:
        return report
    grouped_major = major_admission_df.groupby(["school_code", "province", "year"])["min_admission_score"].min()
    grouped_school = school_line_df.groupby(["school_code", "province", "year"])["min_admission_score"].first()
    for key in grouped_major.index.intersection(grouped_school.index):
        report["total"] += 1
        if grouped_major[key] < grouped_school[key] - 5:
            report["inconsistent"] += 1
            report["details"].append({"key": key, "major_min": grouped_major[key], "school_min": grouped_school[key]})
    return report


def generate_cleaning_log(stage, df_name, before_shape, after_shape, issues=None):
    """生成清洗日志"""
    log = {
        "stage": stage,
        "table": df_name,
        "before_rows": before_shape[0] if before_shape else 0,
        "after_rows": after_shape[0] if after_shape else 0,
        "removed_rows": before_shape[0] - after_shape[0] if before_shape else 0,
        "issues_found": issues or [],
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    return log


def clean_numeric_column(series, col_name, lower=None, upper=None):
    """清洗数值列：转数值 + 边界检查"""
    s = pd.to_numeric(series, errors="coerce")
    if lower is not None:
        s = s.where((s >= lower) | s.isna(), np.nan)
    if upper is not None:
        s = s.where((s <= upper) | s.isna(), np.nan)
    return s


def detect_outliers_iqr(series, multiplier=3.0):
    """使用 IQR 方法检测异常值"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return series[(series < lower) | (series > upper)]


def clean_segment_data(df, province, year):
    """清洗一分一段表"""
    df = df.copy()

    col_map = {
        "分数": "score", "分数段": "score",
        "本段人数": "segment_count", "累计人数": "cumulative_count",
        "位次": "rank", "批次线": "batch_line",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["score"] = clean_numeric_column(df.get("score"), "score", 0, 750)
    df["segment_count"] = clean_numeric_column(df.get("segment_count"), "segment_count", 0)
    df["cumulative_count"] = clean_numeric_column(df.get("cumulative_count"), "cumulative_count", 0)

    missing_rate = df[["score", "cumulative_count"]].isnull().mean().max()
    df["review_status"] = 0
    if missing_rate > 0.05:
        df["review_status"] = 2
        print(f"[WARN] 缺失率 {missing_rate:.2%}, 标记为待审核")

    if "score" in df.columns:
        df["score"] = df["score"].fillna(method="ffill")
        df = df.dropna(subset=["score"])

    df["province"] = province
    df["year"] = year
    df["crawl_time"] = pd.Timestamp.now()

    df = df.sort_values("score", ascending=False)
    df = df.drop_duplicates(subset=["score"], keep="first")

    if "cumulative_count" in df.columns:
        total = df["cumulative_count"].max()
        df["percentile"] = df["cumulative_count"] / total
        df["total_exam_count"] = total

        if not df["cumulative_count"].is_monotonic_decreasing:
            df["cumulative_count"] = df["cumulative_count"].cummax()

    return df.reset_index(drop=True)


def clean_admission_line_data(df, province, segment_df_lookup=None):
    """清洗院校投档线数据"""
    df = df.copy()

    if "school_name" in df.columns:
        df["school_name"] = df["school_name"].apply(standardize_school_name)
    if "province" not in df.columns:
        df["province"] = province

    df["min_admission_score"] = clean_numeric_column(
        df.get("min_admission_score"), "min_admission_score", 0, 750
    )
    df["min_admission_rank"] = clean_numeric_column(
        df.get("min_admission_rank"), "min_admission_rank", 1
    )
    df["plan_count"] = clean_numeric_column(df.get("plan_count"), "plan_count", 0)
    df["admission_count"] = clean_numeric_column(df.get("admission_count"), "admission_count", 0)

    df = df.dropna(subset=["school_code", "year", "min_admission_score"])

    if "min_admission_rank" in df.columns and "min_admission_score" in df.columns:
        rank_from_score = segment_df_lookup(df["min_admission_score"]) if segment_df_lookup else None
        if rank_from_score is not None:
            diff = abs(df["min_admission_rank"] - rank_from_score)
            mask = diff > diff.quantile(0.95)
            if mask.any():
                df.loc[mask, "review_status"] = 2

    key_cols = ["province", "year", "batch", "subject_type", "school_code"]
    available = [c for c in key_cols if c in df.columns]
    if available:
        df = df.drop_duplicates(subset=available, keep="first")

    return df.reset_index(drop=True)


def clean_major_admission_data(df):
    """清洗专业录取数据，确保 min <= avg <= max"""
    df = df.copy()

    score_cols = ["min_admission_score", "avg_score", "max_score"]
    for col in score_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col], col, 0, 750)

    df["min_admission_rank"] = clean_numeric_column(df.get("min_admission_rank"), "min_admission_rank", 1)
    df["plan_count"] = clean_numeric_column(df.get("plan_count"), "plan_count", 0)
    df["admission_count"] = clean_numeric_column(df.get("admission_count"), "admission_count", 0)

    if "major_name" in df.columns:
        df["major_name"] = df["major_name"].apply(standardize_major_name)

    if all(c in df.columns for c in score_cols):
        mask1 = df["min_admission_score"] > df["avg_score"]
        mask2 = df["avg_score"] > df["max_score"]
        invalid = mask1 | mask2
        if invalid.any():
            print(f"[WARN] min<=avg<=max 违反: {invalid.sum()} 行")
            df.loc[mask1, "avg_score"] = df.loc[mask1, "min_admission_score"]
            df.loc[mask2, "max_score"] = df.loc[mask2, "avg_score"]

    key_cols = ["school_code", "major_code", "major_group_code", "province", "year"]
    available = [c for c in key_cols if c in df.columns]
    if available:
        df = df.drop_duplicates(subset=available, keep="first")

    return df.reset_index(drop=True)


def clean_employment_data(df):
    """清洗就业数据：比例类字段必须在 [0, 1]"""
    df = df.copy()
    ratio_cols = ["employment_rate", "postgraduate_rate", "job_growth_rate"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col], col, lower=0, upper=1)

    salary_cols = ["average_salary", "median_salary"]
    for col in salary_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col], col, lower=0)

    score_cols = ["industry_growth_score", "stability_score", "sentiment_warning_score"]
    for col in score_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col], col, lower=0, upper=10)

    if "major_name" in df.columns:
        df["major_name"] = df["major_name"].apply(standardize_major_name)

    return df.dropna(subset=["major_code", "data_year"]).reset_index(drop=True)


def validate_cross_table(school_line_df, major_admission_df):
    """跨表一致性校验：专业最低分 >= 院校投档线（允许小幅偏差）"""
    report = {"total": 0, "inconsistent": 0, "details": []}

    if "school_code" not in major_admission_df.columns:
        return report
    if "school_code" not in school_line_df.columns:
        return report

    grouped_major = major_admission_df.groupby(["school_code", "province", "year"])[
        "min_admission_score"
    ].min()
    grouped_school = school_line_df.groupby(["school_code", "province", "year"])[
        "min_admission_score"
    ].first()

    for key in grouped_major.index.intersection(grouped_school.index):
        report["total"] += 1
        major_min = grouped_major[key]
        school_min = grouped_school[key]
        if major_min < school_min - 5:
            report["inconsistent"] += 1
            report["details"].append(
                {"key": key, "major_min": major_min, "school_min": school_min}
            )

    return report


def prepare_model_input(segment_df, school_line_df, major_admission_df,
                        admission_plan_df, employment_df, city_df, candidate_profile):
    """将所有清洗后的数据整合为模型输入字典"""
    return {
        "segment_table": segment_df,
        "school_admission_line": school_line_df,
        "major_admission": major_admission_df,
        "admission_plan": admission_plan_df,
        "major_employment": employment_df,
        "city_data": city_df,
        "candidate_profile": candidate_profile,
    }


if __name__ == "__main__":
    print("数据清洗模块加载完毕")
    print("可用函数:")
    print("  - clean_segment_data()")
    print("  - clean_admission_line_data()")
    print("  - clean_major_admission_data()")
    print("  - clean_employment_data()")
    print("  - validate_cross_table()")
