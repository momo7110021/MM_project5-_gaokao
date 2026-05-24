"""
UI 辅助函数 (Project 5 v3.1.0 UI)

为 streamlit_app.py 提供格式化、颜色标签、数据转换等工具函数。
不包含模型逻辑。
"""

import json
import pandas as pd


# ================================================================
# 颜色常量
# ================================================================

TIER_COLORS = {
    "rush": "#E67E22",
    "stable": "#3498DB",
    "safe": "#2ECC71",
    "bottom": "#1ABC9C",
}

TIER_BG = {
    "rush": "rgba(230,126,34,0.15)",
    "stable": "rgba(52,152,219,0.15)",
    "safe": "rgba(46,204,113,0.15)",
    "bottom": "rgba(26,188,156,0.15)",
}

TIER_LABEL = {"rush": "冲", "stable": "稳", "safe": "保", "bottom": "垫"}

RISK_COLORS = {
    "low": "#2ECC71",
    "medium": "#F39C12",
    "high": "#E74C3C",
    "critical": "#8E44AD",
    "very_high": "#8E44AD",
}

RISK_LABELS = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "极高风险",
    "very_high": "极高风险",
}

LABEL_COLORS = {
    "green": "#2ECC71",
    "yellow": "#F39C12",
    "red": "#E74C3C",
}

LABEL_BG = {
    "green": "rgba(46,204,113,0.2)",
    "yellow": "rgba(243,156,18,0.2)",
    "red": "rgba(231,76,60,0.2)",
}


# ================================================================
# 格式化函数
# ================================================================

def fmt_prob(p):
    """格式化概率为百分比字符串"""
    return f"{p:.0%}"


def fmt_interval(lo, hi):
    """格式化概率区间，确保 lower <= upper"""
    lo = float(lo); hi = float(hi)
    if lo > hi: lo, hi = hi, lo
    lo = max(0, min(lo, 0.99))
    hi = max(0, min(hi, 0.99))
    return f"{lo:.0%}-{hi:.0%}"


def format_probability_interval(interval):
    if not interval or len(interval) != 2:
        return "-"
    return fmt_interval(interval[0], interval[1])


def fmt_score(s):
    """格式化评分为两位小数"""
    return f"{s:.2f}"


def tier_html(tier):
    """冲稳保垫彩色标签 HTML"""
    c = TIER_COLORS.get(tier, "#999")
    n = TIER_LABEL.get(tier, tier)
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">{n}</span>'


def risk_html(level):
    """风险等级彩色标签 HTML"""
    c = RISK_COLORS.get(level, "#999")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">{level}</span>'


def label_html(label):
    """红黄绿标签 HTML"""
    c = LABEL_COLORS.get(label, "#999")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold">{label}</span>'


def prob_bar_html(p):
    """概率进度条 HTML"""
    w = max(0, min(100, p * 100))
    c = "#E74C3C" if p < 0.20 else "#F39C12" if p < 0.45 else "#3498DB" if p < 0.70 else "#2ECC71"
    return f"""
    <div style="width:100px;height:8px;background:#eee;border-radius:4px;display:inline-block;vertical-align:middle;margin-right:6px">
        <div style="width:{w}%;height:8px;background:{c};border-radius:4px"></div>
    </div>
    """


# ================================================================
# 数据转换
# ================================================================

def volunteers_to_dataframe(volunteers_json):
    """将 pipeline 输出的 JSON volunteer 列表转为更友好的 DataFrame"""
    if not volunteers_json:
        return pd.DataFrame()
    rows = []
    for v in volunteers_json:
        rows.append({
            "序号": v.get("volunteer_id", 0),
            "院校": v.get("school_name", ""),
            "专业": v.get("major_name", ""),
            "录取概率": v.get("admit_probability", 0),
            "概率区间": format_probability_interval(v.get('probability_interval', [0,1])),
            "推荐等级": v.get("recommendation_tier", ""),
            "就业评分": v.get("career_score", 0),
            "城市评分": v.get("city_score", 0),
            "匹配度": v.get("fit_score", 0),
            "综合效用": v.get("overall_utility", 0),
            "风险等级": v.get("risk_level", ""),
            "复查": "是" if v.get("review_required") else "否",
            "解释": v.get("explanation", ""),
            "建议": v.get("modification_suggestion", ""),
            "院校代码": v.get("school_code", ""),
            "专业代码": v.get("major_code", ""),
            "source_trace": v.get("source_trace", {}),
            # raw for detail expander
            "_raw": v,
        })
    return pd.DataFrame(rows)


def risk_to_dataframe(risk_assessment):
    """将风险评估 dict 展平为 DataFrame"""
    if not risk_assessment:
        return pd.DataFrame()
    risk_names = {
        "slip_risk": "滑档风险", "withdrawal_risk": "退档风险",
        "adjustment_risk": "调剂风险", "cold_major_risk": "冷门专业风险",
        "employment_risk": "就业风险", "region_risk": "地域风险",
    }
    rows = []
    for key, name in risk_names.items():
        item = risk_assessment.get(key, {})
        if isinstance(item, dict):
            rows.append({
                "风险类别": name,
                "评分": item.get("score", 0),
                "等级": item.get("level", ""),
                "触发原因": item.get("trigger_reason", ""),
                "修改建议": item.get("modification_suggestion", ""),
                "需复核": "是" if item.get("review_required") else "否",
            })
    return pd.DataFrame(rows)


def export_json_bytes(parsed_output):
    """将 Python dict 转为 JSON bytes 供下载"""
    return json.dumps(parsed_output, ensure_ascii=False, indent=2).encode("utf-8")


def validate_inputs(province, score, rank):
    """校验输入并返回错误列表"""
    errors = []
    if not province:
        errors.append("请选择省份")
    if score is None or score < 0 or score > 750:
        errors.append("分数必须在 0-750 之间")
    if rank is None or rank < 1:
        errors.append("位次必须为正整数")
    return errors


def validate_subject_selection(exam_mode, selected_subjects=None, first_subject=None, selected_subjects_12=None):
    """校验选科输入。3+1+2 时传入 first_subject + selected_subjects_12，不传合并后的 selected_subjects"""
    errors = []
    all_six = ["物理", "化学", "生物", "历史", "地理", "政治"]
    re_elect = ["化学", "生物", "地理", "政治"]

    if exam_mode in ("3+1+2", ""):
        # 校验首选
        if not first_subject or first_subject not in ("物理", "历史"):
            errors.append("3+1+2 模式下必须选择首选科目(物理或历史)")
        # 校验再选
        if selected_subjects_12 is None:
            selected_subjects_12 = selected_subjects or []
        if not isinstance(selected_subjects_12, list):
            selected_subjects_12 = list(selected_subjects_12) if selected_subjects_12 else []
        if len(selected_subjects_12) != 2:
            errors.append("3+1+2 模式下再选科目必须选择 2 门")
        if not all(s in re_elect for s in selected_subjects_12):
            errors.append("再选科目只能从化学/生物/地理/政治中选择")
        if first_subject and first_subject in selected_subjects_12:
            errors.append("首选科目不应重复出现在再选科目中")

    elif exam_mode == "3+3":
        subs = selected_subjects or selected_subjects_12 or []
        if len(subs) != 3:
            errors.append("3+3 模式下必须选择 3 门科目")
        if not all(s in all_six for s in subs):
            errors.append("包含无效科目")

    return errors


def safe_multiselect_default(current_values, options, fallback=None, max_count=None):
    """确保 multiselect 的 default 值全部在 options 内"""
    if current_values is None:
        current_values = fallback or []
    if not isinstance(current_values, list):
        current_values = list(current_values) if isinstance(current_values, (tuple, set)) else [current_values]
    valid = [x for x in current_values if x in options]
    if not valid and fallback:
        valid = [x for x in fallback if x in options]
    if max_count is not None:
        valid = valid[:max_count]
    return valid


def derive_subject_type(exam_mode, first_subject=None):
    """根据选科模式推导兼容 subject_type"""
    if exam_mode == "3+1+2":
        if first_subject == "物理":
            return "物理类"
        elif first_subject == "历史":
            return "历史类"
    return "综合类"
