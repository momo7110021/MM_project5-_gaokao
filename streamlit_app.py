"""
高考志愿智能推荐系统 - 交互式页面 (Streamlit)
Project 5 Demo v3.1.0 UI

运行方式: streamlit run streamlit_app.py

当前为模拟数据演示版本，仅供教学和系统原型展示。
"""

import sys, os, json, pandas as pd, numpy as np, streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_all_data
from src.pipeline import GaoKaoPipeline
from src.ui_helpers import (
    TIER_COLORS, TIER_LABEL, RISK_COLORS, RISK_LABELS, LABEL_COLORS, TIER_BG,
    fmt_prob, fmt_interval, fmt_score,
    tier_html, risk_html, label_html,
    volunteers_to_dataframe, risk_to_dataframe, export_json_bytes,
    validate_inputs, validate_subject_selection, derive_subject_type,
    safe_multiselect_default,
)

VERSION = "v3.1.4 UI"

# ================================================================
st.set_page_config(page_title="高考志愿智能推荐系统 | Project 5", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
    .sub-header { font-size: 14px; color: #666; margin-bottom: 12px; }
    .disclaimer { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #856404; margin: 10px 0; }
    .statusbar { background: #1a1a2e; color: #fff; border-radius: 8px; padding: 10px 18px; margin: 8px 0; font-size: 13px; display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
    .statusbar span { color: #aaa; margin-right: 4px; }
    .statusbar .val { color: #fff; font-weight: 600; }
    .statusbar .badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
    .card { background: #fff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 18px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .section-title { font-size: 18px; font-weight: 600; color: #1a1a2e; margin: 20px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3498DB; }
    .risk-critical { color: #8E44AD; font-weight: 700; }
    .risk-high { color: #E74C3C; font-weight: 700; }
    .risk-medium { color: #F39C12; font-weight: 700; }
    .risk-low { color: #2ECC71; font-weight: 700; }
    .stButton > button { border-radius: 8px; font-weight: 600; padding: 8px 24px; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; color: #fff; }
    .high-risk-bar { background: #ffe0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# DEFAULTS: 仅非 widget 状态，widget 默认值由组件参数控制
DEFAULTS = {"result": None, "show_export": False}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

def reset_inputs():
    """安全重置：pop 所有 widget key，然后 rerun 让组件重建"""
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    st.rerun()

# ================================================================
# Sidebar
# ================================================================
def load_demo_case():
    """设置演示案例参数（在 widget 创建前调用）"""
    for k in ["data_mode","province","score","rank","year","sel33","sel12",
              "first_subject","risk_preference","family_budget","accept_adjustment",
              "accept_sino_foreign","accept_far_city","employment_first",
              "postgraduate_first","preferred_cities","excluded_sel","excluded_custom","preferred_sel","preferred_custom"]:
        st.session_state.pop(k, None)
    st.session_state["data_mode"] = "mock"
    st.session_state["province"] = "广东省"
    st.session_state["score"] = 620
    st.session_state["rank"] = 11500
    st.session_state["year"] = 2024
    st.session_state["first_subject"] = "物理"
    st.session_state["sel12"] = ["化学", "生物"]
    st.session_state["sel33"] = ["物理", "化学", "生物"]
    st.session_state["risk_preference"] = "balanced"
    st.session_state["family_budget"] = 20000
    st.session_state["accept_adjustment"] = True
    st.session_state["accept_sino_foreign"] = False
    st.session_state["accept_far_city"] = True
    st.session_state["employment_first"] = False
    st.session_state["postgraduate_first"] = True
    st.session_state["preferred_cities"] = ["广州", "深圳", "杭州", "南京"]
    st.session_state["excluded_sel"] = []
    st.session_state["excluded_custom"] = ""
    st.session_state["preferred_sel"] = []
    st.session_state["preferred_custom"] = ""
    st.session_state["result"] = None

def build_sidebar():
    """构建侧边栏，返回所有用户输入的 dict（绝不写入 widget key）"""
    st.sidebar.button(" 加载演示案例", on_click=load_demo_case, use_container_width=True)
    st.sidebar.markdown("##  数据模式")
    data_mode = st.sidebar.selectbox("选择数据源", ["mock", "gd_sample", "real"],
        format_func=lambda x: {"mock":" 模拟数据 Demo","gd_sample":" 广东样例数据","real":" 真实数据模式"}[x],
        index=0, key="data_mode")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  考生信息")
    provinces = ["河北省","山东省","广东省","江苏省","浙江省","四川省","湖北省","湖南省","河南省","安徽省","福建省","辽宁省","北京市","上海市","天津市","重庆市","陕西省","山西省"]
    province = st.sidebar.selectbox("省份", provinces, index=2, key="province")
    c1, c2, c3 = st.sidebar.columns(3)
    score = c1.number_input("分数", 0, 750, 620, key="score")
    rank = c2.number_input("位次", 1, 99999999, 11500, key="rank")
    year = c3.number_input("年份", 2020, 2030, 2026, key="year")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  新高考选科")
    PROVINCE_MODE = {
        "北京":"3+3","北京市":"3+3","天津":"3+3","天津市":"3+3",
        "上海":"3+3","上海市":"3+3","浙江":"3+3","浙江省":"3+3",
        "山东":"3+3","山东省":"3+3","海南":"3+3","海南省":"3+3",
        "河北":"3+1+2","河北省":"3+1+2","辽宁":"3+1+2","辽宁省":"3+1+2",
        "江苏":"3+1+2","江苏省":"3+1+2","福建":"3+1+2","福建省":"3+1+2",
        "湖北":"3+1+2","湖北省":"3+1+2","湖南":"3+1+2","湖南省":"3+1+2",
        "广东":"3+1+2","广东省":"3+1+2","重庆":"3+1+2","重庆市":"3+1+2",
        "吉林":"3+1+2","吉林省":"3+1+2","黑龙江":"3+1+2","黑龙江省":"3+1+2",
        "安徽":"3+1+2","安徽省":"3+1+2","江西":"3+1+2","江西省":"3+1+2",
        "广西":"3+1+2","贵州":"3+1+2","甘肃省":"3+1+2",
        "山西":"3+1+2","山西省":"3+1+2","内蒙古":"3+1+2",
        "河南":"3+1+2","河南省":"3+1+2","四川":"3+1+2","四川省":"3+1+2",
        "云南":"3+1+2","云南省":"3+1+2","陕西":"3+1+2","陕西省":"3+1+2",
        "青海":"3+1+2","青海省":"3+1+2","宁夏":"3+1+2",
        "新疆":"文理分科","西藏":"文理分科",
    }
    auto_mode = PROVINCE_MODE.get(province, "3+1+2")
    st.sidebar.caption(f" 当前模式: {auto_mode}（根据省份自动确定）")

    all_six = ["物理", "化学", "生物", "历史", "地理", "政治"]
    RE_ELECT_12 = ["化学", "生物", "地理", "政治"]

    selected_final = []
    subject_type = ""
    subject_combo = ""

    if auto_mode == "文理分科":
        st_type = st.sidebar.radio("科类", ["理科", "文科"], index=0, key="subj_type_radio")
        first_s = "物理" if st_type == "理科" else "历史"
        re_s = ["化学","生物"] if st_type == "理科" else ["地理","政治"]
        selected_final = [first_s] + re_s
        subject_type = "物理类" if st_type == "理科" else "历史类"
        subject_combo = "理科综合" if st_type == "理科" else "文科综合"

    elif auto_mode == "3+3":
        def_33 = safe_multiselect_default(
            st.session_state.get("sel33"), all_six,
            fallback=["物理", "化学", "生物"], max_count=3)
        sel33 = st.sidebar.multiselect("选择 3 门科目 (六选三)", all_six,
            default=def_33, max_selections=3, key="sel33")
        selected_final = list(sel33)
        subject_type = "综合类"
        subject_combo = "/".join(sel33) if sel33 else ""

    else:  # 3+1+2
        first_s = st.sidebar.radio("首选科目 (必选一门)", ["物理", "历史"], index=0, key="first_subject")
        def_12 = safe_multiselect_default(
            st.session_state.get("sel12"), RE_ELECT_12,
            fallback=["化学", "生物"], max_count=2)
        sel12 = st.sidebar.multiselect("再选科目 (四选二)", RE_ELECT_12,
            default=def_12, max_selections=2, key="sel12")
        selected_final = [first_s] + list(sel12)
        subject_type = "物理类" if first_s == "物理" else "历史类"
        subject_combo = "/".join(selected_final) if selected_final else ""
        st.sidebar.caption(f" 选科: {subject_combo or '未选择'}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  方案偏好")
    risk_preference = st.sidebar.radio("风险偏好", ["balanced","aggressive","conservative"],
        format_func=lambda x: {"balanced":" 均衡型","aggressive":" 激进型","conservative":" 保守型"}[x],
        index=0, key="risk_preference")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  约束条件")
    family_budget = st.sidebar.number_input("家庭预算 (元/年)", 0, 200000, 20000, 1000, key="family_budget")
    accept_adjustment = st.sidebar.checkbox("接受调剂", True, key="accept_adjustment")
    accept_sino_foreign = st.sidebar.checkbox("接受中外合作", False, key="accept_sino_foreign")
    accept_far_city = st.sidebar.checkbox("接受远程城市", True, key="accept_far_city")
    employment_first = st.sidebar.checkbox("优先就业", False, key="employment_first")
    postgraduate_first = st.sidebar.checkbox("优先升学", True, key="postgraduate_first")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  偏好城市")
    all_cities = ["北京","上海","广州","深圳","杭州","南京","武汉","成都","西安","天津","重庆","苏州","青岛","厦门","长沙","大连","合肥","郑州"]
    preferred_cities = st.sidebar.multiselect("选择偏好城市", all_cities,
        default=["广州","深圳","杭州","南京"], key="preferred_cities")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  排斥专业")
    excluded_opts = ["医学","师范","土木","化学","哲学","历史学","心理学","英语","工商管理","中外合作"]
    excluded_sel = st.sidebar.multiselect("勾选排斥类别", excluded_opts, default=[], key="excluded_sel")
    excluded_custom = st.sidebar.text_input("手动输入(逗号分隔)", value="", key="excluded_custom")
    excluded_majors = list(excluded_sel)
    if excluded_custom:
        excluded_majors += [x.strip() for x in excluded_custom.split(",") if x.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("##  心仪专业")
    preferred_opts = ["计算机","软件","电子信息","人工智能","数据科学","自动化","电气","机械","土木","建筑","临床医学","口腔医学","数学","物理","化学","生物","金融","会计","法学","汉语言","英语","心理学"]
    preferred_sel = st.sidebar.multiselect("勾选心仪方向(加权推荐)", preferred_opts, default=[], key="preferred_sel")
    preferred_custom = st.sidebar.text_input("或手动输入(逗号分隔)", key="preferred_custom")
    preferred_majors = list(preferred_sel)
    if preferred_custom:
        preferred_majors += [x.strip() for x in preferred_custom.split(",") if x.strip()]

    st.sidebar.markdown("---")
    ca, cb = st.sidebar.columns(2)
    generate_btn = ca.button(" 生成方案", type="primary", use_container_width=True)
    if cb.button(" 重置", use_container_width=True): reset_inputs()

    return {
        "generate_btn": generate_btn,
        "data_mode": data_mode, "province": province, "score": score, "rank": rank, "year": year,
        "exam_mode": auto_mode, "selected_subjects": selected_final,
        "subject_combo": subject_combo, "subject_type": subject_type,
        "risk_preference": risk_preference, "family_budget": family_budget,
        "accept_adjustment": accept_adjustment, "accept_sino_foreign": accept_sino_foreign,
        "accept_far_city": accept_far_city, "employment_first": employment_first,
        "postgraduate_first": postgraduate_first, "preferred_cities": preferred_cities,
        "excluded_majors": excluded_majors, "preferred_majors": preferred_majors,
        "first_subject": selected_final[0] if selected_final else None,
    }

# ================================================================
# Model Runner
# ================================================================
@st.cache_data(show_spinner=False)
def _load_real_gd_data(province, year, subject_type):
    """加载广东真实数据，校验省份和科类"""
    if "广东" not in str(province):
        return None, "广东样例数据仅支持广东省"
    if "综合" in str(subject_type) or "3+3" in str(subject_type):
        return None, "广东样例数据暂不支持综合类，请切换为模拟数据或3+1+2省份"
    import os
    seg_parts = []
    is_phys = "物理" in str(subject_type) or "physics" in str(subject_type).lower()
    is_hist = "历史" in str(subject_type) or "history" in str(subject_type).lower()
    if not is_phys and not is_hist:
        return None, f"不支持的科类: {subject_type}"
    for y in [2021, 2022, 2023, 2024, 2025, 2026]:
        for f in os.listdir("data"):
            if f"segment_{y}" in f:
                df = pd.read_csv(f"data/{f}")
                st = str(df["subject_type"].iloc[0])
                if (is_phys and ("物理" in st or "physics" in st.lower())) or \
                   (is_hist and ("历史" in st or "history" in st.lower())):
                    seg_parts.append(df)
                    break
    if not seg_parts:
        return None, "未找到广东段表数据"
    seg_all = pd.concat(seg_parts, ignore_index=True)
    # 按 subject_type 列物理类
    def _is_physics(val):
        s = str(val)
        return "physics" in s.lower() or (len(s) >= 2 and ord(s[0]) in (29289, 29702))
    if is_phys:
        seg_all = seg_all[seg_all["subject_type"].apply(_is_physics)].copy()
    elif is_hist:
        seg_all = seg_all[~seg_all["subject_type"].apply(_is_physics)].copy()

    adm_parts = []
    for f in os.listdir("data"):
        if "admission_line" in f:
            df = pd.read_csv(f"data/{f}")
            st = str(df["subject_type"].iloc[0])
            if (is_phys and ("物理" in st or "physics" in st.lower())) or \
               (is_hist and ("历史" in st or "history" in st.lower())):
                adm_parts.append(df)
    if not adm_parts:
        return None, "未找到广东投档线数据"
    adm_all = pd.concat(adm_parts, ignore_index=True)

    sch = [f for f in os.listdir("data") if "school_info" in f]
    plan = [f for f in os.listdir("data") if "admission_plan" in f]
    plan_file = plan[0] if plan else None
    maj = [f for f in os.listdir("data") if "major_admission" in f]
    emp = [f for f in os.listdir("data") if "major_employment" in f]
    city = [f for f in os.listdir("data") if "city_data" in f]
    if sch and plan_file and maj and emp and city:
        return {
            "segment_table": seg_all,
            "school_info": pd.read_csv(f"data/{sch[0]}"),
            "school_admission_line": adm_all,
            "major_admission": pd.read_csv(f"data/{maj[0]}"),
            "admission_plan": pd.read_csv(f"data/{plan_file}"),
            "major_employment": pd.read_csv(f"data/{emp[0]}"),
            "city_data": pd.read_csv(f"data/{city[0]}"),
        }, None
    return None, "数据文件不完整"


@st.cache_data(show_spinner=False)
def run_model(province, score, rank, year, subject_type, risk_preference, family_budget,
               accept_adjustment, accept_sino_foreign, accept_far_city,
               employment_first, postgraduate_first, preferred_cities, excluded_majors,
               exam_mode="", subject_combo="", data_mode="mock"):
    err_msg = None
    pipeline_data = None

    if data_mode == "mock":
        data = generate_all_data(province=province, year=year, subject_type=subject_type, candidate_score=score, candidate_rank=rank)
        profile = data["candidate_profile"]
        profile.update({
            "plan_type": risk_preference, "year": year, "province": province,
            "subject_type": subject_type, "family_budget": family_budget,
            "accept_adjustment": 1 if accept_adjustment else 0,
            "accept_sino_foreign": 1 if accept_sino_foreign else 0,
            "accept_far_city": 1 if accept_far_city else 0,
            "employment_first": 1 if employment_first else 0,
            "postgraduate_first": 1 if postgraduate_first else 0,
            "preferred_cities": preferred_cities, "excluded_majors": excluded_majors,
            "preferred_majors": st.session_state.get("preferred_majors", []),
            "exam_mode": exam_mode, "subject_combo": subject_combo,
            "selected_subjects": st.session_state.get("_selected_subjects", []),
            "first_subject": st.session_state.get("_first_subject"),
            "interest_direction": ["计算机", "电子信息"],
        })
        pipeline_data = {"segment_table": data["segment_table"], "school_info": data["school_info"],
            "school_admission_line": data["school_admission_line"], "major_admission": data["major_admission"],
            "admission_plan": data["admission_plan"], "major_employment": data["major_employment"],
            "city_data": data["city_data"]}
    elif data_mode == "gd_sample":
        gd_data, gd_err = _load_real_gd_data(province, year, subject_type)
        if gd_data:
            pipeline_data = gd_data
            # 使用投档线最新年份作为模型 year（兼容 year 不匹配）
            model_year = int(gd_data["school_admission_line"]["year"].max())
            profile = {
                "candidate_id": "gd_ui", "province": province, "year": model_year,
                "subject_type": subject_type, "score": score, "rank": rank,
                "plan_type": risk_preference, "family_budget": family_budget,
                "accept_adjustment": 1 if accept_adjustment else 0,
                "accept_sino_foreign": 1 if accept_sino_foreign else 0,
                "accept_far_city": 1 if accept_far_city else 0,
                "employment_first": 1 if employment_first else 0,
                "postgraduate_first": 1 if postgraduate_first else 0,
                "preferred_cities": preferred_cities, "excluded_majors": excluded_majors,
                "preferred_majors": st.session_state.get("preferred_majors", []),
                "exam_mode": exam_mode, "subject_combo": subject_combo,
                "selected_subjects": st.session_state.get("_selected_subjects", []),
                "first_subject": st.session_state.get("_first_subject"),
                "interest_direction": ["计算机", "电子信息"],
            }
        else:
            err_msg = gd_err
    else:  # real
        err_msg = "真实数据模式需要手动配置 data/processed 目录。请先使用模拟数据 Demo 或广东样例数据。"

    if err_msg:
        raise RuntimeError(err_msg)

    pipeline = GaoKaoPipeline()
    pipeline.load_data(pipeline_data)
    output = pipeline.run(profile=profile, plan_type=risk_preference)
    return json.loads(output)

# ================================================================
# Status Bar
# ================================================================
def render_status_bar(parsed, ui_meta=None):
    ui_meta = ui_meta or {}
    plan_cn = {"aggressive":"激进型","balanced":"均衡型","conservative":"保守型"}
    plan = parsed["recommendation_plan"]["plan_type"]
    r = parsed["recommendation_plan"]["risk_assessment"]
    risk_level = r.get("risk_level", "low")
    rc = RISK_COLORS.get(risk_level, "#999")
    review = "是" if r.get("review_required") else "否"
    exam = ui_meta.get("exam_mode", "")
    subj = ui_meta.get("subject_combo", "")
    dm = ui_meta.get("data_mode", "mock")
    dm_label = {"mock":"模拟数据 Demo","gd_sample":"广东样例数据","real":"真实数据模式"}.get(dm, dm)
    dm_color = {"mock":"#F39C12","gd_sample":"#3498DB","real":"#2ECC71"}.get(dm, "#999")
    st.markdown(f"""
    <div class="statusbar">
        <span>数据:</span><span class="badge" style="background:{dm_color}">{dm_label}</span>
        <span>方案:</span><span class="val">{plan_cn.get(plan,plan)}</span>
        <span>选科:</span><span class="val">{exam} {subj}</span>
        <span>版本:</span><span class="val">{VERSION}</span>
        <span>风险:</span><span class="badge" style="background:{rc}">{risk_level}</span>
        <span>需复核:</span><span class="val">{review}</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# Render: Overview
# ================================================================
def render_overview(parsed):
    s = parsed["recommendation_plan"]["statistics"]
    r = parsed["recommendation_plan"]["risk_assessment"]
    vols = parsed["recommendation_plan"]["volunteers"]
    avg_prob = np.mean([v.get("admit_probability",0) for v in vols]) if vols else 0
    review_count = sum(1 for v in vols if v.get("review_required"))

    st.markdown('<div class="section-title"> 推荐总览</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric(" 总分", f"{parsed['candidate']['score']}", f"位次 {parsed['candidate']['rank']:,}")
    cols[1].metric(" 等效分", f"{parsed['candidate'].get('equivalent_score','')}", f"置信度 {parsed['candidate'].get('confidence_level','')}")
    cols[2].metric(" 综合风险", r.get("risk_level",""), f"评分 {r.get('overall_risk_score',0):.2f}")
    cols[3].metric(" 需复核", f"{review_count} 个", "高风险冲刺")

    ui_meta = st.session_state.get("result_meta", {})
    st.markdown(f"**新高考模式**: {ui_meta.get('exam_mode','')}  |  **选科组合**: {ui_meta.get('subject_combo','')}  |  **模型口径**: {ui_meta.get('subject_type','')}")

    st.markdown("---")
    st.markdown("### 冲稳保垫分布")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" 冲", s["rush_count"], delta=f"{s['rush_count']/max(1,sum([s['rush_count'],s['stable_count'],s['safe_count'],s['bottom_count']])):.0%}", delta_color="off")
    c2.metric(" 稳", s["stable_count"])
    c3.metric(" 保", s["safe_count"])
    c4.metric(" 垫", s["bottom_count"])

    chart_data = pd.DataFrame({"冲":[s["rush_count"]],"稳":[s["stable_count"]],"保":[s["safe_count"]],"垫":[s["bottom_count"]]})
    st.bar_chart(chart_data, use_container_width=True, color=["#E67E22","#3498DB","#2ECC71","#1ABC9C"])

# ================================================================
# Render: Volunteer Table
# ================================================================
def render_volunteer_table(parsed):
    st.markdown('<div class="section-title"> 志愿推荐表</div>', unsafe_allow_html=True)

    df = volunteers_to_dataframe(parsed["recommendation_plan"]["volunteers"])
    if df.empty: st.warning("暂无志愿数据"); return

    # Filters
    c1, c2, c3 = st.columns(3)
    tier_filter = c1.multiselect("按等级筛选", ["rush","stable","safe","bottom"], default=["rush","stable","safe","bottom"], format_func=lambda x: TIER_LABEL.get(x,x))
    risk_filter = c2.multiselect("按风险筛选", ["low","medium","high","very_high"], default=["low","medium","high","very_high"])
    search = c3.text_input("搜索院校/专业", placeholder="关键词...")

    df["_tier_cn"] = df["推荐等级"].map({"rush":"冲","stable":"稳","safe":"保","bottom":"垫"}).fillna("")
    filtered = df[(df["推荐等级"].isin(tier_filter)) & (df["风险等级"].isin(risk_filter))]
    if search:
        filtered = filtered[filtered["院校"].str.contains(search,na=False) | filtered["专业"].str.contains(search,na=False)]

    st.caption(f"共 {len(filtered)} 个志愿")

    # Render as HTML cards for better visuals
    for _, row in filtered.iterrows():
        p = row["录取概率"]
        tier = row["推荐等级"]
        tc = TIER_COLORS.get(tier, "#999")
        risk = row["风险等级"]
        rc = RISK_COLORS.get(risk, "#999")
        is_high_risk = p < 0.20
        high_risk_badge = '<span class="tag" style="background:#E74C3C;margin-left:6px">高风险冲刺</span>' if is_high_risk else ''
        review_badge = '<span class="tag" style="background:#8E44AD;margin-left:6px">需复核</span>' if row["复查"] == "是" else ''

        with st.expander(f"#{row['序号']} {row['院校']} — {row['专业']} | {fmt_prob(p)} | {TIER_LABEL.get(tier,tier)}"):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"<h4>{row['院校']} · {row['专业']}</h4>", unsafe_allow_html=True)
                st.markdown(f'<span class="tag" style="background:{tc}">{TIER_LABEL.get(tier,tier)}</span> <span class="tag" style="background:{rc}">{risk}</span>{high_risk_badge}{review_badge}', unsafe_allow_html=True)
                st.markdown(f"**推荐理由**\n\n{row['解释']}")
                st.markdown(f"**修改建议**\n\n{row['建议']}")
            with col_b:
                st.metric("录取概率", fmt_prob(p), delta=f"区间 {row['概率区间']}", delta_color="off")
                st.metric("就业评分", f"{row['就业评分']:.2f}")
                st.metric("城市评分", f"{row['城市评分']:.2f}")
                st.metric("综合效用", f"{row['综合效用']:.3f}")

# ================================================================
# Render: Risk Panel
# ================================================================
def render_risk_panel(parsed):
    st.markdown('<div class="section-title"> 风险分析</div>', unsafe_allow_html=True)
    risk = parsed["recommendation_plan"]["risk_assessment"]
    risk_df = risk_to_dataframe(risk)
    if risk_df.empty: st.info("暂无风险数据"); return

    for _, row in risk_df.iterrows():
        score = row["评分"]
        level = row["等级"]
        color = RISK_COLORS.get(level, "#999")
        label = RISK_LABELS.get(level, level)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{row['风险类别']}**")
        c1.progress(min(1.0, score))
        c2.markdown(f'<span class="tag" style="background:{color}">{label} ({score:.2f})</span>', unsafe_allow_html=True)
        if row.get("触发原因"): st.caption(f"  {row['触发原因']}")
        if row.get("修改建议"): st.caption(f"  {row['修改建议']}")

    reasons = risk.get("risk_reason", [])
    if reasons:
        st.markdown("**识别到的问题:**")
        for rr in reasons: st.warning(rr)
    sug = risk.get("modification_suggestion", "")
    if sug: st.info(f"**修改建议**: {sug}")

def render_consultant_checklist(parsed):
    st.markdown('<div class="section-title"> 咨询师复核清单</div>', unsafe_allow_html=True)
    vols = parsed["recommendation_plan"]["volunteers"]
    r = parsed["recommendation_plan"]["risk_assessment"]
    items = []
    high_risk = [v for v in vols if v.get("admit_probability",0) < 0.20]
    if high_risk: items.append(f"  **{len(high_risk)} 个志愿 < 20%** (高风险冲刺)，建议逐项复核是否保留")
    review_items = [v for v in vols if v.get("review_required")]
    if review_items: items.append(f"  **{len(review_items)} 个志愿触发人工复核**")
    red_career = [v for v in vols if v.get("career_score",1) < 0.3]
    if red_career: items.append(f"  **{len(red_career)} 个专业就业评分 < 0.3**")
    adj = r.get("adjustment_risk",{})
    if isinstance(adj,dict) and adj.get("score",0) > 0.4: items.append("  **调剂风险较高**，检查专业组内冷热差异")
    reg = r.get("region_risk",{})
    if isinstance(reg,dict) and reg.get("score",0) > 0.4: items.append("  **地域约束未完全满足**")
    if not items: items.append("  未发现明显需复核事项")
    else: items.append(""); items.append("所有数据为模拟数据，真实使用前须接入官方公开数据复核。")
    for it in items: st.markdown(it)

# ================================================================
# Render: Career Panel
# ================================================================
def render_career_panel(parsed):
    st.markdown('<div class="section-title"> 专业就业景气度</div>', unsafe_allow_html=True)
    vols = parsed["recommendation_plan"]["volunteers"]
    if not vols: st.info("暂无就业数据"); return

    career_data = []
    seen = set()
    for v in vols:
        mc = v.get("major_code","")
        if mc and mc not in seen:
            seen.add(mc)
            cs = v.get("career_score", 0)
            # 从 career_score 计算标签 (与 CareerEvaluator._employment_level 阈值一致)
            if cs >= 0.66:
                label = "green"
            elif cs >= 0.33:
                label = "yellow"
            else:
                label = "red"
            career_data.append({
                "专业": v.get("major_name",""),
                "就业评分": round(cs, 2),
                "景气度": label,
                "评级": {"green":"高","yellow":"中","red":"低"}.get(label, label),
                "需复核": "是" if v.get("review_required") else "否",
            })

    if career_data:
        cdf = pd.DataFrame(career_data).sort_values("就业评分", ascending=False)
        def label_bg(val):
            c = LABEL_COLORS.get(str(val), "#999")
            return f'background-color: {c}; color: white; font-weight: 600'
        styled = cdf.style.format({"就业评分":"{:.2f}"}).map(label_bg, subset=["景气度"])
        st.dataframe(styled, use_container_width=True)
        green_n = (cdf["景气度"] == "green").sum()
        yellow_n = (cdf["景气度"] == "yellow").sum()
        red_n = (cdf["景气度"] == "red").sum()
        st.caption(f"绿(高) {green_n} | 黄(中) {yellow_n} | 红(低) {red_n}。当前为模拟就业数据，社媒舆情仅辅助预警。")
    else:
        st.info("暂无就业数据")

# ================================================================
# Render: Explanation & Downloads
# ================================================================
def render_explanation(parsed):
    st.markdown('<div class="section-title"> 家长端解释报告</div>', unsafe_allow_html=True)
    s = parsed["recommendation_plan"]["statistics"]
    r = parsed["recommendation_plan"]["risk_assessment"]
    plan = parsed["recommendation_plan"]["plan_type"]
    plan_cn = {"aggressive":"激进型","balanced":"均衡型","conservative":"保守型"}
    parts = [
        f"### 推荐方案说明", "",
        f"本方案采用**{plan_cn.get(plan,plan)}**策略，为您推荐了 **{s['rush_count']} 冲 + {s['stable_count']} 稳 + {s['safe_count']} 保 + {s['bottom_count']} 垫**。",
        f"综合风险等级：**{r.get('risk_level','')}**（{r.get('overall_risk_score',0):.2f}）。",
    ]
    reasons = r.get("risk_reason",[])
    if reasons:
        parts.append(""); parts.append("**需要关注：**")
        for rr in reasons: parts.append(f"- {rr}")
    parts.append(""); parts.append(f"**是否建议复核**：{'是' if r.get('review_required') else '否'}。")
    parts.append(""); parts.append("> 以上分析基于历史数据和公开就业数据，**不构成录取或就业保证**。所有数据均为模拟数据，仅供教学和系统原型展示。")
    st.markdown("\n".join(parts))

def render_downloads(parsed):
    st.markdown('<div class="section-title">  导出</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    json_bytes = export_json_bytes(parsed)
    c1.download_button(" 下载 JSON", data=json_bytes, file_name="gaokao_result.json", mime="application/json", use_container_width=True)
    c2.download_button(" 下载 MD 报告", data=build_report(parsed), file_name="gaokao_report.md", mime="text/markdown", use_container_width=True)
    df = volunteers_to_dataframe(parsed["recommendation_plan"]["volunteers"])
    if not df.empty:
        csv = df[["序号","院校","专业","录取概率","推荐等级","就业评分","城市评分","风险等级","复查"]].to_csv(index=False).encode("utf-8-sig")
        c3.download_button(" 下载 CSV", data=csv, file_name="volunteers.csv", mime="text/csv", use_container_width=True)

def build_report(parsed):
    ui_meta = st.session_state.get("result_meta", {})
    s = parsed["recommendation_plan"]["statistics"]
    r = parsed["recommendation_plan"]["risk_assessment"]
    c = parsed["candidate"]
    lines = [
        "# 高考志愿推荐报告", f"生成时间: {parsed['meta']['generate_time']} | 版本: {VERSION}", "",
        f"**考生**: {c['province']} | {c['score']}分 | 位次{c['rank']} | {c['subject_type']} | {parsed['recommendation_plan']['plan_type']}方案", "",
        f"**选科模式**: {ui_meta.get('exam_mode','')} | **选科组合**: {ui_meta.get('subject_combo','')}", "",
        f"**等效分**: {c.get('equivalent_score','')} | 区间: {c.get('equivalent_score_interval','')} | 置信度: {c.get('confidence_level','')}", "",
        f"**志愿结构**: 冲{s['rush_count']} | 稳{s['stable_count']} | 保{s['safe_count']} | 垫{s['bottom_count']}", "",
        f"**风险**: {r.get('risk_level','')} ({r.get('overall_risk_score',0):.2f}) | 需复核: {'是' if r.get('review_required') else '否'}", "",
        "## 推荐志愿 (前10个)", "",
        "| # | 院校 | 专业 | 概率 | 等级 | 就业 | 风险 |",
        "|---|------|------|------|------|------|------|",
    ]
    for v in parsed["recommendation_plan"]["volunteers"][:10]:
        lines.append(f"| {v['volunteer_id']} | {v['school_name']} | {v['major_name']} | {fmt_prob(v['admit_probability'])} | {TIER_LABEL.get(v['recommendation_tier'],'')} | {v.get('career_score',0):.2f} | {v.get('risk_level','')} |")
    lines += ["", "## 风险 & 复核", ""]
    for rr in r.get("risk_reason",[]): lines.append(f"- {rr}")
    lines += ["", "## 免责声明", "", "本报告由系统基于模拟数据自动生成，仅供教学和系统原型展示，**不构成录取保证**。真实使用前必须接入官方公开数据并由咨询师复核。"]
    return "\n".join(lines)

# ================================================================
# Main
# ================================================================
def main():
    st.markdown(f'<div class="main-header"> 基于大数据和 AI 的高考志愿智能推荐系统</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">输入分数、位次与偏好，生成冲稳保垫志愿方案 | {VERSION}</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer"><strong> 模拟数据演示版本</strong> — 推荐结果不能作为真实高考志愿填报依据。真实使用前须接入官方公开数据并由咨询师复核。</div>', unsafe_allow_html=True)

    ui = build_sidebar()
    # 将选科结果存入 session_state（非 widget key，安全）
    st.session_state["_first_subject"] = ui.get("first_subject")
    st.session_state["_selected_subjects"] = ui["selected_subjects"]

    if ui["generate_btn"]:
        errors = validate_inputs(ui["province"], ui["score"], ui["rank"])
        if ui["exam_mode"] in ("3+1+2", ""):
            subj_errors = validate_subject_selection(
                exam_mode=ui["exam_mode"],
                first_subject=ui.get("first_subject"),
                selected_subjects_12=[s for s in ui["selected_subjects"] if s in ("化学","生物","地理","政治")],
            )
        else:
            subj_errors = validate_subject_selection(
                exam_mode=ui["exam_mode"],
                selected_subjects=ui["selected_subjects"],
            )
        errors += subj_errors
        if ui["data_mode"] == "gd_sample" and ui["province"] != "广东省":
            errors.append("广东样例数据模式仅支持广东省，请切换省份或使用模拟数据 Demo。")
        if errors:
            for e in errors: st.error(e)
        else:
            with st.spinner("正在分析数据、计算录取概率、优化志愿方案..."):
                try:
                    parsed = run_model(
                        ui["province"], ui["score"], ui["rank"],
                        ui["year"], ui["subject_type"], ui["risk_preference"],
                        ui["family_budget"], ui["accept_adjustment"],
                        ui["accept_sino_foreign"], ui["accept_far_city"],
                        ui["employment_first"], ui["postgraduate_first"],
                        ui["preferred_cities"], ui["excluded_majors"],
                        exam_mode=ui["exam_mode"],
                        subject_combo=ui["subject_combo"],
                        data_mode=ui["data_mode"],
                    )
                    st.session_state.result = parsed
                    st.session_state.result_meta = ui  # save UI inputs for display
                except Exception as e:
                    st.error(f"模型运行失败: {e}")

    if st.session_state.result:
        parsed = st.session_state.result
        ui_meta = st.session_state.get("result_meta", {})
        render_status_bar(parsed, ui_meta)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([" 推荐总览"," 志愿表"," 风险分析"," 专业就业"," 解释报告"," 数据说明"])

        with tab1:
            render_overview(parsed)
        with tab2:
            render_volunteer_table(parsed)
        with tab3:
            render_risk_panel(parsed)
            st.markdown("---")
            render_consultant_checklist(parsed)
        with tab4:
            render_career_panel(parsed)
        with tab5:
            render_explanation(parsed)
        with tab6:
            st.markdown("### 数据来源与免责声明")
            st.markdown("- 模拟数据由 `data_generator.py` 生成，不反映真实院校/专业/考生信息\n- 录取概率上限 0.99\n- 社媒舆情仅辅助预警\n- 所有推荐不构成录取或就业保证")
            st.json(parsed["meta"])
        if st.session_state.show_export:
            st.markdown("---")
            render_downloads(parsed)
    else:
        # Landing page
        st.markdown("---")

        # Demo params card
        with st.expander(" 推荐演示参数", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.markdown("**数据模式**: 模拟数据 Demo  \n**省份**: 广东省  \n**分数**: 620  \n**位次**: 11500")
            col2.markdown("**新高考模式**: 3+1+2  \n**选科**: 物理 / 化学 / 生物  \n**风险偏好**: 均衡型")
            col3.markdown("**偏好城市**: 广州、深圳  \n**家庭预算**: 20000 元/年  \n**接受调剂**: 是")
            st.caption(" 点击左侧 **加载演示案例** 按钮可自动填入以上参数。")

        # System flow
        with st.expander(" 系统流程说明", expanded=False):
            steps = [
                ("1. 输入考生信息", "省份、分数、位次、选科组合、个人偏好"),
                ("2. 分数—位次等效换算", "分位数映射 + 线差修正 → 跨年可比等效分"),
                ("3. 院校/专业录取概率预测", "LR + XGBoost + Bayes + MC 四引擎融合"),
                ("4. 专业就业景气度评价", "AHP + 熵权 + TOPSIS → 红黄绿标签"),
                ("5. 冲稳保志愿组合优化", "硬约束过滤 + 贪心 + 局部搜索 → 完整志愿表"),
                ("6. 志愿填报风险评估", "6 类风险评分 + 10 项问题扫描"),
                ("7. 生成解释报告", "家长端 + 咨询师端 + 系统后台三层解释"),
            ]
            for title, desc in steps:
                st.markdown(f"**{title}**  \n{desc}  \n")

        # Landing cards
        st.markdown("###  系统能力")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="card"><h4> 位次换算</h4><p>通过分位数映射和线差修正，将您的分数科学换算为往年等效分，解决跨年分数不可比的问题。</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="card"><h4> 录取概率</h4><p>LR+XGBoost+Bayes+MC 四方法融合，给出录取概率和置信区间。</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="card"><h4> 志愿优化</h4><p>基于多目标效用函数，自动生成冲稳保垫合理结构，考虑排斥专业、预算和城市偏好。</p></div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        c4.markdown('<div class="card"><h4> 风险预警</h4><p>识别滑档/退档/调剂/冷门/就业/地域六类风险，扫描十大典型问题。</p></div>', unsafe_allow_html=True)
        c5.markdown('<div class="card"><h4> 就业评价</h4><p>AHP+熵权+TOPSIS 多维评价，输出红黄绿标签和细分评分。</p></div>', unsafe_allow_html=True)
        st.info(" 在左侧栏输入分数、位次和偏好，点击 **生成方案** 开始。")

if __name__ == "__main__":
    main()
