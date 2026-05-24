"""
广东真实高考志愿推荐 — 直接运行脚本
用法: python run_gd.py                          (默认 620/11500/平衡)
      python run_gd.py 650 3000 aggressive        (650分/3000名/激进)
      python run_gd.py 580 35000 conservative --export
"""
import sys, os, json, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import GaoKaoPipeline

# 解析参数
score = int(sys.argv[1]) if len(sys.argv) > 1 else 620
rank  = int(sys.argv[2]) if len(sys.argv) > 2 else 11500
plan  = sys.argv[3] if len(sys.argv) > 3 else "balanced"
export = "--export" in sys.argv

# 加载真实广东数据
seg_parts = []
for y, tags in [(2021, ("2021_物理","2021_历史")),(2022, ("2022_物理","2022_历史")),
                (2023, ("2023_物理","2023_历史")),(2024, ("2024_physics","2024_history")),
                (2025, ("2025_物理","2025_历史")),(2026, ("2026_物理","2026_历史"))]:
    for tag in tags:
        files = [f for f in os.listdir("data") if "segment" in f and tag in f]
        if files: seg_parts.append(pd.read_csv(f"data/{files[0]}"))
seg_all = pd.concat(seg_parts, ignore_index=True)

adm24 = pd.read_csv("data/gd_admission_line_2024_物理类.csv")
adm23 = pd.read_csv("data/gd_admission_line_2023_物理类.csv")

pipeline = GaoKaoPipeline()
pipeline.load_data({
    "segment_table": seg_all,
    "school_info": pd.read_csv("data/gd_school_info.csv"),
    "school_admission_line": pd.concat([adm24, adm23]),
    "major_admission": pd.read_csv("data/gd_major_admission.csv"),
    "admission_plan": pd.read_csv("data/gd_admission_plan_2026.csv"),
    "major_employment": pd.read_csv("data/gd_major_employment.csv"),
    "city_data": pd.read_csv("data/gd_city_data.csv"),
})

profile = {
    "candidate_id": "gd_user", "province": "广东省", "year": 2026,
    "subject_type": "物理类", "score": score, "rank": rank, "plan_type": plan,
    "interest_direction": ["计算机","电子信息"], "excluded_majors": [],
    "preferred_cities": ["广州","深圳"], "family_budget": 25000,
    "accept_adjustment": 1, "accept_sino_foreign": 0, "accept_far_city": 1,
    "employment_first": 0, "postgraduate_first": 1,
    "selected_subjects": ["化学", "生物"], "first_subject": "物理",
    "preferred_majors": ["计算机", "电子信息"],
}

print(f"\n  广东真实数据 · {score}分/{rank}名 · {plan}")
print(f"  段表2021-2025  投档线2023-2024  938校\n")

output = pipeline.run(profile=profile, plan_type=plan)
parsed = json.loads(output)
c = parsed["candidate"]
s = parsed["recommendation_plan"]["statistics"]
r = parsed["recommendation_plan"]["risk_assessment"]
vols = parsed["recommendation_plan"]["volunteers"]

eq = c.get("equivalent_score", "?")
eq_range = c.get("equivalent_score_interval", ["?","?"])
conf = c.get("confidence_level", "?")
print(f"  等效分: {eq} [{eq_range[0]}, {eq_range[1]}]  置信度: {conf}")
print(f"  冲{s['rush_count']} 稳{s['stable_count']} 保{s['safe_count']} 垫{s['bottom_count']}  风险: {r['risk_level']}({r.get('overall_risk_score',0):.2f})\n")

for tier in ["rush", "stable", "safe", "bottom"]:
    tv = [v for v in vols if v["recommendation_tier"] == tier]
    if tv:
        print(f" ── {tier} ({len(tv)}个) ──")
        for v in tv[:5]:
            print(f"  {v['school_name'][:14]:14s} {v['major_name'][:10]:10s}  {v['admit_probability']:.0%}  复核:{'是' if v['review_required'] else '否'}")
        if len(tv) > 5: print(f"  ... 还有 {len(tv)-5} 个")

reasons = r.get("risk_reason", [])
if reasons:
    print(f"\n  风险提醒:")
    for rr in reasons[:3]: print(f"  ! {rr[:100]}")

if export:
    with open("gd_result.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已导出到 gd_result.json")
