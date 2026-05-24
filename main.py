"""
基于大数据和 AI 的高考志愿填报多目标推荐模型
主入口程序
12345678910
用法:
    python main.py                              # 默认运行均衡方案演示
    python main.py --plan aggressive             # 激进型方案
    python main.py --plan conservative           # 保守型方案
    python main.py --score 650 --rank 3000       # 指定考生分数和位次
    python main.py --test                        # 运行测试案例集
    python main.py --export-json result.json     # 导出 JSON 结果
    python main.py --export-report report.md     # 导出 Markdown 报告
    python main.py --use-mock-data               # 使用模拟数据(默认)
    python main.py --help                        # 查看帮助

所有数据均为模拟数据，仅供教学和测试使用。
"""

import sys
import json
import argparse
import os

from src.data_generator import generate_all_data
from src.cleaner import clean_segment_data, clean_admission_line_data, clean_major_admission_data
from src.pipeline import GaoKaoPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于大数据和AI的高考志愿填报多目标推荐模型 v3.0.0"
    )
    parser.add_argument(
        "--plan", type=str, default="balanced",
        choices=["aggressive", "balanced", "conservative"],
        help="风险偏好方案类型 (默认: balanced)",
    )
    parser.add_argument(
        "--score", type=int, default=620,
        help="考生分数 (默认: 620)",
    )
    parser.add_argument(
        "--rank", type=int, default=8500,
        help="考生位次 (默认: 8500)",
    )
    parser.add_argument(
        "--province", type=str, default="河北省",
        help="考生省份 (默认: 河北省)",
    )
    parser.add_argument(
        "--subject", type=str, default="物理类",
        help="科类 (默认: 物理类)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出JSON文件路径 (默认: 打印到控制台)",
    )
    parser.add_argument(
        "--export-json", type=str, default=None,
        help="导出完整JSON到文件 (默认: 不导出)",
    )
    parser.add_argument(
        "--export-report", type=str, default=None,
        help="导出Markdown评估报告到文件 (默认: 不导出)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="运行测试案例集",
    )
    parser.add_argument(
        "--use-mock-data", action="store_true", default=True,
        help="使用模拟数据 (默认: True，仅用于测试)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印详细中间结果",
    )
    return parser.parse_args()


def run_test_cases(pipeline, args):
    """运行测试案例集"""
    print("\n" + "=" * 60)
    print("  测试案例集")
    print("=" * 60)

    test_cases = [
        {"name": "案例1:高分冲名校", "score": 680, "rank": 500, "plan": "aggressive", "province": "河北省"},
        {"name": "案例2:中分均衡型", "score": 550, "rank": 50000, "plan": "balanced", "province": "河北省"},
        {"name": "案例3:低分保底型", "score": 450, "rank": 150000, "plan": "conservative", "province": "河北省"},
        {"name": "案例4:不接受调剂", "score": 620, "rank": 8500, "plan": "balanced", "province": "河北省"},
        {"name": "案例5:排斥专业", "score": 620, "rank": 8500, "plan": "balanced", "province": "河北省"},
        {"name": "案例6:地域约束", "score": 620, "rank": 8500, "plan": "balanced", "province": "河北省"},
        {"name": "案例7:小样本专业", "score": 620, "rank": 8500, "plan": "balanced", "province": "河北省"},
    ]

    all_passed = True
    for tc in test_cases:
        print(f"\n  {tc['name']}: 分数={tc['score']}, 位次={tc['rank']}, 方案={tc['plan']}")
        try:
            data = generate_all_data(
                province=tc["province"],
                year=2024,
                subject_type=args.subject,
                candidate_score=tc["score"],
                candidate_rank=tc["rank"],
            )
            profile = data["candidate_profile"]
            profile.update({
                "plan_type": tc["plan"],
                "year": 2024,
                "province": tc["province"],
            })
            if tc["name"] == "案例4:不接受调剂":
                profile["accept_adjustment"] = 0
            if tc["name"] == "案例5:排斥专业":
                profile["excluded_majors"] = ["哲学", "历史学", "心理学", "英语", "工商管理"]
            if tc["name"] == "案例6:地域约束":
                profile["preferred_cities"] = ["北京", "上海"]

            p = GaoKaoPipeline()
            processed = {
                "segment_table": data["segment_table"],
                "school_info": data["school_info"],
                "major_info": data["major_info"],
                "school_admission_line": data["school_admission_line"],
                "major_admission": data["major_admission"],
                "admission_plan": data["admission_plan"],
                "major_employment": data["major_employment"],
                "city_data": data["city_data"],
            }
            p.load_data(processed)
            output = p.run(profile=profile, plan_type=tc["plan"])
            parsed = json.loads(output)
            risk = parsed["recommendation_plan"]["risk_assessment"]
            stats = parsed["recommendation_plan"]["statistics"]
            print(f"    结果: 冲{stats['rush_count']} 稳{stats['stable_count']} "
                  f"保{stats['safe_count']} 垫{stats['bottom_count']} "
                  f"风险:{risk['risk_level']}({risk['overall_risk_score']})")
        except Exception as e:
            print(f"    异常: {e}")
            all_passed = False

    print(f"\n  测试完成: {'全部通过' if all_passed else '存在异常'}")
    return 0 if all_passed else 1


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  基于大数据和 AI 的高考志愿填报多目标推荐模型")
    print("  Model Version: 3.0.0 (Project 5)")
    print("  所有数据均为模拟数据，仅供教学和测试使用")
    print("=" * 60)

    if args.test:
        pipeline = GaoKaoPipeline()
        return run_test_cases(pipeline, args)

    print(f"\n  考生配置:")
    print(f"    省份: {args.province}")
    print(f"    科类: {args.subject}")
    print(f"    分数: {args.score}")
    print(f"    位次: {args.rank}")
    print(f"    方案: {args.plan}")
    print()

    print("正在生成模拟数据...")
    data = generate_all_data(
        province=args.province,
        year=2024,
        subject_type=args.subject,
        candidate_score=args.score,
        candidate_rank=args.rank,
    )

    profile = data["candidate_profile"]
    profile["plan_type"] = args.plan
    profile["year"] = 2024

    pipeline = GaoKaoPipeline()

    processed_data = {
        "segment_table": data["segment_table"],
        "school_info": data["school_info"],
        "major_info": data["major_info"],
        "school_admission_line": data["school_admission_line"],
        "major_admission": data["major_admission"],
        "admission_plan": data["admission_plan"],
        "major_employment": data["major_employment"],
        "city_data": data["city_data"],
    }
    pipeline.load_data(processed_data)

    json_output = pipeline.run(
        profile=profile,
        plan_type=args.plan,
    )

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"\nJSON 结果已导出到: {args.export_json}")

    if args.export_report:
        report = pipeline.generate_markdown_report()
        with open(args.export_report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Markdown 报告已导出到: {args.export_report}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"\n结果已保存到: {args.output}")
    else:
        parsed = json.loads(json_output)
        stats = parsed["recommendation_plan"]["statistics"]
        risk = parsed["recommendation_plan"]["risk_assessment"]

        print(f"\n{'='*60}")
        print(f"  推荐结果摘要")
        print(f"{'='*60}")
        print(f"  方案类型: {parsed['recommendation_plan']['plan_type']}")
        print(f"  冲{stats['rush_count']} | 稳{stats['stable_count']} | "
              f"保{stats['safe_count']} | 垫{stats['bottom_count']}")
        print(f"  综合评分: {stats['overall_score']}")
        print(f"  综合风险: {risk['risk_level']} "
              f"(评分: {risk['overall_risk_score']})")
        print(f"  需人工复核: {risk['review_required']}")

        if risk.get("risk_reason"):
            print(f"\n  风险提醒:")
            for reason in risk["risk_reason"]:
                print(f"    ! {reason}")

        volunteers = parsed["recommendation_plan"]["volunteers"]
        if volunteers and args.verbose:
            print(f"\n  志愿表详细 (前5个):")
            print(f"  {'序号':<4} {'院校':<12} {'专业':<16} {'概率':<8} {'等级':<6}")
            print(f"  {'-'*50}")
            for i, v in enumerate(volunteers[:5]):
                print(f"  {i+1:<4} {v['school_name']:<12} {v['major_name']:<16} "
                      f"{v['admit_probability']:.0%}  {v['recommendation_tier']:<6}")

        if args.verbose:
            print(f"\n  前5个志愿完整JSON:")
            print(json.dumps(volunteers[:5], ensure_ascii=False, indent=2))

    print(f"\n{'='*60}")
    print(f"  运行完毕")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
