"""
核心模型二：院校/专业录取概率预测模型 (Project 5 编号)

功能：预测考生报考某院校/专业的录取概率。
方法：多模型融合 (Logistic回归 + XGBoost + 贝叶斯修正 + 蒙特卡洛模拟)

当前为模拟数据训练版本，仅用于教学、测试和流程验证。
真实上线前必须使用按省份、科类、选科组合分层后的历史投档线、
专业录取位次和招生计划数据重新训练、回测和校准。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


class AdmissionProbabilityPredictor:
    """录取概率预测器 (Project 5 核心模型二)"""

    def __init__(self, rush_threshold=0.20, stable_threshold=0.45,
                 safe_threshold=0.70, bottom_threshold=0.88,
                 mc_iterations=10000):
        """
        Args:
            rush_threshold: 冲刺上界(p<此值为rush)
            stable_threshold: 稳上界(p<此值为stable)
            safe_threshold: 保上界(p<此值为safe)
            bottom_threshold: 垫下界(p>=此值为bottom)
            mc_iterations: 蒙特卡洛模拟次数

        注意：以上阈值仅为默认值，实际使用时应按省份、批次
        和年份回测数据进行校准。
        """
        self.rush_threshold = rush_threshold
        self.stable_threshold = stable_threshold
        self.safe_threshold = safe_threshold
        self.bottom_threshold = bottom_threshold
        self.mc_iterations = mc_iterations
        self.model_lr = None
        self.model_xgb = None
        self.feature_columns = []
        self.is_trained = False
        self.training_meta = {
            "is_simulated": True,
            "note": "当前为模拟数据训练版本，仅用于教学、测试和流程验证。"
                    "真实上线前必须使用按省份、科类、选科组合分层的"
                    "历史数据重新训练、回测和校准。",
        }

    # ================================================================
    # 特征工程
    # ================================================================

    def extract_features(self, candidate_rank, historical_records,
                         plan_series=None, pop_factor=1.0):
        """从历史记录中提取特征向量"""
        records = historical_records.copy()
        if isinstance(records, pd.DataFrame) and len(records) == 0:
            return self._empty_features()

        ranks = np.array(records["min_admission_rank"].values, dtype=float)
        n_years = len(ranks)

        rank_gap_mean = float(candidate_rank - ranks.mean())
        rank_gap_std = float(ranks.std()) if n_years >= 2 else float(abs(rank_gap_mean) * 0.3)
        rank_gap_min = float(candidate_rank - ranks.min())
        rank_gap_max = float(candidate_rank - ranks.max())

        if n_years >= 3:
            x = np.arange(n_years)
            slope = np.polyfit(x, ranks, 1)[0]
        else:
            slope = 0

        rank_volatility = float(ranks.std() / ranks.mean()) if ranks.mean() > 0 else 0

        score_col = "min_admission_score"
        if score_col in records.columns:
            scores = records[score_col].values.astype(float)
            score_diff_mean = float(candidate_rank / 100 - scores.mean())
            score_diff_std = float(scores.std()) if n_years >= 2 else 0
        else:
            score_diff_mean = rank_gap_mean / 100
            score_diff_std = rank_gap_std / 100

        if plan_series is not None and len(plan_series) >= 2:
            plan_change_rate = float(
                (plan_series.iloc[0] - plan_series.iloc[1]) / max(plan_series.iloc[1], 1)
            )
            plan_count_mean = float(plan_series.mean())
        else:
            plan_change_rate = 0
            plan_count_mean = 0

        subject_match = 1 if "subject_requirement" in records.columns and \
            records["subject_requirement"].iloc[0] in ("物理", "物理,化学", "物理,化学,生物") else 0

        features = {
            "rank_gap_mean": rank_gap_mean,
            "rank_gap_std": rank_gap_std,
            "rank_gap_min": rank_gap_min,
            "rank_gap_max": rank_gap_max,
            "rank_gap_trend": slope,
            "rank_volatility": rank_volatility,
            "score_diff_mean": score_diff_mean,
            "score_diff_std": score_diff_std,
            "plan_change_rate": plan_change_rate,
            "plan_count_mean": plan_count_mean,
            "years_with_data": n_years,
            "popularity_factor": pop_factor,
            "subject_requirement_match": subject_match,
        }
        self.feature_columns = list(features.keys())
        return features

    def _empty_features(self):
        defaults = {
            "rank_gap_mean": 0, "rank_gap_std": 0, "rank_gap_min": 0,
            "rank_gap_max": 0, "rank_gap_trend": 0, "rank_volatility": 0,
            "score_diff_mean": 0, "score_diff_std": 0,
            "plan_change_rate": 0, "plan_count_mean": 0,
            "years_with_data": 0, "popularity_factor": 1.0,
            "subject_requirement_match": 0,
        }
        self.feature_columns = list(defaults.keys())
        return defaults

    def features_to_array(self, features_dict):
        """将特征字典转换为 numpy 数组"""
        return np.array([features_dict.get(c, 0) for c in self.feature_columns])

    # ================================================================
    # 模型训练
    # ================================================================

    def fit_logistic(self, X_train, y_train):
        """训练 Logistic 回归模型"""
        self.model_lr = LogisticRegression(max_iter=1000, C=1.0)
        self.model_lr.fit(X_train, y_train)
        if X_train.shape[1] > 0:
            y_pred = self.model_lr.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, y_pred)
            print(f"[LR] Train AUC: {auc:.4f}")

    def fit_xgboost(self, X_train, y_train):
        """训练 XGBoost 模型"""
        import xgboost as xgb
        self.model_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, objective="binary:logistic",
            eval_metric="auc", verbosity=0, random_state=42,
        )
        self.model_xgb.fit(X_train, y_train)
        y_pred = self.model_xgb.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_pred)
        print(f"[XGB] Train AUC: {auc:.4f}")

    def _generate_simulated_training_data(self, n_samples=5000):
        """生成模拟训练数据（仅供演示和测试）"""
        np.random.seed(42)
        n = n_samples
        X = np.zeros((n, len(self.feature_columns)))
        for i, col in enumerate(self.feature_columns):
            if col == "rank_gap_mean":
                X[:, i] = np.random.normal(0, 3000, n)
            elif col == "rank_gap_std":
                X[:, i] = np.abs(np.random.normal(500, 300, n))
            elif col == "rank_gap_min":
                X[:, i] = np.random.normal(-1000, 4000, n)
            elif col == "rank_gap_max":
                X[:, i] = np.random.normal(5000, 3000, n)
            elif col == "rank_gap_trend":
                X[:, i] = np.random.normal(0, 200, n)
            elif col == "rank_volatility":
                X[:, i] = np.abs(np.random.normal(0.05, 0.1, n))
            elif col == "score_diff_mean":
                X[:, i] = np.random.normal(0, 30, n)
            elif col == "score_diff_std":
                X[:, i] = np.abs(np.random.normal(10, 5, n))
            elif col == "plan_change_rate":
                X[:, i] = np.random.normal(0, 0.15, n)
            elif col == "plan_count_mean":
                X[:, i] = np.abs(np.random.normal(15, 8, n))
            elif col == "years_with_data":
                X[:, i] = np.random.randint(1, 6, n).astype(float)
            elif col == "popularity_factor":
                X[:, i] = np.random.uniform(0.5, 2.0, n)
            elif col == "subject_requirement_match":
                X[:, i] = np.random.choice([0, 1], n)
            else:
                X[:, i] = np.random.normal(0, 1, n)

        log_odds = (
            0.001 * X[:, 0] +
            (-0.005) * X[:, 1] +
            0.0005 * X[:, 2] +
            0.0002 * X[:, 3] +
            (-0.001) * X[:, 4] +
            (-3.0) * X[:, 5] +
            0.02 * X[:, 6] +
            (-0.01) * X[:, 7] +
            (-0.5) * X[:, 8] +
            0.03 * X[:, 9] +
            0.4 * (X[:, 10] - 1) +
            0.3 * X[:, 11] +
            0.5 * X[:, 12]
        )
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        prob = np.clip(prob, 0.01, 0.99)
        y = (np.random.random(n) < prob).astype(int)
        return X, y

    def fit_all(self, X_train=None, y_train=None, use_simulated=True):
        """
        训练全部模型（Logistic + XGBoost）

        如果未提供真实训练数据且 use_simulated=True，
        使用模拟数据训练。模拟训练结果不代表真实上线效果。

        Returns:
            dict: 训练元信息
        """
        if X_train is None or y_train is None:
            if use_simulated:
                print("[WARN] 使用模拟数据训练，结果不代表真实效果")
                if not self.feature_columns:
                    self._empty_features()
                X_train, y_train = self._generate_simulated_training_data()
            else:
                print("[WARN] 无训练数据，将使用启发式规则预测")
                self.is_trained = False
                return self.training_meta

        if X_train.shape[0] == 0:
            self.is_trained = False
            return self.training_meta

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        self.fit_logistic(X_train, y_train)
        try:
            self.fit_xgboost(X_train, y_train)
        except Exception as e:
            print(f"[WARN] XGBoost 训练失败: {e}，将回退到 LR")
            self.model_xgb = None

        self.is_trained = True
        if X_train is not None:
            self.training_meta["n_train_samples"] = X_train.shape[0]
            self.training_meta["n_features"] = X_train.shape[1]
        return self.training_meta

    # ================================================================
    # 概率估计核心方法
    # ================================================================

    def _bayesian_adjust(self, p_ml, n_years, prior_alpha=2, prior_beta=2):
        """贝叶斯修正（小样本平滑）"""
        if n_years > 2:
            return p_ml
        effective_successes = p_ml * n_years + 0.1
        p_bayes = (prior_alpha + effective_successes) / (
            prior_alpha + prior_beta + n_years
        )
        return p_bayes

    def bayesian_correction(self, p_ml, n_years, prior_alpha=2, prior_beta=2):
        """贝叶斯修正（公开接口）"""
        return self._bayesian_adjust(p_ml, n_years, prior_alpha, prior_beta)

    def _monte_carlo_simulate(self, candidate_rank, historical_ranks,
                               plan_change_rate=0, n_years=1, eta=0.5):
        """蒙特卡洛模拟录取概率（内部实现）"""
        ranks = np.array(historical_ranks, dtype=float)
        if n_years <= 1:
            mean_rank = float(ranks.mean()) if len(ranks) > 0 else float(candidate_rank + 5000)
            std_rank = max(500, abs(mean_rank - candidate_rank) * 0.3)
        else:
            mean_rank = float(ranks.mean())
            std_rank = float(ranks.std()) if ranks.std() > 0 else 500

        mc_samples = np.random.normal(mean_rank, std_rank, self.mc_iterations)
        plan_adjustment = eta * plan_change_rate * mean_rank
        mc_samples += plan_adjustment

        results = candidate_rank <= mc_samples
        p_mc = float(np.mean(results))
        ci_lower = float(np.percentile(results.astype(float), 2.5))
        ci_upper = float(np.percentile(results.astype(float), 97.5))

        return p_mc, ci_lower, ci_upper

    def monte_carlo_simulation(self, candidate_rank, historical_ranks,
                                plan_change_rate=0, n_years=1, eta=0.5):
        """蒙特卡洛模拟录取概率（公开接口）"""
        return self._monte_carlo_simulate(
            candidate_rank, historical_ranks, plan_change_rate, n_years, eta
        )

    def estimate_probability_interval(self, p_center, ci_lower, ci_upper):
        """估计概率区间（应用小样本扩展等规则）"""
        return [round(max(0.0, ci_lower), 4), round(min(0.99, ci_upper), 4)]

    def _classify_tier(self, p):
        """将录取概率映射为冲/稳/保/垫标签"""
        if p >= self.bottom_threshold:
            return "bottom"
        if p >= self.safe_threshold:
            return "safe"
        if p >= self.stable_threshold:
            return "stable"
        return "rush"

    def map_to_recommendation_tier(self, p):
        """公开接口：概率映射为推荐等级"""
        return self._classify_tier(p)

    # ================================================================
    # 省份/科类/选科组合校正
    # ================================================================

    def province_subject_correction(self, p, province, subject_type,
                                     school_level=None, n_years_data=3):
        """
        省份、科类、选科组合校正

        不同省份、不同科类、不同选科组合的录取概率模型
        不能直接混用未经校正的结果。

        当前为简化实现：根据院校层次和样本量做粗略修正。
        真实上线需使用分层回测校正系数。

        Returns:
            float: 校正后的概率
        """
        p_corrected = p

        if n_years_data <= 2:
            p_corrected -= 0.05

        if school_level and school_level == "985":
            p_corrected += 0.03

        if p < 0.20 and p_corrected < 0.20:
            p_corrected = p_corrected - 0.02

        p_corrected = max(0.01, min(0.99, p_corrected))
        return p_corrected

    # ================================================================
    # 小样本专业处理
    # ================================================================

    def handle_small_sample_major(self, n_years, ci_lower, ci_upper,
                                   center=None):
        """
        小样本专业处理：扩大不确定性区间

        Returns:
            tuple: (ci_lower, ci_upper, uncertainty_level, review_required)
        """
        if center is None:
            center = (ci_lower + ci_upper) / 2

        if n_years <= 1:
            expansion = 1 + (3 - max(n_years, 0.5)) * 2.0
            uncertainty = "high"
            review = True
        elif n_years <= 2:
            expansion = 1 + (3 - n_years) * 1.5
            uncertainty = "high"
            review = True
        elif n_years <= 3:
            expansion = 1.0
            uncertainty = "medium"
            review = False
        else:
            expansion = 1.0
            uncertainty = "low"
            review = False

        half = (ci_upper - ci_lower) / 2 * expansion
        ci_lower_new = max(0.0, center - half)
        ci_upper_new = min(0.99, center + half)

        return ci_lower_new, ci_upper_new, uncertainty, review

    # ================================================================
    # 预测主方法
    # ================================================================

    def predict(self, candidate_rank, historical_records, plan_series=None,
                pop_factor=1.0, profile_meta=None):
        """
        预测录取概率（主方法，兼容 pipeline.py 调用）

        Args:
            candidate_rank: 考生位次
            historical_records: DataFrame, 历史录取记录
            plan_series: Series, 历史招生计划
            pop_factor: 热度因子
            profile_meta: dict, 考生元信息(province/subject_type/school_level等)

        Returns:
            dict: 含录取概率、区间、推荐等级、三类解释、修改建议
        """
        features = self.extract_features(
            candidate_rank, historical_records, plan_series, pop_factor
        )
        n_years = features["years_with_data"]

        X = self.features_to_array(features).reshape(1, -1)

        # --- Logistic 回归 ---
        if self.model_lr is not None and X.shape[1] > 0:
            p_lr = float(self.model_lr.predict_proba(X)[0, 1])
        else:
            gap_mean = features["rank_gap_mean"]
            if gap_mean > 5000:
                p_lr = 0.95
            elif gap_mean > 2000:
                p_lr = 0.70 + (gap_mean - 2000) / 3000 * 0.25
            elif gap_mean > 0:
                p_lr = 0.35 + gap_mean / 2000 * 0.35
            elif gap_mean > -3000:
                p_lr = 0.10 + (gap_mean + 3000) / 3000 * 0.25
            else:
                p_lr = 0.02
            p_lr = max(0.01, min(0.99, p_lr))

        # --- XGBoost ---
        if self.model_xgb is not None and X.shape[1] > 0:
            p_xgb = float(self.model_xgb.predict_proba(X)[0, 1])
        else:
            p_xgb = p_lr

        # --- 贝叶斯修正 ---
        p_bayes = self._bayesian_adjust(p_lr, n_years)

        # --- 蒙特卡洛模拟 ---
        hist_ranks = historical_records["min_admission_rank"].values
        p_mc, ci_lower, ci_upper = self._monte_carlo_simulate(
            candidate_rank, hist_ranks,
            plan_change_rate=features["plan_change_rate"],
            n_years=n_years,
        )

        # --- 加权融合 ---
        p_final = 0.15 * p_lr + 0.35 * p_xgb + 0.20 * p_bayes + 0.30 * p_mc
        p_final = max(0.005, min(0.99, p_final))
        ci_lower_final, ci_upper_final = ci_lower, ci_upper

        # --- 小样本处理 ---
        ci_lower_final, ci_upper_final, uncertainty, review_required = \
            self.handle_small_sample_major(n_years, ci_lower_final, ci_upper_final)

        # --- 省份/科类校正 ---
        if profile_meta:
            p_final = self.province_subject_correction(
                p_final,
                province=profile_meta.get("province", ""),
                subject_type=profile_meta.get("subject_type", ""),
                school_level=profile_meta.get("school_level", None),
                n_years_data=n_years,
            )

        # --- 等级映射 ---
        tier = self._classify_tier(p_final)

        # --- 额外复核触发条件 ---
        if not review_required:
            interval_width = ci_upper_final - ci_lower_final
            if interval_width > 0.5:
                review_required = True
            if features.get("rank_volatility", 0) > 0.3:
                review_required = True
            if 0.15 < p_final < 0.25:
                review_required = True

        # --- Top 特征 ---
        top_features = self.extract_top_features(features)

        # --- 概率区间 ---
        prob_interval = self.estimate_probability_interval(
            p_final, ci_lower_final, ci_upper_final
        )

        return {
            "admit_probability": round(p_final, 4),
            "probability_interval": prob_interval,
            "recommendation_tier": tier,
            "top_features": top_features,
            "uncertainty_level": uncertainty,
            "review_required": review_required,
            "n_years_data": n_years,
            "component_probabilities": {
                "lr": round(p_lr, 4),
                "xgb": round(p_xgb, 4),
                "bayes": round(p_bayes, 4),
                "mc": round(p_mc, 4),
            },
        }

    def predict_probability(self, candidate_rank, historical_records,
                             plan_series=None, pop_factor=1.0, profile_meta=None,
                             school_name="", major_name=""):
        """
        预测录取概率并附带三类解释（完整输出接口）

        Returns:
            dict: 含 admit_probability + 三类解释 + modification_suggestion
        """
        pred = self.predict(
            candidate_rank, historical_records, plan_series,
            pop_factor, profile_meta
        )

        pred["explanation"] = self.generate_parent_explanation(
            pred, school_name, major_name
        )
        pred["consultant_explanation"] = self.generate_consultant_explanation(pred)
        pred["backend_explanation"] = self.generate_backend_explanation(pred)

        pred["modification_suggestion"] = self._generate_modification_suggestion(
            pred, profile_meta
        )

        return pred

    # ================================================================
    # Top 特征提取
    # ================================================================

    def extract_top_features(self, features_dict, top_k=5):
        """输出影响录取概率的 Top 特征（按绝对值排序）"""
        features_copy = features_dict.copy()
        features_copy.pop("years_with_data", None)
        features_copy.pop("subject_requirement_match", None)
        sorted_features = sorted(
            features_copy.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        result = []
        for name, val in sorted_features[:top_k]:
            direction = "+" if val > 0 else "-"
            result.append({
                "feature": name,
                "value": round(val, 4),
                "direction": direction,
            })
        return result

    # ================================================================
    # 推荐等级映射
    # ================================================================

    def map_to_recommendation_tier(self, p):
        """
        将录取概率映射为推荐等级

        rush   (冲): p < self.rush_threshold        (默认 < 0.20)
        stable (稳): rush_threshold <= p < stable_threshold (默认 0.20-0.45)
        safe   (保): stable_threshold <= p < bottom_threshold (默认 0.45-0.88)
        bottom (垫): p >= bottom_threshold           (默认 >= 0.88)
        """
        return self._classify_tier(p)

    # ================================================================
    # 三类业务解释
    # ================================================================

    def generate_parent_explanation(self, prediction, school_name="", major_name=""):
        """家长端解释：通俗、克制、不承诺录取"""
        p = prediction["admit_probability"]
        lo, hi = prediction["probability_interval"]
        tier = prediction["recommendation_tier"]
        n_years = prediction.get("n_years_data", 0)
        uncertainty = prediction.get("uncertainty_level", "medium")

        tier_cn = {
            "rush": "冲（录取有一定难度，但值得尝试）",
            "stable": "稳（有较大希望被录取）",
            "safe": "保（基本能够被录取）",
            "bottom": "垫（几乎确保录取，为兜底志愿）",
        }

        parts = []
        if school_name and major_name:
            parts.append(
                f"报考{school_name}的{major_name}专业，预估录取概率约{int(p * 100)}%"
                f"（区间：{int(lo * 100)}% - {int(hi * 100)}%），"
                f"属于{tier_cn.get(tier, tier)}。"
            )
        else:
            parts.append(
                f"预估录取概率约{int(p * 100)}%"
                f"（区间：{int(lo * 100)}% - {int(hi * 100)}%），"
                f"推荐等级：{tier}。"
            )

        if tier == "rush":
            parts.append("该志愿录取难度较大，建议作为冲刺志愿且同时填报足够保底志愿。")
        elif tier == "stable":
            parts.append("该志愿在可争取范围内，建议同时搭配保底志愿以确保安全。")
        elif tier == "safe":
            parts.append("该志愿有较高考取把握，可作为主要选择之一。")
        elif tier == "bottom":
            parts.append("该志愿作为兜底选择，能有效降低滑档风险。")

        if n_years <= 2:
            parts.append(f"注意：该专业仅有{n_years}年录取历史数据，预测不确定性较大。")
        if uncertainty == "high":
            parts.append("建议咨询专业人士进行人工复核。")

        return "".join(parts)

    def generate_consultant_explanation(self, prediction):
        """咨询师端解释：模型依据、风险条件、可调整项"""
        p = prediction["admit_probability"]
        lo, hi = prediction["probability_interval"]
        tier = prediction["recommendation_tier"]
        uncertainty = prediction.get("uncertainty_level", "medium")
        n_years = prediction.get("n_years_data", 0)
        review = prediction.get("review_required", False)
        top = prediction.get("top_features", [])
        comps = prediction.get("component_probabilities", {})

        lines = [
            f"录取概率: {p:.4f} ({lo:.4f}-{hi:.4f})",
            f"推荐等级: {tier}",
            f"不确定性: {uncertainty}",
            f"历史数据年数: {n_years}",
            f"分项概率: LR={comps.get('lr', 0):.4f} XGB={comps.get('xgb', 0):.4f} "
            f"Bayes={comps.get('bayes', 0):.4f} MC={comps.get('mc', 0):.4f}",
        ]

        if top:
            feature_str = ", ".join(
                f"{f['feature']}({f['value']:+.2f})" for f in top[:3]
            )
            lines.append(f"Top影响因素: {feature_str}")

        if review:
            lines.append("已触发人工复核标志，原因: ")
            if n_years <= 2:
                lines.append("  - 历史数据不足(<=2年)")
            if (hi - lo) > 0.5:
                lines.append("  - 概率区间过宽(>50%)")
            if tier == "rush" and p < 0.15:
                lines.append("  - 录取概率极低(<15%)，可考虑移除")

        lines.append(f"阈值配置: rush<{self.rush_threshold} "
                      f"stable<{self.stable_threshold} "
                      f"safe<{self.safe_threshold} "
                      f"bottom>={self.bottom_threshold}")
        lines.append("注意：不同省份/科类/选科组合不可混用未经校正的模型。")

        return "; ".join(lines)

    def generate_backend_explanation(self, prediction):
        """系统后台解释：结构化字段，便于追溯和审核"""
        return {
            "method": "multi_model_fusion(LR+XGBoost+Bayes+MC)",
            "fusion_weights": {"lr": 0.15, "xgb": 0.35, "bayes": 0.20, "mc": 0.30},
            "mc_iterations": self.mc_iterations,
            "thresholds": {
                "rush": self.rush_threshold,
                "stable": self.stable_threshold,
                "safe": self.safe_threshold,
                "bottom": self.bottom_threshold,
            },
            "is_trained": self.is_trained,
            "training_meta": self.training_meta,
            "prediction": prediction,
        }

    def _generate_modification_suggestion(self, prediction, profile_meta=None):
        """生成修改建议"""
        p = prediction["admit_probability"]
        tier = prediction["recommendation_tier"]
        uncertainty = prediction.get("uncertainty_level", "medium")
        n_years = prediction.get("n_years_data", 0)
        suggestions = []

        if tier == "rush" and p < 0.10:
            suggestions.append("录取概率极低(不足10%)，建议替换为录取可能性更高的志愿。")
        elif tier == "rush":
            suggestions.append("该志愿为冲刺志愿，建议搭配至少2个保底志愿。")

        if uncertainty == "high":
            suggestions.append(f"仅有{n_years}年数据，建议寻找更多历史参考数据或选择数据更充分的高校。")

        if prediction.get("probability_interval", [0, 1])[1] - prediction.get("probability_interval", [0, 1])[0] > 0.6:
            suggestions.append("概率区间过宽，录取结果高度不确定，建议咨询师复核。")

        if not suggestions:
            suggestions.append("无明显需要修改的问题，建议关注该专业近年录取位次波动趋势。")

        return "；".join(suggestions)

    # ================================================================
    # 废弃方法（向后兼容）
    # ================================================================

    def generate_explanation(self, prediction, school_name="", major_name=""):
        """
        [已废弃] 请使用 generate_parent_explanation()

        保留此方法以向后兼容 pipeline.py 旧版本调用。
        """
        return self.generate_parent_explanation(prediction, school_name, major_name)


if __name__ == "__main__":
    from data_generator import generate_major_admission, generate_admission_plan

    print("=" * 50)
    print("  核心模型二：录取概率预测模型 演示 (Project 5)")
    print("=" * 50)

    major_df = generate_major_admission(province="河北省", years=(2020, 2021, 2022, 2023))
    plan_df = generate_admission_plan(province="河北省", year=2024)

    predictor = AdmissionProbabilityPredictor()

    print("\n[训练] 使用模拟数据训练模型...")
    meta = predictor.fit_all(use_simulated=True)
    print(f"训练元信息: {meta}")

    school_code = "10003"
    major_code = "080901"
    hist = major_df[
        (major_df["school_code"] == school_code) &
        (major_df["major_code"] == major_code) &
        (major_df["province"] == "河北省")
    ].sort_values("year")

    print(f"\n目标: 院校{school_code}, 专业{major_code}")
    print(f"历史数据: {len(hist)} 年")
    print(f"历史录取位次: {list(hist['min_admission_rank'])}")

    result = predictor.predict(
        candidate_rank=8500,
        historical_records=hist,
        profile_meta={"province": "河北省", "subject_type": "物理类"},
    )

    print(f"\n预测结果:")
    print(f"  录取概率: {result['admit_probability']}")
    print(f"  概率区间: {result['probability_interval']}")
    print(f"  推荐等级: {result['recommendation_tier']}")
    print(f"  不确定性: {result['uncertainty_level']}")
    print(f"  需人工复核: {result['review_required']}")
    print(f"  分项概率: {result['component_probabilities']}")
    print(f"  Top特征: {result['top_features']}")

    parent_exp = predictor.generate_parent_explanation(
        result, "A理工大学", "计算机科学与技术"
    )
    print(f"\n[家长端解释]\n{parent_exp}")

    consultant_exp = predictor.generate_consultant_explanation(result)
    print(f"\n[咨询师端解释]\n{consultant_exp}")

    backend_exp = predictor.generate_backend_explanation(result)
    print(f"\n[后台端解释]\n{backend_exp}")

    suggestion = predictor._generate_modification_suggestion(result)
    print(f"\n[修改建议]\n{suggestion}")
