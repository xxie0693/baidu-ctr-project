"""模型可解释性分析与业务模拟设计。"""

from __future__ import annotations

import logging
from pathlib import Path

import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd

from ctr_project.models.data_loader import load_and_preprocess_data, CAT_FEATURES, NUM_FEATURES

# 解决 Mac/Windows 图表中文变成方块的问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("data/processed/models")


def run_shap_analysis():
    """使用 SHAP 分析 LightGBM 特征重要性并绘图。"""
    logger.info("加载模型与抽样数据用于 SHAP 分析...")
    
    # 读取之前训练好的模型
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
    
    # 抽样 10000 条数据用于计算 SHAP（避免计算量爆炸）
    X_train, _, _, _, X_test, _ = load_and_preprocess_data(sample_ratio=0.01)
    X_sample = X_test.sample(10000, random_state=42)
    
    logger.info("计算 SHAP 值 (耗时约 1 分钟)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 图 1：SHAP 摘要图（蜂群图）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("图 9：特征对 CTR 预估的 SHAP 影响值", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_shap_summary.png", dpi=150)
    plt.close()
    
    logger.info(f"SHAP 图表已保存至 {FIG_DIR}/09_shap_summary.png")
    
    # 找出最重要的特征名
    feature_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)
    
    logger.info("Top 5 最重要特征:")
    for idx, row in feature_importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return feature_importance


def simulate_ab_test():
    """模拟 A/B 测试收益计算。"""
    # 假设基线 CTR 为 16.98%（来自第二阶段全局统计）
    # 模型给 Top 10% 的高潜人群打分，假设这部分人原本的 CTR 是 30%（因为模型挑出了容易点击的）
    # 我们通过提升这部分人的曝光量来提升大盘 CTR
    
    baseline_ctr = 0.1698
    high_potential_ctr = 0.3200  # 假设模型识别出的高潜人群天然点击率更高
    traffic_ratio = 0.10         # 只占全站 10% 流量
    
    # 预估大盘 CTR 变化
    new_ctr = (1 - traffic_ratio) * baseline_ctr + traffic_ratio * high_potential_ctr
    
    logger.info("=== A/B 测试业务模拟方案 ===")
    logger.info(f"实验组策略：使用 LightGBM 模型预测 CTR，对 Top 10% 高潜用户提权。")
    logger.info(f"基线 CTR: {baseline_ctr:.2%}")
    logger.info(f"高潜用户 CTR: {high_potential_ctr:.2%}")
    logger.info(f"预期大盘 CTR 提升至: {new_ctr:.2%} (相对提升 {((new_ctr - baseline_ctr)/baseline_ctr):.2%})")
    
    if new_ctr > baseline_ctr:
        logger.info("✅ 收益评估：该策略具有正向业务价值，建议开启 A/B 实验。")
    else:
        logger.info("❌ 收益评估：未带来显著提升，暂不推荐上线。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    run_shap_analysis()
    print("\n")
    simulate_ab_test()