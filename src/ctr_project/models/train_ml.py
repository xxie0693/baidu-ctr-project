"""传统机器学习模型训练与评估：LightGBM 和 Logistic Regression。"""

from __future__ import annotations

import logging
import time
import joblib
from pathlib import Path

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from ctr_project.models.data_loader import load_and_preprocess_data

logger = logging.getLogger(__name__)
MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_lgbm(X_train, y_train, X_val, y_val):
    """训练 LightGBM 模型。"""
    logger.info("开始训练 LightGBM 模型...")
    
    # 构造 LightGBM 数据集
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # 设置参数（针对 CTR 预估常用参数）
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    # 训练
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)]
    )
    
    # 保存模型
    model_path = MODEL_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    logger.info(f"LightGBM 模型已保存至 {model_path}")
    return model


def evaluate_model(model, X, y, model_name: str):
    """评估模型指标 (AUC, LogLoss)。"""
    y_pred_prob = model.predict(X)
    auc = roc_auc_score(y, y_pred_prob)
    loss = log_loss(y, y_pred_prob)
    logger.info(f"[{model_name}] 评估结果 -> AUC: {auc:.4f}, LogLoss: {loss:.4f}")
    return auc, loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 1. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(sample_ratio=0.1)
    
    # 2. 训练 LightGBM
    start = time.time()
    lgb_model = train_lgbm(X_train, y_train, X_val, y_val)
    logger.info(f"LightGBM 训练耗时: {time.time() - start:.2f} 秒")
    
    # 3. 在测试集上评估
    evaluate_model(lgb_model, X_test, y_test, "LightGBM (Test)")
    
    # 4. 训练逻辑回归（作为基线参考）
    logger.info("开始训练逻辑回归模型 (基线)...")
    # 逻辑回归对特征分布敏感，需先做标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(solver='lbfgs', max_iter=200)
    lr_model.fit(X_train_scaled, y_train)
    
    # 评估 LR
    y_pred_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    logger.info(f"[Logistic Regression] AUC: {roc_auc_score(y_test, y_pred_lr):.4f}")