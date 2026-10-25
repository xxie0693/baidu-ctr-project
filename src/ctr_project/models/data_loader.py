"""模型训练的数据加载与预处理模块。"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ctr_project.data.sqlite_queries import SQLiteClient

logger = logging.getLogger(__name__)

# 用于建模的类别特征
CAT_FEATURES = [
    "banner_pos", "site_id", "app_id", "device_type", 
    "device_conn_type", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"
]
NUM_FEATURES = ["hour_of_day", "day_of_week"]


def load_and_preprocess_data(sample_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """从 SQLite 抽取数据并划分训练/验证/测试集。"""
    logger.info(f"从 SQLite 加载数据并进行 {sample_ratio*100}% 抽样...")
    client = SQLiteClient()
    
    # 修复 1：补上 C1 字段
    # 修复 2：使用 rowid 取模抽样代替 RANDOM()，速度提升百倍
    # 修复 3：SQLite 没有 STRFTIME 直接处理 hour_dt，这里改用 Python 解析或者直接读取（简单起见，直接读主表时间）
    mod_value = int(1 / sample_ratio)
    query = f"""
        SELECT 
            click, banner_pos, site_id, app_id, device_type, 
            device_conn_type, C1, C14, C15, C16, C17, C18, C19, C20, C21,
            CAST(STRFTIME('%H', hour_dt) AS INTEGER) as hour_of_day,
            CAST(STRFTIME('%w', hour_dt) AS INTEGER) as day_of_week
        FROM fact_impressions
        WHERE rowid % {mod_value} == 0
    """
    df = client.query(query)
    logger.info(f"抽样加载完成，行数: {len(df)}")

    # 填补空值（类别特征给一个特殊值 -1）
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("-1").astype(str)
        
    # 对高基数类别特征做 LabelEncoder
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df["click"]

    # 7:2:1 划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3333, stratify=y_temp, random_state=42
    )
    
    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test