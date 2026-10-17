"""特征工程：基础聚合特征构建与异常值处理。"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from ctr_project.data.sqlite_queries import SQLiteClient

logger = logging.getLogger(__name__)


def handle_iqr_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """对聚合出的连续指标使用 IQR 法则进行异常值截断。
    
    Args:
        df: 包含目标列的 DataFrame。
        col: 列名。
        
    Returns:
        处理后的 DataFrame。
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # 记录处理前的极值
    before_max = df[col].max()
    
    # 截断
    df[col] = df[col].clip(lower, upper)
    
    logger.info(f"[{col}] IQR 异常检测: 阈值区间 [{lower:.4f}, {upper:.4f}]，修正了 {before_max - df[col].max():.4f} 的极值。")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    client = SQLiteClient()
    
    logger.info("从 SQLite 读取维度特征...")
    
    # 1. 读取广告位特征
    df_ad = client.query("SELECT * FROM dim_ad_position")
    
    # 2. 应用 IQR 异常值处理（重点针对 CTR 等连续指标）
    df_ad = handle_iqr_outliers(df_ad, "ctr")
    
    # 3. 读取媒体特征
    df_media = client.query("SELECT * FROM dim_media")
    # 过滤掉曝光量极低的长尾媒体，并处理 CTR
    df_media = df_media[df_media["impressions"] > 100].copy() # 限制最小曝光量
    df_media = handle_iqr_outliers(df_media, "ctr")
    
    print("\n广告位聚合特征（IQR 处理后）：")
    print(df_ad)
    
    print("\n媒体聚合特征（IQR 处理后 Top 5）：")
    print(df_media.sort_values("ctr", ascending=False).head())
    