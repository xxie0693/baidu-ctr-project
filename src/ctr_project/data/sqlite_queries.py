"""Python 与 SQLite 交互及基础分析查询封装。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd
from tabulate import tabulate

DB_PATH = Path("data/processed/ctr.db")

class SQLiteClient:
    """轻量级 SQLite 查询客户端。"""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}，请先运行 sqlite_loader.py")

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """执行查询并返回 DataFrame。

        Args:
            sql: SQL 语句。
            params: 参数化查询参数。

        Returns:
            查询结果 DataFrame。
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def print_query(self, sql: str, title: str = "") -> None:
        """执行查询并以美观的表格打印。"""
        df = self.query(sql)
        print(f"\n=== {title} ===")
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    client = SQLiteClient()
    
    # 示例 1：统计总曝光量与点击率
    client.print_query("""
        SELECT 
            COUNT(*) AS total_impressions,
            SUM(click) AS total_clicks,
            ROUND(CAST(SUM(click) AS FLOAT) / COUNT(*), 4) AS overall_ctr
        FROM fact_impressions;
    """, title="全局曝光与点击率 (CTR)")
    
    # 示例 2：各广告位点击率对比 (关联维度表)
    client.print_query("""
        SELECT 
            d.banner_pos,
            d.impressions,
            d.clicks,
            ROUND(d.ctr * 100, 2) || '%' AS ctr_percent
        FROM dim_ad_position d
        ORDER BY d.impressions DESC
        LIMIT 5;
    """, title="Top 5 广告位表现")

    # 示例 3：时段投放趋势（按小时）
    client.print_query("""
        SELECT 
            hour_of_day,
            SUM(impressions) AS total_imp,
            ROUND(CAST(SUM(clicks) AS FLOAT) / SUM(impressions), 4) AS hour_ctr
        FROM dim_time
        GROUP BY hour_of_day
        ORDER BY hour_of_day;
    """, title="全天时段 CTR 趋势")