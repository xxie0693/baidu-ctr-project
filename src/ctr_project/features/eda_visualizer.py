"""多维度探索性数据分析可视化
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ctr_project.data.sqlite_queries import SQLiteClient

# 设置全局绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 适配 Mac/Windows 中文
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def plot_hourly_trend(client: SQLiteClient) -> None:
    """图 1：全天各时段曝光量与 CTR 趋势（双轴图）。"""
    df = client.query("""
        SELECT hour_of_day, 
               SUM(impressions) as imp, 
               SUM(clicks) * 1.0 / SUM(impressions) as ctr 
        FROM dim_time 
        GROUP BY hour_of_day
    """)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df['hour_of_day'], df['imp'], color='#A0CBE8', label='曝光量')
    ax1.set_ylabel('曝光量', color='#4C72B0', fontsize=12)
    ax1.set_xlabel('小时 (0-23)', fontsize=12)
    ax1.set_xticks(range(0, 24))
    
    ax2 = ax1.twinx()
    ax2.plot(df['hour_of_day'], df['ctr'] * 100, color='#C44E52', marker='o', linewidth=2, label='CTR')
    ax2.set_ylabel('点击率 CTR (%)', color='#C44E52', fontsize=12)
    
    plt.title("图 1：全天各时段曝光量与 CTR 趋势", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_hourly_trend.png", dpi=150)
    plt.close()


def plot_banner_ctr(client: SQLiteClient) -> None:
    """图 2：各广告位 CTR 对比。"""
    df = client.query("""
        SELECT banner_pos, ctr 
        FROM dim_ad_position 
        WHERE impressions > 10000 
        ORDER BY ctr DESC
    """)
    
    plt.figure(figsize=(8, 4))
    # 消除警告：加入 hue 和 legend=False
    sns.barplot(data=df, x='banner_pos', y=df['ctr']*100, palette='viridis', hue='banner_pos', legend=False)
    plt.title("图 2：高曝光广告位 CTR 对比", fontsize=14)
    plt.xlabel("广告位编码", fontsize=12)
    plt.ylabel("点击率 (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_banner_ctr.png", dpi=150)
    plt.close()


def plot_media_top(client: SQLiteClient) -> None:
    """图 3：曝光量 Top 10 的媒体站点 CTR。"""
    df = client.query("""
        SELECT media_id, impressions, ctr 
        FROM dim_media 
        WHERE media_type='site' 
        ORDER BY impressions DESC 
        LIMIT 10
    """)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, y='media_id', x=df['ctr']*100, palette='magma', hue='media_id', legend=False)
    plt.title("图 3：曝光 Top 10 站点 CTR", fontsize=14)
    plt.xlabel("点击率 (%)", fontsize=12)
    plt.ylabel("站点 ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_media_top_ctr.png", dpi=150)
    plt.close()


def plot_device_distribution(client: SQLiteClient) -> None:
    """图 4：主要设备类型曝光量分布（使用横向柱状图 + 长尾合并，彻底解决重叠）。"""
    df = client.query("""
        SELECT device_type, SUM(impressions) as imp 
        FROM dim_device 
        GROUP BY device_type 
        ORDER BY imp DESC
    """)
    # 强制转为字符串，防止绘图时被误认为连续数值
    df['device_type'] = df['device_type'].astype(str)
    
    # 计算百分比
    total_imp = df['imp'].sum()
    df['imp_pct'] = df['imp'] / total_imp * 100
    
    # 核心优化：将占比小于 0.5% 的项合并为“其他”
    threshold = 0.5
    main_df = df[df['imp_pct'] >= threshold].copy()
    other_pct = df[df['imp_pct'] < threshold]['imp_pct'].sum()
    
    if other_pct > 0:
        other_row = pd.DataFrame({'device_type': ['其他'], 'imp_pct': [other_pct]})
        main_df = pd.concat([main_df, other_row], ignore_index=True)
    
    # 按占比升序排列，这样最大的会在图表最上方
    main_df = main_df.sort_values('imp_pct', ascending=True)
    
    plt.figure(figsize=(10, 5))
    # 画横向柱状图
    bars = sns.barplot(
        data=main_df, 
        y='device_type', 
        x='imp_pct', 
        palette='Set2',
        hue='device_type', 
        legend=False
    )
    
    # 在每个柱子右侧添加文本标签
    for p in bars.patches:
        width = p.get_width()
        # 如果占比极小，就不显示数字了，避免挤在一起
        if width > 0.5:
            bars.text(
                width + 1, 
                p.get_y() + p.get_height() / 2,
                f'{width:.1f}%', 
                va='center', 
                fontsize=11
            )
        elif width > 0:
             bars.text(
                width + 1, 
                p.get_y() + p.get_height() / 2,
                f'{width:.2f}%', 
                va='center', 
                fontsize=9
            )

    plt.title("图 4：Top 设备类型曝光占比分布", fontsize=15, pad=15)
    plt.xlabel("曝光占比 (%)", fontsize=12)
    plt.ylabel("设备类型编码", fontsize=12)
    
    # 根据最大占比动态扩展 x 轴，给文本留出空间
    plt.xlim(0, main_df['imp_pct'].max() * 1.2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_device_bar.png", dpi=150)
    plt.close()


def plot_weekend_vs_weekday(client: SQLiteClient) -> None:
    """图 5：工作日 vs 周末 CTR 对比。"""
    df = client.query("""
        SELECT 
            CASE WHEN day_of_week >= 5 THEN '周末' ELSE '工作日' END as day_type,
            SUM(impressions) as imp,
            SUM(clicks) * 1.0 / SUM(impressions) as ctr
        FROM dim_time
        GROUP BY day_type
    """)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='day_type', y=df['ctr']*100, palette='pastel', hue='day_type', legend=False)
    plt.title("图 5：工作日 vs 周末 CTR 对比", fontsize=14)
    plt.ylabel("点击率 (%)", fontsize=12)
    plt.xlabel("")
    for i, v in enumerate(df['ctr']*100):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
    plt.ylim(0, df['ctr'].max() * 100 * 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_weekend_compare.png", dpi=150)
    plt.close()


def plot_category_ctr(client: SQLiteClient) -> None:
    """图 6：不同媒体类型的 CTR 对比（修复版：基于 dim_media 分析）。"""
    # 由于事实表没有保留 site_category，这里通过 dim_media 聚合不同媒体类型的表现
    df = client.query("""
        SELECT media_type, 
               SUM(impressions) as imp, 
               SUM(clicks) * 1.0 / SUM(impressions) as ctr 
        FROM dim_media 
        GROUP BY media_type
    """)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='media_type', y=df['ctr']*100, palette='crest', hue='media_type', legend=False)
    plt.title("图 6：不同媒体类型 (Site vs App) CTR 对比", fontsize=14)
    plt.xlabel("媒体类型", fontsize=12)
    plt.ylabel("点击率 (%)", fontsize=12)
    
    for i, v in enumerate(df['ctr']*100):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
        
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_category_ctr.png", dpi=150)
    plt.close()


def plot_conversion_funnel(client: SQLiteClient) -> None:
    """图 7：全局转化漏斗（曝光 -> 点击）。"""
    df = client.query("SELECT COUNT(*) as imp, SUM(click) as clk FROM fact_impressions")
    imp, clk = df.iloc[0]['imp'], df.iloc[0]['clk']
    
    stages = ['曝光 (Impression)', '点击 (Click)']
    values = [imp, clk]
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=values, y=stages, palette='Blues_r', hue=stages, legend=False)
    plt.title("图 7：广告曝光-点击转化漏斗", fontsize=14)
    
    for i, v in enumerate(values):
        pct = (v / imp) * 100
        plt.text(v + imp*0.01, i, f"{v:,} ({pct:.2f}%)", va='center')
    
    plt.xlim(0, imp * 1.2)
    plt.xlabel("数量", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_conversion_funnel.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(client: SQLiteClient) -> None:
    """图 8：关键数值特征与目标变量的相关性热力图。"""
    # 抽样 10 万行计算相关性，避免 OOM
    df = client.query("""
        SELECT banner_pos, device_type, device_conn_type, click 
        FROM fact_impressions 
        LIMIT 100000
    """)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    corr = df.corr()
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
    plt.title("图 8：数值特征相关性热力图", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_correlation_heatmap.png", dpi=150)
    plt.close()


def generate_all_plots() -> None:
    """执行全部可视化。"""
    logger.info("开始生成 EDA 图表...")
    client = SQLiteClient()
    
    plot_hourly_trend(client)
    logger.info("已生成图 1：时段趋势")
    
    plot_banner_ctr(client)
    logger.info("已生成图 2：广告位对比")
    
    plot_media_top(client)
    logger.info("已生成图 3：Top 媒体对比")
    
    plot_device_distribution(client)
    logger.info("已生成图 4：设备分布（条形图）")
    
    plot_weekend_vs_weekday(client)
    logger.info("已生成图 5：工作日周末对比")
    
    plot_category_ctr(client)
    logger.info("已生成图 6：媒体类型对比（修复版）")
    
    plot_conversion_funnel(client)
    logger.info("已生成图 7：转化漏斗")
    
    plot_correlation_heatmap(client)
    logger.info("已生成图 8：特征相关性热力图")
    
    logger.info(f"全部图表已保存至 {FIG_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_all_plots()