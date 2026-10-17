"""多维度探索性数据分析可视化。

产出 8 张核心图表，保存至 docs/figures/。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from ctr_project.data.sqlite_queries import SQLiteClient

# 设置全局绘图风格
sns.set_theme(style="whitegrid")
# 解决中文字体显示问题 (Mac 环境)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def plot_hourly_trend(client: SQLiteClient) -> None:
    """图 1：全天各时段曝光量与 CTR 趋势。"""
    df = client.query("SELECT hour_of_day, SUM(impressions) as imp, SUM(clicks)/SUM(impressions) as ctr FROM dim_time GROUP BY hour_of_day")
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df['hour_of_day'], df['imp'], color='lightgray', label='曝光量')
    ax1.set_ylabel('曝光量', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    ax2 = ax1.twinx()
    ax2.plot(df['hour_of_day'], df['ctr'], color='red', marker='o', label='CTR')
    ax2.set_ylabel('点击率 (CTR)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("图 1：全天各时段曝光量与 CTR 趋势")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_hourly_trend.png", dpi=150)
    plt.close()


def plot_banner_ctr(client: SQLiteClient) -> None:
    """图 2：各广告位 CTR 对比（剔除极低曝光位）。"""
    df = client.query("SELECT banner_pos, ctr FROM dim_ad_position WHERE impressions > 10000 ORDER BY ctr DESC")
    
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='banner_pos', y='ctr', palette='viridis')
    plt.title("图 2：高曝光广告位 CTR 对比")
    plt.xlabel("广告位编码")
    plt.ylabel("点击率 (CTR)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_banner_ctr.png", dpi=150)
    plt.close()


def plot_media_top(client: SQLiteClient) -> None:
    """图 3：曝光量 Top 10 的媒体站点 CTR。"""
    df = client.query("SELECT media_id, impressions, ctr FROM dim_media WHERE media_type='site' ORDER BY impressions DESC LIMIT 10")
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, y='media_id', x='ctr', palette='magma')
    plt.title("图 3：曝光 Top 10 站点 CTR")
    plt.xlabel("点击率 (CTR)")
    plt.ylabel("站点 ID")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_media_top_ctr.png", dpi=150)
    plt.close()


def plot_device_distribution(client: SQLiteClient) -> None:
    """图 4：主要设备类型曝光量分布。"""
    df = client.query("SELECT device_type, SUM(impressions) as imp FROM dim_device GROUP BY device_type ORDER BY imp DESC LIMIT 10")
    
    plt.figure(figsize=(8, 5))
    df.set_index('device_type')['imp'].plot(kind='pie', autopct='%1.1f%%', cmap='Set2')
    plt.title("图 4：Top 10 设备类型曝光占比")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_device_pie.png", dpi=150)
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
    logger.info("已生成图 4：设备分布")
    
    logger.info(f"全部图表已保存至 {FIG_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_all_plots()