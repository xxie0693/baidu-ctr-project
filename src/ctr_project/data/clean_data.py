"""千万级广告数据的全流程清洗模块。

覆盖去重、时间格式统一、异常值过滤与关键字段完整性校验。
所有清洗操作基于 Dask 惰性计算，需调用 compute() 触发实际执行。
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from ctr_project.common.constants import (
    ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_FORMAT,
)
from ctr_project.common.exceptions import DataCleanError

logger = logging.getLogger(__name__)

MIN_DATETIME: pd.Timestamp = pd.Timestamp("2014-01-01")
MAX_DATETIME: pd.Timestamp = pd.Timestamp("2015-12-31")


def drop_duplicates(ddf: dd.DataFrame) -> dd.DataFrame:
    """基于曝光记录唯一标识去重。

    Args:
        ddf: 原始 Dask DataFrame。

    Returns:
        去重后的 Dask DataFrame。

    Raises:
        DataCleanError: id 列缺失时抛出。
    """
    if ID_COLUMN not in ddf.columns:
        raise DataCleanError(f"缺少唯一标识列: {ID_COLUMN}")

    before = len(ddf)
    ddf_dedup = ddf.drop_duplicates(subset=[ID_COLUMN])
    logger.info("去重完成: 原始行数=%d", before)
    return ddf_dedup


def parse_time_column(ddf: dd.DataFrame) -> dd.DataFrame:
    """将 hour 字段从 YYMMDDHH 字符串解析为 datetime64。

    采用 Dask 原生的 map_partitions 实现并行解析，避免全量物化。

    Args:
        ddf: 含 hour 字符串列的 Dask DataFrame。

    Returns:
        新增 hour_dt 列的 Dask DataFrame。

    Raises:
        DataCleanError: hour 列缺失或解析异常时抛出。
    """
    if TIME_COLUMN not in ddf.columns:
        raise DataCleanError(f"缺少时间列: {TIME_COLUMN}")

    def _parse_partition(part: pd.DataFrame) -> pd.DataFrame:
        parsed = pd.to_datetime(
            part[TIME_COLUMN], format=TIME_FORMAT, errors="coerce"
        )
        part = part.assign(hour_dt=parsed)
        return part

    try:
        ddf = ddf.map_partitions(_parse_partition, meta=_meta_with_hour_dt(ddf))
    except Exception as exc:  # noqa: BLE001 - 包装为业务异常
        logger.error("时间解析失败: %s", exc)
        raise DataCleanError("hour 字段解析异常") from exc

    logger.info("时间解析完成: 新增 hour_dt 列")
    return ddf


def _meta_with_hour_dt(ddf: dd.DataFrame) -> pd.DataFrame:
    """构造 map_partitions 所需的 meta DataFrame。

    Args:
        ddf: 原 Dask DataFrame。

    Returns:
        含 hour_dt 列的空白 meta DataFrame。
    """
    meta = ddf._meta.copy()
    meta["hour_dt"] = pd.Series([], dtype="datetime64[ns]")
    return meta


def filter_invalid_time(ddf: dd.DataFrame) -> dd.DataFrame:
    """过滤时间解析失败或超出合理范围的记录。

    Args:
        ddf: 含 hour_dt 列的 Dask DataFrame。

    Returns:
        时间有效的 Dask DataFrame。
    """
    if "hour_dt" not in ddf.columns:
        raise DataCleanError("请先调用 parse_time_column 生成 hour_dt 列")

    valid_mask = (
        ddf["hour_dt"].notnull()
        & (ddf["hour_dt"] >= MIN_DATETIME)
        & (ddf["hour_dt"] <= MAX_DATETIME)
    )
    ddf_valid = ddf[valid_mask]

    # 打印时间范围，便于诊断
    min_t = ddf["hour_dt"].min().compute()
    max_t = ddf["hour_dt"].max().compute()
    logger.info(
        "时间范围: [%s, %s], 阈值: [%s, %s]",
        min_t, max_t, MIN_DATETIME, MAX_DATETIME,
    )
    return ddf_valid


def filter_invalid_target(ddf: dd.DataFrame) -> dd.DataFrame:
    """过滤目标变量非 0/1 的异常记录。

    Args:
        ddf: 原始 Dask DataFrame。

    Returns:
        目标变量取值合法的 Dask DataFrame。
    """
    if TARGET_COLUMN not in ddf.columns:
        raise DataCleanError(f"缺少目标列: {TARGET_COLUMN}")

    ddf_valid = ddf[ddf[TARGET_COLUMN].isin([0, 1])]
    logger.info("目标变量过滤完成: 保留 click in {0, 1}")
    return ddf_valid


def drop_critical_nulls(ddf: dd.DataFrame) -> dd.DataFrame:
    """剔除关键字段存在空值的记录。

    关键字段包括 id、click、hour。非关键字段的空值保留，
    交由特征工程阶段处理。

    Args:
        ddf: 原始 Dask DataFrame。

    Returns:
        关键字段非空的 Dask DataFrame。
    """
    critical_cols = [ID_COLUMN, TARGET_COLUMN, TIME_COLUMN]
    ddf_clean = ddf.dropna(subset=critical_cols)
    logger.info("关键字段空值过滤完成: %s", critical_cols)
    return ddf_clean


def clean_pipeline(ddf: dd.DataFrame) -> dd.DataFrame:
    """执行完整清洗流程，并打印真实行数变化。

    Args:
        ddf: 原始 Dask DataFrame。

    Returns:
        清洗完成的 Dask DataFrame。
    """
    before = len(ddf)
    logger.info("清洗前总行数: %d", before)

    ddf = drop_duplicates(ddf)
    ddf = parse_time_column(ddf)
    ddf = filter_invalid_time(ddf)
    ddf = filter_invalid_target(ddf)
    ddf = drop_critical_nulls(ddf)

    after = len(ddf)
    logger.info(
        "清洗后总行数: %d (移除 %d 条, %.2f%%)",
        after,
        before - after,
        100 * (before - after) / before if before else 0.0,
    )
    return ddf


def save_cleaned(ddf: dd.DataFrame, out_dir: str | Path) -> None:
    """将清洗结果保存为 Parquet 格式。

    Args:
        ddf: 清洗后的 Dask DataFrame。
        out_dir: 输出目录。
    """
    row_count = int(ddf.shape[0].compute())
    logger.info("准备写入 Parquet, 行数=%d", row_count)
    if row_count == 0:
        raise DataCleanError("清洗后数据为空，请检查上游过滤条件")

    # 规避 Dask category 跨分区编码不一致
    cat_cols = [c for c in ddf.columns if str(ddf[c].dtype) == "category"]
    if cat_cols:
        ddf = ddf.astype({col: "string" for col in cat_cols})
        logger.info("写盘前 category -> string: %s", cat_cols)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ddf.to_parquet(str(out_path), engine="pyarrow", write_index=False)
    logger.info("清洗结果已保存: %s (行数=%d)", out_path, row_count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from ctr_project.data.load_data import load_raw_data

    ddf_ = load_raw_data("data/raw/train.csv")
    ddf_ = clean_pipeline(ddf_)
    save_cleaned(ddf_, "data/interim/cleaned")