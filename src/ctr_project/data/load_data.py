"""千万级广告曝光-点击数据的分块加载与类型优化模块。

使用 Dask DataFrame 直接加载原始 CSV，避免本地 Pandas 预加载导致
内存溢出。加载后立即执行列裁剪与数据类型压缩，降低内存占用。

Example:
    >>> from ctr_project.data.load_data import load_raw_data
    >>> ddf = load_raw_data("data/raw/train.csv", blocksize="100MB")
    >>> print(ddf.npartitions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import dask.dataframe as dd
import pandas as pd

from ctr_project.common.constants import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)
from ctr_project.common.exceptions import DataLoadError

logger = logging.getLogger(__name__)

DEFAULT_BLOCKSIZE: Final[str] = "100MB"
DEFAULT_DTYPE: Final[dict[str, str]] = {
    "id": "string",
    "click": "int8",
    "hour": "string",
    "banner_pos": "uint8",
    "device_type": "uint16",
    "device_conn_type": "float32", 
    "site_domain": "category",
    "app_domain": "category",
}


def _build_dtype_map() -> dict[str, str]:
    """构建 Dask 读取时的列类型映射。

    Returns:
        列名到 dtype 字符串的映射字典。
    """
    dtype_map: dict[str, str] = dict(DEFAULT_DTYPE)
    for col in CATEGORICAL_COLUMNS:
        dtype_map[col] = "category"
    return dtype_map


def load_raw_data(
    file_path: str | Path,
    blocksize: str = DEFAULT_BLOCKSIZE,
    usecols: list[str] | None = None,
) -> dd.DataFrame:
    """分块加载原始 CSV 广告数据。

    Args:
        file_path: CSV 文件路径。
        blocksize: 每个分区目标大小，默认 100MB。
        usecols: 需要读取的列，None 表示读取全部必需列。

    Returns:
        Dask DataFrame 对象，已完成基础类型转换。

    Raises:
        FileNotFoundError: 文件路径不存在时抛出。
        DataLoadError: Dask 加载失败时抛出。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    selected_cols = usecols if usecols is not None else list(REQUIRED_COLUMNS)
    dtype_map = _build_dtype_map()
    active_dtype = {
        col: dtype_map[col]
        for col in selected_cols
        if col in dtype_map
    }

    try:
        ddf = dd.read_csv(
            str(path),
            blocksize=blocksize,
            dtype=active_dtype,
            assume_missing=True,
            include_path_column=False,
        )
    except Exception as exc:  # noqa: BLE001 - 需包装为业务异常
        logger.error("Dask 加载失败: %s", exc)
        raise DataLoadError(f"加载 {path} 失败") from exc

    logger.info(
        "加载完成: %s, 分区数=%d, 列数=%d",
        path.name,
        ddf.npartitions,
        len(ddf.columns),
    )
    return ddf


def optimize_dtypes(ddf: dd.DataFrame) -> dd.DataFrame:
    """对已加载的 Dask DataFrame 执行进一步类型压缩。

    将 float64 转为 float32，将 object 类型的高基数列转为 category。

    Args:
        ddf: 待优化的 Dask DataFrame。

    Returns:
        类型优化后的 Dask DataFrame。
    """
    float_cols = [
        col for col in ddf.columns
        if ddf[col].dtype == "float64"
    ]
    if float_cols:
        ddf = ddf.astype({col: "float32" for col in float_cols})
        logger.info("压缩 float64 -> float32: %s", float_cols)

    object_cols = [
        col for col in ddf.columns
        if ddf[col].dtype == "object" and col not in CATEGORICAL_COLUMNS
    ]
    if object_cols:
        ddf = ddf.astype({col: "category" for col in object_cols})
        logger.info("object -> category: %s", object_cols)

    return ddf


def get_basic_info(ddf: dd.DataFrame) -> pd.DataFrame:
    """获取数据集的列类型概览。

    Args:
        ddf: Dask DataFrame。

    Returns:
        包含列名、dtype、非空数的 Pandas DataFrame。
    """
    return pd.DataFrame(
        {
            "column": ddf.columns,
            "dtype": [str(ddf[col].dtype) for col in ddf.columns],
        }
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ddf_ = load_raw_data("data/raw/train.csv")
    ddf_ = optimize_dtypes(ddf_)
    print(get_basic_info(ddf_).to_string(index=False))