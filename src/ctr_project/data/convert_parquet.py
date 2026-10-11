"""CSV 到 Parquet 的格式转换与存储优化模块。

将清洗后的 CSV 数据分块转换为 Parquet 格式，通过列式存储、
zstd 压缩与合理分区提升后续读写效率。
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.dataframe as dd

from ctr_project.common.exceptions import ParquetConvertError

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSION: str = "zstd"
DEFAULT_ENGINE: str = "pyarrow"


def csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    blocksize: str = "100MB",
    compression: str = DEFAULT_COMPRESSION,
) -> None:
    """将 CSV 文件转换为 Parquet 格式。

    Args:
        csv_path: 源 CSV 文件路径。
        parquet_path: 目标 Parquet 目录路径。
        blocksize: 分块大小，默认 100MB。
        compression: 压缩算法，默认 zstd。

    Raises:
        FileNotFoundError: 源文件不存在时抛出。
        ParquetConvertError: 转换过程失败时抛出。
    """
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")

    dst = Path(parquet_path)
    dst.mkdir(parents=True, exist_ok=True)

    try:
        ddf = dd.read_csv(
            str(src),
            blocksize=blocksize,
            assume_missing=True,
        )
        ddf.to_parquet(
            str(dst),
            engine=DEFAULT_ENGINE,
            compression=compression,
            write_index=False,
        )
    except Exception as exc:  # noqa: BLE001 - 包装为业务异常
        logger.error("Parquet 转换失败: %s", exc)
        raise ParquetConvertError(f"转换 {src} 失败") from exc

    logger.info(
        "转换完成: %s -> %s (压缩=%s, 分区=%d)",
        src,
        dst,
        compression,
        ddf.npartitions,
    )


def dask_to_parquet(
    ddf: dd.DataFrame,
    parquet_path: str | Path,
    compression: str = DEFAULT_COMPRESSION,
    partition_size: str = "100MB",
) -> None:
    """将 Dask DataFrame 直接写出为 Parquet。

    适用于已完成清洗与类型优化、尚未物化落盘的数据流。

    Args:
        ddf: Dask DataFrame。
        parquet_path: 目标 Parquet 目录。
        compression: 压缩算法。
        partition_size: 目标分区大小。

    Raises:
        ParquetConvertError: 写出失败时抛出。
    """
    dst = Path(parquet_path)
    dst.mkdir(parents=True, exist_ok=True)

    try:
        ddf.to_parquet(
            str(dst),
            engine=DEFAULT_ENGINE,
            compression=compression,
            write_index=False,
            partition_on=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Dask 写出 Parquet 失败: %s", exc)
        raise ParquetConvertError(f"写出 {dst} 失败") from exc

    logger.info("Dask 写出完成: %s (压缩=%s)", dst, compression)


def read_parquet(
    parquet_path: str | Path,
    columns: list[str] | None = None,
) -> dd.DataFrame:
    """读取 Parquet 文件为 Dask DataFrame。

    Args:
        parquet_path: Parquet 目录路径。
        columns: 需要读取的列，None 表示全部。

    Returns:
        Dask DataFrame 对象。

    Raises:
        FileNotFoundError: 路径不存在时抛出。
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet 路径不存在: {path}")

    ddf = dd.read_parquet(
        str(path),
        columns=columns,
        engine=DEFAULT_ENGINE,
    )
    logger.info(
        "读取 Parquet: %s (分区=%d, 列=%d)",
        path,
        ddf.npartitions,
        len(ddf.columns),
    )
    return ddf


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # 本模块作为工具库使用，主要导出以下函数：
    #   - read_parquet(path, columns=None)      读取 Parquet 为 Dask DataFrame
    #   - dask_to_parquet(ddf, path, ...)       写出 Dask DataFrame 为 Parquet
    #   - csv_to_parquet(src, dst, ...)         一次性 CSV -> Parquet 转换
    #
    # 实际流程中，清洗后的数据已由 clean_data.py 输出到
    # data/interim/cleaned/，无需再次转换。此处仅打印提示，避免误操作。
    logger.info(
        "convert_parquet 作为工具库使用。"
        "如需转换 Parquet，请在其他模块中导入相应函数。"
    )