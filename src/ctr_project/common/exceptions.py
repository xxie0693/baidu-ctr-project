"""自定义异常层级，所有业务异常继承自 CTRError。"""


class CTRError(Exception):
    """CTR 项目基础业务异常。"""


class DataLoadError(CTRError):
    """数据加载失败时抛出。"""


class DataValidationError(CTRError):
    """数据质量校验失败时抛出。"""


class DataCleanError(CTRError):
    """数据清洗过程异常时抛出。"""


class ParquetConvertError(CTRError):
    """Parquet 格式转换失败时抛出。"""