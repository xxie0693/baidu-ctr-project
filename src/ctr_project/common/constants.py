"""项目全局常量定义。"""

from typing import Final

# 数据集字段定义
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "click",
    "hour",
    "banner_pos",
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_type",
    "device_model",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)

# 类别型字段（高基数，转为 category 类型）
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_model",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
)

# 数值型字段（压缩为更小的整型）
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "banner_pos",
    "device_type",
)

# 目标变量
TARGET_COLUMN: Final[str] = "click"

# 主键
ID_COLUMN: Final[str] = "id"

# 时间字段
TIME_COLUMN: Final[str] = "hour"

# 时间格式
TIME_FORMAT: Final[str] = "%y%m%d%H"

# 数据质量阈值
MAX_NULL_RATIO: Final[float] = 0.05