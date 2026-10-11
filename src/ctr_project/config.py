"""项目路径与配置管理。"""

from pathlib import Path

# 项目根目录（根据本文件位置向上回溯两级）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 数据目录
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = DATA_DIR / "reports"

# 原始文件
TRAIN_CSV: Path = RAW_DIR / "train.csv"
TEST_CSV: Path = RAW_DIR / "test.csv"

# 中间产物
CLEANED_DIR: Path = INTERIM_DIR / "cleaned"
PARQUET_DIR: Path = INTERIM_DIR / "parquet"