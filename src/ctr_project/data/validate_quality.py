"""数据质量校验模块。

从 schema 一致性、完整性、唯一性、有效性、正确性、业务合理性
六个维度执行批量检测，生成可追溯的质量报告。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd

from ctr_project.common.constants import (
    ID_COLUMN,
    MAX_NULL_RATIO,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

# 新增全局阈值常量
MAX_DUPLICATE_RATIO: float = 0.0001  # id 重复率警戒线：0.01%
MAX_TIME_PARSE_FAIL_RATIO: float = 0.005  # 时间解析失败率警戒线：0.5%

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """单条质量检查结果。

    Attributes:
        name: 检查项名称。
        passed: 是否通过。
        detail: 详细说明或异常统计。
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    """数据质量报告聚合对象。"""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        """追加一条检查结果。"""
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """全部检查项是否通过。"""
        return all(r.passed for r in self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 Pandas DataFrame 便于导出。"""
        return pd.DataFrame(
            [
                {"check": r.name, "passed": r.passed, "detail": r.detail}
                for r in self.results
            ]
        )


def check_schema(ddf: dd.DataFrame) -> CheckResult:
    """检查列名与必需字段是否一致。

    Args:
        ddf: 待校验的 Dask DataFrame。

    Returns:
        CheckResult 对象。
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in ddf.columns]
    if missing:
        return CheckResult(
            "schema_consistency",
            passed=False,
            detail=f"缺少字段: {missing}",
        )
    return CheckResult("schema_consistency", passed=True, detail="字段完整")


def check_completeness(ddf: dd.DataFrame) -> CheckResult:
    """检查关键字段空值比例是否低于阈值。

    Args:
        ddf: 待校验的 Dask DataFrame。

    Returns:
        CheckResult 对象。
    """
    critical = [ID_COLUMN, TARGET_COLUMN, "hour"]
    try:
        null_counts = ddf[critical].isnull().sum().compute()
        total = len(ddf)
        ratios = (null_counts / total).to_dict()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "completeness", passed=False, detail=f"计算失败: {exc}"
        )

    violations = {
        k: v for k, v in ratios.items() if v > MAX_NULL_RATIO
    }
    if violations:
        return CheckResult(
            "completeness",
            passed=False,
            detail=f"空值比例超阈值: {violations}",
        )
    return CheckResult(
        "completeness", passed=True, detail=f"空值比例: {ratios}"
    )


def check_uniqueness(ddf: dd.DataFrame) -> CheckResult:
    """检查 id 字段是否全局唯一（引入重复率告警阈值）。"""
    if ID_COLUMN not in ddf.columns:
        return CheckResult("uniqueness", passed=False, detail=f"缺少 {ID_COLUMN}")

    try:
        total = int(ddf.shape[0].compute())
        unique = int(ddf[ID_COLUMN].nunique().compute())
    except Exception as exc:
        return CheckResult("uniqueness", passed=False, detail=f"计算失败: {exc}")

    if total == 0:
        return CheckResult("uniqueness", passed=False, detail="数据为空")

    dup_count = total - unique
    dup_ratio = dup_count / total
    
    # 如果重复率超过警戒线，则失败告警
    if dup_ratio > MAX_DUPLICATE_RATIO:
        return CheckResult(
            "uniqueness",
            passed=False,
            detail=f"id 重复率 {dup_ratio*100:.4f}% 超过阈值 {MAX_DUPLICATE_RATIO*100}%",
        )
    
    return CheckResult(
        "uniqueness",
        passed=True,
        detail=f"id 重复 {dup_count} 条 (重复率 {dup_ratio*100:.4f}%)",
    )

def check_validity(ddf: dd.DataFrame) -> CheckResult:
    """检查目标变量与时间字段取值有效性（引入解析失败率告警）。"""
    issues: list[str] = []
    
    if TARGET_COLUMN in ddf.columns:
        invalid_target = int((~ddf[TARGET_COLUMN].isin([0, 1])).sum().compute())
        if invalid_target > 0:
            # 目标变量非 0/1 极其严重，0容忍
            issues.append(f"click 非 0/1 异常: {invalid_target} 条")

    if "hour_dt" in ddf.columns:
        total = int(ddf.shape[0].compute())
        invalid_time = int(ddf["hour_dt"].isnull().sum().compute())
        invalid_ratio = invalid_time / total if total > 0 else 0
        
        if invalid_ratio > MAX_TIME_PARSE_FAIL_RATIO:
            issues.append(
                f"hour_dt 解析失败率 {invalid_ratio*100:.2f}% 超过阈值 {MAX_TIME_PARSE_FAIL_RATIO*100}%"
            )

    if issues:
        return CheckResult("validity", passed=False, detail="; ".join(issues))
    return CheckResult("validity", passed=True, detail="取值有效")

def check_correctness(ddf: dd.DataFrame) -> CheckResult:
    """检查逻辑一致性（click 取值仅为 0 或 1）。

    Args:
        ddf: 待校验的 Dask DataFrame。

    Returns:
        CheckResult 对象。
    """
    if TARGET_COLUMN not in ddf.columns:
        return CheckResult(
            "correctness", passed=False, detail="缺少 click 列"
        )

    try:
        value_counts = ddf[TARGET_COLUMN].value_counts().compute()
        allowed = set(value_counts.index.tolist())
        if not allowed.issubset({0, 1}):
            return CheckResult(
                "correctness",
                passed=False,
                detail=f"click 取值异常: {allowed}",
            )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "correctness", passed=False, detail=f"计算失败: {exc}"
        )

    return CheckResult(
        "correctness", passed=True, detail="click 仅含 0/1"
    )


def run_quality_checks(ddf: dd.DataFrame) -> QualityReport:
    """执行全部质量检查项。

    Args:
        ddf: 待校验的 Dask DataFrame。

    Returns:
        聚合后的 QualityReport。
    """
    total = int(len(ddf))
    unique_id = int(ddf[ID_COLUMN].nunique().compute())
    null_id = int(ddf[ID_COLUMN].isnull().sum().compute())
    null_click = int(ddf[TARGET_COLUMN].isnull().sum().compute())
    null_hour = int(ddf["hour"].isnull().sum().compute())

    report = QualityReport()
    report.add(check_schema(ddf))
    report.add(
        CheckResult(
            "completeness",
            passed=all(
                r <= MAX_NULL_RATIO
                for r in [null_id / total, null_click / total, null_hour / total]
            ),
            detail=f"id 缺失={null_id}, click 缺失={null_click}, hour 缺失={null_hour}",
        )
    )
    report.add(
        CheckResult(
            "uniqueness",
            passed=(unique_id == total),
            detail=f"id 唯一值={unique_id}, 总行数={total}",
        )
    )
    report.add(check_validity(ddf))
    report.add(check_correctness(ddf))
    return report


def export_report(report: QualityReport, out_path: str | Path) -> None:
    """将质量报告导出为 CSV。

    Args:
        report: QualityReport 对象。
        out_path: 输出 CSV 路径。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_dataframe().to_csv(path, index=False)
    logger.info("质量报告已导出: %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from ctr_project.data.convert_parquet import read_parquet

    ddf_ = read_parquet("data/interim/cleaned")
    report_ = run_quality_checks(ddf_)
    print(report_.to_dataframe().to_string(index=False))
    export_report(report_, "data/reports/quality_report.csv")