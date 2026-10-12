一、文件写入命令

bash
cd ~/Downloads/baidu-ctr-projrct

# 确保 docs 目录存在
mkdir -p docs

# 用 Python heredoc 写入（避免 shell heredoc 卡住）
python << 'PY_EOF'
content = '''# 数据处理报告

> 项目：百度广告点击率预测与投放优化分析
> 阶段：第一阶段 — 数据工程体系搭建
> 数据：Avazu 广告曝光-点击数据集
> 报告日期：2026-10-12

---

## 一、数据处理逻辑说明

### 1.1 数据来源与规模

| 项 | 值 |
|----|----|
| 数据集 | Avazu CTR Prediction Dataset |
| 文件 | `data/raw/train.csv` |
| 原始大小 | 约 6 GB（CSV，未压缩） |
| 记录数 | 40,428,967 条 |
| 字段数 | 24 列 |
| 时间跨度 | 2014-10-21 00:00 至 2014-10-30 23:00（10 天） |
| 数据格式 | hour 字段编码为 YYMMDDHH |

### 1.2 处理流程总览
原始 CSV (6 GB)
↓ [load_data.py] Dask 分块加载 + 类型压缩
↓ [clean_data.py] 去重 → 时间解析 → 时间过滤 → click 过滤 → 空值过滤
↓ [validate_quality.py] 五维质量校验
↓ [convert_parquet.py] Parquet + zstd 压缩
标准化 Parquet 数据集 (1.5 GB, 63 分区)

text

### 1.3 加载阶段（load_data.py）

**核心问题**：40,428,967 行 × 24 列的 CSV 约 6 GB，单次 Pandas 全量加载会触发 OOM（macOS 内存限制）。

**技术方案**：

1. **Dask 分块加载**：使用 `dd.read_csv(blocksize="100MB")` 将数据切分为 63 个分区。每个分区独立加载，内存峰值控制在单分区大小以内。
2. **类型压缩**：读取时显式声明 dtype，避免 Dask 自动推断为 `int64` 或 `object`。

| 字段 | 原始类型 | 优化后类型 | 压缩理由 |
|------|---------|-----------|---------|
| click | int64 | int8 | 仅 0/1，1 字节足够 |
| banner_pos | int64 | uint8 | 取值 0-7 |
| device_type | int64 | uint16 | 取值有限 |
| device_conn_type | int64 | float32 | 含少量空值，暂保留 float |
| C1, C14-C21 | object | category | 高基数类别，字典编码 |
| site_id, app_id 等 | object | category | 同上 |
| site_domain, app_domain | object | string | 中基数，保留 string |

**效果**：内存占用从理论上的 ~12 GB（未优化）降至 **3-4 GB 级别**，可稳定在单机运行。

### 1.4 清洗阶段（clean_data.py）

清洗流程按以下顺序执行，每一步都有明确依据。

#### 1.4.1 去重（基于 id 字段）

```python
ddf.drop_duplicates(subset=["id"])
依据：id 是广告曝光记录的唯一标识，理论上全局唯一。
实际结果：移除 0 条。Avazu 数据集本身已保证 id 唯一，清洗逻辑正确但数据本身无重复。
1.4.2 时间解析

python
pd.to_datetime(part["hour"], format="%y%m%d%H", errors="coerce")
依据：hour 字段格式为 YYMMDDHH（如 14102100 表示 2014-10-21 00 时）。
新增字段：hour_dt（datetime64 类型），供后续时间维度分析使用。
容错：errors="coerce" 将无法解析的值转为 NaT，供下一步过滤。
1.4.3 时间有效性过滤

python
MIN_DATETIME = pd.Timestamp("2014-01-01")
MAX_DATETIME = pd.Timestamp("2015-12-31")
依据：Avazu 数据集时间范围为 2014-10-21 至 2014-10-30。设定合理区间 [2014-01-01, 2015-12-31]，覆盖数据实际范围的同时排除异常值。
踩坑记录：初版代码误写为 2024-01-01 至 2030-12-31，导致所有 2014 年数据被过滤为空。修复后正常。
实际结果：移除 0 条，数据时间全部在有效范围内。
1.4.4 目标变量过滤

python
ddf[ddf["click"].isin([0, 1])]
依据：click 是二分类目标变量，取值仅应为 0（未点击）或 1（点击）。
实际结果：移除 0 条。
1.4.5 关键字段空值过滤

python
ddf.dropna(subset=["id", "click", "hour"])
依据：id、click、hour 是核心字段，任一为空则该记录无法用于分析或建模。
实际结果：移除 0 条。
1.5 清洗前后对比

阶段	行数	移除数	移除比例
原始数据	40,428,967	—	—
去重后	40,428,967	0	0.00%
时间过滤后	40,428,967	0	0.00%
click 过滤后	40,428,967	0	0.00%
空值过滤后	40,428,967	0	0.00%
最终	40,428,967	0	0.00%
结论：Avazu 是高质量竞赛数据集，清洗前后行数不变。这不是清洗逻辑失效，而是数据本身干净。清洗逻辑的价值在于：

可复用性：同样的清洗流程可用于生产环境中质量参差不齐的真实数据。
防御性：清洗代码中包含的过滤条件，能捕获未来数据源可能出现的异常。
可验证性：通过五维质量校验脚本证实了"清洗前后无变化"这一事实。
1.6 格式转换阶段（convert_parquet.py）

转换目的：

维度	CSV	Parquet
存储方式	行式	列式
压缩	无	zstd
读取特定列	需扫描全文件	只读相关列
数据类型保留	需重新推断	保留 schema
分析型查询性能	慢	快 5-10 倍
实际效果：

项	值
原始 CSV	~6 GB
转换后 Parquet	1.5 GB
压缩率	75%
分区数	63
压缩算法	zstd
踩坑记录：Dask 的 category 类型在不同分区可能推断出不同编码宽度（如 int8 与 int16），写 Parquet 时 schema 冲突报错。解决方案：写盘前将 category 列统一转为 string，由 pyarrow 在写盘时重新做字典编码。

二、质量评估结果

2.1 五维质量校验体系

数据质量校验从以下五个维度展开：

维度	含义	检查内容
Schema 一致性	字段结构符合预期	必需字段是否完整
完整性	关键字段无缺失	id/click/hour 的空值比例
唯一性	主键不重复	id 字段唯一值数量
有效性	取值符合格式	hour 可正常解析为 datetime
正确性	逻辑上无矛盾	click 取值仅为 0/1
2.2 校验结果

检查项	通过	详情
schema_consistency	✅	字段完整（24 + hour_dt = 25 列）
completeness	✅	id 缺失=0, click 缺失=0, hour 缺失=0
uniqueness	✅	id 唯一值=40,428,967, 总行数=40,428,967
validity	✅	时间解析全部有效
correctness	✅	click 仅含 0/1
结论：5/5 项全部通过，数据质量达到建模要求。

2.3 关键统计摘要

指标	值
总记录数	40,428,967
时间跨度	2014-10-21 00:00 至 2014-10-30 23:00
天数	10 天
分区数	63
字段数	25（含 hour_dt）
文件大小	1.5 GB（Parquet + zstd）
click 正样本率	约 17%（待补充精确值）
2.4 数据质量报告

自动化校验脚本产出可追溯的质量报告，路径：data/reports/quality_report.csv

csv
check,passed,detail
schema_consistency,True,字段完整
completeness,True,"id 缺失=0, click 缺失=0, hour 缺失=0"
uniqueness,True,"id 唯一值=40428967, 总行数=40428967"
validity,True,取值有效
correctness,True,click 仅含 0/1
三、中间表设计说明

3.1 设计目标

基于清洗后的标准化数据，构建四维基础聚合中间表，为第二阶段特征工程与探索性分析提供预计算基础。四张表分别对应广告位、媒体、用户设备、时间四个维度。

3.2 星型模型设计

采用星型模型（Star Schema）组织数据：

text
                ┌──────────────────┐
                │  fact_impression │  ← 事实表
                │  (40,428,967 行) │    每条曝光一条记录
                └────────┬─────────┘
                         │
        ┌────────────────┼────────────────┬────────────────┐
        │                │                │                │
┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼──────┐ ┌──────▼───────┐
│ dim_ad_position│ │  dim_media  │ │ dim_device   │ │  dim_time    │
│  (广告位维度)  │ │  (媒体维度) │ │ (设备维度)   │ │ (时间维度)   │
└───────────────┘ └─────────────┘ └──────────────┘ └──────────────┘
3.3 表结构定义

3.3.1 广告位维度表 dim_ad_position

字段	类型	说明
banner_pos	int	广告位编码（主键）
impressions	int	曝光总量
clicks	int	点击总量
ctr	float	点击率 = clicks / impressions
position_share	float	该广告位流量占比
3.3.2 媒体维度表 dim_media

字段	类型	说明
media_key	string	site_id 或 app_id（主键）
media_type	string	'site' 或 'app'
media_category	string	site_category 或 app_category
impressions	int	曝光总量
clicks	int	点击总量
ctr	float	点击率
heat_score	float	热度评分（曝光量的归一化）
3.3.3 用户设备维度表 dim_device

字段	类型	说明
device_type	int	设备类型编码（主键）
device_model	string	设备型号
impressions	int	曝光总量
clicks	int	点击总量
ctr	float	点击率
device_share	float	该设备类型流量占比
3.3.4 时间维度表 dim_time

字段	类型	说明
hour_dt	datetime	小时级时间戳（主键）
hour_of_day	int	0-23
day_of_week	int	0=周一, 6=周日
is_weekend	bool	是否周末
impressions	int	曝光总量
clicks	int	点击总量
ctr	float	点击率
3.4 中间表落地方案

方案	优点	缺点	采用
A. 独立 Parquet 文件	与主数据集一致	与第二阶段 SQLite 重复	备选
B. 第二阶段 SQLite 建表	避免重复 I/O，支持 SQL 查询	需等第二阶段	✅ 主选
C. CSV 文件	通用	解析慢	不推荐
决定：中间表统一在第二阶段 SQLite 阶段实现。理由：

第二阶段 SQLite 本身就需要设计星型模型，中间表设计可以复用。
避免第一阶段生成一次 Parquet、第二阶段再导入 SQLite 的重复 I/O。
SQL 聚合的性能在 SQLite 中优于 Dask，适合维度统计。
当前状态：中间表设计已完成（本文档），实现延后到第二阶段。