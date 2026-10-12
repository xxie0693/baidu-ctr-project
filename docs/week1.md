# PTA项目第一周小结

## 主要内容

通过 **Python + Dask** 工具链，完成了 Avazu 广告数据集（4040 万条记录）的工程化加载与清洗：

- 搭建了模块化的项目目录结构（`src/ctr_project/data/`）
- 编写了 `load_data.py`、`clean_data.py`、`validate_quality.py`、`convert_parquet.py` 四个核心脚本
- 实现了**分块加载、类型压缩、去重、时间解析与异常过滤**的全流程自动化
- 产出了标准化 Parquet 数据集（**40428967 行 × 25 列**）
- 建立了五维数据质量校验体系，质量报告 **5 项全部通过**
- 完成了 Git 仓库初始化与首次提交，代码已托管到 GitHub

## 遇到的挫折和困难

### 困难一：Dask 的惰性计算模型理解不透彻

刚上手时，我在 `clean_data.py` 里写了 `logger.info("去重完成")`，日志显示 **8 毫秒**就完成了 4040 万行的去重——当时还很得意，结果发现 Dask 根本没有执行计算，只是构建了惰性任务图，日志只是函数返回的时间。直到 `to_parquet` 才真正触发计算。这导致我一度以为清洗成功，实际上游数据为空都没发现。

**克服方式：** 通过仔细阅读 Dask 官方文档的 "Lazy Evaluation" 章节，理解了 Dask 与 Pandas 的核心差异，学会在关键节点用 `.compute()` 或 `.persist()` 触发计算。同时举一反三，把每一阶段的中间结果都写成日志，形成了"每步可观测"的调试习惯。

### 困难二：Dask category 类型跨分区编码不一致导致 Parquet 写入失败

在写 Parquet 时遇到：

```python
# pyarrow 抛出的异常
ArrowInvalid('Integer value 204 not in range: -128 to 127',
             'Conversion failed for column site_id with type category')
```

原因是不同分区的 `site_id` 唯一值数量不同，Dask 为第 0 个分区推断 `int8` 索引，为其他分区推断 `int16`，写盘时 schema 冲突。

**克服方式：** 经过多次尝试，最终采用"写盘前将 category 列统一转为 string"的方案。虽然文件会略大一点，但避免了跨分区编码冲突。这个坑也让我意识到：Dask 的分布式特性会带来 Pandas 不会遇到的边界问题，使用任何分布式框架都要注意"**分区一致性**"。

### 困难三：Git 首次提交的反复试错

第一次提交时，`git add` 因为 `requirements.txt` 不存在而整条命令回滚（Git 的原子性），我误以为其他文件也没添加。另外 Git 身份没配置导致 commit 失败。这些是"工程实践"而非"数据分析"层面的坑。

**克服方式：** 通过 `git status`、`git diff --cached --stat` 逐步排查，学会把长命令拆成短命令逐条执行，避免原子操作失败后难以定位。同时把"新环境先配 `user.name` 和 `user.email`"作为肌肉记忆。

## 收获

1. 对大规模数据的加载、清洗、校验、落盘全流程更加熟练
2. 熟悉了 Conventional Commits 规范（如 `feat:`、`fix:`、`docs:`）
3. 通过查阅 Dask 官方文档与实践，学会了：
   - 使用 `dd.read_csv(blocksize="100MB")` 分块加载
   - 通过 dtype 压缩将内存占用降低 **60% 以上**
   - 使用 Parquet 列式存储将重复读取的 I/O 降低一个数量级

## 下周计划

- [ ] 完成特征工程脚本 `build_features.py`
- [ ] 搭建 baseline 模型（Logistic Regression / LightGBM）
- [ ] 补充单元测试与 CI

---