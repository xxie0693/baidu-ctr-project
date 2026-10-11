# 百度广告点击率预测与投放优化分析

基于 Avazu 广告曝光-点击数据集（约 40428967 条记录，覆盖 10 天连续投放周期），
构建从数据治理、特征挖掘、多模型 CTR 预估到投放优化的全链路分析项目。

## 项目背景

点击率（CTR）预估是广告精准投放、流量价值最大化的核心技术底座。
本项目对标百度凤巢搜索广告与信息流广告的真实业务流程，完成以下核心目标：

1. 千万级广告行为数据的工程化治理
2. 覆盖时间、广告位、媒体内容、用户设备四大维度的全量特征体系
3. 传统机器学习与深度学习结合的多模型 CTR 预估体系
4. 广告投放优化、定向策略调整的可落地建议
5. 可复用的广告数据分析与建模工具包

## 项目结构
baidu-ctr-project/
├── src/ctr_project/ # 源码
│ ├── init.py
│ ├── config.py # 配置管理
│ ├── common/ # 公共组件
│ │ ├── exceptions.py # 自定义异常
│ │ ├── constants.py # 常量定义
│ │ └── utils.py # 工具函数
│ ├── data/ # 数据层
│ ├── features/ # 特征层
│ └── models/ # 模型层
├── tests/ # 测试
├── scripts/ # 运维脚本
├── configs/ # 配置文件
├── docs/ # 项目文档
├── data/
│ ├── raw/ # 原始数据（不纳入版本控制）
│ ├── interim/ # 中间产物（不纳入版本控制）
│ ├── processed/ # 最终数据集（不纳入版本控制）
│ └── reports/ # 质量报告
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore

## 环境安装

```bash
conda create -n ctr python=3.10 -y
conda activate ctr
pip install -e .

运行流程

第一阶段：数据工程

bash
python -m ctr_project.data.load_data
python -m ctr_project.data.clean_data
python -m ctr_project.data.validate_quality
python -m ctr_project.data.convert_parquet
