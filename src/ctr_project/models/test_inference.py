"""模型推理性能与资源消耗测试。"""

import time
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

MODEL_DIR = Path("data/processed/models")

def test_model_inference():
    model_path = MODEL_DIR / "lgbm_model.txt"
    if not model_path.exists():
        print("找不到模型文件，请先运行 train_ml.py")
        return

    # 1. 测试模型大小
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"=== 模型大小 ===")
    print(f"文件大小: {model_size_mb:.2f} MB")

    # 2. 加载模型
    model = lgb.Booster(model_file=str(model_path))

    # 3. 构造模拟数据（16 个特征）
    # 14个类别特征 + 2个数值特征
    dummy_data = np.random.randint(0, 50, size=(1000, 16))
    dummy_df = pd.DataFrame(dummy_data, columns=[
        "banner_pos", "site_id", "app_id", "device_type", 
        "device_conn_type", "C1", "C14", "C15", "C16", "C17", 
        "C18", "C19", "C20", "C21", "hour_of_day", "day_of_week"
    ])

    # 4. 测试单条推理
    print(f"\n=== 推理速度测试 ===")
    single_sample = dummy_df.iloc[[0]]
    
    start = time.perf_counter()
    for _ in range(1000):
        model.predict(single_sample)
    end = time.perf_counter()
    print(f"单条样本推理平均耗时: {(end - start) * 1000:.3f} ms/条")

    # 5. 测试批量推理 (1000条)
    start = time.perf_counter()
    model.predict(dummy_df)
    end = time.perf_counter()
    print(f"批量推理 (1000条) 总耗时: {(end - start) * 1000:.3f} ms")
    print(f"批量平均单条耗时: {((end - start) * 1000) / 1000:.3f} ms/条")

if __name__ == "__main__":
    test_model_inference()
    