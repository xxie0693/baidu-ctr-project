"""深度学习模型 (Wide & Deep) 的 PyTorch 实现。"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from ctr_project.models.data_loader import load_and_preprocess_data, CAT_FEATURES

logger = logging.getLogger(__name__)


class WideAndDeep(nn.Module):
    """Wide & Deep 联合模型。"""

    def __init__(self, vocab_sizes: list[int], num_numeric_features: int, embed_dim: int = 8):
        super().__init__()
        # Deep 部分：每个类别特征一个 Embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # Deep 部分的 MLP
        deep_input_dim = len(vocab_sizes) * embed_dim + num_numeric_features
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Wide 部分（简化版：给每个类别特征一个一阶权重）
        self.wide_linear = nn.ModuleList([
            nn.Embedding(vocab_size, 1) for vocab_size in vocab_sizes
        ])
        self.numeric_linear = nn.Linear(num_numeric_features, 1)

    def forward(self, cat_x, num_x):
        # Deep
        embed_list = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        deep_input = torch.cat(embed_list, dim=1)
        deep_input = torch.cat([deep_input, num_x], dim=1)
        deep_out = self.mlp(deep_input)
        
        # Wide
        wide_out = sum(w(cat_x[:, i]) for i, w in enumerate(self.wide_linear))
        wide_out += self.numeric_linear(num_x)
        
        return torch.sigmoid(wide_out + deep_out).squeeze()


def train_dl_model(X_train, y_train, X_val, y_val):
    """训练 PyTorch 深度学习模型。"""
    logger.info("准备 PyTorch 张量...")
    
    # 获取 Embedding 词汇表大小
    vocab_sizes = [int(X_train[col].max()) + 2 for col in CAT_FEATURES]
    
    # 转换为 Tensor
    cat_cols = CAT_FEATURES
    num_cols = ["hour_of_day", "day_of_week"]
    
    train_dataset = TensorDataset(
        torch.tensor(X_train[cat_cols].values, dtype=torch.long),
        torch.tensor(X_train[num_cols].values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    val_cat = torch.tensor(X_val[cat_cols].values, dtype=torch.long)
    val_num = torch.tensor(X_val[num_cols].values, dtype=torch.float32)
    val_y = torch.tensor(y_val.values, dtype=torch.float32)
    
    # 初始化模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model = WideAndDeep(vocab_sizes, num_numeric_features=2, embed_dim=8).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger.info("开始训练 Wide & Deep 模型...")
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_cat, batch_num, batch_y) in enumerate(train_loader):
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_cat, batch_num)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 每 100 个 batch 打印一次进度
            if (i + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_pred = model(val_cat.to(device), val_num.to(device)).cpu().numpy()
            val_auc = roc_auc_score(val_y.numpy(), val_pred)
            
        logger.info(f"Epoch {epoch+1}/{epochs} 结束 | 平均 Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 抽样 5%（约 200 万行），避免 CPU 训练太慢
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(sample_ratio=0.05)
    
    dl_model = train_dl_model(X_train, y_train, X_val, y_val)
    logger.info("深度学习模型训练完成。")