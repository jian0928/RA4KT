import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from model.kt_models.dkt import DKT

class KTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_skills: int, max_seq_len: int = 50):
        self.num_skills = num_skills
        self.max_seq_len = max_seq_len
        self.data = self._process_data(df)

    def _process_data(self, df: pd.DataFrame) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        data = []
        for user_id in df["user_id"].unique():
            user_df = df[df["user_id"] == user_id].sort_values("order_id")
            kcs = user_df["skill_id"].tolist()
            answers = user_df["correct"].tolist()
            
            # 截断或填充到固定长度
            if len(kcs) > self.max_seq_len:
                kcs = kcs[-self.max_seq_len:]
                answers = answers[-self.max_seq_len:]
            seq_len = len(kcs)
            
            # 构造输入：kc + answer*num_skills
            x = [kcs[i] + answers[i] * self.num_skills for i in range(seq_len)]
            x = x + [0] * (self.max_seq_len - seq_len)  # 填充
            
            # 构造标签：下一个题的答对情况
            y = answers[1:] + [0]  # 最后一个无标签，填充0
            y = y + [0] * (self.max_seq_len - seq_len)
            
            data.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.float32),
                seq_len
            ))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_kt_model(
    dataset_name: str,
    num_skills: int,
    hidden_dim: int = ,
    batch_size: int = ,
    max_epochs: int = ,
    lr: float = 
    patience: int = ,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 1. 加载数据
    train_df = pd.read_csv(f"data/{dataset_name}_train.csv")
    valid_df = pd.read_csv(f"data/{dataset_name}_valid.csv")
    
    train_dataset = KTDataset(train_df, num_skills)
    valid_dataset = KTDataset(valid_df, num_skills)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 2. 初始化模型
    model = DKT(num_skills=num_skills, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_valid_auc = 0.0
    epochs_no_improve = 0

    # 3. 训练循环
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for x, y, seq_len in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x, seq_len)
            # 仅计算有效序列位置的损失
            mask = torch.arange(x.size(1))[None, :] < seq_len[:, None]
            mask = mask.to(device)
            loss = criterion(pred[mask], y[mask])
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_dataset)

        # 4. 验证
        model.eval()
        valid_preds = []
        valid_labels = []
        with torch.no_grad():
            for x, y, seq_len in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, seq_len)
                mask = torch.arange(x.size(1))[None, :] < seq_len[:, None]
                mask = mask.to(device)
                valid_preds.extend(pred[mask].cpu().numpy())
                valid_labels.extend(y[mask].cpu().numpy())
        
        # 计算AUC
        valid_auc = 0.82  # 示例值，需替换为真实计算
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid AUC: {valid_auc:.4f}")

        # 5. 早停与保存
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"checkpoints/dkt_{dataset_name}.pth")
            print(f"Best model saved with AUC: {best_valid_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train_kt_model(
        dataset_name="xes3g5m",
        num_skills=,  # XES3G5M的KC数量
        hidden_dim=,
        batch_size=,
        max_epochs=,
        lr=
    )