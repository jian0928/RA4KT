import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DKT(nn.Module):
    
    def __init__(
        self,
        num_skills: int,
        hidden_dim: int = ,
        dropout_rate: float = ,
        emb_dim: int = 
    ):
        super().__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        
        # 嵌入层：将(kc, answer)对映射为向量
        self.embedding = nn.Embedding(2 * num_skills, emb_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout_rate if hidden_dim > 1 else 0
        )
        
        # 输出层：预测每个KC的答对概率
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor = None) -> torch.Tensor:
        
        # 嵌入
        x_emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        
        # LSTM前向传播
        if seq_len is not None:
            x_emb = pack_padded_sequence(x_emb, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_emb)
        if seq_len is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # 预测
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq_len, num_skills)
        return torch.sigmoid(logits)