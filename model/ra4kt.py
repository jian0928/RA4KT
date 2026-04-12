import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Optional

def hamming_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    return torch.sum(torch.abs(torch.round(x) - y))

def gumbel_sigmoid_relaxation(
    r_orig: torch.Tensor,
    temperature: float = ,
    noise_scale: float = 
) -> torch.Tensor:
   
    # 1. 保留原始序列信息，添加高斯噪声
    init_logits = r_orig.float() + noise_scale * torch.randn_like(r_orig)
    
    # 2. Gumbel-Max 重参数化实现可微采样
    gumbel1 = -torch.log(-torch.log(torch.rand_like(init_logits) + 1e-8) + 1e-8)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(init_logits) + 1e-8) + 1e-8)
    
    return torch.sigmoid((init_logits + gumbel1 - gumbel2) / temperature)

# ===================== RA4KT 核心类 =====================
class RA4KT:
    def __init__(
        self,
        kt_model: nn.Module,
        kc_graph: nx.Graph,
        num_skills: int,
        lambda_cons: float = ,        
        lambda_coh: float = ,        
        max_iter: int = ,              
        lr: float = ,                 
        early_stop_threshold: float = , 
        target_pred_threshold: float = , 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        
        self.kt_model = kt_model.to(device)
        self.kc_graph = kc_graph
        self.num_skills = num_skills
        self.lambda_cons = lambda_cons
        self.lambda_coh = lambda_coh
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop_threshold = early_stop_threshold
        self.target_pred_threshold = target_pred_threshold
        self.device = device

        # 冻结 KT 模型参数（黑盒模式，不修改原模型）
        for param in self.kt_model.parameters():
            param.requires_grad = False
        self.kt_model.eval()

        # 预计算所有 KC 对的最短路径距离（用于一致性损失）
        print("Precomputing shortest paths between knowledge concepts...")
        self.kc_shortest_paths = dict(nx.all_pairs_shortest_path_length(kc_graph))
        print("Shortest paths precomputed!")

    def _build_actionability_mask(self, r_orig: torch.Tensor) -> torch.Tensor:
        
        return (r_orig == 0).float().to(self.device)

    def _convert_seq_to_kt_input(
        self,
        kc_seq: List[int],
        r_seq: torch.Tensor
    ) -> torch.Tensor:
        n: KT 模型输入张量，shape=(seq_len,)
        
        r_binary = torch.round(r_seq).long()  # 连续值转二值
        kt_input = [
            kc_seq[i] + r_binary[i].item() * self.num_skills
            for i in range(len(kc_seq))
        ]
        return torch.tensor(kt_input, dtype=torch.long, device=self.device)

    def _compute_validity_loss(
        self,
        kc_seq: List[int],
        r_cf: torch.Tensor,
        target_kc: int
    ) -> torch.Tensor:
        
        # 1. 构造 KT 模型输入
        kt_input = self._convert_seq_to_kt_input(kc_seq, r_cf)
        
        # 2. 前向传播获取预测（取最后一步目标 KC 的预测概率）
        with torch.no_grad():
            # 先禁用梯度计算获取预测值，再单独对 r_cf 求导（数值梯度方案，兼容黑盒模型）
            pred = self.kt_model(kt_input.unsqueeze(0))[0, -1, target_kc]
        
        # 3. 构造可微的有效性损失（最大化预测概率等价于最小化负对数）
        # 注意：为了兼容黑盒模型，这里使用数值梯度近似，实际可根据 KT 模型可微性调整
        return -torch.log(torch.clamp(pred, min=1e-8, max=1-1e-8))

    def _compute_constraint_loss(
        self,
        r_cf: torch.Tensor,
        r_orig: torch.Tensor
    ) -> torch.Tensor:
        
        return hamming_distance(r_cf, r_orig) / len(r_orig)

    def _compute_coherence_loss(
        self,
        kc_seq: List[int],
        r_cf: torch.Tensor,
        r_orig: torch.Tensor,
        target_kc: int
    ) -> torch.Tensor:
        
        # 1. 找出修改的时间步
        modified_steps = torch.where(torch.round(r_cf) != r_orig)[0]
        if len(modified_steps) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 2. 提取修改的 KC 集合
        modified_kcs = set([kc_seq[i] for i in modified_steps.cpu().numpy()])
        
        # 3. 计算总最短路径距离
        total_distance = 0.0
        for kc in modified_kcs:
            if kc in self.kc_shortest_paths and target_kc in self.kc_shortest_paths[kc]:
                total_distance += self.kc_shortest_paths[kc][target_kc]
            else:
                total_distance += 10.0  # 无关联时的惩罚项（避免选择无关 KC）
        
        return torch.tensor(total_distance, dtype=torch.float32, device=self.device)

    def generate(
        self,
        orig_seq: List[Tuple[int, int]]
    ) -> Tuple[List[int], Set[int]]:
        
        # ===================== 步骤 1：解析输入序列 =====================
        kc_seq = [kc for kc, r in orig_seq]
        r_orig = torch.tensor([r for kc, r in orig_seq], dtype=torch.long, device=self.device)
        target_kc = kc_seq[-1]
        seq_len = len(orig_seq)

        # 验证输入合法性
        assert r_orig[-1].item() == 0, "Target KC must be answered incorrectly in original sequence!"

        # ===================== 步骤 2：初始化与掩码构建 =====================
        # Gumbel-Sigmoid 初始化
        r_cf = gumbel_sigmoid_relaxation(r_orig, temperature=0.5, noise_scale=0.1)
        r_cf.requires_grad = True  # 仅对 r_cf 求导，KT 模型保持黑盒

        # 构建可操作性掩码
        action_mask = self._build_actionability_mask(r_orig)

        # 初始化 Adam 优化器
        optimizer = torch.optim.Adam([r_cf], lr=self.lr)

        # ===================== 步骤 3：迭代优化 =====================
        for iter_idx in range(self.max_iter):
            optimizer.zero_grad()

            # 应用可操作性掩码硬约束
            # 仅允许修改原始答错的位置，强制重置答对位置为原始值
            r_cf_masked = r_cf * action_mask + r_orig.float() * (1 - action_mask)
            r_cf_masked = torch.clamp(r_cf_masked, min=0.0, max=1.0)  # 限制在 [0,1] 范围内

            # 计算多目标总损失
            # 注意：为兼容黑盒 KT 模型，这里使用数值梯度近似，可微模型可直接链式求导
            with torch.no_grad():
                # 先计算当前预测值
                kt_input = self._convert_seq_to_kt_input(kc_seq, r_cf_masked)
                current_pred = self.kt_model(kt_input.unsqueeze(0))[0, -1, target_kc].item()
            
            # 构造可微损失
            # 实际生产中若 KT 模型可微，可直接替换为链式求导以提高效率
            val_loss = -torch.log(torch.tensor(current_pred, device=self.device) + 1e-8)
            cons_loss = self._compute_constraint_loss(r_cf_masked, r_orig.float())
            coh_loss = self._compute_coherence_loss(kc_seq, r_cf_masked, r_orig.float(), target_kc)
            total_loss = val_loss + self.lambda_cons * cons_loss + self.lambda_coh * coh_loss

            
            with torch.no_grad():
                # 数值梯度近似：对每个可修改位置微小扰动
                grad = torch.zeros_like(r_cf)
                eps = 1e-4
                for i in range(seq_len):
                    if action_mask[i].item() == 1:
                        # 正向扰动
                        r_cf_plus = r_cf_masked.clone()
                        r_cf_plus[i] = torch.clamp(r_cf_plus[i] + eps, 0.0, 1.0)
                        kt_input_plus = self._convert_seq_to_kt_input(kc_seq, r_cf_plus)
                        pred_plus = self.kt_model(kt_input_plus.unsqueeze(0))[0, -1, target_kc].item()
                        
                        # 负向扰动
                        r_cf_minus = r_cf_masked.clone()
                        r_cf_minus[i] = torch.clamp(r_cf_minus[i] - eps, 0.0, 1.0)
                        kt_input_minus = self._convert_seq_to_kt_input(kc_seq, r_cf_minus)
                        pred_minus = self.kt_model(kt_input_minus.unsqueeze(0))[0, -1, target_kc].item()
                        
                        # 数值梯度（最大化预测概率）
                        grad[i] = (pred_plus - pred_minus) / (2 * eps)
                
                # 更新 r_cf（梯度上升最大化预测，同时考虑约束损失的梯度符号）
                r_cf.data += self.lr * grad * action_mask
                r_cf.data = torch.clamp(r_cf.data, 0.0, 1.0)

            # 再次应用掩码确保硬约束不被违反
            with torch.no_grad():
                r_cf.data = r_cf.data * action_mask + r_orig.float().data * (1 - action_mask)

            # 早停检查
            if total_loss.item() < self.early_stop_threshold:
                print(f"Early stopping at iteration {iter_idx+1}")
                break

            # 检查是否达到目标预测阈值
            with torch.no_grad():
                r_cf_binary = torch.round(r_cf_masked)
                kt_input_final = self._convert_seq_to_kt_input(kc_seq, r_cf_binary)
                final_pred = self.kt_model(kt_input_final.unsqueeze(0))[0, -1, target_kc].item()
                if final_pred >= self.target_pred_threshold:
                    print(f"Target prediction threshold reached at iteration {iter_idx+1}, final pred: {final_pred:.4f}")
                    break

        # ===================== 步骤 4：生成最终结果 =====================
        with torch.no_grad():
            r_cf_binary = torch.round(r_cf_masked).detach().cpu().numpy().astype(int).tolist()
        
        # 提取需要复习的去重 KC 集合（对应论文 Generality 属性）
        modified_kcs = set([
            kc_seq[i] for i in range(seq_len)
            if r_cf_binary[i] != r_orig[i].item()
        ])

        return r_cf_binary, modified_kcs