import torch
import numpy as np
import networkx as nx
from typing import List, Tuple, Set

def hamming_distance(x torch.Tensor, y torch.Tensor) - torch.Tensor
    计算汉明距离，对应论文公式(3)
    return torch.sum(torch.abs(torch.round(x) - y))

def gumbel_sigmoid(logits torch.Tensor, temperature float = 0.5) - torch.Tensor
    Gumbel-Sigmoid重参数化，对应论文最优初始化策略
    gumbel1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    gumbel2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    return torch.sigmoid((logits + gumbel1 - gumbel2)  temperature)

class RA4KT
    def __init__(
        self,
        kt_model,
        kc_graph nx.Graph,
        lambda_cons float = ,
        lambda_coh float = 
        max_iter int = 
        lr float = 
        early_stop_threshold float = 
        target_threshold float = 
        device str = cuda if torch.cuda.is_available() else cpu
    )
        self.kt_model = kt_model.to(device)
        self.kc_graph = kc_graph
        self.lambda_cons = lambda_cons
        self.lambda_coh = lambda_coh
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop_threshold = early_stop_threshold
        self.target_threshold = target_threshold
        self.device = device

        # 冻结KT模型参数（黑盒模式）
        for param in self.kt_model.parameters()
            param.requires_grad = False
        self.kt_model.eval()

        # 预计算所有KC对的最短路径距离
        self.kc_shortest_paths = dict(nx.all_pairs_shortest_path_length(kc_graph))

    def _build_actionability_mask(self, r_orig torch.Tensor) - torch.Tensor
        构建可操作性掩码，对应论文公式(10)：仅允许修改原始答错的位置
        return (r_orig == 0).float().to(self.device)

    def _compute_val_loss(self, cf_seq torch.Tensor, target_kc int) - torch.Tensor
        计算有效性损失，对应论文公式(6)：引导预测翻转
        # 构造KT模型输入：将cf_seq转换为(kc, answer)对的索引
        # 注意：此处需根据实际KT模型输入格式调整，示例假设输入为kc + answernum_skills
        kt_input = self._convert_to_kt_input(cf_seq)
        pred = self.kt_model(kt_input.unsqueeze(0))[0, -1, target_kc]  # 取最后一步目标KC的预测
        return -torch.log(pred + 1e-8)  # 二元交叉熵损失

    def _compute_cons_loss(self, cf_seq torch.Tensor, r_orig torch.Tensor) - torch.Tensor
        计算约束损失，对应论文公式(7)：最小化修改量
        return hamming_distance(cf_seq, r_orig)  len(r_orig)

    def _compute_coh_loss(self, cf_seq torch.Tensor, r_orig torch.Tensor, kc_seq List[int], target_kc int) - torch.Tensor
        计算一致性损失，对应论文公式(9)：优先选择与目标KC关联强的概念
        modified_steps = torch.where(torch.round(cf_seq) != r_orig)[0]
        modified_kcs = set([kc_seq[i] for i in modified_steps.cpu().numpy()])
        
        total_distance = 0.0
        for kc in modified_kcs
            if kc in self.kc_shortest_paths and target_kc in self.kc_shortest_paths[kc]
                total_distance += self.kc_shortest_paths[kc][target_kc]
            else
                total_distance += 10.0  # 无关联时的惩罚项
        return torch.tensor(total_distance, dtype=torch.float32, device=self.device)

    def _convert_to_kt_input(self, r_seq torch.Tensor) - torch.Tensor
        将答题序列转换为KT模型输入（需根据实际模型调整）
        # 示例：假设kc_seq已保存，此处简化处理
        # 实际使用时需传入kc_seq，构造kc + answernum_skills的输入
        return r_seq.long()  # 仅为示例，需替换为真实输入构造逻辑

    def generate(
        self,
        orig_seq List[Tuple[int, int]],  # [(kc1, r1), (kc2, r2), ..., (target_kc, 0)]
    ) - Tuple[List[int], Set[int]]
        
        生成反向归因序列，对应论文算法1
        param orig_seq 原始学习序列，最后一个元素为目标KC且答错
        return (反向归因答题序列, 需要复习的KC集合)
        
        # 1. 解析输入序列
        kc_seq = [kc for kc, r in orig_seq]
        r_orig = torch.tensor([r for kc, r in orig_seq], dtype=torch.float32, device=self.device)
        target_kc = kc_seq[-1]
        seq_len = len(orig_seq)

        # 2. 初始化反向归因序列（Gumbel-Sigmoid策略）
        logits = torch.zeros(seq_len, device=self.device)
        r_cf = gumbel_sigmoid(logits, temperature=0.5)
        r_cf.requires_grad = True

        # 3. 构建可操作性掩码
        action_mask = self._build_actionability_mask(r_orig)

        # 4. 初始化优化器
        optimizer = torch.optim.Adam([r_cf], lr=self.lr)

        # 5. 迭代优化
        for iter in range(self.max_iter)
            optimizer.zero_grad()

            # 应用可操作性掩码：仅允许修改原始答错的位置
            r_cf_masked = r_cf  action_mask + r_orig  (1 - action_mask)

            # 计算多目标损失，对应论文公式(5)
            val_loss = self._compute_val_loss(r_cf_masked, target_kc)
            cons_loss = self._compute_cons_loss(r_cf_masked, r_orig)
            coh_loss = self._compute_coh_loss(r_cf_masked, r_orig, kc_seq, target_kc)
            total_loss = val_loss + self.lambda_cons  cons_loss + self.lambda_coh  coh_loss

            # 反向传播与更新
            total_loss.backward()
            optimizer.step()

            # 再次应用掩码确保约束
            with torch.no_grad()
                r_cf.data = r_cf.data  action_mask + r_orig.data  (1 - action_mask)
                r_cf.data = torch.clamp(r_cf.data, 0.0, 1.0)

            # 早停检查
            if total_loss.item()  self.early_stop_threshold
                break

            # 检查是否达到目标预测阈值
            with torch.no_grad()
                r_cf_binary = torch.round(r_cf_masked)
                kt_input = self._convert_to_kt_input(r_cf_binary)
                final_pred = self.kt_model(kt_input.unsqueeze(0))[0, -1, target_kc].item()
                if final_pred = self.target_threshold
                    break

        # 6. 生成最终结果
        r_cf_final = torch.round(r_cf_masked).detach().cpu().numpy().astype(int).tolist()
        modified_kcs = set([
            kc_seq[i] for i in range(seq_len) 
            if r_cf_final[i] != r_orig[i].item()
        ])
        return r_cf_final, modified_kcs