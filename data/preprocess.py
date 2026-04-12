import os
import pickle
import pandas as pd
import networkx as nx
from pykt.utils import preprocess_data

def build_kc_graph(df: pd.DataFrame, kc_col: str = "skill_id") -> nx.Graph:
    
    G = nx.Graph()
    # 1. 添加所有KC节点
    kcs = df[kc_col].unique()
    G.add_nodes_from(kcs)
    
    # 2. 基于题目共现和教学顺序添加边（模拟专家标注的先验依赖）
    # 实际使用时可替换为真实的教学大纲依赖关系
    for user_id in df["user_id"].unique():
        user_seq = df[df["user_id"] == user_id].sort_values("order_id")[kc_col].tolist()
        for i in range(len(user_seq)-1):
            if G.has_edge(user_seq[i], user_seq[i+1]):
                G[user_seq[i]][user_seq[i+1]]["weight"] += 1
            else:
                G.add_edge(user_seq[i], user_seq[i+1], weight=1)
    
    # 3. 保留权重高的边（过滤噪声）
    threshold = 2
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < threshold]
    G.remove_edges_from(edges_to_remove)
    return G

def preprocess_dataset(dataset_name: str, max_seq_len: int = 50):
    
    # 1. 使用pykt预处理原始数据
    if dataset_name == "xes3g5m":
        # 需先从pykt下载XES3G5M数据集
        raw_data = preprocess_data(dataset_name, "data/raw/xes3g5m")
    elif dataset_name == "assistments2017":
        raw_data = preprocess_data(dataset_name, "data/raw/assistments2017")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 2. 构建并保存KC关联图
    kc_graph = build_kc_graph(raw_data["train"])
    nx.write_gpickle(kc_graph, f"data/kc_graph_{dataset_name}.gpkl")
    
    # 3. 保存预处理后的序列数据
    for split in ["train", "valid", "test"]:
        raw_data[split].to_csv(f"data/{dataset_name}_{split}.csv", index=False)
    
    print(f"Preprocessing complete for {dataset_name}!")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    preprocess_dataset("xes3g5m", max_seq_len=50)
    preprocess_dataset("assistments2017", max_seq_len=50)