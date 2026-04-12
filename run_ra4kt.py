import torch
import networkx as nx
import pandas as pd
from model.kt_models.dkt import DKT
from model.ra4kt import RA4KT
from model.postprocess import TSPPPathGenerator

def load_student_sequence(df: pd.DataFrame, user_id: int, target_kc: int) -> list:
    """从数据集中加载指定学生的学习序列"""
    user_df = df[df["user_id"] == user_id].sort_values("order_id")
    # 截取到目标KC之前的序列（包含目标KC）
    target_idx = user_df[user_df["skill_id"] == target_kc].index[0]
    user_df = user_df.loc[:target_idx]
    return list(zip(user_df["skill_id"], user_df["correct"]))

def main():
    # 1. 配置参数
    dataset_name = "xes3g5m"
    num_skills = 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载训练好的KT模型
    kt_model = DKT(num_skills=num_skills, hidden_dim=100).to(device)
    kt_model.load_state_dict(torch.load(f"checkpoints/dkt_{dataset_name}.pth", map_location=device))
    kt_model.eval()

    # 3. 加载KC关联图
    kc_graph = nx.read_gpickle(f"data/kc_graph_{dataset_name}.gpkl")

    # 4. 初始化RA4KT和路径生成器
    ra4kt = RA4KT(
        kt_model=kt_model,
        kc_graph=kc_graph,
        lambda_cons=,
        lambda_coh=,
        max_iter=,
        lr=,
        device=device
    )
    path_generator = TSPPPathGenerator(kc_graph)

    # 5. 加载测试数据，选择一个需要帮助的学生
    test_df = pd.read_csv(f"data/{dataset_name}_test.csv")
    example_user = test_df[test_df["correct"] == 0]["user_id"].iloc[0]
    example_target_kc = test_df[(test_df["user_id"] == example_user) & (test_df["correct"] == 0)]["skill_id"].iloc[0]
    orig_seq = load_student_sequence(test_df, example_user, example_target_kc)

    # 6. 生成反向归因
    print(f"Generating reverse attribution for student {example_user}, target KC {example_target_kc}...")
    cf_seq, modified_kcs = ra4kt.generate(orig_seq)
    print(f"Modified KCs to review: {modified_kcs}")

    # 7. 生成有序教学路径
    teaching_path = path_generator.generate(modified_kcs, example_target_kc)
    print(f"Recommended teaching path (from basics to target): {teaching_path}")

if __name__ == "__main__":
    main()