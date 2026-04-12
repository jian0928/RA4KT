import networkx as nx
from typing import List, Set

class TSPPPathGenerator:
    def __init__(self, kc_graph: nx.Graph):
        self.kc_graph = kc_graph
        # 预计算最短路径
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(kc_graph))
        self.shortest_path_nodes = dict(nx.all_pairs_shortest_path(kc_graph))

    def generate(self, kc_set: Set[int], target_kc: int) -> List[int]:
        
        if not kc_set:
            return [target_kc]
        
        # 1. 构造路径规划节点集
        node_set = kc_set.copy()
        node_set.add(target_kc)
        node_list = list(node_set)
        n = len(node_list)
        target_idx = node_list.index(target_kc)

        # 2. 构造最短路径距离矩阵
        dist_matrix = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                u, v = node_list[i], node_list[j]
                if u in self.shortest_paths and v in self.shortest_paths[u]:
                    dist_matrix[i][j] = self.shortest_paths[u][v]
                else:
                    dist_matrix[i][j] = 100.0  # 无关联时的大距离

        # 3. 贪心算法求解以目标KC为起点的最短哈密顿路径
        visited = [False] * n
        visited[target_idx] = True
        path = [target_idx]
        current = target_idx

        for _ in range(n-1):
            # 找到最近的未访问节点
            min_dist = float('inf')
            next_node = -1
            for neighbor in range(n):
                if not visited[neighbor] and dist_matrix[current][neighbor] < min_dist:
                    min_dist = dist_matrix[current][neighbor]
                    next_node = neighbor
            if next_node == -1:
                break  # 无法继续，提前终止
            visited[next_node] = True
            path.append(next_node)
            current = next_node

        # 4. 反转路径，得到从基础到目标的学习顺序
        path.reverse()
        kc_path = [node_list[i] for i in path]

        # 5. 扩展为完整的教学路径（包含中间依赖KC）
        full_path = []
        for i in range(len(kc_path)-1):
            u, v = kc_path[i], kc_path[i+1]
            if u in self.shortest_path_nodes and v in self.shortest_path_nodes[u]:
                full_path.extend(self.shortest_path_nodes[u][v][:-1])  # 避免重复
            else:
                full_path.append(u)
        full_path.append(kc_path[-1])
        
        # 去重并保持顺序
        seen = set()
        final_path = []
        for kc in full_path:
            if kc not in seen:
                seen.add(kc)
                final_path.append(kc)
        return final_path