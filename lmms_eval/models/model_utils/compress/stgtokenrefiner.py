import torch
import torch.nn.functional as F
import numpy as np
import time

class STGTokenRefiner:
    def __init__(self, tau=0.5, alpha=0.5, beta=0.5, lambda_coeff=0.5, p=0.5):
        """
        3D时空图Token精炼模块初始化
        :param tau: 邻居筛选阈值
        :param alpha: 时空权重平衡系数
        :param beta: 时空依赖度平衡系数
        :param lambda_coeff: 权重/依赖度平衡系数
        :param p: 保留Token百分比
        """
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.lambda_coeff = lambda_coeff
        self.p = p

    def compute_similarity(self, x1, x2):
        """计算两个Token矩阵的余弦相似度矩阵"""
        # x1: (N, D), x2: (N, D) -> 输出 (N, N)
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)
        return torch.matmul(x1_norm, x2_norm.t())

    def build_similarity_matrices(self, X):
        """构建帧内和帧间相似度矩阵"""
        T, N, D = X.shape
        S_intra = []  # 帧内相似度: (T, N, N)
        S_inter = []  # 帧间相似度: (T-1, N, N)

        # 计算帧内相似度
        for t in range(T):
            x_t = X[t]  # (N, D)
            s_intra = self.compute_similarity(x_t, x_t)
            S_intra.append(s_intra)
        S_intra = torch.stack(S_intra, dim=0)

        # 计算相邻帧间相似度（仅相邻帧）
        for t in range(T - 1):
            x_t = X[t]  # 当前帧
            x_t1 = X[t + 1]  # 下一帧
            s_inter = self.compute_similarity(x_t, x_t1)
            S_inter.append(s_inter)
        S_inter = torch.stack(S_inter, dim=0) if T > 1 else torch.empty(0)

        return S_intra, S_inter

    def build_adjacency_matrices(self, S_intra, S_inter):
        """构建帧内和帧间邻接矩阵（0-1矩阵）"""
        T = S_intra.shape[0]
        N = S_intra.shape[1]

        # 帧内邻接矩阵: 相似度大于阈值的为1
        # import pdb;pdb.set_trace();
        A_intra = (S_intra > self.tau).float()
        # 确保自环（避免孤立节点）
        for t in range(T):
            A_intra[t].fill_diagonal_(1.0)

        # 帧间邻接矩阵
        A_inter = []
        if S_inter.numel() > 0:  # 存在相邻帧
            A_inter = (S_inter > self.tau).float()

        return A_intra, A_inter

    def compute_spatial_weights(self, X):
        """计算空间权重（L2范数归一化）"""
        T, N, D = X.shape
        w_s = torch.norm(X, p=2, dim=2)  # (T, N)
        max_ws = torch.max(w_s)
        return w_s / max_ws if max_ws > 0 else w_s

    def compute_temporal_weights(self, S_inter, T, N):
        """计算时间权重（时间变化显著性）"""
        w_t = torch.zeros(T, N, device=S_inter.device if S_inter.numel() > 0 else None)

        for t in range(T):
            if T == 1:
                # 单帧无时间信息
                wt_raw = 0.0
            elif t == 0:
                # 第一帧只有后向帧
                wt_raw = 1 - S_inter[t, :, :].diag()  # (N,)
            elif t == T - 1:
                # 最后一帧只有前向帧
                wt_raw = 1 - S_inter[t - 1, :, :].diag()  # (N,)
            else:
                # 中间帧取前后平均
                wt_raw = 1 - (S_inter[t - 1, :, :].diag() + S_inter[t, :, :].diag()) / 2

            w_t[t] = wt_raw

        # 全局归一化
        max_wt = torch.max(w_t)
        return w_t / max_wt if max_wt > 0 else w_t

    def compute_dependency(self, A_intra, A_inter, T, N):
        """计算时空依赖度（入度统计）"""
        # 空间依赖度（同帧内入度）
        dep_s = A_intra.sum(dim=2)  # (T, N)
        max_ds = torch.max(dep_s)
        dep_s_norm = dep_s / max_ds if max_ds > 0 else dep_s

        # 时间依赖度（跨帧入度）
        dep_t = torch.zeros(T, N, device=A_intra.device)
        if A_inter.numel() > 0:
            for t in range(T):
                # 前一帧指向当前帧的入度
                prev_in = A_inter[t - 1, :, :].sum(dim=0) if t > 0 else 0
                # 当前帧指向下一帧的入度（被下一帧指向）
                next_in = A_inter[t, :, :].sum(dim=1) if t < T - 1 else 0
                dep_t[t] = prev_in + next_in

        max_dt = torch.max(dep_t)
        dep_t_norm = dep_t / max_dt if max_dt > 0 else dep_t

        # 联合依赖度
        dep_st = self.beta * dep_s_norm + (1 - self.beta) * dep_t_norm
        return dep_st

    def compute_importance_scores(self, X, S_inter, A_intra, A_inter):
        """计算综合重要性分数"""
        T, N, D = X.shape

        # 时空权重
        w_s = self.compute_spatial_weights(X)
        w_t = self.compute_temporal_weights(S_inter, T, N)
        w_st = self.alpha * w_s + (1 - self.alpha) * w_t

        # 时空依赖度
        dep_st = self.compute_dependency(A_intra, A_inter, T, N)

        # 综合分数
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * dep_st
        return scores

    def refine_tokens(self, X, T):
        """
        主函数：对视频Token进行精炼压缩
        :param X: 输入Token矩阵 (T, N, D)
        :return:
            refined_tokens: 精炼后的Token矩阵 (M, D)
            frame_counts: 每帧保留的Token数量字典
            kept_ids: 保留的Token在全局序列中的id列表（已考虑文本Token插入的偏移）
        """
        N_total, D = X.shape
        N = N_total//T
        X = X.resize(T, N, D)
        M = max(1, int(self.p * N_total))  # 目标保留数量

        if M >= N_total:
            # 无需压缩
            frame_counts = {t: N for t in range(T)}
            # 计算包含文本Token偏移的id
            kept_ids = []
            for t in range(T):
                for s in range(N):
                    # 原始id加上偏移量：第t帧的Token需要加t+1
                    original_id = t * N + s
                    shifted_id = original_id + (t + 1)
                    kept_ids.append(shifted_id)
            return X.reshape(N_total, D), frame_counts, kept_ids

        # 1. 构建相似度矩阵
        S_intra, S_inter = self.build_similarity_matrices(X)

        # 2. 构建邻接矩阵
        A_intra, A_inter = self.build_adjacency_matrices(S_intra, S_inter)

        # 3. 计算重要性分数
        scores = self.compute_importance_scores(X, S_inter, A_intra, A_inter)
        scores_flat = scores.reshape(-1)  # 展平为 (T*N,)

        # 4. 初始化Token集合和特征
        current_tokens = X.reshape(N_total, D).clone()  # 展平特征
        current_indices = set(range(N_total))  # 当前保留的Token索引（原始id）
        current_scores = scores_flat.clone()

        # 5. 迭代合并Token
        while len(current_indices) > M:
            # 找到当前分数最低的Token
            valid_indices = list(current_indices)
            valid_scores = current_scores[valid_indices]
            min_idx = valid_indices[torch.argmin(valid_scores)]

            # 解析帧和空间索引
            t = min_idx // N
            s = min_idx % N

            # 寻找有效邻居（同帧内+相邻帧）
            neighbors = set()

            # 帧内邻居
            intra_neighbors = torch.where(A_intra[t, s] == 1)[0].tolist()
            intra_neighbors = [t * N + s_idx for s_idx in intra_neighbors]
            neighbors.update(intra_neighbors)

            # 帧间邻居（前一帧）
            if t > 0:
                inter_prev_neighbors = torch.where(A_inter[t - 1, :, s] == 1)[0].tolist()
                inter_prev_neighbors = [(t - 1) * N + s_prev for s_prev in inter_prev_neighbors]
                neighbors.update(inter_prev_neighbors)

            # 帧间邻居（后一帧）
            if t < T - 1 and A_inter.numel() > 0:
                inter_next_neighbors = torch.where(A_inter[t, s, :] == 1)[0].tolist()
                inter_next_neighbors = [(t + 1) * N + s_next for s_next in inter_next_neighbors]
                neighbors.update(inter_next_neighbors)

            # 筛选有效邻居（仍在当前集合中）
            valid_neighbors = [n for n in neighbors if n in current_indices and n != min_idx]
            if not valid_neighbors:
                current_indices.remove(min_idx)
                continue

            # 找到最相似的邻居
            sim_values = []
            for n in valid_neighbors:
                tn = n // N
                sn = n % N
                if tn == t:
                    # 帧内相似度
                    sim = S_intra[t, s, sn]
                elif tn == t - 1:
                    # 前一帧帧间相似度
                    sim = S_inter[t - 1, sn, s]
                else:  # tn == t + 1
                    # 后一帧帧间相似度
                    sim = S_inter[t, s, sn]
                sim_values.append(sim)

            best_neighbor = valid_neighbors[torch.argmax(torch.tensor(sim_values))]

            # 合并特征
            sim = max(sim_values)
            # current_tokens[best_neighbor] = ((1-sim) * current_tokens[min_idx] + current_tokens[best_neighbor]) / (1-sim + 1)
            current_tokens[best_neighbor] = (1-sim) * current_tokens[min_idx] + current_tokens[best_neighbor]
            # 合并分数
            current_scores[best_neighbor] += current_scores[min_idx]

            # 移除被合并的Token
            current_indices.remove(min_idx)

        return current_tokens, list(current_indices)

if __name__ == "__main__":
    T, N, D = 4, 16, 768
    X = torch.randn(T, N, D)

    refiner = STGTokenRefiner(tau=0.3, alpha=0.6, beta=0.4, lambda_coeff=0.5, p=0.5)

    refined_tokens, frame_counts, kept_ids = refiner.refine_tokens(X)

    print(f"原始Token数量: {T * N}")
    print(f"精炼后Token数量: {refined_tokens.shape[0]}")
    print("每帧保留的Token数量:", frame_counts)
    print("保留的Token全局id:", kept_ids)