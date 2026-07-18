import torch
import torch.nn.functional as F
import heapq

class STGTokenRefiner:
    def __init__(self, tau=0.5, alpha=0.0, beta=0.5, lambda_coeff=0.5, p=0.5):
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

    
    def compute_similarity_batch(self, X):
        """批量计算所有相似度矩阵"""
        T, N, D = X.shape
        
        # 批量归一化所有token
        X_norm = F.normalize(X, p=2, dim=2)  # (T, N, D)
        
        # 批量计算帧内相似度: X_norm @ X_norm^T
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))  # (T, N, N)
        
        # 批量计算帧间相似度
        if T > 1:
            # 准备相邻帧对
            X_curr = X_norm[:-1]  # (T-1, N, D)
            X_next = X_norm[1:]   # (T-1, N, D)
            S_inter = torch.bmm(X_curr, X_next.transpose(1, 2))  # (T-1, N, N)
        else:
            S_inter = torch.empty(0, device=X.device)
            
        return S_intra, S_inter

    def compute_temporal_weights_batch(self, S_inter):
        """批量计算时间权重"""
        T = S_inter.shape[0] + 1 if S_inter.numel() > 0 else 1
        N = S_inter.shape[1] if S_inter.numel() > 0 else 0
        
        if T == 1:
            return torch.zeros(1, N, device=S_inter.device)
        
        # 批量提取对角线元素
        diag_indices = torch.arange(N, device=S_inter.device)
        
        # 所有帧的时间变化
        time_changes = 1.0 - S_inter[:, diag_indices, diag_indices]  # (T-1, N)
        
        # 构建时间权重矩阵
        w_t = torch.zeros(T, N, device=S_inter.device)
        
        # 第一帧
        w_t[0] = time_changes[0]
        
        # 中间帧（如果有）
        if T > 2:
            # 批量计算中间帧的平均变化
            w_t[1:-1] = (time_changes[:-1] + time_changes[1:]) / 2
        
        # 最后一帧
        w_t[-1] = time_changes[-1]
        
        # 归一化
        max_val = w_t.max()
        return w_t / max_val if max_val > 0 else w_t

    def build_adjacency_matrices_batch(self, S_intra, S_inter):
        """批量构建邻接矩阵"""
        # 帧内邻接矩阵
        A_intra = (S_intra > self.tau).float()
        
        # 添加自环
        N = A_intra.shape[1]
        diag_mask = torch.eye(N, device=A_intra.device).unsqueeze(0).bool()
        A_intra = A_intra.masked_fill(diag_mask, 1.0)
        
        # 帧间邻接矩阵
        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).float()
        else:
            A_inter = torch.empty(0, device=S_intra.device)
            
        return A_intra, A_inter

    def compute_dependency_batch(self, A_intra, A_inter):
        """批量计算时空依赖度"""
        T, N = A_intra.shape[:2]
        
        # 空间依赖度（批量求和）
        dep_s = A_intra.sum(dim=2)  # (T, N)
        dep_s_norm = dep_s / dep_s.max()
        
        # 时间依赖度
        dep_t = torch.zeros(T, N, device=A_intra.device)
        
        if A_inter.numel() > 0:
            # 前一帧对当前帧的贡献
            if T > 1:
                # A_inter[t-1, :, :] 的列和
                prev_in = A_inter.sum(dim=1)  # (T-1, N)
                dep_t[:-1] += prev_in
            
            # 后一帧对当前帧的贡献
            if T > 1:
                # A_inter[t, :, :] 的行和
                next_in = A_inter.sum(dim=2)  # (T-1, N)
                dep_t[1:] += next_in
        
        dep_t_norm = dep_t / (dep_t.max() + 1e-8)
        
        # 联合依赖度
        dep_st = self.beta * dep_s_norm + (1 - self.beta) * dep_t_norm
        return dep_st

    def compute_importance_scores_batch(self, X, S_inter, A_intra, A_inter):
        """批量计算重要性分数"""
        T, N, D = X.shape
        
        # 时空权重
        w_s = torch.norm(X, p=2, dim=2)  # (T, N)
        w_s = w_s / w_s.max()
        
        w_t = self.compute_temporal_weights_batch(S_inter)
        
        w_st = self.alpha * w_s + (1 - self.alpha) * w_t
        
        # 时空依赖度
        dep_st = self.compute_dependency_batch(A_intra, A_inter)
        
        # 综合分数
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * dep_st
        return scores

    def create_neighbor_masks(self, A_intra, A_inter, N):
        """预计算邻居掩码，避免重复计算"""
        T = A_intra.shape[0]
        device = A_intra.device
        
        # 为每个位置预计算邻居位置
        neighbor_masks = torch.zeros(T * N, T * N, device=device, dtype=torch.bool)
        
        for i in range(T * N):
            t_i, s_i = divmod(i, N.item())
            
            # 帧内邻居
            intra_mask = A_intra[t_i, s_i].bool()
            intra_indices = t_i * N + torch.where(intra_mask)[0]
            neighbor_masks[i, intra_indices] = True
            
            # 前向帧间邻居
            if t_i > 0 and A_inter.numel() > 0:
                inter_prev_mask = A_inter[t_i - 1, :, s_i].bool()
                inter_prev_indices = (t_i - 1) * N + torch.where(inter_prev_mask)[0]
                neighbor_masks[i, inter_prev_indices] = True
            
            # 后向帧间邻居
            if t_i < T - 1 and A_inter.numel() > 0:
                inter_next_mask = A_inter[t_i, s_i, :].bool()
                inter_next_indices = (t_i + 1) * N + torch.where(inter_next_mask)[0]
                neighbor_masks[i, inter_next_indices] = True
        
        return neighbor_masks
    
    
    def refine_tokens_gpu_optimized(self, X, T):
        """
        真正GPU优化的版本：从低分开始删除，删除时合并到相似邻居
        """
        N_total, D = X.shape
        N = N_total // T
        # 重塑为3D
        X_3d = X.view(T, N, D)
        M = max(1, int(self.p * N_total))
        if M >= N_total:
            return X.clone(), torch.arange(N_total, device=X.device)
        # 1. 批量计算相似度矩阵（完全向量化）
        X_norm = F.normalize(X_3d, p=2, dim=2)
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))  # (T, N, N)
        # 批量计算帧间相似度
        if T > 1:
            S_inter = torch.bmm(X_norm[:-1], X_norm[1:].transpose(1, 2))  # (T-1, N, N)
        else:
            S_inter = torch.empty(0, device=X.device)
        # 2. 批量构建邻接矩阵
        A_intra = (S_intra > self.tau).float()
        eye_matrix = torch.eye(N, device=X.device).unsqueeze(0).expand(T, N, N)
        A_intra = torch.where(eye_matrix.bool(), torch.ones_like(A_intra), A_intra)
        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).float()
        else:
            A_inter = torch.empty(0, device=X.device)
        # 3. 批量计算重要性分数
        # 空间权重
        w_s = torch.norm(X_3d, p=2, dim=2)  # (T, N)
        w_s = w_s / (w_s.max() + 1e-8)
        # 时间权重
        if T > 1:
            diag_values = S_inter[:, torch.arange(N, device=X.device), torch.arange(N, device=X.device)]
            w_t = torch.zeros(T, N, device=X.device)
            w_t[0] = 1 - diag_values[0]
            if T > 2:
                w_t[1:-1] = (1 - diag_values[:-1] + 1 - diag_values[1:]) / 2
            w_t[-1] = 1 - diag_values[-1]
            w_t = w_t / (w_t.max() + 1e-8)
        else:
            w_t = torch.zeros(1, N, device=X.device)
        w_st = self.alpha * w_s + (1 - self.alpha) * w_t
        # 空间依赖度
        dep_s = A_intra.sum(dim=2)  # (T, N)
        dep_s = dep_s / (dep_s.max() + 1e-8)
        # 时间依赖度
        dep_t = torch.zeros(T, N, device=X.device)
        if T > 1:
            # 前一帧对当前帧的贡献
            if A_inter.numel() > 0:
                prev_in = A_inter.sum(dim=1)  # (T-1, N)
                dep_t[:-1] += prev_in
            # 后一帧对当前帧的贡献
            if A_inter.numel() > 0:
                next_in = A_inter.sum(dim=2)  # (T-1, N)
                dep_t[1:] += next_in
        dep_t = dep_t / (dep_t.max() + 1e-8)
        dep_st = self.beta * dep_s + (1 - self.beta) * dep_t
        # 综合分数
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * (1-dep_st)
        scores_flat = scores.view(-1)
        # 4. 原始逻辑：从低分开始删除，删除时合并到相似邻居
        # 初始化所有token
        tokens = X.clone()  # 保存当前token特征
        token_scores = scores_flat.clone()  # 保存当前分数
        # 活跃状态掩码
        active_mask = torch.ones(N_total, dtype=torch.bool, device=X.device)
        # 创建相似度查找表（加速邻居查找）
        sim_lookup = torch.zeros(T * N, T * N, device=X.device)
        # 填充相似度查找表（只填充活跃时可能访问的部分）
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧相似度
                sim_lookup[idx, t * N:(t + 1) * N] = S_intra[t, s]
                # 前一帧相似度
                if t > 0 and S_inter.numel() > 0:
                    sim_lookup[idx, (t - 1) * N:t * N] = S_inter[t - 1, :, s]
                # 后一帧相似度
                if t < T - 1 and S_inter.numel() > 0:
                    sim_lookup[idx, (t + 1) * N:(t + 2) * N] = S_inter[t, s, :]
        # 创建邻接矩阵查找表
        adj_lookup = torch.zeros(T * N, T * N, dtype=torch.bool, device=X.device)
        # 填充邻接查找表
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧邻居
                adj_lookup[idx, t * N:(t + 1) * N] = A_intra[t, s].bool()
                # 前一帧邻居
                if t > 0 and A_inter.numel() > 0:
                    adj_lookup[idx, (t - 1) * N:t * N] = A_inter[t - 1, :, s].bool()
                # 后一帧邻居
                if t < T - 1 and A_inter.numel() > 0:
                    adj_lookup[idx, (t + 1) * N:(t + 2) * N] = A_inter[t, s, :].bool()
        # 删除低分token直到达到目标数量M
        while active_mask.sum() > M:
            # 找到当前最低分的活跃token
            # 将不活跃token的分数设为无穷大，这样argmin不会选中它们
            masked_scores = torch.where(
                active_mask,
                token_scores,
                torch.tensor(float('inf'), device=X.device)
            )
            min_idx = torch.argmin(masked_scores).item()
            if not active_mask[min_idx]:
                continue
            # 找到min_idx的活跃邻居
            neighbor_mask = adj_lookup[min_idx] & active_mask
            neighbor_indices = torch.where(neighbor_mask)[0]
            # 排除自己
            neighbor_indices = neighbor_indices[neighbor_indices != min_idx]
            if len(neighbor_indices) == 0:
                # 如果没有邻居，直接丢弃
                active_mask[min_idx] = False
                continue
            # 找到最相似的邻居
            sim_values = sim_lookup[min_idx, neighbor_indices]
            best_sim_idx = torch.argmax(sim_values)
            best_neighbor = neighbor_indices[best_sim_idx]
            best_sim = sim_values[best_sim_idx]
            # 将低分token的特征合并到最相似的邻居
            # 使用加权平均：相似度越高，低分token的贡献越大
            if best_sim > self.tau:
            # if False:
                # 计算合并权重
                alpha = best_sim  # 相似度作为权重因子
                # 合并特征
                tokens[best_neighbor] = (
                    alpha * tokens[min_idx] +
                    (1.0 - alpha) * tokens[best_neighbor]
                )
                # 合并分数
                token_scores[best_neighbor] = (
                    token_scores[best_neighbor] +
                    token_scores[min_idx] * alpha
                )
            else:
                # 如果相似度不高，简单相加
                tokens[best_neighbor] = tokens[min_idx] + tokens[best_neighbor]
                token_scores[best_neighbor] = (
                    token_scores[best_neighbor] +
                    token_scores[min_idx]
                )
            # 标记低分token为不活跃（删除）
            active_mask[min_idx] = False
        # 收集最终结果
        final_indices = torch.where(active_mask)[0]
        return tokens, final_indices
    
    def refine_tokens_gpu_included_depstack(self, X, T, deepstack_feature_lists):
        N_total, D = X.shape
        N = N_total // T
        # 重塑为3D
        X_3d = X.view(T, N, D)
        M = max(1, int(self.p * N_total))
        if M >= N_total:
            # 如果不需要合并，直接返回原始数据
            return X.clone(), torch.arange(N_total, device=X.device), deepstack_feature_lists
        # 1. 批量计算相似度矩阵（完全向量化）
        X_norm = F.normalize(X_3d, p=2, dim=2)
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))  # (T, N, N)
        # 批量计算帧间相似度
        if T > 1:
            S_inter = torch.bmm(X_norm[:-1], X_norm[1:].transpose(1, 2))  # (T-1, N, N)
        else:
            S_inter = torch.empty(0, device=X.device)
        # 2. 批量构建邻接矩阵
        A_intra = (S_intra > self.tau).float()
        eye_matrix = torch.eye(N, device=X.device).unsqueeze(0).expand(T, N, N)
        A_intra = torch.where(eye_matrix.bool(), torch.ones_like(A_intra), A_intra)

        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).float()
        else:
            A_inter = torch.empty(0, device=X.device)
        # 3. 批量计算重要性分数
        # 空间权重
        w_s = torch.norm(X_3d, p=2, dim=2)  # (T, N)
        w_s = w_s / (w_s.max() + 1e-8)
        # 时间权重
        if T > 1:
            diag_values = S_inter[:, torch.arange(N, device=X.device), torch.arange(N, device=X.device)]
            w_t = torch.zeros(T, N, device=X.device)
            w_t[0] = 1 - diag_values[0]
            if T > 2:
                w_t[1:-1] = (1 - diag_values[:-1] + 1 - diag_values[1:]) / 2
            w_t[-1] = 1 - diag_values[-1]
            w_t = w_t / (w_t.max() + 1e-8)
        else:
            w_t = torch.zeros(1, N, device=X.device)
        w_st = self.alpha * w_s + (1 - self.alpha) * w_t
        # 空间依赖度
        dep_s = A_intra.sum(dim=2)  # (T, N)
        dep_s = dep_s / (dep_s.max() + 1e-8)
        # 时间依赖度
        dep_t = torch.zeros(T, N, device=X.device)
        if T > 1:
            # 前一帧对当前帧的贡献
            if A_inter.numel() > 0:
                prev_in = A_inter.sum(dim=1)  # (T-1, N)
                dep_t[:-1] += prev_in
            # 后一帧对当前帧的贡献
            if A_inter.numel() > 0:
                next_in = A_inter.sum(dim=2)  # (T-1, N)
                dep_t[1:] += next_in
        dep_t = dep_t / (dep_t.max() + 1e-8)
        dep_st = self.beta * dep_s + (1 - self.beta) * dep_t
        # 综合分数
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * dep_st
        scores_flat = scores.view(-1)
        # 4. 原始逻辑：从低分开始删除，删除时合并到相似邻居
        # 初始化所有token
        tokens = X.clone()  # 保存当前token特征
        token_scores = scores_flat.clone()  # 保存当前分数
        # 对每个VIT层的特征进行同样的初始化
        deepstack_tokens_list = []
        for feat in deepstack_feature_lists:
            # 确保特征维度与X一致
            if feat.shape[0] == N_total:
                deepstack_tokens_list.append(feat.clone())
            else:
                # 如果维度不匹配，使用适当的reshape或处理
                # 这里假设特征可以reshape到(T, N, D')的形式
                feat_D = feat.shape[1]
                feat_3d = feat.view(T, N, feat_D)
                deepstack_tokens_list.append(feat_3d.clone())
        # 活跃状态掩码
        active_mask = torch.ones(N_total, dtype=torch.bool, device=X.device)
        # 创建相似度查找表（加速邻居查找）
        sim_lookup = torch.zeros(T * N, T * N, device=X.device)
        # 填充相似度查找表（只填充活跃时可能访问的部分）
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧相似度
                sim_lookup[idx, t * N:(t + 1) * N] = S_intra[t, s]
                # 前一帧相似度
                if t > 0 and S_inter.numel() > 0:
                    sim_lookup[idx, (t - 1) * N:t * N] = S_inter[t - 1, :, s]
                # 后一帧相似度
                if t < T - 1 and S_inter.numel() > 0:
                    sim_lookup[idx, (t + 1) * N:(t + 2) * N] = S_inter[t, s, :]
        # 创建邻接矩阵查找表
        adj_lookup = torch.zeros(T * N, T * N, dtype=torch.bool, device=X.device)
        # 填充邻接查找表
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧邻居
                adj_lookup[idx, t * N:(t + 1) * N] = A_intra[t, s].bool()
                # 前一帧邻居
                if t > 0 and A_inter.numel() > 0:
                    adj_lookup[idx, (t - 1) * N:t * N] = A_inter[t - 1, :, s].bool()
                # 后一帧邻居
                if t < T - 1 and A_inter.numel() > 0:
                    adj_lookup[idx, (t + 1) * N:(t + 2) * N] = A_inter[t, s, :].bool()
        # 删除低分token直到达到目标数量M
        while active_mask.sum() > M:
            # 找到当前最低分的活跃token
            # 将不活跃token的分数设为无穷大，这样argmin不会选中它们
            masked_scores = torch.where(
                active_mask,
                token_scores,
                torch.tensor(float('inf'), device=X.device)
            )
            min_idx = torch.argmin(masked_scores).item()
            if not active_mask[min_idx]:
                continue
            # 找到min_idx的活跃邻居
            neighbor_mask = adj_lookup[min_idx] & active_mask
            neighbor_indices = torch.where(neighbor_mask)[0]
            # 排除自己
            neighbor_indices = neighbor_indices[neighbor_indices != min_idx]
            if len(neighbor_indices) == 0:
                # 如果没有邻居，直接删除
                active_mask[min_idx] = False
                continue
            # 找到最相似的邻居
            sim_values = sim_lookup[min_idx, neighbor_indices]
            best_sim_idx = torch.argmax(sim_values)
            best_neighbor = neighbor_indices[best_sim_idx]
            best_sim = sim_values[best_sim_idx]
            # 将低分token的特征合并到最相似的邻居
            # 使用加权平均：相似度越高，低分token的贡献越大
            if best_sim > self.tau:
                # 计算合并权重
                alpha = best_sim  # 相似度作为权重因子

                # 合并主特征
                tokens[best_neighbor] = (
                    alpha * tokens[min_idx] +
                    (1.0 - alpha) * tokens[best_neighbor]
                )

                # 合并所有VIT层的特征
                for i in range(len(deepstack_tokens_list)):
                    # 根据特征的形状决定如何访问
                    if len(deepstack_tokens_list[i].shape) == 2:
                        # 2D形状: (N_total, D')
                        deepstack_tokens_list[i][best_neighbor] = (
                            alpha * deepstack_tokens_list[i][min_idx] +
                            (1.0 - alpha) * deepstack_tokens_list[i][best_neighbor]
                        )
                    else:
                        # 3D形状: (T, N, D')
                        t_idx = min_idx // N
                        s_idx = min_idx % N
                        t_neighbor = best_neighbor // N
                        s_neighbor = best_neighbor % N
                        
                        deepstack_tokens_list[i][t_neighbor, s_neighbor] = (
                            alpha * deepstack_tokens_list[i][t_idx, s_idx] +
                            (1.0 - alpha) * deepstack_tokens_list[i][t_neighbor, s_neighbor]
                        )

                # 合并分数
                token_scores[best_neighbor] = (
                    token_scores[best_neighbor] +
                    token_scores[min_idx] * alpha
                )
            else:
                # 如果相似度不高，简单相加
                tokens[best_neighbor] = tokens[min_idx] + tokens[best_neighbor]
                
                # 合并所有VIT层的特征
                for i in range(len(deepstack_tokens_list)):
                    if len(deepstack_tokens_list[i].shape) == 2:
                        deepstack_tokens_list[i][best_neighbor] += deepstack_tokens_list[i][min_idx]
                    else:
                        t_idx = min_idx // N
                        s_idx = min_idx % N
                        t_neighbor = best_neighbor // N
                        s_neighbor = best_neighbor % N
                        deepstack_tokens_list[i][t_neighbor, s_neighbor] += deepstack_tokens_list[i][t_idx, s_idx]

                token_scores[best_neighbor] = (
                    token_scores[best_neighbor] +
                    token_scores[min_idx]
                )
            # 标记低分token为不活跃（删除）
            active_mask[min_idx] = False
        # 收集最终结果
        final_indices = torch.where(active_mask)[0]
        return tokens, final_indices, deepstack_tokens_list
    
    # def compute_spatial_weights_fully_vectorized(self, S_intra, H, W):
        T = S_intra.shape[0]
        w_s = torch.zeros((T, H, W), dtype=S_intra.dtype, device=S_intra.device)
        # 1. 计算内部区域（8-邻域）
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                # 上一行
                adj = S_intra[:, h * W + w, (h - 1) * W + w - 1:(h - 1) * W + w + 2].sum(dim=1)
                # 当前行
                adj += S_intra[:, h * W + w, h * W + w - 1:h * W + w + 2:2].sum(dim=1)  # 跳过中间自己
                # 下一行
                adj += S_intra[:, h * W + w, (h + 1) * W + w - 1:(h + 1) * W + w + 2].sum(dim=1)
                w_s[:, h, w] = adj
        # 2. 处理四个角点（3-邻域）
        # 左上角 (0, 0)
        if H > 1 and W > 1:
            # 右、下、右下
            adj = S_intra[:, 0, 1] + S_intra[:, 0, W] + S_intra[:, 0, W + 1]
            w_s[:, 0, 0] = adj
        # 右上角 (0, W-1)
        if H > 1 and W > 1:
            # 左、下、左下
            adj = S_intra[:, W - 1, W - 2] + S_intra[:, W - 1, 2 * W - 1] + S_intra[:, W - 1,2 * W - 2]
            w_s[:, 0, W - 1] = adj
        # 左下角 (H-1, 0)
        if H > 1 and W > 1:
            # 右、上、右上
            adj = S_intra[:, (H - 1) * W, (H - 1) * W + 1] + S_intra[:, (H - 1) * W,(H - 2) * W] + S_intra[:, (H - 1) * W, (H - 2) * W + 1]
            w_s[:, H - 1, 0] = adj
        # 右下角 (H-1, W-1)
        if H > 1 and W > 1:
            # 左、上、左上
            adj = S_intra[:, H * W - 1, H * W - 2] + S_intra[:, H * W - 1, (H - 1) * W - 1] + S_intra[:, H * W - 1, (H - 1) * W - 2]
            w_s[:, H - 1, W - 1] = adj
        # 3. 处理上边缘（排除角点，5-邻域）
        if H > 1 and W > 2:
            for w in range(1, W - 1):
                # 左、右、下、左下、右下
                adj = (S_intra[:, w, w - 1] + S_intra[:, w, w + 1] +
                       S_intra[:, w, W + w - 1] + S_intra[:, w, W + w] +
                       S_intra[:, w, W + w + 1])
                w_s[:, 0, w] = adj
        # 4. 处理下边缘（排除角点，5-邻域）
        if H > 1 and W > 2:
            for w in range(1, W - 1):
                h = H - 1
                # 左、右、上、左上、右上
                adj = (S_intra[:, h * W + w, h * W + w - 1] + S_intra[:, h * W + w, h * W + w + 1] +
                       S_intra[:, h * W + w, (h - 1) * W + w - 1] + S_intra[:, h * W + w, (h - 1) * W + w] +
                       S_intra[:, h * W + w, (h - 1) * W + w + 1])
                w_s[:, H - 1, w] = adj
        # 5. 处理左边缘（排除角点，5-邻域）
        if H > 2 and W > 1:
            for h in range(1, H - 1):
                # 上、下、右、右上、右下
                adj = (S_intra[:, h * W, (h - 1) * W] + S_intra[:, h * W, (h + 1) * W] +
                       S_intra[:, h * W, h * W + 1] + S_intra[:, h * W, (h - 1) * W + 1] +
                       S_intra[:, h * W, (h + 1) * W + 1])
                w_s[:, h, 0] = adj
        # 6. 处理右边缘（排除角点，5-邻域）
        if H > 2 and W > 1:
            for h in range(1, H - 1):
                w = W - 1
                # 上、下、左、左上、左下
                adj = (S_intra[:, h * W + w, (h - 1) * W + w] + S_intra[:, h * W + w, (h + 1) * W + w] +
                       S_intra[:, h * W + w, h * W + w - 1] + S_intra[:, h * W + w, (h - 1) * W + w - 1] +
                       S_intra[:, h * W + w, (h + 1) * W + w - 1])
                w_s[:, h, W - 1] = adj
        # 归一化
        w_s = w_s.view(T, H*W) / (w_s.max() + 1e-8)
        return w_s
    
    # def refine_tokens_gpu_optimized2(self, X, grid_thw):
        """
        GPU优化的版本：将所有低于目标百分数的Token一次性合并到高分Token
        """
        T = grid_thw[0]
        H = grid_thw[1] // 2
        W = grid_thw[2] // 2
        N_total, D = X.shape
        N = N_total // T
        M = max(1, int(self.p * N_total))
        
        if M >= N_total:
            return X.clone(), torch.arange(N_total, device=X.device)
        
        # 重塑为3D
        X_3d = X.view(T, N, D)
        
        # 1. 批量计算相似度矩阵（完全向量化）
        X_norm = F.normalize(X_3d, p=2, dim=2)
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))  # (T, N, N)
        
        # 批量计算帧间相似度
        if T > 1:
            S_inter = torch.bmm(X_norm[:-1], X_norm[1:].transpose(1, 2))  # (T-1, N, N)
        else:
            S_inter = torch.empty(0, device=X.device)
        
        # 2. 批量构建邻接矩阵
        A_intra = (S_intra > self.tau).float()
        eye_matrix = torch.eye(N, device=X.device).unsqueeze(0).expand(T, N, N)
        A_intra = torch.where(eye_matrix.bool(), torch.ones_like(A_intra), A_intra)
        
        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).float()
        else:
            A_inter = torch.empty(0, device=X.device)

        # 将相似度矩阵重塑为 (T, H, W, H, W)
        S_intra_reshaped = S_intra.view(T, H, W, H, W)
        
        # 为每个位置创建偏移量掩码（用于8邻域）
        neighbor_mask = torch.zeros(H, W, H, W, device=X.device, dtype=torch.bool)
        for h in range(H):
            for w in range(W):
                # 计算当前位置的8邻域
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        if dh == 0 and dw == 0:
                            continue  # 跳过自己
                        nh, nw = h + dh, w + dw
                        if 0 <= nh < H and 0 <= nw < W:
                            neighbor_mask[h, w, nh, nw] = True
        
        # 扩展掩码到batch维度
        neighbor_mask_expanded = neighbor_mask.unsqueeze(0).expand(T, -1, -1, -1, -1)
        
        # 应用掩码获取邻居相似度
        neighbor_sims_masked = torch.where(
            neighbor_mask_expanded,
            S_intra_reshaped,
            torch.tensor(0.0, device=X.device)
        )
        
        # 计算每个位置的邻居数量
        neighbor_counts = neighbor_mask.sum(dim=(2, 3)).unsqueeze(0).expand(T, -1, -1)
        
        # 计算相似度总和
        neighbor_sums = neighbor_sims_masked.sum(dim=(3, 4))
        
        # 计算平均值（避免除0）
        w_s_2d = torch.where(
            neighbor_counts > 0,
            neighbor_sums / neighbor_counts,
            torch.tensor(0.0, device=X.device)
        )
        
        # 重新展平为 (T, N)
        w_s = w_s_2d.view(T, N)
        w_s = w_s / (w_s.max() + 1e-8)

        # 时间权重
        if T > 1:
            diag_values = S_inter[:, torch.arange(N, device=X.device), torch.arange(N, device=X.device)]
            w_t = torch.zeros(T, N, device=X.device)
            w_t[0] = 1 - diag_values[0]
            if T > 2:
                w_t[1:-1] = (1 - diag_values[:-1] + 1 - diag_values[1:]) / 2
            w_t[-1] = 1 - diag_values[-1]
            w_t = w_t / (w_t.max() + 1e-8)
        else:
            w_t = torch.zeros(1, N, device=X.device)

        w_st = self.alpha * (1 - w_s) + (1 - self.alpha) * w_t

        # 空间依赖度
        dep_s = A_intra.sum(dim=2)  # (T, N)
        dep_s = dep_s / (dep_s.max() + 1e-8)

        # 时间依赖度
        dep_t = torch.zeros(T, N, device=X.device)
        if T > 1:
            # 前一帧对当前帧的贡献
            if A_inter.numel() > 0:
                prev_in = A_inter.sum(dim=1)  # (T-1, N)
                dep_t[:-1] += prev_in

            # 后一帧对当前帧的贡献
            if A_inter.numel() > 0:
                next_in = A_inter.sum(dim=2)  # (T-1, N)
                dep_t[1:] += next_in

        dep_t = dep_t / (dep_t.max() + 1e-8)
        dep_st = self.beta * dep_s + (1 - self.beta) * dep_t

        # 综合分数
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * (1 - dep_st)
        scores_flat = scores.view(-1)
        
        # 4. 一次性合并所有低分token到高分token
        # 克隆原始token和分数
        tokens = X.clone()
        token_scores = scores_flat.clone()
        
        # 根据分数排序，找出高分token（保留前M个）
        sorted_scores, sorted_indices = torch.sort(scores_flat, descending=True)
        
        # 高分token（要保留的）
        high_score_indices = sorted_indices[:M]  # 保留前M个高分token
        low_score_indices = sorted_indices[M:]   # 要合并的低分token
        
        # 如果低分token数量为0，直接返回
        if len(low_score_indices) == 0:
            return tokens, high_score_indices
        
        # 创建活跃状态掩码
        active_mask = torch.ones(N_total, dtype=torch.bool, device=X.device)
        active_mask[low_score_indices] = False  # 低分token初始为不活跃
        
        # 创建相似度查找表（只计算需要合并的部分）
        # 我们可以只计算低分token和高分token之间的相似度
        sim_lookup = torch.zeros(N_total, N_total, device=X.device)
        
        # 填充相似度查找表
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧相似度
                sim_lookup[idx, t * N:(t + 1) * N] = S_intra[t, s]
                
                # 前一帧相似度
                if t > 0 and S_inter.numel() > 0:
                    sim_lookup[idx, (t - 1) * N:t * N] = S_inter[t - 1, :, s]
                
                # 后一帧相似度
                if t < T - 1 and S_inter.numel() > 0:
                    sim_lookup[idx, (t + 1) * N:(t + 2) * N] = S_inter[t, s, :]
        
        # 创建邻接矩阵查找表
        adj_lookup = torch.zeros(N_total, N_total, dtype=torch.bool, device=X.device)
        
        # 填充邻接查找表
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧邻居
                adj_lookup[idx, t * N:(t + 1) * N] = A_intra[t, s].bool()
                
                # 前一帧邻居
                if t > 0 and A_inter.numel() > 0:
                    adj_lookup[idx, (t - 1) * N:t * N] = A_inter[t - 1, :, s].bool()
                
                # 后一帧邻居
                if t < T - 1 and A_inter.numel() > 0:
                    adj_lookup[idx, (t + 1) * N:(t + 2) * N] = A_inter[t, s, :].bool()
        
        # 批量处理低分token：为每个低分token找到最相似的高分邻居并合并
        for low_idx in low_score_indices:
            # 找到低分token的活跃邻居（只在高分token中寻找）
            # 获取所有邻居
            all_neighbors = torch.where(adj_lookup[low_idx])[0]
            
            # 只保留活跃的高分token邻居
            high_score_neighbors = [n for n in all_neighbors if active_mask[n].item()]
            
            if len(high_score_neighbors) == 0:
                # 如果没有活跃邻居，直接丢弃
                continue
            
            # 找到最相似的高分邻居
            high_score_neighbors_tensor = torch.tensor(high_score_neighbors, device=X.device)
            sim_values = sim_lookup[low_idx, high_score_neighbors_tensor]
            best_sim_idx = torch.argmax(sim_values)
            best_neighbor = high_score_neighbors_tensor[best_sim_idx]
            best_sim = sim_values[best_sim_idx]
            
            # 将低分token的特征合并到最相似的高分邻居
            # if best_sim > self.tau:
                # 使用相似度作为权重因子进行加权平均
            alpha = best_sim
            
            # 合并特征
            tokens[best_neighbor] = (
                alpha * tokens[low_idx] +
                (1.0 - alpha) * tokens[best_neighbor]
            )
        
        # 收集最终结果（所有高分token）
        final_indices = high_score_indices
        
        
        return tokens, final_indices

    def refine_tokens_gpu_optimized2(self, X, grid_thw):
        """
        GPU优化的版本：从低分开始删除，删除时合并到相似邻居
        """
        T=grid_thw[0]
        H=grid_thw[1]//2
        W=grid_thw[2]//2
        N_total, D = X.shape
        N = N_total // T
        M = max(1, int(self.p * N_total))
        if M >= N_total:
            return X.clone(), torch.arange(N_total, device=X.device)
        # 重塑为3D
        X_3d = X.view(T, N, D)
        # 1. 批量计算相似度矩阵（完全向量化）
        X_norm = F.normalize(X_3d, p=2, dim=2)
        S_intra = torch.bmm(X_norm, X_norm.transpose(1, 2))  # (T, N, N)
        # 批量计算帧间相似度
        if T > 1:
            S_inter = torch.bmm(X_norm[:-1], X_norm[1:].transpose(1, 2))  # (T-1, N, N)
        else:
            S_inter = torch.empty(0, device=X.device)
        # 2. 批量构建邻接矩阵
        A_intra = (S_intra > self.tau).float()
        eye_matrix = torch.eye(N, device=X.device).unsqueeze(0).expand(T, N, N)
        A_intra = torch.where(eye_matrix.bool(), torch.ones_like(A_intra), A_intra)
        if S_inter.numel() > 0:
            A_inter = (S_inter > self.tau).float()
        else:
            A_inter = torch.empty(0, device=X.device)
        # 将相似度矩阵重塑为 (T, H, W, H, W)
        S_intra_reshaped = S_intra.view(T, H, W, H, W)
        # 为每个位置创建偏移量掩码（用于8邻域）
        neighbor_mask = torch.zeros(H, W, H, W, device=X.device, dtype=torch.bool)
        for h in range(H):
            for w in range(W):
                # 计算当前位置的8邻域
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        if dh == 0 and dw == 0:
                            continue  # 跳过自己
                        nh, nw = h + dh, w + dw
                        if 0 <= nh < H and 0 <= nw < W:
                            neighbor_mask[h, w, nh, nw] = True
        # 扩展掩码到batch维度
        neighbor_mask_expanded = neighbor_mask.unsqueeze(0).expand(T, -1, -1, -1, -1)
        # 应用掩码获取邻居相似度
        neighbor_sims_masked = torch.where(
            neighbor_mask_expanded,
            S_intra_reshaped,
            torch.tensor(0.0, device=X.device)
        )
        # 计算每个位置的邻居数量
        neighbor_counts = neighbor_mask.sum(dim=(2, 3)).unsqueeze(0).expand(T, -1, -1)
        # 计算相似度总和
        neighbor_sums = neighbor_sims_masked.sum(dim=(3, 4))
        # 计算平均值（避免除0）
        w_s_2d = torch.where(
            neighbor_counts > 0,
            neighbor_sums / neighbor_counts,
            torch.tensor(0.0, device=X.device)
        )
        # 重新展平为 (T, N)
        w_s = w_s_2d.view(T, N)
        # 归一化
        w_s = w_s / (w_s.max() + 1e-8)
        # w_s =self.compute_spatial_weights_fully_vectorized(S_intra, H, W)
        # 时间权重（保持原逻辑不变）
        if T > 1:
            diag_values = S_inter[:, torch.arange(N, device=X.device), torch.arange(N, device=X.device)]
            w_t = torch.zeros(T, N, device=X.device)
            w_t[0] = 1 - diag_values[0]
            if T > 2:
                w_t[1:-1] = (1 - diag_values[:-1] + 1 - diag_values[1:]) / 2
            w_t[-1] = 1 - diag_values[-1]
            w_t = w_t / (w_t.max() + 1e-8)
        else:
            w_t = torch.zeros(1, N, device=X.device)
        w_st = self.alpha * (1-w_s) + (1 - self.alpha) * w_t
        # 空间依赖度（保持原逻辑不变）
        dep_s = A_intra.sum(dim=2)  # (T, N)
        dep_s = dep_s / (dep_s.max() + 1e-8)
        # 时间依赖度（保持原逻辑不变）
        dep_t = torch.zeros(T, N, device=X.device)
        if T > 1:
            # 前一帧对当前帧的贡献
            if A_inter.numel() > 0:
                prev_in = A_inter.sum(dim=1)  # (T-1, N)
                dep_t[:-1] += prev_in
            # 后一帧对当前帧的贡献
            if A_inter.numel() > 0:
                next_in = A_inter.sum(dim=2)  # (T-1, N)
                dep_t[1:] += next_in
        dep_t = dep_t / (dep_t.max() + 1e-8)
        dep_st = self.beta * dep_s + (1 - self.beta) * dep_t
        # 综合分数（保持原逻辑不变）
        scores = self.lambda_coeff * w_st + (1 - self.lambda_coeff) * (1-dep_st)
        scores_flat = scores.view(-1)
        # 4. 原始逻辑：从低分开始删除，删除时合并到相似邻居
        # 初始化所有token
        tokens = X.clone()  # 保存当前token特征
        token_scores = scores_flat.clone()  # 保存当前分数
        # represented token number: n_i represents how many original tokens are aggregated into token i
        represented_token_num = torch.ones(N_total, dtype=torch.float32, device=X.device)
        # 活跃状态掩码
        active_mask = torch.ones(N_total, dtype=torch.bool, device=X.device)
        # 创建相似度查找表（加速邻居查找）
        sim_lookup = torch.zeros(T * N, T * N, device=X.device)
        # 填充相似度查找表（只填充活跃时可能访问的部分）
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧相似度
                sim_lookup[idx, t * N:(t + 1) * N] = S_intra[t, s]

                # 前一帧相似度
                if t > 0 and S_inter.numel() > 0:
                    sim_lookup[idx, (t - 1) * N:t * N] = S_inter[t - 1, :, s]

                # 后一帧相似度
                if t < T - 1 and S_inter.numel() > 0:
                    sim_lookup[idx, (t + 1) * N:(t + 2) * N] = S_inter[t, s, :]
        # 创建邻接矩阵查找表
        adj_lookup = torch.zeros(T * N, T * N, dtype=torch.bool, device=X.device)
        # 填充邻接查找表
        for t in range(T):
            for s in range(N):
                idx = t * N + s
                # 同帧邻居
                adj_lookup[idx, t * N:(t + 1) * N] = A_intra[t, s].bool()
                # 前一帧邻居
                if t > 0 and A_inter.numel() > 0:
                    adj_lookup[idx, (t - 1) * N:t * N] = A_inter[t - 1, :, s].bool()
                # 后一帧邻居
                if t < T - 1 and A_inter.numel() > 0:
                    adj_lookup[idx, (t + 1) * N:(t + 2) * N] = A_inter[t, s, :].bool()
        # 删除低分token直到达到目标数量M
        while active_mask.sum() > M:
            # 找到当前最低分的活跃token
            # 将不活跃token的分数设为无穷大，这样argmin不会选中它们
            masked_scores = torch.where(
                active_mask,
                token_scores,
                torch.tensor(float('inf'), device=X.device)
            )
            min_idx = torch.argmin(masked_scores).item()
            if not active_mask[min_idx]:
                continue
            # 找到min_idx的活跃邻居
            neighbor_mask = adj_lookup[min_idx] & active_mask
            neighbor_indices = torch.where(neighbor_mask)[0]
            # 排除自己
            neighbor_indices = neighbor_indices[neighbor_indices != min_idx]

            if len(neighbor_indices) == 0:
                # 如果没有邻居，直接丢弃
                active_mask[min_idx] = False
                continue
            # 找到最相似的邻居
            sim_values = sim_lookup[min_idx, neighbor_indices]
            best_sim_idx = torch.argmax(sim_values)
            best_neighbor = neighbor_indices[best_sim_idx]
            best_sim = sim_values[best_sim_idx]
            # 将低分token的特征合并到最相似的邻居
            # 使用加权平均：相似度越高，低分token的贡献越大
            # if best_sim > self.tau:
            #     # 计算合并权重
            #     alpha = best_sim  # 相似度作为权重因子
            #     # 合并特征
            #     tokens[best_neighbor] = (alpha * tokens[min_idx] + (1.0 - alpha) * tokens[best_neighbor])
            #     # 合并分数
            #     token_scores[best_neighbor] = (token_scores[best_neighbor] + token_scores[min_idx] * alpha)
            # else:
            #     # 如果相似度不高，简单相加
            #     tokens[best_neighbor] = tokens[min_idx] + tokens[best_neighbor]
            #     token_scores[best_neighbor] = (token_scores[best_neighbor] + token_scores[min_idx])
            # 标记低分token为不活跃（删除）
            # current represented token numbers
            n_i = represented_token_num[min_idx]
            n_j = represented_token_num[best_neighbor]
            # similarity weighted aggregation
            merge_weight = best_sim
            # -------- Feature Merge --------
            tokens[best_neighbor] = (merge_weight * n_i * tokens[min_idx] + n_j * tokens[best_neighbor] ) / ( merge_weight * n_i+n_j)
            # -------- Importance Score Merge --------
            token_scores[best_neighbor] = (n_j * token_scores[best_neighbor]+n_i * token_scores[min_idx]) / (n_j + n_i)
            # -------- Update represented token number --------
            represented_token_num[best_neighbor] = n_i + n_j
            active_mask[min_idx] = False
        # 收集最终结果
        final_indices = torch.where(active_mask)[0]
        return tokens, final_indices