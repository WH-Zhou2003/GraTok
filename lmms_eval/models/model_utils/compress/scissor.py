import torch
import einops
import numpy as np
from typing import Tuple, List



class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int32)
    
    def find(self, x):
        # 迭代式路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x
    
    def batch_union(self, x_arr, y_arr):
        # 批量合并优化
        for x, y in zip(x_arr, y_arr):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                continue
            
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1


# Semantic Connected Components Module
def approximate_components(adj_matrix, epsilon=0.05):
    n = adj_matrix.shape[0]  # 节点总数
    all_nodes = np.ones(n)  # 标记所有节点为"未处理"
    all_indices = np.arange(0, n)
    if n == 0:
        return []

    sample_size = min(n, int(np.ceil(np.log(n) / epsilon**2)))
    sampled_nodes = np.random.choice(n, size=sample_size, replace=False)
    all_nodes[sampled_nodes] = 0
    
    # 创建稀疏邻接表  # 记录采样节点的邻居
    neighbor_dict = defaultdict(list)
    for i in sampled_nodes:
        neighbors = np.nonzero(adj_matrix[i])[0]  # 找到所有相邻节点
        valid_neighbors = np.intersect1d(neighbors, all_indices, assume_unique=True)
        neighbor_dict[i] = valid_neighbors
        all_nodes[neighbors] = 0  # 标记邻居为"已处理"
    
    remain_nodes = np.nonzero(all_nodes)[0] # 找到未被采样的孤立节点
    remain_nodes = [[element] for element in remain_nodes] # 每个作为独立分量
    # 批量合并优化
    uf = UnionFind(n) # 初始化并查集
    all_x, all_y = [], []
    for i in sampled_nodes:
        for j in neighbor_dict[i]:
            all_x.append(i)
            all_y.append(j)
    uf.batch_union(np.array(all_x), np.array(all_y)) # 批量合并
    
    sampled_roots = np.array([uf.find(i) for i in sampled_nodes])
    unique_roots, counts = np.unique(sampled_roots, return_counts=True)
    
    components = []
    for root in unique_roots:
        mask = (sampled_roots == root)
        cluster = np.where(uf.parent == root)[0].tolist()
        if len(cluster) > 0:
            components.append(cluster)
    components.extend(remain_nodes) # 添加孤立节点

    degrees = np.count_nonzero(adj_matrix, axis=1)  # 预计算度数
    
    def get_sort_key(cluster):
        max_degree = -1
        min_node = float('inf')
        for node in cluster:
            current_degree = degrees[node]
            if (current_degree > max_degree) or \
               (current_degree == max_degree and node < min_node):
                max_degree = current_degree
                min_node = node
        return min_node
    
    components.sort(key=get_sort_key)

    return components


def compress_video_features(
    video_features: torch.Tensor,
    T: int,
    H: int,
    W: int,
    tau: float = 0.95,
    epsilon: float = 0.05,
    second_zip: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    """
    对视频帧特征进行空间+时间维度的相似Token压缩，返回合并后的展平特征和保留Token的ID
    
    Args:
        video_features: 视频帧特征，形状为 (T, H*W, C) 或 (T*H*W, C)
        T: 视频帧数
        H: 每帧特征的高度（patch数）
        W: 每帧特征的宽度（patch数）
        tau: 相似度阈值，高于该值的Token会被聚类合并
        epsilon: 连通分量计算的误差容忍度
        second_zip: 是否开启时间维度的二次压缩
    
    Returns:
        merged_flatten_features: 合并了相似Token信息的展平特征 (T*H*W, C)，原始Token未被抛弃
        kept_token_ids: 压缩后需要保留的Token的展平ID列表（基于原始展平特征的索引）
    """
    # 1. 预处理：确保输入形状为 (T, H*W, C)
    ori_token_num_per_frame = H * W
    if video_features.ndim == 2:
        # 如果输入是展平的 (T*H*W, C)，先恢复为 (T, H*W, C)
        video_features = einops.rearrange(video_features, '(n hw) c -> n hw c', n=T, hw=ori_token_num_per_frame)
    elif video_features.ndim != 3 or video_features.shape[0] != T or video_features.shape[1] != ori_token_num_per_frame:
        raise ValueError(f"输入特征形状错误！期望 (T, H*W, C) 或 (T*H*W, C)，实际 {video_features.shape}")
    
    device = video_features.device
    C = video_features.shape[-1]
    
    # -------------------------- 第一步：空间压缩（单帧内） --------------------------
    # 归一化计算相似度矩阵
    norm_image_feat = torch.norm(video_features, p=2, dim=-1, keepdim=True)  # (T, H*W, 1)
    image_feat_normalized = video_features / norm_image_feat  # (T, H*W, C)
    similarity_matrix = torch.matmul(image_feat_normalized, image_feat_normalized.transpose(1, 2))  # (T, H*W, H*W)
    high_similarity_indices = (similarity_matrix > tau)  # (T, H*W, H*W)
    
    all_fused_feature = []
    all_connected_components = []  # 记录每帧的连通分量（用于后续生成保留ID）
    
    # 逐帧处理空间压缩
    for frame_idx in range(T):
        high_similarity_indices_per_frame = high_similarity_indices[frame_idx].cpu().numpy()
        # 计算连通分量（相似Token聚类）
        connected_components = approximate_components(high_similarity_indices_per_frame, epsilon=epsilon)
        all_connected_components.append(connected_components)
        
        # 对每个聚类取均值，生成压缩后的Token
        image_feat_per_frame = video_features[frame_idx]
        fused_features = []
        for component in connected_components:
            selected_features = image_feat_per_frame[component]
            fused_feature = torch.mean(selected_features, dim=0)
            fused_features.append(fused_feature)
        fused_features = torch.stack(fused_features)
        all_fused_feature.append(fused_features)
    
    # 拼接所有帧的空间压缩结果 (total_spatial_compressed, C)
    spatial_compressed_features = torch.cat(all_fused_feature, dim=0)
    # 生成空间压缩后的Token对应的原始展平ID（每聚类取第一个Token作为代表）
    spatial_kept_ids = []
    for frame_idx, components in enumerate(all_connected_components):
        frame_base_id = frame_idx * ori_token_num_per_frame  # 该帧在展平特征中的起始ID
        for component in components:
            # 取聚类中第一个Token作为保留ID（也可改为取均值/中心等）
            spatial_kept_ids.append(frame_base_id + component[0])
    
    # -------------------------- 第二步：时间压缩（帧间） --------------------------
    if second_zip and len(spatial_compressed_features) > 1:
        # 对空间压缩后的Token做时间维度压缩
        norm_select_tokens = torch.norm(spatial_compressed_features, p=2, dim=1, keepdim=True)
        select_tokens_normalized = spatial_compressed_features / norm_select_tokens
        similarity_matrix_select_tokens = torch.matmul(select_tokens_normalized, select_tokens_normalized.t())
        high_similarity_indices_select_tokens = (similarity_matrix_select_tokens > tau)
        
        # 计算跨帧的连通分量
        connected_component_select_tokens = approximate_components(
            high_similarity_indices_select_tokens.cpu().numpy(), 
            epsilon=epsilon
        )
        
        # 时间压缩后的Token
        fused_features_select_tokens = []
        temporal_kept_spatial_ids = []  # 时间压缩后保留的空间压缩Token的索引
        for component in connected_component_select_tokens:
            selected_features = spatial_compressed_features[component]
            fused_feature = torch.mean(selected_features, dim=0)
            fused_features_select_tokens.append(fused_feature)
            # 取聚类中第一个空间压缩Token作为代表
            temporal_kept_spatial_ids.append(component[0])
        
        # 时间压缩后的最终保留Token
        temporal_compressed_features = torch.stack(fused_features_select_tokens)
        # 映射回原始展平ID
        kept_token_ids = [spatial_kept_ids[idx] for idx in temporal_kept_spatial_ids]
    else:
        # 不开启时间压缩，直接使用空间压缩的结果
        temporal_compressed_features = spatial_compressed_features
        kept_token_ids = spatial_kept_ids
    
    # -------------------------- 第三步：合并原始Token信息（核心需求） --------------------------
    # 展平原始特征 (T*H*W, C)
    flatten_original_features = einops.rearrange(video_features, 'n hw c -> (n hw) c')
    # 归一化计算原始Token与压缩Token的相似度
    norm_selected_tokens = torch.norm(temporal_compressed_features, p=2, dim=1, keepdim=True)
    norm_remaining_values = torch.norm(flatten_original_features, p=2, dim=1, keepdim=True)
    selected_tokens_normalized = temporal_compressed_features / norm_selected_tokens
    remaining_values_normalized = flatten_original_features / norm_remaining_values
    
    # 计算每个原始Token与压缩Token的相似度，找到最匹配的压缩Token
    similarity_matrix = torch.matmul(remaining_values_normalized, selected_tokens_normalized.t())  # (T*H*W, num_kept)
    closest_indices = torch.argmax(similarity_matrix, dim=1)  # (T*H*W,)
    
    # 将原始Token特征累加到匹配的压缩Token上（但保留原始Token不抛弃）
    merged_tokens = torch.zeros_like(temporal_compressed_features)
    merged_tokens.scatter_add_(
        dim=0,
        index=closest_indices.view(-1, 1).expand(-1, merged_tokens.shape[1]),
        src=flatten_original_features
    )
    
    # 计算累加次数（+1是因为要包含压缩Token本身）
    counts = torch.bincount(closest_indices, minlength=len(temporal_compressed_features)).float() + 1
    # 加权平均得到最终的压缩Token
    final_compressed_features = temporal_compressed_features + merged_tokens
    final_compressed_features /= counts.unsqueeze(1)
    
    # -------------------------- 生成合并后的展平特征（原始Token未抛弃） --------------------------
    # 创建与原始展平特征同形状的矩阵，将压缩后的Token信息回填到对应位置
    merged_flatten_features = flatten_original_features.clone()
    for kept_idx, kept_token_id in enumerate(kept_token_ids):
        merged_flatten_features[kept_token_id] = final_compressed_features[kept_idx]
    
    # 确保返回的ID是列表形式，且为整数
    kept_token_ids = [int(id) for id in kept_token_ids]
    
    return merged_flatten_features, kept_token_ids


# -------------------------- 依赖函数（必须包含） --------------------------
def approximate_components(adj_matrix: np.ndarray, epsilon: float = 0.05) -> List[List[int]]:
    """
    基于相似度矩阵计算连通分量（简化实现，与原代码逻辑对齐）
    Args:
        adj_matrix: 相似度邻接矩阵 (N, N)
        epsilon: 误差容忍度
    Returns:
        连通分量列表，每个分量是Token ID的列表
    """
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []
    
    for i in range(n):
        if not visited[i]:
            # BFS找连通分量
            queue = [i]
            visited[i] = True
            component = [i]
            
            while queue:
                node = queue.pop(0)
                # 找到相似度高于阈值的邻居
                neighbors = np.where(adj_matrix[node])[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        component.append(neighbor)
                        queue.append(neighbor)
            components.append(component)
    
    return components


# -------------------------- 测试用例 --------------------------
if __name__ == "__main__":
    # 模拟输入：8帧视频，每帧 24x24=576个patch，特征维度4096
    T, H, W, C = 8, 24, 24, 4096
    video_features = torch.randn(T, H*W, C).cuda()  # (8, 576, 4096)
    
    # 调用压缩函数
    merged_flatten_feat, kept_ids = compress_video_features(
        video_features=video_features,
        T=T,
        H=H,
        W=W,
        tau=0.95,
        epsilon=0.05
    )
    
    # 输出结果验证
    print(f"原始展平特征形状: {merged_flatten_feat.shape}")  # 应输出 (8*576=4608, 4096)
    print(f"保留的Token数量: {len(kept_ids)}")
    print(f"保留的Token ID示例: {kept_ids[:5]}")
    print(f"合并后特征是否与原始展平特征同形状: {merged_flatten_feat.shape == (T*H*W, C)}")