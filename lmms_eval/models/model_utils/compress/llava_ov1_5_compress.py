from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache
from transformers.utils import is_torchdynamo_compiling
from transformers.modeling_outputs import ModelOutput
from .vidcom2 import *

tau=0.7
alpha=0.3
beta=0.5
lambda_coeff=0.5
p=0.3

def compute_similarity_batch(X):
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

def compute_temporal_weights_batch(S_inter):
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

def build_adjacency_matrices_batch(S_intra, S_inter):
    """批量构建邻接矩阵"""
    # 帧内邻接矩阵
    A_intra = (S_intra > tau).float()
    
    # 添加自环
    N = A_intra.shape[1]
    diag_mask = torch.eye(N, device=A_intra.device).unsqueeze(0).bool()
    A_intra = A_intra.masked_fill(diag_mask, 1.0)
    
    # 帧间邻接矩阵
    if S_inter.numel() > 0:
        A_inter = (S_inter > tau).float()
    else:
        A_inter = torch.empty(0, device=S_intra.device)
        
    return A_intra, A_inter

def compute_dependency_batch(A_intra, A_inter):
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
    dep_st = beta * dep_s_norm + (1 - beta) * dep_t_norm
    return dep_st

def compute_importance_scores_batch(X, S_inter, A_intra, A_inter):
    """批量计算重要性分数"""
    T, N, D = X.shape
    
    # 时空权重
    w_s = torch.norm(X, p=2, dim=2)  # (T, N)
    w_s = w_s / w_s.max()
    
    w_t = compute_temporal_weights_batch(S_inter)
    
    w_st = alpha * w_s + (1 - alpha) * w_t
    
    # 时空依赖度
    dep_st = compute_dependency_batch(A_intra, A_inter)
    
    # 综合分数
    scores = lambda_coeff * w_st + (1 - lambda_coeff) * dep_st
    return scores

def create_neighbor_masks(A_intra, A_inter, N):
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


def refine_tokens_gpu_optimized2(X, grid_thw):
    """
    真正GPU优化的版本：从低分开始删除，删除时合并到相似邻居
    """
    T=grid_thw[0]
    H=grid_thw[1]//2
    W=grid_thw[2]//2
    N_total, D = X.shape
    N = N_total // T
    M = max(1, int(p * N_total))
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
    A_intra = (S_intra > tau).float()
    eye_matrix = torch.eye(N, device=X.device).unsqueeze(0).expand(T, N, N)
    A_intra = torch.where(eye_matrix.bool(), torch.ones_like(A_intra), A_intra)
    if S_inter.numel() > 0:
        A_inter = (S_inter > tau).float()
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
    w_st = alpha * (1-w_s) + (1 - alpha) * w_t
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
    dep_st = beta * dep_s + (1 - beta) * dep_t
    # 综合分数（保持原逻辑不变）
    scores = lambda_coeff * w_st + (1 - lambda_coeff) * (1-dep_st)
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
        if best_sim > tau:
            # 计算合并权重
            weight_alpha = best_sim  # 相似度作为权重因子
            # 合并特征
            tokens[best_neighbor] = (
                weight_alpha * tokens[min_idx] +
                (1.0 - weight_alpha) * tokens[best_neighbor]
            )
            # 合并分数
            token_scores[best_neighbor] = (
                token_scores[best_neighbor] +
                token_scores[min_idx] * weight_alpha
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


@dataclass
class LLaVAOneVision1_5_ModelOutputWithPast(ModelOutput):
    """
    Base class for Llava outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

def get_kept_token_indices(video_mask, kept_ids):
    """
    获取需要保留的所有Token的全局索引

    参数:
    video_mask: 布尔向量，True表示视频Token，False表示非视频Token
    kept_ids: 一维数组，压缩后保留的视频Token的相对索引（相对于所有视频Token）
    N_per_frame: 每帧的Token数

    返回:
    kept_indices: 需要保留的所有Token的全局索引
    """
    # 确保输入是Tensor
    if not isinstance(video_mask, torch.Tensor):
        video_mask = torch.tensor(video_mask)
    if not isinstance(kept_ids, torch.Tensor):
        kept_ids = torch.tensor(kept_ids)
    # 1. 找出所有非视频Token的位置（需要全部保留）
    non_video_indices = torch.where(~video_mask)[0]
    # 2. 找出所有视频Token的位置
    video_indices = torch.where(video_mask)[0]
    # 3. 检查kept_ids是否有效
    total_video_tokens = len(video_indices)
    if len(kept_ids) > 0:
        if kept_ids.max() >= total_video_tokens:
            raise ValueError(f"kept_ids中的最大索引{kept_ids.max().item()}超出了视频Token总数{total_video_tokens}")
    # 4. 将kept_ids（视频Token的相对索引）转换为全局索引
    if len(kept_ids) > 0:
        # 使用索引选择
        video_kept_indices = video_indices[kept_ids.long()]
    else:
        video_kept_indices = torch.tensor([], dtype=torch.long, device=video_mask.device)
    # 5. 合并非视频Token索引和保留的视频Token索引
    kept_indices = torch.cat([non_video_indices, video_kept_indices])
    # 6. 按顺序排序
    kept_indices, _ = torch.sort(kept_indices)
    return kept_indices

def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, LLaVAOneVision1_5_ModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
        [`Qwen2VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)

            #######################我的压缩######################
            video_embeds, keep_ids=refine_tokens_gpu_optimized2(video_embeds,video_grid_thw[0])
            ####################################################

            ################################### vidcom2压缩 ##################################
            # base_scale=0.15
            # t = video_grid_thw[:, 0]
            # h = video_grid_thw[:, 1]
            # w = video_grid_thw[:, 2]
            # resize_h = h // 2
            # resize_w = w // 2
            # token_per_frame = resize_h * resize_w
            # model = "llava_ov1_5_compress"
            # indices = vidcom2_compression(flattened_feat=video_embeds, model=model, base_scale=base_scale,
            #                                frame_token_len=token_per_frame)
            # global_indices = []
            # for i, local_idx in enumerate(indices):
            #     offset = i * token_per_frame   # token_per_frame 即每帧 token 数
            #     # 确保 local_idx 和 offset 在同一设备
            #     offset_tensor = torch.tensor(offset, device=local_idx.device)
            #     global_indices.append(local_idx + offset_tensor)
            # keep_ids = torch.cat(global_indices)
            ########################################################################################
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            ########得到适应get_kept_token_indices的mask############
            video_token_mask_1d = (input_ids == self.config.video_token_id)  # 形状: (batch_size, seq_len)
            video_mask = video_token_mask_1d.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            # video_mask = (
            #     (input_ids == self.config.video_token_id)
            #     .unsqueeze(-1)
            #     .expand_as(inputs_embeds)
            #     .to(inputs_embeds.device)
            # )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    #############################将所有序列都只保留需要的id####################
    if pixel_values_videos is not None:
        retention_video_indice_in_all_tokens=get_kept_token_indices(video_token_mask_1d[0],keep_ids)
        print("before_prune_input", inputs_embeds.shape)
        inputs_embeds = inputs_embeds[:, retention_video_indice_in_all_tokens, :]
        print("after_prune_input", inputs_embeds.shape)

        input_ids = input_ids[:, retention_video_indice_in_all_tokens]
        position_ids = position_ids[:, retention_video_indice_in_all_tokens]
        attention_mask = attention_mask[:, retention_video_indice_in_all_tokens]
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    output = LLaVAOneVision1_5_ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()



