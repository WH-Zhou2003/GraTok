from typing import Optional, List, Tuple, Union
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextDecoderLayer, Qwen3VLTextModel, Qwen3VLModelOutputWithPast,Qwen3VLTextAttention
import torch
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2_vl.modeling_qwen2_vl import TransformersKwargs
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_torchdynamo_compiling

merging_ratio_global=0.5

class DycokeConfigs():
    def __init__(self):
        self.dycoke_layer_idx = 3
        self.dycoke_radio = 0.5
        self.image_token_start_index = 14
        self.image_token_length = None
        self.similarity = None
        self.attention_score = None
        self.dycoke_l = 3

class Qwen3VLTextModel_Dycoke(Qwen3VLTextModel):
    # def __init__(self):
    #     super().__init__()

    def __init__(self, config):
        super().__init__(config)
        self.Dycokeconfig = DycokeConfigs()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        # ---------- 原有前置处理 ----------
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        ########################################################
        # 设置参数
        past_key_values.dycoke_layer_idx = 3
        past_key_values.dycoke_radio = 0.8
        past_key_values.image_token_start_index = 14
        past_key_values.image_token_length = None
        past_key_values.similarity = None
        past_key_values.attention_score = None
        past_key_values.dycoke_l = 3
        #####################################################
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 处理位置编码（保持不变）
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ---------- DyCoke 配置准备 ----------
        dycoke_layer_idx = self.Dycokeconfig.dycoke_layer_idx
        dycoke_l = self.Dycokeconfig.dycoke_l
        dycoke_enabled = (dycoke_layer_idx >= 0 and dycoke_l >= 0 and use_cache and
                          isinstance(past_key_values, PrunableDynamicCache))
        if dycoke_enabled:
            # 创建一个简单的配置对象（可根据需要扩展）
            class DyCokeConfig:
                pass
            dycoke_config = DyCokeConfig()
            # 当前序列长度 + 历史长度
            past_len = past_key_values.get_seq_length() if past_key_values else 0
            dycoke_config.seq_length_with_past = past_len + inputs_embeds.shape[1]
        else:
            dycoke_config = None

        # ---------- 解码层循环 ----------
        for layer_idx, decoder_layer in enumerate(self.layers):
            # ---------- DyCoke 剪枝条件检查 ----------
            if dycoke_enabled:
                # 如果当前层小于起始剪枝层，清空kv_cache（或执行其他操作）
                if layer_idx < dycoke_layer_idx:
                    past_key_values.kv_cache = None
                # 如果当前层是触发层，且kv_cache为空，且处于生成第一个token阶段（序列长度=1），则执行剪枝
                elif (layer_idx == dycoke_l and
                      past_key_values.kv_cache is None and
                      position_ids.shape[-1] == 1):  # 生成阶段序列长度为1
                    past_key_values.dycoke_pruning(layer_idx, dycoke_config)

            # ---------- 调用当前层 ----------
            # 注意：decoder_layer 必须支持接收 output_attentions 等参数，并返回 hidden_states
            # 若需在剪枝中使用注意力分数，应在此处设置 output_attentions=True，并修改 decoder_layer 返回更多信息
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs  # 假设 decoder_layer 只返回 hidden_states

            # ---------- DeepStack 融合 ----------
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[:,visual_pos_masks, :] + visual_embeds
        hidden_states[:,visual_pos_masks, :] = local_this
        return hidden_states


class PrunableDynamicCache(DynamicCache):
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.kv_cache = None

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self.kv_cache is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            return torch.gather(self.key_cache[layer_idx], dim=2,
                                index=torch.tensor(self.kv_cache, device=self.key_cache[layer_idx].device).view(1, 1,
                                                                                                                -1,
                                                                                                                1).expand(
                                    self.key_cache[layer_idx].size(0), self.key_cache[layer_idx].size(1), -1,
                                    self.key_cache[layer_idx].size(3))), torch.gather(self.value_cache[layer_idx],
                                                                                      dim=2,
                                                                                      index=torch.tensor(self.kv_cache,
                                                                                                         device=
                                                                                                         self.value_cache[
                                                                                                             layer_idx].device).view(
                                                                                          1, 1, -1, 1).expand(
                                                                                          self.value_cache[
                                                                                              layer_idx].size(0),
                                                                                          self.value_cache[
                                                                                              layer_idx].size(1), -1,
                                                                                          self.value_cache[
                                                                                              layer_idx].size(3)))

    def update_cache(self, image_attention, config):
        # Pre-calculate values to avoid repeated computation
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        num_keep = int(img_len * (1 - config.dycoke_radio))

        # Get top indices in one operation
        top_indices = torch.topk(image_attention, num_keep, sorted=False)[1] + start_idx

        # Create ranges efficiently using single arange call
        device = image_attention.device
        full_range = torch.arange(config.seq_length_with_past, device=device)
        keep_indexs = torch.cat([
            full_range[:start_idx],
            top_indices,
            full_range[start_idx + img_len:]
        ])

        # Convert to list once at end
        self.kv_cache = keep_indexs.tolist()

    def dycoke_pruning(self, attn, layer_idx, config):
        attention_avg = attn[1].mean(1)[0, -1]
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        image_attention = attention_avg[start_idx:start_idx + img_len]

        if config.attention_score is not None:
            config.similarity = F.cosine_similarity(
                image_attention,
                config.attention_score,
                dim=0
            )
        else:
            config.similarity = 0
        config.attention_score = image_attention

        if config.similarity < 0.9:
            self.update_cache(image_attention, config)







class Qwen3VLModel_Dycoke(Qwen3VLModel):

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3VLTextModel_Dycoke._from_config(config.text_config)

    # ---------- 新增：视频Token压缩相关方法 ----------
    def dycole_ttm(self, image_feature: torch.Tensor, video_grid_thw: torch.Tensor, merging_ratio: float = 0.7) -> torch.Tensor:
        """
        动态合并视频帧中的Token。

        Args:
            image_feature: [total_tokens, hidden_dim] 的视频特征（属于单个视频）。
            video_grid_thw: [1, 3] 的网格信息 [t, h, w]。
            merging_ratio: 保留Token的比例 (0.7表示保留70%)。

        Returns:
            kept_indices: 保留的Token在image_feature中的索引（一维张量）。
        """
        t, h, w = video_grid_thw[0].tolist()
        num_tokens_per_frame = (h // self.config.vision_config.spatial_merge_size) * (w // self.config.vision_config.spatial_merge_size)
        # 重塑为 [frames, tokens_per_frame, hidden_dim]
        video_features = image_feature.view(t, num_tokens_per_frame, -1)

        similarities = []
        for i in range(0, t - 1, 2):
            if i + 1 >= t:
                break
            frame1 = video_features[i]      # [tokens_per_frame, hidden]
            frame2 = video_features[i + 1]
            # 余弦相似度
            frame1_norm = F.normalize(frame1, p=2, dim=1)
            frame2_norm = F.normalize(frame2, p=2, dim=1)
            sim = F.cosine_similarity(frame1_norm, frame2_norm, dim=1)  # [tokens_per_frame]
            similarities.append(sim)

        if not similarities:
            # 没有可合并的帧对，返回全部索引
            return torch.arange(image_feature.size(0), device=image_feature.device)

        similarities = torch.stack(similarities)  # [num_pairs, tokens_per_frame]

        kept_indices_list = []
        for pair_idx, i in enumerate(range(0, t - 1, 2)):
            if i + 1 >= t:
                break
            # 第一帧全部保留
            start1 = i * num_tokens_per_frame
            end1 = (i + 1) * num_tokens_per_frame
            kept_indices_list.append(torch.arange(start1, end1, device=image_feature.device))

            # 第二帧保留相似度最低的 merging_ratio 比例的token
            num_keep = int(merging_ratio * num_tokens_per_frame)
            # 相似度越低，信息越丰富，取topk largest=False
            _, keep_idx = similarities[pair_idx].topk(num_keep, largest=False)
            start2 = (i + 1) * num_tokens_per_frame
            kept_indices_list.append(start2 + keep_idx)

        # 处理剩余的奇数帧（未配对的最后一帧）
        last_processed = (len(similarities) * 2)
        for i in range(last_processed, t):
            start = i * num_tokens_per_frame
            end = (i + 1) * num_tokens_per_frame
            kept_indices_list.append(torch.arange(start, end, device=image_feature.device))

        kept_indices = torch.cat(kept_indices_list)
        return kept_indices

    def get_kept_token_indices(self, video_mask: torch.Tensor, kept_ids: torch.Tensor) -> torch.Tensor:
        """
        获取最终需要保留的所有Token的全局索引（包括非视频Token和保留的视频Token）。

        Args:
            video_mask: 布尔向量，长度为原始序列长度，True表示视频Token。
            kept_ids: 一维数组，需要保留的视频Token在全体视频Token中的相对索引（从0开始）。

        Returns:
            kept_indices: 保留Token的全局索引（升序排列）。
        """
        non_video_indices = torch.where(~video_mask)[0]
        video_indices = torch.where(video_mask)[0]
        if kept_ids.numel() > 0:
            video_kept = video_indices[kept_ids.long()]
        else:
            video_kept = torch.tensor([], dtype=torch.long, device=video_mask.device)
        kept_indices = torch.cat([non_video_indices, video_kept])
        kept_indices, _ = torch.sort(kept_indices)
        return kept_indices


    # def get_rope_index(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     # 检查 input_ids 是否包含视觉起始标记
    #     vision_start_token_id = getattr(self.config, 'vision_start_token_id', None)
    #     has_vision_start = False
    #     if input_ids is not None and vision_start_token_id is not None:
    #         has_vision_start = (input_ids == vision_start_token_id).any()

    #     # 如果没有视觉 token 或者没有提供图像/视频网格信息，则使用简化版本
    #     import pdb;pdb.set_trace()
    #     if not has_vision_start or (image_grid_thw is None and video_grid_thw is None):
    #         # fallback: 生成标准的 3D position_ids（与 Qwen3VL 的默认行为一致）
            
    #         if attention_mask is not None:
    #             # 根据 attention_mask 生成位置索引
    #             position_ids = attention_mask.long().cumsum(-1) - 1
    #             position_ids.masked_fill_(attention_mask == 0, 1)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
    #             max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    #             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    #         else:
    #             seq_len = input_ids.shape[1] if input_ids is not None else attention_mask.shape[1]
    #             device = input_ids.device if input_ids is not None else attention_mask.device
    #             position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
    #             mrope_position_deltas = torch.zeros(input_ids.shape[0], 1, device=device, dtype=input_ids.dtype)
    #         return position_ids, mrope_position_deltas
    #     else:
    #         # 否则调用父类的原始实现（处理正常视觉输入）
    #         return super().get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

    # ---------- 修改forward ----------
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    #     # 确保使用支持修剪的缓存
    #     if past_key_values is not None and not isinstance(past_key_values, Cache):
    #         past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
    #     elif past_key_values is None and kwargs.get("use_cache", False):
    #         # 如果需要缓存但未提供，初始化为空PrunableDynamicCache
    #         past_key_values = PrunableDynamicCache()

    #     # 原始前向逻辑：处理输入嵌入
    #     if (input_ids is None) ^ (inputs_embeds is not None):
    #         raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    #     if inputs_embeds is None:
    #         inputs_embeds = self.get_input_embeddings()(input_ids)

    #     image_mask = None
    #     video_mask = None
    #     deepstack_image_embeds = None
    #     deepstack_video_embeds = None

    #     # 图像处理
    #     if pixel_values is not None:
    #         image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    #         image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    #         image_mask, _ = self.get_placeholder_mask(
    #             input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    #         )
    #         inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    #     # 视频处理（原始填充）
    #     if pixel_values_videos is not None:
    #         video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
    #         video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    #         _, video_mask = self.get_placeholder_mask(
    #             input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
    #         )
    #         inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    #         # ---------- 视频Token压缩（仅推理时启用，且batch size=1）----------
    #         if not self.training and input_ids.shape[0] == 1:
    #             # 获取视频掩码的一维形式
    #             video_mask_1d = video_mask[..., 0]  # [1, seq_len]

    #             # 计算每个视频的Token数量
    #             spatial_merge_size = self.config.vision_config.spatial_merge_size
    #             video_token_counts = (video_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
    #             # 将拼接的视频特征按视频拆分
    #             video_embeds_list = torch.split(video_embeds, video_token_counts)
    #             # 同样拆分deepstack特征（每个级别）
    #             num_levels = len(deepstack_video_embeds)
    #             deepstack_video_embeds_per_level = [torch.split(level, video_token_counts) for level in deepstack_video_embeds]

    #             kept_indices_per_video = []
    #             compressed_video_embeds_list = []
    #             compressed_deepstack_per_video = [[] for _ in range(num_levels)]

    #             for i, (vid_embed, grid_thw) in enumerate(zip(video_embeds_list, video_grid_thw)):
    #                 # 压缩单个视频
    #                 kept = self.dycole_ttm(vid_embed, grid_thw.unsqueeze(0), merging_ratio=0.7)
    #                 kept_indices_per_video.append(kept)
    #                 # 压缩后的特征
    #                 compressed_video_embeds_list.append(vid_embed[kept])
    #                 for level_idx in range(num_levels):
    #                     compressed_deepstack_per_video[level_idx].append(
    #                         deepstack_video_embeds_per_level[level_idx][i][kept]
    #                     )

    #             # 重新拼接压缩后的特征
    #             compressed_video_embeds = torch.cat(compressed_video_embeds_list, dim=0)
    #             compressed_deepstack_video_embeds = [
    #                 torch.cat(level_list, dim=0) for level_list in compressed_deepstack_per_video
    #             ]

    #             # 计算每个视频的起始索引（用于全局视频索引）
    #             video_start_indices = [0]
    #             for cnt in video_token_counts:
    #                 video_start_indices.append(video_start_indices[-1] + cnt)
    #             # 将所有保留的视频索引转换为全局视频索引（相对于所有视频Token）
    #             kept_global_video_indices = []
    #             for i, kept in enumerate(kept_indices_per_video):
    #                 kept_global_video_indices.append(kept + video_start_indices[i])
    #             kept_global_video_indices = torch.cat(kept_global_video_indices)  # [new_total_video_tokens]

    #             # 获得最终需要保留的全局序列索引
    #             kept_global_indices = self.get_kept_token_indices(video_mask_1d[0], kept_global_video_indices)

    #             # 裁剪inputs_embeds和相关张量
    #             inputs_embeds = inputs_embeds[:, kept_global_indices, :]
    #             if attention_mask is not None:
    #                 attention_mask = attention_mask[:, kept_global_indices]
    #             if position_ids is not None:
    #                 position_ids = position_ids[:, :, kept_global_indices]
    #             if input_ids is not None:
    #                 input_ids = input_ids[:, kept_global_indices]

    #             # 裁剪past_key_values（如果存在）
    #             if past_key_values is not None and past_key_values.get_seq_length() > 0:
    #                 past_key_values.prune(kept_global_indices)

    #             # 更新visual_pos_masks和deepstack_visual_embeds
    #             # 获取裁剪后的图像掩码（如果存在）
    #             if pixel_values is not None:
    #                 image_mask_1d = image_mask[..., 0]  # [1, seq_len]
    #                 image_mask_1d = image_mask_1d[:, kept_global_indices]
    #             else:
    #                 image_mask_1d = torch.zeros_like(video_mask_1d)[:, kept_global_indices]

    #             video_mask_1d = video_mask_1d[:, kept_global_indices]
    #             visual_pos_masks = (image_mask_1d | video_mask_1d)[0]  # 取第一个样本的一维掩码

    #             # 重新构建deepstack_visual_embeds
    #             # 图像特征（未压缩）
    #             image_features_by_level = deepstack_image_embeds if pixel_values is not None else [None] * num_levels
    #             # 视频特征（压缩后）
    #             video_features_by_level = compressed_deepstack_video_embeds

    #             new_deepstack_visual_embeds = []
    #             is_image = image_mask_1d[0]
    #             is_video = video_mask_1d[0]

    #             # for level_idx in range(num_levels):
    #             #     embed_joint = torch.zeros(
    #             #         visual_pos_masks.sum(),
    #             #         inputs_embeds.shape[-1],
    #             #         device=inputs_embeds.device,
    #             #         dtype=inputs_embeds.dtype
    #             #     )
    #             #     if image_features_by_level[level_idx] is not None:
    #             #         embed_joint[is_image, :] = image_features_by_level[level_idx]
    #             #     embed_joint[is_video, :] = video_features_by_level[level_idx]
    #             #     new_deepstack_visual_embeds.append(embed_joint)

    #             ############## 先获取只对应视觉token的掩码  ##########3
    #             visual_pos_mask = visual_pos_masks          # [new_seq_len] bool
    #             is_image_vis = is_image[visual_pos_mask]   # [num_visual_tokens] bool
    #             is_video_vis = is_video[visual_pos_mask]   # [num_visual_tokens] bool

    #             for level_idx in range(num_levels):
    #                 embed_joint = torch.zeros(
    #                     visual_pos_mask.sum(),
    #                     inputs_embeds.shape[-1],
    #                     device=inputs_embeds.device,
    #                     dtype=inputs_embeds.dtype
    #                 )
    #                 if image_features_by_level[level_idx] is not None:
    #                     embed_joint[is_image_vis, :] = image_features_by_level[level_idx]
    #                 embed_joint[is_video_vis, :] = video_features_by_level[level_idx]
    #                 new_deepstack_visual_embeds.append(embed_joint)
    #             ###################################################
    #             deepstack_visual_embeds = new_deepstack_visual_embeds
    #         else:
    #             # 不压缩时，按照原有逻辑合并视觉掩码和deepstack
    #             visual_pos_masks = None
    #             deepstack_visual_embeds = None  # 将在后续合并
    #     # 原有逻辑：合并图像和视频的视觉掩码及deepstack（当未压缩或跳过时）
    #     if pixel_values_videos is None or self.training or input_ids.shape[0] != 1:
    #         # 沿用原代码的合并逻辑
    #         if image_mask is not None and video_mask is not None:
    #             image_mask = image_mask[..., 0]
    #             video_mask = video_mask[..., 0]
    #             visual_pos_masks = image_mask | video_mask
    #             deepstack_visual_embeds = []
    #             image_mask_joint = image_mask[visual_pos_masks]
    #             video_mask_joint = video_mask[visual_pos_masks]
    #             for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
    #                 embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
    #                 embed_joint[image_mask_joint, :] = img_embed
    #                 embed_joint[video_mask_joint, :] = vid_embed
    #                 deepstack_visual_embeds.append(embed_joint)
    #         elif image_mask is not None:
    #             image_mask = image_mask[..., 0]
    #             visual_pos_masks = image_mask
    #             deepstack_visual_embeds = deepstack_image_embeds
    #         elif video_mask is not None:
    #             video_mask = video_mask[..., 0]
    #             visual_pos_masks = video_mask
    #             deepstack_visual_embeds = deepstack_video_embeds

    #     # 后续计算position_ids等（保持原样）
    #     if position_ids is None:
    #         attention_mask_tensor = (
    #             attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
    #         )
    #         if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
    #             attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
    #             if attention_mask_tensor.dtype.is_floating_point:
    #                 attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
    #                 attention_mask_tensor = (1.0 - attention_mask_tensor).int()

    #         prefill_compiled_stage = is_torchdynamo_compiling() and (
    #             (input_ids is not None and input_ids.shape[1] != 1)
    #             or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
    #         )
    #         prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
    #             (cache_position is not None and cache_position[0] == 0)
    #             or (past_key_values is None or past_key_values.get_seq_length() == 0)
    #         )
    #         if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
    #             position_ids, rope_deltas = self.get_rope_index(
    #                 input_ids,
    #                 image_grid_thw,
    #                 video_grid_thw,
    #                 attention_mask=attention_mask_tensor,
    #             )
    #             self.rope_deltas = rope_deltas
    #         else:
    #             batch_size, seq_length, _ = inputs_embeds.shape
    #             delta = (
    #                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
    #                 if cache_position is not None
    #                 else 0
    #             )
    #             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
    #             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    #             if cache_position is not None:
    #                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    #             position_ids = position_ids.add(delta)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    #     outputs = self.language_model(
    #         input_ids=None,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         cache_position=cache_position,
    #         visual_pos_masks=visual_pos_masks,
    #         deepstack_visual_embeds=deepstack_visual_embeds,
    #         **kwargs,
    #     )

    #     return Qwen3VLModelOutputWithPast(
    #         last_hidden_state=outputs.last_hidden_state,
    #         past_key_values=outputs.past_key_values,
    #         rope_deltas=self.rope_deltas,
    #     )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        # 确保使用支持修剪的缓存
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
        elif past_key_values is None and kwargs.get("use_cache", False):
            past_key_values = PrunableDynamicCache()

        # 原始前向逻辑：处理输入嵌入
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        # 图像处理
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 视频处理（原始填充）
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ==================== 关键修改1：提前计算 position_ids（使用原始长度）====================
        # 这段代码原本在压缩之后，现在移到压缩之前，以保证 get_rope_index 能正确处理特殊token
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        # ===================================================================================

        # ---------- 视频Token压缩（仅推理时启用，且batch size=1）----------
        if not self.training and input_ids.shape[0] == 1 and pixel_values_videos is not None:
            # 获取视频掩码的一维形式
            video_mask_1d = video_mask[..., 0]  # [1, seq_len]

            # 计算每个视频的Token数量
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            video_token_counts = (video_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
            # 将拼接的视频特征按视频拆分
            video_embeds_list = torch.split(video_embeds, video_token_counts)
            # 同样拆分deepstack特征（每个级别）
            num_levels = len(deepstack_video_embeds)
            deepstack_video_embeds_per_level = [torch.split(level, video_token_counts) for level in deepstack_video_embeds]

            kept_indices_per_video = []
            compressed_video_embeds_list = []
            compressed_deepstack_per_video = [[] for _ in range(num_levels)]

            for i, (vid_embed, grid_thw) in enumerate(zip(video_embeds_list, video_grid_thw)):
                # 压缩单个视频
                kept = self.dycole_ttm(vid_embed, grid_thw.unsqueeze(0), merging_ratio=merging_ratio_global)
                kept_indices_per_video.append(kept)
                # 压缩后的特征
                compressed_video_embeds_list.append(vid_embed[kept])
                for level_idx in range(num_levels):
                    compressed_deepstack_per_video[level_idx].append(
                        deepstack_video_embeds_per_level[level_idx][i][kept]
                    )

            # 重新拼接压缩后的特征
            compressed_video_embeds = torch.cat(compressed_video_embeds_list, dim=0)
            compressed_deepstack_video_embeds = [
                torch.cat(level_list, dim=0) for level_list in compressed_deepstack_per_video
            ]

            # 计算每个视频的起始索引（用于全局视频索引）
            video_start_indices = [0]
            for cnt in video_token_counts:
                video_start_indices.append(video_start_indices[-1] + cnt)
            # 将所有保留的视频索引转换为全局视频索引（相对于所有视频Token）
            kept_global_video_indices = []
            for i, kept in enumerate(kept_indices_per_video):
                kept_global_video_indices.append(kept + video_start_indices[i])
            kept_global_video_indices = torch.cat(kept_global_video_indices)  # [new_total_video_tokens]

            # 获得最终需要保留的全局序列索引
            kept_global_indices = self.get_kept_token_indices(video_mask_1d[0], kept_global_video_indices)

            # ==================== 关键修改2：同时裁剪所有相关张量，包括 position_ids ====================
            print("befor:",inputs_embeds.shape)
            inputs_embeds = inputs_embeds[:, kept_global_indices, :]
            print("after:",inputs_embeds.shape)
            if attention_mask is not None:
                attention_mask = attention_mask[:, kept_global_indices]
            if position_ids is not None:
                position_ids = position_ids[:, :, kept_global_indices]   # 裁剪位置编码
            if input_ids is not None:
                input_ids = input_ids[:, kept_global_indices]            # 可选，保持一致性
            # 裁剪past_key_values（如果存在）
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                past_key_values.prune(kept_global_indices)
            # =======================================================================================

            # 更新visual_pos_masks和deepstack_visual_embeds
            # 获取裁剪后的图像掩码（如果存在）
            if pixel_values is not None:
                image_mask_1d = image_mask[..., 0]  # [1, seq_len]
                image_mask_1d = image_mask_1d[:, kept_global_indices]
            else:
                image_mask_1d = torch.zeros_like(video_mask_1d)[:, kept_global_indices]

            video_mask_1d = video_mask_1d[:, kept_global_indices]
            visual_pos_masks = (image_mask_1d | video_mask_1d)[0]  # 取第一个样本的一维掩码

            # 重新构建deepstack_visual_embeds
            # 图像特征（未压缩）
            image_features_by_level = deepstack_image_embeds if pixel_values is not None else [None] * num_levels
            # 视频特征（压缩后）
            video_features_by_level = compressed_deepstack_video_embeds

            new_deepstack_visual_embeds = []
            # ==================== 关键修改3：修复索引错误，使用 visual_pos_masks 过滤 ====================
            is_image = image_mask_1d[0]   # shape [new_seq_len]
            is_video = video_mask_1d[0]   # shape [new_seq_len]
            # 只保留视觉token位置的掩码
            is_image_vis = is_image[visual_pos_masks]   # [num_visual_tokens]
            is_video_vis = is_video[visual_pos_masks]   # [num_visual_tokens]

            for level_idx in range(num_levels):
                embed_joint = torch.zeros(
                    visual_pos_masks.sum(),
                    inputs_embeds.shape[-1],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )
                if image_features_by_level[level_idx] is not None:
                    embed_joint[is_image_vis, :] = image_features_by_level[level_idx]
                embed_joint[is_video_vis, :] = video_features_by_level[level_idx]
                new_deepstack_visual_embeds.append(embed_joint)

            deepstack_visual_embeds = new_deepstack_visual_embeds
            # ===========================================================================================
        else:
            # 不压缩时，按照原有逻辑合并视觉掩码和deepstack
            if pixel_values_videos is None or self.training or input_ids.shape[0] != 1:
                if image_mask is not None and video_mask is not None:
                    image_mask = image_mask[..., 0]
                    video_mask = video_mask[..., 0]
                    visual_pos_masks = image_mask | video_mask
                    deepstack_visual_embeds = []
                    image_mask_joint = image_mask[visual_pos_masks]
                    video_mask_joint = video_mask[visual_pos_masks]
                    for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                        embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                        embed_joint[image_mask_joint, :] = img_embed
                        embed_joint[video_mask_joint, :] = vid_embed
                        deepstack_visual_embeds.append(embed_joint)
                elif image_mask is not None:
                    image_mask = image_mask[..., 0]
                    visual_pos_masks = image_mask
                    deepstack_visual_embeds = deepstack_image_embeds
                elif video_mask is not None:
                    video_mask = video_mask[..., 0]
                    visual_pos_masks = video_mask
                    deepstack_visual_embeds = deepstack_video_embeds
                else:
                    visual_pos_masks = None
                    deepstack_visual_embeds = None

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )