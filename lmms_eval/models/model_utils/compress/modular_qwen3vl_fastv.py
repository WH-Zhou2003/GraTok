from typing import Optional, Union
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextDecoderLayer, Qwen3VLTextModel, Qwen3VLModelOutputWithPast,Qwen3VLTextAttention
import torch
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2_vl.modeling_qwen2_vl import TransformersKwargs
from .stgtokenrefiner2 import STGTokenRefiner
from .vidcom2 import vidcom2_compression
from .scissor import compress_video_features
# import numpy as np
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
import os
import torch.nn as nn



class Qwen3VLTextDecoderLayer_Fastv(Qwen3VLTextDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        # import pdb;pdb.set_trace()
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights

class Qwen3VLTextModel_fastv(Qwen3VLTextModel):
############fastv###################
    def __init__(self, config):
        super().__init__(config)
        self.last_attention = None  # 保存注意力分数
        self.prune_layer_idx = 3    # 在第4层（索引3）后剪枝
        self.prune_ratio = 0.3      # 保留50%的视觉tokens
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer_Fastv(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    def _fastv_prune(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        text_position_ids: torch.Tensor,
        visual_embeds: Optional[torch.Tensor] = None,
    ):
        """
        执行FastV剪枝：基于注意力分数筛选重要的视觉tokens
        """
        if self.last_attention is None or visual_pos_masks.sum() == 0:
            return hidden_states, attention_mask, position_ids, cache_position, visual_pos_masks
        
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        device = hidden_states.device
        
        # 获取视觉tokens的位置索引
        visual_indices = torch.where(visual_pos_masks[0])[0]  # 假设batch维度相同
        num_visual_tokens = len(visual_indices)
        
        if num_visual_tokens == 0:
            return hidden_states, attention_mask, position_ids, cache_position, visual_pos_masks
        
        # 计算要保留的视觉tokens数量
        keep_num = int(num_visual_tokens * self.prune_ratio)
        
        # 初始化保留索引（包含所有文本tokens）
        all_indices = torch.arange(seq_length, device=device)
        keep_indices = all_indices[~visual_pos_masks[0].bool()].tolist()  # 所有文本tokens
        
        # 对每个batch单独处理
        batch_keep_indices = []
        for b in range(batch_size):
            # 提取该batch的视觉tokens注意力分数
            # 假设注意力形状: [batch, heads, target_seq, source_seq]
            # 使用最后一个文本token的注意力分布
            if self.last_attention.dim() == 4:  # 标准注意力形状
                # 取最后一个文本token对视觉tokens的注意力
                last_text_idx = -1
                while last_text_idx >= -seq_length and visual_pos_masks[b, last_text_idx]:
                    last_text_idx -= 1
                
                if last_text_idx < -seq_length:
                    # 没有找到文本token，使用平均注意力
                    visual_attention = self.last_attention[b, :, :, visual_indices].mean(dim=(1, 2))
                else:
                    visual_attention = self.last_attention[b, :, last_text_idx, visual_indices].mean(dim=0)
            else:
                # 其他注意力格式，使用简单平均
                visual_attention = self.last_attention.mean(dim=(0, 1))[visual_indices]
            
            # 选择最重要的视觉tokens
            _, top_indices = torch.topk(visual_attention, keep_num)
            selected_visual_indices = visual_indices[top_indices].tolist()
            
            # 合并文本和选中的视觉tokens
            batch_indices = sorted(keep_indices + selected_visual_indices)
            batch_keep_indices.append(batch_indices)
        
        # 找到所有batch都保留的索引（或选择第一个batch的索引）
        if batch_size > 1:
            # 简单处理：取第一个batch的索引
            keep_indices_tensor = torch.tensor(batch_keep_indices[0], device=device)
        else:
            keep_indices_tensor = torch.tensor(batch_keep_indices[0], device=device)
        
        # 更新hidden_states
        pruned_hidden_states = hidden_states[:, keep_indices_tensor, :]
        
        # 更新attention_mask
        if attention_mask is not None:
            pruned_seq_len = len(keep_indices_tensor)
            if attention_mask.dim() == 4:  # 4D mask [batch, 1, target, source]
                attention_mask = attention_mask[:, :, keep_indices_tensor, :]
                attention_mask = attention_mask[:, :, :, keep_indices_tensor]
            elif attention_mask.dim() == 2:  # 2D mask [batch, seq]
                attention_mask = attention_mask[:, keep_indices_tensor]
        
        # 更新position_ids
        if position_ids is not None:
            # position_ids形状为[3, batch, seq]
            pruned_position_ids = position_ids[:, :, keep_indices_tensor]
        else:
            pruned_position_ids = position_ids
        
        # 更新text_position_ids
        pruned_text_position_ids = text_position_ids[:, keep_indices_tensor]
        
        # 更新cache_position
        if cache_position is not None:
            pruned_cache_position = cache_position[keep_indices_tensor]
        else:
            pruned_cache_position = cache_position
        
        # 更新visual_pos_masks
        pruned_visual_pos_masks = torch.zeros(batch_size, len(keep_indices_tensor), 
                                            dtype=visual_pos_masks.dtype, device=device)
        for b in range(batch_size):
            # 将选中的视觉tokens位置标记为True
            selected_visual_indices = batch_keep_indices[b]
            visual_mask = torch.isin(torch.tensor(selected_visual_indices, device=device), 
                                   torch.tensor(visual_indices.tolist(), device=device))
            pruned_visual_pos_masks[b] = visual_mask
        
        # 清空保存的注意力分数
        self.last_attention = None
        
        return (pruned_hidden_states, attention_mask, pruned_position_ids, 
                pruned_cache_position, pruned_visual_pos_masks)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
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
        
        # 初始位置编码 - 在剪枝前计算一次
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            # 在第prune_layer_idx-1层获取注意力分数
            if layer_idx == self.prune_layer_idx - 1:
                # 临时修改以获取注意力输出
                with torch.no_grad():
                    # 保存当前配置
                    original_output_attentions = kwargs.get('output_attentions', False)
                    kwargs['output_attentions'] = True
                    # 运行第prune_layer_idx-1层
                    layer_outputs, self.last_attention = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,  # 使用当前的position_embeddings
                        **kwargs,
                    )
                    # 恢复原始配置
                    kwargs['output_attentions'] = original_output_attentions
            else:        
                # 正常的前向传播
                layer_outputs, _ = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,  # 使用当前的position_embeddings
                    **kwargs,
                )
            hidden_states = layer_outputs
            
            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
            
            # 在第3层（索引2）后执行剪枝
            if layer_idx == self.prune_layer_idx and visual_pos_masks is not None:
                print("before_fastv_prune_input", hidden_states.shape)           
                hidden_states, attention_mask, position_ids, cache_position, visual_pos_masks = self._fastv_prune(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    visual_pos_masks=visual_pos_masks,
                    text_position_ids=text_position_ids,
                    visual_embeds=deepstack_visual_embeds[layer_idx] if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds) else None
                )
                print("after_fastv_prune_input", hidden_states.shape)
                
                # 关键：剪枝后需要重新计算 position_embeddings
                # 因为序列长度发生了变化，cos 和 sin 需要对应新的序列长度
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor):
        """处理DeepStack视觉特征注入（保持不变）"""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states



class Qwen3VLModel_fastv(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3VLTextModel_fastv._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        # image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
        # return image_embeds, deepstack_image_embeds   `   `


    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask


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
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
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

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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