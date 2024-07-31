

from typing import Optional
from ...layers.attention import Attention
from ...functional import AllReduceFusionParams,Tensor


class BypassAttention(Attention):

    # def __init__(self, *args,**kwargs):
    #     super().__init__(*args,**kwargs)



    def forward(self,
                hidden_states: Tensor,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                encoder_output: Optional[Tensor] = None,
                position_embedding=None,
                norm_before_bmm1=False,
                lora_layer_params=None,
                cross_kv_cache_gen: Optional[Tensor] = None,
                cross_qkv_reuse: Optional[Tensor] = None,
                reduce_fusion_params: Optional[AllReduceFusionParams] = None):

        assert isinstance(hidden_states, Tensor)
        '''
        spec_decoding_params = SpecDecodingParams(
        ) if spec_decoding_params is None else spec_decoding_params

        alibi_slopes = None
        if self.position_embedding_type.is_alibi():
            alibi_slopes = self.alibi_slopes.value
            if default_net().plugin_config.gpt_attention_plugin:
                alibi_slopes = cast(alibi_slopes, hidden_states.dtype)

        qkv_lora_params = None
        if lora_layer_params is not None:
            if not self.cross_attention:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_qkv")
            else:
                qkv_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_qkv")

        unfuse_qkv_gemm = self.qkv is None
        if unfuse_qkv_gemm:
            qkv_gemm = [self.q, self.k, self.v]
            qkv = [gemm(hidden_states) for gemm in qkv_gemm]
            if default_net(
            ).plugin_config.lora_plugin and qkv_lora_params is not None:
                lora = self.qkv.lora(hidden_states, qkv_lora_params)
                kv_size = self.attention_head_size * self.num_attention_kv_heads
                qkv_lora = split(lora,
                                 [self.attention_hidden_size, kv_size, kv_size],
                                 dim=1)
                qkv = [tensor + lora for tensor, lora in zip(qkv, qkv_lora)]
        else:
            qkv = self.qkv(hidden_states, qkv_lora_params)

        if self.clip_qkv is not None:
            qkv = clip(qkv, -self.clip_qkv, self.clip_qkv)

        if default_net().plugin_config.remove_input_padding:
            if unfuse_qkv_gemm:
                for tensor in qkv:
                    assert tensor.ndim() == 2
            else:
                assert qkv.ndim() == 2

        if default_net(
        ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
            if not self.cross_attention:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "attn_v")
            else:
                q_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_q")
                k_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_k")
                v_lora_params = lora_layer_params.get_runtime_params(
                    0, "cross_attn_v")

            assert (q_lora_params is not None and k_lora_params is not None and v_lora_params is not None) or \
                (q_lora_params is None and k_lora_params is None and v_lora_params is None), "q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time."

            if q_lora_params is not None and k_lora_params is not None and v_lora_params is not None:
                qkv_lora_runtime_params = LoraRuntimeParams(
                    lora_ranks=[
                        q_lora_params.lora_ranks[0],
                        k_lora_params.lora_ranks[0],
                        v_lora_params.lora_ranks[0],
                    ],
                    lora_weights_pointers=[
                        q_lora_params.lora_weights_pointers[0],
                        k_lora_params.lora_weights_pointers[0],
                        v_lora_params.lora_weights_pointers[0],
                    ],
                    host_request_types=q_lora_params.host_request_types,
                    host_context_lengths=q_lora_params.host_context_lengths,
                    max_context_length=q_lora_params.max_context_length,
                    max_encoder_context_length=q_lora_params.
                    max_encoder_context_length,
                    host_encoder_input_lengths=q_lora_params.
                    host_encoder_input_lengths,
                )

                q_lora, k_lora, v_lora = self.qkv_lora(hidden_states,
                                                       qkv_lora_runtime_params)
                qkv_lora = concat([q_lora, k_lora, v_lora],
                                  dim=q_lora.rank() - 1)
                qkv = qkv + qkv_lora
        if self.qk_layernorm:
            base_shape = shape(qkv, 0) if qkv.ndim() == 2 else concat(
                [shape(qkv, 0), shape(qkv, 1)])
            # here we assume that q, k and v have the same number of attention heads
            # TODO: allow different number of attention heads for q, k and v.
            qkv = qkv.view(
                concat([
                    base_shape, self.num_attention_heads, 3,
                    self.attention_head_size
                ]))

            query, key, value = split(qkv, 1, dim=qkv.ndim() - 2)
            q_shape = concat([
                base_shape, self.num_attention_heads, self.attention_head_size
            ])
            query = query.view(q_shape)
            key = key.view(q_shape)
            value = value.view(q_shape)

            query = self.q_layernorm(query)
            key = self.k_layernorm(key)
            qkv = concat([query, key, value], dim=query.ndim() - 2)
            qkv = qkv.view(concat([base_shape, self.attention_hidden_size * 3]))
        if self.position_embedding_type == PositionEmbeddingType.chatglm:
            qkv = RopeEmbeddingUtils.apply_rotary_pos_emb_chatglm(
                qkv,
                position_embedding,
                self.num_attention_heads,
                self.attention_head_size,
                self.max_position_embeddings,
                self.rotary_embedding_scale,
                default_net().plugin_config.remove_input_padding,
            )
            self.rotary_embedding_scale_type = RotaryScalingType.none
            self.rotary_embedding_scale = 1.0

        paged_kv_cache = default_net().plugin_config.paged_kv_cache

        assert attention_params is None or attention_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin,
            default_net().plugin_config.remove_input_padding)
        assert kv_cache_params is None or kv_cache_params.is_valid(
            default_net().plugin_config.gpt_attention_plugin)

        past_key_value = None if kv_cache_params is None else kv_cache_params.get_first_past_key_value(
        )

        # if cross attention, cross QKV only needs to be calculated once in the
        # 1st decoding step --> write to cross KV cache --> remains constant
        # during the entire decoding steps.
        # 1st and >1st steps are distinguished by a boolean tensor `cross_kv_cache_gen` passed at runtime
        # also, cross KV cache max length is set from encoder output seqlen,
        # this maps to the max context length concept in decoder-only models
        cross_qkv = None
        if self.cross_attention and encoder_output:
            assert isinstance(encoder_output, Tensor)

            def compute_cross_qkv(encoder_output):
                cross_qkv = self.qkv(encoder_output, qkv_lora_params)

                if default_net(
                ).plugin_config.lora_plugin and qkv_lora_params is None and lora_layer_params is not None:
                    cross_q_lora, cross_k_lora, cross_v_lora = self.qkv_lora(
                        encoder_output,
                        qkv_lora_runtime_params,
                        is_cross_attention=True)
                    cross_qkv_lora = concat(
                        [cross_q_lora, cross_k_lora, cross_v_lora],
                        dim=cross_q_lora.rank() - 1)
                    cross_qkv = cross_qkv + cross_qkv_lora

                return cross_qkv

            if self.skip_cross_qkv:
                conditional = Conditional(cross_kv_cache_gen)
                cond_in1 = conditional.add_input(encoder_output)
                cond_in2 = conditional.add_input(cross_qkv_reuse)

                ## True branch: context phase, compute cross qkv
                cross_qkv_true = compute_cross_qkv(cond_in1)

                ## False branch: generation phase, no compute but need to obey shape constraints
                # because TRT's IfConditional requires the output shape of two subgraphs to be identical
                # our 1st attempt was to stack encoder_output [B, S, H] or [N, H] --> cross qkv [B, S, 3*H] or [N, 3*H],
                # but it still introduces unnecessary concat. A better solution is to create a dummy torch tensor `cross_qkv_resue`
                # with the correct shape and reuse it in every generation step
                cross_qkv_false = cond_in2
                cross_qkv = conditional.add_output(cross_qkv_true,
                                                   cross_qkv_false)
            else:
                cross_qkv = compute_cross_qkv(encoder_output)

        if default_net().plugin_config.gpt_attention_plugin:
            if self.cross_attention and (past_key_value is not None):
                past_key_value = kv_cache_params.past_key_value[1]
            assert self.attention_mask_type in [
                AttentionMaskType.causal, AttentionMaskType.bidirectional,
                AttentionMaskType.bidirectionalglm,
                AttentionMaskType.blocksparse
            ], 'Plugin only support masked MHA.'

            # KV cache scales.
            if self.kv_cache_scaling_factor is not None:
                kv_orig_quant_scale = constant(fp32_array(
                    [1.0])) / self.kv_cache_scaling_factor.value
                kv_quant_orig_scale = self.kv_cache_scaling_factor.value
            else:
                kv_orig_quant_scale = None
                kv_quant_orig_scale = None

            # Attention output scales
            assert (
                not default_net().plugin_config.use_fp8_context_fmha
            ) or self.quant_mode.has_fp8_qdq(
            ), "FP8 Context FMHA must be used together with the fp8 quantization workflow."

            attention_output_orig_quant_scale = self.attention_output_orig_quant_scale.value if self.attention_output_orig_quant_scale is not None else None

            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                short = slice(
                    self.embed_positions_short_factors_for_attention_plugin.
                    value, concat([0, 0, 0]),
                    concat([
                        max(attention_params.sequence_length,
                            self.original_max_position_embeddings),
                        self.rotary_embedding_dim // 2, 2
                    ]))
                long = slice(
                    self.embed_positions_long_factors_for_attention_plugin.
                    value, concat([0, 0, 0]),
                    concat([
                        max(attention_params.sequence_length,
                            self.original_max_position_embeddings),
                        self.rotary_embedding_dim // 2, 2
                    ]))
                short = short.view((1, -1))
                long = long.view((1, -1))
                embed_positions = concat([short, long], dim=0)
                select = where(
                    fmax(attention_params.sequence_length, dim=0) <=
                    self.original_max_position_embeddings, 0, 1)
                rotary_cos_sin = slice(embed_positions,
                                       concat([select, 0]),
                                       sizes=concat([1, shape(long, 1)]))
                short_factors = self.rope_scaling_short_factors.value
                long_factors = self.rope_scaling_long_factors.value
                scale_factors = concat([short_factors, long_factors], dim=0)
                rope_scaling_factors = slice(scale_factors,
                                             concat([select, 0]),
                                             sizes=concat(
                                                 [1, shape(long_factors, 1)]))
                rope_scaling_factors = rope_scaling_factors.view((-1, ))
            else:
                # Rotary cos/sin cache.
                rotary_cos_sin = self.embed_positions_for_gpt_attention.value if self.position_embedding_type.is_rope(
                ) else None
                rope_scaling_factors = None

            if self.position_embedding_type == PositionEmbeddingType.long_rope:
                short_mscale, long_mscale = self.short_mscale, self.long_mscale
            else:
                short_mscale, long_mscale = None, None

            context, past_key_value = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=attention_params.sequence_length,
                host_past_key_value_lengths=kv_cache_params.
                host_past_key_value_lengths,
                host_max_attention_window_sizes=kv_cache_params.
                host_max_attention_window_sizes,
                host_sink_token_length=kv_cache_params.host_sink_token_length,
                context_lengths=attention_params.context_lengths,
                cache_indirection=kv_cache_params.cache_indirection,
                host_request_types=attention_params.host_request_types,
                layer_idx=self.local_layer_idx,
                num_heads=self.num_attention_heads,
                num_kv_heads=self.num_attention_kv_heads,
                hidden_size_per_head=self.attention_head_size,
                q_scaling=self.q_scaling,
                rotary_embedding_dim=self.rotary_embedding_dim,
                rotary_embedding_base=self.rotary_embedding_base,
                rotary_embedding_scale_type=self.rotary_embedding_scale_type,
                rotary_embedding_scaling_factors=rope_scaling_factors,
                rotary_embedding_short_m_scale=short_mscale,
                rotary_embedding_long_m_scale=long_mscale,
                rotary_embedding_scale=self.rotary_embedding_scale,
                rotary_embedding_max_positions=self.max_position_embeddings,
                rotary_embedding_original_max_positions=self.
                original_max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                rotary_cos_sin=rotary_cos_sin,
                kv_orig_quant_scale=kv_orig_quant_scale,
                kv_quant_orig_scale=kv_quant_orig_scale,
                attention_output_orig_quant_scale=
                attention_output_orig_quant_scale,
                kv_cache_quant_mode=self.quant_mode,
                max_context_length=attention_params.max_context_length,
                mask_type=self.attention_mask_type,
                block_sparse_block_size=self.block_sparse_params.block_size,
                block_sparse_homo_head_pattern=self.block_sparse_params.
                homo_head_pattern,
                block_sparse_num_local_blocks=self.block_sparse_params.
                num_local_blocks,
                block_sparse_vertical_stride=self.block_sparse_params.
                vertical_stride,
                alibi_slopes=alibi_slopes,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets
                if not self.cross_attention else
                kv_cache_params.cross_kv_cache_block_offsets,
                host_kv_cache_block_offsets=kv_cache_params.
                host_kv_cache_block_offsets if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=kv_cache_params.
                host_kv_cache_pool_pointers if not self.cross_attention else
                kv_cache_params.host_cross_kv_cache_pool_pointers,
                do_cross_attention=self.cross_attention,
                cross_qkv=cross_qkv,
                cross_qkv_length=attention_params.encoder_max_input_length,
                encoder_input_lengths=attention_params.encoder_input_lengths,
                relative_attention_bias=self.rel_attn_table.value
                if self.relative_attention else None,
                max_distance=self.max_distance,
                host_context_lengths=attention_params.host_context_lengths,
                use_cache=use_cache,
                spec_decoding_is_generation_length_variable=spec_decoding_params
                .spec_decoding_is_generation_length_variable,
                spec_decoding_max_generation_length=spec_decoding_params.
                spec_decoding_max_generation_length,
                spec_decoding_generation_lengths=spec_decoding_params.
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_params.
                spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_params.
                spec_decoding_packed_mask,
                qk_tanh_scale=self.max_attn_value)

        else:
            # plain TensorRT mode
            assert paged_kv_cache == False

            def transpose_for_scores(x,
                                     rotary: bool = False,
                                     is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])
                if rotary:
                    return x.view(new_x_shape)
                else:
                    return x.view(new_x_shape).permute([0, 2, 1, 3])

            # qkv after projection is of shape
            #   [bs, seqlen, (num_attention_heads + 2 * num_attention_kv_heads), attention_head_size].
            # The projected and split qkv after transpose_for_scores():
            #   Q[bs, num_attention_heads, seqlen, attention_head_size]
            #   K[bs, num_attention_kv_heads, seqlen, attention_head_size]
            #   V[bs, num_attention_kv_heads, seqlen, attention_head_size]
            kv_size = self.attention_head_size * self.num_attention_kv_heads
            if unfuse_qkv_gemm:
                query, key, value = qkv[0], qkv[1], qkv[2]
            else:
                query, key, value = split(
                    qkv, [self.attention_hidden_size, kv_size, kv_size], dim=2)

            # in cross attention mode, replace kv by encoder_output
            if self.cross_attention and encoder_output is not None:
                encoder_qkv = self.qkv(encoder_output)
                _, key, value = split(
                    encoder_qkv, [self.attention_hidden_size, kv_size, kv_size],
                    dim=2)

            query = transpose_for_scores(
                query, rotary=self.position_embedding_type.is_rope())
            key = transpose_for_scores(
                key, is_kv=True, rotary=self.position_embedding_type.is_rope())
            value = transpose_for_scores(value, is_kv=True)

            if self.position_embedding_type.is_rope():
                if self.position_embedding_type == PositionEmbeddingType.long_rope:
                    sequence_length = shape(hidden_states, 1)
                    short = slice(
                        self.embed_positions_short_factors.value,
                        concat([0, 0, 0]),
                        concat([
                            1,
                            max(sequence_length,
                                self.original_max_position_embeddings),
                            self.rotary_embedding_dim
                        ]))
                    long = slice(
                        self.embed_positions_long_factors.value,
                        concat([0, 0, 0]),
                        concat([
                            1,
                            max(sequence_length,
                                self.original_max_position_embeddings),
                            self.rotary_embedding_dim
                        ]))
                    embed_positions = concat([short, long], dim=0)
                    select = where(
                        sequence_length <=
                        self.original_max_position_embeddings, 0, 1)
                    embed_positions = slice(embed_positions,
                                            concat([select, 0, 0]),
                                            sizes=shape(short))
                    embed_positions = cast(embed_positions, self.dtype)
                elif is_same_dtype(self.dtype, trt.bfloat16):
                    embed_positions = cast(self.embed_positions.value,
                                           trt.bfloat16)
                else:
                    embed_positions = cast(self.embed_positions.value,
                                           query.dtype)

                if self.rotary_embedding_dim is not None:
                    # When shape(hidden_states, 1) > 1(Context phase), the embedding start from 0,
                    # otherwise (Generation phase) move start to position
                    start = where(
                        shape(hidden_states, 1) > 1, 0,
                        shape(past_key_value, 3))
                    size = where(
                        shape(hidden_states, 1) > 1, shape(hidden_states, 1), 1)
                    sincos = slice(embed_positions, concat([0, start, 0]),
                                   concat([1, size, self.rotary_embedding_dim]))
                    sin, cos = split(sincos,
                                     self.rotary_embedding_dim // 2,
                                     dim=-1)

                    key_rot_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), self.rotary_embedding_dim
                    ])
                    query_rot_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), self.rotary_embedding_dim
                    ])
                    remaining = shape(key, 3) - self.rotary_embedding_dim
                    key_pass_size = concat([
                        shape(key, 0),
                        shape(key, 1),
                        shape(key, 2), remaining
                    ])
                    query_pass_size = concat([
                        shape(query, 0),
                        shape(query, 1),
                        shape(query, 2), remaining
                    ])
                    k_rot = slice(key, [0, 0, 0, 0], key_rot_size)
                    k_pass = slice(key, [0, 0, 0, self.rotary_embedding_dim],
                                   key_pass_size)

                    q_rot = slice(query, [0, 0, 0, 0], query_rot_size)
                    q_pass = slice(query, [0, 0, 0, self.rotary_embedding_dim],
                                   query_pass_size)

                    k_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        k_rot, [cos, sin], self.position_embedding_type)
                    q_rot = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        q_rot, [cos, sin], self.position_embedding_type)

                    key = concat([k_rot, k_pass], dim=3)
                    query = concat([q_rot, q_pass], dim=3)
                else:
                    key = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        key, [cos, sin], self.position_embedding_type)
                    query = RopeEmbeddingUtils.apply_rotary_pos_emb(
                        query, [cos, sin], self.position_embedding_type)

                key = key.permute([0, 2, 1, 3])
                query = query.permute([0, 2, 1, 3])

            if past_key_value is not None and not self.cross_attention:
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = dequantize(
                        past_key_value,
                        self.kv_cache_scaling_factor.value,
                        output_type=self.dtype)

                # past_key_value [bs, 2, num_heads, max_seq_len, head_dim]
                past_key, past_value = split(past_key_value, 1, dim=1)

                key_shape = concat([
                    shape(past_key, 0),
                    shape(past_key, 2),
                    shape(past_key, 3),
                    shape(past_key, 4)
                ])
                past_key = past_key.view(key_shape, zero_is_placeholder=False)
                past_value = past_value.view(key_shape,
                                             zero_is_placeholder=False)

                key = concat([past_key, key], dim=2)
                value = concat([past_value, value], dim=2)

            if use_cache:
                key_inflated_shape = concat([
                    shape(key, 0), 1,
                    shape(key, 1),
                    shape(key, 2),
                    shape(key, 3)
                ])
                inflated_key = key.view(key_inflated_shape,
                                        zero_is_placeholder=False)
                inflated_value = value.view(key_inflated_shape,
                                            zero_is_placeholder=False)
                past_key_value = concat([inflated_key, inflated_value], dim=1)

                # TRT quantizes the tensor value by doing `cast(clip(fp_value / scale))` while
                # the plugin quantizes it by doing `cast(clip(fp_value * scale))`.
                if self.kv_cache_scaling_factor is not None:
                    past_key_value = quantize(
                        past_key_value,
                        self.kv_cache_scaling_factor.value,
                        dtype='fp8'
                        if self.quant_mode.has_fp8_kv_cache() else 'int8')

            # MQA broadcast
            if self.num_attention_heads // self.num_attention_kv_heads > 1:
                key = repeat_interleave(
                    key,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)
                value = repeat_interleave(
                    value,
                    self.num_attention_heads // self.num_attention_kv_heads, 1)

            key_length = shape(key, 2)

            # The following code creates a 2D tensor with 0s in the lower triangular (including the diagonal) and
            # +INF in the upper triangular parts. This bias tensor will be added to the output of the Q*K^T matrix
            # multiplication (BMM1). The +INF elements will be transformed to 0s by the Softmax operator that
            # follows. The elements that corresponds to 0s in the bias are unaffected by the bias tensor.
            #
            # Note that when we added to another bias tensor B (for example, with AliBi), the values in the lower-
            # triangular part of the B tensor are not affected and the upper-triangular ones are set to +INF.
            if self.attention_mask_type == AttentionMaskType.causal and not self.cross_attention:
                if self.position_embedding_type.is_alibi():
                    query_length = shape(query, 2)
                    # bsz, tatget_length, past_key_value_length
                    buffer = make_causal_mask(shape(query, 0), query_length,
                                              key_length - query_length,
                                              trt.float32)
                    starts = concat([0, 0, 0, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    generated_mask = slice(buffer, starts, sizes)

                else:
                    query_length = shape(query, 2)
                    starts = concat([0, 0, key_length - query_length, 0])
                    sizes = concat([1, 1, query_length, key_length])
                    if self.position_embedding_type == PositionEmbeddingType.long_rope:
                        buf_shape = (self.original_max_position_embeddings,
                                     self.original_max_position_embeddings)
                    else:
                        buf_shape = (self.max_position_embeddings,
                                     self.max_position_embeddings)
                    select_buf = np.expand_dims(
                        np.tril(np.ones(buf_shape)).astype(bool), (0, 1))

                    select_buf = np.logical_not(select_buf)
                    mask_buf = np.zeros_like(select_buf, np.float32)
                    mask_buf[select_buf] = float('-inf')
                    buffer = constant(mask_buf)
                    generated_mask = slice(buffer, starts, sizes)

            elif self.attention_mask_type == AttentionMaskType.bidirectional and not self.cross_attention:
                query_length = shape(query, 2)
                zero_buf = np.expand_dims(
                    np.zeros((self.max_position_embeddings,
                              self.max_position_embeddings),
                             dtype=np.float32), (0, 1))

                zero_buf[:, :, :-1, -1] = 1
                zero_buf *= -10000

                mask = constant(zero_buf)

                # context phase, query_length
                mask_size = where(query_length > 1, query_length, 1)
                mask_start = where(query_length > 1,
                                   self.max_position_embeddings - mask_size, 1)
                start = concat([0, 0, mask_start, mask_start])
                size = concat([1, 1, mask_size, mask_size])
                generated_mask = slice(mask, start, size)

            if attention_mask is not None:
                if self.cross_attention:
                    batch_size = shape(attention_mask, 0)
                    query_len = shape(attention_mask, 1)
                    encoder_input_len = shape(attention_mask, 2)
                    attention_mask = attention_mask.view(
                        concat([batch_size, 1, query_len, encoder_input_len]))
                    attention_mask = where(attention_mask == 0, float('-inf'),
                                           0.0)
                else:
                    attention_mask = expand_mask(attention_mask,
                                                 shape(query, 2))
            bias = attention_mask
            if self.position_embedding_type.is_alibi():
                alibi_biases = generate_alibi_biases(alibi_slopes, key_length)
                bias = alibi_biases if bias is None else bias + alibi_biases

            if self.relative_attention:
                query_length = shape(query, 2)
                if self.use_implicit_relative_attention:
                    relative_bias = compute_relative_bias(
                        query_length + key_length - 1,
                        key_length,
                        self.num_buckets,
                        self.max_distance,
                        False,  # bidirectional
                        self.rel_attn_table.value.transpose(1, 0),
                        tp_size=self.tp_size,
                        tp_group=self.tp_group,
                        tp_rank=self.tp_rank)
                else:
                    relative_bias = unsqueeze(self.rel_attn_table.value, 0)
                start = concat([0, 0, query_length + key_length - 2, 0])
                size = concat([
                    shape(relative_bias, 0),
                    shape(relative_bias, 1), 1, key_length
                ])
                relative_bias = slice(relative_bias, start, size)

            key = key.permute([0, 1, 3, 2])
            with precision('float32'):
                if norm_before_bmm1:
                    # Apply norm on query earlier to prevent matmul fp16 overflow.
                    query /= (self.q_scaling * self.norm_factor)
                if trt_gte_10() or self.position_embedding_type.is_alibi():
                    attention_scores = matmul(query, key)
                else:
                    # For TRT 9.x, OOTB need this WAR to fuse mha.
                    attention_scores = matmul(cast(query, 'float32'),
                                              cast(key, 'float32'))
                if not norm_before_bmm1:
                    attention_scores = attention_scores / (self.q_scaling *
                                                           self.norm_factor)
                if self.max_attn_value > 0:
                    attention_scores = self.max_attn_value * ACT2FN['tanh'](
                        attention_scores / self.max_attn_value)

                if self.attention_mask_type in [
                        AttentionMaskType.causal,
                        AttentionMaskType.bidirectional
                ] and not self.cross_attention:

                    bias = generated_mask if bias is None else bias + generated_mask

                if bias is not None:
                    bias = cast(bias, attention_scores.dtype)
                    attention_scores = attention_scores + bias

                if self.relative_attention:
                    attention_scores = attention_scores + relative_bias

            attention_probs = softmax(attention_scores, dim=-1)

            if trt_gte_10() or self.position_embedding_type.is_alibi():
                # For trt_version() == 9.x and pos_embed == alibi, TRT has gpu buffer management issues. Need this WAR to avoid peak gpu mem regression.
                # A dummy reshape WAR for mha fusion for 10.0
                attention_probs = attention_probs.view(
                    concat([
                        shape(attention_probs, 0),
                        shape(attention_probs, 1),
                        shape(attention_probs, 2),
                        shape(value, 2)
                    ]))
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute([0, 2, 1, 3])
            else:
                # For TRT 9.x, need this WAR to fuse mha.
                context = matmul(attention_probs,
                                 cast(value, 'float32')).permute([0, 2, 1, 3])
                if context.dtype != value.dtype:
                    context = cast(context, value.dtype)
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1), self.attention_hidden_size
                ]))

        dense_lora_params = None
        if lora_layer_params is not None:
            dense_lora_params = lora_layer_params.get_runtime_params(
                0, "attn_dense")

        if self.inner_layernorm is not None:
            context = self.inner_layernorm(context)
        context = self.dense(context,
                             lora_runtime_params=dense_lora_params,
                             reduce_fusion_params=reduce_fusion_params)

        if use_cache:
            return (context, past_key_value)
        else:
            return context
        '''

        if use_cache:
            # TODO finish past key value 填充 
            return hidden_states,hidden_states
        else:
            return hidden_states

    # def set_rel_attn_table(self, max_seq_len, precomputed_relative_attention):
    #     self.rel_attn_table = Parameter(shape=(self.num_attention_heads,
    #                                            max_seq_len + 1,
    #                                            max_seq_len + 1),
    #                                     dtype=self.dtype)
    #     self.rel_attn_table.value = precomputed_relative_attention
