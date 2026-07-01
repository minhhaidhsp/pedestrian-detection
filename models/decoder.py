"""RT-DETR transformer decoder, copied and adapted from RT-DETR.

Source: https://github.com/lyuwenyu/RT-DETR (rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py
and utils.py), commit at clone time 2026-07-01. License: Apache License 2.0
(see D:\\Projects\\_external\\RT-DETR\\LICENSE; original authorship notice
"by lyuwenyu" retained per the license's attribution requirement).

Changes vs. upstream (Giai doan D, Buoc 5):
- Dropped the `@register`/`src.core` YAML-config-registry integration.
- # DESIGN DECISION: contrastive denoising (DN) training is removed entirely
  -- no `num_denoising`/`label_noise_ratio`/`box_noise_scale` params, no
  `denoising_class_embed`, no `get_contrastive_denoising_training_group` call
  in forward. The paper's Eq. 7-8 loss has no DN term; DN is an RT-DETR
  training-convergence trick orthogonal to what's being reproduced here.
- # DESIGN DECISION: upstream RT-DETR actually has TWO DISTINCT auxiliary-loss
  mechanisms, which this project treats differently. Do not conflate them:
    (1) Per-DECODER-LAYER aux loss (supervises every intermediate decoder
        layer's box/cls output, in addition to the final layer). This is an
        RT-DETR convergence-speed trick with no counterpart in the paper's
        Eq. 7-8 -- REMOVED per Quyet dinh D. `TransformerDecoder.forward`
        still runs all decoder layers (needed for the iterative
        box-refinement recurrence itself -- core architecture, not an
        aux-loss artifact) but only computes/returns classification logits
        for the FINAL layer. `RTDETRTransformer.forward`'s main output
        ("pred_logits"/"pred_boxes") is the final layer only, not a list of
        per-layer outputs. Per-layer classification heads (`dec_score_head`)
        were only ever needed to supervise the removed per-layer aux loss, so
        they're collapsed to one shared final score head (kept as a
        ModuleList per-layer only for the box-refinement `dec_bbox_head`,
        since every layer's bbox head is actually used in the recurrence,
        not just for aux supervision).
    (2) Encoder QUERY-SELECTION loss (supervises the encoder's own top-k
        proposals, `enc_topk_logits`/`enc_topk_bboxes`, returned by forward()
        as "enc_pred_logits"/"enc_pred_boxes"). This is KEPT, and is NOT
        optional the way (1) is: `enc_score_head`'s output only ever feeds
        `torch.topk(...).indices` (query selection) -- topk's indices are
        not differentiable w.r.t. which elements got picked, so without some
        loss directly supervising enc_topk_logits/enc_topk_bboxes,
        `enc_score_head` (and by the same argument, part of the signal to
        `enc_output`/`enc_bbox_head`) would receive literally zero gradient,
        forever, regardless of any `.detach()` choices elsewhere -- verified
        empirically in Buoc 5 (test_decoder.py backward test). The paper
        does not define its own formula for this term (query selection is an
        RT-DETR implementation detail, not one of the paper's contributions),
        so it is supervised with the SAME SetCriterion/weight_dict as the
        main decoder loss, per RT-DETR upstream's own convention. Total
        training loss = Eq. 8 decoder loss + this encoder query-selection
        loss (see models/fa_promptdetr.py, Buoc 6).
- # DESIGN DECISION: removed the `.detach()` calls upstream applied to the
  inter-layer reference-point recurrence (`ref_points_detach` in
  `TransformerDecoder.forward`). Upstream paired this detach with the
  per-decoder-layer aux loss (1) above -- the detach existed to stop that
  aux-supervised branch from *also* being trained through the main
  recurrence path, not to gate training entirely. With that aux loss removed
  but the detach left in place, this was verified (Buoc 5 test_decoder.py,
  item g) to leave every non-final `dec_bbox_head[i]` with **zero
  gradient** -- i.e. permanently untrained dead parameters, since they had no
  other path to the loss. Removing the detach lets gradients from the final
  loss flow through the whole box-refinement recurrence instead. Note this
  is unrelated to the encoder query-selection `.detach()` question -- that
  path never had a stray detach to begin with; it needed loss (2) above
  instead, since detach-removal alone cannot fix a non-differentiable
  `torch.topk` index-selection.
- # DESIGN DECISION: `feat_channels`/`feat_strides` are reconfigured for this
  project's 3-scale [P2, P3, P4] decoder input (channels [128,128,128] --
  HybridEncoder's hidden_dim output -- strides [4,8,16], P5 dropped per
  Quyet dinh A). The caller (models/fa_promptdetr.py, Buoc 6) is responsible
  for slicing HybridEncoder's 4-scale [P2,P3,P4,P5] output down to
  `encoder_outs[:3]` before calling this decoder -- this module itself stays
  a generic N-scale decoder (like upstream) and does not know about "P5" as a
  special concept, it just requires `len(feats) == len(feat_channels)`.
  Verified (Buoc 5 task item 2b): anchor generation (`_generate_anchors`) and
  deformable attention (`deformable_attention_core_func`, grid_sample-based)
  read spatial shapes from actual runtime tensor shapes or from
  `self.feat_strides` generically -- there is no hardcoded assumption of a
  minimum stride of 8 anywhere in this file. The one calibration caveat (not
  a code bug) is `_generate_anchors`' `grid_size=0.05` constant, which sets
  the assumed anchor box size at the *finest* level and was tuned upstream
  assuming that finest level is stride 8; with stride 4 now finest, this
  constant is unverified for our setup. Flagged for the user, not changed
  here.
- `hidden_dim` defaults to 128 (project config) instead of upstream's 256;
  it was already a plain constructor parameter propagated everywhere
  upstream, so this required no structural change (verified in this file's
  actual code below, not just by inspection of the unmodified upstream copy).
- `num_classes` defaults to 1 ("person"), consistent with Giai doan C's
  sigmoid-focal / no-background-class SetCriterion: `pred_logits` is
  `[B, num_queries, 1]`, not `[B, num_queries, num_classes+1]`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ["RTDETRTransformer"]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """value: [bs, value_length, n_head, c]. value_spatial_shapes: [n_levels, 2].
    sampling_locations: [bs, query_length, n_head, n_levels, n_points, 2].
    attention_weights: [bs, query_length, n_head, n_levels, n_points].
    Returns: [bs, Length_query, C].
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, Len_q, n_levels * n_points)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )
    return output.permute(0, 2, 1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else getattr(F, act) if isinstance(act, str) else act

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, dropout=0.0, activation="relu", n_levels=4, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self,
        tgt,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory, memory_spatial_shapes, memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))
        return tgt


class TransformerDecoder(nn.Module):
    """Iterative box-refinement decoder stack. No aux-loss output (see module docstring)."""

    def __init__(self, hidden_dim, decoder_layer, num_layers):
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = tgt
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output, ref_points_input, memory, memory_spatial_shapes, memory_level_start_index, attn_mask, memory_mask, query_pos_embed
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if i == self.num_layers - 1:
                final_logits = score_head(output)
                final_bboxes = inter_ref_bbox
                break

            ref_points_detach = inter_ref_bbox

        return final_bboxes, final_logits


class RTDETRTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        hidden_dim: int = 128,
        num_queries: int = 300,
        position_embed_type: str = "sine",
        feat_channels: list = [128, 128, 128],
        feat_strides: list = [4, 8, 16],
        num_levels: int = 3,
        num_decoder_points: int = 4,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        learnt_init_query: bool = False,
        eval_spatial_size: tuple | None = None,
        eps: float = 1e-2,
    ):
        super().__init__()
        assert position_embed_type in ["sine", "learned"], f"unsupported position_embed_type {position_embed_type}"
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size

        self._build_input_proj_layer(feat_channels)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers)

        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # DESIGN DECISION: a single shared final score head, not one per decoder
        # layer -- per-layer score heads upstream existed only to supervise the
        # now-removed aux losses. dec_bbox_head stays per-layer: every layer's
        # bbox head is used in the iterative refinement recurrence itself.
        self.dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)]
        )

        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.dec_score_head.bias, bias)
        for reg_ in self.dec_bbox_head:
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False), nn.BatchNorm2d(self.hidden_dim)
                )
            )
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim),
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        assert len(feats) == len(self.input_proj), (
            f"RTDETRTransformer expects exactly {len(self.input_proj)} feature maps "
            f"(matching feat_channels); got {len(feats)}. Caller must slice the "
            f"encoder's output to the scales this decoder is configured for "
            f"(e.g. encoder_outs[:3] to drop P5)."
        )
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            level_start_index.append(h * w + level_start_index[-1])

        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype), torch.arange(end=w, dtype=dtype), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes):
        bs, _, _ = memory.shape
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )

        # enc_topk_bboxes/enc_topk_logits: the encoder's own top-k proposals,
        # kept (and returned by forward() as enc_pred_boxes/enc_pred_logits)
        # specifically so they can be supervised by a dedicated encoder
        # query-selection loss -- see module docstring, "encoder query-selection
        # loss" design decision. Without that supervision enc_score_head would
        # never receive gradient at all (torch.topk's indices are not
        # differentiable w.r.t. which elements were selected).
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits

    def forward(self, feats: list[torch.Tensor]) -> dict:
        """feats: list of exactly `len(feat_channels)` tensors (e.g. [P2, P3, P4],
        HybridEncoder output with P5 already sliced off by the caller).

        Returns a dict with:
        - "pred_logits" [B, num_queries, num_classes], "pred_boxes" [B, num_queries, 4]:
          the final decoder layer's output -- supervise with the paper's Eq. 7-8
          SetCriterion (Giai doan C).
        - "enc_pred_logits", "enc_pred_boxes" (same shapes): the encoder's own
          top-k query-selection proposals -- supervise with the SAME SetCriterion
          (RT-DETR upstream convention; the paper does not define a separate
          formula for this term, since query selection is an RT-DETR
          implementation detail, not one of the paper's own contributions).
          This is structurally required (see _get_decoder_input), NOT the
          per-decoder-layer aux loss that Quyet dinh D removed.
        """
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes
        )

        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
        )

        return {
            "pred_logits": out_logits,
            "pred_boxes": out_bboxes,
            "enc_pred_logits": enc_topk_logits,
            "enc_pred_boxes": enc_topk_bboxes,
        }
