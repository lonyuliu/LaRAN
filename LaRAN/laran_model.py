"""Neural association model used by LaRAN network-level evaluation."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key_padding_mask=None):
        attn_output, _ = self.mha(query, query, query, key_padding_mask=key_padding_mask)
        return attn_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return self.dropout(x)


class LaRANAssociationBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        state_dim=6,
        meas_dim=4,
        rnn_layers=2,
        num_interactions: int = 2,
        transformer_layers: int = 2,
        num_newborns: int = 32,
        use_physical_consistency: bool = True,
        use_query_point_interaction: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0 <= num_interactions <= num_layers:
            raise ValueError(
                "num_interactions must be between 0 and num_layers"
            )

        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.num_interactions = num_interactions
        self.num_newborns = num_newborns
        self.use_physical_consistency = bool(use_physical_consistency)
        self.use_query_point_interaction = bool(use_query_point_interaction)
        self.tau = 0.1

        self.history_input_proj = nn.Linear(state_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=transformer_layers)
        self.track_history_encoder_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.track_encoder = nn.Linear(embed_dim, embed_dim)
        self.point_encoder = nn.Linear(meas_dim, embed_dim)
        self.newborn_queries = nn.Parameter(torch.randn(1, num_newborns, embed_dim))

        self.query_self_attn_layers = nn.ModuleList([SelfAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.track_update_layers = nn.ModuleList([CrossAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.point_update_layers = nn.ModuleList([CrossAttention(embed_dim, num_heads) for _ in range(num_layers)])

        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_point = nn.LayerNorm(embed_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.diff_scorer = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
        )

    def _compute_affinity_and_context(
        self,
        track_history,
        point_features,
        point_mask,
        track_inputs_raw,
        prev_hidden_states=None,
    ):
        batch_size, num_tracks, history_len, state_dim = track_history.shape

        if prev_hidden_states is None:
            if num_tracks > 0:
                history_flat = track_history.reshape(
                    batch_size * num_tracks, history_len, state_dim
                )
                history_embed = self.history_input_proj(history_flat)
                history_embed_pos = self.pos_encoder(history_embed)
                transformer_output = self.transformer_encoder(history_embed_pos)
                _, hidden = self.track_history_encoder_gru(transformer_output)
                target_query_state = hidden[-1]
            else:
                target_query_state = point_features.new_empty(
                    0, self.embed_dim
                )
        else:
            target_query_state = prev_hidden_states.reshape(
                batch_size * num_tracks, self.embed_dim
            )

        if num_tracks > 0:
            target_query_embed = self.track_encoder(
                target_query_state
            ).reshape(batch_size, num_tracks, self.embed_dim)
        else:
            target_query_embed = point_features.new_empty(
                batch_size, 0, self.embed_dim
            )

        newborn_queries_expanded = self.newborn_queries.expand(
            batch_size, -1, -1
        )
        all_queries = torch.cat([target_query_embed, newborn_queries_expanded], dim=1)

        point_keys = self.point_encoder(point_features)
        has_points = point_mask.any(dim=1)
        safe_point_mask = point_mask
        if point_mask.shape[1] > 0 and not has_points.all():
            safe_point_mask = point_mask.clone()
            safe_point_mask[~has_points, 0] = True

        interaction_rounds = self.num_interactions if self.use_query_point_interaction else 0
        for i in range(interaction_rounds):
            query_self = self.query_self_attn_layers[i](all_queries)
            all_queries = self.norm_query(all_queries + query_self)

            if point_keys.shape[1] > 0:
                track_update = self.track_update_layers[i](
                    all_queries,
                    point_keys,
                    point_keys,
                    key_padding_mask=~safe_point_mask,
                )
                track_update = track_update.masked_fill(
                    ~has_points[:, None, None], 0.0
                )
                all_queries = self.norm_query(
                    all_queries + track_update
                )

                point_update = self.point_update_layers[i](
                    point_keys, all_queries, all_queries
                )
                point_keys = self.norm_point(point_keys + point_update)

        track_proj = self.projection_head(all_queries)
        point_proj = self.projection_head(point_keys)

        track_norm = F.normalize(track_proj, p=2, dim=-1, eps=1e-7)
        point_norm = F.normalize(point_proj, p=2, dim=-1, eps=1e-7)

        cos_sim_matrix = torch.matmul(track_norm, point_norm.transpose(1, 2))

        if num_tracks > 0:
            phys_pred_expanded = torch.cat(
                [
                    track_inputs_raw,
                    track_inputs_raw.new_zeros(
                        batch_size,
                        self.num_newborns,
                        self.state_dim,
                    ),
                ],
                dim=1,
            )
        else:
            phys_pred_expanded = point_features.new_zeros(
                batch_size, self.num_newborns, self.state_dim
            )

        pos_diff = point_features[..., :3].unsqueeze(1) - phys_pred_expanded[..., :3].unsqueeze(2)
        raw_phys_dist = torch.sqrt(torch.sum(pos_diff ** 2, dim=-1) + 1e-7)

        if self.use_physical_consistency:
            diff_score = self.diff_scorer(pos_diff).squeeze(-1)
        else:
            diff_score = torch.zeros_like(cos_sim_matrix)
        final_affinity = (cos_sim_matrix + 0.5 * diff_score) / self.tau

        mask_expanded = ~point_mask.unsqueeze(1)
        final_affinity = final_affinity.masked_fill(mask_expanded, -1e4)

        return final_affinity, point_keys, target_query_state, raw_phys_dist

    def forward_association(
        self,
        track_history,
        point_features,
        point_mask,
        track_inputs_raw,
        prev_hidden_states=None,
        return_raw_phys_dist=False,
    ):
        _, num_tracks, _, _ = track_history.shape
        final_affinity, _, _, raw_phys_dist = self._compute_affinity_and_context(
            track_history=track_history,
            point_features=point_features,
            point_mask=point_mask,
            track_inputs_raw=track_inputs_raw,
            prev_hidden_states=prev_hidden_states,
        )

        association_scores = final_affinity[:, :num_tracks, :]
        association_phys_dist = raw_phys_dist[:, :num_tracks, :]

        if return_raw_phys_dist:
            return association_scores, association_phys_dist
        return association_scores


class LaRANAssociationNet(LaRANAssociationBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_target_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, 1),
        )

    def forward(
        self,
        track_history,
        point_features,
        point_mask,
        track_inputs_raw,
        prev_hidden_states=None,
        return_raw_phys_dist=False,
    ):
        _, num_tracks, _, _ = track_history.shape
        association_logits, point_features_encoded, _, physical_distance = self._compute_affinity_and_context(
            track_history=track_history,
            point_features=point_features,
            point_mask=point_mask,
            track_inputs_raw=track_inputs_raw,
            prev_hidden_states=prev_hidden_states,
        )

        association_logits = association_logits[:, :num_tracks, :]
        physical_distance = physical_distance[:, :num_tracks, :]
        targetness_logits = self.point_target_head(point_features_encoded).squeeze(-1)
        targetness_logits = targetness_logits.masked_fill(~point_mask, -1e4)

        if return_raw_phys_dist:
            return association_logits, physical_distance
        return association_logits, targetness_logits, physical_distance

    def load_full_state_dict(self, state_dict):
        """Load the association-compatible subset of a full checkpoint."""
        current_state = self.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        missing_keys = [key for key in current_state if key not in filtered_state]
        if missing_keys:
            missing_preview = ", ".join(missing_keys[:8])
            raise RuntimeError(
                "Checkpoint is missing LaRAN network parameters: "
                f"{missing_preview}"
            )
        return self.load_state_dict(filtered_state, strict=False)

DEFAULT_DEPLOY_MODEL_CLASS = LaRANAssociationNet


def build_deployment_model(checkpoint=None, map_location="cpu", eval_mode=True, **kwargs):
    model = DEFAULT_DEPLOY_MODEL_CLASS(**kwargs)
    if checkpoint is not None:
        if isinstance(checkpoint, (str, os.PathLike)):
            state_dict = torch.load(
                checkpoint,
                map_location=map_location,
                weights_only=True,
            )
        else:
            state_dict = checkpoint
        model.load_full_state_dict(state_dict)
    if eval_mode:
        model.eval()
    return model
