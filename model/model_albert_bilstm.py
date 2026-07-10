# model_main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel

from model.model_component import DilatedConvResidualBlockWithAttention, MultiHeadAttentionFusion


class AlbertBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        tfidf_dim,
        metadata_dim=4,
        dropout_rate=0.3,
        lstm_dropout=0.6,
        pos_weight=None,
        pretrained_model_name="albert-base-chinese",
        fusion_mode="aensf",
        use_dilated_conv=True,
        use_residual=True,
        use_channel_attention=True,
        use_metadata_features=True,
        use_tfidf_features=True,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.use_metadata_features = use_metadata_features
        self.use_tfidf_features = use_tfidf_features
        self.albert = AlbertModel.from_pretrained(pretrained_model_name)
        self.albert_embedding_size = self.albert.config.hidden_size
        self.lstm = nn.LSTM(
            self.albert_embedding_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=lstm_dropout)
        self.residual_block = DilatedConvResidualBlockWithAttention(
            in_channels=hidden_size * 2,
            out_channels=hidden_size,
            use_dilated_conv=use_dilated_conv,
            use_residual=use_residual,
            use_channel_attention=use_channel_attention,
        )

        self.multiattention_fusion = MultiHeadAttentionFusion(
            metadata_dim=metadata_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
        )
        self.tfidf_reduction = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.additional_dropout = nn.Dropout(p=dropout_rate)
        self.loss_fun = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.concat_projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.inner_fusion_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.outer_fusion_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
        )

        for _, param in self.albert.named_parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids,
        attention_mask,
        metadata_features,
        tfidf_features,
        labels=None,
        threshold=0.5,
        return_interpretation=False,
    ):
        albert_output = self.albert(input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_output, _ = self.lstm(albert_output)
        lstm_output = self.dropout(lstm_output)

        lstm_output = lstm_output.permute(0, 2, 1).unsqueeze(-1)
        residual_output = self.residual_block(lstm_output).squeeze(-1).permute(0, 2, 1)

        pooled_output = residual_output.mean(dim=1)
        metadata_attention = None

        if self.use_metadata_features:
            if return_interpretation:
                weighted_features, metadata_attention = self.multiattention_fusion(
                    metadata_features,
                    return_attention=True,
                )
            else:
                weighted_features = self.multiattention_fusion(metadata_features)
        else:
            weighted_features = torch.zeros_like(pooled_output)

        if self.use_tfidf_features:
            reduced_tfidf = self.tfidf_reduction(tfidf_features)
        else:
            reduced_tfidf = torch.zeros_like(pooled_output)

        combined_features = self._fuse_features(
            pooled_output,
            weighted_features,
            reduced_tfidf,
        )
        combined_features = self.additional_dropout(combined_features)
        logits = self.classifier(combined_features)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= threshold).long()

        outputs = {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "text_features": pooled_output,
            "metadata_features": weighted_features,
            "tfidf_features": reduced_tfidf,
            "fused_features": combined_features,
        }
        if metadata_attention is not None:
            outputs["metadata_attention_weights"] = metadata_attention
        if labels is not None:
            outputs["loss"] = self.loss_fun(logits, labels.float().unsqueeze(1))
        return outputs

    def _skip_connection(self, feature1, feature2):
        if feature1.size(-1) != feature2.size(-1):
            raise ValueError("Skip connection 要求两个特征维度一致。")
        return feature1 + feature2

    def _nested_skip_connection(self, feature1, feature2, feature3):
        # Inner skip: text feature + aligned metadata feature.
        inner_skip = self._skip_connection(feature1, feature2)
        inner_fused = self.inner_fusion_block(inner_skip)
        # Outer skip: inner fused feature + aligned TF-IDF feature.
        outer_skip = self._skip_connection(inner_fused, feature3)
        return self.outer_fusion_block(outer_skip)

    def _weighted_sum_fusion(self, feature1, feature2, feature3):
        weights = torch.softmax(self.fusion_weights, dim=0)
        return weights[0] * feature1 + weights[1] * feature2 + weights[2] * feature3

    def _concat_fusion(self, feature1, feature2, feature3):
        return self.concat_projection(torch.cat([feature1, feature2, feature3], dim=-1))

    def _fuse_features(self, text_feature, metadata_feature, tfidf_feature):
        if self.fusion_mode == "sf":
            return text_feature
        if self.fusion_mode == "ws":
            return self._weighted_sum_fusion(text_feature, metadata_feature, tfidf_feature)
        if self.fusion_mode == "c":
            return self._concat_fusion(text_feature, metadata_feature, tfidf_feature)
        if self.fusion_mode == "aensf":
            return self._nested_skip_connection(text_feature, metadata_feature, tfidf_feature)
        raise ValueError(f"不支持的 fusion_mode: {self.fusion_mode}")

