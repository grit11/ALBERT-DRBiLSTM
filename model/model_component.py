# model_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedConvLayer(nn.Module):
    """
    扩张卷积层，用于扩展感受野。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DilatedConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation_rate, dilation=dilation_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 交换维度，使得 [batch_size, seq_len, features] -> [batch_size, features, seq_len]
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 恢复回原来的维度
        return x

class AttentionFusion(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.attention_w = nn.Linear(feature_dims[-1], 1) 
    def forward(self, pooled_output, weighted_features, batch_sentiment, reduced_tfidf):
        device = pooled_output.device
        features = torch.stack((pooled_output, weighted_features, batch_sentiment, reduced_tfidf), dim = 1)
        attention_scores = self.attention_w(features).squeeze(-1)
        
        attention_weights = torch.softmax(attention_scores, dim = 1)
        combined_features = torch.sum(attention_weights.unsqueeze(-1) * features, dim = 1)
        return combined_features
   
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return torch.sigmoid(y)
     
class DilatedConvResidualBlockWithAttention(nn.Module):
    ''' 注意力机制的空洞卷积残差块'''
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation_rate=2,
        use_dilated_conv=True,
        use_residual=True,
        use_channel_attention=True,
    ):
        super(DilatedConvResidualBlockWithAttention, self).__init__()
        actual_dilation = dilation_rate if use_dilated_conv else 1
        self.use_residual = use_residual
        self.use_channel_attention = use_channel_attention
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=actual_dilation,
            dilation=actual_dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=actual_dilation,
            dilation=actual_dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = ChannelAttention(out_channels)
        if in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        attention_weight = None
        if self.use_channel_attention:
            attention_weight = self.attention(out)
            out = out * attention_weight.expand_as(out)
        if self.use_residual:
            out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    '''
    残差
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class RMSNorm(nn.Module):
    """RMSNorm实现"""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.weight


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation模块"""

    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        reduced_dim = hidden_dim // reduction
        self.fc1 = nn.Linear(hidden_dim, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, hidden_dim)

    def forward(self, x):
        w = x.mean(dim=1, keepdim=True)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MultiHeadAttentionBlock(nn.Module):
    """多头注意力Block，可选择rmsnorm或layernorm"""

    def __init__(self, embed_dim, num_heads, dropout=0.1, use_pre_norm=True, norm_type='layernorm'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)
        else:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'.")

        self.use_pre_norm = use_pre_norm
        self.res_scale_mha = nn.Parameter(torch.ones(1))
        self.res_scale_ffn = nn.Parameter(torch.ones(1))

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.se = SqueezeExcitation(embed_dim)

    def forward(self, x, mask=None):
        if self.use_pre_norm:
            x_norm = self.norm1(x)
        else:
            x_norm = x

        attn_out, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        attn_out = self.dropout(attn_out)
        x = x + self.res_scale_mha * attn_out

        if not self.use_pre_norm:
            x = self.norm1(x)

        if self.use_pre_norm:
            x_norm = self.norm2(x)
        else:
            x_norm = x

        ffn_out = self.ffn(x_norm)
        ffn_out = self.se(ffn_out)
        ffn_out = self.dropout(ffn_out)
        x = x + self.res_scale_ffn * ffn_out

        if not self.use_pre_norm:
            x = self.norm2(x)

        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = 1
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # probs = torch.sigmoid(inputs)
        # print("probs shape:", probs.shape)
        # # 如果 targets 是 one-hot 编码，将其转换为类别索引
        # if targets.dim() > 1:  # one-hot 编码
        #     targets = targets.argmax(dim=1)  # 转换为类别索引 [batch_size]
        # print("targets shape:", targets.shape)
        # pt = probs.gather(1, targets.unsqueeze(1))  # shape: [batch_size, 1] -> pt 是正确的类别概率
        # loss = - (1 - pt) ** self.gamma * torch.log(pt)
        
        probs = torch.sigmoid(inputs)[:, 1]
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = - (1 - pt) ** self.gamma * torch.log(pt)
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
        else:
            at = self.alpha

        F_loss = at * loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 自定义自注意力机制  目前使用:MultiheadAttention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)  # 定义 Dropout 层

    def forward(self, x):
        # x: [batch_size, num_features, feature_dim]
        attn_output, attn_weights = self.attn(x, x, x)  # Self-attention mechanism
        attn_output = self.layer_norm(attn_output + x) 
        # attn_output = self.dropout(attn_output)  # Dropout 操作
        return attn_output, attn_weights 


# class AttentionFusion(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(AttentionFusion, self).__init__()
#         self.query = nn.Linear(input_size, hidden_size)
#         self.key = nn.Linear(input_size, hidden_size)
#         self.value = nn.Linear(input_size, hidden_size)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, similarity_score, suspicion_score, timestamp):
#         combined_input = torch.cat((similarity_score, suspicion_score, timestamp), dim=-1)
#         query = self.query(combined_input)
#         key = self.key(combined_input)
#         value = self.value(combined_input)
#         query_key_dot = torch.matmul(query, key.transpose(-1, -2))
#         attention_weights = self.softmax(query_key_dot.unsqueeze(1))
#         weighted_value = torch.sum(attention_weights * value.unsqueeze(1), dim=1)
#         return weighted_value

class MultiHeadAttentionFusion(nn.Module):
    """
    对元数据特征做轻量自注意力编码。

    输入张量形状为 [batch_size, metadata_dim]，其中 metadata_dim 对应：
    [similarity, suspicion, timestamp, sentiment]。
    """

    def __init__(self, metadata_dim, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttentionFusion, self).__init__()
        self.metadata_dim = metadata_dim
        self.hidden_size = hidden_size
        self.feature_projection = nn.Linear(1, hidden_size)
        self.feature_type_embedding = nn.Parameter(torch.randn(metadata_dim, hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, metadata_features, return_attention=False):
        if metadata_features.dim() != 2:
            raise ValueError(
                f"metadata_features 应为二维张量 [batch_size, metadata_dim]，实际得到 {metadata_features.shape}"
            )
        if metadata_features.size(1) != self.metadata_dim:
            raise ValueError(
                f"metadata_features 最后一维应为 {self.metadata_dim}，实际得到 {metadata_features.size(1)}"
            )

        x = metadata_features.unsqueeze(-1)
        x = self.feature_projection(x)
        x = x + self.feature_type_embedding.unsqueeze(0)

        attn_output, attn_weights = self.attention(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        pooled = x.mean(dim=1)
        if return_attention:
            return pooled, attn_weights
        return pooled


