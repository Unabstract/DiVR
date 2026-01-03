import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, return_atts=False):
        """
        :param q: Query tensor (batch_size, query_length, d_model)
        :param kv: Key-Value tensor (batch_size, key_value_length, d_model)
        :param return_atts: Boolean to return attention weights
        :return: Output tensor and optionally attention weights
        """
        attn_output, attn_weights = self.attention(q.permute(1, 0, 2), kv.permute(1, 0, 2), kv.permute(1, 0, 2))
        attn_output = attn_output.permute(1, 0, 2)
        output = q + self.dropout(attn_output)
        return (output, attn_weights) if return_atts else output


class PositionwiseFeedForward(nn.Module):
    """
    A two-feed-forward-layer module
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        return x + residual


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module
    """

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class SelfAttention(nn.Module):
    """
    Self-Attention with normalization and dropout
    """

    def __init__(self, num_heads, num_channels, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.self_att = MultiHeadAttention(num_heads, num_channels, num_channels, dropout)

    def forward(self, enc_input, return_atts=False):
        x = self.norm(enc_input)
        return self.self_att(x, x, return_atts)


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention Layer composed of Self-Attention and Feed-Forward layers
    """

    def __init__(self, num_heads, num_q_channels, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.self_att = SelfAttention(num_heads, num_q_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, qkv, return_atts=False):
        if return_atts:
            r = self.self_att(qkv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.self_att(qkv))


class CrossAttention(nn.Module):
    """
    Cross-Attention with normalization and dropout
    """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.cross_att = MultiHeadAttention(num_heads, num_q_channels, num_kv_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        q = self.q_norm(q)
        kv = self.kv_norm(kv)
        return self.cross_att(q, kv, return_atts)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer composed of Cross-Attention and Feed-Forward layers
    """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_att = CrossAttention(num_heads, num_q_channels, num_kv_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        if return_atts:
            r = self.cross_att(q, kv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.cross_att(q, kv))


class PerceiveEncoder(nn.Module):
    """
    Encoder model with self-attention mechanism
    """

    def __init__(self, n_input_channels, n_latent, n_latent_channels=512,
                 n_cross_att_heads=1, n_self_att_heads=8, n_self_att_layers=6,
                 dropout=0.1, n_position=400):
        super().__init__()

        self.position_enc = PositionalEncoding(n_input_channels, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.cross_att = CrossAttentionLayer(
            num_q_channels=n_latent_channels,
            num_kv_channels=n_input_channels,
            num_heads=n_cross_att_heads,
            dropout=dropout
        )
        self.self_att = nn.Sequential(*[
            SelfAttentionLayer(num_q_channels=n_latent_channels, num_heads=n_self_att_heads, dropout=dropout)
            for _ in range(n_self_att_layers)
        ])
        self.latent = nn.Parameter(torch.empty(n_latent, n_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, feats_embedding):
        """
        :param feats_embedding: Input features (batch_size, sequence_length, dim)
        :return: Latent representation
        """
        b = feats_embedding.shape[0]
        enc_output = self.dropout(self.position_enc(feats_embedding))
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        x_latent = self.cross_att(x_latent, enc_output)
        x_latent = self.self_att(x_latent)
        return x_latent


class PerceiveDecoder(nn.Module):
    """
    Decoder model with cross-attention mechanism
    """

    def __init__(self, n_query, n_query_channels, n_latent_channels, n_cross_att_heads=1, dropout=0.1):
        super().__init__()

        self.cross_att = CrossAttentionLayer(
            num_q_channels=n_query_channels,
            num_kv_channels=n_latent_channels,
            num_heads=n_cross_att_heads,
            dropout=dropout
        )
        self.query_latent = nn.Parameter(torch.empty(n_query, n_query_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.query_latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, query, latent, return_atts=False):
        """
        :param query: Query tensor (batch_size, num_queries, query_dim)
        :param latent: Latent tensor (batch_size, sequence_length, latent_dim)
        :param return_atts: Boolean to return attention weights
        :return: Decoded output and optionally attention weights
        """
        b = query.shape[0]
        x_latent = repeat(self.query_latent, "... -> b ...", b=b)
        query_embedding = query + x_latent
        return self.cross_att(query_embedding, latent, return_atts)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    motion_feats = torch.randn((3, 6, 256))
    gaze_feats = torch.randn((3, 6, 256))

    gaze_encoder = PerceiveEncoder(n_input_channels=256, n_latent=10, n_latent_channels=512)
    gaze_embedding = gaze_encoder(gaze_feats)
    print(gaze_embedding.shape)

    motion_encoder = PerceiveEncoder(n_input_channels=256, n_latent=10, n_latent_channels=512)
    motion_embedding = motion_encoder(motion_feats)
    print(motion_embedding.shape)

    decoder = PerceiveDecoder(n_query=10, n_query_channels=512, n_latent_channels=512)
    print(decoder(gaze_embedding, motion_embedding).shape)
