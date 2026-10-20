import math
import torch
import torch.nn as nn

from .attention import Attention2D
from .embedding import Embedding2D

class FeatureJitter(nn.Module):
    def __init__(self, scale=0.05): 
        super().__init__()
        self.scale = scale

    def forward(self, feature_tokens):
        if self.training:
            # DINOv3 features are approximately on the unit sphere (norm = 1).
            # Scale noise by the L2 norm instead of dividing by the feature dimension.
            feature_norms = feature_tokens.norm(dim=-1, keepdim=True).detach() 
            jitter = torch.randn_like(feature_tokens) * feature_norms * self.scale
            # Add the noise to the features.
            feature_tokens = feature_tokens + jitter
        return feature_tokens

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, d_model, d_feed_foward=None, dropout=0.1):
        super().__init__()
        d_feed_foward = d_feed_foward or 4 * d_model
        self.self_attention = self_attention
        self.ffn = Mlp(d_model, d_feed_foward, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention repairs corrupted features using their context.
        # The attention module may return a tuple; use its output tensor.
        sa_out = self.self_attention(x, x, x, return_attention=False, use_mask=False)
        if isinstance(sa_out, tuple):
            sa_out = sa_out[0]
            
        new_x = x + self.dropout(sa_out)

        y = x = self.norm1(new_x)
        y = self.ffn(y)
        out = self.norm2(x + y)

        return out

class Encoder(nn.Module):
    def __init__(self, encode_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.encode_layers = nn.ModuleList(encode_layers)
        self.norm = norm_layer

    def forward(self, x):
        for layer in self.encode_layers:
            x = layer(x)

        if self.norm is not None:
            x = self.norm(x)
        return x

class MVPI(nn.Module):
    def __init__(self,
                 seq_len,
                 in_channels,
                 out_channels,
                 d_model=512,
                 n_heads=4,
                 n_layers=3,
                 d_feed_foward_scale=4,
                 dropout=0.0,
                 args=None,
                 layer_name=None,
                 jitter_scale=0.05):
        super(MVPI, self).__init__()

        d_feed_foward = d_model * d_feed_foward_scale
        device = getattr(args, 'device', torch.device('cuda')) if args is not None else torch.device('cuda')

        self.embedding = Embedding2D(in_channels, d_model, dropout, device=device)
        self.jitter = FeatureJitter(scale=jitter_scale)

        # Lightweight encoder containing the self-attention layers used for reconstruction.
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attention2D(seq_len, d_model, n_heads, dropout=dropout, device=device),
                    d_model,
                    d_feed_foward,
                    dropout=dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels, bias=True)

    def forward(self, x, train=True):
        emb = self.embedding(x)
        
        if train:
            emb = self.jitter(emb)
            
        emb = self.encoder(emb)
        x_out = self.projection(emb)

        # Project the reconstructed features back to the output dimension.
        return x_out