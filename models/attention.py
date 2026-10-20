import os
import math
import torch
import torch.nn as nn
import numpy as np

# repo root: <root>/models/attention.py -> <root>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# distance maps live next to the repo; allow overriding for read-only installs
_DIST_DIR = os.environ.get('MVPI_DISTANCES_DIR', os.path.join(_REPO_ROOT, 'distances'))


def _create_distance_map(seq_len, path_x, path_y):
    """Create (and cache) the squared position-distance maps for ``seq_len``."""
    width = int(math.sqrt(seq_len))
    if width * width != seq_len:
        raise ValueError('seq_len must be a perfect square, got {}'.format(seq_len))
    idx = np.arange(seq_len)
    h = (idx // width).astype(np.float32)
    w = (idx % width).astype(np.float32)
    distances_x = (w[:, None] - w[None, :]) ** 2
    distances_y = (h[:, None] - h[None, :]) ** 2
    np.save(path_x, distances_x.astype(np.float32))
    np.save(path_y, distances_y.astype(np.float32))
    return distances_x.astype(np.float32), distances_y.astype(np.float32)


def _load_distance_maps(seq_len):
    """Load distances_{x,y}_{seq_len}.npy, generating them on the fly if absent.

    Paths are resolved from the repository root (or $MVPI_DISTANCES_DIR) so the
    model builds correctly no matter what the current working directory is
    (multi-GPU spawn workers, cron jobs, running from another folder, ...).
    """
    os.makedirs(_DIST_DIR, exist_ok=True)
    path_x = os.path.join(_DIST_DIR, 'distances_x_{}.npy'.format(seq_len))
    path_y = os.path.join(_DIST_DIR, 'distances_y_{}.npy'.format(seq_len))
    if os.path.exists(path_x) and os.path.exists(path_y):
        return np.load(path_x), np.load(path_y)
    print('[Attention2D] generating distance maps for seq_len={} in {}'.format(seq_len, _DIST_DIR))
    return _create_distance_map(seq_len, path_x, path_y)


class Attention2D(nn.Module):
    def __init__(self, 
                 seq_len=256,
                 d_model=512,
                 num_heads=8,
                 scale_factor=None,
                 dropout=0.0,
                 device=torch.device('cuda'),
                 mask_neighbor_size=9): # Default to 9 (a 3x3 neighborhood).
        super(Attention2D, self).__init__()
        
        self.seq_len = seq_len
        self.width = int(math.sqrt(seq_len))
        self.scale_factor = scale_factor or 1. / math.sqrt(d_model // num_heads)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.device = device
        
        
        self.mask_neighbor_size = mask_neighbor_size
        if self.mask_neighbor_size > 0:
            self.register_buffer('attn_mask', self._generate_neighbor_mask(self.width, mask_neighbor_size))

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.sigma_projection = nn.Linear(d_model, 2 * num_heads)
        self.out_projection = nn.Linear(d_model, d_model)
        
        distances_x, distances_y = _load_distance_maps(seq_len)
        distances_x = torch.from_numpy(distances_x)
        distances_y = torch.from_numpy(distances_y)
        self.distances_x = distances_x.to(device)
        self.distances_y = distances_y.to(device)

    def _generate_neighbor_mask(self, length, mask_size):
        """Generate an (L, L) mask for each local neighborhood."""
        L = length * length
        mask = torch.zeros(L, L)
        
        # Convert sequence indices to (h, w) coordinates.
        idx = torch.arange(L)
        h = idx // length
        w = idx % length
        
        # Compute coordinate differences.
        # (L, 1) - (1, L) -> (L, L)
        h_diff = torch.abs(h.unsqueeze(1) - h.unsqueeze(0))
        w_diff = torch.abs(w.unsqueeze(1) - w.unsqueeze(0))
        
        # Compute the Chebyshev distance.
        dist = torch.maximum(h_diff, w_diff)
        
        # Determine the masking radius.
        # mask_size = 1 -> radius = 0 (mask the token itself only).
        # mask_size = 9 (3x3) -> radius = 1.
        # mask_size = 25 (5x5) -> radius = 2.
        radius = int((math.sqrt(mask_size) - 1) / 2)
        
        # Set distances within the radius to -inf (zero after softmax).
        mask[dist <= radius] = float('-inf')
        
        return mask

    def forward(self, query, key, value, return_attention=True, use_mask=False):
        """
        Args:
            query: [B, L, C]
            key: [B, S, C]
            value: [B, S, C]
            return_attention: Whether to return the attention map and target.
            use_mask: Enable the mask when mask_neighbor_size > 0 and L == S.
        """
        B, L, _ = query.shape
        _, S, _ = key.shape

        if return_attention: 
            sigma = self.sigma_projection(query).view(B, L, self.num_heads, -1)
        query = self.query_projection(query).view(B, L, self.num_heads, -1)
        key = self.key_projection(key).view(B, S, self.num_heads, -1)
        value = self.value_projection(value).view(B, S, self.num_heads, -1)
        
        scores = torch.einsum("blhe,bshe->bhls", query, key)
    
        # attn: (N, n_heads, L, L)
        attn = self.scale_factor * scores
        
        # Apply the neighborhood mask.
        
        if self.mask_neighbor_size > 0 and L == S and use_mask:
            attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(0)

        if return_attention:
            sigma = sigma.transpose(1, 2)
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            
            sigma1 = sigma[:, :, :, 0]
            sigma2 = sigma[:, :, :, 1]
            sigma1 = sigma1.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)
            sigma2 = sigma2.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)
            
            distances_x = self.distances_x.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
            distances_y = self.distances_y.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(self.device)
            
            target = 1.0 / (2 * math.pi * sigma1 * sigma2) * torch.exp(-distances_y / (2 * sigma1 ** 2) -distances_x / (2 * sigma2 ** 2))
        else:
            target = None

        softmax_scores = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", softmax_scores, value)

        out = out.contiguous().view(B, L, -1)
        out = self.out_projection(out)

        if return_attention:
            return out, softmax_scores, target
        else:
            return out, None, None
