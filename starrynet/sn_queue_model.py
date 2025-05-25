import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class StateEmbedding(nn.Module):
    def __init__(self, num_states: int = 125, state_dim: int = 35, emb_size: int = 64):
        super().__init__()
        self.projection = nn.Linear(state_dim, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(num_states + 1, emb_size))
    
    def forward(self, x: Tensor) :
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 64, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 64,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))        

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# Average pooling for classification head
# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size: int = 64, n_classes: int = 4):
#         super().__init__(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size), 
#             nn.Linear(emb_size, n_classes),
#             nn.Softmax(dim=-1)
#         )

# just use the cls token for classification
class ClassificationHead(nn.Module):
    def __init__(self, emb_size:int=64, n_classes:int=4):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.fc   = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        # x.shape == [batch, num_tokens+1, emb_size]
        cls_token = x[:, 0, :]
        other_tokens = x[:, 1:, :]
        out = self.norm(cls_token)
        logits = self.fc(out)
        return logits, other_tokens # return logits and other tokens for future use


class ViT(nn.Sequential):
    def __init__(self,     
                num_states: int = 5,  # number of tokens
                state_dim: int = 30,  # dimension of each token
                emb_size: int = 64,   # embedding dimension
                depth: int = 12,      # num of transformer encoder blocks
                n_classes: int = 4,
                **kwargs):
        super().__init__(
            StateEmbedding(num_states, state_dim, emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes=n_classes)
        )            