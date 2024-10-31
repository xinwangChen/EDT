import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import random

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
    def get_flops(self):
        #fc1 [1,frequency_embedding_size]x[frequency_embedding_size,hidden_size]
        fc1 = self.frequency_embedding_size * self.hidden_size
        #fc2 [1,hidden_size]x[hidden_size,hidden_size]
        fc2 = self.hidden_size * self.hidden_size
        return fc1+fc2

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = (dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_flops(self):
        return 0

class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.seqlen= self.window_size[0]*self.window_size[1]
        self.num_heads = num_heads
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)
        self.relative_position_index.requires_grad = False
        # self.register_buffer("relative_position_index",relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.seqlen,self.seqlen, self.num_heads)  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()

def euclidean_distances(row_n,col_n):
    # 使用 meshgrid 生成行和列索引
    row_indices, col_indices = torch.meshgrid(torch.arange(row_n), torch.arange(col_n), indexing='ij')
    # 展开行和列索引为一维向量
    row_indices_flat = row_indices.flatten()
    col_indices_flat = col_indices.flatten()
    # 计算行和列的差的平方，利用广播
    row_diffs_squared = (row_indices_flat[:, None] - row_indices_flat) ** 2
    col_diffs_squared = (col_indices_flat[:, None] - col_indices_flat) ** 2
    # 计算欧式距离
    euclidean_distances = torch.sqrt(row_diffs_squared + col_diffs_squared)
    return euclidean_distances

def modulation_matrix(n,k=47):
    distances_matrix = euclidean_distances(n,n)
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    T = 4*(n-1)*math.sqrt(2) # T = 4*(n-1)*sqrt(2)
    f= 2*math.pi/T #T=2*PI/f
    matrix = torch.cos(f * distances_matrix) # cos(f*d)
    # R = sqrt(4+N^2) is the distance between token 0 and token 47 in 16x16 tokens grid
    if k%n==0:
        k=k-1
    matrix = torch.exp(matrix)/2 # 0.5*exp(cos(f*d))
    bound = matrix[0][k] # bound = 0.5*exp(cos(f*R))
    matrix[matrix<bound] = 0 # d<R part
    return matrix

#use for 512×512
# def modulation_matrix(n,k=47):
#     distances_matrix = euclidean_distances(n,n)
#     i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
#     T = 4*(n-1)*math.sqrt(2) # T = 4*(n-1)*sqrt(2)
#     f= 2*math.pi/T #T=2*PI/f
#     matrix = 1.3*torch.cos(f * distances_matrix) # 1.3cos(f*d)
#     # R = sqrt(4+N^2) is the distance between token 0 and token 47 in 16x16 tokens grid
#     if k%n==0:
#         k=k-1
#     bound = matrix[0][k] # bound = 1.3*cos(f*R))
#     matrix[matrix<bound] = 0 # d<R part
#     return matrix

class AttentionBlock(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,local=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.local = local
        if self.local:
            self.modulation_matrix = modulation_matrix(input_resolution, 3*input_resolution).unsqueeze(0).unsqueeze(0)

        # define a parameter table of relative position bias
        self.rel_pos_bias = RelativePositionBias(window_size=[input_resolution, input_resolution], num_heads=num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + self.rel_pos_bias().unsqueeze(0)
        
        if self.local and not self.training:
            attn = self.softmax(attn)*(self.modulation_matrix.to(attn.device))
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                                 Core EDT Model                                #
#################################################################################
class EDTBlock(nn.Module):
    """
    A EDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, input_resolution=16, cond_dim=384, local=False, **block_kwargs):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = AttentionBlock(hidden_size, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=True,local=local,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=self.mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        H= self.input_resolution
        W = self.input_resolution
        B, L, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        shortcut = x
        x=modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.attn(x)
        x = shortcut + gate_msa.unsqueeze(1) * x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def get_flops(self):
        #adaln [1,cond_dim]x[cond_dim,6 * hidden_size]
        adaln = self.cond_dim * 6 * self.hidden_size
        #kqv [input_resolution^2,hidden_size]x[hidden_size,3*hidden_size]
        kqv = (self.input_resolution*self.input_resolution)*self.hidden_size*(3*self.hidden_size)
        #att [num_heads,input_resolution^2,hidden_size//num_heads]x[num_heads,hidden_size//num_heads,input_resolution^2]
        dim = self.hidden_size//self.num_heads
        att = self.num_heads * (self.input_resolution*self.input_resolution) * dim * (self.input_resolution*self.input_resolution)
        #new_x [num_heads,input_resolution^2,input_resolution^2]x[num_heads,input_resolution^2,hidden_size//num_heads]
        new_x = self.num_heads * (self.input_resolution*self.input_resolution) * (self.input_resolution*self.input_resolution) * dim
        #proj [input_resolution^2, hidden_size]x[hidden_size,hidden_size]
        proj = (self.input_resolution*self.input_resolution) *self.hidden_size * self.hidden_size
        #fc1 [input_resolution^2, hidden_size] x [hidden_size, mlp_hidden_dim]
        fc1 = (self.input_resolution*self.input_resolution) *self.hidden_size * self.mlp_hidden_dim
        #fc2 [input_resolution^2, mlp_hidden_dim] x [mlp_hidden_dim, hidden_size]
        fc2 = (self.input_resolution*self.input_resolution) * self.mlp_hidden_dim * self.hidden_size
        return adaln+kqv+att+new_x+proj+fc1+fc2



class FinalLayer(nn.Module):
    """
    The final layer of EDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)#表示只进行归一化，部进行仿射变换
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DownSample(nn.Module):
    def __init__(self, hidden_size, out_hidden_size, input_resolution, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.input_resolution = input_resolution
        self.hidden_size = hidden_size
        self.output_resolution = int(input_resolution/2)
        self.out_hidden_size = out_hidden_size
        self.norm = nn.LayerNorm(4*hidden_size, elementwise_affine=False, eps=1e-6)#表示只进行归一化，部进行仿射变换
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 8 * hidden_size, bias=True)
        )
        self.reduction = nn.Linear(4 * hidden_size, out_hidden_size, bias=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.output_resolution**2, out_hidden_size), requires_grad=False)

    def forward(self, x, c, mask=None):
        """
        x: B, H*W, C
        """
        H = self.input_resolution
        W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        if mask!=None:
            mask = mask.expand(B,L,C).view(B, H, W, C)
            mask0 = mask[:, 0::2, 0::2, :]  # B H/2 W/2 C
            mask1 = mask[:, 1::2, 0::2, :]  # B H/2 W/2 C
            mask2 = mask[:, 0::2, 1::2, :]  # B H/2 W/2 C
            mask3 = mask[:, 1::2, 1::2, :]  # B H/2 W/2 C
            mask = torch.cat([mask0, mask1, mask2, mask3], -1).view(B, -1, 4 * C)
            x = x*mask
        x = self.reduction(x)+self.pos_embed
        return x
    
    def get_flops(self):
        # [1,cond_dim] x [cond_dim,8*hidden_size]
        adaln = self.cond_dim * 8 * self.hidden_size
        # [output_resolution^2,4*hidden_size] x [4*hidden_size,out_hidden_size]
        reduction = (self.output_resolution*self.output_resolution)*(4*self.hidden_size)*self.out_hidden_size
        return adaln + reduction

class UpSample(nn.Module):
    def __init__(self, hidden_size, out_hidden_size, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.expand = nn.Linear(hidden_size, 4*out_hidden_size, bias=False)
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H = self.input_resolution
        W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        return x

    def get_flops(self):
        # [input_resolution^2,hidden_size] x [hidden_size,4*out_hidden_size]
        expand = self.input_resolution*self.input_resolution * self.hidden_size * 4 * self.out_hidden_size
        return expand

class ConcatLayer(nn.Module):
    """
    The final layer of EDT.
    """
    def __init__(self, hidden_size, cond_dim, input_resolution):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.input_resolution = input_resolution
        self.norm_final = nn.LayerNorm(2*hidden_size, elementwise_affine=False, eps=1e-6)#表示只进行归一化，部进行仿射变换
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 4 * hidden_size, bias=True)
        )
        self.cat = nn.Linear(2*hidden_size, hidden_size, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_resolution**2, hidden_size), requires_grad=False)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.cat(x)+self.pos_embed
        return x
    
    def get_flops(self):
        # [1,cond_dim] x [cond_dim,4*hidden_size]
        adaln = self.cond_dim* 4 * self.hidden_size
        # [input_resolution^2,2*hidden_size] x [2*hidden_size,hidden_size]
        cat = (self.input_resolution*self.input_resolution)*(2*self.hidden_size)*self.hidden_size
        return adaln+cat


class EDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=312,
        depth=12,
        num_heads=[6,8,10,8,6],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        stages_depth=[2,2,2,3,3],
        dims = [312,416,520,416,312],
        mid_mask=False,
        size="S",
        local=True):
        super().__init__()

        total_depths = 0
        for i in stages_depth:
            total_depths +=i
        assert total_depths == depth

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.stages_depth = stages_depth
        self.hidden_size = hidden_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.input_resolution=int(input_size/patch_size)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.size = size
        self.stages = len(stages_depth)
        self.down_stages = int(self.stages/2)

        self.mid_mask = mid_mask

        input_resolution = self.input_resolution
        self.blocks = nn.ModuleList()
        self.concat_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        '''
        The number '1' in the 'is_local' array indicates that the corresponding block uses AMM, 
        otherwise AMM is not used. 
        Small and Base models have the same number of layers, 
        so they have the same arrangement form of AMM.
        '''
        if self.size == "S" or self.size == "B":
            # for edt-s, edt-b, when cfg=1, cfg=1.25 or cfg=1.5
            is_local = [0,0, 0,0, 0,0, 0,1,0, 0,1,0] 

            # for edt-s, edt-b, when cfg=2 or cfg>=3
            # is_local = [0,0, 0,0, 0,0, 0,0,1, 0,0,0]
            # for edt-s 512×512, and use function at line 155
            # is_local = [0,0, 0,0, 0,0, 0,0,0, 0,1,0]
        elif self.size == "XL":
            # for edt-xl
            is_local = [0,0,0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,0,0,  0,0,1,0,1,0,0] 
            
            # is_local = [0,0,0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,0,0,  0,0,0,1,0,0,0] 
            self.mid_mask = True 
        if not local:
            is_local = [0,0,0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
        
        depth = 0
        for curr_stage in range(self.stages):
            stage_depth = stages_depth[curr_stage]

            if curr_stage> self.down_stages:
                up_block = UpSample(dims[curr_stage-1], dims[curr_stage], input_resolution)
                self.up_blocks.append(up_block)
                input_resolution = int(input_resolution*2)
                concat_block=ConcatLayer(dims[curr_stage], cond_dim=self.hidden_size, input_resolution = input_resolution)
                self.concat_blocks.append(concat_block)
            
            for j in range(stage_depth):
                if is_local[depth]==1:
                    local=True
                else:
                    local=False
                block = EDTBlock(dims[curr_stage], num_heads[curr_stage], mlp_ratio=mlp_ratio,input_resolution=input_resolution,cond_dim=self.hidden_size,local=local)
                self.blocks.append(block)
                depth +=1
                
            if curr_stage< self.down_stages:
                down_block = DownSample(dims[curr_stage], dims[curr_stage+1], input_resolution, self.hidden_size)
                self.down_blocks.append(down_block)
                input_resolution = int(input_resolution/2)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in EDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        for block in self.concat_blocks:
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            pos_embed = get_2d_sincos_pos_embed(block.pos_embed.shape[-1], int(block.input_resolution))
            block.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        for block in self.down_blocks:
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            pos_embed = get_2d_sincos_pos_embed(block.pos_embed.shape[-1], int(block.input_resolution/2))
            block.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)#模型输入点
            return outputs
        return ckpt_forward

    def forward(self, x, t, y, enable_input_mask=False, enable_down_mask=False):
        """
        Forward pass of EDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        x_skips = []
        mask=None
        mid_mask = False
        if enable_down_mask:
            mid_mask = self.mid_mask
        depth = 0
        a=0.4
        b=0.5
        for curr_stage in range(self.stages):
            if curr_stage<self.down_stages:
                x_skips.append(x)
            
            # an up-sampling module
            if curr_stage>self.down_stages:
                x = self.up_blocks[curr_stage-self.down_stages-1](x)
                x = torch.cat([x,x_skips[-1]],-1)
                x = self.concat_blocks[curr_stage-self.down_stages-1](x,c)
                x_skips.pop()
            
            # an EDT stage
            for j in range(self.stages_depth[curr_stage]):
                if mid_mask and j==4:
                    p=random.uniform(0.15, 0.25) # mid_mask only use for EDT-xl
                    mask = torch.rand(x.shape[:-1]).unsqueeze(-1).to(x.device)
                    mask = (mask >= p).float()
                    x = x*mask + (1-mask)*self.pos_embed
                    mask = None
                    mid_mask = False 
                x = self.blocks[depth](x,c)  # (N, T, D)
                depth+=1
            
            # a down-sampling modules
            if curr_stage<self.down_stages:
                if enable_down_mask:
                    p=random.uniform(a, b) #[0.1,b]
                    mask = torch.rand(x.shape[:-1]).unsqueeze(-1).to(x.device)
                    mask = (mask >= p).float()
                    a=0.1
                    b=0.2
                x = self.down_blocks[curr_stage](x,c,mask)
                mask = None
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of EDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]#将输入x分成两半，只取前一半。这假设输入批量中的前一半是有条件的，而后一半是无条件的
        combined = torch.cat([half, half], dim=0)#将处理过的前一半复制并拼接，实际上是在准备进行同时有条件和无条件的前向传播


        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :3], model_out[:, 3:]#将模型输出分为两部分：前三个通道（假设用于CFG）和剩余的通道
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)#将eps（前三个通道）分成有条件部分和无条件部分
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)#应用CFG：通过将有条件生成部分与无条件生成部分的差异乘以cfg_scale（CFG缩放因子）并加到无条件部分上，来调整无条件生成部分。
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)#将调整后的eps复制并拼接，准备与rest合并

    def get_flops(self):
        #73935272,4548771840
        # [in_channels,input_size,input_size] conv [hidden_size, in_channels, patch_size, patch_size] → [hidden_size,input_resolution,input_resolution]
        x_embed = self.input_resolution * self.input_resolution * self.patch_size * self.patch_size * self.in_channels * self.hidden_size
        t_embed = self.t_embedder.get_flops()
        y_embed = self.y_embedder.get_flops()
        # [1,hidden_size] x [hidden_size,2*hidden_size];[input_resolution^2,hidden_size] x [hidden_size,out_hidden_size]
        out_hidden_size = self.patch_size * self.patch_size * self.out_channels
        final_layer = self.hidden_size * 2 * self.hidden_size + (self.input_resolution*self.input_resolution)*self.hidden_size*out_hidden_size
        flops = x_embed+t_embed+y_embed+final_layer
        for block in self.blocks:
            flops+=block.get_flops()
        
        for block in self.concat_blocks:
            flops+=block.get_flops()
        
        for block in self.down_blocks:
            flops+=block.get_flops()
        
        for block in self.up_blocks:
            flops+=block.get_flops()

        return flops



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   EDT Configs                                  #
#################################################################################

def EDT_XL_2_noAMM(**kwargs):
    return EDT(depth=28, hidden_size=936, patch_size=2, num_heads=[18,24,30,24,18], stages_depth=[6,4,4,7,7], dims=[936,1248,1560,1248,936], size="XL", local = False, **kwargs)#708.44、51.83

def EDT_B_2_noAMM(**kwargs):
    return EDT(depth=12, hidden_size=624, patch_size=2, num_heads=[12,16,20,16,12], stages_depth=[2,2,2,3,3], dims=[624,832,1040,832,624], size="B", local = False, **kwargs)#152.2、10.2 #b


def EDT_S_2_noAMM(**kwargs):
    return EDT(depth=12, hidden_size=312, patch_size=2, num_heads=[6,8,10,8,6], stages_depth=[2,2,2,3,3], dims=[312,416,520,416,312], size="S", local = False, **kwargs)#38.32、2.66

def EDT_XL_2(**kwargs):
    return EDT(depth=28, hidden_size=936, patch_size=2, num_heads=[18,24,30,24,18], stages_depth=[6,4,4,7,7], dims=[936,1248,1560,1248,936], size="XL", **kwargs)#708.44、51.83

def EDT_B_2(**kwargs):
    return EDT(depth=12, hidden_size=624, patch_size=2, num_heads=[12,16,20,16,12], stages_depth=[2,2,2,3,3], dims=[624,832,1040,832,624], size="B", **kwargs)#152.2、10.2 #b


def EDT_S_2(**kwargs):
    return EDT(depth=12, hidden_size=312, patch_size=2, num_heads=[6,8,10,8,6], stages_depth=[2,2,2,3,3], dims=[312,416,520,416,312], size="S", **kwargs)#38.32、2.66


EDT_models = {
    'EDT-XL/2_noAMM': EDT_XL_2_noAMM, 
    'EDT-B/2_noAMM':  EDT_B_2_noAMM,
    'EDT-S/2_noAMM':  EDT_S_2_noAMM,

    'EDT-XL/2': EDT_XL_2, 
    'EDT-B/2':  EDT_B_2,
    'EDT-S/2':  EDT_S_2,
}
