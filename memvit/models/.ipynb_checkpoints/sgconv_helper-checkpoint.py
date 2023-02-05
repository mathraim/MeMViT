import numpy as np
import torch
import torch.nn as nn

import math
import torch.nn.functional as F

from memvit.datasets import loader
from memvit.datasets import ava_helper
from memvit.models import build_model

from memvit.models.common import DropPath2, Mlp

from memvit.config.defaults import get_cfg
from einops import rearrange, repeat

class PoolingLayer(nn.Module):
    
    def __init__(self,
                 mode,
                 kernel,
                 stride,
                 padding,
                 input_dim,
                 head_num):
        super().__init__()
        
        # IF not active just return the input
        self.active = np.prod(kernel) != 1 or np.prod(stride) != 1
        
        # the mode of the pooling as well as 
        # if we want to share pooling across the heads
        self.mode = mode
        
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_dim = input_dim // head_num
        
        if "avg" in mode:
            self.pool_op = nn.AvgPool3d(kernel,
                                        stride,
                                        padding,
                                        ceil_mode = False)
        elif "max" in mode:
            self.pool_op = nn.MaxPool3d(kernel,
                                        stride,
                                        padding,
                                        ceil_mode = False)
        elif mode == "conv":
            self.pool_op = nn.Conv3d(input_dim,
                                     input_dim,
                                     kernel,
                                     stride,
                                     padding,
                                     groups=input_dim,
                                     bias=False)
        elif mode == "conv_unshared":
            self.pool_op = nn.Conv3d(input_dim,
                                     input_dim,
                                     kernel,
                                     stride,
                                     padding,
                                     groups=self.head_num,
                                     bias=False)
            
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
            
        
    def forward(self, x):
        """
        Shape of x is (Batch, Channel, Lenght, Height, Width)
        Output is (Batch, Channel, new Length, new Height, new Width)
        """
        if not self.active:
            return x
        
        #First check if input_dim is correct 
        assert self.input_dim == x.shape[1]
        
        # reshaping for pooling
        x = self.pool_op(x)
        
        # reshaping for layer norm
        x = rearrange(x, "b c l h w -> b l h w c")
        x = self.norm(x)
        x = rearrange(x, "b l h w c -> b c l h w")
        return x  
    
class LinearLayerKeepingShape(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 qkv_bias,
                 drop_qkv_rate):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        
        if drop_qkv_rate > 0.0:
            self.qkv_drop = nn.Dropout(drop_qkv_rate)
        else:
            self.qkv_drop = nn.Identity()
        
    def forward(self, x, class_token):
        """
        Input:
        Shape of x is (Batch, Channel, Lenght, Height, Width)
        Shape of class_token is (Batch, Channel)
        
        Output:
        Shape of x_out is (Batch, new Channel, Length, Height, Width)
        Shape of class_token_out is (Batch, new Channel)
        """
        length = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        
        x = rearrange(x, "b c l h w -> b (l h w) c")
        
        if class_token is not None:
            class_token = repeat(class_token, "b c -> b one c", one = 1)
            x = torch.cat((class_token, x), dim = 1)
            
        x = self.layer(x)
        x = self.qkv_drop(x)
        
        if class_token is not None:
            class_token = x[:, 0, :]
            x = x[:, 1:, :]
        
        x = rearrange(x, "b (l h w) c -> b c l h w", 
                      l = length, h = height, w = width)
        return x, class_token 
    
class Attention(nn.Module):
    
    def __init__(self,
                 output_dim,
                 head_num,
                 drop_rate,
                 drop_attn_rate):
        super().__init__()
        
        self.output_dim = output_dim
        self.head_num = head_num
        self.head_dim = output_dim // head_num
        self.scale = self.head_dim**-0.5
        
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)
        else:
            self.proj_drop = nn.Identity()
            
        if drop_attn_rate > 0.0:
            self.attn_drop = nn.Dropout(drop_attn_rate)
        else:
            self.attn_drop = nn.Identity()
        
        self.proj = nn.Linear(output_dim, output_dim)
        
    
    def forward(self, q, k, v, cls_q, cls_k, cls_v): 
        """
        Input:
        Shape of q is (Batch, Channel, q Lenght, q Height, q Width)
        Shape of k, v is (Batch, Channel, Lenght, Height, Width)
        Shape of cls_q, cls_k, cls_v is (Batch, Channel)
        
        Output:
        Shape of x_out is (Batch, Channel, Length, Height, Width)
        Shape of class_token_out is (Batch, Channel)
        """
        #start_time = time.time()
        
        out_length = q.shape[2]
        out_height = q.shape[3]
        out_width = q.shape[4]
        
        #print("1 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        
        q = rearrange(q, "b (head c) l h w -> b head (l h w) c",
                     head = self.head_num)
        k = rearrange(k, "b (head c) l h w -> b head (l h w) c",
                     head = self.head_num)
        v = rearrange(v, "b (head c) l h w -> b head (l h w) c",
                     head = self.head_num)
        
        #print("2 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        
        if cls_q is not None:
            cls_q = rearrange(cls_q, "b (head c) -> b head c",
                              head = self.head_num)
            cls_q = repeat(cls_q, "b head c -> b head one c",
                           one = 1)
            q = torch.cat((cls_q, q), dim = 2)
            
        if cls_k is not None:
            cls_k = rearrange(cls_k, "b (head c) -> b head c",
                              head = self.head_num)
            cls_k = repeat(cls_k, "b head c -> b head one c",
                           one = 1)
            k = torch.cat((cls_k, k), dim = 2)
            
        if cls_v is not None:
            cls_v = rearrange(cls_v, "b (head c) -> b head c",
                              head = self.head_num)
            cls_v = repeat(cls_v, "b head c -> b head one c",
                           one = 1)
            v = torch.cat((cls_v, v), dim = 2)
        
        #print("3 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        
        k_tran = rearrange(k, "b head seq c -> b head c seq")
        """
        Current q shape - b head seq c
        Current k shape - b head c seq
        """
        x = (q @ k_tran) * self.scale
        x = x.softmax(dim=-1)
        x = self.attn_drop(x)
        
        #print("4 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()

        
        """
        Current x shape is - batch head seq seq
        Current v shape is - batch head seq head_dim
        """
        x = (x @ v)
        
        #print("5 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        
        # Current x shape is - batch head seq head_dim
        # Reshaping for the projection
        x = rearrange(x, "b h seq c -> b seq (h c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # Current x shape is - batch seq out_dim
        
        #print("6 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()

        
        class_token = None
        if cls_q is not None:
            class_token = x[:, 0, :]
            x = x[:, 1:, :]
            
        #print("7 --- %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        
        #Returning to shape
        x = rearrange(x, "b (l h w) c -> b c l h w", 
                      l = out_length, h = out_height, w = out_width)
        
        #print("8 --- %s seconds ---" % (time.time() - start_time))
 
        return x, class_token

class MultiScaleAttentionNew(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 head_num,
                 qkv_bias,
                 drop_rate,
                 kernel_q,
                 kernel_kv,
                 stride_q,
                 stride_kv,
                 mode,
                 pool_first,
                 drop_attn_rate,
                 drop_qkv_rate):
        super().__init__()
        self.pool_first = pool_first
        
        
        # Initialising poolings
        self.input_dim = input_dim
        self.head_num = head_num
        self.mode = mode
        self.kernel_q = kernel_q
        self.kernel_kv = kernel_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        
        self.initialize_poolings()
        # Finished initialise pooling
        
        #-----------------------------------
        
        #Initialize qkv
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.qkv_bias = qkv_bias
        self.drop_qkv_rate = drop_qkv_rate
        
        self.initialize_qkv()
        # Finished initialise qkv
        
        #-----------------------------------
        
        # Initialize attention
        self.output_dim = output_dim
        self.head_num = head_num
        self.drop_rate = drop_rate
        self.drop_attn_rate = drop_attn_rate
        
        self.initialize_attention()
        # Finished initialise attention
        
        #-----------------------------------
        
        
    def forward(self, x, class_token):
        """
        Input:
        Shape of x is (Batch, Channel, Lenght, Height, Width)
        Shape of class_token is (Batch, Channel, Lenght, Height, Width)
            
        Output:
        Shape of x_out is (Batch, Channel, Length, Height, Width)
        Shape of class_token_out is (Batch, Channel)
        """
        #First layer
        if self.pool_first:
            q = self.q_pooling(x)
            k = self.k_pooling(x)
            v = self.v_pooling(x)
        else:
            q, cls_q = self.q_layer(x, class_token)
            k, cls_k = self.k_layer(x, class_token)
            v, cls_v = self.v_layer(x, class_token)
        #-----------------------------------
                
        #Second layer
        if not self.pool_first:
            q = self.q_pooling(q)
            k = self.k_pooling(k)
            v = self.v_pooling(v)
        else:
            q, cls_q = self.q_layer(q, class_token)
            k, cls_k = self.k_layer(k, class_token)
            v, cls_v = self.v_layer(v, class_token)
        #-----------------------------------
            
        x, class_token = self.attn_layer(q, k, v,
                                         cls_q, cls_k, cls_v)
        return x, class_token
                                             
                
    def initialize_attention(self):
        self.attn_layer = Attention(self.output_dim,
                                    self.head_num, 
                                    self.drop_rate, 
                                    self.drop_attn_rate)
            
            
    def initialize_qkv(self):
        self.q_layer = LinearLayerKeepingShape(self.input_dim,
                                               self.output_dim,
                                               self.qkv_bias,
                                               self.drop_qkv_rate)
            
        self.k_layer = LinearLayerKeepingShape(self.input_dim,
                                               self.output_dim,
                                               self.qkv_bias,
                                               self.drop_qkv_rate)
            
        self.v_layer = LinearLayerKeepingShape(self.input_dim,
                                               self.output_dim,
                                               self.qkv_bias,
                                               self.drop_qkv_rate)
            
    def initialize_poolings(self):
        
        if self.pool_first:
            pooling_input_dim = self.input_dim
        else:
            pooling_input_dim = self.output_dim
            
        padding_q = [int(i // 2) for i in self.kernel_q]
        self.q_pooling = PoolingLayer(self.mode,
                                      self.kernel_q,
                                      self.stride_q,
                                      padding_q,
                                      pooling_input_dim,
                                      self.head_num)

        padding_kv = [int(i // 2) for i in self.kernel_kv]
        self.k_pooling = PoolingLayer(self.mode,
                                      self.kernel_kv,
                                      self.stride_kv,
                                      padding_kv,
                                      pooling_input_dim,
                                      self.head_num)

        padding_kv = [int(i // 2) for i in self.kernel_kv]
        self.v_pooling = PoolingLayer(self.mode,
                                      self.kernel_kv,
                                      self.stride_kv,
                                      padding_kv,
                                      pooling_input_dim,
                                      self.head_num)
        
        
        
class MultiScaleBlockNew(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads,
                 qkv_bias,
                 drop_rate,
                 kernel_q,
                 kernel_kv,
                 stride_q,
                 stride_kv,
                 mode,
                 pool_first,
                 drop_attn_rate,
                 drop_qkv_rate,
                 mlp_ratio,
                 act_layer,
                 dim_mul_in_attn,
                 drop_path):
        super().__init__()       
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-6)
        
        attn_dim = output_dim if dim_mul_in_attn else input_dim
        self.attn = MultiScaleAttentionNew(input_dim,
                                        attn_dim,
                                        num_heads,
                                        qkv_bias,
                                        drop_rate,
                                        kernel_q,
                                        kernel_kv,
                                        stride_q,
                                        stride_kv,
                                        mode,
                                        pool_first,
                                        drop_attn_rate,
                                        drop_qkv_rate)
        
        self.drop_path = DropPath2(drop_path) if drop_path > 0.0 else None
        self.block_norm = nn.LayerNorm(attn_dim, eps=1e-6)
        
        mlp_hidden_dim = int(attn_dim * mlp_ratio)
        
        self.mlp = Mlp(
            in_features=attn_dim,
            hidden_features=mlp_hidden_dim,
            out_features=output_dim,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        
        kernel_skip = kernel_q#[s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(i // 2) for i in kernel_q]#[int(skip // 2) for skip in kernel_skip]
        
        self.proj1 = nn.Linear(input_dim, attn_dim)
        self.proj2 = nn.Linear(attn_dim, output_dim)
        
        self.pool_skip = PoolingLayer("max",
                                      kernel_skip,
                                      stride_skip,
                                      padding_skip,
                                      attn_dim,
                                      num_heads)
        
    def forward(self, x, cls_token):
        """
        Input:
        Shape of x is (Batch, Channel, Lenght, Height, Width)
        Shape of class_token is (Batch, Channel, Lenght, Height, Width)
            
        Output:
        Shape of x_out is (Batch, Channel, Length, Height, Width)
        Shae of class_token_out is (Batch, Channel)
        """
            
        x = rearrange(x, "b c l h w -> b l h w c")
        x_norm = self.input_norm(x)
        cls_norm = self.input_norm(cls_token)
        x_norm = rearrange(x_norm, "b l h w c -> b c l h w")
            
        x_skip = rearrange(x_norm, "b c l h w -> b l h w c")
        x_skip = self.proj1(x_skip)
        cls_skip = self.proj1(cls_norm)
        x_skip = rearrange(x_skip, "b l h w c -> b c l h w")
            
        x_skip = self.pool_skip(x_skip)
        x_attn, cls_attn = self.attn(x_norm, cls_norm)
        if self.drop_path is not None:
            x_attn, cls_attn = self.drop_path(x_attn, cls_attn)
        
        x = x_skip + x_attn
        cls_token = cls_skip + cls_attn
            
        x = rearrange(x, "b c l h w -> b l h w c")
        x_norm = self.block_norm(x)
        cls_norm = self.block_norm(cls_token)
            
        x_mlp = self.mlp(x_norm)
        cls_mlp = self.mlp(cls_norm)
        x_mlp = rearrange(x_mlp, "b l h w c -> b c l h w")
        if self.drop_path is not None:
            x_mlp, cls_mlp = self.drop_path(x_mlp, cls_mlp)
            
        x_skip = self.proj2(x_norm)
        cls_skip = self.proj2(cls_norm)
        x_skip = rearrange(x_skip, "b l h w c -> b c l h w")
            
        x = x_skip + x_mlp
        cls_token = cls_skip + cls_mlp
            
        return x, cls_token
    
    
class SGConv(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 height,
                 width,
                 length,
                 kernel_dim,
                 decay_min = 2,
                 decay_max = 2):
        super().__init__()
        self.height = height
        self.width = width
        self.length = length
        self.kernel_dim = kernel_dim
        
        self.height_conv = nn.Parameter(torch.randn(input_dim, height))
        self.width_conv = nn.Parameter(torch.randn(input_dim, width))
        
        if length > kernel_dim:
            self.num_scales = 1 + math.ceil(math.log2(length/kernel_dim))
            self.kernel_list = nn.ParameterList()

            for _ in range(self.num_scales):
                kernel = nn.Parameter(torch.randn(1, input_dim, kernel_dim))
                self.kernel_list.append(kernel)

            self.decay = nn.Parameter(
                torch.rand(input_dim).view(1, -1, 1) * (decay_max - decay_min) + decay_min
            )
        else:
            self.lenght_conv = nn.Parameter(torch.randn(input_dim, length))
        
        
        
    def forward(self, x):
        """
        Input:
        Shape of x is (Batch, Channel, Lenght, Height, Width)
        Shape of class_token is (Batch, Channel, Lenght, Height, Width)

        Output:
        Shape of x_out is (Batch, Channel, Length, Height, Width)
        Shape of class_token_out is (Batch, Channel)
        """
        
        x = rearrange(x, "b c l h w -> b l h c w")
        
        x_f = torch.fft.rfft(x, n = 2 * self.width)
        k_f = torch.fft.rfft(self.width_conv, n = 2 * self.width)
        
        x = torch.einsum('blhcw,cw->blhcw', x_f, k_f)
        x = torch.fft.irfft(x, n = 2 * self.width)[..., :self.width]
        
        x = rearrange(x, "b l h c w -> b l w c h")
        
        x_f = torch.fft.rfft(x, n = 2 * self.height)
        k_f = torch.fft.rfft(self.height_conv, n = 2 * self.height)
        
        x = torch.einsum('blwch,ch->blwch', x_f, k_f)
        x = torch.fft.irfft(x, n = 2 * self.height)[..., :self.height]
         
        x = rearrange(x, "b l w c h -> b h w c l")
        
        if self.length > self.kernel_dim:
            kernel_interpolated = []
            #print(self.num_scales)
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor=2**i,
                    mode='linear',
                    )
                kernel *= self.decay ** (self.num_scales - i - 1)
                kernel_interpolated.append(kernel)
            k = torch.cat(kernel_interpolated, dim=-1)
            k = k[0, :, :self.length]
            kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            k = k / kernel_norm
        else:
            k = self.lenght_conv
        
        x_f = torch.fft.rfft(x, n = 2 * self.length)
        k_f = torch.fft.rfft(k, n = 2 * self.length)
        
        x = torch.einsum('bhwcl,cl->bhwcl', x_f, k_f)
        x = torch.fft.irfft(x, n = 2 * self.length)[..., : self.length]
        
        x = rearrange(x, "b h w c l -> b c l h w")
        return x
    
    
class ConvMVITBlock(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads,
                 input_size,
                 qkv_bias,
                 drop_rate,
                 drop_path,
                 kernel_q,
                 kernel_kv,
                 stride_q,
                 stride_kv,
                 mode,
                 has_cls_embed,
                 pool_first,
                 drop_attn_rate,
                 mlp_ratio,
                 act_layer,
                 dim_mul_in_attn,
                 part_length,
                 twh):
        super().__init__()
        
        self.input_size = input_size
        self.has_cls_embed = has_cls_embed
        self.part_length = part_length
        self.parts = input_size[0] // part_length
        
        self.mvit =  MultiScaleBlockNew(input_dim,
                 output_dim,
                 num_heads,
                 qkv_bias,
                 drop_rate,
                 kernel_q,
                 kernel_kv,
                 stride_q,
                 stride_kv,
                 mode,
                 pool_first,
                 drop_attn_rate,
                 0.0,
                 mlp_ratio,
                 act_layer,
                 dim_mul_in_attn,
                 drop_path)
        
        self.conv = SGConv(output_dim,
                           output_dim,
                           twh[1],
                           twh[2],
                           twh[0],
                           kernel_dim = 32,
                           decay_min = 2,
                           decay_max = 2)
        
        self.csl_conv = torch.nn.Conv2d(1, 
                                        1, 
                                        kernel_size = [self.parts, 1], 
                                        stride=[1, 1], 
                                        padding=0)
        
    def forward(self, x):
        """
        Input:
        Shape of x is (Batch, (Lenght, Height, Width + 1), Channel

        Output:
        Shape of x_out is (Batch, (Lenght, Height, Width + 1), Channel
        """
        length, height, width = self.input_size
        
        cls_token = None
        if self.has_cls_embed:
            cls_token = x[:, 0,:]
            x = x[:, 1:,:]
            
        x = rearrange(x, "b (l h w) c -> b c l h w", 
                      l = length, h = height, w = width)
        
        x = rearrange(x, "b c (p pl) h w -> p b c pl h w", pl = self.part_length)
        
        out_x = []
        out_cls = []
        for i in range(x.shape[0]):
            curr_x, curr_cls = self.mvit(x[i], cls_token)
            out_x.append(curr_x)
            out_cls.append(curr_cls)
            
        x = torch.cat(out_x, dim = 2)
        x = self.conv(x)
        #print(x.shape)
        x = rearrange(x, " b c l h w -> b (l h w) c")
        
        if self.has_cls_embed:
            cls_token = torch.stack(out_cls, dim = 1)
            cls_token = repeat(cls_token, "b p c -> b one p c", one = 1)
            cls_token = self.csl_conv(cls_token)[:, 0, :, :]
            x = torch.cat((cls_token, x), dim = 1)
        return x