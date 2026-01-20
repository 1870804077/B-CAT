import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import (
    BertIntermediate, BertSelfAttention, BertLayer
)
import math
import numpy as np
from scipy.linalg import lstsq

# ==================== 【1】配置 ====================
L_SIGMOID = 4.0 
L_INV = 4.0
NUM_SAMPLES = 3000
# =====================================================

# ==================== 【2】工具函数 ====================
def build_design_matrix_torch(x, K1, K2, L, device='cpu'):
    x = x.to(device)
    terms = [torch.ones_like(x)]
    for k in range(1, K1 + 1):
        terms.append(x ** (2 * k - 1))
    for k in range(1, K2 + 1):
        terms.append(torch.sin(torch.pi * k * x / L))
    return torch.stack(terms, dim=-1)

def fit_coeffs(func, x_range, K1, K2, L):
    x_train = np.linspace(x_range[0], x_range[1], NUM_SAMPLES)
    y_train = func(x_train)
    X = [np.ones_like(x_train)]
    for k in range(1, K1 + 1):
        X.append(x_train ** (2 * k - 1))
    for k in range(1, K2 + 1):
        X.append(np.sin(np.pi * k * x_train / L))
    A = np.vstack(X).T
    coeffs, _, _, _ = lstsq(A, y_train)
    return coeffs.astype(np.float32)
# =====================================================

# ==================== 【3】近似模块 ====================
class ApproxSigmoid(nn.Module):
    def __init__(self, K1 = 1, K2 = 4, L = L_INV):
        super().__init__()
        coeffs = fit_coeffs(lambda x: 1.0 / (1.0 + np.exp(-x)), (-L, L), K1, K2, L)
        self.coeffs = nn.Parameter(torch.from_numpy(coeffs), requires_grad=False)
        self.K1 = 1
        self.K2 = 12
        self.L = 8
    def forward(self, x):
            X = build_design_matrix_torch(x, self.K1, self.K2, self.L, x.device)
            return torch.sum(X * self.coeffs, dim=-1)


class ApproxInvSqrt(nn.Module):
    def __init__(self, K1=1, K2=12, split_point=8.0, L_large=3000.0):
        super().__init__()
        self.split_point = split_point
        self.L_small = split_point 
        self.L_large = L_large
        
        def inv_sqrt_func(x):
            return 1.0 / np.sqrt(np.abs(x) + 1e-10)

        coeffs_small = fit_coeffs(inv_sqrt_func, (0.1, self.L_small), K1, K2, self.L_small)
        self.coeffs_small = nn.Parameter(torch.from_numpy(coeffs_small), requires_grad=False)
        
        coeffs_large = fit_coeffs(inv_sqrt_func, (self.L_small, self.L_large), K1, K2, self.L_large)
        self.coeffs_large = nn.Parameter(torch.from_numpy(coeffs_large), requires_grad=False)

        self.K1 = K1
        self.K2 = K2

    def newton_refine(self, y, x):
        simhalfnumber = 0.500438180 * x
        y = y * (1.50131454 - simhalfnumber * y * y)
        y = y * (1.50000086 - 0.999124984 * simhalfnumber * y * y)
        return y

    def forward(self, x):
        out = torch.zeros_like(x)
        mask_small = x < self.split_point
        mask_large = ~mask_small

        if mask_small.any():
            x_small = x[mask_small]
            X_s = build_design_matrix_torch(x_small, self.K1, self.K2, self.L_small, x.device)
            y_small = torch.sum(X_s * self.coeffs_small, dim=-1)
            out[mask_small] = y_small

        if mask_large.any():
            x_large = x[mask_large]
            X_l = build_design_matrix_torch(x_large, self.K1, self.K2, self.L_large, x.device)
            y_large = torch.sum(X_l * self.coeffs_large, dim=-1)
            out[mask_large] = y_large
        
        out = self.newton_refine(out, x)
        return out
    
class ApproxInv(nn.Module):
    def __init__(self, K1 = 1, K2 = 12, L = L_INV):
        super().__init__()
        self.inv_sqrt = ApproxInvSqrt(1, 12, L)
        
    def forward(self, x):
        return self.inv_sqrt(x) ** 2


class ApproxExpPlusOne(nn.Module):
    def __init__(self, K1=1, K2=12, L=16):
        super().__init__()
        coeffs = fit_coeffs(np.exp, (-16, -2), K1, K2, L)
        self.coeffs = nn.Parameter(torch.from_numpy(coeffs), requires_grad=False)
        self.K1 = 1
        self.K2 = 12
        self.L = 16
        
    def forward(self, x):      
        X = build_design_matrix_torch(x, self.K1, self.K2, self.L, x.device)
        return torch.sum(X * self.coeffs, dim=-1) + 1.0
    

def approx_softmax_from_expplus1(x, exp_plus_one_module, approx_inv_module, dim=-1):
    MAXCL = 10 
    x_clamped = torch.clamp(x, min=-4.0, max=MAXCL)
    x_shifted = x_clamped - MAXCL - 2
    
    exp_approx = exp_plus_one_module(x_shifted) - 1.0
    exp_approx = torch.clamp(exp_approx, min=1e-12)
    
    exp_sum = exp_approx.sum(dim=dim, keepdim=True)
    exp_sum = exp_sum * 1000
    
    inv_exp_sum = approx_inv_module(exp_sum)
    softmax_approx = exp_approx * inv_exp_sum * 1000
    return softmax_approx


class CustomBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        coeffs_tanh = fit_coeffs(np.tanh, (-13.0, 13.0), 1, 12, 13)
        self.tanh_coeffs = nn.Parameter(torch.from_numpy(coeffs_tanh), requires_grad=False)
        self.K1 = 1
        self.K2 = 12
        self.L = 13

    def approx_tanh(self, x):
        original_dtype = x.dtype
        x_f32 = x.float()
        x_clamped = torch.clamp(x_f32, -self.L, self.L)
        X = build_design_matrix_torch(x_clamped, self.K1, self.K2, self.L, x.device)
        y = torch.sum(X * self.tanh_coeffs, dim=-1)
        return y.to(original_dtype)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x_float = x.float()
        const_factor = 0.79788456
        arg_float = const_factor * (x_float + 0.044715 * torch.pow(x_float, 3))
        arg = arg_float.to(x.dtype)
        out = torch.zeros_like(x)

        mask_left  = arg < -13.0
        mask_right = arg > 13.0
        mask_mid   = (~mask_left) & (~mask_right)

        if mask_right.any():
            out[mask_right] = x[mask_right]

        if mask_mid.any():
            tanh_val = self.approx_tanh(arg[mask_mid])
            out[mask_mid] = 0.5 * x[mask_mid] * (1.0 + tanh_val)

        return out
    
class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.exp_plus_one = ApproxExpPlusOne()
        self.approx_inv = ApproxInv(1, 12, L_INV)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, past_key_value=None, output_attentions=False, cache_position=None):
        past_key_value = past_key_values if past_key_values is not None else past_key_value
        
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = approx_softmax_from_expplus1(
            attention_scores, self.exp_plus_one, self.approx_inv ,dim=-1
        )
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class CustomBertLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-12):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.inv_sqrt = ApproxInvSqrt(1, 12 , 8.0)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        val = var + self.eps
        inv_std = self.inv_sqrt(val)
        x_norm = (x - mean) * inv_std
        return self.weight * x_norm + self.bias

def replace_bert_modules(model):
    config = model.config
    for module in model.modules():
        if isinstance(module, BertLayer):
            module.intermediate = CustomBertIntermediate(config)
            module.attention.self = CustomBertSelfAttention(config)
            module.attention.output.LayerNorm = CustomBertLayerNorm(
                normalized_shape=config.hidden_size, 
                eps=config.layer_norm_eps
            )
            module.output.LayerNorm = CustomBertLayerNorm(
                normalized_shape=config.hidden_size, 
                eps=config.layer_norm_eps
            )
    return model