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
from scipy.special import erf

# ==================== 【0】监控工具 ====================
STAT_COUNTERS = {}

def log_stats(name, x):
    """
    监控输入张量的统计信息，用于观察数值范围。
    策略：每种类型打印前 20 次，之后每 500 次打印 1 次，防止刷屏。
    """
    global STAT_COUNTERS
    if name not in STAT_COUNTERS:
        STAT_COUNTERS[name] = 0
    STAT_COUNTERS[name] += 1
    
    count = STAT_COUNTERS[name]
    if count <= 20 or count % 500 == 0:
        with torch.no_grad():
            x_min = x.min().item()
            x_max = x.max().item()
            x_mean = x.mean().item()
            x_std = x.std().item()
            print(f"【DEBUG】[{name}] Count={count} | Range=[{x_min:.6f}, {x_max:.6f}] | Mean={x_mean:.6f} | Std={x_std:.6f}")

# ==================== 【1】用户配置 ====================
L_SIGMOID = 4.0   # ← 改为 4.0，匹配你的拟合区间 [-4, 0]
L_INV = 8.0
L_inv_LARGE = 3000
L_sqrtinv_LARGE = 3000
NUM_SAMPLES = 3000
# =====================================================

# ==================== 【2】工具函数 ====================
def build_design_matrix_torch(x, K1, K2, L, device='cpu'):
    x = x.to(device)
    terms = [torch.ones_like(x)]
    for k in range(1, K1 + 1):
        terms.append(x ** (k))
    for k in range(1, K2 + 1):
        terms.append(torch.sin(torch.pi * k * x / L))
    return torch.stack(terms, dim=-1)

def fit_coeffs(func, x_range, K1, K2, L):
    x_train = np.linspace(x_range[0], x_range[1], NUM_SAMPLES)
    y_train = func(x_train)
    X = [np.ones_like(x_train)]
    for k in range(1, K1 + 1):
        X.append(x_train ** (k))
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


# class ApproxInvSqrt(nn.Module):
#     def __init__(self, K1=1, K2=12, split_point=8.0, L_large=L_sqrtinv_LARGE):
#         """
#         分段拟合 InvSqrt:
#         1. Small Range [0.1, 8.0 + 2]: 捕捉剧烈变化的曲线
#         2. Large Range [8.0, 3000.0]: 捕捉平缓的长尾曲线
#         """
#         super().__init__()
#         self.split_point = split_point
#         self.L_small = split_point  # 第一段的 L 就是分界点 8.0
#         self.L_large = L_sqrtinv_LARGE      # 第二段的 L 设为 3000.0
        
#         # 定义目标函数
#         def inv_sqrt_func(x):
#             return 1.0 / np.sqrt(np.abs(x) + 1e-10)

#         coeffs_small = fit_coeffs(inv_sqrt_func, (0.1, self.L_small), K1, K2, self.L_small)
#         self.coeffs_small = nn.Parameter(torch.from_numpy(coeffs_small), requires_grad=False)
        
#         coeffs_large = fit_coeffs(inv_sqrt_func, (self.L_small, self.L_large), K1, K2, self.L_large)
#         self.coeffs_large = nn.Parameter(torch.from_numpy(coeffs_large), requires_grad=False)

#         self.K1 = K1
#         self.K2 = K2

#     def newton_refine(self, y, x):

#         simhalfnumber = 0.500438180 * x
#         y = y * (1.50131454 - simhalfnumber * y * y)
#         y = y * (1.50000086 - 0.999124984 * simhalfnumber * y * y)
#         return y

#     def forward(self, x):
#         # 初始化输出
#         out = torch.zeros_like(x)
        
#         # 1. 生成掩码
#         mask_small = x < self.split_point
#         mask_large = ~mask_small

#         if mask_small.any():
#             x_small = x[mask_small]
            
#             X_s = build_design_matrix_torch(
#                 x_small, self.K1, self.K2, self.L_small, x.device
#             )
            
#             # 计算初始猜测值
#             y_small = torch.sum(X_s * self.coeffs_small, dim=-1)
#             out[mask_small] = y_small

#         if mask_large.any():
#             x_large = x[mask_large]
            
#             X_l = build_design_matrix_torch(
#                 x_large, self.K1, self.K2, self.L_large, x.device
#             )
            
#             # 计算初始猜测值
#             y_large = torch.sum(X_l * self.coeffs_large, dim=-1)
#             out[mask_large] = y_large
        
#         out = self.newton_refine(out, x)
        
#         return out
    
    
# ##############现在倒数计算
# class ApproxInv(nn.Module):
#     def __init__(self, K1=3, K2=12, split_point=8.0, L_large=L_inv_LARGE):
#         super().__init__()
#         self.split_point = split_point
#         self.L_small = split_point 
#         self.L_large = L_inv_LARGE 
#         self.K1 = K1
#         self.K2 = K2
        
#         # 目标函数 1/x
#         def reciprocal_func(x):
#             return 1.0 / (np.abs(x) + 1e-6)
            
#         # 拟合小范围 [0.1, 8.0 + 2]
#         coeffs_small = fit_coeffs(reciprocal_func, (0.1, self.L_small), K1, K2, self.L_small)
#         self.coeffs_small = nn.Parameter(torch.from_numpy(coeffs_small), requires_grad=False)
        
#         # 拟合大范围 [8.0, 3000.0]
#         coeffs_large = fit_coeffs(reciprocal_func, (self.L_small, self.L_large), K1, K2, self.L_large)
#         self.coeffs_large = nn.Parameter(torch.from_numpy(coeffs_large), requires_grad=False)

#     def newton_refine(self, y, x):
#         return y * (2.0 - x * y)

#     def forward(self, x):
#         # 1. 计算 Small 分支
#         X_s = build_design_matrix_torch(x, self.K1, self.K2, self.L_small, x.device)
#         y_small = torch.sum(X_s * self.coeffs_small, dim=-1)

#         # 2. 计算 Large 分支
#         X_l = build_design_matrix_torch(x, self.K1, self.K2, self.L_large, x.device)
#         y_large = torch.sum(X_l * self.coeffs_large, dim=-1)

#         # 3. 生成 Mask
#         mask_small = (x < self.split_point).to(x.dtype)
        
#         # 4. 合成初始猜测 
#         y_init = y_small * mask_small + y_large * (1.0 - mask_small)
        
#         # 5. 牛顿迭代 
#         y = self.newton_refine(y_init, x)
#         y = self.newton_refine(y, x) 
#         y = self.newton_refine(y, x) 
        
#         # log_stats("Inv",y)
        
#         return y


# SHAFT invsqrt    
class ApproxInvSqrt(nn.Module):
    def __init__(self, a = None , b =None , c = None ):
        """
        基于 Newton-Raphson 迭代的 inv_sqrt 近似实现 (复刻 Crypten 逻辑)
        Args:
            iters (int): 牛顿迭代次数，Crypten 默认为配置值，通常 5 次已足够高精度。
        """
        super().__init__()
        self.iters = 5

    def forward(self, x):

        x_safe = x 
        exponent = -(0.5 * x_safe + 0.2)

        y = torch.exp(exponent) * 2.2 + 0.2

        y = y - (x_safe / 1024.0)
        
        for _ in range(self.iters):

            term = 3.0 - x_safe * (y * y)

            y = y * term * 0.5
            
        return y

class ApproxInv(nn.Module):
    def __init__(self, K1 = 1, K2 = 4, L = L_INV):
        super().__init__()
        self.inv_sqrt = ApproxInvSqrt(1, 9, L)
    def forward(self, x):
        return self.inv_sqrt(x) ** 2
    
class ApproxExpPlusOne(nn.Module):
    def __init__(self, K1=3, K2=12, L=16):
        """
        近似 Exp 函数: f(x) = e^x
        注意：Exp 函数增长极快，拟合区间 L 不宜过大，否则多项式在边界处误差会很大。
        对于 Softmax 场景，通常输入已减去最大值 (x <= 0)，建议拟合 [-L, 0] 或使用较小的 L。
        """
        super().__init__()
        coeffs = fit_coeffs(np.exp, (-16, -2), K1, K2, L)
        
        self.coeffs = nn.Parameter(torch.from_numpy(coeffs), requires_grad=False)
        self.K1 = 3
        self.K2 = 12
        self.L = 16
        
    def forward(self, x):      
        X = build_design_matrix_torch(x, self.K1, self.K2, self.L, x.device)
        return torch.sum(X * self.coeffs, dim=-1) + 1.0
    

# def approx_softmax_from_expplus1(x, exp_plus_one_module, approx_inv_module, dim=-1):
#     MAXCL = 10
#     x_clamped = torch.clamp(x, min=-2.0, max=MAXCL)
#     x_shifted = x_clamped - MAXCL - 2
#     # log_stats("EXP", x_shifted)
    
#     # 2. 计算 Exp 近似
#     exp_approx = exp_plus_one_module(x_shifted) - 1.0
#     # exp_approx = torch.exp(x_shifted)
#     # exp_approx = torch.clamp(exp_approx, min = 1e-12)
#     # log_stats("EXP", exp_approx)
    
#     # 3. 计算分母 Sum
#     exp_sum = exp_approx.sum(dim=dim, keepdim=True)
#     exp_sum = exp_sum * 300

#     # log_stats("exp_sum",exp_sum)
    
#     inv_exp_sum = approx_inv_module(exp_sum)
    
#     # log_stats("Inv",inv_exp_sum)
    
#     softmax_approx = exp_approx * inv_exp_sum * 300
    
#     # log_stats("Softmax", softmax_approx)
    
#     return softmax_approx

# # ODE
def approx_softmax_from_expplus1(x, exp_plus_one_module, dim=-1):

    """

    使用 ODE (常微分方程) 方法近似 Softmax。

    注意：此方法不需要 exp_plus_one_module，保留该参数仅为了兼容接口。

    """

    iter_num = 16  

    x_clamped = torch.clamp(x, min=-4.0, max=12.0)
    # log_stats("Softmax_ODE_Input_Clamped", x_clamped)

    x_scaled = x_clamped / iter_num  

    size = x.size(dim)
    g = torch.ones_like(x) / size
    # =======================================================
    # ODE 迭代更新
    # update rule: g += (x_scaled - E[x_scaled]) * g
    # =======================================================

    for _ in range(iter_num):
        expectation = (g * x_scaled).sum(dim=dim, keepdim=True)     
        g = g + (x_scaled - expectation) * g
    return g


# # 统计x在不同区间的个数
# def count_intervals(tensor, intervals):
#     """统计张量在指定区间的元素个数"""
#     counts = []
#     for i, (low, high) in enumerate(intervals):
#         if i < len(intervals) - 1:
#             # 左闭右开区间 [low, high)
#             mask = (tensor >= low) & (tensor < high)
#         else:
#             # 最后一个区间包括右端点 [low, high]
#             mask = (tensor >= low) & (tensor <= high)
#         counts.append(mask.sum().item())
#     return counts


# 计算GeLU函数
# class CustomBertIntermediate(BertIntermediate):
#     def __init__(self, config):
#         super().__init__(config)
#         self.L = 5.0 
#         coeffs_erf = fit_coeffs(erf, (-self.L, self.L), 1, 12, self.L)
#         self.erf_coeffs = nn.Parameter(torch.from_numpy(coeffs_erf), requires_grad=False)
#         self.K1 = 1
#         self.K2 = 12

#     def approx_erf(self, z):
#         original_dtype = z.dtype
#         z_f32 = z.float()
#         z_clamped = torch.clamp(z_f32, -self.L, self.L)
#         X = build_design_matrix_torch(z_clamped, self.K1, self.K2, self.L, z.device)
#         y = torch.sum(X * self.erf_coeffs, dim=-1)
#         return y.to(original_dtype)

#     def forward(self, hidden_states):

#         x = self.dense(hidden_states)
#         x_float = x.float()
#         const_factor = 0.7071067812 
#         arg = (x_float * const_factor).to(x.dtype)
#         mask_left = (x < -self.L).to(x.dtype)
#         mask_right = (x > self.L).to(x.dtype)
#         mask_mid = 1.0 - mask_left - mask_right
#         erf_val = self.approx_erf(arg)
#         res_mid = 0.5 * x * (1.0 + erf_val)
#         out = (x * mask_right) + (res_mid * mask_mid)

#         return out
  
# SHAFT GELU
class CustomBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        beta_sin = [-0.0818, -0.0809, -0.0424, -0.0176, -0.0079, -0.0043, -0.0026, -0.0017]
        self.register_buffer("beta_sin", torch.tensor(beta_sin))
        
        # 2. 定义拟合参数
        self.width = 8.0  # 拟合区间宽度，与之前的 L=8.0 保持一致
        self.terms = len(beta_sin)

    def _fourier_series_sum(self, x):
        """
        计算傅里叶正弦级数求和: sum(beta_k * sin((k+1) * pi * x / width))
        使用了广播机制以提高效率。
        """
        # 构造 k = [1, 2, ..., terms]
        k = torch.arange(1, self.terms + 1, device=x.device, dtype=x.dtype)
        # 计算正弦项的参数: theta = k * pi * x / width
        # x.unsqueeze(-1) shape: [..., 1]
        # k shape: [terms]
        # theta shape: [..., terms]
        theta = (x.unsqueeze(-1) * k * torch.pi) / self.width
        
        # 计算正弦值
        sines = torch.sin(theta)
        
        # 加权求和: sum(beta * sin)
        return torch.sum(sines * self.beta_sin, dim=-1)

    def forward(self, hidden_states):
        # 1. 全连接层 (Dense Layer)
        hidden_states = self.dense(hidden_states)
        
        # 2. Fourier GELU 近似逻辑
        # 公式: GELU(x) ≈ ReLU(x) + FourierSeries(|x|)  (当 |x| < width)
        
        # 计算 ReLU 和 |x|
        relu_x = F.relu(hidden_states)
        abs_x = torch.abs(hidden_states)
        
        # 判断是否在拟合区间内
        do_fs = abs_x < self.width
        
        # 计算傅里叶级数修正项 (Correction Term)
        # 注意：这里我们对所有输入都计算了级数，然后用 mask 过滤，
        # 这在 GPU 上通常比布尔索引更快，保持了 tensor 的形状。
        correction = self._fourier_series_sum(abs_x)
        
        # 组合结果: ReLU(x) + mask * Correction
        out = relu_x + do_fs.float() * correction
        
        return out
       
    
class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.exp_plus_one = ApproxExpPlusOne()
        self.approx_inv = ApproxInv()
        # 继承父类的关键参数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    # 核心修复：补充transpose_for_scores方法（和父类实现一致）
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        past_key_value=None,
        output_attentions=False,
        cache_position=None,
    ):
        # 兼容处理
        past_key_value = past_key_values if past_key_values is not None else past_key_value
        
        # 以下逻辑和父类一致，仅替换softmax为近似版本
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # clamp + Invsqrt开这个
        attention_probs = approx_softmax_from_expplus1(
            attention_scores, self.exp_plus_one ,dim=-1
        )
        attention_probs = self.dropout(attention_probs)

        # 应用head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# 计算 LayerNorm
class CustomBertLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-12):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.inv_sqrt = ApproxInvSqrt()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)   
        
        val = var + self.eps

        inv_std = self.inv_sqrt(val) 
        x_norm = (x - mean) * inv_std  
        return self.weight * x_norm + self.bias

# =====================================================

# ==================== 【5】替换模型 ====================
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
# =====================================================

# ==================== 【6】主程序 ====================
if __name__ == "__main__":
    print("Loading BERT-base...")
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Replacing with approximation modules...")
    model = replace_bert_modules(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print("Testing forward pass...")
    text = "Hello, this is a test sentence."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    print("✅ Forward pass successful!")
    print("Output shape:", outputs.last_hidden_state.shape)
    print("Testing backward pass...")
    dummy_label = torch.randn_like(outputs.last_hidden_state[:, 0, :])
    loss = F.mse_loss(outputs.last_hidden_state[:, 0, :], dummy_label)
    loss.backward()
    print("✅ Backward pass successful!")
# =====================================================