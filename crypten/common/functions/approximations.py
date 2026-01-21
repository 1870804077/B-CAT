#!/usr/bin/env python3

# Modified by Andes Y. L. Kei: Implemented alternative approximations for Sigmoid, Tanh, Erf, GELU, and Softmax
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import crypten
import torch
from crypten.config import cfg
import numpy as np
from scipy.linalg import lstsq
SIGMOID_K1 = 1  
SIGMOID_K2 = 4  
TANH_K1 = 1  
TANH_K2 = 9  
INVSQRT_K1 = 1  
INVSQRT_K2 = 12  
SILU_K1 = 1
SILU_K2 = 5
INV_K1 = 1
INV_K2 = 9
GELU_K1 = 1
GELU_K2 = 9
EXP_K1 = 1
EXP_K2 = 9
_SIGMOID_PARAMS_TABLE = {
    # K1=0: 纯傅里叶拟合 (波动较大)
    (0, 6): {
        "poly": [0.5],
        "beta": [0.56819311, -0.09202582, 0.12546985, -0.07389024, 0.06527241, -0.05254178]
    },
    
    # K1=1: 线性辅助 (0.5 + a1*x) + 傅里叶 (推荐)
    (1, 1): {
        "poly": [0.5, 0.05390335],
        "beta": [0.29366572]
    },
    (1, 4): {
        "poly": [0.5, 0.06269516],
        "beta": [0.2488894, 0.06762586, 0.01903559, 0.00593525]
    },
    (1, 5): {
        "poly": [0.5, 0.06239207],
        "beta": [0.25043303, 0.06685405, 0.01955013, 0.00554935, 0.00172095]
    },
    (1, 6): {
        "poly": [0.5, 0.0624801151],
        "beta": [0.249984614, 0.0670782568, 0.0194006578, 0.00566145155, 0.00163126762, 0.000492295253]
    },
    (1, 12): {
        "poly": [0.5, 0.0624586460],
        "beta": [
            2.50093955e-01, 6.70235864e-02, 1.94371047e-02, 5.63411647e-03, 
            1.65313561e-03, 4.74071997e-04, 1.42648050e-04, 3.85294631e-05, 
            1.32673657e-05, 2.43363945e-06, 1.72815558e-06, -2.34130542e-07
        ]
    },

    # K1=2: 三次项辅助 (0.5 + a1*x + a3*x^3)
    (2, 6): {
        "poly": [0.5, 0.0722486267, 0.0, -0.000153353636],
        "beta": [0.219831958, 0.0707592544, 0.0183534903, 0.00607753424, 0.00143514528, 0.000593830126]
    },
    (2, 12): {
        "poly": [0.5, 0.0629122725, 0.0, -7.09690622e-06],
        "beta": [
            0.248690610, 0.0671979055, 0.0193859974, 0.00565535681, 
            0.00164247158, 0.000480094052, 0.000138966817, 4.09097382e-05, 
            1.16639670e-05, 3.54683299e-06, 9.38036210e-07, 3.35455062e-07
        ]
    }
}

_TANH_PARAMS_TABLE = {
    # K1=1: 线性项 + 傅里叶
    (1, 6): {
        "poly": [-8.1079e-18, 0.126287526],
        "beta": [0.591412722, 0.253190535, 0.124365345, 0.0687274931, 0.0347071534, 0.0205011553]
    },
    (1, 9): {
        "poly": [-8.1079e-18, 0.12479694],
        "beta": [0.5990042, 0.2493948, 0.1268958, 0.0668296, 0.0362254, 0.0192359, 0.0106167, 0.0055195, 0.0031631]
    },
    (1, 12): {
        "poly": [-8.1079e-18, 0.125031860],
        "beta": [
            0.597807772, 0.249993014, 0.126497022, 0.0671287393, 0.0359861523, 
            0.0194353272, 0.0104457855, 0.00566907941, 0.00303016567, 0.00166115831, 
            0.000872903587, 0.000492551782
        ]
    },

    # K1=2: 三次项 + 傅里叶
    (2, 12): {
        "poly": [-4.8664385e-17, 0.184664352,0.0, -0.0009329],
        "beta": [
            0.413327858, 0.272908519, 0.119778593, 0.0699209355, 0.0345842884, 0.0202269700, 
            0.00996186071, 0.00598198382, 0.00281938730, 0.00180749566, 0.000769036671, 0.000567427937
        ]
    }
}

_INVSQRT_PARAMS_TABLE = {
    # K1=1: 线性辅助 (bias + a1*x) + 傅里叶
    (1, 1): {
        "poly": [1.55253719, -0.13239971],
        "beta": [-0.58977142]
    },
    (1, 4): {
        "poly": [2.22204530, -0.24123693],
        "beta": [-0.88762077, -0.37348595, -0.17342802, -0.11718896]
    },
    (1, 6): {
        "poly": [2.50225653, -0.27376923],
        "beta": [-1.07859805, -0.45610262, -0.23678581, -0.15815875, -0.09430349, -0.07103915]
    },
    (1, 9): {
        "poly": [2.81648107, -0.30464850],
        "beta": [
            -1.32129577, -0.53449902, -0.31736923, -0.19700209, -0.14227563, 
            -0.09654192, -0.07495807, -0.05143852, -0.04133197
        ]
    },
    (1, 12): {
        "poly": [3.05891722, -0.34037281],
        "beta": [
            -1.44794806, -0.62530109, -0.35936132, -0.2421499, -0.16720123, 
            -0.12636014, -0.09247449, -0.07350949, -0.05465957, -0.04457874, 
            -0.03279499, -0.02706155
        ]
    },

    # K1=2: 三次项辅助 (a0 + a1*x + a2*x^2 + a3*x^3)
    (2, 12): {
        "poly": [3.0660039, -4.57841701, 0.0, 0.06628997,],
        "beta": [
            11.65562655, -2.25574941, 0.11647787, -0.44163638, -0.06850735, -0.18332714, 
            -0.0587351, -0.09627229, -0.04017675, -0.05539183, -0.02581052,  -0.03271929
        ]
    }
}
_EXP_PARAMS_TABLE = {
    # K1=1: 线性项 + 傅里叶
    (1, 6): {
        "poly": [-8.1079e-18, 0.126287526],
        "beta": [0.591412722, 0.253190535, 0.124365345, 0.0687274931, 0.0347071534, 0.0205011553]
    },
    (1, 9): {
        "poly": [-8.1079e-18, 0.12479694],
        "beta": [0.5990042, 0.2493948, 0.1268958, 0.0668296, 0.0362254, 0.0192359, 0.0106167, 0.0055195, 0.0031631]
    },
    (1, 12): {
        "poly": [-8.1079e-18, 0.125031860],
        "beta": [
            0.597807772, 0.249993014, 0.126497022, 0.0671287393, 0.0359861523, 
            0.0194353272, 0.0104457855, 0.00566907941, 0.00303016567, 0.00166115831, 
            0.000872903587, 0.000492551782
        ]
    },

    # K1=2: 三次项 + 傅里叶
    (2, 12): {
        "poly": [-4.8664385e-17, 0.184664352,0.0, -0.0009329],
        "beta": [
            0.413327858, 0.272908519, 0.119778593, 0.0699209355, 0.0345842884, 0.0202269700, 
            0.00996186071, 0.00598198382, 0.00281938730, 0.00180749566, 0.000769036671, 0.000567427937
        ]
    }
}
__all__ = [
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "odrelu",
]

# Iterative methods:
def exp(self, k1=None, k2=None,fit_min=-16,fit_max=-4):
    r"""Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.作为 Gramine-TDX 的介绍 (基于 CCS '24 论文)
    """  # noqa: W605
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.exp_method
    iters = cfg.functions.exp_iterations

    if method == "ideal":
        return crypten.cryptensor(torch.exp(self.get_plain_text()), device=self.device)
    elif method == "limit":
        result = 1 + self.div(2**iters)
        for _ in range(iters):
            result = result.square()
        return result
    elif method == "newer":
        upper = -4.0
        lower = -16.0
        
        diffs = crypten.cat([self - upper, lower - self]).relu().split(self.shape[0])
        
        safe_x = self + diffs[1] - diffs[0]

        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2
        

        params = _EXP_PARAMS_TABLE[(use_k1, use_k2)]
        poly_coeffs = params["poly"]
        beta_sin_coeffs = params["beta"]
        period = 16.0
        
        poly_bias = poly_coeffs[0]
        poly_part = safe_x.polynomial(poly_coeffs[1:]) + poly_bias

        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        fourier_part = _fourier_series_x3(safe_x, len(beta_sin_coeffs), period, beta_sin=beta_sin)

        res = poly_part + fourier_part
        
        return res
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2
        
        # === 核心修改：锁定范围 [-16, -4] ===
        L_fit = 16.0 
        f_min = fit_min if fit_min is not None else -16
        f_max = fit_max if fit_max is not None else -2
        # 1. 现场计算拟合系数
        a0_val, poly_vals, beta_vals = _get_dynamic_params("exp",use_k1, use_k2, L_fit, fit_min, fit_max)

        # 2. 范围控制 (Range Reduction)
        upper = f_max  # -4.0
        lower = f_min  # -16.0
        
        # 使用 stack 必须小心，这里 diffs 计算逻辑保持原样即可
        diffs = crypten.stack([self - upper, lower - self]).relu()
        safe_x = self + diffs[1] - diffs[0]
        
        # 3. 计算多项式部分
        poly_part = safe_x.polynomial(list(poly_vals)) + a0_val
        
        # 4. 计算傅里叶部分
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_vals, device=self.device, dtype=torch.float)
        y_final = _fourier_series_x2x3(
            safe_x, 
            len(beta_vals), 
            period, 
            alpha=a0_val,          # 传入常数项
            beta_sin=beta_sin, 
            poly_coeffs=full_poly  # 【关键】传入多项式系数
        )
        # 5. 合成
        y_final = poly_part + fourier_part
        return y_final
    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else EXP_K1
        use_k2 = k2 if k2 is not None else EXP_K2
        
        L_fit = 16.0 
        fit_min = -16.0
        fit_max = -4.0
        
        # 1. 现场计算拟合系数
        a0_val, poly_vals, beta_vals = _get_dynamic_params("exp",use_k1, use_k2, L_fit, fit_min, fit_max)
        # 2. 范围控制 (Range Reduction)
        upper = fit_max  # -4.0
        lower = fit_min  # -16.0
        
        # 使用 stack 必须小心，这里 diffs 计算逻辑保持原样即可
        diffs = crypten.stack([self - upper, lower - self]).relu()
        safe_x = self + diffs[1] - diffs[0]
        full_poly = [0.0] + poly_vals
        # 3. 计算多项式部分
        poly_part = self.mul(full_poly[1])
        
        # 4. 计算傅里叶部分
        period = 2 * L_fit 
        beta_sin = torch.tensor(beta_vals, device=self.device, dtype=torch.float)
        fourier_part = _fourier_series_x2x3(safe_x, len(beta_vals), period,alpha=a0_val, beta_sin=beta_sin,poly_coeffs=full_poly)
        
        # 5. 合成
        y_final = poly_part + fourier_part
 
        return y_final
    else:
        raise ValueError(f"Invalid method {method} given for exp function")

def log(self, input_in_01=False):
    r"""
    Approximates the natural logarithm using 8th order modified
    Householder iterations. This approximation is accurate within 2% relative
    error on [0.0001, 250].

    Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

    .. math::

        y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the domain [0, 1],
            causing the function optimize for this domain. This is useful for computing
            log-probabilities for entropy functions.

            We shift the domain of convergence by a constant :math:`a` using the following identity:

            .. math::

                \ln{u} = \ln {au} - \ln{a}

            Since the domain of convergence for CrypTen's log() function is approximately [1e-4, 1e2],
            we can set :math:`a=100`.

    Configuration parameters:
        iterations (int): number of Householder iterations for the approximation
        exp_iterations (int): number of iterations for limit approximation of exp
        order (int): number of polynomial terms used (order of Householder approx)
    """
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    iterations = cfg.functions.log_iterations
    exp_iterations = cfg.functions.log_exp_iterations
    order = cfg.functions.log_order

    term1 = self.div(120)
    term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
    y = term1 - term2 + 3.0

    # 8th order Householder iterations
    with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
        for _ in range(iterations):
            h = 1 - self * exp(-y)
            y -= h.polynomial([1 / (i + 1) for i in range(order)])
    return y


def reciprocal(self, input_in_01=False, k1=None, k2=None, L=3000.0):
    r"""
    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

    Configuration params:
        reciprocal_method (str):  One of 'NR' or 'log'.
        reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method
        reciprocal_log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method
        reciprocal_all_pos (bool): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input.
        reciprocal_initial (tensor): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    pos_override = {"functions.reciprocal_all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64),k1=k1,k2=k2,L=L).mul(64)
        return rec

    # Get config options
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug" 
    # else:
    method = cfg.functions.reciprocal_method
    all_pos = cfg.functions.reciprocal_all_pos
    initial = cfg.functions.reciprocal_initial

    if method == "ideal":
        return crypten.cryptensor(torch.reciprocal(self.get_plain_text()), device=self.device)

    if not all_pos:
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos,k1=k1,k2=k2,L=L)

    if method == "NR":
        nr_iters = cfg.functions.reciprocal_nr_iters
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result += result - result.square().mul_(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal_log_iters
        with cfg.temp_override({"functions.log_iters": log_iters}):
            return exp(-log(self))
    elif method == "newer":
        use_k1 = k1 if k1 is not None else INV_K1
        use_k2 = k2 if k2 is not None else INV_K2
        i_sqrt = self.inv_sqrt(k1=use_k1, k2=use_k2)
        return i_sqrt.square()
    elif method == "newer_1":
        use_k1 = k1 if k1 is not None else INV_K1 
        use_k2 = k2 if k2 is not None else INV_K2
        
        threshold = 2.0
        L_small = threshold
        current_L = 32.0 
        L_large = current_L if current_L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        # --- Group A: 小范围 [0.1, 2.0] ---
        a0_s, poly_s, beta_s = _get_dynamic_params("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)
        period_s = 2 * L_small
        t_s, u_s, v_s = provider.generate_trig_triple(self.size(), period_s, len(beta_s), device=device)

        # --- Group B: 大范围 [2.0, 32.0] ---
        a0_l, poly_l, beta_l = _get_dynamic_params("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)
        period_l = 2 * L_large
        t_l, u_l, v_l = provider.generate_trig_triple(self.size(), period_l, len(beta_l), device=device)

        # 生成掩码
        is_large = (self - threshold).od_sign()
        is_small = 1.0 - is_large

        # 计算两段的 delta
        delta_s_share = self - t_s + period_s
        delta_l_share = self - t_l + period_l
        
        # 优化通信：打包 reveal
        stacked_shares = crypten.stack([delta_s_share, delta_l_share], dim=0)
        stacked_plain = stacked_shares.get_plain_text()
        
        delta_s_plain = stacked_plain[0]
        delta_l_plain = stacked_plain[1]

        # 内部函数：计算 Fourier + Polynomial
        # 这部分逻辑和 inv_sqrt 完全一样，可以直接复用
        def _compute_local_fourier(delta_p, period, beta_list, u_share, v_share, poly_coeffs, a0_val):
            delta = torch.remainder(delta_p, period)
            k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
            
            delta_k = torch.stack([i * delta for i in k_list])
            p = torch.sin(delta_k).to(device)
            q = torch.cos(delta_k).to(device)
            
            beta_tensor = torch.tensor(beta_list, device=device)
            view_shape = [-1] + [1] * self.dim()
            beta_tensor = beta_tensor.view(view_shape)

            fourier_val = ((v_share * p + u_share * q) * beta_tensor).sum(dim=0)
            poly_val = self.polynomial(poly_coeffs) + a0_val
            
            return poly_val + fourier_val

        # 计算初始猜测值 y0
        y_small = _compute_local_fourier(delta_s_plain, period_s, beta_s, u_s, v_s, poly_s, a0_s)
        y_large = _compute_local_fourier(delta_l_plain, period_l, beta_l, u_l, v_l, poly_l, a0_l)

        values = crypten.stack([y_small, y_large], dim=0)

        masks = crypten.stack([is_small, is_large], dim=0)
        products = values.mul(masks)
        y0 = products.sum(dim=0)
        
        y0_sq = y0.square()
        term1 = self.mul(y0_sq) # x * y0^2
        y1 = y0.mul(2.0) - term1

        y1_sq = y1.square()
        term2 = self.mul(y1_sq)
        y2 = y1.mul(2.0) - term2

        
        return y2
    elif method == "newer_time":
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        # 2. 获取拟合参数 (Small & Large)
        a0_s, poly_s, beta_s = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)
        a0_l, poly_l, beta_l = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)

        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        # 4. 计算分段掩码 (CMP Protocol)
        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        # 5. 一次性通信 Reveal
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
            
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        # 6. 本地重构
        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)
        is_small = 1.0 - is_large
        
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        # 7. 定义混合计算分支函数
        def _compute_hybrid_branch(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device) # sin(kd)
                q = torch.cos(delta_k).to(device) # cos(kd)
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            # 设置缩放因子防止溢出
            if i_flag == 0.0:
                s_sq = 32.0  
                s_cu = 32.0
            else:           
                s_sq = 256.0
                s_cu = 1024.0

            s_sq_pow2 = s_sq * s_sq
            s_cu_pow3 = s_cu * s_cu * s_cu 

            # --- 计算 x^2 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]
                
                # 变量缩放
                d_s2  = delta_raw / s_sq
                t_s2  = t / s_sq
                t2_s2 = t2 / s_sq_pow2
                d_s2_sq = d_s2.square()
                
                # x^2 = (d - t)^2 = d^2 - 2dt + t^2
                sq_term1 = d_s2_sq
                sq_term2 = t_s2 * (2 * d_s2) * -1.0
                sq_term3 = t2_s2
                x_sq_scaled = sq_term1 + sq_term2 + sq_term3
                
                # 融合乘法
                combined_c2 = c2 * s_sq_pow2
                poly_val += x_sq_scaled * combined_c2

            # --- 计算 x^3 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                # 变量缩放
                d_s3  = delta_raw / s_cu
                t_s3  = t / s_cu
                t2_s3 = t2 / (s_cu * s_cu)
                t3_s3 = t3 / (s_cu * s_cu * s_cu)
                d_s3_sq = d_s3.square()
                
                cu_term1 = d_s3 * d_s3_sq
                cu_term2 = t_s3 * (3 * d_s3_sq) * -1.0
                cu_term3 = t2_s3 * (3 * d_s3)
                x_cu_scaled = cu_term1 + cu_term2 + cu_term3 - t3_s3
                
                # 融合乘法
                combined_c3 = c3 * s_cu_pow3
                poly_val += x_cu_scaled * combined_c3

            total = poly_val + fourier_val
            return total

        # 8. 执行并行计算
        y_small = _compute_hybrid_branch(
            delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        y_large = _compute_hybrid_branch(
            delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        # 9. 合并结果
        y0 = y_large + is_small * (y_small - y_large)

        # 10. Newton 迭代 (2 Iterations)
        
        # Iteration 1
        y3_term1 = self.cube(y=y0, scale_y=False) 
        y1 = y0.mul(2.0) - y3_term1
        
        # Iteration 2
        y3_term2 = self.cube(y=y1, scale_y=False)
        y2 = y1.mul(2.0) - y3_term2

        return y2
    elif method == "newer_debug":

        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 3  # 建议默认提高 K1
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        a0_s, poly_s, beta_s = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_small, min_val=0.1)
        a0_l, poly_l, beta_l = _get_dynamic_params_1("reciprocal", use_k1, use_k2, L=L_large, min_val=L_small)
        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
            x_real=self.get_plain_text()
            
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)
        is_small = 1.0 - is_large
        
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        def _compute_hybrid_branch(x, delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device)
                q = torch.cos(delta_k).to(device)
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = x * poly_coeffs[0] 
                poly_val += lin_term

            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]
                poly_val += c2 * x * x

            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                poly_val += c3 * x * x * x

            total = poly_val + fourier_val
            return total

        y_small = _compute_hybrid_branch(
            x_real,delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        # Large Branch: 传入原始 t_l, 内部会进行除法缩放
        y_large = _compute_hybrid_branch(
            x_real,delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        # 8. 合并结果
        y0 = y_small * is_small + y_large * is_large
        _d("y0 (Initial)", y0)
        with crypten.no_grad():
            y0_real=y0.get_plain_text()
        # 9. Newton 迭代 
        # Iteration 1
        y3_term1=self.mul(y0_real*y0_real)
        y1 = -y3_term1+y0_real.mul(2.0)
        
        _d("y1 (Iter 1)", y1)
        # Iteration 2
        y3_term2=self.mul(y1*y1)
        y2 = -y3_term2+y1.mul(2.0)
        _d("y2 (Final)", y2)
        return y2
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self, k1=None, k2=None, L=8.0):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sqrt_method

    if method == "ideal":
        return crypten.cryptensor(torch.rsqrt(self.get_plain_text()), device=self.device)
    elif method == "NR":
        # Initialize using decent approximation
        if initial is None:
            y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
            y -= self.div(1024)
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iters):
            y = y.mul_(3 - self * y.square()).div_(2)
        return y
    elif method == "newer":
        use_k1 = k1 if k1 is not None else INVSQRT_K1
        use_k2 = k2 if k2 is not None else INVSQRT_K2
        
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L)
        
        period = 2 * L
        poly_part = self.polynomial(poly_body) + a0
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device)
        y0 = poly_part + _fourier_series_x3(self, len(beta_sin_coeffs), period, beta_sin=beta_sin)

        y0_cube = y0.cube()
        
        y1 = y0.mul(1.50131454).sub(self.mul(y0_cube.mul(0.500438180)))
        
        y2 = y1.mul(1.50000086).sub(self.mul(y1.cube().mul(0.499999)))
        
        return y2
    elif method == "newer_time":
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        # 2. 获取动态拟合参数
        a0_s, poly_s, beta_s = _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_small, min_val=0.1)
        a0_l, poly_l, beta_l = _get_dynamic_params_1("inv_sqrt", use_k1, use_k2, L=L_large, min_val=L_small)

        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        # 4. 计算分段掩码
        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        # 一次性通信 Reveal
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
            
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        # 5. 本地重构变量
        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)
        is_small = 1.0 - is_large
        
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        # 6. 定义混合计算分支函数
        def _compute_hybrid_branch(delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            # --- Fourier Part ---
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device) 
                q = torch.cos(delta_k).to(device) 
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            # --- Polynomial Part ---
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            # 设置缩放因子
            if i_flag == 0.0:
                s_sq = 32.0  
                s_cu = 32.0
            else:           
                s_sq = 256.0
                s_cu = 1024.0

            s_sq_pow2 = s_sq * s_sq
            s_cu_pow3 = s_cu * s_cu * s_cu 

            # --- 计算 x^2 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]
                
                # 变量缩放
                d_s2  = delta_raw / s_sq
                t_s2  = t / s_sq
                t2_s2 = t2 / s_sq_pow2
                d_s2_sq = d_s2.square()
                
                # x^2 = (d - t)^2 = d^2 - 2dt + t^2
                sq_term1 = d_s2_sq
                sq_term2 = t_s2 * (2 * d_s2) * -1.0
                sq_term3 = t2_s2
                x_sq_scaled = sq_term1 + sq_term2 + sq_term3
                
                # 融合乘法
                combined_c2 = c2 * s_sq_pow2
                poly_val += x_sq_scaled * combined_c2

            # --- 计算 x^3 (缩放 + 融合乘法) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                # 变量缩放
                d_s3  = delta_raw / s_cu
                t_s3  = t / s_cu
                t2_s3 = t2 / (s_cu * s_cu)
                t3_s3 = t3 / (s_cu * s_cu * s_cu)
                d_s3_sq = d_s3.square()
                
                cu_term1 = d_s3 * d_s3_sq
                cu_term2 = t_s3 * (3 * d_s3_sq) * -1.0
                cu_term3 = t2_s3 * (3 * d_s3)
                x_cu_scaled = cu_term1 + cu_term2 + cu_term3 - t3_s3
                
                # 融合乘法
                combined_c3 = c3 * s_cu_pow3
                poly_val += x_cu_scaled * combined_c3

            total = poly_val + fourier_val
            return total

        # 7. 执行并行计算
        y_small = _compute_hybrid_branch(
            delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        y_large = _compute_hybrid_branch(
            delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        # 8. 合并结果
        values = crypten.stack([y_small, y_large], dim=0)

        masks = crypten.stack([is_small, is_large], dim=0)

        products = values.mul(masks)

        y0 = products.sum(dim=0)

        # 9. Newton 迭代 
        
        # Iteration 1
        y3_term1 = y0.cube() 
        half_x = self.mul(0.500438180)
        
        # 计算被减数: 0.5 * x * y^3
        sub_term1 = half_x.mul(y3_term1)
        
        y1 = y0.mul(1.50131454) - sub_term1
        
        # Iteration 2
        y3_term2 = y1.cube().mul(0.999124984) 
        y2 = y1.mul(1.50000086) - half_x.mul(y3_term2)

        return y2
    elif method == "newer_debug":
        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        
        # 1. 基础配置 & 分段阈值
        use_k1 = k1 if k1 is not None else 5  # 建议默认提高 K1
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 8.0 
        L_small = threshold
        L_large = L if L > threshold else 32.0 
        
        provider = crypten.mpc.get_default_provider()
        device = self.device
        a0_s, poly_s, beta_s = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L=L_small, min_val=0.1)
        a0_l, poly_l, beta_l = _get_dynamic_params("inv_sqrt", use_k1, use_k2, L=L_large, min_val=L_small)
        period_s = 2 * L_small
        period_l = 2 * L_large

        # 3. 获取 Triples
        t_s, u_s, v_s, t2_s, t3_s = provider.generate_hybrid_triple(self.size(), period_s, len(beta_s), device=device)
        t_l, u_l, v_l, t2_l, t3_l = provider.generate_hybrid_triple(self.size(), period_l, len(beta_l), device=device)

        diff = self - threshold
        cmp_a, cmp_b, cmp_r, cmp_c = provider.generate_cmp_aux(self.size(), device=device)
        masked_sign_share = diff.mul(cmp_a).add(cmp_b)

        delta_s_share = self + t_s + period_s
        delta_l_share = self + t_l + period_l
        
        stacked_shares = crypten.stack([masked_sign_share, delta_s_share, delta_l_share], dim=0)
        
        with crypten.no_grad():
            stacked_plain = stacked_shares.get_plain_text()
            x_real=self.get_plain_text()
            
        masked_sign_plain = stacked_plain[0]
        delta_s_plain_raw = stacked_plain[1]
        delta_l_plain_raw = stacked_plain[2]

        V = (masked_sign_plain > 0).float()
        
        is_large = cmp_r.mul(V).add(cmp_c)
        is_small = 1.0 - is_large
        
        delta_s_exact = delta_s_plain_raw
        delta_l_exact = delta_l_plain_raw

        delta_s_mod = torch.fmod(delta_s_plain_raw, period_s)
        delta_l_mod = torch.fmod(delta_l_plain_raw, period_l)

        delta_s_mod[delta_s_mod < 0] += period_s
        delta_l_mod[delta_l_mod < 0] += period_l

        def _compute_hybrid_branch(x,delta_mod, delta_exact, period, beta_list, poly_coeffs, a0_val, 
                                   t, u, v, t2, t3, label, i_flag):
            
            fourier_val = 0
            if len(beta_list) > 0:
                k_list = [i * 2 * math.pi / period for i in range(1, len(beta_list) + 1)]
                delta_k = torch.stack([i * delta_mod for i in k_list])
                p = torch.sin(delta_k).to(device) # sin(kd)
                q = torch.cos(delta_k).to(device) # cos(kd)
                beta_tensor = torch.tensor(beta_list, device=device).view([-1] + [1] * self.dim())
                fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
            
            delta_raw = delta_exact - period
            
            poly_val = 0
            poly_val += a0_val
            
            if len(poly_coeffs) > 0:
                lin_term = self * poly_coeffs[0]
                poly_val += lin_term

            with crypten.no_grad():
                stacked_plain = stacked_shares.get_plain_text()
            # --- 计算 x^2 (使用 Fusion Mul) ---
            if len(poly_coeffs) > 1 and poly_coeffs[1] != 0:
                c2 = poly_coeffs[1]

                poly_val += x*x * c2

            # --- 计算 x^3 (使用 Fusion Mul) ---
            if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
                c3 = poly_coeffs[2]
                
                poly_val += x*x*x * c3 

            total = poly_val + fourier_val
            return total

        # 7. 执行并行计算
        y_small = _compute_hybrid_branch(
            x_real,delta_s_mod, delta_s_exact, period_s, beta_s, poly_s, a0_s, 
            t_s, u_s, v_s, t2_s, t3_s, "SMALL", 0.0
        )
        
        # Large Branch: 传入原始 t_l, 内部会进行除法缩放
        y_large = _compute_hybrid_branch(
            x_real,delta_l_mod, delta_l_exact, period_l, beta_l, poly_l, a0_l, 
            t_l, u_l, v_l, t2_l, t3_l, "LARGE", 1.0
        )

        # 8. 合并结果
        values = crypten.stack([y_small, y_large], dim=0)
        masks = crypten.stack([is_small, is_large], dim=0)
        products = values.mul(masks)
        y0 = products.sum(dim=0)
        _d("y0 (Initial)", y0)
        with crypten.no_grad():
            y0_real=y0.get_plain_text()
        # 9. Newton 迭代 

        y3_term1 = y0_real*y0_real*y0_real
        half_x = self.mul(0.500438180)
        
        sub_term1 = half_x.mul(y3_term1)
        
        y1 = y0.mul(1.50131454) - sub_term1
        with crypten.no_grad():
            y1_real=y1.get_plain_text()

        # Iteration 2 (逻辑保持不变，如果 y1 已经崩了，这里也会崩)
        y3_term2 = y1_real*y1_real*y1_real.mul(0.999124984) 
        y2 = y1.mul(1.50000086) - half_x.mul(y3_term2)
        return y2
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")


def sqrt(self, k1=None, k2=None):
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run
        sqrt_initial (tensor): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return self.inv_sqrt(k1=k1, k2=k2).mul(self)


def _eix(self):
    r"""Computes e^(i * self) where i is the imaginary unit.
    Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
    """
    iterations = cfg.functions.trig_iterations

    re = 1
    im = self.div(2**iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im.square()
    im *= 2

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul_(re)
        im._tensor *= 2
        re = a2 - b2

    return re, im


def cossin(self):
    r"""Computes cosine and sine of input via exp(i * x).

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return self._eix()


def cos(self):
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[0]


def sin(self):
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[1]


# Logistic Functions
def sigmoid(self, k1=None, k2=None,L=8.0):
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    If a valid method is given, this function will compute sigmoid
        using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with
        truncation and uses the identity:

    .. math::
        \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

    "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
        the reciprocal

    """  # noqa: W605
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.sigmoid(self.get_plain_text()), device=self.device)
    elif method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sign = 1 - 2 * ltz

        pos_input = self.mul(sign)
        denominator = pos_input.neg().exp().add(1)

        # TODO: Set these with configurable parameters
        with cfg.temp_override(
            {
                "functions.exp_iterations": 9,
                "functions.reciprocal_nr_iters": 3,
                "functions.reciprocal_all_pos": True,
                "functions.reciprocal_initial": 0.75,
            }
        ):
            pos_output = denominator.reciprocal()

        result = pos_output.where(1 - ltz, 1 - pos_output)
        # TODO: Support addition with different encoder scales
        # result = pos_output + ltz - 2 * pos_output * ltz
        return result
    elif method == "fourier":    
        m = cfg.functions.sigmoid_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.sigmoid_fs_terms

        # note that beta_cos = 0 for tanh
        alpha, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, alpha=alpha, beta_sin=beta_sin)
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        # 3. 调用混合评估器 (x3 版本)
        mixed_part = _fourier_series_x3_time(
            self, 
            terms=len(beta_sin_coeffs), 
            period=period, 
            alpha=a0, 
            beta_sin=beta_sin, 
            poly_coeffs=full_poly
        )
        
        # 4. 合并结果
        final_res = linear_term + mixed_part
        
        return final_res

    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else SIGMOID_K1
        use_k2 = k2 if k2 is not None else SIGMOID_K2

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L)
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        # 3. 调用混合评估器 (x3 版本)
        mixed_part = _fourier_series_x3(
            self, 
            terms=len(beta_sin_coeffs), 
            period=period, 
            alpha=a0, 
            beta_sin=beta_sin, 
            poly_coeffs=full_poly
        )
        
        # 4. 合并结果
        final_res = linear_term + mixed_part
        return final_res
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")

def tanh(self, k1=None, k2=None, L=6.0):
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    If a valid method is given, this function will compute tanh using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with truncation.

    .. math::
        tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

    where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
    The approximation is truncated to +/-1 outside [-1, 1].

    Args:
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    """
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.tanh(self.get_plain_text()), device=self.device)
    if method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh_terms
        coeffs = crypten.common.util.chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return out.hardtanh()
    elif method == "poly":
        drelu_x = self >= 0
        sign_x = 2 * drelu_x - 1
        abs_x = sign_x * self
        do_poly = abs_x < 2.95
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([1.1950192,-0.49313435,0.0737858,-0.00147019]) - 0.01758266
        out = sign_x * (do_poly * (poly_x - 1) + 1)
        return out
    elif method == "fourier":
        m = cfg.functions.tanh_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.tanh_fs_terms

        # note that alpha, beta_cos = 0 for tanh
        _, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, beta_sin=beta_sin)
    elif method == "ode":
        iter_num = cfg.functions.tanh_ode_iter_num
        x = self / iter_num
        y = self.new(torch.zeros_like(self.data), device=self.device)
        for _ in range(iter_num):
            y += (1 - y * y) * x
        return y
    elif method == "newer_time":
        use_k1 = k1 if k1 is not None else TANH_K1
        use_k2 = k2 if k2 is not None else TANH_K2

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", use_k1, use_k2, L)
        
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        # 3. 调用混合评估器 (x3 版本)
        mixed_part = _fourier_series_x3_time(
            self, 
            terms=len(beta_sin_coeffs), 
            period=period, 
            alpha=a0, 
            beta_sin=beta_sin, 
            poly_coeffs=full_poly
        )
        
        # 4. 合并结果
        final_res = linear_term + mixed_part
        
        return final_res

    elif method == "newer_debug":
        use_k1 = k1 if k1 is not None else TANH_K1
        use_k2 = k2 if k2 is not None else TANH_K2

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", use_k1, use_k2, L)
        # 1. 计算线性项
        linear_term = self.mul(poly_body[0])

        # 2. 准备参数
        full_poly = [0.0] + poly_body
        period = 2 * L
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        
        # 3. 调用混合评估器 (x3 版本)
        mixed_part = _fourier_series_x3(
            self, 
            terms=len(beta_sin_coeffs), 
            period=period, 
            alpha=a0, 
            beta_sin=beta_sin, 
            poly_coeffs=full_poly
        )
        
        # 4. 合并结果
        final_res = linear_term + mixed_part
        return final_res
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def _chebyshev_polynomials(self, terms):
    r"""Evaluates odd degree Chebyshev polynomials at x

    Chebyshev Polynomials of the first kind are defined as

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        self (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return crypten.stack(polynomials)

def _fourier_series_x3_time(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (x3 Only - Production Version):
    - Communication: Reveals (delta + period) to handle scale and positive range.
    - Fourier: Uses (delta + period) directly (periodicity holds).
    - Polynomial: Recovers (delta) locally to ensure x^3 correctness.
    """
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    # 1. 获取混合三元组
    provider = crypten.mpc.get_default_provider()
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)

    with crypten.no_grad():
        delta_share = self + t + period 
        
        delta_mod = delta_share.get_plain_text() 
        
    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    poly_res = 0
    if poly_coeffs is not None:
        delta_raw = delta_mod - period

        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]

            delta_sq = delta_raw.square()
            
            cube_term1 = delta_raw * delta_sq
            
            cube_term2 = t * (3 * delta_sq)
            cube_term3 = t2 * (3 * delta_raw)
            
            x_cube = cube_term1 + cube_term2 + cube_term3 + t3
            
            poly_res = poly_res + (x_cube * c3)

    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
        
    return final_res

def _fourier_series_x3(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (Debug Version with Error Analysis):
    Includes a debug block to find the input x with the maximum approximation error.
    """
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    # 1. 获取混合三元组
    provider = crypten.mpc.get_default_provider()
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)

    # 2. 掩码与明文获取 (Debug模式: 获取 x_mod 用于计算 Ground Truth)
    with crypten.no_grad():
        delta_share = self + t + period 
        x_mod = self.get_plain_text()  # 获取明文 x，用于后续计算误差
        delta_mod = delta_share.get_plain_text() 
        
    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    # 3. 傅里叶部分计算
    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    # 4. 多项式部分计算 (基于明文 x^3 的 Golden Baseline)
    poly_res = 0
    if poly_coeffs is not None:
        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]
            # 这里使用了明文 x 计算 x^3，精度是最高的
            poly_res = poly_res + (x_mod * x_mod * x_mod * c3)

    # 5. 合并结果
    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
    # =========================================================================

    return final_res

def _fourier_series_x2x3_time(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (Final Correct Version)
    """
    # 参数校验
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    provider = crypten.mpc.get_default_provider()
    
    # 1. 获取预计算的三元组/辅助参数 (加密状态)
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)
    
    # 2. 计算 Delta (揭示掩码后的差值)
    with crypten.no_grad():
        delta_share = self + t +period
        delta_mod = delta_share.get_plain_text() 
    delta_raw = delta_mod - period


    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    poly_res = 0
    if poly_coeffs is not None:
        delta_sq = delta_raw.square() 
        # 计算 x^2
        if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
            c2 = poly_coeffs[2]
            
            sq_term1 = delta_sq                    # d^2
            sq_term2 = t * (2 * delta_raw)         # 2dt (Crypto * Plain)
            
            sq_term3 = t2                          # t^2
            
            x_square = sq_term1 - sq_term2 + sq_term3
            poly_res = poly_res + (x_square * c2)

        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]
            
            cube_term1 = delta_raw * delta_sq      # d^3
            cube_term2 = t * (3 * delta_sq)        # 3d^2t
            cube_term3 = t2 * (3 * delta_raw)      # 3dt^2

            x_cube = cube_term1 - cube_term2 + cube_term3 - t3
            poly_res = poly_res + (x_cube * c3)

    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
        
    return final_res

def _fourier_series_x2x3(self, terms, period, alpha=None, beta_cos=None, beta_sin=None, poly_coeffs=None):
    r"""
    Hybrid Evaluator (Final Correct Version)
    """
    # 参数校验
    if beta_cos is not None:
        raise NotImplementedError("Fourier series with cosine is currently not supported")
    if beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device
    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    provider = crypten.mpc.get_default_provider()
    
    # 1. 获取预计算的三元组/辅助参数 (加密状态)
    t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, terms, device=device)
    
    # 2. 计算 Delta (揭示掩码后的差值)
    with crypten.no_grad():
        delta_share = self + t +period
        delta_mod = delta_share.get_plain_text() 
        x_plain=self.get_plain_text() 
    delta_raw = delta_mod - period


    delta_k = torch.stack([i * delta_mod for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    fourier_res = ((v * p - u * q) * beta_sin).sum(dim=0)

    poly_res = 0
    if poly_coeffs is not None:
        delta_sq = delta_raw.square() 
        # 计算 x^2
        if len(poly_coeffs) > 2 and poly_coeffs[2] != 0:
            c2 = poly_coeffs[2]
            poly_res = poly_res + x_plain*x_plain*c2

        if len(poly_coeffs) > 3 and poly_coeffs[3] != 0:
            c3 = poly_coeffs[3]
            
            poly_res = poly_res + x_plain*x_plain*x_plain*c3

    if isinstance(poly_res, (int, float)) and poly_res == 0:
        final_res = fourier_res
    else:
        final_res = fourier_res + poly_res

    if alpha is not None:
        final_res = final_res + alpha
        
    return final_res
def erf(tensor):
    r"""
    Approximates the error function of the input tensor.
    """
    method = cfg.functions.erf_method

    if method == "ideal":
        return crypten.cryptensor(torch.erf(tensor.get_plain_text()), device=tensor.device)
    elif method == "taylor":
        iters = cfg.functions.erf_iterations

        output = tensor.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(tensor.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
        # NOTE: This approximation is not unstable for large tensor values.
    elif method == "tanh":
        return tanh(math.sqrt(4 / math.pi) * (tensor + 0.044715 * tensor.pow(3)))
    elif method == "fourier":
        period = cfg.functions.erf_fs_period
        width = period / 2
        terms = cfg.functions.erf_fs_terms

        # note that alpha, beta_cos = 0 for erf
        _, _, beta_sin = crypten.common.util.fourier_series(torch.erf, width, terms)
        return _fourier_series(tensor, terms, period, beta_sin=beta_sin)
    else:
        raise ValueError(f"Unrecognized method {method} for erf")

def _diff_gelu(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="none") - torch.nn.functional.relu(x))

def _diff_gelu_tanh(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="tanh") - torch.nn.functional.relu(x))

def _diff_silu(x):
    return torch.sign(x) * (torch.nn.functional.silu(x) - torch.nn.functional.relu(x))

def gelu(self, approximate="none", k1=None, k2=None):
    r"""Compute the Gaussian error linear unit of a tensor"""
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.gelu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.gelu(self.get_plain_text(), approximate=approximate), device=self.device)
    elif method == "fourier":
        period = cfg.functions.gelu_fs_period
        width = period / 2
        terms = cfg.functions.gelu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        if approximate == "tanh":
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu_tanh, width, terms)
            beta_sin = torch.tensor([-0.0817,-0.0812,-0.0424,-0.0175,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        else:
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu, width, terms)
            beta_sin = torch.tensor([-0.0818,-0.0809,-0.0424,-0.0176,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    elif method == "secformer":
        # set erf_fs_period: 20, erf_fs_terms: 7
        b0, b1 = self > -1.7 * math.sqrt(2), self < 1.7 * math.sqrt(2)
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    elif method == "poly":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 3
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([-0.55386347,0.5658561,-0.19719836,0.02328962]) + 0.00410626
        return relu_x + do_poly * poly_x
    elif method == "bolt":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 2.7
        # Motzkin's polynomial preprocessing
        g = [0.14439048359960427, -0.7077117131613893, 4.5702822654246535, -8.15444702051307, 16.382265425072532]
        poly_x = (g[0] * abs_x + g[1]) * abs_x + g[2]
        poly_x = (poly_x + g[0] * abs_x + g[3]) * abs_x + g[4] + 0.5 * self
        # The g's provided by BOLT are wrong, uncomment the following line to get the correct approximation
        #poly_x = abs_x.polynomial([-0.53798164612714154,0.5410550166368381,-0.18352506127082727,0.020848611754127593]) + 0.001620808531841547
        return relu_x + do_poly * poly_x
    elif method == "erf":
        # set erf_fs_period: 16, erf_fs_terms: 5
        b0, b1 = self > -2, self < 2
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    elif method == "newer":
        use_k1 = k1 if k1 is not None else GELU_K1
        use_k2 = k2 if k2 is not None else GELU_K2

        threshold = 4.0 
        stacked_inputs = crypten.stack([self - threshold, self.neg() - threshold])
        indicators = stacked_inputs.od_sign() 
        
        is_pos_large = indicators[0]
        is_neg_large = indicators[1] 
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0

        # --- 2. 计算 Tanh 参数 ---
        sqrt_2_pi = math.sqrt(2 / math.pi)
        
        x3 = self.cube()
        
        # 公式: sqrt(2/pi) * (0.044715 * x^3 + x)
        inner_poly = x3.mul(0.044715).add(self).mul(sqrt_2_pi)
        
        tanh_out = inner_poly.tanh(k1=use_k1, k2=use_k2)
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))

        # --- 3. 最终合成 ---
        return self.mul(is_pos_large) + mid_res.mul(is_mid)
    elif method == "newer_debug":
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        tanh_k1, tanh_k2, tanh_L = TANH_K1, TANH_K2, 6.0 
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", tanh_k1, tanh_k2, tanh_L)
        full_poly = [0.0] + poly_body
        period = 2 * tanh_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        # Cube 和 Tanh 的混合三元组
        t_cube, t2_cube, t3_cube = provider.cube(self.size(), device=device)
        
        t_tanh, u_tanh, v_tanh, t2_tanh, t3_tanh = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        
        inputs_cmp = crypten.stack([self - threshold, self.neg() - threshold])
        masked_share = inputs_cmp.mul(a_all).add(b_all)
        
        delta_cube_share = self
        
        with crypten.no_grad():
            delta_cube_plain = delta_cube_share.get_plain_text()
            
        delta_sq = delta_cube_plain.square()
        delta_cu = delta_sq * delta_cube_plain
        
        x3 = delta_cube_plain**3
        
        # 计算 Tanh 输入 z
        z_share = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        
        delta_tanh_share = z_share + t_tanh + period
        
        delta_tanh_expanded = delta_tanh_share.unsqueeze(0)
        comm_block_1 = crypten.cat([masked_share, delta_tanh_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            
        masked_plain = plain_block_1[0:2] # 前两个是 CMP Mask
        delta_tanh_raw = plain_block_1[2] # 第三个是 Tanh Delta
        
        V = (masked_plain > 0).float()
        indicators_ast = r_all.mul(V).add(c_all)
        
        indicators = indicators_ast
        is_pos_large = indicators[0]
        is_neg_large = indicators[1]
        
        delta_mod = delta_tanh_raw
        
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_tanh * p - u_tanh * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
        with crypten.no_grad():
            z_plain = z_share.get_plain_text()

        if len(full_poly) > 3 and full_poly[3] != 0:
            
            poly_res = poly_res + (z_plain *z_plain *z_plain * c3)
        
        tanh_out = poly_res + fourier_res 
        
        # =========================================================
        # 6. Final GeLU Mix
        # =========================================================
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))
        
        # 计算 is_mid: 1 - pos - neg
        is_mid = self + 1.0 - is_pos_large - is_neg_large- self
        
        # 计算 pos_mask: self * is_pos_large 的逻辑等价
        pos_mask = self + is_pos_large - self
        
        final_lhs = crypten.stack([self, mid_res])
        # is_pos_large 已经被转回 MPCTensor，可以安全 stack
        final_rhs = crypten.stack([pos_mask, is_mid])
        
        gelu_res = final_lhs.mul(final_rhs).sum(dim=0)
        
        return gelu_res
    elif method == "newer_time":
        # --- 0. 准备常数与提供者 ---
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        

        tanh_k1, tanh_k2, tanh_L = TANH_K1, TANH_K2, 6.0 # 假设 L=4.0
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", tanh_k1, tanh_k2, tanh_L)
        full_poly = [0.0] + poly_body
        period = 2 * tanh_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        u_cmp, v_cmp, w_cmp = provider.generate_triples(double_size, device=device)
        
        t_cube, t2_cube, t3_cube = provider.cube(self.size(), device=device)
        
        t_tanh, u_tanh, v_tanh, t2_tanh, t3_tanh = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        
        inputs_cmp = crypten.stack([self - threshold, self.neg() - threshold])
        
        # Beaver Mul 核心公式: x*y = [w] + e_x*[v] + e_y*[u] + e_x*e_y
        # 我们需要 Reveal: eps_x = inputs - u, eps_y = a_all - v
        eps_x_share = inputs_cmp - u_cmp
        eps_y_share = self+a_all - v_cmp-self
        
        # --- B. 准备 Cube 的输入 ---
        # 需要 Reveal: delta = x - t
        delta_cube_share = self - t_cube
        
        # --- C. 打包并通信 (Stack & Reveal) ---
        delta_cube_expanded = delta_cube_share.unsqueeze(0) # (1, ...)
        
        comm_block_1 = crypten.cat([eps_x_share, eps_y_share, delta_cube_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            
        eps_x = plain_block_1[0:2]
        eps_y = plain_block_1[2:4]
        delta_cube_plain = plain_block_1[4]
        
        # =================================================================
        # [Local Computation 1] 完成乘法 & 准备 Tanh 输入
        # =================================================================
        
        # 1. 完成 CMP 的乘法 (masked = inputs * a + b)
        # Beaver Reconstruction: w + eps_x * v + eps_y * u + eps_x * eps_y
        product = w_cmp + v_cmp.mul(eps_x) + u_cmp.mul(eps_y) + eps_x * eps_y
        masked_share = self+product + b_all-self
        
        # 2. 完成 Cube 计算 (x^3)
        delta_sq = delta_cube_plain.square()
        delta_cu = delta_sq * delta_cube_plain
        x3 = t3_cube + t2_cube.mul(3 * delta_cube_plain) + t_cube.mul(3 * delta_sq) + delta_cu
        
        # 3. 计算 Tanh 输入 z
        # z = sqrt(2/pi) * (x + coeff * x^3)
        # 注意: 这里直接用 self (x) 和 x3 组合
        z_share = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        
        # =================================================================
        # [Round 2] 并行处理: (CMP Reveal) & (Tanh Delta Reveal)
        # =================================================================
        
        
        delta_tanh_share = z_share - t_tanh + period
        
        # --- C. 打包并通信 ---
        delta_tanh_expanded = delta_tanh_share.unsqueeze(0)
        comm_block_2 = crypten.cat([masked_share, delta_tanh_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
            
        # --- D. 解包数据 ---
        masked_plain = plain_block_2[0:2]
        delta_tanh_plain = plain_block_2[2]
        
        # =================================================================
        # [Local Computation 2] 重构结果
        # =================================================================
        
        # 1. 重构 CMP 指示器
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0]
        is_neg_large = indicators[1]
        
        # 2. 重构 Tanh 结果 (Hybrid Fourier)
        delta_mod = delta_tanh_plain
        delta_raw = delta_mod - period
        
        # 傅里叶部分
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        
        fourier_res = ((v_tanh * p - u_tanh * q) * beta_view).sum(dim=0)
        
        # 多项式部分 (这里用 x3 逻辑)
        poly_res = a0
        # 线性项 (c1 * z)
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
            
        # 三次项 (c3 * z^3) -> 需要利用 hybrid triples t, t2, t3
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            d_sq = delta_raw.square()
            cube_t1 = delta_raw * d_sq
            cube_t2 = t_tanh * (3 * d_sq)
            cube_t3 = t2_tanh * (3 * delta_raw)
            z_cube = cube_t1 + cube_t2 + cube_t3 + t3_tanh
            poly_res += z_cube * c3
            
        tanh_out = poly_res+fourier_res
        
        # =================================================================
        # [Round 3] 最终合成 (必须的一轮乘法)
        # =================================================================

        
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0)) # 1轮通信

        is_mid = self+(is_pos_large + is_neg_large).neg() + 1.0-self
        
        # 显式 Stack 优化最后一轮
        final_lhs = crypten.stack([self, mid_res])
        final_rhs = crypten.stack([self+is_pos_large-self, is_mid])
        
        final_terms = final_lhs.mul(final_rhs) # Round 3 Comm
        
        gelu_res = final_terms[0] + final_terms[1]
        
        return gelu_res
    elif method == "newer_debug_1":
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]
        
        t, t2, t3 = provider.cube(self.size(), device=self.device, mode="cube")
        
        masked_pos = (self - threshold).mul(a1).add(b1)
        masked_neg = (self.neg() - threshold).mul(a2).add(b2)
        delta_share = self - t
        
        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            x_plain = self.get_plain_text()
        # --- 4. 解包与重构 ---
        V_pos = (all_plain[0] > 0).float()
        V_neg = (all_plain[1] > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        is_neg_large = r2.mul(V_neg).add(c2)
        
        delta_plain = all_plain[2] 
        x3 = x_plain*x_plain*x_plain

        # --- 5. Tanh 计算 ---
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0
        inner_poly_ast = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        tanh_out = inner_poly_ast.tanh(k1=k1, k2=k2)
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))

        # --- 6. 最终合成 ---
        effective_share_ast = self.mul(is_pos_large)+mid_res * is_mid 
        gelu_res = effective_share_ast
        return gelu_res
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")

def silu(self, k1=None, k2=None,L=4.0):
    r"""Compute the Sigmoid linear unit of a tensor with global variable support"""
    # 自动判定 method
    # if k1 is not None or k2 is not None:
    #     method = "newer_debug"
    # else:
    method = cfg.functions.silu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.silu(self.get_plain_text()), device=self.device)
    elif method == "fourier":
        period = cfg.functions.silu_fs_period
        width = period / 2
        terms = cfg.functions.silu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        #_, _, beta_sin = crypten.common.util.fourier_series(_diff_silu, width, terms)
        beta_sin = torch.tensor([-0.1299, -0.1220, -0.0743, -0.0394, -0.0216, -0.0118, \
                                 -0.0074, -0.0044, -0.0033, -0.0021, -0.0018, -0.0011], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    
    elif method == "newer":
        use_k1 = k1 if k1 is not None else SILU_K1
        use_k2 = k2 if k2 is not None else SILU_K2

        # --- Debug 准备：预先获取输入明文 ---
        x_plain = self.get_plain_text()

        # --- 1. 分段逻辑 ---
        threshold = 12.0
        stacked_inputs = crypten.stack([self - threshold, self.neg() - threshold])
        # 必须对 stacked_inputs 调用 od_sign 以获得两个指示器
        indicators = stacked_inputs.od_sign()
        
        is_pos_large = indicators[0]
        is_neg_large = indicators[1] 
        is_mid = (is_pos_large + is_neg_large).neg() + 1.0
        fit_L=threshold+1.0
        # --- 2. 现场拟合 Sigmoid 部分 ---
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("sigmoid", use_k1, use_k2, fit_L)
        
        period = 2 * fit_L
        poly_part = self.polynomial(poly_body) + a0
        beta_sin = torch.tensor(beta_sin_coeffs, device=self.device, dtype=torch.float)
        fourier_part = _fourier_series_x3(self, len(beta_sin_coeffs), period, beta_sin=beta_sin)
        
        sig_x_approx = poly_part + fourier_part
        # 计算中间段：x * sigmoid(x)
        mid_res = self.mul(sig_x_approx)

        # --- 3. 最终结果合成 ---
        # x > 5 -> x; x < -5 -> 0; mid -> x * sig_approx
        res = self.mul(is_pos_large) + mid_res.mul(is_mid)

        # # ==================== TOP 10 ERRORS DEBUG START ====================
        # # 计算标准真值 (使用 torch.nn.functional)
        import torch.nn.functional as F_debug # 使用别名避免冲突
        true_silu = F_debug.silu(x_plain)
        
        # 提取中间项明文用于分析
        res_plain = res.get_plain_text()
        is_pos_plain = is_pos_large.get_plain_text()
        is_mid_plain = is_mid.get_plain_text()
        sig_x_plain = sig_x_approx.get_plain_text()
        mid_res_plain = mid_res_plain = mid_res.get_plain_text()


        return res
    elif method == "newer_debug":
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        
        tanh_k1, tanh_k2, tanh_L = TANH_K1, TANH_K2, 6.0 
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", tanh_k1, tanh_k2, tanh_L)
        full_poly = [0.0] + poly_body
        period = 2 * tanh_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        # Cube 和 Tanh 的混合三元组
        t_cube, t2_cube, t3_cube = provider.cube(self.size(), device=device)
        
        t_tanh, u_tanh, v_tanh, t2_tanh, t3_tanh = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        
        inputs_cmp = crypten.stack([self - threshold, self.neg() - threshold])
        masked_share = inputs_cmp.mul(a_all).add(b_all)
        
        delta_cube_share = self
        
        with crypten.no_grad():
            delta_cube_plain = delta_cube_share.get_plain_text()
            
        delta_sq = delta_cube_plain.square()
        delta_cu = delta_sq * delta_cube_plain
        
        x3 = delta_cube_plain**3
        
        # 计算 Tanh 输入 z
        z_share = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        
        delta_tanh_share = z_share + t_tanh + period
        
        delta_tanh_expanded = delta_tanh_share.unsqueeze(0)
        comm_block_1 = crypten.cat([masked_share, delta_tanh_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            
        masked_plain = plain_block_1[0:2] # 前两个是 CMP Mask
        delta_tanh_raw = plain_block_1[2] # 第三个是 Tanh Delta
        
        V = (masked_plain > 0).float()
        indicators_ast = r_all.mul(V).add(c_all)
        
        indicators = indicators_ast
        is_pos_large = indicators[0]
        is_neg_large = indicators[1]
        
        delta_mod = delta_tanh_raw
        
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        fourier_res = ((v_tanh * p - u_tanh * q) * beta_view).sum(dim=0)
        
        poly_res = a0
        
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
        with crypten.no_grad():
            z_plain = z_share.get_plain_text()

        if len(full_poly) > 3 and full_poly[3] != 0:
            
            poly_res = poly_res + (z_plain *z_plain *z_plain * c3)
        
        tanh_out = poly_res + fourier_res 
        
        # =========================================================
        # 6. Final GeLU Mix
        # =========================================================
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0))
        
        # 计算 is_mid: 1 - pos - neg
        is_mid = self + 1.0 - is_pos_large - is_neg_large- self
        
        # 计算 pos_mask: self * is_pos_large 的逻辑等价
        pos_mask = self + is_pos_large - self
        
        final_lhs = crypten.stack([self, mid_res])
        # is_pos_large 已经被转回 MPCTensor，可以安全 stack
        final_rhs = crypten.stack([pos_mask, is_mid])
        
        gelu_res = final_lhs.mul(final_rhs).sum(dim=0)
        
        return gelu_res
    elif method == "newer_time":
        # --- 0. 准备常数与提供者 ---
        coeff_x3 = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        threshold = 4.0
        provider = crypten.mpc.get_default_provider()
        device = self.device
        

        tanh_k1, tanh_k2, tanh_L = TANH_K1, TANH_K2, 6.0 # 假设 L=4.0
        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("tanh", tanh_k1, tanh_k2, tanh_L)
        full_poly = [0.0] + poly_body
        period = 2 * tanh_L
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)

        double_size = (2,) + self.size()
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        u_cmp, v_cmp, w_cmp = provider.generate_triples(double_size, device=device)
        
        t_cube, t2_cube, t3_cube = provider.cube(self.size(), device=device)
        
        t_tanh, u_tanh, v_tanh, t2_tanh, t3_tanh = provider.generate_hybrid_triple(
            self.size(), period, len(beta_sin_coeffs), device=device
        )
        
        inputs_cmp = crypten.stack([self - threshold, self.neg() - threshold])
        
        # Beaver Mul 核心公式: x*y = [w] + e_x*[v] + e_y*[u] + e_x*e_y
        # 我们需要 Reveal: eps_x = inputs - u, eps_y = a_all - v
        eps_x_share = inputs_cmp - u_cmp
        eps_y_share = self+a_all - v_cmp-self
        
        # --- B. 准备 Cube 的输入 ---
        # 需要 Reveal: delta = x - t
        delta_cube_share = self - t_cube
        
        # --- C. 打包并通信 (Stack & Reveal) ---
        delta_cube_expanded = delta_cube_share.unsqueeze(0) # (1, ...)
        
        comm_block_1 = crypten.cat([eps_x_share, eps_y_share, delta_cube_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_1 = comm_block_1.get_plain_text()
            
        eps_x = plain_block_1[0:2]
        eps_y = plain_block_1[2:4]
        delta_cube_plain = plain_block_1[4]
        
        # =================================================================
        # [Local Computation 1] 完成乘法 & 准备 Tanh 输入
        # =================================================================
        
        # 1. 完成 CMP 的乘法 (masked = inputs * a + b)
        # Beaver Reconstruction: w + eps_x * v + eps_y * u + eps_x * eps_y
        product = w_cmp + v_cmp.mul(eps_x) + u_cmp.mul(eps_y) + eps_x * eps_y
        masked_share = self+product + b_all-self
        
        # 2. 完成 Cube 计算 (x^3)
        delta_sq = delta_cube_plain.square()
        delta_cu = delta_sq * delta_cube_plain
        x3 = t3_cube + t2_cube.mul(3 * delta_cube_plain) + t_cube.mul(3 * delta_sq) + delta_cu
        
        # 3. 计算 Tanh 输入 z
        # z = sqrt(2/pi) * (x + coeff * x^3)
        # 注意: 这里直接用 self (x) 和 x3 组合
        z_share = self.add(x3.mul(coeff_x3)).mul(sqrt_2_pi)
        
        # =================================================================
        # [Round 2] 并行处理: (CMP Reveal) & (Tanh Delta Reveal)
        # =================================================================
        
        
        delta_tanh_share = z_share - t_tanh + period
        
        # --- C. 打包并通信 ---
        delta_tanh_expanded = delta_tanh_share.unsqueeze(0)
        comm_block_2 = crypten.cat([masked_share, delta_tanh_expanded], dim=0)
        
        with crypten.no_grad():
            plain_block_2 = comm_block_2.get_plain_text()
            
        # --- D. 解包数据 ---
        masked_plain = plain_block_2[0:2]
        delta_tanh_plain = plain_block_2[2]
        
        # =================================================================
        # [Local Computation 2] 重构结果
        # =================================================================
        
        # 1. 重构 CMP 指示器
        V = (masked_plain > 0).float()
        indicators = r_all.mul(V).add(c_all)
        is_pos_large = indicators[0]
        is_neg_large = indicators[1]
        
        # 2. 重构 Tanh 结果 (Hybrid Fourier)
        delta_mod = delta_tanh_plain
        delta_raw = delta_mod - period
        
        # 傅里叶部分
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([d * delta_mod for d in k_list]) 
        
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        
        beta_view = beta_sin.view([-1] + [1] * self.dim())
        
        fourier_res = ((v_tanh * p - u_tanh * q) * beta_view).sum(dim=0)
        
        # 多项式部分 (这里用 x3 逻辑)
        poly_res = a0
        # 线性项 (c1 * z)
        if len(full_poly) > 1:
            poly_res += z_share * full_poly[1]
            
        # 三次项 (c3 * z^3) -> 需要利用 hybrid triples t, t2, t3
        if len(full_poly) > 3 and full_poly[3] != 0:
            c3 = full_poly[3]
            d_sq = delta_raw.square()
            cube_t1 = delta_raw * d_sq
            cube_t2 = t_tanh * (3 * d_sq)
            cube_t3 = t2_tanh * (3 * delta_raw)
            z_cube = cube_t1 + cube_t2 + cube_t3 + t3_tanh
            poly_res += z_cube * c3
            
        tanh_out = poly_res+fourier_res
        
        # =================================================================
        # [Round 3] 最终合成 (必须的一轮乘法)
        # =================================================================

        
        mid_res = self.mul(0.5).mul(tanh_out.add(1.0)) # 1轮通信

        is_mid = self+(is_pos_large + is_neg_large).neg() + 1.0-self
        
        # 显式 Stack 优化最后一轮
        final_lhs = crypten.stack([self, mid_res])
        final_rhs = crypten.stack([self+is_pos_large-self, is_mid])
        
        final_terms = final_lhs.mul(final_rhs) # Round 3 Comm
        
        gelu_res = final_terms[0] + final_terms[1]
        
        return gelu_res
    elif method == "newer_time_1":
        # Debug 工具
        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 12.0 
        L_fit = max(float(L), threshold + 1.0)
        period = 2 * L_fit
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params("sigmoid", use_k1, use_k2, L=L_fit)
        full_poly_coeffs = [0.0] + poly_body 
        
        # --- [优化修改开始] 批量生成随机数 (Batch Generation) ---
        # 1. 构造双倍尺寸：(2, Batch_Size, ...)
        double_size = (2,) + self.size()
        
        # 2. 一次性生成两组辅助参数
        # 返回的 a_all, b_all 等都是 shape 为 (2, ...) 的 Share
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        # 3. 切片分离 (Slicing)
        # 注意：Crypten 的切片操作通常是零拷贝的，非常快
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]
        # --- [优化修改结束] ---

        diff_pos = self - threshold
        masked_pos = diff_pos.mul(a1).add(b1)

        diff_neg = self + threshold
        masked_neg = diff_neg.mul(a2).add(b2) # M2 = (x+12)*a2 + b2

        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, len(beta_sin_coeffs), device=device)
        delta_share = self + t + period

        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            
        masked_pos_plain = all_plain[0]
        masked_neg_plain = all_plain[1]
        delta_plain_raw = all_plain[2]

        V_pos = (masked_pos_plain > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        
        V_neg = (masked_neg_plain > 0).float()
        is_larger_than_neg12 = r2.mul(V_neg).add(c2)
        
        is_neg_large = 1.0 - is_larger_than_neg12
        
        is_mid = 1.0 - is_pos_large - is_neg_large

        # --- B. 还原拟合结果 (Hybrid Fitting) ---
        delta_exact = delta_plain_raw
        delta_mod = torch.fmod(delta_plain_raw, period)
        delta_mod[delta_mod < 0] += period
        
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        
        # --- 计算 Sigmoid 近似值 (内联 Hybrid Branch) ---
        fourier_val = 0
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([i * delta_mod for i in k_list])
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_tensor = beta_sin.view([-1] + [1] * self.dim())
        fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
        
        delta_raw = delta_exact - period
        poly_val = a0
        
        # 线性项
        if len(full_poly_coeffs) > 1:
            poly_val += self * full_poly_coeffs[1]
            
        delta_sq = delta_raw.square()
        
        # x^2 = (d-t)^2
        if len(full_poly_coeffs) > 2 and full_poly_coeffs[2] != 0:
            c2 = full_poly_coeffs[2]
            term_sq = delta_sq - t * (2 * delta_raw) + t2
            poly_val += term_sq * c2
            
        # x^3 = (d-t)^3
        if len(full_poly_coeffs) > 3 and full_poly_coeffs[3] != 0:
            c3 = full_poly_coeffs[3]
            cube_term1 = delta_raw * delta_sq
            cube_term2 = t * (3 * delta_sq)
            cube_term3 = t2 * (3 * delta_raw)
            term_cu = cube_term1 - cube_term2 + cube_term3 - t3
            poly_val += term_cu * c3
            
        sig_approx = poly_val + fourier_val

        effective_share = sig_approx.mul(is_mid)+ is_pos_large

        silu_res = self.mul(effective_share)
        return silu_res
    elif method == "newer_debug_1":
        # Debug 工具
        debug_history = []
        def _d(name, var):
            if isinstance(var, torch.Tensor):
                debug_history.append((name, var.clone().detach()))
            else:
                debug_history.append((name, var))

        use_k1 = k1 if k1 is not None else 5 
        use_k2 = k2 if k2 is not None else 12
        
        threshold = 12.0 
        L_fit = max(float(L), threshold + 1.0)
        period = 2 * L_fit
        
        provider = crypten.mpc.get_default_provider()
        device = self.device

        a0, poly_body, beta_sin_coeffs = _get_dynamic_params_odd("sigmoid", use_k1, use_k2, L=L_fit)
        full_poly_coeffs = [0.0] + poly_body 
        
        double_size = (2,) + self.size()
        
        a_all, b_all, r_all, c_all = provider.generate_cmp_aux(double_size, device=device)
        
        a1, a2 = a_all[0], a_all[1]
        b1, b2 = b_all[0], b_all[1]
        r1, r2 = r_all[0], r_all[1]
        c1, c2 = c_all[0], c_all[1]

        diff_pos = self - threshold
        masked_pos = diff_pos.mul(a1).add(b1)

        diff_neg = self + threshold
        masked_neg = diff_neg.mul(a2).add(b2) 

        t, u, v, t2, t3 = provider.generate_hybrid_triple(self.size(), period, len(beta_sin_coeffs), device=device)
        delta_share = self + t + period

        stacked_shares = crypten.stack([masked_pos, masked_neg, delta_share], dim=0)
        
        with crypten.no_grad():
            all_plain = stacked_shares.get_plain_text()
            
        masked_pos_plain = all_plain[0]
        masked_neg_plain = all_plain[1]
        delta_plain_raw = all_plain[2]

        V_pos = (masked_pos_plain > 0).float()
        is_pos_large = r1.mul(V_pos).add(c1)
        
        V_neg = (masked_neg_plain > 0).float()
        is_larger_than_neg12 = r2.mul(V_neg).add(c2)
        
        is_neg_large = 1.0 - is_larger_than_neg12
        
        is_mid = 1.0 - is_pos_large - is_neg_large
        delta_exact = delta_plain_raw
        delta_mod = torch.fmod(delta_plain_raw, period)
        delta_mod[delta_mod < 0] += period
        
        beta_sin = torch.tensor(beta_sin_coeffs, device=device, dtype=torch.float)
        
        fourier_val = 0
        k_list = [i * 2 * math.pi / period for i in range(1, len(beta_sin_coeffs) + 1)]
        delta_k = torch.stack([i * delta_mod for i in k_list])
        p = torch.sin(delta_k).to(device)
        q = torch.cos(delta_k).to(device)
        beta_tensor = beta_sin.view([-1] + [1] * self.dim())
        fourier_val = ((v * p - u * q) * beta_tensor).sum(dim=0)
        
        delta_raw = delta_exact - period
        poly_val = a0
        
        # 线性项
        if len(full_poly_coeffs) > 1:
            poly_val += self * full_poly_coeffs[1]
            
        delta_sq = delta_raw.square()
        with crypten.no_grad():
            x_plain = self.get_plain_text()
        # x^2 = (d-t)^2
        if len(full_poly_coeffs) > 2 and full_poly_coeffs[2] != 0:
            c2 = full_poly_coeffs[2]
            poly_val += x_plain*x_plain * c2
            
        # x^3 = (d-t)^3
        if len(full_poly_coeffs) > 3 and full_poly_coeffs[3] != 0:
            c3 = full_poly_coeffs[3]
            poly_val += x_plain*x_plain *x_plain * c3
            
        sig_approx = poly_val + fourier_val

        effective_share = sig_approx.mul(is_mid)+ is_pos_large

        silu_res = self.mul(effective_share)
        return silu_res
    else:
        raise ValueError(f"Unrecognized method {method} for silu")

def softmax(self, dim, k1=None, k2_exp=None, k2_recip=None, **kwargs):
    r"""Compute the softmax of a tensor's elements along a given dimension"""
    if kwargs.get('k2') is not None:
        if k2_exp is None: k2_exp = kwargs['k2']
        if k2_recip is None: k2_recip = kwargs['k2']
        
    if k1 is not None or k2_exp is not None:
        method = "newer"
    else:
        method = cfg.functions.softmax_method
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.ones_like((self.data)))

    if self.size(dim) == 1:
        return self.new(torch.ones_like(self.data))

    if method == "ideal":
        return crypten.cryptensor(torch.softmax(self.get_plain_text(), dim=dim), device=self.device)
    if method == "reciprocal":
        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    elif method == "ode":
        iter_num = cfg.functions.softmax_ode_iter_num
        clip = cfg.functions.softmax_ode_clip
        upper, lower = cfg.functions.softmax_ode_ub, cfg.functions.softmax_ode_lb

        if clip:
            # clip the input within the range [lower, upper] for numerical stability
            diff = crypten.cat([self - upper, lower - self]).relu().split(self.shape[0])#.split([1,1])
            self += diff[1] - diff[0]

        # initialize ode approximation
        x = self / iter_num
        g = self.new(torch.ones_like(self.data) / self.size(dim), device=self.device)

        # compute ode update formula
        for _ in range(iter_num):
            g += (x - g.mul(x).sum(dim=dim).unsqueeze(-1)).squeeze(-1) * g
        return g
    elif method == "newer":
        alpha_star = -4.0 
        beta_star = 10.0
        
        x_input = self

        cmp_inputs = crypten.stack([alpha_star - x_input, x_input - beta_star], dim=0)
        
        indicators = cmp_inputs.od_sign()
        
        t1 = indicators[0]
        t2 = indicators[1] 
        
        diff_lower = alpha_star - x_input
        diff_upper = x_input - beta_star
        
        mult_inputs_a = crypten.stack([t1, t2], dim=0)
        mult_inputs_b = crypten.stack([diff_lower, diff_upper], dim=0)
        
        mult_results = mult_inputs_a.mul(mult_inputs_b)
        
        term_lower = mult_results[0]
        term_upper = mult_results[1]
        
        x_clamped = x_input + term_lower - term_upper-12
        
        t3 = x_clamped.exp(k1=k1, k2=k2_exp,fit_min=alpha_star-beta_star-2,fit_max=-2)
        
        t4 = t3.sum(dim, keepdim=True)
        
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            t5 = t4.reciprocal(k1=k1, k2=k2_recip)
            
        # --- Step 10: Final Multiplication (SMP) ---
        y = t3.mul(t5)
        
        return y
    else:
        raise ValueError(f"Unrecognized method {method} for softmax")

def log_softmax(self, dim, **kwargs):
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.zeros((), device=self.device))

    if self.size(dim) == 1:
        return self.new(torch.zeros_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - normalize_term.log()
    return result

def od_sign(self):
    """
    Implements Protocol 4: CMP+ (Optimized Comparison) -> Sign Bit (0 or 1)
    """
    provider = crypten.mpc.get_default_provider()
    # 1. 获取辅助数据 (ArithmeticSharedTensor)
    a_share, b_share, r_share, c_share = provider.generate_cmp_aux(self.size(), device=self.device)
    
    # 2. 计算 Masked = x * a + b
    # 直接利用 MPCTensor 对底层 Share 的运算支持
    masked = self.mul(a_share).add(b_share)
    
    # 3. Reveal 掩码 (使用 get_plain_text 自动处理 scale)
    with crypten.no_grad():
        masked_plain = masked.get_plain_text()
    
    # 4. 明文比较 (得到 0.0 或 1.0)
    V = (masked_plain > 0).float()
    
    # 5. 计算 Indicator = r * V + c
    # r_share 是 AST，V 是 Tensor，结果自动为 AST
    indicator = r_share.mul(V).add(c_share)
    
    return indicator


def odrelu(self):
    """
    Implements Protocol 4: CMP+ (Optimized Comparison) -> ReLU
    """
    provider = crypten.mpc.get_default_provider()
    a_share, b_share, r_share, c_share = provider.generate_cmp_aux(self.size(), device=self.device)
    
    masked = self.mul(a_share).add(b_share)
    
    with crypten.no_grad():
        masked_plain = masked.get_plain_text()
    
    V = (masked_plain > 0).float()
    
    indicator = r_share.mul(V).add(c_share)
    
    return self.mul(indicator)


def odrelu(self):
    """
    Debug Version: Protocol 4 CMP+ -> ReLU
    复用上面的逻辑，但在最后多一步 x * indicator
    """
    indicator = self.od_sign()
    
    res = self.mul(indicator)
        
    return res
def _get_dynamic_params(func_type, K1, K2, L, min_val=None, max_val=None): 
    # 1. 基础配置 (保持不变)
    num_samples = 20000 
    L_val = float(L)
    ODD_FUNCS = ["tanh,sigmoid"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围 (保持不变)
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # 采样逻辑 (针对 exp 的负半轴可以维持线性，因为权重会处理重要性)
    x_fit = np.linspace(start, end, num_samples, dtype=np.float64)

    # 3. 目标函数 Ground Truth
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    y_fit = target_map[f_key](x_fit).astype(np.float64)

    # 4. 构建设计矩阵 X (保持不变)
    X_list = [np.ones_like(x_fit, dtype=np.float64)] 
    for k in range(1, K1 + 1):
        X_list.append(x_fit.astype(np.float64) ** k)
    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float64)
        X_list.append(np.sin(arg)) 
    X = np.vstack(X_list).T

    if func_type == "exp":
        
        safe_y = y_fit + 1e-20 
        
        weights = 1.0 / (safe_y ** 2)
        
        W = np.sqrt(weights)[:, np.newaxis] 
        
        X_weighted = X * W
        y_weighted = y_fit * W.flatten()
    
        coeffs, residuals, rank, s = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    else:
        # 5. 求解普通最小二乘
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y_fit, rcond=None)

    # 6. 结果提取 (保持不变)
    a0 = coeffs[0].item()
    poly_coeffs_raw = coeffs[1 : 1 + K1]
    beta_sin = coeffs[1 + K1 :].tolist()
    
    poly_body = []
    for i, c in enumerate(poly_coeffs_raw):
        power = i + 1
        if is_odd_mode and (power % 2 == 0):
            poly_body.append(0.0)
        else:
            poly_body.append(c.item())
            
    return a0, poly_body, beta_sin

def _get_dynamic_params_odd(func_type, K1, K2, L, min_val=None, max_val=None): 
    # 1. 基础配置
    num_samples = 20000 
    L_val = float(L)
    ODD_FUNCS = ["tanh", "sigmoid"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围 (保持不变)
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # 采样逻辑
    x_fit = np.linspace(start, end, num_samples, dtype=np.float64)

    # 3. 目标函数 Ground Truth
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    y_fit = target_map[f_key](x_fit).astype(np.float64)

    X_list = [np.ones_like(x_fit, dtype=np.float64)]
    
    if is_odd_mode:
        active_powers = [p for p in range(1, K1 + 1) if p % 2 != 0]
    else:
        active_powers = list(range(1, K1 + 1))

    for k in active_powers:
        X_list.append(x_fit.astype(np.float64) ** k)

    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float64)
        X_list.append(np.sin(arg)) 
        
    X = np.vstack(X_list).T

    if func_type == "exp":
        weights = 1.0 / (y_fit + 1e-12)
        W = np.sqrt(weights)[:, np.newaxis]
        X_weighted = X * W
        y_weighted = y_fit * W.flatten()
        coeffs, residuals, rank, s = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    else:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y_fit, rcond=None)

    a0 = coeffs[0].item()
    
    num_poly_fitted = len(active_powers)
    
    poly_coeffs_computed = coeffs[1 : 1 + num_poly_fitted]
    
    beta_sin = coeffs[1 + num_poly_fitted :].tolist()

    poly_body = []
    
    coeff_iter = iter(poly_coeffs_computed)
    
    for k in range(1, K1 + 1):
        if k in active_powers:
            poly_body.append(next(coeff_iter).item())
        else:
            poly_body.append(0.0)
            
    return a0, poly_body, beta_sin

def _get_dynamic_params_1(func_type, K1, K2, L, min_val=None, max_val=None): 
    import numpy as np
    # 1. 基础配置
    num_samples = 20000 
    L_val = float(L)
    
    # 修正：ODD_FUNCS 应该是列表包含多个字符串，而不是一个逗号分隔的字符串
    ODD_FUNCS = ["tanh", "sigmoid"] 
    is_odd_mode = (func_type in ODD_FUNCS)

    # 2. 确定拟合范围
    start_v = float(min_val) if min_val is not None else None
    end_v = float(max_val) if max_val is not None else None

    if func_type == "exp":
        start = start_v if start_v is not None else -L_val
        end = end_v if end_v is not None else 0.0
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        start = start_v if start_v is not None else 0.1
        end = L_val
    else:
        start = start_v if start_v is not None else -L_val
        end = L_val

    # --- 优点保留：混合采样策略 ---
    if func_type in ["inv_sqrt", "inv", "reciprocal"] and start > 1e-9:
        if L_val < 20.0:
            ratio_geo = 0.7
        else:
            ratio_geo = 0.8
        n_geo = int(num_samples * ratio_geo)
        n_lin = num_samples - n_geo
        # geomspace 可能会包含 start，linspace 也包含，合并需注意
        x_geo = np.geomspace(start, end, n_geo, dtype=np.float64)
        x_lin = np.linspace(start, end, n_lin, dtype=np.float64)
        x_fit = np.concatenate([x_geo, x_lin])
        x_fit = np.sort(x_fit)
    else:
        x_fit = np.linspace(start, end, num_samples, dtype=np.float64)

    # 3. 目标函数 Ground Truth
    target_map = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "tanh": lambda x: np.tanh(x),
        "inv_sqrt": lambda x: 1.0 / np.sqrt(x),
        "exp": lambda x: np.exp(x),
        "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
        "reciprocal": lambda x: 1.0 / x,
        "inv": lambda x: 1.0 / x
    }
    
    f_key = func_type
    if func_type == "inv": f_key = "reciprocal"
    y_fit = target_map[f_key](x_fit).astype(np.float64)

    # 4. 构建设计矩阵 X (Bias + Poly + Fourier)
    X_list = [np.ones_like(x_fit, dtype=np.float64)] 
    
    # 优化：如果是奇函数模式，建议直接在这里跳过偶数项，或者后续置零
    # 这里保持你原本的逻辑（先拟合再置零），但为了正则化效果，最好这里只生成需要的列
    # 不过为了代码改动最小，维持原状
    for k in range(1, K1 + 1):
        X_list.append(x_fit.astype(np.float64) ** k)
    for k in range(1, K2 + 1):
        arg = (np.pi * k * x_fit / L_val).astype(np.float64)
        X_list.append(np.sin(arg)) 
    X = np.vstack(X_list).T

    # =========================================================================
    # 【核心修正】加权逻辑覆盖 Reciprocal/Inv_Sqrt
    # =========================================================================
    
    # 确定正则化强度 alpha (稍微调小了一点大 alpha，防止欠拟合)
    if L_val < 20.0:
        alpha = 0.1 # 之前是 2.0，可能太大了导致拟合不上，改成 0.1 试试
    else:
        alpha = 1e-6
    
    weights = None

    # A. 针对 exp 的加权 (保持不变)
    if func_type == "exp":
        # exp 变化极大，权重设为 1/y 比较平滑
        weights = 1.0 / (np.abs(y_fit) + 1e-15)
        
    # B. 【新增】针对 inv, reciprocal, inv_sqrt 的加权
    # 目标：最小化相对误差 ((y_pred - y)/y)^2 -> 权重 w = 1/y^2
    elif func_type in ["inv", "reciprocal", "inv_sqrt"]:
        weights = 1.0 / (y_fit ** 2 + 1e-15)

    # 应用权重
    if weights is not None:
        W_sqrt = np.sqrt(weights)[:, np.newaxis]
        X_w = X * W_sqrt
        y_w = y_fit * np.sqrt(weights)
    else:
        X_w = X
        y_w = y_fit

    # 5. 求解 (Ridge Regression)
    n_features = X.shape[1]
    
    XT_w = X_w.T
    XTX_w = XT_w @ X_w
    A = XTX_w + alpha * np.eye(n_features)
    b = XT_w @ y_w
    
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None) # 注意这里也要用加权后的数据
    
    # 6. 结果提取
    a0 = coeffs[0].item()
    poly_coeffs_raw = coeffs[1 : 1 + K1]
    beta_sin = coeffs[1 + K1 :].tolist()
    
    poly_body = []
    for i, c in enumerate(poly_coeffs_raw):
        power = i + 1
        if is_odd_mode and (power % 2 == 0):
            poly_body.append(0.0)
        else:
            poly_body.append(c.item())
            
    return a0, poly_body, beta_sin