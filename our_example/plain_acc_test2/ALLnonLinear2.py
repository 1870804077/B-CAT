import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import math

# ==================== 基础函数定义 ====================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh(x): return np.tanh(x)
def SiLU(x): return x * sigmoid(x)
def inv_sqrt(x): return 1.0 / np.sqrt(np.abs(x) + 1e-10)
def inv(x): return 1.0 / (x + np.sign(x) * 1e-10)
def e_x(x): return np.exp(x)
def GeLU(x): return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ==================== 私有化函数定义 ====================
def build_design_matrix_g(x, K1, K2, L):
    """构建设计矩阵"""
    X = [np.ones_like(x)]
    for k in range(1, K1+1):
        X.append(x**(k))
    for k in range(1, K2+1):
        X.append(np.sin(np.pi * k * x / L))
        # X.append(np.cos(np.pi * k * x / L))
    return np.vstack(X).T

def fit_g(x, y, K1, K2, L):
    """拟合系数"""
    X = build_design_matrix_g(x, K1, K2, L)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs

def eval_g(x, coeffs, K1, K2, L):
    """评估函数值"""
    X = build_design_matrix_g(x, K1, K2, L)
    return X @ coeffs

def newton_refine(y, x):
    """牛顿迭代细化"""
    simhalfnumber = 0.500438180 * x
    y = y * (1.50131454 - simhalfnumber * y * y)
    y = y * (1.50000086 - 0.999124984 * simhalfnumber * y * y)
    # y = y * (1.50000086 - 0.999124984 * simhalfnumber * y * y)
    # y = y * (1.50000086 - 0.999124984 * simhalfnumber * y * y)
    return y

def private_sigmoid(x_train, y_train, x_test, K1, K2, L):
    """私有化sigmoid"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    print(coeffs)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    return y_test

def private_tanh(x_train, y_train, x_test, K1, K2, L):
    """私有化tanh"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    print(coeffs)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    return y_test

def private_SiLU(x_train, y_train, x_test, K1, K2, L):
    # """私有化SiLU"""
    # y_test = private_sigmoid(x_train, y_train, x_test, K1, K2, L)
    # y_test = y_test * x_test
    # return y_test 
    """
    分段GELU近似：
    - x < -4: 0
    - x > 4: x
    - -4 ≤ x ≤ 4: GeLU近似公式
    """
    # 创建结果数组
    y_test = np.zeros_like(x_test)
    
    # 分段处理
    mask_low = x_test < -12   # x < -4 的部分
    mask_high = x_test > 12   # x > 4 的部分
    mask_mid = (~mask_low) & (~mask_high)  # -4 ≤ x ≤ 4 的部分
    
    # 1. x < -4: y = 0
    y_test[mask_low] = 0
    
    # 2. x > 4: y = x
    y_test[mask_high] = x_test[mask_high]
    
    # 3. -4 ≤ x ≤ 4: GeLU近似
    if np.any(mask_mid):
        x_mid = x_test[mask_mid]
        
        # 使用私有sigmoid近似（你提供的函数）
        sig_approx = private_sigmoid(x_train, y_train, x_mid, K1, K2, L)
        
        # GeLU公式：x * sigmoid(x)
        y_test[mask_mid] = x_mid * sig_approx
    
    return y_test

def private_inv_sqrt(x_train, y_train, x_test, K1, K2, L):
    """私有化平方根倒数"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    print(coeffs)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    y_test = newton_refine(y_test, x_test)
    return y_test

def private_inv(x_train, y_train, x_test, K1, K2, L):
    """私有化倒数"""
    y_test = private_inv_sqrt(x_train, y_train, x_test, K1, K2, L)
    y_test = y_test * y_test
    return y_test

def private_erf(x_train, y_train, x_test, K1, K2, L):
    """私有化误差函数"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    print(coeffs)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    return y_test

def newton_refine2(y, x, t):
    """牛顿迭代细化"""
    for i in range(t):
        y = y * (2 - x * y)
    return y

def private_inv2(x_train, y_train, x_test, K1, K2, L):
    """私有化倒数"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    print(coeffs)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    y_test = newton_refine2(y_test, x_test, 3)
    return y_test

# 对比GPT方案
# def private_e_x(x_fixed, s):
    """
    简化的整数指数函数
    
    基于：exp(x) ≈ 2^(x/ln2)
    分解为：2^(z + p) = 2^z * 2^p，其中z是整数，p是小数
    用二次多项式近似 2^p
    """
    # 常数定义
    INV_LN2 = 1.0 / math.log(2)  # 1/ln2 ≈ 1.442695
    LN2 = math.log(2)            # ln2 ≈ 0.693147
    
    # 1. 将定点数转换回浮点数进行计算
    x_float = (x_fixed / (1 << s))
    
    # 2. 计算 x / ln2
    x_div_ln2 = x_float * INV_LN2
    
    # 3. 分离整数和小数部分
    z_int = int(math.floor(x_div_ln2))  # 整数部分
    p = x_div_ln2 - z_int                # 小数部分，范围[0, 1)
    
    # 4. 用二次多项式近似 2^p (p在[0,1)范围内)
    # 使用近似: 2^p ≈ 1 + p*ln2 + p^2*(ln2)^2/2
    # 或者更简单的: 2^p ≈ 0.355*p^2 + 0.69*p + 0.983
    p_poly = 0.355 * p * p + 0.69 * p + 0.983
    
    # 5. 计算 2^z * 2^p
    result_float = p_poly / (1 << (-z_int))
    
    # 6. 转换为定点数
    result_fixed = int(result_float * (1 << s))
    return result_fixed

# 2段多项式逼近与三段多项式逼近反复
def private_e2_x(xei, s):
    """
    Algorithm 1: Secure Integer-only Exponential Function (High Accuracy Fix)
    """
    scale = 1 << s 
    
    # 1. 常数准备
    invNegLt_const = int((-1.0 / math.log(2)) * scale)
    ln2_const = int(math.log(2) * scale)
    
    # 【修正】系数从 0.385 改为 0.358
    # 论文(I-BERT)中的精确系数通常是 a=0.35815147
    c_0_358 = int(0.35815147 * scale)  # <--- 关键修正
    c_1_353 = int(1.353 * scale)
    c_0_344 = int(0.34426511 * scale)  # 稍微增加一点精度

    # 2. 算法实现
    # Range Reduction: x = -z*ln2 + p
    z_fixed = (xei * invNegLt_const) >> s
    z = z_fixed >> s  # 整数部分 k
    
    # Clip z (根据论文，指数范围限制)
    if z < 0: z = 0
    elif z > (s + 1): z = s + 1 # 这里的 clip 是为了位移安全

    # 计算残差 p (residual)
    # p = x + z*ln2
    p = xei + (z * ln2_const)

    # Polynomial: 0.358 * (p + 1.353)^2 + 0.344
    t1 = p + c_1_353
    t2 = (t1 * t1) >> s
    t3 = (c_0_358 * t2) >> s
    poly_res = t3 + c_0_344
    
    # Reconstruction: result = poly >> z
    result = poly_res >> z
    
    return result

def private_e3_x(xei, s):
    """
    Algorithm 1 Ultimate: 
    - 3rd Degree Polynomial (Cubic) for ~20x better accuracy.
    - Horner's Method for efficient computation.
    - Rounding for all shifts.
    """
    scale = 1 << s 
    round_bias = 1 << (s - 1)
    
    # --- 1. 常数准备 (3rd Degree Coeffs) ---
    # Coeffs fitted on [-ln2, 0] for exp(x)
    c3_const = int(0.11864127 * scale)
    c2_const = int(0.47995675 * scale)
    c1_const = int(0.99701588 * scale)
    # c0 理论上非常接近 1.0，直接使用 1.0 可以保证 exp(0)=1 的精确性
    c0_const = int(1.00000000 * scale) 
    
    invNegLt_const = int((-1.0 / math.log(2)) * scale)
    ln2_const = int(math.log(2) * scale)

    # --- 2. 算法实现 ---
    
    # [Range Reduction] x = -z*ln2 + p
    # z = floor(x / -ln2)
    # 使用 2*s 精度进行中间计算以减少误差
    z = (xei * invNegLt_const + round_bias) >> (2 * s)
    
    # Clip z
    if z < 0: z = 0
    elif z > (s + 1): z = s + 1

    # p = x + z*ln2
    p = xei + (z * ln2_const)

    # [Polynomial: Cubic Horner]
    # Poly = ((c3 * p + c2) * p + c1) * p + c0
    
    # Step 1: term1 = c3 * p + c2
    term1_mul = (c3_const * p + round_bias) >> s
    term1 = term1_mul + c2_const
    
    # Step 2: term2 = term1 * p + c1
    term2_mul = (term1 * p + round_bias) >> s
    term2 = term2_mul + c1_const
    
    # Step 3: term3 = term2 * p + c0
    term3_mul = (term2 * p + round_bias) >> s
    poly_res = term3_mul + c0_const
    
    # [Reconstruction] result = poly_res >> z
    if z > 0:
        result = (poly_res + (1 << (z - 1))) >> z
    else:
        result = poly_res
    
    return result

def private_GeLU(x_train, y_train, x_test, K1, K2, L):
    """私有化GeLU"""
    # x_arg = np.sqrt(2/np.pi) * (x_test + 0.044715 * x_test**3)
    # y_test = private_tanh(x_train, y_train, x_arg, K1, K2, L)
    # y_test = 0.5 * x_test * (1.0 + y_test)
    """
    分段GELU近似：
    - x < -4: 0
    - x > 4: x
    - -4 ≤ x ≤ 4: GeLU近似公式
    """
    # 创建结果数组
    y_test = np.zeros_like(x_test)
    
    # 分段处理
    mask_low = x_test < -4   # x < -4 的部分
    mask_high = x_test > 4   # x > 4 的部分
    mask_mid = (~mask_low) & (~mask_high)  # -4 ≤ x ≤ 4 的部分
    
    # 1. x < -4: y = 0
    y_test[mask_low] = 0
    
    # 2. x > 4: y = x
    y_test[mask_high] = x_test[mask_high]
    
    # 3. -4 ≤ x ≤ 4: GeLU近似
    if np.any(mask_mid):
        x_mid = x_test[mask_mid]
        
        # 计算x_arg（GeLU的近似参数）
        x_arg = np.sqrt(2/np.pi) * (x_mid + 0.044715 * x_mid**3)
        
        # 使用私有tanh近似（你提供的函数）
        tanh_approx = private_tanh(x_train, y_train, x_arg, K1, K2, L)
        
        # GeLU公式：0.5 * x * (1 + tanh(√(2/π)(x + 0.044715x³)))
        y_test[mask_mid] = 0.5 * x_mid * (1.0 + tanh_approx)
    
    return y_test

# 对比方案失败版本
def private_GeLU2(x_train, y_train, x_test, K1, K2, L):
    """私有化GeLU"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    y_test = eval_g(np.abs(x_test), coeffs, K1, K2, L)
    k = np.array([0] * len(x_test))
    for i in range(0, len(x_test)):
        if (np.abs(x_test[i]) < L):
            k[i] = 1
    y_test = np.maximum(x_test, 0.0) + k * y_test
    return y_test


def private_Softmax(x_train, y_train, x_test, K1, K2, L, s):
    """私有化Softmax"""
    # 转换为定点数
    x_fixed = np.array([int(x * (1 << s)) for x in x_test])
    
    # 计算指数
    ex = np.array([private_e_x(i, s) for i in x_fixed])
    
    # 计算指数和
    ex_sum = np.sum(ex)
    
    # 计算倒数（需要先准备训练数据）
    # 这里简化处理，实际应该用专门训练好的倒数函数
    ex_sum_fixed = int(ex_sum)
    ex_sum_float = ex_sum_fixed / (1 << s)
    
    # 使用私有化倒数
    ex_sum_inv = private_inv(x_train, y_train, np.array([ex_sum_float]), K1, K2, L)[0]
    
    # 计算softmax
    y_test = ex * ex_sum_inv / len(ex)
    return y_test


def sigmoid_compare():
    """sigmoid函数不同参数对比"""
    L = 8.0
    num_samples = 10000
    K1_list = [0, 1, 3]
    K2_list = [8, 12]
    # K1_list = [1]
    # K2_list = [4,5,6,7,8,9]
    x_fit = np.linspace(-L, L, num_samples)
    y = sigmoid(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse, 'max_error': max_error, 'mean_error': mean_error}
            print(f"[sigmoid] K1={K1}, K2={K2}, RMSE={rmse:.6e}, ME={max_error:.6e}, AE={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            max_error = results[(K1, K2)]['max_error']
            mean_error = results[(K1, K2)]['mean_error']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}, ME={max_error:.1e}, AE={mean_error:.1e}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=1, label='True sigmoid')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-0.1, 1.1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("Sigmoid approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("sigmoid(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("sigmoid_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def tanh_compare():
    """tanh函数不同参数对比"""
    L = 8.0
    num_samples = 3000
    K1_list = [0, 1, 3]
    K2_list = [6, 12]
    # K1_list = [1]
    # K2_list = [7,8,9, 10, 12]
    x_fit = np.linspace(-L, L, num_samples)
    y = tanh(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_tanh(x_fit, y, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse, 'max_error': max_error, 'mean_error': mean_error}
            print(f"[tanh] K1={K1}, K2={K2}, RMSE={rmse:.6e}, ME={max_error:.6e}, AE={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            max_error = results[(K1, K2)]['max_error']
            mean_error = results[(K1, K2)]['mean_error']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}, ME={max_error:.1e}, AE={mean_error:.1e}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=1, label='True tanh')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-1.5, 1.5)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("Tanh approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("tanh(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("tanh_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def SiLU_compare():
    """SiLU函数不同参数对比"""
    L = 12.0
    num_samples = 5000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [1,2,3]
    K2_list = [8,12]
    x_fit = np.linspace(-L, L, num_samples)
    # y = SiLU(x_fit)
    y_train = sigmoid(x_fit)
    x_test = np.linspace(-13, 13, num_samples)
    y = SiLU(x_test)
    
    print(y[-1])

    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_SiLU(x_fit, y_train, x_test, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[SiLU] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_test, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_test, y, 'k-', linewidth=3, label='True SiLU')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-0.5, 8)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("SiLU approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("SiLU(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("SiLU_comparison.pdf", dpi=300)
    # plt.show()
    plt.close()

def SiLU_compare2():
    """SiLU函数使用固定参数对比"""
    L = 8
    num_samples = 3000
    K1 = 0  # 因为你的参数只有正弦项
    K2 = 8  # 因为有8个正弦项
    
    belta = np.array([-0.1299, -0.1220, -0.0743, -0.0394, -0.0216, -0.0118, -0.0074, -0.0044, -0.0033, -0.0021, -0.0018, -0.0011])

    x_test = np.linspace(-13, 13, num_samples)
    y_true = SiLU(x_test)

    # 直接计算加权正弦项的和
    sin_sum = np.zeros_like(x_test)
    for k in range(1, 13):
        sin_sum += belta[k-1] * np.sin(np.pi * k * np.abs(x_test) / L)
    
    # 直接计算近似值
    y_approx = sin_sum
    
    # 应用阈值函数
    k = np.zeros_like(x_test)
    for i in range(len(x_test)):
        if np.abs(x_test[i]) < L:
            k[i] = 1
    
    y_approx = np.maximum(x_test, 0.0) + k * y_approx
    
    # 计算误差
    max_error = np.max(np.abs(y_true - y_approx))
    mean_error = np.mean(np.abs(y_true - y_approx))
    rmse = np.sqrt(np.mean((y_true - y_approx)**2))
    
    print(f"[SiLU with fixed params] K1={K1}, K2={K2}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Mean Error: {mean_error:.6e}")
    
    # 绘制对比图
    plt.figure(figsize=(14, 8))
    
    # 子图1：函数对比
    plt.subplot(2, 2, 1)
    plt.plot(x_test, y_true, 'k-', linewidth=3, label='True SiLU')
    plt.plot(x_test, y_approx, 'r--', linewidth=2, label='Approximation')
    plt.title(f"SiLU Approximation (K1={K1}, K2={K2})", fontdict={"family": "Times New Roman", "size": 16})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 12})
    plt.ylabel("SiLU(x)", fontdict={"family": "Times New Roman", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.close()


def inv_compare():
    """inv函数不同参数对比"""
    L = 8.0
    num_samples = 3000
    # K1_list = [0, 1]
    # K2_list = [1, 6, 12]
    K1_list = [1,3]
    K2_list = [6, 9, 12]
    x_fit = np.linspace(0.1, L, num_samples)  # 避免除以0
    y = inv(x_fit)
    y_train = inv_sqrt(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_inv(x_fit, y_train, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[inv] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True 1/x')
    
    # 设置坐标轴
    plt.xlim(0, 6)
    plt.ylim(0, 10)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("1/x approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("1/x", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("inv_comparison.pdf", dpi=300)
    plt.show()
    plt.close()
    
def inv_compare2():
    """inv函数不同参数对比"""
    L = 8
    num_samples = 3000
    # K1_list = [0, 1]
    # K2_list = [1, 6, 12]
    K1_list = [0,1,2,3]
    K2_list = [12]
    x_fit = np.linspace(0.1, 12, num_samples)  # 避免除以0
    y_fit = inv(x_fit)
    x = np.linspace(0.1, 8, num_samples)
    y = inv(x)
    
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_inv2(x_fit, y_fit, x, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[inv] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x, y, 'k-', linewidth=3, label='True 1/x')
    
    # 设置坐标轴
    plt.xlim(8, L)
    plt.ylim(0, 0.3)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("1/x approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("1/x", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("inv_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def inv_sqrt_compare():
    """inv_sqrt函数不同参数对比"""
    L = 8
    num_samples = 3000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [1, 2, 3]
    K2_list = [8, 12]
    x_fit = np.linspace(0.1, L, num_samples)  # 避免负数开方
    y = inv_sqrt(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_inv_sqrt(x_fit, y, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[inv_sqrt] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True 1/√x')
    
    # 设置坐标轴
    plt.xlim(0, L)
    plt.ylim(0, 3.5)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("1/√x approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("1/√x", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("inv_sqrt_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def inv_crypten(x, te, t):
    """简化的整数指数函数对比"""
    # 生成定点数
    y0 = 3 * ex(0.5-x, te) + 0.003
    
    for i in range(t):
        y0 = y0 * (2.0 - x * y0)
        
    return y0

def inv_sqrt_crypten(x, te, t):
    """简化的整数指数函数对比"""
    # 生成定点数
    y0 = 2.2*ex(-(x/2+0.2), te) + 0.2
    y0 = y0 - y0 / 1024
    
    for i in range(t):
        y0 = 0.5 * y0 * (3 - x * (y0**2))
        
    return y0

def inv_sqrt_crypten_compare():
    """倒数函数"""
    """生成对比可视化图表"""
    # 生成测试点
    x_continuous = np.linspace(0.1, 8, 5000)
    y_sqrt = np.array([inv_sqrt_crypten(x, 8, 2) for x in x_continuous])
    
    y_true = inv_sqrt(x_continuous)
    y_simple = y_sqrt
    
    
    max_error = np.max(np.abs(y_true - y_simple))
    mean_error = np.mean(np.abs(y_true - y_simple))
    rmse = np.sqrt(np.mean((y_true - y_simple)**2))
    print(f"[inv_sqrt], RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    plt.plot(x_continuous, y_simple, color='red', linestyle='-', linewidth=2, label=f"inv_sqrt Approximation")
    
    # 绘制真实曲线
    plt.plot(x_continuous, y_true, 'k-', linewidth=2, label='True inv_sqrt')
    
    # 设置坐标轴
    plt.xlim(1, 8)
    # plt.ylim(0, 0.1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("inv_sqrt approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("inv_sqrt(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("inv_sqrt_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def inv_crypten_compare():
    """倒数函数"""
    """生成对比可视化图表"""
    # 生成测试点
    x_continuous = np.linspace(0.1, 3000, 5000)
    y_true = 1 / x_continuous
    y_simple = np.array([inv_crypten(x, 8, 3) for x in x_continuous])
    
    max_error = np.max(np.abs(y_true - y_simple))
    mean_error = np.mean(np.abs(y_true - y_simple))
    rmse = np.sqrt(np.mean((y_true - y_simple)**2))
    print(f"[inv], RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    plt.plot(x_continuous, y_simple, color='red', linestyle='-', linewidth=2, label=f"inv Approximation")
    
    # 绘制真实曲线
    plt.plot(x_continuous, y_true, 'k-', linewidth=2, label='True inv')
    
    # 设置坐标轴
    plt.xlim(1, 100)
    # plt.ylim(0, 0.1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("inv approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("inv(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("inv_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def GeLU_compare():
    """GeLU函数不同参数对比"""
    L = 6
    num_samples = 3000
    # K1_list = [0, 1, 2]
    # K2_list = [6, 12]
    K1_list = [1,2,3]
    K2_list = [8, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-5, 5, 3000)
    y = GeLU(x_test)
    y_train = tanh(x_fit)
    
    results = {}
    print(x_test[0])
    print(y[0])
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_GeLU(x_fit, y_train, x_test, K1, K2, L)
            print(y_test[0])
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[GeLU] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True GeLU')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-3, 8)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("GeLU approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("GeLU(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("GeLU_comparison.pdf", dpi=300)
    # plt.show()
    plt.close()

def GeLU_compare2():
    """GeLU函数不同参数对比"""
    L = 4
    num_samples = 3000
    K1_list = [0]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-4, 4, 3000)
    y = GeLU(x_test)
    y_train = GeLU(np.abs(x_fit)) - np.maximum(np.abs(x_fit), 0.0)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_GeLU2(x_fit, y_train, x_test, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[GeLU] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True GeLU')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-3, 8)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("GeLU approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("GeLU(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("GeLU_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

# 对比方案
def GeLU_compare3():
    """GeLU函数使用固定参数对比"""
    L = 4
    num_samples = 3000
    K1 = 0  # 因为你的参数只有正弦项
    K2 = 8  # 因为有8个正弦项
    
    belta = np.array([-0.0818,-0.0809,-0.0424,-0.0176,-0.0079,-0.0043,-0.0026,-0.0017])

    x_test = np.linspace(-5, 5, num_samples)
    y_true = GeLU(x_test)

    # 直接计算加权正弦项的和
    sin_sum = np.zeros_like(x_test)
    for k in range(1, 9):
        sin_sum += belta[k-1] * np.sin(np.pi * k * np.abs(x_test) / L)
    
    # 直接计算近似值
    y_approx = sin_sum
    
    # 应用阈值函数
    k = np.zeros_like(x_test)
    for i in range(len(x_test)):
        if np.abs(x_test[i]) < L:
            k[i] = 1
    
    y_approx = np.maximum(x_test, 0.0) + k * y_approx
    
    # 计算误差
    max_error = np.max(np.abs(y_true - y_approx))
    mean_error = np.mean(np.abs(y_true - y_approx))
    rmse = np.sqrt(np.mean((y_true - y_approx)**2))
    
    print(f"[GeLU with fixed params] K1={K1}, K2={K2}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Mean Error: {mean_error:.6e}")
    
    # 绘制对比图
    plt.figure(figsize=(14, 8))
    
    # 子图1：函数对比
    plt.subplot(2, 2, 1)
    plt.plot(x_test, y_true, 'k-', linewidth=3, label='True GeLU')
    plt.plot(x_test, y_approx, 'r--', linewidth=2, label='Approximation')
    plt.title(f"GeLU Approximation (K1={K1}, K2={K2})", fontdict={"family": "Times New Roman", "size": 16})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 12})
    plt.ylabel("GeLU(x)", fontdict={"family": "Times New Roman", "size": 12})
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.close()



# 这个方法是先计算sigmoid 然后计算sigmoid的倒数
def ex_compare():
    L = 16
    num_samples = 5000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [1]
    K2_list = [12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-16, 0, num_samples)
    y = e_x(x_test) 

    y_train = sigmoid(x_fit)
    x_fit2 = np.linspace(0.1, 4, num_samples)
    y_train2 = inv_sqrt(x_fit2)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y_train, x_test, K1, K2, L)

            y_test = np.abs(1 - y_test)
            
            y_test = private_inv(x_fit2, y_train2, y_test, K1, K2, L) - 1

            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[ex] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_test, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_test, y, 'k-', linewidth=3, label='True ex')
    
    # 设置坐标轴
    plt.xlim(-4, 0)
    plt.ylim(0, 1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("ex approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("ex(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("ex_comparison.pdf", dpi=300)
    plt.show()
    plt.close()


def ex_compare2():
    """GeLU函数不同参数对比"""
    """生成对比可视化图表"""
    # 生成测试点
    x_continuous = np.linspace(-8, 0, 3000)
    s = 20
    y_true = np.exp(x_continuous)
    y_simple = np.array([private_e2_x(int(x * (1 << s)), s) / (1 << s) for x in x_continuous])
    # y_simple = np.array([private_e3_x(int(x * (1 << s)), s) / (1 << s) for x in x_continuous])
    
    max_error = np.max(np.abs(y_true - y_simple))
    mean_error = np.mean(np.abs(y_true - y_simple))
    rmse = np.sqrt(np.mean((y_true - y_simple)**2))
    print(f"[ex], RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    plt.plot(x_continuous, y_simple, color='red', linestyle='-', linewidth=2, label=f"ex Approximation")
    
    # 绘制真实曲线
    plt.plot(x_continuous, y_true, 'k-', linewidth=2, label='True ex')
    
    # 设置坐标轴
    plt.xlim(-4, 0)
    plt.ylim(0, 1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("ex approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("ex(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("ex_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

# 这个方法是直接用sin+x函数去近似ex
def ex_compare3():
    L = 15
    num_samples = 10000000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [3]
    K2_list = [12]
    x_fit = np.linspace(-15, -2, num_samples)
    x_test = np.linspace(-14, -10, num_samples)
    y = e_x(x_fit) 
    print("e-14", y[0])
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y, x_test, K1, K2, L)
            print("min", np.min(y_test))
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[ex] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # # 绘制对比图
    # colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    # linestyles = ['-', '--', '-.', ':']
    
    # plt.figure(figsize=(14, 6))
    
    # # 绘制近似曲线
    # for i, K1 in enumerate(K1_list):
    #     for j, K2 in enumerate(K2_list):
    #         y_hat = results[(K1, K2)]['y_hat']
    #         rmse = results[(K1, K2)]['rmse']
    #         color = colors[i % len(colors)]
    #         linestyle = linestyles[j % len(linestyles)]
    #         # plt.plot(x_test, y_hat, color=color, linestyle=linestyle,
    #                 #  linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # # 绘制真实曲线
    # plt.plot(x_fit, y, 'k-', linewidth=3, label='True ex')
    
    # # 设置坐标轴
    # plt.xlim(-16, 0)
    # plt.ylim(0, 1)
    
    # # 设置字体和坐标轴粗细
    # plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    # plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    # ax = plt.gca()
    
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    
    # # 设置标题和标签
    # plt.title("ex approximation", fontdict={"family": "Times New Roman", "size": 40})
    # plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    # plt.ylabel("ex(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    # plt.grid(True, alpha=0.3)
    # plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    # plt.tight_layout()
    # plt.savefig("ex_comparison.pdf", dpi=300)
    # # plt.show()
    # plt.close()

def ex(x, n_limit=50):
    """
    计算极限定义下的 e^x: ex = lim n→∞ (1 + x/(2n))^(2n)
    
    参数:
    x: 输入值或数组
    n_limit: 极限计算中的n值，越大越精确
    
    返回:
    e^x 的近似值
    """
    if isinstance(x, (int, float)):
        return (1 + x/(2**n_limit))**(2**n_limit)
    else:
        # 处理数组输入
        return (1 + x/(2**n_limit))**(2**n_limit)

def ex_compare4():
    """GeLU函数不同参数对比"""
    """生成对比可视化图表"""
    # 生成测试点
    x_continuous = np.linspace(-10, -2, 5000)
    y_true = np.exp(x_continuous)
    y_simple = np.array([ex(x, 8) for x in x_continuous])
    
    max_error = np.max(np.abs(y_true - y_simple))
    mean_error = np.mean(np.abs(y_true - y_simple))
    rmse = np.sqrt(np.mean((y_true - y_simple)**2))
    print(f"[ex], RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    plt.plot(x_continuous, y_simple, color='red', linestyle='-', linewidth=2, label=f"ex Approximation")
    
    # 绘制真实曲线
    plt.plot(x_continuous, y_true, 'k-', linewidth=2, label='True ex')
    
    # 设置坐标轴
    plt.xlim(-20, 4)
    plt.ylim(0, 0.1)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("ex approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("ex(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("ex_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def Softmax_compare():
    L = 8
    num_samples = 3000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [1]
    K2_list = [1,5,12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-L, 0, 3000)
    y = softmax(x_test)
    x_max = np.max(x_test)
    exp_x = np.exp(x_test - x_max)
    y = softmax(x_test)
    print(y)

    y_train = sigmoid(x_fit)
    x_fit2 = np.linspace(0.1, L, num_samples)
    y_train2 = inv_sqrt(x_fit2)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y_train, x_test, K1, K2, L)

            y_test = np.abs(1 - y_test)
            
            y_test = private_inv(x_fit2, y_train2, y_test, K1, K2, L)
            y_test = y_test - 1

            y_test_sum = np.sum(y_test) / 3000
            y_test_sum = private_inv(x_fit2, y_train2, y_test_sum, K1, K2, L)
            y_test = [i * y_test_sum / 3000 for i in y_test]

            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[Softmax] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True ex')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-3, 8)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("Softmax approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("Softmax(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("Softmax_comparison.pdf", dpi=300)
    plt.show()
    plt.close()

def Softmax_compare2():
    L = 8
    num_samples = 1000
    # K1_list = [0, 1, 2]
    # K2_list = [1, 6, 12]
    K1_list = [1]
    K2_list = [1,2,3,4,5,12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-L, 0, num_samples)
    y_train1 = e_x(x_test)
    y = softmax(x_test)

    x_fit2 = np.linspace(0.1, L, num_samples)
    y_train2 = inv_sqrt(x_fit2)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_test, y_train1, x_test, K1, K2, L)

            y_test_sum = np.sum(y_test) / num_samples
            y_test_sum = private_inv(x_fit2, y_train2, y_test_sum, K1, K2, L)
            y_test = [i * y_test_sum / num_samples for i in y_test]

            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[Softmax] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
    # 绘制对比图
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(14, 6))
    
    # 绘制近似曲线
    for i, K1 in enumerate(K1_list):
        for j, K2 in enumerate(K2_list):
            y_hat = results[(K1, K2)]['y_hat']
            rmse = results[(K1, K2)]['rmse']
            color = colors[i % len(colors)]
            linestyle = linestyles[j % len(linestyles)]
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True ex')
    
    # 设置坐标轴
    plt.xlim(-8, 8)
    plt.ylim(-3, 8)
    
    # 设置字体和坐标轴粗细
    plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
    plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
    ax = plt.gca()
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    # 设置标题和标签
    plt.title("Softmax approximation", fontdict={"family": "Times New Roman", "size": 40})
    plt.xlabel("x", fontdict={"family": "Times New Roman", "size": 40})
    plt.ylabel("Softmax(x)", fontdict={"family": "Times New Roman", "size": 40})
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12, prop={'size': 12, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig("Softmax_comparison.pdf", dpi=300)
    plt.show()
    plt.close()


# 运行所有比较
print("Starting function comparisons...")

# ### 我们的sigmoid
# sigmoid_compare()

# ### 我们的tanh
# tanh_compare()

# ### 我们的SiLU函数
# SiLU_compare()
# SiLU_compare2()

# ### 我们的inv函数
# inv_compare()
# inv_compare2()
# inv_crypten_compare()

# ### 我们的inv_sqrt函数
# inv_sqrt_compare()
# inv_sqrt_crypten_compare()

# ### 我们的GeLU函数
# GeLU_compare()
# GeLU_compare2()
# GeLU_compare3()

# ex_compare()
# ex_compare2()
# ### 我们的ex函数
ex_compare3()
# ex_compare4()


# Softmax_compare()
# Softmax_compare2()