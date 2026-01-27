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
        X.append(x**(2*k-1))
    for k in range(1, K2+1):
        X.append(np.sin(np.pi * k * x / L))
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
    return y

def private_sigmoid(x_train, y_train, x_test, K1, K2, L):
    """私有化sigmoid"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    return y_test

def private_tanh(x_train, y_train, x_test, K1, K2, L):
    """私有化tanh"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    return y_test

def private_SiLU(x_train, y_train, x_test, K1, K2, L):
    """私有化SiLU"""
    y_test = private_sigmoid(x_train, y_train, x_test, K1, K2, L)
    y_test = y_test * x_test
    return y_test 

def private_inv_sqrt(x_train, y_train, x_test, K1, K2, L):
    """私有化平方根倒数"""
    A = build_design_matrix_g(x_train, K1, K2, L)
    coeffs, _, _, _ = lstsq(A, y_train)
    y_test = eval_g(x_test, coeffs, K1, K2, L)
    y_test = newton_refine(y_test, x_test)
    return y_test

def private_inv(x_train, y_train, x_test, K1, K2, L):
    """私有化倒数"""
    y_test = private_inv_sqrt(x_train, y_train, x_test, K1, K2, L)
    y_test = y_test * y_test
    return y_test

# 对比方案
def private_e_x(x_fixed, s):
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
    x_float = x_fixed / (1 << s)
    
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

def private_GeLU(x_train, y_train, x_test, K1, K2, L):
    """私有化GeLU"""
    x_arg = np.sqrt(2/np.pi) * (x_test + 0.044715 * x_test**3)
    y_test = private_tanh(x_train, y_train, x_arg, K1, K2, L)
    y_test = 0.5 * x_test * (1.0 + y_test)
    return y_test

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
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    y = sigmoid(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[sigmoid] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
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
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True sigmoid')
    
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
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    y = tanh(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_tanh(x_fit, y, x_fit, K1, K2, L)
            max_error = np.max(np.abs(y - y_test))
            mean_error = np.mean(np.abs(y - y_test))
            rmse = np.sqrt(np.mean((y - y_test)**2))
            results[(K1, K2)] = {'y_hat': y_test, 'rmse': rmse}
            print(f"[tanh] K1={K1}, K2={K2}, RMSE={rmse:.6e}, max_error={max_error:.6e}, mean_error={mean_error:.6e}")
    
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
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True tanh')
    
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
    L = 8.0
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    y = SiLU(x_fit)
    y_train = sigmoid(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_SiLU(x_fit, y_train, x_fit, K1, K2, L)
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
            plt.plot(x_fit, y_hat, color=color, linestyle=linestyle,
                     linewidth=2, label=f"$K_1$={K1}, $K_2$={K2}")
    
    # 绘制真实曲线
    plt.plot(x_fit, y, 'k-', linewidth=3, label='True SiLU')
    
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
    plt.show()
    plt.close()

def inv_compare():
    """inv函数不同参数对比"""
    L = 4.0
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
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

def inv_sqrt_compare():
    """inv_sqrt函数不同参数对比"""
    L = 8.0
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
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
    plt.xlim(0, 8)
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

def GeLU_compare():
    """GeLU函数不同参数对比"""
    L = 8
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-4, 4, 3000)
    y = GeLU(x_test)
    y_train = tanh(x_fit)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_GeLU(x_fit, y_train, x_test, K1, K2, L)
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

def GeLU_compare2():
    """GeLU函数不同参数对比"""
    L = 8
    num_samples = 3000
    K1_list = [0]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-5, 5, 3000)
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

def ex_compare():
    L = 8
    num_samples = 3000
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-4, 0, 3000)
    y = e_x(x_test) + 1

    y_train = sigmoid(x_fit)
    x_fit2 = np.linspace(0.1, 4, num_samples)
    y_train2 = inv_sqrt(x_fit2)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y_train, x_test, K1, K2, L)

            y_test = np.abs(1 - y_test)
            
            y_test = private_inv(x_fit2, y_train2, y_test, K1, K2, L)

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
    x_continuous = np.linspace(-4, 0, 500)
    s = 20
    y_true = np.exp(x_continuous)
    y_simple = np.array([private_e_x(int(x * (1 << s)), s) / (1 << s) for x in x_continuous])
    
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
    plt.ylim(0, 2)
    
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
    K1_list = [0, 1, 2]
    K2_list = [1, 6, 12]
    x_fit = np.linspace(-L, L, num_samples)
    x_test = np.linspace(-4, 0, 3000)
    y = softmax(x_test)
    x_max = np.max(x_test)
    exp_x = np.exp(x_test - x_max)
    y = softmax(x_test)
    print(y)

    y_train = sigmoid(x_fit)
    x_fit2 = np.linspace(0.1, 4, num_samples)
    y_train2 = inv_sqrt(x_fit2)
    
    results = {}
    
    for K1 in K1_list:
        for K2 in K2_list:
            y_test = private_sigmoid(x_fit, y_train, x_test, K1, K2, L)

            y_test = np.abs(1 - y_test)
            
            y_test = private_inv(x_fit2, y_train2, y_test, K1, K2, L)
            y_test = y_test - 1

            y_test_sum = np.sum(y_test) / 3000
            print(y_test_sum)
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

# 运行所有比较
print("Starting function comparisons...")
# sigmoid_compare()
# tanh_compare()
# SiLU_compare()
# inv_compare()
# inv_sqrt_compare()
# GeLU_compare()
# # GeLU_compare2()
# # GeLU_compare3()
ex_compare()
# # ex_compare2()
# Softmax_compare()