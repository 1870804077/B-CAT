import torch
import crypten
import time
import math
from crypten.config import cfg
from crypten.mpc import MPCTensor 

# 导入 approximations 模块
import crypten.common.functions.approximations as approx
from crypten.common.functions.approximations import odrelu, od_sign

# 补丁代码
def _od_sign_patch(self):
    return od_sign(self)

if not hasattr(MPCTensor, 'od_sign'):
    MPCTensor.od_sign = _od_sign_patch
    # 确保把所有近似函数都挂载上，防止部分环境漏掉
    MPCTensor.inv_sqrt = approx.inv_sqrt
    MPCTensor.reciprocal = approx.reciprocal
    MPCTensor.silu = approx.silu
    MPCTensor.sigmoid = approx.sigmoid
    MPCTensor.tanh = approx.tanh
    MPCTensor.gelu = approx.gelu
    MPCTensor.exp = approx.exp
    print("[Setup] Successfully patched MPCTensor with .od_sign()")

# ==========================================
# 辅助函数: 打印测试表头
# ==========================================
def print_header(name, range_info):
    print("\n" + "="*60)
    print(f"      Testing Secure {name} (Auto-Fit Mode)")
    print(f"      Range: {range_info}")
    print("="*60)
    # 修改列名为 Mean Err
    print(f"{'K1':<4} | {'K2':<4} | {'Mean Err':<12} | {'Max Error':<12} | {'Time':<8}")
    print("-" * 60)

# ==========================================
# 测试函数: Sigmoid
# ==========================================
def test_sigmoid_range():
    print_header("Sigmoid", "K1=1, K2=[4, 12], L=8.0")
    x = (torch.rand(1, 128, 768) * 16) - 8 
    y_true = torch.sigmoid(x)
    x_enc = crypten.cryptensor(x)
    for k1 in (1, 3):
        for k2 in range(4, 13):
            start = time.time()
            y_enc = x_enc.sigmoid(k1=k1, k2=k2, L=8.0)
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            # 改为 Mean Error
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: Tanh
# ==========================================
def test_tanh_range():
    print_header("Tanh", "K1=1, K2=[4, 12], L=6.0")
    x = (torch.rand(1, 128, 768) * 12) - 6
    y_true = torch.tanh(x)
    x_enc = crypten.cryptensor(x)
    for k1 in (1, 3):
        for k2 in range(4, 13):
            start = time.time()
            y_enc = x_enc.tanh(k1=1, k2=k2, L=6.0)
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            # 改为 Mean Error
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: GeLU (内部调用 Tanh)
# ==========================================
def test_gelu_range():
    print_header("GeLU", "K1=1, K2=[4, 12], Internal Tanh L=4.0")
    cfg.functions.gelu_method = "newer_time"
    x = (torch.rand(1, 128, 768) * 10) - 5
    y_true = torch.nn.functional.gelu(x)
    x_enc = crypten.cryptensor(x)
    for k1 in (1,3):
        for k2 in range(12, 13):
            start = time.time()
            y_enc = x_enc.gelu(k1=k1, k2=k2) 
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            # 改为 Mean Error
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: Exponential
# ==========================================
def test_exp_range():
    print_header("Exponential", "K1=1, K2=[4, 12], Range=[-16, -4]")
    x = (torch.rand(1, 128, 768) * 12) - 16 
    y_true = torch.exp(x)
    x_enc = crypten.cryptensor(x)
    cfg.functions.exp_method = "newer_debug"
    for k1 in (1, 3):
        for k2 in range(4, 13):
            start = time.time()
            y_enc = x_enc.exp(k1=k1, k2=k2)
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            # 改为 Mean Error
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: InvSqrt
# ==========================================
def test_inv_sqrt_range():
    total_L = 3000.0 
    start_val = 0.1
    
    print_header("InvSqrt (Split-Range)", f"K1=1, K2=[4, 12], Range=[{start_val}, {total_L}]")
    
    x = torch.rand(1, 128, 768) * (total_L - start_val) + start_val
    
    y_true = torch.rsqrt(x)
    x_enc = crypten.cryptensor(x)
    
    cfg.functions.sqrt_method = "newer_debug"
    for k1 in (1,3):
        for k2 in range(4,13):
            start = time.time()
            y_enc = x_enc.inv_sqrt(k1=k1, k2=k2, L=total_L)
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: SiLU
# ==========================================
def test_silu_range():
    print_header("SiLU", "K1=1, K2=[4, 12], L=10.0")
    x = (torch.rand(1, 128, 768) * 30) - 15
    y_true = torch.nn.functional.silu(x)
    x_enc = crypten.cryptensor(x)
    cfg.functions.silu_method = "newer_debug"
    k1_val = 1
    for k2_val in range(4, 13):
        start = time.time()
        y_enc = x_enc.silu(k1=k1_val, k2=k2_val, L=10.0)
        t = time.time() - start
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        
        # 改为 Mean Error
        mean_err = diff.mean().item()
        max_err = diff.max().item()
        print(f"{k1_val:<4} | {k2_val:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: Reciprocal
# ==========================================
def test_reciprocal_range():
    total_L = 3000.0 
    start_val = 0.1
    
    print_header("Reciprocal (Split-Range)", f"K1=1, K2=[4, 12], Range=[{start_val}, {total_L}]")
    
    x = torch.rand(1, 128, 768) * (total_L - start_val) + start_val
    
    y_true = 1.0 / x
    x_enc = crypten.cryptensor(x)
    
    cfg.functions.reciprocal_method = "newer_debug"
    for k1 in (1,3):
        for k2 in range(4,13):
            start = time.time()
            y_enc = x_enc.reciprocal(k1=k1, k2=k2, L=total_L)
            t = time.time() - start
            
            y_pred = y_enc.get_plain_text()
            diff = (y_true - y_pred).abs()
            
            mean_err = diff.mean().item()
            max_err = diff.max().item()
            
            print(f"{k1:<4} | {k2:<4} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s")

# ==========================================
# 测试函数: Softmax
# ==========================================
def test_softmax_range():
    print("========================================================================")
    print("      Testing Softmax (Grid Search: Exp K2 vs Recip K2)")
    print("      Range: K1=1, Exp_K2=[4..12], Recip_K2=[2..10]")
    print("========================================================================")
    
    # 生成测试数据
    torch.manual_seed(0) # 固定随机种子以便复现
    x = (torch.rand(1, 128, 768) * 12) - 16  # Range approx [-16, -4]
    y_true = torch.softmax(x, dim=-1)
    
    # 加密数据
    x_enc = crypten.cryptensor(x)
    cfg.functions.softmax_method = "newer_debug"
    # 表头
    print(f"{'Exp K2':<8} | {'Recip K2':<8} | {'Mean Err':<12} | {'Max Err':<12} | {'Time':<8}")
    print("-" * 65)

    # 定义我们要搜索的参数范围
    # Exp 通常比较难，建议从 6 开始测；Recip 收敛快，可以从 2 或 4 开始
    exp_k2_list = list(range(4, 13))   # [4, 5, ..., 12]
    recip_k2_list = list(range(4, 13)) # [4, 5, ..., 12]
    for k1 in (1,3):
        for k2_exp in exp_k2_list:
            for k2_recip in recip_k2_list:
                
                # 计时开始
                start = time.time()
                
                # 调用修改后的 softmax，传入两个不同的 K2
                y_enc = x_enc.softmax(dim=-1, k1=k1, k2_exp=k2_exp, k2_recip=k2_recip)
                
                # 强制同步一下(如果是GPU)并停止计时，或者直接解密包含在时间内也行
                y_pred = y_enc.get_plain_text()
                t = time.time() - start
                
                # 计算误差
                diff = (y_true - y_pred).abs()
                mean_err = diff.mean().item()
                max_err = diff.max().item()
                
                # 打印结果
                # 使用高亮标记出特别好的结果（例如 Max Err < 1e-3 且速度快）
                marker = ""
                if max_err < 1e-3: marker = " *" # 标记可用组合
                
                print(f"{k2_exp:<8} | {k2_recip:<8} | {mean_err:.6e} | {max_err:.6e} | {t:.3f}s {marker}")
            
            # 每个 Exp K2 测完后打印一个分割线，方便阅读
            print("-" * 65)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("Initializing CrypTen...")
    # crypten.config.cfg.encoder.precision_bits = 20  

    # 初始化 CrypTen
    crypten.init()

    # (可选) 打印一下确认修改生效
    print(f"[Config] Current Precision Bits: {crypten.config.cfg.encoder.precision_bits}")
    
    # 根据需要取消注释
    test_sigmoid_range()
    test_tanh_range()

    test_inv_sqrt_range()
    test_reciprocal_range()

    test_gelu_range()
    test_silu_range()
    
    test_exp_range()
    # test_softmax_range()
    
    print("\n" + "="*60)
    print("      All Range Tests Completed!")
    print("="*60)