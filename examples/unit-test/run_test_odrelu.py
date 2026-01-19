import torch
import crypten
import time
import math
from crypten.config import cfg
from crypten.mpc import MPCTensor 

# 导入 approximations 模块
import crypten.common.functions.approximations as approx
from crypten.common.functions.approximations import odrelu

# ==========================================
# 临时补丁区域: 挂载 OD-ReLU
# ==========================================
def _odrelu_patch(self):
    return odrelu(self)

def _od_sign_patch(self):
    # 注意：这里需要从 approximations 模块导入之前写的 od_sign 函数
    from crypten.common.functions.approximations import od_sign
    return od_sign(self)

if not hasattr(MPCTensor, 'odrelu'):
    MPCTensor.odrelu = _odrelu_patch
    print("[Setup] Successfully patched MPCTensor with .odrelu()")

if not hasattr(MPCTensor, 'od_sign'):
    MPCTensor.od_sign = _od_sign_patch
    print("[Setup] Successfully patched MPCTensor with .od_sign()")

# ==========================================
# 测试函数: GELU (newer Mode)
# ==========================================
def test_gelu():
    print("\n" + "="*40)
    print("      Testing Secure GELU (newer)")
    print("="*40)
    cfg.functions.gelu_method = "newer"
    
    B, L, D = 1, 128, 768 
    x = (torch.rand(B, L, D) * 8) - 4 # 覆盖 [-10, 10]，包含边界 8 的截断
    y_true = torch.nn.functional.gelu(x)
    x_enc = crypten.cryptensor(x)

    # 注意：GELU 内部调用了 Tanh，因此其精度受 GELU_K 影响
    test_params = [(1, 9), (1, 12)] 

    for k1, k2 in test_params:
        print(f"\n--- Testing GELU: K1={k1}, K2={k2} ---")
        approx.GELU_K1, approx.GELU_K2 = k1, k2
        
        start_time = time.time()
        y_enc = x_enc.gelu()
        end_time = time.time()
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        
        # --- 新增：输出前 5 个最大误差及其对应的 x ---
        diff_flat = diff.view(-1)
        x_flat = x.view(-1)
        top_errors, indices = torch.topk(diff_flat, 5)
        top_x = x_flat[indices]

        print(f"Time Cost: {end_time - start_time:.4f}s | RMSE: {torch.sqrt((diff**2).mean()).item():.6e}")
        print("Top 5 Max Errors and corresponding x:")
        for i in range(5):
            print(f"  [{i+1}] Error: {top_errors[i].item():.6e} | x: {top_x[i].item():.6f}")
# ==========================================
# 测试函数: SiLU (newer Mode)
# ==========================================
def test_silu():
    print("\n" + "="*40)
    print("      Testing Secure SiLU (newer)")
    print("="*40)
    cfg.functions.silu_method = "newer"
    
    B, L, D = 1, 128, 768 
    x = (torch.rand(B, L, D) * 8) - 4 # 覆盖 [-6, 6]，包含边界 4 的截断
    y_true = torch.nn.functional.silu(x)
    x_enc = crypten.cryptensor(x)

    test_params = [(1, 5), (1, 12)]

    for k1, k2 in test_params:
        print(f"\n--- Testing SiLU: K1={k1}, K2={k2} ---")
        approx.SILU_K1, approx.SILU_K2 = k1, k2
        
        start_time = time.time()
        y_enc = x_enc.silu()
        end_time = time.time()
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        
        # --- 新增：输出前 5 个最大误差及其对应的 x ---
        diff_flat = diff.view(-1)
        x_flat = x.view(-1)
        top_errors, indices = torch.topk(diff_flat, 5)
        top_x = x_flat[indices]

        print(f"Time Cost: {end_time - start_time:.4f}s | RMSE: {torch.sqrt((diff**2).mean()).item():.6e}")
        print("Top 5 Max Errors and corresponding x:")
        for i in range(5):
            print(f"  [{i+1}] Error: {top_errors[i].item():.6e} | x: {top_x[i].item():.6f}")

# ==========================================
# 测试函数: InvSqrt (newer Mode)
# ==========================================
def test_inv_sqrt():
    print("\n" + "="*40)
    print("      Testing Secure InvSqrt (newer)")
    print("="*40)
    cfg.functions.sqrt_method = "newer"
    
    # 准备正数数据，避免 0 附近的极端不稳定性
    B, L, D = 1, 128, 768 
    x = torch.rand(B, L, D) * 10 + 0.1 # [0.1, 10.1]
    y_true = torch.rsqrt(x)
    x_enc = crypten.cryptensor(x)

    # InvSqrt 依赖优化的牛顿迭代
    test_params = [(1, 6), (1, 12)] 

    for k1, k2 in test_params:
        print(f"\n--- Testing InvSqrt: K1={k1}, K2={k2} ---")
        approx.INVSQRT_K1, approx.INVSQRT_K2 = k1, k2
        
        start_time = time.time()
        y_enc = x_enc.inv_sqrt()
        end_time = time.time()
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        print(f"Time Cost: {end_time - start_time:.4f}s | Max Error: {diff.max().item():.6e} | RMSE: {torch.sqrt((diff**2).mean()).item():.6e}")

# ==========================================
# 测试函数: Reciprocal (newer Mode)
# ==========================================
def test_reciprocal():
    print("\n" + "="*40)
    print("      Testing Secure Reciprocal (newer)")
    print("="*40)
    # 使用 newer 模式，内部调用 (inv_sqrt)^2
    cfg.functions.reciprocal_method = "newer"
    
    B, L, D = 1, 128, 768 
    x = torch.rand(B, L, D) * 9 + 1.0 # [1.0, 10.0]
    y_true = 1.0 / x
    x_enc = crypten.cryptensor(x)

    test_params = [(1, 9), (1, 12)]

    for k1, k2 in test_params:
        print(f"\n--- Testing Reciprocal: K1={k1}, K2={k2} ---")
        approx.INV_K1, approx.INV_K2 = k1, k2
        
        start_time = time.time()
        y_enc = x_enc.reciprocal()
        end_time = time.time()
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        print(f"Time Cost: {end_time - start_time:.4f}s | Max Error: {diff.max().item():.6e} | RMSE: {torch.sqrt((diff**2).mean()).item():.6e}")


def test_sigmoid():
    print("\n" + "="*60)
    print("      Testing Secure Sigmoid (Forced newer_debug)")
    print("      Range: K1=1, K2=[4, 12], L=8.0")
    print("="*60)
    
    # 构造测试数据 [-8, 8]
    x = (torch.rand(1, 128, 768) * 16) - 8 
    y_true = torch.sigmoid(x)
    x_enc = crypten.cryptensor(x)
    
    L_val = 8.0
    k1_val = 1
    
    print(f"{'K1':<4} | {'K2':<4} | {'RMSE':<12} | {'Max Error':<12}")
    print("-" * 50)
    
    for k2_val in range(4, 13): 
        # 这里通过 method="newer_debug" 强制指定使用动态拟合分支
        y_enc = x_enc.sigmoid(k1=k1_val, k2=k2_val, L=L_val)
        
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        
        rmse = torch.sqrt((diff**2).mean()).item()
        max_err = diff.max().item()
        
        print(f"{k1_val:<4} | {k2_val:<4} | {rmse:.6e} | {max_err:.6e}")

def test_tanh():
    print("\n" + "="*40)
    print("      Testing Secure Tanh (newer)")
    print("="*40)
    cfg.functions.sigmoid_tanh_method = "newer"
    x = (torch.rand(1, 128, 768) * 16) - 8
    y_true = torch.tanh(x)
    x_enc = crypten.cryptensor(x)
    for k1, k2 in [(1, 9), (1, 12)]:
        approx.TANH_K1, approx.TANH_K2 = k1, k2
        y_enc = x_enc.tanh()
        y_pred = y_enc.get_plain_text()
        diff = (y_true - y_pred).abs()
        print(f"K1={k1}, K2={k2} | Max Error: {diff.max().item():.6e} | RMSE: {torch.sqrt((diff**2).mean()).item():.6e}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("Initializing CrypTen...")
    crypten.init()
    
    # 依次运行所有算子测试
    # test_sigmoid()
    # test_tanh()
    test_gelu()
    # test_silu()
    # test_inv_sqrt()
    # test_reciprocal()
    
    print("\n" + "="*40)
    print("      All Tests Completed!")
    print("="*40)