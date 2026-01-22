# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher

# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 10
#     gelu_time, gelu_bytes, gelu_rounds = {}, {}, {}
#     approximate = "none"
#     crypten.cfg.functions.gelu_method = "newer_time"

#     x = torch.arange(-5, 5, 0.001)
#     y_original = torch.nn.functional.gelu(x, approximate=approximate)
#     y_actual = crypten.cryptensor(x).gelu(approximate=approximate).get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()
    
#     for gelu_size in [(128, 3072), (128, 4096)]:
#         gelu_in = crypten.cryptensor(torch.zeros(gelu_size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             gelu_in.gelu(approximate=approximate)
#         gelu_time[gelu_size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         gelu_bytes[gelu_size[1]] = stats["bytes"]
#         gelu_rounds[gelu_size[1]] = stats["rounds"]
    
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.4f}, avg error: {avg_err:.6f}")
#         for gelu_size in [[128, 3072], [128, 4096]]:
#             print(f"({gelu_size[0]}, {gelu_size[1]}) "
#                 f"time: {gelu_time[gelu_size[1]] / runs:.4f}s, "
#                 f"bytes: {gelu_bytes[gelu_size[1]] / 1048576 / runs:.0f} MB, "
#                 f"rounds: {gelu_rounds[gelu_size[1]] / runs:.0f}"
#             )

# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()

#silu
import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def main():
    crypten.init()
    device = "cuda" 
    runs = 1
    silu_time, silu_bytes, silu_rounds = {}, {}, {}
    
    # 设置 SiLU 方法
    crypten.cfg.functions.silu_method = "newer_time"

    x = torch.arange(14, 15, 0.001, device=device)
    
    # 2. 计算 PyTorch 原生结果 (Ground Truth)
    y_original = torch.nn.functional.silu(x) 
    
    # 3. 计算 CrypTen MPC 结果
    # 注意：这里会调用你写的 silu 函数
    x_enc = crypten.cryptensor(x, device=device)
    y_mpc = x_enc.silu()
    
    # 4. 解密结果 (Get Plain Text)
    y_actual = y_mpc.get_plain_text()
    
    # 5. 计算误差
    diff = (y_original - y_actual).abs()
    max_err = diff.max()
    avg_err = diff.mean()
    
    # 6. 找出误差最大的那个点 (调试神器)
    max_err_idx = diff.argmax()
    max_err_x = x[max_err_idx]
    max_err_y_true = y_original[max_err_idx]
    max_err_y_mpc = y_actual[max_err_idx]


    # # --- 性能测试 ---
    # ✅ 2. 定义好要测试的列表，确保计算和打印使用同一个列表
    test_sizes = [(128, 3072)] # 如果想测 4096，请写成 [(128, 3072), (128, 4096)]

    for silu_size in test_sizes:
        # ✅ 3. 修正：使用 silu_size 动态生成数据，不要写死 768
        # 生成形状为 (128, 3072) 的数据
        x_perf = (torch.rand(*silu_size) * 16) - 8 
        
        silu_in = crypten.cryptensor(x_perf, device=device)
        crypten.reset_communication_stats()
        
        # 预热一次 (可选，防止第一次初始化耗时影响)
        # silu_in.silu() 
        
        start_time = time.time()
        for _ in range(runs):
            silu_in.silu() 
        
        # 记录时间
        current_dim = silu_size[1] # 取第二维 3072 作为 Key
        silu_time[current_dim] = time.time() - start_time
        stats = crypten.get_communication_stats()
        silu_bytes[current_dim] = stats["bytes"]
        silu_rounds[current_dim] = stats["rounds"]

    # --- 打印结果 ---
    # 仅 rank 0 打印
    if crypten.comm.get().get_rank() == 0:
        print("="*40, flush=True)
        print(f"Precision Check:\nmax error: {max_err:.4f}, avg error: {avg_err:.6f}", flush=True)
        print("-" * 40, flush=True)
        print("Performance:", flush=True)
        
        # ✅ 4. 修正：遍历同一个 test_sizes 列表，防止 KeyError
        for silu_size in test_sizes:
            dim = silu_size[1]
            print(f"Shape {silu_size}: "
                  f"time: {silu_time[dim] / runs:.4f}s, "
                  f"bytes: {silu_bytes[dim] / 1048576 / runs:.2f} MB, "
                  f"rounds: {silu_rounds[dim] / runs:.0f}", flush=True)
        print("="*40, flush=True)

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher


# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 10
#     exp_time, exp_bytes, exp_rounds = {}, {}, {}
    
#     # 设置 exp 方法（确保您的 CrypTen 实现中已注册 "newer_time"）
#     crypten.cfg.functions.exp_method = "newer_time"

#     # 精度验证：比较明文 PyTorch exp 与 CrypTen exp
#     # 注意：exp 输入范围需合理（避免溢出），这里用 [-2, 2]
#     x = torch.linspace(-2.0, 2.0, 10000)
#     y_original = torch.exp(x)  # PyT Torch 原生 exp
#     y_actual = crypten.cryptensor(x).exp().get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()

#     # 性能测试：不同输入尺寸
#     for exp_size in [(128, 3072), (128, 4096)]:
#         exp_in = crypten.cryptensor(torch.zeros(exp_size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             exp_in.exp()  # 调用 exp
        
#         exp_time[exp_size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         exp_bytes[exp_size[1]] = stats["bytes"]
#         exp_rounds[exp_size[1]] = stats["rounds"]

#     # 仅 rank 0 打印结果
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
#         for exp_size in [[128, 3072], [128, 4096]]:
#             print(f"({exp_size[0]}, {exp_size[1]}) "
#                   f"time: {exp_time[exp_size[1]] / runs:.4f}s, "
#                   f"bytes: {exp_bytes[exp_size[1]] / 1048576 / runs:.0f} MB, "
#                   f"rounds: {exp_rounds[exp_size[1]] / runs:.0f}")


# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()
# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher


# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 10
#     sigmoid_time, sigmoid_bytes, sigmoid_rounds = {}, {}, {}
    
#     # 设置 sigmoid 方法（请确保您的 CrypTen 实现中已注册 "newer_time"）
#     crypten.cfg.functions.sigmoid_tanh_method = "newer_time"

#     # 精度验证：比较明文 PyTorch sigmoid 与 CrypTen sigmoid
#     x = torch.arange(-5, 5, 0.001)
#     y_original = torch.sigmoid(x)  # PyTorch 原生 sigmoid
#     y_actual = crypten.cryptensor(x).sigmoid().get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()

#     # 性能测试：不同输入尺寸
#     for sigmoid_size in [(128, 3072), (128, 4096)]:
#         sigmoid_in = crypten.cryptensor(torch.zeros(sigmoid_size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             sigmoid_in.sigmoid()  # 调用 sigmoid
        
#         sigmoid_time[sigmoid_size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         sigmoid_bytes[sigmoid_size[1]] = stats["bytes"]
#         sigmoid_rounds[sigmoid_size[1]] = stats["rounds"]

#     # 仅 rank 0 打印结果
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
#         for sigmoid_size in [[128, 3072], [128, 4096]]:
#             print(f"({sigmoid_size[0]}, {sigmoid_size[1]}) "
#                   f"time: {sigmoid_time[sigmoid_size[1]] / runs:.4f}s, "
#                   f"bytes: {sigmoid_bytes[sigmoid_size[1]] / 1048576 / runs:.0f} MB, "
#                   f"rounds: {sigmoid_rounds[sigmoid_size[1]] / runs:.0f}")


# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()

# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher


# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 10
#     tanh_time, tanh_bytes, tanh_rounds = {}, {}, {}
    
#     # 设置 tanh 方法（请确保您的 CrypTen 实现中已注册 "newer_time"）
#     crypten.cfg.functions.sigmoid_tanh_method = "newer_time"

#     # 精度验证：比较明文 PyTorch tanh 与 CrypTen tanh
#     x = torch.arange(-5, 5, 0.001)
#     y_original = torch.tanh(x)  # PyTorch 原生 tanh
#     y_actual = crypten.cryptensor(x).tanh().get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()

#     # 性能测试：不同输入尺寸
#     for tanh_size in [(128, 3072), (128, 4096)]:
#         tanh_in = crypten.cryptensor(torch.zeros(tanh_size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             tanh_in.tanh()  # 调用 tanh
        
#         tanh_time[tanh_size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         tanh_bytes[tanh_size[1]] = stats["bytes"]
#         tanh_rounds[tanh_size[1]] = stats["rounds"]

#     # 仅 rank 0 打印结果
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
#         for tanh_size in [[128, 3072], [128, 4096]]:
#             print(f"({tanh_size[0]}, {tanh_size[1]}) "
#                   f"time: {tanh_time[tanh_size[1]] / runs:.4f}s, "
#                   f"bytes: {tanh_bytes[tanh_size[1]] / 1048576 / runs:.0f} MB, "
#                   f"rounds: {tanh_rounds[tanh_size[1]] / runs:.0f}")


# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()

# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher


# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 10
#     reciprocal_time, reciprocal_bytes, reciprocal_rounds = {}, {}, {}
    
#     # 设置 reciprocal 方法（请确保您的 CrypTen 实现中已注册 "newer_time"）
#     crypten.cfg.functions.reciprocal_method = "newer_time"

#     # 精度验证：比较明文 PyTorch reciprocal 与 CrypTen reciprocal
#     # 注意：reciprocal 在 x=0 处无定义，且对小值敏感 → 选择安全范围 [0.1, 5]
#     x = torch.arange(0.1, 5.0, 0.001)
#     y_original = 1.0 / x  # 明文倒数
#     y_actual = crypten.cryptensor(x).reciprocal().get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()

#     # 性能测试：不同输入尺寸（使用正值避免除零）
#     for size in [(128, 3072), (128, 4096)]:
#         # 输入初始化为 1.0（安全值），避免接近零
#         reciprocal_in = crypten.cryptensor(torch.ones(size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             reciprocal_in.reciprocal()  # 调用 reciprocal
        
#         reciprocal_time[size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         reciprocal_bytes[size[1]] = stats["bytes"]
#         reciprocal_rounds[size[1]] = stats["rounds"]

#     # 仅 rank 0 打印结果
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
#         for size in [[128, 3072], [128, 4096]]:
#             print(f"({size[0]}, {size[1]}) "
#                   f"time: {reciprocal_time[size[1]] / runs:.4f}s, "
#                   f"bytes: {reciprocal_bytes[size[1]] / 1048576 / runs:.0f} MB, "
#                   f"rounds: {reciprocal_rounds[size[1]] / runs:.0f}")


# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()

# import time
# import torch
# import crypten
# from multiprocess_launcher import MultiProcessLauncher


# def main():
#     crypten.init()
#     device = "cuda"
#     runs = 1
#     inv_sqrt_time, inv_sqrt_bytes, inv_sqrt_rounds = {}, {}, {}
    
#     # 设置 inv_sqrt 方法（请确保您的 CrypTen 实现中已注册 "newer_time"）
#     crypten.cfg.functions.sqrt_method = "newer_time"

#     # 精度验证：比较明文 PyTorch inv_sqrt 与 CrypTen inv_sqrt
#     # 注意：x 必须 > 0；选择典型范围 [0.1, 5.0]（覆盖小值和大值）
#     x = torch.arange(0.1, 5.0, 0.001)
#     y_original = 1.0 / torch.sqrt(x)  # 明文 1/sqrt(x)
#     y_actual = crypten.cryptensor(x).inv_sqrt().get_plain_text()
#     max_err = (y_original - y_actual).abs().max()
#     avg_err = (y_original - y_actual).abs().mean()

#     # 性能测试：不同输入尺寸（使用正值避免无效输入）
#     for size in [(128, 3072), (128, 4096)]:
#         # 初始化为 1.0（安全值），确保 x > 0
#         inv_sqrt_in = crypten.cryptensor(torch.ones(size), device=device)
#         crypten.reset_communication_stats()
#         start_time = time.time()
        
#         for _ in range(runs):
#             inv_sqrt_in.inv_sqrt()  # 调用 inv_sqrt
        
#         inv_sqrt_time[size[1]] = time.time() - start_time
#         stats = crypten.get_communication_stats()
#         inv_sqrt_bytes[size[1]] = stats["bytes"]
#         inv_sqrt_rounds[size[1]] = stats["rounds"]

#     # 仅 rank 0 打印结果
#     if crypten.comm.get().get_rank() == 0:
#         print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
#         for size in [[128, 3072], [128, 4096]]:
#             print(f"({size[0]}, {size[1]}) "
#                   f"time: {inv_sqrt_time[size[1]] / runs:.4f}s, "
#                   f"bytes: {inv_sqrt_bytes[size[1]] / 1048576 / runs:.0f} MB, "
#                   f"rounds: {inv_sqrt_rounds[size[1]] / runs:.0f}")


# if __name__ == "__main__":
#     launcher = MultiProcessLauncher(2, main)
#     launcher.start()
#     launcher.join()
#     launcher.terminate()