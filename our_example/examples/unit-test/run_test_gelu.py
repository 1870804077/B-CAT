import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

# ================== ⚙️ 全局配置 (Global Config) ==================
DEVICE = "cuda"
RUNS = 1 # 性能测试运行次数

K_PARAMS = [(1, 8)]
# K_PARAMS = [(1, 8),(1, 12),(2, 12),(3, 12)]
TEST_SIZES = [(1,2000)]
# TEST_SIZES = [(1,500),(1,1000),(1,1500),(1,2000)]
# ================== 🛠️ 辅助工具 (Helpers) ==================

def run_perf_benchmark(input_generator_func, operation_func):
    """
    执行具体的性能测试循环 (修正版：增加 CUDA 同步和预热)
    """
    times, bytes_stats, rounds_stats = {}, {}, {}

    for size in TEST_SIZES:
        # 1. 生成输入
        x_enc = input_generator_func(size)
        
        operation_func(x_enc)
        if DEVICE == "cuda":
            torch.cuda.synchronize() # 确保预热真正完成

        # 2. 重置通信统计
        crypten.reset_communication_stats()

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        # 3. 计时开始
        start_time = time.time()
        
        for _ in range(RUNS):
            operation_func(x_enc)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        # 4. 记录数据
        dim = size[1]
        times[dim] = time.time() - start_time
        
        stats = crypten.get_communication_stats()
        bytes_stats[dim] = stats["bytes"]
        rounds_stats[dim] = stats["rounds"]
        
    return times, bytes_stats, rounds_stats

def print_report(op_name, config_str, max_err, avg_err, times, bytes_stats, rounds_stats):
    """
    打印报告：先打印配置，再打印精度和性能
    """
    if crypten.comm.get().get_rank() == 0:
        print(f"\n{'='*20} 🧪 {op_name} Test {'='*20}", flush=True)
        print(f"⚙️  Configuration: {config_str}", flush=True)
        print("-" * 50, flush=True)
        
        print(f"✅ Precision Check:", flush=True)
        print(f"   Max Error: {max_err:.8f}", flush=True)
        print(f"   Avg Error: {avg_err:.8f}", flush=True)
        
        print(f"🚀 Performance (Avg over {RUNS} runs):", flush=True)
        for size in TEST_SIZES:
            dim = size[1]
            t = times[dim] / RUNS
            comm = bytes_stats[dim] / 1048576 / RUNS
            rnd = rounds_stats[dim] / RUNS
            print(f"   Shape {size}: Time: {t:.7f}s | Comm: {comm:.7f}MB | Rounds: {rnd:.0f}", flush=True)
        print("="*60, flush=True)

# ================== 🧪 具体测试函数 (Test Functions) ==================

def test_sigmoid():
    # 配置方法名 (确保和你底层实现一致)
    crypten.cfg.functions.sigmoid_tanh_method = "newer_debug"
    
    print(f"\n{'='*40}")
    print(f"🧪 Sigmoid 精度验证模式 (Skip Performance)")
    print(f"{'='*40}")

    # 遍历每一组 K1, K2 参数
    for k1, k2 in K_PARAMS:
        # 1. 构造测试数据
        # 建议范围覆盖 Sigmoid 的非线性区 [-5, 5] 和饱和区
        x = torch.arange(-7, 7, 0.001, device=DEVICE)
        
        # 2. 计算标准答案 (Ground Truth)
        y_original = torch.sigmoid(x)
        
        # 3. 计算 MPC 近似值
        # 注意：这里调用的是你修改过的包含 _fourier_series_x3 的逻辑
        y_enc = crypten.cryptensor(x, device=DEVICE)
        y_actual = y_enc.sigmoid(k1=k1, k2=k2).get_plain_text()
        
        # 4. 计算误差
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err = diff.max().item()
        avg_err = diff.mean().item()

        # 5. 直接打印结果
        print(f"\n🔹 配置: k1={k1}, k2={k2}")
        print(f"   输入范围: [{x.min():.1f}, {x.max():.1f}]")
        print(f"   ✅ Max Error: {max_err:.8f}")
        print(f"   ✅ Avg Error: {avg_err:.8f}")
        
        # 简单的 Pass/Fail 提示 (阈值可按需调整，例如 0.01)
        if max_err < 0.01:
            print("   ✨ 精度达标 (Excellent)")
        else:
            print("   ⚠️ 精度可能有问题 (Check Truncation/Mask)")
            
    print(f"\n{'='*40}\n")

def test_tanh():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time+"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(-5, 5, 0.001, device=DEVICE)
        y_original = torch.tanh(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).tanh(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.tanh(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Tanh", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_gelu():
    crypten.cfg.functions.gelu_method = "newer_time"
    approximate = "none"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(-5, 5, 0.001, device=DEVICE)
        y_original = torch.nn.functional.gelu(x, approximate=approximate)
        # 假设你的 gelu 实现也接受 k1, k2
        y_actual = crypten.cryptensor(x, device=DEVICE).gelu(approximate=approximate, k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试 (随机输入)
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 10) - 5
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.gelu(approximate=approximate, k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("GeLU", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_silu():
    crypten.cfg.functions.silu_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(-15, 15, 0.001, device=DEVICE)
        y_original = torch.nn.functional.silu(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).silu(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试 (随机输入)
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 16) - 8
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.silu(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("SiLU", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_exp():
    crypten.cfg.functions.exp_method = "newer_time+"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.linspace(-15.0, -3.0, 10000, device=DEVICE)
        y_original = torch.exp(x)
        # 假设 exp 实现接受参数（可能是迭代次数或多项式参数）
        y_actual = crypten.cryptensor(x, device=DEVICE).exp(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.exp(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Exp", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)

def test_erf():
    # 1. 配置方法名 (假设你在 approximations.py 中也使用了 erf_method 并在 cfg 中注册了它)
    # 如果 CrypTen 默认没注册 erf_method，确保你的实现能通过参数透传
    crypten.cfg.functions.erf_method = "newer_time+"

    for k1, k2 in K_PARAMS:
        # 2. 精度验证 (erf 在 [-3, 3] 之外基本就饱和到 -1 或 1 了)
        x = torch.linspace(-5.0, 5.0, 10000, device=DEVICE)
        y_original = torch.erf(x)
        
        # 假设你的 erf 实现接受 k1, k2 参数
        y_actual = crypten.cryptensor(x, device=DEVICE).erf(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 3. 性能测试 (输入: 随机 [-4, 4] 覆盖非线性最剧烈的区域)
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 8) - 4
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.erf(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Erf", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)

def test_reciprocal():
    crypten.cfg.functions.reciprocal_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(0.1, 5.0, 0.001, device=DEVICE)
        y_original = 1.0 / x
        y_actual = crypten.cryptensor(x, device=DEVICE).reciprocal(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.reciprocal(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Reciprocal", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_inv_sqrt():
    crypten.cfg.functions.sqrt_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(0.1, 5.0, 0.001, device=DEVICE)
        y_original = 1.0 / torch.sqrt(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).inv_sqrt(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.inv_sqrt(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("InvSqrt", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)

def test_softmax():
    # 1. 配置方法名 (假设你的实现中 softmax 会透传 k1, k2 给内部的 exp 和 reciprocal)
    # 注意：Softmax 通常在特定维度上做，这里默认 dim=-1
    crypten.cfg.functions.softmax_method = "newer"
    crypten.cfg.functions.reciprocal_method = "newer_time"
    crypten.cfg.functions.exp_method = "newer_time"
    for k1, k2 in K_PARAMS:
        # 2. 精度验证
        # 生成一些类似模型输出的 logits 数据
        x = torch.randn(1, 128, device=DEVICE) 
        y_original = torch.nn.functional.softmax(x, dim=-1)
        
        # 这里的 .softmax() 内部应当使用了你拟合参数后的 exp 和 reciprocal
        x_enc = crypten.cryptensor(x, device=DEVICE)
        y_mpc = x_enc.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)
        y_actual = y_mpc.get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 3. 性能测试
        def input_gen(size):
            # size 传入的是 TEST_SIZES 里的 (1, 500) 等
            return crypten.cryptensor(torch.randn(*size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Softmax", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)
# ================== 🚀 主程序 (Main) ==================

def main():
    crypten.init()
    for i in range(1,2) :
        print("第{i}次测试",i)
        test_sigmoid()
        # test_tanh()

        # test_exp()
        # test_erf()
        # test_reciprocal()
        # test_inv_sqrt()
        # test_gelu()
        # test_silu()

        # test_softmax()
if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()