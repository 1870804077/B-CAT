import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

# ================== ⚙️ 全局配置 (Global Config) ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = 1  # 性能测试运行次数

# 参数配置 (K1, K2)
K_PARAMS = [(1, 12), (3, 12)]

# ✅ 修改：加入 BERT 常用的三维形状测试
TEST_SIZES = [
    (1, 2000),             # 2D 基础测试
    (1, 128, 768),         # 模拟 BERT: Batch=1, Seq=128, Hidden=768
    (8, 128, 768)          # 模拟 BERT: Batch=8, Seq=128, Hidden=768 (高负载)
]

# ================== 🛠️ 辅助工具 (Helpers) =====================

def inspect_cryptensor(name, enc_tensor):
    """
    🔍 调试探针：打印 CrypTen 张量的详细内部结构
    """
    # 只让 Rank 0 打印，防止多进程输出混乱
    if crypten.comm.get().get_rank() == 0:
        print(f"\n🔍 [INSPECT] {name}")
        print(f"  1. Wrapper Type (外层类型): {type(enc_tensor)}")
        print(f"  2. Logical Size (逻辑形状): {enc_tensor.size()}")
        
        # 访问 .share (ArithmeticSharedTensor)
        if hasattr(enc_tensor, 'share'):
            share = enc_tensor.share
            print(f"  3. Share Type   (分片类型): {type(share)}")
            
            # 访问 ._tensor (实际存储数据的 PyTorch Tensor)
            if hasattr(share, '_tensor'):
                raw_tensor = share._tensor
                print(f"  4. Raw ._tensor (底层数据): {type(raw_tensor)}")
                print(f"  5. Raw Size     (物理形状): {raw_tensor.size()}")
                print(f"  6. Device       (设备位置): {raw_tensor.device}")
        print("-" * 40)

def run_perf_benchmark(input_generator_func, operation_func):
    """
    执行具体的性能测试循环 (修正版：增加 CUDA 同步和预热)
    """
    times, bytes_stats, rounds_stats = {}, {}, {}

    for size in TEST_SIZES:
        # 1. 生成输入
        x_enc = input_generator_func(size)
        
        # 🔍 插入检查点：确认输入是否为 3D
        inspect_cryptensor(f"Input for size {size}", x_enc)
        
        # 预热 run
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
        # 使用元素总数作为 key，或者直接用 tuple
        dim_key = str(size) 
        times[dim_key] = time.time() - start_time
        
        stats = crypten.get_communication_stats()
        bytes_stats[dim_key] = stats["bytes"]
        rounds_stats[dim_key] = stats["rounds"]
        
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
            dim_key = str(size)
            if dim_key in times:
                t = times[dim_key] / RUNS
                comm = bytes_stats[dim_key] / 1048576 / RUNS
                rnd = rounds_stats[dim_key] / RUNS
                # [FIX] 这里加上 str(size)，否则元组无法使用 :<15 格式化
                print(f"   Shape {str(size):<15}: Time: {t:.7f}s | Comm: {comm:.7f}MB | Rounds: {rnd:.0f}", flush=True)
        print("="*60, flush=True)

# ================== 🧪 具体测试函数 (Test Functions) ==================

def test_sigmoid():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time"
    
    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(-8, 8, 0.001, device=DEVICE)
        y_original = torch.sigmoid(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).sigmoid(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.sigmoid(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Sigmoid", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_tanh():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.arange(-7, 7, 0.001, device=DEVICE)
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
        y_actual = crypten.cryptensor(x, device=DEVICE).gelu(approximate=approximate, k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
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

        # 2. 性能测试
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 16) - 8
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.silu(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("SiLU", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_exp():
    crypten.cfg.functions.exp_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.linspace(-16.0, -2.0, 10000, device=DEVICE)
        y_original = torch.exp(x)
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
    crypten.cfg.functions.erf_method = "newer_time"

    for k1, k2 in K_PARAMS:
        # 1. 精度验证
        x = torch.linspace(-5.0, 5.0, 10000, device=DEVICE)
        y_original = torch.erf(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).erf(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 2. 性能测试
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
    # 1. 配置
    crypten.cfg.functions.softmax_method = "newer"
    crypten.cfg.functions.reciprocal_method = "newer_time"
    crypten.cfg.functions.exp_method = "newer_time"
    
    for k1, k2 in K_PARAMS:
        # 2. 精度验证
        x = torch.randn(1, 128, device=DEVICE) 
        y_original = torch.nn.functional.softmax(x, dim=-1)
        
        x_enc = crypten.cryptensor(x, device=DEVICE)
        y_mpc = x_enc.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)
        y_actual = y_mpc.get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        # 3. 性能测试
        def input_gen(size):
            return crypten.cryptensor(torch.randn(*size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Softmax", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)

# ================== 🚀 主程序 (Main) ==================

def main():
    crypten.init()
    print(f"🚀 Running tests on device: {DEVICE}")
    
    # 你可以注释掉不需要跑的测试
    test_sigmoid()
    test_tanh()
    test_exp()
    test_erf()
    test_reciprocal()
    test_inv_sqrt()
    test_gelu()
    test_silu()
    test_softmax()

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()