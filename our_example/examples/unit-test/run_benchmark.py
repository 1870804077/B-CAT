import time
import torch
import crypten
import os
from multiprocess_launcher import MultiProcessLauncher

# ================== ⚙️ 全局配置 (Global Config) ==================
DEVICE = "cuda"
RUNS = 100  # 性能测试运行次数

# 定义日志保存的文件名
LOG_FILE = "benchmark_results.txt"

K_PARAMS = [(1, 8), (1, 12), (2, 12), (3, 12)]
TEST_SIZES = [(1, 500), (1, 1000), (1, 1500), (1, 2000)]

# ================== 🛠️ [新增] 日志与输出工具 ==================

def tee_print(*args, **kwargs):
    """
    双向输出函数：
    1. 打印到控制台 (stdout)
    2. 追加到文件 (LOG_FILE)
    
    安全机制：
    内部检查 Rank，只有 Rank 0 才会执行文件写入，防止多进程冲突。
    """
    # 1. 正常打印到控制台
    print(*args, **kwargs)

    # 2. 文件写入逻辑 (仅限 Rank 0)
    try:
        rank = crypten.comm.get().get_rank()
    except:
        # 如果 crypten 尚未初始化 (例如在 launcher 启动前)，默认为 0
        rank = 0

    if rank == 0:
        # 提取 print 的格式化参数
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        
        # 拼接字符串
        content = sep.join(map(str, args)) + end
        
        # 追加写入文件
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            # 防止文件写入报错导致程序崩溃，只在控制台提示
            print(f"[Error writing to log]: {e}")

# ================== 🛠️ 性能测试辅助工具 ==================

def run_perf_benchmark(input_generator_func, operation_func):
    """
    执行具体的性能测试循环
    """
    times, bytes_stats, rounds_stats = {}, {}, {}

    for size in TEST_SIZES:
        # 生成输入
        x_enc = input_generator_func(size)
        
        # 重置通信统计
        crypten.reset_communication_stats()
        
        # 计时开始
        start_time = time.time()
        for _ in range(RUNS):
            operation_func(x_enc)
        
        # 记录数据
        dim = size[1]
        times[dim] = time.time() - start_time
        stats = crypten.get_communication_stats()
        bytes_stats[dim] = stats["bytes"]
        rounds_stats[dim] = stats["rounds"]
        
    return times, bytes_stats, rounds_stats

def print_report(op_name, config_str, max_err, avg_err, times, bytes_stats, rounds_stats):
    """
    打印报告：使用 tee_print 替代 print
    """
    # 只让 Rank 0 负责打印和记录，避免 Rank 1 重复输出
    if crypten.comm.get().get_rank() == 0:
        tee_print(f"\n{'='*20} 🧪 {op_name} Test {'='*20}", flush=True)
        tee_print(f"⚙️  Configuration: {config_str}", flush=True)
        tee_print("-" * 50, flush=True)
        
        tee_print(f"✅ Precision Check:", flush=True)
        tee_print(f"   Max Error: {max_err:.8f}", flush=True)
        tee_print(f"   Avg Error: {avg_err:.8f}", flush=True)
        
        tee_print(f"🚀 Performance (Avg over {RUNS} runs):", flush=True)
        for size in TEST_SIZES:
            dim = size[1]
            t = times[dim] / RUNS
            comm = bytes_stats[dim] / 1048576 / RUNS
            rnd = rounds_stats[dim] / RUNS
            tee_print(f"   Shape {size}: Time: {t:.7f}s | Comm: {comm:.7f}MB | Rounds: {rnd:.0f}", flush=True)
        tee_print("="*60, flush=True)

# ================== 🧪 具体测试函数 (Test Functions) ==================

def test_sigmoid():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time+"
    for k1, k2 in K_PARAMS:
        x = torch.arange(-5, 5, 0.001, device=DEVICE)
        y_original = torch.sigmoid(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).sigmoid(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.sigmoid(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Sigmoid", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_tanh():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time+"
    for k1, k2 in K_PARAMS:
        x = torch.arange(-5, 5, 0.001, device=DEVICE)
        y_original = torch.tanh(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).tanh(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

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
        x = torch.arange(-5, 5, 0.001, device=DEVICE)
        y_original = torch.nn.functional.gelu(x, approximate=approximate)
        y_actual = crypten.cryptensor(x, device=DEVICE).gelu(approximate=approximate, k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

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
        x = torch.arange(-15, 15, 0.001, device=DEVICE)
        y_original = torch.nn.functional.silu(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).silu(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

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
        x = torch.linspace(-15.0, -3.0, 10000, device=DEVICE)
        y_original = torch.exp(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).exp(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.exp(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Exp", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_erf():
    crypten.cfg.functions.erf_method = "newer_time+"
    for k1, k2 in K_PARAMS:
        x = torch.linspace(-5.0, 5.0, 10000, device=DEVICE)
        y_original = torch.erf(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).erf(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 8) - 4
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.erf(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Erf", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_reciprocal():
    crypten.cfg.functions.reciprocal_method = "newer_time+"
    for k1, k2 in K_PARAMS:
        x = torch.arange(0.1, 5.0, 0.001, device=DEVICE)
        y_original = 1.0 / x
        y_actual = crypten.cryptensor(x, device=DEVICE).reciprocal(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.reciprocal(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Reciprocal", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_inv_sqrt():
    crypten.cfg.functions.sqrt_method = "newer_time+"
    for k1, k2 in K_PARAMS:
        x = torch.arange(0.1, 5.0, 0.001, device=DEVICE)
        y_original = 1.0 / torch.sqrt(x)
        y_actual = crypten.cryptensor(x, device=DEVICE).inv_sqrt(k1=k1, k2=k2).get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.inv_sqrt(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("InvSqrt", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)


def test_softmax():
    crypten.cfg.functions.softmax_method = "newer"
    crypten.cfg.functions.reciprocal_method = "newer_time"
    crypten.cfg.functions.exp_method = "newer_time"
    for k1, k2 in K_PARAMS:
        x = torch.randn(1, 128, device=DEVICE) 
        y_original = torch.nn.functional.softmax(x, dim=-1)
        
        x_enc = crypten.cryptensor(x, device=DEVICE)
        y_mpc = x_enc.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)
        y_actual = y_mpc.get_plain_text()
        
        diff = (y_original.cpu() - y_actual.cpu()).abs()
        max_err, avg_err = diff.max(), diff.mean()

        def input_gen(size):
            return crypten.cryptensor(torch.randn(*size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Softmax", f"k1={k1}, k2={k2}", max_err, avg_err, times, comms, rounds)

# ================== 🚀 主程序 (Main) ==================

def main():
    crypten.init()
    
    # 获取 rank，只有 rank 0 负责打印循环进度
    rank = crypten.comm.get().get_rank()

    for i in range(1, 6):
        if rank == 0:
            tee_print(f"\n📢 第 {i} 次测试循环", flush=True)

        test_sigmoid()
        test_tanh()

        test_exp()
        # test_erf()

        # test_gelu()
        # test_silu()

        test_reciprocal()
        test_inv_sqrt()
        # test_softmax()

if __name__ == "__main__":
    # 初始化：清空旧日志文件，写入新标题
    # 这步操作在 Launcher 启动多进程之前执行，所以是单进程操作，很安全
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Benchmark Report - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    print(f"📄 日志将保存至: {os.path.abspath(LOG_FILE)}")
    
    # 启动多进程
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()