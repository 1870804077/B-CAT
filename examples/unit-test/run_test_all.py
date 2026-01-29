import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = 5  

K_PARAMS = [(1, 12) ,(3, 12)] 

TEST_SIZES = [
    (1, 128, 768),         
    (8, 128, 768)       
]


def run_perf_benchmark(input_generator_func, operation_func):
    times, bytes_stats, rounds_stats = {}, {}, {}

    for size in TEST_SIZES:
        x_enc = input_generator_func(size)
        
        operation_func(x_enc)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        crypten.reset_communication_stats()
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        
        for _ in range(RUNS):
            operation_func(x_enc)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time

        dim_key = str(size) 
        times[dim_key] = total_time
        
        stats = crypten.get_communication_stats()
        bytes_stats[dim_key] = stats["bytes"]
        rounds_stats[dim_key] = stats["rounds"]
        
    return times, bytes_stats, rounds_stats

def print_report(op_name, config_str, times, bytes_stats, rounds_stats):
    if crypten.comm.get().get_rank() == 0:
        print(f"\n{'='*20}   {op_name} Performance Test {'='*20}", flush=True)
        print(f"   Configuration: {config_str}", flush=True)
        print("-" * 65, flush=True)
        print(f"  Average Performance over {RUNS} runs:", flush=True)
        print(f" {'Shape':<20} | {'Time (s)':<12} | {'Comm (MB)':<12} | {'Rounds':<8}")
        print("-" * 65, flush=True)
        
        for size in TEST_SIZES:
            dim_key = str(size)
            if dim_key in times:
                t = times[dim_key] / RUNS
                comm = bytes_stats[dim_key] / 1048576 / RUNS # Convert to MB
                rnd = rounds_stats[dim_key] / RUNS
                print(f" {dim_key:<20} | {t:<12.5f} | {comm:<12.5f} | {rnd:<8.0f}", flush=True)
        print("="*65 + "\n", flush=True)

# ==================   具体测试函数 (Test Functions) ==================

def test_sigmoid():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time"
    
    for k1, k2 in K_PARAMS:
        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.sigmoid(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Sigmoid", f"k1={k1}, k2={k2}", times, comms, rounds)


def test_tanh():
    crypten.cfg.functions.sigmoid_tanh_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            return crypten.cryptensor(torch.zeros(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.tanh(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Tanh", f"k1={k1}, k2={k2}", times, comms, rounds)


def test_gelu():
    crypten.cfg.functions.gelu_method = "newer_time"
    approximate = "none"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 10) - 5
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.gelu(approximate=approximate, k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("GeLU", f"k1={k1}, k2={k2}", times, comms, rounds)


def test_silu():
    crypten.cfg.functions.silu_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 16) - 8
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.silu(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("SiLU", f"k1={k1}, k2={k2}", times, comms, rounds)


def test_exp():
    crypten.cfg.functions.exp_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 10) - 10 
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.exp(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Exp", f"k1={k1}, k2={k2}", times, comms, rounds)

def test_erf():
    crypten.cfg.functions.erf_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            data = (torch.rand(*size, device=DEVICE) * 8) - 4
            return crypten.cryptensor(data, device=DEVICE)

        def op_func(enc_x):
            enc_x.erf(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Erf", f"k1={k1}, k2={k2}", times, comms, rounds)

def test_reciprocal():
    crypten.cfg.functions.reciprocal_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.reciprocal(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Reciprocal", f"k1={k1}, k2={k2}", times, comms, rounds)


def test_inv_sqrt():
    crypten.cfg.functions.sqrt_method = "newer_time"

    for k1, k2 in K_PARAMS:
        def input_gen(size):
            return crypten.cryptensor(torch.ones(size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.inv_sqrt(k1=k1, k2=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("InvSqrt", f"k1={k1}, k2={k2}", times, comms, rounds)

def test_softmax():
    crypten.cfg.functions.softmax_method = "newer"
    crypten.cfg.functions.reciprocal_method = "newer_time"
    crypten.cfg.functions.exp_method = "newer_time"
    
    for k1, k2 in K_PARAMS:
        def input_gen(size):
            return crypten.cryptensor(torch.randn(*size, device=DEVICE), device=DEVICE)

        def op_func(enc_x):
            enc_x.softmax(dim=-1, k1=k1, k2_exp=k2, k2_recip=k2)

        times, comms, rounds = run_perf_benchmark(input_gen, op_func)
        print_report("Softmax", f"k1={k1}, k2={k2}", times, comms, rounds)


def main():
    crypten.init()
    if crypten.comm.get().get_rank() == 0:
        print(f"  Running Performance Benchmarks on device: {DEVICE}")
        print(f"  Runs per test: {RUNS}")
    
    test_sigmoid()
    test_tanh()
    test_exp()
    test_erf()

    test_reciprocal()
    test_inv_sqrt()
    test_gelu()
    test_softmax()

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()