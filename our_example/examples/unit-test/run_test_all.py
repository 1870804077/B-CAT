import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher
import argparse


def test_function(func_name, device="cuda", runs=10):
    # 配置方法（假设都使用 "newer_time"）
    method_map = {
        "gelu": "newer_time",
        "silu": "newer_time",
        "exp": "newer_time",
        "sigmoid": "newer_time",
        "tanh": "newer_time",
        "reciprocal": "newer_time",
        "inv_sqrt": "newer_time",
    }

    if func_name in ["gelu", "silu"]:
        crypten.cfg.functions.gelu_method = method_map[func_name]
        crypten.cfg.functions.silu_method = method_map[func_name]
    elif func_name in ["sigmoid", "tanh"]:
        crypten.cfg.functions.sigmoid_tanh_method = method_map[func_name]
    elif func_name == "exp":
        crypten.cfg.functions.exp_method = method_map[func_name]
    elif func_name == "reciprocal":
        crypten.cfg.functions.reciprocal_method = method_map[func_name]
    elif func_name == "inv_sqrt":
        crypten.cfg.functions.inv_sqrt_method = method_map[func_name]

    # --- 精度验证 ---
    if func_name == "gelu":
        x = torch.arange(-5, 5, 0.001)
        y_original = torch.nn.functional.gelu(x, approximate="none")
        y_actual = crypten.cryptensor(x).gelu(approximate="none").get_plain_text()
    elif func_name == "silu":
        x = torch.arange(-5, 5, 0.001)
        y_original = torch.nn.functional.silu(x)
        y_actual = crypten.cryptensor(x).silu().get_plain_text()
    elif func_name == "exp":
        x = torch.linspace(-16.0, -4.0, 10000)
        y_original = torch.exp(x)
        y_actual = crypten.cryptensor(x).exp().get_plain_text()
    elif func_name == "sigmoid":
        x = torch.arange(-5, 5, 0.001)
        y_original = torch.sigmoid(x)
        y_actual = crypten.cryptensor(x).sigmoid().get_plain_text()
    elif func_name == "tanh":
        x = torch.arange(-5, 5, 0.001)
        y_original = torch.tanh(x)
        y_actual = crypten.cryptensor(x).tanh().get_plain_text()
    elif func_name == "reciprocal":
        x = torch.arange(0.1, 5.0, 0.001)
        y_original = 1.0 / x
        y_actual = crypten.cryptensor(x).reciprocal().get_plain_text()
    elif func_name == "inv_sqrt":
        x = torch.arange(0.1, 5.0, 0.001)
        y_original = 1.0 / torch.sqrt(x)
        y_actual = crypten.cryptensor(x).inv_sqrt().get_plain_text()
    else:
        raise ValueError(f"Unsupported function: {func_name}")

    max_err = (y_original - y_actual).abs().max()
    avg_err = (y_original - y_actual).abs().mean()

    # --- 性能测试 ---
    sizes = [(128, 3072), (128, 4096)]
    time_dict, bytes_dict, rounds_dict = {}, {}, {}

    for size in sizes:
        if func_name == "reciprocal":
            inp = crypten.cryptensor(torch.ones(size), device=device)  # avoid near-zero
        elif func_name == "inv_sqrt":
            inp = crypten.cryptensor(torch.ones(size), device=device)  # x > 0
        else:
            inp = crypten.cryptensor(torch.zeros(size), device=device)

        crypten.reset_communication_stats()
        start_time = time.time()

        for _ in range(runs):
            if func_name == "gelu":
                inp.gelu(k1=1,k2=12)
            elif func_name == "silu":
                inp.silu(k1=1,k2=12)
            elif func_name == "exp":
                inp.exp(k1=1, k2=12)
            elif func_name == "sigmoid":
                inp.sigmoid(k1=1, k2=12)
            elif func_name == "tanh":
                inp.tanh(k1=1, k2=12)
            elif func_name == "reciprocal":
                inp.reciprocal(k1=1, k2=12)
            elif func_name == "inv_sqrt":
                inp.inv_sqrt(k1=1, k2=12)

        elapsed = time.time() - start_time
        stats = crypten.get_communication_stats()

        time_dict[size[1]] = elapsed
        bytes_dict[size[1]] = stats["bytes"]
        rounds_dict[size[1]] = stats["rounds"]

    # --- 打印结果（仅 rank 0）---
    if crypten.comm.get().get_rank() == 0:
        print(f"\n=== {func_name.upper()} ===")
        if func_name in ["reciprocal", "inv_sqrt"]:
            print(f"max error: {max_err:.6f}, avg error: {avg_err:.8f}")
        else:
            print(f"max error: {max_err:.4f}, avg error: {avg_err:.6f}")

        for size in sizes:
            s0, s1 = size
            t = time_dict[s1] / runs
            b = bytes_dict[s1] / 1048576 / runs
            r = rounds_dict[s1] / runs
            print(f"({s0}, {s1}) time: {t:.4f}s, bytes: {b:.0f} MB, rounds: {r:.0f}")


def main(args):
    crypten.init()
    
    # 如果命令行输入了函数名，就测单个；否则测所有
    if args.func:
        test_function(args.func, runs=args.runs)
    else:
        all_functions = ["gelu", "silu", "exp", "sigmoid", "tanh", "reciprocal", "inv_sqrt"]
        for func in all_functions:
            print(f"\n🚀 Running test for: {func}")
            test_function(func, runs=args.runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 将 positional argument 改为 optional argument
    parser.add_argument("--func", type=str, choices=[
        "gelu", "silu", "exp", "sigmoid", "tanh", "reciprocal", "inv_sqrt"
    ], help="指定单个函数测试，若不指定则运行所有测试")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    args = parser.parse_args()

    launcher = MultiProcessLauncher(2, main, args)
    launcher.start()
    launcher.join()
    launcher.terminate()